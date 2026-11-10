"""
PFM: follow ferroelectric DOMAIN WALLS instead of topographic edges (Run_PFM_DomainWalls.ipynb).

Signals. Igor's PFM amplitude and/or phase (the lock-in, PFM mode -- the drive stays on
while engaged) are copied to BNC outputs the same way as the X/Y sensors (a PIDS loop as a
straight copy -> a free DAC -> a BNC, locked), cabled to PyScanner AIs, and read from the
same AI record as everything else (ContinuousExecutor.extra_channels). Each one is optional
(cfg.pfm.amplitude / .phase .enabled): amplitude only, phase only, or both. Defaults:
    amplitude: Cypher loop 5 -> OutFastA -> BNCOut2 -> AI0 (where the Z copy went)
    phase:     ARC loop 2 -> ARC OutA -> ARC BNCOut0 -> AI3      (CONFIRM ON HARDWARE)
(the Cypher's six loops are taken: 0/1 Igor's LVDT X/Y, 2 the Z feedback, 3/4 the X/Y
sensor copies; the ARC's 2..5 were free, 2026-10-16). BNC volts = gain x signal, so
signal = AI volts / gain (the executor converts: amplitude in V, phase in degrees).

What a wall is (DomainModel -- the same interface as substrate.SubstrateModel, so the
edge explorer's whole machinery -- following, corners, closing, the area border, the
sensors -- is reused unchanged):
  - phase (with or without amplitude): the phase forms two populations (the two domains,
    ideally 180 deg apart -- any separation >= min_phase_separation_deg), found on the fly
    with circular statistics (wrap-around safe) and tracking slow drift. Every sample gets
    a score s = w * d in [-1, 1]: d = which domain's phase it's nearer (cos-difference,
    normalised), w = its amplitude / the domains' amplitude (1 without amplitude), so where
    the amplitude vanishes (at the wall) the phase -- pure noise there -- counts for
    nothing. "On" = domain B. With amplitude, a phase flip only counts as a wall if the
    amplitude dips within confirm_window_V of it (confirm_with_amplitude).
  - amplitude only: no domains, the wall itself is "on": amplitude below amp_dip_fraction x
    the domains' -- a thin strip the explorer follows round (both its sides).
The explorer compares |height| with a threshold: height = 2 + 2 s, threshold 2 (s > 0:
domain B), with its hysteresis (+-25% -> s beyond +-0.25) against flicker.

Simulation: PFMSampleBackend -- domains (bubbles, stripes, a labyrinth), amplitude dipping
over a wall width, phase 0/180 deg + noise (more where the amplitude is low) + drift.
"""

from __future__ import annotations
import json
import time
from typing import Optional

import numpy as np

from .config import PFMConfig, EdgeExplorerConfig
from .planners.edge_explorer_planner import EdgeExplorerPlanner
from .backends.simulate_backend import SimulateBackend

THRESHOLD = 2.0


# ====================================================================================
# channels
# ====================================================================================
def channel_map(pfm: PFMConfig) -> dict:
    """{name: (ai_channel, signal units per AI volt)} of the enabled channels."""
    return {k: (c.ai_channel, 1.0 / c.gain) for k, c in pfm.channels().items()}


def primary(pfm: PFMConfig) -> str:
    """The channel recorded as "z" (live view, reconstruction, raster): phase if on, else amplitude."""
    ch = pfm.channels()
    if not ch:
        raise ValueError("cfg.pfm: enable the amplitude and/or the phase channel")
    return "phase" if "phase" in ch else "amplitude"


def z_info(pfm: PFMConfig, ai_matches=None) -> dict:
    """What z is (executor.set_z_units): the primary channel, in its own units."""
    c = pfm.channels()[primary(pfm)]
    return dict(source=c.source, ai_channel=c.ai_channel, unit=c.unit, scale=1.0 / c.gain, label=c.label,
                ai_matches=ai_matches, pfm=True)


def configure(cfg, backend, executor=None) -> None:
    """Point the run at the PFM channels: readback = the primary channel's AI, the others read
    with it (executor.extra_channels), z in the primary's units."""
    p = cfg.pfm
    cfg.readback.ai_channel = p.channels()[primary(p)].ai_channel
    if getattr(backend, "z_signal_info", None) is None or not backend.z_signal_info.get("pfm"):
        backend.z_signal_info = z_info(p)
    backend.readback = getattr(backend, "readback", cfg.readback)
    if executor is not None:
        executor.extra_channels = channel_map(p)
        executor.z_signal_info = backend.z_signal_info
        executor.set_z_units()


# ====================================================================================
# routing (asylum.AsylumControl): each channel copied to its BNC, then checked at rest
# ====================================================================================
def route(afm, verbose: bool = True) -> dict:
    """Copy every enabled PFM channel to its BNC (asylum._copy_to_bnc, Cypher or ARC side),
    then -- at rest -- check its AI reads the copy. Returns {name: {dac, ai_matches, ...}};
    the primary's goes to backend.z_signal_info."""
    p = afm.cfg.pfm
    out = {}
    from .asylum import PANEL, _side_name
    taken_c = afm._xy_sensor_dacs() if hasattr(afm, "_xy_sensor_dacs") else []
    # our own earlier copies this run doesn't use may be taken over: the Z copy when
    # cfg.z_signal.route is off (the topography notebook leaves it running in Igor), and a
    # PFM signal that's switched off now
    zs = afm.cfg.z_signal
    spare = ([zs.source, "ZSensor"] if not zs.route else []) + \
        [o.source for o in (p.amplitude, p.phase) if not o.enabled]
    for name, c in p.channels().items():
        sn = _side_name(c.side)
        taken = taken_c if sn == "Cypher" else []
        dac = afm._copy_to_bnc(c.source, c.loop, c.bnc, -c.gain, prefer=(c.dac,), taken=taken, side=sn,
                               replaceable=spare)
        if sn == "Cypher":
            taken_c = list(taken_c) + [dac]
        afm.ex("WriteXPT", PANEL, 0, "")
        info = dict(source=c.source, side=sn, loop=c.loop, dac=dac, bnc=c.bnc, ai_channel=c.ai_channel,
                    gain=c.gain, unit=c.unit, ai_matches=None)
        if afm.igor is not None:
            time.sleep(0.2)
            afm._copy_matches(c.source, dac[3:], -c.gain, tol_V=max(p.check_tol, 0.05), verbose=True, side=sn)
            info["ai_matches"] = check_ai(afm, name, c, dac, sn)
        out[name] = info
    afm.pfm_info = out
    pm = primary(p)
    afm.backend.z_signal_info = z_info(p, out.get(pm, {}).get("ai_matches"))
    configure(afm.cfg, afm.backend, afm.executor)
    return out


def check_ai(afm, name, c, dac, side) -> Optional[bool]:
    """At rest: does c.ai_channel read the copy on the DAC? (a wrong cable -> a warning)"""
    n = max(int(0.2 * afm.cfg.velocity.sample_rate_hz), 20)
    p = getattr(afm.backend, "current_pos", None)
    x0, y0 = (float(p[0]), float(p[1])) if p is not None else (0.0, 0.0)
    afm.backend.measure_path(np.full(n, x0), np.full(n, y0))
    try:
        ai = np.asarray(afm.backend.read_channels([c.ai_channel])[0], dtype=float)
    except Exception as e:   # noqa: BLE001
        print(f"  {name}: can't read {c.ai_channel} ({e})")
        return None
    dv = afm.read_value(f"{side}.Output.{dac[3:]}")
    sv = afm.read_value(c.source)
    if not len(ai) or dv is None:
        return None
    a = float(np.mean(ai))
    ok = abs(a - dv) <= max(afm.cfg.pfm.check_tol, 3 * float(np.std(ai)))
    print(f"  {c.ai_channel} reads {a:+.4f} V (std {np.std(ai) * 1e3:.2f} mV); {side} {dac} = {dv:+.4f} V "
          f"({c.source} {sv:+.4g} {c.unit} x {c.gain:g})" + (f"  -> {c.ai_channel} is the {c.label}" if ok else ""))
    if not ok:
        print(f"  WARNING: {c.ai_channel} does NOT read the {c.source} copy -- is its cable on the {side}'s {c.bnc}?")
    return ok


def routing_ok(afm) -> Optional[bool]:
    """Every PFM copy still in place (loop running on its source, the BNC on its DAC)?"""
    if afm.igor is None:
        return None
    from .asylum import _g, _active, _p_ok, _side_name
    for name, c in afm.cfg.pfm.channels().items():
        sn = _side_name(c.side)
        g = afm.read_loop(c.loop, side=sn)
        out = str(_g(g, "OutputChannel")).split(".")[-1]
        if str(_g(g, "InputChannel")).casefold() != c.source.casefold() or not _active(g) \
                or str(afm.read_route(c.bnc, sn)).casefold() != f"out{out}".casefold() or not _p_ok(g, -c.gain):
            return False
    return True


# ====================================================================================
# what a wall is
# ====================================================================================
def _wrap(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


class DomainModel:
    """The on/off classifier for domain walls -- the interface of substrate.SubstrateModel
    (classify / update / height / threshold / plane / summary ...) on PFM signals. The
    signals of the segment being classified are given first (set_data); x, y, z are the
    explorer's (z = the primary channel, unused here)."""

    degree = 0                       # nothing to flatten in the images (viz)

    def __init__(self, walls, has_amp: bool, has_phase: bool):
        self.w = walls
        self.has_amp, self.has_phase = has_amp, has_phase
        self.amp = self.phase = None
        self.phA = self.phB = None       # the two domains' phases (deg)
        self.Aref = None                 # the domains' amplitude
        self.threshold = THRESHOLD
        self.coef = np.array([0.0])
        self.sigma = 0.0
        self.gap_active = False          # = both domains found
        self.history = []
        self._seen = 0
        self.fitted = False

    # ------------------------------------------------------------------
    def set_data(self, amp, phase):
        self.amp = None if amp is None else np.asarray(amp, dtype=float)
        self.phase = None if phase is None else np.asarray(phase, dtype=float)

    def plane(self, x, y):
        return np.zeros(np.shape(x))

    def _weights(self, n):
        if self.amp is None or self.Aref is None or not self.Aref > 0:
            return np.ones(n)
        return np.clip(self.amp[:n] / self.Aref, 0.0, 1.0)

    def score(self, n):
        """s in [-1, 1] per sample: > 0 domain B (phase), or the wall (amplitude only)."""
        if not self.has_phase:          # amplitude only: the wall is "on"
            if self.amp is None or self.Aref is None:
                return -np.ones(n)
            f = self.w.amp_dip_fraction
            # s = +1 at zero amplitude, 0 at the dip level, -1 at the domains' amplitude
            return np.clip((f - self.amp[:n] / self.Aref) / max(f, 1 - f), -1.0, 1.0)
        if self.phase is None or self.phA is None:
            return -np.ones(n)
        ph = self.phase[:n]
        if self.phB is None:            # one domain so far: everything is A
            return -self._weights(n)
        sep = np.deg2rad(abs(_wrap(self.phB - self.phA)))
        d = (np.cos(np.deg2rad(ph - self.phB)) - np.cos(np.deg2rad(ph - self.phA))) / max(1 - np.cos(sep), 1e-6)
        return np.clip(d, -1.0, 1.0) * self._weights(n)

    def height(self, x, y, z):
        return THRESHOLD + 2.0 * self.score(len(np.atleast_1d(z)))

    def classify(self, x, y, z):
        if not self.fitted:
            self.calibrate(x, y, z)
        return np.abs(self.height(x, y, z)) > self.threshold

    # ------------------------------------------------------------------
    def calibrate(self, x, y, z):
        self.fitted = True
        self._learn(first=True)

    def update(self, x, y, z, on):
        self._learn(first=False, on=on)

    def _learn(self, first, on=None):
        w = self.w
        if self.amp is not None and len(self.amp):
            a = float(np.quantile(self.amp, w.amp_reference_quantile))
            if a > 0:
                self.Aref = a if self.Aref is None or first else (w.phase_memory * self.Aref + (1 - w.phase_memory) * a)
        if self.has_phase and self.phase is not None and len(self.phase):
            ph = self.phase
            wt = self._weights(len(ph))
            good = wt > 0.5
            if good.sum() < 10:
                good = np.ones(len(ph), dtype=bool)
            if self.phA is None:
                # the most populated phase (circular histogram, weighted)
                h, e = np.histogram(_wrap(ph[good]), bins=72, range=(-180, 180), weights=wt[good])
                self.phA = float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))
            # samples clearly away from A: a second domain?
            dA = np.abs(_wrap(ph - self.phA))
            far = good & (dA > w.min_phase_separation_deg)
            if self.phB is None:
                if far.sum() >= max(20, w.min_second_fraction * len(ph)):
                    self.phB = float(np.rad2deg(np.angle(np.sum(wt[far] * np.exp(1j * np.deg2rad(ph[far]))))))
            if self.phB is not None:
                dB = np.abs(_wrap(ph - self.phB))
                inA, inB = good & (dA < dB) & (dA < 45), good & (dB < dA) & (dB < 45)
                m = w.phase_memory
                for attr, sel in (("phA", inA), ("phB", inB)):
                    if sel.sum() >= 10:
                        new = np.angle(np.sum(wt[sel] * np.exp(1j * np.deg2rad(ph[sel]))))
                        old = np.deg2rad(getattr(self, attr))
                        setattr(self, attr, float(np.rad2deg(np.angle(m * np.exp(1j * old) + (1 - m) * np.exp(1j * new)))))
                spread = np.concatenate([_wrap(ph[inA] - self.phA), _wrap(ph[inB] - self.phB)])
                self.sigma = float(np.median(np.abs(spread)) * 1.4826) if len(spread) else self.sigma
        self.gap_active = (self.phB is not None) if self.has_phase else (self.Aref is not None)
        self._seen += 0 if self.amp is None and self.phase is None else len(self.amp if self.amp is not None else self.phase)
        self.history.append((self._seen, self.threshold, self.sigma, self.gap_active, self.coef.copy()))

    def wall_dip(self, i, half_samples):
        """Does the amplitude dip within half_samples of sample i (a wall, not phase noise)?"""
        if self.amp is None or self.Aref is None:
            return True
        a = self.amp[max(0, i - half_samples):i + half_samples + 1]
        return bool(len(a)) and float(np.min(a)) < self.w.amp_dip_fraction * self.Aref

    def summary(self):
        return dict(kind="domain_walls", has_amplitude=self.has_amp, has_phase=self.has_phase,
                    phase_A_deg=self.phA, phase_B_deg=self.phB, amplitude_ref=self.Aref,
                    phase_spread_deg=self.sigma, threshold=self.threshold, coef=self.coef.tolist(),
                    degree=0, sigma=self.sigma, gap_active=self.gap_active)


class InPlaneWallModel(DomainModel):
    """The on/off classifier for IN-PLANE walls: "on" = the higher of two amplitude levels
    (2026-11-04: in-plane walls are amplitude steps, ~3.1 vs 2.0 mV, the phase unchanged within
    1 deg; the level carries on across 180 deg walls). The amplitude also depends on the tip's
    direction of motion (friction; trace vs retrace 3.55 vs 2.76 mV) -- so each level is a curve
    over the direction: log A = c_k + sum_h (a_kh cos h t + b_kh sin h t), two components fitted
    by hard EM on a reservoir of every waveform's samples. Samples in a 180 deg wall's amplitude
    dip (below inplane_dip_fraction x the low level) are ignored: the state is held across them,
    so 180 deg walls are not edges. Until two levels are seen everything is "off"."""

    def __init__(self, walls, has_amp: bool, has_phase: bool):
        super().__init__(walls, has_amp, has_phase)
        self.co = None                   # (2, n_terms): the two levels' direction curves
        self.ratio = None
        self._res = None                 # learning reservoir: (log A, direction)
        self._last_s = -1.0              # the state at the end of the last classified waveform
        self.xy = None
        self.ph_axis = None              # the two phase domains' axis (deg; domains at axis +- 90)

    def _theta(self, x, y):
        """Direction of motion per sample, from the positions smoothed over ~inplane_smooth_V."""
        from scipy.ndimage import uniform_filter1d
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        if n < 3:
            return np.zeros(n)
        ds = float(np.median(np.hypot(np.diff(x), np.diff(y)))) or 1e-6
        w = int(np.clip(self.w.inplane_smooth_V / ds, 5, max(5, n // 2)))
        return np.arctan2(np.gradient(uniform_filter1d(y, w)), np.gradient(uniform_filter1d(x, w)))

    def _X(self, t):
        cols = [np.ones_like(t)]
        for h in range(1, int(self.w.inplane_direction_harmonics) + 1):
            cols += [np.cos(h * t), np.sin(h * t)]
        return np.column_stack(cols)

    def _mu(self, t):
        X = self._X(t)
        return X @ self.co[0], X @ self.co[1]

    def _fit(self):
        la, t = self._res
        X = self._X(t)
        lab = la > np.median(la)
        co = np.zeros((2, X.shape[1]))
        for _ in range(40):
            for k in (0, 1):
                m = lab == k
                if m.sum() < X.shape[1] * 5:
                    return
                co[k] = np.linalg.lstsq(X[m], la[m], rcond=None)[0]
            new = np.abs(la - X @ co[1]) < np.abs(la - X @ co[0])
            if np.array_equal(new, lab):
                break
            lab = new
        frac = float(lab.mean())
        gap = float(np.median(X @ co[1] - X @ co[0]))
        self.sigma = float(np.std(np.where(lab, la - X @ co[1], la - X @ co[0])))
        if gap > np.log(self.w.inplane_min_level_ratio) and min(frac, 1 - frac) > self.w.inplane_min_fraction:
            self.co, self.ratio = co, float(np.exp(gap))
        self.gap_active = self.co is not None

    def _learn_axis(self):
        """The two phase domains' axis (angle doubling: phi and phi + 180 are one axis), mixed with
        the earlier estimate."""
        if self.phase is None or not len(self.phase):
            return
        z = np.mean(np.exp(2j * np.deg2rad(self.phase)))
        if abs(z) < 0.3:                 # not two clear domains (yet) in this waveform
            return
        new = np.exp(1j * np.angle(z))
        old = None if self.ph_axis is None else np.exp(2j * np.deg2rad(self.ph_axis))
        m = self.w.phase_memory
        zz = new if old is None else m * old + (1 - m) * new
        self.ph_axis = float(0.5 * np.rad2deg(np.angle(zz)))

    def phase_flip_near(self, i, half_samples) -> bool:
        """Does the phase change domain within half_samples of sample i (a 180 deg wall)?"""
        if self.phase is None or self.ph_axis is None:
            return False
        p = self.phase[max(0, i - half_samples):i + half_samples + 1]
        if len(p) < 3:
            return False
        side = np.cos(np.deg2rad(p - self.ph_axis)) > 0
        f = float(side.mean())
        return 0.05 < f < 0.95

    def _add(self, x, y):
        self._learn_axis()
        if self.amp is None or not len(self.amp):
            return
        a = self.amp[:len(x)]
        t = self._theta(x, y)[:len(a)]
        keep = a > 0
        if self.co is not None:          # leave the 180 deg walls' dips out of the learning
            keep &= a > self.w.inplane_dip_fraction * np.exp(self._mu(t)[0])
        else:                            # before the levels are known: the lowest 3% out
            keep &= a > np.quantile(a, 0.03)
        step = max(1, len(a) // 20000)
        la, tt = np.log(a[keep][::step]), t[keep][::step]
        self._res = (la, tt) if self._res is None else (np.concatenate([self._res[0], la]),
                                                        np.concatenate([self._res[1], tt]))
        n = len(self._res[0])
        if n > self.w.inplane_max_samples:            # a random subset, old and new alike
            k = np.sort(np.random.default_rng(n).choice(n, self.w.inplane_max_samples, replace=False))
            self._res = (self._res[0][k], self._res[1][k])
        self._fit()

    def score(self, n):
        """s in [-1, 1] per sample: > 0 = the high amplitude level; held across 180 deg dips."""
        from scipy.ndimage import uniform_filter1d
        if self.co is None or self.amp is None or self.xy is None:
            return -np.ones(n)
        x, y = self.xy[0][:n], self.xy[1][:n]
        a = self.amp[:n]
        m0, m1 = self._mu(self._theta(x, y))
        ok = a > self.w.inplane_dip_fraction * np.exp(m0)       # not in a 180 deg wall's dip
        ds = float(np.median(np.hypot(np.diff(x), np.diff(y)))) if n > 2 else 1.0
        g = int(round(self.w.inplane_dip_guard_V / max(ds, 1e-9)))
        if g > 0 and not ok.all():                              # ... nor on its flanks
            from scipy.ndimage import binary_dilation
            ok &= ~binary_dilation(~ok, iterations=min(g, n))
        w = int(np.clip(self.w.inplane_smooth_V / max(ds, 1e-9), 1, max(1, n)))
        la = np.log(np.maximum(a, 1e-12))
        num, den = uniform_filter1d(np.where(ok, la, 0.0), w), uniform_filter1d(ok.astype(float), w)
        ls = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
        s = np.clip((np.abs(ls - m0) - np.abs(ls - m1)) / np.maximum(m1 - m0, 1e-6), -1.0, 1.0)
        # hold the state across the dips (forward fill; before the first good sample: the
        # state the last waveform ended in)
        idx = np.maximum.accumulate(np.where(np.isfinite(s), np.arange(n), -1))
        return np.where(idx >= 0, s[np.maximum(idx, 0)], self._last_s)

    def height(self, x, y, z):
        self.xy = (np.asarray(x, float), np.asarray(y, float))
        return THRESHOLD + 2.0 * self.score(len(np.atleast_1d(z)))

    def calibrate(self, x, y, z):
        self.fitted = True
        self._add(x, y)
        self._hist()

    def update(self, x, y, z, on):
        self._add(x, y)
        self._hist()
        s = self.score(len(x))
        if len(s):
            self._last_s = float(s[-1])

    def _hist(self):
        self._seen += 0 if self.amp is None else len(self.amp)
        self.history.append((self._seen, self.threshold, self.sigma, self.gap_active, self.coef.copy()))

    def wall_dip(self, i, half_samples):
        return True

    def summary(self):
        d = super().summary()
        d.update(kind="in_plane_walls", level_ratio=self.ratio,
                 levels_mV=None if self.co is None else [float(np.exp(c[0]) * 1e3) for c in self.co],
                 direction_curves=None if self.co is None else self.co.tolist())
        return d


# ====================================================================================
# the planner
# ====================================================================================
class DomainWallExplorerPlanner(EdgeExplorerPlanner):
    """The edge explorer, following domain walls (DomainModel) instead of topographic
    edges. Features = domains (phase) or walls (amplitude only); their insides aren't
    scanned, nor taken as explored -- a domain can hold others, and the inside of one
    closed by the area's border is ambiguous -- so exploring goes on everywhere."""

    INSIDE_IS_KNOWN = False

    def __init__(self, executor, cfg: EdgeExplorerConfig, pfm: PFMConfig):
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.scan_interior = False
        cfg.calibration_radius_V = cfg.calibration_radius_V   # (the circle samples both signals first)
        super().__init__(executor, cfg)
        self.pfm = pfm

    def _make_model(self):
        ch = self.pfm.channels()
        if self.pfm.walls.target == "in_plane":
            if "amplitude" not in ch:
                raise ValueError("in-plane walls need the PFM amplitude (cfg.pfm.amplitude.enabled)")
            return InPlaneWallModel(self.pfm.walls, True, "phase" in ch)
        return DomainModel(self.pfm.walls, "amplitude" in ch, "phase" in ch)

    def _learn(self, x, y, z):
        ch = self.executor.last_channels or {}
        n = len(z)
        amp = ch.get("amplitude")
        ph = ch.get("phase")
        self.substrate.set_data(None if amp is None else amp[:n], None if ph is None else ph[:n])
        crossings = super()._learn(x, y, z)
        m = self.substrate
        if self.pfm.walls.target == "in_plane" and m.has_phase and crossings:
            # an amplitude-level change with a phase flip next to it: a 180 deg wall, not followed
            half = max(1, int(round(self.pfm.walls.inplane_flip_guard_V / self.executor.velocity.sample_spacing_V)))
            P = np.column_stack([x, y])
            crossings = [c for c in crossings
                         if not m.phase_flip_near(int(np.argmin(np.sum((P - c.p) ** 2, axis=1))), half)]
        if self.pfm.walls.target != "in_plane" and m.has_phase and m.has_amp \
                and self.pfm.walls.confirm_with_amplitude and crossings:
            # a phase flip without an amplitude dip near it: noise, not a wall
            half = max(1, int(round(self.pfm.walls.confirm_window_V / self.executor.velocity.sample_spacing_V)))
            P = np.column_stack([x, y])
            keep = []
            for c in crossings:
                i = int(np.argmin(np.sum((P - c.p) ** 2, axis=1)))
                if m.wall_dip(i, half):
                    keep.append(c)
            crossings = keep
        return crossings


# ====================================================================================
# simulation
# ====================================================================================
def domain_field(kind: str = "bubbles", seed: int = 0, extent: float = 3.0):
    """f(x, y): the domain pattern, walls where f = 0 (B where f > 0)."""
    rng = np.random.default_rng(seed)
    if kind == "stripes":
        th, period = rng.uniform(0, np.pi), 0.35 * extent
        def f(x, y):
            u = x * np.cos(th) + y * np.sin(th)
            v = -x * np.sin(th) + y * np.cos(th)
            return np.sin(2 * np.pi * u / period + 0.6 * np.sin(2 * np.pi * v / (1.3 * extent)))
        return f
    if kind == "labyrinth":
        k = rng.normal(0, 1, (12, 2)) * (3.0 / extent)
        ph = rng.uniform(0, 2 * np.pi, 12)
        def f(x, y):
            return sum(np.cos(kk[0] * x + kk[1] * y + p) for kk, p in zip(k, ph))
        return f
    # bubbles: B disks / ellipses in an A matrix
    C = rng.uniform(-0.7 * extent, 0.7 * extent, (5, 2))
    R = rng.uniform(0.12, 0.3, 5) * extent
    E = rng.uniform(0.7, 1.4, 5)
    def f(x, y):
        out = None
        for (cx, cy), r, e in zip(C, R, E):
            v = r - np.sqrt(((x - cx) * e) ** 2 + ((y - cy) / e) ** 2)
            out = v if out is None else np.maximum(out, v)
        return out
    return f


class PFMSampleBackend(SimulateBackend):
    """Simulated PFM: amplitude A0 dipping to ~0 over the wall width, phase phi0 / phi0+dphi
    on the two domains + noise (larger where the amplitude is low) + slow drift. AI volts =
    gain x signal for every enabled channel of cfg.pfm; z = the primary one."""

    def __init__(self, pfm: PFMConfig, field_fn, wall_width_V: float = 0.03, A0: float = 0.05,
                 phi0: float = 30.0, dphi: float = 180.0, phase_noise_deg: float = 6.0, amp_noise: float = 0.002,
                 drift_deg_per_s: float = 0.02, seed: int = 1, **kw):
        super().__init__(topography_fn=lambda x, y: 0 * x, noise_std=0.0, seed=seed, **kw)
        self.pfm, self.f = pfm, field_fn
        self.ww, self.A0, self.phi0, self.dphi = wall_width_V, A0, phi0, dphi
        self.pn, self.an, self.drift = phase_noise_deg, amp_noise, drift_deg_per_s
        self.t = 0.0
        self._last_ch = {}
        self.z_signal_info = z_info(pfm)

    def signals(self, x, y, t=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        f = self.f(x, y)
        e = 1e-4
        g = np.hypot((self.f(x + e, y) - f) / e, (self.f(x, y + e) - f) / e)
        dist = np.abs(f) / np.maximum(g, 1e-9)                  # distance to the wall
        amp = self.A0 * (1 - 0.95 * np.exp(-0.5 * (dist / self.ww) ** 2))
        dr = 0.0 if t is None else self.drift * t
        ph = self.phi0 + dr + np.where(f > 0, self.dphi, 0.0)
        ph = ph + self._rng.normal(0, 1, len(x)) * self.pn * np.clip(self.A0 / np.maximum(amp, 1e-6), 1, 20)
        amp = amp + self._rng.normal(0, self.an, len(x))
        return amp, _wrap(ph)

    def measure_path(self, x_vec, y_vec):
        x_vec, y_vec = np.asarray(x_vec, float), np.asarray(y_vec, float)
        t = self.t + np.arange(len(x_vec)) / 1e4
        self.t = float(t[-1]) if len(t) else self.t
        amp, ph = self.signals(x_vec, y_vec, t)
        self._last_ch = {}
        for name, c in self.pfm.channels().items():
            self._last_ch[c.ai_channel] = (amp if name == "amplitude" else ph) * c.gain
        self.current_pos = np.array([x_vec[-1], y_vec[-1]])
        prim = self._last_ch[self.pfm.channels()[primary(self.pfm)].ai_channel]
        self.history_x.append(x_vec); self.history_y.append(y_vec); self.history_z.append(prim)
        return prim

    def read_channels(self, ai_channels):
        return [self._last_ch[a] for a in ai_channels]

    def run_pyscanner_scan(self, scan, sample_rate_hz):
        from . import pyscanner_scans as ps
        x, y = ps.predict_path(scan, sample_rate_hz)
        return x, y, self.measure_path(x, y)
