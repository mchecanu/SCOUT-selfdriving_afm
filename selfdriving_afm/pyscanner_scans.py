"""
PyScanner's OWN scan types (the "pyscanner_defined" scan generator).

Instead of sending a user-defined waveform (scan_type 0), PyScanner generates the path
itself from its scan-type parameters -- the tabs on its front panel:

    scan_type 1 = spiral        (spiral_scan_control_cluster)
        spiral_inner_radius_x_V_00, spiral_outer_radius_x_V_01, spiral_inner_radius_y_V_02,
        spiral_outer_radius_y_V_03, spiral_N_cycles_04, spiral_duration_05,
        spiral_dose_distribution_06 (greyed out), spiral_direction_07 (0 = clockwise out),
        spiral_return_opt_08 (0 = direct return)
    scan_type 2 = fast raster   (fast_raster_scan_control_cluster)
        raster_scan_size_x_V_00, raster_scan_size_y_V_01, raster_N_scan_lines_02,
        raster_scan_duration_s_03, raster_type_04 (2 = sinusoidal)
    scan_type 3 = Lissajous     (lissajous_scan_control_cluster)
        lissajous_scan_size_x_V_00, lissajous_scan_size_y_V_01, lissajous_scan_w1_Hz_02,
        lissajous_scan_w2_Hz_03, lissajous_scan_phi_rad_04, lissajous_scan_duration_s_05
        x = A sin(2 pi w1 t + phi),  y = B sin(2 pi w2 t)
        -- the "size" fields are the AMPLITUDES A, B (half the width): checked on hardware,
        size 0.2 gave a path 0.4 V wide. PyScannerScan.size_x_V/size_y_V stay full widths;
        cluster_values() writes half of them.

and every scan is placed with scan_x_offset_V / scan_y_offset_V / scan_rotation_deg.

This module turns "scan this box / disk with lines `spacing` apart at tip speed v" into
those parameters (`PyScannerScan`), and predicts the path PyScanner will run
(`predict_path`) -- for simulation, for the safety check, and for where the scan
STARTS (the package moves the tip there smoothly first, so PyScanner never jumps).
The real path is always read back from PyScanner (AO_vs_time) after the scan, and
compared with the prediction (see PyScannerBackend.run_pyscanner_scan).

Fast raster, checked on hardware (probes 2026-09-24 and 2026-10-05): every line is a
full cosine cycle -- trace AND retrace, x = cx - A cos(2 pi t / T_line), T_line =
duration / n_lines -- starting at the box's low-x, low-y corner; the slow axis steps
instantly by one spacing at the start of each line (at x = cx - A). scan_rotation_deg
turns CLOCKWISE. A scan shorter than 150,000 samples is padded with zeros in
AO_vs_time up to 150,000 (the tip then goes to (0, 0)) -- keep rasters longer than
that (reference_raster does).

Speed: PyScanner's scans are sinusoidal (raster lines, Lissajous) or constant angular
speed (spiral), so the tip speed varies along them. The duration is set so the tip
speed is at most VelocityConfig.tip_velocity_V_s ("peak", the default -- the tip never
goes faster than you set) or equals it on average ("mean").

Checked on hardware (probe_pyscanner_scans, 2026-09-24): the Lissajous starts where
predicted and has duration x rate samples; AO_vs_time_shifted is the output WITH the
offset (the real path). Still to confirm with the probe: the spiral's timing (assumed
constant angular speed) and its "direct return".
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np

SCAN_TYPE = {"spiral": 1, "raster": 2, "lissajous": 3}
CLUSTER = {"spiral": "spiral_scan_control_cluster",
           "raster": "fast_raster_scan_control_cluster",
           "lissajous": "lissajous_scan_control_cluster"}
# cluster field -> index in the cluster tuple (the _NN suffix of each field name)
RASTER_FIELDS = dict(size_x_V=0, size_y_V=1, n_lines=2, duration_s=3, raster_type=4)
LISSAJOUS_FIELDS = dict(size_x_V=0, size_y_V=1, w1_hz=2, w2_hz=3, phi_rad=4, duration_s=5)
# from the spiral tab (field name suffix _NN = index NN)
SPIRAL_FIELDS = dict(inner_radius_x_V=0, outer_radius_x_V=1, inner_radius_y_V=2, outer_radius_y_V=3,
                     n_cycles=4, duration_s=5, direction=7, return_opt=8)
SPIRAL_DIRECTION = 0  # spiral_direction_07: 0 = clockwise spiral out
# spiral_return_opt_08 (checked on hardware 2026-11-02): 0 = out to the radius in ~95% of the duration,
# then a straight line back to the centre at ~5x the speed; 1 = out in the first half, back IN in the
# second (n_cycles each way, the same sense of rotation -- the way back crosses the way out), smooth
# all the way; 2 = not usable (AI_vs_time longer than the scan)
SPIRAL_RETURN = 1
MIN_SCAN_SAMPLES = 150_000   # PyScanner pads shorter scans with zeros up to this (hardware, 2026-09-24)


@dataclass
class PyScannerScan:
    pattern: str                         # "raster" | "lissajous" | "spiral"
    center_xy: Tuple[float, float]       # -> scan_x_offset_V / scan_y_offset_V
    size_x_V: float = 0.0                # raster / Lissajous box (spiral: 2 x radius)
    size_y_V: float = 0.0
    duration_s: float = 1.0
    rotation_deg: float = 0.0            # -> scan_rotation_deg (raster with lines along y: 90)
    n_lines: int = 0                     # raster
    raster_type: int = 2                 # raster: 2 = sinusoidal
    w1_hz: float = 0.0                   # Lissajous
    w2_hz: float = 0.0
    phi_rad: float = 0.0
    radius_V: float = 0.0                # spiral
    n_cycles: float = 0.0
    info: dict = field(default_factory=dict)

    def describe(self) -> str:
        c = f"at ({self.center_xy[0]:.3g}, {self.center_xy[1]:.3g}) V"
        if self.pattern == "raster":
            rot = f", rotated {self.rotation_deg:g} deg" if self.rotation_deg else ""
            return (f"PyScanner raster {self.size_x_V:.3g} x {self.size_y_V:.3g} V {c}, "
                    f"{self.n_lines} lines in {self.duration_s:.3g} s{rot}")
        if self.pattern == "lissajous":
            return (f"PyScanner Lissajous {self.size_x_V:.3g} x {self.size_y_V:.3g} V {c}, "
                    f"w1 = {self.w1_hz:.4g} Hz, w2 = {self.w2_hz:.4g} Hz, {self.duration_s:.3g} s")
        return (f"PyScanner spiral r = {self.radius_V:.3g} V {c}, {self.n_cycles:.1f} cycles "
                f"in {self.duration_s:.3g} s")


# ------------------------------------------------------------------------------------
# "scan this area at this spacing and tip speed" -> PyScanner parameters
# ------------------------------------------------------------------------------------
def _speed_factor(speed_basis: str, peak_over_mean: float) -> float:
    """Duration multiplier: 'peak' -> the fastest point is at v; 'mean' -> average = v."""
    if speed_basis == "peak":
        return 1.0
    if speed_basis == "mean":
        return 1.0 / peak_over_mean
    raise ValueError(f"speed_basis must be 'peak' or 'mean', not {speed_basis!r}")


def plan_raster(center_xy, size_x: float, size_y: float, spacing: float, v: float,
                fast_axis: str = "x", raster_type: int = 2, speed_basis: str = "peak") -> PyScannerScan:
    """Sinusoidal raster: each line is trace + retrace, x = -A cos(2 pi t / T) ->
    peak speed 2 pi A / T (checked on hardware)."""
    if fast_axis == "y":  # PyScanner's lines run along x -- rotate the scan 90 deg
        size_x, size_y, rot = size_y, size_x, 90.0
    else:
        rot = 0.0
    n_lines = max(2, int(round(size_y / max(spacing, 1e-12))) + 1)
    A = 0.5 * size_x
    T_line = 2 * np.pi * A / v * _speed_factor(speed_basis, np.pi / 2)
    return PyScannerScan("raster", tuple(center_xy), size_x, size_y, n_lines * T_line, rot,
                         n_lines=n_lines, raster_type=raster_type,
                         info=dict(spacing=size_y / (n_lines - 1), fast_axis=fast_axis))


def plan_lissajous(center_xy, size_x: float, size_y: float, spacing: float, v: float,
                   phi_rad: Optional[float] = None, speed_basis: str = "peak") -> PyScannerScan:
    """a x-cycles and b = a + 1 y-cycles over the duration (a closed curve), a chosen
    like trajectory.lissajous_scan (same coverage as a raster `spacing` apart).
    Peak speed <= 2 pi hypot(a A, b B) / duration."""
    A, B = 0.5 * size_x, 0.5 * size_y
    a = max(1, int(np.ceil(0.75 * max(size_x, size_y) / max(spacing, 1e-12))))
    b = a + 1
    peak = 2 * np.pi * np.hypot(a * A, b * B)
    # mean speed of a Lissajous ~ 2/pi of the peak bound (numerically ~0.64-0.7)
    D = peak / v * _speed_factor(speed_basis, np.pi / 2)
    return PyScannerScan("lissajous", tuple(center_xy), size_x, size_y, D,
                         w1_hz=a / D, w2_hz=b / D, phi_rad=0.0 if phi_rad is None else phi_rad,
                         info=dict(a=a, b=b, spacing=spacing))


def plan_spiral(center_xy, radius: float, spacing: float, v: float,
                speed_basis: str = "peak") -> PyScannerScan:
    """Spiral out to `radius`, turns `spacing` apart. Assumed constant angular speed
    (CONFIRM ON HARDWARE) -> fastest at the outer edge: 2 pi R n_cycles / duration."""
    n_cycles = max(int(round(radius / max(spacing, 1e-12))), 1)  # PyScanner's N cycles: whole turns
    D = 2 * np.pi * radius * n_cycles / v * _speed_factor(speed_basis, 2.0)
    if SPIRAL_RETURN == 1:
        D *= 2.0      # out AND back in, at the same speed
    return PyScannerScan("spiral", tuple(center_xy), 2 * radius, 2 * radius, D,
                         radius_V=radius, n_cycles=n_cycles, info=dict(spacing=spacing))


# ------------------------------------------------------------------------------------
# what PyScanner is expected to run (simulation, safety check, start point)
# ------------------------------------------------------------------------------------
def predict_path(scan: PyScannerScan, sample_rate_hz: float):
    """(x_vec, y_vec) we expect PyScanner to output for `scan` -- our model of its
    scan types (CONFIRM ON HARDWARE with probe_pyscanner_scans): offset + rotation
    applied, one sample per 1/sample_rate_hz."""
    n = max(4, int(round(scan.duration_s * sample_rate_hz)))
    t = np.arange(n) / sample_rate_hz
    if scan.pattern == "raster":
        T = scan.duration_s / scan.n_lines
        u = t / T                                   # line coordinate 0..n_lines
        x = -0.5 * scan.size_x_V * np.cos(2 * np.pi * u)   # trace + retrace per line
        y = -0.5 * scan.size_y_V + scan.size_y_V / max(scan.n_lines - 1, 1) * np.minimum(np.floor(u), scan.n_lines - 1)
    elif scan.pattern == "lissajous":
        x = 0.5 * scan.size_x_V * np.sin(2 * np.pi * scan.w1_hz * t + scan.phi_rad)
        y = 0.5 * scan.size_y_V * np.sin(2 * np.pi * scan.w2_hz * t)
    elif scan.pattern == "spiral":  # clockwise out from the centre (spiral_direction 0)
        f = t / scan.duration_s
        if SPIRAL_RETURN == 1:       # out in the first half, back in in the second (hardware, 2026-11-02)
            r = scan.radius_V * (1.0 - np.abs(1.0 - 2.0 * f))
            th = 0.5 * np.pi - 2 * np.pi * 2 * scan.n_cycles * f
        else:
            r = scan.radius_V * f
            th = 0.5 * np.pi - 2 * np.pi * scan.n_cycles * f   # starts a quarter turn on (hardware, 2026-11-02)
        x, y = r * np.cos(th), r * np.sin(th)
    else:
        raise ValueError(f"Unknown pattern {scan.pattern!r}")
    # PyScanner's scan_rotation_deg turns clockwise (checked on hardware)
    c, s = np.cos(np.deg2rad(-scan.rotation_deg)), np.sin(np.deg2rad(-scan.rotation_deg))
    xr, yr = c * x - s * y, s * x + c * y
    return scan.center_xy[0] + xr, scan.center_xy[1] + yr


def fit_fast_raster_delay(scan: PyScannerScan, z, sample_rate_hz: float, guess_s: float,
                          max_fit_samples: int = 100_000) -> dict:
    """The time shift of a fast raster's Z record against its path: Z sample i belongs to
    path sample i + delay_s * rate (on the instrument, 2026-10-06: +0.127 s). Fitted from
    trace and retrace agreement (reference_raster._mismatch: every line's two halves
    binned along the line, mean |trace - retrace|), searched over guess_s +- half a line.
    Shifts half a line apart fit about equally well (one mirrors the image along the
    fast axis): the one nearest guess_s is kept. When no shift clearly beats none (a flat
    area, too few samples, or the shift is a multiple of half a line), guess_s is used.
    Returns dict(delay_s, fitted, note)."""
    from .reference_raster import _mismatch
    z = np.asarray(z, dtype=float)
    T = scan.duration_s / max(scan.n_lines, 1)
    if scan.raster_type == 1:   # sawtooth: the flyback isn't a mirror of the trace
        return dict(delay_s=float(guess_s), fitted=False,
                    note=f"sawtooth raster: not fitted, shifted by {guess_s * 1e3:+.1f} ms")
    i = np.arange(len(z))
    if len(z) > max_fit_samples:                     # plenty for the fit
        i = np.linspace(0, len(z) - 1, max_fit_samples).astype(int)
    t, zz = i / sample_rate_hz, z[i]
    g = dict(coords=np.zeros(int(scan.n_lines)), T=T)   # all _mismatch needs
    base = _mismatch(t, zz, g, 0.0)
    # coarse search (on a quarter of the samples): the dip at the right shift is narrow
    # (a few ms at a fast tip), so the grid has to be fine -- 1/400 line
    q = slice(None, None, 4)
    coarse = guess_s + np.arange(-0.5, 0.5, 0.0025) * T
    sc = np.array([_mismatch(t[q], zz[q], g, tau) for tau in coarse])
    if not np.isfinite(sc).any():
        return dict(delay_s=float(guess_s), fitted=False,
                    note=f"too few samples to fit; shifted by {guess_s * 1e3:+.1f} ms")
    fine = coarse[np.nanargmin(sc)] + np.arange(-1, 1.0001, 0.05) * 0.0025 * T
    scf = np.array([_mismatch(t, zz, g, tau) for tau in fine])
    best = float(fine[np.nanargmin(scf)])
    after = float(np.nanmin(scf))
    if not (np.isfinite(base) and after < 0.8 * base):
        return dict(delay_s=float(guess_s), fitted=False,
                    note=(f"no clear shift (trace vs retrace differ by {base:.3g} V unshifted, {after:.3g} V "
                          f"at best); shifted by {guess_s * 1e3:+.1f} ms"))
    # the alias (best + k half lines) nearest the estimate
    k = np.round((guess_s - best) / (0.5 * T))
    tau = float(best + k * 0.5 * T)
    return dict(delay_s=tau, fitted=True,
                note=(f"shifted by {tau * 1e3:+.1f} ms ({tau / T:+.3f} line): trace vs retrace differ by "
                      f"{base:.3g} -> {_mismatch(t, zz, g, tau):.3g} V"))


def shift_by_delay(n: int, delay_samples: int):
    """(path slice, Z slice) pairing Z sample i with path sample i + delay_samples; the
    samples left without a partner at the ends are dropped."""
    d = int(delay_samples)
    d = max(-(n - 2), min(d, n - 2))
    if d >= 0:
        return slice(d, n), slice(0, n - d)
    return slice(0, n + d), slice(-d, n)


def extent(scan: PyScannerScan) -> float:
    """Largest |x| or |y| the scan can reach (for the safety limit)."""
    if scan.rotation_deg % 90:
        half = 0.5 * np.hypot(scan.size_x_V, scan.size_y_V)
        return max(abs(scan.center_xy[0]), abs(scan.center_xy[1])) + half
    hx, hy = 0.5 * scan.size_x_V, 0.5 * scan.size_y_V
    if round(scan.rotation_deg / 90) % 2:   # 90 / 270 deg: the box's sides swap
        hx, hy = hy, hx
    return max(abs(scan.center_xy[0]) + hx, abs(scan.center_xy[1]) + hy)


def within(scan: PyScannerScan, safety) -> bool:
    """The scan stays inside the safety limits (per axis: config.amplitude_limits)."""
    from .config import amplitude_limits
    lx, ly = amplitude_limits(safety)
    if scan.rotation_deg % 90:
        half = 0.5 * np.hypot(scan.size_x_V, scan.size_y_V)
        return abs(scan.center_xy[0]) + half <= lx and abs(scan.center_xy[1]) + half <= ly
    hx, hy = 0.5 * scan.size_x_V, 0.5 * scan.size_y_V
    if round(scan.rotation_deg / 90) % 2:
        hx, hy = hy, hx
    return abs(scan.center_xy[0]) + hx <= lx and abs(scan.center_xy[1]) + hy <= ly


def cluster_values(scan: PyScannerScan) -> dict:
    """{cluster index: value} to write into the scan type's control cluster."""
    if scan.pattern == "raster":
        f = RASTER_FIELDS
        return {f["size_x_V"]: scan.size_x_V, f["size_y_V"]: scan.size_y_V, f["n_lines"]: int(scan.n_lines),
                f["duration_s"]: scan.duration_s, f["raster_type"]: int(scan.raster_type)}
    if scan.pattern == "lissajous":
        f = LISSAJOUS_FIELDS
        # PyScanner's Lissajous "size" = amplitude (half the width) -- see module docstring
        return {f["size_x_V"]: 0.5 * scan.size_x_V, f["size_y_V"]: 0.5 * scan.size_y_V, f["w1_hz"]: scan.w1_hz,
                f["w2_hz"]: scan.w2_hz, f["phi_rad"]: scan.phi_rad, f["duration_s"]: scan.duration_s}
    f = SPIRAL_FIELDS
    return {f["inner_radius_x_V"]: 0.0, f["outer_radius_x_V"]: scan.radius_V,
            f["inner_radius_y_V"]: 0.0, f["outer_radius_y_V"]: scan.radius_V,
            f["n_cycles"]: int(scan.n_cycles), f["duration_s"]: scan.duration_s,
            f["direction"]: SPIRAL_DIRECTION, f["return_opt"]: SPIRAL_RETURN}


# ------------------------------------------------------------------------------------
# one-off hardware check (Run notebook, "Check PyScanner's own scan types")
# ------------------------------------------------------------------------------------
def probe_pyscanner_scans(executor, center_xy=(0.1, 0.05), size_V: float = 0.2, lines: int = 10,
                          save_dir="scan_data"):
    """Run small PyScanner-defined scans around center_xy (a size_V square, `lines`
    lines apart-equivalent, at the executor's tip speed) and report, for each: samples
    vs duration x rate, predicted vs actual start point, whether AO_vs_time includes
    the offset, and wall time (per-scan overhead). Then checks that a user-defined move
    afterwards is NOT shifted by the offset (it's reset after every PyScanner scan).
    Also prints the current spiral / raster / Lissajous clusters. Everything printed,
    plus every scan's full path (as run, and as predicted), is saved to
    save_dir/pyscanner_probe_<time>.npz and .txt for analysis. Returns the reports."""
    import contextlib
    import io
    import pathlib
    import sys
    import time
    log = io.StringIO()
    out = sys.stdout

    class _Tee(io.TextIOBase):
        def write(self, text):
            log.write(text)
            return out.write(text)

    with contextlib.redirect_stdout(_Tee()):
        reports, paths = _probe(executor, center_xy, size_V, lines)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = pathlib.Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"pyscanner_probe_{stamp}.npz", sample_rate_hz=executor.velocity.sample_rate_hz,
             tip_velocity_V_s=executor.velocity.tip_velocity_V_s, **paths)
    (d / f"pyscanner_probe_{stamp}.txt").write_text(log.getvalue(), encoding="utf-8")
    print(f"Saved {d / f'pyscanner_probe_{stamp}'}.npz and .txt")
    return reports


def _probe(executor, center_xy, size_V, lines):
    import time
    be = executor.backend
    v = executor.velocity.tip_velocity_V_s
    rate = executor.velocity.sample_rate_hz
    spacing = size_V / lines
    vi = getattr(be, "vi", None)
    if vi is not None:
        for p, nm in CLUSTER.items():
            try:
                print(f"{nm} (now): {tuple(vi.getcontrolvalue(nm))}")
            except Exception as e:  # noqa: BLE001
                print(f"{nm}: could not read ({e})")
    max_n = executor.safety.max_samples_per_waveform

    def fit(make):
        """fewer lines until the scan fits in one PyScanner waveform (and <= 2 s)"""
        n_l = lines
        s = make(size_V / n_l)
        while n_l > 2 and (s.duration_s * rate > max_n or s.duration_s > 2.0):
            n_l -= 1
            s = make(size_V / n_l)
        return s
    scans = [fit(lambda sp: plan_raster(center_xy, size_V, size_V, sp, v, "x")),
             fit(lambda sp: plan_raster(center_xy, size_V, 0.5 * size_V, sp, v, "y")),   # rotated 90 deg
             fit(lambda sp: plan_lissajous(center_xy, size_V, size_V, sp, v))]
    scans.append(fit(lambda sp: plan_spiral(center_xy, 0.5 * size_V, sp, v)))
    reports, paths = [], {}
    for i_s, s in enumerate(scans):
        t0 = time.perf_counter()
        x, y, z = executor.send_pyscanner(s, "grid_sparse", correct_delay=False,   # the raw record, as run
                                          meta={"label": f"probe: {s.describe()}"})
        wall = time.perf_counter() - t0
        step_v = np.hypot(np.diff(x), np.diff(y)) * rate
        rep = dict(getattr(be, "last_pyscanner_report", {}) or {}, wall_time_s=wall, duration_s=s.duration_s,
                   z_samples=len(z), tip_speed_max_V_s=float(step_v.max()),
                   tip_speed_mean_V_s=float(step_v.mean()), tip_speed_set_V_s=v)
        if vi is not None:
            try:  # does PyScanner report the offset-applied output here?
                sh = np.asarray(vi.getcontrolvalue('AO_vs_time_shifted'), dtype=float)
                rep["AO_vs_time_shifted"] = (f"shape {sh.shape}, x {sh[0].min():.3f}..{sh[0].max():.3f}, "
                                             f"y {sh[1].min():.3f}..{sh[1].max():.3f}, first ({sh[0, 0]:.3f}, "
                                             f"{sh[1, 0]:.3f}), last ({sh[0, -1]:.3f}, {sh[1, -1]:.3f})")
            except Exception as e:  # noqa: BLE001
                rep["AO_vs_time_shifted"] = f"could not read ({e})"
        reports.append(rep)
        px, py = predict_path(s, rate)
        k = 10  # saved every 10th sample (10 us at 1 MS/s), float32 -- keeps the file ~20 MB
        paths.update({f"scan{i_s}_x": x[::k].astype(np.float32), f"scan{i_s}_y": y[::k].astype(np.float32),
                      f"scan{i_s}_z": z[::k].astype(np.float32),
                      f"scan{i_s}_pred_x": px[::k].astype(np.float32), f"scan{i_s}_pred_y": py[::k].astype(np.float32),
                      f"scan{i_s}_desc": np.array(s.describe()), "saved_every_n_samples": k})
        print(f"\n{s.describe()}")
        for k, val in rep.items():
            if k != "scan":
                print(f"   {k}: {val}")
    # a short user-defined line: is it where we asked (offset reset)?
    t0 = time.perf_counter()
    p0 = executor.last_pos.copy()
    xs = np.linspace(p0[0], p0[0] + 0.02, max(4, int(0.02 / executor.velocity.sample_spacing_V)))
    executor.send(xs, np.full_like(xs, p0[1]), kind="connector", meta={"label": "probe: user-defined line"})
    if vi is not None:
        ao = np.asarray(vi.getcontrolvalue('AO_vs_time'), dtype=float)
        n = min(len(xs), ao.shape[1])
        err = float(np.max(np.hypot(ao[0, :n] - xs[:n], ao[1, :n] - p0[1])))
        print(f"\nuser-defined line after the PyScanner scans: AO_vs_time differs from what was sent by at most "
              f"{err:.4f} V ({'OK -- offsets are reset' if err < 1e-3 else 'SHIFTED -- tell me this number'}); "
              f"{time.perf_counter() - t0:.2f} s wall for {len(xs) / rate:.3f} s of motion")
    return reports, paths
