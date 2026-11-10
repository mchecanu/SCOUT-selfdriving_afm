"""
Single choke point through which every planner sends a trajectory to a backend.

Three things live here, deliberately in one place instead of duplicated per planner:

1. **Continuity.** Per your spec: scans always start at (0, 0) in bias-volt XY space,
   and are continuous -- wherever one waveform finishes, the next one starts. Planners
   compute trajectory primitives (a spiral, a circle, a line segment) independently,
   and those primitives don't always share endpoints (e.g. a tracing circle of radius
   R is centered ON a detected edge point, so its first sample is R away from that
   edge point, not AT it). Rather than special-casing every transition inside every
   planner, `ContinuousExecutor.send()` checks the gap before every primitive and, if
   it's nonzero, automatically inserts a straight connecting line first -- so the
   waveform actually sent to hardware is always one unbroken curve. This also means
   `EdgeFollowPlanner`'s old "step off the feature" nudge (see README) is now a real,
   measured motion instead of a bookkeeping-only position jump.
2. **Live view hook.** Every send (including auto-inserted connectors) calls
   `on_step(step_index, step, executor)` after the measurement returns, which is what
   `viz.LiveEdgeFollowView` / `viz.LiveGPPathView` hang off of. Since this is the only
   path through which planners talk to hardware, both scan modalities get live
   updating for free with no per-planner plumbing.
3. **Safety.** `trajectory.clamp_to_safety` is applied here, once, to every waveform
   (including connectors) -- planners never call it directly any more.
4. **Position-sensor readback (optional).** If `PositionSensorConfig.enabled`, every
   step also calls `backend.read_position_sensors(x_vec, y_vec)` right after
   `measure_path`, storing the result as `TrajectoryStep.x_sensor`/`y_sensor`.
   `all_points()` then prefers sensor position over commanded whenever it's
   present, so saved/reconstructed data is corrected for open-loop discrepancy
   automatically -- with the option off (the default), `x_sensor`/`y_sensor` are
   just `None` everywhere and behavior is exactly what it was before this existed.
   This is scoped to DATA correction only: the planners' own live decisions
   (where to center the next circle, which direction the GP explores) still use
   the commanded position they asked for, not sensor position -- feeding sensor
   noise/lag back into that live loop is a bigger change with its own stability
   questions, not something this adds implicitly.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from . import trajectory
from .config import SafetyConfig, VelocityConfig, PositionSensorConfig, ReadbackConfig, ZSignalConfig
from .backends.base import ScanBackend

OnStep = Callable[[int, "TrajectoryStep", "ContinuousExecutor"], None]
# called right BEFORE a waveform is sent: (kind, meta, executor) -- meta["label"] is a
# human-readable description of the scan about to run (viz uses it for the status line)
OnBeforeStep = Callable[[str, dict, "ContinuousExecutor"], None]


@dataclass
class TrajectoryStep:
    kind: str  # "spiral" | "circle" | "line" | "connector"
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    meta: dict = field(default_factory=dict)
    x_sensor: Optional[np.ndarray] = None  # set only if PositionSensorConfig.enabled -- the measured
    y_sensor: Optional[np.ndarray] = None  # position in the drive's frame (V; sensors.SensorCalibration)
    x_sensor_raw: Optional[np.ndarray] = None  # the AI1 / AI2 volts it came from
    y_sensor_raw: Optional[np.ndarray] = None
    channels: Optional[dict] = None            # extra signals from the same AI record ({name: array},
                                               #   in their units -- executor.extra_channels; PFM)


def _line_including_endpoint(p0: np.ndarray, p1: np.ndarray, resolution: float):
    """Straight line from p0 to p1 inclusive of BOTH endpoints (unlike
    trajectory.line_segment, which is built for the GP planner's convention of
    excluding the far end so consecutive segments don't duplicate a sample).
    A connector's whole job is to close an exact gap, so its last sample must land
    exactly on p1 -- not "close enough" after an integer number of resolution-sized
    steps.
    """
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-12:
        return np.array([p0[0]]), np.array([p0[1]])
    n_points = max(int(round(length / resolution)) + 1, 2)
    trajectory._check_point_count(n_points, length, resolution, "connector")
    t = np.linspace(0.0, 1.0, n_points)
    x_vec = p0[0] + t * (p1[0] - p0[0])
    y_vec = p0[1] + t * (p1[1] - p0[1])
    return x_vec, y_vec


def z_units(z_signal: ZSignalConfig, info: Optional[dict]):
    """(scale, unit, why): measured Z x scale is in `unit` ("V" or "nm") -- cfg.z_signal.units
    and the setup's Z routing (asylum.AsylumControl.route_z -> info); `why` says why Z
    stays in volts when nm was asked for. The sign flip (z_signal.invert) is in the scale."""
    if info and "unit" in info and "scale" in info:     # stated outright (PFM: signal units per AI volt)
        return float(info["scale"]), str(info["unit"]), ""
    scale, unit, why = 1.0, "V", ""
    if z_signal.units == "nm":
        m = (info or {}).get("m_per_V")
        if not info:
            why = "the Z signal wasn't routed in this session (cfg.z_signal.route + the setup cell)"
        elif not (m is not None and np.isfinite(m) and m > 0):
            why = f"no nm factor for {info.get('source')} (set cfg.z_signal.m_per_V, in m/V)"
        elif info.get("ai_matches") is False:
            why = f"{info.get('ai_channel')} didn't read the {info.get('source')} copy in the setup check"
        else:
            scale, unit = float(m) * float(info.get("divider", 1.0)) * 1e9, "nm"
    # cfg.z_signal.invert: the routed signal (ZSensor) reads opposite to the height
    if info and z_signal.invert:
        scale = -scale
    return scale, unit, why


class ContinuousExecutor:
    """Wraps a ScanBackend so planners never call it (or clamp_to_safety) directly.

    `start_xy` is where the executor believes the tip currently sits -- pass the
    same `start_xy` the planner's `run()` was given (default (0, 0), per spec) so the
    very first primitive is checked against it too.
    """

    def __init__(self, backend: ScanBackend, safety: SafetyConfig,
                 velocity: VelocityConfig, start_xy=(0.0, 0.0),
                 on_step: Optional[OnStep] = None, position_tol_V: float = 1e-9,
                 position_sensors: Optional[PositionSensorConfig] = None,
                 on_before_step: Optional[OnBeforeStep] = None,
                 readback: Optional[ReadbackConfig] = None,
                 z_signal: Optional[ZSignalConfig] = None):
        self.backend = backend
        # Z units (cfg.z_signal.units): every measured z is multiplied by z_scale as it comes in
        # (1, or nm per AI volt), so planners, history and plots all see z_unit. The factor comes
        # from the setup's Z routing (asylum.AsylumControl.route_z -> z_signal_info, also left on
        # the backend so every executor on it -- the reference raster's too -- gets it)
        self.z_signal = z_signal or ZSignalConfig()
        self.z_signal_info = getattr(backend, "z_signal_info", None)
        self.z_scale = 1.0
        self.z_unit = "V"
        self.z_label = None              # e.g. "PFM phase (deg)"; None = "AI0 (V)" / "Z (nm)"
        self._z_units_said = None
        # extra signals read from the same AI record as Z, every waveform: {name: (ai_channel,
        # signal units per AI volt)} -- empty for topography (PFM: amplitude / phase)
        self.extra_channels = {}
        self.last_channels = None        # the last send()'s, joined like its z
        self.readback = readback or ReadbackConfig()
        # fast rasters: the current estimate of the Z record's time shift (s) -- the
        # backend's own (simulation: 0) or readback.fast_raster_delay_s, then each clear fit
        self.fast_raster_delay_s = float(getattr(backend, "fast_raster_delay_guess_s",
                                                 self.readback.fast_raster_delay_s))
        self.on_before_step = on_before_step
        self.safety = safety
        self.velocity = velocity
        self.on_step = on_step
        self.position_tol_V = position_tol_V
        self.position_sensors = position_sensors or PositionSensorConfig()
        self.last_pos = np.array(start_xy, dtype=float)
        self.history: List[TrajectoryStep] = []
        self._last_full_step: Optional[TrajectoryStep] = None   # the last step with every sample
        self.step_index = 0
        # tip speed = tip_velocity_V_s only if sample_rate_hz is the hardware's real output
        # rate -- backends that can check it do so here (PyScannerBackend.check_sample_rate)
        check = getattr(backend, "check_sample_rate", None)
        if callable(check):
            check(velocity.sample_rate_hz)
        # > 0: connectors start and end at rest, speeding up/slowing down smoothly over
        # this many samples (trajectory.ease_ends) -- set by the grid+refine planner
        self.connector_ease_samples = 0
        # > 0: every send() ends with the tip held at rest at the path's end for this long
        # (2026-11-09). PyScanner's AI record can come back short at the END (35k-154k
        # samples seen = 35-155 ms at 1 MS/s): the dwell absorbs that loss, so the path
        # itself is recorded to its last sample. The dwell's samples are measured (in
        # history) but not returned.
        self.tail_dwell_s = 0.0
        # True: if the last piece still comes back short, the rest of the path is re-measured
        # (from where the record stopped). user_raster turns this off: it re-scans whole
        # lines itself, so a raster is never cut in the middle of a line.
        self.remeasure_short_tail = True
        # position sensors: AI volts -> the measured position in drive volts
        # (sensors.SensorCalibration, measured in the setup -- asylum.AsylumControl). Until
        # it's set the sensors are recorded (raw) but not used for positions
        self.sensor_cal = getattr(backend, "sensor_cal", None)   # the setup's (asylum.calibrate_sensors)
        self._warned_no_cal = False
        # drive correction (cfg.position_sensors.correct_linear / correct_offset): planners give
        # TARGET positions (where the tip should be, in the calibrated frame); the drive sent is
        # A^-1 (target - u). u is eased from u_applied (the last waveform's) to u_target over a
        # waveform's start, so the drive never jumps. last_pos is a target position.
        self._u_applied = np.zeros(2)
        self._u_target = np.zeros(2)
        self.correction_log: List[tuple] = []   # (step index, u_target x, y, measured error x, y)
        # the measured positions (or, without sensors, the targets) of the last send(): what
        # the planners pair the returned z with -- measured_xy()
        self.last_xy = None

    # ------------------------------------------------------------------
    # drive correction (a: the calibration's drive map; b: the measured offset)
    # ------------------------------------------------------------------
    def _cal_ok(self) -> bool:
        return self.sensor_cal is not None and getattr(self.sensor_cal, "ok", True)

    def _map(self):
        """(A, A^-1) of the drive map used for the correction -- identity without (a)."""
        ps = self.position_sensors
        if self._cal_ok() and getattr(ps, "correct_linear", False):
            A = np.asarray(self.sensor_cal.A, dtype=float)
            if np.all(np.isfinite(A)) and abs(np.linalg.det(A)) > 1e-6:
                return A, np.linalg.inv(A)
        return np.eye(2), np.eye(2)

    @property
    def correcting(self) -> bool:
        ps = self.position_sensors
        return self._cal_ok() and (getattr(ps, "correct_linear", False) or getattr(ps, "correct_offset", False))

    def to_drive(self, x_t, y_t, ease: bool = True):
        """Target positions -> the drive to send: A^-1 (target - u), u eased from the last
        waveform's value to the new one over the first correction_ramp_s (no jump)."""
        x_t, y_t = np.asarray(x_t, dtype=float), np.asarray(y_t, dtype=float)
        if not self.correcting:
            return x_t, y_t
        _, Ai = self._map()
        n = len(x_t)
        u0, u1 = self._u_applied, self._u_target
        if ease and n > 1 and np.any(np.abs(u1 - u0) > 1e-12):
            m = max(2, min(n, int(round(self.position_sensors.correction_ramp_s * self.velocity.sample_rate_hz))))
            w = np.ones(n)
            w[:m] = 0.5 - 0.5 * np.cos(np.pi * np.arange(m) / (m - 1))   # 0 -> 1, smooth
            ux, uy = u0[0] + (u1[0] - u0[0]) * w, u0[1] + (u1[1] - u0[1]) * w
        else:
            ux, uy = np.full(n, u1[0]), np.full(n, u1[1])
        dx, dy = x_t - ux, y_t - uy
        self._u_applied = u1.copy()
        return Ai[0, 0] * dx + Ai[0, 1] * dy, Ai[1, 0] * dx + Ai[1, 1] * dy

    def to_target(self, x_d, y_d):
        """Drive sent -> the target it stood for (A drive + u, the current u)."""
        x_d, y_d = np.asarray(x_d, dtype=float), np.asarray(y_d, dtype=float)
        if not self.correcting:
            return x_d, y_d
        A, _ = self._map()
        u = self._u_applied
        return A[0, 0] * x_d + A[0, 1] * y_d + u[0], A[1, 0] * x_d + A[1, 1] * y_d + u[1]

    def _update_correction(self, step) -> None:
        """(b): the measured error at the end of this waveform -> the next waveforms' offset."""
        ps = self.position_sensors
        if not (self._cal_ok() and getattr(ps, "correct_offset", False)) or step.x_sensor is None:
            return
        n = len(step.x_sensor)
        if n < 2:
            return
        k = max(1, min(n, int(round(ps.correction_window_s * self.velocity.sample_rate_hz))))
        A, _ = self._map()
        d = np.column_stack([step.x[-k:], step.y[-k:]])
        m = np.column_stack([step.x_sensor[-k:], step.y_sensor[-k:]])
        u_obs = np.nanmean(m - d @ A.T, axis=0)          # where it went minus A drive = c + error
        if not np.all(np.isfinite(u_obs)):
            return
        u_new = self._u_target + ps.correction_gain * (u_obs - self._u_target)
        lim = ps.correction_max_V
        r = float(np.hypot(*u_new))
        if r > lim:
            u_new = u_new * lim / r
        self._u_target = u_new
        self.correction_log.append((self.step_index, float(u_new[0]), float(u_new[1]),
                                    float(u_obs[0]), float(u_obs[1])))

    def measured_xy(self, x_vec, y_vec, z_vec):
        """The positions to pair the last send()'s z with: the MEASURED ones (sensors) when
        cfg.position_sensors.plan_on_sensors and a good calibration, else the targets sent."""
        n = len(z_vec)
        if self.last_xy is not None and len(self.last_xy[0]) == n:
            return self.last_xy
        return np.asarray(x_vec)[:n], np.asarray(y_vec)[:n]

    def _sensor_positions(self, raw_x, raw_y):
        """(x_sensor, y_sensor) in the drive frame from the AI volts -- None without a calibration."""
        if raw_x is not None and self.sensor_cal is not None and not self._cal_ok():
            if not self._warned_no_cal:
                print("[position sensors] recorded, but not used: the calibration didn't pass its checks "
                      "(rerun the setup cell)")
                self._warned_no_cal = True
            return None, None
        if raw_x is None or self.sensor_cal is None:
            if raw_x is not None and not self._warned_no_cal:
                print("[position sensors] recorded, but not used for positions: no calibration "
                      "(executor.sensor_cal -- the setup cell measures it)")
                self._warned_no_cal = True
            return None, None
        from .sensors import smooth
        n = getattr(self.position_sensors, "smooth_samples", 0)
        return self.sensor_cal.to_frame(smooth(raw_x, n), smooth(raw_y, n))

    def reset(self, start_xy=(0.0, 0.0)):
        """Called at the top of a planner's run() so re-running reuses the executor
        (and its live-view callback) starting fresh from start_xy."""
        self.last_pos = np.array(start_xy, dtype=float)
        self.history.clear()
        self.step_index = 0
        # the tip is assumed AT start_xy (the drive sent = start_xy): u such that A^-1 (start - u)
        # = start, i.e. no correction yet -- it's measured from the first waveform on
        A, _ = self._map()
        s0 = np.asarray(start_xy, dtype=float)
        self._u_applied = s0 - A @ s0
        self._u_target = self._u_applied.copy()
        u0 = getattr(self.position_sensors, "initial_offset_V", None)
        if u0 is not None and self._cal_ok() and getattr(self.position_sensors, "correct_offset", False):
            # start from a known offset: eased in (to_drive) over the first waveform's start
            self._u_target = self._u_applied + np.asarray(u0, dtype=float)
        self.correction_log = []
        self.last_xy = None
        self.set_z_units()

    def set_z_units(self) -> None:
        """z_scale / z_unit from cfg.z_signal.units and the setup's Z routing (printed when
        they change)."""
        info = self.z_signal_info or getattr(self.backend, "z_signal_info", None)
        self.z_signal_info = info
        scale, unit, why = z_units(self.z_signal, info)
        self.z_label = (info or {}).get("label") if (info and "unit" in info) else None
        if self.z_label:
            self.z_label = f"{self.z_label} ({unit})"
        said = (scale, unit, why)
        if said != self._z_units_said:
            flip = " -- sign inverted (cfg.z_signal.invert)" if scale < 0 else ""
            if unit == "nm":
                print(f"[Z] heights in nm: {abs(scale):.5g} nm per {info.get('ai_channel', 'AI')} volt "
                      f"({info.get('source')}) -- thresholds in cfg are in nm too{flip}")
            elif why:
                print(f"[Z] cfg.z_signal.units = 'nm' but Z stays in volts: {why}{flip}")
            elif flip:
                print(f"[Z] {info.get('source')} on {info.get('ai_channel', 'AI')}, in volts{flip}")
            self._z_units_said = said
        self.z_scale, self.z_unit = scale, unit

    # ------------------------------------------------------------------
    def send(self, x_vec, y_vec, kind: str, meta: Optional[dict] = None) -> np.ndarray:
        """Execute a planner-computed primitive (spiral / circle / line), inserting
        a connecting line first if it doesn't start where the tip currently is.
        Returns the primitive's own z_vec (the connector's z, if any, is recorded in
        history but not returned -- callers only want the primitive's measurement).

        Waveforms longer than SafetyConfig.max_samples_per_waveform are sent to the
        backend in consecutive pieces (each its own history step), and the returned
        z is the pieces' readback joined back together. If the backend records fewer
        samples than a piece had (PyScanner's AI_vs_time can come back short at the
        end), the next piece starts again from the first unrecorded sample, so no
        stretch of the path goes unmeasured. z can still be shorter than x_vec at the
        very end (last piece) -- z[i] always pairs with x_vec[i].
        """
        x_vec = np.asarray(x_vec, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float)
        meta = meta or {}
        n = len(x_vec)
        if n == 0:
            return np.array([])
        max_n = max(2, int(self.safety.max_samples_per_waveform))
        n_pieces = int(np.ceil(n / max_n))
        z_parts = []
        xy_parts = []
        ch_parts = []
        self.last_xy = None
        self.last_channels = None
        i = 0
        piece = 0
        retries = 0
        # a cut inside a motion is never a sudden stop / start: each piece slows down to rest
        # along the path after its end (tail) and the next speeds up onto it from rest
        # (head), over the connector ease time (at least 20 ms) -- both measured, not returned
        m_ease = max(int(self.connector_ease_samples), int(round(0.02 * self.velocity.sample_rate_hz)))
        while i < n:
            piece += 1
            j = min(i + max_n, n)
            piece_meta = meta
            if n_pieces > 1 or piece > 1:
                piece_meta = dict(meta, piece=piece,
                                  label=f"{meta.get('label', kind)} [part {piece}, samples {i:,}-{j - 1:,} of {n:,}]")
            hx, hy = trajectory.ramp_into(x_vec, y_vec, i, m_ease) if i > 0 else (np.empty(0), np.empty(0))
            tx, ty = trajectory.ramp_out_of(x_vec, y_vec, j - 1, m_ease) if j < n else (np.empty(0), np.empty(0))
            n_dw = int(round(float(self.tail_dwell_s or 0.0) * self.velocity.sample_rate_hz))
            if n_dw > 0:          # hold at rest where this piece stops (absorbs the AI tail loss)
                ex_, ey_ = (tx[-1], ty[-1]) if len(tx) else (x_vec[j - 1], y_vec[j - 1])
                tx, ty = np.concatenate([tx, np.full(n_dw, ex_)]), np.concatenate([ty, np.full(n_dw, ey_)])
            nh = len(hx)
            px = np.concatenate([hx, x_vec[i:j], tx])
            py = np.concatenate([hy, y_vec[i:j], ty])
            start = np.array([px[0], py[0]])
            # a gap of up to one sample step is just the next sample of a continuous
            # path -- not a move
            gap_tol = max(self.position_tol_V, 1.5 * self.velocity.sample_spacing_V)
            if np.linalg.norm(start - self.last_pos) > gap_tol:
                self._send_connector(start, meta={"label": f"Moving to the start of: "
                                                           f"{piece_meta.get('label', kind)}"})
            zp = self._measure(px, py, kind, piece_meta)
            got = int(np.clip(len(zp) - nh, 0, j - i))     # path samples i.. recorded
            if got == 0 and j < n:
                raise RuntimeError(f"piece {piece}: nothing of the path recorded ({len(zp)} samples back)")
            z_parts.append(zp[nh:nh + got])
            full = self._last_full_step        # every sample (history may keep block means)
            sl = slice(nh, nh + got)
            ch_parts.append({k: v[sl] for k, v in full.channels.items()} if full.channels else full.channels)
            fx = full.x_sensor[sl] if full.x_sensor is not None else None
            fy = full.y_sensor[sl] if full.y_sensor is not None else None
            core = TrajectoryStep(kind, full.x[sl], full.y[sl], full.z[sl], piece_meta, fx, fy)
            xy_parts.append(self._planning_xy(core, x_vec[i:i + got], y_vec[i:i + got]))
            if j >= n and got >= j - i:
                break
            if got < j - i and j >= n:
                if not self.remeasure_short_tail:
                    break
                retries += 1
                if got == 0 or retries > 3:
                    print(f"[executor] '{meta.get('label', kind)}': the record stops at sample {i + got:,} of "
                          f"{n:,} and re-measuring didn't help -- keeping what was recorded")
                    break
                print(f"[executor] the last piece of '{meta.get('label', kind)}' was recorded to sample "
                      f"{i + got:,} of {n:,} -- re-measuring the rest (raise executor.tail_dwell_s)")
            i += got  # re-measure whatever this piece didn't record, as part of the next
        self.last_xy = (np.concatenate([a for a, _ in xy_parts]), np.concatenate([b for _, b in xy_parts]))
        if ch_parts and all(c is not None for c in ch_parts):
            self.last_channels = {k: np.concatenate([c[k] for c in ch_parts]) for k in ch_parts[0]}
        return np.concatenate(z_parts)

    def _grab_channels(self, n: int):
        """The extra channels of the waveform just measured (same AI record as Z), first n
        samples, in their units -- None if none are configured or the backend can't."""
        if not self.extra_channels:
            return None
        read = getattr(self.backend, "read_channels", None)
        if not callable(read):
            return None
        names = list(self.extra_channels)
        raw = read([self.extra_channels[k][0] for k in names])
        return {k: np.asarray(r[:n], dtype=float) * float(self.extra_channels[k][1]) for k, r in zip(names, raw)}

    def _planning_xy(self, step, x_t, y_t):
        """A step's positions for the planners: measured if wanted and available, else the targets."""
        if getattr(self.position_sensors, "plan_on_sensors", True) and step.x_sensor is not None \
                and len(step.x_sensor) == len(x_t):
            return np.asarray(step.x_sensor), np.asarray(step.y_sensor)
        return np.asarray(x_t), np.asarray(y_t)

    def _store(self, step: TrajectoryStep) -> TrajectoryStep:
        """The step as kept in history: block means of readback.store_every samples (RAM);
        the step itself when store_every <= 1 or it's too short."""
        k = int(getattr(self.readback, "store_every", 1) or 1)
        n = len(step.x)
        if k <= 1 or n < 2 * k:
            return step
        m = n // k

        def red(a):
            if a is None:
                return None
            a = np.asarray(a, dtype=float)
            head = a[:m * k].reshape(m, k).mean(axis=1)
            return np.concatenate([head, [a[m * k:].mean()]]) if n > m * k else head

        return TrajectoryStep(step.kind, red(step.x), red(step.y), red(step.z), dict(step.meta, store_every=k, n_raw=n),
                              red(step.x_sensor), red(step.y_sensor), red(step.x_sensor_raw),
                              red(step.y_sensor_raw),
                              channels=None if step.channels is None else {c: red(v) for c, v in step.channels.items()})

    def move_to(self, x: float, y: float, meta: Optional[dict] = None) -> None:
        """Explicit repositioning without acquiring a planned trajectory -- routed
        through the same connector machinery so it's still a continuous, measured,
        live-viewable motion rather than a silent teleport."""
        target = np.array([float(x), float(y)])
        if np.linalg.norm(target - self.last_pos) > self.position_tol_V:
            self._send_connector(target, meta=meta)

    def send_pyscanner(self, scan, kind: str, meta: Optional[dict] = None, check_samples: bool = True,
                       correct_delay: bool = True):
        """Run one PyScanner-DEFINED scan (pyscanner_scans.PyScannerScan -- PyScanner
        generates the path from its scan-type parameters; check_samples=False skips the
        max_samples_per_waveform check, e.g. for one long reference raster). The tip is first moved
        smoothly (a measured connector, as for every other scan) to where the scan is
        predicted to start, so PyScanner doesn't jump there. Returns (x, y, z) of the
        path PyScanner actually ran, which is what goes into history.

        Fast rasters (correct_delay, readback.fast_raster_delay != "off"): Z is re-paired
        with the path at its time shift (pyscanner_scans.fit_fast_raster_delay) -- without
        it trace and retrace put every feature at two places along the fast axis. The
        shift is kept in the step's meta (fast_raster_delay_s, fast_raster_delay_note)."""
        from . import pyscanner_scans as ps
        meta = meta or {}
        run = getattr(self.backend, "run_pyscanner_scan", None)
        if run is None:
            raise NotImplementedError(f"{type(self.backend).__name__} can't run PyScanner-defined scans")
        if not ps.within(scan, self.safety):
            raise trajectory.UnsafeWaveformError(
                f"PyScanner scan reaches beyond the safety limits ({scan.describe()})")
        n_expected = int(round(scan.duration_s * self.velocity.sample_rate_hz))
        if check_samples and n_expected > self.safety.max_samples_per_waveform:
            raise ValueError(f"PyScanner scan would be {n_expected:,} samples, more than "
                             f"max_samples_per_waveform={self.safety.max_samples_per_waveform:,} "
                             f"({scan.describe()}) -- split it (smaller area) or raise the limit")
        gap_tol = max(self.position_tol_V, 1.5 * self.velocity.sample_spacing_V)
        scan_t = scan                       # as the planner asked for it (target frame)
        if self.correcting:
            # PyScanner draws the path itself: the correction can only place it -- its centre
            # through A^-1 (target - u) and its size through the map's scale (shear ignored). A
            # new u is eased in by the move to its start; without a move, the current u is kept
            import copy as _copy
            A, Ai = self._map()

            def _placed(u):
                sc = _copy.copy(scan)
                c = Ai @ (np.asarray(scan.center_xy, dtype=float) - u)
                sc.center_xy = (float(c[0]), float(c[1]))
                rot90 = round(scan.rotation_deg / 90) % 2 == 1
                kx, ky = (A[1, 1], A[0, 0]) if rot90 else (A[0, 0], A[1, 1])
                sc.size_x_V, sc.size_y_V = scan.size_x_V / abs(kx), scan.size_y_V / abs(ky)
                if scan.pattern == "spiral":
                    sc.radius_V = scan.radius_V / abs(0.5 * (A[0, 0] + A[1, 1]))
                return sc

            px, py = ps.predict_path(scan, self.velocity.sample_rate_hz)
            if np.linalg.norm(np.array([px[0], py[0]]) - self.last_pos) <= gap_tol:
                self._u_target = self._u_applied.copy()
            scan = _placed(self._u_target)
            meta = dict(meta, drive_correction_V=[float(v) for v in self._u_target])
        if not ps.within(scan, self.safety):
            raise trajectory.UnsafeWaveformError(
                f"PyScanner scan (corrected) reaches beyond the safety limits ({scan.describe()})")
        px, py = ps.predict_path(scan, self.velocity.sample_rate_hz)     # the drive PyScanner will output
        start_t = np.array([px[0], py[0]])
        if self.correcting:
            A, _ = self._map()
            start_t = A @ np.array([px[0], py[0]]) + self._u_target     # its start, as a target
        if np.linalg.norm(start_t - self.last_pos) > gap_tol:
            self._send_connector(start_t, meta={"label": f"Moving to the start of: {meta.get('label', kind)}"})
        self._u_applied = self._u_target.copy()
        if self.on_before_step is not None:
            self.on_before_step(kind, meta, self)
        x, y, z = run(scan, self.velocity.sample_rate_hz)
        z = np.asarray(z, dtype=float) * self.z_scale
        # last_pos is a target position: where the outputs really are after the scan (the
        # backend's readback) -- not the end of AO_vs_time, which can differ (PyScanner may stop
        # or hold the outputs elsewhere): the next move then starts there, never with a jump
        end = getattr(self.backend, "last_out", None)
        end = (x[-1], y[-1]) if end is None else end
        self.last_pos = np.array(self.to_target([end[0]], [end[1]])).ravel()
        x_sensor = y_sensor = None
        if self.position_sensors.enabled:   # from the same AI_vs_time as z, like _measure
            x_sensor, y_sensor = self.backend.read_position_sensors(x, y)
        n = min(len(z), len(x))
        x, y, z = np.asarray(x[:n]), np.asarray(y[:n]), np.asarray(z[:n])
        if x_sensor is not None:
            x_sensor, y_sensor = np.asarray(x_sensor[:n]), np.asarray(y_sensor[:n])
        chans = self._grab_channels(n)
        meta = dict(meta, pyscanner_scan=scan.describe())
        mode = self.readback.fast_raster_delay
        if scan.pattern == "raster" and correct_delay and mode != "off":
            rate = self.velocity.sample_rate_hz
            if mode == "fit":
                fit = ps.fit_fast_raster_delay(scan, z, rate, self.fast_raster_delay_s)
                if fit["fitted"]:
                    self.fast_raster_delay_s = fit["delay_s"]
            elif mode == "fixed":
                fit = dict(delay_s=float(self.readback.fast_raster_delay_s),
                           note=f"fixed shift {self.readback.fast_raster_delay_s * 1e3:+.1f} ms")
            else:
                raise ValueError(f"Unknown readback.fast_raster_delay {mode!r} (use 'fit', 'fixed' or 'off')")
            # Z (and the sensors, read with it) at sample i <-> the path at i + delay
            p_sl, z_sl = ps.shift_by_delay(n, int(round(fit["delay_s"] * rate)))
            x, y, z = x[p_sl], y[p_sl], z[z_sl]
            if x_sensor is not None:
                x_sensor, y_sensor = x_sensor[z_sl], y_sensor[z_sl]
            if chans is not None:
                chans = {k: v[z_sl] for k, v in chans.items()}
            meta.update(fast_raster_delay_s=float(fit["delay_s"]), fast_raster_delay_note=fit["note"])
        xs, ys = self._sensor_positions(x_sensor, y_sensor)
        step = TrajectoryStep(kind, x, y, z, meta, xs, ys, x_sensor, y_sensor, channels=chans)
        self.step_index += 1
        self._last_full_step = step
        self.history.append(self._store(step))
        xt, yt = self.to_target(x, y)
        self.last_xy = self._planning_xy(step, xt, yt)
        self.last_channels = step.channels
        self._update_correction(step)
        if self.on_step is not None:
            self.on_step(self.step_index, self.history[-1], self)
        return step.x, step.y, step.z

    # ------------------------------------------------------------------
    def _send_connector(self, target_xy: np.ndarray, meta: Optional[dict] = None) -> None:
        x_vec, y_vec = _line_including_endpoint(self.last_pos, target_xy, self.velocity.sample_spacing_V)
        # always from rest to rest (at least 20 ms speeding up / slowing down) -- never a jump
        # in speed, also for executors no planner set connector_ease_samples on
        n_ease = max(int(self.connector_ease_samples), int(round(0.02 * self.velocity.sample_rate_hz)))
        x_vec, y_vec = trajectory.ease_ends(x_vec, y_vec, n_ease)
        max_n = max(2, int(self.safety.max_samples_per_waveform))
        if len(x_vec) <= max_n:
            self._measure(x_vec, y_vec, "connector", meta or {})
        else:
            # a long transit is still one physical move -- split it the same way
            for k in range(0, len(x_vec), max_n):
                self._measure(x_vec[k:k + max_n], y_vec[k:k + max_n], "connector", meta or {})

    def _measure(self, x_vec: np.ndarray, y_vec: np.ndarray, kind: str, meta: dict) -> np.ndarray:
        # x_vec, y_vec: TARGET positions; sent as the (corrected) drive -- the same without a correction
        target_end = np.array([x_vec[-1], y_vec[-1]], dtype=float)
        if self.correcting:
            meta = dict(meta, drive_correction_V=[float(v) for v in self._u_target])
        x_vec, y_vec = self.to_drive(x_vec, y_vec)
        x_vec, y_vec = trajectory.clamp_to_safety(x_vec, y_vec, self.safety)
        if self.on_before_step is not None:
            self.on_before_step(kind, meta, self)
        z_vec = np.asarray(self.backend.measure_path(x_vec, y_vec), dtype=float) * self.z_scale
        # the whole waveform was OUTPUT, so the tip really ends at x_vec[-1] (as a target) ...
        self.last_pos = target_end

        x_sensor = y_sensor = None
        if self.position_sensors.enabled:
            x_sensor, y_sensor = self.backend.read_position_sensors(x_vec, y_vec)

        # ... but the backend may have RECORDED fewer samples than it output (see
        # PyScannerBackend._extract_raw_channel -- missing ones are at the end), so
        # keep only the measured part in history: every stored step has x, y, z
        # (and sensors) of equal length, with z[i] measured at (x[i], y[i]).
        n = len(z_vec)
        x_vec, y_vec = x_vec[:n], y_vec[:n]
        if x_sensor is not None:
            x_sensor, y_sensor = x_sensor[:n], y_sensor[:n]
        chans = self._grab_channels(n)

        xs, ys = self._sensor_positions(x_sensor, y_sensor)
        step = TrajectoryStep(kind, x_vec, y_vec, z_vec, meta, xs, ys, x_sensor, y_sensor, channels=chans)
        self.step_index += 1
        self._last_full_step = step
        self.history.append(self._store(step))
        self._update_correction(step)
        if self.on_step is not None:
            self.on_step(self.step_index, self.history[-1], self)
        return z_vec

    # convenience for plotting / analysis, mirrors SimulateBackend.all_points().
    # Prefers sensor position over commanded whenever it was collected -- see
    # PositionSensorConfig / point 4 in this module's docstring.
    def all_points(self) -> np.ndarray:
        if not self.history:
            return np.empty((0, 3))
        x = np.concatenate([s.x_sensor if s.x_sensor is not None else s.x for s in self.history])
        y = np.concatenate([s.y_sensor if s.y_sensor is not None else s.y for s in self.history])
        z = np.concatenate([s.z for s in self.history])
        return np.column_stack([x, y, z])
