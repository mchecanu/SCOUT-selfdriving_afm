"""
PyScanner hardware backend.

Wraps the ActiveX calls shown in `PyScanner_examples_042023_01.py` and the Hedgehog
notebook behind the same `measure_path(x_vec, y_vec) -> z_vec` interface the
planners use, so the exact same planner code runs on simulation or real hardware.

Everything here is written directly from your example scripts EXCEPT the items
flagged "# CONFIRM ON HARDWARE" -- those are educated guesses (matching the naming
convention of everything else you've shown me) that need a quick check against the
live LabVIEW front panel before a real run. Nothing else in the package needs to
change if these turn out to have slightly different names -- it's all isolated here.

**Channel map** (per your example scripts -- confirm this assignment against the
front panel before a real run, same as the two items below):
  - `AO_X_CONTROL` ("UI_AO0_V") drives the X piezo axis.
  - `AO_Y_CONTROL` ("UI_AO1_V") drives the Y piezo axis.
  - The Z channel is deflection READBACK, not an output -- there's no AO for it.
    Which AI channel it comes in on is `config.ReadbackConfig.ai_channel`
    (default `"AI0"`); change that one config value, not this file, if your
    deflection signal is wired to a different AI input.

Two readback strategies, selected via config.ReadbackConfig.mode:
  - "raw":   uses a per-sample AI readout that lines up 1:1 with the UI_AO0_V /
             UI_AO1_V waveform you sent. This is the natural fit for arbitrary
             (non-raster) trajectories like spirals and tracing circles, since
             there's no binning/interpolation ambiguity. Required if you're using
             config.PositionSensorConfig (below) -- there's no "image" equivalent
             for sensor position.
  - "image": uses the existing binned `image_AI0` reconstruction (like your raster
             and spiral examples) and maps each requested (x, y) sample back to its
             nearest pixel. Provided so you can cross-check the raw readout against
             the imaging pipeline you already trust, and because it's the only
             option if `AI_vs_time` doesn't turn out to exist under that name.

**Position sensors** (config.PositionSensorConfig, opt-in): if enabled, every
`measure_path` call also fetches `x_ai_channel`/`y_ai_channel` (default AI1/AI2)
from the SAME already-fetched `AI_vs_time` array Z came from -- one ActiveX call
per segment, not one per channel, since re-querying the control a second time isn't
guaranteed to return the same snapshot. `read_position_sensors` reuses that cached
array; it only makes sense called right after `measure_path` for the same segment
(that's exactly how `execution.ContinuousExecutor` uses it) and raises if
readback.mode isn't "raw".
"""

from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple

from .base import ScanBackend
from ..config import SafetyConfig, ReadbackConfig, PositionSensorConfig
from ..trajectory import UnsafeWaveformError

AO_X_CONTROL = "UI_AO0_V"  # X piezo drive -- CONFIRM ON HARDWARE (see module docstring)
AO_Y_CONTROL = "UI_AO1_V"  # Y piezo drive -- CONFIRM ON HARDWARE (see module docstring)


def connect(exe_path: str, vi_path: str, boot_wait_s: float = 5.0):
    """Start (if needed) and connect to the PyScanner LabVIEW executable via ActiveX.
    Mirrors PyScanner_examples_042023_01.py exactly.
    """
    import win32com.client
    import os

    os.startfile(exe_path)
    time.sleep(boot_wait_s)

    labview = win32com.client.Dispatch("BEPyAE.Application")
    vi = labview.getvireference(vi_path)
    return vi


def _ai_channel_index(ai_channel: str) -> int:
    """"AI0" -> 0, "AI1" -> 1, "AI2" -> 2, ... -- generalizes the old AI0/AI1-only
    lookup now that position sensors can name AI1/AI2 too."""
    return int(ai_channel.upper().replace("AI", ""))


def _wait_while(vi, control_name: str, poll_s: float = 0.0):
    """Poll a boolean 'busy' control until it goes False (the do_scan / do_scan_update
    / do_probe_move pattern used throughout your examples)."""
    busy = True
    while busy:
        busy = vi.getcontrolvalue(control_name)
        if poll_s:
            time.sleep(poll_s)


class PyScannerBackend(ScanBackend):
    def __init__(self, vi, safety: SafetyConfig, readback: ReadbackConfig,
                 position_sensors: Optional[PositionSensorConfig] = None):
        self.vi = vi
        self.safety = safety
        self.readback = readback
        self.position_sensors = position_sensors or PositionSensorConfig()
        self._last_raw_readback: Optional[np.ndarray] = None
        self._last_shifted: Optional[np.ndarray] = None
        self.last_pyscanner_report: dict = {}
        # where the X/Y outputs were left by the last COMPLETED output (None = not known --
        # e.g. before the first one, or after one was interrupted): move_to then goes there
        # smoothly with a user-defined waveform instead of PyScanner's line-scan move
        self.last_out: Optional[np.ndarray] = None
        self.sample_rate_hz = 1.0e6
        # no-jump guard: before every output the X/Y outputs are read back (get_AO0_V/AO1_V);
        # if they aren't where the output starts (> guard_tol_V), they're first driven there
        # smoothly at guard_speed_V_s. After every PyScanner scan last_out is the readback.
        self.guard = True
        self.guard_tol_V = 0.005
        self.guard_speed_V_s = 10.0
        self.guard_log: list = []

    # PyScanner's data_rate enum -> samples per second (only the setting checked on this
    # instrument so far; add others here as you confirm them on the front panel)
    DATA_RATE_HZ = {0: 1.0e6, 1: 1.0e5, 2: 1.0e4}   # 1 / 2: from the user, 2026-11-02

    def check_sample_rate(self, sample_rate_hz: float) -> None:
        """The tip speed is tip_velocity_V_s only if VelocityConfig.sample_rate_hz is
        PyScanner's actual output rate (waveforms are built at tip_velocity/sample_rate
        volts per sample). Called by ContinuousExecutor when it's built: raises if they
        differ, notes it if the data_rate setting isn't one we know the rate of."""
        try:
            setting = int(self.vi.getcontrolvalue('data_rate'))
        except Exception as e:  # noqa: BLE001 -- a readback problem shouldn't stop the run
            print(f"[sample rate] could not read PyScanner's data_rate ({e}) -- make sure "
                  f"cfg.velocity.sample_rate_hz = {sample_rate_hz:g} matches it")
            return
        rate = self.DATA_RATE_HZ.get(setting)
        if rate is None:
            print(f"[sample rate] PyScanner data_rate setting = {setting} (rate not known to the "
                  f"package) -- make sure cfg.velocity.sample_rate_hz = {sample_rate_hz:g} matches it")
        if rate is not None and abs(rate - sample_rate_hz) <= 1e-6 * rate:
            self.sample_rate_hz = float(rate)          # smooth moves (move_to) at the real rate
        if rate is None:
            pass
        elif abs(rate - sample_rate_hz) > 1e-6 * rate:
            raise ValueError(
                f"cfg.velocity.sample_rate_hz = {sample_rate_hz:g} but PyScanner's data_rate is "
                f"{rate:g} samples/s -- the tip would move {rate / sample_rate_hz:g}x faster than "
                f"tip_velocity_V_s. Set cfg.velocity.sample_rate_hz = {rate:g} (or change "
                f"data_rate in PyScanner) and rebuild.")

    # ------------------------------------------------------------------
    # Core measurement
    # ------------------------------------------------------------------
    def measure_path(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        x_vec = np.asarray(x_vec, dtype=float)
        y_vec = np.asarray(y_vec, dtype=float)

        # defense in depth -- planners/trajectory.py should already have clamped this,
        # but a hardware backend should never trust its caller blindly.
        from ..config import amplitude_limits
        lx, ly = amplitude_limits(self.safety)
        mx, my = float(np.max(np.abs(x_vec))), float(np.max(np.abs(y_vec)))
        if mx > lx or my > ly:
            raise UnsafeWaveformError(
                f"Refusing to send waveform: x {mx:.4f} / y {my:.4f} V exceeds the limits "
                f"x {lx:.4f} / y {ly:.4f} V"
            )
        self._drive_to_start(x_vec[0], y_vec[0], "user-defined waveform")
        self.last_out = None          # unknown until this one has finished

        vi = self.vi
        vi.setcontrolvalue('scan_type', 0)  # user-defined scan

        vi.setcontrolvalue(AO_X_CONTROL, x_vec.tolist())
        vi.setcontrolvalue(AO_Y_CONTROL, y_vec.tolist())

        if self.readback.mode == "image":
            vi.setcontrolvalue('N_image_pixels', self.readback.n_image_pixels)  # CONFIRM ON HARDWARE

        vi.setcontrolvalue('do_scan_update', 1)
        _wait_while(vi, 'do_scan_update')

        vi.setcontrolvalue('do_scan', 1)
        _wait_while(vi, 'do_scan')
        self.last_out = np.array([x_vec[-1], y_vec[-1]])
        self._check_end(self.last_out, "user-defined waveform")

        if self.readback.mode == "raw":
            # cached so read_position_sensors (called right after, by
            # ContinuousExecutor, when enabled) doesn't re-query the control --
            # see module docstring.
            self._last_raw_readback = np.asarray(self.vi.getcontrolvalue('AI_vs_time'))
            return self._extract_raw_channel(self._last_raw_readback, self.readback.ai_channel, len(x_vec))
        else:
            self._last_raw_readback = None
            return self._read_image(x_vec, y_vec)

    def move_to(self, x: float, y: float, transition_time_s: float = 0.2) -> None:
        """Reposition the tip without acquiring data. Where the outputs are is known
        (last_out): a user-defined waveform that starts and ends at rest (cosine profile,
        1.6 x transition_time_s long so its peak speed is about distance / transition_time_s)
        -- never a jump in position or speed. Not known: PyScanner's line-scan move from the
        current output (the Hedgehog notebook's move_tip_to_pos())."""
        target = np.array([float(x), float(y)])
        if self.last_out is not None:
            d = float(np.hypot(*(target - self.last_out)))
            if d < 1e-6:
                return
            T = 1.6 * max(float(transition_time_s), 0.05)
            n = max(int(round(T * self.sample_rate_hz)), 2)
            w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))
            p0 = self.last_out.copy()
            self.measure_path(p0[0] + w * (target[0] - p0[0]), p0[1] + w * (target[1] - p0[1]))
            return
        vi = self.vi
        line_scan_cluster = list(vi.getcontrolvalue('line_scan_control_cluster'))
        line_scan_cluster[2] = x
        line_scan_cluster[3] = y
        line_scan_cluster[4] = transition_time_s
        vi.setcontrolvalue('line_scan_control_cluster', tuple(line_scan_cluster))
        vi.setcontrolvalue('make_cur_pos_start_pos', True)
        vi.setcontrolvalue('do_probe_move_update', True)
        _wait_while(vi, 'do_probe_move_update')
        vi.setcontrolvalue('do_probe_move', True)
        _wait_while(vi, 'do_probe_move')
        self.last_out = target

    def read_ao_xy(self) -> Optional[Tuple[float, float]]:
        """The FPGA's X/Y outputs right now (AO0, AO1, V), from the front panel's
        get_AO0_V / get_AO1_V indicators (as in the Hedgehog notebook) -- CONFIRM ON
        HARDWARE; None if they can't be read."""
        try:
            return float(self.vi.getcontrolvalue('get_AO0_V')), float(self.vi.getcontrolvalue('get_AO1_V'))
        except Exception:  # noqa: BLE001
            return None

    def _settled_ao(self, expected, timeout_s: float = 1.5):
        """The AO readback, given time to catch up: get_AO0_V/AO1_V lag (read right after an
        output they can still show the one before -- hardware, 2026-11-03). Polled until it is
        at `expected` or stays put for timeout_s; (readback, seconds waited)."""
        t0 = time.perf_counter()
        now = self.read_ao_xy()
        while now is not None and np.hypot(now[0] - expected[0], now[1] - expected[1]) > self.guard_tol_V                 and time.perf_counter() - t0 < timeout_s:
            time.sleep(0.02)
            now = self.read_ao_xy()
        return now, time.perf_counter() - t0

    def _drive_to_start(self, x0: float, y0: float, what: str) -> None:
        """The no-jump guard: if the outputs (read back) aren't at (x0, y0), drive them
        there first -- a smooth cosine move from where they really are, rest to rest."""
        if not self.guard:
            return
        now, waited = self._settled_ao((x0, y0))
        if now is None:
            return
        d = float(np.hypot(x0 - now[0], y0 - now[1]))
        self.guard_log.append(dict(what=what, ao=now, start=(float(x0), float(y0)), gap_V=d, waited_s=round(waited, 3)))
        if d <= self.guard_tol_V:
            self.last_out = np.array(now, dtype=float)
            return
        print(f"[no-jump guard] outputs at ({now[0]:+.4f}, {now[1]:+.4f}) V, the {what} starts at "
              f"({x0:+.4f}, {y0:+.4f}) V ({d:.4f} V away) -- driving there smoothly first")
        T = max(d / self.guard_speed_V_s, 0.05)
        n = max(int(round(T * self.sample_rate_hz)), 2)
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))
        self.guard = False
        try:
            self.measure_path(now[0] + w * (x0 - now[0]), now[1] + w * (y0 - now[1]))
        finally:
            self.guard = True

    def _check_end(self, expected, what: str) -> None:
        """After an output: where the outputs really are. last_out = the readback (the next
        move starts from there); reported if it isn't where the output was expected to end."""
        now, waited = self._settled_ao(expected)
        if now is None:
            return
        d = float(np.hypot(now[0] - expected[0], now[1] - expected[1]))
        self.guard_log.append(dict(what=what + " (end)", ao=now, expected=tuple(map(float, expected)), gap_V=d,
                                   waited_s=round(waited, 3)))
        if d > self.guard_tol_V:
            print(f"[no-jump guard] the {what} ended with the outputs at ({now[0]:+.4f}, {now[1]:+.4f}) V, "
                  f"not ({expected[0]:+.4f}, {expected[1]:+.4f}) V ({d:.4f} V) -- the next move starts from the readback")
        self.last_out = np.array(now, dtype=float)

    # ------------------------------------------------------------------
    # PyScanner's own scan types (the "pyscanner_defined" generator -- see
    # pyscanner_scans.py): PyScanner generates the path; we set its parameters, place
    # it with the scan offsets, and read the path it actually ran back from AO_vs_time.
    # ------------------------------------------------------------------
    def run_pyscanner_scan(self, scan, sample_rate_hz: float):
        """Run one PyScanner-defined scan (pyscanner_scans.PyScannerScan). Returns
        (x_vec, y_vec, z_vec): the path PyScanner actually output (AO_vs_time rows 0/1,
        placed at the scan's offset if AO_vs_time turns out not to include it) and the
        measured Z along it. The offsets, rotation and scan_type are put back to 0
        afterwards, so user-defined waveforms (moves, other modes) stay absolute.
        self.last_pyscanner_report has what was checked (samples, start point, placement)."""
        from .. import pyscanner_scans as ps
        if not ps.within(scan, self.safety):
            raise UnsafeWaveformError(f"Refusing PyScanner scan: it reaches beyond the safety limits ({scan.describe()})")
        import dataclasses as _dc
        p0x, p0y = ps.predict_path(_dc.replace(scan, duration_s=8.0 / sample_rate_hz), sample_rate_hz)
        self._drive_to_start(float(p0x[0]), float(p0y[0]), f"PyScanner {scan.pattern}")
        self.last_out = None
        vi = self.vi
        name = ps.CLUSTER[scan.pattern]
        cluster = list(vi.getcontrolvalue(name))
        for i, v in ps.cluster_values(scan).items():
            cluster[i] = v
        try:
            vi.setcontrolvalue('scan_type', ps.SCAN_TYPE[scan.pattern])
            vi.setcontrolvalue(name, tuple(cluster))
            vi.setcontrolvalue('scan_x_offset_V', float(scan.center_xy[0]))
            vi.setcontrolvalue('scan_y_offset_V', float(scan.center_xy[1]))
            vi.setcontrolvalue('scan_rotation_deg', float(scan.rotation_deg))
            if self.readback.mode == "image":
                vi.setcontrolvalue('N_image_pixels', self.readback.n_image_pixels)
            t = [time.perf_counter()]
            vi.setcontrolvalue('do_scan_update', 1)
            _wait_while(vi, 'do_scan_update')
            t.append(time.perf_counter())
            vi.setcontrolvalue('do_scan', 1)
            _wait_while(vi, 'do_scan')
            t.append(time.perf_counter())

            ao = np.asarray(vi.getcontrolvalue('AO_vs_time'), dtype=float)
            # checked on hardware: AO_vs_time_shifted is the output WITH the scan offset
            # (a spiral at offset (0.1, 0.05) started at (0.100, 0.050) there, at (0, 0) in
            # AO_vs_time) -- the real tip path; AO_vs_time is kept for the image lookup
            try:
                sh = np.asarray(vi.getcontrolvalue('AO_vs_time_shifted'), dtype=float)
                self._last_shifted = sh if sh.ndim == 2 and sh.shape[1] == ao.shape[1] else None
            except Exception:  # noqa: BLE001
                self._last_shifted = None
            t.append(time.perf_counter())
            x_raw, y_raw = ao[0], ao[1]  # AO0 = X, AO1 = Y (as for user-defined scans)
            if self.readback.mode == "raw":
                self._last_raw_readback = np.asarray(vi.getcontrolvalue('AI_vs_time'))
                z = self._extract_raw_channel(self._last_raw_readback, self.readback.ai_channel, len(x_raw))
            else:
                self._last_raw_readback = None
                # PyScanner's image is binned over the same coordinates AO_vs_time reports
                z = self._read_image(x_raw, y_raw)
            t.append(time.perf_counter())
        finally:
            vi.setcontrolvalue('scan_x_offset_V', 0.0)
            vi.setcontrolvalue('scan_y_offset_V', 0.0)
            vi.setcontrolvalue('scan_rotation_deg', 0.0)
            vi.setcontrolvalue('scan_type', 0)
        if self._last_shifted is not None:
            x, y = self._place_pyscanner_path(scan, self._last_shifted[0], self._last_shifted[1], sample_rate_hz)
        else:
            x, y = self._place_pyscanner_path(scan, x_raw, y_raw, sample_rate_hz)
        self.last_out = np.array([x[-1], y[-1]], dtype=float)
        self._check_end(self.last_out, f"PyScanner {scan.pattern}")      # last_out = the readback
        self.last_pyscanner_report["end_ao"] = tuple(float(v) for v in self.last_out)
        self.last_pyscanner_report["timing_s"] = dict(
            do_scan_update=round(t[1] - t[0], 3), do_scan=round(t[2] - t[1], 3),
            read_AO_vs_time=round(t[3] - t[2], 3), read_Z=round(t[4] - t[3], 3))
        return x, y, z

    def _place_pyscanner_path(self, scan, x_raw, y_raw, sample_rate_hz):
        """AO_vs_time may or may not include the scan offset/rotation (CONFIRM ON
        HARDWARE -- probe_pyscanner_scans reports which): if the path is centred on the
        requested centre, use it as is; if it's centred on (0, 0), apply the offset and
        rotation here. Also compares the start point and length with the prediction."""
        from .. import pyscanner_scans as ps
        cx, cy = scan.center_xy
        mid = np.array([0.5 * (x_raw.min() + x_raw.max()), 0.5 * (y_raw.min() + y_raw.max())])
        size = max(scan.size_x_V, scan.size_y_V, 1e-6)
        tol = 0.05 * size + 1e-3
        if np.hypot(mid[0] - cx, mid[1] - cy) <= tol:
            x, y, placed = x_raw, y_raw, "AO_vs_time includes the offset"
        elif np.hypot(*mid) <= tol:
            # centred on 0: the offset (and so presumably the rotation) isn't in AO_vs_time
            c, s = np.cos(np.deg2rad(-scan.rotation_deg)), np.sin(np.deg2rad(-scan.rotation_deg))  # clockwise
            x, y = cx + c * x_raw - s * y_raw, cy + s * x_raw + c * y_raw
            placed = "AO_vs_time is centred on 0 -- offset and rotation added by the package"
        else:
            x, y = x_raw, y_raw
            placed = (f"WARNING: AO_vs_time is centred on ({mid[0]:.3f}, {mid[1]:.3f}) V, neither the "
                      f"requested ({cx:.3f}, {cy:.3f}) nor 0 -- used as is")
            print(f"[PyScanner scan] {placed}")
        px, py = ps.predict_path(scan, sample_rate_hz)
        start_err = float(np.hypot(x[0] - px[0], y[0] - py[0]))
        self.last_pyscanner_report = dict(
            scan=scan.describe(), samples=len(x_raw), expected_samples=len(px), placement=placed,
            start=(float(x[0]), float(y[0])), predicted_start=(float(px[0]), float(py[0])),
            start_error_V=start_err, end=(float(x[-1]), float(y[-1])),
            x_range=(float(x.min()), float(x.max())), y_range=(float(y.min()), float(y.max())))
        if start_err > tol:
            print(f"[PyScanner scan] its first sample is {start_err:.3f} V from where the package moved the "
                  f"tip ({px[0]:.3f}, {py[0]:.3f}) -- the tip JUMPED at the start. Run probe_pyscanner_scans "
                  f"and tell me the report so the start-point model can be fixed.")
        return x, y

    # ------------------------------------------------------------------
    # Readback strategy 1: raw per-sample channel
    # ------------------------------------------------------------------
    def _extract_raw_channel(self, raw: np.ndarray, ai_channel: str, expected_len: int) -> np.ndarray:
        # CONFIRM ON HARDWARE: exact control name + whether it returns one channel's
        # worth of samples or a [n_channels, N_points] array like 'AO_vs_time_shifted'
        # does for outputs. Adjust the indexing below once you've checked the panel.
        if raw.ndim == 2:
            channel = raw[_ai_channel_index(ai_channel), :]
        else:
            channel = raw

        # Checked on hardware: AO_vs_time is always the full waveform we sent, but
        # AI_vs_time can come back SHORTER (e.g. 3,623,205 of 3,777,537) -- the
        # missing samples are at the END (AI sample i <-> AO sample i; this pairing
        # reproduces image_AI0 to ~0.3 mV, the start-missing pairing does not). So
        # return the recorded head as-is: z may be shorter than x_vec, and callers
        # pair z[i] with x_vec[i] (ContinuousExecutor truncates the step to match).
        n = len(channel)
        if n == 0 or n > expected_len:
            raise RuntimeError(
                f"AI_vs_time channel {ai_channel} returned {n} samples but "
                f"the sent waveform had {expected_len} samples."
            )
        if n < expected_len:
            print(f"[raw readback] AI_vs_time recorded {n:,} of {expected_len:,} samples "
                  f"({expected_len - n:,} missing at the end); using the recorded part")
        return channel

    # ------------------------------------------------------------------
    # Position sensors (optional -- config.PositionSensorConfig)
    # ------------------------------------------------------------------
    def read_position_sensors(self, x_vec: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.readback.mode != "raw" or self._last_raw_readback is None:
            raise RuntimeError(
                "read_position_sensors needs readback.mode='raw', and must be called "
                "right after measure_path for the same segment -- it reuses that "
                "segment's already-fetched AI_vs_time array rather than re-querying "
                "the front panel a second time."
            )
        x_sensor = self._extract_raw_channel(self._last_raw_readback, self.position_sensors.x_ai_channel, len(x_vec))
        y_sensor = self._extract_raw_channel(self._last_raw_readback, self.position_sensors.y_ai_channel, len(y_vec))
        return x_sensor, y_sensor

    def read_channels(self, ai_channels):
        """Other AI channels (e.g. PFM amplitude / phase) of the waveform just measured, from
        the same already-fetched AI_vs_time as Z (no second query)."""
        if self.readback.mode != "raw" or self._last_raw_readback is None:
            raise RuntimeError("read_channels needs readback.mode='raw', right after measure_path / a PyScanner scan")
        raw = self._last_raw_readback
        n = raw.shape[-1]
        return [self._extract_raw_channel(raw, a, n) for a in ai_channels]

    # ------------------------------------------------------------------
    # Readback strategy 2: binned image, mapped back onto the trajectory
    # ------------------------------------------------------------------
    def _read_image(self, x_vec: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
        key = f'image_{self.readback.ai_channel.upper()}'
        image = np.matrix.transpose(np.asarray(self.vi.getcontrolvalue(key)))

        x_min, x_max = float(np.min(x_vec)), float(np.max(x_vec))
        y_min, y_max = float(np.min(y_vec)), float(np.max(y_vec))
        # guard against a degenerate (zero-span) trajectory
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_max - y_min, 1e-9)

        n_rows, n_cols = image.shape
        col_idx = np.clip(((x_vec - x_min) / x_span * (n_cols - 1)).round().astype(int), 0, n_cols - 1)
        row_idx = np.clip(((y_vec - y_min) / y_span * (n_rows - 1)).round().astype(int), 0, n_rows - 1)
        z_vec = np.asarray(image[row_idx, col_idx], dtype=float).ravel()

        # Pixels the path never crossed are NOT NaN in PyScanner's image -- they hold a
        # constant fill value (checked on hardware: ~80% of a spiral's pixels), and
        # 'image_mask' (0/1, same shape) marks which pixels were actually visited.
        # Treat unvisited ones as missing and fill from the nearest valid samples along
        # the path, reporting how many there were. A high fraction means the
        # pixel<->(x, y) mapping above doesn't match PyScanner's image frame.
        visited = np.matrix.transpose(np.asarray(self.vi.getcontrolvalue('image_mask'))) > 0
        z_vec[~np.asarray(visited[row_idx, col_idx]).ravel()] = np.nan
        nan_mask = ~np.isfinite(z_vec)
        if nan_mask.all() and len(z_vec) < 100 and visited.any():
            # a very short waveform (e.g. a tiny connecting move) can leave nothing in
            # PyScanner's image under its samples -- use the nearest visited pixel
            # rather than failing; only real scans (spirals/circles) need to be exact
            vr, vc = np.nonzero(visited)
            nearest = np.argmin((vr[None, :] - row_idx.ravel()[:, None]) ** 2 +
                                (vc[None, :] - col_idx.ravel()[:, None]) ** 2, axis=1)
            return np.asarray(image[vr[nearest], vc[nearest]], dtype=float).ravel()
        if nan_mask.all():
            raise RuntimeError(
                f"image_{self.readback.ai_channel.upper()} lookup returned only NaN for "
                f"all {len(z_vec)} samples -- the pixel mapping doesn't match PyScanner's image."
            )
        if nan_mask.any():
            print(f"[image readback] {nan_mask.mean():.1%} of samples landed on NaN pixels; "
                  f"filled from neighbouring samples along the path")
            idx = np.arange(len(z_vec))
            z_vec[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], z_vec[~nan_mask])
        return z_vec
