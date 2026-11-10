"""
X/Y position sensors (cfg.position_sensors): the real tip position, recorded next to Z.

The controller copies its XSensor / YSensor (the LVDT position sensors, in sensor volts) to
BNC outputs (asylum.AsylumControl.route_sensors: a PIDS loop per axis -> a free DAC -> the
BNC), wired to PyScanner's AI1 / AI2, so every waveform records them sample by sample with Z.

    AI volts --(ai_gain, ai_offset)--> sensor volts --(m_per_V, offset_V: Igor's XLVDTSens /
    XLVDTOffset)--> metres --(k, b: the drive calibration)--> "drive volts"

The last step expresses the measured position in the drive's own frame (volts, the frame the
planner, priors and pixel grid use): x_V = (x_m - b) / k, with k, b a straight-line fit of the
sensor position against the commanded drive position, measured at rest on a small grid in
the setup (calibrate). Every image built from the sensors is then geometrically right --
the piezo's own errors (non-linearity, hysteresis, creep) are gone -- and a pixel grid in
drive volts is a pixel grid in metres: 1 V = k metres along that axis (to_um, axis ticks).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np


@dataclass
class AxisCal:
    ai_gain: float = 1.0        # sensor V = ai_gain * AI V + ai_offset (the BNC copy + the AI input)
    ai_offset: float = 0.0
    m_per_V: float = 1e-6       # sensor V -> metres (Igor: XLVDTSens)
    offset_V: float = 0.0       # sensor V at 0 m (Igor: XLVDTOffset)
    k: float = 1e-6             # metres per drive volt (fitted)
    b: float = 0.0              # metres at drive 0 V (fitted)

    def metres(self, ai):
        return ((self.ai_gain * np.asarray(ai, dtype=float) + self.ai_offset) - self.offset_V) * self.m_per_V

    def drive_V(self, ai):
        return (self.metres(ai) - self.b) / self.k


@dataclass
class SensorCalibration:
    x: AxisCal = field(default_factory=AxisCal)
    y: AxisCal = field(default_factory=AxisCal)
    report: dict = field(default_factory=dict)
    # where the tip goes, in this frame, for a drive command d: A @ d + c (fitted at rest on the
    # calibration grid -- A ~ identity by construction of the frame; its off-diagonal terms are
    # the axes' cross-talk). The drive correction sends A^-1 (target - u) (execution.py)
    A: list = field(default_factory=lambda: [[1.0, 0.0], [0.0, 1.0]])
    c: list = field(default_factory=lambda: [0.0, 0.0])
    ok: bool = True        # passed the checks (AI gain as expected): corrections may use it

    def to_frame(self, ai_x, ai_y) -> Tuple[np.ndarray, np.ndarray]:
        """AI1/AI2 samples -> the measured position in drive volts."""
        return self.x.drive_V(ai_x), self.y.drive_V(ai_y)

    def um(self, x_V, y_V):
        """Drive-frame position (V) -> micrometres along the sensors' axes."""
        return (self.x.k * np.asarray(x_V) + self.x.b) * 1e6, (self.y.k * np.asarray(y_V) + self.y.b) * 1e6

    @property
    def um_per_V(self) -> Tuple[float, float]:
        return self.x.k * 1e6, self.y.k * 1e6

    def as_dict(self) -> dict:
        return dict(x=asdict(self.x), y=asdict(self.y), A=[list(map(float, r)) for r in self.A],
                    c=list(map(float, self.c)), ok=bool(self.ok))

    @classmethod
    def from_dict(cls, d: dict) -> "SensorCalibration":
        return cls(AxisCal(**d["x"]), AxisCal(**d["y"]), A=d.get("A", [[1.0, 0.0], [0.0, 1.0]]),
                   c=d.get("c", [0.0, 0.0]), ok=d.get("ok", True))


def smooth(a: np.ndarray, n: int) -> np.ndarray:
    """Moving average over n samples (edges: over what's there) -- sensor noise."""
    a = np.asarray(a, dtype=float)
    if n is None or n <= 1 or len(a) < 2:
        return a
    n = int(min(n, len(a)))
    k = np.ones(n)
    return np.convolve(a, k, mode="same") / np.convolve(np.ones(len(a)), k, mode="same")


def calibrate(backend, cfg, read_sensor_V=None, igor_gv=None, verbose: bool = True) -> SensorCalibration:
    """Measure the calibration at rest (tip WITHDRAWN): the drive is moved to a 3 x 3 grid
    of points cfg.position_sensors.calibration_span_V around (0, 0); at each, after settling,
    a short stationary waveform records AI1/AI2 (their mean), and read_sensor_V(name) (Igor:
    ir_ReadValue("XSensor")) gives the sensor volts directly -> AI -> sensor V (checks the
    wiring: gain ~ 1, offset ~ 0), and sensor metres vs drive volts -> k, b. Without Igor
    (read_sensor_V None: simulation) the AI volts are taken as sensor volts. Back at (0, 0)
    afterwards."""
    ps = cfg.position_sensors
    v = cfg.velocity.tip_velocity_V_s
    rate = cfg.velocity.sample_rate_hz
    s = min(ps.calibration_span_V, 0.9 * cfg.safety.max_amplitude_V)
    if ps.calibration_area_half_nm:       # inside the area to scan (the priors, set before the setup)
        s = min(s, ps.calibration_area_fraction * ps.calibration_area_half_nm * 1e-3 / ps.calibration_um_per_V_guess)
    pts = [(x, y) for j, y in enumerate((-s, 0.0, s)) for x in ((-s, 0.0, s) if j % 2 == 0 else (s, 0.0, -s))]
    n = max(int(ps.calibration_dwell_s * rate), 20)
    gx = igor_gv("XLVDTSens") if igor_gv else (ps.x_m_per_V or 1e-6)
    gy = igor_gv("YLVDTSens") if igor_gv else (ps.y_m_per_V or 1e-6)
    ox = igor_gv("XLVDTOffset") if igor_gv else 0.0
    oy = igor_gv("YLVDTOffset") if igor_gv else 0.0
    if ps.x_m_per_V:
        gx = ps.x_m_per_V
    if ps.y_m_per_V:
        gy = ps.y_m_per_V
    rows = []
    here = np.zeros(2)
    if verbose:
        print(f"  Sensor calibration: 3 x 3 points, +-{s:g} V, {ps.calibration_settle_s:g} s settling + "
              f"{n / rate * 1e3:.0f} ms recording each")
    for (x, y) in pts:
        d = float(np.hypot(x - here[0], y - here[1]))
        backend.move_to(x, y, transition_time_s=max(d / v, 0.2))
        here = np.array([x, y])
        time.sleep(ps.calibration_settle_s)
        xv, yv = np.full(n, x), np.full(n, y)
        backend.measure_path(xv, yv)
        ax_, ay_ = backend.read_position_sensors(xv, yv)
        sx = read_sensor_V("XSensor") if read_sensor_V else float(np.mean(ax_))
        sy = read_sensor_V("YSensor") if read_sensor_V else float(np.mean(ay_))
        rows.append((x, y, float(np.mean(ax_)), float(np.mean(ay_)), float(np.std(ax_)), float(np.std(ay_)), sx, sy))
        if verbose:
            print(f"    drive ({x:+.2f}, {y:+.2f}) V: AI1 {rows[-1][2]:+.5f} V, AI2 {rows[-1][3]:+.5f} V | "
                  f"XSensor {sx:+.5f} V, YSensor {sy:+.5f} V")
    backend.move_to(0.0, 0.0, transition_time_s=max(float(np.hypot(*here)) / v, 0.2))
    R = np.array(rows)
    dx, dy, aix, aiy, sdx, sdy, sx, sy = R.T
    # AI -> sensor volts
    gax, oax = np.polyfit(aix, sx, 1)
    gay, oay = np.polyfit(aiy, sy, 1)
    cal = SensorCalibration(AxisCal(gax, oax, gx, ox), AxisCal(gay, oay, gy, oy))
    mx, my = cal.x.metres(aix), cal.y.metres(aiy)
    cal.x.k, cal.x.b = np.polyfit(dx, mx, 1)
    cal.y.k, cal.y.b = np.polyfit(dy, my, 1)
    # how good: residuals of the straight-line fits, and the cross-talk (x moving with y drive)
    rx = mx - (cal.x.k * dx + cal.x.b)
    ry = my - (cal.y.k * dy + cal.y.b)
    A = np.column_stack([dx, dy, np.ones_like(dx)])
    cx = np.linalg.lstsq(A, mx, rcond=None)[0]
    cy = np.linalg.lstsq(A, my, rcond=None)[0]
    # the drive map in this frame: measured (frame) = A @ drive + c, a full 2 x 2 fit
    fx, fy = cal.to_frame(aix, aiy)
    M = np.linalg.lstsq(np.column_stack([dx, dy, np.ones_like(dx)]), np.column_stack([fx, fy]), rcond=None)[0]
    cal.A = M[:2].T.tolist()
    cal.c = M[2].tolist()
    noise_x = float(np.median(sdx)) * abs(gax) * gx
    noise_y = float(np.median(sdy)) * abs(gay) * gy
    cal.report = dict(points=R.tolist(), ai_gain=(gax, gay), ai_offset=(oax, oay), um_per_V=cal.um_per_V,
                      residual_nm=(float(np.sqrt(np.mean(rx ** 2))) * 1e9, float(np.sqrt(np.mean(ry ** 2))) * 1e9),
                      crosstalk=(float(cx[1] / cx[0]) if cx[0] else np.nan, float(cy[0] / cy[1]) if cy[1] else np.nan),
                      noise_nm=(noise_x * 1e9, noise_y * 1e9),
                      copy_gain=float(getattr(ps, "copy_gain", -1.0)))   # the routing it was measured with
    if verbose:
        r = cal.report
        div = -1.0 / ps.copy_gain if hasattr(ps, "copy_gain") else 1.0
        print(f"  AI -> sensor volts: X gain {gax:.4f}, offset {oax:+.4f} V; Y gain {gay:.4f}, offset {oay:+.4f} V "
              f"(expected ~{div:g} and ~0: the BNC copies the sensor" + (f" / {div:g})" if div != 1 else ")"))
        print(f"  sensor -> length: X {gx * 1e6:.4g} um/V (offset {ox:g} V), Y {gy * 1e6:.4g} um/V (offset {oy:g} V)")
        print(f"  drive calibration: X {r['um_per_V'][0]:+.4f} um per drive V, Y {r['um_per_V'][1]:+.4f} um per drive V "
              f"(straight-line residual {r['residual_nm'][0]:.1f} / {r['residual_nm'][1]:.1f} nm; cross-talk "
              f"{100 * r['crosstalk'][0]:+.2f}% / {100 * r['crosstalk'][1]:+.2f}%)")
        print(f"  sensor noise at rest (one sample): X {r['noise_nm'][0]:.2f} nm, Y {r['noise_nm'][1]:.2f} nm rms")
        print(f"  drive map (frame = A drive + c): A = [[{cal.A[0][0]:.4f}, {cal.A[0][1]:+.4f}], "
              f"[{cal.A[1][0]:+.4f}, {cal.A[1][1]:.4f}]], c = ({cal.c[0]:+.4f}, {cal.c[1]:+.4f}) V")
    div_ = -1.0 / ps.copy_gain if hasattr(ps, "copy_gain") else 1.0
    # the AI gain is only a check with Igor's own sensor reading (without it, it's 1 by construction)
    gain_ok = read_sensor_V is None or (abs(abs(gax) / div_ - 1) <= 0.2 and abs(abs(gay) / div_ - 1) <= 0.2)
    cal.ok = bool(np.isfinite(gax) and np.isfinite(gay) and gain_ok and np.all(np.isfinite(cal.A))
                  and abs(cal.x.k) > 0 and abs(cal.y.k) > 0)
    if verbose:
        if not cal.ok:
            print("  -> calibration NOT OK: sensor positions, planning on them and the drive correction are OFF "
                  "for this calibration (fix the routing / wiring and rerun the setup cell)")
        if read_sensor_V is not None and (abs(abs(gax) / div - 1) > 0.1 or abs(abs(gay) / div - 1) > 0.1):
            print(f"  WARNING: AI gain far from +-{div:g} -- check the wiring (BNCOut0 -> AI1, BNCOut1 -> AI2), "
                  f"the routing and cfg.position_sensors.divide_output")
    return cal
