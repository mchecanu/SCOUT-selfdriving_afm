"""
Trajectory / waveform synthesis.

Every function here returns plain numpy arrays (x_vec, y_vec[, t_vec]) in the SAME
units the caller passed in (we work in "bias volts" throughout this project per the
open-loop decision, but nothing here assumes that -- these are pure geometry helpers).

Safety clamping (`clamp_to_safety`) should be applied to every waveform right before
it is handed to `pyscanner_io.send_path`, not before -- keep the planning/geometry
code free of hardware limits so it stays testable in simulation.
"""

from __future__ import annotations
import numpy as np

from .config import SafetyConfig


class UnsafeWaveformError(ValueError):
    pass


def clamp_to_safety(x_vec: np.ndarray, y_vec: np.ndarray, safety: SafetyConfig):
    """Enforce the amplitude limit on a generated waveform.

    Raises UnsafeWaveformError if any sample exceeds max_amplitude_V and
    safety.clamp_mode == "raise" (the default). If clamp_mode == "clip", the
    waveform is clipped in place and returned (with a printed warning), which
    is useful for interactive/simulation work but NOT recommended as a default
    for real hardware runs -- silently clipping changes the geometry of the scan.
    """
    from .config import amplitude_limits
    lx, ly = amplitude_limits(safety)
    mx, my = float(np.max(np.abs(x_vec))), float(np.max(np.abs(y_vec)))
    if mx > lx or my > ly:
        if safety.clamp_mode == "raise":
            raise UnsafeWaveformError(
                f"Waveform amplitude x {mx:.4f} V / y {my:.4f} V exceeds the limits x {lx:.4f} V / "
                f"y {ly:.4f} V. Reduce spiral/circle radius or "
                f"path length, or raise the safety limit deliberately."
            )
        else:
            print(f"[safety] WARNING: clipping waveform (x {mx:.4f}, y {my:.4f} V) to x {lx:.4f}, y {ly:.4f} V")
            x_vec = np.clip(x_vec, -lx, lx)
            y_vec = np.clip(y_vec, -ly, ly)
    return x_vec, y_vec


# --------------------------------------------------------------------------------
# Straight-line path segment (used by the GP path planner)
# --------------------------------------------------------------------------------
def line_segment(start_xy, direction_xy, length: float, resolution: float):
    """A straight segment from start_xy, walking in unit direction_xy for `length`,
    sampled every `resolution`. Returns (x_vec, y_vec) including the start point.
    """
    direction_xy = np.asarray(direction_xy, dtype=float)
    norm = np.linalg.norm(direction_xy)
    if norm == 0:
        raise ValueError("direction_xy must be nonzero")
    direction_xy = direction_xy / norm

    n_points = max(int(round(length / resolution)), 1)
    _check_point_count(n_points, length, resolution, "line_segment")
    steps = np.arange(n_points) * resolution
    x_vec = start_xy[0] + steps * direction_xy[0]
    y_vec = start_xy[1] + steps * direction_xy[1]
    return x_vec, y_vec


# --------------------------------------------------------------------------------
# Point-count safety cap -- shared by every primitive below. A slow tip_velocity_V_s
# combined with a fast sample_rate_hz (see VelocityConfig) is physically valid but
# can imply an enormous point count for a large trajectory (total_points =
# arc_length / (velocity / rate)) -- that should error with a clear explanation of
# the tradeoff, not silently hang or exhaust memory building a 100M-point array.
# --------------------------------------------------------------------------------
_MAX_TRAJECTORY_POINTS = 50_000_000  # generation sanity cap; hardware
# waveforms are split into SafetyConfig.max_samples_per_waveform pieces by the executor


def _check_point_count(n_points: int, arc_length: float, ds: float, what: str):
    if n_points > _MAX_TRAJECTORY_POINTS:
        raise RuntimeError(
            f"{what} would need {n_points:,} points (arc_length={arc_length:.4g} V / "
            f"ds={ds:.4g} V) -- exceeds the safety cap of {_MAX_TRAJECTORY_POINTS:,}. "
            f"Raise VelocityConfig.tip_velocity_V_s, lower sample_rate_hz, or shrink "
            f"the trajectory's spatial extent."
        )


# --------------------------------------------------------------------------------
# Search spiral (Archimedean, growing outward)
# --------------------------------------------------------------------------------
def spiral_out(center_xy, r_start: float, r_end: float, n_cycles: float,
                ds: float, direction: str = "ccw", _oversample: int = 20):
    """Archimedean spiral from radius r_start to r_end around center_xy, resampled to
    constant ARC-LENGTH spacing `ds` -- pass VelocityConfig.sample_spacing_V so a
    fixed DAQ output rate gives a constant tip velocity in V/s, all the way from the
    center out (this package always starts spirals at r_start=0.0 for continuity).

    For an Archimedean spiral (dr/dtheta = b, constant), the arc-length element is
    ds_dtheta = sqrt(r^2 + b^2) dtheta -- NOT constant, since it grows with r, so
    naive uniform-dtheta sampling would NOT give uniform-arc-length (constant
    velocity) samples. This computes the cumulative arc length on a fine uniform-
    theta grid (vectorized trapezoid integration) and resamples (`np.interp`) to
    land samples at even ds increments along that arc length -- fully vectorized,
    no per-sample Python loop, so it stays fast even at real-hardware sample counts.

    Returns (x_vec, y_vec, r_vec, theta_vec). r_vec/theta_vec are handy for edge
    detection (they give you a clean 1D "distance traveled" axis).
    """
    if r_end <= r_start:
        raise ValueError("r_end must be > r_start")
    if ds <= 0:
        raise ValueError("ds must be > 0")

    b = (r_end - r_start) / (2 * np.pi * n_cycles)  # dr/dtheta, constant for this spiral
    theta_max = (r_end - r_start) / b

    approx_arc_length = 0.5 * (r_start + r_end) * theta_max  # coarse estimate, sizing only
    n_final_est = max(int(approx_arc_length / ds), 2)
    n_fine = int(np.clip(n_final_est * _oversample, 2000, 10_000_000))  # fine grid only -- interp does the rest

    theta_fine = np.linspace(0.0, theta_max, n_fine)
    r_fine = r_start + b * theta_fine
    ds_dtheta = np.sqrt(r_fine ** 2 + b ** 2)
    s_fine = np.concatenate(([0.0], np.cumsum(
        (ds_dtheta[:-1] + ds_dtheta[1:]) / 2 * np.diff(theta_fine))))
    total_arc_length = float(s_fine[-1])

    n_points = max(int(round(total_arc_length / ds)) + 1, 2)
    _check_point_count(n_points, total_arc_length, ds, "spiral_out")

    s_target = np.linspace(0.0, total_arc_length, n_points)
    theta_vec = np.interp(s_target, s_fine, theta_fine)
    r_vec = r_start + b * theta_vec

    if direction == "cw":
        theta_vec = -theta_vec

    x_vec = center_xy[0] + r_vec * np.cos(theta_vec)
    y_vec = center_xy[1] + r_vec * np.sin(theta_vec)
    return x_vec, y_vec, r_vec, theta_vec


# --------------------------------------------------------------------------------
# Tracing circle (used to hop along a detected perimeter)
# --------------------------------------------------------------------------------
def circle(center_xy, radius: float, ds: float,
           start_angle: float = 0.0, arc_fraction: float = 1.0, direction: str = "ccw"):
    """A full (or partial) circle around center_xy, sampled at constant arc-length
    spacing `ds` (pass VelocityConfig.sample_spacing_V) -- a circle has constant
    curvature, so unlike the search spiral, uniform angular spacing IS exactly
    uniform arc-length spacing here; no integration needed.

    arc_fraction=1.0 traces the full circle. start_angle lets the caller begin
    the sweep pointing away from where it just came from, which keeps the
    "acquired-so-far" trace looking continuous and avoids instantly re-crossing
    the edge it just found.
    Returns (x_vec, y_vec, theta_vec) with theta_vec absolute (not relative to
    start_angle), which is what edge-crossing code wants.
    """
    arc_length = 2 * np.pi * radius * arc_fraction
    n_points = max(int(round(arc_length / ds)), 4)
    _check_point_count(n_points, arc_length, ds, "circle")
    sweep = 2 * np.pi * arc_fraction
    if direction == "cw":
        sweep = -sweep
    theta_vec = start_angle + np.linspace(0.0, sweep, n_points)

    x_vec = center_xy[0] + radius * np.cos(theta_vec)
    y_vec = center_xy[1] + radius * np.sin(theta_vec)
    return x_vec, y_vec, theta_vec


# --------------------------------------------------------------------------------
# Smooth start/stop -- used by the grid+refine mode (and its connectors) so every
# scan and every move between scans starts and ends at REST: the piezo never sees a
# jump in position or in velocity where two waveforms meet.
# --------------------------------------------------------------------------------
def ease_ends(x_vec: np.ndarray, y_vec: np.ndarray, n_ramp: int):
    """Re-time a path (same shape) so its speed ramps smoothly (sin^2) from 0 up to
    the path's own speed over the first n_ramp samples, and back to 0 over the last
    n_ramp. The path geometry is unchanged; the waveform gets ~n_ramp samples longer.
    n_ramp < 2 (or a very short path) returns the path unchanged."""
    x_vec = np.asarray(x_vec, dtype=float)
    y_vec = np.asarray(y_vec, dtype=float)
    n = len(x_vec)
    n_ramp = int(n_ramp)
    if n_ramp < 2 or n < 4:
        return x_vec, y_vec
    if n < n_ramp:
        # short path: one cosine from rest to rest, as long as it takes to stay within the
        # full ramps' acceleration (and the path's own speed)
        m = int(np.ceil(max(np.sqrt(np.pi * n * n_ramp), 0.5 * np.pi * n))) + 1
        u = (n - 1) * (0.5 - 0.5 * np.cos(np.pi * np.arange(m) / (m - 1)))
        idx = np.arange(n, dtype=float)
        return np.interp(u, idx, x_vec), np.interp(u, idx, y_vec)
    m = n + n_ramp  # each ramp covers ~n_ramp/2 samples' worth of path
    tau = np.arange(m, dtype=float)
    ramp = np.minimum(np.minimum((tau + 0.5) / n_ramp, (m - 0.5 - tau) / n_ramp), 1.0)
    speed = np.sin(0.5 * np.pi * ramp) ** 2
    u = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    u *= (n - 1) / u[-1]
    idx = np.arange(n, dtype=float)
    return np.interp(u, idx, x_vec), np.interp(u, idx, y_vec)


def ramp_into(x_vec, y_vec, s: int, m: int):
    """Speed-up onto a path at sample s: the path's own samples s-m .. s-1, re-timed to
    ~2m samples whose speed rises smoothly (sin^2) from rest to the path's (one path
    sample per sample) -- the next sample is x_vec[s] at full speed. m is cut to what's
    there (s); returns empty arrays if s == 0."""
    m = int(min(m, s))
    if m < 1:
        return np.empty(0), np.empty(0)
    u = _ramp_u(m)
    idx = np.arange(len(x_vec), dtype=float)
    return np.interp(s - m + u, idx, x_vec), np.interp(s - m + u, idx, y_vec)


def ramp_out_of(x_vec, y_vec, e: int, m: int):
    """Slow-down off a path after its sample e (at full speed): along the path's own
    samples e+1 .. e+m, re-timed to ~2m samples whose speed falls smoothly to rest. m is
    cut to what's there; returns empty arrays at the path's end."""
    n = len(x_vec)
    m = int(min(m, n - 1 - e))
    if m < 1:
        return np.empty(0), np.empty(0)
    u = e + m - _ramp_u(m)[::-1]
    idx = np.arange(n, dtype=float)
    return np.interp(u, idx, x_vec), np.interp(u, idx, y_vec)


def _ramp_u(m: int) -> np.ndarray:
    """Index offsets 0 .. m-1 over 2m samples, speed sin^2-rising from 0 to 1 per sample."""
    k = 2 * m
    speed = np.sin(0.5 * np.pi * (np.arange(k) + 0.5) / k) ** 2
    u = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]))])
    return u * (m - 1) / u[-1] if m > 1 else u * 0.0


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * s))


# --------------------------------------------------------------------------------
# Area scans (grid+refine mode): raster, Lissajous, spiral -- the same shapes as
# PyScanner's own scan types, generated here as user-defined waveforms so the start and
# end of every scan is known exactly and moves between scans can join them smoothly.
# ALL of them move the tip at constant speed = one sample spacing `ds` per sample, i.e.
# VelocityConfig.tip_velocity_V_s (ds = tip_velocity_V_s / sample_rate_hz), everywhere
# on the scan -- including round the raster turnarounds (unless a larger, gentler
# turn_radius is asked for). add_runup() then adds the speed-up / slow-down OUTSIDE the
# scan, so the scan itself is at full speed from its first sample to its last.
# --------------------------------------------------------------------------------
def add_runup(x_vec: np.ndarray, y_vec: np.ndarray, ds: float, n_ease: int):
    """Put a straight run-up before the path (along its starting direction) and a
    run-out after it (along its final direction), each long enough to speed up from /
    slow down to rest over n_ease samples (ease_ends), and apply the ramps there -- so
    the whole ORIGINAL path runs at its own full speed and the waveform still starts
    and ends at rest. Run-up/out length = n_ease * ds / 2 each.
    Returns (x_vec, y_vec, runup_length)."""
    x_vec = np.asarray(x_vec, dtype=float)
    y_vec = np.asarray(y_vec, dtype=float)
    m = int(n_ease) // 2   # run-up samples (at full speed, before re-timing)
    if m < 1 or len(x_vec) < 3:
        return x_vec, y_vec, 0.0

    def unit(dx, dy):
        n = np.hypot(dx, dy)
        return (dx / n, dy / n) if n > 0 else (1.0, 0.0)

    k = min(10, len(x_vec) - 1)
    u0 = unit(x_vec[k] - x_vec[0], y_vec[k] - y_vec[0])
    u1 = unit(x_vec[-1] - x_vec[-1 - k], y_vec[-1] - y_vec[-1 - k])
    t = np.arange(m, 0, -1) * ds
    x = np.concatenate([x_vec[0] - t * u0[0], x_vec, x_vec[-1] + t[::-1] * u1[0]])
    y = np.concatenate([y_vec[0] - t * u0[1], y_vec, y_vec[-1] + t[::-1] * u1[1]])
    # a ramp of 2m output samples covers m samples' worth of path = exactly the run-up
    x, y = ease_ends(x, y, 2 * m)
    return x, y, m * ds


def raster_scan(center_xy, size_x: float, size_y: float, spacing: float, ds: float,
                fast_axis: str = "x", start_corner=(-1, -1), turn_radius=None):
    """Serpentine raster over a size_x x size_y box, lines `spacing` apart (rounded so
    they span the box exactly, at least 2 lines), at CONSTANT speed ds/sample:
      - every scan line crosses the whole box at full speed,
      - each turnaround is a half-turn just outside the box: with turn_radius=None it
        is a semicircle of radius spacing/2, still at full speed (the tip never slows
        down); a larger turn_radius makes it a gentler, wider half-ellipse that slows
        down near its apex (to spacing / (2 turn_radius) x full speed).
    The turnarounds stick out max(turn_radius, spacing/2) past the box ends.
    start_corner = (sign of fast-axis start, sign of slow-axis start), e.g. (-1, -1)
    starts at the minus/minus corner. Starts and ends AT SPEED -- pass through add_runup().
    Returns (x_vec, y_vec, info) -- info: n_lines, spacing, samples, overshoot."""
    if fast_axis not in ("x", "y"):
        raise ValueError("fast_axis must be 'x' or 'y'")
    size_fast, size_slow = (size_x, size_y) if fast_axis == "x" else (size_y, size_x)
    n_lines = max(2, int(round(size_slow / max(spacing, 1e-12))) + 1)
    h = size_slow / (n_lines - 1)
    A = 0.5 * size_fast
    e = max(0.5 * h, turn_radius or 0.0)        # turnaround excursion past the box
    n_line = max(1, int(round(2 * A / ds)))      # samples per line
    n_turn = max(4, int(round(np.pi * e / ds)))  # samples per turnaround
    est = n_lines * n_line + (n_lines - 1) * n_turn + 1
    _check_point_count(est, n_lines * size_fast, ds, "raster_scan")

    fs, ss = [], []
    tau = np.arange(1, n_turn + 1) / n_turn
    line = np.arange(0, n_line + 1) / n_line  # 0..1 inclusive
    for k in range(n_lines):
        d = 1.0 if k % 2 == 0 else -1.0
        s_k = -0.5 * size_slow + k * h
        f = -d * A + d * 2 * A * (line if k == 0 else line[1:])
        fs.append(f)
        ss.append(np.full(len(f), s_k))
        if k < n_lines - 1:
            # half-turn: fast axis out by e and back (speed pi e / n_turn = ds at both
            # ends), slow axis steps h smoothly; e = h/2 -> an exact semicircle at speed ds
            fs.append(d * (A + e * np.sin(np.pi * tau)))
            ss.append(s_k + h * 0.5 * (1.0 - np.cos(np.pi * tau)))
    fast = np.concatenate(fs)
    slow = np.concatenate(ss)
    sf, sl = start_corner
    fast = fast * (1 if sf < 0 else -1)
    slow = slow * (1 if sl < 0 else -1)
    if fast_axis == "x":
        x_vec, y_vec = center_xy[0] + fast, center_xy[1] + slow
    else:
        x_vec, y_vec = center_xy[0] + slow, center_xy[1] + fast
    return x_vec, y_vec, dict(n_lines=n_lines, spacing=h, samples=len(x_vec), overshoot=e)


def clipped_raster_scan(slow0: float, spacing: float, extents, ds: float, fast_axis: str = "x",
                        start_xy=None, turn_radius=None):
    """Serpentine raster whose lines each cover only [f0, f1] of the fast axis -- a
    raster clipped to an irregular region. Line k sits at slow = slow0 + k * spacing and
    covers extents[k] = (f0, f1). Two neighbouring lines share their turnaround, so on
    that side both reach the outer of their two ends; the turnaround is a semicircle of
    radius spacing/2 at full speed, or a wider half-ellipse with turn_radius (as in
    raster_scan). Equal extents = raster_scan.
    start_xy: the path (or its reverse -- both are valid scans) starts at the end nearer
    this point. Starts and ends AT SPEED -- pass through add_runup().
    Returns (x_vec, y_vec, info) -- info: n_lines, spacing, samples, overshoot."""
    if fast_axis not in ("x", "y"):
        raise ValueError("fast_axis must be 'x' or 'y'")
    ext = np.asarray(extents, dtype=float).reshape(-1, 2)
    n_lines = len(ext)
    if n_lines < 1:
        raise ValueError("clipped_raster_scan: no lines")
    h = float(spacing)
    e = max(0.5 * h, turn_radius or 0.0)
    lo, hi = ext[:, 0].copy(), ext[:, 1].copy()
    # line k runs +fast for even k, -fast for odd k; its END side is shared with line k+1
    for k in range(n_lines - 1):
        if k % 2 == 0:
            hi[k] = hi[k + 1] = max(hi[k], hi[k + 1])
        else:
            lo[k] = lo[k + 1] = min(lo[k], lo[k + 1])
    n_turn = max(4, int(round(np.pi * e / ds)))
    est = int(np.sum((hi - lo) / ds)) + (n_lines - 1) * n_turn + 1
    _check_point_count(est, float(np.sum(hi - lo)), ds, "clipped_raster_scan")
    tau = np.arange(1, n_turn + 1) / n_turn
    fs, ss = [], []
    for k in range(n_lines):
        s_k = slow0 + k * h
        a, b = (lo[k], hi[k]) if k % 2 == 0 else (hi[k], lo[k])
        n_line = max(1, int(round(abs(b - a) / ds)))
        u = np.arange(0, n_line + 1) / n_line
        f = a + (b - a) * (u if k == 0 else u[1:])
        fs.append(f)
        ss.append(np.full(len(f), s_k))
        if k < n_lines - 1:
            d = 1.0 if k % 2 == 0 else -1.0
            fs.append(b + d * e * np.sin(np.pi * tau))
            ss.append(s_k + h * 0.5 * (1.0 - np.cos(np.pi * tau)))
    fast, slow = np.concatenate(fs), np.concatenate(ss)
    x_vec, y_vec = (fast, slow) if fast_axis == "x" else (slow, fast)
    if start_xy is not None:
        p = np.asarray(start_xy, dtype=float)
        if np.hypot(x_vec[-1] - p[0], y_vec[-1] - p[1]) < np.hypot(x_vec[0] - p[0], y_vec[0] - p[1]):
            x_vec, y_vec = x_vec[::-1].copy(), y_vec[::-1].copy()
    return x_vec, y_vec, dict(n_lines=n_lines, spacing=h, samples=len(x_vec), overshoot=e)


def lissajous_scan(center_xy, size_x: float, size_y: float, spacing: float, ds: float,
                   phi_rad=None):
    """Lissajous over a size_x x size_y box -- PyScanner's form
        x = A sin(2 pi w1 t + phi),   y = B sin(2 pi w2 t)
    with whole numbers of cycles a (x) and b = a + 1 (y) over the scan, so the curve
    closes on itself. a is chosen so the largest unscanned gap matches a raster with
    lines `spacing` apart (a ~ 0.75 x size / spacing; checked numerically). phi = 0
    (the default) covers most evenly for b = a + 1 -- other phases leave gaps up to
    ~2x bigger.
    Re-timed to CONSTANT speed ds/sample along the curve: the shape is exactly the
    Lissajous, only the timing differs from pure sines (so w1/w2 are averages).
    Starts and ends at speed -- pass through add_runup().
    Returns (x_vec, y_vec, info) -- info: a, b cycles, phi_rad, samples."""
    A, B = 0.5 * size_x, 0.5 * size_y
    a = max(1, int(np.ceil(0.75 * max(size_x, size_y) / max(spacing, 1e-12))))
    b = a + 1
    if phi_rad is None:
        phi_rad = 0.0
    # arc length on a fine grid of the curve parameter, then invert it (as spiral_out does)
    n_fine = int(np.clip(4000 * (a + b), 20_000, 5_000_000))
    u = np.linspace(0.0, 1.0, n_fine)
    speed = 2 * np.pi * np.hypot(a * A * np.cos(2 * np.pi * a * u + phi_rad),
                                 b * B * np.cos(2 * np.pi * b * u))
    s_cum = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(u))])
    n = max(16, int(round(s_cum[-1] / ds)) + 1)
    _check_point_count(n, s_cum[-1], ds, "lissajous_scan")
    uu = np.interp(np.linspace(0.0, s_cum[-1], n), s_cum, u)
    x_vec = center_xy[0] + A * np.sin(2 * np.pi * a * uu + phi_rad)
    y_vec = center_xy[1] + B * np.sin(2 * np.pi * b * uu)
    return x_vec, y_vec, dict(a=a, b=b, phi_rad=phi_rad, samples=n)


def spiral_scan(center_xy, radius: float, spacing: float, ds: float, direction: str = "ccw",
                min_radius: float = None):
    """Archimedean spiral out to `radius`, turns `spacing` apart, constant speed
    (ds/sample). Starts one spacing out from the center (the very center of a
    constant-speed spiral would need an impossibly fast turn) -- or at min_radius if
    that's larger (the tightest curve: acceleration v^2 / r, e.g. min_radius = v^2 / a_max;
    the disc inside it is left out). Starts and ends at speed -- pass through add_runup().
    Returns (x_vec, y_vec, info)."""
    r0 = min(max(spacing, min_radius or 0.0), 0.5 * radius)
    n_cycles = max((radius - r0) / max(spacing, 1e-12), 0.5)
    x_vec, y_vec, _, _ = spiral_out(center_xy, r0, radius, n_cycles, ds, direction)
    return x_vec, y_vec, dict(n_cycles=n_cycles, spacing=(radius - r0) / n_cycles, samples=len(x_vec))
