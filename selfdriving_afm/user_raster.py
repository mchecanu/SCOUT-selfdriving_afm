"""The reference raster for a grid_refine run, as USER-DEFINED waveforms through the run's
own executor: the same tip speed, the same smooth run-ups / run-outs and turnarounds
(trajectory.raster_scan with cfg.grid_refine.raster_turn_radius_V), the same drive
correction and position sensors as the run -- only the path differs. Unlike PyScanner's
fast raster (reference_raster.plan/acquire) nothing is placed with PyScanner's scan
offsets, and every motion starts and ends at rest.

As many lines (along x) over the run's window as it takes to measure as many pixels of
its pixel grid as the run did. Sent in bands of a few lines (each band one send(), at
most cfg.safety.max_samples_per_waveform samples per do_scan), so RAM holds one band's
record at a time; history keeps block means (cfg.readback.store_every) like the run.

    rp = user_raster.plan(RUN_NPZ, cfg, executor)      # nothing moves
    user_raster.ram_check(rp, cfg)
    ref = user_raster.acquire(rp, executor, cfg)       # engaged (afm.run)
    reference_raster.save_into(RUN_NPZ, ref)
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import trajectory
from .config import ExperimentConfig, amplitude_limits
from .reconstruction import bin_image, inpaint_binned
from .reference_raster import TRANSFER_BYTES_PER_SAMPLE

BYTES_STORED = 56
# Every band ends with the tip held at rest (past the end of its last line, outside the
# area) for this long: PyScanner's AI record can come back short at the END of a waveform
# (35k-154k samples seen, i.e. up to ~155 ms at 1 MS/s), which cut the end of each band's
# last line (manuscript Fig. S13). The dwell absorbs that loss; and after every band the
# lines are checked -- a line not recorded to its end is re-scanned WHOLE in the next band,
# so the raster is only ever cut at the end of a line (2026-11-09).
TAIL_DWELL_S = 0.3     # per stored sample: x, y, z, 2 sensor positions, 2 raw sensor volts (float64)


def plan(npz_path, cfg: ExperimentConfig, executor, n_lines: Optional[int] = None, area=None,
         tail_dwell_s: float = TAIL_DWELL_S) -> dict:
    """Lines, bands and the predicted samples / time / extent. Nothing moves. area: the
    run's scanned area (session.planner.bounds) -- the raster covers that (binned on the
    run's pixel grid over its reconstruction window); None = the whole window."""
    with np.load(npz_path, allow_pickle=True) as d:
        if "binned_Z_sensor" in d.files or "binned_Z" in d.files:
            key = "binned_Z_sensor" if "binned_Z_sensor" in d.files else "binned_Z"
            binned = np.asarray(d[key])
            bounds = tuple(float(v) for v in d["recon_bounds"])
        else:   # gp_path runs: bin the (measured) points on the live view's grid over their window
            pts = d["points"]
            b = tuple(float(v) for v in d["recon_bounds"]) if "recon_bounds" in d.files else None
            binned, bounds = bin_image(pts[:, :2], pts[:, 2], cfg.live_view.n_image_pixels, b)
            bounds = tuple(float(v) for v in bounds)
    n = binned.shape[0]
    M = int(np.isfinite(binned).sum())
    k = int(n_lines) if n_lines else max(2, int(np.ceil(M / n)))
    x0, x1, y0, y1 = bounds
    if area is not None:
        x0, x1 = max(x0, area[0]), min(x1, area[1])
        y0, y1 = max(y0, area[2]), min(y1, area[3])
    w, h = x1 - x0, y1 - y0
    ys = np.linspace(y0, y1, k)
    sp = float(ys[1] - ys[0])
    v = cfg.velocity.tip_velocity_V_s
    rate = cfg.velocity.sample_rate_hz
    ds = v / rate
    r = cfg.grid_refine.raster_turn_radius_V
    e = max(0.5 * sp, r or 0.0)
    n_ease = int(round(cfg.grid_refine.ease_time_s * rate))
    line_n = w / ds + np.pi * e / ds                      # one line + one turnaround
    max_n = int(cfg.safety.max_samples_per_waveform)
    per_band = max(2, int((0.9 * max_n - tail_dwell_s * rate - 2 * n_ease) // line_n))   # one do_scan per band
    bands = [(int(a), int(min(a + per_band, k))) for a in range(0, k, per_band)]
    if bands[-1][1] - bands[-1][0] < 2 and len(bands) > 1:   # a band needs 2 lines: merge
        a, _ = bands.pop()
        bands[-1] = (bands[-1][0], k)
    run_out = 0.5 * n_ease * ds
    lx, ly = amplitude_limits(cfg.safety)
    u = np.asarray(getattr(executor, "_u_target", np.zeros(2)), dtype=float) if executor.correcting else np.zeros(2)
    reach_x = max(abs(x0), abs(x1)) + e + run_out
    reach_y = max(abs(y0), abs(y1))
    drive_x = max(abs(x0 - e - run_out - u[0]), abs(x1 + e + run_out - u[0]))    # drive ~ target - u
    drive_y = max(abs(y0 - u[1]), abs(y1 - u[1]))
    samples = int(k * line_n + len(bands) * (2 * n_ease + tail_dwell_s * rate))
    return dict(npz_path=str(npz_path), bounds=bounds, area=(x0, x1, y0, y1), n=n, run_pixels=M, n_lines=k, spacing_V=sp,
                fast_axis="x", line_coords_V=ys, speed_V_s=v, line_time_s=line_n / rate, bands=bands,
                turn_V=e, samples=samples, scan_time_s=samples / rate, path_V=float(k * w + (k - 1) * np.pi * e),
                samples_per_scan=min(max_n, int(per_band * line_n + 2 * n_ease + tail_dwell_s * rate)), scans=[],
                points_per_line=n, reach_V=(reach_x, reach_y), drive_reach_V=(drive_x, drive_y),
                limits_V=(lx, ly), tail_dwell_s=float(tail_dwell_s))


def line_end_indices(x, cx: float, w: float, tol: float = 1e-9) -> np.ndarray:
    """Sample indices at which each raster line reaches the far end of the area: every
    exit of the fast-axis coordinate from [cx - w/2, cx + w/2] (the turnarounds, run-up and
    run-out are all outside it). The last sample still inside is returned."""
    inside = np.abs(np.asarray(x) - cx) <= 0.5 * w + tol
    return np.flatnonzero(inside[:-1] & ~inside[1:])


def describe(p: dict) -> str:
    x0, x1, y0, y1 = p["area"]
    return (f"User-defined reference raster over x {x0:+.3f}..{x1:+.3f}, y {y0:+.3f}..{y1:+.3f} V: {p['n_lines']} lines "
            f"along x, {p['spacing_V']:.4f} V apart, {p['speed_V_s']:g} V/s, turnarounds {p['turn_V']:.3g} V wide; "
            f"{len(p['bands'])} band(s) of <= {p['bands'][0][1] - p['bands'][0][0]} lines; ~{p['scan_time_s']:.0f} s "
            f"of scanning ({p['samples'] / 1e6:.0f} M samples) for the run's {p['run_pixels']:,} pixels\n"
            f"  reaches x {p['reach_V'][0]:.3f} / y {p['reach_V'][1]:.3f} V (target); with the current drive "
            f"correction ~x {p['drive_reach_V'][0]:.3f} / y {p['drive_reach_V'][1]:.3f} V; limits x "
            f"{p['limits_V'][0]:.3f} / y {p['limits_V'][1]:.3f} V")


def ram_check(p: dict, cfg: ExperimentConfig, fraction: float = 0.5) -> bool:
    """Before acquiring: one do_scan's record in transfer + what history keeps, against the
    free RAM now -- and the drive inside the safety limits. True = fine."""
    import psutil
    vm = psutil.virtual_memory()
    k = max(int(cfg.readback.store_every), 1)
    need = p["samples_per_scan"] * TRANSFER_BYTES_PER_SAMPLE + p["samples"] / k * BYTES_STORED * 3
    ok_ram = need < fraction * vm.available
    ok_lim = p["drive_reach_V"][0] < p["limits_V"][0] and p["drive_reach_V"][1] < p["limits_V"][1]
    print(f"RAM: {vm.available / 1e9:.1f} GB free of {vm.total / 1e9:.1f} GB; the raster needs ~{need / 1e9:.2f} GB "
          f"(one do_scan of <= {p['samples_per_scan']:,} samples in transfer + {p['samples'] / k / 1e6:.1f} M stored "
          f"samples) -- " + ("OK" if ok_ram else f"TOO MUCH (over {fraction:.0%} of the free RAM): close programs or "
                                                 f"raise cfg.readback.store_every"))
    print("Range: " + ("OK, inside the safety limits" if ok_lim else "TOO CLOSE to the safety limits"))
    return ok_ram and ok_lim


def acquire(p: dict, executor, cfg: ExperimentConfig, min_free_GB: float = 1.5) -> dict:
    """Run the raster band by band on `executor` (engaged). The raster's steps are taken
    out of executor.history afterwards (the run's history is left as it was). Stops before
    a band if less than min_free_GB of RAM is free (keeps what it has)."""
    import gc
    import psutil
    ex = executor
    v = cfg.velocity.tip_velocity_V_s
    rate = cfg.velocity.sample_rate_hz
    ds = v / rate
    n_ease = int(round(cfg.grid_refine.ease_time_s * rate))
    ex.connector_ease_samples = max(ex.connector_ease_samples, n_ease)
    x0, x1, y0, y1 = p["area"]
    cx, w = 0.5 * (x0 + x1), x1 - x0
    ys = p["line_coords_V"]
    hooks = (ex.on_step, ex.on_before_step)
    dwell0, rem0 = getattr(ex, "tail_dwell_s", 0.0), getattr(ex, "remeasure_short_tail", True)
    dwell = float(p.get("tail_dwell_s", TAIL_DWELL_S))
    k = int(p["n_lines"])
    per_band = max(2, max(e_ - a_ for a_, e_ in p["bands"]))
    n0 = len(ex.history)
    t0 = time.time()
    done = 0
    band_log = []
    print(f"Acquiring the reference raster: {k} lines in ~{len(p['bands'])} band(s) of <= {per_band} lines, "
          f"~{p['scan_time_s']:.0f} s of scanning; every band ends with {dwell:.2f} s at rest, cut only at line ends ...")
    try:
        ex.on_step = ex.on_before_step = None           # no live view (the run's image stays)
        ex.tail_dwell_s, ex.remeasure_short_tail = dwell, False
        a = 0
        fails = 0
        while a < k:
            free = psutil.virtual_memory().available / 1e9
            if free < min_free_GB:
                print(f"  STOPPED before line {a + 1}: only {free:.1f} GB of RAM free -- keeping lines 1-{a}")
                break
            e = min(a + per_band, k)
            if e - a < 2:                  # a band needs 2 lines: re-scan the one before as well
                a = e - 2
            yc = 0.5 * (ys[a] + ys[e - 1])
            hgt = float(ys[e - 1] - ys[a])
            # every band runs upwards (line a first), so the lines recorded to their end are always
            # lines a .. a+complete-1; the fast-axis start is the end nearest the tip
            best = None
            for corner in ((-1, -1), (1, -1)):
                st = np.array([cx + 0.5 * corner[0] * w, yc + 0.5 * corner[1] * hgt])
                dist = np.linalg.norm(st - ex.last_pos)
                if best is None or dist < best[0]:
                    best = (dist, corner)
            x, y, info = trajectory.raster_scan((cx, yc), w, hgt, p["spacing_V"], ds, "x", best[1],
                                                cfg.grid_refine.raster_turn_radius_V)
            x, y, _ = trajectory.add_runup(x, y, ds, n_ease)
            ends = line_end_indices(x, cx, w)          # sample at which each line reaches its far end
            nb = e - a
            if len(ends) != nb:
                raise RuntimeError(f"band {a + 1}-{e}: found {len(ends)} line ends, expected {nb}")
            z = ex.send(x, y, kind="raster", meta={"label": f"Reference raster, lines {a + 1}-{e} of {k}"})
            rec = len(z)
            complete = int(np.sum(ends < rec))           # lines recorded to their end
            band_log.append(dict(lines=(a + 1, e), sent=len(x), recorded=rec, complete=complete))
            del x, y, z
            gc.collect()
            if complete < nb:
                print(f"  lines {a + 1}-{e}: recorded {rec:,} samples, {ends[-1] + 1:,} needed to the "
                      f"last line's end -- {complete} of {nb} lines complete; re-scanning from line {a + complete + 1}")
                fails += 1
                ex.tail_dwell_s = min(max(1.5 * ex.tail_dwell_s, ex.tail_dwell_s + 0.1), 1.5)
                print(f"  (dwell at the end of each band raised to {ex.tail_dwell_s:.2f} s)")
                if fails > 5:
                    print("  STOPPED: the record keeps stopping before the end of the band's lines")
                    done = a + complete
                    break
            else:
                fails = 0
            done = a + complete
            a = done
            print(f"  lines 1-{done} of {k} done ({time.time() - t0:.0f} s; python "
                  f"{psutil.Process().memory_info().rss / 1e9:.1f} GB, {psutil.virtual_memory().available / 1e9:.1f} GB free)")
    finally:
        ex.on_step, ex.on_before_step = hooks
        ex.tail_dwell_s, ex.remeasure_short_tail = dwell0, rem0
        steps = ex.history[n0:]
        del ex.history[n0:]
    wall = time.time() - t0
    st = [s_ for s_ in steps if s_.kind == "raster"]
    if not st:
        raise RuntimeError("No raster lines acquired")
    xd = np.concatenate([s_.x for s_ in st]); yd = np.concatenate([s_.y for s_ in st])
    z = np.concatenate([s_.z for s_ in st])
    xt, yt = ex.to_target(xd, yd)          # the drive -> target positions (the run's frame)
    b = p["bounds"]
    n = p["n"]
    binned, _ = bin_image(np.column_stack([xt, yt]), z, n, b)
    recon = inpaint_binned(binned, b, cfg.live_view, mode="raster_columns")
    out = dict(binned=binned, recon=recon, x=xt.astype(np.float32), y=yt.astype(np.float32), z=z.astype(np.float32),
               t=np.arange(len(z), dtype=np.float32) * float(st[0].meta.get("store_every", 1)) / rate,
               delay_s=0.0, delay_note="user-defined raster: Z read with the sensors (no shift)",
               every_V=float(st[0].meta.get("store_every", 1)) * ds, wall_time_s=wall, plan=p, lines_done=done,
               samples=int(sum(len(s_.z) for s_ in st) * st[0].meta.get("store_every", 1)),
               generator="user_defined_raster", band_log=band_log)
    if all(s_.x_sensor is not None for s_ in st):
        xs = np.concatenate([s_.x_sensor for s_ in st]); ysn = np.concatenate([s_.y_sensor for s_ in st])
        ok = np.isfinite(xs) & np.isfinite(ysn)
        bs, _ = bin_image(np.column_stack([xs[ok], ysn[ok]]), z[ok], n, b)
        out.update(binned_sensor=bs, recon_sensor=inpaint_binned(bs, b, cfg.live_view, mode="raster_columns"),
                   xs=xs[ok].astype(np.float32), ys=ysn[ok].astype(np.float32), zs=z[ok].astype(np.float32))
        print(f"  at the measured positions (sensors): {int(np.isfinite(bs).sum()):,} pixels; the tip went "
              f"x {np.nanmin(xs):+.3f}..{np.nanmax(xs):+.3f}, y {np.nanmin(ysn):+.3f}..{np.nanmax(ysn):+.3f} V")
    # every extra signal (PFM amplitude / phase: executor.extra_channels), at the commanded and the
    # measured positions -- the same format as reference_raster's (save_into: raster_ch_<name>_*)
    names = sorted({k_ for s_ in st if s_.channels for k_ in s_.channels})
    chans = {}
    for k_ in names:
        vc = np.concatenate([s_.channels[k_] if s_.channels and k_ in s_.channels else np.full(len(s_.z), np.nan)
                             for s_ in st])
        Bc, _ = bin_image(np.column_stack([xt, yt]), vc, n, b)
        d_ = dict(binned=Bc, recon=inpaint_binned(Bc, b, cfg.live_view, mode="raster_columns"),
                  x=xt.astype(np.float32), y=yt.astype(np.float32), v=vc.astype(np.float32))
        if "xs" in out:
            Bs, _ = bin_image(np.column_stack([xs[ok], ysn[ok]]), vc[ok], n, b)
            d_.update(binned_sensor=Bs, recon_sensor=inpaint_binned(Bs, b, cfg.live_view, mode="raster_columns"),
                      xs=out["xs"], ys=out["ys"], vs=vc[ok].astype(np.float32))
        chans[k_] = d_
        print(f"  {k_}: {np.nanpercentile(vc, 1):.4g} .. {np.nanpercentile(vc, 99):.4g} (1..99%)")
    if chans:
        out["channels"] = chans
    print(f"  done in {wall:.0f} s: lines 1-{done} of {p['n_lines']}, {int(np.isfinite(binned).sum()):,} pixels "
          f"(the run: {p['run_pixels']:,})")
    return out
