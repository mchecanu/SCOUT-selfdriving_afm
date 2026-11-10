"""
A reference raster, measured on the instrument right after a self-driving run with
PyScanner's own FAST RASTER (scan_type 2, fast_raster_scan_control_cluster): over the
same area, on the same pixel grid, with (about) the SAME NUMBER OF MEASURED PIXELS as
the run -- the classic scan you'd have done with the same budget. It's saved into the
run's own .npz (raster_* arrays), so Analyze_SelfDriving_Scan.ipynb can compare the two
on real data.

    plan = reference_raster.plan("scan_data/scan_edge_explorer_....npz", cfg)
    print(reference_raster.describe(plan))          # lines, spacing, time -- before moving
    ref = reference_raster.acquire(plan, backend, cfg, tip_xy=executor.last_pos)
    reference_raster.save_into("scan_data/scan_edge_explorer_....npz", ref)
    reference_raster.plot("scan_data/scan_edge_explorer_....npz")

Pixels: a full raster line crosses the whole area, so it measures one pixel row (n
pixels); a run that measured M pixels gets ceil(M / n) lines, spread evenly over the
area, the first on the area's low edge and the last on its high edge (the fair
comparison for the reconstruction -- the same as Simulation.ipynb's "sparse" raster).

PyScanner's fast raster (checked on hardware, see pyscanner_scans.py): every line is
trace + retrace (a full cosine cycle, both measured), the slow axis steps by one line at
the line's start; lines along y are the same raster rotated 90 deg. PyScanner generates
the path itself, so it's one do_scan per raster instead of many user-defined pieces.
The whole raster is ONE fast raster (one do_scan), like setting the tab by hand:
raster_scan_size_x/y_V = the area, raster_N_scan_lines = the lines, raster_scan_duration_s
= duration_s (default 1 s per line), raster_type = 2 (sinusoidal), scan_x/y_offset_V = the
area's centre. It's kept at least 150,000 samples long (PyScanner pads shorter scans
with zeros, sending the tip to (0, 0)).

Position sensors (cfg.position_sensors on, calibrated in the setup): the raster records
X/Y sensors with Z, from the same AI record -- Z sample i was measured at sensor sample
i, no time shift to fit. A second image is made at the MEASURED positions: along each
line the sensors' fast-axis position, across it the line's measured (mean) position, so
each line stays one pixel row (its column-wise reconstruction and the Analyze
animation work as for the commanded one). load() returns that one when it's there
(positions="auto"); the commanded image (time-shift fitted) is kept as before. The
raster itself is sent uncorrected (no drive correction), as planned.
"""

from __future__ import annotations
import dataclasses
import json
import os
import pathlib
import time
from typing import Optional

import numpy as np

from . import pyscanner_scans as ps
from .config import ExperimentConfig
from .execution import ContinuousExecutor
from .reconstruction import bin_image, inpaint_binned, pixel_index

MIN_SAMPLES_PER_SCAN = 150_000          # PyScanner pads shorter scans with zeros (tip -> (0, 0))
# Points per line: every line is sampled at the full rate (1 M samples per 1 s line), far
# more than the run's pixel grid (200) can show. The raster is binned ALONG its lines on
# a finer grid -- this many points per line -- for its own images (load(); the lines and
# their count are unchanged, so is the scan time). The pixel budget (lines) is still
# counted on the run's grid.
POINTS_PER_LINE = 1000


# RAM: what one PyScanner do_scan costs while its record comes back (COM hands AI_vs_time --
# 4 channels -- and AO_vs_time / AO_vs_time_shifted as Python floats, then numpy copies):
# a conservative estimate per sample, used to size the raster's chunks (ram_check)
TRANSFER_BYTES_PER_SAMPLE = 400


def max_lines_for_ram(cfg: ExperimentConfig, line_time_s: float = 1.0, fraction: float = 0.35,
                      available_bytes: Optional[float] = None) -> int:
    """Lines per do_scan so one chunk's transfer stays under `fraction` of the free RAM."""
    if available_bytes is None:
        import psutil
        available_bytes = psutil.virtual_memory().available
    per_line = line_time_s * cfg.velocity.sample_rate_hz * TRANSFER_BYTES_PER_SAMPLE
    return int(max(2, fraction * available_bytes // per_line))


def _chunks(k: int, max_lines: Optional[int]):
    """[(first, end)] line ranges, as even as possible, each at least 2 lines."""
    if not max_lines or max_lines >= k:
        return [(0, k)]
    n = int(np.ceil(k / max(2, int(max_lines))))
    n = max(1, min(n, k // 2))
    e = np.round(np.linspace(0, k, n + 1)).astype(int)
    return [(int(e[i]), int(e[i + 1])) for i in range(n)]


def plan(npz_path, cfg: ExperimentConfig, fast_axis: str = "x", n_lines: Optional[int] = None,
         duration_s: Optional[float] = None, line_time_s: float = 1.0,
         points_per_line: int = POINTS_PER_LINE, max_lines_per_scan="auto") -> dict:
    """PyScanner fast raster(s) matching the run saved in npz_path: over its
    reconstruction window (recon_bounds), as many lines as it takes to measure as many
    pixels of its pixel grid (binned_Z, e.g. 200 x 200) as the run did -- ceil(M / n)
    lines -- (n_lines overrides). Duration: duration_s, or line_time_s per line (1 s
    per line by default, like 20 lines in 20 s). Nothing moves.
    max_lines_per_scan: the raster is cut into consecutive fast rasters (one do_scan each)
    of at most this many lines -- RAM: each do_scan's record comes back whole. "auto" =
    from the free RAM now (max_lines_for_ram); None = one raster."""
    if fast_axis not in ("x", "y"):
        raise ValueError(f"fast_axis must be 'x' or 'y', not {fast_axis!r}")
    with np.load(npz_path, allow_pickle=True) as d:   # closed again: save_into rewrites this file
        if "binned_Z" in d.files and "recon_bounds" in d.files:
            binned = np.asarray(d["binned_Z"])
            bounds = tuple(float(v) for v in d["recon_bounds"])
        else:   # gp_path runs: bin the points on the live view's grid over their window
            pts = d["points"]
            b = tuple(float(v) for v in d["recon_bounds"]) if "recon_bounds" in d.files else None
            binned, bounds = bin_image(pts[:, :2], pts[:, 2], cfg.live_view.n_image_pixels, b)
    n = binned.shape[0]
    M = int(np.isfinite(binned).sum())
    x0, x1, y0, y1 = bounds
    w, h = x1 - x0, y1 - y0
    k = int(n_lines) if n_lines else max(2, int(np.ceil(M / n)))
    k = int(np.clip(k, 2, n))
    rate = cfg.velocity.sample_rate_hz
    fast_len, slow_len, (s0, s1) = (w, h, (y0, y1)) if fast_axis == "x" else (h, w, (x0, x1))
    coords = np.linspace(s0, s1, k)                  # the lines' slow-axis positions
    dur = float(duration_s) if duration_s else k * float(line_time_s)
    dur = max(dur, MIN_SAMPLES_PER_SCAN / rate)      # PyScanner pads shorter scans with zeros
    T_line = dur / k
    if max_lines_per_scan == "auto":
        max_lines_per_scan = max_lines_for_ram(cfg, T_line)
    c = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    scans, notes, worst = [], [], 0.0
    for a, e in _chunks(k, max_lines_per_scan):
        lc = coords[a:e]
        mid = 0.5 * (lc[0] + lc[-1])
        cc = (c[0], mid) if fast_axis == "x" else (mid, c[1])
        # lines along y: PyScanner's raster rotated 90 deg (clockwise -- its lines run along -y)
        sc = ps.PyScannerScan("raster", cc, fast_len, float(lc[-1] - lc[0]), (e - a) * T_line,
                              0.0 if fast_axis == "x" else 90.0, n_lines=e - a, raster_type=2)
        sc.info = dict(lines=(a, e), fast_axis=fast_axis)
        if ps.extent(sc) > cfg.safety.max_amplitude_V:
            raise ValueError(f"The reference raster reaches beyond max_amplitude_V={cfg.safety.max_amplitude_V} V")
        # the chunk must run where the timing model (_path_at) puts lines a..e-1
        r_ = min(rate, 2e5)
        qx, qy = ps.predict_path(sc, r_)
        gx, gy = _path_at(np.arange(len(qx)) / r_, _geometry(bounds, fast_axis, lc, T_line))
        worst = max(worst, float(np.max(np.hypot(qx - gx, qy - gy))))
        scans.append(sc)
    if worst > 0.02 * fast_len:
        notes.append(f"the chunks' predicted paths differ from the raster model by up to {worst:.3g} V -- "
                     f"check the timing model before trusting the commanded-position image")
    # what it should measure: the predicted path, binned like the real one will be
    pix = set()
    for sc in scans:
        px, py = ps.predict_path(sc, min(rate, 2e5))
        pix.update(np.unique(pixel_index(*_inside(px, py, bounds), n, bounds)).tolist())
    samples = int(round(dur * rate))
    per_scan = int(round(max(s.duration_s for s in scans) * rate))
    if per_scan > cfg.safety.max_samples_per_waveform:
        notes.append(f"{per_scan:,} samples per do_scan -- more than max_samples_per_waveform="
                     f"{cfg.safety.max_samples_per_waveform:,} (the limit for user-defined waveforms); "
                     f"run as fast rasters anyway")
    return dict(npz_path=str(npz_path), bounds=bounds, n=n, run_pixels=M, n_lines=k,
                spacing_V=float(coords[1] - coords[0]), fast_axis=fast_axis, line_coords_V=coords,
                speed_V_s=float(2 * np.pi * 0.5 * fast_len / T_line), line_time_s=T_line, scans=scans,
                planned_pixels=len(pix), samples=samples, samples_per_scan=per_scan, scan_time_s=dur,
                path_V=float(k * 2 * fast_len), notes=notes, points_per_line=int(max(points_per_line, n)),
                fast_len_V=float(fast_len), model_error_V=worst)


def ram_check(p: dict, cfg: ExperimentConfig, fraction: float = 0.5) -> bool:
    """Before acquiring: the RAM one chunk (do_scan) needs vs what's free now, and what's
    kept for the whole raster. True = fine."""
    import psutil
    vm = psutil.virtual_memory()
    per_scan = p.get("samples_per_scan", p["samples"])
    need = per_scan * TRANSFER_BYTES_PER_SAMPLE
    every = 0.25 * p["fast_len_V"] / (p["points_per_line"] - 1)
    kept = p["path_V"] / every * 6 * 8 * 3     # thinned t, x, y, z, sensors (float64), x3 working copies
    ok = need + kept < fraction * vm.available
    print(f"RAM: {vm.available / 1e9:.1f} GB free of {vm.total / 1e9:.1f} GB. The raster: {len(p['scans'])} "
          f"do_scan(s) of <= {per_scan:,} samples -> ~{need / 1e9:.2f} GB while each one's record comes back "
          f"(freed after it), ~{kept / 1e9:.2f} GB kept for the whole raster -- "
          + ("OK" if ok else f"TOO MUCH (over {fraction:.0%} of the free RAM): re-plan with a smaller "
                              f"max_lines_per_scan, or close other programs"))
    return ok


def _inside(x, y, bounds, tol=1e-9):
    x0, x1, y0, y1 = bounds
    m = (x >= x0 - tol) & (x <= x1 + tol) & (y >= y0 - tol) & (y <= y1 + tol)
    return np.clip(x[m], x0, x1), np.clip(y[m], y0, y1)


def describe(p: dict) -> str:
    x0, x1, y0, y1 = p["bounds"]
    return (f"Reference fast raster for {pathlib.Path(p['npz_path']).name}: {p['n_lines']} lines along "
            f"{p['fast_axis']}, {p['spacing_V']:.3g} V apart, over X {x0:.3g}..{x1:.3g}, Y {y0:.3g}..{y1:.3g} V "
            f"-- {p['planned_pixels']:,} pixels for the run's {p['run_pixels']:,} ({p['n']}x{p['n']} grid). "
            f"One PyScanner fast raster: {p['scan_time_s']:.3g} s ({p['line_time_s']:.3g} s per line, trace + "
            f"retrace; peak tip speed {p['speed_V_s']:.3g} V/s)"
            + "".join(f"\n  NOTE: {t}" for t in p.get("notes", [])))


def acquire(p: dict, backend, cfg: ExperimentConfig, tip_xy=(0.0, 0.0), return_to_origin: bool = True,
            keep_every_V: Optional[float] = None, sensor_cal=None, min_free_GB: float = 1.5) -> dict:
    """Measure the planned fast raster(s) on `backend` (the same one the run used), through
    a fresh ContinuousExecutor -- the run's own history is untouched. The tip is first
    moved smoothly to the raster's start (PyScanner's line-scan move, at the run's tip
    speed from wherever `tip_xy` says it is -- or from (0, 0), whichever is farther), and
    back to (0, 0) afterwards (return_to_origin). One do_scan per chunk of lines
    (plan(max_lines_per_scan)): each chunk's record is thinned right away (one sample per
    keep_every_V of path, default a quarter point) and dropped, so RAM holds one chunk at
    a time. Stops early (keeping what it has) if less than min_free_GB of RAM is free
    before a chunk. Returns the binned image, its raster_columns reconstruction and the
    kept samples, in scan order (float32)."""
    import gc
    import psutil
    if getattr(backend, "run_pyscanner_scan", None) is None:
        raise NotImplementedError(f"{type(backend).__name__} can't run PyScanner's fast raster")
    v = cfg.velocity.tip_velocity_V_s
    rate = cfg.velocity.sample_rate_hz
    px, py = ps.predict_path(p["scans"][0], rate)
    start = np.array([px[0], py[0]])
    del px, py
    dist = max(np.hypot(*(start - np.asarray(tip_xy, dtype=float))), np.hypot(*start))
    backend.move_to(float(start[0]), float(start[1]), transition_time_s=max(dist / v, 0.2))
    # the sensors recorded (positions from the setup's calibration, on the backend), but the
    # raster sent as planned -- no drive correction, so the commanded image is as before
    ps_cfg = dataclasses.replace(cfg.position_sensors, correct_linear=False, correct_offset=False)
    ex = ContinuousExecutor(backend, cfg.safety, cfg.velocity, start_xy=start, position_sensors=ps_cfg,
                            z_signal=cfg.z_signal)
    if sensor_cal is not None:
        ex.sensor_cal = sensor_cal
    # PFM (cfg.pfm on): every signal that's on (amplitude / phase) read with Z, from the same AI record
    pfm_on = getattr(cfg, "pfm", None) is not None and cfg.pfm.enabled
    if pfm_on:
        from .pfm import channel_map
        ex.extra_channels = channel_map(cfg.pfm)
    ex.set_z_units()      # Z in the run's units (nm with cfg.z_signal.units = "nm")
    b = p["bounds"]
    x0, x1, y0, y1 = b
    T = p["line_time_s"]
    P = int(p.get("points_per_line", p["n"]))
    every = keep_every_V if keep_every_V is not None else 0.25 * p.get("fast_len_V", x1 - x0) / (P - 1)
    n_sc = len(p["scans"])
    t0 = time.time()
    print(f"Acquiring the reference fast raster: {p['n_lines']} lines in {p['scan_time_s']:.3g} s, "
          f"{n_sc} do_scan(s) ...")
    parts = []                               # per chunk: thinned samples + its geometry
    n_total = 0
    peak_rss = 0.0
    proc = psutil.Process()
    for j, sc in enumerate(p["scans"]):
        a, e = sc.info["lines"]
        free = psutil.virtual_memory().available / 1e9
        if free < min_free_GB:
            print(f"  STOPPED before lines {a + 1}-{e}: only {free:.1f} GB of RAM free (min_free_GB="
                  f"{min_free_GB}) -- keeping lines 1-{a}")
            break
        x, y, z = ex.send_pyscanner(sc, "raster", check_samples=False, correct_delay=False,
                                    meta={"label": f"Reference raster, lines {a + 1}-{e} of {p['n_lines']}"})
        st = ex._last_full_step
        m = min(len(z), int(round(sc.duration_s * rate)))     # never PyScanner's zero padding
        x, y, z = np.asarray(x[:m]), np.asarray(y[:m]), np.asarray(z[:m])
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        keep = np.unique(np.searchsorted(arc, np.arange(0.0, arc[-1], every)))
        keep = keep[keep < m]
        part = dict(a=a, e=e, t=(keep / rate), z=z[keep].copy(),
                    g=_geometry(b, p["fast_axis"], p["line_coords_V"][a:e], T))
        if st.x_sensor is not None and len(st.x_sensor) >= m:
            part.update(sx=np.asarray(st.x_sensor[:m])[keep].copy(), sy=np.asarray(st.y_sensor[:m])[keep].copy())
        if st.channels:
            part["ch"] = {k_: np.asarray(v_[:m])[keep].copy() for k_, v_ in st.channels.items()}
        parts.append(part)
        n_total += m
        peak_rss = max(peak_rss, proc.memory_info().rss / 1e9)
        # free this chunk's record before the next one
        del x, y, z, arc, keep, st
        ex.history.clear()
        ex._last_full_step = None
        if hasattr(backend, "_last_raw_readback"):
            backend._last_raw_readback = None
        if hasattr(backend, "_last_shifted"):
            backend._last_shifted = None
        gc.collect()
        print(f"  lines {a + 1}-{e} of {p['n_lines']} done ({time.time() - t0:.0f} s; python "
              f"{proc.memory_info().rss / 1e9:.1f} GB, {psutil.virtual_memory().available / 1e9:.1f} GB free)")
    wall = time.time() - t0
    end = ex.last_pos.copy()
    del ex
    if return_to_origin:
        backend.move_to(0.0, 0.0, transition_time_s=max(np.hypot(*end) / v, 0.2))
    if not parts:
        raise RuntimeError("No raster lines acquired")
    # the time shift of Z vs the commanded path: fitted on the chunk with the most structure
    fit_on = max(parts, key=lambda q: float(np.nanpercentile(q["z"], 98) - np.nanpercentile(q["z"], 2)))
    tau, note = fit_delay(fit_on["t"], fit_on["z"], fit_on["g"], _run_binned(p["npz_path"]))
    if len(parts) > 1:
        note += f" (fitted on lines {fit_on['a'] + 1}-{fit_on['e']})"
    print(f"  Z vs the commanded path: {note}")
    X, Y, Z, TT = [], [], [], []
    XS, YS, ZS = [], [], []
    CH = {}
    for q in parts:
        xk, yk, zk, tk = place(q["t"], q["z"], q["g"], tau)
        X.append(xk); Y.append(yk); Z.append(zk); TT.append(tk + q["a"] * T)   # scan time of the whole raster
        if "sx" in q:
            xm, ym = sensor_line_positions(q["sx"], q["sy"], q["t"], q["g"])
            ok = np.isfinite(xm) & np.isfinite(ym)
            XS.append(xm[ok]); YS.append(ym[ok]); ZS.append(q["z"][ok])
        for k_, vv in q.get("ch", {}).items():
            c = CH.setdefault(k_, dict(x=[], y=[], v=[], xs=[], ys=[], vs=[]))
            xc_, yc_, vc_, _ = place(q["t"], vv, q["g"], tau)
            c["x"].append(xc_); c["y"].append(yc_); c["v"].append(vc_)
            if "sx" in q:
                c["xs"].append(xm[ok]); c["ys"].append(ym[ok]); c["vs"].append(vv[ok])
    xk, yk, zk, tk = (np.concatenate(v_) for v_ in (X, Y, Z, TT))
    binned, _ = bin_image(np.column_stack([xk, yk]), zk, p["n"], b)
    recon = inpaint_binned(binned, b, cfg.live_view, mode="raster_columns")
    lines_done = parts[-1]["e"]
    print(f"  done in {wall:.0f} s wall time: {n_total:,} samples (lines 1-{lines_done} of {p['n_lines']}), "
          f"{int(np.isfinite(binned).sum()):,} pixels measured (the run: {p['run_pixels']:,}); python peaked at "
          f"~{peak_rss:.1f} GB")
    out = dict(binned=binned, recon=recon, x=xk.astype(np.float32), y=yk.astype(np.float32),
               z=zk.astype(np.float32), t=tk, delay_s=tau, delay_note=note,
               every_V=every, wall_time_s=wall, plan=p, lines_done=lines_done, samples=n_total)
    if XS and len(XS) == len(parts):
        xm, ym, zm = (np.concatenate(v_) for v_ in (XS, YS, ZS))
        binned_s, _ = bin_image(np.column_stack([xm, ym]), zm, p["n"], b)
        recon_s = inpaint_binned(binned_s, b, cfg.live_view, mode="raster_columns")
        sxa = np.concatenate([q["sx"] for q in parts]); sya = np.concatenate([q["sy"] for q in parts])
        print(f"  at the measured positions (sensors): {int(np.isfinite(binned_s).sum()):,} pixels; the tip went "
              f"{np.nanmin(sxa):+.3f}..{np.nanmax(sxa):+.3f} V in x, {np.nanmin(sya):+.3f}..{np.nanmax(sya):+.3f} V "
              f"in y (planned {b[0]:+.3f}..{b[1]:+.3f}, {b[2]:+.3f}..{b[3]:+.3f})")
        out.update(binned_sensor=binned_s, recon_sensor=recon_s, xs=xm.astype(np.float32), ys=ym.astype(np.float32),
                   zs=zm.astype(np.float32))
    elif cfg.position_sensors.enabled:
        print("  (position sensors on, but no calibrated sensor positions -- commanded positions only)")
    # every PFM signal: at the commanded positions (Z's time shift -- the same AI record) and at
    # the measured ones (sensors, no shift)
    chans = {}
    for k_, c in CH.items():
        xc_, yc_, vc_ = (np.concatenate(c[q]) for q in ("x", "y", "v"))
        Bc, _ = bin_image(np.column_stack([xc_, yc_]), vc_, p["n"], b)
        d_ = dict(binned=Bc, recon=inpaint_binned(Bc, b, cfg.live_view, mode="raster_columns"),
                  x=xc_.astype(np.float32), y=yc_.astype(np.float32), v=vc_.astype(np.float32))
        if c["xs"] and "xs" in out:
            xm2, ym2, vs2 = (np.concatenate(c[q]) for q in ("xs", "ys", "vs"))
            Bs, _ = bin_image(np.column_stack([xm2, ym2]), vs2, p["n"], b)
            d_.update(binned_sensor=Bs, recon_sensor=inpaint_binned(Bs, b, cfg.live_view, mode="raster_columns"),
                      xs=xm2.astype(np.float32), ys=ym2.astype(np.float32), vs=vs2.astype(np.float32))
        chans[k_] = d_
        print(f"  {k_}: {np.nanpercentile(vc_, 1):.4g} .. {np.nanpercentile(vc_, 99):.4g} (1..99%)")
    if chans:
        out["channels"] = chans
    elif pfm_on:
        print("  (cfg.pfm on, but no PFM signals came back with the raster)")
    return out


def sensor_line_positions(sx, sy, t, g):
    """The raster at its MEASURED positions, line by line: along the line the sensors'
    fast-axis position; across it the line's measured slow-axis position (its median), so
    each line stays one straight row of pixels. sx, sy: drive-frame sensor positions,
    aligned with Z (the same AI record); t: scan time of each sample."""
    sx, sy = np.asarray(sx, dtype=float), np.asarray(sy, dtype=float)
    k = len(g["coords"])
    line = np.clip(np.floor(np.asarray(t) / g["T"]), 0, k - 1).astype(int)
    fast, slow = (sx, sy) if g["fast_axis"] == "x" else (sy, sx)
    med = np.full(k, np.nan)
    for j in range(k):
        v = slow[line == j]
        v = v[np.isfinite(v)]
        if len(v):
            med[j] = np.median(v)
    slow_l = med[line]
    return (fast, slow_l) if g["fast_axis"] == "x" else (slow_l, fast)


# ------------------------------------------------------------------------------------
# Z vs position on a sinusoidal fast raster
# ------------------------------------------------------------------------------------
# On the instrument (2026-10-06, 26 lines of 1 s) the AI0 record was shifted in time
# against the commanded path (AO_vs_time): Z at sample i belonged to the tip position at
# time i/rate + ~0.127 s. Trace and retrace then put every feature at two different
# places -- the image looked doubled along the fast axis. The shift is fitted from the
# data (trace and retrace must agree) and Z is re-placed on the path at the shifted
# time. A shift and the same shift minus half a line both make trace and retrace agree
# (the second mirrors the image along the fast axis): the one that matches the
# self-driving run's own image (its reconstruction) is kept (or, if that's unclear, the one with Z ahead in
# time, as seen on the instrument).

def _geometry(bounds, fast_axis, line_coords, line_time_s) -> dict:
    x0, x1, y0, y1 = bounds
    if fast_axis == "x":    # PyScanner: x = cx - A cos(2 pi u), lines at y = coords
        cf, A = 0.5 * (x0 + x1), 0.5 * (x1 - x0)
    else:                   # rotated 90 deg clockwise: y = cy + A cos(2 pi u), lines at x = coords
        cf, A = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
    return dict(bounds=tuple(bounds), fast_axis=fast_axis, coords=np.asarray(line_coords, dtype=float),
                T=float(line_time_s), cf=cf, A=A)


def _path_at(tt, g):
    """The commanded fast-raster position at scan time tt."""
    k = len(g["coords"])
    line = np.clip(np.floor(tt / g["T"]), 0, k - 1).astype(int)
    u = tt / g["T"] - line
    s = -g["A"] * np.cos(2 * np.pi * u)               # along the line, from its start
    slow = g["coords"][line]
    if g["fast_axis"] == "x":
        return g["cf"] + s, slow
    return slow, g["cf"] - s


def times_from_path(x, y, g):
    """Scan time of each (commanded) sample, from where it is on the raster -- for saved
    data that kept positions only (thinned along the path, so the clock is lost)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fast, slow = (x, y) if g["fast_axis"] == "x" else (y, x)
    s = (fast - g["cf"]) if g["fast_axis"] == "x" else (g["cf"] - fast)
    line = np.argmin(np.abs(slow[:, None] - g["coords"][None, :]), axis=1)
    outward = np.gradient(s) > 0                      # first half of the cycle
    ph = np.arccos(np.clip(-s / g["A"], -1, 1)) / (2 * np.pi)
    return (line + np.where(outward, ph, 1 - ph)) * g["T"]


def _mismatch(t, z, g, tau, n=200):
    """mean |trace - retrace| on a (line, n) grid, Z placed at time t + tau"""
    tt = t + tau
    k = len(g["coords"])
    ok = (tt >= 0) & (tt < k * g["T"])
    tt, zz = tt[ok], z[ok]
    line = np.floor(tt / g["T"]).astype(int)
    u = tt / g["T"] - line
    col = np.clip(((1 - np.cos(2 * np.pi * u)) / 2 * n).astype(int), 0, n - 1)
    out = u < 0.5
    img = []
    for m in (out, ~out):
        idx = line[m] * n + col[m]
        c = np.bincount(idx, minlength=k * n)[:k * n]      # (a last sample can round into line k)
        sm = np.bincount(idx, weights=zz[m], minlength=k * n)[:k * n]
        img.append(np.where(c > 0, sm / np.maximum(c, 1), np.nan))
    dd = np.abs(img[0] - img[1])
    good = np.isfinite(dd)
    return float(np.mean(dd[good])) if good.sum() > 0.25 * k * n else np.inf


def _detrended(img):
    m = np.isfinite(img)
    if m.sum() < 10:
        return img
    r, c = np.nonzero(m)
    A = np.column_stack([np.ones(len(r)), r, c])
    coef = np.linalg.lstsq(A, img[m], rcond=None)[0]
    out = np.full(img.shape, np.nan)
    out[m] = img[m] - A @ coef
    return out


def fit_delay(t, z, g, run_binned=None):
    """(tau, note): the Z sample at scan time t belongs to the path at t + tau."""
    t, z = np.asarray(t, dtype=float), np.asarray(z, dtype=float)
    T = g["T"]
    if len(t) > 400_000:                              # plenty for the fit
        sub = np.linspace(0, len(t) - 1, 400_000).astype(int)
        t, z = t[sub], z[sub]
    grid = np.arange(-0.5, 0.5, 1 / 400) * T
    sc = np.array([_mismatch(t, z, g, tau) for tau in grid])
    best = grid[np.argmin(sc)]
    fine = best + np.arange(-1, 1.0001, 0.05) / 400 * T
    scf = np.array([_mismatch(t, z, g, tau) for tau in fine])
    best = float(fine[np.argmin(scf)])
    base = _mismatch(t, z, g, 0.0)
    if _mismatch(t, z, g, best) > 0.8 * base:         # no clear shift: leave it
        return 0.0, f"no time shift found (trace vs retrace differ by {base:.3g} V unshifted)"
    other = best - 0.5 * T if best >= 0 else best + 0.5 * T
    cands = sorted([best, other], key=lambda v: -v)   # Z ahead (tau >= 0) first
    choice, why = cands[0], "Z ahead of the path, as seen on the instrument"
    if run_binned is not None and np.isfinite(run_binned).sum() > 50:
        n = run_binned.shape[0]
        ref = _detrended(run_binned)   # tilt removed from both: the run's may be flattened
        cc = []
        for tau in cands:
            xx, yy, zz, _ = place(t, z, g, tau)
            B, _ = bin_image(np.column_stack([xx, yy]), zz, n, g["bounds"])
            B = _detrended(B)
            ok = np.isfinite(B) & np.isfinite(ref)
            cc.append(np.corrcoef(B[ok], ref[ok])[0, 1] if ok.sum() > 20 else np.nan)
        if np.all(np.isfinite(cc)) and abs(cc[0] - cc[1]) > 0.03:
            choice = cands[int(np.argmax(cc))]
            why = f"matches the self-driving image (correlation {max(cc):.2f}, mirrored {min(cc):.2f})"
        else:
            why += f" (the self-driving image doesn't decide: correlation {cc[0]:.2f} vs {cc[1]:.2f})"
    after = _mismatch(t, z, g, choice)
    return float(choice), (f"shifted by {choice * 1e3:+.1f} ms ({choice / T:+.3f} line): trace vs retrace "
                           f"differ by {base:.3g} -> {after:.3g} V; {why}")


def place(t, z, g, tau):
    """(x, y, z, t): Z at the path position of time t + tau (samples falling outside the
    scan dropped)."""
    tt = np.asarray(t, dtype=float) + tau
    ok = (tt >= 0) & (tt < len(g["coords"]) * g["T"])
    x, y = _path_at(tt[ok], g)
    return x, y, np.asarray(z)[ok], np.asarray(t)[ok]


def _run_binned(npz_path):
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            return _run_image(d)
    except Exception:   # noqa: BLE001
        return None


def _run_image(d):
    """the self-driving run's image to tell a shift from its mirror: its full
    reconstruction if saved (sharper test than the sparse binned pixels), else binned"""
    for k in ("recon_Z", "binned_Z"):
        if k in d.files and np.asarray(d[k]).ndim == 2:
            return np.asarray(d[k], dtype=float)
    return None


def load(d, verbose: bool = True, points_per_line="saved", positions: str = "auto") -> dict:
    """The reference raster of a loaded run file `d` (np.load): binned image, bounds,
    samples. Files acquired before the time-shift correction (no raster_delay_s) with a
    PyScanner fast raster are corrected here, in memory, from their saved samples.
    points_per_line: the image is binned on a points_per_line x points_per_line grid
    (the lines still where they were scanned) -- "saved" = what the acquisition asked for
    (POINTS_PER_LINE; older files: the run's grid), None = the run's grid. Limited to
    what the saved samples support (one sample per point at least).
    positions: "auto" = at the measured positions (sensors) if the file has them, else
    commanded; "sensor" / "commanded" to choose. out["positions"] says which."""
    b = tuple(float(v) for v in d["raster_bounds"])
    use_sensor = positions in ("auto", "sensor") and "raster_xs" in d.files
    if positions == "sensor" and not use_sensor and verbose:
        print("Reference raster: no sensor positions in this file -- commanded positions")
    out = dict(bounds=b, binned=np.asarray(d["raster_binned_Z"]), x=d["raster_x"], y=d["raster_y"],
               z=d["raster_z"], delay_s=float(d["raster_delay_s"]) if "raster_delay_s" in d.files else None,
               corrected_here=False)
    fast = "raster_generator" in d.files and str(d["raster_generator"]) == "pyscanner_fast_raster"
    if fast and out["delay_s"] is None and "raster_line_coords_V" in d.files:
        g = _geometry(b, str(d["raster_fast_axis"]), d["raster_line_coords_V"], float(d["raster_line_time_s"]))
        x, y, z = (np.asarray(d[k], dtype=float) for k in ("raster_x", "raster_y", "raster_z"))
        t = times_from_path(x, y, g)
        tau, note = fit_delay(t, z, g, _run_image(d))
        xk, yk, zk, _ = place(t, z, g, tau)
        binned, _ = bin_image(np.column_stack([xk, yk]), zk, out["binned"].shape[0], b)
        out.update(binned=binned, x=xk, y=yk, z=zk, delay_s=tau, corrected_here=True)
        if verbose:
            print(f"Reference raster (saved before the time-shift correction): {note}")
    out["positions"] = "commanded"
    if use_sensor:     # Z at the measured positions: no time shift involved
        out.update(binned=np.asarray(d["raster_binned_Z_sensor"]), x=d["raster_xs"], y=d["raster_ys"],
                   z=d["raster_zs"], positions="sensor")
    n_run = out["binned"].shape[0]
    if points_per_line == "saved":
        points_per_line = int(d["raster_points_per_line"]) if "raster_points_per_line" in d.files else None
    out.update(points_per_line=n_run, run_grid=n_run)
    if points_per_line and int(points_per_line) != n_run:
        fa = str(d["raster_fast_axis"]) if "raster_fast_axis" in d.files else "x"
        fast_len = (b[1] - b[0]) if fa == "x" else (b[3] - b[2])
        P = int(points_per_line)
        if "raster_samples_every_V" in d.files:      # what the kept samples support
            P_max = int(0.9 * fast_len / float(d["raster_samples_every_V"]))
            if P > P_max:
                if verbose:
                    print(f"Reference raster: {P} points per line asked, the saved samples support "
                          f"{P_max} -- using {P_max}")
                P = P_max
        if P > n_run:
            xs, ys, zs = (np.asarray(out[k], dtype=float) for k in ("x", "y", "z"))
            binned, _ = bin_image(np.column_stack([xs, ys]), zs, P, b)
            out.update(binned=_fill_along_lines(binned, fa), points_per_line=P)
    return out



def for_display(img, rows: int = 200):
    """For showing a binned raster on a fine grid: each measured line drawn as a band
    (about 1/rows of the image tall) instead of one pixel row, which disappears when the
    figure has fewer pixels than the grid. Display only -- never reconstruct from it."""
    from .reconstruction import raster_lines_along
    img = np.asarray(img, dtype=float)
    w = int(img.shape[0] // max(rows, 1))
    if w <= 1:
        return img
    along = raster_lines_along(img)
    A = img if along == "x" else img.T
    out = np.array(A)
    for r in np.nonzero(np.isfinite(A).mean(axis=1) > 0.5)[0]:
        for k in range(max(0, r - w // 2), min(A.shape[0], r + w // 2 + 1)):
            if not np.isfinite(A[k]).any():
                out[k] = A[r]
    return out if along == "x" else out.T

def _fill_along_lines(img, fast_axis):
    """A fine grid can leave a few points of a line without a sample: fill them along the
    line (linear between its neighbours), so the reconstruction across lines doesn't skip
    that line there (streaks). Only the measured lines; nothing between them."""
    out = np.array(img, dtype=float)
    A = out if fast_axis == "x" else out.T          # a view: rows = lines
    for r in np.nonzero(np.isfinite(A).mean(axis=1) > 0.5)[0]:
        m = np.isfinite(A[r])
        if not m.all():
            k = np.arange(A.shape[1])
            A[r, ~m] = np.interp(k[~m], k[m], A[r, m])
    return out


def save_into(npz_path, ref: dict) -> None:
    """Add the reference raster to the run's .npz (raster_* arrays; any earlier ones
    replaced). The file is rewritten -- via a temporary file, so an interrupted save
    can't damage the original."""
    p = ref["plan"]
    path = pathlib.Path(npz_path)
    with np.load(path, allow_pickle=True) as d:
        data = {k: d[k] for k in d.files if not k.startswith("raster_")}
    data.update(
        raster_binned_Z=ref["binned"], raster_recon_Z=ref["recon"], raster_recon_mode="raster_columns",
        raster_generator=ref.get("generator", "pyscanner_fast_raster"), raster_bounds=np.array(p["bounds"]),
        raster_n_lines=p["n_lines"], raster_spacing_V=p["spacing_V"], raster_fast_axis=p["fast_axis"],
        raster_line_coords_V=np.asarray(p["line_coords_V"]), raster_speed_V_s=p["speed_V_s"],
        raster_line_time_s=p["line_time_s"], raster_n_scans=len(p["scans"]),
        raster_scans=np.array([s.describe() for s in p["scans"]]), raster_run_pixels=p["run_pixels"],
        raster_pixels=int(np.isfinite(ref["binned"]).sum()), raster_path_V=p["path_V"],
        raster_scan_time_s=p["scan_time_s"], raster_wall_time_s=ref["wall_time_s"],
        raster_lines_done=int(ref.get("lines_done", p["n_lines"])),
        raster_x=ref["x"], raster_y=ref["y"], raster_z=ref["z"], raster_t=ref["t"],
        raster_delay_s=ref["delay_s"], raster_delay_note=ref["delay_note"], raster_samples_every_V=ref["every_V"],
        raster_points_per_line=int(p.get("points_per_line", p["n"])))
    if "binned_sensor" in ref:      # the same raster at the measured positions (sensors)
        data.update(raster_binned_Z_sensor=ref["binned_sensor"], raster_recon_Z_sensor=ref["recon_sensor"],
                    raster_xs=ref["xs"], raster_ys=ref["ys"], raster_zs=ref["zs"])
    if ref.get("channels"):         # PFM: every signal that was on, raster_ch_<name>_*
        data["raster_channels"] = np.array(list(ref["channels"]))
        for k_, c in ref["channels"].items():
            for key, val in c.items():
                data[f"raster_ch_{k_}_{key}"] = val
    tmp = path.with_name(path.stem + "_tmp_saving.npz")
    np.savez(tmp, **data)
    os.replace(tmp, path)
    print(f"Saved the reference raster into {path.name} ({path.stat().st_size / 1e6:.0f} MB)")


def plot(npz_path, cmap: str = "copper", points_per_line: Optional[int] = POINTS_PER_LINE):
    """The run's reconstruction next to the reference raster (path, binned, reconstruction),
    on one colour scale; the raster binned with points_per_line points along each line."""
    import matplotlib.pyplot as plt
    with np.load(npz_path, allow_pickle=True) as d:
        if "raster_binned_Z" not in d.files:
            raise ValueError(f"{npz_path} has no reference raster")
        R = load(d, points_per_line=points_per_line)   # time-shift corrected (files saved before the correction: here), fine along lines
        b = R["bounds"]
        rb = R["binned"]
        z_label = "Z (nm)" if ("z_unit" in d.files and str(d["z_unit"]) == "nm") else "AI0 (V)"
        if R["corrected_here"] or R["points_per_line"] != R["run_grid"]:
            from .config import LiveViewConfig
            rr_ = inpaint_binned(rb, b, LiveViewConfig(), mode="raster_columns")
        else:
            rr_ = d["raster_recon_Z_sensor"] if R.get("positions") == "sensor" else d["raster_recon_Z"]
        panels = []
        if "recon_Z" in d.files:
            mode = str(d["recon_mode"]) if "recon_mode" in d.files else ""
            panels.append((d["recon_Z"], f"Self-driving run: reconstruction ({mode})\n"
                                         f"{int(d['raster_run_pixels']):,} pixels measured"))
        # the run's saved reconstruction has the fitted substrate (tilt) removed when the
        # live view flattened it -- remove the same plane from the raster, for one scale
        plane = 0.0
        flat = bool(d["recon_flattened"]) if "recon_flattened" in d.files else False
        if flat and "substrate_json" in d.files:
            coef = json.loads(str(d["substrate_json"]))["coef"]
            n = rb.shape[0]
            X, Y = np.meshgrid(np.linspace(b[0], b[1], n), np.linspace(b[2], b[3], n))
            plane = coef[0] + (coef[1] * X + coef[2] * Y if len(coef) >= 3 else 0.0)
            if len(coef) >= 6:
                plane = plane + coef[3] * X * X + coef[4] * X * Y + coef[5] * Y * Y
        tilt = ", tilt removed" if flat else ""
        tilt += ", measured positions" if R.get("positions") == "sensor" else ""
        panels.append((rb - plane, f"Reference raster: binned{tilt}\n"
                                                      f"{int(d['raster_n_lines'])} lines x {R['points_per_line']} points"))
        panels.append((rr_ - plane, f"Reference raster: reconstruction (raster_columns){tilt}"))
        rx, ry = R["x"], R["y"]
        ref = rb - plane
    vmin, vmax = np.nanpercentile(ref, [1, 99])
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.6 * (len(panels) + 1), 4.4), layout="constrained")
    st = max(1, len(rx) // 20000)
    axes[0].plot(rx[::st], ry[::st], lw=0.4, color="tab:red")
    axes[0].set_title("Reference raster path")
    for ax in axes:
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect("equal"); ax.set_xlabel("X (V)")
    for ax, (img, title) in zip(axes[1:], panels):
        im = ax.imshow(img, origin="lower", extent=b, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=list(axes[1:]), label=z_label + (" above the substrate" if flat else ""), shrink=0.85)
    plt.show()
    plot_channels(npz_path)
    return fig


def plot_channels(npz_path):
    """The raster's PFM signals (raster_ch_<name>_*, if any): binned and column-wise
    reconstruction, at the commanded and (if recorded) the measured positions."""
    import matplotlib.pyplot as plt
    with np.load(npz_path, allow_pickle=True) as d:
        if "raster_channels" not in d.files:
            return None
        names = [str(n) for n in d["raster_channels"]]
        pf = json.loads(str(d["pfm_json"])) if "pfm_json" in d.files else {}
        b = tuple(float(v) for v in d["raster_bounds"])
        rows = ["commanded"] + (["measured"] if f"raster_ch_{names[0]}_binned_sensor" in d.files else [])
        fig, axs = plt.subplots(len(rows), 2 * len(names), figsize=(4.4 * 2 * len(names), 4.0 * len(rows)),
                                squeeze=False, layout="constrained")
        for k, n in enumerate(names):
            c = pf.get(n, {})
            cmap = c.get("cmap", "inferno" if n == "phase" else "viridis")
            lab = f"{c.get('label', n)} ({c.get('unit', '')})"
            for r, where in enumerate(rows):
                tag = "" if where == "commanded" else "_sensor"
                B, R = np.asarray(d[f"raster_ch_{n}_binned{tag}"]), np.asarray(d[f"raster_ch_{n}_recon{tag}"])
                lo, hi = (-180.0, 180.0) if n == "phase" else tuple(np.nanpercentile(B, [1, 99]))
                for j, (img, what) in enumerate(((for_display(B), "binned"), (R, "column-wise PCHIP"))):
                    ax = axs[r, 2 * k + j]
                    im = ax.imshow(img, origin="lower", extent=b, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
                    fig.colorbar(im, ax=ax, label=lab, shrink=0.85)
                    ax.set_title(f"raster {n}, {what} -- {where} positions", fontsize=9)
                    ax.set_xlabel("X (V)"); ax.set_ylabel("Y (V)")
    plt.show()
    return fig
