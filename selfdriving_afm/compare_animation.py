"""
Side-by-side animations of a saved run (Analyze_SelfDriving_Scan.ipynb, section 5b/5c):
one row per (scan, positions) -- the self-driving run and its reference raster, each at
the COMMANDED positions (open loop: the drive volts sent) and at the MEASURED ones
(closed loop: the X/Y position sensors, in the calibrated frame) -- like the run's live
view, all rows advancing together:

    path | binned Z | reconstruction | uncertainty

    compare_animation.animate(d, "run_compare.gif", rows=("run_cmd", "run_sensor",
                              "raster_cmd", "raster_sensor"), recon_cfg=cfg)

Frames follow the self-driving run (its waveforms, about max_frames of them); the raster
rows show, in every frame, the raster as far as it takes to measure AS MANY PIXELS (of
the run's grid) as the run has measured so far -- equal budgets, counted at the commanded
positions (the sensors' jitter spreads a path over more pixels), and both raster rows at
the same point of the raster scan. Sensor rows are drawn in the calibrated drive frame with their axes
labelled in um or nm along the sensors (viz.length_axis_ticks). Z: the file's units
(z_unit); the fitted substrate (tilt) removed if the run's reconstruction was flattened.
Rows the file doesn't have (no sensors, no raster) are skipped, with a note.
"""

from __future__ import annotations
import json
import pathlib
import time
from typing import Optional, Sequence

import numpy as np

from .config import LiveViewConfig
from .reconstruction import pixel_index, uncertainty_map, inpaint_binned

ROW_LABELS = {
    "run_cmd": "Self-driving\ncommanded (open loop)",
    "run_sensor": "Self-driving\nmeasured (closed loop)",
    "raster_cmd": "Raster\ncommanded (open loop)",
    "raster_sensor": "Raster\nmeasured (closed loop)",
}


def _sensor_cal(d):
    if "sensor_calibration_json" not in d.files:
        return None
    from .sensors import SensorCalibration
    return SensorCalibration.from_dict(json.loads(str(d["sensor_calibration_json"])))


def _plane(d, bounds, n):
    """The run's fitted substrate on the image grid, if its reconstruction was flattened."""
    if not ("recon_flattened" in d.files and bool(d["recon_flattened"]) and "substrate_json" in d.files):
        return 0.0
    coef = json.loads(str(d["substrate_json"]))["coef"]
    X, Y = np.meshgrid(np.linspace(bounds[0], bounds[1], n), np.linspace(bounds[2], bounds[3], n))
    p = coef[0] + (coef[1] * X + coef[2] * Y if len(coef) >= 3 else 0.0)
    if len(coef) >= 6:
        p = p + coef[3] * X * X + coef[4] * X * Y + coef[5] * Y * Y
    return p


class _Row:
    def __init__(self, key, x, y, z, bounds, n, kinds=None, offsets=None):
        self.key, self.n, self.bounds = key, n, bounds
        self.x, self.y, self.z = x, y, z
        self.kinds, self.offsets = kinds, offsets
        x0, x1, y0, y1 = bounds
        self.inside = np.isfinite(x) & np.isfinite(y) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        self.pix = np.where(self.inside, pixel_index(np.nan_to_num(x), np.nan_to_num(y), n, bounds), 0).astype(np.int32)
        # sample index at which each pixel is first measured (sorted): the pixel budget
        idx = np.nonzero(self.inside)[0]
        _, first = np.unique(self.pix[idx], return_index=True)
        self.first = np.sort(idx[first])
        self.sums = np.zeros(n * n)
        self.counts = np.zeros(n * n)
        self.done = 0

    def pixels_at(self, cut):
        return int(np.searchsorted(self.first, cut, side="left"))

    def cut_for_pixels(self, m):
        if m <= 0:
            return 0
        if m >= len(self.first):
            return len(self.x)
        return int(self.first[m - 1]) + 1

    def advance(self, cut):
        a, b = self.done, max(cut, self.done)
        if b > a:
            sel = self.inside[a:b]
            p = self.pix[a:b][sel]
            self.counts += np.bincount(p, minlength=self.n * self.n)
            self.sums += np.bincount(p, weights=np.asarray(self.z[a:b], dtype=float)[sel], minlength=self.n * self.n)
        self.done = b
        return a, b

    def binned(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, self.sums / self.counts, np.nan).reshape(self.n, self.n)


def build_rows(d, rows: Sequence[str], bounds, n_pixels: int = 200, verbose: bool = True,
               channel: Optional[str] = None):
    """The rows the file has, as _Row objects (self-driving: every sample; raster: its
    saved samples, in scan order). channel: a saved extra channel (e.g. PFM "amplitude" /
    "phase": ch_<channel>) instead of z -- the raster has z only, so its rows need channel
    None or the one z is."""
    out, notes = [], []
    L = np.asarray(d["segment_lengths"])
    offsets = np.concatenate([[0], np.cumsum(L)])
    kinds = [str(k) for k in d["kinds"]]
    has_raster = "raster_binned_Z" in d.files
    zsrc = d["z"]
    if channel is not None:
        if f"ch_{channel}" not in d.files:
            raise ValueError(f"no channel {channel!r} in this file (have: "
                             f"{[k[3:] for k in d.files if k.startswith('ch_')]})")
        zsrc = d[f"ch_{channel}"]
        # the raster recorded z only: its rows show this channel only if z is it
        zl = str(d["z_label"]).lower() if "z_label" in d.files else ""
        if channel not in zl:
            has_raster = False
            notes.append(f"the reference raster recorded the other signal, not {channel}: no raster rows")
    for key in rows:
        if key == "run_cmd":
            out.append(_Row(key, np.asarray(d["x_cmd"]), np.asarray(d["y_cmd"]), zsrc, bounds, n_pixels, kinds, offsets))
        elif key == "run_sensor":
            if "x_sensor" not in d.files or "sensor_calibration_json" not in d.files:
                notes.append("no position-sensor data / calibration in this file: no self-driving measured row")
                continue
            out.append(_Row(key, np.asarray(d["x_sensor"]), np.asarray(d["y_sensor"]), zsrc, bounds, n_pixels,
                            kinds, offsets))
        elif key in ("raster_cmd", "raster_sensor"):
            if not has_raster:
                notes.append("no reference raster in this file: no raster rows")
                continue
            from . import reference_raster
            pos = "sensor" if key == "raster_sensor" else "commanded"
            if pos == "sensor" and "raster_xs" not in d.files:
                notes.append("the reference raster has no sensor positions: no raster measured row")
                continue
            R = reference_raster.load(d, verbose=False, points_per_line=None, positions=pos)
            out.append(_Row(key, np.asarray(R["x"], dtype=float), np.asarray(R["y"], dtype=float),
                            np.asarray(R["z"], dtype=float), bounds, n_pixels))
        else:
            raise ValueError(f"unknown row {key!r} (use {list(ROW_LABELS)})")
    if verbose:
        for t in dict.fromkeys(notes):
            print("  " + t)
    return out


def animate(d, out_path, rows: Sequence[str] = ("run_cmd", "run_sensor", "raster_cmd", "raster_sensor"),
            bounds=None, n_pixels: int = 200, recon_cfg: Optional[LiveViewConfig] = None,
            recon_mode: Optional[str] = None, raster_recon: str = "raster_columns", max_frames: int = 60,
            fps: int = 4, dpi: int = 80, recon_every: int = 1, unc_scale_V: Optional[float] = None,
            unc_label: Optional[str] = None, length_unit: str = "um", panel_size=(3.7, 3.3),
            title: str = "", verbose: bool = True, cmap: str = "copper", channel: Optional[str] = None,
            z_label: Optional[str] = None) -> Optional[pathlib.Path]:
    """Render the animation to out_path (.gif, or .mp4 with ffmpeg) and return its path
    (None if the file has none of the rows). See the module docstring."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, writers
    from .viz import _KIND_STYLE, length_axis_ticks
    t0 = time.time()
    recon_cfg = recon_cfg or LiveViewConfig()
    recon_mode = recon_mode or recon_cfg.reconstruction
    if bounds is None:
        bounds = tuple(float(v) for v in d["recon_bounds"])
    n = n_pixels
    R = build_rows(d, rows, bounds, n, verbose, channel=channel)
    if not R:
        print("  nothing to animate")
        return None
    cal = _sensor_cal(d)
    z_unit = str(d["z_unit"]) if "z_unit" in d.files else "V"
    zlab = z_label or (str(d["z_label"]) if "z_label" in d.files and channel is None else
                       "Z (nm)" if z_unit == "nm" else "AI0 (V)")
    plane = _plane(d, bounds, n)
    if unc_scale_V is None and "priors_json" in d.files:
        unc_scale_V = json.loads(str(d["priors_json"])).get("feature_min_V")
    unc_label = unc_label or (f"{unc_scale_V:.3g} V" if unc_scale_V else "5% of the width")

    # frames: the self-driving run's waveforms (about max_frames of them)
    run_rows = [r for r in R if r.key.startswith("run")]
    ref = run_rows[0] if run_rows else None
    if ref is not None:
        nw = len(ref.offsets) - 1
        idx = np.unique(np.linspace(0, nw - 1, min(max_frames, nw)).round().astype(int))
        cuts = [int(ref.offsets[i + 1]) for i in idx]
    else:      # rasters only: by sample count
        cuts = list(np.linspace(0, len(R[0].x), max_frames + 1)[1:].astype(int))
    # the raster's progress per frame (a fraction of its scan): the pixel budget of the run,
    # counted at the commanded positions (rows built here if not shown), else the sensors'
    frac = None
    if any(r.key.startswith("raster") for r in R):
        have = {r.key: r for r in R}
        for run_k, ras_k in (("run_cmd", "raster_cmd"), ("run_sensor", "raster_sensor")):
            try:
                rr = have.get(run_k) or (build_rows(d, (run_k,), bounds, n, False) or [None])[0]
                ra = have.get(ras_k) or (build_rows(d, (ras_k,), bounds, n, False) or [None])[0]
            except Exception:   # noqa: BLE001
                rr = ra = None
            if rr is not None and ra is not None:
                frac = [ra.cut_for_pixels(rr.pixels_at(c)) / max(len(ra.x), 1) for c in cuts]
                break
        if frac is None:
            frac = list(np.linspace(0, 1, len(cuts) + 1)[1:])

    # colour scale: the final self-driving (else first) image, substrate removed
    fin = ref or R[0]
    img = np.full(n * n, np.nan)
    c = np.bincount(fin.pix[fin.inside], minlength=n * n)
    s = np.bincount(fin.pix[fin.inside], weights=np.asarray(fin.z, dtype=float)[fin.inside], minlength=n * n)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.where(c > 0, s / c, np.nan).reshape(n, n) - plane
    vmin, vmax = np.nanpercentile(img, [1, 99])

    nr = len(R)
    fig, axs = plt.subplots(nr, 4, figsize=(4 * panel_size[0], nr * panel_size[1]), layout="constrained", squeeze=False)
    plt.close(fig)
    art = []
    blank = np.full((n, n), np.nan)
    for r, row in zip(R, axs):
        aP, aB, aR, aU = row
        sensor = r.key.endswith("sensor")
        iB = aB.imshow(blank, origin="lower", extent=bounds, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        iR = aR.imshow(blank, origin="lower", extent=bounds, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        iU = aU.imshow(np.ones((n, n)) * 100, origin="lower", extent=bounds, cmap="magma", vmin=0, vmax=100,
                       interpolation="nearest")
        fig.colorbar(iB, ax=aB, label=zlab); fig.colorbar(iR, ax=aR, label=zlab)
        fig.colorbar(iU, ax=aU, label="uncertainty (%)")
        mode = raster_recon if r.key.startswith("raster") else recon_mode
        for ax, t in ((aP, "path"), (aB, "binned"), (aR, f"reconstruction ({mode})"), (aU, "uncertainty")):
            ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3]); ax.set_aspect("equal")
            ax.set_title(t, fontsize=9)
            if sensor and cal is not None:
                length_axis_ticks(ax, cal, unit=length_unit)
                ax.set_xlabel(f"X ({length_unit}, sensor)", fontsize=8); ax.set_ylabel(f"Y ({length_unit}, sensor)", fontsize=8)
            else:
                ax.set_xlabel("X (V)", fontsize=8); ax.set_ylabel("Y (V)", fontsize=8)
            ax.tick_params(labelsize=7)
        aP.set_ylabel(ROW_LABELS[r.key] + "\n" + aP.get_ylabel(), fontsize=9, fontweight="bold")
        aP.grid(True, alpha=0.3)
        tip, = aP.plot([], [], "ro", markersize=5, zorder=5)
        art.append(dict(aP=aP, aU=aU, iB=iB, iR=iR, iU=iU, tip=tip, mode=mode))
    status = fig.suptitle("", fontsize=11, fontweight="bold")

    def draw(k):
        cut_run = cuts[k]
        last = k == len(cuts) - 1
        texts = []
        for r, a in zip(R, art):
            if r.key.startswith("run"):
                cut = cut_run
            else:
                cut = int(round(frac[k] * len(r.x)))
            lo, hi = r.advance(cut)
            # path since the last frame (strided for drawing)
            if hi > lo:
                if r.offsets is not None:
                    i0 = int(np.searchsorted(r.offsets, lo, side="right")) - 1
                    for i in range(max(i0, 0), len(r.offsets) - 1):
                        s0, s1 = max(int(r.offsets[i]), lo), min(int(r.offsets[i + 1]), hi)
                        if s0 >= hi:
                            break
                        if s1 > s0:
                            st = max(1, (s1 - s0) // 1500)
                            style = {q: v for q, v in _KIND_STYLE.get(r.kinds[i], {}).items() if q != "label"}
                            a["aP"].plot(r.x[s0:s1:st], r.y[s0:s1:st], **style)
                else:
                    st = max(1, (hi - lo) // 3000)
                    a["aP"].plot(r.x[lo:hi:st], r.y[lo:hi:st], "-", color="tab:red", lw=0.6)
                a["tip"].set_data([r.x[hi - 1]], [r.y[hi - 1]])
            B = r.binned()
            nm_ = int(np.isfinite(B).sum())
            a["iB"].set_data(B - plane)
            u = 100.0 * uncertainty_map(B, bounds, unc_scale_V)
            a["iU"].set_data(u)
            a["aU"].set_title(f"uncertainty (100% = {unc_label})\nmean {u.mean():.0f}%", fontsize=9)
            if (k % max(recon_every, 1) == 0 or last) and nm_ >= 10:
                rec = inpaint_binned(B, bounds, recon_cfg, mode=a["mode"])
                if rec is not None:
                    rec = np.array(rec, dtype=float) - plane
                    if r.key.startswith("raster"):
                        # nothing beyond the line being scanned (along the slow axis)
                        M = np.isfinite(B)
                        rows_ = np.nonzero(M.any(axis=1))[0]
                        cols_ = np.nonzero(M.any(axis=0))[0]
                        along_x = M.any(axis=1).sum() < M.any(axis=0).sum()
                        if along_x and len(rows_):
                            rec[:rows_.min()] = np.nan; rec[rows_.max() + 1:] = np.nan
                        elif len(cols_):
                            rec[:, :cols_.min()] = np.nan; rec[:, cols_.max() + 1:] = np.nan
                    a["iR"].set_data(rec)
            texts.append(f"{r.key.replace('_', ' ')}: {nm_:,} px")
        status.set_text((title + "\n" if title else "") + f"frame {k + 1}/{len(cuts)}  |  " + "   ".join(texts))
        return []

    out_path = pathlib.Path(out_path)
    ext = out_path.suffix.lower()
    if ext == ".mp4" and not writers.is_available("ffmpeg"):
        print("  ffmpeg not found -- saving a GIF instead")
        out_path = out_path.with_suffix(".gif"); ext = ".gif"
    writer = FFMpegWriter(fps=fps) if ext == ".mp4" else PillowWriter(fps=fps)

    def _progress(i, total):
        if verbose and ((i + 1) % 10 == 0 or i + 1 == total):
            print(f"  rendered frame {i + 1}/{total}")

    anim = FuncAnimation(fig, draw, frames=len(cuts), blit=False, repeat=False)
    anim.save(str(out_path), writer=writer, dpi=dpi, progress_callback=_progress)
    if verbose:
        print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, {len(cuts)} frames, {nr} rows) "
              f"in {time.time() - t0:.0f} s")
    return out_path
