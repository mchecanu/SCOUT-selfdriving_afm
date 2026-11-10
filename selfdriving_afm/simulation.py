"""
Scoring scan modes against a known ground truth -- the tools behind Simulation.ipynb.

    ev = evaluate(session, sample)          # after modes.run(session)
    plot_evaluation(ev)
    table = compare([...variants...], cfg, sample)     # several modes / settings, same sample
    sweep("grid_refine", "grid_refine.detect_k", [3, 5, 8], cfg, sample)

Everything is scored on the SAME window (default: the sample's area) and the same
pixel grid, so modes that scan different areas are compared fairly: the question is
"how good is the image of this area, and what did it cost".

Metrics (see `evaluate`):
  quality   rmse / bad_px (fraction of pixels off by more than bad_frac x the sample's
            height range) over the whole window; the same on the EDGES only
            (edge_bad_px); edge_cov = fraction of true edge pixels a sample passed
            within edge_tol_V of; per-feature edge coverage (text samples).
  cost      samples, waveforms (= do_scan calls on the instrument), path length,
            waveform time (samples / rate) and an ESTIMATED instrument time that adds
            the per-do_scan and readback overheads measured on the instrument
            (2026-09-24 probe: ~3 s per do_scan, ~0.5 us per sample read back).
"""

from __future__ import annotations
import copy
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import modes
from .backends.simulate_backend import SimulateBackend
from .config import ExperimentConfig
from .reconstruction import bin_image, reconstruct_image

# measured on the instrument -- change if you re-measure
# PyScanner-DEFINED scans (raster / spiral / Lissajous, grid_refine): probe_pyscanner_scans,
# 2026-09-24 -- ~2 M-sample scans took 3.2-4.2 s beyond their motion
OVERHEAD_PER_WAVEFORM_S = 3.0      # do_scan_update + do_scan + readback beyond the waveform itself
# USER-DEFINED waveforms (edge_explorer, domain_wall_explorer, gp_path, edge_follow): the same
# probe's user-defined line took 0.38 s wall for 0.02 s of motion, and a real edge_explorer run
# (500 waveforms, 38.5 M samples, 253 s) works out to 0.39 s per waveform beyond motion + readback
# (the PFM domain-wall run of 2026-11-04: <= 0.68 s with its extra channels and live view)
OVERHEAD_USER_WAVEFORM_S = 0.4
READBACK_S_PER_SAMPLE = 0.5e-6     # reading AO/AI_vs_time back: ~1 s per 2 M samples


# ------------------------------------------------------------------------------------
# config helpers
# ------------------------------------------------------------------------------------
def set_param(cfg: ExperimentConfig, path: str, value) -> None:
    """set_param(cfg, "grid_refine.detect_k", 3.0) -- a dotted path into the config."""
    obj = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise AttributeError(f"config has no {path!r}")
    setattr(obj, parts[-1], value)


def get_param(cfg: ExperimentConfig, path: str):
    obj = cfg
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def model_notes(cfg: ExperimentConfig, modality: str) -> List[str]:
    """What this simulation does NOT reproduce for this mode/config -- printed so a
    good simulated result isn't over-trusted."""
    notes = []
    if modality == "grid_refine" and cfg.grid_refine.scan_generator == "pyscanner_defined":
        notes.append("grid_refine with scan_generator='pyscanner_defined' is simulated with "
                     "pyscanner_scans.predict_path. Its rasters match the hardware probes (trace + retrace, "
                     "clockwise rotation); the 2026-09-24 probe showed Lissajous w1/w2 act as cycles-per-scan "
                     "(not Hz) and the spiral returns to the centre inside its duration. The per-do_scan overhead "
                     "(many tiles = many do_scans) is only in the time ESTIMATE.")
    if cfg.readback.mode == "image":
        notes.append("readback.mode='image' is not simulated: the simulator returns the exact z at every "
                     "sample (like 'raw'). On the instrument, image readback mis-maps many samples.")
    return notes


# ------------------------------------------------------------------------------------
# the classic raster a self-driving scan is compared with
# ------------------------------------------------------------------------------------
class EquivalentRaster:
    """A plain raster scan of the same sample, sized to match a self-driving scan so
    far: every line left to right (the trace; the retrace isn't measured), lines one
    after another from the bottom of the area.

    match   "pixels": as many measured pixels as the self-driving scan has (on the same
                      pixel grid); "path": as much tip path as the self-driving scan has
                      travelled (moves included, raster retraces counted) -- the same
                      scan time at the same tip speed.
    layout  "sparse": the lines spread evenly over the whole area (a coarser raster --
                      the fair comparison for the reconstruction); "progressive": a
                      full-resolution raster (one line per pixel row) that has got as
                      far as that budget, like an ordinary scan stopped part-way.

    reconstruction  how the raster's gaps between lines are filled: "raster_columns"
                      (default -- column-wise PCHIP between the lines, the best method
                      for raster data), or None = the self-driving scan's own method
                      (cfg.live_view.reconstruction), or any other live_view method name.

    The raster is measured the way the self-driving scan is: the sample's topography
    along each line, sub-sampled within each pixel, with the per-sample noise averaged
    over the samples a pixel gets at the configured tip speed and sample rate."""

    def __init__(self, sample, cfg: ExperimentConfig, noise_std: float = 0.0, layout: str = "sparse",
                 match: str = "pixels", seed: int = 0, reconstruction: Optional[str] = "raster_columns"):
        if layout not in ("sparse", "progressive"):
            raise ValueError(f"layout must be 'sparse' or 'progressive', not {layout!r}")
        if match not in ("pixels", "path"):
            raise ValueError(f"match must be 'pixels' or 'path', not {match!r}")
        self.sample = sample
        self.recon_cfg = copy.deepcopy(cfg.live_view)
        self.reconstruction = reconstruction or self.recon_cfg.reconstruction
        if self.reconstruction == "interpolated":  # needs raw points -- on raster pixels it's linear
            self.reconstruction = "linear"
        self.ds = cfg.velocity.sample_spacing_V
        self.v = cfg.velocity.tip_velocity_V_s
        self.noise_std = noise_std
        self.layout = layout
        self.match = match
        self.seed = seed
        self._cache_key = None
        self._cache_img = None

    def _full_image(self, bounds, n: int) -> np.ndarray:
        """What a full raster (every pixel row) measures on this pixel grid."""
        key = (tuple(round(float(b), 12) for b in bounds), n)
        if key == self._cache_key:
            return self._cache_img
        x0, x1, y0, y1 = bounds
        pix = (x1 - x0) / (n - 1)
        spp = max(pix / self.ds, 1.0)                 # samples per pixel along a line
        m = int(np.clip(round(spp), 1, 8))            # sub-samples actually evaluated
        xc = x0 + np.arange(n) * pix                  # pixel centres (bin_image convention)
        off = (np.arange(m) + 0.5) / m - 0.5
        xs = (xc[:, None] + off[None, :] * pix).ravel()
        ys = y0 + np.arange(n) * (y1 - y0) / (n - 1)
        X, Y = np.meshgrid(xs, ys)
        img = self.sample(X, Y).reshape(n, n, m).mean(axis=2)
        if self.noise_std > 0:
            rng = np.random.default_rng(self.seed)
            img = img + rng.normal(0.0, self.noise_std / np.sqrt(spp), img.shape)
        self._cache_key, self._cache_img = key, img
        return img

    def build(self, bounds, n: int, n_measured: int = 0, path_V: float = 0.0) -> dict:
        """The equivalent raster on an n x n grid over `bounds`: binned image (NaN =
        not measured), reconstruction (self.reconstruction), and what it took: lines,
        pixels, trace path, time at the same tip speed."""
        from .reconstruction import inpaint_binned
        x0, x1, y0, y1 = bounds
        width = x1 - x0
        if self.match == "pixels":
            M = int(n_measured)
        else:  # a full line costs its width twice (trace + retrace) and covers n pixels
            M = int(round(path_V / (2 * width) * n)) if width > 0 else 0
        M = int(np.clip(M, 0, n * n))
        binned = np.full((n, n), np.nan)
        n_rows = int(np.ceil(M / n)) if M else 0
        if self.layout == "progressive" or n_rows == 0:
            rows = np.arange(n_rows)
        else:
            rows = np.unique(np.round(np.linspace(0, n - 1, n_rows)).astype(int))
        full = self._full_image(bounds, n) if M else None
        left = M
        for r in rows:  # the last line (the top one) is only as long as what's left
            k = min(n, left)
            binned[r, :k] = full[r, :k]
            left -= k
        recon = None
        if M >= 3:
            recon = inpaint_binned(binned, bounds, self.recon_cfg, mode=self.reconstruction)
        ys = y0 + rows * (y1 - y0) / (n - 1)
        last_len = ((M - n * (len(rows) - 1)) / n) * width if len(rows) else 0.0
        trace = width * max(len(rows) - 1, 0) + last_len
        spacing = (ys[1] - ys[0]) if len(ys) > 1 else float("nan")
        return dict(binned=binned, recon=recon, rows_y=ys, last_len=last_len, bounds=tuple(bounds),
                    n_lines=len(rows), n_pixels=int(np.isfinite(binned).sum()), spacing_V=spacing,
                    trace_V=trace, time_s=2 * trace / self.v, layout=self.layout, match=self.match)

    def draw_path(self, ax, info: dict, color: str = "tab:red") -> None:
        """The raster's lines (trace, left to right), the last one partial."""
        x0 = info["bounds"][0]
        x1 = info["bounds"][1]
        ys = info["rows_y"]
        many = len(ys) > 40  # dense rasters: thin, faint lines so the ground truth stays visible
        lw, alpha = (0.3, 0.45) if many else (1.0, 0.85)
        for i, y in enumerate(ys):
            xe = x0 + info["last_len"] if i == len(ys) - 1 else x1
            ax.plot([x0, xe], [y, y], "-", color=color, lw=lw, alpha=alpha,
                    label="raster lines (left to right)" if i == 0 else None)
        if len(ys):  # where the raster has got to: the end of its last line
            ax.plot(x0 + info["last_len"], ys[-1], "o", color=color, markersize=6)

    def describe(self, info: dict) -> str:
        sp = f", {info['spacing_V']:.3g} V apart" if np.isfinite(info["spacing_V"]) else ""
        return (f"{info['n_lines']} line(s){sp}, {info['n_pixels']:,} px, "
                f"~{info['time_s']:.2g} s at the same tip speed (trace + retrace)")


class EquivalentSpiral:
    """A plain Archimedean spiral scan of the same sample -- out from the centre of the
    area to the circle inscribed in it -- sized to match a self-driving scan so far,
    like EquivalentRaster (same `match` / `layout` meaning):

    match   "pixels": as many measured pixels (same pixel grid); "path": as much tip path.
    layout  "sparse": the whole spiral, its turns spread out to fit the budget (the fair
                      comparison for the reconstruction); "progressive": a full-resolution
                      spiral (turns one pixel apart) that has got as far as the budget.
    reconstruction  how the gaps are filled: None = the self-driving scan's own method
                      (cfg.live_view.reconstruction), or any live_view method name.

    Measured like the raster: the sample's topography along the path, with the
    per-sample noise averaged over the samples between two evaluated points."""

    def __init__(self, sample, cfg: ExperimentConfig, noise_std: float = 0.0, layout: str = "sparse",
                 match: str = "pixels", seed: int = 0, reconstruction: Optional[str] = None):
        if layout not in ("sparse", "progressive"):
            raise ValueError(f"layout must be 'sparse' or 'progressive', not {layout!r}")
        if match not in ("pixels", "path"):
            raise ValueError(f"match must be 'pixels' or 'path', not {match!r}")
        self.sample = sample
        self.recon_cfg = copy.deepcopy(cfg.live_view)
        self.reconstruction = reconstruction or self.recon_cfg.reconstruction
        if self.reconstruction in ("interpolated", "raster_columns"):
            self.reconstruction = "linear"
        self.ds = cfg.velocity.sample_spacing_V
        self.v = cfg.velocity.tip_velocity_V_s
        self.noise_std = noise_std
        self.layout = layout
        self.match = match
        self.seed = seed

    @staticmethod
    def _path(center, R, pitch, h):
        """Archimedean spiral r = pitch * theta / 2 pi, 0..R, one point every ~h of path."""
        n_turns = max(R / max(pitch, 1e-12), 1e-3)
        th_end = 2 * np.pi * n_turns
        b = pitch / (2 * np.pi)
        L = 0.5 * b * (th_end * np.sqrt(1 + th_end ** 2) + np.arcsinh(th_end))   # arc length
        n = int(np.clip(L / h, 16, 4_000_000))
        s = np.linspace(0.0, L, n)
        th = np.sqrt(2 * s / max(b, 1e-15))       # s ~ b th^2 / 2: fine for the pixel count
        th = np.minimum(th, th_end)
        r = b * th
        return center[0] + r * np.cos(th), center[1] + r * np.sin(th), np.full(n, L / max(n - 1, 1))

    def _pixels(self, x, y, bounds, n):
        from .reconstruction import pixel_index
        return pixel_index(x, y, n, bounds)

    def build(self, bounds, n: int, n_measured: int = 0, path_V: float = 0.0) -> dict:
        """The equivalent spiral on an n x n grid over `bounds`: binned image, its
        reconstruction, the path (for drawing) and what it took."""
        from .reconstruction import inpaint_binned
        x0, x1, y0, y1 = bounds
        c = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        R = 0.5 * min(x1 - x0, y1 - y0)
        pix = (x1 - x0) / (n - 1)
        h = 0.5 * pix
        empty = dict(binned=np.full((n, n), np.nan), recon=None, x=np.empty(0), y=np.empty(0), bounds=tuple(bounds),
                     n_pixels=0, pitch_V=float("nan"), length_V=0.0, time_s=0.0, layout=self.layout, match=self.match)
        if (self.match == "pixels" and n_measured < 3) or (self.match == "path" and path_V <= 0):
            return empty
        if self.layout == "progressive":
            x, y, _ = self._path(c, R, pix, h)
            if self.match == "pixels":
                flat = self._pixels(x, y, bounds, n)
                _, first = np.unique(flat, return_index=True)
                k = np.sort(first)[min(int(n_measured), len(first)) - 1] + 1
            else:
                k = int(np.clip(path_V / h, 2, len(x)))
            x, y = x[:k], y[:k]
            pitch = pix
        elif self.match == "path":
            pitch = max(np.pi * R * R / path_V, pix)
            x, y, _ = self._path(c, R, pitch, h)
        else:   # the pitch whose whole spiral measures n_measured pixels (bisection)
            lo_p, hi_p = pix, 2 * R
            for _ in range(22):
                mid = np.sqrt(lo_p * hi_p)
                x, y, _ = self._path(c, R, mid, h)
                if len(np.unique(self._pixels(x, y, bounds, n))) > n_measured:
                    lo_p = mid
                else:
                    hi_p = mid
            pitch = np.sqrt(lo_p * hi_p)
            x, y, _ = self._path(c, R, pitch, h)
        z = self.sample(x, y)
        if self.noise_std > 0:
            rng = np.random.default_rng(self.seed)
            z = z + rng.normal(0.0, self.noise_std / np.sqrt(max(h / self.ds, 1.0)), z.shape)
        binned, _ = bin_image(np.column_stack([x, y]), z, n, bounds)
        recon = inpaint_binned(binned, bounds, self.recon_cfg, mode=self.reconstruction) \
            if np.isfinite(binned).sum() >= 3 else None
        length = float(np.hypot(np.diff(x), np.diff(y)).sum()) if len(x) > 1 else 0.0
        return dict(binned=binned, recon=recon, x=x, y=y, bounds=tuple(bounds), n_pixels=int(np.isfinite(binned).sum()),
                    pitch_V=float(pitch), length_V=length, time_s=length / self.v, layout=self.layout, match=self.match)

    def draw_path(self, ax, info: dict, color: str = "tab:purple") -> None:
        x, y = info["x"], info["y"]
        if not len(x):
            return
        st = max(1, len(x) // 6000)
        ax.plot(x[::st], y[::st], "-", color=color, lw=0.4, alpha=0.8, label="spiral (from the centre)")
        ax.plot(x[-1], y[-1], "o", color=color, markersize=6)

    def describe(self, info: dict) -> str:
        return (f"spiral, turns {info['pitch_V']:.3g} V apart, {info['n_pixels']:,} px, "
                f"{info['length_V']:.3g} V of path (~{info['time_s']:.2g} s at the same tip speed)")


def score_against(recon, truth_img, bad_frac: float = 0.2) -> dict:
    """RMSE and bad-pixel fraction of a reconstruction against a ground-truth image on the
    same grid (NaN = not reconstructed counts as bad); bad = off by more than bad_frac x
    the truth's 1-99% height range."""
    t = np.asarray(truth_img, dtype=float)
    lo, hi = np.nanpercentile(t, [1, 99])
    bad_abs = bad_frac * max(hi - lo, 1e-12)
    if recon is None:
        return dict(rmse=float("nan"), bad_px=1.0)
    err = np.asarray(recon, dtype=float) - t
    return dict(rmse=float(np.sqrt(np.nanmean(err ** 2))) if np.isfinite(err).any() else float("nan"),
                bad_px=float(np.mean(~(np.abs(err) <= bad_abs))))


# ------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------
def _edge_mask(gt: np.ndarray, edge_frac: float) -> np.ndarray:
    """Pixels where the ground truth jumps by more than edge_frac x its height range
    to a neighbour -- the edges a scan should find."""
    lo, hi = np.percentile(gt, [1, 99])
    thr = edge_frac * max(hi - lo, 1e-30)
    m = np.zeros(gt.shape, dtype=bool)
    dx = np.abs(np.diff(gt, axis=1)) > thr
    dy = np.abs(np.diff(gt, axis=0)) > thr
    m[:, 1:] |= dx
    m[:, :-1] |= dx
    m[1:, :] |= dy
    m[:-1, :] |= dy
    return m


def evaluate(session: modes.Session, sample, eval_bounds=None, n: int = 200, bad_frac: float = 0.2,
             edge_frac: float = 0.2, edge_tol_V: Optional[float] = None,
             overhead_per_waveform_s: float = OVERHEAD_PER_WAVEFORM_S,
             overhead_user_waveform_s: float = OVERHEAD_USER_WAVEFORM_S,
             readback_s_per_sample: float = READBACK_S_PER_SAMPLE,
             raster: Optional["EquivalentRaster"] = None) -> dict:
    """Score a finished run against `sample` (the ground truth) over eval_bounds
    (default: the sample's area) on an n x n grid. The reconstruction is recomputed on
    that grid with the run's own method (cfg.live_view.reconstruction; GP path: its GP).
    raster: an EquivalentRaster -> the same scores for the classic raster with the same
    budget, in ev["raster"]."""
    s, cfg = session, session.cfg
    b = tuple(eval_bounds or sample.bounds)
    pix = max((b[1] - b[0]) / (n - 1), (b[3] - b[2]) / (n - 1))
    tol = edge_tol_V if edge_tol_V is not None else 2 * pix
    X, Y = np.meshgrid(np.linspace(b[0], b[1], n), np.linspace(b[2], b[3], n))
    gt = sample(X, Y)
    pts = s.executor.all_points()

    t0 = time.perf_counter()
    rc = copy.deepcopy(cfg.live_view)
    rc.n_image_pixels = n
    binned, _ = bin_image(pts[:, :2], pts[:, 2], n, b)
    if modes.MODES[s.modality].live_view == "gp":
        recon, _ = s.planner.predict_surface(X, Y)
        method = "GP path's GP"
    else:
        _, recon = reconstruct_image(pts[:, :2], pts[:, 2], b, rc)
        method = rc.reconstruction
    recon_time = time.perf_counter() - t0

    lo, hi = np.percentile(gt, [1, 99])
    zr = max(hi - lo, 1e-30)
    edges = _edge_mask(gt, edge_frac)
    score = _score(recon, binned, gt, edges, X, Y, sample, pix, tol, bad_frac * zr)

    hist = s.executor.history
    # true sample counts (readback.store_every keeps block means in RAM; n_raw = what was measured)
    n_samples = int(sum((st.meta or {}).get("n_raw", len(st.x)) for st in hist))
    path_len = float(sum(np.hypot(np.diff(st.x), np.diff(st.y)).sum() for st in hist if len(st.x) > 1))
    wave_s = n_samples / cfg.velocity.sample_rate_hz
    n_wave = len(hist)
    n_builtin = sum(1 for st in hist if st.meta and "pyscanner_scan" in st.meta)   # PyScanner-defined scans
    est_s = (wave_s + n_builtin * overhead_per_waveform_s + (n_wave - n_builtin) * overhead_user_waveform_s
             + n_samples * readback_s_per_sample)
    ev = dict(
        modality=s.modality, session=s, sample=sample, bounds=b, X=X, Y=Y, gt=gt, binned=binned,
        recon=recon, edges=edges, method=method, **score,
        samples=n_samples, waveforms=n_wave, path_V=path_len, waveform_s=wave_s, est_instrument_s=est_s,
        sim_wall_s=s.run_time_s, recon_s=recon_time, z_range=zr, tol_V=tol,
    )
    if raster is not None:
        # the classic raster with the same budget (pixels on THIS grid, or path length),
        # scored the same way
        info = raster.build(b, n, n_measured=int(np.isfinite(binned).sum()), path_V=path_len)
        r_recon = info["recon"] if info["recon"] is not None else np.full(gt.shape, np.nan)
        ev["raster"] = dict(info=info, binned=info["binned"], recon=r_recon, raster=raster,
                            **_score(r_recon, info["binned"], gt, edges, X, Y, sample, pix, tol, bad_frac * zr))
    return ev


def _score(recon, binned, gt, edges, X, Y, sample, pix, tol, bad_abs) -> dict:
    """Quality metrics of one reconstruction against the ground truth (see evaluate)."""
    from scipy.ndimage import distance_transform_edt, binary_dilation
    err = recon - gt
    bad = ~(np.abs(err) <= bad_abs)          # NaN (not reconstructed) counts as bad
    # distance (V) from every pixel to the nearest pixel a sample landed in
    measured = np.isfinite(binned)
    dist = distance_transform_edt(~measured) * pix if measured.any() else np.full(gt.shape, np.inf)
    edge_zone = binary_dilation(edges, iterations=2) if edges.any() else edges
    feats = []
    for i, f in enumerate(getattr(sample, "features", []) or []):
        ring = np.abs(f.signed_distance(X, Y)) <= pix
        if not ring.any():
            continue
        feats.append(dict(index=i + 1, shape=f.shape, edge_cov=float(np.mean(dist[ring] <= tol)),
                          edge_bad_px=float(np.mean(bad[binary_dilation(ring, iterations=2)]))))
    fin = np.isfinite(err)
    return dict(err=err, bad=bad, dist=dist, features=feats,
                rmse=float(np.sqrt(np.mean(err[fin] ** 2))) if fin.any() else np.nan,
                mae=float(np.mean(np.abs(err[fin]))) if fin.any() else np.nan,
                bad_px=float(bad.mean()),
                edge_bad_px=float(bad[edge_zone].mean()) if edge_zone.any() else np.nan,
                edge_cov=float(np.mean(dist[edges] <= tol)) if edges.any() else np.nan,
                measured_px=float(measured.mean()))


def format_evaluation(ev: dict) -> str:
    lines = [f"{ev['modality']}: reconstruction ({ev['method']}) vs ground truth over "
             f"X {ev['bounds'][0]:.3g}..{ev['bounds'][1]:.3g}, Y {ev['bounds'][2]:.3g}..{ev['bounds'][3]:.3g} V",
             f"  quality: RMSE {ev['rmse']:.4g} ({100 * ev['rmse'] / ev['z_range']:.1f}% of the height range), "
             f"bad pixels {100 * ev['bad_px']:.1f}% (edges: {100 * ev['edge_bad_px']:.1f}%), "
             f"edges visited {100 * ev['edge_cov']:.1f}% (within {ev['tol_V']:.3g} V), "
             f"pixels measured {100 * ev['measured_px']:.1f}%",
             f"  cost: {ev['samples']:,} samples in {ev['waveforms']} waveform(s), path {ev['path_V']:.1f} V, "
             f"{ev['waveform_s']:.1f} s of waveforms -> ~{ev['est_instrument_s']:.0f} s estimated on the instrument "
             f"(simulated in {ev['sim_wall_s']:.1f} s)"]
    if ev["features"]:
        missed = [f for f in ev["features"] if f["edge_cov"] < 0.5]
        lines.append("  per feature (edge visited / edge bad px): " + ", ".join(
            f"#{f['index']} {f['shape']} {100 * f['edge_cov']:.0f}%/{100 * f['edge_bad_px']:.0f}%" for f in ev["features"]))
        if missed:
            lines.append("  MISSED (less than half of the edge visited): " +
                         ", ".join(f"#{f['index']} {f['shape']}" for f in missed))
    r = ev.get("raster")
    if r is not None:
        info = r["info"]
        what = "measured pixels" if info["match"] == "pixels" else "tip path"
        lines.append(f"  classic raster with the same {what} ({info['layout']}, reconstructed with "
                     f"{r['raster'].reconstruction}): {r['raster'].describe(info)}")
        d = r["bad_px"] - ev["bad_px"]
        verdict = "BETTER than" if d > 0.001 else "WORSE than" if d < -0.001 else "about the SAME as"
        lines.append(f"    quality: RMSE {r['rmse']:.4g}, bad pixels {100 * r['bad_px']:.1f}% "
                     f"(edges: {100 * r['edge_bad_px']:.1f}%), edges visited {100 * r['edge_cov']:.1f}%")
        lines.append(f"    -> {ev['modality']} is {verdict} the raster: {100 * ev['bad_px']:.1f}% vs "
                     f"{100 * r['bad_px']:.1f}% bad pixels; scan time {ev['waveform_s']:.1f} s of waveforms "
                     f"(~{ev['est_instrument_s']:.0f} s on the instrument) vs ~{info['time_s']:.1f} s for the raster")
    return "\n".join(lines)


def _draw_path(ax, session, max_points_per_step: int = 3000):
    from .viz import _KIND_STYLE
    seen = set()
    for st in session.executor.history:
        style = dict(_KIND_STYLE.get(st.kind, dict(color="k", linewidth=1.0)))
        if st.kind in seen:
            style.pop("label", None)
        seen.add(st.kind)
        k = max(1, len(st.x) // max_points_per_step)
        style["linewidth"] = min(style.get("linewidth", 1.0), 0.8)
        ax.plot(st.x[::k], st.y[::k], **style)


def _plot_row(axes, ev, sc, binned, recon, draw_path, path_title, method, cmap):
    fig = axes[0].figure
    b = ev["bounds"]
    vmin, vmax = np.percentile(ev["gt"], [0.5, 99.5])
    kw = dict(origin="lower", extent=b, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].imshow(ev["gt"], alpha=0.75, **kw)
    draw_path(axes[0])
    axes[0].set_title(path_title)
    axes[1].imshow(binned, **kw)
    axes[1].set_title(f"Measured pixels ({100 * sc['measured_px']:.1f}%)")
    im = axes[2].imshow(recon, **kw)
    fig.colorbar(im, ax=axes[2], shrink=0.85)
    axes[2].set_title(f"Reconstruction ({method})")
    lim = ev["z_range"]
    im = axes[3].imshow(sc["err"], origin="lower", extent=b, cmap="RdBu_r", vmin=-lim, vmax=lim,
                        interpolation="nearest")
    axes[3].contour(ev["X"], ev["Y"], ev["edges"].astype(float), levels=[0.5], colors="k", linewidths=0.4)
    fig.colorbar(im, ax=axes[3], shrink=0.85, label="recon - truth")
    axes[3].set_title(f"Error: {100 * sc['bad_px']:.1f}% bad px, edges {100 * sc['edge_cov']:.0f}% visited")
    for ax in axes:
        ax.set_xlim(b[0], b[1]); ax.set_ylim(b[2], b[3]); ax.set_aspect("equal")
        ax.set_xlabel("X (V)")
    axes[0].set_ylabel("Y (V)")


def plot_evaluation(ev: dict, axes=None, cmap: str = "copper", title: bool = True):
    """Ground truth + path | measured (binned) | reconstruction | error (edges outlined).
    If the evaluation has the equivalent classic raster (evaluate(raster=...)) and no
    axes are given: a second row, the same four panels for that raster."""
    import matplotlib.pyplot as plt
    r = ev.get("raster")
    if axes is None:
        rows = 2 if r is not None else 1
        fig, ax_all = plt.subplots(rows, 4, figsize=(18, 4.4 * rows), layout="constrained", squeeze=False)
        axes = ax_all[0]
        raster_axes = ax_all[1] if r is not None else None
    else:
        fig = axes[0].figure
        raster_axes = None
    _plot_row(axes, ev, ev, ev["binned"], ev["recon"], lambda ax: _draw_path(ax, ev["session"]),
              f"Ground truth + {ev['modality']} path ({ev['samples']:,} samples)", ev["method"], cmap)
    if raster_axes is not None:
        info = r["info"]
        _plot_row(raster_axes, ev, r, r["binned"], r["recon"], lambda ax: r["raster"].draw_path(ax, info),
                  f"Classic raster, same {'pixels' if info['match'] == 'pixels' else 'path'}: "
                  f"{info['n_lines']} lines", r["raster"].reconstruction, cmap)
        axes[0].set_ylabel(f"{ev['modality']}\nY (V)", fontweight="bold")
        raster_axes[0].set_ylabel("classic raster\nY (V)", fontweight="bold")
    if title:
        extra = (f"; classic raster: {100 * r['bad_px']:.1f}% vs {ev['modality']}: {100 * ev['bad_px']:.1f}% bad px"
                 if r is not None else "")
        fig.suptitle(f"{ev['modality']} -- ~{ev['est_instrument_s']:.0f} s estimated on the instrument, "
                     f"RMSE {ev['rmse']:.3g}{extra}", fontweight="bold")
    return fig


# ------------------------------------------------------------------------------------
# several modes / settings on the same sample
# ------------------------------------------------------------------------------------
Variant = Union[str, Tuple[str, str], Tuple[str, str, Dict[str, object]]]


def _variant(v: Variant):
    """"grid_refine" | ("label", "grid_refine") | ("label", "grid_refine", {"grid_refine.detect_k": 3})"""
    if isinstance(v, str):
        return v, v, {}
    if len(v) == 2:
        return v[0], v[1], {}
    return v[0], v[1], dict(v[2])


def simulate(modality: str, cfg: ExperimentConfig, sample, noise_std: float = 0.001, seed: int = 1,
             live: bool = False, overrides: Optional[Dict[str, object]] = None, verbose: bool = False,
             raster: Optional[dict] = None, **eval_kw):
    """Build + run + evaluate one mode on `sample` (a copy of cfg, with overrides).
    raster: e.g. {"layout": "sparse", "match": "pixels"} -> also score the classic
    raster with the same budget (EquivalentRaster). Returns the evaluation dict (its
    "session" has the planner, executor, ...)."""
    c = copy.deepcopy(cfg)
    for k, val in (overrides or {}).items():
        set_param(c, k, val)
    backend = SimulateBackend(sample, noise_std=noise_std, seed=seed)
    s = modes.build(modality, c, backend, live=live, ground_truth_fn=sample)
    modes.run(s, verbose=verbose)
    eq = EquivalentRaster(sample, c, noise_std, seed=seed, **raster) if raster is not None else None
    return evaluate(s, sample, raster=eq, **eval_kw)


def compare(variants: Sequence[Variant], cfg: ExperimentConfig, sample, noise_std: float = 0.001,
            seed: int = 1, plot: bool = True, raster: Optional[dict] = None, **eval_kw) -> List[dict]:
    """Run every variant (no live view) on the same sample and print one table row each;
    plot: one row of evaluation panels per variant. raster (e.g. {"layout": "sparse",
    "match": "pixels"}): the table also scores, for each variant, the classic raster with
    the same budget. A failing variant is reported, not fatal. Returns the evaluation
    dicts (failed ones: {"label", "error"})."""
    import matplotlib.pyplot as plt
    evs = []
    for v in variants:
        label, modality, ov = _variant(v)
        t0 = time.time()
        try:
            ev = simulate(modality, cfg, sample, noise_std, seed, overrides=ov, raster=raster, **eval_kw)
            ev["label"] = label
            print(f"  done: {label} ({time.time() - t0:.1f} s)")
        except Exception as e:  # noqa: BLE001 -- one bad variant shouldn't stop the comparison
            import traceback
            ev = dict(label=label, modality=modality, error=f"{type(e).__name__}: {e}",
                      trace=traceback.format_exc())
            print(f"  FAILED: {label} -- {ev['error']}")
        evs.append(ev)
    print()
    print(comparison_table(evs))
    ok = [e for e in evs if "error" not in e]
    if plot and ok:
        fig, axes = plt.subplots(len(ok), 4, figsize=(18, 4.2 * len(ok)), layout="constrained", squeeze=False)
        for row, ev in zip(axes, ok):
            plot_evaluation(ev, row, title=False)
            row[0].set_ylabel(f"{ev['label']}\nY (V)", fontweight="bold")
        plt.show()
    return evs


def comparison_table(evs: Sequence[dict]) -> str:
    with_raster = any("raster" in e for e in evs)
    head = (f"{'variant':28s} {'bad px':>7s} {'edge bad':>8s} {'edges hit':>9s} {'RMSE':>9s} "
            f"{'samples':>11s} {'waveforms':>9s} {'wave s':>7s} {'~instr s':>8s} {'sim s':>6s}")
    if with_raster:
        head += f" | {'raster bad':>10s} {'raster edge':>11s} {'lines':>5s}"
    rows = [head, "-" * len(head)]
    for e in evs:
        if "error" in e:
            rows.append(f"{e['label'][:28]:28s} FAILED: {e['error'][:90]}")
            continue
        row = (f"{e['label'][:28]:28s} {100 * e['bad_px']:6.1f}% {100 * e['edge_bad_px']:7.1f}% "
               f"{100 * e['edge_cov']:8.1f}% {e['rmse']:9.3g} {e['samples']:11,} {e['waveforms']:9d} "
               f"{e['waveform_s']:7.1f} {e['est_instrument_s']:8.0f} {e['sim_wall_s']:6.1f}")
        if "raster" in e:
            r = e["raster"]
            row += f" | {100 * r['bad_px']:9.1f}% {100 * r['edge_bad_px']:10.1f}% {r['info']['n_lines']:5d}"
        rows.append(row)
    rows.append("bad px / edge bad: pixels off by > bad_frac x height range (all / near edges); "
                "edges hit: true edges a sample passed near; ~instr s: waveforms + measured do_scan/readback overhead")
    if with_raster:
        info = next(e["raster"]["info"] for e in evs if "raster" in e)
        rows.append(f"raster: a classic left-to-right raster ({info['layout']}) with the same "
                    f"{'measured pixels' if info['match'] == 'pixels' else 'tip path'} as each variant")
    return "\n".join(rows)


def sweep(modality: str, param: str, values: Iterable, cfg: ExperimentConfig, sample,
          metrics: Sequence[str] = ("bad_px", "edge_cov", "est_instrument_s"), plot: bool = True,
          **kw) -> List[dict]:
    """Run `modality` once per value of one config parameter (dotted path, e.g.
    "grid_refine.detect_k") and plot the chosen metrics against it."""
    import matplotlib.pyplot as plt
    values = list(values)
    evs = compare([(f"{param.split('.')[-1]}={v!r}", modality, {param: v}) for v in values], cfg, sample,
                  plot=False, **kw)
    ok = [(v, e) for v, e in zip(values, evs) if "error" not in e]
    if plot and ok:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.4), layout="constrained")
        for ax, m in zip(np.atleast_1d(axes), metrics):
            xs = [v for v, _ in ok]
            ys = [e[m] for _, e in ok]
            numeric = all(isinstance(x, (int, float)) for x in xs)
            ax.plot(xs if numeric else range(len(xs)), ys, "o-")
            if not numeric:
                ax.set_xticks(range(len(xs)), [str(x) for x in xs], rotation=30)
            ax.set_xlabel(param)
            ax.set_title(m)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{modality}: sweep of {param}", fontweight="bold")
        plt.show()
    return evs
