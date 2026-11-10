"""
Grid + refine planner: a sparse scan of the whole area, then denser scans only where
the data shows structure, level by level.

    1. sparse scan: ONE scan over the whole area (raster / Lissajous / spiral, lines
       `sparse spacing` apart; sparse_crosshatch adds a second raster pass the other way)
    2. fast inpainting of the binned image (normalized convolution, ~20 ms)
    3. detection: where does the data show structure the current line spacing
       can't resolve? -- local height variation after removing the LOCAL tilt (a
       plane fitted in a small window around every pixel), compared with the noise
       level. A tilted substrate or a flat feature top is not flagged; the edges of
       features, steps and rough patches are. cfg.detect_on picks what it's
       evaluated on: "measured" (the measured pixels only -- the scan path is then
       independent of any reconstruction) or "reconstruction" (the image inpainted
       with cfg.detect_reconstruction, so structure predicted between the scan lines
       counts too -- the path then depends on that reconstruction method).
    4. tiles: small squares (at most tile_max_V -- the priors: half the smallest
       feature) with flagged pixels. A closed ring of tiles round unflagged ones -- a
       feature's outline round its flat top -- gets its inside added (fill_interiors, up
       to fill_max_size_V wide = the largest feature): the top gets measured and
       internal edges found. Tiles that touch (sides or corners) form ONE region.
    5. refinement: one scan per region, lines refine_factor x closer, its pattern from
       its shape (refine_pattern "auto"): round -> spiral (when shorter than a raster),
       rectangular -> raster / Lissajous (auto_rect_pattern), irregular -> raster with
       every line clipped to the region's tiles. All regions of a level are joined by
       smooth moves into ONE waveform (one do_scan) by default
    6. every refinement adds its points to the same global data set (the executor's
       history) -- the live view reconstructs from all of it
    7. back to 2. at the new spacing, looking only inside the regions just refined;
       stops when nothing is flagged, at min_spacing_V, or at the time limit.

Tip speed: every scan moves the tip at constant VelocityConfig.tip_velocity_V_s --
along the raster lines AND round their turnarounds (semicircles of half a line spacing,
just outside the tile), and along the Lissajous / spiral. Each scan gets a short
straight run-up and run-out (tip_velocity x ease_time_s / 2 long) where it speeds up
from / slows down to rest, and moves between scans start and end at rest too, so the
piezo never sees a jump in position or velocity.

The planner never computes anything expensive between scans: binning is a
bincount, detection is a handful of box filters, inpainting a few Gaussian
filters.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter, binary_fill_holes, binary_erosion, find_objects, label as _label

from .. import trajectory
from .. import pyscanner_scans as pys
from ..config import amplitude_limits, GridRefineConfig
from ..execution import ContinuousExecutor
from ..config import LiveViewConfig
from ..reconstruction import bin_image, fast_inpaint, inpaint_binned  # noqa: F401 (re-exported)


@dataclass(eq=False)  # identity comparison -- holds numpy arrays
class ScanRect:
    """One scan: the whole area (the sparse scan) or one refinement REGION -- a group of
    touching tiles (x0..y1 = its bounding box, `cells` = the tiles themselves)."""
    x0: float
    x1: float
    y0: float
    y1: float
    fast_axis: str = "x"   # raster line direction
    pattern: Optional[str] = None   # this region's scan pattern (None = the level's)
    shape: str = ""                 # what "auto" saw: "round", "rectangular", "irregular"
    cells: Optional[np.ndarray] = None     # (n, 4) tiles: x0, x1, y0, y1
    interior: Optional[np.ndarray] = None  # (n,) bool: tile added as a feature's inside
    spiral_center: Optional[np.ndarray] = None
    radius: Optional[float] = None  # spiral radius (covers the feature in it -- _feature_disk)
    roundness: Optional[float] = None  # what "auto" measured (feature area / its circle's)

    @property
    def center(self):
        if self.spiral_center is not None:
            return np.asarray(self.spiral_center, dtype=float)
        return np.array([0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1)])

    @property
    def size(self):
        return self.x1 - self.x0, self.y1 - self.y0

    def area(self, pattern: Optional[str] = None) -> float:
        """Area the scan covers: the spiral's disk, the tiles (clipped raster), or the
        bounding box."""
        pattern = self.pattern or pattern
        if pattern == "spiral" and self.radius is not None:
            return float(np.pi * self.radius ** 2)
        if pattern == "raster" and self.cells is not None:
            c = self.cells
            return float(np.sum((c[:, 1] - c[:, 0]) * (c[:, 3] - c[:, 2])))
        return float(self.size[0] * self.size[1])

    def contains_points(self, X, Y) -> np.ndarray:
        """Which of the points (arrays X, Y) lie in this region's tiles (bbox if none)."""
        cells = self.cells if self.cells is not None else np.array([[self.x0, self.x1, self.y0, self.y1]])
        out = np.zeros(np.shape(X), dtype=bool)
        for x0, x1, y0, y1 in cells:
            out |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        return out


@dataclass
class RefineLevel:
    index: int                     # 0 = the sparse scan
    spacing_V: float               # line spacing of this level's scans
    pattern: str
    rects: List[ScanRect] = field(default_factory=list)
    n_samples: int = 0             # waveform samples sent (scans + moves)
    scan_time_s: float = 0.0       # n_samples / sample rate
    plan_time_s: float = 0.0       # computing before this level (inpaint + detect + plan)
    threshold: Optional[float] = None  # detection threshold that selected these tiles
    noise: Optional[float] = None
    area_fraction: float = 1.0     # fraction of the whole area this level scanned
    tile_size_V: Optional[float] = None  # the tiles these regions were made of
    n_interior_tiles: int = 0      # tiles added as features' insides (fill_interiors)


def local_plane_rms(img: np.ndarray, r: int, min_count: int = 5) -> np.ndarray:
    """RMS residual of a plane fitted (least squares) to the measured pixels in the
    (2r+1)^2 window around every pixel -- i.e. the local height variation with the
    local tilt removed. NaN pixels in `img` are unmeasured; windows with fewer than
    min_count measured pixels give NaN. All box filters: O(pixels), ~ms."""
    M = np.isfinite(img)
    if not M.any():
        return np.full(img.shape, np.nan)
    z0 = float(np.mean(img[M]))
    zs = float(np.std(img[M])) or 1.0
    Z = np.where(M, (img - z0) / zs, 0.0)
    Mf = M.astype(float)
    ny, nx = img.shape
    X, Y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    size = 2 * int(r) + 1

    def box(a):
        return uniform_filter(a, size, mode="constant")  # window MEAN (x size^2 = sum)

    n = box(Mf)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = 1.0 / n
        mx, my, mz = box(Mf * X) * inv, box(Mf * Y) * inv, box(Z) * inv
        cxx = box(Mf * X * X) * inv - mx * mx
        cyy = box(Mf * Y * Y) * inv - my * my
        cxy = box(Mf * X * Y) * inv - mx * my
        cxz = box(Z * X) * inv - mx * mz
        cyz = box(Z * Y) * inv - my * mz
        czz = box(Z * Z) * inv - mz * mz
        ridge = 1e-3 * (cxx + cyy) + 1e-9   # a single line in the window: fit along it only
        a, d = cxx + ridge, cyy + ridge
        det = a * d - cxy * cxy
        # explained variance of the best plane: [cxz cyz] C^-1 [cxz cyz]^T
        expl = (d * cxz * cxz - 2 * cxy * cxz * cyz + a * cyz * cyz) / det
        var = np.maximum(czz - expl, 0.0)
    rms = np.sqrt(var) * zs
    rms[~np.isfinite(rms) | (n * size * size < min_count)] = np.nan
    return rms


def _remove_plane(img: np.ndarray) -> np.ndarray:
    """Subtract a robust global plane (two-pass least squares, features excluded on
    the second pass) -- used only to choose raster line directions."""
    M = np.isfinite(img)
    ny, nx = img.shape
    X, Y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    keep = M.copy()
    coef = np.zeros(3)
    for _ in range(2):
        if keep.sum() < 3:
            break
        A = np.column_stack([X[keep], Y[keep], np.ones(keep.sum())])
        coef, *_ = np.linalg.lstsq(A, img[keep], rcond=None)
        res = img - (coef[0] * X + coef[1] * Y + coef[2])
        mad = np.nanmedian(np.abs(res[M])) + 1e-12
        keep = M & (np.abs(res) < 3 * mad)
    return img - (coef[0] * X + coef[1] * Y + coef[2])


class GridRefinePlanner:
    def __init__(self, executor: ContinuousExecutor, cfg: GridRefineConfig,
                 recon_cfg: Optional[LiveViewConfig] = None):
        """recon_cfg: the reconstruction parameters (CS lambda, GP kernel, ...) used when
        cfg.detect_on == "reconstruction" -- pass cfg.live_view (default: LiveViewConfig())."""
        self.executor = executor
        self.cfg = cfg
        self.recon_cfg = recon_cfg or LiveViewConfig()
        self.detect_image = None  # what the last detection ran on (binned or reconstructed)
        self.levels: List[RefineLevel] = []
        # latest internal maps (row = y, col = x, over self.bounds) -- for plots/saving
        self.binned = self.inpainted = self.roughness = self.flagged = None
        self.bounds = None

    # ------------------------------------------------------------------
    @property
    def _ds(self) -> float:
        return self.executor.velocity.sample_spacing_V

    @property
    def _rate(self) -> float:
        return self.executor.velocity.sample_rate_hz

    def _ease_n(self) -> int:
        return int(round(self.cfg.ease_time_s * self._rate))

    def _lim(self) -> float:
        """Half-width of the largest centred square inside the target limits (round scans)."""
        x0, x1, y0, y1 = self._lims()
        return min(-x0, x1, -y0, y1)

    def _lims(self):
        """(x_lo, x_hi, y_lo, y_hi): where the scans' TARGET positions may go -- the safety
        limits on the drive (drive ~ target - u), less cfg.drive_margin_V, shifted by the
        offset u the run starts with (position_sensors.initial_offset_V, if correcting)."""
        lx, ly = amplitude_limits(self.executor.safety)
        mx, my = self.cfg.drive_margin_V
        ex = self.executor
        u = np.zeros(2)
        u0 = getattr(ex.position_sensors, "initial_offset_V", None)
        if u0 is not None and ex.correcting:
            u = np.asarray(u0, dtype=float)
        ax, ay = 0.999 * lx - mx, 0.999 * ly - my
        return u[0] - ax, u[0] + ax, u[1] - ay, u[1] + ay

    # ------------------------------------------------------------------
    def run(self, start_xy=(0.0, 0.0)) -> List[RefineLevel]:
        cfg = self.cfg
        ex = self.executor
        ex.reset(start_xy)
        ex.connector_ease_samples = self._ease_n()
        self.levels = []
        t0 = time.monotonic()

        # the area, kept inside the safety limit
        xl, xh, yl, yh = self._lims()
        cx = float(np.clip(cfg.center_V[0], xl, xh))
        cy = float(np.clip(cfg.center_V[1], yl, yh))
        # room for what sticks out past the area: raster turnarounds (half the sparse
        # spacing, or raster_turn_radius_V) and the run-up / run-out -- along the lines'
        # axis; with every raster along x (sparse + raster_direction "x") nothing sticks out in y
        h0 = 2 * cfg.half_size_V / max(cfg.sparse_n_lines - 1, 1)
        margin = max(0.5 * h0, cfg.raster_turn_radius_V or 0.0) + 0.5 * self._ease_n() * self._ds
        # y: rasters all along x and a sparse x-raster or spiral (the spiral: inside the area, only its
        # run-out sticks out; refinement spirals are kept inside the range by _room) -- just the run-out
        x_only = (cfg.sparse_pattern in ("raster", "spiral") and not cfg.sparse_crosshatch
                  and cfg.raster_direction == "x" and cfg.refine_pattern in ("raster", "auto", "spiral")
                  and cfg.scan_generator == "user_defined")
        run_out = 0.5 * self._ease_n() * self._ds
        my_ = run_out if x_only else margin
        H = min(cfg.half_size_V, cx - xl - margin, xh - cx - margin, cy - yl - my_, yh - cy - my_)
        if H <= 0:
            raise ValueError("grid_refine: the area is outside the safety limit")
        self.bounds = (cx - H, cx + H, cy - H, cy + H)
        S = 2 * H
        h = S / max(cfg.sparse_n_lines - 1, 1)
        h_min = cfg.min_spacing_V if cfg.min_spacing_V is not None else S / 255
        n_pix = cfg.detect_pixels or int(np.clip(np.ceil(2 * S / h_min) + 1, 128, 512))
        self._n_pix = n_pix
        self._pix = S / (n_pix - 1)

        # 1. sparse scan over the whole area
        whole = ScanRect(*self.bounds, fast_axis="x")
        self._scan_level(0, h, [whole], cfg.sparse_pattern, plan_time=0.0, sparse=True)
        allowed = [whole]

        for k in range(1, cfg.max_levels + 1):
            if cfg.max_run_time_s is not None and time.monotonic() - t0 > cfg.max_run_time_s:
                break
            if h <= h_min * (1 + 1e-9):
                break  # already at the finest spacing
            h_next = max(h / cfg.refine_factor, h_min)  # the last level lands on h_min exactly
            t_plan = time.perf_counter()
            # 2. + 3. inpaint, detect
            self._update_maps(h)
            rects, thr, noise = self._plan_tiles(k, h, allowed, h_next)
            plan_time = time.perf_counter() - t_plan
            if not rects:
                break
            # 4. + 5. refine (points go into the executor's history = the global data)
            self._scan_level(k, h_next, rects, cfg.refine_pattern, plan_time, threshold=thr, noise=noise)
            self.levels[-1].tile_size_V = self._tile_size
            self.levels[-1].n_interior_tiles = self._n_interior
            allowed = rects   # 6. next level looks only inside what was just refined
            h = h_next

        self._update_maps(h)
        return self.levels

    # ------------------------------------------------------------------
    # 2. + 3.: maps and detection
    # ------------------------------------------------------------------
    def _update_maps(self, h: float) -> None:
        pts = self.executor.all_points()
        self.binned, _ = bin_image(pts[:, :2], pts[:, 2], self._n_pix, self.bounds)
        self.inpainted = fast_inpaint(self.binned, 0.5 * h / self._pix)
        r = max(1, int(round(self.cfg.detect_window_lines * h / self._pix)))
        if self.cfg.detect_on == "reconstruction":
            mode = self.cfg.detect_reconstruction
            if mode == "live_view":
                mode = self.recon_cfg.reconstruction
            if mode == "interpolated":
                mode = "linear"
            recon = inpaint_binned(self.binned, self.bounds, self.recon_cfg, mode=mode)
            # keep the measured pixels as measured; the reconstruction fills the gaps
            self.detect_image = np.where(np.isfinite(self.binned), self.binned, recon)
            # restrict to the scanned square's data extent (no extrapolation far out):
            # pixels further than one line spacing from any measurement are left out
            from scipy.ndimage import distance_transform_edt
            far = distance_transform_edt(~np.isfinite(self.binned)) * self._pix > h
            img = np.where(far, np.nan, self.detect_image)
            self.inpainted = self.detect_image
        elif self.cfg.detect_on == "measured":
            img = self.detect_image = self.binned
        else:
            raise ValueError(f"Unknown detect_on {self.cfg.detect_on!r} (use 'measured' or 'reconstruction')")
        self.roughness = local_plane_rms(img, r)
        # the noise level always comes from the MEASURED pixels -- an inpainting's own
        # artefacts in the gaps (e.g. CS ringing) must not raise the threshold
        self._rough_measured = (self.roughness if self.cfg.detect_on == "measured"
                                else local_plane_rms(self.binned, r))

    def _plan_tiles(self, k: int, h: float, allowed: List[ScanRect], h_next: Optional[float] = None):
        """The next level's regions: flagged tiles (+ the insides they enclose), grouped
        into regions of touching tiles, each with its own pattern."""
        cfg = self.cfg
        h_next = h / cfg.refine_factor if h_next is None else h_next
        self._n_interior = 0
        rough = self.roughness
        valid = np.isfinite(rough)
        rm = self._rough_measured
        if not valid.any() or not np.isfinite(rm).any():
            return [], None, None
        noise = float(np.quantile(rm[np.isfinite(rm)], cfg.detect_noise_quantile))
        zrange = float(np.nanmax(self.binned) - np.nanmin(self.binned)) or 1.0
        thr = cfg.detect_threshold if cfg.detect_threshold is not None else max(cfg.detect_k * noise, 1e-6 * zrange)
        flag = valid & (rough > thr)
        self.flagged = flag

        x0, x1, y0, y1 = self.bounds
        T = cfg.tile_size_lines * h
        if cfg.tile_max_V is not None:
            T = min(T, cfg.tile_max_V)        # the priors: half the smallest feature
        T = max(T, 2 * h_next, 2 * self._pix)  # at least a couple of the new lines per tile
        self._tile_size = T
        nt_x = max(1, int(np.ceil((x1 - x0) / T - 1e-9)))
        nt_y = max(1, int(np.ceil((y1 - y0) / T - 1e-9)))
        n = self._n_pix
        px = x0 + np.arange(n) * self._pix  # pixel centres (bin_image convention)
        py = y0 + np.arange(n) * self._pix
        tcol = np.minimum(((px - x0) / T).astype(int), nt_x - 1)
        trow = np.minimum(((py - y0) / T).astype(int), nt_y - 1)
        rr, cc = np.nonzero(flag)
        counts = np.bincount(trow[rr] * nt_x + tcol[cc], minlength=nt_x * nt_y).reshape(nt_y, nt_x)
        sel = counts >= cfg.min_flagged_pixels

        # only inside what the previous level scanned (tile centre test)
        tx = x0 + (np.arange(nt_x) + 0.5) * T
        ty = y0 + (np.arange(nt_y) + 0.5) * T
        TX, TY = np.meshgrid(tx, ty)
        inside = np.zeros_like(sel)
        for a in allowed:
            inside |= a.contains_points(TX, TY)
        sel &= inside
        if not sel.any():
            return [], thr, noise

        # a feature's flat top is not flagged, only its outline: scan the inside a
        # closed ring of tiles encloses too (not insides wider than the largest feature --
        # that's substrate between features -- nor ones open to the area's border)
        interior = np.zeros_like(sel)
        if cfg.fill_interiors and k <= cfg.fill_interior_levels:
            holes = binary_fill_holes(sel) & ~sel & inside
            lab, n_holes = _label(holes)  # 4-connected: the background can't leak diagonally
            for j, sl in enumerate(find_objects(lab), start=1):
                width = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * T
                if cfg.fill_max_size_V is None or width <= cfg.fill_max_size_V:
                    interior[sl] |= lab[sl] == j
            sel |= interior
            self._n_interior = int(interior.sum())

        # raster line direction: lines along the steepest direction = across the edges
        if cfg.raster_direction == "auto":
            flat = _remove_plane(self.inpainted)
            gy, gx = np.gradient(flat)
            idx = (trow[:, None] * nt_x + tcol[None, :]).ravel()
            gx2 = np.bincount(idx, weights=(gx ** 2).ravel(), minlength=nt_x * nt_y).reshape(nt_y, nt_x)
            gy2 = np.bincount(idx, weights=(gy ** 2).ravel(), minlength=nt_x * nt_y).reshape(nt_y, nt_x)

        def direction(cells_mask):
            if cfg.raster_direction == "auto":
                return "y" if gy2[cells_mask].sum() > gx2[cells_mask].sum() else "x"
            if cfg.raster_direction == "alternate":
                return "y" if k % 2 == 1 else "x"
            return cfg.raster_direction

        # regions: tiles that touch (sides or corners) are ONE region, scanned as one scan
        if cfg.merge_tiles:
            lab, n_reg = _label(sel, structure=np.ones((3, 3), dtype=bool) if cfg.merge_diagonal else None)
        else:
            lab = np.zeros(sel.shape, dtype=int)
            lab[sel] = np.arange(1, int(sel.sum()) + 1)
        if cfg.merge_tiles and cfg.region_max_V:
            lab = self._split_regions(lab, T, cfg.region_max_V)
        self._px_tiles = (px, py, trow, tcol)   # pixel centres + their tile, for _feature_disk
        rects = []
        for j, sl in enumerate(find_objects(lab), start=1):
            if sl is None:
                continue
            m = lab == j
            self._region_mask = m
            rr, cc = np.nonzero(m)
            cells = np.column_stack([x0 + cc * T, np.minimum(x0 + (cc + 1) * T, x1),
                                     y0 + rr * T, np.minimum(y0 + (rr + 1) * T, y1)])
            reg = ScanRect(float(cells[:, 0].min()), float(cells[:, 1].max()),
                           float(cells[:, 2].min()), float(cells[:, 3].max()),
                           direction(m), cells=cells, interior=interior[rr, cc])
            self._choose_pattern(reg, h_next, h)
            rects.append(reg)
        return rects, thr, noise

    @staticmethod
    def _split_regions(lab: np.ndarray, T: float, max_V: float) -> np.ndarray:
        """Regions (labels on the tile grid) wider than max_V split into their objects: the
        region's tiles eroded until they fall apart, then every tile given to the nearest piece."""
        from scipy.ndimage import distance_transform_edt
        out = np.zeros_like(lab)
        nxt = 1
        for j, sl in enumerate(find_objects(lab), start=1):
            if sl is None:
                continue
            m = lab == j
            if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * T <= max_V:
                out[m] = nxt
                nxt += 1
                continue
            parts, k = None, 0
            for it in range(1, 20):
                e = binary_erosion(m, iterations=it)
                if not e.any():
                    break
                pl, k = _label(e)
                if k >= 2:
                    parts = pl
                    break
            if parts is None:
                out[m] = nxt
                nxt += 1
                continue
            _, (ir, ic) = distance_transform_edt(parts == 0, return_indices=True)
            near = parts[ir, ic]
            for q in range(1, k + 1):
                sel = m & (near == q)
                if sel.any():
                    out[sel] = nxt
                    nxt += 1
        return out

    def _choose_pattern(self, reg: ScanRect, spacing: float, h_prev: Optional[float] = None) -> None:
        """Pick a region's pattern from its shape (refine_pattern = "auto").

        The tiles are NOT the feature's shape: structure is flagged with a square window
        about one line spacing wide, so even a small disk comes out as a rounded SQUARE of
        tiles. So the shape is judged on the feature itself, segmented from the image
        inside the region (_feature_disk):
          round        roundness (feature area / its enclosing circle's) >=
                       spiral_min_roundness, and a spiral covering it -- plus the edge's
                       position uncertainty (half the previous spacing) and one new line
                       -- is a shorter path than a raster of the tiles
          rectangular  the tiles fill >= rect_fill_min of their bounding box -> auto_rect_pattern
          irregular    anything else -> raster, each line clipped to the tiles.
        A spiral must stay inside the area (and the scan range)."""
        cfg = self.cfg
        h_prev = cfg.refine_factor * spacing if h_prev is None else h_prev
        c = reg.cells
        a = (c[:, 1] - c[:, 0]) * (c[:, 3] - c[:, 2])
        disk = self._feature_disk(spacing, h_prev)
        if disk is None:  # no clear feature: a circle round the tiles' centroid
            centroid = np.array([np.sum(a * 0.5 * (c[:, 0] + c[:, 1])),
                                 np.sum(a * 0.5 * (c[:, 2] + c[:, 3]))]) / a.sum()
            R = float(np.max(np.hypot(np.concatenate([c[:, 0], c[:, 1]])[:, None] - centroid[0],
                                      np.concatenate([c[:, 2], c[:, 3]])[None, :] - centroid[1])))
            roundness = 0.0
        else:
            centroid, R, roundness = disk
        room = self._room(centroid)
        if cfg.refine_pattern == "spiral":
            R = min(R, room)
        fits = R <= room + 1e-9
        if fits or cfg.refine_pattern == "spiral":
            reg.spiral_center, reg.radius = centroid, R
        if cfg.refine_pattern != "auto":
            reg.pattern = cfg.refine_pattern
            return
        sx, sy = reg.size
        fill = float(a.sum() / max(sx * sy, 1e-24))
        _, ext, h = self._clipped_lines(reg, spacing)
        raster_len = float(np.sum(ext[:, 1] - ext[:, 0])) + (len(ext) - 1) * 0.5 * np.pi * h
        spiral_len = np.pi * R ** 2 / spacing
        if (fits and roundness >= cfg.spiral_min_roundness and R >= 3 * spacing   # >= ~2 turns
                and spiral_len < raster_len):
            reg.pattern, reg.shape = "spiral", "round"
        elif fill >= cfg.rect_fill_min:
            reg.pattern, reg.shape = cfg.auto_rect_pattern, "rectangular"
        else:
            reg.pattern, reg.shape = "raster", "irregular"
        reg.roundness = roundness

    def _room(self, p) -> float:
        """How big a circle round p may be: inside the area and the scan range."""
        x0, x1, y0, y1 = self.bounds
        xl, xh, yl, yh = self._lims()
        return float(min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1],
                         p[0] - xl, xh - p[0], p[1] - yl, yh - p[1]))

    def _feature_disk(self, spacing: float, h_prev: float):
        """The feature inside the current region (self._region_mask on the tile grid):
        its pixels in the image the detection ran on, split into two levels halfway
        between the region's 10th and 90th height percentiles; the SUBSTRATE is the
        level most of the region's outer tiles are on, the feature the other one
        (a bump or a hole). Returns (centre, spiral radius, roundness) or None:
          centre     middle of the feature's extent
          radius     spiral_coverage of its pixels lie within this, + h_prev / 2 (where
                     the edge really is, between the previous lines) + one new line
          roundness  feature area / pi r^2, r = that coverage radius (disk ~0.9-1,
                     square 0.64, a thin bar much less)."""
        img = self.inpainted
        if img is None:
            return None
        px, py, trow, tcol = self._px_tiles
        m = self._region_mask
        P = m[trow[:, None], tcol[None, :]] & np.isfinite(img)
        edge_tiles = m & ~binary_erosion(m)
        B = edge_tiles[trow[:, None], tcol[None, :]] & P
        if P.sum() < 16 or B.sum() < 4:
            return None
        z = img[P]
        p10, p90 = np.percentile(z, [10, 90])
        if not p90 > p10:
            return None
        hi = img > 0.5 * (p10 + p90)
        substrate_hi = hi[B].mean() > 0.5
        feat = P & (hi != substrate_hi)
        if feat.sum() < 4:
            return None
        rr, cc = np.nonzero(feat)
        X, Y = px[cc], py[rr]
        centre = np.array([0.5 * (X.min() + X.max()), 0.5 * (Y.min() + Y.max())])
        d = np.hypot(X - centre[0], Y - centre[1])
        r = float(np.quantile(d, self.cfg.spiral_coverage)) + 0.5 * self._pix
        roundness = float(len(d) * self._pix ** 2 / (np.pi * r * r))
        return centre, r + 0.5 * h_prev + spacing, min(roundness, 1.0)

    @staticmethod
    def _clipped_lines(reg: ScanRect, spacing: float):
        """Raster lines over a region, each clipped to the tiles it crosses:
        (first line's slow coordinate, (n, 2) fast-axis extents, actual spacing)."""
        c = reg.cells
        if reg.fast_axis == "x":
            s0, s1, slo, shi, flo, fhi = reg.y0, reg.y1, c[:, 2], c[:, 3], c[:, 0], c[:, 1]
        else:
            s0, s1, slo, shi, flo, fhi = reg.x0, reg.x1, c[:, 0], c[:, 1], c[:, 2], c[:, 3]
        n = max(2, int(round((s1 - s0) / max(spacing, 1e-12))) + 1)
        h = (s1 - s0) / (n - 1)
        tol = 1e-9 + 0.5 * h   # a line also covers the tiles within half a spacing of it
        ext = np.empty((n, 2))
        for i in range(n):
            s = s0 + i * h
            on = (slo <= s + tol) & (shi >= s - tol)
            if not on.any():
                on = np.abs(0.5 * (slo + shi) - s) == np.min(np.abs(0.5 * (slo + shi) - s))
            ext[i] = flo[on].min(), fhi[on].max()
        return s0, ext, h

    # ------------------------------------------------------------------
    # 1. + 4.: building and sending the scans
    # ------------------------------------------------------------------
    def _pattern(self, rect: ScanRect, spacing: float, pattern: str, start_xy, sparse: bool = False):
        """Waveform for one rect, starting at whichever start option is nearest
        start_xy. Starts and ends at rest. A region's own pattern (rect.pattern) wins
        over `pattern`; a region's raster has each line clipped to its tiles."""
        cfg, ds = self.cfg, self._ds
        pattern = rect.pattern or pattern
        sx, sy = rect.size
        c = np.array([0.5 * (rect.x0 + rect.x1), 0.5 * (rect.y0 + rect.y1)])  # bounding-box centre
        if pattern == "raster" and rect.cells is not None and not sparse:
            s0, ext, h = self._clipped_lines(rect, spacing)
            x, y, info = trajectory.clipped_raster_scan(s0, h, ext, ds, rect.fast_axis, start_xy,
                                                        cfg.raster_turn_radius_V)
            desc = (f"raster {sx:.3g} x {sy:.3g} V clipped to {len(rect.cells)} tiles, "
                    f"{info['n_lines']} lines along {rect.fast_axis}")
        elif pattern == "raster":
            best = None
            for corner in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                fs, ss = corner if rect.fast_axis == "x" else corner[::-1]
                start = c + 0.5 * np.array([corner[0] * sx, corner[1] * sy])
                dist = np.linalg.norm(start - start_xy)
                if best is None or dist < best[0]:
                    best = (dist, (fs, ss))
            x, y, info = trajectory.raster_scan(c, sx, sy, spacing, ds, rect.fast_axis, best[1],
                                                cfg.raster_turn_radius_V)
            desc = f"raster {sx:.3g} x {sy:.3g} V, {info['n_lines']} lines along {rect.fast_axis}"
            if sparse and cfg.sparse_crosshatch:
                other = "y" if rect.fast_axis == "x" else "x"
                end = np.array([x[-1], y[-1]])
                sgn = np.sign(end - c)
                corner2 = (sgn[1], sgn[0]) if other == "y" else (sgn[0], sgn[1])
                x2, y2, info2 = trajectory.raster_scan(c, sx, sy, spacing, ds, other, corner2,
                                                       cfg.raster_turn_radius_V)
                # two passes at right angles: each gets its own run-up / run-out (full speed
                # on every line), joined by a smooth move at the corner
                xa, ya, _ = trajectory.add_runup(x, y, ds, self._ease_n())
                xb, yb, _ = trajectory.add_runup(x2, y2, ds, self._ease_n())
                mx, my = self._eased_move((xa[-1], ya[-1]), (xb[0], yb[0]))
                x, y = np.concatenate([xa, mx, xb]), np.concatenate([ya, my, yb])
                desc = (f"raster crosshatch {sx:.3g} x {sy:.3g} V, "
                        f"{info['n_lines']} + {info2['n_lines']} lines")
                return x, y, desc
        elif pattern == "lissajous":
            x, y, info = trajectory.lissajous_scan(c, sx, sy, spacing, ds, cfg.lissajous_phi_rad)
            dur = info["samples"] / self._rate
            desc = (f"Lissajous {sx:.3g} x {sy:.3g} V, w1 = {info['a'] / dur:.3g} Hz, "
                    f"w2 = {info['b'] / dur:.3g} Hz over {dur:.3g} s")
        elif pattern == "spiral":
            lim = self._lim()
            c = rect.center   # a region's centroid
            # a region: its own radius; the sparse scan (or a bare rect): the circle inside
            # it -- never outside the area to explore
            R = rect.radius if rect.radius is not None else 0.5 * min(sx, sy)
            R = min(R, lim - abs(c[0]), lim - abs(c[1]))
            x, y, info = trajectory.spiral_scan(c, R, spacing, ds, cfg.spiral_direction,
                                                min_radius=cfg.raster_turn_radius_V)
            desc = f"spiral r = {R:.3g} V, {info['n_cycles']:.1f} turns"
        else:
            raise ValueError(f"Unknown pattern {pattern!r} (use 'raster', 'lissajous' or 'spiral')")
        # speed up / slow down on a short run-up / run-out, so the scan itself is all at
        # tip_velocity_V_s and the waveform still starts and ends at rest
        x, y, _ = trajectory.add_runup(x, y, ds, self._ease_n())
        return x, y, desc

    def _eased_move(self, p0, p1):
        """Straight move p0 -> p1 that starts and ends at rest (speeding up / slowing
        down over ease_time_s, tip_velocity_V_s in between). Returns the samples strictly
        between p0 and p1 (empty if they're within a sample step)."""
        p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
        gap = p1 - p0
        L = float(np.linalg.norm(gap))
        if L <= 1.5 * self._ds:
            return np.empty(0), np.empty(0)
        n = max(int(np.ceil(L / self._ds)) + 1, 2)
        t = np.linspace(0.0, 1.0, n)
        mx, my = trajectory.ease_ends(p0[0] + t * gap[0], p0[1] + t * gap[1], self._ease_n())
        return mx[1:-1], my[1:-1]

    def _fit_one(self, make, spacing: float, v: float, max_n: int, rate: float):
        """make(spacing, speed) -> one scan of at most max_n samples: faster first (up to
        cfg.pyscanner_max_speed_V_s), then lines / turns further apart. The scan's info says
        what it took (fit_speed_V_s, fit_spacing_V)."""
        vmax = max(v, self.cfg.pyscanner_max_speed_V_s)
        s = make(spacing, v)
        if s.duration_s * rate > max_n:
            vv = min(vmax, v * s.duration_s * rate / (0.98 * max_n))
            s = make(spacing, vv)
            sp = spacing
            while s.duration_s * rate > max_n and sp < 1e3:
                sp *= 1.05
                s = make(sp, vv)
            v, spacing = vv, sp
        # PyScanner pads a scan shorter than MIN_SCAN_SAMPLES with zeros (the tip then goes to
        # (0, 0) -- a jump): short scans run slower instead
        if s.duration_s * rate < pys.MIN_SCAN_SAMPLES:
            v = v * s.duration_s * rate / (1.02 * pys.MIN_SCAN_SAMPLES)
            s = make(spacing, v)
        s.info = dict(s.info or {}, fit_speed_V_s=float(v), fit_spacing_V=float(spacing))
        return s

    def _pyscanner_scans(self, rect: ScanRect, spacing: float, pattern: str, sparse: bool = False):
        """PyScanner-defined scan(s) covering one rect: a raster (two, at right angles, for
        a crosshatch sparse scan -- and split into bands along its slow axis if it would
        exceed max_samples_per_waveform), a Lissajous, or a spiral round the rect."""
        cfg, ex = self.cfg, self.executor
        v = ex.velocity.tip_velocity_V_s
        rate = self._rate
        max_n = int(ex.safety.max_samples_per_waveform)
        basis = cfg.pyscanner_speed_basis
        pattern = rect.pattern or pattern   # a region's own pattern (PyScanner rasters: its bbox)
        sx, sy = rect.size
        c = np.array([0.5 * (rect.x0 + rect.x1), 0.5 * (rect.y0 + rect.y1)])
        if cfg.pyscanner_one_scan and pattern in ("raster", "spiral"):
            if pattern == "raster":
                def make(sp, vv):
                    return pys.plan_raster(c, sx, sy, sp, vv, rect.fast_axis, cfg.pyscanner_raster_type, basis)
            else:
                cs = rect.center
                R = rect.radius if rect.radius is not None else 0.5 * min(sx, sy)
                if sparse and cfg.sparse_spiral_radius_V:
                    R = cfg.sparse_spiral_radius_V
                xl, xh, yl, yh = self._lims()
                R = min(R, cs[0] - xl, xh - cs[0], cs[1] - yl, yh - cs[1])

                def make(sp, vv):
                    return pys.plan_spiral(cs, R, sp, vv, basis)
            return [self._fit_one(make, spacing, v, max_n, rate)]
        if pattern == "raster":
            axes = [rect.fast_axis]
            if sparse and cfg.sparse_crosshatch:
                axes.append("y" if rect.fast_axis == "x" else "x")
            out = []
            for fa in axes:
                s = pys.plan_raster(c, sx, sy, spacing, v, fa, cfg.pyscanner_raster_type, basis)
                n = s.duration_s * rate
                if n <= max_n:
                    out.append(s)
                    continue
                # too many samples for one scan: split into bands along the slow axis
                k = int(np.ceil(n / (0.95 * max_n)))
                slow = sy if fa == "x" else sx
                while True:    # a band's lines are rounded: more bands until every one fits
                    bands = []
                    for b in range(k):
                        lo, hi = -0.5 * slow + b * slow / k, -0.5 * slow + (b + 1) * slow / k
                        mid = 0.5 * (lo + hi)
                        cb = (c[0], c[1] + mid) if fa == "x" else (c[0] + mid, c[1])
                        bx, by = (sx, hi - lo) if fa == "x" else (hi - lo, sy)
                        bands.append(pys.plan_raster(cb, bx, by, spacing, v, fa, cfg.pyscanner_raster_type, basis))
                    if all(b_.duration_s * rate <= max_n for b_ in bands) or k > 200:
                        break
                    k += 1
                out += bands
            return out
        if pattern in ("lissajous", "spiral"):
            if pattern == "lissajous":
                s = pys.plan_lissajous(c, sx, sy, spacing, v, cfg.lissajous_phi_rad, basis)
            else:
                lim = self._lim()
                cs = rect.center   # a region's centroid
                # a region: its own radius; the sparse scan: the circle inside the area
                R = rect.radius if rect.radius is not None else 0.5 * min(sx, sy)
                if sparse and cfg.sparse_spiral_radius_V:
                    R = cfg.sparse_spiral_radius_V
                xl, xh, yl, yh = self._lims()
                R = min(R, cs[0] - xl, xh - cs[0], cs[1] - yl, yh - cs[1])
                s = pys.plan_spiral(cs, R, spacing, v, basis)
            if sparse or s.duration_s * rate <= max_n or min(sx, sy) < 4 * spacing:
                return [s]            # the sparse scan: always ONE scan (RAM checked before the run)
            # too many samples for one scan: split the rect into quarters (recursively)
            out = []
            for qx in (-1, 1):
                for qy in (-1, 1):
                    q = ScanRect(rect.x0 if qx < 0 else c[0], c[0] if qx < 0 else rect.x1,
                                 rect.y0 if qy < 0 else c[1], c[1] if qy < 0 else rect.y1, rect.fast_axis)
                    out += self._pyscanner_scans(q, spacing, pattern, sparse)
            return out
        raise ValueError(f"Unknown pattern {pattern!r} (use 'raster', 'lissajous' or 'spiral')")

    @staticmethod
    def _what(r: ScanRect) -> str:
        """' (spiral, round, 12 tiles incl. 3 inside a feature)' for the waveform labels."""
        if r.cells is None:
            return ""
        n_in = int(r.interior.sum()) if r.interior is not None else 0
        bits = [b for b in (r.pattern, r.shape) if b]
        if r.roundness:
            bits.append(f"roundness {r.roundness:.2f}")
        bits.append(f"{len(r.cells)} tiles" + (f" incl. {n_in} inside a feature" if n_in else ""))
        return " (" + ", ".join(bits) + ")"

    @staticmethod
    def _level_mix(lv: RefineLevel) -> str:
        """' [3 raster, 1 spiral; tiles 0.25 V, 14 inside features]' for the summary."""
        if lv.index == 0 or not lv.rects:
            return ""
        pats = [r.pattern or lv.pattern for r in lv.rects]
        mix = ", ".join(f"{pats.count(p)} {p}" for p in dict.fromkeys(pats))
        if lv.tile_size_V is not None:
            mix += f"; tiles {lv.tile_size_V:.3g} V"
        if lv.n_interior_tiles:
            mix += f", {lv.n_interior_tiles} inside features"
        return f" [{mix}]"

    def _order(self, rects: List[ScanRect], pos) -> List[ScanRect]:
        """Nearest-neighbour tour, starting from the tip."""
        left = list(rects)
        out = []
        pos = np.asarray(pos, dtype=float)
        while left:
            i = int(np.argmin([np.linalg.norm(r.center - pos) for r in left]))
            out.append(left.pop(i))
            pos = out[-1].center
        return out

    def _scan_level(self, k: int, spacing: float, rects: List[ScanRect], pattern: str,
                    plan_time: float, threshold=None, noise=None, sparse: bool = False) -> None:
        cfg, ex = self.cfg, self.executor
        S2 = (self.bounds[1] - self.bounds[0]) * (self.bounds[3] - self.bounds[2])
        pys_mode = cfg.scan_generator == "pyscanner_defined"
        covered = sum(r.size[0] * r.size[1] if pys_mode and (r.pattern or pattern) == "raster"
                      else r.area(pattern) for r in rects)
        level = RefineLevel(k, spacing, pattern, rects, plan_time_s=plan_time, threshold=threshold,
                            noise=noise, area_fraction=min(covered / S2, 1.0) if not sparse else 1.0)
        self.levels.append(level)
        n_before = sum(len(s.x) for s in ex.history)
        kind = "grid_sparse" if sparse else "refine"
        name = "Sparse scan" if sparse else f"Refinement level {k}"

        max_n = int(ex.safety.max_samples_per_waveform)
        if cfg.scan_generator == "pyscanner_defined":
            order = rects if sparse else self._order(rects, ex.last_pos)
            level.rects = order
            head = (f"{name} (PyScanner {pattern}, lines {spacing:.3g} V apart)" if sparse else
                    f"{name} (PyScanner {pattern}, lines {spacing:.3g} V apart): {len(order)} region(s), "
                    f"{100 * level.area_fraction:.1f}% of the area")
            scans = [(i, s) for i, r in enumerate(order) for s in self._pyscanner_scans(r, spacing, pattern, sparse)]
            for j, (i, s) in enumerate(scans):
                where = "" if sparse else f" -- region {i + 1}/{len(order)}{self._what(order[i])}"
                part = f" [scan {j + 1}/{len(scans)}]" if len(scans) > 1 else ""
                ex.send_pyscanner(s, kind, meta={"label": f"{head}{where}{part}: {s.describe()}", "level": k},
                                  check_samples=not sparse)
        elif cfg.scan_generator != "user_defined":
            raise ValueError(f"Unknown scan_generator {cfg.scan_generator!r} "
                             f"(use 'user_defined' or 'pyscanner_defined')")
        elif sparse:
            x, y, desc = self._pattern(rects[0], spacing, pattern, ex.last_pos, sparse=True)
            label = f"{name}: {desc}, lines {spacing:.3g} V apart"
            if pattern == "raster" and cfg.sparse_crosshatch and len(x) > max_n:
                # too long for one waveform: send the two passes separately -- they meet at a
                # corner where the tip is at rest, so the split doesn't jerk the piezo
                x1, y1, _ = self._pattern(rects[0], spacing, pattern, ex.last_pos, sparse=False)
                ex.send(x1, y1, kind=kind, meta={"label": label + " (pass 1 of 2)", "level": k})
                ex.send(x[len(x1) - 1:], y[len(y1) - 1:], kind=kind,
                        meta={"label": label + " (pass 2 of 2)", "level": k})
            else:
                ex.send(x, y, kind=kind, meta={"label": label, "level": k})
        else:
            order = self._order(rects, ex.last_pos)
            level.rects = order
            head = (f"{name} ({pattern}, lines {spacing:.3g} V apart): {len(order)} region(s), "
                    f"{100 * level.area_fraction:.1f}% of the area")
            if cfg.batch_tiles:
                # tiles joined by smooth moves into as few waveforms as possible; a waveform
                # that would exceed max_samples_per_waveform is cut BETWEEN tiles (where the
                # tip is at rest), never in the middle of a motion
                chunks, xs, ys, n_cur = [], [], [], 0
                pos = ex.last_pos.copy()
                for r in order:
                    x, y, _ = self._pattern(r, spacing, pattern, pos)
                    mx = my = np.empty(0)
                    if xs:  # smooth move from the previous tile's end (at rest) to this start
                        mx, my = self._eased_move(pos, (x[0], y[0]))
                    if xs and n_cur + len(mx) + len(x) > max_n:
                        chunks.append((xs, ys))
                        xs, ys, n_cur = [], [], 0
                        mx = my = np.empty(0)  # the executor's own (eased) move starts the next one
                    xs += [mx, x]
                    ys += [my, y]
                    n_cur += len(mx) + len(x)
                    pos = np.array([x[-1], y[-1]])
                chunks.append((xs, ys))
                for i, (cx_, cy_) in enumerate(chunks):
                    part = " -- one waveform" if len(chunks) == 1 else f" -- waveform {i + 1}/{len(chunks)}"
                    ex.send(np.concatenate(cx_), np.concatenate(cy_), kind=kind,
                            meta={"label": head + part, "level": k})
            else:
                for i, r in enumerate(order):
                    x, y, desc = self._pattern(r, spacing, pattern, ex.last_pos)
                    ex.send(x, y, kind=kind, meta={"label": f"{head} -- region {i + 1}/{len(order)}{self._what(r)}: {desc}",
                                                   "level": k})
        level.n_samples = sum(len(s.x) for s in ex.history) - n_before
        level.scan_time_s = level.n_samples / self._rate

    # ------------------------------------------------------------------
    def summary(self) -> str:
        v = self.executor.velocity.tip_velocity_V_s
        if self.cfg.scan_generator == "pyscanner_defined":
            lines = [f"  PyScanner-defined scans: tip speed {'at most' if self.cfg.pyscanner_speed_basis == 'peak' else 'on average'} "
                     f"{v:g} V/s (PyScanner's sinusoidal profiles); moves between scans at {v:g} V/s"]
        else:
            lines = [f"  tip speed {v:g} V/s on every scan line, turnaround and curve "
                     f"(speed-up/slow-down only on the run-ups and the moves)"]
        for lv in self.levels:
            what = "sparse scan" if lv.index == 0 else f"refinement {lv.index}"
            lines.append(f"  level {lv.index} ({what}, {lv.pattern}): lines {lv.spacing_V:.3g} V apart, "
                         f"{len(lv.rects)} region(s){self._level_mix(lv)} = {100 * lv.area_fraction:.1f}% of the area, "
                         f"{lv.n_samples:,} samples = {lv.scan_time_s:.2f} s of scanning"
                         + (f", planning {1000 * lv.plan_time_s:.0f} ms" if lv.index else ""))
        return "\n".join(lines)
