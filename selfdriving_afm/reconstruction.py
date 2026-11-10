"""
Image reconstruction from a sparse scan path.

Everything starts from the same PyScanner-style binned image (`bin_image`: mean z
per pixel, unvisited pixels NaN -- identical convention to image_AI0), then fills
in the unvisited pixels with one of (config.LiveViewConfig.reconstruction):

  - "compressed_sensing": the image (minus its best-fit plane -- a tilt is NOT sparse
    in the DCT) is assumed sparse in the 2D DCT basis; solves
        min_c  1/2 ||M * IDCT(c) - y||^2 + lam * ||c||_1
    over the visited-pixel mask M with FISTA. ~0.2 s at 200x200 with 500 iterations.
  - "gp": Gaussian-process regression trained on the visited pixels (subsampled to
    gp_max_points), predicted on every pixel. Matern 5/2 + white noise by default, as
    in Checa et al., Nat. Commun. 14, 7196 (2023); with gp_optimize=True the kernel
    hyperparameters (length scale, noise) are learned from the data (~40 s for 3000
    points) -- a fixed RBF length scale rings around sharp edges. O(n^3) in training
    points, so gp_max_points is the speed knob.
  - "linear": linear interpolation between the measured pixels (Delaunay), nearest
    measured pixel outside their hull. ~0.3 s; on step-like topography (flat
    terraces, sharp edges) it had the fewest wrong pixels in simulation.
  - "fast": normalized convolution (fast_inpaint) -- a Gaussian-weighted average of
    the nearest measured pixels; ~20 ms, purely local (no ghosts far from the data).
  - "tv": total-variation inpainting (tv_inpaint): flat plateaus with sharp edges,
    measured pixels weighted by tv_mu (lower = more of the noise smoothed). On the good
    grid_refine dataset (2026-11-03), scored on the equivalent raster's pixels, it had
    the lowest shape and edge errors and the best object overlap (linear 2nd, CS last).
    ~10 s at 256x256 with 1500 iterations -- for saving/analysis, not the live view.
  - "interpolated": the older griddata fill (viz.reconstruct_topography).

`reconstruct_image` is the single entry point used by both the live view and the
notebook's save cell, so what you watch is exactly what gets saved.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.fft import dctn, idctn

from .config import LiveViewConfig


def pixel_index(x: np.ndarray, y: np.ndarray, n: int, bounds) -> np.ndarray:
    """Flat pixel index (row * n + col) of every sample on bin_image's grid: pixel =
    round(t * (n - 1)), clipped to the frame -- PyScanner's image_AI0 convention."""
    xmin, xmax, ymin, ymax = bounds
    col = np.clip(np.round((x - xmin) / max(xmax - xmin, 1e-12) * (n - 1)), 0, n - 1).astype(np.int64)
    row = np.clip(np.round((y - ymin) / max(ymax - ymin, 1e-12) * (n - 1)), 0, n - 1).astype(np.int64)
    return row * n + col


def bin_image(points_xy: np.ndarray, z: np.ndarray, n_pixels: int = 200, bounds=None):
    """PyScanner-style binned image: the mean z of every sample in each pixel, NO
    interpolation -- unvisited pixels are NaN. Same pixel convention PyScanner uses
    for image_AI0 (checked against its image_counts on hardware): frame = min/max
    of the samples unless `bounds` is given, pixel index = round(t * (n - 1)).
    Returns (image[row=y, col=x], (xmin, xmax, ymin, ymax)) -- plot with
    imshow(origin="lower", extent=...)."""
    x, y = points_xy[:, 0], points_xy[:, 1]
    if bounds is None:
        bounds = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    n = n_pixels
    flat = pixel_index(x, y, n, bounds)
    counts = np.bincount(flat, minlength=n * n)
    sums = np.bincount(flat, weights=z, minlength=n * n)
    with np.errstate(invalid="ignore", divide="ignore"):
        img = np.where(counts > 0, sums / counts, np.nan)
    return img.reshape(n, n), bounds


def uncertainty_map(binned: np.ndarray, bounds, scale_V: Optional[float] = None) -> np.ndarray:
    """How far each pixel's reconstruction is from any measurement, 0..1: 0 on a measured
    pixel (finite in `binned`), rising linearly with the distance to the nearest measured
    pixel, 1 at scale_V or more -- a gap that wide could hide a whole feature, so use the
    smallest expected feature (priors.feature_min_V); None = 5% of the frame width. The
    same for every scan pattern and reconstruction method: it says where the data is,
    not how well a method guesses between it."""
    from scipy.ndimage import distance_transform_edt
    binned = np.asarray(binned, dtype=float)
    M = np.isfinite(binned)
    xmin, xmax, ymin, ymax = bounds
    if scale_V is None or not scale_V > 0:
        scale_V = 0.05 * max(xmax - xmin, ymax - ymin)
    if not M.any():
        return np.ones(binned.shape)
    ny, nx = binned.shape
    dy, dx = (ymax - ymin) / max(ny - 1, 1), (xmax - xmin) / max(nx - 1, 1)
    d = distance_transform_edt(~M, sampling=(dy, dx))
    return np.clip(d / scale_V, 0.0, 1.0)


def fit_plane(img: np.ndarray) -> np.ndarray:
    """Least-squares plane through the measured (finite) pixels of img, evaluated on
    every pixel (row = y, col = x) -- e.g. a sample tilt to remove before CS."""
    M = np.isfinite(img)
    ny, nx = img.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    if M.sum() < 3:
        return np.full(img.shape, float(np.nanmean(img)) if M.any() else 0.0)
    A = np.column_stack([X[M], Y[M], np.ones(M.sum())])
    co, *_ = np.linalg.lstsq(A, img[M], rcond=None)
    return co[0] * X + co[1] * Y + co[2]


def _soft(c: np.ndarray, t: float) -> np.ndarray:
    # soft threshold sign(c) * max(|c| - t, 0), written as c - clip(c, -t, t): the
    # identical result (bit for bit), ~20% faster CS overall
    return c - np.clip(c, -t, t)


def cs_reconstruct(binned: np.ndarray, lam_frac: float = 0.005, n_iter: int = 500,
                   detrend: bool = True) -> np.ndarray:
    """Compressed-sensing inpainting of a binned image (NaN = unmeasured), assuming
    sparsity in the orthonormal 2D DCT. FISTA with step size 1 (the masked
    orthonormal operator has norm <= 1). `lam_frac` sets the L1 weight relative to
    the largest DCT coefficient of the zero-filled data: higher = smoother/sparser,
    lower = follows the measured pixels more tightly. detrend=True removes the
    best-fit plane first (and adds it back) -- a tilted sample is not sparse in the
    DCT, so without this CS spends its sparsity on the tilt and blurs the features."""
    mask = np.isfinite(binned)
    if not mask.any():
        return np.full_like(binned, np.nan)
    offset = fit_plane(binned) if detrend else float(np.mean(binned[mask]))
    y = np.where(mask, binned - offset, 0.0)

    c = dctn(y, norm="ortho")
    lam = lam_frac * float(np.max(np.abs(c)))
    v, t = c.copy(), 1.0
    for _ in range(n_iter):
        resid = mask * (idctn(v, norm="ortho") - y)
        c_new = _soft(v - dctn(resid, norm="ortho"), lam)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        v = c_new + ((t - 1.0) / t_new) * (c_new - c)
        c, t = c_new, t_new
    return idctn(c, norm="ortho") + offset


def gp_reconstruct(binned: np.ndarray, bounds, max_points: int = 1500,
                   length_scale_frac: float = 0.05, optimize: bool = True,
                   rng: np.random.Generator = None, kernel: str = "matern52") -> np.ndarray:
    """GP regression over the visited pixels of a binned image, predicted on every
    pixel. Trained on at most `max_points` visited pixels (random subset) -- this is
    the speed knob, the fit is O(max_points^3). kernel = "matern52" (Matern 5/2, as
    in the SS-KPFM paper -- less over-smooth than RBF, no ringing at edges) or "rbf".
    The length scale starts at `length_scale_frac` x the frame width; optimize=True
    learns it (and the noise level) from the data, optimize=False keeps it fixed
    (fast, for the live view)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

    rng = rng or np.random.default_rng(0)
    mask = np.isfinite(binned)
    if mask.sum() < 3:
        return np.full_like(binned, np.nan)
    n_rows, n_cols = binned.shape
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, n_cols)
    ys = np.linspace(ymin, ymax, n_rows)
    Xg, Yg = np.meshgrid(xs, ys)

    rows, cols = np.nonzero(mask)
    if len(rows) > max_points:
        keep = rng.choice(len(rows), size=max_points, replace=False)
        rows, cols = rows[keep], cols[keep]
    X_train = np.column_stack([Xg[rows, cols], Yg[rows, cols]])
    z_train = binned[rows, cols]

    width = max(xmax - xmin, ymax - ymin, 1e-9)
    ls = length_scale_frac * width
    if kernel == "matern52":
        base = Matern(ls, (ls / 20, width), nu=2.5)
    elif kernel == "rbf":
        base = RBF(ls, (ls / 20, width))
    else:
        raise ValueError(f"Unknown GP kernel {kernel!r} (use 'matern52' or 'rbf')")
    kern = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(1e-2, (1e-6, 1e0))
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, alpha=1e-8,
                                  optimizer="fmin_l_bfgs_b" if optimize else None)
    gp.fit(X_train, z_train)

    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    out = np.empty(len(pts))
    for i in range(0, len(pts), 5000):  # chunked -- a full 200x200 K_* is ~40k x n_train
        out[i:i + 5000] = gp.predict(pts[i:i + 5000])
    return out.reshape(binned.shape)


def linear_inpaint(img: np.ndarray) -> np.ndarray:
    """Fill the unmeasured (NaN) pixels by linear interpolation between the measured
    ones (Delaunay triangulation), nearest measured pixel outside their hull. Measured
    pixels are kept. ~0.3 s at 200^2."""
    from scipy.interpolate import griddata
    M = np.isfinite(img)
    if M.sum() < 3:
        return np.where(M, img, float(np.nanmean(img)) if M.any() else 0.0)
    rows, cols = np.nonzero(M)
    pts = np.column_stack([cols, rows]).astype(float)
    vals = img[M]
    out = img.copy()
    hr, hc = np.nonzero(~M)
    if len(hr):
        try:
            out[hr, hc] = griddata(pts, vals, (hc, hr), method="linear")
        except Exception:  # all measured pixels collinear (e.g. one line) -- nearest only
            pass
        left = ~np.isfinite(out)
        if left.any():
            lr, lc = np.nonzero(left)
            out[lr, lc] = griddata(pts, vals, (lc, lr), method="nearest")
    return out


def tv_inpaint(img: np.ndarray, mu: Optional[float] = 0.02, n_iter: int = 1500) -> np.ndarray:
    """Total-variation inpainting (Chambolle-Pock primal-dual):
        min_u  TV(u) + mu/2 ||M * (u - f)||^2
    over the measured-pixel mask M (mu None = measured pixels kept exactly). Starts from
    linear_inpaint. Favours flat plateaus with sharp edges (step-like topography).
    Scale-free: the measured values are scaled to a 1..99% range of 100 first (mu was tuned on
    ~100 nm heights; a PFM amplitude of ~0.003 V came out flat without it, 2026-11-04)."""
    M = np.isfinite(img)
    if M.sum() < 3:
        return linear_inpaint(img)
    lo, hi = np.percentile(img[M], [1, 99])
    s = (hi - lo) / 100.0 if hi > lo else 1.0
    f = np.where(M, (img - lo) / s, 0.0)
    u = (linear_inpaint(img).astype(float) - lo) / s
    ub = u.copy()
    px = np.zeros_like(u); py = np.zeros_like(u)
    tau = sigma = 1 / np.sqrt(8)
    for _ in range(n_iter):
        px += sigma * np.diff(ub, axis=1, append=ub[:, -1:])
        py += sigma * np.diff(ub, axis=0, append=ub[-1:, :])
        nrm = np.maximum(1.0, np.hypot(px, py)); px /= nrm; py /= nrm
        div = np.diff(px, axis=1, prepend=0 * px[:, :1]) + np.diff(py, axis=0, prepend=0 * py[:1, :])
        u_old = u
        u = u + tau * div
        u = np.where(M, f, u) if mu is None else np.where(M, (u + tau * mu * f) / (1 + tau * mu), u)
        ub = 2 * u - u_old
    return u * s + lo


def fast_inpaint(img: np.ndarray, sigma_px: float = 2.0) -> np.ndarray:
    """Fill the unmeasured (NaN) pixels by normalized convolution: a Gaussian-weighted
    average of the measured pixels around each hole (sigma ~ half the line spacing),
    widening the Gaussian (x3) for holes further than that from any data. Measured
    pixels are kept as they are. Box/Gaussian filters only: ~10-30 ms at 512^2 --
    purely local, so no ghost copies of features in sparsely scanned areas (as the
    DCT-based CS can give); edges are blurred over ~ the gap between scan lines."""
    from scipy.ndimage import gaussian_filter
    M = np.isfinite(img)
    if not M.any():
        return np.zeros(img.shape)
    Mf = M.astype(float)
    Z = np.where(M, img, 0.0)
    out = img.copy()
    hole = ~M
    s = max(float(sigma_px), 0.5)
    for _ in range(6):
        if not hole.any():
            break
        w = gaussian_filter(Mf, s, mode="constant", truncate=3.0)
        v = gaussian_filter(Z, s, mode="constant", truncate=3.0)
        fill = hole & (w > 1e-3)
        out[fill] = v[fill] / w[fill]
        hole &= ~fill
        s *= 3.0
    out[hole] = float(np.mean(img[M]))
    return out


def raster_lines_along(img: np.ndarray) -> str:
    """Which way a raster image's scan lines run: "x" if its measured pixels form (mostly)
    whole pixel ROWS, "y" if whole COLUMNS -- the lines are dense along themselves and
    missing between them. Ties / neither (not a raster): "x"."""
    M = np.isfinite(img)
    rows_full = int((M.mean(axis=1) > 0.5).sum())
    cols_full = int((M.mean(axis=0) > 0.5).sum())
    return "y" if cols_full > rows_full else "x"


def column_inpaint(img: np.ndarray, lines_along: str = "auto") -> np.ndarray:
    """Fill the unmeasured (NaN) pixels of a RASTER image across its scan lines: with
    lines along x, each pixel column is interpolated in y between the lines that crossed
    it (lines along y: each pixel row, in x), with a shape-preserving cubic (PCHIP -- no
    overshoot at steps), constant beyond the first and last line. A raster is dense along
    its lines and missing only between them, so this is the natural fill (much better
    than CS, whose DCT prior aliases on regularly skipped lines). Measured pixels are kept.
    lines_along: "x", "y" or "auto" (detected from which way the measured pixels run --
    raster_lines_along). ~0.01-0.05 s at 200^2."""
    if lines_along == "auto":
        lines_along = raster_lines_along(img)
    if lines_along == "y":
        return column_inpaint(np.asarray(img).T, "x").T
    if lines_along != "x":
        raise ValueError(f"lines_along must be 'x', 'y' or 'auto', not {lines_along!r}")
    from scipy.interpolate import PchipInterpolator
    M = np.isfinite(img)
    out = img.copy()
    rows = np.arange(img.shape[0])
    has = M.any(axis=0)
    if not has.any():
        return np.zeros(img.shape)
    # columns crossed by the same lines are interpolated together (a raster has one or
    # two such groups: the full lines, plus the part the last, partial line reached)
    groups = {}
    for c in np.nonzero(has)[0]:
        groups.setdefault(M[:, c].tobytes(), []).append(c)
    for cols in groups.values():
        cols = np.asarray(cols)
        m = M[:, cols[0]]
        r, v = rows[m], img[m][:, cols]
        if len(r) == 1:
            out[:, cols] = v[0]
            continue
        f = PchipInterpolator(r, v, axis=0, extrapolate=False)(rows)
        f[rows < r[0]], f[rows > r[-1]] = v[0], v[-1]
        out[:, cols] = np.where(M[:, cols], img[:, cols], f)
    if not has.all():  # columns no line reached (a partial last line only) -- nearest column
        hc = np.nonzero(has)[0]
        for c in np.nonzero(~has)[0]:
            out[:, c] = out[:, hc[np.argmin(np.abs(hc - c))]]
    return out


def inpaint_binned(binned: np.ndarray, bounds, cfg: LiveViewConfig, rng=None, mode: str = None):
    """Fill a binned image's unmeasured pixels with `mode` (default cfg.reconstruction)
    and cfg's parameters. Returns None for "interpolated" (which needs the raw points
    -- see reconstruct_image)."""
    mode = mode or cfg.reconstruction
    if mode == "compressed_sensing":
        return cs_reconstruct(binned, cfg.cs_lambda, cfg.cs_iterations, cfg.cs_detrend)
    if mode == "gp":
        return gp_reconstruct(binned, bounds, cfg.gp_max_points, cfg.gp_length_scale_frac,
                              cfg.gp_optimize, rng, cfg.gp_kernel)
    if mode == "linear":
        return linear_inpaint(binned)
    if mode == "fast":
        return fast_inpaint(binned, 1.0)
    if mode == "tv":
        return tv_inpaint(binned, cfg.tv_mu, cfg.tv_iterations)
    if mode == "raster_columns":
        return column_inpaint(binned)
    if mode == "interpolated":
        return None
    raise ValueError(f"Unknown reconstruction {mode!r} (use 'compressed_sensing', 'gp', 'linear', "
                     f"'fast', 'tv', 'raster_columns' or 'interpolated')")


def reconstruct_image(points_xy: np.ndarray, z: np.ndarray, bounds, cfg: LiveViewConfig,
                      rng: np.random.Generator = None):
    """Bin every point onto an n_image_pixels grid over `bounds`, then fill it in
    with cfg.reconstruction. Returns (binned, reconstructed) -- both
    [row=y, col=x] arrays over `bounds`; plot with imshow(origin="lower",
    extent=bounds)."""
    binned, _ = bin_image(points_xy, z, cfg.n_image_pixels, bounds)
    recon = inpaint_binned(binned, bounds, cfg, rng)
    if recon is None:  # "interpolated": griddata from the raw points
        from .viz import reconstruct_topography, subsample_for_reconstruction
        xmin, xmax, ymin, ymax = bounds
        n = cfg.n_image_pixels
        Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
        xy, zz = subsample_for_reconstruction(points_xy, z, cfg.max_reconstruction_points, rng)
        recon = reconstruct_topography(xy, zz, (Xg, Yg))
    return binned, recon


def flattened_reconstruction(points_xy: np.ndarray, z: np.ndarray, bounds, cfg: LiveViewConfig,
                             n_iter: Optional[int] = None, max_plane_points: int = 200_000):
    """Binned image + reconstruction (cfg.reconstruction) of scattered samples with the
    substrate plane removed (robust plane: least squares, then refit within 3 robust sigma --
    the features left out), substrate level at 0. For path data (gp_path) whose map should
    keep flat plateaus and sharp steps: use cfg.reconstruction = "tv".
    Returns (binned, recon, plane) -- plane = (a, b, c): z_substrate = a + b x + c y."""
    pts = np.asarray(points_xy, float)
    z = np.asarray(z, float)
    # the substrate plane from the BINNED image (every place counts once -- dense follow /
    # interior scans on the features must not outweigh the substrate), iterated on the lower
    # level: least squares, then keep the pixels below the 40th percentile of the residual +
    # 3 robust sigma of that lower part, repeat (features protrude; the substrate is the floor)
    B0, _ = bin_image(pts, z, cfg.n_image_pixels, bounds)
    n = B0.shape[0]
    yy, xx = np.nonzero(np.isfinite(B0))
    x0, x1, y0, y1 = bounds
    px = x0 + xx / max(n - 1, 1) * (x1 - x0)
    py = y0 + yy / max(n - 1, 1) * (y1 - y0)
    zz = B0[yy, xx]
    A = np.column_stack([np.ones(len(zz)), px, py])
    keep = np.ones(len(zz), bool)
    coef = np.zeros(3)
    for _ in range(10):
        if keep.sum() < 3:
            break
        coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
        r_ = zz - A @ coef
        lo_ = r_ <= np.percentile(r_, 40)
        sig = 1.4826 * np.median(np.abs(r_[lo_] - np.median(r_[lo_]))) + 1e-12
        new_keep = r_ <= np.percentile(r_, 40) + 3 * sig
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    r = z - (coef[0] + pts @ coef[1:])
    lo = float(np.median((zz - A @ coef)[keep])) if keep.any() else 0.0
    r = r - lo
    coef = np.array([coef[0] + lo, coef[1], coef[2]])
    binned, _ = bin_image(pts, r, cfg.n_image_pixels, bounds)
    if (cfg.reconstruction == "tv") and n_iter is not None:
        recon = tv_inpaint(binned, cfg.tv_mu, n_iter)
    else:
        recon = inpaint_binned(binned, bounds, cfg)
    return binned, recon, coef
