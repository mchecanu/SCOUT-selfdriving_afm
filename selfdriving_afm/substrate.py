"""
On-the-fly substrate (tilt) correction for edge detection.

The usual AFM post-processing -- mask out the features, fit a low-order polynomial
(degree 1: a plane) to what's left, subtract it -- done DURING the scan, so the
planners decide "on a feature / on the substrate" from the height above the
substrate instead of from the raw signal.

Why it's needed (seen on the instrument, 2026-10-05): a sample tilted by ~0.26 V over
the 8 V area, with features ~0.3 V tall. The planners compared raw z with ONE level
taken where the tip started, and a threshold of 6 x the noise there (8 mV): the tilt
crossed that threshold all over the area, and the edge explorer followed the lines
of constant height on the tilted substrate as if they were edges (9 "features" for
2 real ones). After removing the plane the substrate is flat to ~7 mV and the
features stand 0.26-0.34 V above it, with an empty gap in between.

    model = SubstrateModel(cfg.edge_detect)
    model.calibrate(x, y, z)          # e.g. a small circle at the start: plane + noise
    on = model.classify(x, y, z)      # |z - plane| > threshold, per sample
    model.update(x, y, z, on)         # refine the plane from the substrate samples

The model:
  - PLANE (or degree-2 surface, `substrate_degree`): a weighted least-squares fit to
    every substrate (off-feature) sample measured so far, accumulated as normal
    equations -- each new waveform adds its samples (thinned to one per
    `substrate_sample_spacing_V` of path, so long waveforms don't dominate), and the
    fit stays a tiny (3x3) solve however long the run gets. A single straight line
    only tells the slope along itself; the other direction comes from what was
    measured before (the calibration circle gives both).
  - THRESHOLD, with no feature height known in advance (mode "auto_mad"):
      floor  = mad_k x the substrate's roughness: robust (MAD) spread of the height
               above the plane, over EVERYTHING measured so far -- not a 0.1 V
               stretch at the start, which only sees the fine noise (0.4 mV on the
               instrument) and not the substrate's own texture (~7 mV);
      gap    = once a second population of heights shows up (the features), the
               threshold moves into the gap between the two (Otsu's split of the
               |height| histogram, accepted only if the two populations are well
               separated -- see `_gap_threshold`); never below the floor.
    "manual": the fixed manual_threshold, on the height above the plane.
  - tilt_correction=False: the plane is replaced by a constant level (the old
    behaviour, kept for comparison).
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from .config import EdgeDetectConfig


def _basis(x, y, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cols = [np.ones_like(x)]
    if degree >= 1:
        cols += [x, y]
    if degree >= 2:
        cols += [x * x, x * y, y * y]
    return np.column_stack(cols)


def _thin(x, y, z, spacing: float, extra=None):
    """One sample per `spacing` of path length (at least a few) -- the fit weights path,
    not sample count, so a slow tip doesn't count a stretch many times over."""
    n = len(z)
    if n == 0:
        return x, y, z, extra
    step = np.hypot(np.diff(x), np.diff(y))
    L = float(step.sum())
    k = int(max(1, min(n // 4 if n >= 4 else 1, round(n * spacing / max(L, 1e-12)))))
    sl = slice(0, None, k)
    return x[sl], y[sl], z[sl], (None if extra is None else extra[sl])


def robust_fit(x, y, z, degree: int = 1, iters: int = 15):
    """IRLS fit of a degree-`degree` surface, Huber-like weights (residuals beyond
    2 sigma down-weighted): features covering a minority of the samples barely move
    it. Returns (coefficients, robust sigma of the residuals)."""
    A = _basis(x, y, degree)
    z = np.asarray(z, dtype=float)
    w = np.ones(len(z))
    c = np.zeros(A.shape[1])
    s = 0.0
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)
        r = z - A @ c
        s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        w = np.minimum(1.0, 2.0 * max(s, 1e-12) / np.maximum(np.abs(r), 1e-12))
    return c, s


def slope_fit(x, y, z, window_V: float = 0.01):
    """The substrate slope (dz/dx, dz/dy) from the DERIVATIVE of z along the path,
    robustly: over every `window_V` of path, dz = gx dx + gy dy (+ noise), fitted with
    Tukey's biweight. A feature edge is a step in z -- a tilt to any fit of z itself
    when it covers one side of a calibration circle (seen on the instrument: a circle
    a third on a feature gave 0.29 V/V for a 0.005 V/V tilt) -- but in dz it is only
    the few windows across the edge, which the biweight rejects. Needs the path to
    turn (a circle): a straight line gives the slope along itself only. Returns
    (gx, gy), or None if the path is too short or straight."""
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    if len(z) < 8:
        return None
    ds = np.hypot(np.diff(x), np.diff(y))
    L = float(ds.sum())
    k = int(max(1, round(len(z) * window_V / max(L, 1e-12))))
    if len(z) <= 4 * k:
        return None
    dx, dy, dz = x[k:] - x[:-k], y[k:] - y[:-k], z[k:] - z[:-k]
    D = np.column_stack([dx, dy])
    if np.linalg.cond(D.T @ D) > 1e6:
        return None
    g = np.zeros(2)
    for _ in range(25):
        r = dz - D @ g
        s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        c = 4.685 * max(s, 1e-15)
        u = r / c
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        Wd = D * w[:, None]
        try:
            g_new = np.linalg.solve(Wd.T @ D + 1e-18 * np.eye(2), Wd.T @ dz)
        except np.linalg.LinAlgError:
            return None
        if np.allclose(g_new, g, rtol=1e-6, atol=1e-12):
            g = g_new
            break
        g = g_new
    return float(g[0]), float(g[1])


def _mode_level(r: np.ndarray):
    """(level, spread) of the most populated height in r: the substrate under the start
    -- or a feature top, if the start sits on one; the run's later updates sort it out."""
    lo, hi = np.quantile(r, [0.005, 0.995])
    if hi - lo < 1e-12:
        return float(np.median(r)), 0.0
    h, e = np.histogram(r, bins=60, range=(lo, hi))
    k = int(np.argmax(h))
    w = e[1] - e[0]
    near = np.abs(r - 0.5 * (e[k] + e[k + 1])) <= 1.5 * w
    level = float(np.median(r[near]))
    spread = 1.4826 * float(np.median(np.abs(r[near] - level)))
    return level, spread


class SubstrateModel:
    """The substrate under the features, learned while scanning (see module docstring)."""

    # |height| samples kept for the roughness / gap estimates (a random reservoir)
    RESERVOIR = 200_000
    # gap threshold: the high population must be at least this fraction of the samples...
    GAP_MIN_FRACTION = 0.002
    # ...its mean at least this many floors above the low one's...
    GAP_MIN_SEPARATION = 3.0
    # ...and the histogram at the split this empty compared with the substrate's peak
    GAP_MAX_VALLEY = 0.02

    def __init__(self, cfg: EdgeDetectConfig, seed: int = 0):
        self.cfg = cfg
        self.degree = int(cfg.substrate_degree) if cfg.tilt_correction else 0
        m = self.degree
        self.n_coef = 1 + (2 if m >= 1 else 0) + (3 if m >= 2 else 0)
        self._AtA = np.zeros((self.n_coef, self.n_coef))
        self._Atz = np.zeros(self.n_coef)
        self.coef: Optional[np.ndarray] = None
        self._pts = np.empty((0, 3))      # reservoir of measured (x, y, z): heights re-evaluated
                                          #   against the CURRENT surface at every update
        self._seen = 0
        self._rng = np.random.default_rng(seed)
        self.sigma: Optional[float] = None
        self.threshold: Optional[float] = None
        self.gap_active = False           # the threshold sits in the gap between two populations
        self.history = []                 # (n samples so far, threshold, sigma, gap_active, coef)

    # ------------------------------------------------------------------
    @property
    def fitted(self) -> bool:
        return self.coef is not None

    def plane(self, x, y) -> np.ndarray:
        """The substrate level at (x, y) (0 before anything was measured)."""
        if self.coef is None:
            return np.zeros(np.shape(x))
        return (_basis(np.ravel(x), np.ravel(y), self.degree) @ self.coef).reshape(np.shape(x))

    def height(self, x, y, z) -> np.ndarray:
        """z above the substrate."""
        return np.asarray(z, dtype=float) - self.plane(x, y)

    # ------------------------------------------------------------------
    def calibrate(self, x, y, z) -> None:
        """Start the model from one waveform (the calibration circle, or the first
        scan). The slope from the derivative along the path (slope_fit -- immune to a
        feature edge on the circle, unlike a fit of z itself); the level from the most
        populated height once that slope is removed. Its samples at that level seed
        the running fit."""
        x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
        xt, yt, zt, _ = _thin(x, y, z, self.cfg.substrate_sample_spacing_V)
        if len(zt) < 2 * self.n_coef:
            xt, yt, zt = x, y, z
        g = slope_fit(x, y, z) if self.degree >= 1 else None
        if self.degree >= 1 and g is None:
            coef, _ = robust_fit(xt, yt, zt, self.degree)   # a straight first path: no better option
            r = zt - _basis(xt, yt, self.degree) @ coef
            level, spread = _mode_level(r)
            coef[0] += level
        else:
            gx, gy = g if g is not None else (0.0, 0.0)
            r = zt - (gx * xt + gy * yt if self.degree >= 1 else 0.0)
            level, spread = _mode_level(r)
            coef = np.zeros(self.n_coef)
            coef[0] = level
            if self.degree >= 1:
                coef[1:3] = gx, gy
        self.coef = coef
        r = zt - _basis(xt, yt, self.degree) @ coef
        off = np.abs(r) <= max(4.0 * spread, 1e-9)
        A = _basis(xt[off], yt[off], self.degree)
        self._AtA += A.T @ A
        self._Atz += A.T @ zt[off]
        self._add_points(x, y, z)
        self._update_threshold(len(z))

    def classify(self, x, y, z) -> np.ndarray:
        """On-feature mask: |height above the substrate| > threshold."""
        if self.coef is None:
            self.calibrate(x, y, z)
        return np.abs(self.height(x, y, z)) > self.threshold

    def update(self, x, y, z, on: np.ndarray) -> None:
        """Add a measured waveform: its substrate samples (~on) refine the surface, all
        its heights feed the roughness / gap estimates, and the threshold is re-set."""
        x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
        on = np.asarray(on, dtype=bool)
        if len(z) == 0:
            return
        if self.coef is None:
            self.calibrate(x, y, z)
            return
        xt, yt, zt, ont = _thin(x, y, z, self.cfg.substrate_sample_spacing_V, on)
        keep = ~ont
        if keep.sum() >= 2:
            A = _basis(xt[keep], yt[keep], self.degree)
            self._AtA += A.T @ A
            self._Atz += A.T @ zt[keep]
            # ridge toward the current fit: one straight line alone can't move the slope
            # across itself
            lam = 1e-9 * max(np.trace(self._AtA), 1.0)
            try:
                self.coef = np.linalg.solve(self._AtA + lam * np.eye(self.n_coef), self._Atz + lam * self.coef)
            except np.linalg.LinAlgError:
                pass
        self._add_points(x, y, z)
        self._update_threshold(len(z))

    # ------------------------------------------------------------------
    def _add_points(self, x, y, z) -> None:
        """Reservoir sample of the measured (x, y, z) (thinned along the path first)."""
        k = max(1, len(z) // 20_000)
        P = np.column_stack([x[::k], y[::k], z[::k]])
        self._seen += len(P)
        room = self.RESERVOIR - len(self._pts)
        if room > 0:
            self._pts = np.vstack([self._pts, P[:room]])
            P = P[room:]
        if len(P):
            # replace random entries with probability RESERVOIR / seen
            pick = self._rng.random(len(P)) < self.RESERVOIR / self._seen
            if pick.any():
                self._pts[self._rng.integers(0, self.RESERVOIR, pick.sum())] = P[pick]

    def _update_threshold(self, n_new: int) -> None:
        P = self._pts
        a = np.abs(self.height(P[:, 0], P[:, 1], P[:, 2])) if len(P) else np.empty(0)
        # roughness: robust spread of the substrate heights. The plane is fitted to the
        # substrate, so its heights scatter around 0. From the lower quartile of |h|
        # (= 0.3186 sigma for a Gaussian), not the median: early in a run the features can
        # be close to half of what was measured (a first line across one), and the median
        # would then measure the features instead of the substrate
        self.sigma = float(np.quantile(a, 0.25)) / 0.3186 if len(a) else 0.0
        if self.cfg.mode == "manual":
            if self.cfg.manual_threshold is None:
                raise ValueError("EdgeDetectConfig.mode == 'manual' requires manual_threshold to be set")
            self.threshold = float(self.cfg.manual_threshold)
            self.gap_active = False
        else:
            floor = self.cfg.mad_k * max(self.sigma, 1e-12)
            gap = self._gap_threshold(a, floor) if self.cfg.use_height_gap else None
            self.gap_active = gap is not None
            self.threshold = max(floor, gap) if gap is not None else floor
        self.history.append((self._seen, self.threshold, self.sigma, self.gap_active,
                             None if self.coef is None else self.coef.copy()))

    def _gap_threshold(self, a: np.ndarray, floor: float) -> Optional[float]:
        """Otsu split of the |height| histogram, if it separates two real populations:
        the upper one a non-negligible fraction of the samples, its mean well above the
        lower one's, and an (almost) empty histogram at the split. None otherwise --
        no features seen yet, or features only marginally taller than the roughness."""
        if len(a) < 1000:
            return None
        hi = float(np.quantile(a, 0.9995))
        if hi <= 2 * floor:
            return None
        bins = np.linspace(0.0, hi, 257)
        h, e = np.histogram(a, bins=bins)
        c = 0.5 * (e[1:] + e[:-1])
        w0 = np.cumsum(h)
        w1 = w0[-1] - w0
        m0 = np.cumsum(h * c)
        mu0 = m0 / np.maximum(w0, 1)
        mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
        between = (w0 * w1 * (mu0 - mu1) ** 2)[:-1]
        # every split inside an empty stretch of the histogram scores the same: take the
        # middle of the best-scoring stretch, i.e. the middle of the gap
        best = np.flatnonzero(between >= (1 - 1e-6) * between.max())
        k = int(best[len(best) // 2])
        t = float(e[k + 1])
        frac_hi = w1[k] / w0[-1]
        if frac_hi < self.GAP_MIN_FRACTION or (mu1[k] - mu0[k]) < self.GAP_MIN_SEPARATION * floor:
            return None
        # an (almost) empty valley at the split: compare the density around t with the
        # substrate's peak
        win = max(2, len(h) // 64)
        lo_i, hi_i = max(0, k - win), min(len(h), k + win + 1)
        valley = h[lo_i:hi_i].mean() / max(h.max(), 1)
        if valley > self.GAP_MAX_VALLEY:
            return None
        return t

    def summary(self) -> dict:
        return dict(coef=None if self.coef is None else self.coef.tolist(), degree=self.degree,
                    sigma=self.sigma, threshold=self.threshold, gap_active=self.gap_active)
