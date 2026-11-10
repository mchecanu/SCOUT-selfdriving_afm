"""
GP back ends for the gp_path planner (GPPathConfig.gp_backend).

All of them use the same kernel as before -- C * RBF + White, with normalize_y -- and
the same length-scale prior/bounds from the priors; they differ only in how much work
each refit does. Benchmarked in simulation on the hexagon sample at the Run-notebook
settings (gp_benchmark/, 2026-11-06): at ~6,000 training points one decision costs

  "sklearn_exact"  ~1,100 s   what the planner always did: re-optimise the kernel
                              hyperparameters (L-BFGS) on ALL points at every refit
  "frozen"         ~2-5 s     learn the hyperparameters while n < freeze_after_n (on at
                              most hyper_opt_subset points), then freeze them: a refit is
                              one Cholesky, O(n^3) but no optimiser -- still grows with n
  "local"          ~0.02 s    "frozen", but each refit uses only the local_n_points
                              training points nearest the tip -- constant cost per refit
                              however long the run (a global sparse model is built on
                              demand for whole-area maps: live view, stop check, save)
  "sparse"         ~0.03 s    sparse variational GP (Titsias 2009 / SGPR) with inducing
                              points on a grid over the cells the path has visited, frozen
                              hyperparameters: O(n m^2), whole-area model at all times

(2-core CPU; worst per-segment stall over a 60-segment run: sklearn_exact 256-263 s,
frozen 1.4-6.8 s, local / sparse 0.7-1.9 s -- the worst step is the warm-up while the
hyperparameters are still being learned, typical steps are 0.02-0.06 s.) "sparse" made
the same direction choice as "sklearn_exact" at 6/6 replay checkpoints, "frozen" and
"local" at 5/6 (the one miss was the neighbouring direction).

Pure numpy / scipy / scikit-learn -- no torch, no GPU needed.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

BACKENDS = ("sklearn_exact", "frozen", "local", "sparse")
_ALPHA = 1e-8   # same diagonal regulariser as the sklearn GPR always used


NU = {"rbf": np.inf, "matern52": 2.5, "matern32": 1.5, "matern12": 0.5}


def make_kernel(length_scale: float, bounds, kind: str = "rbf"):
    """bounds = (lo, hi), or "fixed" to keep the length scale at length_scale. kind: "rbf"
    (smooth -- rings at sharp steps) or "matern52"/"matern32"/"matern12" (rougher; 1/2 does
    not overshoot at steps)."""
    b = "fixed" if bounds == "fixed" else tuple(bounds)
    nu = NU[kind]
    k = (RBF(length_scale=length_scale, length_scale_bounds=b) if np.isinf(nu)
         else Matern(length_scale=length_scale, length_scale_bounds=b, nu=nu))
    return (C(1.0, (1e-2, 1e2)) * k
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0)))


def _kernel_params(kernel):
    """(constant, length_scale, noise) out of a fitted C * RBF + White kernel."""
    p = kernel.get_params()
    return float(p["k1__k1__constant_value"]), float(p["k1__k2__length_scale"]), float(p["k2__noise_level"])


def _rbf(A, B, c, ls, nu=np.inf):
    """Stationary covariance: RBF (nu = inf) or Matern nu = 0.5 / 1.5 / 2.5."""
    if np.isinf(nu):
        return c * np.exp(-0.5 * cdist(A / ls, B / ls, "sqeuclidean"))
    d = cdist(A / ls, B / ls)
    if nu == 0.5:
        return c * np.exp(-d)
    if nu == 1.5:
        q = np.sqrt(3) * d
        return c * (1 + q) * np.exp(-q)
    q = np.sqrt(5) * d
    return c * (1 + q + q * q / 3) * np.exp(-q)


def _chol(K, jitter=0.0):
    """Lower Cholesky; adds growing jitter if K is numerically not positive definite."""
    I = np.eye(len(K))
    for j in [jitter] + [1e-10 * 10 ** k for k in range(8)]:
        try:
            return np.linalg.cholesky(K + j * I)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("GP covariance not positive definite even with jitter 1e-3")


class _Base:
    """fit(X, y, center=None) / predict(X, return_std=True) / kernel_ -- the subset of the
    sklearn GaussianProcessRegressor interface the planner, live view and save use."""
    name = "base"

    def __init__(self, cfg, length_scale, bounds):
        self.cfg, self.ls0 = cfg, length_scale
        self.nu = NU[getattr(cfg, 'gp_kernel', 'rbf')]
        self.kernel = make_kernel(length_scale, bounds, getattr(cfg, 'gp_kernel', 'rbf'))
        self.kernel_ = self.kernel
        self.n_train = 0
        self.fit_log = []            # (n points, seconds, hyperparameters optimised?)

    def describe(self) -> str:
        return f"{self.name}: {self.kernel_}"

    def prior_std(self) -> float:
        """The GP's prior standard deviation in z units (sigma far from any data)."""
        c, _, noise = _kernel_params(self.kernel_)
        return float(np.sqrt(c + noise) * self._y_scale())

    def _y_scale(self) -> float:
        return float(getattr(self, "ys", 1.0))


class SklearnExactGP(_Base):
    """The planner's original GP, unchanged (kept as the default and as the reference)."""
    name = "sklearn_exact"

    def __init__(self, cfg, length_scale, bounds):
        super().__init__(cfg, length_scale, bounds)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=cfg.gp_n_restarts_optimizer,
                                           alpha=_ALPHA, normalize_y=True)

    def fit(self, X, y, center=None):
        t = time.perf_counter()
        self.gp.fit(X, y)
        self.kernel_, self.n_train = self.gp.kernel_, len(X)
        self.fit_log.append((len(X), time.perf_counter() - t, True))
        return self

    def predict(self, Xs, return_std=True):
        return self.gp.predict(Xs, return_std=return_std)

    def predict_global(self, Xs):
        return self.predict(Xs)

    def _y_scale(self):
        return float(np.atleast_1d(getattr(self.gp, "_y_train_std", 1.0))[0])


class _FrozenHyper(_Base):
    """Hyperparameters: optimised with sklearn (same kernel, same bounds) on at most
    hyper_opt_subset random points until n >= freeze_after_n; then frozen. Every learn
    starts from the PRIOR kernel (the priors' length scale), as the original planner's
    fits did -- warm-starting from the previous result lets the length scale drift to its
    lower bound while the path is still on bare substrate (seen in simulation, 2026-11-06).
    Subclasses build their own fast predictor from the frozen values; whole-area maps
    (predict_global: live view, stop check, saved reconstruction) always use a sparse
    model on all the data, built on demand -- an exact solve over a 300 x 300 map is
    minutes at ~6000 points."""

    def __init__(self, cfg, length_scale, bounds):
        super().__init__(cfg, length_scale, bounds)
        self.frozen = False
        self._rng = np.random.default_rng(0)

    def _learn(self, X, y):
        if self.frozen:
            return False
        m = self.cfg.hyper_opt_subset
        idx = slice(None) if len(X) <= m else self._rng.choice(len(X), m, replace=False)
        gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=self.cfg.gp_n_restarts_optimizer,
                                      alpha=_ALPHA, normalize_y=True).fit(X[idx], y[idx])
        self.kernel_ = gp.kernel_
        if len(X) >= self.cfg.freeze_after_n:
            self.frozen = True
        return True

    def _spacing(self):
        return self.cfg.sparse_inducing_spacing_V or self.ls0 / 4

    def _keep_all(self, X, y):
        self._all, self._global = (X, y), None

    def predict_global(self, Xs):
        if self._global is None:
            X, y = self._all
            ym, ys = float(np.mean(y)), float(np.std(y)) or 1.0
            self._global = (_SparseSolver(X, (y - ym) / ys, self.c, self.ls, self.noise, self._spacing(), self.nu), ym, ys)
        s, ym, ys = self._global
        mu, sd = s.predict(np.asarray(Xs, float))
        return mu * ys + ym, sd * ys

    def variance_reduction(self, G, C, noise_div: float = 1.0) -> np.ndarray:
        """How much the posterior variance at the points G would drop if the candidate
        points C[d] (shape (D, k, 2)) were measured -- for each candidate d, before measuring
        anything: a GP's posterior variance depends only on WHERE it has data, not on the
        values. Returns (D, len(G)) in normalized-variance units. noise_div: each candidate
        point stands for this many real samples (their noise averages down)."""
        s = self.solver
        D, k, _ = C.shape
        Cf = C.reshape(-1, 2)
        Kgc = s.cov(G, Cf)
        Kcc = s.cov(Cf, Cf)
        nz = (self.noise + _ALPHA) / max(noise_div, 1.0)
        out = np.empty((D, len(G)))
        for d in range(D):
            sl = slice(d * k, (d + 1) * k)
            L = _chol(Kcc[sl, sl] + nz * np.eye(k))
            V = solve_triangular(L, Kgc[:, sl].T, lower=True, check_finite=False)
            out[d] = np.einsum("ij,ij->j", V, V)
        return out

    def _set_data(self, X, y):
        self.ym = float(np.mean(y))
        self.ys = float(np.std(y)) or 1.0          # sklearn normalize_y does the same
        self.c, self.ls, self.noise = _kernel_params(self.kernel_)
        return (y - self.ym) / self.ys


class _ExactSolver:
    """Exact GP solve on (X, yn) with fixed (c, ls, noise) -- shared by frozen and local."""

    def __init__(self, X, yn, c, ls, noise, nu=np.inf):
        self.X, self.c, self.ls, self.noise, self.nu = X, c, ls, noise, nu
        K = _rbf(X, X, c, ls, nu)
        K[np.diag_indices_from(K)] += noise + _ALPHA
        self.L = _chol(K)
        self.alpha = cho_solve((self.L, True), yn)

    def predict(self, S, chunk=4000):
        mu, var = np.empty(len(S)), np.empty(len(S))
        for i in range(0, len(S), chunk):
            Ks = _rbf(self.X, S[i:i + chunk], self.c, self.ls, self.nu)
            mu[i:i + chunk] = Ks.T @ self.alpha
            v = solve_triangular(self.L, Ks, lower=True, check_finite=False)
            var[i:i + chunk] = self.c + self.noise - np.einsum("ij,ij->j", v, v)
        return mu, np.sqrt(np.clip(var, 1e-12, None))

    def cov(self, A, B):
        """Posterior covariance of the latent surface between point sets A and B."""
        va = solve_triangular(self.L, _rbf(self.X, A, self.c, self.ls, self.nu), lower=True, check_finite=False)
        vb = va if B is A else solve_triangular(self.L, _rbf(self.X, B, self.c, self.ls, self.nu),
                                                lower=True, check_finite=False)
        return _rbf(A, B, self.c, self.ls, self.nu) - va.T @ vb


class _SparseSolver:
    """Titsias (2009) sparse GP / SGPR predictive with fixed hyperparameters.
    Inducing points: centres of the `spacing`-sized grid cells that contain data."""

    def __init__(self, X, yn, c, ls, noise, spacing, nu=np.inf):
        self.c, self.ls, self.s2, self.nu = c, ls, noise + _ALPHA, nu
        Z = np.unique(np.round(X / spacing), axis=0) * spacing
        self.Z = Z
        Kuu = _rbf(Z, Z, c, ls, nu)
        self.Lu = _chol(Kuu, 1e-8 * c)
        A = solve_triangular(self.Lu, _rbf(Z, X, c, ls, nu), lower=True, check_finite=False) / np.sqrt(self.s2)
        B = A @ A.T
        B[np.diag_indices_from(B)] += 1.0
        self.LB = _chol(B)
        self.cvec = solve_triangular(self.LB, A @ yn, lower=True, check_finite=False) / np.sqrt(self.s2)

    def predict(self, S, chunk=20000):
        mu, var = np.empty(len(S)), np.empty(len(S))
        for i in range(0, len(S), chunk):
            v1 = solve_triangular(self.Lu, _rbf(self.Z, S[i:i + chunk], self.c, self.ls, self.nu), lower=True, check_finite=False)
            v2 = solve_triangular(self.LB, v1, lower=True, check_finite=False)
            mu[i:i + chunk] = v2.T @ self.cvec
            var[i:i + chunk] = (self.c + self.s2 - np.einsum("ij,ij->j", v1, v1)
                                + np.einsum("ij,ij->j", v2, v2))
        return mu, np.sqrt(np.clip(var, 1e-12, None))

    def cov(self, A, B):
        """Posterior covariance of the latent surface between A and B (SGPR)."""
        def v(P):
            v1 = solve_triangular(self.Lu, _rbf(self.Z, P, self.c, self.ls, self.nu), lower=True, check_finite=False)
            return v1, solve_triangular(self.LB, v1, lower=True, check_finite=False)
        a1, a2 = v(A)
        b1, b2 = (a1, a2) if B is A else v(B)
        return _rbf(A, B, self.c, self.ls, self.nu) - a1.T @ b1 + a2.T @ b2


class FrozenExactGP(_FrozenHyper):
    name = "frozen"

    def fit(self, X, y, center=None):
        t = time.perf_counter()
        X, y = np.asarray(X, float), np.asarray(y, float)
        opt = self._learn(X, y)
        self._keep_all(X, y)
        yn = self._set_data(X, y)
        self.solver = _ExactSolver(X, yn, self.c, self.ls, self.noise, self.nu)
        self.n_train = len(X)
        self.fit_log.append((len(X), time.perf_counter() - t, opt))
        return self

    def predict(self, Xs, return_std=True):
        mu, sd = self.solver.predict(np.asarray(Xs, float))
        mu, sd = mu * self.ys + self.ym, sd * self.ys
        return (mu, sd) if return_std else mu


class SparseGP(_FrozenHyper):
    name = "sparse"

    def fit(self, X, y, center=None):
        t = time.perf_counter()
        X = np.asarray(X, float)
        opt = self._learn(X, y)
        yn = self._set_data(X, y)
        self.solver = _SparseSolver(X, yn, self.c, self.ls, self.noise, self._spacing(), self.nu)
        self.n_train = len(X)
        self.fit_log.append((len(X), time.perf_counter() - t, opt))
        return self

    def predict(self, Xs, return_std=True):
        mu, sd = self.solver.predict(np.asarray(Xs, float))
        mu, sd = mu * self.ys + self.ym, sd * self.ys
        return (mu, sd) if return_std else mu

    def predict_global(self, Xs):
        return self.predict(Xs)


class LocalGP(_FrozenHyper):
    """Exact GP on the local_n_points nearest to `center` (the tip) for the planner's
    decisions; whole-area maps (predict_global: live view, stop check, saved
    reconstruction) come from a sparse model on all the data, built only when asked."""
    name = "local"

    def fit(self, X, y, center=None):
        t = time.perf_counter()
        X, y = np.asarray(X, float), np.asarray(y, float)
        opt = self._learn(X, y)
        self._keep_all(X, y)
        M = self.cfg.local_n_points
        if center is not None and len(X) > M:
            idx = np.argpartition(((X - np.asarray(center, float)) ** 2).sum(1), M)[:M]
            Xl, yl = X[idx], y[idx]
        else:
            Xl, yl = X, y
        yn = self._set_data(Xl, yl)
        self.solver = _ExactSolver(Xl, yn, self.c, self.ls, self.noise, self.nu)
        self._local_norm = (self.ym, self.ys)
        self.n_train = len(X)
        self.fit_log.append((len(X), time.perf_counter() - t, opt))
        return self

    def predict(self, Xs, return_std=True):
        ym, ys = self._local_norm
        mu, sd = self.solver.predict(np.asarray(Xs, float))
        mu, sd = mu * ys + ym, sd * ys
        return (mu, sd) if return_std else mu


def robust_plane(X, y):
    """Substrate plane z = a + b x + c y: least squares, then refit on the points within 3
    robust sigma (features left out)."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    med = np.median(r)
    keep = np.abs(r - med) <= 3 * (1.4826 * np.median(np.abs(r - med)) + 1e-12)
    if keep.sum() > 3:
        coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
    return coef


class Detrended:
    """GP on the residuals from a robust substrate plane (GPPathConfig.gp_detrend): away
    from data the prediction reverts to the tilted substrate, not to a flat mean -- on a
    tilted sample (0.033 V/V on the instrument) that is the difference between a usable and
    a useless map of the unvisited ground."""

    def __init__(self, model):
        self.model, self.coef = model, np.zeros(3)

    def __getattr__(self, k):            # kernel_, fit_log, frozen, ... from the wrapped model
        return getattr(self.model, k)

    def _plane(self, X):
        X = np.asarray(X, float)
        return self.coef[0] + X @ self.coef[1:]

    def fit(self, X, y, center=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.coef = robust_plane(X, y)
        self.model.fit(X, y - self._plane(X), center=center)
        return self

    def predict(self, Xs, return_std=True):
        out = self.model.predict(Xs, return_std=return_std)
        if return_std:
            return out[0] + self._plane(Xs), out[1]
        return out + self._plane(Xs)

    def predict_global(self, Xs):
        mu, sd = self.model.predict_global(Xs)
        return mu + self._plane(Xs), sd

    def prior_std(self):
        return self.model.prior_std()

    def describe(self):
        return self.model.describe() + f" + plane {np.round(self.coef, 4).tolist()}"


def make_gp(cfg, length_scale: float, bounds):
    """GP back end for a GPPathConfig; length_scale/bounds = the kernel's start and fit range
    (cfg.learn_length_scale = False keeps the length scale fixed at length_scale)."""
    if not getattr(cfg, "learn_length_scale", True):
        bounds = "fixed"
    kind = cfg.gp_backend
    if kind not in BACKENDS:
        raise ValueError(f"GPPathConfig.gp_backend must be one of {BACKENDS}, not {kind!r}")
    m = {"sklearn_exact": SklearnExactGP, "frozen": FrozenExactGP,
         "local": LocalGP, "sparse": SparseGP}[kind](cfg, length_scale, bounds)
    return Detrended(m) if getattr(cfg, "gp_detrend", False) else m
