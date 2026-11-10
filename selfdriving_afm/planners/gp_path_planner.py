"""
GP (Bayesian-optimization) path planner -- same algorithm as your Phase-2 notebook
cell, refactored so `measure_segment` calls a backend instead of the synthetic
`get_topography`. Kept as a real alternative planner (selectable alongside
EdgeFollowPlanner) rather than folded into it, since the two have different
strengths: this one gives a full GP surface + uncertainty map, the edge-follow
planner is cheaper per-step and tuned specifically for tracing sharp boundaries.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import GPPathConfig
from ..gp_models import make_gp
from .. import trajectory
from ..execution import ContinuousExecutor


@dataclass
class PathSegment:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


class GPPathPlanner:
    def __init__(self, executor: ContinuousExecutor, cfg: GPPathConfig):
        self.executor = executor
        self.cfg = cfg
        # length scale ~ the feature size: from the priors (priors.py) if given
        ls = cfg.length_scale_V if cfg.length_scale_V is not None else cfg.path_segment_length_V * 3
        ls_bounds = (cfg.length_scale_bounds_V if cfg.length_scale_bounds_V is not None
                     else (cfg.gradient_eval_resolution_V, cfg.path_segment_length_V * 50))
        if cfg.length_scale_V is not None:
            # floor: length_scale_min_frac x the smallest feature (length_scale_V, from the
            # priors) -- enforced here too, so a preset / override takes effect without
            # re-running the priors
            ls_bounds = (cfg.length_scale_min_frac * cfg.length_scale_V, float(ls_bounds[1]))
        ls = float(np.clip(ls, *ls_bounds))
        # kernel C * RBF + White with normalize_y in every back end (AI0 is not zero-mean,
        # about -5.7 V on the instrument; without it the GP's zero prior makes the boundary
        # of the measured data look like a huge gradient). cfg.gp_backend picks how much
        # work a refit does -- see gp_models.py; "sklearn_exact" is the original behaviour.
        self.gp = make_gp(cfg, ls, ls_bounds)
        self.refit_every = max(1, cfg.refit_every)
        self._fitted = False
        # per-segment compute: (segment index, n training points, fit s, acquisition s)
        self.timing: List[tuple] = []
        self._slow_warned = False
        self._edge_win = cfg.edge_window_V if cfg.edge_window_V is not None else 0.25 * ls
        self._edge_h = 1.0
        self._plane = np.zeros(3)
        self._feat_sign = 1.0
        self.path_length_V = 0.0
        self._last_dir = None
        self._trace = None
        self._traced_pts, self._traced_tree = [], None
        self.traced_features: List[dict] = []
        self._trace_cap = (cfg.trace_max_length_V if cfg.trace_max_length_V is not None
                           else 1.5 * np.pi * ls_bounds[1] / 2)   # ls upper bound = 2 x largest feature
        self._trace_min_loop = 1.5 * np.pi * ls   # ls = smallest feature (priors)
        self._ls0 = ls
        self._trace_excl = 3 * cfg.trace_step_V
        self._thr, self._features_seen = np.inf, False
        self._explore_r = cfg.explore_radius_V if cfg.explore_radius_V is not None else 2 * ls
        self._tree = None

    # ------------------------------------------------------------------
    @staticmethod
    def _hermite(start, heading, direction, length, n):
        """Cubic Hermite curve from start (leaving along `heading`) to start + direction *
        length (arriving along `direction`) -- the same connector the edge explorer's weave
        uses: no corner at either end. n points, uniform in the curve parameter."""
        u = np.linspace(0.0, 1.0, n)[:, None]
        h00, h10, h01, h11 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u, -2*u**3 + 3*u**2, u**3 - u**2
        p0 = np.asarray(start, float)
        p1 = p0 + np.asarray(direction, float) * length
        return h00 * p0 + h10 * (length * np.asarray(heading, float)) + h01 * p1 + h11 * (length * np.asarray(direction, float))

    def _smooth_this(self, heading, direction) -> bool:
        return (self.cfg.smooth_turns and heading is not None
                and float(np.dot(heading, direction)) < np.cos(np.deg2rad(1.0)))

    def _segment_curve(self, start, direction, length, heading, ds):
        """The path the tip actually travels for one segment, resampled at constant spacing
        ds (constant tip speed): a straight line, or with smooth_turns the Hermite curve
        that starts tangent to the end of the previous segment."""
        if not self._smooth_this(heading, direction):
            return trajectory.line_segment(start, direction, length, ds)
        C = self._hermite(start, heading, direction, length, max(200, int(4 * length / ds)))
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
        m = np.arange(0.0, arc[-1], ds)
        return np.interp(m, arc, C[:, 0]), np.interp(m, arc, C[:, 1])

    def _turn_ok(self, start, heading, direction, length) -> bool:
        """smooth_turns / max_turn_deg: is this direction allowed from here?"""
        cfg = self.cfg
        if heading is None:
            return True
        if cfg.max_turn_deg is not None:
            if float(np.dot(heading, direction)) < np.cos(np.deg2rad(cfg.max_turn_deg)) - 1e-9:
                return False
        if self._smooth_this(heading, direction) and self._area is not None:
            C = self._hermite(start, heading, direction, length, 40)
            a = self._area
            if not ((C[:, 0] >= a[0]) & (C[:, 0] <= a[1]) & (C[:, 1] >= a[2]) & (C[:, 1] <= a[3])).all():
                return False
        return True

    def _measure_segment(self, start_xy: np.ndarray, direction: np.ndarray, length: float,
                         heading: Optional[np.ndarray] = None) -> PathSegment:
        """Measures the segment at full fidelity (VelocityConfig.sample_spacing_V --
        the physical hardware motion, fully preserved in executor.history/all_points()
        for saving/analysis), then returns it subsampled to gradient_eval_resolution_V
        for the GP to actually train on. GP regression is O(n^2) memory / O(n^3) fit
        cost, so training on every raw hardware sample -- which can be thousands per
        short segment at a slow tip_velocity_V_s -- doesn't just get slow, it can
        exhaust memory outright. gradient_eval_resolution_V is the GP's own working
        resolution, deliberately decoupled from physical sample spacing (config.py)."""
        x_vec, y_vec = self._segment_curve(start_xy, direction, length, heading, self.executor.velocity.sample_spacing_V)
        self._last_arc_V = float(np.sum(np.hypot(np.diff(x_vec), np.diff(y_vec)))) if len(x_vec) > 1 else 0.0
        self._n_segments = getattr(self, "_n_segments", 0) + 1
        z_vec = self.executor.send(x_vec, y_vec, kind="line", meta={
            "label": f"GP path segment {self._n_segments} (length {length:.3f} V)"})
        # z may be short at the end (raw readback); the GP learns at the measured positions
        # when there are sensors
        x_vec, y_vec = self.executor.measured_xy(x_vec, y_vec, z_vec)
        stride = max(1, int(round(self.cfg.gradient_eval_resolution_V / self.executor.velocity.sample_spacing_V)))
        return PathSegment(x_vec[::stride], y_vec[::stride], z_vec[::stride])

    def _candidate_directions(self) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, self.cfg.n_candidate_directions, endpoint=False)
        return np.array([[np.cos(a), np.sin(a)] for a in angles])

    def _estimate_path_gradient(self, start_pos, direction, length, resolution):
        n_points = max(int(length / resolution), 2)
        pts = np.array([start_pos + i * resolution * direction for i in range(n_points)])
        mu, sigma = self.gp.predict(pts, return_std=True)
        gradients = np.abs(np.diff(mu))
        avg_gradient = np.mean(gradients) if len(gradients) else 0.0
        avg_uncertainty = np.mean(sigma)
        return avg_gradient, avg_uncertainty

    def _inside_area(self, p) -> bool:
        a = getattr(self, "_area", None)
        return a is None or (a[0] <= p[0] <= a[1] and a[2] <= p[1] <= a[3])

    def _update_edge_scale(self, X, Z):
        """edge_uncertainty: the z change that counts as a full edge. Detrended (plane fit)
        z range of the data so far / 2, floored at edge_noise_k x the median z change over
        edge_window_V along the path (substrate roughness + noise), so roughness never
        reads as an edge before the first real one is found."""
        cfg = self.cfg
        if cfg.edge_height is not None:
            self._edge_h = float(cfg.edge_height)
            return
        A = np.column_stack([np.ones(len(X)), X])
        # substrate plane: least squares, then refit on the points near it (features out)
        coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
        r = Z - A @ coef
        keep = np.abs(r - np.median(r)) <= 3 * (np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-12)
        if keep.sum() > 3:
            coef, *_ = np.linalg.lstsq(A[keep], Z[keep], rcond=None)
            r = Z - A @ coef
        self._plane = coef
        k = max(1, int(round(self._edge_win / cfg.gradient_eval_resolution_V)))
        floor = cfg.edge_noise_k * float(np.median(np.abs(Z[k:] - Z[:-k]))) if len(Z) > k else 0.0
        base = float(np.median(r))
        # feature height above (or below) the substrate: the most extreme 0.1% of residuals
        up, dn = np.percentile(r, 99.9) - base, base - np.percentile(r, 0.1)
        self._feat_sign = 1.0 if up >= dn else -1.0
        self._edge_h = max(max(up, dn), floor, 1e-12)
        # on/off-feature threshold for trace mode: half the feature height, but never below
        # edge_noise_k x the substrate's robust scatter (roughness + noise)
        mad = 1.4826 * float(np.median(np.abs(r - base)))
        self._sub_sigma = mad
        self._thr = max(0.5 * self._edge_h, cfg.edge_noise_k * mad)
        self._features_seen = max(up, dn) > 2 * cfg.edge_noise_k * mad

    # ---------------- trace mode (trace_edges) ----------------
    def _on_feature(self, seg) -> np.ndarray:
        A = np.column_stack([np.ones(len(seg.x)), seg.x, seg.y])
        return self._feat_sign * (seg.z - A @ self._plane) > self._thr

    def _traced_near(self, p, r) -> bool:
        return self._traced_tree is not None and self._traced_tree.query(p)[0] < r

    def _maybe_start_trace(self, seg):
        """After an exploration segment: did it cross an outline of a feature not traced yet?"""
        if not (self._fitted and getattr(self, "_features_seen", False)) or len(seg.z) < 3:
            return
        on = self._on_feature(seg)
        for i in np.nonzero(np.diff(on.astype(int)))[0]:
            p = np.array([seg.x[i], seg.y[i]])
            if not self._traced_near(p, self._trace_excl):
                self._trace = dict(start=p, length=0.0, pts=[])
                return

    def _end_trace(self):
        if self._trace is not None and self._trace["pts"]:
            from scipy.spatial import cKDTree
            self._traced_pts.append(np.vstack(self._trace["pts"]))
            self._traced_tree = cKDTree(np.vstack(self._traced_pts))
            self.traced_features.append(dict(start=self._trace["start"], length_V=self._trace["length"],
                                             closed=self._trace.get("closed", False)))
        self._trace = None

    def _trace_direction(self, pos, last_seg):
        """Next heading while tracing: turn toward the feature when off it, away when on it
        (feature kept on the left); None = tracing is over."""
        cfg, tr = self.cfg, self._trace
        if tr["length"] >= self._trace_cap:
            return self._end_trace()
        if tr["length"] > self._trace_min_loop and np.hypot(*(pos - tr["start"])) < 1.5 * cfg.trace_step_V:
            tr["closed"] = True
            return self._end_trace()
        on = self._on_feature(last_seg)
        tail = on[-max(1, len(on) // 5):]
        side = -1.0 if tail.mean() > 0.5 else 1.0       # on the feature: steer out; off: steer in
        rot = lambda v, a: np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])
        # GP-guided: the outline runs perpendicular to the GP mean's gradient at the tip
        # (gradient x feature sign points into the feature; tangent with the feature on the left)
        e = 0.25 * cfg.trace_step_V
        P = pos + np.array([[e, 0], [-e, 0], [0, e], [0, -e]])
        m = self.gp.predict(P, return_std=False)
        g = self._feat_sign * np.array([m[0] - m[1], m[2] - m[3]]) / (2 * e)
        gn = float(np.hypot(*g))
        if gn * self._ls0 > 0.2 * self._edge_h:                            # a usable edge signal
            t = rot(g / gn, -np.pi / 2)
            d = rot(t, side * np.deg2rad(cfg.trace_correction_deg))
            if self._last_dir is not None and d @ self._last_dir < -0.5:  # never double back
                d = rot(self._last_dir, side * np.deg2rad(cfg.trace_turn_deg))
        else:                                                              # fall back: bang-bang
            h = self._last_dir if self._last_dir is not None else np.array([1.0, 0.0])
            d = rot(h, side * np.deg2rad(cfg.trace_turn_deg))
        if not self._inside_area(pos + d * cfg.trace_step_V):
            return self._end_trace()          # the outline runs out of the area
        return d

    def _near_feature(self, seg) -> bool:
        """Adaptive segments: did the last segment touch a feature? (any measured height
        more than half the feature height away from the substrate plane)"""
        if len(seg.z) == 0 or self._edge_h <= 0 or not self._fitted:
            return False
        A = np.column_stack([np.ones(len(seg.x)), seg.x, seg.y])
        r = seg.z - A @ self._plane
        return bool(np.any(np.abs(r) > 0.5 * self._edge_h))

    def _acquisition_edge_uncertainty(self, current_pos, dirs, length):
        cfg = self.cfg
        res = cfg.gradient_eval_resolution_V
        n = max(int(length / res), 2)
        s = np.arange(n) * res
        P = np.concatenate([current_pos + s[:, None] * d for d in dirs])
        mu, sd = self.gp.predict(P, return_std=True)
        mu, sd = mu.reshape(len(dirs), n), sd.reshape(len(dirs), n)
        sdn = np.clip(sd / max(self.gp.prior_std(), 1e-12), 0, 1)
        if cfg.acquisition == "straddle":
            # level-set estimation (Bryan et al. 2005): kappa*sigma - |mu - t|, t = substrate
            # plane + half the feature height; normalised by the feature height
            t = np.column_stack([np.ones(len(P)), P]) @ self._plane + self._feat_sign * 0.5 * self._edge_h
            st = (cfg.straddle_kappa * sd.ravel() - np.abs(mu.ravel() - t)) / self._edge_h
            edge_term = np.clip(st.reshape(len(dirs), n), -1, None).mean(1)
        else:
            k = min(n - 1, max(1, int(round(self._edge_win / res))))
            edge = np.clip(np.abs(mu[:, k:] - mu[:, :-k]) / self._edge_h, 0, 1)      # (D, n-k)
            sd_mid = 0.5 * (sdn[:, k:] + sdn[:, :-k])
            edge_term = (edge * sd_mid).mean(1)
        explore = sdn.mean(1)
        look = cfg.lookahead_V if cfg.lookahead_V is not None else length
        if cfg.explore_measure == "distance":
            # coverage: distance to the nearest measured point, saturating at explore_radius_V
            # -- decouples how wide a pass "clears" the ground from the GP length scale
            r2 = max(res, look / 60)
            s2 = np.arange(0, look, r2)
            P2 = np.concatenate([current_pos + s2[:, None] * d for d in dirs])
            dist, _ = self._tree.query(P2)
            cov = np.clip(dist / self._explore_r, 0, 1)
            if self._area is not None:
                a = self._area
                cov = cov * ((P2[:, 0] >= a[0]) & (P2[:, 0] <= a[1]) & (P2[:, 1] >= a[2]) & (P2[:, 1] <= a[3]))
            explore = cov.reshape(len(dirs), len(s2)).mean(1)
        elif look > length:
            # exploration looks further than the committed segment: sigma along the line out
            # to lookahead_V; points outside the area count as known (0), so the border of
            # the area stops looking attractive
            r2 = max(res, look / 60)
            s2 = np.arange(0, look, r2)
            P2 = np.concatenate([current_pos + s2[:, None] * d for d in dirs])
            _, sd2 = self.gp.predict(P2, return_std=True)
            sd2 = np.clip(sd2 / max(self.gp.prior_std(), 1e-12), 0, 1)
            if self._area is not None:
                a = self._area
                inside = (P2[:, 0] >= a[0]) & (P2[:, 0] <= a[1]) & (P2[:, 1] >= a[2]) & (P2[:, 1] <= a[3])
                sd2 = sd2 * inside
            explore = sd2.reshape(len(dirs), len(s2)).mean(1)
        values = cfg.edge_weight * edge_term + cfg.exploration_weight * explore
        if cfg.straight_bonus and self._last_dir is not None:
            values = values + cfg.straight_bonus * (dirs @ self._last_dir)
        ends = current_pos + dirs * length
        for i, e in enumerate(ends):
            if not self._inside_area(e):
                values[i] = -np.inf
        return values

    def _acquisition_ivr(self, pos, dirs, length):
        """Pure GP: weighted integrated variance reduction (IVR). For each candidate segment,
        how much would measuring along it shrink the GP's posterior variance over the map
        around the tip -- computed exactly before measuring (the variance depends only on
        where the data are). Each map point is weighted by
            w = (1 - ivr_lambda) + ivr_lambda * interest,
        interest = |grad mu| (detrended) / (feature height / length scale), clipped to 0..1:
        ivr_lambda = 0 spreads the measurements to minimize the total uncertainty
        (exploration); ivr_lambda -> 1 spends them where the GP predicts edges
        (exploitation)."""
        cfg, ls = self.cfg, self._ls0
        # receding horizon: score each direction as if the tip kept going for lookahead_V
        # (the whole planned line), but only the first segment is executed, then re-plan
        horizon = max(length, cfg.lookahead_V or length)
        k = max(2, int(round(cfg.ivr_points_per_segment * horizon / length)))
        s = (np.arange(k) + 0.5) * horizon / k
        C = np.stack([pos + s[:, None] * d for d in dirs])                 # (D, k, 2)
        h = getattr(self, "_heading", None)
        if cfg.smooth_turns and h is not None:
            # the first segment is the Hermite curve the tip will really travel
            k1 = int(np.sum(s <= length))
            for i, d in enumerate(dirs):
                if self._smooth_this(h, d) and k1 > 0:
                    Hc = self._hermite(pos, h, d, length, 200)
                    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Hc, axis=0).T))])
                    t = s[:k1] / length * arc[-1]
                    C[i, :k1] = np.column_stack([np.interp(t, arc, Hc[:, 0]), np.interp(t, arc, Hc[:, 1])])
        noise_div = (horizon / k) / cfg.gradient_eval_resolution_V       # samples per point
        gs = cfg.ivr_grid_spacing_V or ls / 2
        R = cfg.ivr_radius_V or (horizon + 3 * ls)
        xs = np.arange(pos[0] - R, pos[0] + R + 1e-9, gs)
        ys = np.arange(pos[1] - R, pos[1] + R + 1e-9, gs)
        GX, GY = np.meshgrid(xs, ys)
        G = np.column_stack([GX.ravel(), GY.ravel()])
        keep = np.hypot(G[:, 0] - pos[0], G[:, 1] - pos[1]) <= R
        if self._area is not None:
            a = self._area
            keep &= (G[:, 0] >= a[0]) & (G[:, 0] <= a[1]) & (G[:, 1] >= a[2]) & (G[:, 1] <= a[3])
        G = G[keep]
        red = self.gp.variance_reduction(G, C, noise_div)                  # (D, g)
        lam = float(np.clip(cfg.ivr_lambda, 0.0, 1.0))
        w = np.ones(len(G))
        if lam > 0:
            e = 0.5 * gs
            m = self.gp.predict(np.concatenate([G + [e, 0], G - [e, 0], G + [0, e], G - [0, e]]),
                                return_std=False).reshape(4, -1)
            gx = (m[0] - m[1]) / (2 * e) - self._plane[1]
            gy = (m[2] - m[3]) / (2 * e) - self._plane[2]
            interest = np.clip(np.hypot(gx, gy) / max(self._edge_h / ls, 1e-12), 0, 1)
            gam = float(getattr(cfg, "ivr_interest_power", 1.0) or 1.0)
            w = (1 - lam) + lam * interest ** gam
        values = red @ w
        for i, end in enumerate(pos + dirs * length):
            if not self._inside_area(end):
                values[i] = -np.inf
        return values

    def _acquisition(self, current_pos, candidate_directions, length, exploration_weight):
        if self.cfg.acquisition == "ivr":
            return self._acquisition_ivr(current_pos, candidate_directions, length)
        if self.cfg.acquisition in ("edge_uncertainty", "straddle"):
            return self._acquisition_edge_uncertainty(current_pos, candidate_directions, length)
        values = []
        for d in candidate_directions:
            if not self._inside_area(current_pos + d * length):
                values.append(-np.inf)  # would leave the area to explore (the priors)
                continue
            grad, unc = self._estimate_path_gradient(current_pos, d, length, self.cfg.gradient_eval_resolution_V)
            values.append(grad + exploration_weight * unc)
        return np.array(values)

    def _mean_uncertainty_over_check_grid(self, start_xy: np.ndarray) -> float:
        cfg = self.cfg
        g = np.linspace(-cfg.stop_check_extent_V, cfg.stop_check_extent_V, cfg.stop_check_grid_n)
        Xg, Yg = np.meshgrid(g + start_xy[0], g + start_xy[1])
        _, sigma = self.predict_surface(Xg, Yg)
        return float(np.mean(sigma))

    # ------------------------------------------------------------------
    def run(self, start_xy=(0.0, 0.0), initial_direction=(1.0, 0.0)) -> List[PathSegment]:
        """Runs until whichever of these fires first (see README "When does a scan
        stop?"): `max_segments` (hard cap, always enforced), `max_run_time_s`
        (wall-clock safety net, if set), or the GP's mean uncertainty over
        `stop_check_extent_V` dropping at/below `target_mean_uncertainty` (if set --
        the "the reconstruction is good enough" criterion)."""
        cfg = self.cfg
        if cfg.target_mean_uncertainty is not None and cfg.stop_check_extent_V is None:
            raise ValueError("GPPathConfig.stop_check_extent_V must be set when "
                              "target_mean_uncertainty is used")

        self.executor.reset(start_xy)
        self._n_segments = 0
        self._area = None
        current_pos = np.array(start_xy, dtype=float)
        if cfg.area_half_width_V is not None:
            # stay inside the area; the first segment starts at its center
            c = np.array(cfg.area_center_V if cfg.area_center_V is not None else start_xy, dtype=float)
            h = cfg.area_half_width_V
            self._area = (c[0] - h, c[0] + h, c[1] - h, c[1] + h)
            current_pos = c
        check_center = current_pos.copy()  # the uncertainty check is centered on where the path starts
        t0 = time.monotonic()

        segments: List[PathSegment] = []
        seg0 = self._measure_segment(current_pos, np.array(initial_direction), cfg.initial_path_length_V)
        self.path_length_V = cfg.initial_path_length_V
        segments.append(seg0)
        current_pos = np.array(self.executor.last_pos, dtype=float)   # where the path goes on from

        X_sampled = np.column_stack([seg0.x, seg0.y])
        Z_sampled = seg0.z.copy()

        candidate_directions = self._candidate_directions()

        for seg_i in range(cfg.max_segments):
            if cfg.max_run_time_s is not None and time.monotonic() - t0 > cfg.max_run_time_s:
                break
            if (cfg.max_path_length_V is not None and
                    cfg.max_path_length_V - self.path_length_V < 0.1 * cfg.path_segment_length_V):
                break       # budget used up (less than a tenth of a segment left)

            t_fit0 = time.perf_counter()
            fit_s = 0.0
            if seg_i % self.refit_every == 0:
                self.gp.fit(X_sampled, Z_sampled, center=current_pos)
                self._fitted = True
                if cfg.acquisition in ("edge_uncertainty", "straddle", "ivr"):
                    self._update_edge_scale(X_sampled, Z_sampled)
                    if cfg.explore_measure == "distance":
                        from scipy.spatial import cKDTree
                        self._tree = cKDTree(X_sampled)
                fit_s = time.perf_counter() - t_fit0
                if fit_s > 5.0 and cfg.gp_backend == "sklearn_exact" and not self._slow_warned:
                    print(f"[gp_path] GP refit took {fit_s:.1f}s at {len(X_sampled)} points and grows "
                          f"~n^3 -- set cfg.gp_path.gp_backend = 'sparse' (or 'local') to keep "
                          f"each refit well under a second")
                    self._slow_warned = True

                if (cfg.target_mean_uncertainty is not None
                        and seg_i >= cfg.min_segments_before_stop_check
                        and self._mean_uncertainty_over_check_grid(check_center) <= cfg.target_mean_uncertainty):
                    break

            t_acq0 = time.perf_counter()
            self._heading = self._last_dir if self._last_dir is not None else np.asarray(initial_direction, float)
            tracing = False
            if self._trace is not None:
                d = self._trace_direction(current_pos, segments[-1])
                if d is not None:
                    best_dir, seg_len, tracing = d, cfg.trace_step_V, True
            if not tracing:
                look_len = cfg.path_segment_length_V
                if cfg.min_segment_length_V is not None and self._near_feature(segments[-1]):
                    look_len = cfg.min_segment_length_V
                values = self._acquisition(current_pos, candidate_directions, look_len, cfg.exploration_weight)
                if cfg.smooth_turns or cfg.max_turn_deg is not None:
                    for i, d in enumerate(candidate_directions):
                        if np.isfinite(values[i]) and not self._turn_ok(current_pos, self._heading, d, look_len):
                            values[i] = -np.inf
                if not np.isfinite(values).any():
                    break  # every direction leaves the area
                best_dir = candidate_directions[int(np.argmax(values))]
                seg_len = look_len
            self.timing.append((seg_i, len(X_sampled), fit_s, time.perf_counter() - t_acq0))
            self._last_dir = best_dir
            if cfg.max_path_length_V is not None:
                seg_len = min(seg_len, cfg.max_path_length_V - self.path_length_V)
            heading = self._heading
            if tracing and not self._turn_ok(current_pos, heading, best_dir, seg_len):
                heading = None                      # tracing: fall back to a straight step
            seg = self._measure_segment(current_pos, best_dir, seg_len, heading=heading if cfg.smooth_turns else None)
            # actual tip travel (a curve is longer than its chord; a sampled line can come out a
            # sample short of seg_len, which must not stall the budget)
            self.path_length_V += max(self._last_arc_V, seg_len)
            if len(seg.x) == 0:
                break

            segments.append(seg)
            if cfg.trace_edges:
                if tracing:
                    self._trace["length"] += seg_len
                    self._trace["pts"].append(np.column_stack([seg.x, seg.y]))
                else:
                    self._maybe_start_trace(seg)
            X_sampled = np.vstack([X_sampled, np.column_stack([seg.x, seg.y])])
            Z_sampled = np.concatenate([Z_sampled, seg.z])
            current_pos = np.array(self.executor.last_pos, dtype=float)

        self.gp.fit(X_sampled, Z_sampled, center=current_pos)  # final fit: predict_surface() sees all data
        self._fitted = True
        return segments

    def predict_surface(self, x_grid: np.ndarray, y_grid: np.ndarray):
        """Reconstruct the full surface (mean + std) over a grid, for plotting."""
        pts = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        mu, sigma = self.gp.predict_global(pts)   # "local": a whole-area sparse model
        return mu.reshape(x_grid.shape), sigma.reshape(x_grid.shape)

    def timing_summary(self) -> str:
        """Per-segment GP compute (fit + acquisition): the tip's wait between segments."""
        if not self.timing:
            return f"GP back end {self.cfg.gp_backend!r}: no segments yet"
        t = np.array([f + a for _, _, f, a in self.timing])
        return (f"GP back end {self.cfg.gp_backend!r}: per-segment compute median {np.median(t):.3f}s, "
                f"worst {t.max():.2f}s, total {t.sum():.1f}s over {len(t)} segments "
                f"(final n = {self.timing[-1][1]})")

    @property
    def fitted(self) -> bool:
        """True once the GP has been fit at least once -- predict_surface() is only
        meaningful after this (used by viz.LiveGPPathView to skip the first redraw)."""
        return self._fitted
