"""
GP between features, edge tracker on them (2026-11-09).

EdgeExplorerPlanner with ONE change: where to explore next. Between features the edge
explorer picks the direction that leads into the least-measured part of its coverage map;
here a Gaussian process of everything measured so far picks it instead -- the GP path
planner's integrated variance reduction (IVR) acquisition, weighted by ivr_lambda between
total uncertainty (0, pure exploration) and predicted edges (1). As soon as a segment crosses
an outline, the edge explorer takes over exactly as in contour following (L2): it follows the
edge (zigzag / weave) until the loop closes or the edge is lost, scans the inside if
configured, then hands the choice back to the GP. Stopping, connectors, substrate model and
detection are the edge explorer's own.

    planner = GPEdgeExplorerPlanner(executor, cfg.edge_explorer, cfg.gp_path)

cfg.gp_path supplies the GP (gp_backend, gp_kernel, gp_detrend, length scale from the
priors) and the acquisition settings (ivr_lambda, lookahead_V, n_candidate_directions).
Each exploring step is cfg.edge_explorer.segment_length_V long (straight), as in L2.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from ..config import EdgeExplorerConfig, GPPathConfig
from ..execution import ContinuousExecutor
from .edge_explorer_planner import EdgeExplorerPlanner, _Crossing, _unit, HEIGHT_STRETCH_V, DIR_SPAN_V
from .. import edge_detect
from .gp_path_planner import GPPathPlanner


class GPEdgeExplorerPlanner(EdgeExplorerPlanner):
    MIN_POINTS = 50          # GP needs this many training points; before that: the coverage map
    HANDOVER_UNEXPLORED_FRAC = 0.03   # less than this much of the area unexplored (coverage map):
                                      # the map finishes (jumps to what's left, then stops)
    RECHECK_EVERY = 8        # every this many exploring choices: re-check all earlier exploring
                             # data with the CURRENT substrate model for outline crossings that were
                             # never traced (seen 2026-11-09: a hexagon crossed by the initial path,
                             # before the threshold was calibrated, was never followed -- and its
                             # spot counted as measured, so nothing went back)
    TRAIN_SPACING_V = 0.02   # GP training points: one per this much path (follow / interior
                             # scans are dense -- at the GP path's 0.005 V the GP reached 181k
                             # points and ~5 s per choice in simulation)

    def __init__(self, executor: ContinuousExecutor, cfg: EdgeExplorerConfig, gp_cfg: GPPathConfig):
        super().__init__(executor, cfg)
        self.gp_cfg = gp_cfg
        self.gp_helper = GPPathPlanner(executor, gp_cfg)    # its GP + IVR acquisition (never run)
        self.timing: List[tuple] = []                       # (step, n, fit_s, acquisition_s)
        self.choices: List[str] = []                        # "gp" | "map" per exploring choice

    def run(self, start_xy=(0.0, 0.0)):
        self._gpX, self._gpZ = [], []
        self.timing, self.choices = [], []
        return super().run(start_xy)

    # every measured path (explore, follow, interior) trains the GP
    def _do_path(self, x_vec, y_vec, mode, label):
        crossings = super()._do_path(x_vec, y_vec, mode, label)
        sg = self.segments[-1]
        stride = max(1, int(round(self.TRAIN_SPACING_V / self.executor.velocity.sample_spacing_V)))
        n = len(sg.z)
        if n:
            self._gpX.append(np.column_stack([sg.x[:n:stride], sg.y[:n:stride]]))
            self._gpZ.append(np.asarray(sg.z[:n:stride], float))
        return crossings

    def _recheck(self, pos):
        """Untraced crossings in the exploring data so far -> the queue (nearest first). Returns
        the nearest new one, or None."""
        model = self.substrate
        if not model.fitted:
            return None
        ds = self.executor.velocity.sample_spacing_V
        min_run = max(1, int(round(self.cfg.edge_detect.min_run_V / ds)))
        found = []
        for sg in self.segments:
            if sg.mode not in ("initial", "explore", "calibration") or len(sg.z) < 3 * min_run:
                continue
            x, y, z = np.asarray(sg.x), np.asarray(sg.y), np.asarray(sg.z)
            n = min(len(x), len(z)); x, y, z = x[:n], y[:n], z[:n]
            on = edge_detect.clean_mask(model.classify(x, y, z), min_run)
            h = np.abs(model.height(x, y, z))
            span = max(2, int(round(DIR_SPAN_V / ds))); w = max(4 * min_run, int(round(HEIGHT_STRETCH_V / ds)))
            for i in edge_detect.transitions(on):
                p = np.array([x[i], y[i]])
                if self._known(p) or any(np.linalg.norm(q.p - p) < self._follow_limits()[0] for q in self._queue):
                    continue
                ent = bool(on[i])
                j0, j1 = max(i - span, 0), min(i + span, n - 1)
                d = _unit([x[j1] - x[j0], y[j1] - y[j0]]) if j1 > j0 else np.array([1.0, 0.0])
                side = h[i:i + w] if ent else h[max(i - w, 0):i]
                height = float(np.median(side)) if len(side) else np.inf
                if height < model.threshold:
                    continue
                found.append(_Crossing(p, d if ent else -d, height=height))
        if not found:
            return None
        self._queue.extend(found)
        self.rechecked = getattr(self, "rechecked", 0) + len(found)
        print(f"[gp_edge_explorer] {len(found)} untraced crossing(s) in earlier exploring data -> queued")
        return min(found, key=lambda c: np.linalg.norm(c.p - pos))

    def _choose(self, pos, heading):
        # every RECHECK_EVERY choices: untraced crossings found earlier -> queue; head toward the
        # nearest (the run loop follows it once the tip is there -- the executor drives the move)
        if len(self.choices) % self.RECHECK_EVERY == self.RECHECK_EVERY - 1:
            c = self._recheck(pos)
            if c is not None:
                self.choices.append("recheck")
                d = c.p - np.asarray(pos, float)
                L = float(np.linalg.norm(d))
                if L > 1e-9:
                    return d / L, float("inf")
        if not self._gpX or sum(len(z) for z in self._gpZ) < self.MIN_POINTS:
            self.choices.append("map")
            return super()._choose(pos, heading)
        # stop exactly as contour following does: once its coverage map has no unexplored area
        # left (nothing >= half the explore scale from a measurement), the map decides -- it
        # jumps to what's left or returns None (stop). The GP would always find some variance.
        if (self._unexplored() >= 0.5).mean() < self.HANDOVER_UNEXPLORED_FRAC:
            self.choices.append("map")
            return super()._choose(pos, heading)
        h = self.gp_helper
        X = np.vstack(self._gpX); Z = np.concatenate(self._gpZ)
        t0 = time.perf_counter()
        h._area = self._bounds
        h.gp.fit(X, Z, center=pos)
        h._fitted = True
        h._update_edge_scale(X, Z)
        t1 = time.perf_counter()
        h._heading = np.asarray(heading, float)
        h._last_dir = h._heading
        dirs = h._candidate_directions()
        L = self._L
        values = h._acquisition_ivr(np.asarray(pos, float), dirs, L)
        lim = self._scan_limit()
        x0, x1, y0, y1 = self._bounds
        for i, d in enumerate(dirs):
            end = pos + d * L
            if not (x0 <= end[0] <= x1 and y0 <= end[1] <= y1) or np.any(np.abs(end) > lim):
                values[i] = -np.inf
            elif self.cfg.max_turn_deg is not None and float(np.dot(heading, d)) < np.cos(np.deg2rad(self.cfg.max_turn_deg)):
                values[i] = -np.inf
        self.timing.append((len(self.timing), len(X), t1 - t0, time.perf_counter() - t1))
        if not np.isfinite(values).any():
            self.choices.append("map")
            return super()._choose(pos, heading)
        k = int(np.argmax(values))
        # nothing left the GP considers worth measuring near here: the coverage map decides
        # (it returns None when everything is explored -> jump / stop, as in L2)
        if values[k] <= 1e-6 * max(np.nanmax(np.where(np.isfinite(values), values, np.nan)), 1e-12) or values[k] <= 0:
            self.choices.append("map")
            return super()._choose(pos, heading)
        self.choices.append("gp")
        return dirs[k], float(values[k])

    MAX_RESUMES = 4          # a trace that ends without closing is restarted from its open end
                             # (fresh probe across the edge) up to this many times before the GP
                             # explores again -- finish the object first (2026-11-09)
    MIN_CROSSINGS_TO_RESUME = 20   # specks (the small particle) are not chased

    def _inside_area_xy(self, p) -> bool:
        x0, x1, y0, y1 = self._bounds
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def _holder(self, fid):
        """The feature that now holds feature fid (itself, or the one it was merged into)."""
        for g in self.features:
            if g.id == fid or fid in g.merged_ids:
                return g
        return None

    def _follow(self, seed, pos):
        n0 = len(self.features)
        out = super()._follow(seed, pos)
        if len(self.features) <= n0 and not self.features:
            return out
        fid = self.features[-1].id if len(self.features) > n0 else None
        if fid is None:
            return out
        for attempt in range(1, self.MAX_RESUMES + 1):
            f = self._holder(fid)
            if f is None or f.closed_loop or f.closed_by_border or f.hit_border:
                break
            if len(f.points) < self.MIN_CROSSINGS_TO_RESUME:
                break
            tip = np.asarray(out[0], float)
            legs = [np.asarray(l, float).reshape(-1, 2) for l in f.legs if len(l) >= 2]
            if not legs:
                break
            # the open end nearest the tip, and the edge direction there (continuing the leg)
            L = min(legs, key=lambda q: np.linalg.norm(q[-1] - tip))
            p_end = L[-1]
            t = _unit(L[-1] - L[max(0, len(L) - 6)])
            P = f.pts()
            k = int(np.argmin(np.linalg.norm(P - p_end, axis=1)))
            n_in = _unit(f.dirs[k]) if k < len(f.dirs) else None
            if n_in is None:
                break
            # step PAST the end before probing: at the end itself the probe only crosses the
            # trace's own outline, which counts as known (within the follow step of it) -- seen
            # on the instrument: every resume was two failed probes. Further / turned toward the
            # feature on later attempts (a convex corner turns the edge toward its inside).
            step, ang = [(0.4, 0.0), (0.6, 30.0), (0.8, 60.0), (0.5, -30.0)][(attempt - 1) % 4]
            a = np.deg2rad(ang)
            d = _unit(t * np.cos(a) + n_in * np.sin(a))
            p_new = p_end + step * d
            if not self._inside_area_xy(p_new):
                break
            print(f"[gp_edge_explorer] feature {f.id} not closed (lost) -- resuming {step:.1f} V past its open end "
                  f"({p_end[0]:+.2f}, {p_end[1]:+.2f}) V, turned {ang:+.0f} deg, attempt {attempt}/{self.MAX_RESUMES}")
            self.resumes = getattr(self, "resumes", 0) + 1
            self._prev_on = None                       # re-find the edge with a probe across it
            out = super()._follow(_Crossing(p_new, n_in, height=np.inf), tip)
            # done once a CLOSED outline passes this open end (the resumed trace closed it, merged
            # with this one or on its own)
            if any(g.closed_loop and g.distance(p_end) < 2 * self._follow_limits()[0] for g in self.features):
                print(f"[gp_edge_explorer] the object at ({p_end[0]:+.2f}, {p_end[1]:+.2f}) V is CLOSED after {attempt} resume(s)")
                break
        return out

    # live view / save: the GP map like gp_path
    @property
    def fitted(self) -> bool:
        return self.gp_helper._fitted

    def predict_surface(self, x_grid, y_grid):
        return self.gp_helper.predict_surface(x_grid, y_grid)

    def timing_summary(self) -> str:
        if not self.timing:
            return "GP: no exploring choices"
        t = np.array([f + a for _, _, f, a in self.timing])
        return (f"GP-chosen exploring steps: {self.choices.count('gp')} (coverage map: {self.choices.count('map')}); "
                f"GP compute median {np.median(t):.3f}s, worst {t.max():.2f}s, final n = {self.timing[-1][1]}")
