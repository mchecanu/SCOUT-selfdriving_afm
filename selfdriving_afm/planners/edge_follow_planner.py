"""
Edge-follow planner: spiral search for an edge, then a chain of circular scans
centered on the edge that walks along it, one feature per edge.

    SEARCH  spiral out from the current tip position until the trace changes state
            (off-feature <-> on-feature) somewhere that isn't an already-known edge.
            The FIRST search can instead be a raster or a Lissajous over the whole
            square area (first_scan_pattern) -- corners included, so the later
            "relocate" spirals have nothing left to cover.
            Nothing new found: spiral_search_mode="relocate" (default) moves the next
            spiral -- same size -- to the nearest area not scanned yet (inside
            explore_extent_V of the start); the run ends once everything there has been
            covered. "grow" instead makes the spiral bigger around the same center
            (spiral_growth_factor, up to max_search_radius_V).
            A spiral that crosses SEVERAL separate edges (follow_all_spiral_edges):
            each is traced in turn -- the next one nearest the tip, skipping any a
            trace has covered meanwhile -- before the next search spiral.
      -> TRACE  circles centered ON the edge. Each circle is split into on-feature
            arcs; the arc continuing the edge we're following gives the forward
            crossing, and the next circle is centered along the chord toward it
            (tangential to the edge, never more than one radius, so the new center
            stays on the edge). Continues until:
              - the loop closes (back onto an earlier circle of this pass, within
                ~half a step)                                       -> feature done
              - the edge disappears (no crossings on a circle)      -> if
                trace_both_directions, go back to where the edge was first found and
                follow it the OTHER way, still the same feature; then SEARCH again
              - a SEPARATE edge appears on a circle (another on-feature arc, farther
                than same_feature_distance_V from every known edge) -> a NEW feature:
                on_new_edge="switch" traces it right away and then comes back to
                finish this one; "queue" finishes this one first
              - the trace runs into another feature's edge          -> it was the
                same edge all along: merged into that feature (one edge = one feature)
            Circle size (circle.radius_mode): fixed, "adaptive" (shrinks after sharp
            turns), or "curvature" -- each circle's on-feature arc spans 2 acos(r / 2R)
            round an edge of curvature radius R, so it measures R and the next circle
            gets r = curvature_radius_fraction x R, between adaptive_min/max_radius_V
            (the feature-size priors, priors.py). The first circle on a feature a
            spiral found is sized from how far the spiral ran on it.

On/off-feature is decided from the height ABOVE THE SUBSTRATE (substrate.SubstrateModel):
a small calibration circle at the start (calibration_radius_V) gives the tilt and the
roughness; the plane is refined after every spiral and circle from its substrate
samples, and the threshold moves into the gap between substrate and feature heights
once features have been seen. Each feature also tracks a small local offset of the
substrate next to its edge (Feature.baseline, from each circle's off-feature samples)
-- not each circle's own median, which sits between the two levels when a circle is
half on the feature and would hide the edge.

Works against any backend.ScanBackend (simulation or real hardware) -- this file
has no hardware-specific code in it at all.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from matplotlib.path import Path as _Polygon

from ..config import EdgeFollowConfig
from .. import trajectory
from .. import edge_detect
from ..substrate import SubstrateModel
from ..execution import ContinuousExecutor


def _unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return np.array([1.0, 0.0])
    return vec / n


def _angle_of(vec: np.ndarray) -> float:
    return float(np.arctan2(vec[1], vec[0]))


def _angular_distance(a: float, b: float) -> float:
    """Wrapped difference between two angles, in [0, pi]."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(d)


@dataclass
class SpiralAttempt:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    edge_found: bool
    edge_index: Optional[int] = None


@dataclass
class CircleStep:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    center: np.ndarray
    radius: float
    crossing_xy: Optional[np.ndarray] = None  # forward crossing the next circle heads for
    leg: int = 0                               # 0 = first direction, 1 = the other way
    new_edge_xy: Optional[np.ndarray] = None   # a separate edge this circle discovered


@dataclass(eq=False)  # identity comparison -- features hold numpy arrays
class Feature:
    """One edge. Every circle that followed it -- across both directions, and across
    being interrupted to trace another edge and resumed -- lands in the same Feature."""
    spiral_attempts: List[SpiralAttempt] = field(default_factory=list)
    circle_steps: List[CircleStep] = field(default_factory=list)
    closed_loop: bool = False
    edge_lost: bool = False
    edge_points: List[np.ndarray] = field(default_factory=list)  # every crossing on this edge
    baseline: float = 0.0    # local substrate offset above the fitted plane, tracked while tracing
    threshold: float = 0.0   # |height - baseline| > threshold = on-feature (the model's, when last used)
    _other_way: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (first center, other crossing)
    id: int = -1             # stable feature number (1, 2, ...) -- used in the live-view labels
    hit_scan_limit: bool = False  # the edge ran off the +/-max_amplitude_V scan range

    def distance_to_edge(self, p: np.ndarray) -> float:
        """Distance from p to this feature's traced edge LINE: the chain of its
        circle centers (each one sits on the edge) plus each chain's last forward
        crossing. A point on the same edge lands within a hair of it; a point on a
        neighbouring feature's edge is a whole gap away -- unlike the distance to the
        nearest scattered crossing point, which can be ~half a step even for the same
        edge and so can't tell a close neighbour from the same edge."""
        segs = self._trace_segments()
        if segs is None:
            if not self.edge_points:
                return np.inf
            return float(np.min(np.linalg.norm(np.asarray(self.edge_points) - p, axis=1)))
        a, b = segs
        ab = b - a
        t = np.clip(np.einsum('ij,ij->i', np.asarray(p) - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-24), 0, 1)
        return float(np.min(np.linalg.norm(a + t[:, None] * ab - np.asarray(p), axis=1)))

    def _trace_segments(self):
        """(starts, ends) of the edge-line segments, or None if no circles yet.
        Consecutive circles of the same leg are joined; a jump of more than two
        radii (a different leg, or merged-in circles) starts a new chain."""
        if not self.circle_steps:
            return None
        starts, ends = [], []
        prev = None
        for cs in self.circle_steps:
            c = np.asarray(cs.center, dtype=float)
            if prev is not None and prev.leg == cs.leg and np.linalg.norm(c - prev.center) <= 2 * max(prev.radius, cs.radius):
                starts.append(prev.center); ends.append(c)
            if cs.crossing_xy is not None:
                starts.append(c); ends.append(np.asarray(cs.crossing_xy, dtype=float))
            prev = cs
        if not starts:
            return None
        return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)

    def contains(self, p: np.ndarray) -> bool:
        """Inside a closed traced loop (so a point there is this feature's interior,
        not a new edge)."""
        if not self.closed_loop or len(self.edge_points) < 3:
            return False
        return bool(_Polygon(np.asarray(self.edge_points)).contains_point(p))

    def exclusion_region(self, margin: float):
        """(centroid, radius) disk covering the traced circles plus a margin."""
        centers = np.array([cs.center for cs in self.circle_steps])
        centroid = centers.mean(axis=0)
        max_r = np.linalg.norm(centers - centroid, axis=1).max() if len(centers) > 1 else 0.0
        return centroid, max_r + margin


class _Coverage:
    """Which parts of the exploration region have been scanned already: every
    spiral's and tracing circle's disk, plus the inside of every closed feature.
    A boolean occupancy grid -- used to send the next search spiral somewhere new."""

    def __init__(self, bounds, n: int):
        x0, x1, y0, y1 = bounds
        self.bounds = bounds
        self.X, self.Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
        self.explored = np.zeros(self.X.shape, dtype=bool)

    def _disk(self, c, r):
        return (self.X - c[0]) ** 2 + (self.Y - c[1]) ** 2 <= r * r

    def add_disk(self, c, r) -> None:
        self.explored |= self._disk(c, r)

    def add_rect(self, x0, x1, y0, y1) -> None:
        self.explored |= (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1)

    def add_polygon(self, pts) -> None:
        if len(pts) >= 3:
            inside = _Polygon(np.asarray(pts)).contains_points(np.column_stack([self.X.ravel(), self.Y.ravel()]))
            self.explored |= inside.reshape(self.X.shape)

    def unexplored_fraction(self, c, r) -> float:
        disk = self._disk(c, r)
        if not disk.any():
            return 0.0
        return float((~self.explored[disk]).mean())


@dataclass
class _TraceTask:
    """Where to (re)start following an edge -- a stack of these is what lets a
    trace be interrupted by a new edge and resumed afterwards."""
    feature: Feature
    center: np.ndarray
    backward_angle: float
    radius: float
    leg: int = 0
    first: bool = False           # first circle of a feature: approached from the side
    prev_direction_angle: Optional[float] = None
    steps_in_leg: int = 0
    chain_start: Optional[np.ndarray] = None


class EdgeFollowPlanner:
    def __init__(self, executor: ContinuousExecutor, cfg: EdgeFollowConfig):
        self.executor = executor
        self.cfg = cfg

    # ------------------------------------------------------------------
    def run(self, start_xy=(0.0, 0.0), max_features: Optional[int] = None) -> List[Feature]:
        cfg = self.cfg
        self.executor.reset(start_xy)
        center0 = np.array(cfg.explore_center_V if cfg.explore_center_V is not None else start_xy, dtype=float)
        current_pos = center0.copy()   # the first spiral: at the area's center
        search_radius = cfg.first_search_radius_V or cfg.initial_search_radius_V
        self._target = max_features or cfg.max_features
        features: List[Feature] = []
        tasks: List[_TraceTask] = []  # stack: tasks[-1] runs next
        pending_attempts: List[SpiralAttempt] = []  # failed spirals, carried into the next feature
        spiral_seeds = []  # the last spiral's other new edges: (xy, chord, spiral center, baseline, threshold)
        self._n_features_made = 0
        self._n_spirals = 0
        self._relocated = False
        traced = False  # a trace just finished (-> "relocate" searches the nearest unexplored area)
        lim = self._scan_limit()
        e = cfg.explore_extent_V
        self._coverage = _Coverage((max(center0[0] - e, -lim), min(center0[0] + e, lim),
                                    max(center0[1] - e, -lim), min(center0[1] + e, lim)),
                                   cfg.explore_grid_n)

        # calibration: a small circle round the start -- the substrate tilt and roughness
        # before the first search decides anything
        self.substrate = SubstrateModel(cfg.edge_detect)
        r_cal = cfg.calibration_radius_V
        if r_cal:
            r_cal = min(r_cal, max(0.0, lim - float(np.max(np.abs(center0)))))
        if r_cal:
            xc, yc, _ = trajectory.circle(center0, r_cal, ds=self.executor.velocity.sample_spacing_V)
            zc = self.executor.send(xc, yc, kind="calibration", meta={
                "label": f"Calibration circle (radius {r_cal:.3g} V) -- substrate tilt + roughness"})
            self.substrate.calibrate(*self.executor.measured_xy(xc, yc, zc), zc)

        t0 = time.monotonic()
        while tasks or len(features) < self._target:
            if cfg.max_run_time_s is not None and time.monotonic() - t0 > cfg.max_run_time_s:
                break  # wall-clock safety net -- see README "When does a scan stop?"

            if tasks:
                task = tasks.pop()
                end_pos = self._trace(task, features, tasks)
                if not tasks:
                    # nothing left to follow -- step off the edge we're sitting on
                    # before searching, otherwise the spiral instantly re-finds it
                    current_pos = self._step_off_feature(end_pos, task.feature)
                    search_radius = cfg.initial_search_radius_V
                    traced = True
                continue

            if spiral_seeds:
                # the last search spiral crossed other edges too: follow the one nearest
                # the tip, unless tracing has covered it since
                tip = self.executor.last_pos
                k = int(np.argmin([np.linalg.norm(s[0] - tip) for s in spiral_seeds]))
                seed, chord, from_xy, baseline, threshold = spiral_seeds.pop(k)
                if self._is_new_edge(seed, features, self._separation(self._initial_radius()), tasks):
                    feature = Feature(baseline=baseline, threshold=threshold, id=self._new_feature_id())
                    features.append(feature)
                    tasks.append(_TraceTask(feature, center=seed, radius=self._first_circle_radius(chord),
                                            backward_angle=_angle_of(from_xy - seed),
                                            first=True, chain_start=seed.copy()))
                continue

            if traced and cfg.spiral_search_mode == "relocate":
                # after tracing, search next where nothing has been scanned yet (inside
                # the area) -- not just next to the feature just traced
                traced = False
                nxt = self._next_unexplored_center(self.executor.last_pos, search_radius)
                if nxt is None:
                    break  # the whole exploration region has been covered -- done
                current_pos = nxt
                self._relocated = True

            found = self._search_for_edge(current_pos, search_radius, features)
            edge_xy, attempt, baseline, threshold = found
            spiral_center = np.asarray(current_pos, dtype=float)
            spiral_seeds = [(p, c, frm, baseline, threshold) for p, c, frm in self._last_spiral_edges[1:]]
            pending_attempts.append(attempt)
            if edge_xy is None:
                if cfg.spiral_search_mode == "grow":
                    # older behaviour: bigger spiral, same center
                    search_radius *= cfg.spiral_growth_factor
                    if search_radius > cfg.max_search_radius_V:
                        break  # exhausted the search budget around this position -- done
                    continue
                # "relocate": same-size spiral, centered on the nearest unexplored area
                nxt = self._next_unexplored_center(self.executor.last_pos, cfg.initial_search_radius_V)
                if nxt is None:
                    break  # the whole exploration region has been covered -- done
                current_pos = nxt
                search_radius = cfg.initial_search_radius_V
                self._relocated = True
                continue

            feature = Feature(spiral_attempts=pending_attempts, baseline=baseline, threshold=threshold,
                              id=self._new_feature_id())
            pending_attempts = []
            features.append(feature)
            tasks.append(_TraceTask(feature, center=edge_xy, radius=self._first_circle_radius(self._last_spiral_edges[0][1]),
                                    backward_angle=_angle_of(self._last_spiral_edges[0][2] - edge_xy),
                                    first=True, chain_start=edge_xy.copy()))

        return features

    # ------------------------------------------------------------------
    def _next_unexplored_center(self, tip: np.ndarray, radius: float) -> Optional[np.ndarray]:
        """Nearest spiral center (to the tip -- least travel) on a grid one search
        radius apart, inside the exploration region, whose spiral would be at least
        explore_min_unexplored_fraction unexplored. None = nothing left to explore."""
        cov = self._coverage
        x0, x1, y0, y1 = cov.bounds
        lim = self._scan_limit()
        step = max(radius, 1e-6)
        xs = np.arange(x0, x1 + 0.5 * step, step)
        ys = np.arange(y0, y1 + 0.5 * step, step)
        cands = np.array([(x, y) for x in xs for y in ys])
        # keep the whole spiral inside the exploration area where it fits
        if x1 - x0 > 2 * radius:
            cands[:, 0] = np.clip(cands[:, 0], x0 + radius, x1 - radius)
        if y1 - y0 > 2 * radius:
            cands[:, 1] = np.clip(cands[:, 1], y0 + radius, y1 - radius)
        cands = np.clip(cands, -lim + radius, lim - radius)  # where the spiral will actually go
        order = np.argsort(np.linalg.norm(cands - np.asarray(tip), axis=1))
        for c in cands[order]:
            if cov.unexplored_fraction(c, radius) >= self.cfg.explore_min_unexplored_fraction:
                return c
        return None

    def _new_feature_id(self) -> int:
        self._n_features_made += 1
        return self._n_features_made

    def _scan_limit(self) -> float:
        """Largest |X| or |Y| any planned waveform may reach -- just inside
        SafetyConfig.max_amplitude_V so float round-off can't trip the safety check."""
        return 0.999 * self.executor.safety.max_amplitude_V

    def _initial_radius(self) -> float:
        c = self.cfg.circle
        if c.radius_mode == "curvature":
            return c.adaptive_min_radius_V  # nothing known about the edge yet: the safe size
        return c.adaptive_max_radius_V if c.radius_mode == "adaptive" else c.fixed_radius_V

    def _first_circle_radius(self, chord: Optional[float]) -> float:
        """First circle on a feature a search spiral found. "curvature": the spiral
        crossed it over `chord` V -- about its width in that direction (less if the
        spiral only grazed it, which errs on the safe, small side) -- so start at
        curvature_radius_fraction x half of that, as for a round feature that wide."""
        c = self.cfg.circle
        if c.radius_mode != "curvature" or not chord:
            return self._initial_radius()
        return float(np.clip(c.curvature_radius_fraction * 0.5 * chord,
                             c.adaptive_min_radius_V, c.adaptive_max_radius_V))

    def _curvature_radius(self, span: float, radius: float) -> float:
        """Next circle's radius from this one's on-feature arc: it spans
        2 acos(r / 2R) round a convex edge of curvature radius R (180 deg = straight;
        a concave edge is the same seen from the substrate side, so use the smaller of
        the on- and off-feature spans)."""
        c = self.cfg.circle
        s = min(span, 2 * np.pi - span)
        cos_half = np.cos(0.5 * s)
        R = radius / (2 * cos_half) if cos_half > 1e-6 else np.inf
        r = c.curvature_radius_fraction * R
        return float(np.clip(r, c.adaptive_min_radius_V, min(c.adaptive_max_radius_V, c.curvature_max_growth * radius)))

    def _separation(self, radius: float) -> float:
        """How close to a traced edge LINE a point must be to count as that same edge."""
        d = self.cfg.same_feature_distance_V
        return 0.25 * radius if d is None else d

    def _min_run(self) -> int:
        return max(1, int(round(self.cfg.edge_detect.min_run_V / self.executor.velocity.sample_spacing_V)))

    def _is_new_edge(self, p: np.ndarray, features: List[Feature], sep: float,
                     tasks: List[_TraceTask]) -> bool:
        """p is on an edge nobody has traced (or queued) yet."""
        for f in features:
            if f.distance_to_edge(p) < sep or f.contains(p):
                return False
        for t in tasks:
            if t.first and np.linalg.norm(t.center - p) < sep:
                return False
        return True

    def _step_off_feature(self, chain_end: np.ndarray, feature: Feature) -> np.ndarray:
        if not feature.circle_steps:
            return chain_end
        centers = np.array([cs.center for cs in feature.circle_steps])
        outward = _unit(chain_end - centers.mean(axis=0))
        return chain_end + outward * 2.0 * self.cfg.initial_search_radius_V

    # ------------------------------------------------------------------
    # SEARCH phase
    # ------------------------------------------------------------------
    def _search_for_edge(self, center: np.ndarray, search_radius: float, features: List[Feature]):
        """Returns (edge_xy or None, SpiralAttempt, substrate baseline, threshold)."""
        cfg = self.cfg
        # keep the whole spiral inside the +/-max_amplitude_V scan range: shrink the
        # radius if it's bigger than the range itself, and slide the center inward so
        # center +/- radius fits. The executor inserts the (measured) move from the
        # tip to the shifted center automatically.
        lim = self._scan_limit()
        area_scan = self._n_spirals == 0 and cfg.first_scan_pattern in ("raster", "lissajous")
        if area_scan:
            x_vec, y_vec, z_vec = self._first_area_scan(np.array(center, dtype=float), search_radius)
        else:
            requested = np.array(center, dtype=float)
            search_radius = min(search_radius, lim)
            center = np.clip(requested, -lim + search_radius, lim - search_radius)
            shifted = np.linalg.norm(center - requested) > 1e-9
            # r_start=0.0 so the spiral's first sample is exactly `center`
            n_cycles = (max(1.0, search_radius / cfg.spiral_pitch_V) if cfg.spiral_pitch_V
                        else cfg.spiral.n_cycles)
            x_vec, y_vec, r_vec, theta_vec = trajectory.spiral_out(
                center, r_start=0.0, r_end=search_radius,
                n_cycles=n_cycles, ds=self.executor.velocity.sample_spacing_V,
                direction=cfg.spiral.direction,
            )
            self._n_spirals += 1
            where = ", moved to an unexplored area" if getattr(self, "_relocated", False) else ""
            self._relocated = False
            z_vec = self.executor.send(x_vec, y_vec, kind="spiral", meta={
                "label": f"Spiral search #{self._n_spirals} -- looking for an edge "
                         f"(radius {search_radius:.3f} V, centered at ({center[0]:.2f}, {center[1]:.2f}) V"
                         f"{where}{', moved inward to stay within the scan range' if shifted else ''})"})
            if hasattr(self, "_coverage"):
                self._coverage.add_disk(center, search_radius)
            # z may be short at the end (raw readback); positions: measured when there are sensors
            x_vec, y_vec = self.executor.measured_xy(x_vec, y_vec, z_vec)

        # height above the substrate (substrate.SubstrateModel): classify, refine the model
        # from this scan's substrate samples, classify again with the refined plane/threshold
        model = self.substrate
        fresh = not model.fitted
        on = model.classify(x_vec, y_vec, z_vec)
        if not fresh:
            model.update(x_vec, y_vec, z_vec, on)
            on = model.classify(x_vec, y_vec, z_vec)
        baseline, threshold = 0.0, model.threshold   # local offset 0: right on the plane
        on = edge_detect.clean_mask(on, self._min_run())
        sep = self._separation(self._initial_radius())

        # every edge crossing not on an already-known edge, in spiral order (a feature
        # crossed on several turns gives several of these -- once one of them has been
        # traced, the others lie on its edge and are dropped before any scanning)
        new_idx: List[int] = []
        for i in edge_detect.transitions(on):  # real edges, not every sample on a plateau
            p = np.array([x_vec[i], y_vec[i]])
            if all(f.distance_to_edge(p) >= sep and not f.contains(p) for f in features) and \
                    all(np.hypot(x_vec[j] - p[0], y_vec[j] - p[1]) >= sep for j in new_idx):
                new_idx.append(int(i))
                if not cfg.follow_all_spiral_edges:
                    break

        edge_idx = new_idx[0] if new_idx else None
        attempt = SpiralAttempt(x=x_vec, y=y_vec, z=z_vec,
                                edge_found=edge_idx is not None, edge_index=edge_idx)
        # (edge point, chord): the chord is how far the spiral ran on the feature there --
        # about the feature's width, sizes the first tracing circle ("curvature" radius)
        ds = self.executor.velocity.sample_spacing_V

        # ... and where the trace was off the feature next to it ("came from" -- the first
        # circle takes the arc away from it): the spiral's center as before; for a raster /
        # Lissajous a few samples along the path, on the off-feature side
        def came_from(i):
            if not area_scan:
                return np.asarray(center, dtype=float)
            m = max(3, self._min_run())
            j = max(i - m, 0) if on[min(i, len(on) - 1)] else min(i + m, len(on) - 1)
            return np.array([x_vec[j], y_vec[j]])
        self._last_spiral_edges = [(np.array([x_vec[i], y_vec[i]]), self._on_run_length(on, i) * ds, came_from(i))
                                   for i in new_idx]
        if edge_idx is None:
            return None, attempt, baseline, threshold
        return np.array([x_vec[edge_idx], y_vec[edge_idx]]), attempt, baseline, threshold

    def _first_area_scan(self, center: np.ndarray, half: float):
        """The first search as a raster / Lissajous over the whole square area: half-width
        `half` (first_search_radius_V -- the priors: the area's half-width), lines
        spiral_pitch_V apart, kept inside the scan range (raster turnarounds and run-ups
        included). Returns (x, y, z)."""
        cfg = self.cfg
        if cfg.first_search_radius_V is None:
            half = cfg.explore_extent_V   # no priors: the whole exploration square
        ds = self.executor.velocity.sample_spacing_V
        lim = self._scan_limit()
        pitch = cfg.spiral_pitch_V or half / max(cfg.spiral.n_cycles, 1)
        n_ease = max(self.executor.connector_ease_samples, 20)
        margin = (0.5 * pitch if cfg.first_scan_pattern == "raster" else 0.0) + 0.5 * n_ease * ds
        # the whole path -- turnarounds and run-ups included -- stays inside the area too
        H = min(half - margin, lim - margin - abs(center[0]), lim - margin - abs(center[1]))
        if H <= 0:
            raise ValueError("edge_follow: the first search area is outside the scan range")
        if cfg.first_scan_pattern == "raster":
            x, y, info = trajectory.raster_scan(center, 2 * H, 2 * H, pitch, ds, "x", (-1, -1))
            desc = f"raster {2 * H:.3g} x {2 * H:.3g} V, {info['n_lines']} lines {info['spacing']:.3g} V apart"
        else:
            x, y, info = trajectory.lissajous_scan(center, 2 * H, 2 * H, pitch, ds)
            desc = (f"Lissajous {2 * H:.3g} x {2 * H:.3g} V ({info['a']}:{info['b']} cycles), "
                    f"gaps like lines {pitch:.3g} V apart")
        x, y, _ = trajectory.add_runup(x, y, ds, n_ease)   # starts and ends at rest
        self._n_spirals += 1
        z = self.executor.send(x, y, kind="search_area", meta={
            "label": f"First search: {desc} over the whole area, centered at ({center[0]:.2f}, {center[1]:.2f}) V"})
        if hasattr(self, "_coverage"):
            self._coverage.add_rect(center[0] - H, center[0] + H, center[1] - H, center[1] + H)
        return (*self.executor.measured_xy(x, y, z), z)

    @staticmethod
    def _on_run_length(on: np.ndarray, i: int) -> int:
        """Samples in the on-feature run that starts or ends at transition i."""
        n = len(on)
        k = i if on[min(i, n - 1)] else min(i + 1, n - 1)   # a sample on the feature, next to the edge
        if not on[k]:
            k = max(i - 1, 0)
            if not on[k]:
                return 0
        change = np.flatnonzero(np.diff(on.astype(np.int8)))  # a run ends at each of these
        j = int(np.searchsorted(change, k))
        a = change[j - 1] + 1 if j > 0 else 0
        b = change[j] if j < len(change) else n - 1
        return int(b - a + 1)

    # ------------------------------------------------------------------
    # TRACE phase
    # ------------------------------------------------------------------
    def _adaptive_radius(self, recent_turn_angle: Optional[float]) -> float:
        cfg = self.cfg.circle
        if recent_turn_angle is None:
            return cfg.adaptive_max_radius_V
        shrink = min(1.0, cfg.adaptive_curvature_gain * recent_turn_angle / np.pi)
        return cfg.adaptive_max_radius_V - shrink * (cfg.adaptive_max_radius_V - cfg.adaptive_min_radius_V)

    def _schedule_other_way(self, f: Feature, radius: float, tasks: List[_TraceTask]) -> None:
        """Edge lost (or circle budget spent) going one way: follow it the other way
        from where it was first found -- same feature."""
        if not self.cfg.trace_both_directions or f._other_way is None:
            return
        c0, other_xy = f._other_way
        f._other_way = None  # only once
        step = min(2 * radius * (1 - self.cfg.circle.overlap_fraction), radius)
        next_center = c0 + _unit(other_xy - c0) * step
        tasks.append(_TraceTask(f, center=next_center, backward_angle=_angle_of(c0 - next_center),
                                radius=radius, leg=1, chain_start=c0.copy()))

    def _closes_loop(self, f: Feature, forward_xy: np.ndarray, leg: int, tol: float) -> bool:
        """forward_xy is back on an EARLIER part of this pass along the edge: within
        tol of the center of a circle from this same leg, excluding the last
        loop_closure_min_steps circles (the ones right behind us, which are always
        close). Only the same leg: the other direction starts next to leg 0's first
        circles by construction, which isn't a closed loop."""
        centers = [cs.center for cs in f.circle_steps if cs.leg == leg]
        earlier = centers[:-self.cfg.loop_closure_min_steps]
        return any(np.linalg.norm(forward_xy - c) < tol for c in earlier)

    def _merge(self, src: Feature, dst: Feature, features: List[Feature], tasks: List[_TraceTask]) -> None:
        """src turned out to be the same edge as dst -- fold it in (one edge = one feature)."""
        dst.spiral_attempts.extend(src.spiral_attempts)
        dst.circle_steps.extend(src.circle_steps)
        dst.edge_points.extend(src.edge_points)
        features[:] = [g for g in features if g is not src]
        for t in tasks:
            if t.feature is src:
                t.feature = dst

    def _trace(self, task: _TraceTask, features: List[Feature], tasks: List[_TraceTask]) -> np.ndarray:
        """Follow one edge from `task` until the loop closes, the edge is lost, it
        joins another feature, or (on_new_edge="switch") a separate edge shows up.
        Returns the last circle center (where the tip is, give or take a radius)."""
        cfg = self.cfg
        f = task.feature
        center = np.array(task.center, dtype=float)
        backward = task.backward_angle
        radius = task.radius
        prev_dir = task.prev_direction_angle
        first = task.first
        steps = task.steps_in_leg
        chain_start = task.chain_start if task.chain_start is not None else center.copy()
        min_run = self._min_run()

        if first and not f.circle_steps:
            # a queued/new edge that another trace has covered since it was found --
            # nothing new to follow, drop it rather than tracing a duplicate
            sep = self._separation(radius)
            if any(g is not f and (g.distance_to_edge(center) < sep or g.contains(center)) for g in features):
                features[:] = [g for g in features if g is not f]
                return center

        while steps < cfg.max_circles_per_feature:  # budget per direction (leg)
            if np.any(np.abs(center) + radius > self._scan_limit()):
                # the edge runs out of the scan range this way -- stop this direction
                # (the other way still gets traced) instead of raising a safety error
                f.hit_scan_limit = True
                break
            # the circle starts AT the backward direction, so the crossing we came in
            # on sits at the start/end of the trace
            x_vec, y_vec, theta_vec = trajectory.circle(
                center, radius, ds=self.executor.velocity.sample_spacing_V,
                start_angle=backward, arc_fraction=1.0,
            )
            z_vec = self.executor.send(x_vec, y_vec, kind="circle", meta={
                "feature": f.id, "circle": len(f.circle_steps) + 1, "leg": task.leg,
                "label": f"Feature {f.id}: circle {len(f.circle_steps) + 1} following the edge"
                         f"{' (other direction)' if task.leg == 1 else ''} -- radius {radius:.3f} V"})
            if hasattr(self, "_coverage"):
                self._coverage.add_disk(center, radius)
            # z may be short at the end (raw readback) -- then the recorded part is an
            # open arc, not a closed loop, so no wraparound
            full_loop = len(z_vec) == len(x_vec)
            d_theta = 2 * np.pi / len(x_vec)   # the circle's samples are evenly spaced in angle
            n = len(z_vec)
            # the crossings' positions: measured when there are sensors (the angles stay the
            # circle's own, for the order round it)
            x_vec, y_vec = self.executor.measured_xy(x_vec, y_vec, z_vec)
            theta_vec = theta_vec[:n]

            model = self.substrate
            h = model.height(x_vec, y_vec, z_vec)
            f.threshold = model.threshold
            on = edge_detect.clean_mask(np.abs(h - f.baseline) > f.threshold, min_run)
            arcs = edge_detect.on_arcs(on, closed=full_loop)
            if not arcs or on.all():
                # entirely off (or entirely on) the feature -> the edge is gone here
                f.circle_steps.append(CircleStep(x_vec, y_vec, z_vec, center.copy(), radius, leg=task.leg))
                f.edge_lost = True
                self._schedule_other_way(f, radius, tasks)
                return center
            if (~on).sum() > 0.1 * n:
                f.baseline = float(np.median(h[~on]))  # local substrate offset next to this edge
            model.update(x_vec, y_vec, z_vec, on)      # refine the plane from the substrate samples

            if first:
                # approached from the side (a spiral, or another edge): the feature lies
                # beyond the edge, so take the arc farthest from where we came from --
                # its two ends are this edge's two directions
                def mid_angle(a):
                    s, e = a
                    m = ((s + e + (n if s > e else 0)) // 2) % n
                    return theta_vec[m]
                cur = max(arcs, key=lambda a: _angular_distance(mid_angle(a), backward))
                fwd_i, other_i = cur[1], cur[0]
            else:
                # the arc with an end nearest where we came from continues this edge;
                # its OTHER end is the way forward
                def near_back(a):
                    return min(_angular_distance(theta_vec[a[0]], backward),
                               _angular_distance(theta_vec[a[1]], backward))
                # skip arcs that lie on ANOTHER feature's traced edge (both ends on it)
                # -- a close neighbour showing up in this circle mustn't be mistaken
                # for the continuation of the edge we're following
                own_tol = self._separation(radius)
                def on_other_feature(a):
                    pts = [np.array([x_vec[k], y_vec[k]]) for k in a]
                    return any(g is not f and all(g.distance_to_edge(q) < own_tol for q in pts) for g in features)
                candidates = [a for a in arcs if not on_other_feature(a)] or arcs
                cur = min(candidates, key=near_back)
                s, e = cur
                if _angular_distance(theta_vec[s], backward) <= _angular_distance(theta_vec[e], backward):
                    fwd_i, other_i = e, s
                else:
                    fwd_i, other_i = s, e
            forward_xy = np.array([x_vec[fwd_i], y_vec[fwd_i]])
            other_xy = np.array([x_vec[other_i], y_vec[other_i]])
            f.edge_points.extend([other_xy, forward_xy])

            step_rec = CircleStep(x_vec, y_vec, z_vec, center.copy(), radius,
                                  crossing_xy=forward_xy, leg=task.leg)
            f.circle_steps.append(step_rec)

            # a SEPARATE edge on this circle: any other on-feature arc whose ends are
            # away from every known edge
            sep = self._separation(radius)
            new_seed = None
            if len(features) < self._target:
                for a in arcs:
                    if a is cur:
                        continue
                    for idx in a:
                        p = np.array([x_vec[idx], y_vec[idx]])
                        if self._is_new_edge(p, features, sep, tasks):
                            new_seed = p
                            break
                    if new_seed is not None:
                        break
            step_rec.new_edge_xy = new_seed

            steps += 1
            # next circle goes along the chord toward the forward crossing, at most one
            # radius so the new center stays on the edge
            step_size = min(2 * radius * (1 - cfg.circle.overlap_fraction), radius)
            # "close enough to count as the same place" has to scale with the step:
            # consecutive crossings/centers are ~step_size apart along the edge, so a
            # fixed 0.02 V almost never hits when circles are big (seen on hardware:
            # r ~0.4 V circles went round a closed feature and just kept going)
            meet_tol = max(cfg.loop_closure_distance_V, 0.6 * step_size)
            # back onto an earlier stretch of this same pass -> closed loop
            if not first and steps >= cfg.loop_closure_min_steps and                     self._closes_loop(f, forward_xy, task.leg, meet_tol):
                f.closed_loop = True
                if hasattr(self, "_coverage"):
                    self._coverage.add_polygon(f.edge_points)  # its inside is known now
                return forward_xy
            # ran into ANOTHER feature's edge -> it's the same edge: merge. The
            # crossing must sit on that feature's traced edge LINE itself -- being
            # merely close to a neighbouring feature is not the same edge (seen on
            # hardware: a feature next to an already-traced one got merged into it
            # and stopped)
            join_tol = 0.1 * radius
            for g in features:
                if g is not f and g.distance_to_edge(forward_xy) < join_tol:
                    self._merge(f, g, features, tasks)
                    return forward_xy

            next_center = center + _unit(forward_xy - center) * step_size
            new_dir = _angle_of(next_center - center)
            if prev_dir is not None and cfg.circle.radius_mode == "adaptive":
                radius = self._adaptive_radius(_angular_distance(new_dir, prev_dir))
            elif cfg.circle.radius_mode == "curvature":
                # the angle the on-feature arc we're following spans on this circle
                span = ((cur[1] - cur[0]) % n + 1) * d_theta
                radius = self._curvature_radius(span, radius)
            prev_dir = new_dir
            if first and task.leg == 0:
                f._other_way = (center.copy(), other_xy)
            first = False
            backward = _angle_of(center - next_center)
            center = next_center

            if new_seed is not None:
                g = Feature(baseline=f.baseline, threshold=f.threshold, id=self._new_feature_id())
                features.append(g)
                new_task = _TraceTask(g, center=new_seed, radius=self._initial_radius(),
                                      backward_angle=_angle_of(step_rec.center - new_seed),
                                      first=True, chain_start=new_seed.copy())
                if cfg.on_new_edge == "switch":
                    # come back to THIS edge afterwards, right where we left it
                    tasks.append(_TraceTask(f, center=center, backward_angle=backward, radius=radius,
                                            leg=task.leg, prev_direction_angle=prev_dir,
                                            steps_in_leg=steps, chain_start=chain_start))
                    tasks.append(new_task)
                    return center
                tasks.insert(0, new_task)  # "queue": after everything already pending

        self._schedule_other_way(f, radius, tasks)  # circle budget spent in this direction
        return center
