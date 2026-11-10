"""
Edge-explorer planner: a continuous path of straight segments that EXPLORES until it
crosses an edge, then FOLLOWS that edge until it is closed (or lost at both ends),
then explores again. Fast between segments -- a few milliseconds of numpy on
fixed-size grids, no model fitting -- so the tip barely waits between waveforms.

  EXPLORE  straight segments of segment_length_V, each in the direction that looks
           most unexplored (distance from everything measured so far, looking
           explore_lookahead_V ahead, a small bonus for going straight -> long sweeps).
           Stops once nothing scores above min_score (the region is explored).
      -> a segment crosses an edge nobody has traced: FOLLOW it, right away.
  FOLLOW   a zigzag of short segments back and forth ACROSS the edge, moving along it:
           each segment is aimed so it crosses the edge at about its middle (the angle
           to the edge set from how far the tip is from it), so every segment adds an
           edge point. The edge direction comes from the last few crossings; the
           segment length adapts -- longer (up to follow_segment_max_V) while the edge
           runs straight, shorter (down to follow_segment_min_V) where it turns.
           A segment that misses the edge (a corner): turn toward where the edge must
           have gone (follow_recovery_turn_deg) and try again, up to
           follow_recovery_tries times; then the edge is lost that way.
           follow_style = "weave": instead of straight segments, a smooth sine weaving
           back and forth across the edge (follow_weave_amplitude_V either side), its
           cycles close together (follow_weave_wavelength_V along the edge each), several
           per waveform (follow_weave_cycles). After every waveform its axis is re-fitted
           to the crossings it just made -- all on the edge -- so the next one is centred
           on the edge and aimed along it, bent by the curvature of a circle through them.
           Each waveform starts where the last one ended, moving the same way (no
           corners: easier on the piezo). A corner shows as crossings that stop part-way
           through a waveform: the next one restarts at the last crossing, turned toward
           the feature (away from it at a concave corner), and the edge direction is
           learned again from the crossings past the corner.
           Everything else (closure, merging, border, recovery, interior) as above.
           Ends when:
             - the edge closes (back to where it was first found, having gone round
               it: points covering loop_closure_min_coverage_deg)  -> feature done
             - it's lost / runs out of the area / max_segments_per_feature
               -> follow it the OTHER way from where it was first found, then done
  INTERIOR once an edge closes (scan_interior): parallel lines across its inside, in
           one waveform, interior_line_spacing_V apart along the feature's long axis,
           interior_margin_V inside the outline -- measures the top (a much better
           reconstruction than the edge band alone) and finds edges INSIDE it: where a
           line drops to the substrate (a hole, an opening), that edge is queued and
           followed like any other (and its own inside scanned if it closes).
      -> other edges crossed while following are QUEUED, and traced one after
         another (nearest first) once this one is finished; then EXPLORE again.

On/off-feature is decided from the height ABOVE THE SUBSTRATE (substrate.SubstrateModel):
a small calibration circle at the start (calibration_radius_V) gives the tilt and the
roughness, the plane is refined after every segment from its substrate samples, and the
threshold moves into the gap between substrate and feature heights once features have
been seen. Queued edges are dropped if they no longer clear the (raised) threshold.

A closed feature's inside counts as explored, and later crossings on a finished edge
are ignored -- the path doesn't re-trace known edges.

Stopping: max_segments / max_features (closed) / max_run_time_s are checked only
BETWEEN features -- an edge being followed is always finished first (its own cap is
max_segments_per_feature). Also stops once exploring finds nothing left to explore.

Works against any backend (simulation or real hardware) through
execution.ContinuousExecutor -- no hardware-specific code here.
"""

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from matplotlib.path import Path as _Polygon
from scipy.ndimage import distance_transform_edt

from ..config import EdgeExplorerConfig
from .. import trajectory
from .. import edge_detect
from ..substrate import SubstrateModel
from ..execution import ContinuousExecutor


WEAVE_MIN_V = 0.2   # follow_style "weave": smallest default amplitude / wavelength (V)
HEIGHT_STRETCH_V = 0.1  # a crossing's height: measured over up to this much path on the feature side
DIR_SPAN_V = 0.02       # a crossing's path direction: over this much path either side (sensor noise)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([1.0, 0.0])


def _rot(v, a) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


@dataclass(eq=False)
class ExplorerFeature:
    id: int
    points: List[np.ndarray] = field(default_factory=list)  # edge crossings, in acquisition order
    dirs: List[np.ndarray] = field(default_factory=list)    # into-the-feature direction at each crossing
    closed_loop: bool = False
    edge_lost: bool = False     # lost (both ways) before it closed
    hit_border: bool = False    # ran out of the exploration area
    finished: bool = False      # followed as far as it goes (closed or not)
    legs: List[List[np.ndarray]] = field(default_factory=list)  # crossings per direction, in order along the edge
    interior_lines: int = 0     # lines scanned across its inside once it closed
    interior_note: str = ""     # why its inside was NOT scanned, if it closed but wasn't
    merged_ids: List[int] = field(default_factory=list)  # other features found to be this same edge
    closed_by_border: bool = False  # runs out of the area: closed along the area's border
    follow_segments: int = 0    # segments spent following it (probes included)
    turn: List[float] = field(default_factory=lambda: [0.0, 0.0])  # signed turn of the estimated
                                # edge direction along each direction of the trace (radians)
    closed_by: str = ""         # which rule closed it (diagnostics)
    border_closure: List[np.ndarray] = field(default_factory=list)  # that stretch of border, from
                                    # where the first direction left the area to where the other did
    inside_of: Optional[int] = None  # an internal edge (e.g. a hole): the feature it's inside
    enclosed: float = float("nan")   # how well its outline encloses it, when last checked (1 = fully)

    @property
    def circle_steps(self):  # the run cell's summary counts these; here: edge points
        return self.points

    def pts(self) -> np.ndarray:
        return np.asarray(self.points, dtype=float).reshape(-1, 2)

    def distance(self, p) -> float:
        if not self.points:
            return np.inf
        return float(np.min(np.linalg.norm(self.pts() - np.asarray(p), axis=1)))

    def contains(self, p) -> bool:
        if not self.closed_loop or len(self.points) < 3:
            return False
        return bool(_Polygon(self._ordered()).contains_point(p))

    def _ordered(self) -> np.ndarray:
        """The outline, in order along the edge: the second direction reversed, then
        the first (both start where the edge was found). Without legs: points sorted
        by angle round their centroid."""
        if self.legs and len(self.legs[0]) >= 2:
            seq = (self.legs[1][:0:-1] if len(self.legs) > 1 else []) + self.legs[0]
            # closed at the area's border: from where leg 0 left the area, along the
            # border, back to where the other direction did
            return np.asarray(seq + list(self.border_closure), dtype=float)
        P = self.pts()
        c = P.mean(axis=0)
        return P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]

    def angular_coverage(self) -> float:
        """How far around their own centroid the edge points go (radians, 0..2pi):
        ~2pi for an edge traced all the way round a feature."""
        P = self.pts()
        if len(P) < 3:
            return 0.0
        ang = np.sort(np.arctan2(P[:, 1] - P[:, 1].mean(), P[:, 0] - P[:, 0].mean()))
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
        return float(2 * np.pi - gaps.max())


@dataclass
class ExplorerSegment:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    mode: str               # "initial" | "follow" | "explore"
    edge_points: List[np.ndarray] = field(default_factory=list)


@dataclass(eq=False)  # identity comparison -- holds numpy arrays
class _Crossing:
    p: np.ndarray           # where the segment crossed the edge
    into: np.ndarray        # unit vector pointing into the feature there
    inside_of: Optional[int] = None  # found by the interior scan of this feature
    height: float = np.inf  # |height above the substrate| on its feature side, when found


class EdgeExplorerPlanner:
    # a closed feature's inside: counts as explored, and edges found there as known (they
    # belong to it) -- topography. Off for domains (pfm.DomainWallExplorerPlanner): domains
    # nest, and the inside of one closed by the border is ambiguous
    INSIDE_IS_KNOWN = True

    def __init__(self, executor: ContinuousExecutor, cfg: EdgeExplorerConfig):
        self.executor = executor
        self.cfg = cfg
        self.features: List[ExplorerFeature] = []
        self.segments: List[ExplorerSegment] = []
        self.step_times_s: List[float] = []  # planning time before each EXPLORING segment (the map search)

    # ------------------------------------------------------------------
    # derived lengths
    # ------------------------------------------------------------------
    @property
    def _L(self) -> float:
        return self.cfg.segment_length_V

    def _follow_L(self) -> float:
        return self.cfg.follow_segment_length_V if self.cfg.follow_segment_length_V is not None else 0.5 * self._L

    def _follow_limits(self):
        lo = self.cfg.follow_segment_min_V if self.cfg.follow_segment_min_V is not None else self._follow_L()
        hi = self.cfg.follow_segment_max_V if self.cfg.follow_segment_max_V is not None else self._follow_L()
        return min(lo, hi), max(lo, hi)

    def _explore_scale(self) -> float:
        return self.cfg.explore_scale_V if self.cfg.explore_scale_V is not None else self._L

    def _scan_limit(self) -> float:
        return 0.999 * self.executor.safety.max_amplitude_V

    # ------------------------------------------------------------------
    def run(self, start_xy=(0.0, 0.0)) -> List[ExplorerFeature]:
        cfg = self.cfg
        self.executor.reset(start_xy)
        self.features, self.segments, self.step_times_s = [], [], []
        self._next_id = 1
        self.substrate = self._make_model()
        self._baseline = None   # (kept for inspection: the substrate level at the start ...
        self._threshold = None  #  ... and the current on/off threshold -- see self.substrate)
        self._prev_on = None
        self._queue: List[_Crossing] = []   # edges seen but not traced yet

        lim = self._scan_limit()
        e = cfg.explore_extent_V
        c0 = np.array(cfg.explore_center_V if cfg.explore_center_V is not None else start_xy, dtype=float)
        self._bounds = (max(c0[0] - e, -lim), min(c0[0] + e, lim),
                        max(c0[1] - e, -lim), min(c0[1] + e, lim))
        n = cfg.grid_n
        x0, x1, y0, y1 = self._bounds
        self._gx = np.linspace(x0, x1, n)
        self._gy = np.linspace(y0, y1, n)
        self._cell = max((x1 - x0) / (n - 1), (y1 - y0) / (n - 1))
        self._measured = np.zeros((n, n), dtype=bool)   # row = y, col = x

        pos = c0.copy()  # the initial path starts at the area's center (the executor moves there)
        a = np.deg2rad(cfg.initial_direction_deg)
        heading = np.array([np.cos(a), np.sin(a)])
        t0 = time.monotonic()

        # calibration: a small circle round the start -- the substrate tilt and roughness
        # before any decision (a straight line alone only shows the slope along itself).
        # It ends where the initial straight path begins, so no extra move.
        r_cal = cfg.calibration_radius_V
        if r_cal:
            r_cal = min(r_cal, max(0.0, self._scan_limit() - float(np.max(np.abs(c0)))))
        if r_cal:
            xc, yc, _ = trajectory.circle(c0, r_cal, ds=self.executor.velocity.sample_spacing_V,
                                          start_angle=float(a))
            self._enqueue(self._do_path(xc, yc, "calibration",
                                        f"Edge explorer: calibration circle (radius {r_cal:.3g} V) -- "
                                        f"substrate tilt + roughness"))
            pos = np.array([xc[-1], yc[-1]])

        # the initial straight path
        self.step_times_s.append(0.0)
        length = self._fit(pos, heading, cfg.initial_path_length_V)
        pos, crossings = self._do_segment(pos, heading, length, "initial", "Edge explorer: initial straight path")
        self._enqueue(crossings)
        jumps = 0   # consecutive moves to a far unexplored region
        self.quiet_path_V = 0.0   # exploring path since the last new edge was followed
        self.stop_reason = None
        x0, x1, y0, y1 = self._bounds
        sweep_V = (x1 - x0) * (y1 - y0) / (cfg.explore_scale_V or self._L)

        while True:
            # between features: the stopping criteria (a trace is never cut short)
            if cfg.max_run_time_s is not None and time.monotonic() - t0 > cfg.max_run_time_s:
                self.stop_reason = f"max_run_time_s ({cfg.max_run_time_s:g} s)"
                break
            if (cfg.stop_after_quiet_sweeps is not None and not self._queue
                    and self.quiet_path_V >= cfg.stop_after_quiet_sweeps * sweep_V):
                self.stop_reason = (f"no new edge over {self.quiet_path_V:.3g} V of exploring "
                                    f"({cfg.stop_after_quiet_sweeps:g} sweeps of the area, {sweep_V:.3g} V each)")
                break
            if len(self.segments) >= cfg.max_segments:
                break
            if sum(f.closed_loop for f in self.features) >= cfg.max_features:
                break
            if self._queue:
                # the queued edge nearest the tip -- unless tracing covered it meanwhile
                k = int(np.argmin([np.linalg.norm(c.p - pos) for c in self._queue]))
                seed = self._queue.pop(k)
                # found while the threshold was lower (before the features' height was
                # known): no longer an edge -- e.g. a line of constant height on the tilt
                if seed.height < self.substrate.threshold:
                    continue
                if not self._known(seed.p, inside_of=seed.inside_of):
                    pos, heading = self._follow(seed, pos)
                    self.quiet_path_V = 0.0
                continue
            t_plan = time.perf_counter()
            choice = self._choose(pos, heading)
            self.step_times_s.append(time.perf_counter() - t_plan)
            if choice is None:
                # nothing unexplored within reach of here -- e.g. the tip ended in a corner
                # after scanning the inside of a feature cut by the border. If unexplored
                # area is left elsewhere, go there in one straight (measured) line; stop
                # only when there is none (or jumping doesn't help)
                far = self._nearest_unexplored(pos) if jumps < 3 else None
                if far is None:
                    self.stop_reason = "everything explored"
                    break
                jumps += 1
                d = _unit(far - pos)
                length = self._fit(pos, d, float(np.linalg.norm(far - pos)))
                if length < 0.5 * self._L:
                    break
                label = f"Edge explorer segment {len(self.segments) + 1}: moving to an unexplored region"
                pos, crossings = self._do_segment(pos, d, length, "explore", label)
                heading = d
                self._enqueue(crossings)
                continue
            jumps = 0
            d, score = choice
            label = f"Edge explorer segment {len(self.segments) + 1}: exploring (score {score:.2f})"
            pos, crossings = self._do_segment(pos, d, self._L, "explore", label)
            self.quiet_path_V += self._L
            heading = d
            self._enqueue(crossings)
        if self.stop_reason is None:
            self.stop_reason = "max_segments / max_features / everything explored"
        print(f"Edge explorer stopped: {self.stop_reason}")
        return self.features

    def _make_model(self):
        """The on/off classifier: height above the fitted substrate (subclasses: other signals,
        e.g. pfm.DomainWallExplorerPlanner)."""
        return SubstrateModel(self.cfg.edge_detect)

    # ------------------------------------------------------------------
    # measure + detect
    # ------------------------------------------------------------------
    def _fit(self, pos, d, length) -> float:
        """The longest part of `length` along d that stays inside the area (and the
        safety limit)."""
        x0, x1, y0, y1 = self._bounds
        lim = self._scan_limit()
        lo = np.maximum([x0, y0], -lim)
        hi = np.minimum([x1, y1], lim)
        t = length
        for k in range(2):
            if d[k] > 1e-12:
                t = min(t, (hi[k] - pos[k]) / d[k])
            elif d[k] < -1e-12:
                t = min(t, (lo[k] - pos[k]) / d[k])
        return max(0.0, float(t))

    def _line(self, a, b):
        """Samples from a to b at the tip's sample spacing, b included."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-12:
            return np.array([b[0]]), np.array([b[1]])
        x, y = trajectory.line_segment(a, (b - a) / L, L, self.executor.velocity.sample_spacing_V)
        # line_segment excludes its far end -- append it so the tip lands exactly on b
        # (the next segment starts there, no connector needed)
        return np.append(x, b[0]), np.append(y, b[1])

    def _do_segment(self, pos, direction, length, mode, label):
        """Measure a straight segment; returns (end, crossings on it)."""
        pos = np.asarray(pos, dtype=float)
        end = pos + direction * length
        x_vec, y_vec = self._line(pos, end)
        return end, self._do_path(x_vec, y_vec, mode, label)

    def _do_path(self, x_vec, y_vec, mode, label) -> List[_Crossing]:
        """Measure any path (one waveform); returns its edge crossings, in order."""
        start = np.array([x_vec[0], y_vec[0]])
        if self.executor.last_pos is not None and np.linalg.norm(np.asarray(self.executor.last_pos) - start) > 1e-9:
            self._prev_on = None  # a jump (the executor inserts the move): no state carried over
        z_vec = self.executor.send(x_vec, y_vec, kind="explore", meta={"label": label, "mode": mode})
        # where each z was measured: the sensors' positions when there are (cfg.position_sensors
        # .plan_on_sensors) -- crossings, substrate, coverage and closing all work from them
        x_vec, y_vec = self.executor.measured_xy(x_vec, y_vec, z_vec)
        crossings = self._learn(x_vec, y_vec, z_vec)
        self.segments.append(ExplorerSegment(x_vec, y_vec, z_vec, mode, [c.p for c in crossings]))
        return crossings

    def _learn(self, x, y, z) -> List[_Crossing]:
        """Mark the segment as explored; return its edge crossings, in order."""
        cfg = self.cfg
        model = self.substrate
        # on/off-feature from the height above the substrate: classify with the current
        # model (the first waveform calibrates it), refine the model from this segment's
        # substrate samples, classify again with the refined plane / threshold
        fresh = not model.fitted
        on = model.classify(x, y, z)
        if not fresh:
            model.update(x, y, z, on)
            on = model.classify(x, y, z)
        self._baseline = float(model.coef[0])   # substrate level at (0, 0)
        self._threshold = model.threshold
        h = np.abs(model.height(x, y, z))
        min_run = max(1, int(round(cfg.edge_detect.min_run_V / self.executor.velocity.sample_spacing_V)))
        thr, hf = model.threshold, cfg.edge_detect.hysteresis_fraction
        if hf > 0:
            # two thresholds: no burst of flickering edges along a sidewall
            on = edge_detect.hysteresis_mask(h, thr * (1 - hf), thr * (1 + hf), initial=self._prev_on)
        on = edge_detect.clean_mask(on, min_run)
        idx = list(edge_detect.transitions(on))
        if self._prev_on is not None and len(on) and on[0] != self._prev_on:
            idx = [0] + idx  # the crossing fell exactly between two segments
        entering = [bool(on[i]) for i in idx]   # the new state, where the mask switched
        if hf > 0:
            # ... each one put back where the height passes the single threshold
            idx = [edge_detect.refine_crossing(h, i, thr, e, i - (idx[k - 1] if k else 0))
                   for k, (i, e) in enumerate(zip(idx, entering))]
        if len(on):
            self._prev_on = bool(on[-1])

        # everything measured along the way counts as explored
        stride = max(1, int(self._cell / 2 / self.executor.velocity.sample_spacing_V))
        r, c = self._rc(x[::stride], y[::stride])
        self._measured[r, c] = True

        out = []
        # crossing height: median |height| over up to HEIGHT_STRETCH_V of the feature side,
        # up to the next / previous crossing. (A few samples only -- as before -- sit on the
        # sidewall: real 0.28 V edges read ~0.10 V, and once the threshold rose past that,
        # their queued crossings were dropped as "no longer an edge".)
        w = max(4 * min_run, int(round(HEIGHT_STRETCH_V / self.executor.velocity.sample_spacing_V)))
        # the local path direction is taken over DIR_SPAN_V of path either side, not a couple
        # of samples: with measured (sensor) positions the sample-to-sample noise (~0.002 V on
        # the instrument, 2026-10-13) is ~40x the step at 50 V/s and 1 MHz -- over +-2 samples
        # half the "into" directions came out backwards, so no outline passed the enclosure
        # check and nothing closed
        span = max(2, int(round(DIR_SPAN_V / self.executor.velocity.sample_spacing_V)))
        for k, (i, ent) in enumerate(zip(idx, entering)):
            # the local path direction there (paths can turn: the interior scan)
            j0, j1 = max(i - span, 0), min(i + span, len(x) - 1)
            d = _unit([x[j1] - x[j0], y[j1] - y[j0]]) if j1 > j0 else np.array([1.0, 0.0])
            # entering the feature (off -> on): "into" is along the path, leaving: against it
            if ent:
                stop = idx[k + 1] if k + 1 < len(idx) else len(h)
                side = h[i:min(i + w, stop)]
            else:
                start = idx[k - 1] if k > 0 else 0
                side = h[max(i - w, start):i]
            height = float(np.median(side)) if len(side) else np.inf
            out.append(_Crossing(np.array([x[i], y[i]]), d if ent else -d, height=height))
        return out

    def _rc(self, x, y):
        x0, x1, y0, y1 = self._bounds
        n = self.cfg.grid_n
        c = np.clip(np.round((np.asarray(x) - x0) / max(x1 - x0, 1e-12) * (n - 1)), 0, n - 1).astype(int)
        r = np.clip(np.round((np.asarray(y) - y0) / max(y1 - y0, 1e-12) * (n - 1)), 0, n - 1).astype(int)
        return r, c

    def _known(self, p, exclude=None, inside_of=None) -> bool:
        """p is on an edge already traced (or being traced), or inside a closed one --
        except inside `inside_of`: an interior scan looks for edges inside it."""
        r = self._follow_limits()[0]
        return any(f is not exclude and (f.distance(p) < r or (self.INSIDE_IS_KNOWN and f.contains(p)
                                                                and f.id != inside_of))
                   for f in self.features)

    def _enqueue(self, crossings: List[_Crossing], exclude=None) -> None:
        r = self._follow_limits()[0]
        for c in crossings:
            if (self._known(c.p, exclude, c.inside_of)
                    or any(np.linalg.norm(q.p - c.p) < r for q in self._queue)):
                continue
            self._queue.append(c)

    # ------------------------------------------------------------------
    # FOLLOW: zigzag along one edge until it closes (or is lost both ways)
    # ------------------------------------------------------------------
    def _follow(self, seed: _Crossing, pos):
        cfg = self.cfg
        f = ExplorerFeature(id=self._next_id, points=[seed.p], dirs=[seed.into], inside_of=seed.inside_of)
        self._next_id += 1
        self.features.append(f)
        lo, hi = self._follow_limits()
        into0 = _unit(seed.into)
        t0 = _rot(into0, -np.pi / 2)   # leg 0 runs with the feature on its left
        tip = np.asarray(pos, dtype=float)
        heading = t0
        start_p = seed.p.copy()        # where leg 0 started (moved if the edge is re-found elsewhere)
        leg0_end = start_p
        for leg in (0, 1):
            Lf = lo
            p, t, n_in = start_p.copy(), (t0 if leg == 0 else -t0), into0.copy()
            if leg == 0 and np.linalg.norm(tip - p) < 3 * max(self._L, Lf) and self._prev_on is not None:
                side = self._prev_on          # carry straight on from where we crossed it
            else:
                # (re)start: find the edge again with a probe line across it -- after a
                # long move the open-loop piezo puts it a few tenths of a V elsewhere
                found = self._reacquire(f, p, n_in)
                if found is None:
                    f.edge_lost = True
                    continue
                c, tip = found
                p = c.p
                n_in = _unit(c.into)
                t = _rot(n_in, -np.pi / 2) if leg == 0 else _rot(n_in, np.pi / 2)
                if leg == 0:
                    start_p = leg0_end = p.copy()
                f.points.append(p)
                f.dirs.append(c.into)
                side = bool(self._prev_on) if self._prev_on is not None else True
            leg_pts = [p]
            f.legs.append(leg_pts)
            misses = 0
            t_meas = t.copy()   # edge direction at the last crossing (before recovery guesses)
            kappa = 0.0         # weave: signed curvature of the edge (1/V, + = turning left)
            weave = cfg.follow_style == "weave"
            since = 0           # weave: the edge direction is fitted to leg_pts[since:] (a corner resets it)
            just_turned = False  # weave: the last waveform started at a corner, turned
            t_before = None      # weave: the edge direction before the last corner / recovery turn
            bad_closes = 0       # a closure rule fired but the outline doesn't enclose the feature
            for _ in range(cfg.max_segments_per_feature):
                # aim at where the edge should be next -- one step along it from the last
                # crossing -- and stop a short, fixed distance past it: the segment
                # crosses the edge there and the tip never drifts far from it
                label = (f"Edge explorer segment {len(self.segments) + 1}: following the edge of feature {f.id}"
                         f"{' (other direction)' if leg else ''} -- {Lf:.3g} V steps")
                if weave:
                    built = self._weave_chunk(tip, heading, p, t, kappa, Lf)
                    if built is None:
                        f.hit_border = True
                        break
                    x_vec, y_vec, center, amp, axis_len = built
                    crossings = self._do_path(x_vec, y_vec, "follow", label.replace("V steps", "V weave"))
                    tip_new = np.array([x_vec[-1], y_vec[-1]])   # the whole waveform was output
                    k2 = max(0, len(x_vec) - 1 - max(2, len(x_vec) // 200))
                    heading = _unit(tip_new - np.array([x_vec[k2], y_vec[k2]]))
                    target = center[-1]
                    steer = None
                else:
                    target = p + t * Lf
                    steer = self._steer_target(f, leg, leg_pts, start_p, tip, Lf) if cfg.fast_closure else None
                    if steer is not None:
                        target = steer   # nearly round: aim straight across the start of the trace
                    d = _unit(target - tip)
                    length = self._fit(tip, d, np.linalg.norm(target - tip) + 0.5 * Lf)
                    if length < 0.2 * lo:
                        f.hit_border = True
                        break
                    tip_new, crossings = self._do_segment(tip, d, length, "follow",
                                                          label + (" -- closing the loop" if steer is not None else ""))
                    heading = d
                f.follow_segments += 1
                side = bool(self._prev_on) if self._prev_on is not None else side
                reach = 1.5 * Lf + 0.5 * lo
                for c in crossings:  # an internal edge (a hole) lies inside its parent's outline
                    c.inside_of = f.inside_of
                if weave:   # this edge: crossings near the weave's axis (the predicted edge)
                    near = [c for c in crossings
                            if np.min(np.linalg.norm(center - c.p, axis=1)) < amp + 0.5 * lo]
                else:
                    near = [c for c in crossings if np.linalg.norm(c.p - target) < reach]
                mine = [c for c in near if not self._known(c.p, f, f.inside_of)]
                if weave and len(leg_pts) > 1:
                    # no backtracking: a crossing on the stretch of edge this direction already
                    # traced (not the newest crossings) is not progress -- after a corner
                    # misjudged, following it would go back round the way it came. (Closing
                    # the loop is checked on the waveform's own crossings, not on these.)
                    recent = int(4 * max(cfg.follow_weave_cycles, 1))
                    old = np.asarray(leg_pts[:-recent], dtype=float).reshape(-1, 2)
                    if len(old):
                        # at least the weave's own half-width: on the instrument a reversed
                        # trace came back ~0.4 V outside its earlier crossings
                        r_old = max(0.5 * self._weave_wavelength(), self._weave_amplitude())
                        mine = [c for c in mine if np.min(np.linalg.norm(old - c.p, axis=1)) > r_old]
                self._enqueue([c for c in crossings if not any(c is m for m in mine)], exclude=f)
                tip = tip_new
                if cfg.fast_closure and self._back_at_start(f, leg, leg_pts, mine, Lf):
                    for c in mine:
                        f.points.append(c.p)
                        f.dirs.append(c.into)
                    if mine:
                        leg_pts.append(min(mine, key=lambda c: np.linalg.norm(c.p - target)).p)
                    f.closed_loop = True
                    f.closed_by = "back at the start"
                    self._mark_inside(f)
                    break
                if not mine and cfg.merge_features:
                    # ran into ANOTHER open edge: the same edge, traced from elsewhere
                    g = self._other_open_feature(f, near)
                    if g is not None:
                        goal = start_p if leg == 0 else leg0_end
                        if self._merge(f, g, leg, leg_pts, goal, Lf) and self._loop_ok(f):
                            f.closed_loop = True
                            f.closed_by = "merged outline"
                            self._mark_inside(f)
                        break
                if mine:
                    misses = 0
                    for c in mine:
                        f.points.append(c.p)
                        f.dirs.append(c.into)
                    if weave:   # every crossing is a point along the edge, in order
                        q = mine[-1]
                        leg_pts.extend(c.p for c in mine)
                        t_new = self._weave_tangent(leg_pts[since:], t)
                        # a direction fitted to a few crossings past a corner can come out
                        # backwards (seen on the instrument: the trace then went all the way
                        # back round): never more than ~120 deg from the direction before the
                        # corner -- that's the same line, the other way
                        ref = t_before if t_before is not None else t
                        if np.dot(t_new, ref) < -0.5:
                            t_new = -t_new
                        if t_before is not None and len(leg_pts) - since >= 8:
                            t_before = None      # the new direction is established
                    else:
                        q = min(mine, key=lambda c: np.linalg.norm(c.p - target))
                        leg_pts.append(q.p)
                        t_new = self._tangent(leg_pts, t, Lf)
                    f.turn[leg] += float(np.clip(np.arctan2(t_meas[0] * t_new[1] - t_meas[1] * t_new[0],
                                                            np.dot(t_meas, t_new)), -np.pi / 2, np.pi / 2))
                    t_meas = t_new.copy()
                    turn = np.arccos(np.clip(np.dot(t, t_new), -1, 1))
                    t = t_new
                    n_in = np.array([-t[1], t[0]])
                    if np.dot(n_in, q.into) < 0:
                        n_in = -n_in
                    p = q.p
                    if weave:
                        kappa = 0.5 * kappa + 0.5 * self._curvature(leg_pts[since:], t, Lf)
                        # a corner: the crossings stopped more than a cycle before the end of
                        # the waveform -- the edge turned at the last one
                        lam = self._weave_wavelength()
                        along = float(np.dot(q.p - center[0], _unit(center[-1] - center[0]))) \
                            if len(center) > 1 else axis_len
                        cornered = axis_len - along > 1.25 * lam
                        if cornered and just_turned and len(leg_pts) - since >= 3:
                            # right after a corner turn: its crossings past the corner already
                            # give the new direction (t, fitted above) -- restart from the last
                            # one along it, without turning again
                            kappa = 0.0
                            just_turned = False
                        elif cornered:
                            toward = n_in if not side else -n_in
                            a = np.deg2rad(cfg.follow_weave_corner_deg) * np.sign(t[0] * toward[1] - t[1] * toward[0])
                            t_before = t.copy() if t_before is None else t_before
                            t, n_in = _rot(t, a), _rot(n_in, a)
                            kappa = 0.0
                            since = len(leg_pts) - 1     # learn the direction again from the corner on
                            just_turned = True
                        else:
                            just_turned = False
                    else:
                        Lf = min(Lf * 1.25, hi) if turn < np.deg2rad(15) else max(Lf * 0.6, lo) if turn > np.deg2rad(40) else Lf
                    if cfg.close_at_border and self._border_gap(p) <= 0.5 * lo:
                        # the edge reached the area's border: this direction ends here (going
                        # on, a recovery turn would only send it back along its own edge)
                        f.hit_border = True
                        break
                    if self._closes(f, leg, leg_pts, start_p if leg == 0 else leg0_end, Lf):
                        if self._loop_ok(f):
                            f.closed_loop = True
                            f.closed_by = "strict rules"
                            self._mark_inside(f)
                            break
                        # back round, but the outline doesn't enclose the feature (it reversed,
                        # or is too ragged): not closed. A few times running -> this direction
                        # is done (the other one is tried, from where the edge was found)
                        bad_closes += 1
                        if bad_closes >= 3:
                            f.edge_lost = True
                            break
                else:
                    misses += 1
                    if misses > cfg.follow_recovery_tries:
                        f.edge_lost = True
                        break
                    # missed it: the edge turned -- toward the feature if we're outside
                    # it (a convex corner), away from it if we're on it (a concave one)
                    toward = n_in if not side else -n_in
                    a = np.deg2rad(cfg.follow_recovery_turn_deg) * np.sign(t[0] * toward[1] - t[1] * toward[0])
                    if weave and t_before is None:
                        t_before = t.copy()
                    t, n_in = _rot(t, a), _rot(n_in, a)
                    Lf = max(Lf * 0.6, lo)
                    kappa = 0.0
                    since = len(leg_pts) - 1
            if f.closed_loop:
                break
            if leg == 0:
                leg0_end = leg_pts[-1]  # the other direction closes the loop when it gets here
            elif self._ends_meet(f, leg0_end, leg_pts[-1], Lf) and self._loop_ok(f):
                f.closed_loop = True
                f.closed_by = "both directions' ends meet"
                self._mark_inside(f)
        if not f.closed_loop and cfg.close_at_border and self._close_at_border(f):
            if self._loop_ok(f, trim=False):
                f.closed_loop = True
                f.closed_by = "area border"
                self._mark_inside(f)
            else:
                f.border_closure = []
        f.finished = True
        if not f.legs:
            # the probe found no edge there, either way: not an edge (e.g. a crossing found
            # while the threshold was still low) -- not a feature
            self.features = [g for g in self.features if g is not f]
            return tip, heading
        if f.closed_loop and cfg.scan_interior:
            tip = self._scan_interior(f, tip)
        return tip, heading

    # ------------------------------------------------------------------
    # WEAVE (follow_style = "weave"): a smooth sine across the edge, along its prediction
    # ------------------------------------------------------------------
    def _weave_amplitude(self, Lf=None) -> float:
        a = self.cfg.follow_weave_amplitude_V
        # wider than the open-loop piezo's position error (0.1-0.3 V on the instrument), or
        # the edge slips out of the weave whatever the feature size
        return float(a) if a is not None else max(0.5 * self._follow_limits()[0], WEAVE_MIN_V)

    def _weave_wavelength(self) -> float:
        w = self.cfg.follow_weave_wavelength_V
        return float(w) if w is not None else max(0.5 * self._follow_limits()[0], WEAVE_MIN_V)

    def _weave_tangent(self, pts, t_prev) -> np.ndarray:
        """Edge direction from the newest crossings (up to two cycles' worth), pointing the
        way we're going; t_prev if they don't span a full wavelength of the edge yet (a
        fit to crossings closer together is noise -- the piezo's position error)."""
        P = np.asarray(pts[-5:], dtype=float).reshape(-1, 2)
        if len(P) < 2 or np.ptp(P, axis=0).max() < self._weave_wavelength():
            return t_prev
        _, _, vt = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)
        t = vt[0]
        # pointing the way the trace moves: the crossings are in the order they were made
        # (t_prev can be off after a corner turn -- and a flipped sign sends the weave back
        # along the edge it just did)
        prog = P[-1] - P[0]
        ref = prog if np.linalg.norm(prog) > 0.25 * self._weave_wavelength() else t_prev
        return t if np.dot(t, ref) >= 0 else -t

    def _curvature(self, pts, t, Lf) -> float:
        """Signed curvature of the edge (1/V, + = turning left going along t) from a circle
        through the last few crossings, if they span enough of the edge; 0 otherwise.
        Clipped to 1 / Lf (a corner is handled by the recovery turn, not by bending)."""
        P = np.asarray(pts[-7:], dtype=float).reshape(-1, 2)
        if len(P) < 3 or np.linalg.norm(P[-1] - P[0]) < 1.5 * self._weave_wavelength():
            return 0.0
        a, b, c = P[0], P[len(P) // 2], P[-1]
        ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
        cross = (b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0]
        if ab * bc * ca < 1e-12:
            return 0.0
        k = 2.0 * cross / (ab * bc * ca)          # signed, along a -> b -> c
        if np.dot(c - a, t) < 0:
            k = -k
        return float(np.clip(k, -2.0 / Lf, 2.0 / Lf))

    def _weave_chunk(self, tip, heading, p, t, kappa, Lf):
        """One weave waveform from the tip: the edge predicted from p (last crossing) along t,
        bent by kappa; the path weaves +-amplitude across it, follow_weave_wavelength_V
        along the edge per cycle, for follow_weave_cycles cycles. Starts exactly at the tip, moving across the edge the
        way the tip is already moving. Returns (x, y, predicted edge points, amplitude,
        axis length), or None if it doesn't fit in the area."""
        cfg = self.cfg
        lo = self._follow_limits()[0]
        ds = self.executor.velocity.sample_spacing_V
        tip, p, t = (np.asarray(v, dtype=float) for v in (tip, p, t))
        t = _unit(t)
        lam = self._weave_wavelength()
        A = self._weave_amplitude(Lf)
        S = max(cfg.follow_weave_cycles, 0.25) * lam
        kappa = float(np.clip(kappa, -np.pi / 4 / S, np.pi / 4 / S))   # bend <= 45 deg per waveform
        # the tangent from the last crossings is a chord: it lags round a curve -- turn it
        # forward to the newest crossing
        t = _rot(t, kappa * lam)
        nl = np.array([-t[1], t[0]])
        s0 = float(np.dot(tip - p, t))
        d_tip = float(np.dot(tip - (p + s0 * t), nl))
        # near the prediction (the usual case: carrying on from the last waveform): the
        # weave starts level with the tip, at its offset; otherwise (a new leg, a recovery
        # turn): it starts at the last crossing, crossing the edge there, reached from the
        # tip by a smooth connector
        joined = abs(d_tip) <= A and -0.25 * lam <= s0 <= 0.75 * lam
        if joined:
            base = p + s0 * t
            phi0 = float(np.arcsin(np.clip(d_tip / A, -0.95, 0.95)))
            if np.dot(heading, nl) < 0:
                phi0 = np.pi - phi0
        else:
            base = p.copy()
            phi0 = 0.0 if d_tip < 0 else np.pi     # cross towards the tip's far side
        n = int(min(max(S / (0.25 * ds), 400), 4_000_000))
        s = np.linspace(0.0, S, n)
        th = np.arctan2(t[1], t[0]) + kappa * s
        cx = base[0] + np.concatenate([[0.0], np.cumsum(np.cos(th[:-1]) * np.diff(s))])
        cy = base[1] + np.concatenate([[0.0], np.cumsum(np.sin(th[:-1]) * np.diff(s))])
        off = A * np.sin(phi0 + 2 * np.pi * s / lam)
        qx, qy = cx - np.sin(th) * off, cy + np.cos(th) * off
        if joined:
            # exactly from the tip: the small mismatch blended out over a quarter wavelength
            dx, dy = tip[0] - qx[0], tip[1] - qy[0]
            w = 0.5 * (1 + np.cos(np.pi * np.minimum(s / (0.25 * lam), 1.0)))
            qx, qy = qx + dx * w, qy + dy * w
        else:
            # cubic Hermite connector: leaves the tip along its heading, joins the weave
            # along its direction -- no corner either end
            q0 = np.array([qx[0], qy[0]])
            dq = _unit([qx[1] - qx[0], qy[1] - qy[0]])
            Lc = float(np.linalg.norm(q0 - tip))
            if Lc > 1e-9:
                m = int(max(Lc / (0.25 * ds), 50))
                u = np.linspace(0.0, 1.0, m, endpoint=False)[:, None]
                h00, h10, h01, h11 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u, -2*u**3 + 3*u**2, u**3 - u**2
                C = h00 * tip + h10 * (Lc * _unit(heading)) + h01 * q0 + h11 * (Lc * dq)
                qx, qy = np.concatenate([C[:, 0], qx]), np.concatenate([C[:, 1], qy])
        # constant tip speed: resample at the sample spacing along the path
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(qx), np.diff(qy)))])
        m = np.arange(0.0, arc[-1], ds)
        x, y = np.interp(m, arc, qx), np.interp(m, arc, qy)
        # stay inside the area (and the safety limit)
        x0, x1, y0, y1 = self._bounds
        lim = self._scan_limit()
        out = (x < max(x0, -lim)) | (x > min(x1, lim)) | (y < max(y0, -lim)) | (y > min(y1, lim))
        if out.any():
            k = int(np.argmax(out))
            if k * ds < 0.2 * lo:
                return None
            x, y = x[:k], y[:k]
        if len(x) < 2:
            return None
        center = np.column_stack([cx[::max(1, n // 200)], cy[::max(1, n // 200)]])
        return x, y, center, A, S

    def _loop_ok(self, f, trim: bool = True) -> bool:
        """The outline is fit to be called closed: joined up, it encloses the feature
        (_encloses >= loop_closure_min_enclosed) -- after trimming a single-direction
        trace that wound round more than once to its first full loop. Without this a
        closure rule could accept an outline that reversed and came back (seen on the
        instrument, 2026-10-06: enclosing 0.2, and the inside scan then found room for one
        line or none)."""
        if trim and len(f.legs) == 1 and len(f.legs[0]) >= 8:
            L = np.asarray(f.legs[0], dtype=float)
            c = L.mean(axis=0)
            ang = np.unwrap(np.arctan2(L[:, 1] - c[1], L[:, 0] - c[0]))
            k = np.flatnonzero(np.abs(ang - ang[0]) >= 2 * np.pi)
            if len(k):
                del f.legs[0][int(k[0]) + 1:]
        f.enclosed = float(self._encloses(f, f._ordered()))
        return f.enclosed >= self.cfg.loop_closure_min_enclosed

    def _border_gap(self, p) -> float:
        """Distance from p to the border of the area to explore."""
        x0, x1, y0, y1 = self._bounds
        return float(min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]))

    def _close_at_border(self, f: ExplorerFeature) -> bool:
        """A feature cut by the area's border: both directions of the trace ended at the
        border (within border_close_distance_V of it -- a zigzag can't cross an edge
        right at the border). The border between those two exits then closes the
        outline -- the way round the border that has the feature's inside on the inside
        of the outline (judged from the into-the-feature direction at every crossing).
        Sets f.border_closure; True if it closed."""
        if len(f.legs) < 2 or not f.legs[0] or not f.legs[1] or len(f.points) < 3:
            return False
        lo, hi = self._follow_limits()
        tol = self.cfg.border_close_distance_V
        tol = max(lo, 0.6 * hi) if tol is None else tol
        x0, x1, y0, y1 = self._bounds
        e0, e1 = np.asarray(f.legs[0][-1], dtype=float), np.asarray(f.legs[1][-1], dtype=float)

        def gap(e):
            return min(e[0] - x0, x1 - e[0], e[1] - y0, y1 - e[1])
        if gap(e0) > tol or gap(e1) > tol:
            return False
        if np.linalg.norm(e0 - e1) <= tol:
            # both directions left the area at the same place: the outline is already
            # round the feature -- no stretch of border to add, just check it encloses it
            if self._encloses(f, f._ordered()) < 0.5:
                return False
            f.closed_by_border = True
            return True
        # the border as a loop, counter-clockwise from (x0, y0); s = distance along it
        W, H = x1 - x0, y1 - y0
        per = 2 * (W + H)
        corners = [(0.0, (x0, y0)), (W, (x1, y0)), (W + H, (x1, y1)), (2 * W + H, (x0, y1))]

        def proj(e):     # nearest border point and its s
            cands = [(e[1] - y0, (e[0], y0), e[0] - x0), (x1 - e[0], (x1, e[1]), W + e[1] - y0),
                     (y1 - e[1], (e[0], y1), W + H + x1 - e[0]), (e[0] - x0, (x0, e[1]), 2 * W + H + y1 - e[1])]
            d, q, s = min(cands, key=lambda c: c[0])
            return np.array(q, dtype=float), s % per

        def along(sa, sb, ccw):   # border corners passed going from s=sa to s=sb
            span = (sb - sa) % per if ccw else (sa - sb) % per
            out = []
            for sc, xy in corners:
                ds = (sc - sa) % per if ccw else (sa - sc) % per
                if 1e-12 < ds < span:
                    out.append((ds, np.array(xy, dtype=float)))
            return [xy for _, xy in sorted(out, key=lambda t: t[0])]
        q0, s0 = proj(e0)
        q1, s1 = proj(e1)
        base = list(f._ordered())     # e1 ... seed ... e0 (border_closure still empty)
        P = f.pts()
        D = np.asarray(f.dirs, dtype=float).reshape(-1, 2)
        u = 0.5 * lo * D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        best = None
        for ccw in (True, False):
            closure = [q0] + along(s0, s1, ccw) + [q1]
            poly = _Polygon(np.asarray(base + closure))
            # right way round: just inside every crossing is INSIDE the outline and just
            # outside it (the substrate) is OUTSIDE -- the wrong way round encloses most
            # of the area, the substrate side included
            score = poly.contains_points(P + u).mean() - poly.contains_points(P - u).mean()
            if best is None or score > best[0]:
                best = (score, closure)
        if best[0] < 0.5:
            return False   # neither way round puts the feature inside: not a feature cut by the border
        f.border_closure = best[1]
        f.closed_by_border = True
        return True

    def _ends_meet(self, f: ExplorerFeature, end0, end1, Lf) -> bool:
        """After BOTH directions: the second one stopped (lost the edge, e.g. at a sharp
        corner) within one long step of where the first one ended, and the outline
        already covers loop_closure_min_coverage_deg -- only that last corner is
        missing, the loop is closed."""
        _, hi = self._follow_limits()
        return bool(np.linalg.norm(np.asarray(end1) - np.asarray(end0)) <= max(hi, 2 * Lf)
                    and f.angular_coverage() >= np.deg2rad(self.cfg.loop_closure_min_coverage_deg))

    def _reacquire(self, f: ExplorerFeature, p, n_in):
        """Find the edge near p again: a straight probe across it, from outside
        (p - a n_in) to inside (p + a n_in), a = follow_reacquire_V; doubled, up to
        follow_reacquire_tries probes, while it crosses nothing new. Returns (the crossing
        nearest p, where the tip ended) or None. Without this, a restart a fixed 0.09 V
        outside the stored point started on the wrong side of the edge whenever the
        piezo had shifted it more than that (seen on the instrument, 2026-10-05) -- every
        follow step then stayed on one side, and the edge was lost at once."""
        cfg = self.cfg
        lo = self._follow_limits()[0]
        a = cfg.follow_reacquire_V if cfg.follow_reacquire_V is not None else max(2 * lo, 0.5)
        p = np.asarray(p, dtype=float)
        n_in = _unit(n_in)
        for _ in range(max(1, cfg.follow_reacquire_tries)):
            a0 = self._fit(p, -n_in, a)
            a1 = self._fit(p, n_in, a)
            if a0 + a1 < 0.2 * lo:
                return None
            x_vec, y_vec = self._line(p - n_in * a0, p + n_in * a1)
            label = (f"Edge explorer segment {len(self.segments) + 1}: finding the edge of feature {f.id} "
                     f"again -- probe {a0 + a1:.3g} V across it")
            crossings = self._do_path(x_vec, y_vec, "follow", label)
            f.follow_segments += 1
            for c in crossings:
                c.inside_of = f.inside_of
            # the crossing ENTERING the feature the expected way (into ~ n_in): a probe
            # through a narrow feature also leaves it on the far side, and following from
            # that one would go the same way round as the other direction
            ok = [c for c in crossings if not self._known(c.p, f, f.inside_of) and np.dot(c.into, n_in) > 0]
            self._enqueue([c for c in crossings if not any(c is m for m in ok)], exclude=f)
            tip = np.array([x_vec[-1], y_vec[-1]])
            if ok:
                return min(ok, key=lambda c: np.linalg.norm(c.p - p)), tip
            a *= 2
        return None

    def _other_open_feature(self, f: ExplorerFeature, crossings) -> Optional[ExplorerFeature]:
        """The open (finished, not closed) feature whose traced edge one of these
        crossings lies on -- same parent (inside_of) as f."""
        r = self._follow_limits()[0]
        for c in crossings:
            for g in self.features:
                if (g is not f and g.finished and not g.closed_loop and g.inside_of == f.inside_of
                        and g.distance(c.p) < r):
                    return g
        return None

    def _merge(self, f: ExplorerFeature, g: ExplorerFeature, leg: int, leg_pts, goal, Lf) -> bool:
        """f's trace ran into g's edge: one edge, traced from two places. g's crossings
        join f, its outline is appended to this leg (oriented from where they met), and
        g is dropped. Returns True if the joined outline closes the loop: its far end
        is back at `goal` (where this leg is to close) after going round the feature."""
        seq = [np.asarray(q, dtype=float) for q in g._ordered()]
        if seq and np.linalg.norm(seq[-1] - leg_pts[-1]) < np.linalg.norm(seq[0] - leg_pts[-1]):
            seq = seq[::-1]           # start with g's end where the two met
        f.points.extend(g.points)
        f.dirs.extend(g.dirs)
        leg_pts.extend(seq)
        f.hit_border = f.hit_border or g.hit_border
        f.merged_ids.append(g.id)
        self.features = [h for h in self.features if h is not g]
        if not seq:
            return False
        _, hi = self._follow_limits()
        return bool(np.linalg.norm(leg_pts[-1] - np.asarray(goal)) <= max(hi, 2 * Lf)
                    and f.angular_coverage() >= np.deg2rad(self.cfg.loop_closure_min_coverage_deg))

    def _tangent(self, pts, t_prev, Lf) -> np.ndarray:
        """Edge direction from the last few crossings (principal direction), pointing
        the way we're going."""
        P = np.asarray(pts[-4:], dtype=float)
        if len(P) < 2 or np.ptp(P, axis=0).max() < 0.1 * Lf:
            return t_prev
        _, _, vt = np.linalg.svd(P - P.mean(axis=0), full_matrices=False)
        t = vt[0]
        return t if np.dot(t, t_prev) >= 0 else -t

    def _closes(self, f, leg, leg_pts, goal, Lf) -> bool:
        """The trace got back to `goal` -- where the edge was first found (first
        direction), or where the first direction ended (second direction) -- after
        going round the feature: its points cover loop_closure_min_coverage_deg round
        their centre. (Coverage alone isn't enough: a U round one end of a long feature
        already covers ~300 deg.)"""
        cfg = self.cfg
        if len(f.points) < cfg.loop_closure_min_points or len(leg_pts) < 3:
            return False
        if self._lapped(f, leg_pts, Lf):
            return True
        if leg == 1 and len(f.legs) > 1 and f.angular_coverage() >= np.deg2rad(cfg.loop_closure_min_coverage_deg):
            # the other direction ran onto the far half of the first direction's crossings
            # (it missed `goal`, the first direction's end, by a little and kept going
            # along the stretch already traced)
            P0 = np.asarray(f.legs[0], dtype=float)
            far = P0[len(P0) // 2:]
            if len(far) and np.min(np.linalg.norm(far - np.asarray(leg_pts[-1]), axis=1)) < \
                    max(self._follow_limits()[0], 0.75 * Lf):
                return True
        # the stretch of edge between the last two crossings passes right by the goal
        # (both lie on the edge) -- a tolerance tied to the SHORTEST step, so a big step
        # along a long side can't "reach" a goal across a narrow feature
        a, b = np.asarray(leg_pts[-2]), np.asarray(leg_pts[-1])
        ab = b - a
        u = np.clip(np.dot(goal - a, ab) / max(np.dot(ab, ab), 1e-24), 0.0, 1.0)
        if np.linalg.norm(a + u * ab - goal) >= 0.5 * self._follow_limits()[0]:
            return False
        return f.angular_coverage() >= np.deg2rad(cfg.loop_closure_min_coverage_deg)

    # ------------------------------------------------------------------
    # fast closure (fast_closure): turned round + back at the start of the trace
    # ------------------------------------------------------------------
    def _turned(self, f) -> float:
        """How far the edge direction has turned along the trace (radians): the signed
        turn of the follower's own edge-direction estimate (from the last 4 crossings --
        smooth, so zigzag jitter cancels instead of adding up), per direction, added up.
        2 pi once round any closed shape, convex or not; ~pi round one end of a bar."""
        return float(abs(f.turn[0]) + abs(f.turn[1]))

    def _start_points(self, f, leg, leg_pts) -> np.ndarray:
        """The OTHER end of the traced outline -- closing the loop means the two ends meet:
        leg 0: the first 3 crossings of the trace (never the last 4, right behind the
        tip); leg 1 (which started at leg 0's start and goes the other way): the last 3
        crossings of leg 0."""
        if leg == 0:
            P = np.asarray(leg_pts, dtype=float).reshape(-1, 2)
            # not the seed (P[0]): it was found on another path -- the initial line or an
            # exploring one, with a different piezo history -- and can sit a few tenths of a
            # V off the edge as the trace itself sees it; aiming the closure at it made the
            # trace miss, turn and go back round (simulation with piezo memory, 2026-10-06)
            if len(P) > 5:
                P = P[1:]
            if self.cfg.follow_style == "weave" and len(P) > 1:
                # dense crossings: the first 3 are only a few tenths of a V of edge -- take
                # the first four wavelengths' worth (never the last two waveforms' crossings)
                arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
                k = max(3, int(np.searchsorted(arc, 4 * self._weave_wavelength())))
                recent = int(4 * max(self.cfg.follow_weave_cycles, 1))
                return P[:min(k, max(len(P) - recent, 0))]
            return P[:min(3, max(len(P) - 4, 0))]
        P0 = np.asarray(f.legs[0], dtype=float).reshape(-1, 2)
        return P0[-3:] if len(P0) > 1 else P0[:0]

    def _encloses(self, f, outline) -> float:
        """How well `outline` (closed by joining its ends) is the feature's outline: the
        fraction of crossings with the point just inside the edge INSIDE it, minus the
        fraction with the point just outside (the substrate) inside it. ~1 for a trace
        that went round the feature; low for one that ran along part of it (or round one
        end), whose joined-up ends cut through the feature."""
        O = np.asarray(outline, dtype=float).reshape(-1, 2)
        P = f.pts()
        if len(O) < 3 or len(P) < 3:
            return 0.0
        D = np.asarray(f.dirs, dtype=float).reshape(-1, 2)
        u = 0.5 * self._follow_limits()[0] * D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        poly = _Polygon(O)
        return float(poly.contains_points(P + u).mean() - poly.contains_points(P - u).mean())

    def _outline_length(self, f) -> float:
        """Path length of the traced outline so far (both directions)."""
        total = 0.0
        for pts in f.legs:
            P = np.asarray(pts, dtype=float).reshape(-1, 2)
            if len(P) > 1:
                total += float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
        return total

    def _close_tol(self, Lf) -> float:
        # tied to the SHORTEST step, not the current one: a long step along a straight
        # side mustn't make the tolerance wide enough to reach across a thin feature
        return max(self.cfg.loop_closure_tolerance_V, 0.5 * self._follow_limits()[0])

    def _back_at_start(self, f, leg, leg_pts, mine, Lf) -> bool:
        """Closed: the two ends of the traced outline meet -- the segment just measured
        (an edge crossing on it) passes within the tolerance of the
        outline's other end -- the outline is at least 5 shortest steps long, and joined
        up it encloses the feature (_encloses >= loop_closure_min_enclosed)."""
        if len(f.points) < self.cfg.loop_closure_min_points:
            return False
        # a short, interrupted trace (one direction lost at a corner right away) has its
        # two ends close together without having gone round anything
        if self._outline_length(f) < 5 * self._follow_limits()[0]:
            return False
        S = self._start_points(f, leg, leg_pts)
        if not len(S):
            return False
        tol = self._close_tol(Lf)
        # an edge CROSSING of the segment just measured -- not just any point of it: a
        # segment can cut across the feature right past the start without the trace
        # having come round
        Q = np.asarray(self.segments[-1].edge_points, dtype=float).reshape(-1, 2)
        if not len(Q):
            return False
        d = np.min(np.linalg.norm(Q[:, None, :] - S[None, :, :], axis=2))
        if d >= tol:
            return False
        # ... and the outline, joined up, really goes round the feature
        return self._encloses(f, f._ordered()) >= self.cfg.loop_closure_min_enclosed

    def _steer_target(self, f, leg, leg_pts, start_p, tip, Lf):
        """Nearly round (the outline, joined up, already encloses the feature) and its
        other end is within two steps: aim the next segment straight across it (a little
        past, as always), instead of stepping along the edge and passing it by."""
        if len(f.points) < self.cfg.loop_closure_min_points:
            return None
        if self._outline_length(f) < 5 * self._follow_limits()[0]:
            return None
        if self._encloses(f, f._ordered()) < self.cfg.loop_closure_min_enclosed:
            return None
        # nearly round = the gap left between the outline's two ends is small next to the
        # outline traced so far (two sides of a square joined up "enclose" too)
        S = self._start_points(f, leg, leg_pts)
        ends_gap = np.min(np.linalg.norm(S - np.asarray(leg_pts[-1], dtype=float), axis=1)) if len(S) else np.inf
        if ends_gap > 0.25 * self._outline_length(f):
            return None
        S = self._start_points(f, leg, leg_pts)
        if not len(S):
            return None
        p = np.asarray(leg_pts[-1], dtype=float)
        dist = np.linalg.norm(S - p, axis=1)
        j = int(np.argmin(dist))
        if dist[j] > 2.0 * Lf + self._close_tol(Lf):
            return None
        # the nearest of the trace's own first crossings (not the seed: measured on another
        # path, it can be off the edge as this trace sees it -- see _start_points)
        return np.asarray(S[j], dtype=float)

    def _lapped(self, f, leg_pts, Lf) -> bool:
        """Fallback closure, for when the trace never passes close enough to `goal`
        (seen on the instrument: 120 segments round and round one feature). Closed once
        the edge has been gone all the way round -- the crossings cover the full
        loop_closure_min_coverage_deg -- AND the newest crossing is back next to one
        from earlier in this leg (not the last few, which are always close); or, in any
        case, once the trace has wound more than 1.25 turns round the feature."""
        cfg = self.cfg
        P = np.asarray(leg_pts, dtype=float)
        if len(P) < 8:
            return False
        c = f.pts().mean(axis=0)
        ang = np.unwrap(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
        if abs(ang[-1] - ang[0]) >= 2.5 * np.pi:
            return True
        if f.angular_coverage() < np.deg2rad(cfg.loop_closure_min_coverage_deg):
            return False
        earlier = P[:-4]
        # back where it started, a turn later: near an earlier crossing, having wound most
        # of the way round since then
        d = np.linalg.norm(earlier - P[-1], axis=1)
        near = np.flatnonzero(d < max(self._follow_limits()[0], 0.75 * Lf))
        return bool(len(near) and abs(ang[-1] - ang[near[0]]) >= 1.5 * np.pi)

    def _mark_inside(self, f: ExplorerFeature) -> None:
        if len(f.points) < 3 or not self.INSIDE_IS_KNOWN:
            return
        GX, GY = np.meshgrid(self._gx, self._gy)
        inside = _Polygon(f._ordered()).contains_points(np.column_stack([GX.ravel(), GY.ravel()]))
        self._measured |= inside.reshape(GX.shape)

    # ------------------------------------------------------------------
    # INTERIOR: parallel lines across a closed feature's inside (one waveform)
    # ------------------------------------------------------------------
    def _interior_lines(self, f: ExplorerFeature):
        """[(start, end), ...] of the interior scan lines, in scan order (back and forth):
        along the feature's long axis, interior_line_spacing_V apart, each kept
        interior_margin_V inside the traced outline."""
        cfg = self.cfg
        P = f._ordered()
        if len(P) < 3:
            return []
        lo = self._follow_limits()[0]
        spacing = cfg.interior_line_spacing_V if cfg.interior_line_spacing_V is not None else 1.4 * self._follow_L()
        margin = cfg.interior_margin_V if cfg.interior_margin_V is not None else 0.5 * lo
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        u, v = vt[0], np.array([-vt[0][1], vt[0][0]])        # long axis, across
        a, b = (P - c) @ u, (P - c) @ v
        width = (b.max() - b.min()) - 2 * margin
        if width < 0:
            return []
        n = int(np.floor(width / spacing)) + 1
        if n > cfg.interior_max_lines:
            n = cfg.interior_max_lines
            spacing = width / max(n - 1, 1)
        b_mid = 0.5 * (b.max() + b.min())
        levels = b_mid + (np.arange(n) - 0.5 * (n - 1)) * spacing
        poly = _Polygon(np.vstack([P, P[:1]]))
        seg_a, seg_b = P, np.roll(P, -1, axis=0)                # outline segments
        ab = seg_b - seg_a
        ab2 = np.maximum(np.einsum("mk,mk->m", ab, ab), 1e-24)
        step = max(min(spacing, margin if margin > 0 else spacing, lo) / 4, 1e-4)
        s = np.arange(a.min(), a.max() + step, step)
        lines = []
        for bk in levels:
            pts = c + s[:, None] * u + bk * v
            ok = poly.contains_points(pts)
            if margin > 0 and ok.any():
                # distance from each sample to the outline: stay `margin` inside it
                t = np.clip(np.einsum("nmk,mk->nm", pts[:, None, :] - seg_a[None], ab) / ab2, 0, 1)
                d = np.linalg.norm(seg_a[None] + t[..., None] * ab[None] - pts[:, None, :], axis=2).min(axis=1)
                ok &= d >= margin
            if not ok.any():
                continue
            # the longest run of samples inside
            runs = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))).reshape(-1, 2)
            i0, i1 = max(runs, key=lambda r: r[1] - r[0])
            if (i1 - 1 - i0) * step < margin:
                continue
            p0, p1 = pts[i0], pts[i1 - 1]
            lines.append((p0, p1) if len(lines) % 2 == 0 else (p1, p0))
        return lines

    def _scan_interior(self, f: ExplorerFeature, tip):
        """Scan a little of a closed feature's inside -- parallel lines, one waveform:
        measures its top (for the reconstruction) and finds edges inside it (a hole, an
        opening down to the substrate), which are queued and followed like any other."""
        lines = self._interior_lines(f)
        if not lines and f.legs:
            # a ragged / self-crossing outline (an edge re-found, two traces merged): try
            # the outline of its points ordered by angle round their centre instead
            legs, f.legs = f.legs, []
            lines = self._interior_lines(f)
            f.legs = legs
        if not lines:
            lo = self._follow_limits()[0]
            margin = self.cfg.interior_margin_V if self.cfg.interior_margin_V is not None else 0.5 * lo
            f.interior_note = (f"inside NOT scanned: no line fits {margin:.3g} V (interior_margin_V) inside "
                               f"the traced outline -- too narrow, or a ragged/self-crossing outline")
            return tip
        xs, ys = [], []
        for k, (p0, p1) in enumerate(lines):
            if k:  # the short move to the next line, measured too
                x, y = self._line(lines[k - 1][1], p0)
                xs.append(x[1:]); ys.append(y[1:])
            x, y = self._line(p0, p1)
            xs.append(x if k == 0 else x[1:]); ys.append(y if k == 0 else y[1:])
        x_vec, y_vec = np.concatenate(xs), np.concatenate(ys)
        label = (f"Edge explorer segment {len(self.segments) + 1}: scanning the inside of feature {f.id} "
                 f"({len(lines)} lines)")
        crossings = self._do_path(x_vec, y_vec, "interior", label)
        f.interior_lines = len(lines)
        for c in crossings:
            c.inside_of = f.id
        self._enqueue(crossings)
        return np.array([x_vec[-1], y_vec[-1]])

    # ------------------------------------------------------------------
    # EXPLORE: the most unexplored direction
    # ------------------------------------------------------------------
    def _nearest_unexplored(self, pos) -> Optional[np.ndarray]:
        """The nearest point well away (>= half the explore scale) from everything measured,
        or None if no such area is left."""
        U = self._unexplored()
        r, c = np.nonzero(U >= 0.5)
        if not len(r):
            return None
        P = np.column_stack([self._gx[c], self._gy[r]])
        return P[int(np.argmin(np.linalg.norm(P - np.asarray(pos), axis=1)))]

    def _unexplored(self) -> np.ndarray:
        n = self.cfg.grid_n
        if not self._measured.any():
            return np.ones((n, n))
        dist = distance_transform_edt(~self._measured) * self._cell
        return np.clip(dist / self._explore_scale(), 0.0, 1.0)

    def _choose(self, pos, heading):
        cfg = self.cfg
        unexplored = self._unexplored()
        x0, x1, y0, y1 = self._bounds
        lim = self._scan_limit()
        L = self._L
        look = cfg.explore_lookahead_V if cfg.explore_lookahead_V is not None else 4 * L
        s_look = np.linspace(0.0, look, max(4, int(np.ceil(look / self._cell)) + 1))[1:]
        angles = np.linspace(0, 2 * np.pi, cfg.n_candidate_directions, endpoint=False)
        h0 = np.arctan2(heading[1], heading[0])
        best = None
        for ang in angles:
            if cfg.max_turn_deg is not None:
                turn = abs((ang - h0 + np.pi) % (2 * np.pi) - np.pi)
                if turn > np.deg2rad(cfg.max_turn_deg):
                    continue
            d = np.array([np.cos(ang), np.sin(ang)])
            end = pos + d * L
            if not (x0 <= end[0] <= x1 and y0 <= end[1] <= y1) or np.any(np.abs(end) > lim):
                continue
            # look further ahead than one segment (everything within one short
            # segment looks equally unexplored), and slightly prefer going straight --
            # long sweeps instead of a random walk
            px, py = pos[0] + s_look * d[0], pos[1] + s_look * d[1]
            rr, cc = self._rc(px, py)
            inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            explore = float(np.mean(np.where(inside, unexplored[rr, cc], 0.0)))
            straight = 0.5 * (1 + np.cos(ang - h0))
            score = explore + cfg.straight_bonus * straight * (explore > 0.05)
            if best is None or score > best[1]:
                best = (d, score, explore)
        if best is None or best[2] < cfg.min_score:
            return None
        return best[0], best[1]
