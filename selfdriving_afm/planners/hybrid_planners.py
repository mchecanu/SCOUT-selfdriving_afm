"""
Two more combinations of SCOUT levels (2026-11-10), next to gp_edge_explorer (GP + tracker):

GridTrackerPlanner  ("grid_tracker"): level-0 sparse spiral of grid_refine (one smooth
    Archimedean spiral over the whole area, from the calibration circle outward, turns
    SPARSE_SPACING_V apart) -> every outline it crossed is followed by the edge explorer's
    tracer (Fig. 3: weave, closes the loop, interior lines). After the spiral, nothing more is
    explored: crossings in the spiral data that were never traced are re-checked with the final
    substrate model, then it stops.

GPSpiralPlanner  ("gp_spiral"): GP (IVR) exploration between objects as in gp_edge_explorer;
    when an exploring segment crosses an outline, grid_refine's refinement takes over instead of
    the tracer: one coarse spiral (level 1) round the object's expected centre, then one dense
    spiral (level 2) centred on the object measured by it, out to its half-size + a pad. The
    object's outline (from the dense spiral) counts as a closed feature -- its crossings are
    known from then on -- and the GP explores again.

Every move is driven (no jumps): the spirals start where the tip is (or a measured straight line
leads to their start), and every waveform is at most CHUNK_V of path (<= 0.5 s at 30 V/s: the
PyScanner AI record is complete for those).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .. import trajectory
from .edge_explorer_planner import ExplorerFeature, _Crossing, _unit
from .gp_edge_explorer_planner import GPEdgeExplorerPlanner

CHUNK_V = 14.0   # one waveform: at most this much path (0.47 s at 30 V/s)


def _spiral(center, r0, r1, spacing, ds, start_angle=0.0):
    """Archimedean spiral from radius r0 to r1 (turns `spacing` apart, constant speed), rotated so
    it starts at angle start_angle."""
    n_cycles = max((r1 - r0) / max(spacing, 1e-9), 0.5)
    x, y, _, _ = trajectory.spiral_out((0.0, 0.0), r0, r1, n_cycles, ds)
    c, s = np.cos(start_angle), np.sin(start_angle)
    return center[0] + c * x - s * y, center[1] + s * x + c * y


class _ChunkedPaths:
    """_do_path for long paths, in waveforms of at most CHUNK_V of path, each starting where the
    previous one ended (the edge state carries over)."""

    def _do_long(self, x, y, mode, label):
        ds = self.executor.velocity.sample_spacing_V
        n = max(2, int(CHUNK_V / ds))
        out: List[_Crossing] = []
        k, i0 = 0, 0
        N = len(x)
        nchunks = int(np.ceil((N - 1) / n))
        while i0 < N - 1:
            i1 = min(i0 + n, N - 1)
            k += 1
            out += self._do_path(x[i0:i1 + 1], y[i0:i1 + 1], mode, f"{label} ({k}/{nchunks})")
            i0 = i1
        return out


class GridTrackerPlanner(_ChunkedPaths, GPEdgeExplorerPlanner):
    SPARSE_SPACING_V = 1.07     # level-0 spiral: turns this far apart (2.0 um; objects >= 3 um)
    SPARSE_RADIUS_V = None      # None = the area's half-width (the inscribed circle)

    def run(self, start_xy=(0.0, 0.0)):
        self._sparse_done = False
        return super().run(start_xy)

    def _do_segment(self, pos, direction, length, mode, label):
        if mode != "initial" or self._sparse_done:
            return super()._do_segment(pos, direction, length, mode, label)
        # the initial straight path -> the level-0 sparse spiral, from the calibration circle
        # (which ends at radius r_cal, angle initial_direction) outward
        self._sparse_done = True
        cfg = self.cfg
        c0 = np.array(cfg.explore_center_V, float)
        R = self.SPARSE_RADIUS_V or cfg.explore_extent_V
        R = min(R, self._scan_limit() - float(np.max(np.abs(c0))))
        pos = np.asarray(pos, float)
        r0 = float(np.linalg.norm(pos - c0))
        a0 = float(np.arctan2(pos[1] - c0[1], pos[0] - c0[0])) if r0 > 1e-6 else 0.0
        r0 = max(r0, 0.3)
        x, y = _spiral(c0, r0, R, self.SPARSE_SPACING_V, self.executor.velocity.sample_spacing_V, a0)
        if np.linalg.norm(np.array([x[0], y[0]]) - pos) > 1e-6:      # (only without a calibration circle)
            lx, ly = self._line(pos, (x[0], y[0]))
            x, y = np.concatenate([lx, x[1:]]), np.concatenate([ly, y[1:]])
        L = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        print(f"[grid_tracker] level 0: sparse spiral, radius {R:.2f} V, turns {self.SPARSE_SPACING_V:.2f} V apart, "
              f"{L:.0f} V of path")
        crossings = self._do_long(x, y, "initial", "Level 0: sparse spiral")
        self.sparse_crossings = len(crossings)
        print(f"[grid_tracker] level 0 done: {len(crossings)} outline crossings -> the tracer")
        return np.array([x[-1], y[-1]]), crossings

    def _choose(self, pos, heading):
        # no exploring after level 0: only crossings in its data that were never traced
        self.choices.append("recheck")
        c = self._recheck(pos)
        if c is None:
            return None
        d = c.p - np.asarray(pos, float)
        L = float(np.linalg.norm(d))
        return (d / L if L > 1e-9 else np.array([1.0, 0.0])), float("inf")

    def _nearest_unexplored(self, pos):
        return None                 # nothing to explore: stop once the queue is empty


class GPSpiralPlanner(_ChunkedPaths, GPEdgeExplorerPlanner):
    """GP between objects, level-1 -> level-2 spirals on each (v6: the first good instrument run,
    2026-11-09 23:19 -- 5/5 objects mapped, 34 % of the pixels on objects vs 8 % for its raster;
    v1-v5 failed on the instrument for the reasons noted at each step below).
    0. background: a robust degree-2 surface (objects trimmed out, BG_TRIM) refitted from all
       exploring data before every exploring path -- the piezo bow (~300 nm over 26 um) made a
       plane read whole regions as objects (v3, v4).
    1. GP exploring in short steps (GP_STEP_V) so a detected edge is not overshot; when quiet,
       the tip goes to the nearest hole an object of the smallest size could hide in.
    2. trigger: the crossing's feature side >= TRIGGER_FACTOR x the edge threshold AND a height
       step along the exploring path (bow / tilt are smooth). Path length from the sample index
       (the sensor positions' noise inflates a cumulative length ~100x, v5).
    3. edge probe: a circle (PROBE_R_V) round the crossing; it must show a step; the plane
       through it tilts up the step -> the edge normal. The drive's error depends on where the
       tip is, so the circle is re-centred on where it actually landed and measured again (v5).
    4. level 1: coarse spiral centred one expected radius INTO the object along that normal,
       radius half the largest feature + L1_PAD_V; heights above a plane through the spiral's
       own substrate. The object = a mean-shift window one feature wide (no merging of
       neighbours, v5), checked (size within the priors, height). Repeated once, re-centred, if
       it reaches the level-1 rim.
    5. level 2: dense spiral centred on the level-1 object, slightly bigger than it (L2_PAD_V).
       Closed when the object stays inside the rim in >= CLOSED_FRAC of the angles AND its size
       and height are an object's; else extended outward (only while it is not too big).
    Known limit (v6): the plane inside a level-2 disc can tilt (2 of 5 outlines partial; the
    data itself covers the objects) -- fitting it on the disc's outer ring would fix it.
    Every decision goes to LOG_PATH too (the live view replaces the cell output)."""
    GP_STEP_V = 0.5
    TRIGGER_FACTOR = 1.1        # crossing / probe: x the edge threshold (instrument crossings read 59-79 nm)
    OBJECT_HEIGHT_FACTOR = 1.3  # level-1 / level-2 region: median height >= this x the threshold
    PROBE_R_V = 0.5
    PROBE_ON_RANGE = (0.12, 0.88)
    L1_SPACING_V = 0.30
    L1_PAD_V = 0.80
    L2_SPACING_V = 0.06
    L2_PAD_V = 0.40
    L2_RADIUS_LIMITS = (0.9, 2.4)
    EXTEND_V = 0.6
    MAX_EXTENSIONS = 2
    SIZE_RANGE = (0.5, 1.5)     # object across, in units of (feature_min, feature_max)
    MIN_SPIRAL_R0 = 0.15
    N_ANGLE_BINS = 72
    CLOSED_FRAC = 0.85
    FEATURE_MAX_V = 2.67        # set from the priors by modes.build (priors=...)
    FEATURE_MIN_V = 1.6
    LOG_PATH = None             # a text file: every decision appended (the live view replaces the cell output)

    @property
    def _L(self) -> float:
        return float(self.GP_STEP_V)

    # ---- the GP trains on the dense spirals only every SPIRAL_TRAIN_SPACING_V (they add
    #      hundreds of thousands of points otherwise: 1500 s of GP fits in simulation)
    SPIRAL_TRAIN_SPACING_V = 0.15
    # ---- holes: an object of the smallest size fits where no measurement is within half of
    #      it; after HOLE_AFTER_SWEEPS of quiet GP exploring, the tip heads to the nearest hole
    HOLE_CELL_V = 0.2
    HOLE_AFTER_SWEEPS = 0.5

    # ---- the global substrate: a ROBUST degree-2 surface (piezo bow: -250..+50 nm over the area on
    #      the instrument; a plane fitted to 'off' samples never learns where the bow lifts the
    #      background over the threshold -- whole regions read 'on', objects inside them give no
    #      crossing). Refitted before every exploring path from all exploring data so far,
    #      objects trimmed out (the samples farthest from the surface, BG_TRIM of them).
    BG_TRIM = 0.3
    BG_THIN_V = 0.05
    BG_MIN_SAMPLES = 300

    def _refit_background(self):
        model = self.substrate
        if model.coef is None:
            return
        xs, ys, zs = [], [], []
        for sg in self.segments:
            if sg.mode not in ("initial", "explore", "calibration"):
                continue
            n = min(len(sg.x), len(sg.y), len(sg.z))
            st = max(1, int(round(self.BG_THIN_V / self.executor.velocity.sample_spacing_V)))
            xs.append(np.asarray(sg.x[:n:st], float)); ys.append(np.asarray(sg.y[:n:st], float))
            zs.append(np.asarray(sg.z[:n:st], float))
        if not xs:
            return
        x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
        if len(z) < self.BG_MIN_SAMPLES:
            return
        x0, x1, y0, y1 = self._bounds
        cover = min(np.ptp(x) / (x1 - x0), np.ptp(y) / (y1 - y0))
        deg = 2 if cover >= 0.6 else 1
        cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if deg == 2 else [])
        A = np.column_stack(cols)
        keep = np.ones(len(z), bool)
        for _ in range(6):
            c, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
            r = z - A @ c
            a = np.abs(r - np.median(r[keep]))
            keep = a <= np.quantile(a, 1.0 - self.BG_TRIM)
        c, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        coef = np.zeros(6); coef[:len(c)] = c
        model.degree = 2
        model.coef = coef
        self.bg_fits = getattr(self, "bg_fits", 0) + 1

    def _do_path(self, x_vec, y_vec, mode, label):
        if mode in ("explore", "initial") and self.substrate.coef is not None:
            if not getattr(self, "_bg_frozen", False):
                # the model's own update (normal equations of 'off' samples) would undo the robust
                # fit: from now on only the robust refit moves the surface (threshold: manual)
                self.substrate.update = lambda *a, **k: None
                self._bg_frozen = True
            self._refit_background()
        n_before = len(self._gpX)
        crossings = super()._do_path(x_vec, y_vec, mode, label)
        sg = self.segments[-1]
        if mode in ("follow", "interior") and len(self._gpX) > n_before:
            stride = max(1, int(round(self.SPIRAL_TRAIN_SPACING_V / self.executor.velocity.sample_spacing_V)))
            n = min(len(sg.x), len(sg.y), len(sg.z))
            self._gpX[-1] = np.column_stack([sg.x[:n:stride], sg.y[:n:stride]])
            self._gpZ[-1] = np.asarray(sg.z[:n:stride], float)
        self._cover(sg.x, sg.y)
        return crossings

    def _cover(self, x, y):
        x0, x1, y0, y1 = self._bounds
        c = self.HOLE_CELL_V
        if getattr(self, "_cov", None) is None:
            self._cov = np.zeros((int(np.ceil((y1 - y0) / c)), int(np.ceil((x1 - x0) / c))), bool)
        n = min(len(x), len(y))
        st = max(1, int(0.5 * c / self.executor.velocity.sample_spacing_V))
        i = ((np.asarray(x[:n:st]) - x0) / c).astype(int); j = ((np.asarray(y[:n:st]) - y0) / c).astype(int)
        k = (i >= 0) & (j >= 0) & (i < self._cov.shape[1]) & (j < self._cov.shape[0])
        self._cov[j[k], i[k]] = True

    def _nearest_hole(self, pos):
        from scipy import ndimage
        if getattr(self, "_cov", None) is None:
            return None
        c = self.HOLE_CELL_V
        cov = self._cov.copy()
        x0, x1, y0, y1 = self._bounds
        gx = x0 + (np.arange(cov.shape[1]) + 0.5) * c; gy = y0 + (np.arange(cov.shape[0]) + 0.5) * c
        GX, GY = np.meshgrid(gx, gy)
        for cen, R in self.discs:
            cov |= np.hypot(GX - cen[0], GY - cen[1]) < R
        dist = ndimage.distance_transform_edt(~cov) * c
        hole = dist > 0.5 * float(self.FEATURE_MIN_V)
        if not hole.any():
            return None
        d = np.hypot(GX - pos[0], GY - pos[1])
        k = np.argmin(np.where(hole, d, np.inf))
        return np.array([GX.flat[k], GY.flat[k]]), float(dist.flat[k])

    def _choose(self, pos, heading):
        x0, x1, y0, y1 = self._bounds
        sweep_V = (x1 - x0) * (y1 - y0) / (self.cfg.explore_scale_V or self._L)
        if getattr(self, "quiet_path_V", 0.0) >= self.HOLE_AFTER_SWEEPS * sweep_V:
            h = self._nearest_hole(pos)
            if h is not None:
                p, r = h
                d = p - np.asarray(pos, float)
                L = float(np.linalg.norm(d))
                if L > 1e-9:
                    if getattr(self, "_hole_target", None) is None or np.linalg.norm(self._hole_target - p) > 0.5:
                        self._log(f"quiet: heading to the unexplored hole at ({p[0]:+.2f}, {p[1]:+.2f}) V "
                                  f"({r:.2f} V from any measurement)")
                    self._hole_target = p
                    self.choices.append("hole")
                    return d / L, float("inf")
        c = super()._choose(pos, heading)
        if c is None:
            # the map says nothing is left within reach: a hole an object fits in still counts
            h = self._nearest_hole(pos)
            if h is not None:
                p, r = h
                d = p - np.asarray(pos, float)
                L = float(np.linalg.norm(d))
                if L > 1e-9:
                    self._log(f"map done; heading to the unexplored hole at ({p[0]:+.2f}, {p[1]:+.2f}) V ({r:.2f} V from any measurement)")
                    self.choices.append("hole")
                    return d / L, float("inf")
        return c

    def run(self, start_xy=(0.0, 0.0)):
        self.discs = []             # (centre, radius) of every level-2 spiral: known area
        self.rejected = []          # crossings that are not an object: not re-queued
        self.log_lines = []
        self._cov = None; self._hole_target = None; self._bg_frozen = False; self._resid_off = None
        self._log(f"GP step {self._L:.2f} V, trigger x{self.TRIGGER_FACTOR}, features "
                  f"{self.FEATURE_MIN_V:.2f}-{self.FEATURE_MAX_V:.2f} V")
        return super().run(start_xy)

    def _log(self, msg):
        msg = "[gp_spiral] " + msg
        print(msg)
        self.log_lines.append(msg)
        if self.LOG_PATH:
            try:
                with open(self.LOG_PATH, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass

    def _known(self, p, exclude=None, inside_of=None) -> bool:
        p = np.asarray(p, float)
        if any(np.linalg.norm(p - c) < 0.95 * R for c, R in getattr(self, "discs", [])):
            return True
        if any(np.linalg.norm(p - q) < self.REJECT_RADIUS_V for q in getattr(self, "rejected", [])):
            return True
        return super()._known(p, exclude, inside_of)

    def _heights(self, x, y, z):
        return np.abs(self.substrate.height(x, y, z))

    LOCAL_BG_FRACTION = 0.3     # local substrate: plane through the lowest this fraction of a spiral

    def _local_heights(self, x, y, z):
        """Heights above a plane fitted to THIS spiral's own substrate (robust: refitted to the
        lowest LOCAL_BG_FRACTION of the residuals). The global substrate plane is off near the
        area's edges on the instrument (piezo bow: regions 4 V wide read 'on', 200+ nm tall);
        a disc 4 V wide is flat enough for a plane."""
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        st = max(1, len(z) // 20000)
        xs, ys, zs = x[::st], y[::st], z[::st]
        A = np.column_stack([np.ones_like(xs), xs, ys])
        # the objects' side: the global model's sign of the tall samples (default up)
        hg = self.substrate.height(xs, ys, zs)
        tall = np.abs(hg) > self.substrate.threshold
        sgn = 1.0 if not tall.any() or np.median(hg[tall]) >= 0 else -1.0
        zz = sgn * zs
        keep = np.ones(len(zz), bool)
        for _ in range(5):
            coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
            r = zz - A @ coef
            keep = r <= np.quantile(r, self.LOCAL_BG_FRACTION)
        coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
        r0 = float(np.median((zz - A @ coef)[keep]))
        return sgn * z - (coef[0] + coef[1] * x + coef[2] * y) - r0

    def _lead_in(self, pos, x, y):
        """A measured straight line from pos to the path's start, then the path."""
        lx, ly = self._line(pos, (x[0], y[0]))
        return np.concatenate([lx, x[1:]]), np.concatenate([ly, y[1:]])

    # the executor's drive-offset correction leaves a residual (instrument v5: ~0.5 V -- a
    # 0.35 V probe circle never reached its edge). Every probe / spiral is commanded shifted by
    # the residual measured on the previous ones (commanded - measured, median), so the tip
    # (measured) lands where it should.
    OFFSET_GAIN = 0.7
    OFFSET_MAX_V = 2.0

    def _measure(self, pos, x, y, mode, label):
        off = getattr(self, "_resid_off", None)
        off = np.zeros(2) if off is None else off
        lim = self._scan_limit()
        x = np.clip(np.asarray(x, float) + off[0], -lim, lim)
        y = np.clip(np.asarray(y, float) + off[1], -lim, lim)
        x, y = self._lead_in(pos, x, y)
        n0 = len(self.segments)
        self._do_long(x, y, mode, label)
        segs = self.segments[n0:]
        X = np.concatenate([s.x for s in segs]); Y = np.concatenate([s.y for s in segs])
        Z = np.concatenate([s.z for s in segs])
        n = min(len(X), len(Y), len(Z), len(x))
        if n > 100:
            k = slice(n // 5, n)            # past the lead-in's start
            e = np.array([np.median(x[k] - X[k]), np.median(y[k] - Y[k])])
            # x = desired + off, X = x - E (the drive's error): e = x - X = E; the shift needed is E
            new = (1 - self.OFFSET_GAIN) * off + self.OFFSET_GAIN * e
            if np.linalg.norm(new) > self.OFFSET_MAX_V:
                new = new / np.linalg.norm(new) * self.OFFSET_MAX_V
            if np.linalg.norm(new - off) > 0.05:
                self._log(f"drive residual {np.linalg.norm(e):.2f} V -> spirals / probes shifted by ({new[0]:+.2f}, {new[1]:+.2f}) V")
            self._resid_off = new
        n = min(len(X), len(Y), len(Z))
        return np.array([x[-1], y[-1]]), X[:n], Y[:n], Z[:n]

    def _spiral_at(self, pos, c, R, spacing, mode, label, r0=None):
        ds = self.executor.velocity.sample_spacing_V
        lim = self._scan_limit()
        R = float(min(R, lim - np.max(np.abs(c))))
        if r0 is None:
            r0 = min(max(spacing, self.MIN_SPIRAL_R0), 0.5 * R)
        a0 = float(np.arctan2(pos[1] - c[1], pos[0] - c[0])) if r0 > 0.5 else 0.0
        x, y = _spiral(c, r0, R, spacing, ds, start_angle=a0)
        return self._measure(pos, x, y, mode, label)

    def _circle_at(self, pos, c, r, label):
        """A circle round c: (pos, X, Y, Z) of the CIRCLE's samples only (not the lead-in)."""
        ds = self.executor.velocity.sample_spacing_V
        a0 = float(np.arctan2(pos[1] - c[1], pos[0] - c[0]))
        n = max(16, int(2 * np.pi * r / ds))
        a = a0 + np.linspace(0, 2 * np.pi, n)
        pos, X, Y, Z = self._measure(pos, c[0] + r * np.cos(a), c[1] + r * np.sin(a), "follow", label)
        k = min(n, len(Z))
        return pos, X[-k:], Y[-k:], Z[-k:]

    PROBE_RECENTER_V = 0.12

    def _probe(self, pos, p, label):
        """The edge probe round p. The drive's error depends on where the tip is (v5: the probe
        circles landed ~0.3 V off, missing the edge): the circle's measured centre is compared
        with p, and the circle is measured again shifted by the miss (once)."""
        c = np.asarray(p, float).copy()
        for attempt in range(2):
            pos, X, Y, Z = self._circle_at(pos, c, self.PROBE_R_V, label)
            if len(X) < 50:
                return pos, X, Y, Z
            miss = np.array([np.mean(X), np.mean(Y)]) - p
            if np.linalg.norm(miss) < self.PROBE_RECENTER_V or attempt == 1:
                if attempt == 1 or np.linalg.norm(miss) >= self.PROBE_RECENTER_V:
                    self._log(f"probe at ({p[0]:+.2f}, {p[1]:+.2f}): landed {np.linalg.norm(miss):.2f} V off"
                              + (" after re-centring" if attempt else ""))
                return pos, X, Y, Z
            self._log(f"probe at ({p[0]:+.2f}, {p[1]:+.2f}): landed {np.linalg.norm(miss):.2f} V off -> again, shifted")
            c = c - miss
        return pos, X, Y, Z

    @staticmethod
    def _image(X, Y, v, c, R, cell):
        """Per-cell (on-majority, measured, mean height) images round c (square 2R)."""
        on, h = v
        n = int(np.ceil(2 * R / cell)) + 1
        i = np.clip(((X - c[0] + R) / cell).astype(int), 0, n - 1)
        j = np.clip(((Y - c[1] + R) / cell).astype(int), 0, n - 1)
        cnt = np.zeros((n, n)); hit = np.zeros((n, n)); hs = np.zeros((n, n))
        np.add.at(cnt, (j, i), 1); np.add.at(hit, (j, i), on.astype(float)); np.add.at(hs, (j, i), h)
        meas = cnt > 0
        ON = meas & (hit > 0.5 * np.maximum(cnt, 1))
        H = hs / np.maximum(cnt, 1)
        g = (c[0] - R + (np.arange(n) + 0.5) * cell, c[1] - R + (np.arange(n) + 0.5) * cell)
        return ON, meas, H, g

    @staticmethod
    def _pick(lab, gx, gy, p):
        """Label of the component nearest point p (0: none)."""
        if lab.max() == 0:
            return 0
        jj, ii = np.nonzero(lab)
        k = int(np.argmin(np.hypot(gx[ii] - p[0], gy[jj] - p[1])))
        return int(lab[jj[k], ii[k]])

    REJECT_RADIUS_V = 0.4       # no new trigger this close to a rejected crossing (v6 on the instrument)
    PATH_STEP_HALF_V = 0.5      # the along-path step check: this much path either side of the crossing

    def _path_step(self, p):
        """Height step across the crossing along the exploring paths through p: the spread
        (5-95%) of z about a straight line over +-PATH_STEP_HALF_V of path (by sample index:
        constant speed). Every pass of every exploring path within 0.03 V of p is tried (the
        calibration circle and the initial line both pass some points) -- the largest step
        counts. A real edge is a step (~ the object's height); bow / tilt are smooth (a few nm).
        None: no pass found."""
        p = np.asarray(p, float)
        ds = self.executor.velocity.sample_spacing_V
        w = int(round(self.PATH_STEP_HALF_V / ds))
        best = None
        for sg in self.segments:
            if sg.mode not in ("initial", "explore", "calibration"):
                continue
            x, y, z = np.asarray(sg.x), np.asarray(sg.y), np.asarray(sg.z)
            n = min(len(x), len(y), len(z))
            if n < 10:
                continue
            x, y, z = x[:n], y[:n], z[:n]
            d = np.hypot(x - p[0], y - p[1])
            near = np.flatnonzero(d < 0.03)
            if not len(near):
                continue
            # one candidate per pass: runs of nearby samples, split where the index jumps
            runs = np.split(near, np.flatnonzero(np.diff(near) > max(10, w // 10)) + 1)
            for run in runs:
                i = int(run[np.argmin(d[run])])
                j0, j1 = max(i - w, 0), min(i + w + 1, n)
                if j1 - j0 < 20:
                    continue
                st = max(1, (j1 - j0) // 4000)
                idx = np.arange(j0, j1, st)
                A = np.column_stack([np.ones(len(idx)), idx * ds])
                coef, *_ = np.linalg.lstsq(A, z[idx], rcond=None)
                r = z[idx] - A @ coef
                q5, q95 = np.quantile(r, [0.05, 0.95])
                v = float(q95 - q5)
                best = v if best is None else max(best, v)
        return best

    def _size_ok(self, across):
        lo, hi = self.SIZE_RANGE
        return lo * self.FEATURE_MIN_V <= across <= hi * self.FEATURE_MAX_V

    def _reject(self, f, p, why, disc=None):
        self._log(f"object {f.id}: {why} -- not an object")
        f.edge_lost = True; f.finished = True; f.closed_loop = False; f.closed_by = why
        self.rejected.append(np.asarray(p, float))
        if disc is not None:
            self.discs.append(disc)

    def _level1(self, pos, f, c1, R1, seed_p):
        """Level-1 spiral round c1: (pos, centre, across, reaches the rim) of the connected
        on-region nearest seed_p, or (pos, None, why, False)."""
        from scipy import ndimage
        thr = float(self.substrate.threshold)
        pos, X, Y, Z = self._spiral_at(pos, c1, R1, self.L1_SPACING_V, "follow",
                                       f"Level 1: coarse spiral on object {f.id}")
        hl = self._local_heights(X, Y, Z)
        on = hl > thr
        cell = 0.1
        ON, meas, H, (gx, gy) = self._image(X, Y, (on, hl), c1, R1, cell)
        # ONE object: mean-shift from the expected centre with a window one largest-object wide
        # (v5: a connected region bridged two neighbours 1 V apart into one '3.9 V' object)
        GX, GY = np.meshgrid(gx, gy)
        win = 0.5 * float(self.FEATURE_MAX_V) + 0.2
        c = np.asarray(c1, float).copy()
        comp = np.zeros_like(ON)
        for _ in range(6):
            comp = ON & (np.hypot(GX - c[0], GY - c[1]) < win)
            if comp.sum() < 6:
                break
            jj, ii = np.nonzero(comp)
            cn = 0.5 * np.array([gx[ii].min() + gx[ii].max(), gy[jj].min() + gy[jj].max()])
            if np.linalg.norm(cn - c) < 0.03:
                c = cn
                break
            c = cn
        if comp.sum() < 6:
            return pos, None, "nothing above the threshold on level 1", False
        jj, ii = np.nonzero(comp)
        lo = np.array([gx[ii].min(), gy[jj].min()]) - 0.5 * self.L1_SPACING_V
        hi = np.array([gx[ii].max(), gy[jj].max()]) + 0.5 * self.L1_SPACING_V
        cen = 0.5 * (lo + hi)
        across = float(np.max(hi - lo))
        hmed = float(np.median(H[comp]))
        rim = bool(np.any(np.hypot(gx[ii] - c1[0], gy[jj] - c1[1]) > R1 - 0.25))
        self._log(f"object {f.id}: level 1 -> centre ({cen[0]:+.2f}, {cen[1]:+.2f}) V, {across:.2f} V across, "
                  f"median height {hmed:.0f}, reaches the rim: {rim}")
        if hmed < self.OBJECT_HEIGHT_FACTOR * thr:
            return pos, None, f"level 1: median height {hmed:.0f} < {self.OBJECT_HEIGHT_FACTOR * thr:.0f}", False
        return pos, cen, across, rim

    def _follow(self, seed, pos):
        thr = float(self.substrate.threshold)
        into = _unit(seed.into)
        trig = self.TRIGGER_FACTOR * thr
        if not np.isfinite(seed.height) or seed.height < trig:
            self._log(f"crossing at ({seed.p[0]:+.2f}, {seed.p[1]:+.2f}) V: height {seed.height:.0f} < {trig:.0f} -- no spiral")
            self.rejected.append(np.asarray(seed.p, float))
            return pos, into
        step0 = self._path_step(seed.p)
        if step0 is not None and step0 < self.TRIGGER_FACTOR * thr:
            self._log(f"crossing at ({seed.p[0]:+.2f}, {seed.p[1]:+.2f}) V: step along the path {step0:.0f} < "
                      f"{self.TRIGGER_FACTOR * thr:.0f} (smooth background) -- no spiral")
            self.rejected.append(np.asarray(seed.p, float))
            return pos, into
        f = ExplorerFeature(id=self._next_id, points=[seed.p], dirs=[seed.into])
        self._next_id += 1
        self.features.append(f)
        fmax, fmin = float(self.FEATURE_MAX_V), float(self.FEATURE_MIN_V)
        # ---- edge probe: which way is the object's inside (normal), is it really an edge
        pos, Xr, Yr, Zr = self._probe(pos, seed.p, f"Edge probe on object {f.id}")
        # a real edge is a STEP of about the object's height across the ring; the background
        # (bow, tilt) is smooth: residuals from a plane through the ring itself tell them apart
        if len(Zr) < 50:
            self._reject(f, seed.p, "edge probe: no data")
            return pos, into
        hg = self.substrate.height(Xr, Yr, Zr)
        tall = np.abs(hg) > thr
        sgn = 1.0 if not tall.any() or np.median(hg[tall]) >= 0 else -1.0
        A = np.column_stack([np.ones_like(Xr), Xr, Yr])
        coef, *_ = np.linalg.lstsq(A, sgn * Zr, rcond=None)
        r = sgn * Zr - A @ coef
        q5, q95 = np.quantile(r, [0.05, 0.95])
        step = float(q95 - q5)
        if step < self.TRIGGER_FACTOR * thr:
            self._reject(f, seed.p, f"edge probe at ({seed.p[0]:+.2f}, {seed.p[1]:+.2f}): step {step:.0f} < {self.TRIGGER_FACTOR * thr:.0f} (smooth background)")
            return pos, into
        # the inside: up the step. A plane through a ring half on the object tilts toward the
        # high side (step / ring diameter ~ 150 nm/V >> the bow's few tens): its gradient
        onr = r > 0.5 * (q5 + q95)
        frac = float(onr.mean())
        gvec = np.array([coef[1], coef[2]])
        n = _unit(gvec) if np.linalg.norm(gvec) > 1e-9 else into
        Rexp = 0.25 * (fmin + fmax)   # expected half-size
        c1 = seed.p + n * Rexp
        R1 = 0.5 * fmax + self.L1_PAD_V
        self._log(f"object {f.id}: crossing {seed.height:.0f} at ({seed.p[0]:+.2f}, {seed.p[1]:+.2f}); probe step {step:.0f}, {frac:.0%} on, "
                  f"normal ({n[0]:+.2f}, {n[1]:+.2f}) -> level 1 round ({c1[0]:+.2f}, {c1[1]:+.2f}), R {R1:.2f} V")
        # ---- level 1 (repeated once, re-centred, if the object reaches its rim)
        pos, cen, across, rim = self._level1(pos, f, c1, R1, seed.p + n * 0.15)
        if cen is not None and rim:
            pos, cen2, across2, rim2 = self._level1(pos, f, cen, R1, cen)
            if cen2 is not None:
                cen, across, rim = cen2, across2, rim2
        if cen is None:
            self._reject(f, seed.p, across, disc=(c1, 0.6 * R1))
            return pos, n
        for g in self.features:
            if g is not f and g.closed_loop and len(g.points) >= 3:
                gc = np.mean(np.asarray(g.points, float), axis=0)
                if np.linalg.norm(cen - gc) < 0.5 * float(self.FEATURE_MIN_V):
                    self._reject(f, seed.p, f"level 1: same object as object {g.id} (closed already)")
                    return pos, n
        if not self._size_ok(across) and across > self.SIZE_RANGE[1] * fmax:
            self._reject(f, seed.p, f"level 1: {across:.2f} V across, bigger than any feature", disc=(cen, 0.6 * R1))
            return pos, n
        # ---- level 2: dense spiral on the level-1 object, slightly bigger than it
        R2 = float(np.clip(0.5 * across + self.L2_PAD_V, *self.L2_RADIUS_LIMITS))
        self._log(f"object {f.id}: level 2 round ({cen[0]:+.2f}, {cen[1]:+.2f}), R {R2:.2f} V")
        pos, X, Y, Z = self._spiral_at(pos, cen, R2, self.L2_SPACING_V, "interior",
                                       f"Level 2: dense spiral on object {f.id}")
        ext = 0
        while True:
            res = self._closed_outline(X, Y, Z, cen, R2)
            too_big = 2 * res["half"] > self.SIZE_RANGE[1] * fmax
            if res["closed"] or ext >= self.MAX_EXTENSIONS or too_big:
                break
            ext += 1
            R3 = R2 + self.EXTEND_V
            self._log(f"object {f.id}: outline open ({res['frac']:.0%} closed) -> extend the disc to {R3:.2f} V")
            pos, X3, Y3, Z3 = self._spiral_at(pos, cen, R3, self.L2_SPACING_V, "interior",
                                              f"Level 2: extend disc on object {f.id}", r0=R2 - 0.1)
            X, Y, Z = np.r_[X, X3], np.r_[Y, Y3], np.r_[Z, Z3]
            R2 = R3
        E = res["outline"]
        f.points = [p for p in E] if len(E) else [seed.p]
        f.dirs = [_unit(cen - p) for p in f.points]
        f.finished = True
        f.interior_lines = 0
        f.follow_segments = 3 + ext
        self.discs.append((cen, R2))
        across2 = res["across"]
        if not self._size_ok(across2):
            f.closed_loop = False
            f.closed_by = f"level 2: {across2:.2f} V across -- not an object's size"
        elif res["hmed"] < self.OBJECT_HEIGHT_FACTOR * thr:
            f.closed_loop = False
            f.closed_by = f"level 2: median height {res['hmed']:.0f} < {self.OBJECT_HEIGHT_FACTOR * thr:.0f} -- not an object"
        elif res["closed"]:
            f.closed_loop = True
            f.closed_by = f"level-2 spiral ({res['frac']:.0%} of the outline inside the disc)"
            self._mark_inside(f)
        else:
            f.closed_loop = False
            f.edge_lost = True
            f.closed_by = f"level-2 spiral: outline open ({res['frac']:.0%} of the angles closed inside the disc)"
        self._log(f"object {f.id}: {across2:.2f} V across, median height {res['hmed']:.0f}, {res['frac']:.0%} closed "
                  f"(R {R2:.2f} V) -> {'CLOSED' if f.closed_loop else f.closed_by}")
        self._prev_on = None
        return pos, n

    def _closed_outline(self, X, Y, Z, cen, R):
        """The object round cen in the dense data: the connected on-region (small gaps closed,
        holes filled, noise opened away) nearest the centre; per angle bin its outermost cell;
        closed bins: those that stay inside the rim."""
        from scipy import ndimage
        h = self._local_heights(X, Y, Z)
        on = h > float(self.substrate.threshold)
        cell = 0.05
        ON, meas, H, (gx, gy) = self._image(X, Y, (on, h), cen, R, cell)
        ON = ndimage.binary_closing(ON, iterations=1)
        ON = ndimage.binary_fill_holes(ON)
        ON = ndimage.binary_opening(ON, iterations=1)
        lab, _ = ndimage.label(ON)
        k = self._pick(lab, gx, gy, cen)
        nb = self.N_ANGLE_BINS
        if k == 0:
            return dict(closed=False, frac=0.0, half=0.0, across=0.0, hmed=0.0, outline=np.zeros((0, 2)))
        comp = lab == k
        jj, ii = np.nonzero(comp)
        px, py = gx[ii] - cen[0], gy[jj] - cen[1]
        r = np.hypot(px, py); th = np.arctan2(py, px)
        kb = np.clip(((th + np.pi) / (2 * np.pi) * nb).astype(int), 0, nb - 1)
        rim = R - 0.15
        E, ok = [], 0
        for b in range(nb):
            m = kb == b
            rmax = float(r[m].max()) if m.any() else 0.0
            if rmax < rim:
                ok += 1
            a = -np.pi + (b + 0.5) * 2 * np.pi / nb
            E.append(cen + max(rmax, cell) * np.array([np.cos(a), np.sin(a)]))
        frac = ok / nb
        across = float(max(np.ptp(gx[ii]), np.ptp(gy[jj])) + cell)
        hm = comp & meas
        hmed = float(np.median(H[hm])) if hm.any() else 0.0
        return dict(closed=frac >= self.CLOSED_FRAC, frac=frac, half=float(r.max()), across=across,
                    hmed=hmed, outline=np.array(E))
