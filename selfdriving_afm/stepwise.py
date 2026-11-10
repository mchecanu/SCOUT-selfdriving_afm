"""
Step-by-step grid_refine, levels 0-1-2, SPIRALS ONLY -- the workflow that made the good
grid_refine dataset (2026-11-03, tag 20261103_135746). Run from Run_GridRefine_Stepwise.ipynb:

    run = StepwiseRun()
    run.setup(); run.level0(); run.level1(); run.level2()
    run.save_dataset(); run.figures(); run.animate()
    run.equivalent_raster(); run.compare_raster()       # only when you decide to

Every scan is ONE PyScanner do_scan of a pre-defined spiral (spiral_return_opt 1: out and
smoothly back in), at 1 MS/s (data_rate 0). No PyScanner fast raster (its slow axis steps
instantly at every line) and no Lissajous. The tip is never jumped: every move is a driven
cosine waveform from where the outputs REALLY are (the backend's no-jump guard reads
get_AO0_V/AO1_V before and after every output, waiting for the lagging readback). With no
jumps the sample did not drift between scans (2026-11-03).

    level 0: one sparse spiral over the whole area: r 7.15 V, 6 turns out + 6 back in, 10 s
    level 1: one spiral per object found on the sparse arcs (> 30 nm): r 2 V round each
             object's box, <= 4 s
    level 2: one dense spiral per object, centred on the object measured in level 1 (with
             level 1's real-vs-commanded size and offset corrected): object + 0.35 V, turns
             at h1 / 3, <= 11.5 s

The equivalent raster (user_raster: user-defined smooth waveforms, same number of pixels,
every band starting and ending at rest) is added to the same .npz.

`Stepwise` is the lower-level helper (one planner step at a time, with figures); the
figures are in AR um (UM_PER_V / UM_AT_0 below: the sensor calibration of 2026-11-02).
"""
import contextlib
import gc
import json
import pathlib
import time

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from . import modes, priors, asylum
from . import pyscanner_scans as pss
from .planners.grid_refine_planner import ScanRect, RefineLevel
from .reconstruction import bin_image, inpaint_binned

UM_PER_V = (-1.79, 1.95)        # drive V -> AR um (AR orientation: x flipped) -- figures
UM_AT_0 = (15.1, 15.0)          # AR um at drive (0, 0)


def um(x, y):
    return UM_AT_0[0] + UM_PER_V[0] * np.asarray(x, float), UM_AT_0[1] + UM_PER_V[1] * np.asarray(y, float)


def ext_um(b):
    ux, uy = um(np.array(b[:2]), np.array(b[2:]))
    return (ux[0], ux[1], uy[0], uy[1])   # imshow extent: left = x0 (larger um, x flipped)


def _ram_GB():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return float("nan")


class Stepwise:
    """grid_refine one step at a time, with the user looking at each step before the next:

        st = Stepwise(session, afm)
        st.sparse()                 # the coarse spiral (one PyScanner do_scan) -> figure
        st.propose()                # detection on what was measured -> figure with the proposed regions
        st.approve(keep=None)       # (optionally only some regions) -> the level's scans queued
        st.next_scan()              # one region's PyScanner scan -> figure; repeat until st.queue is empty
        st.end_level()              # -> st.propose() for the next level

    Every step is its own afm.run block: engaged for the scan, then withdrawn and smoothly
    back to (0, 0) while the user looks. The planner's own code does the detection and builds
    the scans (the same as an uninterrupted run); this only splits run() into steps."""

    def __init__(self, session, afm, out_dir="scan_data", tag=None):
        self.s, self.afm = session, afm
        self.pl, self.ex, self.cfg = session.planner, session.executor, session.planner.cfg
        self.out_dir = out_dir
        self.tag = tag or time.strftime("%Y%m%d_%H%M%S")
        self.k = 0
        self.pending = None
        self.queue = []
        self.done = []
        self.level = None
        self.figs = []

    # ------------------------------------------------------------------
    def _xyz(self, steps, sensor=True):
        if sensor and all(s.x_sensor is not None for s in steps):
            x = np.concatenate([s.x_sensor for s in steps]); y = np.concatenate([s.y_sensor for s in steps])
        else:
            x = np.concatenate([s.x for s in steps]); y = np.concatenate([s.y for s in steps])
        z = np.concatenate([s.z for s in steps])
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        return x[ok], y[ok], z[ok]

    def _save(self, fig, name):
        p = f"{self.out_dir}/step_{self.tag}_{name}.png"
        fig.savefig(p, dpi=80)
        self.figs.append(p)
        print("figure:", p)
        return p

    @staticmethod
    def _flat(img):
        """One plane through the lower 60% of the heights removed (substrate ~ 0)."""
        n0, n1 = img.shape
        X, Y = np.meshgrid(np.arange(n1), np.arange(n0))
        m = np.isfinite(img)
        if m.sum() < 10:
            return img
        m &= img < np.nanpercentile(img, 60)
        A = np.column_stack([np.ones(m.sum()), X[m], Y[m]])
        co = np.linalg.lstsq(A, img[m], rcond=None)[0]
        q = img - (co[0] + co[1] * X + co[2] * Y)
        return q - np.nanmedian(q[m])

    # ------------------------------------------------------------------
    def sparse(self):
        """The planner's run() up to and including the sparse scan."""
        pl, ex, cfg = self.pl, self.ex, self.cfg
        ex.reset(self.s.cfg.execution.start_xy_V)
        ex.connector_ease_samples = pl._ease_n()
        pl.levels = []
        xl, xh, yl, yh = pl._lims()
        cx = float(np.clip(cfg.center_V[0], xl, xh))
        cy = float(np.clip(cfg.center_V[1], yl, yh))
        h0 = 2 * cfg.half_size_V / max(cfg.sparse_n_lines - 1, 1)
        margin = max(0.5 * h0, cfg.raster_turn_radius_V or 0.0) + 0.5 * pl._ease_n() * pl._ds
        H = min(cfg.half_size_V, cx - xl - margin, xh - cx - margin, cy - yl - margin, yh - cy - margin)
        pl.bounds = (cx - H, cx + H, cy - H, cy + H)
        S = 2 * H
        self.h = S / max(cfg.sparse_n_lines - 1, 1)
        self.h_min = cfg.min_spacing_V if cfg.min_spacing_V is not None else S / 255
        n_pix = cfg.detect_pixels or int(np.clip(np.ceil(2 * S / self.h_min) + 1, 128, 512))
        pl._n_pix, pl._pix = n_pix, S / (n_pix - 1)
        whole = ScanRect(*pl.bounds, fast_axis="x")
        b = pl.bounds
        print(f"area x {b[0]:+.3f}..{b[1]:+.3f}, y {b[2]:+.3f}..{b[3]:+.3f} V = "
              f"{abs(S * UM_PER_V[0]):.1f} x {abs(S * UM_PER_V[1]):.1f} um (AR x {min(ext_um(b)[:2]):.1f}..{max(ext_um(b)[:2]):.1f}, "
              f"y {ext_um(b)[2]:.1f}..{ext_um(b)[3]:.1f} um)")
        n0 = len(ex.history)
        t0 = time.time()
        with self.afm.run("sparse spiral", ex):
            pl._scan_level(0, self.h, [whole], cfg.sparse_pattern, plan_time=0.0, sparse=True)
        print(f"sparse scan done in {time.time() - t0:.0f} s")
        self.allowed = [whole]
        self.sparse_steps = (n0, len(ex.history))
        return self.show_sparse()

    def show_sparse(self):
        steps = self.ex.history[self.sparse_steps[0]:self.sparse_steps[1]]
        steps = [s for s in steps if s.kind == "grid_sparse"]
        for s in steps:
            print(" ", s.meta.get("label", ""))
        xs, ys, z = self._xyz(steps, sensor=True)
        xc, yc, _ = self._xyz(steps, sensor=False)
        unit = getattr(self.ex, "z_unit", "")
        b = self.pl.bounds
        pad = 1.0
        bb = (b[0] - pad, b[1] + pad, b[2] - pad, b[3] + pad)
        fig = Figure(figsize=(17, 5.6), layout="constrained")
        axs = fig.subplots(1, 3)
        ux, uy = um(xs, ys)
        cux, cuy = um(xc, yc)
        zz = z - np.nanpercentile(z, 30)
        lo, hi = np.nanpercentile(zz, [1, 99.5])
        a = axs[0]
        a.plot(cux, cuy, color="0.75", lw=0.4, label="commanded")
        sc = a.scatter(ux[::5], uy[::5], c=zz[::5], s=1, cmap="copper", vmin=lo, vmax=hi)
        fig.colorbar(sc, ax=a, label=f"Z ({unit}, offset removed)", shrink=0.8)
        a.set_title("the spiral: measured positions coloured by Z\n(grey: commanded path)", fontsize=10)
        img, _ = bin_image(np.column_stack([xs, ys]), z, 200, bb)
        img = self._flat(img)
        e = ext_um(bb)
        h = axs[1].imshow(img, origin="lower", extent=e, cmap="copper", interpolation="nearest",
                          vmin=np.nanpercentile(img, 1), vmax=np.nanpercentile(img, 99.5))
        fig.colorbar(h, ax=axs[1], label=f"Z ({unit}, plane removed)", shrink=0.8)
        axs[1].set_title("measured pixels (sensor positions)", fontsize=10)
        # what the planner sees: its detection image (measured + reconstruction in the gaps)
        self.pl._update_maps(self.h)
        det = self._flat(np.array(self.pl.detect_image, dtype=float))
        h = axs[2].imshow(det, origin="lower", extent=ext_um(b), cmap="copper", interpolation="nearest",
                          vmin=np.nanpercentile(det, 1), vmax=np.nanpercentile(det, 99.5))
        fig.colorbar(h, ax=axs[2], label=f"Z ({unit}, plane removed)", shrink=0.8)
        axs[2].set_title(f"what detection sees: measured + '{self.cfg.detect_reconstruction}' fill "
                         f"({self.cfg.detect_on})", fontsize=10)
        for a in axs:
            a.add_patch(Rectangle((min(ext_um(b)[:2]), ext_um(b)[2]), abs(ext_um(b)[1] - ext_um(b)[0]),
                                  ext_um(b)[3] - ext_um(b)[2], fill=False, ec="c", lw=0.8, ls="--"))
            a.set_xlim(min(e[:2]), max(e[:2]))
            a.set_ylim(e[2], e[3])
            a.set_aspect("equal")
            a.set_xlabel("X (um, AR frame)"); a.set_ylabel("Y (um, AR frame)")
        fig.suptitle(f"Step 0 - coarse spiral: {len(z):,} stored samples; dashed: the grid_refine area")
        return self._save(fig, "0_sparse")

    # ------------------------------------------------------------------
    def propose(self):
        """Detection on what has been measured: the next level's regions (nothing moves)."""
        pl, cfg = self.pl, self.cfg
        k = self.k + 1
        if self.h <= self.h_min * (1 + 1e-9) or k > cfg.max_levels:
            print("no further level (finest spacing / max_levels reached)")
            return None
        h_next = max(self.h / cfg.refine_factor, self.h_min)
        t = time.perf_counter()
        pl._update_maps(self.h)
        rects, thr, noise = pl._plan_tiles(k, self.h, self.allowed, h_next)
        self.pending = (k, h_next, rects, thr, noise, time.perf_counter() - t)
        print(f"level {k}: lines {h_next:.3f} V (~{abs(h_next * UM_PER_V[0]) * 1e3:.0f} nm) apart; threshold "
              f"{thr if thr is None else round(thr, 3)} (noise {noise if noise is None else round(noise, 3)}); "
              f"{len(rects)} region(s)")
        for i, r in enumerate(rects):
            e = ext_um((r.x0, r.x1, r.y0, r.y1))
            scans = pl._pyscanner_scans(r, h_next, cfg.refine_pattern)
            print(f"  region {i + 1}: {r.pattern or cfg.refine_pattern}, x {r.x0:+.2f}..{r.x1:+.2f} y {r.y0:+.2f}..{r.y1:+.2f} V "
                  f"(AR x {min(e[:2]):.1f}..{max(e[:2]):.1f}, y {e[2]:.1f}..{e[3]:.1f} um); "
                  + "; ".join(s.describe() for s in scans))
        return self.show_proposal()

    def show_proposal(self):
        k, h_next, rects, thr, noise, _ = self.pending
        pl = self.pl
        b = pl.bounds
        unit = getattr(self.ex, "z_unit", "")
        fig = Figure(figsize=(17, 5.6), layout="constrained")
        axs = fig.subplots(1, 3)
        e = ext_um(b)
        det = self._flat(np.array(pl.detect_image, dtype=float))
        h = axs[0].imshow(det, origin="lower", extent=e, cmap="copper", interpolation="nearest",
                          vmin=np.nanpercentile(det, 1), vmax=np.nanpercentile(det, 99.5))
        fig.colorbar(h, ax=axs[0], label=f"Z ({unit})", shrink=0.8)
        axs[0].set_title("detection image", fontsize=10)
        rough = np.array(pl.roughness, dtype=float)
        h = axs[1].imshow(rough, origin="lower", extent=e, cmap="viridis", interpolation="nearest",
                          vmax=np.nanpercentile(rough, 99.5))
        fig.colorbar(h, ax=axs[1], label=f"local roughness ({unit})", shrink=0.8)
        if thr is not None:
            axs[1].contour(np.isfinite(rough) & (rough > thr), levels=[0.5], colors="r", linewidths=0.6,
                           extent=e, origin="lower")
        axs[1].set_title(f"roughness; red: above the threshold {thr:.3g}" if thr else "roughness", fontsize=10)
        axs[2].imshow(det, origin="lower", extent=e, cmap="copper", interpolation="nearest",
                      vmin=np.nanpercentile(det, 1), vmax=np.nanpercentile(det, 99.5))
        for i, r in enumerate(rects):
            for c in (r.cells if r.cells is not None else [(r.x0, r.x1, r.y0, r.y1)]):
                ce = ext_um(c)
                axs[2].add_patch(Rectangle((min(ce[:2]), ce[2]), abs(ce[1] - ce[0]), ce[3] - ce[2], fill=False,
                                           ec="y", lw=0.3, alpha=0.6))
            re = ext_um((r.x0, r.x1, r.y0, r.y1))
            axs[2].add_patch(Rectangle((min(re[:2]), re[2]), abs(re[1] - re[0]), re[3] - re[2], fill=False, ec="c", lw=1.2))
            axs[2].text(min(re[:2]), re[3], f" {i + 1}: {r.pattern or self.cfg.refine_pattern}", color="c",
                        fontsize=9, va="bottom")
        axs[2].set_title(f"proposed level {k}: {len(rects)} region(s) (cyan), their tiles (yellow)", fontsize=10)
        for a in axs:
            a.set_xlim(min(e[:2]), max(e[:2])); a.set_ylim(e[2], e[3]); a.set_aspect("equal")
            a.set_xlabel("X (um, AR frame)"); a.set_ylabel("Y (um, AR frame)")
        fig.suptitle(f"Level {k} proposal: lines {h_next:.3f} V apart")
        return self._save(fig, f"{k}_proposal")

    # ------------------------------------------------------------------
    def approve(self, keep=None):
        """Queue the proposed level's regions (keep: 1-based region numbers to keep; None = all)."""
        k, h_next, rects, thr, noise, pt = self.pending
        if keep is not None:
            rects = [rects[i - 1] for i in keep]
        order = self.pl._order(rects, np.zeros(2))     # each scan starts from (0, 0)
        S2 = (self.pl.bounds[1] - self.pl.bounds[0]) * (self.pl.bounds[3] - self.pl.bounds[2])
        cov = sum(r.size[0] * r.size[1] if (r.pattern or self.cfg.refine_pattern) == "raster" else r.area(self.cfg.refine_pattern)
                  for r in order)
        self.level = RefineLevel(k, h_next, self.cfg.refine_pattern, order, plan_time_s=pt, threshold=thr, noise=noise,
                                 area_fraction=min(cov / S2, 1.0))
        self.level.tile_size_V = getattr(self.pl, "_tile_size", None)
        self.level.n_interior_tiles = getattr(self.pl, "_n_interior", 0)
        self.pl.levels.append(self.level)
        self.queue = list(range(len(order)))
        self.done = []
        print(f"level {k}: {len(order)} region(s) queued in this order: "
              + ", ".join(f"{rects.index(r) + 1}" if r in rects else "?" for r in order))

    def next_scan(self):
        """One region's scan(s) (one PyScanner do_scan each), engaged; then the figure."""
        if not self.queue:
            print("nothing queued")
            return None
        i = self.queue.pop(0)
        lv, pl, ex = self.level, self.pl, self.ex
        r = lv.rects[i]
        scans = pl._pyscanner_scans(r, lv.spacing_V, self.cfg.refine_pattern)
        n0 = len(ex.history)
        ns0 = sum(len(s.x) for s in ex.history)
        name = f"Refinement level {lv.index}"
        with self.afm.run(f"level {lv.index} region {i + 1}", ex):
            for j, s in enumerate(scans):
                part = f" [scan {j + 1}/{len(scans)}]" if len(scans) > 1 else ""
                ex.send_pyscanner(s, "refine", meta={"label": f"{name} -- region {i + 1}/{len(lv.rects)}"
                                                              f"{pl._what(r)}{part}: {s.describe()}", "level": lv.index})
        lv.n_samples += sum(len(s.x) for s in ex.history) - ns0
        lv.scan_time_s = lv.n_samples / pl._rate
        self.done.append((i, n0, len(ex.history)))
        return self.show_scan(i, n0, len(ex.history))

    def show_scan(self, i, n0, n1):
        lv = self.level
        r = lv.rects[i]
        steps = [s for s in self.ex.history[n0:n1] if s.kind == "refine"]
        for s in steps:
            print(" ", s.meta.get("label", ""))
        xs, ys, z = self._xyz(steps, sensor=True)
        xc, yc, zc = self._xyz(steps, sensor=False)
        unit = getattr(self.ex, "z_unit", "")
        pad = 0.4
        bb = (r.x0 - pad, r.x1 + pad, r.y0 - pad, r.y1 + pad)
        w = max(bb[1] - bb[0], bb[3] - bb[2])
        nb = int(np.clip(w / (0.5 * lv.spacing_V), 60, 400))
        fig = Figure(figsize=(17, 5.6), layout="constrained")
        axs = fig.subplots(1, 3)
        e = ext_um(bb)
        imgs = []
        for x_, y_, z_ in ((xs, ys, z), (xc, yc, zc)):
            img, _ = bin_image(np.column_stack([x_, y_]), z_, nb, bb)
            imgs.append(self._flat(img))
        lo, hi = np.nanpercentile(imgs[0], [1, 99.5])
        for a, img, t in ((axs[0], imgs[0], "measured (sensor) positions"), (axs[1], imgs[1], "commanded positions")):
            h = a.imshow(img, origin="lower", extent=e, cmap="copper", interpolation="nearest", vmin=lo, vmax=hi)
            fig.colorbar(h, ax=a, label=f"Z ({unit}, plane removed)", shrink=0.8)
            a.set_title(t, fontsize=10)
        ux, uy = um(xs, ys)
        cux, cuy = um(xc, yc)
        axs[2].plot(cux, cuy, color="0.6", lw=0.4, label="commanded")
        axs[2].plot(ux, uy, color="C1", lw=0.4, alpha=0.7, label="measured")
        axs[2].legend(fontsize=8, loc="upper right")
        axs[2].set_title("path", fontsize=10)
        re = ext_um((r.x0, r.x1, r.y0, r.y1))
        for a in axs:
            a.add_patch(Rectangle((min(re[:2]), re[2]), abs(re[1] - re[0]), re[3] - re[2], fill=False, ec="c", lw=0.8, ls="--"))
            a.set_xlim(min(e[:2]), max(e[:2])); a.set_ylim(e[2], e[3]); a.set_aspect("equal")
            a.set_xlabel("X (um, AR frame)"); a.set_ylabel("Y (um, AR frame)")
        zr = np.nanpercentile(imgs[0], 99) - np.nanpercentile(imgs[0], 1)
        fig.suptitle(f"Level {lv.index}, region {i + 1}/{len(lv.rects)} ({r.pattern or lv.pattern}): "
                     f"{len(z):,} stored samples, Z range ~{zr:.0f} {unit}; dashed: the region")
        return self._save(fig, f"{lv.index}_region{i + 1}")

    def end_level(self):
        self.allowed = self.level.rects
        self.h = self.level.spacing_V
        self.k = self.level.index
        self.pending = None
        print(f"level {self.k} done: {self.level.scan_time_s:.1f} s of scanning")


# ======================================================================================
class _NoAFM:
    """Stands in for asylum.AsylumControl in a simulation: no engage / withdraw."""
    @contextlib.contextmanager
    def run(self, label="scan", executor=None):
        yield self


class StepwiseRun:
    """The 2026-11-03 working pipeline (levels 0-1-2, spirals only, no jumps), stage by stage.
    The settings below are the ones of the good dataset; change them on the instance before
    setup() (e.g. run.SPARSE = dict(...))."""
    MAX_SPAN_UM = 28.0                 # the scan range: 28 x 28 um ...
    UM_PER_V_LIMITS = (1.811, 1.956)   # ... at these um/V -> the X / Y safety limits
    V = 30.0                           # tip speed (V/s, peak) for level 1 and 2
    SPARSE = dict(center=(0.3, 0.0), radius_V=7.15, n_cycles=6, duration_s=10.0)
    L1_MAX_S, L2_MAX_S = 4.0, 11.5     # one do_scan each, at most this long
    Z_OBJ_NM = 30.0                    # "object" = this far above the substrate
    L2_PAD_V = 0.35                    # level-2 spiral radius = object half-size + this
    UM_PER_V_AVG = 1.85                # for printing only
    N_PIX = 256                        # the dataset's binned / reconstructed images

    def __init__(self, out_dir="scan_data", config_notebook="Run_SelfDriving_Scan.ipynb",
                 hardware_paths="hardware_paths.json"):
        self.out = pathlib.Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.config_notebook = config_notebook
        self.hardware_paths = hardware_paths
        self.PRIORS = priors.ScanPriors(feature_min_V=2.2, feature_max_V=4.5, area_half_width_V=7.0,
                                        area_center_V=(0.3, 0.0))

    # ------------------------------------------------------------------
    def say_guard(self, title):
        log = getattr(self.backend, "guard_log", None)
        if log is None:
            return
        print(f"guard log ({title}): every output's AO readback vs where it started / ended")
        for g in log:
            print("   ", {k: (tuple(round(v, 4) for v in val) if isinstance(val, tuple) else
                           (round(val, 4) if isinstance(val, float) else val)) for k, val in g.items()})
        big = [g for g in log if "(end)" not in g["what"] and g["gap_V"] > self.backend.guard_tol_V]
        print(f"    -> {len(big)} output(s) did NOT start where the outputs were (each was driven there first)")

    def _clear_guard(self):
        log = getattr(self.backend, "guard_log", None)
        if log is not None:
            log.clear()

    def configure(self, cfg, sample_rate_hz=1e6):
        """The good dataset's configuration (28 x 28 um, 1 MS/s, smooth spirals, RAM-safe)."""
        cfg.safety.max_amplitude_V = 0.5 * self.MAX_SPAN_UM / self.UM_PER_V_LIMITS[0]
        cfg.safety.max_amplitude_y_V = 0.5 * self.MAX_SPAN_UM / self.UM_PER_V_LIMITS[1]
        cfg.safety.clamp_mode = "raise"
        cfg.velocity.sample_rate_hz = sample_rate_hz
        cfg.velocity.tip_velocity_V_s = self.V
        cfg.position_sensors.correct_linear = False
        cfg.position_sensors.correct_offset = False
        cfg.position_sensors.initial_offset_V = None
        gr = cfg.grid_refine
        gr.drive_margin_V = (0.0, 0.0)
        priors.apply(cfg, self.PRIORS)
        gr.scan_generator = "pyscanner_defined"; gr.pyscanner_speed_basis = "peak"
        gr.sparse_pattern = "spiral"; gr.sparse_n_lines = 11; gr.sparse_spiral_radius_V = 8.0
        gr.refine_pattern = "auto"; gr.pyscanner_one_scan = True; gr.pyscanner_max_speed_V_s = self.V
        gr.region_max_V = 5.0; gr.raster_direction = "x"; gr.max_levels = 3; gr.min_spacing_V = 0.04
        gr.detect_on = "reconstruction"; gr.detect_reconstruction = "linear"
        cfg.live_view.n_image_pixels = self.N_PIX
        cfg.readback.store_every = 4
        cfg.readback.fast_raster_delay = "off"
        cfg.safety.max_samples_per_waveform = 11_000_000
        pss.SPIRAL_RETURN = 1          # out and back in, smooth -- never the fast straight return
        return cfg

    def setup(self):
        """Connect, configure, Asylum setup (routing, sensor calibration with the tip
        withdrawn, engage)."""
        from .backends.pyscanner_backend import connect, PyScannerBackend
        hw = json.loads(pathlib.Path(self.hardware_paths).read_text())
        self.vi = connect(exe_path=hw["exe_path"], vi_path=hw["vi_path"])
        self.igor = asylum.connect_igor()
        cfg, _, _ = modes.load_run_config(self.config_notebook)
        self.vi.setcontrolvalue("data_rate", 0)        # 1 MS/s: the only data_rate that is what it says
        self.cfg = self.configure(cfg, 1e6)
        self.backend = PyScannerBackend(self.vi, cfg.safety, cfg.readback, position_sensors=cfg.position_sensors)
        self.afm = asylum.AsylumControl(self.igor, self.backend, cfg)
        ok = self.afm.xy_routing_ok()
        self.afm.xy_from_fpga = bool(ok)   # Igor has the loops: Igor drives to the centre (GoForce) before the handover
        print("X/Y loops already follow the FPGA:", ok, "| AO", self.backend.read_ao_xy())
        self.afm.setup()
        self.tag = time.strftime("%Y%m%d_%H%M%S")
        self.say_guard("setup")
        print("RAM free", round(_ram_GB(), 1), "GB")

    def setup_simulation(self, topography_fn, sample_rate_hz=1e5, noise_std=3.0):
        """The same pipeline on a simulated sample (nothing connected): to check settings."""
        from .backends.simulate_backend import SimulateBackend
        cfg, _, _ = modes.load_run_config(self.config_notebook)
        self.cfg = self.configure(cfg, sample_rate_hz)
        cfg.position_sensors.enabled = False
        cfg.safety.max_samples_per_waveform = int(11e6 * sample_rate_hz / 1e6)
        self.backend = SimulateBackend(topography_fn, noise_std=noise_std, seed=1)
        self.afm = _NoAFM()
        self.tag = "sim_" + time.strftime("%Y%m%d_%H%M%S")

    @property
    def rate(self):
        return self.cfg.velocity.sample_rate_hz

    @staticmethod
    def _xs(s):
        """The step's measured positions (sensors), or the commanded ones without sensors."""
        return (s.x_sensor if s.x_sensor is not None else s.x), (s.y_sensor if s.y_sensor is not None else s.y)

    # ------------------------------------------------------------------
    def level0(self):
        """The sparse spiral (one do_scan, 10 s)."""
        pss.SPIRAL_RETURN = 1
        cfg, P = self.cfg, self.SPARSE
        self.session = modes.build("grid_refine", cfg, self.backend, live=False, priors=self.PRIORS)
        self.executor, self.pl = self.session.executor, self.session.planner
        S = pss.PyScannerScan("spiral", P["center"], 2 * P["radius_V"], 2 * P["radius_V"], P["duration_s"],
                              radius_V=P["radius_V"], n_cycles=P["n_cycles"], info=dict(spacing=P["radius_V"] / P["n_cycles"]))
        assert pss.within(S, cfg.safety)
        orig = self.pl._pyscanner_scans
        self.pl._pyscanner_scans = lambda r, sp, pat, sparse=False: [S] if sparse else orig(r, sp, pat, sparse)
        self.pl._tile_size, self.pl._n_interior = None, 0
        self._clear_guard()
        self.st = Stepwise(self.session, self.afm, out_dir=str(self.out), tag=f"L2run_{self.tag}")
        self.st.sparse()
        self.sparse_scan = S
        self.say_guard("level 0")

    # ------------------------------------------------------------------
    def objects_from_sparse(self):
        """Boxes round the objects on the sparse arcs (measured pixels > Z_OBJ_NM, arcs within
        0.9 V joined, pieces with centres < 3 V apart merged), >= 4 V, shifted inside the range."""
        from scipy.ndimage import label, binary_dilation, find_objects
        st, pl = self.st, self.pl
        B = st._flat(np.array(pl.binned, dtype=float))
        x0, _, y0, _ = pl.bounds; pix = pl._pix
        hi = np.isfinite(B) & (B > self.Z_OBJ_NM)
        lab, _ = label(binary_dilation(hi, iterations=int(round(0.9 / pix))))
        boxes = []
        for j, _sl in enumerate(find_objects(lab), 1):
            m = (lab == j) & hi
            if m.sum() < 5:
                continue
            rr, cc = np.nonzero(m)
            boxes.append([x0 + cc.min() * pix, x0 + cc.max() * pix, y0 + rr.min() * pix, y0 + rr.max() * pix])
        merged = True
        while merged:
            merged = False
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    A, Bx = boxes[a], boxes[b]
                    if np.hypot(0.5 * (A[0] + A[1] - Bx[0] - Bx[1]), 0.5 * (A[2] + A[3] - Bx[2] - Bx[3])) < 3.0:
                        boxes[a] = [min(A[0], Bx[0]), max(A[1], Bx[1]), min(A[2], Bx[2]), max(A[3], Bx[3])]
                        del boxes[b]; merged = True; break
                if merged:
                    break
        lims = pl._lims()
        rects = []
        for bx0, bx1, by0, by1 in boxes:
            pad = 0.6 * st.h
            cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
            wx, wy = max(bx1 - bx0 + 2 * pad, 4.0), max(by1 - by0 + 2 * pad, 4.0)
            cx = float(np.clip(cx, lims[0] + 0.3 + wx / 2, lims[1] - 0.3 - wx / 2))
            cy = float(np.clip(cy, lims[2] + 0.3 + wy / 2, lims[3] - 0.3 - wy / 2))
            r = ScanRect(cx - wx / 2, cx + wx / 2, cy - wy / 2, cy + wy / 2, "x")
            r.pattern = "spiral"; r.spiral_center = np.array([cx, cy]); r.radius = 0.5 * min(wx, wy)
            rects.append(r)
        return rects

    def _run_level(self, k, rects, h, max_s, label):
        cfg, st, pl, ex = self.cfg, self.st, self.pl, self.executor
        cfg.safety.max_samples_per_waveform = ex.safety.max_samples_per_waveform = int(max_s * self.rate) + 1000
        st.pending = (k, h, rects, self.Z_OBJ_NM, None, 0.0)
        st.approve()
        lv = st.level
        plan = []
        for i, r in enumerate(lv.rects):
            sc = pl._pyscanner_scans(r, h, "spiral")
            assert len(sc) == 1 and sc[0].pattern == "spiral" and sc[0].duration_s <= max_s + 1e-9 and pss.within(sc[0], cfg.safety)
            plan.append(sc[0])
            cu = um(*r.center)
            print(f"  scan {i + 1}: {sc[0].describe()} | turns {sc[0].info['fit_spacing_V']:.3f} V, peak "
                  f"{sc[0].info['fit_speed_V_s']:.0f} V/s | AR centre ({cu[0]:.1f}, {cu[1]:.1f}) um")
        print(f"  total scanning {sum(s.duration_s for s in plan):.1f} s; RAM free {_ram_GB():.1f} GB")
        self._clear_guard()
        t0 = time.time()
        with self.afm.run(label, ex):
            for i, s in enumerate(plan):
                n0 = len(ex.history); ns0 = sum(len(q.x) for q in ex.history)
                ex.send_pyscanner(s, "refine", meta={"label": f"Level {k} -- scan {i + 1}/{len(plan)}: {s.describe()}", "level": k})
                lv.n_samples += sum(len(q.x) for q in ex.history) - ns0
                st.done.append((i, n0, len(ex.history)))
                print(f"  scan {i + 1} done ({time.time() - t0:.0f} s), RAM free {_ram_GB():.1f} GB")
                gc.collect()
        lv.scan_time_s = lv.n_samples / pl._rate
        done = list(st.done)
        for i, n0, n1 in done:
            st.show_scan(i, n0, n1)
        self.say_guard(f"level {k}")
        return done

    def level1(self):
        """One spiral (<= L1_MAX_S) per object found on the sparse spiral."""
        rects = self.objects_from_sparse()
        print(f"level 1: {len(rects)} object(s) on the sparse arcs")
        self.H1 = self.st.h / self.cfg.grid_refine.refine_factor
        self.DONE1 = self._run_level(1, rects, self.H1, self.L1_MAX_S, "level 1: one spiral per object")

    def _refine_step(self, n0, n1):
        return [q for q in self.executor.history[n0:n1] if q.kind == "refine"][0]

    def level2(self):
        """One dense spiral per object, placed from level 1's measured data."""
        self.H2 = self.H1 / self.cfg.grid_refine.refine_factor
        L2 = []
        for i, n0, n1 in self.DONE1:
            s = self._refine_step(n0, n1)
            sx, sy = self._xs(s)
            ok = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(s.z)
            xs, ys, z, xc, yc = sx[ok], sy[ok], s.z[ok], s.x[ok], s.y[ok]
            lo = z < np.percentile(z, 30)                         # substrate: the lowest 30 %
            co = np.linalg.lstsq(np.column_stack([np.ones(lo.sum()), xs[lo], ys[lo]]), z[lo], rcond=None)[0]
            zf = z - (co[0] + co[1] * xs + co[2] * ys); zf -= np.median(zf[lo])
            m = zf > max(self.Z_OBJ_NM, 0.5 * np.percentile(zf, 97))   # the object: over half its height
            ox, oy = np.percentile(xs[m], [2, 98]), np.percentile(ys[m], [2, 98])
            cen = np.array([ox.mean(), oy.mean()]); half = 0.5 * max(ox[1] - ox[0], oy[1] - oy[0])
            c_cmd = np.array([0.5 * (xc.min() + xc.max()), 0.5 * (yc.min() + yc.max())])
            c_meas = np.array([0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())])
            off = c_meas - c_cmd                                  # the real path vs the commanded one (level 1)
            ratio = 0.5 * ((xs.max() - xs.min()) / (xc.max() - xc.min()) + (ys.max() - ys.min()) / (yc.max() - yc.min()))
            R = (half + self.L2_PAD_V) / ratio
            c_new = cen - off
            r = ScanRect(c_new[0] - R, c_new[0] + R, c_new[1] - R, c_new[1] + R, "x")
            r.pattern = "spiral"; r.spiral_center = c_new; r.radius = R
            L2.append(r)
            cu = um(*cen)
            print(f"object {i + 1}: centre AR ({cu[0]:.1f}, {cu[1]:.1f}) um, ~{2 * half * self.UM_PER_V_AVG:.1f} um across; "
                  f"level-1 real path {ratio:.2f} x commanded, centre off ({off[0]:+.2f}, {off[1]:+.2f}) V")
        self.DONE2 = self._run_level(2, L2, self.H2, self.L2_MAX_S, "level 2: one dense spiral per object")

    # ------------------------------------------------------------------
    def _lev(self, x, y, z):
        """One scan levelled on its own substrate (plane through the lower 55 %)."""
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); x, y, z = x[ok], y[ok], z[ok]
        m = z < np.percentile(z, 55)
        for _ in range(2):
            co = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), x[m], y[m]]), z[m], rcond=None)[0]
            r = z - (co[0] + co[1] * x + co[2] * y); m = r < np.percentile(r, 55)
        return ok, r - np.median(r[m])

    def parts(self):
        """[(name, level, step, ok-mask, levelled z)] in time order. Levels 0/1 on their own
        substrate; level 2 (mostly on the objects) against the levels 0+1 reconstruction."""
        from scipy.interpolate import RegularGridInterpolator
        ex, pl = self.executor, self.pl
        sp = [q for q in ex.history if q.kind == "grid_sparse"][0]
        P = [("L0", 0, sp) + self._lev(*self._xs(sp), sp.z)]
        for i, n0, n1 in self.DONE1:
            s = self._refine_step(n0, n1); P.append((f"L1_{i + 1}", 1, s) + self._lev(*self._xs(s), s.z))
        b, N = pl.bounds, self.N_PIX
        xy = lambda p: np.column_stack([a[p[3]] for a in self._xs(p[2])])
        B01, _ = bin_image(np.vstack([xy(p) for p in P]), np.concatenate([p[4] for p in P]), N, b)
        R01 = inpaint_binned(B01, b, self.cfg.live_view, mode="linear")
        F = RegularGridInterpolator((np.linspace(b[2], b[3], N), np.linspace(b[0], b[1], N)), R01, bounds_error=False, fill_value=np.nan)
        for i, n0, n1 in self.DONE2:
            s = self._refine_step(n0, n1)
            sx, sy = self._xs(s)
            ok = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(s.z)
            x, y, z = sx[ok], sy[ok], s.z[ok]
            ref = F(np.column_stack([y, x])); m = np.isfinite(ref)
            co = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), x[m], y[m]]), (z - ref)[m], rcond=None)[0]
            zz = z - (co[0] + co[1] * x + co[2] * y)
            print(f"L2 scan {i + 1}: levelled against levels 0+1, residual rms {np.std((zz - ref)[m]):.1f} nm")
            P.append((f"L2_{i + 1}", 2, s, ok, zz))
        return P

    def _pxy(self, p):
        x, y = self._xs(p[2])
        return x[p[3]], y[p[3]]

    def save_dataset(self):
        """Everything in one npz: every scan (commanded + sensor positions, raw and levelled Z,
        its PyScanner scan), the binned images / reconstructions, the guard log, the calibration.
        The npz has binned_Z_sensor / binned_Z / recon_bounds, so user_raster.plan(npz, ...)
        can plan the equivalent raster with the same number of pixels."""
        P = self.PARTS = self.parts()
        pl, cfg, ex = self.pl, self.cfg, self.executor
        b, N = pl.bounds, self.N_PIX
        out = {}
        for name, k, s, ok, zl in P:
            sx, sy = self._xs(s)
            out.update({f"{name}_x_cmd": s.x.astype(np.float32), f"{name}_y_cmd": s.y.astype(np.float32),
                        f"{name}_x_sensor": sx.astype(np.float32), f"{name}_y_sensor": sy.astype(np.float32),
                        f"{name}_z_nm": s.z.astype(np.float32), f"{name}_ok": ok, f"{name}_z_levelled_nm": zl.astype(np.float32),
                        f"{name}_scan": s.meta.get("pyscanner_scan", ""), f"{name}_level": k})
        xs = np.concatenate([self._pxy(p)[0] for p in P]); ys = np.concatenate([self._pxy(p)[1] for p in P])
        xc = np.concatenate([p[2].x[p[3]] for p in P]); yc = np.concatenate([p[2].y[p[3]] for p in P])
        zl = np.concatenate([p[4] for p in P])
        Bs, _ = bin_image(np.column_stack([xs, ys]), zl, N, b)
        Bc, _ = bin_image(np.column_stack([xc, yc]), zl, N, b)
        t = time.time(); Rl = inpaint_binned(Bs, b, cfg.live_view, mode="linear"); tl = time.time() - t
        t = time.time(); Rc = inpaint_binned(Bs, b, cfg.live_view, mode="compressed_sensing"); tc = time.time() - t
        t = time.time(); Rt = inpaint_binned(Bs, b, cfg.live_view, mode="tv"); tt = time.time() - t
        cal = getattr(ex, "sensor_cal", None)
        path = self.out / f"stepwise_L2_{self.tag}.npz"
        np.savez_compressed(path, scans=np.array([p[0] for p in P]), binned_Z_sensor=Bs, binned_Z=Bc, recon_bounds=np.array(b),
                            recon_linear=Rl, recon_cs=Rc, recon_tv=Rt, n_pixels=N, sample_rate_hz=self.rate,
                            store_every=cfg.readback.store_every, tip_velocity_V_s=self.V,
                            h_V=np.array([self.st.h, self.H1, self.H2]), um_per_V=np.array(UM_PER_V),
                            um_at_0V=np.array(UM_AT_0), z_unit="nm", generator="stepwise spirals L0-L2 (no jumps)",
                            sensor_cal=str(getattr(cal, "report", "")), guard_log=str(getattr(self.backend, "guard_log", "")), **out)
        print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB): {len(P)} scans; {int(np.isfinite(Bs).sum()):,} of {N * N:,} "
              f"pixels measured ({N} x {N}); linear {tl:.1f} s, CS {tc:.1f} s, TV {tt:.1f} s")
        self.NPZ, self.RECON = path, (Bs, Rl, Rc, Rt)
        return path

    # ------------------------------------------------------------------
    def figures(self):
        """Overview (levels, paths, reconstructions) + per-object zooms."""
        P, pl, cfg = self.PARTS, self.pl, self.cfg
        b, N = pl.bounds, self.N_PIX
        E = ext_um(b)

        def binned(sel):
            Q = [p for p in P if p[1] in sel]
            B, _ = bin_image(np.column_stack([np.concatenate([self._pxy(p)[0] for p in Q]),
                                              np.concatenate([self._pxy(p)[1] for p in Q])]), np.concatenate([p[4] for p in Q]), N, b)
            return B
        B0, B01, BA = binned({0}), binned({0, 1}), self.RECON[0]
        R0, R01 = (inpaint_binned(B, b, cfg.live_view, mode="linear") for B in (B0, B01))
        fig = Figure(figsize=(17, 11), layout="constrained"); axs = fig.subplots(2, 4)
        panels = [(B0, "level 0: measured pixels"), (B01, "levels 0+1: measured pixels"), (BA, "levels 0+1+2: measured pixels"),
                  (None, "real paths"), (R0, "reconstruction, level 0"), (R01, "reconstruction, levels 0+1"),
                  (self.RECON[1], "reconstruction, levels 0+1+2 (linear)"), (self.RECON[3], "levels 0+1+2 (total variation)")]
        for a, (img, ttl) in zip(axs.ravel(), panels):
            if img is None:
                for p in P:
                    ux, uy = um(self._pxy(p)[0][::10], self._pxy(p)[1][::10])
                    a.plot(ux, uy, lw=(0.3, 0.3, 0.2)[p[1]], color=("0.6", "C0", "C3")[p[1]])
            else:
                h = a.imshow(img, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135)
                fig.colorbar(h, ax=a, label="Z (nm)", shrink=0.7)
            a.set_xlim(min(E[:2]), max(E[:2])); a.set_ylim(E[2], E[3]); a.set_aspect("equal"); a.set_title(ttl, fontsize=10)
            a.set_xlabel("X (um, AR frame)"); a.set_ylabel("Y (um, AR frame)")
        n = [int(np.isfinite(B).sum()) for B in (B0, B01, BA)]
        fig.suptitle(f"Stepwise grid_refine {self.tag}: level 0 (1 spiral) + level 1 ({len(self.DONE1)} spirals) + level 2 "
                     f"({len(self.DONE2)} dense spirals), one PyScanner do_scan each, no jumps | pixels "
                     f"{n[0]:,} / {n[1]:,} / {n[2]:,} of {N * N:,}")
        p1 = self.out / f"stepwise_L2_{self.tag}_overview.png"; fig.savefig(p1, dpi=80)
        n2 = max(len(self.DONE2), 1)
        fig = Figure(figsize=(4.3 * n2, 8.6), layout="constrained"); axs = fig.subplots(2, n2, squeeze=False)
        for k, p in enumerate([p for p in P if p[1] == 2]):
            cu = um(np.median(self._pxy(p)[0]), np.median(self._pxy(p)[1])); w = 3.6
            for row, (img, lab) in enumerate(((R01, "levels 0+1"), (self.RECON[1], "levels 0+1+2"))):
                a = axs[row, k]
                a.imshow(img, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135)
                a.set_xlim(cu[0] - w, cu[0] + w); a.set_ylim(cu[1] - w, cu[1] + w); a.set_aspect("equal")
                a.set_title(f"object {k + 1}: {lab}", fontsize=10); a.set_xlabel("X (um)"); a.set_ylabel("Y (um)")
        fig.suptitle("Per object: reconstruction before (top) and after (bottom) level 2")
        p2 = self.out / f"stepwise_L2_{self.tag}_objects.png"; fig.savefig(p2, dpi=80)
        print("figures:", p1, p2)
        self.FIGS = [str(p1), str(p2)]
        return self.FIGS

    def animate(self, frames_per_scan=(12, 3, 4), fps=4):
        """GIF of the analysis as the run went: real path so far | measured pixels | linear
        reconstruction, the scan being acquired highlighted."""
        from matplotlib.animation import PillowWriter
        P, b, N = self.PARTS, self.pl.bounds, self.N_PIX
        E = ext_um(b)
        chunks = []
        for j, p in enumerate(P):
            n = int(p[3].sum()); f = frames_per_scan[p[1]]
            for c in range(1, f + 1):
                chunks.append((j, int(round(n * c / f))))
        fig = Figure(figsize=(16, 5.6), layout="constrained"); axs = fig.subplots(1, 3)
        path = self.out / f"stepwise_L2_{self.tag}_analysis.gif"
        w = PillowWriter(fps=fps)
        t0 = time.time()
        with w.saving(fig, str(path), dpi=70):
            for j, upto in chunks + [chunks[-1]] * (2 * fps):
                X, Y, Z = [], [], []
                for jj in range(j + 1):
                    p = P[jj]; m = slice(None) if jj < j else slice(0, upto)
                    px, py = self._pxy(p)
                    X.append(px[m]); Y.append(py[m]); Z.append(p[4][m])
                X, Y, Z = np.concatenate(X), np.concatenate(Y), np.concatenate(Z)
                B, _ = bin_image(np.column_stack([X, Y]), Z, N, b)
                R = inpaint_binned(B, b, self.cfg.live_view, mode="linear")
                for a in axs:
                    a.clear()
                for jj in range(j + 1):
                    p = P[jj]; m = slice(None) if jj < j else slice(0, upto)
                    px, py = self._pxy(p)
                    ux, uy = um(px[m][::20], py[m][::20])
                    axs[0].plot(ux, uy, lw=0.5 if jj == j else 0.25, color="C3" if jj == j else ("0.55", "C0", "C2")[p[1]])
                axs[1].imshow(B, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135)
                axs[2].imshow(R, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135)
                for a, ttl in zip(axs, ("real tip path (sensors); red: scan in progress",
                                        f"measured pixels: {int(np.isfinite(B).sum()):,} of {N * N:,}", "reconstruction (linear)")):
                    a.set_xlim(min(E[:2]), max(E[:2])); a.set_ylim(E[2], E[3]); a.set_aspect("equal"); a.set_title(ttl, fontsize=10)
                    a.set_xlabel("X (um, AR frame)")
                axs[0].set_ylabel("Y (um, AR frame)")
                p = P[j]
                fig.suptitle(f"Level {p[1]}" + ("" if p[1] == 0 else f", scan {p[0].split('_')[1]}")
                             + f": {p[2].meta.get('pyscanner_scan', '')}", fontsize=11)
                w.grab_frame()
        print(f"animation: {path} ({len(chunks)} frames + hold, {time.time() - t0:.0f} s to make)")
        self.GIF = str(path)
        return self.GIF

    # ------------------------------------------------------------------
    def plan_raster(self):
        """The equivalent raster (same number of pixels; nothing moves): prints the plan and
        the RAM check. Then equivalent_raster() runs it."""
        from . import user_raster
        cfg = self.cfg
        cfg.grid_refine.raster_turn_radius_V = 0.3   # gentle half-ellipse turnarounds (slow down at the apex)
        cfg.grid_refine.ease_time_s = 0.02           # every band starts / ends at rest, 20 ms speeding up / slowing down
        cfg.safety.max_samples_per_waveform = self.executor.safety.max_samples_per_waveform = int(4.0 * self.rate)
        self.RP = user_raster.plan(self.NPZ, cfg, self.executor, area=self.pl.bounds)
        print(user_raster.describe(self.RP))
        ok = user_raster.ram_check(self.RP, cfg)
        return self.RP, ok

    def equivalent_raster(self):
        """Run the planned equivalent raster (user-defined smooth waveforms, no jumps) and add
        it to the dataset's npz (raster_* arrays)."""
        from . import user_raster, reference_raster
        if getattr(self, "RP", None) is None:
            self.plan_raster()
        assert user_raster.ram_check(self.RP, self.cfg), "not enough RAM / outside the limits -- see above"
        self._clear_guard()
        with self.afm.run("equivalent raster (user-defined, smooth)", self.executor):
            self.REF = user_raster.acquire(self.RP, self.executor, self.cfg)
        self.say_guard("raster")
        reference_raster.save_into(self.NPZ, self.REF)
        print("added to", self.NPZ, "| RAM free", round(_ram_GB(), 1), "GB")
        return self.REF

    def compare_raster(self, zoom_um=None, fps=4):
        """Self-driving vs the equivalent raster: figure (paths, measured pixels, linear
        reconstructions, optional zoom (x0, x1, y0, y1) um) and an animation at equal pixel
        budgets per frame."""
        from matplotlib.animation import PillowWriter
        REF, P, cfg = self.REF, self.PARTS, self.cfg
        b, N = self.pl.bounds, self.N_PIX
        E = ext_um(b)
        k = ("xs", "ys", "zs") if "xs" in REF else ("x", "y", "z")     # sensor positions if recorded
        rx, ry, rz = (np.asarray(REF[q], float) for q in k)
        okr, rzl = self._lev(rx, ry, rz); rx, ry = rx[okr], ry[okr]
        n_lines = REF["plan"]["n_lines"]
        SX = np.concatenate([self._pxy(p)[0] for p in P]); SY = np.concatenate([self._pxy(p)[1] for p in P])
        SZ = np.concatenate([p[4] for p in P])

        def first_new(x, y):
            """cumulative number of distinct pixels measured, sample by sample"""
            ix = np.clip(((x - b[0]) / (b[1] - b[0]) * N).astype(int), 0, N - 1)
            iy = np.clip(((y - b[2]) / (b[3] - b[2]) * N).astype(int), 0, N - 1)
            inside = (x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3])
            k = np.where(inside, iy * N + ix, -1)
            new = np.zeros(len(k), bool)
            _, first = np.unique(k, return_index=True); new[first] = True; new &= k >= 0
            return np.cumsum(new)
        cS, cR = first_new(SX, SY), first_new(rx, ry)
        print(f"pixels: self-driving {cS[-1]:,}, raster {cR[-1]:,}")
        BS, _ = bin_image(np.column_stack([SX, SY]), SZ, N, b); RS = inpaint_binned(BS, b, cfg.live_view, mode="linear")
        BR, _ = bin_image(np.column_stack([rx, ry]), rzl, N, b); RR = inpaint_binned(BR, b, cfg.live_view, mode="linear")
        ncol = 4 if zoom_um else 3
        fig = Figure(figsize=(4.25 * ncol, 9.5), layout="constrained"); axs = fig.subplots(2, ncol)
        for row, (B, R, x, y, name) in enumerate(((BS, RS, SX, SY, f"self-driving (levels 0+1+2, {len(P)} spirals)"),
                                                  (BR, RR, rx, ry, f"equivalent raster ({n_lines} lines, user-defined, smooth)"))):
            ux, uy = um(x[::40], y[::40]); axs[row, 0].plot(ux, uy, lw=0.2, color="C0")
            for a, img, ttl in ((axs[row, 1], B, f"measured pixels: {int(np.isfinite(B).sum()):,}"), (axs[row, 2], R, "reconstruction (linear)")):
                a.imshow(img, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135); a.set_title(ttl, fontsize=10)
            axs[row, 0].set_title(f"{name}\nreal path (sensors)", fontsize=10)
            for a in axs[row, :3]:
                a.set_xlim(min(E[:2]), max(E[:2])); a.set_ylim(E[2], E[3]); a.set_aspect("equal"); a.set_xlabel("X (um)"); a.set_ylabel("Y (um)")
            if zoom_um:
                a = axs[row, 3]; a.imshow(R, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135)
                a.set_xlim(*zoom_um[:2]); a.set_ylim(*zoom_um[2:]); a.set_aspect("equal"); a.set_title("zoom", fontsize=10); a.set_xlabel("X (um)")
        fig.suptitle(f"Same sample, same number of pixels ({cS[-1]:,} vs {cR[-1]:,} of {N * N:,}), no jumps in either", fontsize=12)
        p1 = self.out / f"stepwise_L2_{self.tag}_vs_raster.png"; fig.savefig(p1, dpi=80); print("figure:", p1)
        chunks = []
        starts = np.cumsum([0] + [int(p[3].sum()) for p in P])
        for j, p in enumerate(P):
            n = int(p[3].sum()); f = (12, 3, 4)[p[1]]
            for c in range(1, f + 1):
                chunks.append((j, starts[j] + int(round(n * c / f))))
        fig = Figure(figsize=(16, 10.2), layout="constrained"); axs = fig.subplots(2, 3)
        path = self.out / f"stepwise_L2_{self.tag}_vs_raster.gif"; w = PillowWriter(fps=fps); t0 = time.time()
        with w.saving(fig, str(path), dpi=65):
            for j, upto in chunks + [chunks[-1]] * 8:
                npx = int(cS[upto - 1]); ur = min(int(np.searchsorted(cR, npx)) + 1, len(rx))
                sd_name = f"Self-driving -- level {P[j][1]}" + ("" if P[j][1] == 0 else f", scan {P[j][0].split('_')[1]}")
                for row, (x, y, z, u, name) in enumerate(((SX, SY, SZ, upto, sd_name),
                                                         (rx, ry, rzl, ur, f"Raster -- line {min(n_lines, int(np.ceil(n_lines * ur / len(rx))))} of {n_lines}"))):
                    B, _ = bin_image(np.column_stack([x[:u], y[:u]]), z[:u], N, b); R = inpaint_binned(B, b, cfg.live_view, mode="linear")
                    for a in axs[row]:
                        a.clear()
                    ux, uy = um(x[:u][::25], y[:u][::25]); axs[row, 0].plot(ux, uy, lw=0.25, color="C0")
                    tail = slice(max(0, u - 150000), u); ux, uy = um(x[tail][::10], y[tail][::10]); axs[row, 0].plot(ux, uy, lw=0.6, color="C3")
                    axs[row, 1].imshow(B, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135)
                    axs[row, 2].imshow(R, origin="lower", extent=E, cmap="copper", interpolation="nearest", vmin=-15, vmax=135)
                    for a, ttl in zip(axs[row], (f"{name}: real path", f"measured pixels: {int(np.isfinite(B).sum()):,}", "reconstruction (linear)")):
                        a.set_xlim(min(E[:2]), max(E[:2])); a.set_ylim(E[2], E[3]); a.set_aspect("equal"); a.set_title(ttl, fontsize=10)
                fig.suptitle(f"Equal pixel budgets: {npx:,} of {N * N:,} pixels measured", fontsize=12)
                w.grab_frame()
        print(f"animation: {path} ({time.time() - t0:.0f} s)")
        self.FIG_VS, self.GIF_VS = str(p1), str(path)
        return self.FIG_VS, self.GIF_VS
