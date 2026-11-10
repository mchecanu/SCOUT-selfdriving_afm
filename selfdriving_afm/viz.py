"""Plotting helpers.

The two `plot_*` functions review a *finished* run (simulation or real).
The two `Live*View` classes are the "sit and watch it happen" pieces: pass an
instance's `.on_step` as the `on_step=` callback into `execution.ContinuousExecutor`
(or the planner constructor helpers in this module) and the figure updates itself
as each spiral / circle / line segment comes back from the backend -- on real
hardware exactly the same as in simulation, since both just watch
`ContinuousExecutor.history`.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from .config import LiveViewConfig
from .planners.edge_follow_planner import Feature
from .reconstruction import bin_image, reconstruct_image, inpaint_binned, pixel_index, uncertainty_map  # noqa: F401 (bin_image re-exported)

RECON_TITLES = {"compressed_sensing": "CS reconstruction",
                "gp": "GP reconstruction",
                "fast": "Fast inpainting",
                "linear": "Linear interpolation",
                "raster_columns": "Column-wise PCHIP",
                "interpolated": "Interpolated"}


def _strided(arr, max_points=3000):
    """Downsample purely for plotting -- see viz._LiveViewBase._draw_path's
    docstring for why (real hardware at a slow tip velocity can put millions of
    samples in one segment)."""
    stride = max(1, len(arr) // max_points)
    return arr[::stride]


def plot_edge_follow_result(features, backend, ground_truth_fn=None, extent=0.35, ax=None):
    """Overview plot: spiral search segments (blue) + tracing circles (green),
    optionally over a ground-truth topography if you're in simulation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if ground_truth_fn is not None:
        g = np.linspace(-extent, extent, 300)
        Xg, Yg = np.meshgrid(g, g)
        Zg = ground_truth_fn(Xg, Yg)
        ax.pcolormesh(Xg, Yg, Zg, cmap='copper', shading='auto', alpha=0.6)

    for fi, feature in enumerate(features):
        for att in feature.spiral_attempts:
            ax.plot(_strided(att.x), _strided(att.y), '-', color='tab:blue', linewidth=1, alpha=0.8)
        for cs in feature.circle_steps:
            ax.plot(_strided(cs.x), _strided(cs.y), '-', color='tab:green', linewidth=1.2, alpha=0.8)
            if cs.crossing_xy is not None:
                ax.plot(*cs.crossing_xy, 'r.', markersize=4)

    ax.set_aspect('equal')
    ax.set_xlabel('X (V)')
    ax.set_ylabel('Y (V)')
    ax.set_title(f'Edge-follow scan: {len(features)} feature(s) traced')
    ax.grid(True, alpha=0.3)
    return ax


def plot_gp_path_result(segments, gp_planner, ground_truth_fn=None, extent=0.35):
    g = np.linspace(-extent, extent, 200)
    Xg, Yg = np.meshgrid(g, g)
    mu, sigma = gp_planner.predict_surface(Xg, Yg)

    fig, axes = plt.subplots(1, 3 if ground_truth_fn is None else 4, figsize=(18, 5))
    idx = 0

    if ground_truth_fn is not None:
        Zg_true = ground_truth_fn(Xg, Yg)
        ax = axes[idx]; idx += 1
        c = ax.pcolormesh(Xg, Yg, Zg_true, cmap='copper', shading='auto', alpha=0.6)
        for seg in segments:
            ax.plot(_strided(seg.x), _strided(seg.y), '-', linewidth=1.5, alpha=0.8)
        ax.set_title('Ground truth + scan path')
        ax.set_aspect('equal')
        plt.colorbar(c, ax=ax)

    ax = axes[idx]; idx += 1
    c = ax.pcolormesh(Xg, Yg, mu, cmap='copper', shading='auto')
    ax.set_title('GP reconstruction (mean)')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax)

    ax = axes[idx]; idx += 1
    c = ax.pcolormesh(Xg, Yg, sigma, cmap='hot', shading='auto')
    for seg in segments:
        ax.plot(_strided(seg.x), _strided(seg.y), 'cyan', linewidth=0.8, alpha=0.6)
    ax.set_title('GP uncertainty + path')
    ax.set_aspect('equal')
    plt.colorbar(c, ax=ax)

    ax = axes[idx]
    all_pts = np.column_stack([np.concatenate([s.x for s in segments]),
                                np.concatenate([s.y for s in segments])])
    ax.plot(_strided(all_pts[:, 0]), _strided(all_pts[:, 1]), '.', markersize=1)
    ax.set_title(f'Raw path ({len(all_pts)} samples)')
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# ================================================================================
# Live views -- "sit and watch the reconstruction fill in as the scan runs"
# ================================================================================

_KIND_STYLE = {
    "spiral": dict(color="tab:blue", linewidth=1.0, alpha=0.85, label="spiral search"),
    "search_area": dict(color="tab:blue", linewidth=0.7, alpha=0.8, label="first search scan"),
    "circle": dict(color="tab:green", linewidth=1.3, alpha=0.9, label="edge trace"),
    "line": dict(color="tab:purple", linewidth=1.3, alpha=0.9, label="GP path"),
    "explore": dict(color="tab:orange", linewidth=1.2, alpha=0.9, label="edge explorer path"),
    "grid_sparse": dict(color="tab:blue", linewidth=0.7, alpha=0.8, label="sparse scan"),
    "refine": dict(color="tab:green", linewidth=0.7, alpha=0.8, label="refinement scans"),
    "connector": dict(color="0.6", linewidth=1.0, alpha=0.7, linestyle="--", label="transit (connector)"),
}


def subsample_for_reconstruction(points_xy: np.ndarray, z: np.ndarray, max_points: int,
                                  rng: Optional[np.random.Generator] = None):
    """Cap the point count fed into griddata/GP -- a full scan's raw point cloud
    (millions of samples at a slow tip_velocity_V_s, see VelocityConfig) makes
    reconstruction itself the bottleneck otherwise, live or when saving/re-analyzing
    a finished run. Keeps the most recent third in full (so the newest trace is
    always faithfully shown) and randomly thins the rest. Public so the notebook's
    save cell and Analyze_SelfDriving_Scan.ipynb reuse the exact same logic the live
    view does, rather than each re-deriving their own cap.
    """
    rng = rng or np.random.default_rng(0)
    n = len(z)
    if n <= max_points:
        return points_xy, z
    n_recent = max_points // 3
    recent_idx = np.arange(n - n_recent, n)
    n_older = max_points - n_recent
    older_idx = rng.choice(n - n_recent, size=min(n_older, n - n_recent), replace=False)
    idx = np.concatenate([older_idx, recent_idx])
    return points_xy[idx], z[idx]


def build_grid(extent: float, n: int = 150):
    """The (Xg, Yg) grid every reconstruction in this module is evaluated over.
    Public so the notebook's save cell and Analyze_SelfDriving_Scan.ipynb can
    rebuild the exact same grid a live view used, or a different one entirely."""
    g = np.linspace(-extent, extent, n)
    return np.meshgrid(g, g)


def bounds_from_history(history, fallback_extent: float, pad_frac: float = 0.15):
    """Square (xmin, xmax, ymin, ymax) around everywhere `history` actually covers,
    with padding -- or +/-fallback_extent if history is empty. This is what the live
    view's axes track instead of a fixed guessed extent (see
    `_LiveViewBase._bounds`); public so notebook code -- e.g. the final save-time
    reconstruction in Run_SelfDriving_Scan.ipynb's cell 5 -- can build a grid over
    the same actual-coverage window rather than a fixed extent that may not match
    where the scan ended up."""
    if not history:
        e = fallback_extent
        return -e, e, -e, e
    xs = np.concatenate([s.x for s in history])
    ys = np.concatenate([s.y for s in history])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    half = max(xmax - xmin, ymax - ymin, 1e-6) / 2 * (1.0 + pad_frac)
    cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2
    return cx - half, cx + half, cy - half, cy + half


def grid_from_bounds(bounds, n: int = 150):
    """(Xg, Yg) mesh over an (xmin, xmax, ymin, ymax) tuple, e.g. from
    `bounds_from_history` -- the non-square-around-zero equivalent of `build_grid`."""
    xmin, xmax, ymin, ymax = bounds
    return np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))


def reconstruct_topography(points_xy: np.ndarray, z: np.ndarray, grid_xy, min_points: int = 10,
                            method: Optional[str] = None):
    """Nearest-neighbour fill that upgrades to linear once the points span enough of
    the plane to triangulate -- same trick the exploratory notebooks used so the
    reconstruction doesn't blow up (QhullError) while the path is still ~1D.
    Pass `method="nearest"` or `"linear"` to force one (e.g. from
    Analyze_SelfDriving_Scan.ipynb, where you might want to compare them side by
    side); leave it None for the auto-selection above.
    """
    if len(points_xy) < min_points:
        return None
    if method is None:
        ux = np.unique(np.round(points_xy[:, 0], 9))
        uy = np.unique(np.round(points_xy[:, 1], 9))
        method = "linear" if (len(ux) >= 3 and len(uy) >= 3) else "nearest"
    try:
        Z = griddata(points_xy, z, grid_xy, method=method)
        if method == "linear":
            # fill the outside-the-hull holes linear leaves as NaN
            nan_mask = np.isnan(Z)
            if nan_mask.any():
                Z[nan_mask] = griddata(points_xy, z, grid_xy, method="nearest")[nan_mask]
        return Z
    except Exception:
        return griddata(points_xy, z, grid_xy, method="nearest")


def um_axis_ticks(ax, cal, n: int = 6) -> None:
    """Ticks at round micrometre values on an axes whose data are drive volts (the
    position-sensor frame, sensors.SensorCalibration: um = (k V + b) 1e6 per axis)."""
    length_axis_ticks(ax, cal, n, unit="um")


def length_axis_ticks(ax, cal, n: int = 6, unit: str = "um") -> None:
    """Ticks at round values of `unit` ("um" or "nm") along the sensors, on an axes whose
    data are drive volts (length = (k V + b) per axis, sensors.SensorCalibration)."""
    from matplotlib.ticker import MaxNLocator
    f = {"um": 1e6, "nm": 1e9}[unit]
    for axis, c, lim in ((ax.xaxis, cal.x, ax.get_xlim()), (ax.yaxis, cal.y, ax.get_ylim())):
        L = sorted((c.k * v + c.b) * f for v in lim)
        t_L = [t for t in MaxNLocator(n).tick_values(*L) if L[0] <= t <= L[1]]
        axis.set_ticks([(t / f - c.b) / c.k for t in t_L])
        axis.set_ticklabels([f"{t:g}" for t in t_L])


class _LiveViewBase:
    """Redraw cadence is wall-clock throttled (see LiveViewConfig) so the live view
    can't become the bottleneck in the acquisition loop, regardless of whether the
    backend is simulation (microseconds per segment) or real hardware (milliseconds).
    """

    def __init__(self, cfg: LiveViewConfig, extent: float, ground_truth_fn=None):
        import time
        self._time = time
        self.cfg = cfg
        self.extent = extent
        self.ground_truth_fn = ground_truth_fn
        self._rng = np.random.default_rng(0)
        self._last_path_draw = -np.inf
        self._last_recon_draw = -np.inf
        self._last_status_draw = -np.inf
        self._last_update_end = -np.inf   # time budget -- see _budget_ok
        self._last_update_cost = 0.0
        self.status = ""
        self.log = []  # one line per waveform sent, in order -- print("\n".join(live.log))
        self._cbars = {}
        # optional second row: the classic raster with the same budget (Simulation.ipynb;
        # simulation.EquivalentRaster) -- None on the instrument
        self.compare = None
        self.compare_axes = None
        # extra columns after path | binned | reconstruction (_make_axes): "unc" =
        # uncertainty map, "err" = the true error (simulation) -- per row
        self.extra_axes = {}
        self.compare_extra = {}
        # position sensors: a second row, the same panels from the MEASURED positions, axes in um
        self.sensor_row = False
        self.sensor_axes = None
        self.sensor_extra = {}
        self.channel_rows = []           # extra signals, a row each: [{name, label, cmap, axes, extra}]
        self.last_binned_sensor = self.last_recon_sensor = None
        self.last_compare = None
        self._last_compare_draw = -np.inf
        self._mode_path_V = 0.0
        # Simulation.ipynb: every_step = redraw ALL panels after EVERY waveform (no time
        # throttling -- in simulation a mode can send a few long waveforms, all inside one
        # throttle window, so nothing would show until the end); keep_frames = also keep
        # each of those frames (self.frames) for show_frames() / save_frames() afterwards.
        # Both off on the instrument, where the throttle keeps the live view from slowing
        # the acquisition down.
        self.every_step = False
        self.skip_moves = False   # every-step mode: don't draw after moves between scans
        self.keep_frames = False
        self.frames = []
        self._last_png = None

    # ------------------------------------------------------------------
    # every-step mode + frame replay (Simulation.ipynb)
    # ------------------------------------------------------------------
    def _begin_step(self, step_index: int, step, force: bool):
        """Common start of on_step: returns the effective `force` flag -- or None to
        skip this step (every-step mode with skip_moves, after a move) -- and in
        every-step mode titles the figure with the step that just finished."""
        if self.every_step:
            if step is not None:
                if self.skip_moves and step.kind == "connector":
                    return None  # the next scan's frame shows this move in its path
                self._set_status(f"Step {step_index} ({step.kind}): {self.status}")
            return True
        return force

    def _binned(self, executor, bounds, source: str = "commanded", channel: Optional[str] = None):
        """(binned image of every sample so far over `bounds`, number of samples) --
        identical to bin_image(executor.all_points(), ...), but incremental: running
        per-pixel sums/counts to which only the new waveforms are added, so a redraw
        doesn't gather and re-bin tens of millions of samples every time. Starts over
        when the bounds change (edge modes: the view grows with the scan) or the
        executor was reset."""
        # source: "commanded" (the drive volts sent -- the first row) or "sensor" (the measured
        # positions, sensors.SensorCalibration -- the sensor row; waveforms without them skipped)
        n = self.cfg.n_image_pixels
        hist = executor.history
        key = (tuple(float(b) for b in bounds), n)
        bins = getattr(self, "_bins", None)
        if bins is None:
            bins = self._bins = {}
        bkey = source if channel is None else (source, channel)
        B = bins.get(bkey)
        if B is None or key != B["key"] or len(hist) < B["done"]:
            B = bins[bkey] = dict(key=key, done=0, npts=0, sum=np.zeros(n * n), cnt=np.zeros(n * n))
        for st in hist[B["done"]:]:
            if len(st.z) == 0:
                continue
            if source == "sensor":
                if st.x_sensor is None:
                    continue
                x, y = st.x_sensor, st.y_sensor
            else:
                x, y = st.x, st.y
            z = st.z
            if channel is not None:    # an extra signal of the same samples (PFM amplitude / phase)
                if not st.channels or channel not in st.channels:
                    continue
                z = st.channels[channel]
            if source == "sensor":     # measured positions can fall outside the frame: drop them
                m = (x >= bounds[0]) & (x <= bounds[1]) & (y >= bounds[2]) & (y <= bounds[3])
                x, y, z = x[m], y[m], z[m]
            flat = pixel_index(x, y, n, bounds)
            B["cnt"] += np.bincount(flat, minlength=n * n)
            B["sum"] += np.bincount(flat, weights=z, minlength=n * n)
            B["npts"] += len(z)
        B["done"] = len(hist)
        with np.errstate(invalid="ignore", divide="ignore"):
            img = np.where(B["cnt"] > 0, B["sum"] / B["cnt"], np.nan)
        img = img.reshape(n, n)
        return (img if channel is not None else self._flatten(img, bounds)), B["npts"]

    # the planner whose fitted substrate (planner.substrate, substrate.SubstrateModel)
    # is removed from the images -- set by modes.build for edge_follow / edge_explorer
    substrate_source = None

    def _substrate(self):
        if not getattr(self.cfg, "flatten_substrate", False) or self.substrate_source is None:
            return None
        m = getattr(self.substrate_source, "substrate", None)
        return m if m is not None and m.fitted and m.degree >= 1 else None

    def _flatten(self, img, bounds):
        """img minus the fitted substrate at the pixel centres (pixel_index's grid) --
        exact for the binned mean, since the surface is evaluated per pixel. The
        plane changes as the scan goes, so it's applied at draw time, never binned in."""
        m = self._substrate()
        if m is None or img is None:
            return img
        n = img.shape[0]
        X, Y = np.meshgrid(np.linspace(bounds[0], bounds[1], n), np.linspace(bounds[2], bounds[3], n))
        return img - m.plane(X, Y)

    def _image_label(self) -> str:
        return "Binned AI0, tilt removed" if self._substrate() is not None else "Binned AI0"

    def _recon_from_binned(self, executor, binned, bounds, source: str = "commanded"):
        """The reconstruction reconstruct_image would give, from the binned image --
        except "interpolated", which needs the raw points."""
        if self.cfg.reconstruction == "interpolated":
            pts = executor.all_points() if source == "sensor" else np.column_stack([
                np.concatenate([st.x for st in executor.history]), np.concatenate([st.y for st in executor.history]),
                np.concatenate([st.z for st in executor.history])])
            return reconstruct_image(pts[:, :2], pts[:, 2], bounds, self.cfg, self._rng)[1]
        return inpaint_binned(binned, bounds, self.cfg, self._rng)

    def _record_frame(self, step_index: int, step) -> None:
        if self.keep_frames and self._last_png is not None:
            title = self.fig._suptitle.get_text() if self.fig._suptitle is not None else ""
            self.frames.append(dict(step=step_index, kind=step.kind if step is not None else "final",
                                    title=title, png=self._last_png))

    def show_frames(self, every: int = 1, skip_connectors: bool = False, last: bool = True) -> None:
        """Display the kept frames one after another, each under a caption with its step
        number and what that step was: every `every`-th frame (the final one always,
        with last=True); skip_connectors=True leaves out the moves between scans."""
        from IPython.display import display, Image
        frames = [f for f in self.frames if not (skip_connectors and f["kind"] == "connector")]
        if not frames:
            print("No frames kept -- set live.every_step = live.keep_frames = True before running")
            return
        idx = list(range(0, len(frames), max(1, int(every))))
        if last and idx[-1] != len(frames) - 1:
            idx.append(len(frames) - 1)
        size = sum(len(frames[i]["png"]) for i in idx) / 1e6
        print(f"Step-by-step replay: {len(idx)} of {len(self.frames)} frames "
              f"({size:.0f} MB of images in this cell's output)")
        for i in idx:
            f = frames[i]
            print(f"\n=== step {f['step']} ({f['kind']}) === {f['title']}")
            display(Image(data=f["png"]))

    def save_frames(self, folder) -> None:
        """Write every kept frame as step_NNNN_<kind>.png into `folder` (created)."""
        import pathlib
        d = pathlib.Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        for f in self.frames:
            (d / f"step_{f['step']:04d}_{f['kind']}.png").write_bytes(f["png"])
        print(f"Saved {len(self.frames)} frames to {d.resolve()}")

    def save_gif(self, path, fps: float = 4.0, every: int = 1, skip_connectors: bool = False,
                 max_width: int = 1200, hold_last_s: float = 3.0):
        """Write the kept frames as one animated GIF (loops forever): every `every`-th
        frame (the final one always), `fps` frames per second, the last one held
        `hold_last_s`; skip_connectors=True leaves out the moves between scans. Frames
        are scaled down to at most `max_width` pixels wide (GIF files get big).
        Returns the path written."""
        import io
        import pathlib
        from PIL import Image as PILImage
        frames = [f for f in self.frames if not (skip_connectors and f["kind"] == "connector")]
        if not frames:
            print("No frames kept -- set live.every_step = live.keep_frames = True before running")
            return None
        idx = list(range(0, len(frames), max(1, int(every))))
        if idx[-1] != len(frames) - 1:
            idx.append(len(frames) - 1)
        images, size = [], None
        for i in idx:
            im = PILImage.open(io.BytesIO(frames[i]["png"])).convert("RGB")
            if size is None:  # every frame the same size as the first (scaled to max_width)
                w, h = im.size
                k = min(1.0, max_width / w)
                size = (int(round(w * k)), int(round(h * k)))
            if im.size != size:
                im = im.resize(size, PILImage.LANCZOS)
            images.append(im.quantize(colors=256, method=PILImage.MEDIANCUT, dither=PILImage.NONE))
        p = pathlib.Path(path)
        if p.suffix.lower() != ".gif":
            p = p.with_suffix(".gif")
        p.parent.mkdir(parents=True, exist_ok=True)
        step_ms = int(round(1000 / max(fps, 0.01)))
        durations = [step_ms] * (len(images) - 1) + [int(round(1000 * max(hold_last_s, step_ms / 1000)))]
        images[0].save(p, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False)
        print(f"Saved a {len(images)}-frame GIF ({size[0]} x {size[1]} px, {fps:g} fps, "
              f"{p.stat().st_size / 1e6:.1f} MB) to {p.resolve()}")
        return p

    # ------------------------------------------------------------------
    # figure: 1 x 3, or 2 x 3 with the equivalent-raster row -- + a column for the
    # uncertainty map (cfg.show_uncertainty) and one for the true error (simulation,
    # cfg.show_true_error), in every row
    # ------------------------------------------------------------------
    def _make_axes(self, compare, layout="constrained", extras: bool = True, sensor_row: bool = False,
                   channel_rows=None):
        self.compare = compare
        self.sensor_row = bool(sensor_row)
        self.channel_rows = [dict(c) for c in (channel_rows or [])]
        names = []
        if extras and getattr(self.cfg, "show_uncertainty", False):
            names.append("unc")
        if extras and self.ground_truth_fn is not None and getattr(self.cfg, "show_true_error", False):
            names.append("err")
        nc = 3 + len(names)
        w, h = self.cfg.figsize
        w = w * nc / 3          # cfg.figsize is for 3 columns
        nr = 1 + int(self.sensor_row) + len(self.channel_rows) + int(compare is not None)
        if nr == 1:
            self.fig, axes = plt.subplots(1, nc, figsize=(w, h), layout=layout)
            axes = list(axes)
            self.extra_axes = dict(zip(names, axes[3:]))
            return axes[:3]
        self.fig, axes = plt.subplots(nr, nc, figsize=(w, nr * h), layout="constrained")
        rows = [list(r) for r in axes]
        self.extra_axes = dict(zip(names, rows[0][3:]))
        if self.sensor_row:           # row 2: the measured positions
            self.sensor_axes = rows[1][:3]
            self.sensor_extra = dict(zip(names, rows[1][3:]))
        for j, c in enumerate(self.channel_rows):   # then one row per extra signal
            r = rows[1 + int(self.sensor_row) + j]
            c["axes"], c["extra"] = r[:3], dict(zip(names, r[3:]))
        if compare is not None:       # last row: the classic raster
            self.compare_axes = rows[-1][:3]
            self.compare_extra = dict(zip(names, rows[-1][3:]))
        return rows[0][:3]

    def _recon_title(self, mode=None):
        return RECON_TITLES.get(mode or self.cfg.reconstruction, "Reconstruction")

    def _raster_recon_title(self):
        return self._recon_title(getattr(self.compare, "reconstruction", None))

    def _z_label(self) -> str:
        """The colour bar's label: Z (nm) when the run converts Z to nm (cfg.z_signal.units)."""
        if getattr(self, "z_label_text", None):
            return self.z_label_text
        return "Z (nm)" if self.z_unit == "nm" else "AI0 (V)"

    def _draw_image(self, ax, img, bounds, title, vmin=None, vmax=None, cmap=None, cbar_label=None):
        cbar_label = cbar_label or self._z_label()
        # remove the previous colorbar BEFORE ax.clear() -- see _draw_readback_scatter
        cb = self._cbars.pop(id(ax), None)
        if cb is not None:
            cb.remove()
        ax.clear()
        xmin, xmax, ymin, ymax = bounds
        if img is not None:
            im = ax.imshow(img, origin="lower", extent=bounds, cmap=cmap or self.cfg.cmap,
                           interpolation="nearest", vmin=vmin, vmax=vmax)
            self._cbars[id(ax)] = self.fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X (V)")
        ax.set_ylabel("Y (V)")
        ax.set_title(title)

    def _draw_extras(self, axes, binned, recon, bounds, who: str = "") -> None:
        """The extra columns of one row: the uncertainty map of its binned image, and (with a
        ground truth) the real error of its reconstruction."""
        if not axes:
            return
        if "unc" in axes:
            scale = self.cfg.uncertainty_scale_V
            u = None if binned is None else 100.0 * uncertainty_map(binned, bounds, scale)
            what = f"{scale:.3g} V" if scale else "5% of the width"
            title = f"{who}Uncertainty (100% = {what} from data)"
            if u is not None:
                title += f"\nmean {np.mean(u):.0f}%, {np.mean(u >= 99.9):.0%} of pixels fully unknown"
            self._draw_image(axes["unc"], u, bounds, title, 0.0, 100.0, cmap="magma", cbar_label="uncertainty (%)")
        if "err" in axes:
            err, title, top = None, f"{who}True error |recon - truth|", None
            if recon is not None:
                n = recon.shape[0]
                X, Y = np.meshgrid(np.linspace(bounds[0], bounds[1], n), np.linspace(bounds[2], bounds[3], n))
                truth = self._flatten(np.asarray(self.ground_truth_fn(X, Y), dtype=float), bounds)
                err = np.abs(np.asarray(recon, dtype=float) - truth)
                rng = float(np.nanmax(truth) - np.nanmin(truth)) or 1.0
                top = 0.5 * rng
                ok = np.isfinite(err)
                if ok.any():
                    title += f"\n{np.mean(err[ok] > 0.2 * rng):.1%} of pixels off by > 20% of the height range"
            self._draw_image(axes["err"], err, bounds, title, 0.0, top, cmap="magma", cbar_label=f"|error| ({self.z_unit})")

    def _accumulate_path(self, step) -> None:
        """Tip path so far (every waveform, moves included) -- for match="path"."""
        if self.compare is not None and step is not None and len(step.x) > 1:
            self._mode_path_V += float(np.hypot(np.diff(step.x), np.diff(step.y)).sum())

    def _draw_sensor_empty(self) -> None:
        if not self.sensor_axes:
            return
        b = self._bounds([])
        axp, axb, axr = self.sensor_axes
        self._draw_path(axp, [], "Measured (sensors) -- waiting for a calibration", sensor=True)
        self._draw_image(axb, None, b, "Binned AI0 at the measured positions")
        self._draw_image(axr, None, b, f"{self._recon_title()} (measured positions)")
        self._draw_extras(self.sensor_extra, None, None, b, "Measured: ")

    def _update_sensor_row(self, executor, bounds, path: bool, recon: bool) -> bool:
        """Row 2: path / binned / reconstruction from the sensors' positions (um axes)."""
        if not self.sensor_axes:
            return False
        if self.sensor_cal is None:
            if path:
                self._draw_path(self.sensor_axes[0], [], "Measured (sensors) -- not calibrated: setup cell", sensor=True)
            return path
        axp, axb, axr = self.sensor_axes
        hist = executor.history
        vmin, vmax = self._clim()
        drew = False
        if path:
            self._draw_path(axp, hist, "Measured tip path (sensors)", sensor=True)
            binned, npts = self._binned(executor, bounds, source="sensor")
            self.last_binned_sensor = binned
            self._draw_image(axb, binned, bounds, f"{self._image_label()}, measured positions ({npts:,} pts)",
                             vmin, vmax)
            drew = True
        if recon:
            binned, npts = self._binned(executor, bounds, source="sensor")
            if npts >= self.cfg.min_points_for_reconstruction:
                self.last_binned_sensor = binned
                self.last_recon_sensor = self._recon_from_binned(executor, binned, bounds, source="sensor")
                self._draw_image(axr, self.last_recon_sensor, bounds,
                                 f"{self._recon_title()} (measured positions)", vmin, vmax)
                self._draw_extras(self.sensor_extra, binned, self.last_recon_sensor, bounds, "Measured: ")
                drew = True
        return drew

    def _draw_compare_empty(self) -> None:
        if self.compare_axes is None:
            return
        b = self._bounds([])
        axp, axb, axr = self.compare_axes
        self._draw_raster_path(axp, None, b)
        self._draw_image(axb, None, b, "Classic raster: binned AI0")
        self._draw_image(axr, None, b, f"Classic raster: {self._raster_recon_title()}")
        self._draw_extras(self.compare_extra, None, None, b, "Classic raster: ")

    def _draw_raster_path(self, ax, info, bounds) -> None:
        ax.clear()
        xmin, xmax, ymin, ymax = bounds
        self._draw_ground_truth(ax, bounds)
        title = "Classic raster -- waiting to start"
        if info is not None:
            self.compare.draw_path(ax, info)
            what = "pixels" if info["match"] == "pixels" else "tip path"
            title = f"Classic raster, same {what}: {info['n_lines']} line(s)"
            if info["n_lines"]:
                ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X (V)")
        ax.set_ylabel("Y (V)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _update_compare(self, executor, mode_binned=None, mode_bounds=None) -> None:
        """Second row: the classic raster with the same measured pixels (or tip path)
        as the self-driving scan so far, on the same pixel grid, same color scale."""
        bounds = self._bounds(executor.history)
        n = self.cfg.n_image_pixels
        if mode_binned is None or mode_bounds is None or tuple(mode_bounds) != tuple(bounds):
            mode_binned = self._binned(executor, bounds)[0] if executor.history else None
        n_meas = int(np.isfinite(mode_binned).sum()) if mode_binned is not None else 0
        vmin = vmax = None
        if mode_binned is not None and np.isfinite(mode_binned).any():
            vmin, vmax = float(np.nanmin(mode_binned)), float(np.nanmax(mode_binned))
        info = self.compare.build(bounds, n, n_measured=n_meas, path_V=self._mode_path_V)
        if self._substrate() is not None:  # same tilt removal as the mode's images
            info = dict(info, binned=self._flatten(info["binned"], bounds), recon=self._flatten(info["recon"], bounds))
        self.last_compare = info
        axp, axb, axr = self.compare_axes
        self._draw_raster_path(axp, info, bounds)
        self._draw_image(axb, info["binned"], bounds,
                         f"Classic raster: binned AI0 ({info['n_pixels']:,} px, mode: {n_meas:,})", vmin, vmax)
        self._draw_image(axr, info["recon"], bounds,
                         f"Classic raster: {self._raster_recon_title()} (~{info['time_s']:.2g} s scan)", vmin, vmax)
        self._draw_extras(self.compare_extra, info["binned"], info["recon"], bounds, "Classic raster: ")

    # ------------------------------------------------------------------
    # "What is it doing right now" status line (figure title)
    # ------------------------------------------------------------------
    def on_before_step(self, kind: str, meta: dict, executor) -> None:
        """Pass as ContinuousExecutor(on_before_step=...): runs right BEFORE each
        waveform is sent, so the title shows the scan that's running now (spiral
        search, circle N of feature X, the move between them) rather than the one
        that just finished."""
        label = meta.get("label", kind)
        self.status = label
        self.log.append(f"[{executor.step_index + 1}] {label}")
        if not self.cfg.enabled or self.every_step:
            return  # every-step mode: the step's frame is drawn once it's measured (on_step)
        self._set_status(f"Now: {label}")
        if self._budget_ok(False):
            t_start = self._time.perf_counter()
            self._redraw()
            self._record_update(t_start)

    # ------------------------------------------------------------------
    # Time budget: redrawing (rendering the figure to PNG) takes ~0.2 s, so with
    # fast segments a fixed "every N s" throttle can still spend most of the run
    # redrawing (seen with the edge explorer: 89 s of redraws for 0.4 s of scan
    # planning). Instead, wait at least cost * (1/f - 1) after the END of the last
    # update, so the live view uses at most a fraction f (max_time_fraction) of the
    # run -- and never more often than min_redraw_interval_s.
    # ------------------------------------------------------------------
    def _budget_ok(self, force: bool) -> bool:
        if force:
            return True
        f = min(max(self.cfg.max_time_fraction, 0.01), 1.0)
        wait = max(self.cfg.min_redraw_interval_s, self._last_update_cost * (1.0 / f - 1.0))
        return self._time.perf_counter() - self._last_update_end >= wait

    def _record_update(self, t_start: float) -> None:
        now = self._time.perf_counter()
        self._last_update_cost = now - t_start
        self._last_update_end = now

    def _set_status(self, text: str) -> None:
        self.fig.suptitle(text, fontsize=11, fontweight="bold")

    def _grid(self, n=150):
        return build_grid(self.extent, n)

    def _bounds(self, history, pad_frac: float = 0.15):
        """Square (xmin, xmax, ymin, ymax) around everywhere the tip has actually
        been, with padding -- grows as the scan covers more area, instead of every
        panel being pinned to the fixed `extent` the view was constructed with
        (which is just a guess at construction time, and often doesn't match the
        actual search/path radii once tuned). Falls back to +/-`self.extent` before
        any data exists, so the initial empty frame still has sane axes.
        cfg.fixed_bounds_V set (the priors set it to the area to explore): always that
        area, in every frame, however much of it has been scanned."""
        fb = getattr(self.cfg, "fixed_bounds_V", None)
        if fb is not None:
            x0, x1, y0, y1 = (float(v) for v in fb)
            p = 0.02 * max(x1 - x0, y1 - y0)
            return x0 - p, x1 + p, y0 - p, y1 + p
        return bounds_from_history(history, self.extent, pad_frac)

    def _dynamic_grid(self, history, n: int = 150):
        """(Xg, Yg) mesh over `_bounds(history)` -- the moving-window equivalent of
        `_grid()`/`build_grid()`, which is pinned to the fixed `extent`."""
        return grid_from_bounds(self._bounds(history), n)

    def _due(self, last_attr: str, min_interval_s: float, force: bool) -> bool:
        if force:
            return True
        now = self._time.monotonic()
        if now - getattr(self, last_attr) < min_interval_s:
            return False
        setattr(self, last_attr, now)
        return True

    def _draw_ground_truth(self, ax, bounds, n: int = 200) -> None:
        """Simulation only: the true topography faintly under the path. An image, not a
        pcolormesh -- the same picture, but a 200 x 200 semi-transparent mesh took ~0.3 s
        to render on every frame."""
        if self.ground_truth_fn is None:
            return
        xmin, xmax, ymin, ymax = bounds
        Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
        ax.imshow(self.ground_truth_fn(Xg, Yg), origin="lower", extent=bounds, cmap=self.cfg.cmap,
                  alpha=0.5, interpolation="nearest")

    def _draw_path(self, ax, history, title, max_points_per_step: int = 3000, sensor: bool = False):
        """Real hardware at a slow tip velocity can put millions of samples in a
        single step (see VelocityConfig) -- plotting every one of them is a shape
        matplotlib can't render live at any reasonable frame rate. Stride each
        step's own line down to at most max_points_per_step; this only thins the
        drawn line, never the underlying data (all of it still reaches
        ContinuousExecutor.history / the reconstruction panel)."""
        ax.clear()
        xmin, xmax, ymin, ymax = self._bounds(history)
        self._draw_ground_truth(ax, (xmin, xmax, ymin, ymax))
        seen = set()
        for step in history:
            style = dict(_KIND_STYLE.get(step.kind, dict(color="k", linewidth=1.0)))
            if step.kind in seen:
                style.pop("label", None)
            seen.add(step.kind)
            sx, sy = (step.x_sensor, step.y_sensor) if sensor else (step.x, step.y)
            if sx is None or len(sx) == 0:
                continue
            stride = max(1, len(sx) // max_points_per_step)
            ax.plot(sx[::stride], sy[::stride], **style)
        last = next((st for st in reversed(history) if (st.x_sensor if sensor else st.x) is not None
                     and len(st.x)), None)
        if last is not None:
            lx, ly = (last.x_sensor, last.y_sensor) if sensor else (last.x, last.y)
            ax.plot(lx[-1], ly[-1], "ro", markersize=7, label="tip")
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X (V)")
        ax.set_ylabel("Y (V)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    def _draw_readback_scatter(self, ax, history, title):
        """Middle column: every raw measured point, plotted spatially (same X/Y axes
        and bounds as the path panel) and colored by its own Z/AI0 value -- the
        sparse, un-inpainted counterpart to the reconstruction panel's gridded/
        interpolated version of the same data. Reuses `subsample_for_reconstruction`
        (same cap/recency-weighting the reconstruction panel uses) since a full raw
        point cloud can be millions of samples (see VelocityConfig)."""
        # remove the previous colorbar BEFORE ax.clear() -- Colorbar.remove() needs
        # to restore the host axes' pre-colorbar subplot geometry, using bookkeeping
        # that ax.clear() wipes; removing it after clearing raises AttributeError.
        if getattr(self, "_readback_cbar", None) is not None:
            self._readback_cbar.remove()
            self._readback_cbar = None
        ax.clear()
        xmin, xmax, ymin, ymax = self._bounds(history)
        if history:
            xs = np.concatenate([s.x for s in history])
            ys = np.concatenate([s.y for s in history])
            zs = np.concatenate([s.z for s in history])
            xy, z = subsample_for_reconstruction(np.column_stack([xs, ys]), zs,
                                                  self.cfg.max_reconstruction_points, self._rng)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=z, cmap=self.cfg.cmap, s=4, linewidths=0)
            self._readback_cbar = self.fig.colorbar(sc, ax=ax, label=f"Z / AI0 ({self.z_unit}, measured)")
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("X (V)")
        ax.set_ylabel("Y (V)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    def _redraw(self):
        """Push a fresh static PNG of the figure into the currently-executing cell's
        output, replacing the previous one -- deliberately NOT relying on the
        interactive ipympl/"widget" canvas's own redraw pipeline.

        That pipeline is a request/response handshake: draw_idle() just sends the
        browser a "please redraw" notice, and the browser then has to send a
        follow-up request *back* to the kernel asking for the actual updated pixels.
        The kernel can only answer that while its event loop is free -- but for the
        whole duration of a hardware step, it's blocked inside
        PyScannerBackend.measure_path's synchronous ActiveX polling loop
        (_wait_while), which never yields. So the handshake can't complete until the
        entire run finishes, no matter what combination of draw_idle()/
        flush_events()/plt.pause() is called -- confirmed by inspecting
        FigureCanvasNbAgg directly: it doesn't even override flush_events(), it's
        matplotlib's documented base no-op.

        IPython's display()/clear_output(wait=True), by contrast, is a one-way
        server-to-browser push (the same mechanism a live-updating print() or
        progress bar relies on) -- it doesn't need anything back from the browser,
        so it keeps working even while the kernel is otherwise fully blocked.
        """
        import io
        from IPython.display import display, clear_output, Image

        self._um_ticks()
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=100)
        self._last_png = buf.getvalue()
        clear_output(wait=True)
        display(Image(data=self._last_png))

    # position sensors: every panel's axes labelled in micrometres (the data stays in drive
    # volts -- the calibrated frame, sensors.SensorCalibration; set by on_step from the executor)
    sensor_cal = None
    # Z's unit ("V" or "nm": cfg.z_signal.units, the executor's z_unit -- set by on_step)
    z_unit = "V"

    def _um_ticks(self) -> None:
        cal = self.sensor_cal
        if cal is None or not self.sensor_row:
            return
        ch_axes = [a for c in self.channel_rows if c.get("source") == "sensor"
                   for a in list(c["axes"]) + list(c["extra"].values())]
        for ax in list(self.sensor_axes or []) + list(self.sensor_extra.values()) + ch_axes:
            um_axis_ticks(ax, cal)
            # 1 um the same length along x and y (the data are drive volts: 1 V = k um per axis)
            ax.set_aspect(abs(cal.y.k / cal.x.k), adjustable="box")
            if ax.get_xlabel():
                ax.set_xlabel("X (um, sensor)")
            if ax.get_ylabel():
                ax.set_ylabel("Y (um, sensor)")

    def close(self):
        plt.close(self.fig)


class LiveEdgeFollowView(_LiveViewBase):
    """Live view for EdgeFollowPlanner, three columns, all with the SAME X/Y axis
    limits (the path panel's dynamic `_bounds` -- grows to fit wherever the tip has
    actually been):
      1. the path so far in XY (spiral / edge trace / transit, color-coded)
      2. binned AI0 -- PyScanner-style image (mean per pixel, unmeasured pixels
         blank; same convention as image_AI0, see reconstruction.bin_image)
      3. the reconstruction: column 2's unmeasured pixels filled in with
         cfg.reconstruction ("compressed_sensing" / "gp" / "interpolated")
    Columns 2 and 3 share one color scale (the measured range), so they're directly
    comparable. The latest arrays are kept on the view (`last_binned`,
    `last_recon`, `last_bounds`) for the notebook's save cell.
    """

    path_title = "Edge-follow scan"

    def __init__(self, cfg: LiveViewConfig, extent: float = 0.5, ground_truth_fn=None, compare=None,
                 sensor_row: bool = False, channel_rows=None):
        """compare: a simulation.EquivalentRaster -> a row with the classic raster with the
        same budget (Simulation.ipynb); None (the instrument) -> no such row.
        sensor_row (cfg.position_sensors.enabled): a second row, the same panels built from
        the measured positions, axes in um -- the first row stays the commanded drive volts.
        channel_rows: extra signals recorded with z (PFM: the amplitude when the phase is z),
        a row each -- [dict(name, label, cmap)]: that signal along the last waveform | binned |
        reconstruction | (uncertainty), at the measured positions when there are sensors."""
        super().__init__(cfg, extent, ground_truth_fn)
        # constrained layout: colorbars don't squash panels
        self.ax_path, self.ax_binned, self.ax_recon = self._make_axes(compare, sensor_row=sensor_row,
                                                                      channel_rows=channel_rows)
        self.last_binned = self.last_recon = self.last_bounds = None
        plt.close(self.fig)  # never let the interactive backend auto-display this --
        # _redraw() pushes it explicitly (see _LiveViewBase._redraw's docstring)
        self._draw_path(self.ax_path, [], f"{self.path_title} -- waiting to start")
        self._draw_image(self.ax_binned, None, self._bounds([]), "Binned AI0")
        self._draw_image(self.ax_recon, None, self._bounds([]), self._recon_title())
        self._draw_extras(self.extra_axes, None, None, self._bounds([]))
        self._draw_sensor_empty()
        for c in self.channel_rows:
            self._draw_channel_row(c, None, None)
        self._draw_compare_empty()
        self._redraw()  # initial frame, shown here (the "Build" cell) before Run

    def _clim(self):
        if self.last_binned is None or not np.isfinite(self.last_binned).any():
            return None, None
        return float(np.nanmin(self.last_binned)), float(np.nanmax(self.last_binned))

    def on_step(self, step_index: int, step, executor, force: bool = False) -> None:
        self._accumulate_path(step)  # every step, even ones not redrawn
        self.sensor_cal = getattr(executor, "sensor_cal", None)
        self.z_unit = getattr(executor, "z_unit", "V")
        self.z_label_text = getattr(executor, "z_label", None)
        if not self.cfg.enabled:
            return
        force = self._begin_step(step_index, step, force)
        if force is None or not self._budget_ok(force):
            return
        t_start = self._time.perf_counter()
        redraw_needed = False
        history = executor.history

        path_due = self._due("_last_path_draw", self.cfg.min_redraw_interval_s, force)
        if path_due:
            self._draw_path(self.ax_path, history, f"{self.path_title} -- step {step_index}"
                            + (" (commanded)" if self.sensor_row else ""))
            self._decorate_path(self.ax_path)
            if history:
                bounds = self._bounds(history)
                # every point, no subsampling -- binned incrementally (see _binned)
                self.last_binned, npts = self._binned(executor, bounds)
                self.last_bounds = bounds
                vmin, vmax = self._clim()
                self._draw_image(self.ax_binned, self.last_binned, bounds,
                                 f"{self._image_label()} ({npts:,} pts)", vmin, vmax)
            redraw_needed = True

        recon_due = self._due("_last_recon_draw", self.cfg.min_reconstruction_interval_s, force)
        if recon_due:
            bounds = self._bounds(history)
            binned, npts = self._binned(executor, bounds) if history else (None, 0)
            if npts >= self.cfg.min_points_for_reconstruction:
                self.last_binned = binned
                self.last_recon = self._recon_from_binned(executor, binned, bounds)
                self.last_bounds = bounds
                vmin, vmax = self._clim()
                self._draw_image(self.ax_recon, self.last_recon, bounds, self._recon_title(), vmin, vmax)
                self._draw_extras(self.extra_axes, self.last_binned, self.last_recon, bounds)
                redraw_needed = True

        if self.sensor_row and history and (path_due or recon_due):
            redraw_needed |= self._update_sensor_row(executor, self._bounds(history), path_due, recon_due)
        if self.channel_rows and history and (path_due or recon_due):
            for c in self.channel_rows:
                redraw_needed |= self._update_channel_row(c, executor, self._bounds(history), recon_due)

        if self.compare is not None and self._due("_last_compare_draw", self.cfg.min_reconstruction_interval_s,
                                                  force):
            self._update_compare(executor, self.last_binned, self.last_bounds)
            redraw_needed = True

        if redraw_needed:
            self._redraw()
            self._record_frame(step_index, step)
        self._record_update(t_start)

    def _decorate_path(self, ax) -> None:
        """Hook for subclasses: extra drawing on the path panel after each redraw."""

    def _channel_clim(self, c, img):
        if c.get("vmin") is not None:
            return c["vmin"], c["vmax"]
        if img is None or not np.isfinite(img).any():
            return None, None
        return tuple(float(v) for v in np.nanpercentile(img, [1, 99]))

    def _draw_channel_row(self, c, executor, bounds, recon: bool = True) -> None:
        axl, axb, axr = c["axes"]
        b = bounds or self._bounds([])
        sensor = c.get("source") == "sensor"
        where = "measured" if sensor else "commanded"
        # 1: the path, as in the rows above (commanded, or the sensors' measured one)
        self._draw_path(axl, executor.history if executor is not None else [],
                        f"{c['name'].capitalize()}: " + ("measured path (sensors)" if sensor else "path (commanded)"),
                        sensor=sensor)
        if executor is None:
            self._draw_image(axb, None, b, f"Binned {c['name']}, {where}", cmap=c["cmap"], cbar_label=c["label"])
            self._draw_image(axr, None, b, f"{self._recon_title()} ({c['name']}, {where})", cmap=c["cmap"],
                             cbar_label=c["label"])
            self._draw_extras(c["extra"], None, None, b, f"{c['name']}: ")
            return
        binned, npts = self._binned(executor, b, source=c.get("source", "commanded"), channel=c["name"])
        vmin, vmax = self._channel_clim(c, binned)
        self._draw_image(axb, binned, b, f"Binned {c['name']}, {where} ({npts:,} pts)", vmin, vmax,
                         cmap=c["cmap"], cbar_label=c["label"])
        if recon and npts >= self.cfg.min_points_for_reconstruction:
            rec = self._recon_from_binned(executor, binned, b) if self.cfg.reconstruction != "interpolated" \
                else inpaint_binned(binned, b, self.cfg, self._rng, mode="linear")
            c["last_recon"] = rec
            self._draw_image(axr, rec, b, f"{self._recon_title()} ({c['name']}, {where})", vmin, vmax,
                             cmap=c["cmap"], cbar_label=c["label"])
            self._draw_extras(c["extra"], binned, rec, b, f"{c['name'].capitalize()}: ")
        c["last_binned"] = binned

    def _update_channel_row(self, c, executor, bounds, recon: bool) -> bool:
        if c.get("source") == "sensor" and self.sensor_cal is None:
            c = dict(c, source="commanded")
        self._draw_channel_row(c, executor, bounds, recon)
        return True

    def finalize(self, features=None, executor=None, status: Optional[str] = None):
        """Force one last full redraw (and reconstruction) at the end of a run."""
        if status is not None:
            self._set_status(status)
        elif features is not None:
            n_closed = sum(f.closed_loop for f in features)
            self._set_status(f"Finished -- {len(features)} feature(s) traced "
                             f"({n_closed} closed loop{'s' if n_closed != 1 else ''})")
        if executor is not None:
            self.on_step(executor.step_index, None, executor, force=True)


class LiveGridRefineView(LiveEdgeFollowView):
    """Live view for GridRefinePlanner: the same three columns (path | binned AI0 |
    reconstruction from ALL the data so far), with the tiles of the latest
    refinement level outlined in red on the path panel. Set `.planner` after
    building the planner."""

    path_title = "Grid + refine scan"
    planner = None

    def _bounds(self, history, pad_frac: float = 0.15):
        # the scanned square (fixed for the whole run), not the data's extent + padding --
        # nothing is measured outside it, so reconstructing there only invents structure
        pl = self.planner
        if pl is not None and pl.bounds is not None:
            x0, x1, y0, y1 = pl.bounds
            p = 0.02 * max(x1 - x0, y1 - y0)
            return x0 - p, x1 + p, y0 - p, y1 + p
        return super()._bounds(history, pad_frac)

    def _decorate_path(self, ax) -> None:
        pl = self.planner
        if pl is None or not pl.levels or pl.levels[-1].index == 0:
            return
        from matplotlib.patches import Rectangle, Circle
        lv = pl.levels[-1]
        colors = {"raster": "red", "spiral": "tab:blue", "lissajous": "tab:green"}
        seen = set()
        for r in lv.rects:
            pat = r.pattern or lv.pattern
            col = colors.get(pat, "red")
            lab = f"level {lv.index}: {pat}" if pat not in seen else None
            seen.add(pat)
            if r.cells is None:
                ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, ec=col,
                                       lw=0.6, alpha=0.8, label=lab))
                continue
            # the region's tiles, shaded (darker = a feature's inside, added by fill_interiors)
            inner = r.interior if r.interior is not None else np.zeros(len(r.cells), dtype=bool)
            for (x0, x1, y0, y1), ins in zip(r.cells, inner):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=col, ec="none",
                                       alpha=0.28 if ins else 0.14))
            if pat == "spiral" and r.radius is not None:
                ax.add_patch(Circle(tuple(r.center), r.radius, fill=False, ec=col, lw=0.8, alpha=0.9, label=lab))
            else:
                ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, ec=col,
                                       lw=0.8, alpha=0.9, label=lab))
        ax.legend(loc="upper right", fontsize=8)


class LiveGPPathView(_LiveViewBase):
    """Live view for GPPathPlanner. Left panel: the path so far. Right two panels:
    the GP mean (reconstruction) and uncertainty surfaces, refreshed whenever the
    planner has re-fit the GP (same cadence as `GPPathConfig.refit_every`) rather
    than on every single point.
    """

    def __init__(self, cfg: LiveViewConfig, planner, extent: float = 0.5, ground_truth_fn=None, compare=None):
        """compare: see LiveEdgeFollowView."""
        super().__init__(cfg, extent, ground_truth_fn)
        self.planner = planner
        self.last_binned = self.last_recon = self.last_bounds = self.last_plane = None
        self.ax_path, self.ax_mean, self.ax_std = self._make_axes(compare, layout=None, extras=False)
        plt.close(self.fig)  # see LiveEdgeFollowView.__init__ / _redraw's docstring
        self._draw_path(self.ax_path, [], "GP path scan -- waiting to start")
        self._draw_compare_empty()
        self._redraw()  # initial frame, shown here (the "Build" cell) before Run

    def on_step(self, step_index: int, step, executor, force: bool = False) -> None:
        self._accumulate_path(step)  # every step, even ones not redrawn
        self.sensor_cal = getattr(executor, "sensor_cal", None)
        self.z_unit = getattr(executor, "z_unit", "V")
        self.z_label_text = getattr(executor, "z_label", None)
        if not self.cfg.enabled:
            return
        force = self._begin_step(step_index, step, force)
        if force is None or not self._budget_ok(force):
            return
        t_start = self._time.perf_counter()
        redraw_needed = False

        if self._due("_last_path_draw", self.cfg.min_redraw_interval_s, force):
            self._draw_path(self.ax_path, executor.history,
                             f"GP path scan -- step {step_index}")
            redraw_needed = True

        if self._due("_last_recon_draw", self.cfg.min_reconstruction_interval_s, force) and self.planner.fitted:
            xmin, xmax, ymin, ymax = self._bounds(executor.history)
            Xg, Yg = self._dynamic_grid(executor.history)
            mu, sigma = self.planner.predict_surface(Xg, Yg)
            self.ax_mean.clear()
            if getattr(self.cfg, "gp_view", "reconstruction") == "gp_mean":
                self.ax_mean.pcolormesh(Xg, Yg, mu, cmap=self.cfg.cmap, shading="auto")
                self.ax_mean.set_title("GP mean (planner's surrogate)")
            else:
                # the map: the measured samples, substrate plane removed, filled in with
                # cfg.reconstruction (TV: flat plateaus, sharp steps) -- full iterations at the end
                from .reconstruction import flattened_reconstruction
                pts = executor.all_points()
                if len(pts) >= self.cfg.min_points_for_reconstruction:
                    b = (xmin, xmax, ymin, ymax)
                    it = None if force else getattr(self.cfg, "live_tv_iterations", 300)
                    binned, recon, plane = flattened_reconstruction(pts[:, :2], pts[:, 2], b, self.cfg, n_iter=it)
                    self.last_binned, self.last_recon, self.last_bounds, self.last_plane = binned, recon, b, plane
                    ok = np.isfinite(binned)
                    lo, hi = (np.percentile(binned[ok], [1, 99.5]) if ok.sum() > 10 else (None, None))
                    self.ax_mean.imshow(recon, origin="lower", extent=b, cmap=self.cfg.cmap, vmin=lo, vmax=hi,
                                        interpolation="nearest")
                self.ax_mean.set_title(f"reconstruction ({self.cfg.reconstruction}, plane removed)")
            self.ax_mean.set_aspect("equal")
            self.ax_mean.set_xlim(xmin, xmax)
            self.ax_mean.set_ylim(ymin, ymax)

            self.ax_std.clear()
            self.ax_std.pcolormesh(Xg, Yg, sigma, cmap="hot", shading="auto")
            self.ax_std.set_aspect("equal")
            self.ax_std.set_xlim(xmin, xmax)
            self.ax_std.set_ylim(ymin, ymax)
            self.ax_std.set_title("GP uncertainty")
            redraw_needed = True

        if self.compare is not None and self._due("_last_compare_draw", self.cfg.min_reconstruction_interval_s,
                                                  force):
            self._update_compare(executor)
            redraw_needed = True

        if redraw_needed:
            self._redraw()
            self._record_frame(step_index, step)
        self._record_update(t_start)

    def finalize(self, segments=None, executor=None):
        if segments is not None:
            self._set_status(f"Finished -- {len(segments)} GP path segment(s)")
        if executor is not None:
            self.on_step(executor.step_index, None, executor, force=True)
