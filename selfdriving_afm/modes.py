"""
The scan modes, in ONE place -- used by both Run_SelfDriving_Scan.ipynb (instrument)
and Simulation.ipynb (fake / loaded topography), so what you test in simulation is
exactly the code that runs on the microscope: same planner, same executor, same live
view, same summary, same saved file. The only thing that differs is the backend.

    session = build("grid_refine", cfg, backend)     # executor + planner + live view
    run(session)                                      # blocks; prints the summary
    save(session, "scan_data")                        # .npz (+ .png) -- Analyze notebook reads it

Adding a new mode: write the planner (planners/), give it a config dataclass in
config.py (a field on ExperimentConfig), and add one `Mode(...)` entry to MODES below
with how to build it and how to summarise its result. Both notebooks pick it up.
"""

from __future__ import annotations
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from .config import ExperimentConfig
from .execution import ContinuousExecutor
from .planners.edge_follow_planner import EdgeFollowPlanner
from .planners.gp_path_planner import GPPathPlanner
from .planners.edge_explorer_planner import EdgeExplorerPlanner
from .planners.grid_refine_planner import GridRefinePlanner
from .pfm import DomainWallExplorerPlanner
from .planners.gp_edge_explorer_planner import GPEdgeExplorerPlanner
from .planners.hybrid_planners import GridTrackerPlanner, GPSpiralPlanner


@dataclass
class Session:
    modality: str
    cfg: ExperimentConfig
    backend: object
    executor: ContinuousExecutor
    planner: object
    live: Optional[object] = None      # a viz.Live*View, or None (no live view)
    result: object = None
    run_time_s: float = 0.0
    summary: str = ""
    extra: dict = field(default_factory=dict)
    priors: object = None              # the priors.ScanPriors this run used (saved with it)


# ------------------------------------------------------------------------------------
# summaries (exactly what the Run notebook printed before -- now shared)
# ------------------------------------------------------------------------------------
def _summary_edge_follow(s: Session) -> str:
    lines = [f"{len(s.result)} feature(s) found in {s.run_time_s:.1f}s, "
             f"{len(s.executor.all_points())} points measured"]
    for f in s.result:
        how = ("closed loop" if f.closed_loop else
               "open edge, lost at both ends" if f.edge_lost else "open, circle budget reached")
        if f.hit_scan_limit:
            how += " (edge runs past the scan range)"
        lines.append(f"  feature {f.id}: {len(f.circle_steps)} circles -- {how}")
    return "\n".join(lines)


def _summary_edge_explorer(s: Session) -> str:
    p = s.planner
    n_follow = sum(sg.mode == "follow" for sg in p.segments)
    plan_ms = 1000 * np.array(p.step_times_s[1:] or [0.0])
    lines = [f"{len(s.result)} feature(s) found in {s.run_time_s:.1f}s, {len(s.executor.all_points())} points measured, "
             f"{len(p.segments)} segments ({n_follow} following an edge); "
             f"choosing each exploring segment: median {np.median(plan_ms):.1f} ms, max {plan_ms.max():.1f} ms"]
    for f in s.result:
        how = ("closed at the area's border" if getattr(f, "closed_by_border", False) else
               "closed loop" if f.closed_loop else "open edge")
        if getattr(f, "interior_lines", 0):
            how += f", inside scanned ({f.interior_lines} lines)"
        elif getattr(f, "interior_note", ""):
            how += f", {f.interior_note}"
        if getattr(f, "inside_of", None) is not None:
            how += f" -- INSIDE feature {f.inside_of}"
        if getattr(f, "merged_ids", None):
            how += f" (joined with feature{'s' if len(f.merged_ids) > 1 else ''} "                    f"{', '.join(str(i) for i in f.merged_ids)}: the same edge, traced from two places)"
        lines.append(f"  feature {f.id}: {len(f.points)} edge crossings -- {how}")
    return "\n".join(lines)


def _summary_grid_refine(s: Session) -> str:
    result, p = s.result, s.planner
    scan_s = sum(lv.scan_time_s for lv in result)
    lines = [f"{len(result)} level(s) in {s.run_time_s:.1f}s ({scan_s:.1f}s of waveforms), "
             f"{len(s.executor.all_points()):,} points measured", p.summary()]
    # a full raster at the finest spacing, same peak tip speed (sinusoidal lines), for comparison
    w = p.bounds[1] - p.bounds[0]
    full = (w / result[-1].spacing_V + 1) * (0.5 * np.pi * w) / s.cfg.velocity.tip_velocity_V_s
    lines.append(f"a full raster at the finest spacing would take ~{full:.1f}s of waveforms "
                 f"({full / max(scan_s, 1e-9):.1f}x longer)")
    return "\n".join(lines)


def _summary_gp_path(s: Session) -> str:
    return (f"{len(s.result)} segment(s), {len(s.executor.all_points())} points measured, "
            f"{s.run_time_s:.1f}s\n{s.planner.timing_summary()}")


# ------------------------------------------------------------------------------------
# finalize the live view at the end of a run
# ------------------------------------------------------------------------------------
def _finalize_default(s: Session) -> None:
    s.live.finalize(s.result, s.executor)


def _finalize_grid_refine(s: Session) -> None:
    s.live.finalize(executor=s.executor, status=f"Finished -- {len(s.result) - 1} refinement level(s), "
                                                f"finest lines {s.result[-1].spacing_V:.3g} V apart")


@dataclass
class Mode:
    name: str
    planner_cls: type
    cfg_attr: str                     # the planner's config: getattr(ExperimentConfig, cfg_attr)
    live_view: str                    # "edge" | "gp" | "grid" -- which viz.Live*View
    summary: Callable[[Session], str]
    finalize: Callable[[Session], None] = _finalize_default
    description: str = ""

    def make_planner(self, executor, cfg: ExperimentConfig):
        if self.planner_cls is GridRefinePlanner:
            return GridRefinePlanner(executor, cfg.grid_refine, recon_cfg=cfg.live_view)
        if self.planner_cls in (GPEdgeExplorerPlanner, GridTrackerPlanner, GPSpiralPlanner):
            return self.planner_cls(executor, cfg.edge_explorer, cfg.gp_path)
        if self.planner_cls is DomainWallExplorerPlanner:
            return DomainWallExplorerPlanner(executor, cfg.edge_explorer, cfg.pfm)
        return self.planner_cls(executor, getattr(cfg, self.cfg_attr))


MODES: Dict[str, Mode] = {
    "edge_follow": Mode("edge_follow", EdgeFollowPlanner, "edge_follow", "edge", _summary_edge_follow,
                        description="spiral search -> chain of circles along each edge"),
    "gp_path": Mode("gp_path", GPPathPlanner, "gp_path", "gp", _summary_gp_path,
                    description="Gaussian-process / Bayesian continuous path"),
    "edge_explorer": Mode("edge_explorer", EdgeExplorerPlanner, "edge_explorer", "edge", _summary_edge_explorer,
                          description="straight segments: explore, follow each edge until closed, explore again"),
    "domain_wall_explorer": Mode("domain_wall_explorer", DomainWallExplorerPlanner, "edge_explorer", "edge",
                                 _summary_edge_explorer,
                                 description="PFM: the edge explorer following domain walls (phase flip / amplitude dip)"),
    "gp_edge_explorer": Mode("gp_edge_explorer", GPEdgeExplorerPlanner, "edge_explorer", "edge",
                             lambda s: _summary_edge_explorer(s) + chr(10) + s.planner.timing_summary(),
                             description="GP (IVR) exploration between features, edge explorer tracing on them"),
    "grid_tracker": Mode("grid_tracker", GridTrackerPlanner, "edge_explorer", "edge",
                         lambda s: _summary_edge_explorer(s),
                         description="level-0 sparse spiral (grid_refine), edge explorer tracing every outline it crossed"),
    "gp_spiral": Mode("gp_spiral", GPSpiralPlanner, "edge_explorer", "edge",
                      lambda s: _summary_edge_explorer(s) + chr(10) + s.planner.timing_summary(),
                      description="GP (IVR) exploration between objects, grid_refine level-1/2 spirals on each one"),
    "grid_refine": Mode("grid_refine", GridRefinePlanner, "grid_refine", "grid", _summary_grid_refine,
                        _finalize_grid_refine, description="one sparse scan -> detect -> refine regions (touching tiles), level by level"),
}


def list_modes() -> str:
    return "\n".join(f"  {m.name:14s} {m.description}" for m in MODES.values())


# ------------------------------------------------------------------------------------
# build / run / save
# ------------------------------------------------------------------------------------
def build(modality: str, cfg: ExperimentConfig, backend, live: bool = True,
          live_extent: float = 0.45, ground_truth_fn=None, compare=None, priors=None) -> Session:
    """Executor + planner (+ live view, if `live`) for `modality`, on `backend`.
    live=False: no live view at all (fast -- for comparisons and sweeps).
    compare: a simulation.EquivalentRaster -> the live view gets a second row with the
    classic raster with the same budget (Simulation.ipynb only).
    priors: the priors.ScanPriors already applied to cfg (priors.apply) -- kept with the
    session and saved with the run."""
    if modality not in MODES:
        raise ValueError(f"Unknown MODALITY: {modality!r} -- known modes:\n{list_modes()}")
    m = MODES[modality]
    executor = ContinuousExecutor(backend, cfg.safety, cfg.velocity, start_xy=cfg.execution.start_xy_V,
                                  position_sensors=cfg.position_sensors, readback=cfg.readback,
                                  z_signal=cfg.z_signal)
    if modality == "domain_wall_explorer" or getattr(cfg, "pfm", None) is not None and cfg.pfm.enabled:
        from . import pfm as _pfm
        cfg.pfm.enabled = True
        _pfm.configure(cfg, backend, executor)      # PFM amplitude / phase read with every waveform
    planner = m.make_planner(executor, cfg)
    view = None
    if live:
        from . import viz
        kw = dict(extent=live_extent, ground_truth_fn=ground_truth_fn, compare=compare)
        if m.live_view == "gp":
            view = viz.LiveGPPathView(cfg.live_view, planner, **kw)
        elif m.live_view == "grid":
            view = viz.LiveGridRefineView(cfg.live_view, sensor_row=cfg.position_sensors.enabled, **kw)
            view.planner = planner  # the live view outlines the current refinement tiles
        else:
            rows = None
            if getattr(cfg, "pfm", None) is not None and cfg.pfm.enabled:
                # PFM: the signal(s) other than z get a row each (z: the phase if on)
                from . import pfm as _pfm
                # (rows like z's: at the commanded positions, and at the measured ones with sensors)
                srcs = ["commanded"] + (["sensor"] if cfg.position_sensors.enabled else [])
                rows = [dict(name=k, label=f"{c.label} ({c.unit})", cmap=getattr(c, "cmap", None) or cfg.live_view.cmap,
                             source=src, vmin=-180.0 if k == "phase" else None, vmax=180.0 if k == "phase" else None)
                        for k, c in cfg.pfm.channels().items() if k != _pfm.primary(cfg.pfm) for src in srcs]
            view = viz.LiveEdgeFollowView(cfg.live_view, sensor_row=cfg.position_sensors.enabled,
                                          channel_rows=rows, **kw)
            view.substrate_source = planner  # images shown with the fitted tilt removed
        executor.on_step = view.on_step
        # status line: the live view's title shows the scan running RIGHT NOW; view.log
        # keeps one line per scan
        executor.on_before_step = view.on_before_step
    if priors is not None and hasattr(planner, "FEATURE_MAX_V"):
        # gp_spiral: spiral sizes, size checks and holes scale with the priors' feature sizes
        planner.FEATURE_MIN_V = float(priors.feature_min_V)
        planner.FEATURE_MAX_V = float(priors.feature_max_V)
    return Session(modality, cfg, backend, executor, planner, view, priors=priors)


def run(session: Session, verbose: bool = True):
    """Run the planner from cfg.execution.start_xy_V (blocks until done), finalize the
    live view, and print the mode's summary. Returns the planner's result."""
    s = session
    t0 = time.time()
    s.result = s.planner.run(start_xy=s.cfg.execution.start_xy_V)
    s.run_time_s = time.time() - t0
    if s.live is not None:
        MODES[s.modality].finalize(s)
    s.summary = MODES[s.modality].summary(s)
    if verbose:
        print(s.summary)
    return s.result


def reconstruction_arrays(session: Session, fallback_extent: float = 0.45) -> dict:
    """The reconstruction saved with a run: the live view's own final images when there
    is a live view (what you watched is what's saved), otherwise computed the same way
    (same bounds rule, same cfg.live_view method)."""
    from .viz import bounds_from_history, grid_from_bounds
    from .reconstruction import reconstruct_image
    s, cfg = session, session.cfg
    out = {}
    if MODES[s.modality].live_view in ("edge", "grid"):
        if s.live is not None and s.live.last_recon is not None:
            b, recon, binned = s.live.last_bounds, s.live.last_recon, s.live.last_binned
        else:
            if MODES[s.modality].live_view == "grid" and s.planner.bounds is not None:
                x0, x1, y0, y1 = s.planner.bounds
                p = 0.02 * max(x1 - x0, y1 - y0)
                b = (x0 - p, x1 + p, y0 - p, y1 + p)
            else:
                b = bounds_from_history(s.executor.history, fallback_extent)
            pts = s.executor.all_points()
            binned, recon = reconstruct_image(pts[:, :2], pts[:, 2], b, cfg.live_view)
        n = cfg.live_view.n_image_pixels
        X, Y = np.meshgrid(np.linspace(b[0], b[1], n), np.linspace(b[2], b[3], n))
        out.update(recon_X=X, recon_Y=Y, recon_Z=recon, binned_Z=binned, recon_bounds=np.array(b),
                   recon_mode=cfg.live_view.reconstruction)
        if s.live is not None and getattr(s.live, "last_recon_sensor", None) is not None:
            # the live view's sensor row: the same images from the measured positions (same
            # bounds, drive-volt frame -- um via um_per_V / um_at_0V); recon_Z above is the
            # commanded-position one (the first row)
            out.update(recon_Z_sensor=s.live.last_recon_sensor, binned_Z_sensor=s.live.last_binned_sensor)
    else:
        # gp_path: the area to explore (priors) if set, else the data's extent
        fb = cfg.live_view.fixed_bounds_V
        b = tuple(float(v) for v in fb) if fb is not None else bounds_from_history(s.executor.history, fallback_extent)
        X, Y = grid_from_bounds(b, n=300)
        mean, sigma = s.planner.predict_surface(X, Y)
        out.update(recon_X=X, recon_Y=Y, recon_mean=mean, recon_sigma=sigma, recon_bounds=np.array(b))
        # the map (2026-11-09): the measured samples (sensor positions), substrate plane removed,
        # filled in with cfg.live_view.reconstruction at full iterations -- recon_Z / binned_Z
        # on an n_image_pixels grid over the same bounds (what Analyze and user_raster read)
        from .reconstruction import flattened_reconstruction
        pts = s.executor.all_points()
        if len(pts) >= 10:
            binned, recon, plane = flattened_reconstruction(pts[:, :2], pts[:, 2], b, cfg.live_view)
            n = cfg.live_view.n_image_pixels
            Xn, Yn = np.meshgrid(np.linspace(b[0], b[1], n), np.linspace(b[2], b[3], n))
            out.update(recon_Z=recon, binned_Z=binned, recon_Z_X=Xn, recon_Z_Y=Yn, recon_mode=cfg.live_view.reconstruction,
                       recon_plane=np.asarray(plane), recon_flattened=True)
        if hasattr(s.planner, "timing"):   # gp_path: per-segment compute (seg, n, fit_s, acq_s)
            out.update(gp_timing=np.array(s.planner.timing, dtype=float).reshape(-1, 4),
                       gp_backend=np.array(s.cfg.gp_path.gp_backend))
    return out


def save(session: Session, output_dir="scan_data", prefix: str = "scan", extra: Optional[dict] = None,
         fallback_extent: float = 0.45, save_png: bool = True) -> pathlib.Path:
    """One .npz (+ the live view's .png) per run into output_dir -- the format
    Analyze_SelfDriving_Scan.ipynb reads: the trajectory (x_cmd/y_cmd/z, kinds,
    segment_lengths, sensors if on), `points`, and the reconstruction arrays; plus the
    grid+refine planner's maps and tiles. `extra` adds more arrays (e.g. the
    simulation's ground truth). Returns the path prefix (no extension)."""
    s = session
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_prefix = out_dir / f"{prefix}_{s.modality}_{timestamp}"
    hist = s.executor.history
    points = s.executor.all_points()  # sensor-corrected if position sensors are on
    data = dict(
        modality=s.modality,
        points=points,
        x_cmd=np.concatenate([st.x for st in hist]),
        y_cmd=np.concatenate([st.y for st in hist]),
        z=np.concatenate([st.z for st in hist]),
        kinds=np.array([st.kind for st in hist], dtype=object),
        segment_lengths=np.array([len(st.x) for st in hist]),
        # PyScanner fast rasters: the time shift Z was corrected by (s; NaN = not a fast raster)
        step_fast_raster_delay_s=np.array([st.meta.get("fast_raster_delay_s", np.nan) for st in hist], dtype=float),
        step_fast_raster_delay_note=np.array([st.meta.get("fast_raster_delay_note", "") for st in hist], dtype=object),
        tip_velocity_V_s=float(s.cfg.velocity.tip_velocity_V_s),
        sample_rate_hz=float(s.cfg.velocity.sample_rate_hz),
    )
    # extra channels (PFM amplitude / phase), in their units; NaN where a step had none
    names = sorted({k for st in hist if st.channels for k in st.channels})
    for k in names:
        data[f"ch_{k}"] = np.concatenate([st.channels[k] if st.channels and k in st.channels
                                          else np.full(len(st.x), np.nan) for st in hist]).astype(np.float32)
    if names and getattr(s.cfg, "pfm", None) is not None:
        from dataclasses import asdict as _asdict
        data["pfm_json"] = np.array(json.dumps(_asdict(s.cfg.pfm)))
    if getattr(s.executor, "z_label", None):
        data["z_label"] = np.array(s.executor.z_label)
    if s.cfg.position_sensors.enabled:
        # a step without sensor data (shouldn't happen any more) falls back to commanded
        data["x_sensor"] = np.concatenate([st.x_sensor if st.x_sensor is not None else st.x for st in hist])
        data["y_sensor"] = np.concatenate([st.y_sensor if st.y_sensor is not None else st.y for st in hist])
        data["has_sensor"] = np.concatenate([np.full(len(st.x), st.x_sensor is not None) for st in hist])
        # the AI1/AI2 volts they came from (NaN where none were recorded)
        data["x_sensor_raw_V"] = np.concatenate([st.x_sensor_raw if st.x_sensor_raw is not None
                                                 else np.full(len(st.x), np.nan) for st in hist]).astype(np.float32)
        data["y_sensor_raw_V"] = np.concatenate([st.y_sensor_raw if st.y_sensor_raw is not None
                                                 else np.full(len(st.x), np.nan) for st in hist]).astype(np.float32)
        cal = getattr(s.executor, "sensor_cal", None)
        if cal is not None:
            # drive volts -> micrometres along the sensors' axes: um = um_per_V * V + um_at_0V
            data["sensor_calibration_json"] = np.array(json.dumps(dict(cal.as_dict(), report=cal.report)))
            data["um_per_V"] = np.array(cal.um_per_V)
            data["um_at_0V"] = np.array([cal.x.b * 1e6, cal.y.b * 1e6])
    # what z is (cfg.z_signal, asylum.AsylumControl.route_z): source, divider, AI check, and
    # z_nm_per_V (AI volts -> nm; only when the source's sensitivity is known, e.g. ZSensor)
    data["z_unit"] = np.array(getattr(s.executor, "z_unit", "V"))      # z, recon_Z, binned_Z ... are in this
    data["z_scale_per_V"] = np.array(getattr(s.executor, "z_scale", 1.0))  # = z_unit per AI volt
    zi = getattr(s.executor, "z_signal_info", None)
    if zi:
        data["z_signal_json"] = np.array(json.dumps(zi))
        if zi.get("m_per_V"):
            data["z_nm_per_V"] = np.array(zi["m_per_V"] * zi.get("divider", 1.0) * 1e9)
    data.update(reconstruction_arrays(s, fallback_extent))
    if s.modality == "grid_refine":
        p = s.planner
        # the planner's own maps (over the scanned square) + every level's tiles
        data.update(
            grid_area=np.array(p.bounds), grid_binned=p.binned,
            grid_inpainted=p.inpainted, grid_roughness=p.roughness,
            level_spacing_V=np.array([lv.spacing_V for lv in p.levels]),
            level_pattern=np.array([lv.pattern for lv in p.levels]),
            level_samples=np.array([lv.n_samples for lv in p.levels]),
            level_plan_time_s=np.array([lv.plan_time_s for lv in p.levels]),
            # one row per scan (region): level, x0, x1, y0, y1 (its bounding box) + its pattern
            tiles=np.array([[lv.index, r.x0, r.x1, r.y0, r.y1] for lv in p.levels for r in lv.rects]),
            tile_pattern=np.array([r.pattern or lv.pattern for lv in p.levels for r in lv.rects]),
            # one row per tile of a region: level, region number, x0, x1, y0, y1, inside a feature
            region_cells=np.array([[lv.index, j, *c, bool(ins)] for lv in p.levels
                                   for j, r in enumerate(lv.rects) if r.cells is not None
                                   for c, ins in zip(r.cells, r.interior if r.interior is not None
                                                     else np.zeros(len(r.cells), bool))]).reshape(-1, 7))
    if s.modality in ("gp_edge_explorer", "grid_tracker", "gp_spiral"):
        data["gp_timing"] = np.array(s.planner.timing, dtype=float).reshape(-1, 4)
        data["explore_choices"] = np.array(s.planner.choices, dtype=object)
        data["gp_backend"] = np.array(s.cfg.gp_path.gp_backend)
        data["ivr_lambda"] = np.array(s.cfg.gp_path.ivr_lambda)
    if s.modality in ("edge_explorer", "domain_wall_explorer", "gp_edge_explorer", "grid_tracker", "gp_spiral"):
        # every feature: how it ended, so a run can be checked without replaying it
        pl = s.planner
        feats = []
        for f in pl.features:
            P = f.pts()
            O = f._ordered() if len(f.points) >= 3 else P
            feats.append(dict(
                id=f.id, closed=bool(f.closed_loop), closed_by=f.closed_by, edge_lost=bool(f.edge_lost),
                hit_border=bool(f.hit_border), crossings=len(f.points), legs=[len(l) for l in f.legs],
                follow_waveforms=int(f.follow_segments), interior_lines=int(f.interior_lines),
                interior_note=f.interior_note, merged_ids=list(f.merged_ids), inside_of=f.inside_of,
                enclosed=float(pl._encloses(f, O)) if len(O) >= 3 else 0.0,
                coverage_deg=float(np.degrees(f.angular_coverage())),
                centre=[float(v) for v in P.mean(axis=0)] if len(P) else None,
                outline=np.round(np.asarray(O, dtype=float), 4).tolist()))
        data["features_json"] = np.array(json.dumps(feats))
    model = getattr(s.planner, "substrate", None)
    if model is not None and model.fitted:
        # the substrate the planner decided against (substrate.SubstrateModel): its final
        # surface, roughness, threshold, and how the threshold evolved; whether the saved
        # reconstruction has it removed (cfg.live_view.flatten_substrate)
        hist = model.history
        data.update(substrate_json=np.array(json.dumps(model.summary())),
                    substrate_threshold_history=np.array([[h[0], h[1], h[2], h[3]] for h in hist], dtype=float),
                    recon_flattened=bool(s.live is not None and s.live._substrate() is not None))
    if s.priors is not None:
        from .priors import as_dict
        data["priors_json"] = np.array(json.dumps(as_dict(s.priors)))
    if extra:
        data.update(extra)
    np.savez(f"{out_prefix}.npz", **data)
    if save_png and s.live is not None:
        s.live.fig.savefig(f"{out_prefix}.png", dpi=150)
    print(f"Saved {out_prefix.name}.npz{' and .png' if save_png and s.live is not None else ''} "
          f"({len(points):,} points) to {out_dir.resolve()}")
    return out_prefix


# ------------------------------------------------------------------------------------
# reuse the Run notebook's configuration cell
# ------------------------------------------------------------------------------------
def load_run_config(notebook="Run_SelfDriving_Scan.ipynb", marker: str = "cfg = ExperimentConfig()"):
    """Execute the configuration cell of the Run notebook (the code cell containing
    `marker`) and return (cfg, MODALITY, LIVE_VIEW_EXTENT_V) -- so a simulation uses
    exactly the settings you'd run on the instrument. Only that one cell runs (nothing
    connects to hardware); it creates its OUTPUT_DIR if missing, as it does there."""
    nb = json.loads(pathlib.Path(notebook).read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code" and marker in "".join(c["source"])]
    if not cells:
        raise ValueError(f"No code cell containing {marker!r} in {notebook}")
    ns = {"ExperimentConfig": ExperimentConfig, "pathlib": pathlib, "np": np}
    exec(compile("".join(cells[0]["source"]), f"{notebook} (configuration cell)", "exec"), ns)
    return ns["cfg"], ns.get("MODALITY"), ns.get("LIVE_VIEW_EXTENT_V", 0.45)
