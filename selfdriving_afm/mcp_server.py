"""
MCP server for selfdriving_afm (SCOUT): lets an LLM client (Claude Desktop, Claude Code, any
MCP client) configure, simulate, run on the instrument, and analyze self-driving AFM scans --
every mode (edge_follow, edge_explorer, domain_wall_explorer, grid_refine, gp_path + its
presets, and the hybrids gp_edge_explorer, grid_tracker, gp_spiral), the stepwise grid_refine
workflow, the equivalent raster, and the saved data (full runs, lite / Zenodo copies, rasters).

Start (stdio transport -- what Claude Desktop / Claude Code launch):
    python scout_mcp.py                         # from the package folder (next to the notebooks)
    python -m selfdriving_afm.mcp_server --workdir <folder with the notebooks>
Needs:  pip install "mcp>=1.2"   (works with the 1.x FastMCP and the 2.x MCPServer API)

Design
- One server process = one session: a current ExperimentConfig (loaded from the Run
  notebook's configuration cell, like Simulation.ipynb does), the current mode, the priors,
  the simulated sample, and (once connected) the instrument.
- Long work runs as background JOBS (simulate, compare, run_scan, reference raster): a tool
  returns a job id at once (or the result, if it finished within wait_s); poll job_status.
- Every hardware call runs on ONE dedicated thread: PyScanner (ActiveX) and Igor (COM) objects
  must be used from the thread that created them.
- Tools that move the tip need confirm=True. Scans run inside AsylumControl.run(...), which
  engages for the scan and withdraws + returns to (0, 0) when it ends -- also on an error or a
  stop_job.
- The package prints a lot; prints go to the job's log (and stderr), never to stdout (the MCP
  stdio channel).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import io
import json
import os
import pathlib
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):                       # run as a script: make the package importable
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                                                # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server, Image
except ImportError:                                 # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image
try:
    from mcp.types import ToolAnnotations
except Exception:                                   # very old SDKs
    ToolAnnotations = None

from selfdriving_afm import modes, priors as priors_mod, simulation as sim, gp_presets, datasets
from selfdriving_afm.config import ExperimentConfig
from selfdriving_afm.topography import sample_from_text

# ======================================================================================
# stdout routing: prints -> current job's log (thread-local) + stderr; MCP keeps stdout
# ======================================================================================
_tls = threading.local()


class _Router(io.TextIOBase):
    """Replaces sys.stdout. Text prints go to the calling thread's log (if any) and to
    stderr. `.buffer` is the REAL stdout buffer: the MCP stdio transport writes protocol
    messages there, untouched."""

    def __init__(self, real):
        self._real = real
        self.buffer = real.buffer

    def write(self, s):
        log = getattr(_tls, "log", None)
        if log is not None:
            log.append(s)
        try:
            sys.stderr.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            sys.stderr.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def fileno(self):
        return sys.stderr.fileno()


# ======================================================================================
# session state
# ======================================================================================
SAMPLES = {
    "three_shapes": """
field 4
base -6.0
tilt 0.005 0.033
roughness 0.007 1 4
1 polygon    height=0.28 at=(1.7, 0.5) edge=0.03 points=(0.966,0.259),(0.259,0.966),(-0.707,0.707),(-0.966,-0.259),(-0.259,-0.966),(0.707,-0.707)
1 rectangle  height=0.31 size=1.4x0.5 at=(0.8, -3.75) edge=0.03
1 circle     height=0.30 size=1.2 at=(-2.3, 2.2) edge=0.03
""",
    "original": """
field 4
base -6.0
tilt 0.005 0.033
roughness 0.007 1 4
1 polygon    height=0.28 at=(1.7, 0.5) edge=0.03 points=(0.966,0.259),(0.259,0.966),(-0.707,0.707),(-0.966,-0.259),(-0.259,-0.966),(0.707,-0.707)
1 rectangle  height=0.31 size=1.4x0.5 at=(0.8, -3.75) edge=0.03
""",
    "five_shapes": """
field 4
base -6.0
tilt 0.005 0.033
roughness 0.007 1 4
1 polygon    height=0.28 at=(1.7, 0.5) edge=0.03 points=(0.966,0.259),(0.259,0.966),(-0.707,0.707),(-0.966,-0.259),(-0.259,-0.966),(0.707,-0.707)
1 rectangle  height=0.31 size=1.4x0.5 at=(0.8, -3.75) edge=0.03
1 circle     height=0.30 size=1.2 at=(-2.3, 2.2) edge=0.03
1 triangle   height=0.25 size=1.3 at=(-2.4, -1.6) edge=0.03
1 square     height=0.35 size=0.8 at=(2.6, 2.9) edge=0.03
""",
}


@dataclasses.dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"          # queued / running / done / failed / stopped
    started: float = 0.0
    ended: float = 0.0
    log: List[str] = dataclasses.field(default_factory=list)
    result: Any = None
    images: List[str] = dataclasses.field(default_factory=list)
    error: str = ""
    stop: bool = False
    progress: Dict[str, Any] = dataclasses.field(default_factory=dict)
    session: Any = None             # the running modes.Session (for live snapshots)


class _Stop(Exception):
    pass


class State:
    def __init__(self, workdir: pathlib.Path):
        self.workdir = workdir
        self.cfg: ExperimentConfig = ExperimentConfig()
        self.modality: str = "gp_path"
        self.gp_preset: Optional[str] = None
        self.config_source = "package defaults"
        self.live_extent = 0.45
        self.priors = None              # priors.ScanPriors the user set (None = from the sample in sims)
        self.sample = None
        self.sample_name = None
        self.jobs: Dict[str, Job] = {}
        self.sim_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
        self.hw_pool: Optional[ThreadPoolExecutor] = None
        self.hw: Dict[str, Any] = {}    # vi, igor, backend, afm, dry_run
        self.last_scan: Optional[str] = None
        self.last_session = None        # the last run_scan's modes.Session (equivalent raster)
        self.stepwise = None            # stepwise.StepwiseRun of stepwise_grid_refine
        self.lock = threading.Lock()

    # --------------------------------------------------------------- output folders
    @property
    def out_dir(self) -> pathlib.Path:
        d = self.workdir / "scan_data"          # the notebooks' OUTPUT_DIR
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def mcp_dir(self) -> pathlib.Path:
        p = self.workdir / "scan_data" / "mcp"
        p.mkdir(parents=True, exist_ok=True)
        return p


S: State = None   # set in main()


# ======================================================================================
# helpers
# ======================================================================================
def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else f"array{tuple(x.shape)}"
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, pathlib.Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _dump(obj) -> str:
    return json.dumps(_jsonable(obj), indent=1)


def _save_fig(fig, name: str) -> str:
    path = S.mcp_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_open_figs(name: str) -> List[str]:
    out = []
    for n in plt.get_fignums():
        out.append(_save_fig(plt.figure(n), f"{name}_{n}"))
    return out


def _resolve(path: Optional[str]) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else (S.workdir / p)


def _get_attr_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _coerce(old, new):
    """Match the type of the existing config value where it's unambiguous."""
    if isinstance(old, bool) or new is None or old is None:
        return new
    if isinstance(old, tuple) and isinstance(new, list):
        return tuple(new)
    if isinstance(old, float) and isinstance(new, int) and not isinstance(new, bool):
        return float(new)
    return new


def _apply_overrides(cfg, overrides: Optional[dict]) -> Dict[str, Any]:
    changed = {}
    for k, v in (overrides or {}).items():
        try:
            old = _get_attr_path(cfg, k)
        except AttributeError:
            raise ValueError(f"Unknown config field {k!r} (use get_config to browse sections/fields)")
        if dataclasses.is_dataclass(old) and not isinstance(v, dict):
            raise ValueError(f"{k!r} is a section, not a field")
        sim.set_param(cfg, k, _coerce(old, v))
        changed[k] = {"old": _jsonable(old), "new": _jsonable(_get_attr_path(cfg, k))}
    return changed


# modes whose planner reads cfg.gp_path (the GP presets apply to them)
GP_MODES = ("gp_path", "gp_edge_explorer", "gp_spiral", "grid_tracker")


def _prepare_cfg(mode: Optional[str], gp_preset: Optional[str], overrides: Optional[dict]):
    """A copy of the session config for one run: mode, GP preset, overrides."""
    mode = mode or S.modality
    if mode not in modes.MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose one of {list(modes.MODES)}")
    cfg = copy.deepcopy(S.cfg)
    preset = gp_preset if gp_preset is not None else (S.gp_preset if mode in GP_MODES else None)
    if gp_preset is not None and mode not in GP_MODES:
        raise ValueError(f"gp_preset only applies to the GP modes {GP_MODES}")
    if preset:
        gp_presets.apply_gp_preset(cfg, preset)
    _apply_overrides(cfg, overrides)
    return mode, cfg, preset


def _start_job(kind: str, label: str, fn, pool) -> Job:
    job = Job(id=uuid.uuid4().hex[:8], kind=kind, label=label)
    S.jobs[job.id] = job

    def runner():
        _tls.log = job.log
        job.status, job.started = "running", time.time()
        try:
            job.result = fn(job)
            job.status = "stopped" if job.stop else "done"
        except _Stop:
            job.status = "stopped"
        except BaseException as e:      # noqa: BLE001 -- report everything to the client
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
        finally:
            job.ended = time.time()
            _tls.log = None

    pool.submit(runner)
    return job


def _job_view(job: Job, log_tail: int = 25) -> dict:
    text = "".join(job.log)
    lines = text.splitlines()
    el = (job.ended or time.time()) - job.started if job.started else 0.0
    v = dict(job_id=job.id, kind=job.kind, label=job.label, status=job.status, elapsed_s=round(el, 1),
             progress=job.progress, log_tail=lines[-log_tail:] if log_tail else [])
    if job.error:
        v["error"] = job.error
    if job.status in ("done", "stopped") and job.result is not None:
        v["result"] = job.result
    if job.images:
        v["images"] = job.images
    return v


def _wait(job: Job, wait_s: float):
    t0 = time.time()
    while job.status in ("queued", "running") and time.time() - t0 < wait_s:
        time.sleep(0.25)


def _reply(view: dict, images: Optional[List[str]] = None, max_images: int = 3):
    out: List[Any] = [_dump(view)]
    for p in (images or [])[:max_images]:
        if p and str(p).lower().endswith(".png") and pathlib.Path(p).exists():
            out.append(Image(path=str(p)))
    return out


def _ann(read_only=False, destructive=False, title=None):
    if ToolAnnotations is None:
        return None
    try:
        return ToolAnnotations(title=title, readOnlyHint=read_only, destructiveHint=destructive)
    except Exception:
        return None


def _tool(read_only=False, destructive=False, title=None):
    """Register a tool. Errors come back as readable text ("ERROR: ...") instead of the SDK's
    generic "Error executing tool", so the client sees why (e.g. confirm=True missing)."""
    import functools
    a = _ann(read_only, destructive, title)
    reg = mcp.tool(annotations=a) if a is not None else mcp.tool()

    def deco(fn):
        returns_list = fn.__annotations__.get("return") in (list, "list")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:      # noqa: BLE001
                msg = f"ERROR: {type(e).__name__}: {e}"
                sys.stderr.write(msg + "\n" + traceback.format_exc() + "\n")
                return [msg] if returns_list else msg
        return reg(wrapper)
    return deco


def _summary_of_session(s, ev=None) -> dict:
    hist = s.executor.history
    n_samples = int(sum(len(st.x) for st in hist))
    path_V = float(sum(np.hypot(np.diff(st.x), np.diff(st.y)).sum() for st in hist if len(st.x) > 1))
    out = dict(mode=s.modality, waveforms=len(hist), samples=n_samples, path_V=round(path_V, 3),
               run_time_s=round(float(getattr(s, "run_time_s", 0.0)), 2))
    try:
        out["summary"] = modes.MODES[s.modality].summary(s)
    except Exception:
        pass
    pl = s.planner
    if hasattr(pl, "timing") and pl.timing:
        t = np.array([f + a for _, _, f, a in pl.timing])
        out["gp_compute_per_segment_s"] = dict(median=round(float(np.median(t)), 3), worst=round(float(t.max()), 3),
                                               total=round(float(t.sum()), 1))
    if ev is not None:
        out["evaluation"] = dict(
            bad_px=round(float(ev["bad_px"]), 4), edge_cov=round(float(ev["edge_cov"]), 4), rmse=ev.get("rmse"),
            per_feature=[dict(kind=f.get("kind"), edge_visited=round(float(f["edge_cov"]), 3)) for f in ev.get("features", [])],
            est_instrument_s=round(float(ev["est_instrument_s"]), 1))
        if ev.get("raster"):
            r = ev["raster"]
            out["evaluation"]["raster_same_budget"] = dict(bad_px=round(float(r["bad_px"]), 4),
                                                           edge_cov=round(float(r["edge_cov"]), 4))
    return out


# ======================================================================================
# the server
# ======================================================================================
INSTRUCTIONS = """
selfdriving_afm (SCOUT) controls self-driving AFM scans on an Asylum Cypher driven through
PyScanner (FPGA X/Y waveforms), and simulates them. Coordinates are drive volts (V) on the
X/Y piezo inputs; Z/height is AI0 volts unless the config converts it to nm.

ALWAYS start with server_info, then guide(topic="workflow") the first time.

Typical flow
1. load_run_config  -> the instrument settings from Run_SelfDriving_Scan.ipynb (same as the
   notebooks use). Inspect/change with get_config / set_config; pick the mode with
   set_modality (guide(topic="modes") explains each, incl. the hybrids); for the GP modes pick a
   strategy with apply_gp_preset.
2. Simulation (no hardware): set_sample, then simulate (one run) or compare (several modes /
   presets on the same sample). Results: bad pixels, edge coverage per feature, estimated
   instrument time, and the classic raster with the same budget.
3. Instrument: connect_instrument -> microscope_setup(confirm) -> set_priors ->
   run_scan(confirm) -> poll job_status (live_snapshot for the path so far) ->
   optional acquire_reference_raster(confirm) (the equivalent raster, same pixels) ->
   withdraw_and_park(confirm) when done. grid_refine one level at a time, spirals only, no tip
   jumps: stepwise_grid_refine (also in simulation).
4. Analysis: list_scans, scan_summary, reconstruct_scan, plot_reference_raster, animate_scan,
   run_analysis (the paper's figure scripts in analysis/). Every analysis tool also reads the
   lite / Zenodo copies (pts arrays), the stepwise dataset and plain rasters (ground truth).

SAFETY (hardware)
- Never pass confirm=True to a tool that moves the tip (microscope_setup, run_scan,
  acquire_reference_raster, probe_pyscanner_scans, withdraw_and_park, stepwise_grid_refine on
  the instrument) unless the user has
  explicitly asked for that action in this conversation. Say what will happen first
  (mode, area, path budget / duration) and wait for their go-ahead.
- One hardware job at a time; check instrument_status before starting one.
- stop_job stops a scan at the next waveform; the tip is then withdrawn and returned to
  (0, 0) automatically.
- connect_instrument(dry_run=True) exercises the whole acquisition path with the simulated
  backend (no microscope, Igor commands only printed) -- use it to test a plan.

Long operations return a job id; call job_status(job_id, wait_s=...) until status is done /
failed / stopped. Images are returned as PNGs (paths are in the JSON too).
"""

mcp = _Server("selfdriving-afm", instructions=INSTRUCTIONS)


# ------------------------------------------------------------------ general / config
@_tool(read_only=True, title="Server and session status")
def server_info() -> str:
    """Current session: working folder, config source, mode, GP preset, priors, sample,
    instrument connection, jobs. Call this first."""
    g = S.cfg.gp_path
    info = dict(
        workdir=str(S.workdir), config_source=S.config_source, mode=S.modality, gp_preset=S.gp_preset,
        gp_path=dict(acquisition=g.acquisition, ivr_lambda=g.ivr_lambda, smooth_turns=g.smooth_turns,
                     max_turn_deg=g.max_turn_deg, max_path_length_V=g.max_path_length_V, gp_backend=g.gp_backend),
        velocity=dict(tip_velocity_V_s=S.cfg.velocity.tip_velocity_V_s, sample_rate_hz=S.cfg.velocity.sample_rate_hz),
        priors=_jsonable(S.priors) if S.priors is not None else None,
        sample=S.sample_name, output_dir=str(S.out_dir),
        instrument=dict(connected=bool(S.hw), dry_run=S.hw.get("dry_run") if S.hw else None,
                        has_igor=(S.hw.get("igor") is not None) if S.hw else None),
        last_scan=S.last_scan,
        jobs=[dict(job_id=j.id, kind=j.kind, status=j.status, label=j.label) for j in list(S.jobs.values())[-10:]],
        modes=list(modes.MODES), gp_presets=list(gp_presets.GP_PRESETS))
    return _dump(info)


GUIDE = {
    "workflow": INSTRUCTIONS,
    "modes": "\n".join(f"{k}: {getattr(m, 'description', '')}" for k, m in modes.MODES.items()) + """

Choosing a mode
- edge_explorer: straight exploring segments; follows each edge it crosses until the outline
  closes; then explores again. Robust, thorough (traces most of every outline), long paths.
- edge_follow: search spiral, then a chain of circles along each edge.
- domain_wall_explorer: edge_explorer for PFM domain walls (phase flip / amplitude dip); needs
  the PFM configuration (load_run_config(notebook="Run_PFM_DomainWalls_v2.ipynb")).
- grid_refine: one sparse scan over the area, detect, refine flagged regions level by level
  (PyScanner's own fast rasters/spirals by default).
- gp_path: Gaussian-process pathfinder; strategy via apply_gp_preset (guide topic gp_presets).

Hybrids (combining levels; README "Combining levels"). All three use the edge_explorer settings
for tracing and gp_path's for the GP (so apply_gp_preset works for them too):
- gp_edge_explorer: GP (IVR) chooses where to explore between objects; the edge explorer traces
  every edge it crosses; re-check of earlier data for untraced crossings; coverage map at the end.
- grid_tracker: one sparse survey spiral over the area (grid_refine stage 1), then the edge
  explorer traces every outline that spiral crossed. Finds all objects first.
- gp_spiral: GP exploration between objects; grid_refine object (stage 2) then dense (stage 3)
  spirals on each one.

Stepwise grid_refine (stepwise_grid_refine): the workflow of the 2026-11-03 dataset -- stages 1
(survey) / 2 (object) / 3 (dense spirals) one call each (actions level0 / level1 / level2),
spirals only, no tip jumps, then save / figures / equivalent raster.
""",
    "gp_presets": gp_presets.__doc__,
    "config": """
The config is selfdriving_afm.config.ExperimentConfig (dataclasses). Browse with get_config
(no section = the list of sections), change with set_config({"section.field": value}).
Important sections: velocity (tip_velocity_V_s, sample_rate_hz), safety, execution
(start_xy_V), gp_path, edge_explorer, edge_follow, grid_refine, position_sensors, readback,
live_view (reconstruction), z_signal. Priors (feature sizes, area) are set with set_priors and
rewrite several mode fields (area, step sizes, GP length scale) -- set them before running.
Path budget for gp_path: gp_path.max_path_length_V (V of tip travel).
""",
    "analysis": """
Saved runs are .npz files (modes.save format) in the output folder (scan_data). list_scans
finds them; scan_summary describes one (mode, waveforms, samples, path, GP timing, priors,
sensors, reference raster); reconstruct_scan re-reconstructs (tv -- best for steps --,
linear, compressed_sensing, fast, gp, interpolated) at any size / window, optionally levelled
(substrate plane removed), for z or a PFM channel, next to its equivalent raster;
plot_reference_raster shows the run vs its reference raster; animate_scan makes a GIF replay
(self-driving vs raster, commanded vs sensor rows). run_analysis runs the paper's scripts
(analysis/: fig4_gp, hybrid_fig, combo_fig, discovery_summary, assess_run, run_gif, ...).

Data files: full runs (modes.save), lite / Zenodo copies (pts = x, y, z at the measured
positions, 1 sample in pts_stride; raster_xs/ys/zs), the stepwise dataset, and plain rasters
(ground truth: xs, ys, zs). selfdriving_afm.datasets.load reads all of them.
""",
    "timing": """
Instrument time per waveform: motion (samples / sample rate) + readback (~0.5 us/sample) +
fixed overhead: ~0.4 s for a user-defined waveform (all self-driving modes; measured from a
500-waveform edge_explorer run and the 2026-09-24 probe), ~3 s for a PyScanner-defined scan
(grid_refine's rasters/spirals). gp_path adds its GP compute per segment (printed and saved as
gp_timing). At 30 V/s a 0.5 V segment is 0.017 s of motion: gp_path runs are dominated by
overhead + compute; adaptive scanning pays most for slow scans (PFM/KPFM, spectroscopy).
""",
}


@_tool(read_only=True, title="How-to guide")
def guide(topic: str = "workflow") -> str:
    """Documentation for using this server. topic: workflow, modes, gp_presets, config,
    analysis, timing."""
    if topic not in GUIDE:
        return f"Unknown topic {topic!r}; choose one of {list(GUIDE)}"
    return GUIDE[topic]


@_tool(read_only=True, title="List scan modes and GP presets")
def list_modes() -> str:
    """All scan modes (with descriptions) and all gp_path presets (with their settings)."""
    return _dump(dict(
        modes={k: getattr(m, "description", "") for k, m in modes.MODES.items()},
        gp_presets={k: dict(gp_presets.GP_PRESETS[k]) for k in gp_presets.GP_PRESETS},
        gp_preset_notes=gp_presets.__doc__))


@_tool(title="Load the Run notebook's configuration")
def load_run_config(notebook: str = "Run_SelfDriving_Scan.ipynb") -> str:
    """Load the configuration cell of a Run notebook (default Run_SelfDriving_Scan.ipynb; use
    Run_PFM_DomainWalls_v2.ipynb for PFM) as the session config, mode and GP preset -- exactly
    what the notebooks run. Nothing connects to hardware."""
    nb = _resolve(notebook)
    log = []
    _tls.log = log
    try:
        cfg, modality, extent = modes.load_run_config(str(nb))
    finally:
        _tls.log = None
    S.cfg, S.modality, S.live_extent = cfg, modality or S.modality, extent
    S.config_source = str(nb)
    # which preset the notebook applied (if any)
    src = "".join("".join(c["source"]) for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]
                  if c["cell_type"] == "code" and "cfg = ExperimentConfig()" in "".join(c["source"]))
    import re
    m = re.search(r'^GP_PRESET\s*=\s*"([a-zA-Z0-9_]+)"', src, re.M)
    S.gp_preset = m.group(1) if m else None
    return _dump(dict(loaded=str(nb), mode=S.modality, gp_preset=S.gp_preset,
                      tip_velocity_V_s=cfg.velocity.tip_velocity_V_s, sample_rate_hz=cfg.velocity.sample_rate_hz,
                      gp_max_path_length_V=cfg.gp_path.max_path_length_V, output_dir=str(S.out_dir),
                      notes="".join(log)[-1500:]))


@_tool(read_only=True, title="Read configuration")
def get_config(section: Optional[str] = None, fields: Optional[List[str]] = None) -> str:
    """Read the session config. No section = the list of sections. section = e.g. "gp_path",
    "velocity", "edge_explorer", "grid_refine.detect" (dotted for nested); fields = only those."""
    if not section:
        secs = {f.name: type(getattr(S.cfg, f.name)).__name__ for f in dataclasses.fields(S.cfg)}
        return _dump(dict(sections=secs, hint="get_config(section='gp_path')"))
    obj = _get_attr_path(S.cfg, section)
    d = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else {"value": obj}
    if fields:
        d = {k: d[k] for k in fields if k in d}
    return _dump({section: d})


@_tool(title="Change configuration")
def set_config(updates: Dict[str, Any]) -> str:
    """Change session config fields: {"section.field": value, ...}, e.g.
    {"gp_path.max_path_length_V": 120, "velocity.tip_velocity_V_s": 10}. Returns old -> new.
    Note: safety / readback / position_sensors changes need connect_instrument again to reach
    an already connected instrument."""
    return _dump(dict(changed=_apply_overrides(S.cfg, updates)))


@_tool(title="Choose the scan mode")
def set_modality(mode: str) -> str:
    """Set the session's scan mode: edge_follow, edge_explorer, domain_wall_explorer,
    grid_refine, gp_path, or a hybrid (gp_edge_explorer, grid_tracker, gp_spiral) -- see
    list_modes / guide(topic="modes")."""
    if mode not in modes.MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose one of {list(modes.MODES)}")
    S.modality = mode
    return _dump(dict(mode=mode, description=getattr(modes.MODES[mode], "description", "")))


@_tool(title="Apply a gp_path preset")
def apply_gp_preset(name: str) -> str:
    """Set the gp_path strategy (also used by the GP hybrids) to a named preset (original,
    hybrid, pure_gp, pure_gp_smooth, pure_gp_1V, pure_gp_smooth_1V, pure_gp_edges,
    pure_gp_edges_smooth, pure_gp_edge_focus = lambda 1, gamma 2) in the session config.
    Fine-tune afterwards with set_config (e.g. gp_path.ivr_lambda, gp_path.ivr_interest_power
    = gamma, gp_path.max_turn_deg)."""
    gp_presets.apply_gp_preset(S.cfg, name)
    S.gp_preset = name
    return _dump(dict(gp_preset=name, fields=gp_presets.preset_fields(name)))


@_tool(title="Set the priors (feature sizes, area)")
def set_priors(feature_min: float, feature_max: float, area_half_width: float,
               area_center_x: float = 0.0, area_center_y: float = 0.0, units: str = "V",
               edge_threshold: Optional[float] = None) -> str:
    """What you expect on the sample: smallest feature (its narrowest width), largest feature,
    and the square area to explore (half-width, centre). units "V" (drive volts) or "nm"
    (needs a sensor calibration from microscope_setup with position sensors on).
    edge_threshold: optional fixed on-feature height threshold for the edge modes (Z units).
    Applied to the session config (rewrites mode fields: area, step sizes, GP length scale)."""
    if units == "nm":
        cal = getattr(S.hw.get("afm"), "sensor_cal", None) if S.hw else None
        if cal is None:
            raise ValueError("units='nm' needs the position-sensor calibration: run microscope_setup first")
        p = priors_mod.nm_to_V(feature_min, feature_max, area_half_width, (area_center_x, area_center_y), cal)
    elif units == "V":
        p = priors_mod.ScanPriors(feature_min_V=feature_min, feature_max_V=feature_max,
                                  area_half_width_V=area_half_width, area_center_V=(area_center_x, area_center_y))
    else:
        raise ValueError("units must be 'V' or 'nm'")
    if edge_threshold is not None:
        p.edge_threshold = float(edge_threshold)
    log = []
    _tls.log = log
    try:
        notes = priors_mod.apply(S.cfg, p)
    finally:
        _tls.log = None
    S.priors = p
    return _dump(dict(priors=p, applied=notes))


# ------------------------------------------------------------------ simulation
@_tool(title="Choose the simulated sample")
def set_sample(preset: Optional[str] = "three_shapes", text: Optional[str] = None, ibw: Optional[str] = None,
               ibw_channel: str = "HeightRetrace", dataset: Optional[str] = None, half_size_V: float = 4.0,
               center_x: float = 0.0, center_y: float = 0.0) -> list:
    """The simulated topography. preset: three_shapes (hexagon, rectangle, circle -- the GP
    tuning sample), original (hexagon + rectangle), five_shapes; or text = a sample in the
    topography text format (see Simulation.ipynb section 1), e.g.
      field 4 / base -6 / tilt 0.005 0.033 / roughness 0.007 1 4 /
      1 circle height=0.3 size=1.2 at=(-2,2) edge=0.03   (one item per line)
    ibw: an Asylum .ibw image (channel ibw_channel) laid over a (2 half_size_V)^2 V square
    centred on (center_x, center_y) -- note the AR image is mirrored in x vs the drive frame.
    dataset: a measured raster (.npz: the ground truth / any run with its equivalent raster,
    full or lite) -> its TV map, at its own drive-volt coordinates: simulate on the real sample.
    Image samples have no feature list: set the priors yourself (set_priors).
    Returns its description and an image."""
    if dataset:
        from selfdriving_afm.topography import image_sample
        from selfdriving_afm.reconstruction import bin_image, tv_inpaint
        r = datasets.load(_resolve(dataset))
        src = r if r["format"] == datasets.FORMAT_RASTER or r["raster"] is None else r["raster"]
        b = r["bounds"]
        B, _ = bin_image(np.column_stack([src["x"], src["y"]]), src["z"], 256, b)
        img = tv_inpaint(B, 0.3, 800)
        smp = image_sample(img, b, f"measured: {pathlib.Path(dataset).name} (TV map, {r['z_unit']})",
                           f"Z ({r['z_unit']})")
        name = pathlib.Path(dataset).stem
    elif ibw:
        from selfdriving_afm.topography import sample_from_ibw
        smp = sample_from_ibw(_resolve(ibw), channel=ibw_channel, half_size_V=half_size_V,
                              center_V=(center_x, center_y))
        name = pathlib.Path(ibw).stem
    elif text:
        smp, name = sample_from_text(text), "custom"
    else:
        if preset not in SAMPLES:
            raise ValueError(f"Unknown sample preset {preset!r}; choose {list(SAMPLES)} or pass text")
        smp, name = sample_from_text(SAMPLES[preset]), preset
    S.sample, S.sample_name = smp, name
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(smp.image(250), origin="lower", extent=smp.bounds, cmap="copper")
    ax.set_title(f"sample: {name}")
    img = _save_fig(fig, f"sample_{name}")
    desc = [dict(shape=getattr(f, "shape", "?"), at=_jsonable(getattr(f, "at", None)),
                 size=_jsonable(getattr(f, "size", None)), height=_jsonable(getattr(f, "height", None)))
            for f in smp.features]
    try:
        pri = _jsonable(priors_mod.from_sample(smp)) if smp.features else None
    except Exception:
        pri = None
    return _reply(dict(sample=name, bounds=smp.bounds, features=desc, description=smp.description,
                       priors_from_sample=pri, image=img,
                       note=None if pri else "image sample: call set_priors before simulate"), [img])


def _sim_memory(cfg, mode, store_every):
    if store_every is not None:
        cfg.readback.store_every = int(store_every)
    elif mode == "grid_refine" and int(getattr(cfg.readback, "store_every", 1) or 1) < 10:
        cfg.readback.store_every = 10


def _sim_one(job: Job, mode, cfg, preset, overrides, noise_std, seed, raster, save):
    smp = S.sample
    cfg.readback.mode = "raw"             # the simulator returns exact per-sample Z (Simulation.ipynb)
    if S.priors is None and not smp.features:
        raise ValueError("this sample is an image (ibw / dataset): set_priors first")
    p = S.priors if S.priors is not None else priors_mod.from_sample(smp)
    priors_mod.apply(cfg, p)
    if preset:                            # the preset's choices win over what the priors set...
        gp_presets.apply_gp_preset(cfg, preset)
    _apply_overrides(cfg, overrides)      # ...and explicit overrides win over both
    job.progress = dict(stage="running", mode=mode, gp_preset=preset)
    ev = sim.simulate(mode, cfg, smp, noise_std=noise_std, seed=seed,
                      raster=dict(layout="sparse", match="path", reconstruction="raster_columns") if raster else None)
    print(sim.format_evaluation(ev))
    sim.plot_evaluation(ev)
    job.images += _save_open_figs(f"sim_{mode}")
    out = _summary_of_session(ev["session"], ev)
    out.update(sample=S.sample_name, priors=_jsonable(p))
    if save:
        prefix = modes.save(ev["session"], S.mcp_dir, prefix="sim", fallback_extent=S.live_extent, save_png=False)
        out["saved"] = f"{prefix}.npz"
    return out


@_tool(title="Simulate one run")
def simulate(mode: Optional[str] = None, gp_preset: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
             path_budget_V: Optional[float] = None, noise_std: float = 0.0004, seed: int = 1,
             raster_compare: bool = True, save: bool = False, store_every: Optional[int] = None,
             wait_s: float = 45.0) -> list:
    """Run one mode on the simulated sample (set_sample first; default three_shapes) with the
    session config (+ optional gp_preset / overrides {"section.field": value}). path_budget_V
    sets gp_path.max_path_length_V. Uses the session priors, or the sample's own if none were
    set. Returns a job (and its result + evaluation figure if done within wait_s): bad pixels,
    per-feature edge coverage, estimated instrument time, GP compute per segment, and the
    classic raster with the same path for comparison. save=True writes the run's .npz to
    scan_data/mcp (every sample: ~1 GB for an edge-mode run unless store_every is set).
    GP presets apply to gp_path and the GP hybrids. store_every: keep one block-mean sample
    per N in RAM (None = 10 for grid_refine, whose fine rasters are tens of millions of
    samples -- several GB at full resolution -- else the config's value)."""
    if S.sample is None:
        set_sample("three_shapes")
    overrides = dict(overrides or {})
    if path_budget_V is not None:
        overrides["gp_path.max_path_length_V"] = path_budget_V
    mode, cfg, preset = _prepare_cfg(mode, gp_preset, overrides)
    _sim_memory(cfg, mode, store_every)
    job = _start_job("simulate", f"{mode}" + (f" / {preset}" if preset else ""),
                     lambda j: _sim_one(j, mode, cfg, preset, overrides, noise_std, seed, raster_compare, save),
                     S.sim_pool)
    _wait(job, wait_s)
    return _reply(_job_view(job), job.images)


@_tool(title="Compare modes / presets in simulation")
def compare(variants: Optional[List[Dict[str, Any]]] = None, gp_presets_list: Optional[List[str]] = None,
            path_budget_V: Optional[float] = None, noise_std: float = 0.0004, seed: int = 1,
            store_every: Optional[int] = None, wait_s: float = 45.0) -> list:
    """Several variants on the same simulated sample, one table + one row of panels each.
    variants: [{"label": "...", "mode": "edge_explorer", "gp_preset": null, "overrides": {...}}, ...];
    gp_presets_list: shortcut for gp_path presets, e.g. ["pure_gp", "pure_gp_smooth"] (or ["all"]).
    path_budget_V applies to gp_path variants. Also scores the classic raster with each budget."""
    if S.sample is None:
        set_sample("three_shapes")
    vs = []
    names = list(gp_presets.GP_PRESETS) if gp_presets_list == ["all"] else (gp_presets_list or [])
    for n in names:
        vs.append(dict(label=f"gp_path: {n}", mode="gp_path", gp_preset=n, overrides={}))
    vs += list(variants or [])
    if not vs:
        raise ValueError("give variants and/or gp_presets_list")
    cfg = copy.deepcopy(S.cfg)
    cfg.readback.mode = "raw"             # the simulator returns exact per-sample Z (Simulation.ipynb)
    _sim_memory(cfg, "grid_refine" if any(v.get("mode") == "grid_refine" for v in vs) else "", store_every)
    p = S.priors if S.priors is not None else priors_mod.from_sample(S.sample)
    priors_mod.apply(cfg, p)
    sim_variants = []
    for v in vs:
        mode = v.get("mode", S.modality)
        ov = {}
        if v.get("gp_preset"):
            ov.update(gp_presets.as_overrides(v["gp_preset"]))
        if mode == "gp_path" and path_budget_V is not None:
            ov["gp_path.max_path_length_V"] = path_budget_V
        ov.update(v.get("overrides") or {})
        sim_variants.append((v.get("label") or mode, mode, ov))

    def fn(job):
        job.progress = dict(stage="running", variants=len(sim_variants))
        evs = sim.compare(sim_variants, cfg, S.sample, noise_std=noise_std, seed=seed, plot=True,
                          raster=dict(layout="sparse", match="path", reconstruction="raster_columns"))
        job.images += _save_open_figs("compare")
        rows = []
        for e in evs:
            if "error" in e:
                rows.append(dict(label=e.get("label"), error=e.get("trace", "")[-800:]))
                continue
            r = _summary_of_session(e["session"], e)
            r["label"] = e.get("label")
            rows.append(r)
        return dict(sample=S.sample_name, variants=rows)

    job = _start_job("compare", f"{len(sim_variants)} variants", fn, S.sim_pool)
    _wait(job, wait_s)
    return _reply(_job_view(job), job.images)


# ------------------------------------------------------------------ instrument
def _hw_submit(fn, wait=True, timeout=None):
    if S.hw_pool is None:
        raise RuntimeError("Not connected: call connect_instrument first")
    fut = S.hw_pool.submit(fn)
    return fut.result(timeout=timeout) if wait else fut


def _hw_busy() -> Optional[str]:
    for j in S.jobs.values():
        if j.kind in ("run_scan", "reference_raster", "probe", "setup", "stepwise_hw") and j.status in ("queued", "running"):
            return j.id
    return None


def _need_confirm(confirm: bool, what: str):
    if not confirm:
        raise PermissionError(f"{what} moves the tip. Describe the action to the user, get their explicit "
                              f"go-ahead, then call again with confirm=True.")


@_tool(title="Connect to the instrument (or a dry run)")
def connect_instrument(hardware_paths_file: str = "hardware_paths.json", use_igor: bool = True,
                       dry_run: bool = False) -> str:
    """Connect to PyScanner (exe/vi paths from hardware_paths.json) and, if use_igor, to the
    Asylum software (Igor Pro, COM), and build the backend with the session config's safety /
    readback / position-sensor settings -- as the Run notebook's sections 1 and 3. Does not
    move the tip. dry_run=True: the simulated backend on the session's sample, no Igor
    (commands printed): the whole acquisition path without a microscope."""
    if S.hw_pool is None:
        def _init():
            try:
                import pythoncom          # Windows COM: this thread owns the ActiveX / COM objects
                pythoncom.CoInitialize()
            except Exception:
                pass
        S.hw_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hw", initializer=_init)
    if _hw_busy():
        raise RuntimeError(f"A hardware job is running ({_hw_busy()}); wait or stop_job it first")
    log: List[str] = []

    def _connect():
        _tls.log = log
        try:
            from selfdriving_afm import asylum
            if dry_run:
                from selfdriving_afm.backends.simulate_backend import SimulateBackend
                smp = S.sample if S.sample is not None else sample_from_text(SAMPLES["three_shapes"])
                backend = SimulateBackend(smp, noise_std=0.0004, seed=1)
                vi = igor = None
            else:
                from selfdriving_afm.backends.pyscanner_backend import connect, PyScannerBackend
                hp = json.loads(_resolve(hardware_paths_file).read_text())
                vi = connect(exe_path=hp["exe_path"], vi_path=hp["vi_path"])
                igor = asylum.connect_igor() if use_igor else None
                backend = PyScannerBackend(vi, S.cfg.safety, S.cfg.readback, position_sensors=S.cfg.position_sensors)
            afm = asylum.AsylumControl(igor, backend, S.cfg)
            S.hw = dict(vi=vi, igor=igor, backend=backend, afm=afm, dry_run=dry_run)
        finally:
            _tls.log = None

    _hw_submit(_connect)
    return _dump(dict(connected=True, dry_run=dry_run, igor=S.hw.get("igor") is not None,
                      log="".join(log)[-2000:]))


@_tool(read_only=True, title="Instrument status")
def instrument_status() -> str:
    """Connection, dry-run flag, Igor, tip position (last commanded), sensor calibration, and
    whether a hardware job is running."""
    if not S.hw:
        return _dump(dict(connected=False))
    afm = S.hw.get("afm")
    cal = getattr(afm, "sensor_cal", None)
    st = dict(connected=True, dry_run=S.hw.get("dry_run"), igor=S.hw.get("igor") is not None,
              hardware_job_running=_hw_busy(), sensor_calibration=bool(cal))
    try:
        st["ao_xy"] = _hw_submit(lambda: afm.ao_xy(), timeout=15) if not S.hw.get("dry_run") else None
    except Exception as e:
        st["ao_xy_error"] = repr(e)
    return _dump(st)


@_tool(destructive=True, title="Microscope setup (moves the tip)")
def microscope_setup(confirm: bool = False, withdraw_first: bool = True, wait_s: float = 60.0) -> list:
    """The Run notebook's section 3a: centre, AO0/AO1 to 0, X/Y crosspoint to the FPGA, Z and
    position-sensor routing, sensor calibration (if position_sensors.enabled), engage. MOVES
    THE TIP -- needs confirm=True after the user's go-ahead. Runs as a job."""
    _need_confirm(confirm, "microscope_setup")
    if _hw_busy():
        raise RuntimeError(f"A hardware job is running ({_hw_busy()})")
    afm = S.hw["afm"]

    def fn(job):
        def on_hw():
            _tls.log = job.log
            try:
                afm.setup(withdraw_first=withdraw_first)
                return dict(setup="done", sensor_calibration=bool(getattr(afm, "sensor_cal", None)))
            finally:
                _tls.log = None
        return _hw_submit(on_hw)

    job = _start_job("setup", "microscope setup", fn, _JobPool.pool())
    _wait(job, wait_s)
    return _reply(_job_view(job))


class _JobPool:
    """Waits for hardware jobs (the work itself runs on the hardware thread)."""
    _p = None

    @classmethod
    def pool(cls):
        if cls._p is None:
            cls._p = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hwjob")
        return cls._p


def _run_scan_on_hw(job: Job, mode, cfg, preset, label, save, extra_meta):
    afm, backend = S.hw["afm"], S.hw["backend"]
    gt = S.sample if S.hw.get("dry_run") else None
    session = modes.build(mode, cfg, backend, live=False, live_extent=S.live_extent,
                          ground_truth_fn=gt, priors=S.priors)
    ex = session.executor
    if extra_meta.get("tail_dwell_s") is not None:
        ex.tail_dwell_s = float(extra_meta["tail_dwell_s"])
    prev = ex.on_step

    def on_step(i, step, executor):
        job.progress = dict(stage="scanning", waveforms=i, kind=getattr(step, "kind", None),
                            elapsed_s=round(time.time() - job.started, 1))
        if prev is not None:
            prev(i, step, executor)
        if job.stop:
            raise _Stop("stopped by stop_job")

    ex.on_step = on_step
    job.session = session
    S.last_session = session
    afm.executor = ex
    try:
        with afm.run(label or mode, ex):                 # engaged; withdraws + (0, 0) at the end
            modes.run(session)
    except _Stop:
        print("stopped by stop_job -- tip withdrawn and returned to (0, 0)")
    out = _summary_of_session(session)
    out.update(gp_preset=preset, stopped=job.stop)
    if save and session.executor.history:
        prefix = modes.save(session, S.out_dir, prefix="scan", fallback_extent=S.live_extent)
        out["saved"] = f"{prefix}.npz"
        S.last_scan = out["saved"]
        png = pathlib.Path(f"{prefix}.png")
        if png.exists():
            job.images.append(str(png))
    if session.executor.history:
        try:
            img, _ = _snapshot_png(list(session.executor.history), 150, f"scan_{mode}")
            job.images.append(img)
        except Exception as e:          # an image is a convenience, never a failure
            print(f"(no snapshot image: {e})")
    return out


@_tool(destructive=True, title="Run a self-driving scan (moves the tip)")
def run_scan(confirm: bool = False, mode: Optional[str] = None, gp_preset: Optional[str] = None,
             overrides: Optional[Dict[str, Any]] = None, path_budget_V: Optional[float] = None,
             label: Optional[str] = None, save: bool = True, tail_dwell_s: Optional[float] = None,
             wait_s: float = 10.0) -> list:
    """Run a scan on the instrument (or the dry-run backend) with the session config + mode
    (+ gp_preset / overrides / path_budget_V for gp_path), the session priors, engaged inside
    AsylumControl.run (withdraws and returns to (0, 0) when it ends, fails or is stopped), then
    save the .npz + .png to the output folder. MOVES THE TIP -- confirm=True only after the
    user's explicit go-ahead. Returns a job: poll job_status; live_snapshot shows the path.
    tail_dwell_s: hold the tip at rest this long at the end of every waveform (PyScanner's AI
    record can come back short at the END of a waveform; 0.3 s for rasters, 0 for GP
    segments -- Run_GP_Paper.ipynb); None = the executor's default (0)."""
    _need_confirm(confirm, "run_scan")
    if not S.hw:
        raise RuntimeError("Not connected: call connect_instrument first")
    if _hw_busy():
        raise RuntimeError(f"A hardware job is running ({_hw_busy()})")
    if S.priors is None:
        print("NOTE: no priors set -- the config's own area / sizes are used (set_priors to set them)")
    overrides = dict(overrides or {})
    if path_budget_V is not None:
        overrides["gp_path.max_path_length_V"] = path_budget_V
    # session config already has the priors (set_priors applies them); preset, then overrides
    mode, cfg, preset = _prepare_cfg(mode, gp_preset, overrides)
    if S.hw.get("dry_run"):
        cfg.readback.mode = "raw"         # the simulated backend returns exact per-sample Z

    def fn(job):
        def on_hw():
            _tls.log = job.log
            try:
                return _run_scan_on_hw(job, mode, cfg, preset, label, save, dict(tail_dwell_s=tail_dwell_s))
            finally:
                _tls.log = None
        return _hw_submit(on_hw)

    job = _start_job("run_scan", (label or mode) + (f" / {preset}" if preset else ""), fn, _JobPool.pool())
    _wait(job, wait_s)
    return _reply(_job_view(job), job.images)


@_tool(destructive=True, title="Withdraw and park the tip")
def withdraw_and_park(confirm: bool = False) -> str:
    """Withdraw, then move AO0/AO1 smoothly back to (0, 0) (Run notebook section 4b). Needs
    confirm=True (the user's go-ahead) -- it moves the tip, though toward safety."""
    _need_confirm(confirm, "withdraw_and_park")
    if _hw_busy():
        raise RuntimeError(f"A hardware job is running ({_hw_busy()}); stop_job it -- that withdraws and parks")
    log: List[str] = []

    def fn():
        _tls.log = log
        try:
            S.hw["afm"].finish()
        finally:
            _tls.log = None
    _hw_submit(fn)
    return _dump(dict(done=True, log="".join(log)[-1500:]))


@_tool(destructive=True, title="Acquire a reference raster (moves the tip)")
def acquire_reference_raster(confirm: bool = False, npz: Optional[str] = None, method: str = "user",
                             fast_axis: str = "x", n_lines: Optional[int] = None,
                             duration_s: Optional[float] = None, tail_dwell_s: float = 0.3,
                             wait_s: float = 10.0) -> list:
    """After a run: the EQUIVALENT RASTER -- a raster with the same number of measured pixels as
    the run -- saved INTO the run's .npz for comparison (plot_reference_raster, animate_scan,
    run_analysis). method "user" (default, what the paper uses since 2026-11-09): user-defined
    smooth waveforms through the run's own executor (same speed, drive correction, sensors),
    cut only at line ends, tail_dwell_s at rest after every band (user_raster.py); "pyscanner":
    PyScanner's own fast raster (reference_raster.py; fast_axis, duration_s). npz: the run
    (default the last scan). MOVES THE TIP over the whole area -- confirm=True only after the
    user's go-ahead."""
    _need_confirm(confirm, "acquire_reference_raster")
    if _hw_busy():
        raise RuntimeError(f"A hardware job is running ({_hw_busy()})")
    from selfdriving_afm import reference_raster
    path = str(_resolve(npz)) if npz else S.last_scan
    if not path:
        raise ValueError("No run given and no last scan in this session")
    cfg = copy.deepcopy(S.cfg)
    if method not in ("user", "pyscanner"):
        raise ValueError("method must be 'user' or 'pyscanner'")
    if method == "user":
        from selfdriving_afm import user_raster
        sess = S.last_session
        if sess is None or (S.last_scan and pathlib.Path(path) != pathlib.Path(S.last_scan)):
            # not the run of this session: a fresh executor with the session config
            sess = modes.build(S.modality, copy.deepcopy(S.cfg), S.hw["backend"], live=False, priors=S.priors)
        ex = sess.executor
        if getattr(S.hw.get("afm"), "sensor_cal", None) is not None and getattr(ex, "sensor_cal", None) is None:
            ex.sensor_cal = S.hw["afm"].sensor_cal
        area = getattr(sess.planner, "bounds", None)
        uplan = user_raster.plan(path, sess.cfg, ex, n_lines=n_lines, area=area, tail_dwell_s=tail_dwell_s)
        if not user_raster.ram_check(uplan, sess.cfg):
            raise RuntimeError("equivalent raster refused: not enough RAM / outside the safety limits:\n"
                               + user_raster.describe(uplan))

        def fn(job):
            def on_hw():
                _tls.log = job.log
                try:
                    print(user_raster.describe(uplan))
                    afm = S.hw["afm"]
                    with afm.run("equivalent raster (user-defined)", ex):
                        ref = user_raster.acquire(uplan, ex, sess.cfg)
                    reference_raster.save_into(path, ref)
                    reference_raster.plot(path)
                    job.images += _save_open_figs("equivalent_raster")
                    return dict(saved_into=path, method="user", plan=user_raster.describe(uplan))
                finally:
                    _tls.log = None
            return _hw_submit(on_hw)

        job = _start_job("reference_raster", "equivalent raster (user)", fn, _JobPool.pool())
        _wait(job, wait_s)
        return _reply(_job_view(job), job.images)
    plan = reference_raster.plan(path, cfg, fast_axis=fast_axis, n_lines=n_lines, duration_s=duration_s)

    def fn(job):
        def on_hw():
            _tls.log = job.log
            try:
                print(reference_raster.describe(plan))
                afm, backend = S.hw["afm"], S.hw["backend"]
                tip = getattr(afm.executor, "last_pos", (0.0, 0.0)) if getattr(afm, "executor", None) else (0.0, 0.0)
                with afm.run("reference raster"):
                    ref = reference_raster.acquire(plan, backend, cfg, tip_xy=tip, return_to_origin=False,
                                                   sensor_cal=getattr(afm, "sensor_cal", None))
                reference_raster.save_into(path, ref)
                reference_raster.plot(path)
                job.images += _save_open_figs("reference_raster")
                return dict(saved_into=path, plan=reference_raster.describe(plan))
            finally:
                _tls.log = None
        return _hw_submit(on_hw)

    job = _start_job("reference_raster", "reference raster", fn, _JobPool.pool())
    _wait(job, wait_s)
    return _reply(_job_view(job), job.images)


@_tool(destructive=True, title="Check PyScanner's own scan types (moves the tip)")
def probe_pyscanner_scans(confirm: bool = False, center_x: float = 0.1, center_y: float = 0.05,
                          size_V: float = 0.2, lines: int = 10, wait_s: float = 30.0) -> list:
    """Run notebook section 3d: small PyScanner raster / spiral / Lissajous scans to check how
    PyScanner places and times them (needed once for grid_refine with
    scan_generator='pyscanner_defined'). MOVES THE TIP -- confirm=True after the go-ahead."""
    _need_confirm(confirm, "probe_pyscanner_scans")
    if _hw_busy():
        raise RuntimeError(f"A hardware job is running ({_hw_busy()})")
    from selfdriving_afm.pyscanner_scans import probe_pyscanner_scans as _probe

    def fn(job):
        def on_hw():
            _tls.log = job.log
            try:
                afm = S.hw["afm"]
                session = modes.build(S.modality, copy.deepcopy(S.cfg), S.hw["backend"], live=False, priors=S.priors)
                ex = session.executor
                afm.executor = ex
                with afm.run("PyScanner scan-type check"):
                    reports = _probe(ex, center_xy=(center_x, center_y), size_V=size_V, lines=lines)
                ex.reset(S.cfg.execution.start_xy_V)
                return dict(reports=_jsonable(reports))
            finally:
                _tls.log = None
        return _hw_submit(on_hw)

    job = _start_job("probe", "PyScanner probe", fn, _JobPool.pool())
    _wait(job, wait_s)
    return _reply(_job_view(job))


# ------------------------------------------------------------------ jobs
@_tool(read_only=True, title="Job status")
def job_status(job_id: str, wait_s: float = 0.0, log_tail: int = 25) -> list:
    """Status of a job (simulate, compare, run_scan, ...): status, progress (waveforms so far,
    elapsed), the last log lines, and the result + figures when done. wait_s: wait up to this
    long for it to finish first (e.g. 30-55 s per call while a scan runs)."""
    job = S.jobs.get(job_id)
    if job is None:
        raise ValueError(f"No job {job_id!r}; see server_info for recent jobs")
    _wait(job, wait_s)
    return _reply(_job_view(job, log_tail), job.images if job.status in ("done", "stopped") else [])


@_tool(read_only=True, title="List jobs")
def list_jobs() -> str:
    """All jobs of this session with their status."""
    return _dump([dict(job_id=j.id, kind=j.kind, label=j.label, status=j.status,
                       elapsed_s=round((j.ended or time.time()) - j.started, 1) if j.started else 0)
                  for j in S.jobs.values()])


@_tool(title="Stop a job")
def stop_job(job_id: str) -> str:
    """Stop a running scan / raster at the next waveform: the tip is withdrawn and returned to
    (0, 0) and whatever was measured is still saved. Simulations can't be stopped (they are
    short)."""
    job = S.jobs.get(job_id)
    if job is None:
        raise ValueError(f"No job {job_id!r}")
    job.stop = True
    return _dump(dict(job_id=job_id, stop_requested=True, status=job.status))


@_tool(read_only=True, title="Live snapshot of a running scan")
def live_snapshot(job_id: Optional[str] = None, n_pixels: int = 120) -> list:
    """The path measured so far by a running (or finished) run_scan job, coloured by Z, plus a
    quick binned image -- the notebook's live view as a PNG. Default: the latest run_scan."""
    job = S.jobs.get(job_id) if job_id else next((j for j in reversed(list(S.jobs.values()))
                                                  if j.kind == "run_scan"), None)
    if job is None or job.session is None:
        raise ValueError("No run_scan job with data yet")
    hist = list(job.session.executor.history)
    if not hist:
        return _reply(dict(job_id=job.id, waveforms=0, note="no waveform measured yet"))
    img, n = _snapshot_png(hist, n_pixels, f"live_{job.id}")
    return _reply(dict(job_id=job.id, status=job.status, waveforms=len(hist), samples=int(n), image=img), [img])


def _snapshot_png(hist, n_pixels: int, name: str):
    """Path coloured by Z + binned image of the waveforms measured so far."""
    x = np.concatenate([st.x_sensor if st.x_sensor is not None else st.x for st in hist])
    y = np.concatenate([st.y_sensor if st.y_sensor is not None else st.y for st in hist])
    z = np.concatenate([st.z for st in hist])
    n = min(len(x), len(z))
    x, y, z = x[:n], y[:n], z[:n]
    k = max(1, n // 30000)
    from selfdriving_afm.reconstruction import bin_image
    binned, b = bin_image(np.column_stack([x, y]), z, n_pixels)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    sc = axs[0].scatter(x[::k], y[::k], c=z[::k], s=1, cmap="copper")
    axs[0].set_aspect("equal"); axs[0].set_title(f"path so far ({len(hist)} waveforms)")
    fig.colorbar(sc, ax=axs[0], shrink=0.8)
    axs[1].imshow(binned, origin="lower", extent=b, cmap="copper"); axs[1].set_title("binned Z")
    return _save_fig(fig, name), n


# ------------------------------------------------------------------ analysis
@_tool(read_only=True, title="List saved scans")
def list_scans(directory: Optional[str] = None, limit: int = 30) -> str:
    """Saved data files (.npz) in the output folder (default) or `directory` (e.g. the Zenodo
    data folder), newest first, with mode, format (full / lite / stepwise / raster) and size.
    Simulations from this server are in scan_data/mcp."""
    d = _resolve(directory) if directory else S.out_dir
    files = sorted(d.rglob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    rows = []
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as f:
                mode = str(f["modality"]) if "modality" in f.files else "-"
                fmt = datasets.detect_format(f)
        except Exception:
            mode, fmt = "?", "?"
        rows.append(dict(path=str(p), mode=mode, format=fmt, size_MB=round(p.stat().st_size / 1e6, 2),
                         modified=time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))))
    return _dump(dict(directory=str(d), scans=rows))


@_tool(read_only=True, title="Summarize a saved scan")
def scan_summary(npz: Optional[str] = None) -> str:
    """What a saved data file contains -- a full run, a lite / Zenodo copy, the stepwise
    grid_refine dataset or a plain raster (ground truth): mode, format, samples (and the lite
    stride), positions, bounds, units, PFM channels, equivalent raster, features, priors,
    waveforms (by kind), path length, GP timing, estimated instrument time, and the arrays in
    the file. npz: path (default the last scan of this session)."""
    path = _resolve(npz) if npz else (pathlib.Path(S.last_scan) if S.last_scan else None)
    if path is None:
        raise ValueError("Give npz (see list_scans)")
    r = datasets.load(path)
    out = datasets.summary(r)
    out["mode"] = out["modality"]
    out["path"] = str(path)
    d = np.load(path, allow_pickle=True)
    out["arrays"] = sorted(k for k in d.files if not k.startswith("_lite_"))
    if "segment_lengths" in d.files:
        sl = d["segment_lengths"]
        out.update(waveforms=int(len(sl)), samples_measured=int(sl.sum()))
    if "kinds" in d.files:
        ks, cs = np.unique(d["kinds"].astype(str), return_counts=True)
        out["waveforms_by_kind"] = dict(zip(ks.tolist(), cs.tolist()))
    if "x_cmd" in d.files:
        x, y = d["x_cmd"], d["y_cmd"]
        out["commanded_bounds_V"] = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    out["path_V"] = round(float(np.hypot(np.diff(r["x"]), np.diff(r["y"])).sum()), 2)
    if "tip_velocity_V_s" in d.files and "sample_rate_hz" in d.files and "segment_lengths" in d.files:
        rate = float(d["sample_rate_hz"])
        n = int(d["segment_lengths"].sum())
        nw = int(len(d["segment_lengths"]))
        out["tip_velocity_V_s"] = float(d["tip_velocity_V_s"])
        out["est_instrument_s"] = round(n / rate + n * sim.READBACK_S_PER_SAMPLE + nw * sim.OVERHEAD_USER_WAVEFORM_S, 1)
    for k in ("run_wall_time_s", "raster_wall_time_s", "raster_scan_time_s", "ivr_lambda", "stop_reason"):
        if k in d.files:
            out[k] = _jsonable(datasets._scalar(d[k]))
    if r["priors"] is not None:
        out["priors"] = r["priors"]
    if "gp_timing" in d.files:
        t = d["gp_timing"]
        if t.size:
            tot = t[:, 2] + t[:, 3]
            out["gp_path"] = dict(gp_backend=str(d["gp_backend"]) if "gp_backend" in d.files else None,
                                  segments=int(len(t)), final_points=int(t[-1, 1]),
                                  compute_per_segment_s=dict(median=round(float(np.median(tot)), 3),
                                                             worst=round(float(tot.max()), 3),
                                                             total=round(float(tot.sum()), 1)))
    d.close()
    return _dump(out)


def _recon(x, y, z, b, method, n_pixels, level):
    from selfdriving_afm.config import LiveViewConfig
    from selfdriving_afm.reconstruction import bin_image, reconstruct_image, flattened_reconstruction
    ok = np.isfinite(z)
    pts, zz = np.column_stack([x[ok], y[ok]]), z[ok]
    rc = copy.deepcopy(S.cfg.live_view) if hasattr(S.cfg, "live_view") else LiveViewConfig()
    rc.reconstruction = method
    rc.n_image_pixels = n_pixels
    if method == "tv":
        rc.tv_mu = 0.3                   # the paper's scoring (SUMMARY 2026-11-09)
    if level:
        binned, recon, _ = flattened_reconstruction(pts, zz, b, rc)
    else:
        binned, _ = bin_image(pts, zz, n_pixels, b)
        _, recon = reconstruct_image(pts, zz, b, rc)
    return pts, binned, recon


@_tool(read_only=True, title="Reconstruct a saved scan")
def reconstruct_scan(npz: Optional[str] = None, method: str = "tv", n_pixels: int = 200,
                     bounds: Optional[List[float]] = None, channel: str = "z", level: bool = True,
                     with_raster: bool = True, units: str = "V", save: bool = False) -> list:
    """Re-reconstruct a saved run -- full, lite / Zenodo copy, stepwise dataset or plain raster
    (Analyze notebook sections 3-4): method tv (flat plateaus, sharp steps: the paper's maps),
    linear, compressed_sensing, fast, gp or interpolated; n_pixels image size; bounds
    [xmin, xmax, ymin, ymax] in V (default the run's window); channel "z" or a PFM channel
    ("amplitude", "phase"); level: remove the substrate plane (robust, lower envelope -- for z);
    with_raster: also reconstruct the run's equivalent raster (same pixels) for comparison;
    units "V" or "um" (axes via the run's sensor calibration). Positions: the measured ones
    (sensors) when recorded. save writes <run>_reanalyzed.npz (the original is never modified)."""
    path = _resolve(npz) if npz else (pathlib.Path(S.last_scan) if S.last_scan else None)
    if path is None:
        raise ValueError("Give npz (see list_scans)")
    r = datasets.load(path)
    t0 = time.time()

    def pick(src):
        if channel == "z":
            return src["z"]
        ch = src.get("channels", {})
        if channel not in ch:
            raise ValueError(f"no channel {channel!r} here; channels: {sorted(ch)}")
        return ch[channel]

    b = tuple(bounds) if bounds else r["bounds"]
    lv = level and channel == "z"
    rows = [("run", r["x"], r["y"], pick(r))]
    rs = r["raster"]
    if with_raster and rs is not None and r["format"] != datasets.FORMAT_RASTER:
        try:
            rows.append(("equivalent raster", rs["x"], rs["y"], pick(rs)))
        except ValueError:
            pass
    log: List[str] = []
    _tls.log = log
    res = []
    try:
        for name, x, y, z in rows:
            res.append((name,) + _recon(np.asarray(x), np.asarray(y), np.asarray(z), b, method, n_pixels, lv))
    finally:
        _tls.log = None
    ext = b
    if units == "um":
        if r["um_per_V"] is None:
            raise ValueError("this file has no sensor calibration (um_per_V): use units='V'")
        ux, uy = datasets.to_um(r, [b[0], b[1]], [b[2], b[3]])
        ext = (float(ux[0]), float(ux[1]), float(uy[0]), float(uy[1]))
    fig, axs = plt.subplots(len(res), 3, figsize=(14, 4.6 * len(res)), squeeze=False)
    vmin = vmax = None
    for i, (name, pts, binned, recon) in enumerate(res):
        k = max(1, len(pts) // 30000)
        px, py = (datasets.to_um(r, pts[::k, 0], pts[::k, 1]) if units == "um" else (pts[::k, 0], pts[::k, 1]))
        axs[i, 0].plot(px, py, lw=0.4); axs[i, 0].set_aspect("equal")
        axs[i, 0].set_title(f"{name}: path ({r['positions']} positions, {units})")
        if vmin is None and np.isfinite(recon).any():
            vmin, vmax = np.nanpercentile(recon, [1, 99])
        axs[i, 1].imshow(binned, origin="lower", extent=ext, cmap="copper", vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f"binned ({np.isfinite(binned).mean():.1%} of pixels measured)")
        axs[i, 2].imshow(recon, origin="lower", extent=ext, cmap="copper", vmin=vmin, vmax=vmax)
        axs[i, 2].set_title(f"{method}{' (levelled)' if lv else ''} -- {channel} ({r['z_unit'] if channel == 'z' else ''})")
    fig.tight_layout()
    img = _save_fig(fig, f"recon_{path.stem}")
    out = dict(path=str(path), format=r["format"], modality=r["modality"], method=method, channel=channel,
               levelled=lv, n_pixels=n_pixels, bounds_V=b, positions=r["positions"], image=img,
               rows=[dict(name=nm, samples=int(len(pts)), measured_fraction=round(float(np.isfinite(bn).mean()), 4))
                     for nm, pts, bn, _ in res], seconds=round(time.time() - t0, 1))
    if save:
        outp = path.with_name(path.stem + "_reanalyzed.npz")
        arrs = {}
        for nm, _, bn, rc_ in res:
            tag = "run" if nm == "run" else "raster"
            arrs[f"{tag}_binned"], arrs[f"{tag}_recon"] = bn, rc_
        np.savez(outp, modality=str(r["modality"] or ""), bounds=np.array(b), method=method, channel=channel,
                 levelled=lv, **arrs)
        out["saved"] = str(outp)
    return _reply(out, [img])


@_tool(read_only=True, title="Plot a run vs its reference raster")
def plot_reference_raster(npz: Optional[str] = None) -> list:
    """The run next to the reference raster saved into it (acquire_reference_raster / Run
    notebook section 6)."""
    from selfdriving_afm import reference_raster
    path = _resolve(npz) if npz else (pathlib.Path(S.last_scan) if S.last_scan else None)
    if path is None:
        raise ValueError("Give npz")
    with np.load(path, allow_pickle=True) as f:
        fmt = datasets.detect_format(f)
    if fmt != datasets.FORMAT_FULL:              # lite / stepwise / raster: the generic comparison
        return reconstruct_scan(npz=str(path), method="tv", with_raster=True)
    log: List[str] = []
    _tls.log = log
    try:
        reference_raster.plot(str(path))
    finally:
        _tls.log = None
    imgs = _save_open_figs(f"raster_{path.stem}")
    return _reply(dict(path=str(path), images=imgs, log="".join(log)[-1000:]), imgs)


def _animate_generic(run: dict, out: pathlib.Path, frames: int = 60, fps: int = 4, n_pixels: int = 150):
    """Replay of any data file: the path so far (viridis, start -> end) and the binned image of
    what was measured, for the run and (same pixel count per frame) its equivalent raster."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from selfdriving_afm.reconstruction import bin_image
    b = run["bounds"]
    srcs = [("SCOUT", run)] + ([("equivalent raster", run["raster"])] if run["raster"] is not None
                               and run["format"] != datasets.FORMAT_RASTER else [])
    fig, axs = plt.subplots(len(srcs), 2, figsize=(8.5, 4.2 * len(srcs)), squeeze=False)
    lo, hi = np.nanpercentile(run["z"], [2, 98])
    arts = []
    for i, (nm, src) in enumerate(srcs):
        x, y, z = (np.asarray(src[k]) for k in ("x", "y", "z"))
        k = max(1, len(x) // 30000)
        sc = axs[i, 0].scatter([], [], s=0.4, c=[], cmap="viridis", vmin=0, vmax=1)
        axs[i, 0].set(xlim=b[:2], ylim=b[2:], aspect="equal", title=nm)
        im = axs[i, 1].imshow(np.full((n_pixels, n_pixels), np.nan), origin="lower", extent=b, cmap="copper",
                              vmin=lo, vmax=hi)
        arts.append((x, y, z, k, sc, im, axs[i, 1]))
    def update(f):
        frac = (f + 1) / frames
        for x, y, z, k, sc, im, ax in arts:
            n = max(2, int(frac * len(x)))
            sc.set_offsets(np.column_stack([x[:n:k], y[:n:k]]))
            sc.set_array(np.linspace(0, frac, len(x[:n:k])))
            B, _ = bin_image(np.column_stack([x[:n], y[:n]]), z[:n], n_pixels, b)
            im.set_data(B)
            ax.set_title(f"{frac:.0%} of the samples, {np.isfinite(B).mean():.0%} px measured")
        return []
    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


@_tool(read_only=True, title="Animate a saved run")
def animate_scan(npz: Optional[str] = None, rows: Optional[List[str]] = None, max_frames: int = 60,
                 fps: int = 4, method: str = "linear", wait_s: float = 45.0) -> list:
    """GIF replay of a saved run (Analyze notebook 5b): rows from run_cmd, run_sensor,
    raster_cmd, raster_sensor (rows the file doesn't have are skipped). Runs as a job; the GIF
    path is in the result."""
    from selfdriving_afm import compare_animation
    from selfdriving_afm.config import LiveViewConfig
    path = _resolve(npz) if npz else (pathlib.Path(S.last_scan) if S.last_scan else None)
    if path is None:
        raise ValueError("Give npz")
    rows = rows or ["run_cmd", "run_sensor", "raster_cmd", "raster_sensor"]
    with np.load(path, allow_pickle=True) as f:
        fmt = datasets.detect_format(f)
    if fmt != datasets.FORMAT_FULL:              # lite / stepwise / raster: path-growth replay
        def gfn(job):
            out = S.mcp_dir / (path.stem + "_replay.gif")
            return dict(gif=str(_animate_generic(datasets.load(path), out, max_frames, fps)))
        job = _start_job("animate", path.name, gfn, S.sim_pool)
        _wait(job, wait_s)
        return _reply(_job_view(job))

    def fn(job):
        d = np.load(path, allow_pickle=True)
        out = path.with_name(path.stem + "_replay.gif")
        rc = copy.deepcopy(S.cfg.live_view) if hasattr(S.cfg, "live_view") else LiveViewConfig()
        res = compare_animation.animate(d, out, rows=rows, recon_cfg=rc, recon_mode=method, max_frames=max_frames,
                                        fps=fps, recon_every=2, title=f"{path.stem}")
        return dict(gif=str(res) if res else None)

    job = _start_job("animate", path.name, fn, S.sim_pool)
    _wait(job, wait_s)
    return _reply(_job_view(job))


# ------------------------------------------------------------------ stepwise grid_refine
_STEPWISE_ACTIONS = ("setup", "level0", "level1", "level2", "save", "figures", "animate", "plan_raster",
                     "equivalent_raster", "compare_raster", "status")
_STEPWISE_MOVES = ("setup", "level0", "level1", "level2", "equivalent_raster")


def _new_pngs(since: float) -> List[str]:
    return [str(q) for q in sorted(S.out_dir.glob("*.png"), key=lambda q: q.stat().st_mtime)
            if q.stat().st_mtime >= since - 1]


@_tool(destructive=True, title="Stepwise grid_refine (levels 0-1-2, spirals, no jumps)")
def stepwise_grid_refine(action: str, confirm: bool = False, simulate: bool = False,
                         settings: Optional[Dict[str, Any]] = None, wait_s: float = 30.0) -> list:
    """grid_refine one stage per call -- the workflow of the good 2026-11-03 dataset
    (stepwise.StepwiseRun, Run_GridRefine_Stepwise.ipynb): stage 1 one sparse spiral over the
    area, stage 2 one spiral per object found, stage 3 one dense spiral per object; PyScanner
    spirals only, every move a driven waveform (the tip never jumps). Actions, in order:
      setup            connect the run to the instrument (uses connect_instrument's PyScanner / Igor),
                       configure, Asylum setup + sensor calibration, engage   [moves the tip]
      level0 / level1 / level2   the three stages                          [move the tip]
      save             the dataset .npz (every scan, binned + linear / CS / TV maps)
      figures, animate overview + per-object figures; GIF
      plan_raster, equivalent_raster, compare_raster   the equivalent raster (user-defined,
                       same pixels) -- plan, acquire [moves the tip], compare
      status           what has been done
    simulate=True: the same pipeline on the session's simulated sample (set_sample), nothing
    connected, no confirm needed. settings: StepwiseRun class settings to change before setup,
    e.g. {"SPARSE": {"center": [0.3, 0], "radius_V": 7.15, "n_cycles": 6, "duration_s": 10},
    "L1_MAX_S": 4, "L2_MAX_S": 11.5, "Z_OBJ_NM": 30}. On the instrument every moving action
    needs confirm=True after the user's go-ahead."""
    from selfdriving_afm import stepwise
    if action not in _STEPWISE_ACTIONS:
        raise ValueError(f"action must be one of {_STEPWISE_ACTIONS}")
    run = S.stepwise
    if action == "status":
        if run is None:
            return _reply(dict(stepwise=None, next="setup"))
        done = [a for a, attr in (("setup", "afm"), ("level0", "st"), ("level1", "DONE1"), ("level2", "H2"),
                                  ("save", "NPZ"), ("plan_raster", "RP"), ("equivalent_raster", "REF"))
                if getattr(run, attr, None) is not None]
        return _reply(dict(simulate=getattr(run, "_mcp_sim", None), done=done, tag=getattr(run, "tag", None),
                           dataset=str(getattr(run, "NPZ", "") or "") or None))
    if action != "setup" and run is None:
        raise ValueError("call stepwise_grid_refine(action='setup', ...) first")
    need = dict(level1=("st", "level0"), level2=("DONE1", "level1"), save=("st", "level0"), figures=("NPZ", "save"),
                animate=("NPZ", "save"), plan_raster=("NPZ", "save"), equivalent_raster=("NPZ", "save"),
                compare_raster=("REF", "equivalent_raster"))
    if action in need and getattr(run, need[action][0], None) is None:
        raise ValueError(f"run action={need[action][1]!r} before {action!r} (stepwise_grid_refine(action='status'))")
    sim_run = simulate if action == "setup" else bool(getattr(run, "_mcp_sim", False))
    if action in _STEPWISE_MOVES and not sim_run:
        _need_confirm(confirm, f"stepwise_grid_refine({action})")
    if _hw_busy():
        raise RuntimeError(f"A hardware job is running ({_hw_busy()})")

    def work(job):
        nonlocal run
        t0 = time.time()
        _tls.log = job.log
        try:
            if action == "setup":
                run = stepwise.StepwiseRun(out_dir=str(S.out_dir), config_notebook=str(S.workdir / "Run_SelfDriving_Scan.ipynb"),
                                           hardware_paths=str(S.workdir / "hardware_paths.json"))
                for k, v in (settings or {}).items():
                    if not hasattr(run, k):
                        raise ValueError(f"StepwiseRun has no setting {k!r}")
                    setattr(run, k, type(getattr(run, k))(v) if isinstance(getattr(run, k), tuple) else v)
                if sim_run:
                    smp = S.sample if S.sample is not None else sample_from_text(SAMPLES["three_shapes"])
                    run.setup_simulation(smp)
                else:
                    if not S.hw or S.hw.get("dry_run"):
                        raise RuntimeError("connect_instrument (not a dry run) first, or use simulate=True")
                    from selfdriving_afm import asylum
                    from selfdriving_afm.backends.pyscanner_backend import PyScannerBackend
                    run.vi, run.igor = S.hw["vi"], S.hw["igor"]
                    cfg, _, _ = modes.load_run_config(run.config_notebook)
                    run.vi.setcontrolvalue("data_rate", 0)
                    run.cfg = run.configure(cfg, 1e6)
                    run.backend = PyScannerBackend(run.vi, cfg.safety, cfg.readback, position_sensors=cfg.position_sensors)
                    run.afm = asylum.AsylumControl(run.igor, run.backend, cfg)
                    ok = run.afm.xy_routing_ok()
                    run.afm.xy_from_fpga = bool(ok)
                    run.afm.setup()
                    run.tag = time.strftime("%Y%m%d_%H%M%S")
                    S.hw["afm"], S.hw["backend"] = run.afm, run.backend   # one connection, one controller
                run._mcp_sim = sim_run
                S.stepwise = run
                res = dict(setup="done", simulate=sim_run, tag=run.tag)
            elif action == "save":
                res = dict(dataset=str(run.save_dataset()))
                S.last_scan = res["dataset"]
            elif action == "figures":
                res = dict(figures=[str(f) for f in (run.figures() or [])])
            elif action == "animate":
                res = dict(gif=str(run.animate()))
            elif action == "plan_raster":
                rp, ok = run.plan_raster()
                res = dict(ok=bool(ok), plan={k: _jsonable(v) for k, v in rp.items() if not isinstance(v, (list, np.ndarray)) or k in ("area",)})
            elif action == "compare_raster":
                run.compare_raster()
                res = dict(compared=True)
            else:                               # level0 / level1 / level2 / equivalent_raster
                try:
                    getattr(run, action)()
                except AssertionError as e:
                    raise RuntimeError(f"{action} refused by a safety check (e.g. the spiral would leave the "
                                       f"safety limits: SPARSE center +- radius_V vs cfg.safety): {e}") from e
                res = dict(done=action)
            job.images += [q for q in _new_pngs(t0) if q not in job.images]
            return res
        finally:
            _tls.log = None

    if sim_run:
        job = _start_job("stepwise", f"stepwise {action} (sim)", work, S.sim_pool)
    else:
        def fn(job):
            return _hw_submit(lambda: work(job))
        job = _start_job("stepwise_hw", f"stepwise {action}", fn, _JobPool.pool())
    _wait(job, wait_s)
    return _reply(_job_view(job), job.images)


# ------------------------------------------------------------------ analysis scripts
def _analysis_dir() -> pathlib.Path:
    for d in (S.workdir / "analysis", pathlib.Path(__file__).resolve().parents[1] / "analysis"):
        if d.is_dir():
            return d
    raise FileNotFoundError("no analysis/ folder next to the notebooks")


def _script_doc(p: pathlib.Path) -> str:
    import ast
    try:
        return (ast.get_docstring(ast.parse(p.read_text(encoding="utf-8"))) or "").strip()
    except Exception:
        return ""


@_tool(title="Run a paper analysis script (analysis/)")
def run_analysis(script: Optional[str] = None, args: Optional[List[str]] = None, timeout_s: float = 1800,
                 wait_s: float = 45.0) -> list:
    """The manuscript's figure / metric scripts in analysis/ (fig4_gp, fig4_paper, hybrid_fig,
    combo_fig, discovery_summary, sweep_fig, assess_run, run_gif, gp_recon_study, sim_edge_focus,
    sim_hybrids, tip_check). No script = the list with each one's usage. args: the command-line
    arguments exactly as in the script's usage line (paths relative to the working folder).
    Runs as a separate Python process (a job); new PNGs in the output folders come back as
    images. Read-only on the data; writes figures / json next to the script's --out."""
    import subprocess
    d = _analysis_dir()
    scripts = {q.stem: q for q in sorted(d.glob("*.py"))}
    if not script:
        return _reply(dict(folder=str(d), scripts={k: _script_doc(q)[:600] for k, q in scripts.items()}))
    name = script[:-3] if script.endswith(".py") else script
    if name not in scripts:
        raise ValueError(f"no script {script!r}; choose one of {list(scripts)}")
    cmd = [sys.executable, str(scripts[name])] + [str(a) for a in (args or [])]

    def fn(job):
        t0 = time.time()
        job.progress = dict(stage="running", cmd=" ".join(cmd))
        pr = subprocess.run(cmd, cwd=str(d if name == "tip_check" else S.workdir), capture_output=True, text=True,
                            timeout=timeout_s, env=dict(os.environ, MPLBACKEND="Agg"))
        job.log.append(pr.stdout[-20000:])
        if pr.stderr:
            job.log.append("\n[stderr]\n" + pr.stderr[-8000:])
        new = [str(q) for q in list(S.workdir.rglob("*.png")) + list(d.rglob("*.png")) if q.stat().st_mtime >= t0 - 1]
        job.images += sorted(set(new))[:12]
        if pr.returncode != 0:
            raise RuntimeError(f"{name} exited with {pr.returncode}: {pr.stderr[-1500:]}")
        return dict(script=name, returncode=pr.returncode, new_figures=job.images, seconds=round(time.time() - t0, 1))

    job = _start_job("analysis", name, fn, S.sim_pool)
    _wait(job, wait_s)
    return _reply(_job_view(job), job.images)


# ======================================================================================
def main(argv=None):
    global S
    ap = argparse.ArgumentParser(description="selfdriving_afm MCP server (stdio)")
    ap.add_argument("--workdir", default=None,
                    help="folder with the notebooks / hardware_paths.json / scan_data (default: the "
                         "folder containing the selfdriving_afm package)")
    ap.add_argument("--no-load-config", action="store_true",
                    help="start from package defaults instead of Run_SelfDriving_Scan.ipynb")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    a = ap.parse_args(argv)
    workdir = pathlib.Path(a.workdir).resolve() if a.workdir else pathlib.Path(__file__).resolve().parents[1]
    os.chdir(workdir)                       # notebooks' relative paths (OUTPUT_DIR, hardware_paths.json)
    S = State(workdir)
    sys.stdout = _Router(sys.__stdout__)    # prints -> logs/stderr; the MCP transport keeps stdout
    if not a.no_load_config and (workdir / "Run_SelfDriving_Scan.ipynb").exists():
        try:
            load_run_config("Run_SelfDriving_Scan.ipynb")
        except Exception as e:              # still start; the client can load / fix it
            sys.stderr.write(f"[selfdriving-afm MCP] could not load Run_SelfDriving_Scan.ipynb: {e}\n")
    mcp.run(a.transport)


if __name__ == "__main__":
    main()
