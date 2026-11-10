# SCOUT -- self-driving scan paths for atomic force microscopy

`selfdriving_afm` (SCOUT) lets the AFM tip decide where to go next from what it is measuring,
instead of following a predefined raster. Every mode runs the same planner code in simulation
and on the instrument (an Asylum Cypher driven through PyScanner's FPGA X/Y waveforms), and an
MCP server lets an LLM client (Claude Desktop, Claude Code, ...) configure, simulate, run and
analyze scans.

| mode | what it does | good at |
|---|---|---|
| `grid_refine` (+ `stepwise_grid_refine`) | stage 1 survey scan of the area, stage 2 object scans, stage 3 dense spirals on each object | a quick map, then detail where something is |
| `edge_follow` | search spiral, then a chain of circles along each edge | the original concept |
| `edge_explorer` | straight exploring segments; follows every edge it crosses until the outline closes | complete outlines (contour following) |
| `domain_wall_explorer` | the edge explorer on PFM phase flips / amplitude dips | ferroelectric domain walls |
| `gp_path` (+ presets) | Gaussian-process pathfinder: continuous segments chosen by integrated variance reduction; lambda / gamma trade exploration for edges | fast discovery, an "anytime" map |
| `gp_edge_explorer`, `grid_tracker`, `gp_spiral` | hybrids: GP or survey exploration between objects + edge tracing or spirals on them | combining levels |

Coordinates are drive volts on the X/Y piezo inputs (micrometres with the position-sensor
calibration); see "Units" below.

## Install

```bash
pip install .                 # the package (Python >= 3.8): numpy, scipy, scikit-learn, matplotlib, psutil
pip install ".[mcp]"          # + the MCP server (Python >= 3.10)
pip install ".[hardware]"     # + pywin32, on the instrument PC (PyScanner ActiveX, Igor COM)
pip install ".[dev]" && python -m pytest -q     # the tests (no hardware, ~10 s)
```

## Quick start

```bash
python demo_simulation.py     # two planners on a synthetic sample, no hardware -> *_demo.png
```

- `Simulation.ipynb` -- every mode on a text-defined, `.ibw` or measured topography.
- `Run_SelfDriving_Scan.ipynb` -- **the instrument entry point** ("Running on the microscope").
  `Run_GP_Paper.ipynb` (GP + hybrids, 2026-11-09), `Run_GridRefine_Stepwise.ipynb`,
  `Run_PFM_DomainWalls_v2.ipynb` are the run notebooks of the paper's datasets.
- `Analyze_SelfDriving_Scan.ipynb`, `Analyze_PFM_DomainWalls.ipynb` -- re-analysis of saved runs.
- `analysis/` -- the manuscript's figure scripts (also callable from the MCP server: `run_analysis`).
- `scout_mcp.py` -- the MCP server ("MCP server" below); `python mcp_selftest.py` checks it.

## Data

Saved runs are `.npz` files (`modes.save`). `selfdriving_afm.datasets.load(path)` reads every
kind -- full runs, the lite copies deposited on Zenodo, the stepwise grid_refine dataset and plain
rasters (ground truth) -- as one dict (path at the measured positions, PFM channels, equivalent
raster, calibration, priors, features); `datasets.make_lite` writes a lite copy;
`tools/make_zenodo_data.py` builds the Zenodo data folder (manifest, previews, examples).

## License and citation

License: to be added once the ORNL software release is approved (no license = all rights
reserved). Please cite as in `CITATION.cff`.

## Simulation: testing every mode on a known topography

`Simulation.ipynb` runs **the same scan modes as `Run_SelfDriving_Scan.ipynb`** --
both notebooks build, run and save through `selfdriving_afm/modes.py`, so the only
difference is the backend -- against a sample whose true topography is known:

- **the sample**: described in text (`topography.sample_from_text`), e.g.
  `2 triangles height=0.5 size=1.2` / `1 square height=0.5 size=1.0 at=(2, -2)` /
  `1 rectangle height=0.3 size=1.5x0.6 angle=30` (also circles, ellipses, rings,
  Gaussian bumps, pyramids, terrace steps, polygons; soft sidewalls, tilt, substrate
  level; random non-overlapping placement), **or** one channel of an Asylum `.ibw`
  image (`topography.sample_from_ibw` -- numpy-only reader, no extra install);
- **the configuration**: by default the Run notebook's own configuration cell
  (`modes.load_run_config`), plus per-simulation overrides;
- **scoring** (`simulation.py`): bad pixels of the reconstruction vs the ground truth,
  how much of each feature's edge was visited (MISSED features listed), and cost --
  samples, waveforms, and an estimated instrument time: motion + readback (0.5 us/sample)
  + a fixed overhead per waveform -- ~3 s for a PyScanner-defined scan
  (`OVERHEAD_PER_WAVEFORM_S`), ~0.4 s for a user-defined waveform
  (`OVERHEAD_USER_WAVEFORM_S`), both measured on the instrument;
- **comparisons** of several modes / settings on the same sample, and one-parameter
  **sweeps**;
- **the equivalent classic raster** (`simulation.EquivalentRaster`): the same budget
  spent on plain left-to-right lines, reconstructed by default with
  `"raster_columns"` (`reconstruction.column_inpaint` -- each pixel column
  interpolated between the lines with PCHIP). That is the best method for raster
  data: CS's DCT prior aliases on regularly skipped lines (4-10x more bad pixels at
  sparse line counts). `reconstruction=None` uses the mode's own method instead.

Adding a mode: planner in `planners/`, config dataclass in `config.py`, one entry in
`modes.MODES` -- it's then available in both notebooks. Simulation does not reproduce
PyScanner's own scan types exactly (see `pyscanner_scans.py` and
`simulation.model_notes`) or the image readback mode.

## Priors: feature sizes and the area to explore

Before every scan, both notebooks ask (section 2b; Enter keeps the default) for the
**smallest and largest expected feature** (its narrowest width, in V) and the **most the
scan may explore** (a square: half-width and center, in V). `priors.apply` turns them
into every mode's settings -- the configuration cell no longer sets the areas:

| Mode | From the area | From the feature sizes |
|---|---|---|
| edge_follow | first search covers the whole area: a spiral (inscribed circle; later, smaller spirals fill the rest), or a raster / Lissajous over the whole square (`first_scan_pattern`) | spiral turns 0.5 x smallest apart (no feature fits between two turns); tracing circles `radius_mode="curvature"`, between 0.25 x smallest and 0.75 x largest |
| grid_refine | the sparse scan's square (one scan) | sparse lines 0.7 x smallest apart; refinement tiles at most 0.5 x smallest wide; feature insides up to the largest feature wide are scanned too |
| edge_explorer | exploration square; first path from its center | exploration sweeps ~ the smallest feature apart; zigzag steps 0.5 x smallest .. 0.5 x largest; interior lines 0.7 x smallest apart |
| live view | every frame shows the whole area (`live_view.fixed_bounds_V`) | |
| gp_path | path kept inside the area, starts at its center | GP length scale = smallest, fit range 0.25 x smallest .. 2 x largest |

**Curvature-sized tracing circles.** A circle of radius r centred on the edge of a round
feature of radius R has an on-feature arc spanning 2 acos(r / 2R) (180 deg = straight
edge), so each circle measures R, and the next one gets r = R (clamped to the limits
above, growing at most 1.5x per circle). A circle that's too big for the feature shows it
the same way (its on-feature arc shrinks) and the next one is smaller. The factors were
calibrated in simulation on features of 0.3-2.4 V: all settings found 100% of the edges,
these gave the fewest circles. With mixed sizes this avoids the fixed radius's failure
(one sample: 8.3% bad pixels with fixed 0.6 V circles, 2.5% with the priors); where the
fixed radius happens to suit every feature it can still be ~1.5x cheaper.

The priors are saved with each run (`priors_json` in the .npz). In Simulation.ipynb the
defaults are the sample's true priors (`priors.from_sample`); `USE_PRIORS = False` runs
without them.

## Tilt: edge detection against the substrate, on the fly

`edge_follow` and `edge_explorer` decide "on a feature" from the **height above a
fitted substrate** (`substrate.SubstrateModel`), not from raw AI0 -- the usual
mask + degree-1 plane fit, done during the scan:

1. **Calibration circle** (`calibration_radius_V`, 0.5 V) round the start gives the tilt
   and the substrate roughness before any decision. The tilt comes from the *derivative*
   of z along the circle (`substrate.slope_fit`), not from a fit of z: a feature covering
   one side of the circle looks exactly like a tilt to a plane fit (on the instrument:
   0.29 V/V fitted for a 0.005 V/V tilt, after which edge_explorer circled the feature
   for 120 segments without closing it), but in the derivative it's only the few samples
   across its edge.
2. **Running plane**: after every waveform, its off-feature samples are added to the
   fit (normal equations, one sample per `substrate_sample_spacing_V` of path).
3. **Threshold** (`mode="auto_mad"`), no feature height needed: at least `mad_k` x the
   roughness of the tilt-corrected substrate over everything measured; once features
   have been seen, it moves into the empty gap between the substrate and feature heights
   (`use_height_gap`). Queued edges found under a lower threshold are dropped if they no
   longer clear it. `mode="manual"`: `manual_threshold` is a height above the plane.
4. The live view shows (and saves) the images with the plane removed
   (`live_view.flatten_substrate`); the `.npz` gets `substrate_json` and
   `substrate_threshold_history`.

Why: on 2026-10-05 a sample tilted ~0.26 V over the 8 V area (features ~0.3 V tall)
gave edge_explorer 9 "features" for 2 real ones -- its single start level and a
6 x noise threshold (8 mV) made it follow lines of constant height on the tilted
substrate. Replayed on that run's data, the model flags 0.4% of the substrate (was
88%); in simulation of the same sample both features are found, the hexagon closed.
Settings: `edge_detect.tilt_correction` (False = constant level), `substrate_degree`
(2 = also bow). Limitation: with `use_height_gap`, a feature much lower than the
others can fall below the gap threshold -- turn it off for such samples.


## Edge explorer: open-loop piezo position errors

On the instrument (2026-10-05) the same commanded point read differently before and
after a long move (the hexagon appeared 0.1-0.3 V further to -x after the 5.7 V first
path out and back): the open-loop X/Y piezo's position depends on where it has been.
`edge_explorer` copes with that in three ways:

1. **Re-finding the edge** (`follow_reacquire_V`, `follow_reacquire_tries`): following
   starts (or restarts the other way) with a probe line across the stored edge point,
   ±max(2 x shortest step, 0.5 V), doubled if it crosses nothing. A probe that finds
   nothing either way means it wasn't an edge -- no feature is made.
2. **Merging** (`merge_features`): a trace that runs into another open edge joins it
   (one edge traced from two places); the loop closes if the joined outline gets back
   round. The other direction also closes the loop when it runs onto the far half of
   the first direction's crossings, or stops within one long step of where the first
   one ended with the outline nearly all the way round.
3. **Two thresholds** (`edge_detect.hysteresis_fraction`, 0.25): on above threshold x
   1.25, off below x 0.75 -- a path along a sidewall gives one crossing, not a burst.

4. **Closing at the area's border** (`close_at_border`, `border_close_distance_V`): a
   feature cut by the border of the area to explore can't close on its own. When both
   directions of its trace end at the border, the border between the two exits closes
   the outline -- the way round that has the feature's side of every crossing inside
   and the substrate side outside -- and its inside is scanned like any other.

The inside scan is denser too: lines 0.3 x the smallest feature apart (was 0.7;
`priors.INTERIOR_SPACING`), at most `interior_max_lines` (40).

In simulation with a model of that piezo memory (the error growing with recent travel,
~0.2-0.3 V after the first path, as measured), a hexagon + a rotated square on a 12 V
area gave 2 closed features with their insides scanned (before: duplicates, the square
split in two open halves).

## Edge explorer: smooth edge following ("weave", optional)

`cfg.edge_explorer.follow_style = "weave"` (default `"segments"`, the zigzag) follows an
edge with a smooth sine weaving back and forth across it instead of straight segments
-- no corners, easier on the piezo. Each waveform is `follow_weave_cycles` (3) cycles,
close together (`follow_weave_wavelength_V` along the edge per cycle) and
`follow_weave_amplitude_V` either side (both default to half the shortest follow step,
at least 0.2 V -- wider than the open-loop position error). After every waveform its
axis is re-fitted to the crossings it just made (they all lie on the edge), so the
next one is centred on the edge and aimed along it, bent by the curvature of a circle
through them; each starts where the last one ended, moving the same way (a smooth
cubic connector when it has to restart elsewhere). A corner shows as crossings that
stop part-way through a waveform: the next restarts at the last crossing, turned
`follow_weave_corner_deg` (60) toward the feature, unless its crossings past the corner
already give the new direction; crossings on stretches this direction already traced
don't count as progress (no backtracking). Exploring is unchanged.

In simulation (hexagon, 30 deg square, circle, with and without the piezo-memory
position error) it closes every shape, with fewer waveforms than the zigzag in 5 of 6
cases (e.g. circle 11 vs 24, hexagon with piezo error 13 vs 27), each waveform carrying
2-3x the path. Figures: `weave_tests/`.

On the instrument (2026-10-06) a shorter wavelength (0.1 V) traced noticeably better than
0.2 V: denser crossings keep the direction estimate right despite the open-loop position
error (in simulation, without those errors, it only costs waveforms).

Closing, both styles (after runs whose features "closed" on outlines that reversed and
came back -- the inside scan then found room for one line or none):
- every closure rule now also requires the joined outline to enclose the feature
  (`loop_closure_min_enclosed`, 0.6), after trimming a trace that wound round more than
  once to its first loop; a rule that keeps firing on a bad outline ends that direction
  (the other one is tried);
- the trace closes onto its own first crossings, not onto the seed point (found on another
  path, with a different piezo history, it can be 0.3-0.8 V off the edge as the trace sees
  it -- aiming at it made the trace miss, turn and go back round);
- weave: the direction after a corner is fitted only once the crossings span a
  wavelength, and is never taken more than ~120 deg from the direction before the corner;
  the no-backtracking radius is at least the amplitude.
Each feature's result (closed, by which rule, enclosure, interior lines and note, outline)
is saved in the .npz as `features_json`.

Found while testing, fixed for both styles: a crossing's height is measured over up to
0.1 V of the feature side (was a few samples, on the sidewall: real edges read ~1/3 of
their height and were later dropped as "no longer an edge"); and when exploring finds
nothing within reach (the tip in a corner after an interior scan) but unexplored area
is left elsewhere, it moves there in a straight measured line instead of stopping.

## Package layout

```
selfdriving_afm/
  config.py                    All tunable parameters, as dataclasses (see below)
  trajectory.py                 Waveform synthesis: line segments, search spirals,
                                 tracing circles. Physical-geometry only, no hardware.
  execution.py                   ContinuousExecutor -- the only thing planners call to
                                 reach a backend. Stitches waveforms into one continuous
                                 curve, fires the live-view callback, applies safety
                                 clamping. See "Continuity" below.
  edge_detect.py                 Height-jump / edge detection (manual or auto-MAD threshold)
  substrate.py                   On-the-fly tilt correction: running substrate plane + threshold
  viz.py                          Plotting helpers for reviewing a run, plus
                                 LiveEdgeFollowView / LiveGPPathView for watching one live
  backends/
    base.py                       The ScanBackend interface every planner depends on
    simulate_backend.py            Synthetic topography, for dry-run testing
    pyscanner_backend.py            Real hardware, via PyScanner's ActiveX interface
  planners/
    edge_follow_planner.py          Spiral -> circular chain state machine
    gp_path_planner.py               GP/Bayesian continuous path planner
    edge_explorer_planner.py         Explore -> follow each edge (zigzag) until closed -> queue
    grid_refine_planner.py           Sparse scan -> detect -> refine regions of tiles, level by level
    gp_edge_explorer_planner.py      Hybrid: GP exploration between objects + edge explorer tracing
    hybrid_planners.py               Hybrids grid_tracker (survey spiral + tracer), gp_spiral (GP + spirals)
  priors.py                       Feature-size + area priors -> every mode's settings
  modes.py                        The scan modes (registry + build/run/save), shared by
                                 the Run and Simulation notebooks
  reconstruction.py               Binning + inpainting (CS, GP, linear, fast, TV)
  pyscanner_scans.py              PyScanner's own scan types (scan_type 1-3) + hardware probe
  stepwise.py                     Stepwise grid_refine, levels 0-1-2, spirals only, no jumps
                                 (StepwiseRun: the good grid_refine dataset's workflow)
  user_raster.py                  The equivalent raster from user-defined smooth waveforms
  reference_raster.py             Reference raster (PyScanner fast raster), RAM-chunked
  topography.py                   Synthetic samples from text, .ibw reader
  simulation.py                   Scoring vs ground truth, mode comparison, sweeps
  gp_presets.py                   Named gp_path strategies (pure_gp_smooth, pure_gp_edge_focus, ...)
  datasets.py                     One loader for every data file (full, lite / Zenodo, stepwise, rasters)
  mcp_server.py                   The MCP server (launcher: scout_mcp.py)
  asylum.py, sensors.py, pfm.py   Igor / Asylum control, position-sensor calibration, PFM channels
demo_simulation.py               Runnable demo, both planners, in simulation
Simulation.ipynb                 Test every mode on a text-defined or .ibw topography
Run_SelfDriving_Scan.ipynb       Instrument entry point: connect, configure, run with
                                 live view, save -- real hardware or simulation
Run_GridRefine_Stepwise.ipynb    grid_refine one level per cell (stepwise.py): spirals only,
                                 no tip jumps, + the equivalent raster when you decide
Analyze_SelfDriving_Scan.ipynb   Reloads a saved run's .npz and retunes reconstruction
                                 parameters against it -- no hardware needed
hardware_paths.example.json      Template -- copy to hardware_paths.json (gitignored,
                                 per-computer) and edit for this machine's PyScanner install
```

**The key design point**: every planner only ever calls
`self.executor.send(x_vec, y_vec, kind=...)`, and the executor only ever calls
`backend.measure_path(x_vec, y_vec) -> z_vec`. Which means the exact same planner
code runs against `SimulateBackend` (no hardware) or `PyScannerBackend` (the real
instrument) -- swap one line, nothing else changes. Use simulation to tune
thresholds/radii/overlap cheaply, then point the same planner at hardware.

## Continuity: every scan starts at (0, 0) and never jumps

Per your spec: every `planner.run()` starts at **(0, 0) V**, and wherever one
waveform finishes, the next one starts -- no teleporting the tip between primitives.

This matters because the planners' own geometry doesn't naturally connect end to
end: a tracing circle of radius R is *centered on* a detected edge point, so its
first sample is R away from that edge point, not at it; a bigger retry spiral (after
a smaller one found nothing) restarts from its own center, not from wherever the
failed attempt ended up. Rather than redesigning each primitive's geometry to avoid
every such gap, `execution.ContinuousExecutor.send()` checks the gap before every
primitive and, if it's nonzero, automatically inserts a straight connecting line
first (sampled at `VelocityConfig.sample_spacing_V`, same speed as everything else) --
so the waveform
actually sent to hardware is always one unbroken curve. These connectors are real,
measured motion (not a bookkeeping-only jump), and show up as dashed gray segments
in the live view so you can see exactly where the tip is transiting versus actively
searching/tracing.

This is also what makes `EdgeFollowPlanner`'s "step off the feature" nudge (below)
a physically real move instead of a virtual position update -- see `execution.py`'s
module docstring for the full reasoning.

## Units: everything is in bias volts

Per your call -- the X/Y piezo drive is open loop, so there's no meaningful
nm-per-volt conversion to hang onto right now. Every radius, path length, and
threshold in `config.py` is in the same units PyScanner sends/receives (volts).
`CalibrationConfig` in `config.py` is a passthrough (`scale_x/y/z = 1.0`) so that
if you get closed-loop calibration later, it's a one-line change, not a refactor.

## Tip velocity: one knob, everywhere

`VelocityConfig` (`cfg.velocity`) is the single source of truth for how fast the tip
moves through bias-volt XY space -- shared by every trajectory primitive (search
spiral, tracing circle, GP path segment, and the auto-inserted connectors above):

```python
cfg.velocity.tip_velocity_V_s = 0.05   # target constant tip speed, bias volts/second
cfg.velocity.sample_rate_hz = 5.0e4    # DAQ output rate assumed for every waveform
```

Point density is *derived* from these two numbers, not set directly --
`sample_spacing_V = tip_velocity_V_s / sample_rate_hz` is the distance between
consecutive samples, and every primitive is built at that spacing. This replaces
the old `points_per_cycle` / `points_per_circle` / `scan_resolution_V` knobs, which
set point count directly and had no notion of real time at all (there was an
`io_rate_hz` field in the very first version of this package, but it was never
actually wired into anything -- dead config, not a real rate).

**Both fields are placeholders pending your real instrument numbers -- CONFIRM ON
HARDWARE**, same category as the two items already flagged in
`backends/pyscanner_backend.py`. I don't know your DAQ's actual AO update rate, or
what tip velocity is safe/appropriate for your piezo.

**The tradeoff to know about**: total points for any trajectory scale as
`arc_length / sample_spacing_V`, i.e. `arc_length * sample_rate_hz / tip_velocity_V_s`.
A slow tip on a fast DAQ over a large area is physically reasonable but can imply
tens of millions of samples for one search spiral -- `trajectory.py` raises a clear
error above 10,000,000 points per primitive rather than silently building it (or
worse, silently trying to fit a Gaussian Process on all of them -- see the GP
planner's `gradient_eval_resolution_V` note below for how that's kept separate).
If you hit that error, it's a real configuration decision to make (faster tip,
coarser rate, or a smaller search radius / segment length), not a bug. `demo_simulation.py`
deliberately overrides `tip_velocity_V_s` much higher than the package default purely
so the demo stays fast at this project's small volt-scale synthetic features --
see its comment before copying that override into a real run.

For the search spiral specifically: an Archimedean spiral's arc-length element
grows with radius (`ds/dtheta = sqrt(r^2 + b^2)`), so constant tip velocity means
non-uniform angular spacing -- `trajectory.spiral_out` resamples a fine uniform-theta
grid by cumulative arc length (vectorized, no per-sample Python loop) to get that
right. A tracing circle has constant curvature, so uniform angle already *is*
uniform arc length there -- no resampling needed.

## The edge-follow planner in detail

State machine per feature:

```
SEARCH  (spiral, growing until an edge is found or max_search_radius_V is hit)
  -> TRACE (chain of circles walking around the perimeter)
    -> loop closes, or edge is lost  -> back to SEARCH for the next feature,
       continuing from wherever the tip currently is (per your spec)
```

Key config knobs (`EdgeFollowConfig` in `config.py`):

| Parameter | What it does |
|---|---|
| `circle.radius_mode` | `"fixed"` (one radius), `"adaptive"` (shrinks near corners, grows on smooth edges) or `"curvature"` (each circle measures the edge curvature and sizes the next one -- set by the priors, see "Priors") |
| `circle.overlap_fraction` | User-set overlap between consecutive tracing circles (0 = touching, ~1 = barely moves) |
| `edge_detect.mode` | `"manual"` (you supply `manual_threshold`) or `"auto_mad"` (threshold estimated from the trace's own noise floor via median/MAD -- no prior knowledge of feature height needed, for unknown samples) |
| `edge_detect.mad_k` | How many MADs above the noise floor counts as an edge, for `auto_mad` |
| `spiral_growth_factor` | How much bigger each retry search spiral gets if no edge was found |
| `first_scan_pattern` | The FIRST search: `"spiral"` (default; the circle inside the area) or `"raster"` / `"lissajous"` over the whole square area, corners included, lines `spiral_pitch_V` apart. Every edge it crosses is traced, nearest first. Later searches are always spirals. In a short simulation (5 features, run until all 5 were traced) a first raster or Lissajous needed ~10% fewer circles, and gave 2-3% bad pixels instead of 12%, because the spiral never reached the corners before the run stopped |
| `follow_all_spiral_edges` | `True` (default): a search spiral that crosses several separate edges traces all of them (next = nearest the tip, skipping any a trace has covered meanwhile) before searching again. `False`: only the first one (the older behaviour) |

**A real bug worth knowing about, since it'll matter if you extend this**: after a
feature's loop closes, the tip is sitting exactly *on* the boundary it just traced.
Two things had to be handled so the planner doesn't just immediately re-detect the
same feature:
1. `_step_off_feature` nudges the search starting point outward, away from the
   feature's centroid, before the next spiral begins.
2. Each completed feature contributes an exclusion disk (`Feature.exclusion_region`)
   covering its whole footprint (not just its boundary points!). A per-point
   distance check isn't enough -- a search spiral that grazes back into an
   already-known feature's *interior* stays "above threshold" for many samples in a
   row, and without the region-based check the planner would eventually pick one of
   those interior samples once it stepped far enough from the boundary ring.

Both were caught by running the simulation against a scene with two separated
features and checking the planner actually moved on to the second one -- worth
doing again with any real sample that has features close together, since the
exclusion margin (`2.0 * circle_radius`, in `run()`) is a heuristic, not a proof.

If you have concave/re-entrant shapes or several features close enough that their
exclusion disks might overlap and mask real feature-to-feature gaps, that heuristic
will need revisiting -- it hasn't been tested against that case.

## Microscope control: Asylum Research / Igor (`asylum.py`)

The Run notebook drives the Asylum software (Igor Pro, over COM:
`win32com.client.Dispatch("IgorPro.Application")`) around every scan -- section 3a sets
up, and every run cell (3b probe, 4 run, 6 reference raster) runs inside
`with afm.run(...)`: engaged during the cell, then **withdrawn and back at (0, 0)
smoothly** when it ends, also if it fails or is interrupted. Section 4b (`afm.finish()`)
does the same by hand.

`afm.setup()`, printed step by step with every Igor command:

0. withdraw -- `DoScanFunc("StopEngageButton")`
1. tip to Igor's designated spot (spot 0, the centre of the image) -- Go There,
   `DoForceFunc("GoForce_1")`, then `cfg.asylum.center_transit_s`; skipped once the X/Y
   loops follow the FPGA in this session (`cfg.asylum.center_first`)
2. the FPGA's AO0/AO1 (PyScanner `get_AO0_V`/`get_AO1_V`) at 0, else moved there smoothly
3. crosspoint `InA <- BNCIn0`, `InB <- BNCIn1` (`CypherInAPopup`/`CypherInBPopup`), locked
   (`CypherXPTLock0Box_0`/`1Box_0`, checked in MasterVariablesWave's `CypherXPTLock` bits),
   `WriteXPT`; then the X/Y loops follow inputs A/B:
   `ir_SetPISLoop(0,"always,never","Cypher.Input.A",0,-8,0,0,"Output.X",-10,150)` and the
   same for loop 1 / B / Y (`cfg.asylum.controller = "arc"` for `Arc.Input.A/B`)
4. engage -- `SimpleEngageMe("")`, then `cfg.asylum.engage_wait_s`

Without Igor (`igor=None`, the simulation) the commands are printed, not sent. Nothing hands
the X/Y loops back to Igor afterwards: the piezos keep following the FPGA (at 0) until
that's undone in Igor.

## The grid + refine planner in detail

```
1. ONE sparse scan over the whole area (raster / Lissajous / spiral, lines 0.7 x the
   smallest feature apart; sparse_crosshatch = a second raster pass the other way)
2. bin + inpaint -> 3. flag structure: local height variation (local tilt removed) > k x noise
4. TILES with flagged pixels, at most tile_max_V wide (priors: 0.5 x smallest feature)
     + fill_interiors: tiles a closed ring of tiles encloses (a feature's outline round its
       flat top), insides up to fill_max_size_V wide (priors: the largest feature)
   tiles touching (sides or corners) = ONE REGION
5. one scan per region, lines refine_factor x closer, pattern from the region's shape
6. back to 2. at the new spacing, only inside the regions just refined
```

**Choosing each region's pattern** (`refine_pattern = "auto"`). Each candidate's path
length to cover the region at the new spacing is compared:

| Region | Test | Pattern |
|---|---|---|
| round | a spiral round its centroid (radius covering `spiral_coverage` = 98% of its tiles, at least ~2 turns) is shorter than a raster | spiral |
| rectangular | the tiles fill at least `rect_fill_min` (85%) of their bounding box | `auto_rect_pattern` (raster, or Lissajous) |
| irregular | anything else | raster, every line clipped to the region's tiles (`trajectory.clipped_raster_scan`) |

Rasters run their lines across the edges (`raster_direction = "auto"`, from the
region's gradients). With `scan_generator = "pyscanner_defined"` a raster covers the
region's bounding box (PyScanner's rasters are rectangles), and each region is one do_scan.

**Why regions.** The ~3 s do_scan overhead per waveform made one-scan-per-tile
expensive on the instrument. In a short simulation (5 features in a 4 V field, 2
levels), PyScanner-defined scans went from 345 waveforms (~1040 s estimated on the
instrument) to 21 (~70 s), with the same edges found (100%) and 1.5% bad pixels
instead of 1.6%. With user-defined scans (a level is one waveform either way) it
scanned ~25% less area at the first refinement level.

**Feature insides.** Only the edges get flagged, so a large feature's flat top was
measured only by the sparse lines. With `fill_interiors`, a closed ring of tiles gets
its inside scanned at the first refinement level (`fill_interior_levels`). Later
levels refine only the edges. An enclosed area wider than the largest expected feature
is substrate between features, and is left out. So is an inside that is open to the
area's border.

### PyScanner fast rasters: the Z time shift

On the instrument the AI record of a fast raster is shifted in time against the path
PyScanner reports (+0.127 s, 2026-10-06), so trace and retrace put every feature at two
places -- the image looks doubled along the fast axis. Every fast raster sent through the
executor (`ContinuousExecutor.send_pyscanner`: grid_refine's sparse and refinement rasters)
is corrected: the shift is fitted from trace/retrace agreement
(`pyscanner_scans.fit_fast_raster_delay`) and Z re-paired with the path at that shift.
Shifts half a line apart fit about equally well (one mirrors the image), so the one
nearest the current estimate is kept -- `cfg.readback.fast_raster_delay_s` to start
with, then the last clear fit; that start value has to be within a quarter line of the
truth. When nothing clearly beats no shift (a flat area), the estimate is used as is.
`cfg.readback.fast_raster_delay = "fixed"` always uses `fast_raster_delay_s`, `"off"` turns
it off (the default since 2026-11-03: the fit gave a bogus half-line shift on the instrument,
and grid_refine no longer needs the fast raster -- see the stepwise section below). Each scan's shift is saved (`step_fast_raster_delay_s`, `..._note`, NaN for other
scans). Spirals and Lissajous scans are not corrected. The reference raster has its own
correction (`reference_raster.fit_delay`, which also checks against the run's image).

## The GP path planner in detail

Same algorithm as your notebook: at each step, evaluate several candidate
directions by predicting the local gradient + GP uncertainty along a short path in
that direction, and commit to a continuous segment in whichever direction
maximizes `gradient + exploration_weight * uncertainty`.

### GP back ends (`GPPathConfig.gp_backend`, v4)

Exact GP regression is O(n^3) to fit, and the original planner re-fit on the *entire*
accumulated point set *and* re-optimised the kernel hyperparameters every time. At the
Run-notebook settings (0.5 V segments, 0.005 V training stride = 100 points per
segment) a 60-segment run reaches ~6000 points, where one refit took ~20 minutes in
simulation (2-core CPU) and the tip waited up to ~260 s between segments. `gp_backend`
picks how much work a refit does (`gp_models.py`; same kernel -- C * RBF + White,
normalize_y -- and the same priors in all of them):

| `gp_backend` | each refit | per-segment compute at ~6000 points | notes |
|---|---|---|---|
| `"sklearn_exact"` (default) | re-optimise the kernel on all points + exact solve | minutes, grows ~n^3 | the original behaviour, unchanged (same path, point for point) |
| `"frozen"` | kernel learned until `freeze_after_n` points, then one exact Cholesky | ~2 s (worst ~11 s), grows ~n^3 | |
| `"local"` | frozen kernel, exact GP on the `local_n_points` nearest the tip | ~0.05 s, constant | whole-area maps use a sparse model on all the data |
| `"sparse"` | frozen kernel, sparse variational GP (Titsias 2009) on an inducing grid over the visited cells | ~0.03 s | made the same direction choice as `sklearn_exact` at 6/6 replay checkpoints |

Recommended for the instrument: `"sparse"` (or `"local"`) with `refit_every = 1`.
Both are plain numpy/scipy/scikit-learn -- no torch or GPU needed. A 200-segment
(20,000-point) simulated run with `"local"` stayed at 0.045 s median / 0.73 s worst
per segment. The worst steps are the warm-up: until `freeze_after_n` points, each refit
also re-learns the kernel (~0.4 s on `hyper_opt_subset` random points, always starting
from the priors' length scale -- warm-starting from the previous fit let it drift).

Every run's per-segment compute is printed at the end (`planner.timing_summary()`) and
saved in the .npz as `gp_timing` (segment, n, fit_s, acquisition_s) and `gp_backend`.
With `"sklearn_exact"`, a refit slower than 5 s prints a one-time hint to switch.

Whole-area maps -- the live view, the `target_mean_uncertainty` stop check and the
saved reconstruction -- go through `predict_global`: a sparse model on all the data for
`frozen`/`local`/`sparse` (an exact solve over the 300 x 300 save grid at 6000 points
takes minutes). `"sklearn_exact"` still uses the exact GP for these, as before.

Benchmark scripts and results (CPU, Colab T4): `../gp_benchmark/`.

## MCP server: drive the package from an LLM client (v4)

`selfdriving_afm/mcp_server.py` (launcher: `scout_mcp.py`) exposes the whole package as
MCP tools, so Claude Desktop, Claude Code or any MCP client can configure, simulate, run
on the instrument and analyze -- every mode (edge_follow, edge_explorer,
domain_wall_explorer, grid_refine, gp_path and its presets, the hybrids gp_edge_explorer,
grid_tracker, gp_spiral, and the stepwise grid_refine). It runs the same code as the
notebooks: the config comes from Run_SelfDriving_Scan.ipynb's configuration cell, a scan is
`modes.build` + `modes.run` inside `AsylumControl.run` + `modes.save`, the analysis reads
the same .npz files.

**Install** (the PC that runs the client; on the instrument PC also pywin32):

    pip install "mcp>=1.2"          # works with the 1.x (FastMCP) and 2.x (MCPServer) SDK

**Claude Desktop**: Settings -> Developer -> Edit Config, add (see
`mcp_config.example.json`; `where python` in Anaconda Prompt gives the python path):

    {"mcpServers": {"selfdriving-afm": {
        "command": "C:\\path\\to\\python.exe",
        "args": ["C:\\...\\self_driving_afm_v4\\selfdriving_afm\\scout_mcp.py"]}}}

then restart Claude Desktop. **Claude Code**:
`claude mcp add selfdriving-afm -- python "C:\...\selfdriving_afm\scout_mcp.py"`.
The server's working folder is the launcher's folder (notebooks, hardware_paths.json,
scan_data/).

**Tools** (each documented for the LLM; `server_info` and `guide` explain the workflow):

| group | tools |
|---|---|
| session / config | `server_info`, `guide`, `list_modes`, `load_run_config`, `get_config`, `set_config`, `set_modality`, `apply_gp_preset`, `set_priors` |
| simulation | `set_sample` (three_shapes / original / five_shapes / custom text / an `.ibw` / a measured dataset, e.g. the Zenodo ground truth), `simulate`, `compare` (modes and/or GP presets, + the raster with the same budget) |
| instrument | `connect_instrument` (or `dry_run=True`), `instrument_status`, `microscope_setup`, `run_scan` (`tail_dwell_s`), `acquire_reference_raster` (the equivalent raster: `method="user"` user-defined smooth waveforms, default, or `"pyscanner"`), `probe_pyscanner_scans`, `withdraw_and_park` |
| jobs | `job_status` (waits up to `wait_s`, log tail, result + figures), `list_jobs`, `stop_job`, `live_snapshot` |
| stepwise grid_refine | `stepwise_grid_refine` (setup, stages 1-3, save, figures, animate, equivalent raster -- on the instrument or `simulate=True`) |
| analysis | `list_scans`, `scan_summary`, `reconstruct_scan` (tv / linear / ..., levelled, z or a PFM channel, next to the equivalent raster, V or um), `plot_reference_raster`, `animate_scan`, `run_analysis` (the `analysis/` scripts) -- every one reads full runs, lite / Zenodo copies, the stepwise dataset and rasters |

**Safety.** Every tool that moves the tip (`microscope_setup`, `run_scan`,
`acquire_reference_raster`, `probe_pyscanner_scans`, `withdraw_and_park`, and
`stepwise_grid_refine` on the instrument) refuses unless
called with `confirm=True`, and the server instructs the LLM to set it only after you've
asked for that action in the conversation (clients also see these tools marked
destructive). Scans run inside `AsylumControl.run`: engaged for the scan, withdrawn and
back to (0, 0) at the end -- also on an error or `stop_job` (which stops at the next
waveform and still saves what was measured). One hardware job at a time; all hardware
calls run on one thread (COM objects must stay on the thread that made them).

**Dry run.** `connect_instrument(dry_run=True)` runs the whole acquisition path (setup
with a simulated sensor calibration, priors, scan, withdraw/park, save, analysis) on the
simulated backend with no microscope and Igor commands only printed -- use it to rehearse
a session.

**Long work** returns a job id: `job_status(job_id, wait_s=50)` until done / failed /
stopped; figures come back as images (paths in the JSON). The package's prints go to the
job log, never to the MCP channel. grid_refine simulations keep one block-mean sample per
10 in RAM by default (`store_every`; its fine rasters are tens of millions of samples,
several GB at full resolution); the instrument-time estimate still uses the true counts.

**Example session (testing the GP modes)** -- ask the client, e.g.:
1. "Show me the server info and the GP presets."
2. "Compare pure_gp, pure_gp_smooth and original on the three_shapes sample with a 60 V budget."
3. "Connect to the instrument" -> "Run the microscope setup" (it asks you to confirm) ->
   "Set priors: smallest feature 0.5 V, largest 2 V, area half-width 4 V."
4. "Run gp_path with pure_gp_smooth and a 60 V budget" (confirm) -> "How is it going?"
   (job_status / live_snapshot) -> "Acquire the reference raster" (confirm) ->
   "Withdraw and park."
5. "Summarize the scan and reconstruct it with linear interpolation; make the replay GIF."

### GP presets (`selfdriving_afm/gp_presets.py`, v4)

Every gp_path strategy tested in simulation, by name. In the Run notebook's config cell:

    GP_PRESET = "pure_gp_smooth"
    apply_gp_preset(cfg, GP_PRESET)      # then override any field below it

| preset | what it does |
|---|---|
| `original` | the first scoring (gradient + exploration x sigma) on the fast sparse GP: mostly explores |
| `hybrid` | GP edge-weighted exploration + rule-based edge tracing |
| `pure_gp` | variance reduction (IVR), `ivr_lambda` 0.8, 2 V lookahead, straight segments |
| `pure_gp_smooth` | `pure_gp` with segments tangent to each other (max turn 135 deg) -- the default |
| `pure_gp_1V` / `pure_gp_smooth_1V` | 1 V segments: half the waveforms, coarser map |
| `pure_gp_edges` / `pure_gp_edges_smooth` | `ivr_lambda` 1.0: uncertainty reduction only where the GP sees edges |
| `pure_gp_edge_focus` | `pure_gp_edges_smooth` with `ivr_interest_power` (gamma) 2: weight (1 - lambda) + lambda x interest^gamma -- the most edge-focused run that still finds every object (2026-11-09) |

Each preset sets the same full list of strategy fields, so switching never leaves a setting
behind. Simulation.ipynb section 5b runs all of them on its sample with the same path budget
(`gp_presets.variants()` gives the `sim.compare` variants), plus the matching raster.

### Pure GP planner: integrated variance reduction (`acquisition = "ivr"`, v4)

The Run notebook now uses a pure-GP planner: no edge-tracing rules, no geometric
coverage proxy. At each step, for each of 16 directions:

    score = sum over map points x near the tip of  w(x) * [sigma^2_now(x) - sigma^2_after(x)]
    w(x)  = (1 - ivr_lambda) + ivr_lambda * interest(x),   interest = |grad mu| / (h / l)

sigma^2_after is the GP posterior variance if the tip measured along that direction for
`lookahead_V` (2 V). For a GP it is exact and known *before* measuring: posterior variance
depends only on where the data are, not on their values (verified against a full refit to
1e-16). Only the first segment is executed, then the planner re-plans (receding
horizon). `ivr_lambda` is the single exploration/exploitation knob: 0 = minimize the
total map uncertainty; 1 = spend the uncertainty reduction where the GP predicts edges
(h = feature height, l = length scale). Same GP model as before: sparse, Matern 5/2,
detrended, length scale >= 0.5 x the smallest feature. Compute: ~0.1 s median per segment
at 60 V, ~0.4 s at 120 V (24,000 points), <= 2.3 s worst.

Simulation (3-, 5- and a held-out 3-shape sample; path-matched raster = a classic raster
with the same tip travel):

| 60 V of tip travel | bad px | worst-covered shape |
|---|---|---|
| raster, same path | 8.6% | 0% |
| original GP (`legacy`) | 32% | 0% |
| hybrid (GP + trace rules) | 11% | 4% |
| pure GP (`ivr`, lambda 0.8) | 8.2% | 0% |

At 120 V the raster reaches 5.5% bad px, pure GP 8.4% (lambda 0.8) -- and 2.9% with
lambda = 1.0 on the 3-shape sample (one seed). Pure GP reaches shapes the raster's few lines
miss, but never the rectangle against the bottom border of the area.

**Smooth turns** (`smooth_turns = True`, `max_turn_deg = 135`, on in the Run notebook):
every segment starts tangent to the end of the previous one -- the cubic Hermite connector
of the edge explorer's weave, from the tip (along its heading) to the planned end point
(along the new direction). The planner scores the curve the tip will really travel;
directions whose curve leaves the area or turn more than `max_turn_deg` are skipped. Tighter
turns concentrate the bend: minimum radius ~0.17 V at 45 deg, 0.055 V at 90, 0.012 V at
135 (0.5 V segments). Simulation, 3 samples:

| | 60 V bad px | 60 V worst shape | 120 V bad px | 120 V worst shape |
|---|---|---|---|---|
| pure GP, straight | 8.2% | 0% | 8.4% | 10% |
| pure GP, smooth | 12.2% | 5% | **4.7%** | **22%** |
| raster, same path | 8.6% | 0% | 5.5% | 8% |

With 1 V segments smoothing hurt the map (17-19% bad px): long curved segments leave
wider gaps. Keep 0.5 V segments with smooth turns.

**Does the GP make a real difference on this instrument? For fast topography, not yet --
because of time.** Each segment is one user-defined waveform: ~0.4 s of fixed overhead
(measured: the 2026-09-24 probe's user-defined line, and a real 500-waveform edge_explorer
run, 0.39 s per waveform -- the 3 s figure applies only to PyScanner-defined scans) plus
the GP compute (0.1-0.6 s per segment here). A 0.5 V segment at 30 V/s is 0.017 s of
motion, so >95% of a GP run is waiting: 60 V ~ 1.1 min, 120 V ~ 3-3.7 min. In the same
1.1 min one raster waveform covers ~1,300 V (82 lines, 0.2-0.4% bad px); in 3.6 min a
full 200-line image (0% bad px). Adaptive scanning pays when a segment's motion takes
longer than its overhead + compute -- slow scans (PFM/KPFM at a few V/s or less,
spectroscopy, large areas), or several planned segments sent as one waveform.
Figures and data: `../gp_benchmark/pure_gp/`.

### GP path acquisition: how the pathfinder chooses where to go (v4)

`cfg.gp_path.acquisition = "legacy"` (the package default, unchanged) scores each candidate
direction by mean |d mu| per 0.005 V step + `exploration_weight` x mean sigma. At the
Run-notebook settings the gradient term is ~20x smaller than the uncertainty term, so the
path only explores: in simulation 85-95% of the path was on flat substrate and at most
one feature was ever reached.

The Run notebook now uses the strategy tuned in simulation (2026-11-06):

- **Explore** (`acquisition = "edge_uncertainty"`, `explore_measure = "distance"`): each of
  16 directions is scored by edge strength x normalized GP sigma along the next segment
  (edges not measured yet) + `exploration_weight` x coverage, where coverage = distance to
  the nearest measured point / `explore_radius_V` (0.7 V), averaged `lookahead_V` (1.5 V)
  ahead; ground outside the area counts as covered, so the border stops attracting the path.
- **Trace** (`trace_edges = True`): when a segment crosses an outline (measured height vs
  the fitted substrate plane, threshold = half the feature height, never below 5x the
  substrate scatter), the tip follows it in `trace_step_V` (0.2 V) steps along the GP's
  outline tangent (perpendicular to the GP-mean gradient), steering
  `trace_correction_deg` toward the feature when off it and away when on it, until the
  loop closes, the outline leaves the area, or `trace_max_length_V` (default
  1.5 x pi x the largest feature). A traced feature is never traced again, so no single
  feature eats the budget. Traced outlines: `planner.traced_features`.
- **Model**: Matern 5/2 kernel (`gp_kernel`; RBF rings at sharp steps), GP on the residuals
  from a robust substrate plane (`gp_detrend`; unvisited ground predicts the tilted
  substrate, not a flat mean), length scale >= 0.5 x the smallest feature
  (`length_scale_min_frac`).
- **Budget**: `max_path_length_V` (total tip travel) is the scan-time limit; `max_segments`
  is only a safety cap now (tracing uses short steps).

Simulation results (60 V path budget = 120 x 0.5 V, 3 noise seeds each; edge coverage =
fraction of each outline the path came within 0.08 V of; bad px = map pixels off by
> 20% of the height range):

| sample | legacy: edge coverage per feature | bad px | tuned: edge coverage per feature | bad px |
|---|---|---|---|---|
| A3 (hexagon, rectangle, circle) | 0.32 / 0.00 / 0.02 | 44% | 0.12 / 0.16 / 0.58 | 8% |
| B5 (+ triangle, square) | 0.32 / 0.00 / 0.02 / 0.00 / 0.00 | 43% | 0.34 / 0.00 / 0.56 / 0.39 / 0.36 | 14% |
| C3 (held out: ellipse, square, triangle) | 0.11 / 0.11 / 0.33 | 34% | 0.40 / 0.29 / 0.26 | 17% |
| O2 (the original hexagon + rectangle) | 0.32 / 0.00 | 44% | 0.10 / 0.41 | 6% |

GP compute stayed at ~0.07-0.14 s median, <= 2.1 s worst per segment (sparse back end).
For comparison, `edge_explorer` traces 82-100% of every outline but uses 330-500 V of
path on these samples: the GP pathfinder is the budget-limited mode. Known limit: a
feature whose outline lies within ~0.5 V of the area border is the most often missed
(the rectangle in A3/B5 at 0-16% coverage). Tuning scripts: `../gp_benchmark/tuning/`.

**Instrument time is dominated by per-waveform overhead + GP compute, not motion.**
Every segment is one user-defined waveform (~0.4 s overhead) and one GP step (~0.1-0.6 s).
Fewer, longer waveforms (several planned segments per waveform) are the next lever.

**Known model limitation (all back ends)**: the RBF kernel is smooth, so once the path
crosses a sharp step edge the fitted length scale drops to its lower bound
(0.25 x the smallest feature, from the priors) -- the original exact GP does the same
on the same data. Separately, the acquisition's gradient term (mean |d mu| per
`gradient_eval_resolution_V` step) is ~20x smaller than `exploration_weight` x sigma at
these settings, so the path is currently driven mostly by uncertainty, not edges.
Both are planner-design items for a follow-up, not speed issues.

**`gradient_eval_resolution_V`** is the GP's own working resolution -- how finely
candidate directions are evaluated against the already-fitted GP (`predict()` on
synthetic points, never touches hardware) and, separately, how much each measured
segment is subsampled before being added to the GP's training set. It's
deliberately decoupled from `VelocityConfig.sample_spacing_V`: at a slow tip
velocity the *physical* sample spacing can be far finer than the GP could ever
train on tractably (GP regression is O(n^2) memory / O(n^3) fit cost -- training
directly on tens of thousands of raw samples from one short segment doesn't just
get slow, it can exhaust memory outright, which is what a first version of this
change actually did before I added the subsampling). The full-fidelity measured
data is never lost -- `ContinuousExecutor.history`/`all_points()` still has every
raw sample; only what the GP itself trains on and predicts from is thinned.

## When does a scan stop?

Every criterion below is a plain config field: set one to `None` to turn it off,
set it to a value to turn it on. **Nothing is mutually exclusive** -- every
criterion you've enabled is checked every cycle, and whichever one fires first
ends the run. That's the "toggle" model: there's no single `stop_mode` switch to
flip between them, because on a real, unknown sample you generally want several of
these active simultaneously (e.g. time as a safety net *and* a quality target), not
a choice of exactly one. The notebook's config cell shows all of them commented
with what each does.

**Edge-follow (`EdgeFollowConfig`)**, three criteria:
1. **`max_features`** (count) -- stop once this many features are traced. The
   natural choice when you have a target count in mind. Exhausting
   `max_search_radius_V` (no edge found even after growing the search that far) is
   always active alongside it, as the "nothing else nearby" complement -- it's not
   a separate toggle, it's how the planner gives up on a position.
2. **`max_run_time_s`** (time) -- wall-clock cap in seconds (e.g. `120.0` for 2
   minutes). Runs independently of the above, so it's the right one to lean on when
   you don't have a good a-priori guess for `max_features` on an unknown sample --
   set it generously at first, tighten it once you've seen how long a real feature
   actually takes to trace.
3. What I'd skip for now: a coverage/area-based stop (e.g. "X% of a bounding box
   sampled"). This planner targets discrete features, not blanket coverage, so area
   coverage doesn't map cleanly onto "found what we came for" -- that criterion fits
   the GP planner much better (below). If you want it anyway (e.g. to bound a
   search over a known finite region), it's a small addition against the same live
   reconstruction machinery `viz.LiveEdgeFollowView` already builds.

**GP path (`GPPathConfig`)**, three criteria:
1. **`max_segments`** (count, renamed from `num_path_segments`) -- hard cap, always
   enforced regardless of the other two.
2. **`max_run_time_s`** (time) -- same wall-clock cap as above.
3. **`target_mean_uncertainty`** (quality, **my recommended "good enough"
   criterion**) -- stop once the GP's own mean uncertainty (sigma) over a
   `stop_check_extent_V`-sized region centered on `start_xy` drops at or below this
   value, checked at the same cadence as `refit_every`. This is the most direct
   answer to "is the reconstruction good enough": the GP already carries a
   calibrated uncertainty field, so it's answering exactly that question rather
   than a proxy for it (a fixed segment count gives no signal about whether the
   surface is actually well-resolved, only that a certain amount of *effort* was
   spent). `min_segments_before_stop_check` (default 8) guards against stopping too
   early on a lucky-quiet initial patch.

Two variants of #3 I'd consider next if the flat mean-uncertainty threshold isn't
discriminating enough on real data: (a) **coverage fraction** -- stop when e.g. 95%
of a grid's cells are below the threshold, rather than the mean, so one lingering
poorly-sampled corner can't be averaged away; (b) **acquisition-value decay** --
stop when the best candidate direction's `gradient + exploration_weight*uncertainty`
score itself drops below a threshold, i.e. "nothing left looks informative," which
is a more direct read of the planner's own decision process than the reconstructed
surface. Neither is implemented -- ask if you want one.

## Readback: two modes, your call which to trust

`ReadbackConfig.mode` in `config.py`:
- `"raw"` (default) -- reads a per-sample channel matched 1:1 to the
  `UI_AO0_V`/`UI_AO1_V` waveform you sent. This is the natural fit for the
  non-raster trajectories here (spirals, tracing circles) since there's no binning
  ambiguity.
- `"image"` -- reads the existing binned `image_AI0`/`image_AI1` and maps each
  requested (x, y) sample back to its nearest pixel.

## Position sensors: correcting for open-loop X/Y (optional)

The X/Y piezo drive is open loop (see "Units"), so the commanded waveform isn't
necessarily where the tip actually ends up -- piezo nonlinearity, hysteresis, and
drift all mean the *sensor* reading (if your instrument has one) is the actual
measurement, and the commanded value is only a request. `PositionSensorConfig`
(`cfg.position_sensors`) is how you use that sensor to correct for it:

- **Off by default, not required.** With `enabled = False` (the default), nothing
  about this exists at runtime -- `TrajectoryStep.x_sensor`/`y_sensor` are `None`
  everywhere and behavior is exactly what it was before this option existed.
- **Channel map, per your ask**: `x_ai_channel`/`y_ai_channel` default to `AI1`/`AI2`
  (Z/deflection stays `AI0`, `ReadbackConfig.ai_channel`) -- CONFIRM ON HARDWARE like
  everything else in this list.
- **When enabled**, `ContinuousExecutor` calls `backend.read_position_sensors(x_vec,
  y_vec)` right after every `measure_path`, for every step (search spirals, edge
  traces, GP path segments, and auto-inserted connectors alike), and stores the
  result on that step. `PyScannerBackend` gets this for free from the same
  `AI_vs_time` array Z already came from (one ActiveX call, not two -- see its
  module docstring); needs `readback.mode == "raw"`. `SimulateBackend` fakes a
  plausible open-loop error (fixed gain + offset + noise) purely so you can exercise
  this option before ever touching hardware.
- **Scoped to data, not decisions.** `ContinuousExecutor.all_points()` -- and so
  everything downstream of it: the live reconstruction, the saved `points` array,
  `Analyze_SelfDriving_Scan.ipynb` -- prefers sensor position over commanded
  whenever it's present. The planners' own live decisions (where to center the next
  tracing circle, which direction the GP explores) still use the commanded position
  they asked for. Feeding sensor position back into that live loop is a real,
  bigger design question (sensor noise/lag could destabilize the search state
  machine) -- ask if you want to go there; this is deliberately scoped to
  correcting what gets *reconstructed and saved*, not to changing how the planner
  decides where to go next.

### Position sensors on the Cypher: routing, calibration, micrometres (`sensors.py`)

The controller's XSensor / YSensor (LVDTs, in sensor volts) reach PyScanner through BNC outputs:
`AsylumControl.route_sensors` sets a Cypher PIDS loop per axis (`cfg.position_sensors.x_loop` /
`y_loop`, default 3 / 4 -- never 0 / 1, which are X/Y following the FPGA) as a straight copy
(input the sensor, setpoint 0, P = -1) to a free DAC, routed (locked) to BNCOut0 / BNCOut1,
wired to AI1 / AI2. The setup cell does that and then **calibrates** (tip withdrawn): a 3 x 3
grid of drive positions, at each the AI1/AI2 mean and Igor's own XSensor/YSensor reading ->

    AI V -> sensor V (gain ~1: checks the wiring) -> metres (Igor's XLVDTSens / XLVDTOffset)
         -> drive V (a straight-line fit of the sensor position against the commanded one)

Every waveform's sensor samples go through that chain (`executor.sensor_cal`), so positions
(`all_points`, the live view, the planners' maps, the saved `points`) are the **measured**
ones, expressed in the drive's frame -- the piezo's errors are gone, and the pixel grid stays
in drive volts, 1 V = `um_per_V` micrometres. The live view and the Analyze notebook (4a)
label the axes in um. Saved too: the raw AI volts (`x_sensor_raw_V`), the calibration
(`sensor_calibration_json`, `um_per_V`, `um_at_0V`).

**Sensors past +-10 V:** the BNC outputs (and PyScanner's AI) stop at +-10 V. With
`cfg.position_sensors.divide_output = True` the copy loop's P gain is -1 / `output_divider`
(default 2: OutC/OutD = sensor / 2), so the whole X/Y range fits. The calibration fits AI volts
against Igor's own XSensor/YSensor reading, so it measures the divider (AI gain ~2) and the
positions stay exact; only the AI resolution per nm halves. A calibration measured with a
different divider is dropped at the next run cell (rerun the setup cell).

**Planning on the sensors, and correcting the drive** (with a calibration that passed its checks):

- `plan_on_sensors` (default on): every mode decides where to scan next from the **measured**
  positions. `executor.send()` keeps each returned z's measured position (`executor.measured_xy`),
  and the planners pair z with those, not with what they commanded. That covers edge_follow
  (calibration circle, spirals, first search, tracing circles), edge_explorer (every path: crossings,
  substrate fit, coverage, closing, insides) and gp_path (the GP's data). grid_refine already
  worked from `all_points()`.
- (a) `correct_linear`: the setup fits the drive map (measured = A drive + c, a full 2 x 2 with the
  cross-talk). Every waveform is sent as A^-1 (target - u): the planners give *targets* (where the
  tip should be), and the executor works out the drive.
- (b) `correct_offset`: after each waveform, the measured error over its last `correction_window_s`
  updates u, scaled by `correction_gain` and capped at `correction_max_V`. A new u is eased in over
  the next waveform's first `correction_ramp_s`, so the drive never jumps. PyScanner's own scans
  (grid_refine) can only be *placed*: centre through the map and u, size through its scale. A
  change of u is eased in by the move to the scan's start. The u used is in each step's
  `meta["drive_correction_V"]`, and `executor.correction_log` has every update.
- The live view's first row is still what was applied to the piezos (the corrected drive), and
  the sensor row is where the tip went.

An Igor scan (or a mode change) **resets** the routing -- seen on 2026-10-08: loops 0/1 back
on LVDT.X/Y, InA/InB back on ACDefl/Defl, unlocked. So every run cell (`afm.run`) checks it
first and redoes it (X/Y -> FPGA and the sensors) before engaging. The calibration is kept for
the session (rerun the setup cell after a kernel restart).

## Before running on real hardware

Everything in `backends/pyscanner_backend.py` is written directly from your
example scripts, **except** the lines marked `# CONFIRM ON HARDWARE`:

1. **The raw per-sample readback control name.** I used `'AI_vs_time'`, matching
   the commented-out reference in the Hedgehog notebook -- but it was commented
   out, never called, so I don't know for certain it's the right name, whether it
   returns one channel or a `[n_channels, N_points]` array like
   `AO_vs_time_shifted` does for outputs, or whether it's sample-synced 1:1 with
   what you sent. `measure_path` raises a clear error if the returned length
   doesn't match, so this will fail loudly rather than silently, but check the
   front panel before a real run.
2. **`N_image_pixels`** -- used for `readback.mode == "image"`; confirm this is
   the correct control name for setting the reconstructed image resolution.
3. **The channel map itself.** `AO_X_CONTROL`/`AO_Y_CONTROL` (module-level constants
   at the top of `pyscanner_backend.py`, currently `"UI_AO0_V"`/`"UI_AO1_V"`) drive
   the X and Y piezo axes; Z (tip deflection) has no AO at all -- it's readback only,
   on whichever AI input `cfg.readback.ai_channel` names (default `"AI0"`). All three
   are carried over from your example scripts, not guessed, but confirm the X/Y
   assignment specifically -- if they're swapped on your instrument, every trace in
   this package will be transposed.

Nothing else in the package needs to change if these turn out to have slightly
different names or shapes -- they're isolated to that one file.

**Also per-computer, not per-file**: `hardware_paths.json` (gitignored -- copy
`hardware_paths.example.json` once per computer and edit `exe_path`/`vi_path`) holds
the PyScanner install location, since that's different on every microscope's
machine. `Run_SelfDriving_Scan.ipynb` reads it rather than having the path typed
into the notebook, so the same notebook works unmodified when copied to a different
computer -- you just need that one local file to exist there too.

**Safety**: every generated waveform is clamped to `SafetyConfig.max_amplitude_V`
(10 V by default, matching your instrument) in exactly one place --
`ContinuousExecutor._measure`, which every planner and every auto-inserted
connector goes through, and again, independently, inside
`PyScannerBackend.measure_path` itself, so a bug in a planner (or in the executor)
can't skip the check. Default `clamp_mode` is `"raise"`, not `"clip"` -- an
out-of-range waveform stops the run rather than silently sending something smaller
than what was planned.

## Live view: watching the reconstruction as the scan runs

`viz.LiveEdgeFollowView` and `viz.LiveGPPathView` are the two "sit and watch it"
pieces, one per modality, both wired in exactly the same way: pass the view's
`.on_step` method as `on_step=` to `ContinuousExecutor`, and the figure updates
itself as each segment (spiral, circle, line, or auto-inserted connector) comes
back from the backend -- on real hardware exactly the same as in simulation, since
the view only ever looks at `ContinuousExecutor.history`.

- **Edge-follow**, three columns, all sharing the same X/Y axes: (1) the path so far,
  color-coded by kind (blue = spiral search, green = edge trace, purple = GP path,
  dashed gray = transit connector); (2) every raw measured point, plotted spatially
  and colored by its own Z/AI0 value (`cmap`, `copper` by default) -- the sparse,
  un-inpainted counterpart of column 3, useful for seeing exactly what was measured
  where before any interpolation; (3) the topography reconstructed from every point
  measured so far (nearest-neighbour fill early on, upgrading to linear interpolation
  once the path covers enough of the plane to triangulate).
- **GP path**: left panel is the path so far; the other two are the GP mean
  (reconstruction) and uncertainty surfaces, refreshed whenever the planner has
  actually re-fit the GP (same cadence as `GPPathConfig.refit_every`). Doesn't have
  the raw-scatter column edge-follow's view does -- ask if you want it added there
  too.

All three columns (both modalities) share **dynamic axis bounds**
(`_LiveViewBase._bounds`): a square window, with padding, around everywhere the tip
has actually been so far -- not the fixed `extent` the view was constructed with
(`LIVE_VIEW_EXTENT_V` in the notebook), which is only a guess used before any data
exists. The window grows as the scan covers more area, so the whole scanned region
stays in view without needing to guess its final size up front.

Redraw cadence is **wall-clock throttled** (`LiveViewConfig.min_redraw_interval_s` /
`min_reconstruction_interval_s`), not tied to executor step count -- a single spiral
can be thousands of samples executed in milliseconds on real hardware or
microseconds in simulation, so gating on "every N steps" either redraws far more
than a human can watch or barely at all depending on backend speed. Time-based
throttling keeps it feeling live either way and guarantees the live view itself
can't become the bottleneck in the acquisition loop. The reconstruction panel also
subsamples once the point count passes `max_reconstruction_points` (keeping the
most recent third in full, thinning the rest) so `griddata`/GP re-fits stay fast
late in a long run.

Each redraw is pushed as a static PNG (`display()` + `clear_output(wait=True)`) into
whichever cell is currently executing -- not drawn via the interactive `ipympl`
canvas's own update pipeline, which needs a browser round-trip that can't complete
while the kernel is blocked inside a hardware call. See
`viz._LiveViewBase._redraw`'s docstring for the full reasoning. Practically: the live
view during a run shows up in **cell 4's own output** in the notebook, not layered
onto cell 3's (which only shows the initial empty frame).

`Run_SelfDriving_Scan.ipynb` wires all of this up already -- see below.

### Uncertainty map (and, in simulation, the true error)

Every row of the live view (and so of the Simulation notebook's GIF) and of the Analyze
notebook's animation has a 4th column: **how far each reconstructed pixel is from any
measurement** (`reconstruction.uncertainty_map`): 0% on a measured pixel, rising linearly
with the distance to the nearest measured one, 100% at the smallest expected feature
(`priors.feature_min_V`) or more -- a gap that wide could hide a whole feature. It says
where the data is, not how good a reconstruction method is between it: the same for every
mode, pattern and method. Between raster lines it's low when the lines are close compared
with the smallest feature, and an unscanned area stays at 100% until it's scanned.
In simulation a 5th column shows the **real** error |reconstruction - truth| (tilt removed
as in the images), with the fraction of pixels off by more than 20% of the height range --
to see where the uncertainty map is right and where a reconstruction invents structure
even next to data. `cfg.live_view.show_uncertainty` / `show_true_error` turn them off;
`UNCERTAINTY_COLUMN` / `UNCERTAINTY_SCALE_V` in the Analyze animation cell.

## Comparison to the older spiral + circular-scan notebook

You asked whether `EdgeFollowPlanner` matches the spiral -> circular-scan strategy
in `older scripts and packages/self-exploratory scan/scripts/Adaptative_scanPath_v3.ipynb`
(the one with the circular-scan animation -- `v4`/`v5` moved to the GP-only
path-based approach instead, so `v3` is the actual point of comparison). Same
two-phase idea, several concrete differences:

| | Old (`v3`, `adaptive_circular_scan`) | This package (`EdgeFollowPlanner`) |
|---|---|---|
| Edge criterion | fixed `gradient_threshold` on consecutive-sample differences | `EdgeDetectConfig`: fixed threshold, or `"auto_mad"` -- robust median/MAD estimated from the trace itself, no prior knowledge of feature height needed |
| Circle radius | fixed (`circle_radius = 40`) | fixed, or `"adaptive"` (shrinks near corners, grows on smooth edges) |
| Next circle center | recenter 70% toward the circle's *averaged* edge-crossing centroid, then step a fixed `circle_step` along a tangent estimated from that same average | step from the *specific* crossing farthest from the incoming direction, by `2*radius*(1 - overlap_fraction)` -- no separate recentering pass |
| Avoiding re-detecting a known feature | keep every past edge-crossing *point*; reject a new candidate within `min_distance` of any of them | keep one exclusion *disk* (centroid + footprint radius + margin) per finished feature -- a per-point check alone misses a search spiral grazing back into a feature's *interior*, which is still "above threshold" (this was caught and fixed during this build; see `execution.py`/`edge_follow_planner.py` docstrings) |
| Stepping off a closed feature | none explicit -- relies solely on the point-avoidance check above | explicit outward nudge (`_step_off_feature`) before the next search spiral, now sent as a real connector move (see "Continuity") |
| Search retry | one spiral attempt, fixed `max_radius`; no edge found there ends the whole scan | grows the search radius (`spiral_growth_factor`) and retries from the same position, up to `max_search_radius_V` |
| Units | nm, fixed synthetic grid | bias volts throughout (open-loop piezo -- see "Units" above) |
| Continuity | circle-to-circle and spiral-to-circle both jump (same structural gap as this package originally had) | every transition connected, always starts at (0, 0) -- see "Continuity" above |

None of this is a value judgement on the old version -- it's what it needed to be
for a fast notebook exploration. The MAD threshold, adaptive radius, and
region-based exclusion are genuine upgrades worth keeping for an unknown real
sample; if you'd rather match the old centroid-recenter/tangent-step mechanics
exactly (e.g. to reproduce a specific past result), that's a targeted change
localized to `_trace_perimeter` in `edge_follow_planner.py` -- ask and I'll do it.

## Running on the microscope

`Run_SelfDriving_Scan.ipynb` is the entry point -- notebook, not a script, so you
can run cells one at a time and re-run just the config/run cells to retune without
restarting the kernel:

1. **Backend** -- flip `USE_HARDWARE` to connect via the same BEPyAE/ActiveX bridge
   used elsewhere on this instrument, or leave it `False` to rehearse against
   `SimulateBackend` first. First time on a given computer, copy
   `hardware_paths.example.json` to `hardware_paths.json` and edit `exe_path`/
   `vi_path` for that machine -- this notebook is meant to move between
   microscopes unmodified, so the install path lives in that (gitignored) file,
   not in the notebook itself.
2. **Configuration** -- pick `MODALITY` (`"edge_follow"` or `"gp_path"`) and set your
   tuned values from simulation. `cfg.execution.start_xy_V` stays `(0.0, 0.0)`.
   `OUTPUT_DIR` (default `scan_data/`, gitignored) is created here if it doesn't
   exist yet -- everything from step 5 lands there. `cfg.position_sensors.enabled`
   (default `False`) is also here, if you want open-loop X/Y correction (see
   "Position sensors" above).
3. **Build** -- constructs the backend, a `ContinuousExecutor`, the planner, and the
   matching live view, and displays the live figure.
4. **Run** -- blocks until the scan finishes, updating the figure as it goes.
5. **Save** -- writes one `.npz` + one `.png` into `OUTPUT_DIR`. The `.npz` has both
   **the trajectory scanned** (`x_cmd`/`y_cmd`/`z`, plus `x_sensor`/`y_sensor` if
   position sensors were on) and **the reconstructed data itself** (`recon_Z` for
   edge-follow, `recon_mean`/`recon_sigma` for GP path) -- not just a picture of
   it, so you can reload and retune it later without re-scanning. That's exactly
   what `Analyze_SelfDriving_Scan.ipynb` (next to this notebook) does: point it at
   a saved `.npz` and re-run the reconstruction with different parameters, no
   hardware involved.

Suggested first real-hardware test: small `max_search_radius_V` (well inside a
known safe area), `max_features=1`, on a sample region you already know the
topography of, so you can sanity-check the traced boundary against something you
trust before trusting the planner on an unknown region.

## Stepwise grid_refine with no tip jumps (`Run_GridRefine_Stepwise.ipynb`, 2026-11-03)

Runs with abrupt X/Y piezo jumps (PyScanner's fast straight spiral return, the fast raster's
line steps, undriven starts) showed 1.5-3 um of drift between scans; with every move driven,
the sample did not drift at all. So:

- **No-jump guard** (`PyScannerBackend.guard`, on by default): before every output the AO
  readback (`get_AO0_V/AO1_V`, which lags -- polled up to 1.5 s) is compared with the output's
  start; if it is more than `guard_tol_V` (5 mV) away, a cosine move at `guard_speed_V_s`
  (10 V/s) drives there first. After every output `last_out` is set to the readback, and the
  executor's next move starts from there (PyScanner stops spirals short of where
  `AO_vs_time` says). `guard_log` records every check. `asylum.zero_ao` updates `last_out`;
  `ensure_routing` refuses to hand the loops back if Igor reset the routing with AO not at 0.
- **Spirals only**: `pyscanner_scans.SPIRAL_RETURN = 1` (out and smoothly back in);
  `MIN_SCAN_SAMPLES` keeps every scan above PyScanner's minimum (below it PyScanner pads with
  zeros: a jump to (0, 0)); `within()` checks each scan against the per-axis limits
  (`safety.max_amplitude_y_V`). grid_refine: `pyscanner_one_scan` (one do_scan per region),
  `region_max_V`, `merge_diagonal`, `sparse_spiral_radius_V`, `drive_margin_V`.
- **RAM**: `readback.store_every` keeps block means in the history; the reference raster is
  split into chunks sized to the free RAM.
- **`stepwise.StepwiseRun`**: level 0 (one sparse spiral), level 1 (one spiral per object on
  the sparse arcs), level 2 (one dense spiral per object, placed from level 1's *measured*
  path: level 1's real paths came out 0.78-0.88 x the commanded size and offset). Then the
  dataset, figures, animation, and -- only when you decide -- the equivalent raster
  (`user_raster`: same number of pixels, smooth bands) and the side-by-side comparison.
- **Reconstruction**: `"tv"` (total variation, `tv_mu` 0.02, 1500 iterations) had the lowest
  shape and edge errors on that dataset, scored against the raster's independent pixels
  (linear second, DCT-CS last); it is saved as `recon_tv`. The live view default is unchanged.

## Working on this going forward

The package is a git repository (GitHub). Development notes of the GP day are in
`docs/notes/SUMMARY_2026-11-09.md`. Still to confirm on the hardware: every
`# CONFIRM ON HARDWARE` item in `backends/pyscanner_backend.py` / `config.py` (raw readback
control name, `N_image_pixels`); the GP planner's refit cost grows with the data (cap the
training points / GPU).

## Z signal on a Cypher BNC (`cfg.z_signal`)

PyScanner has no free inputs, so AI0 (`cfg.readback.ai_channel`) stays cabled to one
Cypher BNC (`BNCOut2`) and the setup chooses what's on it: with `cfg.z_signal.route`,
step 3d copies `cfg.z_signal.source` (default `"ZSensor"`, the Z LVDT; or the Z piezo
signal's Cypher name -- CONFIRM ON HARDWARE) through Cypher PIDS loop 5 to OutFastA
(another free DAC if it's taken: never OutA/OutB, which Igor's own LVDT.X/Y loops 0/1
drive, nor the X/Y sensors' OutC/OutD) -> BNCOut2,
locked -- the same straight copy as the X/Y sensors (P = -1 / `output_divider`, the
DAC preloaded first). Then, tip withdrawn and at rest, it records AI0 and compares
it with the DAC: "the readback is ZSensor", or a warning if AI0 reads something else
(cable not on BNCOut2). Before every run the copy is checked and redone if Igor reset
it. The run file gets `z_signal_json` (source, DAC, BNC, divider, the AI check) and,
for ZSensor, `z_nm_per_V` (AI volts -> nm, from Igor's `ZLVDTSens` x the divider).
The planners need no change: the automatic threshold adapts to the signal's scale and
sign (a manual threshold has to be in the new AI volts). Loop states: Igor reports
StartEvent as "Always;" / "Never;" -- the trailing ';' is ignored.

**Z in nm (`cfg.z_signal.units = "nm"`).** Every measured Z is multiplied, as it
comes in (`ContinuousExecutor`), by the setup's nm per AI volt (`m_per_V` x
`output_divider`; for ZSensor `m_per_V` is Igor's `GV("ZLVDTSens")`, or set it by
hand in m/V). Everything downstream -- planners, the substrate fit and threshold, the
live view ("Z (nm)"), the saved `z` / `recon_Z` / `binned_Z`, the reference raster --
is then in nm, and so are the Z-valued settings: `edge_detect.manual_threshold`,
`grid_refine.detect_threshold`, `gp_path.target_mean_uncertainty`. The automatic
thresholds and the reconstructions are scale-free, so a run behaves the same in V or
nm (simulated: identical features, thresholds x the factor). Without the factor (Z
not routed this session, no sensitivity, or AI0 failed the setup check) Z stays in
volts and the run says why. The file records `z_unit` and `z_scale_per_V`; the
Analyze notebook labels its plots from them.

**Sign (`cfg.z_signal.invert`, default True).** ZSensor reads opposite to the height on
this Cypher (2026-10-13: features came out as dips), so with the Z routing on, Z is
multiplied by -1 as it comes in (in V or nm; the copy loop and the setup check are
unchanged). Edge detection uses |height above the substrate|, so it isn't affected.
`z_scale_per_V` in the file includes the sign.

## Priors in nm, and the run notebook's order (2026-10-14)

With `cfg.position_sensors.enabled` the run notebook asks the priors in **nm** along the
position sensors (`priors.ask_for_run` -> `ask_nm`): sizes and the area's half-width
divided by the calibration's nm per drive volt (the mean of the two axes'), the centre
through the calibration's own map (the sensors' coordinates -- the plots' "um, sensor"
axes). Everything after that is in drive volts as before; what was typed is saved with
the run (`priors_json["entered"]`). Sensors off (or not calibrated): in volts. For that,
the notebook now runs **3. backend -> 3a. setup (calibrates the sensors) -> 3b. priors
-> 3c. build -> 3d. probe -> 4. run**; the setup's calibration and Z routing are also
left on the backend, so every executor on it (the run's, the reference raster's) has them.

**Reference raster with the sensors.** It records the X/Y sensors with Z (same AI record:
sample i of Z was measured at sample i of the sensors -- no time shift to fit) and is
saved a second time at the measured positions (`raster_binned_Z_sensor`,
`raster_recon_Z_sensor`, `raster_xs/ys/zs`): along each line the sensors' fast-axis
position, each line at its measured (median) slow-axis position, so lines stay pixel
rows for the column-wise reconstruction and the animation. The raster is sent without
the drive correction (as planned), so the commanded version is unchanged.
`reference_raster.load(d, positions="auto")` returns the sensor version when present;
the Analyze notebook (`RASTER_POSITIONS`) and its animation use it, with um axes when
section 4a ran.

**Exploration settings in the run notebook's config cell**: `explore_lookahead_V = 4.0`,
`min_score = 0.3`, `straight_bonus = 0.3` (package defaults: None = 4 x segment length,
0.02, 0.1) -- longer-range, less gap-filling exploration.

**Open vs closed loop animations (Analyze notebook 5b / 5c, `compare_animation.py`).**
5b: up to 4 rows advancing together -- self-driving at the commanded positions (open
loop, V), self-driving at the measured positions (sensors, um axes), reference raster
commanded, reference raster measured -- each path | binned | reconstruction | uncertainty,
like the live view. 5c: the two measured rows only, axes in nm. Frames follow the run's
waveforms; the raster rows show the raster as far as the run's pixel budget so far takes
it (counted at the commanded positions, both raster rows at the same point of its scan).
Missing rows (no sensors / no raster) are left out. Saved next to the run file as
`*_open_vs_closed.gif` and `*_closed_loop_nm.gif`.

## PFM: following domain walls (`Run_PFM_DomainWalls.ipynb`, `pfm.py`)

A separate notebook (the topography notebook is unchanged), `MODALITY =
"domain_wall_explorer"`: the edge explorer with a different classifier
(`pfm.DomainModel`, plugged in through `EdgeExplorerPlanner._make_model`). Asylum in PFM
mode (the drive on while engaged). The setup (step 3e) copies the PFM signals to BNCs with
the same straight-copy PIDS loops as the X/Y sensors -- amplitude: Cypher loop 5 ->
OutFastA -> BNCOut2 -> AI0; phase: ARC loop 2 -> ARC OutA -> ARC BNCOut0 -> AI3 (the
Cypher's six loops are taken; the ARC's 2-5 were free, 2026-10-16 -- ARC DAC/BNC names
CONFIRM ON HARDWARE) -- checks each AI reads its copy, and re-checks before every run.
Either signal can be off (`cfg.pfm.amplitude/phase.enabled`).
- phase (+ amplitude): two phase populations (domains) found on the fly with circular
  statistics, tracking drift; a sample's score = which domain's phase it's nearer x its
  amplitude / the domains' (so the noisy phase AT the wall counts for nothing); with
  amplitude, a phase flip is a wall only with an amplitude dip within `confirm_window_V`.
- amplitude only: the wall is "on" where the amplitude drops below `amp_dip_fraction` x
  the domains' -- followed round as a thin strip (both sides).
Closed domains' insides are NOT taken as explored (domains nest; one closed by the border
is ambiguous). Both signals are saved (`ch_amplitude`, `ch_phase`, `pfm_json`); "z" (live
view, reconstruction, reference raster) is the phase in degrees, or the amplitude.
Simulated (`pfm.PFMSampleBackend`, walls 0.03 V wide, 400 segments): bubbles -- every
domain closed, 100% of the walls within 0.1 V of the path (all three signal options);
stripes 85% (both) / 82% (phase) / 54% (amplitude only); labyrinth 83% / 83% / 95%.
Known limit: binned images average the phase linearly -- a domain whose phase sits at
+-180 deg would read mixed in the images (the wall detection is wrap-safe).

## PFM: what changed after the 2026-11-04 runs (`Run_PFM_DomainWalls_v2.ipynb`)

The good PFM dataset (3 x 3 um ferroelectric area, the explorer + its smooth equivalent raster) is in
`scan_data/good PFM domain-wall dataset 20261104/` (README there).

- **Copy loops stopped by Igor**: on a withdraw Igor stops its PIDS copy loops (X/Y sensors, Z, PFM
  amplitude / phase: `Status` "Stopped") but leaves StartEvent "Always" -- the outputs then stay frozen
  at their last value. The routing checks now require the loop to be running (`asylum._active`, in
  the sensor / Z checks and `pfm.routing_ok`).
- **Sensor calibration inside the scan area**: `cfg.position_sensors.calibration_area_half_nm` (the
  area's half-width, set BEFORE the setup -- `SCAN_AREA_NM` in the v2 notebook) keeps the 3 x 3
  calibration grid inside it (x `calibration_area_fraction`, nm -> V with `calibration_um_per_V_guess`).
- **Explorer steps scale with the area** (`priors.apply`): exploring step and longest zigzag step at
  most 0.2 x the area's half-width, the calibration circle 0.15 x, the first path 0.5 x.
- **Stop once the walls are explored**: `cfg.edge_explorer.stop_after_quiet_sweeps` (None = off; 2 in
  the v2 notebook) -- no new edge over that many "sweeps of the area" of exploring path. The run prints
  why it stopped (`planner.stop_reason`). Run 2 stopped by itself after 327 s (run 1: 600 s cap).
- **Smooth equivalent raster with the PFM signals**: `user_raster.acquire` records every extra channel
  (`raster_ch_<name>_*`, as `reference_raster`); the v2 notebook's cell 6 uses it (never PyScanner's
  fast raster) and checks every copy loop is running after engaging.
- **Figures**: `pfm_figures.make(npz)` -- run vs raster figure (phase rotated so the two domains sit at
  -90 / +90 deg before binning) and two GIFs.
- **TV reconstruction is scale-free** (the measured 1..99% range scaled to 100 inside `tv_inpaint`).
- Amplitude depends on the tip's direction of motion on this sample (friction; raster trace / retrace
  3.55 / 2.76 mV) -- keep it in mind when using the amplitude level, e.g. for in-plane walls (not
  solved yet: the experimental in-plane-wall classifier is archived with the dataset, `code/`).

## Combining levels: hybrid modes (2026-11-09/10, `planners/gp_edge_explorer_planner.py`, `planners/hybrid_planners.py`)

Three modes in which one SCOUT level explores between objects and another handles each object
(run notebook: `Run_GP_Paper.ipynb`; figure: `analysis/combo_fig.py`):

| mode | explores with | on each object | instrument result (shapes sample, 26 um, 5 objects) |
|---|---|---|---|
| `gp_edge_explorer` | GP (IVR, lambda 0), 1 V steps | edge explorer's tracer (Fig. 3 follow steps: priors 1-4 um) | 5/5 outlines, 31 % of pixels on objects (raster 8 %), 500 s |
| `grid_tracker` | grid_refine level 0: one sparse Archimedean spiral (turns 1.07 V apart) | the same tracer | 5/5 outlines, 33 % on objects, 270 s; all objects found first |
| `gp_spiral` | GP, 0.5 V steps, holes when quiet | edge probe -> level-1 coarse spiral -> level-2 dense spiral | 5/5 objects mapped (4 outlines complete), 34 % on objects, 410 s |

Build them like any mode: `modes.build("gp_spiral", cfg, backend, priors=PRIORS)` -- the priors
set the spirals' sizes (`GPSpiralPlanner.FEATURE_MIN_V/MAX_V`). Settings that the instrument
needed (all defaults now): a fixed edge threshold (58 nm, `edge_detect.mode = "manual"`), the
drive-offset correction un-saturated (`position_sensors.correction_max_V = 2.5`, start from the
last measured offset `initial_offset_V`), and for `gp_spiral` a log file
(`GPSpiralPlanner.LOG_PATH`) because the live view replaces the cell output.

What the instrument showed that simulation did not (each fixed in `GPSpiralPlanner`):
- **Piezo bow** (~300 nm over the area): a global plane reads whole regions as objects ->
  robust degree-2 background, objects trimmed out, refitted before every exploring path, and a
  plane through each spiral's own substrate.
- **Crossings read low** (59-79 nm for 116 nm objects: tip convolution) -> trigger at 1.1 x the
  threshold plus a height STEP along the path and on the probe (the bow is smooth).
- **Path length from the sensors** is ~100x too long (sample-to-sample noise) -> sample index.
- **Position-dependent drive error**: a 0.35 V probe circle landed 0.3 V off -> the probe is
  re-centred on where it measured and repeated once; the probe is 0.5 V.
- **Neighbouring objects** 1 V apart merged in level 1 -> mean-shift window one feature wide.
- Open: the plane inside a level-2 disc can tilt (2 of 5 outlines partial in v6).

`analysis/sim_hybrids.py MODE` simulates each hybrid with the instrument's settings;
`NOISY=1 BOW=3` adds 8 nm noise, tilt, a background bump and a piezo bow.

Other changes of 2026-11-09 (GP, Fig. 4): equivalent rasters cut only at line ends + dwell
(PyScanner's AI record is short at the end of long waveforms), TV maps for GP modes (the GP mean
is a poor map of steps), robust levelling on binned pixels, `gp_path.ivr_interest_power` (gamma).
Details: `SUMMARY_2026-11-09.md`.
