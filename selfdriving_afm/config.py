"""
Central configuration for the self-driving AFM scanning package.

Everything lives in "bias units" (Volts) as sent/received from PyScanner, since the
X/Y piezo drive is open loop. `CalibrationConfig` is a passthrough (scale=1.0) for now
so that plugging in a real nm/V calibration factor later is a one-line change, not a
refactor.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple


# --------------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------------
@dataclass
class SafetyConfig:
    """Hard limits enforced on every generated waveform before it is sent to hardware."""
    max_amplitude_V: float = 10.0   # symmetric +/- limit for X and Y outputs
    max_amplitude_y_V: Optional[float] = None   # a tighter +/- limit for Y only (None = max_amplitude_V)
    max_bias_V: float = 10.0        # limit for tip-bias output, if used
    min_points_per_segment: int = 4
    clamp_mode: Literal["raise", "clip"] = "raise"  # "raise" is the safer default
    max_samples_per_waveform: int = 4_000_000  # longer waveforms are sent to PyScanner in
        # consecutive pieces of at most this many samples (execution.ContinuousExecutor.send).
        # 3.8 M-sample waveforms worked on hardware; an 8.5 M one left PyScanner's X and Y
        # arrays with different lengths, so stay well under that


def amplitude_limits(safety: "SafetyConfig"):
    """(X limit, Y limit), V."""
    ly = safety.max_amplitude_y_V
    return float(safety.max_amplitude_V), float(safety.max_amplitude_V if ly is None else min(ly, safety.max_amplitude_V))


# --------------------------------------------------------------------------------
# Calibration (currently a passthrough - open loop bias world)
# --------------------------------------------------------------------------------
@dataclass
class CalibrationConfig:
    scale_x: float = 1.0   # "physical units" per Volt. 1.0 => physical units ARE volts
    scale_y: float = 1.0
    scale_z: float = 1.0
    units: str = "V"       # display label only; change once closed-loop calibration exists

    def to_volts_xy(self, x_phys: float, y_phys: float):
        return x_phys / self.scale_x, y_phys / self.scale_y

    def to_phys_xy(self, x_v: float, y_v: float):
        return x_v * self.scale_x, y_v * self.scale_y

    def to_phys_z(self, z_v):
        return z_v * self.scale_z


# --------------------------------------------------------------------------------
# Tip velocity -- the single source of truth for how fast the tip moves through
# bias-volt XY space, shared by every trajectory primitive (spiral, circle, line,
# and the auto-inserted connectors in execution.py). Point density is DERIVED from
# this, not set directly, so "how fast does the tip move" and "how finely is it
# sampled" can't drift out of sync with each other.
# --------------------------------------------------------------------------------
@dataclass
class VelocityConfig:
    """Both fields are placeholders pending real instrument numbers -- CONFIRM ON
    HARDWARE (see backends/pyscanner_backend.py). Point count for any trajectory
    scales as arc_length * sample_rate_hz / tip_velocity_V_s, so a large search
    radius combined with a slow tip and a fast DAQ rate can imply tens of millions
    of samples for a single spiral -- trajectory.py raises a clear error rather than
    building that silently; if you hit it, that's a real configuration tradeoff to
    make (faster tip, coarser rate, or a smaller search radius), not a bug.
    """
    tip_velocity_V_s: float = 0.05   # target constant tip speed, bias volts/second
    sample_rate_hz: float = 5.0e4    # DAQ output rate assumed for every waveform

    @property
    def sample_spacing_V(self) -> float:
        """Distance between consecutive samples that gives tip_velocity_V_s at
        sample_rate_hz. Every trajectory primitive is built at this spacing."""
        return self.tip_velocity_V_s / self.sample_rate_hz


# --------------------------------------------------------------------------------
# Trajectory primitives
# --------------------------------------------------------------------------------
@dataclass
class SpiralConfig:
    """Archimedean search spiral, growing outward from a center point. Sample
    spacing (and therefore point count) comes from VelocityConfig, not from here --
    n_cycles only controls the pitch (how tightly wound it is)."""
    n_cycles: float = 8.0
    direction: Literal["cw", "ccw"] = "ccw"


@dataclass
class CircleConfig:
    """Circular scan used to trace a detected edge/perimeter. Sample spacing (and
    therefore point count) comes from VelocityConfig, not from here."""
    radius_mode: Literal["fixed", "adaptive", "curvature"] = "fixed"
        # "fixed":     always fixed_radius_V
        # "adaptive":  shrinks after sharp turns of the trace (adaptive_curvature_gain)
        # "curvature": every circle MEASURES the edge's local radius of curvature R from
        #              the angle its on-feature arc spans (R = r / (2 cos(span / 2)); 180 deg
        #              = straight edge) and the next circle gets r = curvature_radius_fraction
        #              x R, between adaptive_min/max_radius_V (set from the feature-size
        #              priors -- see priors.py)
    fixed_radius_V: float = 0.05
    adaptive_min_radius_V: float = 0.02
    adaptive_max_radius_V: float = 0.15
    adaptive_curvature_gain: float = 4.0   # higher => shrinks radius faster near corners
    curvature_radius_fraction: float = 1.0  # "curvature": r = this x the local radius of curvature
    curvature_max_growth: float = 1.5       # "curvature": the radius grows at most this much per circle
    overlap_fraction: float = 0.5          # user-set; 0 = touching circles, ~1 = little progress


# --------------------------------------------------------------------------------
# Edge detection
# --------------------------------------------------------------------------------
@dataclass
class EdgeDetectConfig:
    mode: Literal["manual", "auto_mad"] = "auto_mad"
    manual_threshold: Optional[float] = None   # required if mode == "manual" (height above the substrate)
    mad_k: float = 6.0                          # auto_mad: threshold at least k x the substrate roughness
    min_run_V: float = 0.002  # an on/off-feature stretch shorter than this (path length, V) is
                              # treated as noise, not an edge -- stops single noisy samples
                              # registering as edges
    # --- substrate (tilt) correction, on the fly (substrate.SubstrateModel; used by the
    # edge_follow and edge_explorer planners) ---
    tilt_correction: bool = True     # decide on/off-feature from the height above a fitted
                                     #   substrate (plane), refined after every waveform from the
                                     #   off-feature samples; False = one constant level (old behaviour)
    substrate_degree: int = 1        # 1 = plane (tilt), 2 = also bow (x^2, xy, y^2)
    substrate_sample_spacing_V: float = 0.005  # the fit uses one sample per this much path
    use_height_gap: bool = True      # auto_mad: once features have been seen (a second population of
                                     #   heights), move the threshold into the gap between substrate
                                     #   and features -- never below mad_k x roughness
    hysteresis_fraction: float = 0.25  # edge_explorer: two thresholds, ON above threshold x (1 + this),
                                     #   OFF below threshold x (1 - this) -- no flickering edges where the
                                     #   path runs along a sidewall; 0 = one threshold


# --------------------------------------------------------------------------------
# Edge-follow (spiral -> circular chain) planner
# --------------------------------------------------------------------------------
@dataclass
class EdgeFollowConfig:
    spiral: SpiralConfig = field(default_factory=SpiralConfig)
    circle: CircleConfig = field(default_factory=CircleConfig)
    edge_detect: EdgeDetectConfig = field(default_factory=EdgeDetectConfig)

    initial_search_radius_V: float = 0.05  # search spiral radius
    calibration_radius_V: Optional[float] = 0.5  # first move: a circle of this radius round the
                                          # start, to measure the substrate tilt + roughness before any
                                          # decision (substrate.SubstrateModel); None = no circle
    first_search_radius_V: Optional[float] = None  # the FIRST spiral's radius (None = initial_search_
                                          # radius_V); the priors set it to cover the whole area
    first_scan_pattern: Literal["spiral", "raster", "lissajous"] = "spiral"
        # the FIRST search scan: "spiral" (radius first_search_radius_V -- the circle inside
        # the area), or "raster" / "lissajous" over the whole SQUARE area (explore_extent_V
        # half-width, corners included), lines spiral_pitch_V apart. Later searches
        # ("relocate") are always spirals
    spiral_pitch_V: Optional[float] = None  # distance between spiral turns; None = radius /
                                          # spiral.n_cycles. The priors set it to half the smallest
                                          # feature, so no feature fits between two turns
    # --- what a spiral that finds nothing new does next ---
    spiral_search_mode: Literal["relocate", "grow"] = "relocate"
        # "relocate": same-size spiral, moved to the nearest area not scanned yet (also
        #             after tracing: the next search goes to the nearest unexplored area)
        # "grow":     bigger spiral around the same center (the older behaviour)
    explore_extent_V: float = 1.0         # "relocate": explore a square of this half-width
                                          # around explore_center_V (kept inside the scan range)
    explore_center_V: Optional[Tuple[float, float]] = None  # None = start_xy; the first
                                          # spiral is centered here
    explore_min_unexplored_fraction: float = 0.5  # "relocate": a new spiral spot must be at least
                                          # this much not-yet-scanned; none left -> run ends
    explore_grid_n: int = 200             # resolution of the explored-area map
    spiral_growth_factor: float = 1.6     # "grow": how much bigger the search spiral gets if no edge found
    max_search_radius_V: float = 1.0      # "grow": stop growing search spirals beyond this

    max_circles_per_feature: int = 150    # per direction along the edge
    loop_closure_distance_V: float = 0.02
    loop_closure_min_steps: int = 6       # don't allow "closure" on the first few steps

    # --- one edge = one feature (see edge_follow_planner module docstring) ---
    on_new_edge: Literal["switch", "queue"] = "switch"  # a circle sees a SEPARATE edge:
        # "switch" = trace the new edge right away, then come back and finish this one;
        # "queue"  = finish this edge first, then trace the new one
    follow_all_spiral_edges: bool = True  # a search spiral crosses SEVERAL separate edges:
        # True  = trace every one of them (next = the one nearest the tip) before
        #         searching again;
        # False = trace only the first one (the older behaviour -- the others are only
        #         found again if a later spiral or circle happens to cross them)
    trace_both_directions: bool = True    # open edge lost in one direction -> go back to where
                                          # it was first found and follow it the other way too,
                                          # as the SAME feature
    same_feature_distance_V: Optional[float] = None  # an edge point closer than this to an
        # already-traced edge LINE is the same edge (same feature), not a new one.
        # None = a quarter of the current circle radius

    # --- stopping criteria -- see README "When does a scan stop?" ---
    max_features: int = 25                # stop once this many features have been traced
    max_run_time_s: Optional[float] = None  # wall-clock safety net; None = no limit


# --------------------------------------------------------------------------------
# Edge explorer (edge-probability map + exploit/explore path) -- see
# planners/edge_explorer_planner.py
# --------------------------------------------------------------------------------
@dataclass
class EdgeExplorerConfig:
    calibration_radius_V: Optional[float] = 0.5  # first move: a circle of this radius round the
                                          # start, to measure the substrate tilt + roughness before any
                                          # decision (substrate.SubstrateModel); None = no circle
    initial_path_length_V: float = 0.3    # the first straight path, before any decision
    initial_direction_deg: float = 0.0    # its direction (0 = +X, 90 = +Y)
    segment_length_V: float = 0.1         # EXPLORING: every straight segment -- the step size
    # --- FOLLOWING an edge: a zigzag across it, until it closes (see the planner) ---
    follow_segment_length_V: Optional[float] = None  # zigzag segment length to start with on a
                                          # new edge (None = 0.5 x segment_length_V)
    follow_segment_min_V: Optional[float] = None  # it shrinks where the edge turns, down to this
                                          # (None = follow_segment_length_V)
    follow_segment_max_V: Optional[float] = None  # and grows where it runs straight, up to this
                                          # (None = follow_segment_length_V). The priors set
                                          # both from the feature sizes
    follow_style: str = "segments"        # "segments": the zigzag of straight segments across the edge;
                                          # "weave": a smooth sine weaving back and forth across it, its
                                          # axis bent to the edge's estimated curvature -- no corners
                                          # (easier on the piezo), one waveform per sine cycle
    follow_weave_amplitude_V: Optional[float] = None  # weave: how far either side of the edge
                                          # (None = 0.5 x shortest step)
    follow_weave_wavelength_V: Optional[float] = None  # weave: distance ALONG the edge per sine cycle
                                          # -- short, so the cycles are close together and the edge is
                                          # re-measured every ~wavelength / 2 (None = 0.5 x shortest step)
    follow_weave_cycles: float = 3.0      # weave: sine cycles per waveform; after each waveform the
                                          # axis is re-fitted to its crossings (centred, re-aimed)
    follow_weave_corner_deg: float = 60.0  # weave: crossings stopped part-way (a corner): restart there
                                          # turned this much toward the feature
    follow_recovery_tries: int = 3        # a segment missed the edge (a corner): turn and retry
                                          # this many times before the edge counts as lost
    follow_recovery_turn_deg: float = 60.0  # ...turning the edge direction this much each time
    follow_reacquire_V: Optional[float] = None  # starting / restarting to follow an edge: a probe line
                                          # this far either side of where it was found, to find it again
                                          # (the open-loop piezo puts it a few tenths of a V elsewhere
                                          # after a long move); None = max(2 x shortest step, 0.5 V)
    follow_reacquire_tries: int = 2       # ...doubled each time it crosses nothing, this many probes
    close_at_border: bool = True          # a feature cut by the area's border: both directions of its
                                          # trace end at the border -> the border closes its outline (and
                                          # its inside is scanned)
    border_close_distance_V: Optional[float] = None  # "at the border" = closer than this (None =
                                          # max(shortest step, 0.6 x longest step))
    merge_features: bool = True           # a trace that runs into another open edge: it's the same
                                          # edge -- merge them (and close the loop if the two meet)
    max_segments_per_feature: int = 300   # per direction along an edge (safety cap)
    # --- once an edge CLOSES: scan a little of its inside (one waveform of parallel lines) ---
    scan_interior: bool = True            # measures the top (better reconstruction) and finds
                                          # edges inside (holes), which are then followed too
    interior_line_spacing_V: Optional[float] = None  # lines this far apart (None = 1.4 x the
                                          # first zigzag step); the priors: 0.7 x smallest feature
    interior_margin_V: Optional[float] = None  # lines stop this far inside the traced outline
                                          # (None = half the shortest zigzag step)
    interior_max_lines: int = 40          # at most this many lines per feature (spread wider)
    exploit_weight: float = 0.8           # UNUSED since following became its own phase (kept so
                                          # older configurations still run)
    n_candidate_directions: int = 32      # EXPLORING: directions scored at every step
    max_turn_deg: Optional[float] = None  # EXPLORING: limit the turn between segments (None = any)
    edge_detect: EdgeDetectConfig = field(default_factory=EdgeDetectConfig)
    edge_along_V: Optional[float] = None  # UNUSED (the old edge-probability map)
    edge_across_V: Optional[float] = None  # UNUSED (the old edge-probability map)
    link_distance_V: Optional[float] = None  # UNUSED (the old edge-probability map)
    loop_closure_min_coverage_deg: float = 300.0  # an edge is CLOSED once its crossings go at
                                          # least this far round their centre (open edge ~180)
    loop_closure_min_points: int = 6      # ...and it has at least this many crossings
    fast_closure: bool = True             # close as soon as the two ends of the traced outline meet: the
                                          # segment just measured passes within loop_closure_tolerance_V of
                                          # the outline's other end -- checked after every segment; and the
                                          # last step aims straight at it. False = only the stricter rules
                                          # (back at one point, 300 deg round the centre)
    loop_closure_min_enclosed: float = 0.6  # ...and the traced outline, its ends joined, must enclose the
                                          # feature: (crossings with the point just inside the edge inside it)
                                          # - (with the point just outside inside it) >= this. The guard
                                          # against closing early on a long, thin or U-shaped feature
    loop_closure_tolerance_V: float = 0.3  # back at the start = within max(this, 0.5 x the shortest step):
                                          # the open-loop piezo moves edges 0.1-0.3 V between visits
    explore_scale_V: Optional[float] = None  # this far from anything measured counts as fully
                                          # unexplored (None = segment_length_V)
    explore_lookahead_V: Optional[float] = None  # exploring scores this far along each direction
                                          # (None = 4 x segment_length_V)
    straight_bonus: float = 0.1           # exploring: small preference for going straight on
    explore_extent_V: float = 1.0         # explore a square of this half-width around explore_center_V
    explore_center_V: Optional[Tuple[float, float]] = None  # None = start_xy; the initial
                                          # straight path starts here
    grid_n: int = 200                     # resolution of the explored-area map
    min_score: float = 0.02               # stop once no direction scores above this (explored)

    # --- stopping criteria ---
    # (all three are checked BETWEEN features -- an edge being followed is always finished)
    max_segments: int = 300
    max_features: int = 25                # stop once this many edges have been CLOSED
    # stop once exploring has found NO NEW edge for this many "sweeps of the area": exploring
    # path since the last new edge >= this x (area / explore_scale_V), the path one sweep of the
    # whole area at explore_scale_V spacing takes. None = off. 2026-11-04 PFM run: the last new
    # wall came at 40% of the run, the gaps between new walls were <= 1.7 sweeps
    stop_after_quiet_sweeps: Optional[float] = None
    max_run_time_s: Optional[float] = None


# --------------------------------------------------------------------------------
# Grid + refine (sparse scan -> fast inpainting -> detect -> refine, level by level)
# -- see planners/grid_refine_planner.py
# --------------------------------------------------------------------------------
@dataclass
class GridRefineConfig:
    # --- the area ---
    center_V: Tuple[float, float] = (0.0, 0.0)
    half_size_V: float = 1.0              # the sparse scan covers a square of this half-width
                                          # (shrunk if needed to stay inside the safety limit)
    # --- 1. sparse scan over the whole area ---
    sparse_pattern: Literal["raster", "lissajous", "spiral"] = "raster"
    sparse_n_lines: int = 12              # sparse density: lines across the area -- spacing =
                                          # area width / (n - 1). Lissajous/spiral: same spacing
    sparse_spiral_radius_V: Optional[float] = None  # spiral sparse scan: its radius (None = the circle
                                          # inside the area; larger reaches into its corners -- kept in range)
    sparse_crosshatch: bool = False       # raster: a second pass with lines the other way (a grid);
                                          # False = ONE scan over the whole area
    # --- who generates the scan paths ---
    scan_generator: Literal["user_defined", "pyscanner_defined"] = "user_defined"
        # "user_defined": the package builds every waveform (scan_type 0) -- constant tip
        #     speed everywhere, tiles of a level joined into one waveform
        # "pyscanner_defined": PyScanner's own scan types (1 spiral, 2 fast raster,
        #     3 Lissajous), placed with scan_x/y_offset_V -- one do_scan per tile; the tip
        #     speed follows PyScanner's sinusoidal profiles (see pyscanner_speed_basis)
    pyscanner_speed_basis: Literal["peak", "mean"] = "peak"
        # "pyscanner_defined" durations: "peak" = the tip never goes faster than
        # tip_velocity_V_s; "mean" = it averages tip_velocity_V_s (peaks ~1.6x higher)
    pyscanner_raster_type: int = 2        # PyScanner raster_type_04 (2 = sinusoidal)
    pyscanner_one_scan: bool = False      # every region (and the sparse scan) ONE PyScanner scan: one that
                                          # would exceed max_samples_per_waveform runs faster (up to
                                          # pyscanner_max_speed_V_s), then with fewer lines / turns -- never split
    pyscanner_max_speed_V_s: float = 50.0
    # --- 4. refinement scans ---
    refine_pattern: Literal["auto", "raster", "lissajous", "spiral"] = "auto"
        # flagged tiles that touch (also diagonally) form ONE region, scanned as one scan.
        # "auto" picks each region's pattern from its shape: round -> spiral (when its
        # circle is a shorter path than a raster of the region), rectangular -> auto_rect_
        # pattern, irregular -> raster with each line clipped to the region. Or force one.
    auto_rect_pattern: Literal["raster", "lissajous"] = "raster"  # "auto": rectangular regions
    rect_fill_min: float = 0.85           # "auto": a region filling at least this fraction of its
                                          # bounding box is rectangular
    spiral_coverage: float = 0.98         # a region's spiral covers this fraction of its feature's
                                          # pixels (+ the edge's position uncertainty)
    spiral_min_roundness: float = 0.75    # "auto": spiral only for features at least this round
                                          # (area / enclosing circle's: disk ~0.9-1, square 0.64)
    fill_interiors: bool = True           # flagged tiles enclosing unflagged ones (a feature's
                                          # outline round its flat top): scan the inside too --
                                          # measures the top, finds internal edges
    fill_interior_levels: int = 1         # ... only on the first N refinement levels (finer levels
                                          # refine the edges only)
    fill_max_size_V: Optional[float] = None  # ... only insides at most this wide (the priors: the
                                          # largest feature -- a bigger enclosed area is substrate
                                          # between features); None = any size
    tile_max_V: Optional[float] = None    # tiles at most this wide (the priors: half the smallest
                                          # feature); touching tiles merge, so small tiles cost
                                          # nothing extra and fit the features tighter
    refine_factor: float = 3.0            # every refinement level has lines this much closer
                                          # (3 was ~25% faster than 2 in simulation, same result)
    min_spacing_V: Optional[float] = None  # finest line spacing = the final resolution; the
                                          # last level lands exactly on it (None = area width
                                          # / 255, i.e. like a 256-line raster)
    max_levels: int = 6                   # refinement levels after the sparse scan
    tile_size_lines: float = 2.0          # refinement tiles are this many (current) line
                                          # spacings wide -- smaller = tighter around features
                                          # (at most tile_max_V, at least 2 of the new lines)
    raster_direction: Literal["auto", "x", "y", "alternate"] = "auto"
                                          # auto = lines cross the local edges (steepest direction)
    merge_tiles: bool = True              # touching tiles (sides or corners) = one region, one scan;
                                          # False = every tile scanned on its own
    merge_diagonal: bool = True           # merge_tiles: corner-touching tiles too (False = sides only)
    region_max_V: Optional[float] = None  # a region wider than this (bounding box) is split into its
                                          # separate objects (eroded apart, every tile to the nearest piece)
    batch_tiles: bool = True              # all tiles of a level in ONE waveform (one do_scan)
    raster_turn_radius_V: Optional[float] = None  # raster turnarounds: None = semicircles of half
                                          # a line spacing at full tip speed; larger = gentler
                                          # turns that slow down at their apex
    lissajous_phi_rad: Optional[float] = None  # None = 0 (covers most evenly)
    spiral_direction: Literal["cw", "ccw"] = "ccw"
    # --- 2./3. fast inpainting + detection ---
    detect_on: Literal["measured", "reconstruction"] = "measured"
                                          # where local variation is evaluated before each level:
                                          # "measured" = the measured pixels only (the scan path
                                          # doesn't depend on any reconstruction); "reconstruction"
                                          # = the image inpainted with detect_reconstruction first,
                                          # so structure it predicts BETWEEN scan lines is found too
    detect_reconstruction: Literal["linear", "fast", "compressed_sensing", "gp", "tv", "live_view"] = "linear"
                                          # inpainting used when detect_on = "reconstruction"
                                          # ("live_view" = the same as cfg.live_view.reconstruction;
                                          # CS/GP use cfg.live_view's parameters and are slower)
    detect_pixels: Optional[int] = None   # resolution of the internal maps (None = 2 pixels per
                                          # finest line spacing, 128..512)
    detect_window_lines: float = 1.0      # local-tilt fit window half-width, in line spacings
    detect_k: float = 5.0                 # structure = local height variation (tilt removed)
                                          # above k x the noise level (auto threshold)
    detect_noise_quantile: float = 0.3    # noise level = this quantile of the variation map
                                          # (most of the area is assumed featureless)
    detect_threshold: Optional[float] = None  # fixed threshold instead (same units as Z)
    min_flagged_pixels: int = 2           # a tile needs at least this many flagged pixels
    # --- smooth motion ---
    drive_margin_V: Tuple[float, float] = (0.0, 0.0)  # the area is kept this far (x, y) inside the
                                          # safety limits too -- room for the drive correction's offset
    ease_time_s: float = 0.005            # every scan and move starts/ends at rest, speeding up
                                          # and slowing down over this long -- on a short
                                          # run-up / run-out outside the scan, so the scan
                                          # itself is all at tip_velocity_V_s
    # --- stopping (also stops when nothing is flagged or at min_spacing_V) ---
    max_run_time_s: Optional[float] = None


# --------------------------------------------------------------------------------
# GP / Bayesian path planner
# --------------------------------------------------------------------------------
@dataclass
class GPPathConfig:
    initial_path_length_V: float = 0.05
    path_segment_length_V: float = 0.03
    gradient_eval_resolution_V: float = 0.0015  # internal only: how finely candidate
        # directions are evaluated against the already-fitted GP when scoring where to
        # go next. Decoupled from VelocityConfig on purpose -- this never touches
        # hardware, it's just how many synthetic points gp.predict() is asked about.
    exploration_weight: float = 0.5
    n_candidate_directions: int = 16
    refit_every: int = 1  # re-fit the GP every N segments; >1 trades staleness for speed
    gp_n_restarts_optimizer: int = 0  # 0 = fast (fixed init hyperparams); raise for better kernel fits
    # --- GP back end: the speed knob (see gp_models.py and README "GP back ends") ---
    gp_backend: Literal["sklearn_exact", "frozen", "local", "sparse"] = "sklearn_exact"
        # "sklearn_exact" = the original: re-optimises the kernel on ALL points at every
        #   refit, O(n^3) per optimiser step -- ~20 min per refit at ~6000 points
        # "frozen" = learn the kernel while n < freeze_after_n, then freeze it (a refit is
        #   one Cholesky): seconds at ~6000 points, still grows ~n^3
        # "local"  = frozen kernel + exact GP on the local_n_points nearest the tip:
        #   constant ~0.02 s per refit at any run length (use refit_every = 1)
        # "sparse" = frozen kernel + sparse GP (inducing points where the path has been):
        #   ~0.01-0.05 s per refit, whole-area model at all times (use refit_every = 1)
        # For the instrument: "sparse" (or "local"); both keep each segment's compute well
        # under a second on a plain CPU (no GPU / torch needed).
    freeze_after_n: int = 1500          # frozen/local/sparse: stop re-learning the kernel once
                                        #   this many training points are in (each learn ~0.4 s)
    hyper_opt_subset: int = 600         # frozen/local/sparse: learn the kernel on at most this
                                        #   many (random) training points -- keeps warm-up cheap
    local_n_points: int = 1000          # "local": training points used per refit (cost ~ this^3)
    sparse_inducing_spacing_V: Optional[float] = None  # "sparse": inducing-point grid spacing
                                        #   (None = length_scale_V / 4)
    length_scale_min_frac: float = 0.25  # priors: the GP length scale may not drop below this x
                                        #   the smallest feature (0.5 keeps a step edge from
                                        #   collapsing it to a uselessly short length scale)
    gp_kernel: Literal["rbf", "matern52", "matern32", "matern12"] = "rbf"  # covariance: "rbf" is
                                        #   smooth and rings (over/undershoots) at sharp steps;
                                        #   Matern is rougher -- "matern12" does not overshoot
    gp_detrend: bool = False            # True = GP on the residuals from a robust substrate plane
                                        #   (refit every refit): unvisited ground predicts the
                                        #   tilted substrate instead of a flat mean
    learn_length_scale: bool = True     # False = keep the GP length scale at length_scale_V
                                        #   (from the priors) instead of fitting it -- a fitted
                                        #   RBF collapses to its lower bound on sharp step edges
    # --- acquisition: how a candidate direction is scored (README "GP path acquisition") ---
    acquisition: Literal["legacy", "edge_uncertainty", "straddle", "ivr"] = "legacy"
        # "legacy" = mean |d mu| per gradient_eval_resolution_V step + exploration_weight x
        #   mean sigma (the original; the gradient term is ~20x smaller than the uncertainty
        #   term, so the path mostly explores)
        # "edge_uncertainty" = mean over the candidate path of [edge strength x normalized
        #   sigma] x edge_weight + exploration_weight x mean normalized sigma -- edge strength
        #   = |mu change over edge_window_V| / edge_height (0..1), sigma normalized by the
        #   GP's prior sigma (0..1). An edge already measured has low sigma, so it stops
        #   paying: the path follows new edges and then moves on to unexplored features.
    edge_weight: float = 1.0
    edge_window_V: Optional[float] = None   # None = 0.25 x length_scale_V
    edge_height: Optional[float] = None     # z change that counts as a full edge (same units as
                                        #   z). None = auto: half the detrended z range measured
                                        #   so far, but at least edge_noise_k x the typical
                                        #   change over edge_window_V along the path
    edge_noise_k: float = 5.0
    # "ivr" (pure GP): next segment = the one whose measurements would most reduce the GP's
    #   posterior variance over the map around the tip, each map point weighted by
    #   (1 - ivr_lambda) + ivr_lambda x edge interest (|grad mu|). ivr_lambda is THE
    #   exploration/exploitation knob: 0 = minimize total uncertainty, 1 = focus on edges.
    #   Needs gp_backend "frozen", "local" or "sparse".
    ivr_lambda: float = 0.5
    # edge interest sharpened (2026-11-09): w = (1 - ivr_lambda) + ivr_lambda * interest ** this.
    # > 1: only clear edges (strong predicted gradient) attract the path; weak / uncertain
    # gradients count less. w stays >= 0, so exploring is never penalised. 1 = as before.
    ivr_interest_power: float = 1.0
    ivr_points_per_segment: int = 8     # candidate measurement points per segment (each stands
                                        #   for the real samples around it)
    ivr_grid_spacing_V: Optional[float] = None  # map grid the reduction is summed over (None =
                                        #   length_scale_V / 2)
    ivr_radius_V: Optional[float] = None  # only map points this close to the tip (None =
                                        #   segment length + 3 x length scale; beyond that a
                                        #   segment barely changes the variance)
    straddle_kappa: float = 1.96        # "straddle": kappa x sigma - |mu - t| (level-set estimation,
                                        #   Bryan et al. 2005); t = substrate + half the feature
                                        #   height -- high on unexplored ground AND on outlines
    lookahead_V: Optional[float] = None     # edge_uncertainty: the exploration term looks this far
                                        #   along each candidate direction (None = one segment)
    explore_measure: Literal["sigma", "distance"] = "sigma"  # edge_uncertainty/straddle exploration
                                        #   term: GP sigma, or distance to the nearest measured
                                        #   point / explore_radius_V (coverage, independent of the
                                        #   GP length scale)
    explore_radius_V: Optional[float] = None  # "distance": ground this far from any measured point
                                        #   counts as fully unexplored (None = 2 x length_scale_V)
    straight_bonus: float = 0.0         # edge_uncertainty: + this x cos(turn angle) -- breaks ties
                                        #   toward going straight (no zigzag in open terrain)
    min_segment_length_V: Optional[float] = None  # adaptive segments (edge_uncertainty/straddle):
                                        #   use this shorter length while the last segment
                                        #   touched a feature; path_segment_length_V on flat
                                        #   ground. None = always path_segment_length_V
    trace_edges: bool = False           # edge_uncertainty/straddle: when an exploration segment
                                        #   crosses a feature outline, trace it in short steps
                                        #   (turn toward the feature when off it, away when on
                                        #   it), until the loop closes, the outline leaves the
                                        #   area, or trace_max_length_V; then explore again. A
                                        #   traced feature is never traced twice.
    trace_step_V: float = 0.2
    trace_turn_deg: float = 45.0          # fallback turn per step when the GP gradient is too weak
    trace_correction_deg: float = 25.0  # GP-guided tracing: steer this much off the GP's outline
                                        #   tangent, toward the feature when off it, away when on
    trace_max_length_V: Optional[float] = None  # per feature (None = 1.5 x pi x the largest
                                        #   feature, from the priors) -- no feature eats the budget
    smooth_turns: bool = False          # each segment starts tangent to the end of the previous one:
                                        #   a cubic Hermite curve from the tip (along its heading) to
                                        #   the planned end point (along the new direction) -- the
                                        #   edge explorer's weave connector; no sudden turns on the
                                        #   piezo. Candidates whose curve leaves the area are skipped
    max_turn_deg: Optional[float] = None  # largest heading change between segments (None = any;
                                        #   ~120 keeps smooth curves gentle -- a 180 deg reversal
                                        #   can't be made smooth in one segment)
    max_path_length_V: Optional[float] = None  # stop once the path is this long (scan-time
                                        #   budget, in V of tip travel); None = no limit
    # --- area + feature size (set by the priors -- see priors.py) ---
    area_center_V: Optional[Tuple[float, float]] = None  # None = start_xy
    area_half_width_V: Optional[float] = None  # keep the path inside this square (None = only
                                          # the safety limit); the first segment starts at its center
    length_scale_V: Optional[float] = None  # GP length scale start (None = 3 x path_segment_length_V)
    length_scale_bounds_V: Optional[Tuple[float, float]] = None  # its fit range (None =
                                          # gradient_eval_resolution_V .. 50 x path_segment_length_V)

    # --- stopping criteria -- see README "When does a scan stop?" ---
    max_segments: int = 100               # hard cap, always enforced
    max_run_time_s: Optional[float] = None  # wall-clock safety net; None = no limit
    target_mean_uncertainty: Optional[float] = None  # stop early once the GP's mean
        # sigma over stop_check_extent_V drops at/below this (same units as z). None
        # disables early stopping -- runs exactly max_segments, like before.
    min_segments_before_stop_check: int = 8  # don't even evaluate the above until
        # this many segments are in, so a lucky-quiet early patch can't end the run
    stop_check_extent_V: Optional[float] = None  # half-width of the square grid the
        # uncertainty check is evaluated over, centered on start_xy. Required if
        # target_mean_uncertainty is set.
    stop_check_grid_n: int = 30


# --------------------------------------------------------------------------------
# Readback -- this is the Z channel: tip deflection, read back on an AI input (there
# is no AO for Z; X/Y are the only outputs -- see backends/pyscanner_backend.py's
# AO_X_CONTROL/AO_Y_CONTROL for those).
# --------------------------------------------------------------------------------
@dataclass
class ReadbackConfig:
    mode: Literal["image", "raw"] = "raw"
    n_image_pixels: int = 200
    ai_channel: str = "AI0"  # Z / deflection input -- CONFIRM ON HARDWARE which AI
                              # your deflection signal is actually wired to
    # PyScanner fast rasters (scan_type 2): the AI record is shifted in time against
    # AO_vs_time (+0.127 s on the instrument, 2026-10-06), so trace and retrace put every
    # feature at two places -- a doubled image along the fast axis. "fit" = fitted on
    # every fast raster from trace/retrace agreement and Z re-placed on the path; "fixed"
    # = always shift by fast_raster_delay_s; "off" = no correction. A shift and the same
    # shift +- half a line both make trace and retrace agree (the second mirrors the
    # image): the one nearest the current estimate is kept -- fast_raster_delay_s to
    # start with (the simulation backends use 0), then the last clear fit.
    # Default "off" since 2026-11-03: on the good grid_refine dataset the fit gave a bogus
    # half-line shift (grid_refine itself no longer uses the PyScanner fast raster).
    fast_raster_delay: Literal["fit", "fixed", "off"] = "off"
    fast_raster_delay_s: float = 0.127
    # RAM: keep one stored sample per store_every recorded ones (block mean of z, sensors,
    # channels and positions) in the run's history -- what's plotted, reconstructed and saved.
    # The planners still get every sample of the waveform just measured. 1 = keep all (at
    # 1 MHz and 30 V/s that's ~2000 samples per 60 nm pixel and ~56 bytes each)
    store_every: int = 1


# --------------------------------------------------------------------------------
# Position sensors (optional) -- the X/Y piezo drive is open loop (see "Units"),
# so the commanded waveform (UI_AO0_V/UI_AO1_V) is not necessarily where the tip
# actually ends up. If your instrument has X/Y position sensors wired to AI inputs,
# enabling this reads them back alongside Z on every segment, so you can correct
# for that open-loop discrepancy (piezo nonlinearity, hysteresis, drift) after the
# fact -- see execution.TrajectoryStep.x_sensor/y_sensor and
# ContinuousExecutor.all_points(), which prefers sensor position over commanded
# whenever it's present. OFF by default -- this is opt-in, not required by the
# package, and needs readback.mode == "raw" (per-sample AI channels; there's no
# sensor-position equivalent of the binned "image" mode).
# --------------------------------------------------------------------------------
@dataclass
class PositionSensorConfig:
    enabled: bool = False
    x_ai_channel: str = "AI1"  # X position sensor -- CONFIRM ON HARDWARE
    y_ai_channel: str = "AI2"  # Y position sensor -- CONFIRM ON HARDWARE
    # where the controller puts them (asylum.AsylumControl.route_sensors, before every run):
    # XSensor -> Cypher PIDS loop x_loop -> a free DAC -> Cypher x_bnc (wired to x_ai_channel)
    x_loop: int = 3
    y_loop: int = 4
    x_bnc: str = "BNCOut0"
    y_bnc: str = "BNCOut1"
    # calibration in the setup (sensors.calibrate, tip withdrawn): a 3 x 3 grid of drive
    # positions +- calibration_span_V, settle, record -> AI -> sensor V -> metres -> drive V
    calibration_span_V: float = 1.0
    # ... but never past the area you'll scan: the priors' half-width (nm), known BEFORE the setup
    # (2026-11-04: a +-2 V grid swept 7 um for a 3 um PFM area). The grid is round drive (0, 0) =
    # where the setup's Go There put the tip; nm -> V with calibration_um_per_V_guess (|um/V|,
    # the last calibration: X 1.72, Y 1.85 on 2026-11-04), x calibration_area_fraction
    calibration_area_half_nm: Optional[float] = None
    calibration_um_per_V_guess: float = 1.85
    calibration_area_fraction: float = 0.9
    calibration_settle_s: float = 0.3
    calibration_dwell_s: float = 0.2     # recording at each point (>= 150k samples at 1 MHz: PyScanner pads shorter)
    x_m_per_V: Optional[float] = None   # sensor V -> metres; None = Igor's XLVDTSens
    y_m_per_V: Optional[float] = None   #                            ... YLVDTSens
    smooth_samples: int = 0             # moving average over this many samples (sensor noise); 0 = none
    # the BNC outputs (and PyScanner's AI) stop at +-10 V, the sensors can go past that: with
    # divide_output the copy loop outputs sensor / output_divider (P = -1 / divider instead of
    # -1). The calibration measures the AI -> sensor gain (then ~ the divider), so the
    # positions stay exact -- only the AI resolution per nm drops by that factor
    divide_output: bool = False
    output_divider: float = 2.0
    # with a (good) calibration:
    plan_on_sensors: bool = True    # every planner decides where to scan next from the MEASURED
                                    # positions (crossings, substrate fit, coverage, GP data, ...)
    correct_linear: bool = False    # (a) send every waveform through the calibration's inverse
                                    #     drive map (scale + cross-talk)
    correct_offset: bool = False    # (b) after each waveform, its measured end error (mean over its
                                    #     last correction_window_s) shifts the next waveforms
    correction_gain: float = 0.5    # (b) fraction of the newly measured error taken each time
    correction_window_s: float = 0.02   # (b) the end stretch the error is averaged over
    correction_ramp_s: float = 0.02     # (b) a new offset is eased in over a waveform's first this-long
    initial_offset_V: Optional[Tuple[float, float]] = None   # (b) the offset to start a run with
                                        #     (e.g. the last run's executor._u_target; eased in over
                                        #     the first waveform's start) -- None = (0, 0)
    correction_max_V: float = 1.0       # (b) never more than this (drive volts) -- a wrong calibration
                                        #     can't run away

    @property
    def copy_gain(self) -> float:
        """The copy loop's P gain: -1 (output = sensor), or -1 / output_divider."""
        return -1.0 / (self.output_divider if self.divide_output and self.output_divider else 1.0)


# --------------------------------------------------------------------------------
# Z signal on a Cypher BNC (-> PyScanner's readback AI, cfg.readback.ai_channel)
# --------------------------------------------------------------------------------
@dataclass
class ZSignalConfig:
    """Which Z signal the readback AI sees, chosen in software: the setup copies `source`
    through a Cypher PIDS loop (straight copy, as for the X/Y sensors) to a free DAC
    (`dac` first) -> `bnc` (locked), and the AI's cable stays on that BNC. With the tip
    withdrawn it then checks that the AI really reads that copy. Off (route False): the AI
    reads whatever it's cabled to, as before."""
    route: bool = False
    source: str = "ZSensor"          # the Cypher input copied: "ZSensor" (the Z LVDT) or the Z piezo
                                     #   signal's Cypher name -- CONFIRM ON HARDWARE (ir_ReadValue of it
                                     #   must give a number)
    loop: int = 5                    # Cypher PIDS loop doing the copy (3/4: the X/Y sensors)
    dac: str = "OutFastA"            # preferred DAC (another free one is used if it's taken; OutA/OutB
                                     #   are Igor's own LVDT.X/Y loops 0/1 -- seen 2026-10-13)
    bnc: str = "BNCOut2"             # the BNC the readback AI is cabled to
    output_divider: float = 1.0      # BNC = source / this (the BNC / AI stop at +-10 V)
    m_per_V: Optional[float] = None  # source V -> metres (saved with the run); None = Igor's
                                     #   ZLVDTSens for "ZSensor", unknown otherwise
    check_tol_V: float = 0.05        # setup: AI vs the copy, at rest
    invert: bool = True              # the routed signal's sign flipped in software (z x -1), so up = taller:
                                     #   ZSensor reads opposite to the height on this Cypher (2026-10-13).
                                     #   Only with the routing on (AI0 = the copy); the copy itself is unchanged
    units: Literal["V", "nm"] = "V"  # "nm": every measured Z is converted to nm as it comes in
                                     #   (x the setup's nm per AI volt: m_per_V x output_divider) --
                                     #   planners, thresholds (edge_detect.manual_threshold,
                                     #   grid_refine.detect_threshold, gp_path.target_mean_uncertainty),
                                     #   plots and the saved z are then in nm. Needs route + a known
                                     #   m_per_V (ZSensor: Igor's ZLVDTSens); otherwise Z stays in V

    @property
    def copy_gain(self) -> float:
        return -1.0 / (self.output_divider or 1.0)


# --------------------------------------------------------------------------------
# PFM (Run_PFM_DomainWalls.ipynb; selfdriving_afm/pfm.py) -- off unless that notebook turns it on
# --------------------------------------------------------------------------------
@dataclass
class PFMChannelConfig:
    """One PFM signal copied by a PIDS loop (straight copy, as the X/Y sensors) to a DAC ->
    a BNC, cabled to a PyScanner AI. BNC volts = gain x the signal (Igor's units: Amplitude
    in V, Phase in degrees)."""
    enabled: bool = True
    source: str = "Amplitude"        # Igor's signal name (ir_ReadValue of it must give a number)
    ai_channel: str = "AI0"          # the PyScanner input it's cabled to
    side: str = "Cypher"             # "Cypher" or "ARC": whose PIDS loop / DAC / BNC
    loop: int = 5
    dac: str = "OutFastA"            # preferred DAC (another free one if taken)
    bnc: str = "BNCOut2"
    gain: float = 10.0               # BNC volts per signal unit (keep |gain x signal| < 10 V)
    unit: str = "V"                  # the signal's unit, for labels
    label: str = "PFM amplitude"
    cmap: str = "viridis"            # its colour map (live view rows of the signals other than z)


def _amp_default():
    return PFMChannelConfig()


def _phase_default():
    return PFMChannelConfig(source="Phase", ai_channel="AI3", side="ARC", loop=2, dac="OutA", bnc="BNCOut0",
                            gain=0.05, unit="deg", label="PFM phase", cmap="inferno")


@dataclass
class DomainWallConfig:
    """How a domain wall is recognised along the path (pfm.DomainModel)."""
    min_phase_separation_deg: float = 60.0   # two phase populations at least this far apart = two domains
    min_second_fraction: float = 0.002       # ...the second one at least this fraction of the samples
    phase_memory: float = 0.8                # the domains' phases follow drift: each waveform's estimate
                                             #   is mixed in with weight (1 - this)
    amp_reference_quantile: float = 0.7      # the domains' amplitude: this quantile of the measured one
    amp_dip_fraction: float = 0.6            # a wall: amplitude below this x the domains' (amplitude only:
                                             #   the wall itself; with the phase: the confirmation)
    confirm_with_amplitude: bool = True      # phase + amplitude: a phase flip is a wall only if the
                                             #   amplitude dips near it
    confirm_window_V: float = 0.05           # ...within this much path of the flip
    # which walls the explorer follows: "180" = phase flips (out-of-plane domains, DomainModel);
    # "in_plane" = amplitude STEPS between two levels with no phase flip (pfm.InPlaneWallModel,
    # 2026-11-04: 3.1 vs 2.0 mV, phase change < 1 deg); the 180 deg walls' amplitude dips are ignored
    target: str = "180"
    inplane_min_level_ratio: float = 1.25    # two amplitude levels at least this far apart (ratio)
    inplane_min_fraction: float = 0.05       # ...each at least this fraction of the samples
    inplane_dip_fraction: float = 0.6        # below this x the low level: a 180 deg wall's dip -> ignored
    inplane_dip_guard_V: float = 0.03        # ...and this much path either side of the dip (its flanks)
    inplane_flip_guard_V: float = 0.06       # an amplitude-level change with a phase flip within this much
                                             #   path of it is a 180 deg wall (some change level too,
                                             #   2026-11-04 run 3): not followed
    inplane_direction_harmonics: int = 2     # amplitude vs the tip's direction of motion: cos/sin up to this
                                             #   (the friction effect, as large as the step; 0 = off)
    inplane_smooth_V: float = 0.01           # running mean of log amplitude along the path (V of path)
    inplane_max_samples: int = 300_000       # learning reservoir (subsampled), all waveforms so far


@dataclass
class PFMConfig:
    enabled: bool = False
    amplitude: PFMChannelConfig = field(default_factory=_amp_default)
    phase: PFMChannelConfig = field(default_factory=_phase_default)
    walls: DomainWallConfig = field(default_factory=DomainWallConfig)
    check_tol: float = 0.05          # setup: |AI - the DAC| at rest (V)

    def channels(self):
        """The enabled channels, {"amplitude": cfg, "phase": cfg}."""
        return {k: c for k, c in (("amplitude", self.amplitude), ("phase", self.phase)) if c.enabled}


# --------------------------------------------------------------------------------
# Execution (continuity between trajectory primitives -- see execution.py)
# --------------------------------------------------------------------------------
@dataclass
class ExecutionConfig:
    start_xy_V: Tuple[float, float] = (0.0, 0.0)   # every run() starts here, per spec
    # Auto-inserted connecting lines are sampled at VelocityConfig.sample_spacing_V too --
    # transit motion is still real physical motion and should honor the same speed limit.


# --------------------------------------------------------------------------------
# Live view (see viz.LiveEdgeFollowView / viz.LiveGPPathView)
# --------------------------------------------------------------------------------
@dataclass
class LiveViewConfig:
    """Redraw cadence is wall-clock throttled, not step-count throttled: a spiral or
    circle primitive can be thousands of samples executed in milliseconds (real
    hardware) or microseconds (simulation), so gating on "every N executor steps"
    either redraws way more than a human can watch or barely at all depending on
    how fast the backend runs. Time-based throttling gives a consistent "few times a
    second" feel regardless of backend speed, and guarantees the live view itself
    never becomes the bottleneck in the acquisition loop.
    """
    enabled: bool = True
    min_redraw_interval_s: float = 0.25       # path panel
    max_time_fraction: float = 0.2           # the live view may use at most this fraction of the run
                                              # (redraws take ~0.2 s each); lower = faster scans, fewer frames
    min_reconstruction_interval_s: float = 1.0  # interpolated/GP panel -- more expensive
    min_points_for_reconstruction: int = 10
    max_reconstruction_points: int = 4000      # subsample beyond this so griddata/GP stays fast
    # --- edge-follow view: column 2 is the PyScanner-style binned AI0 image, column 3
    # fills in its unmeasured pixels with this method (see reconstruction.py) ---
    reconstruction: Literal["compressed_sensing", "gp", "linear", "fast", "raster_columns",
                            "interpolated", "tv"] = "compressed_sensing"
                                     # "raster_columns" = column-wise PCHIP between scan
                                     # lines -- for raster data only (the Simulation
                                     # notebook's classic-raster row uses it)
    n_image_pixels: int = 200        # binned + reconstructed image resolution (match PyScanner's N_image_pixels)
    cs_lambda: float = 0.005         # CS sparsity weight, relative to the largest DCT coefficient:
                                     #   higher = smoother, lower = sticks closer to measured pixels
    cs_iterations: int = 500         # FISTA iterations -- more = converged further, slower
    cs_detrend: bool = True          # remove the best-fit plane (sample tilt) before CS
    tv_mu: float = 0.02              # "tv" (total variation): data weight -- lower = flatter plateaus,
                                     #   sharper edges, more of the measured noise smoothed away
    tv_iterations: int = 1500        # Chambolle-Pock iterations (~10 s for 256 x 256)
    gp_max_points: int = 1500        # GP trains on at most this many measured pixels (O(n^3) -- the speed knob)
    gp_kernel: Literal["matern52", "rbf"] = "matern52"  # Matern 5/2 as in the SS-KPFM paper
    gp_length_scale_frac: float = 0.05  # initial length scale, as a fraction of the image width
    gp_optimize: bool = False        # True = re-fit GP hyperparameters on every redraw (better fit,
                                     #   but ~20x slower: seconds to tens of seconds per update)
    figsize: Tuple[float, float] = (13.5, 4.0)  # inches, for 3 columns (both live views are
        # 3-panel now -- see viz.LiveEdgeFollowView/LiveGPPathView); shrink further (e.g.
        # (10, 3)) if your Jupyter window still shows it wider than the notebook -- set
        # cfg.live_view.figsize before building the executor/planner (Run_SelfDriving_Scan.ipynb, cell 2)
    cmap: str = "copper"
    # 4th column: where the reconstruction has no data nearby (reconstruction.uncertainty_map):
    # 0% on measured pixels -> 100% at uncertainty_scale_V or more from the nearest one
    # (the priors set it to the smallest feature; None = 5% of the frame width)
    show_uncertainty: bool = True
    uncertainty_scale_V: Optional[float] = None
    # simulation only (a ground truth is known): 5th column, the real error |recon - truth|
    show_true_error: bool = True
    flatten_substrate: bool = True   # edge_follow / edge_explorer: show (and save) the binned image and
                                     #   reconstruction with the planner's fitted substrate (tilt) removed
    # gp_path (2026-11-09): what the middle panel shows and what is saved as recon_Z. The GP
    # decides where to go; as a MAP of step-like topography its mean rings and blurs (on the
    # instrument, hold-out RMSE 26 nm vs 14 nm for TV of the same samples), so by default the
    # reconstruction is `reconstruction` (e.g. "tv") of the measured samples, substrate plane
    # removed. "gp_mean" = the old panel (the planner's surrogate). recon_mean / recon_sigma
    # (the GP) are saved either way.
    gp_view: Literal["reconstruction", "gp_mean"] = "reconstruction"
    live_tv_iterations: int = 300    # TV iterations for live redraws (the saved map uses tv_iterations)
    fixed_bounds_V: Optional[Tuple[float, float, float, float]] = None  # (xmin, xmax, ymin, ymax):
                                     # every panel always shows this area (the priors set it
                                     # to the area to explore); None = grow with the scanned area


# --------------------------------------------------------------------------------
# Asylum Research / Igor Pro (the microscope's own software, over COM -- asylum.py):
# engage / withdraw, and routing the FPGA's AO0/AO1 to the X/Y piezos
# --------------------------------------------------------------------------------
@dataclass
class AsylumConfig:
    controller: Literal["cypher", "arc"] = "cypher"   # whose inputs the X/Y loops read:
                                     # "Cypher.Input.A/B" or "Arc.Input.A/B"
    # crosspoint: the panel's input -> the signal routed to it (the FPGA's AO0 / AO1 come
    # in on BNCIn0 / BNCIn1), locked so a mode change doesn't re-route them
    crosspoint: Dict[str, str] = field(default_factory=lambda: {"InA": "BNCIn0", "InB": "BNCIn1"})
    lock_crosspoint: bool = True
    # the X/Y loops follow inputs A/B: ir_SetPISLoop(loop, events, input, setpoint,
    # P, I, S, output, out_min, out_max) -- as typed in Igor (Cypher XY Control Igor Command.txt)
    xy_loop_setpoint: float = 0.0
    xy_loop_gains: Tuple[float, float, float] = (-8.0, 0.0, 0.0)   # P, I, S
    xy_loop_output_range: Tuple[float, float] = (-10.0, 150.0)
    center_first: bool = True        # setup: tip to Igor's designated spot first (Go There,
                                     # DoForceFunc("GoForce_1")), before the loops go to the FPGA
    center_transit_s: float = 2.0    # ... time given to get there
    engage_wait_s: float = 5.0       # SimpleEngageMe: time to let it reach the surface
    withdraw_wait_s: float = 2.0
    ao_zero_tol_V: float = 1e-3      # FPGA AO0/AO1 closer than this to 0 count as at 0


# --------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    readback: ReadbackConfig = field(default_factory=ReadbackConfig)
    position_sensors: PositionSensorConfig = field(default_factory=PositionSensorConfig)
    z_signal: ZSignalConfig = field(default_factory=ZSignalConfig)
    pfm: PFMConfig = field(default_factory=PFMConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    live_view: LiveViewConfig = field(default_factory=LiveViewConfig)
    edge_follow: EdgeFollowConfig = field(default_factory=EdgeFollowConfig)
    gp_path: GPPathConfig = field(default_factory=GPPathConfig)
    edge_explorer: EdgeExplorerConfig = field(default_factory=EdgeExplorerConfig)
    grid_refine: GridRefineConfig = field(default_factory=GridRefineConfig)
    asylum: AsylumConfig = field(default_factory=AsylumConfig)
