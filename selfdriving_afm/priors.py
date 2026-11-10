"""
Priors: what you know about the sample BEFORE scanning, asked once per run (both
notebooks), and turned into every mode's settings:

    feature_min_V / feature_max_V   the smallest and largest feature expected -- its
                                    narrowest width (a disk's diameter, a line's width)
    area_center_V, area_half_width_V   the most the scan may explore: a square

    p = priors.ask(defaults)              # prompts; Enter keeps each default
    notes = priors.apply(cfg, p)          # sets every mode's area + size parameters

With the X/Y position sensors on and calibrated (the setup cell), the run notebook asks
in NANOMETRES instead (ask_for_run / ask_nm): sizes and the area along the sensors' axes,
the centre in the sensors' coordinates (the "X/Y (um, sensor)" axes of the plots),
converted to drive volts with that calibration -- everything downstream stays in volts.

What each mode gets (the rules are the constants below):

  edge_follow   first search spiral = the whole area (a circle inscribed in the square;
                corners by later spirals), spiral turns SPIRAL_PITCH x smallest feature
                apart (no feature fits between two turns); later spirals stay inside the
                area. Tracing circles: radius_mode "curvature" -- each circle measures the
                edge's local radius of curvature R and the next one gets r =
                CURVATURE_FRACTION x R, between CIRCLE_MIN x smallest and CIRCLE_MAX x
                largest feature. With edge_follow.first_scan_pattern = "raster" or
                "lissajous" the first scan covers the whole square instead (lines
                SPIRAL_PITCH x smallest apart).
  grid_refine   sparse scan = ONE scan over the area, lines SPARSE_SPACING x smallest
                feature apart. Refinement tiles at most TILE_MAX x smallest feature wide
                (touching tiles merge into one region, so small tiles cost nothing extra);
                a closed ring of tiles round an inside at most the largest feature wide
                gets that inside scanned too (a bigger enclosed area is substrate).
  edge_explorer explores the area; exploration sweeps ~the smallest feature apart; the
                first straight path goes from the center to the area's edge; edges are
                followed with zigzag steps FOLLOW_MIN x smallest .. FOLLOW_MAX x largest;
                a closed feature's inside is scanned with lines SPARSE_SPACING x smallest
                apart (internal edges at least that big are crossed).
  gp_path       stays inside the area, starts at its center; GP length scale from the
                feature sizes.

Why a radius from the curvature: a circle centred on the edge reaches r into the
feature, so r must stay below the feature's width (else it sees the far side) and about
its radius of curvature (else it takes too few, too big steps round a round feature --
r = R is ~6-8 circles per loop). A too-big circle shows itself: its on-feature arc
shrinks, so the next circle is smaller. Every tracing circle costs ~3 s of do_scan
overhead on the instrument, so the radius is kept as big as that allows.
"""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from .config import ExperimentConfig

SPIRAL_PITCH = 0.5      # spiral turn spacing / smallest feature (a disk that size is always
                        # crossed over >= 87% of its diameter)
CIRCLE_MIN = 0.25       # smallest tracing radius / smallest feature
CIRCLE_MAX = 0.75       # largest tracing radius / largest feature
CURVATURE_FRACTION = 1.0  # tracing radius / the edge's measured radius of curvature
# (CIRCLE_MIN/MAX and CURVATURE_FRACTION calibrated in simulation, 2026-09-29: two samples
# with features 0.3-2.4 V; every setting found 100% of the edges -- these gave the fewest
# circles at ~2.5% bad pixels. The geometric guess, 0.5 x R, took ~40% more circles.)
SPARSE_SPACING = 0.7    # grid_refine sparse line spacing / smallest feature
INTERIOR_SPACING = 0.3  # edge_explorer: inside-scan line spacing / smallest feature (was 0.7)
TILE_MAX = 0.5          # grid_refine refinement tile size (largest) / smallest feature
FOLLOW_MIN = 0.5        # edge_explorer zigzag step (shortest) / smallest feature
FOLLOW_MAX = 0.5        # edge_explorer zigzag step (longest, straight edges) / largest feature
EX_INITIAL = 0.5        # edge_explorer: first straight path / area half-width
EX_CAL_RADIUS = 0.15    # ... calibration circle radius (at most) / area half-width
EX_SEGMENT = 0.2        # ... exploring step and longest zigzag step (at most) / area half-width
RELOCATE_RADIUS = 0.2   # edge_follow: later search spirals, as a fraction of the area half-width
                        # -- they fill the corners the first (inscribed) spiral leaves, so they
                        # must fit there
MAX_SPIRAL_POINTS = 40_000_000  # the first spiral is shrunk to stay under this (trajectory cap 50 M)
OVERHEAD_PER_WAVEFORM_S = 3.0   # do_scan overhead on the instrument, for the time estimates


@dataclass
class ScanPriors:
    feature_min_V: float = 0.5
    feature_max_V: float = 2.0
    area_half_width_V: float = 4.0
    area_center_V: Tuple[float, float] = (0.0, 0.0)
    # what was typed, when the priors were asked in nm (ask_nm): {feature_min_nm, ...,
    # nm_per_V}; None = asked in volts
    entered: Optional[dict] = None
    # edge detection (edge_explorer, edge_follow): a fixed on-feature threshold -- height above
    # the substrate, in edge_threshold_unit (the run's Z unit: "nm" or "V"); None = the
    # configuration cell's setting (automatic by default). ask_edge_threshold
    edge_threshold: Optional[float] = None
    edge_threshold_unit: str = "V"
    edge_mode: Optional[str] = None      # "manual" (edge_threshold), "auto", or None = keep the cell's

    @property
    def bounds(self):
        (cx, cy), h = self.area_center_V, self.area_half_width_V
        return (cx - h, cx + h, cy - h, cy + h)

    def check(self, max_amplitude_V: float = None) -> List[str]:
        """Raises on impossible values; returns warnings."""
        if not (self.feature_min_V > 0 and self.feature_max_V > 0):
            raise ValueError("feature sizes must be > 0")
        if self.feature_max_V < self.feature_min_V:
            raise ValueError(f"largest feature ({self.feature_max_V} V) is smaller than the smallest "
                             f"({self.feature_min_V} V)")
        if self.area_half_width_V <= 0:
            raise ValueError("area half-width must be > 0")
        warn = []
        if 2 * self.area_half_width_V < self.feature_max_V:
            warn.append(f"the area ({2 * self.area_half_width_V:g} V wide) is smaller than the largest "
                        f"feature ({self.feature_max_V:g} V) -- features will run off it")
        if max_amplitude_V is not None:
            b = self.bounds
            if max(abs(v) for v in b) > max_amplitude_V:
                warn.append(f"the area {tuple(round(v, 3) for v in b)} V goes past the safety limit "
                            f"+/-{max_amplitude_V:g} V -- every mode clips it to the limit")
        return warn

    def __str__(self):
        s = (f"features {self.feature_min_V:.4g}..{self.feature_max_V:.4g} V, area: "
             f"{2 * self.area_half_width_V:.4g} x {2 * self.area_half_width_V:.4g} V square centered at "
             f"({self.area_center_V[0]:.4g}, {self.area_center_V[1]:.4g}) V")
        e = self.entered
        if e:
            s += (f"  [entered: features {e['feature_min_nm']:g}..{e['feature_max_nm']:g} nm, area "
                  f"{2 * e['area_half_width_nm']:g} nm wide centered at ({e['area_center_nm'][0]:g}, "
                  f"{e['area_center_nm'][1]:g}) nm (sensors); {e['nm_per_V']:.5g} nm per drive V]")
        if self.edge_mode == "manual" and self.edge_threshold is not None:
            s += f"; edge threshold {self.edge_threshold:g} {self.edge_threshold_unit} above the substrate"
        elif self.edge_mode == "auto":
            s += "; edge threshold automatic"
        return s


# ------------------------------------------------------------------------------------
# asking
# ------------------------------------------------------------------------------------
def _ask_float(prompt: str, default: float) -> float:
    while True:
        s = input(f"{prompt} [{default:g}]: ").strip()
        if not s:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print(f"  not a number: {s!r}")


def ask(defaults: Optional[ScanPriors] = None, interactive: bool = True, what: str = "feature") -> ScanPriors:
    """Ask for the priors (Enter keeps the default shown in brackets). Where no input
    is possible (e.g. running the notebook headless), or interactive=False, the
    defaults are used as they are."""
    d = copy.deepcopy(defaults or ScanPriors())
    if not interactive:
        return d
    try:
        print("Priors for this scan (bias volts) -- press Enter to keep the value in brackets")
        d.feature_min_V = _ask_float(f"  smallest expected {what} (narrowest width, V)", d.feature_min_V)
        d.feature_max_V = _ask_float(f"  largest expected {what} (V)", d.feature_max_V)
        d.area_half_width_V = _ask_float("  max area to explore: half-width of the square (V)", d.area_half_width_V)
        cx = _ask_float("  ...its center X (V)", d.area_center_V[0])
        cy = _ask_float("  ...its center Y (V)", d.area_center_V[1])
        d.area_center_V = (cx, cy)
    except (EOFError, OSError, RuntimeError, NotImplementedError) as e:
        # no stdin (headless run): keep the defaults. (Jupyter raises StdinNotImplementedError,
        # a subclass of NotImplementedError / RuntimeError depending on the version.)
        print(f"  (no input possible -- {type(e).__name__}; using the defaults)")
        return copy.deepcopy(defaults or ScanPriors())
    d.check()
    return d


def _nm_per_V(cal) -> float:
    """Sizes: nm per drive volt, the mean of the two axes' (|um_per_V|; they differ by a few %)."""
    kx, ky = cal.um_per_V
    return 500.0 * (abs(kx) + abs(ky))


def nm_to_V(feature_min_nm, feature_max_nm, area_half_width_nm, area_center_nm, cal) -> ScanPriors:
    """Priors given in nm (sensor frame) -> drive volts, with the sensor calibration `cal`
    (sensors.SensorCalibration): sizes / nm_per_V; the centre through the calibration's
    own map (um = k V + b, per axis, inverted)."""
    k = _nm_per_V(cal)
    cx = (float(area_center_nm[0]) * 1e-9 - cal.x.b) / cal.x.k
    cy = (float(area_center_nm[1]) * 1e-9 - cal.y.b) / cal.y.k
    return ScanPriors(float(feature_min_nm) / k, float(feature_max_nm) / k, float(area_half_width_nm) / k,
                      (float(cx), float(cy)),
                      entered=dict(feature_min_nm=float(feature_min_nm), feature_max_nm=float(feature_max_nm),
                                   area_half_width_nm=float(area_half_width_nm),
                                   area_center_nm=(float(area_center_nm[0]), float(area_center_nm[1])),
                                   nm_per_V=k, um_per_V=[float(v) for v in cal.um_per_V]))


def V_to_nm(p: ScanPriors, cal) -> dict:
    """The nm values (sensor frame) of priors in drive volts -- the defaults offered in nm."""
    k = _nm_per_V(cal)
    ux, uy = cal.um(p.area_center_V[0], p.area_center_V[1])
    return dict(feature_min_nm=p.feature_min_V * k, feature_max_nm=p.feature_max_V * k,
                area_half_width_nm=p.area_half_width_V * k, area_center_nm=(float(ux) * 1e3, float(uy) * 1e3))


def ask_nm(defaults_nm: dict, cal, interactive: bool = True, what: str = "feature") -> ScanPriors:
    """Ask for the priors in nm (Enter keeps the default in brackets), converted to drive
    volts with the sensor calibration `cal` (nm_to_V)."""
    d = dict(defaults_nm)
    if interactive:
        try:
            print(f"Priors for this scan (nm, along the position sensors; {_nm_per_V(cal):.5g} nm per drive V) "
                  f"-- press Enter to keep the value in brackets")
            d["feature_min_nm"] = _ask_float(f"  smallest expected {what} (narrowest width, nm)", round(d["feature_min_nm"]))
            d["feature_max_nm"] = _ask_float(f"  largest expected {what} (nm)", round(d["feature_max_nm"]))
            d["area_half_width_nm"] = _ask_float("  max area to explore: half-width of the square (nm)",
                                                 round(d["area_half_width_nm"]))
            cx = _ask_float("  ...its center X (nm, sensor)", round(d["area_center_nm"][0]))
            cy = _ask_float("  ...its center Y (nm, sensor)", round(d["area_center_nm"][1]))
            d["area_center_nm"] = (cx, cy)
        except (EOFError, OSError, RuntimeError, NotImplementedError) as e:
            print(f"  (no input possible -- {type(e).__name__}; using the defaults)")
            d = dict(defaults_nm)
    p = nm_to_V(d["feature_min_nm"], d["feature_max_nm"], d["area_half_width_nm"], d["area_center_nm"], cal)
    p.check()
    return p


def ask_for_run(cfg: ExperimentConfig, defaults: Optional[ScanPriors] = None, defaults_nm: Optional[dict] = None,
                sensor_cal=None, interactive: bool = True, what: str = "feature") -> ScanPriors:
    """The run notebook's priors: in nm if cfg.position_sensors is on and `sensor_cal` (the
    setup's calibration) is usable, else in volts (ask). defaults_nm: the nm defaults
    (feature_min_nm, feature_max_nm, area_half_width_nm, area_center_nm); None = `defaults`
    converted."""
    defaults = defaults or ScanPriors()
    if cfg.position_sensors.enabled:
        if sensor_cal is not None and getattr(sensor_cal, "ok", True):
            dn = dict(defaults_nm) if defaults_nm else V_to_nm(defaults, sensor_cal)
            return ask_nm(dn, sensor_cal, interactive, what)
        print("NOTE: position sensors on but no usable calibration (run the setup cell first) -- priors in volts")
    return ask(defaults, interactive, what)


def ask_edge_threshold(cfg: ExperimentConfig, p: ScanPriors, z_info: Optional[dict] = None,
                       interactive: bool = True, modality: Optional[str] = None) -> ScanPriors:
    """Ask for the edge-detection threshold (edge_explorer and edge_follow: on a feature =
    height above the substrate beyond it) in the run's Z unit -- nm when Z is converted
    (cfg.z_signal.units = "nm", ZSensor routed in the setup: z_info), else V. Enter keeps
    the configuration cell's setting (shown in brackets: a number = its manual threshold,
    "auto" = the automatic one); a number = that fixed threshold; "auto" = automatic.
    Stored in p (edge_threshold, edge_threshold_unit); apply() sets it."""
    from .execution import z_units
    scale, unit, why = z_units(cfg.z_signal, z_info)
    ed = (cfg.edge_follow if modality == "edge_follow" else cfg.edge_explorer).edge_detect
    cur = ed.manual_threshold if (ed.mode == "manual" and ed.manual_threshold is not None) else None
    p.edge_threshold_unit = unit
    if why:
        print(f"  (Z in volts: {why})")
    p.edge_mode, p.edge_threshold = None, None      # default: keep the configuration cell's
    if not interactive:
        return p
    hint = "; ~1/3 of the features' height is a safe choice, 'auto' = automatic"
    while True:
        try:
            s = input(f"  edge detection: threshold, height above the substrate ({unit}){hint} "
                      f"[{'auto' if cur is None else f'{cur:g}'}]: ").strip().lower()
        except (EOFError, OSError, RuntimeError, NotImplementedError):
            s = ""
        if not s:                    # keep the cell's (cur: shown in brackets)
            break
        if s == "auto":
            p.edge_mode = "auto"
            break
        try:
            v = float(s)
        except ValueError:
            print(f"  not a number (or 'auto'): {s!r}")
            continue
        if v <= 0:
            print("  must be > 0")
            continue
        p.edge_mode, p.edge_threshold = "manual", v
        break
    if p.edge_threshold is not None and unit == "nm":
        print(f"  = {p.edge_threshold / abs(scale):.4g} AI volts ({abs(scale):.5g} nm per volt)")
    return p


def from_sample(sample, area_margin_V: float = 0.0) -> ScanPriors:
    """The true priors of a simulated sample (topography.Sample): its features' smallest
    and largest narrowest widths, and its field as the area -- what a perfect prior
    would say. Ring: its wall width; terrace steps are ignored."""
    sizes = []
    for f in getattr(sample, "features", []):
        if f.shape == "step":
            continue
        if f.shape == "ring" and f.width:
            sizes.append(float(f.width))
        elif f.shape == "polygon" and f.points is not None:
            ext = f.points.max(axis=0) - f.points.min(axis=0)
            sizes.append(float(min(ext)))
        else:
            sizes.append(float(min(f.size)))
    x0, x1, y0, y1 = sample.bounds
    h = 0.5 * max(x1 - x0, y1 - y0) + area_margin_V
    if not sizes:
        sizes = [0.1 * h]
    return ScanPriors(min(sizes), max(sizes), h, (0.5 * (x0 + x1), 0.5 * (y0 + y1)))


# ------------------------------------------------------------------------------------
# applying
# ------------------------------------------------------------------------------------
def apply(cfg: ExperimentConfig, p: ScanPriors) -> List[str]:
    """Set every mode's area and feature-size parameters in `cfg` (in place) from the
    priors. Returns one line per mode saying what was set (and warnings)."""
    lim = 0.999 * cfg.safety.max_amplitude_V
    notes = [f"WARNING: {w}" for w in p.check(cfg.safety.max_amplitude_V)]
    smin, smax, h, c = p.feature_min_V, p.feature_max_V, p.area_half_width_V, tuple(p.area_center_V)
    ds = cfg.velocity.sample_spacing_V
    # --- edge detection (asked with the priors: ask_edge_threshold) ---
    em = getattr(p, "edge_mode", None)
    if em == "manual" and p.edge_threshold is not None:
        for ed in (cfg.edge_explorer.edge_detect, cfg.edge_follow.edge_detect):
            ed.mode, ed.manual_threshold = "manual", float(p.edge_threshold)
        notes.append(f"edge detection (edge_explorer, edge_follow): fixed threshold {p.edge_threshold:g} "
                     f"{p.edge_threshold_unit} above the substrate")
    elif em == "auto":
        for ed in (cfg.edge_explorer.edge_detect, cfg.edge_follow.edge_detect):
            ed.mode = "auto_mad"
        notes.append(f"edge detection (edge_explorer, edge_follow): automatic threshold (in {p.edge_threshold_unit})")
    else:
        for name, ed in (("edge_explorer", cfg.edge_explorer.edge_detect), ("edge_follow", cfg.edge_follow.edge_detect)):
            what = (f"fixed {ed.manual_threshold:g}" if ed.mode == "manual" and ed.manual_threshold is not None
                    else "automatic")
            notes.append(f"edge detection ({name}): as in the configuration cell -- {what} "
                         f"(in {getattr(p, 'edge_threshold_unit', 'V')})")
    v = cfg.velocity.tip_velocity_V_s
    per_wave = cfg.safety.max_samples_per_waveform

    # --- edge_follow ---
    ef = cfg.edge_follow
    pitch = SPIRAL_PITCH * smin
    r1 = min(h, lim)
    first = ef.first_scan_pattern
    # path length / r^2 / pitch: Archimedean spiral pi; raster over the 2r square 4;
    # Lissajous with the same largest gap ~1.6 x the raster
    shape_k = {"spiral": math.pi, "raster": 4.0, "lissajous": 6.4}.get(first, math.pi)
    n_pts = shape_k * r1 ** 2 / pitch / ds
    if n_pts > MAX_SPIRAL_POINTS:
        r1 = math.sqrt(MAX_SPIRAL_POINTS * pitch * ds / shape_k)
        notes.append(f"WARNING: edge_follow: a first {first} over the whole area would need {n_pts / 1e6:.0f} M "
                     f"samples -- it is shrunk to {'radius' if first == 'spiral' else 'half-width'} {r1:.3g} V "
                     f"(later spirals cover the rest). A faster tip / lower sample rate lets it cover everything.")
        n_pts = MAX_SPIRAL_POINTS
    ef.explore_center_V = c
    ef.explore_extent_V = h
    ef.first_search_radius_V = r1
    ef.spiral_pitch_V = pitch
    ef.initial_search_radius_V = min(r1, RELOCATE_RADIUS * h)
    ef.spiral_search_mode = "relocate"
    ef.circle.radius_mode = "curvature"
    ef.circle.curvature_radius_fraction = CURVATURE_FRACTION
    ef.circle.adaptive_min_radius_V = CIRCLE_MIN * smin
    ef.circle.adaptive_max_radius_V = max(CIRCLE_MAX * smax, CIRCLE_MIN * smin)
    spiral_s = n_pts * ds / v + OVERHEAD_PER_WAVEFORM_S * math.ceil(n_pts / per_wave)
    what = (f"first spiral radius {r1:.3g} V" if first == "spiral" else
            f"first scan a {first} over the {2 * r1:.3g} V square")
    notes.append(f"edge_follow: {what} at ({c[0]:g}, {c[1]:g}), {'turns' if first == 'spiral' else 'lines'} "
                 f"{pitch:.3g} V apart "
                 f"({n_pts / 1e6:.2g} M samples, ~{spiral_s:.0f} s on the instrument); later spirals radius "
                 f"{ef.initial_search_radius_V:.3g} V inside the area; tracing circles r = "
                 f"{ef.circle.adaptive_min_radius_V:.3g}..{ef.circle.adaptive_max_radius_V:.3g} V, sized from "
                 f"the measured edge curvature")

    # --- grid_refine ---
    gr = cfg.grid_refine
    gr.center_V = c
    gr.half_size_V = h
    gr.sparse_n_lines = max(3, math.ceil(2 * h / (SPARSE_SPACING * smin)) + 1)
    gr.tile_max_V = TILE_MAX * smin
    gr.fill_max_size_V = smax
    notes.append(f"grid_refine: {'crosshatch' if gr.sparse_crosshatch else 'one'} sparse {gr.sparse_pattern} over "
                 f"the area, {gr.sparse_n_lines} lines ({2 * h / (gr.sparse_n_lines - 1):.3g} V apart); "
                 f"refinement tiles <= {gr.tile_max_V:.3g} V, touching tiles = one region; insides up to "
                 f"{gr.fill_max_size_V:.3g} V wide scanned too")

    # --- edge_explorer ---
    ex = cfg.edge_explorer
    ex.explore_center_V = c
    ex.explore_extent_V = h
    ex.explore_scale_V = smin
    # step sizes relative to the AREA too (2026-11-04: on a 3 um area the 0.5 V circle and
    # steps were too coarse): never more than these fractions of the area's half-width
    ex.initial_path_length_V = EX_INITIAL * h
    ex.calibration_radius_V = min(ex.calibration_radius_V or 0.0, EX_CAL_RADIUS * h) or None
    ex.segment_length_V = min(ex.segment_length_V, EX_SEGMENT * h, 2 * smin)
    if ex.explore_lookahead_V is not None:
        ex.explore_lookahead_V = min(ex.explore_lookahead_V, 2 * ex.segment_length_V)
    ex.follow_segment_length_V = FOLLOW_MIN * smin
    ex.follow_segment_min_V = FOLLOW_MIN * smin
    ex.follow_segment_max_V = min(max(FOLLOW_MAX * smax, FOLLOW_MIN * smin), max(EX_SEGMENT * h, FOLLOW_MIN * smin))
    ex.interior_line_spacing_V = INTERIOR_SPACING * smin   # dense: the top measured, internal edges crossed
    ex.interior_margin_V = 0.25 * smin
    notes.append(f"edge_explorer: explores the area from its center, sweeps ~{smin:.3g} V apart in "
                 f"{ex.segment_length_V:.3g} V steps (calibration circle r {ex.calibration_radius_V or 0:.3g} V, first "
                 f"path {ex.initial_path_length_V:.3g} V); follows each edge with {ex.follow_segment_min_V:.3g}.."
                 f"{ex.follow_segment_max_V:.3g} V zigzag steps, then scans its inside with lines "
                 f"{ex.interior_line_spacing_V:.3g} V apart")

    # --- live view: always the whole area to explore (clipped to the safety limit) ---
    b = p.bounds
    cfg.live_view.fixed_bounds_V = (max(b[0], -lim), min(b[1], lim), max(b[2], -lim), min(b[3], lim))
    cfg.live_view.uncertainty_scale_V = smin   # 100% uncertain: a gap the smallest feature fits in

    # --- gp_path ---
    gp = cfg.gp_path
    gp.area_center_V = c
    gp.area_half_width_V = h
    gp.length_scale_V = smin
    lo = getattr(gp, "length_scale_min_frac", 0.25) * smin
    gp.length_scale_bounds_V = (lo, 2 * smax)
    if gp.stop_check_extent_V is not None:
        gp.stop_check_extent_V = h
    notes.append(f"gp_path: stays inside the area, GP length scale {smin:.3g} V "
                 f"(fit range {lo:.3g}..{2 * smax:.3g} V)")
    return notes


def as_dict(p: ScanPriors) -> dict:
    """For saving with a run."""
    return asdict(p)
