"""
Named gp_path strategies, tested in simulation (2026-11-06/07, README "GP presets").

    from selfdriving_afm.gp_presets import apply_gp_preset
    apply_gp_preset(cfg, "pure_gp_smooth")      # then override any field you like

Every preset sets the SAME list of strategy fields, so switching presets never leaves
a setting behind from the previous one. Hardware-independent: the same names work in
the Run notebook (instrument) and Simulation.ipynb.

  original             the planner's first scoring (mean |d mu| + exploration x sigma) on
                       the fast sparse GP -- in practice pure exploration
  hybrid               GP edge-weighted exploration + rule-based edge tracing
  pure_gp              integrated variance reduction (IVR), ivr_lambda 0.8, 2 V lookahead,
                       straight segments (sharp turns between segments)
  pure_gp_smooth       pure_gp, each segment starting tangent to the previous one
                       (Hermite curves, max turn 135 deg) -- best map at 120 V in simulation
  pure_gp_1V           pure_gp with 1 V segments (half the waveforms, coarser map)
  pure_gp_smooth_1V    pure_gp_1V with smooth turns
  pure_gp_edges        pure_gp with ivr_lambda 1.0 (all uncertainty reduction at edges)
  pure_gp_edges_smooth pure_gp_edges with smooth turns
  pure_gp_edge_focus   pure_gp_edges_smooth with ivr_interest_power (gamma) 2: weight
                       (1 - lambda) + lambda * interest**gamma -- Fig. 4's main run
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

# defaults of every strategy field a preset controls (= GPPathConfig's own defaults
# except the model settings shared by all the tuned presets)
_BASE = dict(
    acquisition="legacy", exploration_weight=0.5, n_candidate_directions=8,
    path_segment_length_V=0.5, initial_path_length_V=0.5, lookahead_V=None,
    # model
    gp_backend="sparse", refit_every=1, gp_kernel="rbf", gp_detrend=False,
    length_scale_min_frac=0.25, learn_length_scale=True,
    # pure GP
    ivr_lambda=0.8, ivr_interest_power=1.0,
    # hybrid
    explore_measure="sigma", explore_radius_V=None, trace_edges=False, trace_step_V=0.2,
    trace_correction_deg=25.0, trace_turn_deg=45.0, edge_weight=1.0, straight_bonus=0.0,
    min_segment_length_V=None,
    # motion
    smooth_turns=False, max_turn_deg=None,
)
_TUNED_MODEL = dict(gp_kernel="matern52", gp_detrend=True, length_scale_min_frac=0.5, n_candidate_directions=16)
_PURE = dict(_TUNED_MODEL, acquisition="ivr", ivr_lambda=0.8, lookahead_V=2.0)
_SMOOTH = dict(smooth_turns=True, max_turn_deg=135)

GP_PRESETS: Dict[str, dict] = {
    "original": dict(),
    "hybrid": dict(_TUNED_MODEL, acquisition="edge_uncertainty", exploration_weight=1.5,
                   explore_measure="distance", explore_radius_V=0.7, lookahead_V=1.5,
                   trace_edges=True, trace_correction_deg=15.0, trace_turn_deg=90.0),
    "pure_gp": dict(_PURE),
    "pure_gp_smooth": dict(_PURE, **_SMOOTH),
    "pure_gp_1V": dict(_PURE, path_segment_length_V=1.0, initial_path_length_V=1.0, lookahead_V=None),
    "pure_gp_smooth_1V": dict(_PURE, path_segment_length_V=1.0, initial_path_length_V=1.0, lookahead_V=None, **_SMOOTH),
    "pure_gp_edges": dict(_PURE, ivr_lambda=1.0),
    "pure_gp_edges_smooth": dict(_PURE, ivr_lambda=1.0, **_SMOOTH),
    # 2026-11-09: lambda 1, gamma 2 -- the most edge-focused run that still finds every object
    # (16 % of pixels on objects on the instrument, 2.2x its raster; Fig. 4)
    "pure_gp_edge_focus": dict(_PURE, ivr_lambda=1.0, ivr_interest_power=2.0, **_SMOOTH),
}


def preset_fields(name: str) -> dict:
    """All strategy fields of a preset (base defaults + its changes)."""
    if name not in GP_PRESETS:
        raise ValueError(f"Unknown GP preset {name!r}; choose one of {list(GP_PRESETS)}")
    return dict(deepcopy(_BASE), **GP_PRESETS[name])


def apply_gp_preset(cfg, name: str):
    """Set cfg.gp_path to the named strategy (in place); returns cfg."""
    for k, v in preset_fields(name).items():
        if not hasattr(cfg.gp_path, k):
            raise AttributeError(f"GPPathConfig has no field {k!r} (preset {name!r})")
        setattr(cfg.gp_path, k, v)
    return cfg


def as_overrides(name: str) -> dict:
    """The preset as Simulation.ipynb / sim.compare overrides: {"gp_path.field": value}."""
    return {f"gp_path.{k}": v for k, v in preset_fields(name).items()}


def variants(names=None):
    """sim.compare variants for several presets: [("gp_path: <name>", "gp_path", overrides)]."""
    return [(f"gp_path: {n}", "gp_path", as_overrides(n)) for n in (names or list(GP_PRESETS))]
