"""
Runnable demo: both planners, in simulation, against the same synthetic sample
(a cylinder + a square) your original notebooks used. No hardware needed.

Run with:  python demo_simulation.py

This is meant as (1) a sanity check that the package works in your environment,
and (2) a template for how to call things -- copy/adapt the two run_* functions
below into a real acquisition notebook once you're ready to swap in
backends.PyScannerBackend.
"""

import warnings
warnings.filterwarnings("ignore")  # sklearn GP kernel-bound convergence warnings are benign here

import time
import matplotlib.pyplot as plt

from selfdriving_afm.config import ExperimentConfig
from selfdriving_afm.execution import ContinuousExecutor
from selfdriving_afm.backends.simulate_backend import SimulateBackend, default_topography
from selfdriving_afm.planners.edge_follow_planner import EdgeFollowPlanner
from selfdriving_afm.planners.gp_path_planner import GPPathPlanner
from selfdriving_afm.viz import plot_edge_follow_result, plot_gp_path_result


def run_edge_follow_demo():
    print("\n=== Edge-follow planner (spiral -> circular chain) ===")
    cfg = ExperimentConfig()

    # cfg.velocity's package defaults are realistic-hardware placeholders (slow tip,
    # fast DAQ -- see VelocityConfig) which at this synthetic scene's small volt-scale
    # features implies tens of millions of points; that's a fine and correct thing for
    # a real, patient acquisition but way more than a quick sanity-check demo needs.
    # Bump the demo's tip speed way up purely for fast iteration here -- set this from
    # your real instrument's numbers once you're pointing at hardware (see the notebook).
    cfg.velocity.tip_velocity_V_s = 5.0

    # --- tune here ---
    cfg.edge_follow.circle.radius_mode = "adaptive"          # or "fixed"
    cfg.edge_follow.circle.fixed_radius_V = 0.025             # used if radius_mode == "fixed"
    cfg.edge_follow.circle.adaptive_min_radius_V = 0.012       # used if radius_mode == "adaptive"
    cfg.edge_follow.circle.adaptive_max_radius_V = 0.028
    cfg.edge_follow.circle.adaptive_curvature_gain = 2.0
    cfg.edge_follow.circle.overlap_fraction = 0.6              # user-set, per your spec
    cfg.edge_follow.edge_detect.mode = "auto_mad"               # or "manual" + manual_threshold
    cfg.edge_follow.edge_detect.mad_k = 6.0
    cfg.edge_follow.initial_search_radius_V = 0.04
    cfg.edge_follow.spiral_growth_factor = 1.5
    cfg.edge_follow.max_search_radius_V = 1.0
    cfg.edge_follow.max_features = 2
    # -----------------

    backend = SimulateBackend(noise_std=0.001, seed=1)
    executor = ContinuousExecutor(backend, cfg.safety, cfg.velocity,
                                   start_xy=cfg.execution.start_xy_V)
    planner = EdgeFollowPlanner(executor, cfg.edge_follow)

    t0 = time.time()
    # Every scan starts at (0, 0) in bias volts, per cfg.execution.start_xy_V.
    features = planner.run(start_xy=cfg.execution.start_xy_V)
    print(f"  {len(features)} feature(s) found in {time.time() - t0:.2f}s, "
          f"{len(executor.all_points())} points measured")
    for i, f in enumerate(features):
        print(f"  feature {i}: {len(f.circle_steps)} circle steps, "
              f"closed_loop={f.closed_loop}, edge_lost={f.edge_lost}")

    plot_edge_follow_result(features, executor, ground_truth_fn=default_topography, extent=0.45)
    plt.savefig("edge_follow_demo.png", dpi=130)
    print("  saved edge_follow_demo.png")


def run_gp_path_demo():
    print("\n=== GP path planner (Bayesian continuous path) ===")
    cfg = ExperimentConfig()

    # --- tune here ---
    cfg.gp_path.max_segments = 25          # hard cap -- keep modest, exact GP fit is O(n^3) in
                                             # total points, see README "Known limitations"
    cfg.gp_path.refit_every = 2
    cfg.gp_path.exploration_weight = 0.5
    cfg.gp_path.gp_n_restarts_optimizer = 0
    # -----------------

    backend = SimulateBackend(noise_std=0.001, seed=2)
    executor = ContinuousExecutor(backend, cfg.safety, cfg.velocity,
                                   start_xy=cfg.execution.start_xy_V)
    planner = GPPathPlanner(executor, cfg.gp_path)

    t0 = time.time()
    segments = planner.run(start_xy=cfg.execution.start_xy_V)
    n_pts = sum(len(s.x) for s in segments)
    print(f"  {len(segments)} segments, {n_pts} points measured, {time.time() - t0:.2f}s")

    plot_gp_path_result(segments, planner, ground_truth_fn=default_topography, extent=0.32)
    plt.savefig("gp_path_demo.png", dpi=130)
    print("  saved gp_path_demo.png")


if __name__ == "__main__":
    run_edge_follow_demo()
    run_gp_path_demo()
    print("\nDone. Both planners ran end-to-end against the simulation backend.")
