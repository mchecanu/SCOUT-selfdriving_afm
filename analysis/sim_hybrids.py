"""Simulation check of the two new hybrids on today's five objects (same area / priors / tracer
settings as the instrument runs).

    python sim_hybrids.py grid_tracker|gp_spiral|gp_edge_explorer   -> sim_hybrids/MODE.png, .json
"""
import copy
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
OUT = HERE / "sim_hybrids"
import sim_edge_focus as S  # noqa: E402

NM_PER_V = 1875.75


def build_cfg(mode):
    from selfdriving_afm import modes, priors
    from selfdriving_afm.gp_presets import apply_gp_preset
    cfg, _, _ = modes.load_run_config("Run_GP_Paper.ipynb")
    cfg.position_sensors.enabled = False; cfg.velocity.sample_rate_hz = 1e5
    cfg.z_signal.route = False; cfg.z_signal.units = "V"
    apply_gp_preset(cfg, "pure_gp_smooth"); cfg.gp_path.ivr_lambda = 0.0
    cfg.live_view.reconstruction = "tv"; cfg.live_view.tv_mu = 0.3
    P = priors.ScanPriors(feature_min_V=3000 / NM_PER_V, feature_max_V=5000 / NM_PER_V,
                          area_half_width_V=S.AREA_H, area_center_V=S.AREA_C)
    EE0 = copy.deepcopy(cfg.edge_explorer)
    # Fig. 3 tracer settings: what priors of 1-4 um give
    tmp = copy.deepcopy(cfg); tmp.edge_explorer = copy.deepcopy(EE0)
    priors.apply(tmp, priors.ScanPriors(feature_min_V=1000 / NM_PER_V, feature_max_V=4000 / NM_PER_V,
                                        area_half_width_V=S.AREA_H, area_center_V=S.AREA_C))
    fig3 = {k: getattr(tmp.edge_explorer, k) for k in ("follow_segment_length_V", "follow_segment_min_V", "follow_segment_max_V",
                                                        "interior_line_spacing_V", "interior_margin_V", "loop_closure_tolerance_V")}
    priors.apply(cfg, P)
    ee = cfg.edge_explorer
    for k, v in fig3.items():
        setattr(ee, k, v)
    ee.max_features = 5
    ee.edge_detect.mode = "manual"; ee.edge_detect.manual_threshold = 58.0
    if ee.follow_segment_length_V is None:
        ee.follow_segment_length_V = 0.5 * ee.segment_length_V
    if ee.explore_scale_V is None:
        ee.explore_scale_V = ee.segment_length_V
    if mode in ("gp_edge_explorer", "gp_spiral"):
        ee.segment_length_V = 1.0
    ee.stop_after_quiet_sweeps = 3.0; ee.max_segments = 2000; ee.max_run_time_s = 1500.0
    return cfg, P


def run(mode):
    os.chdir(ROOT)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from selfdriving_afm import modes, topography
    from selfdriving_afm.backends.simulate_backend import SimulateBackend
    from selfdriving_afm.reconstruction import flattened_reconstruction
    OUT.mkdir(exist_ok=True)
    smp = topography.sample_from_text(S.sample_text())
    cfg, P = build_cfg(mode)
    noisy = os.environ.get("NOISY") == "1"
    BOW = float(os.environ.get("BOW", "0"))   # nm / V^2 piezo bow (instrument-like: 1.5 -> ~75 nm at the area edges)
    if noisy:   # instrument-like: 8 nm noise, stronger tilt, a broad 40 nm background bump at the left edge
        bg = lambda x, y: 0.02 * x - 0.01 * y + 40.0 * np.exp(-((x + 4.5) ** 2 + (y + 2.0) ** 2) / 8.0) + BOW * ((x - 1.9) ** 2 + (y - 0.5) ** 2)
        be = SimulateBackend(lambda x, y: smp(x, y) + bg(x, y), noise_std=8.0, seed=2)
    else:
        be = SimulateBackend(lambda x, y: smp(x, y) + 0.004 * x - 0.002 * y, noise_std=1.0, seed=1)
    s = modes.build(mode, cfg, be, live=False, priors=P)
    t0 = time.time(); modes.run(s); dt = time.time() - t0
    print(s.summary)
    pts = s.executor.all_points()
    b = (S.AREA_C[0] - S.AREA_H, S.AREA_C[0] + S.AREA_H, S.AREA_C[1] - S.AREA_H, S.AREA_C[1] + S.AREA_H)
    n = 200
    X, Y = np.meshgrid(np.linspace(b[0], b[1], n), np.linspace(b[2], b[3], n)); GT = smp(X, Y)
    _, R, _ = flattened_reconstruction(pts[:, :2], pts[:, 2], b, cfg.live_view)
    from selfdriving_afm.reconstruction import pixel_index
    k = np.unique(pixel_index(np.clip(pts[:, 0], b[0], b[1]), np.clip(pts[:, 1], b[2], b[3]), n, b))
    onf = float((GT > 58).ravel()[k].mean())
    pl = s.planner
    feats = [dict(id=f.id, closed=bool(f.closed_loop), by=f.closed_by, n=len(f.points),
                  centre=[round(float(v), 2) for v in f.pts().mean(axis=0)]) for f in pl.features]
    res = dict(mode=mode, wall_s=dt, scan_s=len(pts) / cfg.velocity.sample_rate_hz, pixels=int(len(k)),
               on_object=onf, rmse=float(np.sqrt(np.mean((R - GT) ** 2))), closed=sum(f["closed"] for f in feats),
               features=feats, stop=getattr(pl, "stop_reason", None))
    print(json.dumps(res, indent=1))
    tag = mode + ("_noisy" if noisy else "") + ("_bow" if BOW else "")
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), layout="constrained")
    ax[0].imshow(GT, origin="lower", extent=b, cmap="gray")
    p = pts[::20, :2]; seg = np.stack([p[:-1], p[1:]], 1)
    lc = LineCollection(seg, cmap="viridis", linewidths=1.2); lc.set_array(np.linspace(0, 1, len(seg))); ax[0].add_collection(lc)
    for f in pl.features:
        O = f._ordered() if len(f.points) >= 3 else f.pts()
        ax[0].plot(*np.vstack([O, O[:1]]).T, "-", color="r" if f.closed_loop else "orange", lw=1)
    ax[0].set_title(f"{mode}: {res['closed']}/5 closed, {res['scan_s']:.1f} s scanning, {res['pixels']} px, "
                    f"{onf:.0%} on objects")
    ax[1].imshow(R, origin="lower", extent=b, cmap="copper", vmin=-15, vmax=135)
    ax[1].set_title(f"TV map: {res['rmse']:.1f} nm")
    fig.savefig(OUT / f"{tag}.png", dpi=80)


if __name__ == "__main__":
    run(sys.argv[1])
