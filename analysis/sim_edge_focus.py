"""Simulation: making the GP planner more edge-focused WITHOUT penalising exploration.

    python sim_edge_focus.py NAME LAMBDA GAMMA [BUDGET_V]      -> sim_edge_focus/NAME.npz
    python sim_edge_focus.py --figure                          -> sim_edge_focus/edge_focus.png

Sample: today's five objects (disc + 4 hexagons, 116 nm, ~4 um) at their positions in the
instrument's drive frame (from the 256-line ground truth), same area and priors as the runs.
Planner: pure_gp_smooth, ivr_lambda = LAMBDA, ivr_interest_power = GAMMA, 240 V of path.
Scored: TV map of the samples vs the true topography (RMSE, bad pixels > 23 nm = 20% of the
height), pixels on objects, error vs path length.
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = HERE / "sim_edge_focus"

AREA_C, AREA_H = (1.915, 0.538), 6.93
HEX_R = 1.12      # hexagon circumradius (V), ~4.2 um across
OBJ = [("circle", (3.68, 3.62), 2.3), ("hex", (0.58, 2.56), HEX_R), ("hex", (3.25, -0.10), HEX_R),
       ("hex", (5.39, -2.12), HEX_R), ("hex", (-3.86, -2.02), HEX_R)]


def sample_text():
    lines = []
    for kind, (x, y), s in OBJ:
        if kind == "circle":
            lines.append(f"1 circle height=116 size={s} at=({x},{y})")
        else:
            a = np.deg2rad(np.arange(6) * 60 + 30)
            pts = ",".join(f"({s * np.cos(t):.4f},{s * np.sin(t):.4f})" for t in a)
            lines.append(f"1 polygon height=116 at=({x},{y}) points={pts}")
    return "\n".join(lines)


def run(name, lam, gam, budget=240.0):
    os.chdir(ROOT)
    import matplotlib; matplotlib.use("Agg")
    from selfdriving_afm import modes, priors, topography
    from selfdriving_afm.gp_presets import apply_gp_preset
    from selfdriving_afm.backends.simulate_backend import SimulateBackend
    from selfdriving_afm.reconstruction import flattened_reconstruction
    smp = topography.sample_from_text(sample_text())
    cfg, _, _ = modes.load_run_config("Run_GP_Paper.ipynb")
    cfg.position_sensors.enabled = False; cfg.velocity.sample_rate_hz = 1e5
    cfg.z_signal.route = False; cfg.z_signal.units = "V"
    apply_gp_preset(cfg, "pure_gp_smooth")
    cfg.gp_path.ivr_lambda = lam; cfg.gp_path.ivr_interest_power = gam
    cfg.gp_path.max_path_length_V = budget; cfg.gp_path.max_segments = 1500; cfg.gp_path.max_run_time_s = 3000
    cfg.live_view.reconstruction = "tv"; cfg.live_view.tv_mu = 0.3
    P = priors.ScanPriors(feature_min_V=1.6, feature_max_V=2.67, area_half_width_V=AREA_H, area_center_V=AREA_C)
    priors.apply(cfg, P)
    be = SimulateBackend(lambda x, y: smp(x, y) + 0.004 * x - 0.002 * y, noise_std=1.0, seed=1)
    s = modes.build("gp_path", cfg, be, live=False, priors=P)
    t0 = time.time(); modes.run(s); dt = time.time() - t0
    pts = s.executor.all_points()
    b = (AREA_C[0] - AREA_H, AREA_C[0] + AREA_H, AREA_C[1] - AREA_H, AREA_C[1] + AREA_H)
    n = 200
    X, Y = np.meshgrid(np.linspace(b[0], b[1], n), np.linspace(b[2], b[3], n)); GT = smp(X, Y)
    obj = GT > 58
    # error vs path length: TV of the samples measured up to each fraction of the path
    seg = np.cumsum(np.r_[0, np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))])
    fr, err, bad = np.linspace(0.1, 1, 10), [], []
    for f in fr:
        u = int(np.searchsorted(seg, f * seg[-1])) or len(seg)
        _, R, _ = flattened_reconstruction(pts[:u, :2], pts[:u, 2], b, cfg.live_view)
        err.append(float(np.sqrt(np.mean((R - GT) ** 2)))); bad.append(float(np.mean(np.abs(R - GT) > 23)))
    B, R, _ = flattened_reconstruction(pts[:, :2], pts[:, 2], b, cfg.live_view)
    from selfdriving_afm.reconstruction import pixel_index
    k = np.unique(pixel_index(np.clip(pts[:, 0], b[0], b[1]), np.clip(pts[:, 1], b[2], b[3]), n, b))
    onf = float(obj.ravel()[k].mean())
    # objects whose top was measured (any sample > 58 nm inside the object)
    from scipy import ndimage
    lab, nlab = ndimage.label(obj)
    kk = pixel_index(np.clip(pts[:, 0], b[0], b[1]), np.clip(pts[:, 1], b[2], b[3]), n, b)
    hit = lab.ravel()[kk][pts[:, 2] - np.median(pts[:, 2]) > 58]
    found = int(len(np.unique(hit[hit > 0])))
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", x=pts[::20, 0].astype(np.float32), y=pts[::20, 1].astype(np.float32),
                        recon=R.astype(np.float32), gt=GT.astype(np.float32), bounds=np.array(b), frac=fr,
                        err=np.array(err), bad=np.array(bad),
                        meta=np.array(json.dumps(dict(name=name, lam=lam, gamma=gam, budget_V=budget, sim_s=dt,
                                                      segments=len(s.result), on_object_frac=onf, objects_found=found,
                                                      n_objects=nlab, rmse_nm=err[-1], bad_px=bad[-1]))))
    print(f"{name}: lambda {lam} gamma {gam}: RMSE {err[-1]:.1f} nm, bad px {100 * bad[-1]:.1f}%, "
          f"on objects {onf:.1%}, objects found {found}/{nlab}, {dt:.0f} s")


def figure():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(HERE))
    from fig4_gp import vpath
    files = sorted(OUT.glob("*.npz"))
    R = [dict(np.load(f, allow_pickle=True)) for f in files]
    for r in R:
        r["m"] = json.loads(str(r["meta"]))
    R.sort(key=lambda r: (r["m"]["lam"], r["m"]["gamma"]))
    n = len(R)
    fig = plt.figure(figsize=(4.2 * n, 11), layout="constrained")
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.9])
    b = R[0]["bounds"]; ext = [b[0], b[1], b[2], b[3]]
    for j, r in enumerate(R):
        m = r["m"]
        ax = fig.add_subplot(gs[0, j]); ax.imshow(r["gt"], origin="lower", extent=ext, cmap="gray", vmin=-15, vmax=220)
        vpath(ax, r["x"], r["y"], len(r["x"]), lw=1.4)
        ax.set_title(f"lambda {m['lam']:g}, gamma {m['gamma']:g}\n{m['on_object_frac']:.0%} of pixels on objects, "
                     f"{m['objects_found']}/{m['n_objects']} found", fontsize=10)
        ax2 = fig.add_subplot(gs[1, j]); ax2.imshow(r["recon"], origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135)
        ax2.contour(np.linspace(b[0], b[1], r["gt"].shape[1]), np.linspace(b[2], b[3], r["gt"].shape[0]), r["gt"], [58],
                    colors="w", linewidths=0.4)
        ax2.set_title(f"TV map: {m['rmse_nm']:.1f} nm, {100 * m['bad_px']:.1f}% bad px", fontsize=10)
        for a in (ax, ax2):
            a.set_xlim(b[0], b[1]); a.set_ylim(b[2], b[3]); a.set_aspect("equal"); a.tick_params(labelsize=7)
    axE = fig.add_subplot(gs[2, : n // 2 + 1]); axB = fig.add_subplot(gs[2, n // 2 + 1:])
    for r in R:
        lab = f"lambda {r['m']['lam']:g}, gamma {r['m']['gamma']:g}"
        axE.plot(r["frac"] * r["m"]["budget_V"], r["err"], "o-", ms=3, label=lab)
    axE.set_xlabel("path length (V)"); axE.set_ylabel("RMSE vs truth (nm)"); axE.legend(fontsize=8)
    names = [f"l{r['m']['lam']:g}\ng{r['m']['gamma']:g}" for r in R]
    axB.bar(np.arange(n) - 0.2, [100 * r["m"]["on_object_frac"] for r in R], 0.4, label="pixels on objects (%)")
    axB.bar(np.arange(n) + 0.2, [100 * r["m"]["bad_px"] for r in R], 0.4, label="bad pixels (%)")
    axB.set_xticks(np.arange(n)); axB.set_xticklabels(names, fontsize=8); axB.legend(fontsize=8)
    fig.suptitle("Simulation: more edge-focused GP planning without penalising exploration "
                 "(w = (1 - lambda) + lambda * interest^gamma; today's 5 objects, 240 V each)", fontsize=12)
    fig.savefig(OUT / "edge_focus.png", dpi=100)
    print("figure:", OUT / "edge_focus.png")
    for r in R:
        print(r["m"])


if __name__ == "__main__":
    if sys.argv[1] == "--figure":
        figure()
    else:
        run(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 240.0)
