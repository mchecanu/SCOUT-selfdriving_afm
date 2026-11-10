"""All of today's self-driving runs vs their own equivalent rasters, on the 256-line ground truth:
time to find all 5 objects, pixels on objects, final TV error, wall time.

    python discovery_summary.py   -> final/discovery_summary.png, .pdf, .json
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fig4_gp as F  # noqa: E402
import hybrid_fig as H  # noqa: E402
from scipy import ndimage  # noqa: E402

plt = F.plt
D = HERE.parent / "scan_data"
RUNS = [  # (label, file, kind)
    ("GP lambda 0", "scan_rerun_lam000_pure_gp_smooth_240V_gp_path_20261109_122204.npz", "gp"),
    ("GP lambda 0.25", "scan_rerun_lam025_pure_gp_smooth_240V_gp_path_20261109_123005.npz", "gp"),
    ("GP lambda 0.5", "scan_sweep_lam050_pure_gp_smooth_240V_gp_path_20261109_113659.npz", "gp"),
    ("GP lambda 0.75", "scan_sweep_lam075_pure_gp_smooth_240V_gp_path_20261109_114446.npz", "gp"),
    ("GP lambda 1", "scan_sweep_lam100_pure_gp_smooth_240V_gp_path_20261109_115212.npz", "gp"),
    ("GP lambda 1, gamma 2", "scan_edgefocus_gamma2_lam100_pure_gp_smooth_240V_gp_path_20261109_143003.npz", "gp"),
    ("GP edges preset (hero)", "scan_hero_pure_gp_edges_smooth_240V_gp_path_20261109_103432.npz", "gp"),
    ("hybrid: GP + edge tracker", "scan_hybrid5fig3_lam000_gpL3_edgeL2_gp_edge_explorer_20261109_132517.npz", "edge"),
    ("contour following (L2)", "scan_L2_contour_edge_explorer_20261109_124234.npz", "edge"),
]


def main():
    out = HERE / "final"; out.mkdir(exist_ok=True)
    if "--replot" in sys.argv:
        return plot(json.loads((out / "discovery_summary.json").read_text()), out)
    gt = np.load(D / "ground_truth_256lines_20261109_105402.npz")
    first = H.load(D / RUNS[0][1])
    b = first["area"]
    gx, gy, gz = (gt[k].astype(float) for k in ("xs", "ys", "zs"))
    REF = F.tv(F.binned(gx, gy, H.level_own(gx, gy, gz, b), b), 1500)
    obj = REF > F.OBJ_NM
    lab, n = ndimage.label(obj); sz = ndimage.sum(obj, lab, range(1, n + 1)); keep = np.flatnonzero(sz >= 50) + 1
    lab = np.where(np.isin(lab, keep), np.searchsorted(keep, lab) + 1, 0); n = len(keep)
    rows = []
    for name, fn, kind in RUNS:
        M = H.load(D / fn, area=b, raster=True); R = M["raster"]
        res = dict(name=name, kind=kind, file=fn, wall_s=M["wall"], raster_wall_s=R["wall"], raster_lines=R["lines"])
        for tag, S in (("run", M), ("raster", R)):
            k = F.pix(S["x"], S["y"], b); c = F.cum_pixels(k)
            rec = F.tv(F.binned(S["x"], S["y"], S["z"], b), 1500)
            disc = H.discovery(S["x"], S["y"], S["z"], S["t"], lab, b, n)
            res[tag] = dict(pixels=int(c[-1]), scan_s=float(S["t"][-1]), rmse_nm=F.rmse(rec, REF),
                            on_object=float(np.mean(obj.ravel()[np.unique(k[k >= 0])])),
                            all_found_s=float(disc[-1]) if np.all(np.isfinite(disc)) else None,
                            first_found_s=float(disc[0]) if np.isfinite(disc[0]) else None)
        rows.append(res)
        r, q = res["run"], res["raster"]
        print(f"{name:28s} run: {r['pixels']:5,} px {r['scan_s']:5.1f} s, all 5 at {r['all_found_s']}, on obj {r['on_object']:5.1%}, "
              f"RMSE {r['rmse_nm']:5.1f} | raster: {q['pixels']:5,} px, all 5 at {q['all_found_s']}, on obj {q['on_object']:5.1%}, RMSE {q['rmse_nm']:5.1f}")
    (out / "discovery_summary.json").write_text(json.dumps(rows, indent=1))
    plot(rows, out)


def plot(rows, out):
    names = [r["name"] for r in rows]; y = np.arange(len(rows))
    fig, axs = plt.subplots(1, 4, figsize=(21, 6.2), layout="constrained", sharey=True)
    col = ["C0" if r["kind"] == "gp" else "C2" for r in rows]
    panels = [("time to find all 5 objects (s of scanning)", lambda s: s["all_found_s"] or np.nan),
              ("pixels on objects (%)", lambda s: 100 * s["on_object"]),
              ("TV map error vs ground truth (nm)", lambda s: s["rmse_nm"]),
              ("scanning time (s)", lambda s: s["scan_s"])]
    for ax, (t, fn) in zip(axs, panels):
        ax.barh(y + 0.2, [fn(r["run"]) for r in rows], 0.38, color=col, label="self-driving")
        ax.barh(y - 0.2, [fn(r["raster"]) for r in rows], 0.38, color="0.65", label="its raster (same pixels)")
        ax.set_title(t, fontsize=11); ax.grid(axis="x", alpha=0.3)
    axs[0].set_yticks(y); axs[0].set_yticklabels(names); axs[0].invert_yaxis(); axs[1].legend(fontsize=9, loc="lower right")
    for i, r in enumerate(rows):
        axs[3].text(r["run"]["scan_s"], i + 0.2, f"  wall {r['wall_s']:.0f} s", va="center", fontsize=8)
        axs[3].text(r["raster"]["scan_s"], i - 0.2, f"  wall {r['raster_wall_s']:.0f} s", va="center", fontsize=8, color="0.35")
    fig.suptitle("All self-driving runs of 2026-11-09 on the shapes sample vs their own equivalent rasters "
                 "(256-line ground truth; blue = GP path planning, green = edge tracking)", fontsize=12)
    for e in ("png", "pdf"):
        fig.savefig(out / f"discovery_summary.{e}", dpi=100)
    print("figure:", out / "discovery_summary.png")


if __name__ == "__main__":
    main()
