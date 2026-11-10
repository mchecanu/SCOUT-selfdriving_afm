"""Exploration / exploitation sweep (ivr_lambda) -- summary figure.

    python sweep_fig.py --runs A.npz B.npz ... --gt GROUND_TRUTH.npz --metrics DIR --out DIR

For each run: its path at the measured positions (viridis, start -> end) on the ground truth,
its TV map; then error vs scanning time for all runs + the raster of the first one, and the
final numbers vs lambda (fig4_gp.py must have been run on each with --dense GT, --out DIR).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig4_gp as F  # noqa: E402

plt = F.plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--metrics", default=str(HERE / "fig4"))
    ap.add_argument("--out", default=str(HERE / "sweep"))
    ap.add_argument("--name", default="lambda_sweep")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    met = pathlib.Path(a.metrics)
    runs = []
    for p in a.runs:
        d = np.load(p, allow_pickle=True)
        lam = float(d["ivr_lambda"]) if "ivr_lambda" in d.files else np.nan
        stem = pathlib.Path(p).stem
        m = json.loads((met / f"{stem}_metrics.json").read_text())
        c = np.load(met / f"{stem}_curves.npz", allow_pickle=True)
        runs.append(dict(path=p, lam=lam, m=m, c=c, R=F.load_run(p)))
    runs.sort(key=lambda r: r["lam"])
    REF = runs[0]["c"]["REF"]; ext = list(runs[0]["c"]["ext"])
    xl = (min(ext[:2]), max(ext[:2])); yl = (min(ext[2:]), max(ext[2:]))
    n = len(runs)
    fig = plt.figure(figsize=(3.9 * n, 12.5), layout="constrained")
    gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 1.05])
    for j, r in enumerate(runs):
        R = r["R"]
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(REF, origin="lower", extent=ext, cmap="gray", vmin=-15, vmax=220)
        ux, uy = F.to_um(R, R["x"][::40], R["y"][::40]); F.vpath(ax, ux, uy, len(ux), lw=1.4)
        ax.set_title(f"lambda = {r['lam']:g}\n{r['m']['gp']['pixels']:,} px, {r['m']['gp']['on_object_frac']:.0%} on objects", fontsize=10)
        ax2 = fig.add_subplot(gs[1, j])
        ax2.imshow(r["c"]["recG"], origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135)
        ax2.contour(np.linspace(*ext[:2], REF.shape[1]), np.linspace(*ext[2:], REF.shape[0]), REF, [F.OBJ_NM], colors="w", linewidths=0.4)
        ax2.set_title(f"TV map: {r['m']['gp']['rmse_final_nm']:.1f} nm", fontsize=10)
        for A in (ax, ax2):
            A.set_xlim(*xl); A.set_ylim(*yl); A.set_aspect("equal"); A.tick_params(labelsize=7)
    cmap = plt.get_cmap("plasma")
    axE = fig.add_subplot(gs[2, : max(1, n // 2 + 1)])
    for r in runs:
        col = cmap(0.1 + 0.8 * r["lam"]) if np.isfinite(r["lam"]) else "k"
        axE.plot(r["c"]["times"], r["c"]["errG"], "o-", color=col, ms=3, label=f"GP, lambda {r['lam']:g}")
    r0 = runs[0]
    axE.plot(r0["c"]["times"], r0["c"]["errR"], "s--", color="0.4", ms=3, label="raster (same pixels)")
    axE.set_xlabel("scanning time (s)"); axE.set_ylabel("RMSE vs ground truth (nm)"); axE.legend(fontsize=8)
    axS = fig.add_subplot(gs[2, max(1, n // 2 + 1):])
    lams = np.array([r["lam"] for r in runs])
    axS.plot(lams, [r["m"]["gp"]["rmse_final_nm"] for r in runs], "o-", color="C0", label="final RMSE (nm)")
    axS.plot(lams, [r["m"]["raster"]["rmse_final_nm"] for r in runs], "s--", color="0.4", label="its raster (nm)")
    axS.set_xlabel("ivr_lambda (0 = explore, 1 = edges)"); axS.set_ylabel("RMSE (nm)")
    axT = axS.twinx()
    axT.plot(lams, [100 * r["m"]["gp"]["on_object_frac"] for r in runs], "^-", color="C2", label="pixels on objects (%)")
    axT.set_ylabel("pixels on objects (%)", color="C2")
    axS.legend(loc="upper left", fontsize=8); axT.legend(loc="upper right", fontsize=8)
    fig.suptitle("Exploration vs exploitation: the IVR weight lambda (pure_gp_smooth, 240 V each), on the 256-line ground truth", fontsize=12)
    for e in ("png", "pdf"):
        fig.savefig(out / f"{a.name}.{e}", dpi=110)
    rows = [dict(lam=r["lam"], **{k: r["m"]["gp"][k] for k in ("pixels", "on_object_frac", "rmse_final_nm", "waveforms")},
                 raster_rmse_nm=r["m"]["raster"]["rmse_final_nm"], enrichment=r["m"]["enrichment"]) for r in runs]
    (out / f"{a.name}.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(row)
    print("figure:", out / f"{a.name}.png")


if __name__ == "__main__":
    main()
