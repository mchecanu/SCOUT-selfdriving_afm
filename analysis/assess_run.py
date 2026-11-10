"""Quick assessment of one hybrid run on the 256-line ground truth: path (viridis), the planner's
features (red = closed, orange = open), TV map of the run and of its raster, numbers.

    python assess_run.py RUN.npz GT.npz OUT.png
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


def main(run, gt, outp):
    M = H.load(run, raster=True); b = M["area"]; R = M["raster"]
    D = np.load(gt)
    gx, gy, gz = (D[k].astype(float) for k in ("xs", "ys", "zs"))
    REF = F.tv(F.binned(gx, gy, H.level_own(gx, gy, gz, b), b), 1500)
    obj = REF > F.OBJ_NM
    lab, n = ndimage.label(obj); sz = ndimage.sum(obj, lab, range(1, n + 1)); keep = np.flatnonzero(sz >= 50) + 1
    lab = np.where(np.isin(lab, keep), np.searchsorted(keep, lab) + 1, 0); n = len(keep)
    res = {}
    for tag, S in (("run", M), ("raster", R)):
        k = F.pix(S["x"], S["y"], b); c = F.cum_pixels(k)
        rec = F.tv(F.binned(S["x"], S["y"], S["z"], b), 1500)
        disc = H.discovery(S["x"], S["y"], S["z"], S["t"], lab, b, n)
        res[tag] = dict(pixels=int(c[-1]), scan_s=float(S["t"][-1]), rmse=F.rmse(rec, REF),
                        on_object=float(np.mean(obj.ravel()[np.unique(k[k >= 0])])),
                        found=int(np.isfinite(disc).sum()), rec=rec)
    feats = M["features"] or []
    closed = [f for f in feats if f["closed"]]
    X0, Y0 = F.to_um(M, np.array(b[:2]), np.array(b[2:]))
    ext = [X0[0], X0[1], Y0[0], Y0[1]]; xl = (min(ext[:2]), max(ext[:2])); yl = (min(ext[2:]), max(ext[2:]))
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.6), layout="constrained")
    ax[0].imshow(REF, origin="lower", extent=ext, cmap="gray", vmin=-15, vmax=220)
    ux, uy = F.to_um(M, M["x"][::20], M["y"][::20]); F.vpath(ax[0], ux, uy, len(ux))
    for f in feats:
        O = np.asarray(f["outline"], float).reshape(-1, 2)
        if len(O) >= 2:
            ox, oy = F.to_um(M, O[:, 0], O[:, 1])
            if f["closed"]:
                ox, oy = np.r_[ox, ox[:1]], np.r_[oy, oy[:1]]
            ax[0].plot(ox, oy, "-", color="r" if f["closed"] else "orange", lw=1.6)
    r = res["run"]; q = res["raster"]
    ax[0].set_title(f"{M['name']}\n{len(closed)} closed of {len(feats)} features; {r['found']}/{n} objects found; "
                    f"{r['scan_s']:.1f} s scanning, wall {M['wall']:.0f} s", fontsize=10)
    ax[1].imshow(r["rec"], origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135)
    ax[1].set_title(f"run, TV: {r['rmse']:.1f} nm; {r['pixels']:,} px, {r['on_object']:.1%} on objects", fontsize=10)
    ax[2].imshow(q["rec"], origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135)
    ax[2].set_title(f"its raster ({R['lines']} lines), TV: {q['rmse']:.1f} nm; {q['pixels']:,} px, {q['on_object']:.1%} on objects", fontsize=10)
    for a in ax:
        a.contour(np.linspace(*ext[:2], F.N), np.linspace(*ext[2:], F.N), REF, [F.OBJ_NM], colors="c", linewidths=0.5)
        a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.set_xlabel("X (um)")
    fig.savefig(outp, dpi=90)
    for t in res.values():
        t.pop("rec")
    res["features"] = [{k: f[k] for k in ("id", "closed", "closed_by", "crossings")} for f in feats]
    res["wall_s"] = M["wall"]
    print(json.dumps(res, indent=1))
    pathlib.Path(outp).with_suffix(".json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
