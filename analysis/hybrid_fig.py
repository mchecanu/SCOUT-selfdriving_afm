"""GP exploration + edge tracking (hybrid) vs contour following (L2) vs pure GP (L3) vs raster.

    python hybrid_fig.py --hybrid H.npz --l2 L2.npz --gp GP.npz --gt GT.npz --out DIR [--gif]

Every acquisition at its measured (sensor) positions, levelled with its own substrate plane
(binned, lower-envelope fit: dense scans on the features cannot capture it), TV-reconstructed,
scored against the 256-line ground truth (GT, TV). The raster is the hybrid's equivalent raster.
Panels: each method's path (viridis, start -> end) at three scanning times + its TV map;
error vs scanning time; ground-truth objects discovered vs scanning time; pixels on objects.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fig4_gp as F  # noqa: E402
from scipy import ndimage  # noqa: E402

from selfdriving_afm.config import LiveViewConfig  # noqa: E402
from selfdriving_afm.reconstruction import flattened_reconstruction  # noqa: E402

plt = F.plt
N = F.N
OBJ = F.OBJ_NM


def level_own(x, y, z, b):
    cfg = LiveViewConfig(); cfg.n_image_pixels = N; cfg.reconstruction = "linear"
    _, _, c = flattened_reconstruction(np.column_stack([x, y]), z, b, cfg)
    return z - (c[0] + c[1] * x + c[2] * y)


def load(path, area=None, raster=False):
    d = np.load(path, allow_pickle=True)
    pr = json.loads(str(d["priors_json"])) if "priors_json" in d.files else None
    if area is None:
        c, h = pr["area_center_V"], pr["area_half_width_V"]
        area = (c[0] - h, c[0] + h, c[1] - h, c[1] + h)
    rate = float(d["sample_rate_hz"])
    o = dict(name=pathlib.Path(path).stem, area=area, rate=rate,
             um_per_V=np.asarray(d["um_per_V"], float), um_at_0V=np.asarray(d["um_at_0V"], float),
             wall=float(d["run_wall_time_s"]) if "run_wall_time_s" in d.files else np.nan,
             lam=float(d["ivr_lambda"]) if "ivr_lambda" in d.files else np.nan,
             kinds=d["kinds"], features=json.loads(str(d["features_json"])) if "features_json" in d.files else None)
    STRIDE = 5                                   # RAM: 1 sample in 5 (0.15 nm of path at 30 V/s, 1 MS/s)
    if "pts" in d.files:                         # lite / Zenodo copy: already 1 sample in pts_stride
        p, st = np.array(d["pts"], float), int(d["pts_stride"])
    else:
        p, st = np.array(d["points"][::STRIDE]), STRIDE
    import gc; gc.collect()
    o.update(x=p[:, 0], y=p[:, 1], z=level_own(p[:, 0], p[:, 1], p[:, 2], area), t=np.arange(len(p)) * st / rate)
    del p
    if raster and "raster_xs" in d.files:
        rx, ry, rz = (d[k].astype(float) for k in ("raster_xs", "raster_ys", "raster_zs"))
        dt = float(d["raster_t"][1] - d["raster_t"][0]) if "raster_t" in d.files else float(d["raster_dt"])
        o["raster"] = dict(x=rx, y=ry, z=level_own(rx, ry, rz, area), t=np.arange(len(rx)) * dt,
                           lines=int(d["raster_n_lines"]), wall=float(d["raster_wall_time_s"]))
    return o


def discovery(x, y, z, t, labels, b, n_obj):
    """Scanning time at which each GT object is first measured on top (z > OBJ inside its mask)."""
    k = F.pix(x, y, b)
    lab = np.where(k >= 0, labels.ravel()[np.maximum(k, 0)], 0)
    hit = (lab > 0) & (z > OBJ)
    first = np.full(n_obj, np.nan)
    for i in range(1, n_obj + 1):
        w = np.flatnonzero(hit & (lab == i))
        if len(w):
            first[i - 1] = t[w[0]]
    return np.sort(first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", required=True); ap.add_argument("--l2", required=True)
    ap.add_argument("--gp", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default=str(HERE / "hybrid")); ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    H = load(a.hybrid, raster=True)
    b = H["area"]
    L2 = load(a.l2, area=b); G = load(a.gp, area=b)
    D = np.load(a.gt, allow_pickle=True)
    gx, gy, gz = (D[k].astype(float) for k in ("xs", "ys", "zs"))
    REF = F.tv(F.binned(gx, gy, level_own(gx, gy, gz, b), b), 1500)
    obj = REF > OBJ
    labels, n = ndimage.label(obj)
    sizes = ndimage.sum(obj, labels, range(1, n + 1))
    keep = np.flatnonzero(sizes >= 50) + 1                       # drop specks (the small particle)
    labels = np.where(np.isin(labels, keep), np.searchsorted(keep, labels) + 1, 0)
    n_obj = len(keep)
    print(f"ground truth: {n_obj} objects ({obj.mean():.1%} of the area)")
    R = dict(H["raster"], name="raster (same pixels as the hybrid)")
    methods = [("GP exploration + edge tracking (hybrid)", H, "C0"), ("contour following (L2)", L2, "C2"),
               (f"pure GP path (L3, lambda {G['lam']:g})" + (" gamma 2" if "gamma2" in G["name"] else ""), G, "C1"), (f"raster ({R['lines']} lines)", R, "C3")]
    res = {}
    T_end = max(m[1]["t"][-1] for m in methods)
    times = np.linspace(T_end / 16, T_end, 16)
    for name, M, _ in methods:
        k = F.pix(M["x"], M["y"], b); c = F.cum_pixels(k)
        errs = []
        for tt in times:
            u = int(np.searchsorted(M["t"], tt))
            errs.append(F.rmse(F.tv(F.binned(M["x"][:u], M["y"][:u], M["z"][:u], b)), REF) if u > 10 else np.nan)
        rec = F.tv(F.binned(M["x"], M["y"], M["z"], b), 1500)
        onf = float(np.mean(obj.ravel()[np.unique(k[k >= 0])]))
        res[name] = dict(M=M, err=errs, rec=rec, disc=discovery(M["x"], M["y"], M["z"], M["t"], labels, b, n_obj),
                         pixels=int(c[-1]), on_obj=onf, scan_s=float(M["t"][-1]), rmse=F.rmse(rec, REF))
        print(f"{name:45s} {c[-1]:6,} px  {M['t'][-1]:6.1f} s scan  wall {M.get('wall', np.nan):6.0f} s  on objects {onf:5.1%}  "
              f"RMSE {res[name]['rmse']:5.1f} nm  objects found at (s): {np.round(res[name]['disc'], 1).tolist()}")

    X0, Y0 = F.to_um(H, np.array(b[:2]), np.array(b[2:]))
    ext = [X0[0], X0[1], Y0[0], Y0[1]]; xl = (min(ext[:2]), max(ext[:2])); yl = (min(ext[2:]), max(ext[2:]))
    fig = plt.figure(figsize=(19, 4.3 * len(methods) + 4.8), layout="constrained")
    gs = fig.add_gridspec(len(methods) + 1, 4, height_ratios=[1] * len(methods) + [1.1])
    for i, (name, M, col) in enumerate(methods):
        r = res[name]
        for j, f in enumerate((1 / 3, 2 / 3, 1.0)):
            tt = f * M["t"][-1]; u = int(np.searchsorted(M["t"], tt)) or len(M["t"])
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(REF, origin="lower", extent=ext, cmap="gray", vmin=-15, vmax=220)
            ux, uy = F.to_um(H, M["x"][:u:40], M["y"][:u:40]); F.vpath(ax, ux, uy, len(M["x"][::40]))
            nf = int(np.sum(r["disc"] <= tt))
            ax.set_title(f"{name}\n{tt:.1f} s scanning, {nf}/{n_obj} objects found", fontsize=9)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(r["rec"], origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135)
        ax.contour(np.linspace(*ext[:2], N), np.linspace(*ext[2:], N), REF, [OBJ], colors="w", linewidths=0.4)
        ax.set_title(f"TV map: {r['rmse']:.1f} nm, {r['pixels']:,} px, {r['on_obj']:.0%} on objects", fontsize=9)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    axE = fig.add_subplot(gs[-1, :2]); axD = fig.add_subplot(gs[-1, 2:])
    for name, M, col in methods:
        r = res[name]
        axE.plot(times, r["err"], "o-", ms=3, color=col, label=name)
        dd = np.r_[0, np.sort(r["disc"][np.isfinite(r["disc"])])]
        axD.step(np.r_[dd, M["t"][-1]], np.r_[np.arange(len(dd)), len(dd) - 1], where="post", color=col, label=name)
    axE.set_xlabel("scanning time (s)"); axE.set_ylabel("RMSE vs ground truth (nm)"); axE.legend(fontsize=8)
    axD.set_xlabel("scanning time (s)"); axD.set_ylabel("ground-truth objects found"); axD.set_ylim(0, n_obj + 0.3)
    axD.legend(fontsize=8, loc="lower right")
    fig.suptitle("GP exploration between objects, edge tracking on them -- vs contour following, pure GP and a raster "
                 "(same area, 256-line ground truth)", fontsize=12)
    for e in ("png", "pdf"):
        fig.savefig(out / f"hybrid_comparison.{e}", dpi=100)
    summ = {k: dict(pixels=v["pixels"], scan_s=v["scan_s"], wall_s=v["M"].get("wall"), on_object_frac=v["on_obj"],
                    rmse_nm=v["rmse"], discovery_s=[None if not np.isfinite(q) else float(q) for q in v["disc"]],
                    err_vs_time=list(map(float, v["err"]))) for k, v in res.items()}
    summ["times_s"] = times.tolist()
    summ["features_hybrid"] = [{k: f[k] for k in ("id", "closed", "closed_by", "coverage_deg", "interior_lines")} for f in (H["features"] or [])]
    summ["features_L2"] = [{k: f[k] for k in ("id", "closed", "closed_by", "coverage_deg", "interior_lines")} for f in (L2["features"] or [])]
    (out / "hybrid_comparison.json").write_text(json.dumps(summ, indent=1))
    print("figure:", out / "hybrid_comparison.png")
    if a.gif:
        make_gif(methods, res, REF, H, ext, xl, yl, b, out / "hybrid_comparison.gif")


def make_gif(methods, res, REF, H, ext, xl, yl, b, path, n_frames=40, fps=5):
    """Frames advance in SCANNED PIXELS (distinct pixels of the 200 x 200 grid visited), the same
    number for every method; a method that has used all its pixels stays at its final state."""
    from matplotlib.animation import PillowWriter
    C = {name: F.cum_pixels(F.pix(M["x"], M["y"], b)) for name, M, _ in methods}
    P = max(int(c[-1]) for c in C.values())
    fig, axs = plt.subplots(2, len(methods), figsize=(4.6 * len(methods), 9), layout="constrained")
    w = PillowWriter(fps=fps)
    with w.saving(fig, str(path), dpi=60):
        for f in list(np.linspace(0.02, 1, n_frames)) + [1.0] * 6:
            p = max(1, int(f * P))
            for j, (name, M, col) in enumerate(methods):
                c = C[name]
                u = max(2, min(int(np.searchsorted(c, p)) + 1, len(c)))
                done = p >= c[-1]
                tt = M["t"][u - 1]
                a0, a1 = axs[0, j], axs[1, j]; a0.clear(); a1.clear()
                a0.imshow(REF, origin="lower", extent=ext, cmap="gray", vmin=-15, vmax=220)
                ux, uy = F.to_um(H, M["x"][:u:40], M["y"][:u:40]); F.vpath(a0, ux, uy, len(M["x"][::40]))
                if not done:
                    a0.plot(ux[-1:], uy[-1:], "o", color="r", ms=6, zorder=5)
                nf = int(np.sum(res[name]["disc"] <= tt))
                a0.set_title(f"{name}"+chr(10)+f"{int(c[u - 1]):,} px{' (all, done)' if done else ''}, {nf} objects found", fontsize=9)
                rec = F.tv(F.binned(M["x"][:u], M["y"][:u], M["z"][:u], b), 300)
                a1.imshow(rec, origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135)
                a1.set_title(f"TV map: {F.rmse(rec, REF):.1f} nm", fontsize=9)
                for a_ in (a0, a1):
                    a_.set_xlim(*xl); a_.set_ylim(*yl); a_.set_aspect("equal"); a_.tick_params(labelsize=7)
            fig.suptitle(f"pixels scanned: {p:,} of the {F.N} x {F.N} grid ({p / F.N ** 2:.1%})", fontsize=12)
            w.grab_frame()
    plt.close(fig)
    print("animation:", path)


if __name__ == "__main__":
    main()
