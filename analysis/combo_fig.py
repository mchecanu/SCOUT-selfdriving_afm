"""Combining SCOUT levels: three hybrids on the same sample, each vs its own raster (same pixels).

    python combo_fig.py --runs A.npz B.npz C.npz --names "..." "..." "..." --gt GT.npz --out DIR [--gif]

Rows: one hybrid each -- (path at the measured positions, viridis start -> end, with its traced /
spiralled outlines) | measured pixels coloured by height | TV map | its raster at the same pixels
(path + TV inset). Bottom: pixels on objects vs measured pixels (each hybrid and its raster),
ground-truth objects found vs measured pixels, scanning time + overhead.

The 256-line ground truth (10:54) is registered onto each run by a pure translation (phase
correlation of its TV map with the TV map of the run's own raster): the sample moved up to ~0.7 um
over the afternoon. Every acquisition levelled with its own substrate plane.
"""
from __future__ import annotations

import argparse
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
N = F.N
VMIN, VMAX = -10, 130
GREY = "#f2f2f2"
COLS = ["#1f6fb4", "#2b9348", "#d1495b", "#7b4fa0"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def register(ref_img, img):
    """Shift (rows, cols) that best aligns ref_img onto img (FFT cross-correlation of the
    centred, NaN-free images, integer pixels + parabolic sub-pixel)."""
    a = np.nan_to_num(ref_img - np.nanmean(ref_img)); b = np.nan_to_num(img - np.nanmean(img))
    C = np.fft.ifft2(np.fft.fft2(b) * np.conj(np.fft.fft2(a))).real
    r, c = np.unravel_index(np.argmax(C), C.shape)
    def sub(v, m, k):
        y0, y1, y2 = v[(k - 1) % m], v[k], v[(k + 1) % m]
        d = (y0 - 2 * y1 + y2)
        return k + (0.5 * (y0 - y2) / d if d != 0 else 0.0)
    rs = sub(C[:, c], C.shape[0], r); cs = sub(C[r, :], C.shape[1], c)
    if rs > C.shape[0] / 2: rs -= C.shape[0]
    if cs > C.shape[1] / 2: cs -= C.shape[1]
    return rs, cs


def objects(REF):
    obj = REF > F.OBJ_NM
    lab, n = ndimage.label(obj); sz = ndimage.sum(obj, lab, range(1, n + 1)); keep = np.flatnonzero(sz >= 50) + 1
    lab = np.where(np.isin(lab, keep), np.searchsorted(keep, lab) + 1, 0)
    return obj, lab, len(keep)


def found_vs_pixels(x, y, z, kk, cum, lab, b, n):
    """Measured pixels at which each ground-truth object was first seen on top (> OBJ_NM)."""
    on = z > F.OBJ_NM
    lf = lab.ravel()
    out = []
    for i in range(1, n + 1):
        hit = np.flatnonzero(on & (kk >= 0) & (lf[np.maximum(kk, 0)] == i))
        out.append(int(cum[hit[0]]) if len(hit) >= 1 else np.nan)
    return np.sort(np.array(out, float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--gt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = np.load(a.gt)
    gx, gy, gz = (D[k].astype(float) for k in ("xs", "ys", "zs"))
    runs = []
    for path, name in zip(a.runs, a.names):
        M = H.load(path, raster=True); b = M["area"]; R = M["raster"]
        GT0 = F.tv(F.binned(gx, gy, H.level_own(gx, gy, gz, b), b), 1500)
        recR = F.tv(F.binned(R["x"], R["y"], R["z"], b), 1500)
        dr, dc = register(GT0, recR)
        px = (b[1] - b[0]) / (N - 1)
        REF = F.tv(F.binned(gx + dc * px, gy + dr * px, H.level_own(gx, gy, gz, b), b), 1500)
        obj, lab, n = objects(REF)
        sh_um = F.to_um(M, np.array([b[0] + dc * px]), np.array([b[2] + dr * px]))
        sh0 = F.to_um(M, np.array([b[0]]), np.array([b[2]]))
        shift = (float(sh_um[0][0] - sh0[0][0]), float(sh_um[1][0] - sh0[1][0]))
        res = dict(name=name, file=pathlib.Path(path).name, M=M, REF=REF, obj=obj, shift_um=shift)
        for tag, S in (("run", M), ("raster", R)):
            k = F.pix(S["x"], S["y"], b); c = F.cum_pixels(k)
            new = np.r_[True, np.diff(c) > 0] & (k >= 0)
            onc = np.cumsum(new & obj.ravel()[np.maximum(k, 0)])
            rec = F.tv(F.binned(S["x"], S["y"], S["z"], b), 1500)
            res[tag] = dict(k=k, c=c, on=onc, rec=rec, rmse=F.rmse(rec, REF), pixels=int(c[-1]),
                            on_frac=float(onc[-1] / c[-1]), scan_s=float(S["t"][-1]),
                            found=found_vs_pixels(S["x"], S["y"], S["z"], k, c, lab, b, n))
        res["n_obj"] = n
        feats = M["features"] or []
        res["closed"] = sum(f["closed"] for f in feats if "speck" not in str(f.get("closed_by", "")))
        # outline quality: a closed outline is COMPLETE when it covers >= 70% of the ground-truth
        # object it sits on (registered GT, same grid) -- the same test for every hybrid
        from matplotlib.path import Path as _P
        GXv, GYv = np.meshgrid(np.linspace(b[0], b[1], N), np.linspace(b[2], b[3], N))
        pts = np.column_stack([GXv.ravel(), GYv.ravel()])
        comp = 0
        for f in feats:
            O = np.asarray(f["outline"], float).reshape(-1, 2)
            if not f["closed"] or len(O) < 3:
                continue
            inside = _P(O).contains_points(pts).reshape(N, N)
            ids, cnt = np.unique(lab[inside & (lab > 0)], return_counts=True)
            if len(ids) and cnt.max() >= 0.7 * (lab == ids[np.argmax(cnt)]).sum():
                comp += 1
        res["complete"] = comp
        print(f"{name:40s} GT shift ({shift[0]:+.2f}, {shift[1]:+.2f}) um | run {res['run']['pixels']:,} px, "
              f"{res['run']['on_frac']:.1%} on obj, TV {res['run']['rmse']:.1f} nm, all found at {res['run']['found'][-1]} px | "
              f"raster {res['raster']['pixels']:,} px, {res['raster']['on_frac']:.1%}, {res['raster']['rmse']:.1f} nm, "
              f"all found at {res['raster']['found'][-1]} px | closed {res['closed']}")
        runs.append(res)

    M0 = runs[0]["M"]; b = M0["area"]
    X0, Y0 = F.to_um(M0, np.array(b[:2]), np.array(b[2:]))
    ext = [X0[0], X0[1], Y0[0], Y0[1]]; xl = (min(ext[:2]), max(ext[:2])); yl = (min(ext[2:]), max(ext[2:]))
    cmap_h = plt.get_cmap("copper").copy(); cmap_h.set_bad(GREY)
    nr = len(runs)
    fig = plt.figure(figsize=(21, 5.0 * nr + 5.6))
    gs = fig.add_gridspec(nr + 1, 4, height_ratios=[1] * nr + [1.05], hspace=0.28, wspace=0.12,
                          left=0.08, right=0.97, top=0.935, bottom=0.05)
    L = iter("abcdefghijklmnopqrstuvwxyz")
    for i, r in enumerate(runs):
        M = r["M"]; R = M["raster"]
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(np.where(r["obj"], 0.78, 0.95), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
        ux, uy = F.to_um(M, M["x"][::20], M["y"][::20]); F.vpath(ax, ux, uy, len(ux), lw=1.3)
        for f in (M["features"] or []):
            O = np.asarray(f["outline"], float).reshape(-1, 2)
            if f["closed"] and len(O) >= 3:
                ox, oy = F.to_um(M, O[:, 0], O[:, 1])
                ax.plot(np.r_[ox, ox[:1]], np.r_[oy, oy[:1]], "-", color="r", lw=1.6)
        ax.set_title(f"{r['name']}\n{r['closed']}/{r['n_obj']} outlines closed ({r['complete']} complete) · {r['run']['scan_s']:.0f} s scanning", fontsize=12)
        ax.text(-0.30, 0.5, f"{'ABCD'[i]}", transform=ax.transAxes, fontsize=24, fontweight="bold", color=COLS[i], va="center")
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(F.binned(M["x"], M["y"], M["z"], b), origin="lower", extent=ext, cmap=cmap_h, vmin=VMIN, vmax=VMAX, interpolation="nearest")
        ax2.set_title(f"measured pixels: {r['run']['pixels']:,}\n{r['run']['on_frac']:.0%} on objects", fontsize=12)
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(r["run"]["rec"], origin="lower", extent=ext, cmap="copper", vmin=VMIN, vmax=VMAX)
        ax3.contour(np.linspace(*ext[:2], N), np.linspace(*ext[2:], N), r["REF"], [F.OBJ_NM], colors="c", linewidths=0.6)
        ax3.set_title(f"TV reconstruction\nerror {r['run']['rmse']:.1f} nm", fontsize=12)
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.imshow(np.where(r["obj"], 0.78, 0.95), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
        rx, ry = F.to_um(M, R["x"][::10], R["y"][::10]); ax4.plot(rx, ry, "-", color="0.15", lw=0.6)
        ins = ax4.inset_axes([0.6, 0.6, 0.4, 0.4]); ins.imshow(r["raster"]["rec"], origin="lower", extent=ext, cmap="copper", vmin=VMIN, vmax=VMAX)
        ins.set_xticks([]); ins.set_yticks([]); ins.set_xlim(*xl); ins.set_ylim(*yl)
        ax4.set_title(f"raster, same pixels ({R['lines']} lines)\n{r['raster']['on_frac']:.0%} on objects · error {r['raster']['rmse']:.1f} nm", fontsize=12)
        for A_ in (ax, ax2, ax3, ax4):
            A_.set_xlim(*xl); A_.set_ylim(*yl); A_.set_aspect("equal"); A_.tick_params(labelsize=8)
            A_.text(-0.10, 1.02, next(L), transform=A_.transAxes, fontsize=16, fontweight="bold", ha="right", va="bottom")
        ax.set_ylabel("Y (µm)")
    # bottom
    axk = fig.add_subplot(gs[nr, 0:2]); axf = fig.add_subplot(gs[nr, 2]); axt = fig.add_subplot(gs[nr, 3])
    for i, r in enumerate(runs):
        axk.plot(r["run"]["c"] / 1e3, r["run"]["on"] / 1e3, "-", color=COLS[i], lw=2.2, label=r["name"])
        axk.plot(r["raster"]["c"] / 1e3, r["raster"]["on"] / 1e3, "--", color=COLS[i], lw=1.2, alpha=0.8)
        for tag, ls in (("run", "-"), ("raster", "--")):
            fd = r[tag]["found"]; fd = fd[np.isfinite(fd)]
            axf.step(np.r_[0, fd, r[tag]["pixels"]] / 1e3, np.r_[0, np.arange(1, len(fd) + 1), len(fd)], where="post",
                     color=COLS[i], ls=ls, lw=2 if tag == "run" else 1.2)
        w = float(r["M"]["wall"]); s = r["run"]["scan_s"]
        axt.bar(i - 0.18, s, 0.34, color=COLS[i]); axt.bar(i - 0.18, w - s, 0.34, bottom=s, color="#dde3ea", hatch="//", edgecolor="0.6")
        wr = float(r["M"]["raster"]["wall"]); sr = r["raster"]["scan_s"]
        axt.bar(i + 0.18, sr, 0.34, color="0.45"); axt.bar(i + 0.18, wr - sr, 0.34, bottom=sr, color="#dde3ea", hatch="//", edgecolor="0.6")
        axt.text(i - 0.18, w * 1.05, f"{w:.0f}", ha="center", fontsize=9); axt.text(i + 0.18, wr * 1.05, f"{wr:.0f}", ha="center", fontsize=9)
    xm = max(max(r["run"]["pixels"], r["raster"]["pixels"]) for r in runs)
    axk.plot([0, xm / 1e3], [0, xm * runs[0]["obj"].mean() / 1e3], ":", color="k", label="uniform")
    axk.plot([], [], "--", color="0.4", label="each one's raster (same pixels)")
    axk.set_xlabel("Measured pixels (×10³)"); axk.set_ylabel("Pixels on objects (×10³)"); axk.legend(frameon=False, fontsize=10)
    axk.set_xlim(0, None); axk.set_ylim(0, None)
    axf.set_xlabel("Measured pixels (×10³)"); axf.set_ylabel("Objects found"); axf.set_ylim(0, runs[0]["n_obj"] + 0.4)
    axf.set_xlim(0, None)
    axt.set_xticks(range(nr)); axt.set_xticklabels(["ABCD"[i] for i in range(nr)]); axt.set_yscale("log")
    axt.set_ylabel("Time (s): scanning + overhead"); axt.set_title("left: hybrid, right: its raster", fontsize=10)
    for A_ in (axk, axf, axt):
        A_.text(-0.06 if A_ is axk else -0.12, 1.02, next(L), transform=A_.transAxes, fontsize=16, fontweight="bold", ha="right", va="bottom")
    fig.suptitle("Combining SCOUT levels on the same sample: exploration by one level, objects handled by another "
                 "(each vs a raster with the same pixels; 256-line ground truth, registered)", fontsize=14)
    for e in ("png", "pdf"):
        fig.savefig(out / f"combined_levels.{e}", dpi=110)
    print("figure:", out / "combined_levels.png")
    js = [{k: (v if not isinstance(v, dict) else {kk: (vv.tolist() if isinstance(vv, np.ndarray) and vv.size < 20 else vv)
                                                  for kk, vv in v.items() if kk in ("rmse", "pixels", "on_frac", "scan_s", "found")})
           for k, v in r.items() if k in ("name", "file", "shift_um", "run", "raster", "closed", "complete", "n_obj")} for r in runs]
    for r, j in zip(runs, js):
        j["wall_s"] = float(r["M"]["wall"]); j["raster_wall_s"] = float(r["M"]["raster"]["wall"])
    (out / "combined_levels.json").write_text(json.dumps(js, indent=1, default=float))
    if a.gif:
        make_gif(runs, ext, xl, yl, b, out / "combined_levels.gif")


def make_gif(runs, ext, xl, yl, b, path, n_frames=40, fps=5):
    """Every hybrid and the raster of the first one, frame by frame at the same number of measured pixels."""
    from matplotlib.animation import PillowWriter
    R0 = runs[0]["M"]["raster"]
    items = [(r["name"], r["M"], r["run"]["c"], r["REF"]) for r in runs] + \
            [(f"raster ({R0['lines']} lines)", R0, runs[0]["raster"]["c"], runs[0]["REF"])]
    P = max(int(c[-1]) for _, _, c, _ in items)
    fig, axs = plt.subplots(2, len(items), figsize=(4.6 * len(items), 9), layout="constrained")
    w = PillowWriter(fps=fps)
    M0 = runs[0]["M"]
    with w.saving(fig, str(path), dpi=60):
        for f in list(np.linspace(0.02, 1, n_frames)) + [1.0] * 6:
            p = max(1, int(f * P))
            for j, (name, M, c, REF) in enumerate(items):
                u = max(2, min(int(np.searchsorted(c, p)) + 1, len(c)))
                done = p >= c[-1]
                a0, a1 = axs[0, j], axs[1, j]; a0.clear(); a1.clear()
                a0.imshow(np.where(REF > F.OBJ_NM, 0.78, 0.95), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
                ux, uy = F.to_um(M0, M["x"][:u:20], M["y"][:u:20]); F.vpath(a0, ux, uy, len(M["x"][::20]), lw=1.3)
                if not done:
                    a0.plot(ux[-1:], uy[-1:], "o", color="r", ms=6, zorder=5)
                a0.set_title(f"{name}\n{int(c[u - 1]):,} px{' (done)' if done else ''}", fontsize=9)
                rec = F.tv(F.binned(M["x"][:u], M["y"][:u], M["z"][:u], b), 300)
                a1.imshow(rec, origin="lower", extent=ext, cmap="copper", vmin=VMIN, vmax=VMAX)
                a1.set_title(f"TV map: {F.rmse(rec, REF):.1f} nm", fontsize=9)
                for a_ in (a0, a1):
                    a_.set_xlim(*xl); a_.set_ylim(*yl); a_.set_aspect("equal"); a_.tick_params(labelsize=7)
            fig.suptitle(f"pixels scanned: {p:,} of the {N} x {N} grid ({p / N ** 2:.1%})", fontsize=12)
            w.grab_frame()
    plt.close(fig)
    print("animation:", path)


if __name__ == "__main__":
    main()
