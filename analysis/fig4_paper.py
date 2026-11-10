"""Fig. 4 of the manuscript in the layout of Figs. 2 and 3: GP path planning (L3) vs a raster with
the same number of measured pixels, on the 256-line ground truth.

    python fig4_paper.py RUN.npz --gt GT.npz --curves CURVES.npz --sweep discovery_summary.json --out DIR

Rows: SCOUT (GP) at three pixel budgets (path on the reference, inset TV), its measured pixels,
samples round one object; the raster at the same pixels, likewise. Bottom: (k) pixels on objects
vs measured pixels, (l) reconstruction error vs scanning time, (m) scanning time and overhead,
(n) the explore/exploit sweep (lambda, gamma): share of pixels on objects, with each run's raster.
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

plt = F.plt
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

N = F.N
VMIN, VMAX = -10, 130
GREY = "#f2f2f2"
C_GP, C_R = "#1f6fb4", "0.45"
FRAME = ["#4aa3e8", "#f5a742", "#f0656b"]     # the three budgets (as Fig. 2's levels)
ZOOM = "#c8326e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.linewidth": 1.0})


def own_level(x, y, z):
    c = F.robust_plane(x, y, z)
    r = z - (c[0] * x + c[1] * y + c[2])
    return r - np.median(r[r < np.percentile(r, 50) + 25])


def frame(ax, color, lw=3):
    for s in ax.spines.values():
        s.set_edgecolor(color); s.set_linewidth(lw)
    ax.set_xticks([]); ax.set_yticks([])


def label(ax, s, x=-0.04, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="right")


def tag(ax, s):
    ax.text(0.03, 0.03, s, transform=ax.transAxes, fontsize=10.5, color="w", va="bottom",
            bbox=dict(boxstyle="square,pad=0.25", fc="0.25", ec="none", alpha=0.8))


def scalebar(ax, um, xl, yl, txt):
    x0 = xl[0] + 0.06 * (xl[1] - xl[0]); y0 = yl[0] + 0.07 * (yl[1] - yl[0])
    ax.plot([x0, x0 + um], [y0, y0], "-", color="k", lw=4, solid_capstyle="butt", zorder=6)
    ax.text(x0 + um / 2, y0 + 0.02 * (yl[1] - yl[0]), txt, ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=6)


def measured_img(x, y, z, b):
    return F.binned(x, y, z, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--gt", required=True); ap.add_argument("--curves", required=True)
    ap.add_argument("--sweep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", nargs=3, type=float, default=(1 / 3, 2 / 3, 1.0))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = F.load_run(a.run); b = R["area"]
    R["zl"] = own_level(R["x"], R["y"], R["z"]); R["rzl"] = own_level(R["rx"], R["ry"], R["rz"])
    D = np.load(a.gt, allow_pickle=True)
    gx, gy, gz = (D[k].astype(float) for k in ("xs", "ys", "zs"))
    REF = F.tv(F.binned(gx, gy, own_level(gx, gy, gz), b), 1500)
    obj = REF > F.OBJ_NM
    Cv = np.load(a.curves)
    kG = F.pix(R["x"], R["y"], b); cG = F.cum_pixels(kG)
    kR = F.pix(R["rx"], R["ry"], b); cR = F.cum_pixels(kR)
    tG = np.arange(len(kG)) / R["rate"]; tR = np.arange(len(kR)) * R["r_dt"]
    of = obj.ravel()
    newG = np.r_[True, np.diff(cG) > 0] & (kG >= 0); newR = np.r_[True, np.diff(cR) > 0] & (kR >= 0)
    onG = np.cumsum(newG & of[np.maximum(kG, 0)]); onR = np.cumsum(newR & of[np.maximum(kR, 0)])
    PG = int(cG[-1])
    budgets = [max(1, int(f * PG)) for f in a.fracs]
    uG = [min(int(np.searchsorted(cG, p)) + 1, len(cG)) for p in budgets]
    uR = [min(int(np.searchsorted(cR, p)) + 1, len(cR)) for p in budgets]
    recG = [F.tv(F.binned(R["x"][:u], R["y"][:u], R["zl"][:u], b)) for u in uG]
    recR = []
    for u in uR:
        r_ = F.tv(F.binned(R["rx"][:u], R["ry"][:u], R["rzl"][:u], b))
        # area not scanned yet (above the last raster line): light grey
        yl_ = R["ry"][:u].max()
        rows = np.linspace(b[2], b[3], N) > yl_
        r_ = r_.copy(); r_[rows, :] = np.nan
        recR.append(r_)
    X0, Y0 = F.to_um(R, np.array(b[:2]), np.array(b[2:]))
    ext = [X0[0], X0[1], Y0[0], Y0[1]]
    xl = (min(ext[:2]), max(ext[:2])); yl = (min(ext[2:]), max(ext[2:]))
    xs_um = np.linspace(*ext[:2], N); ys_um = np.linspace(*ext[2:], N)

    # the zoomed object: the one nearest the centre of the field
    from scipy import ndimage
    lab, n = ndimage.label(obj)
    sizes = ndimage.sum(obj, lab, range(1, n + 1))
    cms = ndimage.center_of_mass(obj, lab, range(1, n + 1))
    big = [i for i in range(n) if sizes[i] >= 50]
    cy, cx = min((cms[i] for i in big), key=lambda c: (c[0] - N / 2) ** 2 + (c[1] - N / 2) ** 2)
    zc = (np.interp(cx, np.arange(N), xs_um), np.interp(cy, np.arange(N), ys_um))
    hz = 3.2
    zx = sorted((zc[0] - hz, zc[0] + hz)); zy = sorted((zc[1] - hz, zc[1] + hz))

    cmap_h = plt.get_cmap("copper").copy(); cmap_h.set_bad(GREY)
    fig = plt.figure(figsize=(22, 17))
    W, H = 0.155, 0.205
    xs0 = [0.045, 0.235, 0.425, 0.615, 0.805]
    rows_y = [0.735, 0.445]
    axes = {}
    for r_i, y0 in enumerate(rows_y):
        for c_i, x0 in enumerate(xs0):
            axes[(r_i, c_i)] = fig.add_axes([x0, y0, W, H])
    letters = "abcdefghij"
    sub = 25
    for r_i, (who, xx, yy, zz, kk, uu, recs, t_) in enumerate(
            (("SCOUT", R["x"], R["y"], R["zl"], kG, uG, recG, tG), ("Raster", R["rx"], R["ry"], R["rzl"], kR, uR, recR, tR))):
        cum = cG if r_i == 0 else cR
        for j, u in enumerate(uu):
            ax = axes[(r_i, j)]
            ax.imshow(np.where(obj, 0.78, 0.95), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
            ux, uy = F.to_um(R, xx[:u:sub], yy[:u:sub])
            if r_i == 0:
                F.vpath(ax, ux, uy, len(xx[::sub]), lw=1.6)
            else:
                ax.plot(ux, uy, "-", color="0.15", lw=0.7)
            ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
            frame(ax, FRAME[j] if r_i == 0 else "0.5")
            ins = ax.inset_axes([0.6, 0.6, 0.4, 0.4])
            ins.imshow(recs[j], origin="lower", extent=ext, cmap=cmap_h, vmin=VMIN, vmax=VMAX)
            ins.set_xlim(*xl); ins.set_ylim(*yl); frame(ins, "w", 2)
            ins.text(0.5, 0.97, "TV", transform=ins.transAxes, color="w", ha="center", va="top", fontsize=11, fontweight="bold")
            e = F.rmse(recs[j], REF)
            tag(ax, f"{t_[u - 1]:.1f} s · {int(cum[u - 1]):,} px · {e:.1f} nm")
            ttl = ([f"{int(100 * f)}% of the pixels" if f < 1 else "End of run" for f in a.fracs][j] if r_i == 0
                   else ["Same pixels as (a)", "Same pixels as (b)", "Same pixels as (c)"][j])
            ax.set_title(ttl, fontsize=13, fontweight="bold")
            label(ax, letters[5 * r_i + j])
        # measured pixels, coloured by height
        ax = axes[(r_i, 3)]
        img = measured_img(xx, yy, zz, b)
        ax.imshow(img, origin="lower", extent=ext, cmap=cmap_h, vmin=VMIN, vmax=VMAX, interpolation="nearest")
        ax.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec=ZOOM, lw=2))
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); frame(ax, FRAME[2] if r_i == 0 else "0.5")
        ax.set_title("Measured pixels", fontsize=13, fontweight="bold"); label(ax, letters[5 * r_i + 3])
        if r_i == 0:
            scalebar(ax, 5, xl, yl, "5 µm")
        # samples round the object
        ax = axes[(r_i, 4)]
        ax.set_facecolor(GREY)
        X, Y = F.to_um(R, xx, yy)
        m = (X > zx[0]) & (X < zx[1]) & (Y > zy[0]) & (Y < zy[1])
        st = max(1, int(m.sum() / 120_000))
        ax.scatter(X[m][::st], Y[m][::st], c=zz[m][::st], s=0.6, cmap="copper", vmin=VMIN, vmax=VMAX, rasterized=True, lw=0)
        rec_full = recG[-1] if r_i == 0 else F.tv(F.binned(xx, yy, zz, b), 800)
        ax.contour(xs_um, ys_um, rec_full, [F.OBJ_NM], colors=["#2a7fd4" if r_i == 0 else "#d43a2a"], linewidths=2.2, linestyles="--")
        near = (np.hypot(*np.meshgrid(xs_um - zc[0], ys_um - zc[1])) < hz).ravel()
        onobj = int(np.sum((np.bincount(kk[kk >= 0], minlength=N * N) > 0) & of & near))
        ax.set_xlim(*zx); ax.set_ylim(*zy); ax.set_aspect("equal"); frame(ax, ZOOM)
        ax.set_title(f"{onobj:,} px on object", fontsize=13, fontweight="bold"); label(ax, letters[5 * r_i + 4])
        if r_i == 0:
            scalebar(ax, 1, zx, zy, "1 µm")
        cax = fig.add_axes([xs0[4] + W + 0.006, rows_y[r_i], 0.008, H])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(VMIN, VMAX), cmap="copper"), cax=cax)
        cb.set_label("Height (nm)", fontsize=12)
    fig.text(0.018, rows_y[0] + H / 2, "SCOUT (GP)", rotation=90, fontsize=20, fontweight="bold", va="center", ha="center")
    fig.text(0.018, rows_y[1] + H / 2, "Raster", rotation=90, fontsize=20, fontweight="bold", va="center", ha="center")

    # ---------------- bottom row
    by, bh, bw = 0.07, 0.27, 0.19
    bx = [0.06, 0.305, 0.55, 0.795]
    ax = fig.add_axes([bx[0], by, bw, bh])
    ax.plot(cG / 1e3, onG / 1e3, color=C_GP, lw=2.2, label="SCOUT (GP)")
    ax.plot(cR / 1e3, onR / 1e3, color=C_R, lw=2.2, label="Raster")
    xxm = np.array([0, max(cG[-1], cR[-1])]); ax.plot(xxm / 1e3, xxm * obj.mean() / 1e3, ":", color="k", label="Uniform")
    for j, p in enumerate(budgets[:2]):
        ax.axvline(p / 1e3, color=FRAME[j], ls="--", lw=1.5)
    ax.set_xlabel("Measured pixels (×10³)"); ax.set_ylabel("Pixels on objects (×10³)"); ax.legend(frameon=False)
    ax.set_xlim(0, None); ax.set_ylim(0, None); label(ax, "k", -0.12)
    ax.text(0.98, 0.04, f"objects cover {obj.mean():.1%} of the area", transform=ax.transAxes, ha="right", fontsize=9.5, color="0.3")
    ax = fig.add_axes([bx[1], by, bw, bh])
    ax.plot(Cv["times"], Cv["errG"], "o-", color=C_GP, lw=2, ms=6, label="SCOUT (GP)")
    ax.plot(Cv["times"], Cv["errR"], "s-", color=C_R, lw=2, ms=6, label="Raster")
    for j, u in enumerate(uG[:2]):
        ax.axvline(tG[u - 1], color=FRAME[j], ls="--", lw=1.5)
    ax.set_xlabel("Scanning time (s)"); ax.set_ylabel("Reconstruction RMSE (nm)"); ax.legend(frameon=False)
    ax.set_xlim(0, None); ax.set_ylim(0, None); label(ax, "l", -0.12)
    ax = fig.add_axes([bx[2], by, bw * 0.8, bh])
    d = np.load(a.run, allow_pickle=True)
    wallG = float(d["run_wall_time_s"]); wallR = float(d["raster_wall_time_s"])
    sG, sR = tG[-1], float(R["r_scan_s"])
    ax.bar([0], [sG], 0.55, color=C_GP, label="Scanning")
    ax.bar([0], [wallG - sG], 0.55, bottom=[sG], color="#dde3ea", hatch="//", edgecolor="0.6", label="Overhead")
    ax.bar([1], [sR], 0.55, color=C_R)
    ax.bar([1], [wallR - sR], 0.55, bottom=[sR], color="#dde3ea", hatch="//", edgecolor="0.6")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["SCOUT (GP)", "Raster"]); ax.set_ylabel("Time (s)"); ax.legend(frameon=False)
    ax.set_yscale("log"); ax.set_ylim(1, wallG * 2)
    for xi, s_, w_ in ((0, sG, wallG), (1, sR, wallR)):
        ax.text(xi, w_ * 1.08, f"{w_:.0f} s", ha="center", fontsize=10)
        ax.text(xi, s_ * 0.55, f"{s_:.0f} s", ha="center", fontsize=10, color="w")
    label(ax, "m", -0.15)
    ax = fig.add_axes([bx[3], by, bw, bh])
    S = json.loads(pathlib.Path(a.sweep).read_text())
    lam = []
    for r in S:
        nm = r["name"]
        if not nm.startswith("GP lambda"):
            continue
        g2 = "gamma" in nm
        L = float(nm.split("lambda ")[1].split(",")[0])
        lam.append((L, g2, 100 * r["run"]["on_object"], 100 * r["raster"]["on_object"]))
    l1 = [q for q in lam if not q[1]]; l1.sort()
    ax.plot([q[0] for q in l1], [q[2] for q in l1], "o-", color=C_GP, lw=2, ms=7, label="SCOUT (GP), γ = 1")
    ax.plot([q[0] for q in l1], [q[3] for q in l1], "s", color=C_R, ms=6, label="Its raster")
    for q in lam:
        if q[1]:
            ax.plot([q[0]], [q[2]], "*", color="#e8590c", ms=17, label="SCOUT (GP), γ = 2", zorder=5)
            ax.plot([q[0]], [q[3]], "s", color=C_R, ms=6)
    ax.axhline(100 * obj.mean(), color="k", ls=":", lw=1.2, label="Uniform")
    ax.set_xlabel("Exploration ↔ exploitation weight λ"); ax.set_ylabel("Pixels on objects (%)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_ylim(0, None); ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    label(ax, "n", -0.12)
    for e in ("png", "pdf"):
        fig.savefig(out / f"Fig4_GP.{e}", dpi=150 if e == "png" else None)
    print("figure:", out / "Fig4_GP.png")
    json.dump(dict(run=R["name"], budgets=budgets, rmse_gp=[F.rmse(r_, REF) for r_ in recG],
                   rmse_raster=[F.rmse(r_, REF) for r_ in recR], wall=[wallG, wallR], scan=[sG, sR]),
              open(out / "Fig4_GP.json", "w"), indent=1)


if __name__ == "__main__":
    main()
