"""Fig. 4 analysis: GP path planning (L3) vs the equivalent raster, on the instrument.

    python fig4_gp.py RUN.npz [--others A.npz B.npz ...] [--out DIR] [--gif]

RUN.npz: a gp_path run saved by Run_GP_Paper.ipynb with its equivalent raster (raster_* arrays).
--others: other runs over the SAME area (same session / sensor calibration): their GP and raster
samples build the reference image (TV of all of them), independent of RUN's own data.

Everything at the MEASURED (sensor) positions, in um. Heights levelled jointly (one robust
substrate plane + a constant offset per acquisition). Outputs (in --out):
  <tag>_fig4.png / .pdf      the figure panels
  <tag>_metrics.json         numbers for the text
  <tag>_curves.npz           error / on-object curves vs scanning time and pixels
  <tag>_animation.gif        (--gif) GP path, GP mean and sigma, raster at equal pixel budgets
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

PATH_LW = 1.8          # path lines (figures + animations), coloured start -> end (viridis)


def vpath(ax, x, y, n_total, lw=PATH_LW, cmap="viridis", alpha=1.0):
    """The path as a line coloured by progress through the WHOLE acquisition (viridis:
    start purple -> end yellow), so a partial path shows how far along it is."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return None
    pts = np.column_stack([x, y])[:, None, :]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t = np.linspace(0, (len(x) - 1) / max(n_total - 1, 1), len(x) - 1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1), linewidths=lw, alpha=alpha, capstyle="round")
    lc.set_array(t)
    ax.add_collection(lc)
    return lc

from selfdriving_afm.reconstruction import bin_image, tv_inpaint  # noqa: E402

N = 200                 # pixel grid over the area (as the run)
OBJ_NM = 58.0           # object mask: half the ~116 nm object height (manuscript)
TV_MU = 0.3


# ------------------------------------------------------------------ loading
def load_run(path):
    d = np.load(path, allow_pickle=True)
    pr = json.loads(str(d["priors_json"]))
    c, h = pr["area_center_V"], pr["area_half_width_V"]
    area = (c[0] - h, c[0] + h, c[1] - h, c[1] + h)
    # full runs: every sample; lite / Zenodo copies (selfdriving_afm.datasets): 1 in pts_stride
    lite = "pts" in d.files
    pts = d["pts"] if lite else d["points"]
    st = int(d["pts_stride"]) if lite else 1
    seg = d["segment_lengths"]; kinds = d["kinds"]
    rate_true = float(d["sample_rate_hz"])
    rate = rate_true / st                 # samples per second OF THESE ARRAYS
    r = dict(rate_true=rate_true, path=str(path), area=area, um_per_V=np.asarray(d["um_per_V"], float), um_at_0V=np.asarray(d["um_at_0V"], float),
             x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], rate=rate, seg=seg, kinds=kinds,
             tip_v=float(d["tip_velocity_V_s"]), gp_timing=d["gp_timing"] if "gp_timing" in d.files else None,
             priors=pr, name=pathlib.Path(path).stem)
    from selfdriving_afm.gp_presets import GP_PRESETS
    hits = [k for k in GP_PRESETS if f"_{k}_" in f"_{r['name']}_"]
    r["preset"] = max(hits, key=len) if hits else "pure_gp_smooth"
    mb = re.search(r"_(\d+)V_gp_path", r["name"])
    r["budget_V"] = float(mb.group(1)) if mb else 120.0
    if "raster_xs" in d.files:
        r.update(rx=d["raster_xs"].astype(float), ry=d["raster_ys"].astype(float), rz=d["raster_zs"].astype(float),
                 r_lines=int(d["raster_n_lines"]), r_scan_s=float(d["raster_scan_time_s"]),
                 r_wall_s=float(d["raster_wall_time_s"]), r_every_V=float(d["raster_samples_every_V"]))
        # raster time per kept sample (block means: every r_every_V of path at the tip speed)
        if "raster_t" in d.files:
            r["r_dt"] = float(d["raster_t"][1] - d["raster_t"][0]) if len(d["raster_t"]) > 1 else 1.0 / rate_true
        else:                             # lite copy: the step between the kept raster samples
            r["r_dt"] = float(d["raster_dt"])
    return r


def to_um(r, x, y):
    return r["um_per_V"][0] * x + r["um_at_0V"][0], r["um_per_V"][1] * y + r["um_at_0V"][1]


# ------------------------------------------------------------------ levelling
def robust_plane(x, y, z, above_nm=25.0, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    i = rng.choice(len(z), min(n, len(z)), replace=False)
    X, Y, Z = x[i], y[i], z[i]
    m = np.ones(len(Z), bool)
    c = np.zeros(3)
    for _ in range(8):
        A = np.column_stack([X[m], Y[m], np.ones(m.sum())])
        c, *_ = np.linalg.lstsq(A, Z[m], rcond=None)
        res = Z - (c[0] * X + c[1] * Y + c[2])
        base = np.median(res[m])
        m = res < base + above_nm
    return c


def level(sets):
    """sets: list of (x, y, z) in V / nm. One plane for all + one offset each (substrate median = 0)."""
    X = np.concatenate([s[0] for s in sets]); Y = np.concatenate([s[1] for s in sets]); Z = np.concatenate([s[2] for s in sets])
    c = robust_plane(X, Y, Z)
    out = []
    for x, y, z in sets:
        r = z - (c[0] * x + c[1] * y + c[2])
        sub = r < np.percentile(r, 50) + 25
        out.append(r - np.median(r[sub]))
    return c, out


# ------------------------------------------------------------------ pixels
def pix(x, y, b, n=N):
    ix = np.clip(((x - b[0]) / (b[1] - b[0]) * (n - 1) + 0.5).astype(int), 0, n - 1)
    iy = np.clip(((y - b[2]) / (b[3] - b[2]) * (n - 1) + 0.5).astype(int), 0, n - 1)
    inside = (x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3])
    return np.where(inside, iy * n + ix, -1)


def cum_pixels(k):
    new = np.zeros(len(k), bool)
    _, first = np.unique(k, return_index=True)
    new[first] = True
    new &= k >= 0
    return np.cumsum(new)


def binned(x, y, z, b):
    img, _ = bin_image(np.column_stack([x, y]), z, N, b)
    return img


def tv(img, iters=800):
    return tv_inpaint(img, mu=TV_MU, n_iter=iters)


def rmse(a, ref, mask=None):
    m = np.isfinite(a) & np.isfinite(ref) & (mask if mask is not None else True)
    return float(np.sqrt(np.mean((a[m] - ref[m]) ** 2)))


# ------------------------------------------------------------------ GP at a budget
def gp_maps(r, upto, grid_xy):
    """Posterior mean / sigma (nm) of the planner's GP model refitted on the run's first `upto`
    samples (same preset, priors, training stride), on grid_xy (V)."""
    from selfdriving_afm.config import ExperimentConfig
    from selfdriving_afm.gp_presets import apply_gp_preset
    from selfdriving_afm.gp_models import make_gp
    from selfdriving_afm import priors as P
    cfg = ExperimentConfig()
    cfg.velocity.tip_velocity_V_s = r["tip_v"]; cfg.velocity.sample_rate_hz = r["rate"]
    apply_gp_preset(cfg, r["preset"])
    pr = r["priors"]
    sp = P.ScanPriors(feature_min_V=pr["feature_min_V"], feature_max_V=pr["feature_max_V"],
                      area_half_width_V=pr["area_half_width_V"], area_center_V=tuple(pr["area_center_V"]))
    P.apply(cfg, sp)
    g = cfg.gp_path
    ls = g.length_scale_V
    ls_b = (g.length_scale_min_frac * ls, g.path_segment_length_V * 50 if g.length_scale_bounds_V is None else g.length_scale_bounds_V[1])
    gp = make_gp(g, float(np.clip(ls, *ls_b)), ls_b)
    stride = max(1, int(round(g.gradient_eval_resolution_V / (r["tip_v"] / r["rate"]))))
    sl = slice(0, upto, stride)
    X = np.column_stack([r["x"][sl], r["y"][sl]]); Z = r["zl"][sl]
    gp.fit(X, Z, center=X[-1])
    mu, sd = gp.predict(grid_xy, return_std=True)
    return mu, sd


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--dense", default=None, help="dense reference raster npz (Run_GP_Paper.ipynb): the reference")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fracs", nargs=3, type=float, default=(0.2, 0.5, 1.0))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    R = load_run(a.run)
    O = [load_run(p) for p in a.others]
    b = R["area"]
    tag = R["name"]
    # joint levelling of everything in this session
    sets = [(R["x"], R["y"], R["z"]), (R["rx"], R["ry"], R["rz"])]
    for o in O:
        sets += [(o["x"], o["y"], o["z"])] + ([(o["rx"], o["ry"], o["rz"])] if "rx" in o else [])
    c, lev = level(sets)
    R["zl"], R["rzl"] = lev[0], lev[1]
    ref_sets = lev[2:]
    # reference: TV of all OTHER acquisitions (independent of this run) -- or, without others,
    # of this run's GP + raster together (hold-out scoring then)
    ref_label = "other runs" if O else "GP + raster"
    if a.dense:
        D = np.load(a.dense, allow_pickle=True)
        dx, dy, dz = (D[k].astype(float) for k in (("xs", "ys", "zs") if "xs" in D.files else ("x", "y", "z")))
        # each acquisition levelled with ITS OWN robust substrate plane (substrate = 0): the
        # sample's tilt drifts between acquisitions minutes apart (seen: ~30 nm across the field
        # between the hero run and the ground truth 20 min later, for the GP and its raster alike)
        def own(xx, yy, zz):
            c_ = robust_plane(xx, yy, zz)
            r_ = zz - (c_[0] * xx + c_[1] * yy + c_[2])
            return r_ - np.median(r_[r_ < np.percentile(r_, 50) + 25])
        dl = own(dx, dy, dz)
        R["zl"] = own(R["x"], R["y"], R["z"]); R["rzl"] = own(R["rx"], R["ry"], R["rz"])
        ref_b = binned(dx, dy, dl, b)
        nm_ = pathlib.Path(a.dense).name
        ref_kind = (f"{nm_.split('_')[2] if 'lines' in nm_ else 'dense'} raster of the same area ({nm_}), TV; "
                    f"every acquisition levelled with its own substrate plane")
        ref_label = ("ground truth, " + nm_.split("_")[2].replace("lines", "-line raster")) if "ground_truth" in nm_ else "dense raster"
    elif O:
        xs = np.concatenate([s[0] for s in sets[2:]]); ys = np.concatenate([s[1] for s in sets[2:]])
        zs = np.concatenate(ref_sets)
        ref_b = binned(xs, ys, zs, b)
        ref_kind = f"TV of {len(sets) - 2} other acquisitions of the same area"
    else:
        ref_b = binned(np.concatenate([R["x"], R["rx"]]), np.concatenate([R["y"], R["ry"]]),
                       np.concatenate([R["zl"], R["rzl"]]), b)
        ref_kind = "TV of this run's GP + raster data (hold-out scoring)"
    REF = tv(ref_b, 1500)
    obj = REF > OBJ_NM
    print(f"reference: {ref_kind}; {np.isfinite(ref_b).mean():.1%} of pixels measured; objects {obj.mean():.1%} of the area")

    # pixels over time
    kG = pix(R["x"], R["y"], b); cG = cum_pixels(kG)
    kR = pix(R["rx"], R["ry"], b); cR = cum_pixels(kR)
    tG = np.arange(len(kG)) / R["rate"]                          # scanning time (all motion incl. connectors)
    tR = np.arange(len(kR)) * R["r_dt"]
    objflat = obj.ravel()
    onG = np.cumsum(np.where(kG >= 0, objflat[np.maximum(kG, 0)], False) & (np.r_[True, np.diff(cG) > 0]))
    onR = np.cumsum(np.where(kR >= 0, objflat[np.maximum(kR, 0)], False) & (np.r_[True, np.diff(cR) > 0]))
    PG, PR = int(cG[-1]), int(cR[-1])
    print(f"pixels: GP {PG:,} ({tG[-1]:.1f} s scanning), raster {PR:,} ({tR[-1]:.1f} s); on objects GP {onG[-1] / PG:.1%}, raster {onR[-1] / PR:.1%}")

    # error vs time
    n_t = 14
    times = np.linspace(tG[-1] / n_t, max(tG[-1], tR[-1]), n_t)
    errG, errR, iouG, iouR = [], [], [], []
    for t in times:
        u = int(np.searchsorted(tG, t)); v = int(np.searchsorted(tR, t))
        for (x, y, z, uu, E, I) in ((R["x"], R["y"], R["zl"], u, errG, iouG), (R["rx"], R["ry"], R["rzl"], v, errR, iouR)):
            if uu < 10:
                E.append(np.nan); I.append(np.nan); continue
            rec = tv(binned(x[:uu], y[:uu], z[:uu], b))
            E.append(rmse(rec, REF)); m = rec > OBJ_NM
            I.append(float((m & obj).sum() / max((m | obj).sum(), 1)))
    print("error vs time (s, GP nm, raster nm):")
    for t, e1, e2 in zip(times, errG, errR):
        print(f"  {t:6.1f}  {e1:6.1f}  {e2:6.1f}")

    # budgets (pixels) for the panels
    budgets = [max(1, int(f * PG)) for f in a.fracs]
    uG = [int(np.searchsorted(cG, p)) + 1 for p in budgets]
    uR = [min(int(np.searchsorted(cR, p)) + 1, len(cR)) for p in budgets]
    gx = np.linspace(b[0], b[1], N); gy = np.linspace(b[2], b[3], N)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    maps = []
    for u in uG:
        mu, sd = gp_maps(R, u, grid)
        maps.append((mu.reshape(N, N), sd.reshape(N, N)))
    recG = [tv(binned(R["x"][:u], R["y"][:u], R["zl"][:u], b)) for u in uG]
    recR = [tv(binned(R["rx"][:u], R["ry"][:u], R["rzl"][:u], b)) for u in uR]

    # ---------------------------------------------------------------- figure
    X0, Y0 = to_um(R, np.array(b[:2]), np.array(b[2:]))
    ext = [X0[0], X0[1], Y0[0], Y0[1]]
    xl = (min(ext[:2]), max(ext[:2])); yl = (min(ext[2:]), max(ext[2:]))
    vmin, vmax = -15, 135
    fig = plt.figure(figsize=(18, 17.5), layout="constrained")
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 0.9])
    sub = 40
    for j, (u, v, p) in enumerate(zip(uG, uR, budgets)):
        # GP path (measured positions) on the reference, inset TV
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(REF, origin="lower", extent=ext, cmap="gray", vmin=vmin, vmax=vmax * 1.6, alpha=0.6)
        ux, uy = to_um(R, R["x"][:u:sub], R["y"][:u:sub]); vpath(ax, ux, uy, len(R["x"][::sub]))
        ax.set_title(f"GP path: {p:,} px, {tG[u - 1]:.1f} s\nTV RMSE {rmse(recG[j], REF):.1f} nm", fontsize=10)
        # GP posterior mean / sigma
        ax2 = fig.add_subplot(gs[1, j]); ax2.imshow(recG[j], origin="lower", extent=ext, cmap="copper", vmin=vmin, vmax=vmax)
        ax2.contour(np.linspace(*ext[:2], N), np.linspace(*ext[2:], N), REF, [OBJ_NM], colors="w", linewidths=0.4)
        ax2.set_title(f"map (TV of the GP samples): {rmse(recG[j], REF):.1f} nm", fontsize=10)
        ax3 = fig.add_subplot(gs[2, j]); im3 = ax3.imshow(maps[j][1], origin="lower", extent=ext, cmap="inferno")
        ax3.plot(ux, uy, lw=0.3, color="w", alpha=0.6); ax3.set_title("GP posterior sigma (nm)", fontsize=10)
        fig.colorbar(im3, ax=ax3, shrink=0.7)
        # raster at the same pixels
        ax4 = fig.add_subplot(gs[3, j])
        ax4.imshow(REF, origin="lower", extent=ext, cmap="gray", vmin=vmin, vmax=vmax * 1.6, alpha=0.6)
        rx_, ry_ = to_um(R, R["rx"][:v:sub], R["ry"][:v:sub]); vpath(ax4, rx_, ry_, len(R["rx"][::sub]))
        ax4.set_title(f"raster: {int(cR[v - 1]):,} px, {tR[v - 1]:.1f} s\nTV RMSE {rmse(recR[j], REF):.1f} nm", fontsize=10)
        for A_ in (ax, ax2, ax3, ax4):
            A_.set_xlim(*xl); A_.set_ylim(*yl); A_.set_aspect("equal"); A_.tick_params(labelsize=8)
        for A_, rec in ((ax, recG[j]), (ax4, recR[j])):
            ins = A_.inset_axes([0.62, 0.62, 0.37, 0.37]); ins.imshow(rec, origin="lower", extent=ext, cmap="copper", vmin=vmin, vmax=vmax)
            ins.set_xlim(*xl); ins.set_ylim(*yl); ins.set_xticks([]); ins.set_yticks([])
    # column 4: reference, final reconstructions
    for row, (img, t) in enumerate(((REF, "reference (" + ref_label + ")"),
                                     (recG[-1], "GP data, TV"), (recR[-1], "raster data, TV"))):
        ax = fig.add_subplot(gs[row, 3]); ax.imshow(img, origin="lower", extent=ext, cmap="copper", vmin=vmin, vmax=vmax)
        ax.contour(np.linspace(*ext[:2], N), np.linspace(*ext[2:], N), REF, [OBJ_NM], colors="w", linewidths=0.5)
        ax.set_title(t, fontsize=10); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    ax = fig.add_subplot(gs[3, 3])
    ax.plot(cG, onG, color="C0", label="GP"); ax.plot(cR, onR, color="C3", label="raster")
    xx = np.array([0, max(PG, PR)]); ax.plot(xx, xx * obj.mean(), ":", color="k", label="uniform")
    ax.set_xlabel("measured pixels"); ax.set_ylabel("pixels on objects"); ax.legend(fontsize=8)
    axE = fig.add_subplot(gs[4, :2])
    axE.plot(times, errG, "o-", color="C0", label="GP path (L3)"); axE.plot(times, errR, "s-", color="C3", label="raster (L0)")
    axE.set_xlabel("scanning time (s)"); axE.set_ylabel("RMSE vs reference (nm)"); axE.legend(); axE.set_title("reconstruction error vs scanning time")
    axT = fig.add_subplot(gs[4, 2:])
    if R["gp_timing"] is not None and len(R["gp_timing"]):
        gt = np.asarray(R["gp_timing"], float)
        axT.plot(gt[:, 0], gt[:, 2] + gt[:, 3], ".", ms=3, color="C0", label="GP update (fit + acquisition)")
        seg_t = np.array([s / R.get("rate_true", R["rate"]) for s, k in zip(R["seg"], R["kinds"]) if k != "connector"])
        axT.axhline(np.median(seg_t), color="C2", ls="--", label=f"segment motion ({np.median(seg_t) * 1e3:.0f} ms)")
        axT.set_yscale("log"); axT.set_xlabel("segment"); axT.set_ylabel("time (s)"); axT.legend(fontsize=8)
        axT.set_title("decision latency per segment")
    fig.suptitle(f"GP path planning (L3), preset {R['preset']}, {R['budget_V']:g} V budget -- vs a raster with the same pixels", fontsize=13)
    for ext_ in ("png", "pdf"):
        fig.savefig(out / f"{tag}_fig4.{ext_}", dpi=110)
    plt.close(fig)

    figS, axS = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.6), layout="constrained")
    for j, (a_, (mu_, _)) in enumerate(zip(np.atleast_1d(axS), maps)):
        a_.imshow(mu_, origin="lower", extent=ext, cmap="copper", vmin=vmin, vmax=vmax)
        a_.contour(np.linspace(*ext[:2], N), np.linspace(*ext[2:], N), REF, [OBJ_NM], colors="w", linewidths=0.4)
        a_.set_title(f"GP posterior mean, {budgets[j]:,} px: {rmse(mu_, REF):.1f} nm", fontsize=10)
        a_.set_xlim(*xl); a_.set_ylim(*yl); a_.set_aspect("equal")
    figS.savefig(out / f"{tag}_SI_gp_mean.png", dpi=100); plt.close(figS)
    gt = np.asarray(R["gp_timing"], float) if R["gp_timing"] is not None else np.zeros((0, 4))
    metrics = dict(run=R["name"], preset=R["preset"], budget_V=R["budget_V"], reference=ref_kind,
                   objects_area_frac=float(obj.mean()),
                   gp=dict(pixels=PG, scan_s=float(tG[-1]), on_object_frac=float(onG[-1] / PG), rmse_final_nm=rmse(recG[-1], REF),
                           waveforms=int(sum(k != "connector" for k in R["kinds"])),
                           gp_update_median_s=float(np.median(gt[:, 2] + gt[:, 3])) if len(gt) else None,
                           gp_update_max_s=float(np.max(gt[:, 2] + gt[:, 3])) if len(gt) else None),
                   raster=dict(pixels=PR, lines=R["r_lines"], scan_s=R["r_scan_s"], wall_s=R["r_wall_s"],
                               on_object_frac=float(onR[-1] / PR), rmse_final_nm=rmse(recR[-1], REF)),
                   enrichment=float((onG[-1] / PG) / max(onR[-1] / PR, 1e-9)),
                   budgets_px=budgets, rmse_at_budgets_gp=[rmse(x_, REF) for x_ in recG],
                   rmse_at_budgets_raster=[rmse(x_, REF) for x_ in recR],
                   rmse_gp_mean_at_budgets=[rmse(m_[0], REF) for m_ in maps],
                   time_s=times.tolist(), err_gp=errG, err_raster=errR, iou_gp=iouG, iou_raster=iouR)
    (out / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=1))
    np.savez(out / f"{tag}_curves.npz", times=times, errG=errG, errR=errR, cG=cG[::1000], onG=onG[::1000],
             cR=cR[::1000], onR=onR[::1000], REF=REF, recG=recG[-1], recR=recR[-1], ext=ext)
    print(json.dumps({k: v for k, v in metrics.items() if k in ("gp", "raster", "enrichment")}, indent=1))
    print("figure:", out / f"{tag}_fig4.png")

    if a.gif:
        make_gif(R, REF, ext, xl, yl, cG, cR, tG, tR, b, out / f"{tag}_animation.gif", grid)


def make_gif(R, REF, ext, xl, yl, cG, cR, tG, tR, b, path, grid, n_frames=40, fps=5):
    from matplotlib.animation import PillowWriter
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.2), layout="constrained")
    w = PillowWriter(fps=fps)
    PG = int(cG[-1])
    with w.saving(fig, str(path), dpi=60):
        for f in list(np.linspace(0.03, 1, n_frames)) + [1.0] * 6:
            p = max(1, int(f * PG))
            u = int(np.searchsorted(cG, p)) + 1; v = min(int(np.searchsorted(cR, p)) + 1, len(cR))
            mu, sd = gp_maps(R, u, grid)
            for a_ in axs:
                a_.clear()
            ux, uy = to_um(R, R["x"][:u:40], R["y"][:u:40])
            axs[0].imshow(REF, origin="lower", extent=ext, cmap="gray", alpha=0.5, vmin=-15, vmax=220)
            vpath(axs[0], ux, uy, len(R["x"][::40])); axs[0].plot(ux[-1:], uy[-1:], "o", color="r", ms=6, zorder=5)
            axs[0].set_title(f"GP path: {p:,} px")
            tvm = tv(binned(R["x"][:u], R["y"][:u], R["zl"][:u], b), 400)
            axs[1].imshow(tvm, origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135); axs[1].set_title(f"map (TV): {rmse(tvm, REF):.1f} nm")
            axs[2].imshow(sd.reshape(N, N), origin="lower", extent=ext, cmap="inferno"); axs[2].set_title("GP sigma")
            rx_, ry_ = to_um(R, R["rx"][:v:40], R["ry"][:v:40])
            axs[3].imshow(REF, origin="lower", extent=ext, cmap="gray", alpha=0.5, vmin=-15, vmax=220)
            vpath(axs[3], rx_, ry_, len(R["rx"][::40])); axs[3].plot(rx_[-1:], ry_[-1:], "o", color="r", ms=6, zorder=5); axs[3].set_title(f"raster: {int(cR[v - 1]):,} px")
            for a_ in axs:
                a_.set_xlim(*xl); a_.set_ylim(*yl); a_.set_aspect("equal"); a_.set_xlabel("X (um)")
            fig.suptitle(f"pixels scanned: {p:,} of the {N} x {N} grid ({p / N ** 2:.1%}) -- GP path vs raster at the same pixels", fontsize=12)
            w.grab_frame()
    plt.close(fig)
    print("animation:", path)


if __name__ == "__main__":
    main()
