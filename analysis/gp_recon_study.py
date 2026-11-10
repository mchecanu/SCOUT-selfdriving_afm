"""Which reconstruction of GP-path data is best -- on the instrument's own data.

    python gp_recon_study.py RUN.npz [RUN2.npz ...]

Hold-out: the equivalent raster's pixels that the GP path never visited (both levelled with
one robust plane fitted to the GP data, as a user would). Variants:
  saved      the run's own recon_mean (planner GP: kernel frozen after freeze_after_n points)
  frozen*    the planner's kernel recipe at the end of the run (as saved, recomputed)
  relearn_*  kernel re-learned on ALL the data (stratified subset: half random, half the
             largest |residuals| -- the objects), Matern nu = 5/2, 3/2, 1/2; sparse predictor
  tv / linear  inpainting of the binned GP samples (reconstruction.py)
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402

from selfdriving_afm.gp_models import make_kernel, _kernel_params, _SparseSolver, robust_plane, NU  # noqa: E402
from selfdriving_afm.reconstruction import bin_image, tv_inpaint, linear_inpaint  # noqa: E402

N = 200
OBJ = 58.0


def plane_lev(x, y, z):
    X = np.column_stack([x, y])
    c = robust_plane(X[::50], z[::50])
    return c


def relearn(X, y, ls0, ls_b, kind, n_sub=1500, seed=0):
    rng = np.random.default_rng(seed)
    r = y - np.median(y)
    half = n_sub // 2
    big = np.argsort(-np.abs(r))[: 4 * half]
    idx = np.unique(np.r_[rng.choice(len(y), half, replace=False), rng.choice(big, half, replace=False)])
    gp = GaussianProcessRegressor(kernel=make_kernel(ls0, ls_b, kind), alpha=1e-8, normalize_y=True,
                                  n_restarts_optimizer=2).fit(X[idx], y[idx])
    return _kernel_params(gp.kernel_)


def sparse_predict(X, y, c, ls, noise, nu, grid, spacing):
    ym, ys = float(np.mean(y)), float(np.std(y)) or 1.0
    s = _SparseSolver(X, (y - ym) / ys, c, ls, noise, spacing, nu)
    mu, sd = s.predict(grid)
    return mu * ys + ym, sd * ys


def study(path):
    d = np.load(path, allow_pickle=True)
    pr = json.loads(str(d["priors_json"]))
    cc, h = pr["area_center_V"], pr["area_half_width_V"]
    b = (cc[0] - h, cc[0] + h, cc[1] - h, cc[1] + h)
    pts = d["points"]
    rate, v = float(d["sample_rate_hz"]), float(d["tip_velocity_V_s"])
    stride = max(1, int(round(0.005 / (v / rate))))
    x, y, z = pts[::stride, 0], pts[::stride, 1], pts[::stride, 2]
    cpl = plane_lev(x, y, z)
    lev = lambda X_, Y_, Z_: Z_ - (cpl[0] + cpl[1] * X_ + cpl[2] * Y_)
    zl = lev(x, y, z)
    zl -= np.median(zl[zl < np.percentile(zl, 50) + 25])
    rx, ry, rz = (d[k].astype(float) for k in ("raster_xs", "raster_ys", "raster_zs"))
    rzl = lev(rx, ry, rz); rzl -= np.median(rzl[rzl < np.percentile(rzl, 50) + 25])
    BR, _ = bin_image(np.column_stack([rx, ry]), rzl, N, b)
    # all GP samples (not only the training stride) for the binned images
    xa, ya = pts[:, 0], pts[:, 1]
    za = lev(xa, ya, pts[:, 2]); za -= np.median(zl[zl < np.percentile(zl, 50) + 25]) * 0 + np.median(za[za < np.percentile(za, 50) + 25])
    BG, _ = bin_image(np.column_stack([xa, ya]), za, N, b)
    hold = np.isfinite(BR) & ~np.isfinite(BG)
    print(f"\n{pathlib.Path(path).name}: GP {np.isfinite(BG).sum():,} px, raster {np.isfinite(BR).sum():,} px, "
          f"hold-out {hold.sum():,} px ({(BR[hold] > OBJ).mean():.1%} on objects); training n = {len(x):,}")
    gx = np.linspace(b[0], b[1], N); gy = np.linspace(b[2], b[3], N)
    GX, GY = np.meshgrid(gx, gy); grid = np.column_stack([GX.ravel(), GY.ravel()])
    fmin, fmax = pr["feature_min_V"], pr["feature_max_V"]
    ls_b = (0.25 * fmin, 2 * fmax)

    out = {}

    def score(name, img, t=0.0, extra=""):
        e = img[hold] - BR[hold]
        m_obj = BR[hold] > OBJ
        p, q = img[hold] > OBJ, BR[hold] > OBJ
        iou = (p & q).sum() / max((p | q).sum(), 1)
        out[name] = dict(rmse=float(np.sqrt(np.mean(e ** 2))), rmse_obj=float(np.sqrt(np.mean(e[m_obj] ** 2))) if m_obj.any() else None,
                         iou=float(iou), s=t, note=extra)
        print(f"  {name:22s} RMSE {out[name]['rmse']:6.1f} nm  (on objects {out[name]['rmse_obj'] or 0:6.1f})  IoU {iou:.2f}  {t:5.1f} s  {extra}")
        return img

    imgs = {}
    # saved planner map (300 x 300 over recon_bounds) -> resample to our grid
    if "recon_mean" in d.files:
        from scipy.interpolate import RegularGridInterpolator
        RX, RY, RM = d["recon_X"], d["recon_Y"], d["recon_mean"]
        f = RegularGridInterpolator((RY[:, 0], RX[0]), RM, bounds_error=False, fill_value=np.nan)
        sm = f(np.column_stack([GY.ravel(), GX.ravel()])).reshape(N, N)
        sm = lev(GX, GY, sm); sm -= np.median(za[za < np.percentile(za, 50) + 25]) * 0 + np.nanmedian(sm[sm < np.nanpercentile(sm, 50) + 25])
        imgs["saved"] = score("saved (planner GP)", sm)
    for kind in ("matern52", "matern32", "matern12"):
        t = time.time()
        c, ls, noise = relearn(np.column_stack([x, y]), zl, fmin, ls_b, kind)
        mu, sd = sparse_predict(np.column_stack([x, y]), zl, c, ls, noise, NU[kind], grid, min(ls / 4, 0.15))
        imgs[kind] = score(f"relearn_{kind}", mu.reshape(N, N), time.time() - t, f"ls {ls:.2f} V, noise/c {noise / c:.3g}")
    t = time.time(); imgs["tv"] = score("tv (mu 0.3)", tv_inpaint(BG, 0.3, 1500), time.time() - t)
    t = time.time(); imgs["linear"] = score("linear", linear_inpaint(BG), time.time() - t)
    return dict(file=str(path), results=out), imgs, (b, BG, BR, hold)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allr = []
    for p in sys.argv[1:]:
        r, imgs, (b, BG, BR, hold) = study(p)
        allr.append(r)
        fig, axs = plt.subplots(1, len(imgs) + 1, figsize=(3.6 * (len(imgs) + 1), 3.8), layout="constrained")
        axs[0].imshow(BR, origin="lower", cmap="copper", vmin=-15, vmax=135, interpolation="nearest"); axs[0].set_title("raster (hold-out ref)")
        for a, (k, im) in zip(axs[1:], imgs.items()):
            a.imshow(im, origin="lower", cmap="copper", vmin=-15, vmax=135); a.set_title(f"{k}: {r['results'][[q for q in r['results'] if q.startswith(k[:5]) or k in q][0]]['rmse']:.1f} nm", fontsize=9)
        for a in axs:
            a.set_xticks([]); a.set_yticks([])
        fig.savefig(HERE / f"recon_study_{pathlib.Path(p).stem}.png", dpi=80)
        plt.close(fig)
    (HERE / "recon_study.json").write_text(json.dumps(allr, indent=1))
