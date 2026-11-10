"""Tip / instrument state over the day, from every raster (same kind of scan): substrate noise
(robust sigma of the levelled substrate), object height, and edge sharpness (10-90% rise width
along the fast axis at the object sidewalls)."""
import glob, pathlib, sys, json, zipfile, re, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "..")
import hybrid_fig as H
rows = []
files = sorted(glob.glob("../scan_data/*.npz"), key=lambda f: re.findall(r"(\d{8}_\d{6})", f)[-1])
for f in files:
    z_ = zipfile.ZipFile(f)
    if not any(n.startswith("raster_xs") for n in z_.namelist()) and "ground_truth" not in f and "dense" not in f:
        continue
    d = np.load(f, allow_pickle=True)
    if "ground_truth" in f or "dense" in f:
        x, y, z = (d[k].astype(float) for k in ("xs", "ys", "zs")); b = tuple(d["area"]); dt = 25 / 1e6
    else:
        x, y, z = (d[k].astype(float) for k in ("raster_xs", "raster_ys", "raster_zs"))
        pr = json.loads(str(d["priors_json"])); c, h = pr["area_center_V"], pr["area_half_width_V"]; b = (c[0]-h, c[0]+h, c[1]-h, c[1]+h)
    zl = H.level_own(x, y, z, b)
    sub = zl[zl < 30]; s = 1.4826 * np.median(np.abs(sub - np.median(sub)))
    top = np.median(zl[zl > 80]) if (zl > 80).any() else np.nan
    # edge rise width: along the sample stream, crossings of 58 nm; width between 10% and 90% of top
    ds = np.hypot(np.diff(x), np.diff(y)); sarc = np.r_[0, np.cumsum(ds)]
    up = np.flatnonzero((zl[:-1] < 58) & (zl[1:] >= 58))
    widths = []
    for i in up[:400]:
        j0 = i; j1 = i
        while j0 > 0 and zl[j0] > 0.1 * top and i - j0 < 2000: j0 -= 1
        while j1 < len(zl) - 1 and zl[j1] < 0.9 * top and j1 - i < 2000: j1 += 1
        w = sarc[j1] - sarc[j0]
        if 0 < w < 0.5: widths.append(w)
    t = re.findall(r"(\d{8}_\d{6})", f)[-1]
    rows.append(dict(file=pathlib.Path(f).name, time=t[9:11] + ":" + t[11:13], substrate_sigma_nm=float(s), object_height_nm=float(top),
                     edge_width_nm=float(np.median(widths) * 1876) if widths else np.nan, n_edges=len(widths)))
    print(f"{rows[-1]['time']}  sigma {s:4.1f} nm  height {top:6.1f} nm  edge 10-90% {rows[-1]['edge_width_nm']:6.0f} nm ({len(widths)} edges)  {pathlib.Path(f).name[:60]}")
pathlib.Path("tip_check.json").write_text(json.dumps(rows, indent=1))
