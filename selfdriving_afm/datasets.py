"""
One loader for every SCOUT data file: full runs saved by `modes.save`, the "lite" copies
(Zenodo / make_lite.py: `pts` = 1 sample in `pts_stride`), the stepwise grid_refine dataset
(`stepwise.StepwiseRun.save_dataset`), and plain rasters (ground truth / reference: xs, ys, zs).

    from selfdriving_afm import datasets
    run = datasets.load("scan_edgefocus_gamma2_..._lite.npz")
    run["x"], run["y"], run["z"]          # the path at the MEASURED positions (sensors if recorded), V / z_unit
    run["raster"]                         # the equivalent raster (xs, ys, zs) or None
    run["channels"]                       # PFM channels along the path, e.g. {"amplitude": ..., "phase": ...}
    datasets.to_um(run, run["x"], run["y"])   # drive V -> um (the run's sensor calibration)

Only numpy is needed. Nothing is written.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import numpy as np

FORMAT_LITE = "lite"
FORMAT_FULL = "full"
FORMAT_STEPWISE = "stepwise"
FORMAT_RASTER = "raster"

# files written with numpy 2 pickle their object arrays as "numpy._core...": let numpy 1.x read them
if not hasattr(np, "_core"):          # pragma: no cover - numpy 1.x only
    import importlib
    import numpy.core as _c
    sys.modules.setdefault("numpy._core", _c)
    for _m in ("multiarray", "numeric", "_multiarray_umath", "fromnumeric", "records", "defchararray"):
        try:
            sys.modules.setdefault(f"numpy._core.{_m}", importlib.import_module(f"numpy.core.{_m}"))
        except ImportError:
            pass


def _scalar(v):
    v = np.asarray(v)
    if v.shape == ():
        v = v.item()
    return v.decode() if isinstance(v, bytes) else v


def _json(d, key):
    if key not in d.files:
        return None
    try:
        return json.loads(str(_scalar(d[key])))
    except Exception:
        return None


def detect_format(d) -> str:
    f = set(d.files)
    if "pts" in f:
        return FORMAT_LITE
    if "scans" in f and any(k.endswith("_x_cmd") for k in f):
        return FORMAT_STEPWISE
    if "points" in f or "x_cmd" in f:
        return FORMAT_FULL
    if {"xs", "ys", "zs"} <= f:
        return FORMAT_RASTER
    raise ValueError("not a SCOUT data file (no pts / points / xs,ys,zs arrays)")


def _raster(d, stride_key="raster_stride"):
    if "raster_xs" in d.files:
        x, y, z = (np.asarray(d[k], float) for k in ("raster_xs", "raster_ys", "raster_zs"))
        pos = "sensor"
    elif "raster_x" in d.files:
        x, y, z = (np.asarray(d[k], float) for k in ("raster_x", "raster_y", "raster_z"))
        pos = "commanded"
    else:
        return None
    stride = 1
    if stride_key in d.files:
        stride = int(_scalar(d[stride_key]))
    elif "_lite_block__raster_xs" in d.files:
        stride = int(_scalar(d["_lite_block__raster_xs"]))
    out = dict(x=x, y=y, z=z, positions=pos, stride=stride)
    for k in d.files:            # PFM channels of the raster: raster_ch_<name>_vs (sensor) / _v
        if k.startswith("raster_ch_") and (k.endswith("_vs") or k.endswith("_v")):
            name = k[len("raster_ch_"):].rsplit("_", 1)[0]
            if k.endswith("_vs") or name not in out.get("channels", {}):
                out.setdefault("channels", {})[name] = np.asarray(d[k], float)
    return out


def load(path, keep_open: bool = False) -> dict:
    """Load any SCOUT .npz as one normalized dict (see the module docstring). Keys:
    file, format, modality, x, y, z, positions ("sensor" / "commanded"), stride (1 = every
    sample), n_samples_original, channels, raster, bounds (recon_bounds or the data extent),
    z_unit, um_per_V, um_at_0V, priors, features, images (every 2-D array in the file),
    meta (the other small arrays)."""
    path = pathlib.Path(path)
    d = np.load(path, allow_pickle=True)
    fmt = detect_format(d)
    out = dict(file=str(path), format=fmt, modality=str(_scalar(d["modality"])) if "modality" in d.files else None,
               channels={}, raster=None, stride=1)
    if fmt == FORMAT_LITE:
        p = np.asarray(d["pts"], float)
        has_cal = "sensor_calibration_json" in d.files or "sensor_cal" in d.files
        out.update(x=p[:, 0], y=p[:, 1], z=p[:, 2], positions="sensor" if has_cal else "measured",
                   stride=int(_scalar(d["pts_stride"])) if "pts_stride" in d.files else 1,
                   n_samples_original=int(_scalar(d["pts_n"])) if "pts_n" in d.files else len(p))
        if "pts_level" in d.files:              # stepwise grid_refine: stage / scan of every sample
            out["level"], out["scan_index"] = np.asarray(d["pts_level"]), np.asarray(d["pts_scan_index"])
            out["scan_names"] = [str(v) for v in d["scans"]] if "scans" in d.files else None
        for k in d.files:
            if k.startswith("pts_ch_"):
                out["channels"][k[len("pts_ch_"):]] = np.asarray(d[k], float)
        out["raster"] = _raster(d)
    elif fmt == FORMAT_FULL:
        if "x_sensor" in d.files:
            x, y, pos = d["x_sensor"], d["y_sensor"], "sensor"
        elif "points" in d.files:
            pts = d["points"]
            x, y, pos = pts[:, 0], pts[:, 1], "measured"
        else:
            x, y, pos = d["x_cmd"], d["y_cmd"], "commanded"
        z = d["z"] if "z" in d.files else d["points"][:, 2]
        n = min(len(x), len(z))
        out.update(x=np.asarray(x[:n], float), y=np.asarray(y[:n], float), z=np.asarray(z[:n], float),
                   positions=pos, n_samples_original=n)
        if "_lite_block__points" in d.files:      # older lite copies: block means of N samples
            out["stride"] = int(_scalar(d["_lite_block__points"]))
            out["n_samples_original"] = int(_scalar(d["_lite_len__points"]))
            out["lite_block_means"] = True
        if "x_cmd" in d.files:
            out["x_cmd"], out["y_cmd"] = np.asarray(d["x_cmd"][:n], float), np.asarray(d["y_cmd"][:n], float)
        for k in d.files:
            if k.startswith("ch_"):
                out["channels"][k[3:]] = np.asarray(d[k][:n], float)
        out["raster"] = _raster(d)
    elif fmt == FORMAT_STEPWISE:
        names = [str(s) for s in d["scans"]]
        xs, ys, zs, lv, sc = [], [], [], [], []
        for i, nme in enumerate(names):
            ok = np.asarray(d[f"{nme}_ok"], bool) if f"{nme}_ok" in d.files else slice(None)
            zkey = f"{nme}_z_levelled_nm" if f"{nme}_z_levelled_nm" in d.files else f"{nme}_z_nm"
            zz = np.asarray(d[zkey], float)
            # z_levelled is stored for the ok samples only in some versions
            xx, yy = np.asarray(d[f"{nme}_x_sensor"], float), np.asarray(d[f"{nme}_y_sensor"], float)
            if len(zz) != len(xx):
                xx, yy = xx[ok], yy[ok]
            xs.append(xx); ys.append(yy); zs.append(zz)
            lv.append(np.full(len(zz), int(_scalar(d[f"{nme}_level"])) if f"{nme}_level" in d.files else -1, np.int8))
            sc.append(np.full(len(zz), i, np.int16))
        out.update(x=np.concatenate(xs), y=np.concatenate(ys), z=np.concatenate(zs), positions="sensor",
                   modality="grid_refine (stepwise)", scan_names=names, level=np.concatenate(lv),
                   scan_index=np.concatenate(sc), stride=int(_scalar(d["pts_stride"])) if "pts_stride" in d.files else 1)
        out["n_samples_original"] = int(_scalar(d["pts_n"])) if "pts_n" in d.files else len(out["z"])
        out["raster"] = _raster(d)
    else:   # plain raster (ground truth / reference)
        out.update(x=np.asarray(d["xs"], float), y=np.asarray(d["ys"], float), z=np.asarray(d["zs"], float),
                   positions="sensor", modality="raster", n_samples_original=len(d["zs"]))
    if "recon_bounds" in d.files:
        out["bounds"] = tuple(float(v) for v in d["recon_bounds"])
    elif "area" in d.files and np.asarray(d["area"]).size == 4:
        out["bounds"] = tuple(float(v) for v in d["area"])
    else:
        out["bounds"] = (float(np.nanmin(out["x"])), float(np.nanmax(out["x"])),
                         float(np.nanmin(out["y"])), float(np.nanmax(out["y"])))
    out["z_unit"] = str(_scalar(d["z_unit"])) if "z_unit" in d.files else "V"
    for k in ("um_per_V", "um_at_0V"):
        out[k] = np.asarray(d[k], float) if k in d.files else None
    out["priors"] = _json(d, "priors_json")
    out["features"] = _json(d, "features_json")
    out["images"] = {k: np.asarray(d[k]) for k in d.files if np.asarray(d[k]).ndim == 2 and k not in ("pts", "points")}
    meta = {}
    for k in d.files:
        try:
            a = d[k]
        except Exception:
            continue
        if a.size <= 64 and a.ndim <= 1 and k not in out["images"]:
            meta[k] = _scalar(a) if a.size == 1 else a.tolist()
    out["meta"] = meta
    if not keep_open:
        d.close()
    return out


def to_um(run: dict, x, y):
    """Drive volts -> micrometres with the run's sensor calibration (um_per_V, um_at_0V); the
    sign of um_per_V carries the Asylum orientation (x is mirrored vs the AR image)."""
    k = run.get("um_per_V")
    if k is None:
        raise ValueError("this file has no um_per_V (no position-sensor calibration)")
    c = run.get("um_at_0V")
    c = (0.0, 0.0) if c is None else c
    return c[0] + k[0] * np.asarray(x), c[1] + k[1] * np.asarray(y)


def summary(run: dict) -> dict:
    """Small JSON-able description of a loaded file."""
    r = run["raster"]
    return dict(file=pathlib.Path(run["file"]).name, format=run["format"], modality=run["modality"],
                samples=int(len(run["z"])), stride=run["stride"], samples_original=run.get("n_samples_original"),
                positions=run["positions"], bounds_V=[round(v, 4) for v in run["bounds"]], z_unit=run["z_unit"],
                channels=sorted(run["channels"]), equivalent_raster=None if r is None else
                dict(samples=int(len(r["z"])), stride=r["stride"], channels=sorted(r.get("channels", {}))),
                features=None if run["features"] is None else len(run["features"]),
                um_per_V=None if run["um_per_V"] is None else [round(float(v), 4) for v in run["um_per_V"]])


def make_lite(src, dst, stride: Optional[int] = None, raster_stride: int = 2, dtype=np.float32,
              extra: Optional[dict] = None) -> pathlib.Path:
    """Write a lite copy of any SCOUT file in the lite layout (`pts` = x, y, z at the measured
    positions, 1 sample in `stride`; PFM channels as pts_ch_<name>; the equivalent raster 1 in
    `raster_stride`; every small array and image kept). stride None: 10 for gp_path, 20 for
    longer runs, 1 for plain rasters. extra: arrays to add / override (e.g. {"z_unit": "nm"}).
    The source is never modified."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    run = load(src)
    d = np.load(src, allow_pickle=True)
    big = 200_000
    keep = {}
    for k in d.files:
        try:
            a = d[k]
        except Exception:
            continue
        if a.ndim <= 2 and a.size <= big and not k.startswith("_lite_"):
            keep[k] = a
    n = len(run["z"])
    if stride is None:
        stride = 1 if run["format"] == FORMAT_RASTER else (10 if run["modality"] == "gp_path" else 20)
    st = int(stride)
    if run["format"] == FORMAT_RASTER:
        for k in ("xs", "ys", "zs", "t"):
            if k in d.files:
                keep[k] = np.asarray(d[k])[::st].astype(dtype)
    else:
        keep["pts"] = np.column_stack([run["x"][::st], run["y"][::st], run["z"][::st]]).astype(dtype)
        keep["pts_index"] = np.arange(0, n, st, dtype=np.int64)
        keep["pts_n"] = np.int64(run.get("n_samples_original") or n)
        keep["pts_stride"] = np.int64(st * int(run.get("stride") or 1))
        for c, v in run["channels"].items():
            keep[f"pts_ch_{c}"] = v[::st].astype(dtype)
        if "level" in run:
            keep["pts_level"] = run["level"][::st]
            keep["pts_scan_index"] = run["scan_index"][::st]
        r = run["raster"]
        if r is not None:
            rs = int(raster_stride)
            keep["raster_xs"], keep["raster_ys"], keep["raster_zs"] = (r[k][::rs].astype(dtype) for k in ("x", "y", "z"))
            for c, v in r.get("channels", {}).items():
                keep[f"raster_ch_{c}_vs"] = v[::rs].astype(dtype)
            if "raster_t" in d.files and len(d["raster_t"]) > 1:
                rt = d["raster_t"]
                keep["raster_dt"] = np.float64((rt[1] - rt[0]) * rs)
                keep["raster_n"] = np.int64(len(rt))
            keep["raster_stride"] = np.int64(rs * int(r.get("stride") or 1))
        if run["format"] == FORMAT_STEPWISE and "modality" not in keep:
            keep["modality"] = np.array("grid_refine_stepwise")
    for k in [k for k in keep if k.endswith(("_x_cmd", "_y_cmd", "_x_sensor", "_y_sensor", "_z_nm", "_ok",
                                             "_z_levelled_nm"))]:
        del keep[k]                           # stepwise per-scan arrays: in pts now
    d.close()
    keep.update({k: np.asarray(v) for k, v in (extra or {}).items()})
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **keep)
    return dst
