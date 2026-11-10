"""Build the Zenodo data folder: lite copies of the SCOUT datasets in ONE layout, a manifest
(with SHA-256), a preview image per dataset and the example scripts.

    python tools/make_zenodo_data.py --data-root "C:/.../Self-driving_afm_scan/Data" --out SCOUT_v4_data

--data-root: the folder with 4th/, 14th/, 16th/, SCOUT_v4_export_2026-11-10/ (see DATASETS below:
each source is the full file OR an existing lite copy; lite copies are re-saved unchanged in
resolution, full files are thinned with the stride given). Nothing in --data-root is modified.
Needs only numpy + matplotlib (+ this package).
"""
import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selfdriving_afm import datasets  # noqa: E402

EXP = "SCOUT_v4_export_2026-11-10/lite"
GT_EXTRA = {"z_unit": "nm", "z_label": "ZSensor height (nm)"}

# (destination folder, source relative to --data-root, stride, raster_stride, extra arrays, description)
DATASETS = [
    ("00_ground_truth", f"{EXP}/1_ground_truth/ground_truth_256lines_20261109_105402_lite.npz", 1, 1, GT_EXTRA,
     "256-line raster of the shapes sample (SiO2 disc + 4 hexagons, 116 nm): the ground truth for every 2026-11-09 score"),
    ("00_ground_truth", f"{EXP}/1_ground_truth/dense_reference_150lines_20261109_102815_lite.npz", 1, 1, GT_EXTRA,
     "150-line raster of the same area (earlier reference)"),
    ("01_grid_refine_stepwise", "4th/dataset_grid_refine/good grid_refine dataset/stepwise_L2_20261103_135746.npz", 20, 2, {},
     "grid_refine, stepwise: stage 1 survey spiral, stage 2 one spiral per object, stage 3 dense spiral per object "
     "(PyScanner spirals, no tip jumps) + its equivalent raster"),
    ("02_edge_explorer", "14th/dataset_edge_explorer_1/scan_edge_explorer_20261014_151014_lite.npz", 1, 1, {},
     "edge explorer (contour following) with closed-loop position sensors, 2026-10-14, + its equivalent raster"),
    ("03_edge_explorer_L2_contour", f"{EXP}/5_L2_contour/scan_L2_contour_edge_explorer_20261109_124234_lite.npz", 1, 1, {},
     "edge explorer (contour following) on the shapes sample, 2026-11-09 (5 objects, 2 outlines joined)"),
    ("04_gp_path_lambda_sweep", f"{EXP}/2_fig4_GP_lambda_sweep/scan_rerun_lam000_pure_gp_smooth_240V_gp_path_20261109_122204_lite.npz", 1, 1, {},
     "GP pathfinder, pure_gp_smooth, 240 V budget, lambda = 0 (pure exploration)"),
    ("04_gp_path_lambda_sweep", f"{EXP}/2_fig4_GP_lambda_sweep/scan_rerun_lam025_pure_gp_smooth_240V_gp_path_20261109_123005_lite.npz", 1, 1, {},
     "GP pathfinder, lambda = 0.25"),
    ("04_gp_path_lambda_sweep", f"{EXP}/2_fig4_GP_lambda_sweep/scan_sweep_lam050_pure_gp_smooth_240V_gp_path_20261109_113659_lite.npz", 1, 1, {},
     "GP pathfinder, lambda = 0.5"),
    ("04_gp_path_lambda_sweep", f"{EXP}/2_fig4_GP_lambda_sweep/scan_sweep_lam075_pure_gp_smooth_240V_gp_path_20261109_114446_lite.npz", 1, 1, {},
     "GP pathfinder, lambda = 0.75"),
    ("04_gp_path_lambda_sweep", f"{EXP}/2_fig4_GP_lambda_sweep/scan_sweep_lam100_pure_gp_smooth_240V_gp_path_20261109_115212_lite.npz", 1, 1, {},
     "GP pathfinder, lambda = 1 (variance reduction only where the GP sees edges)"),
    ("05_gp_path_edge_focus", f"{EXP}/3_fig4_GP_gamma2/scan_edgefocus_gamma2_lam100_pure_gp_smooth_240V_gp_path_20261109_143003_lite.npz", 1, 1, {},
     "GP pathfinder, lambda = 1, gamma = 2 (preset pure_gp_edge_focus): the most edge-focused GP run"),
    ("06_gp_path_edges_preset", f"{EXP}/4_GP_first_draft/scan_hero_pure_gp_edges_smooth_240V_gp_path_20261109_103432_lite.npz", 1, 1, {},
     "GP pathfinder, preset pure_gp_edges_smooth"),
    ("07_hybrids", f"{EXP}/6_hybrids/scan_hybrid5fig3_lam000_gpL3_edgeL2_gp_edge_explorer_20261109_132517_lite.npz", 1, 1, {},
     "hybrid gp_edge_explorer: GP exploration between objects + edge tracker on them (5/5 outlines)"),
    ("07_hybrids", f"{EXP}/6_hybrids/scan_hybrid_gridL0_tracker_v2_grid_tracker_20261109_180209_lite.npz", 1, 1, {},
     "hybrid grid_tracker: sparse survey spiral + edge tracker on every outline it crossed (5/5)"),
    ("07_hybrids", f"{EXP}/6_hybrids/scan_hybrid_gp_L2spiral_v6_gp_spiral_20261109_231927_lite.npz", 1, 1, {},
     "hybrid gp_spiral: GP exploration between objects + object and dense spirals on each (5/5 mapped)"),
    ("08_pfm_domain_walls", "16th/dataset_DW_explorer_2/scan_domain_wall_explorer_20261016_151545_lite.npz", 1, 1, {},
     "PFM: domain-wall explorer following walls by phase flip / amplitude dip, 2026-10-16 (amplitude + phase along the path)"),
    ("08_pfm_domain_walls", "4th/good PFM domain-wall dataset 20261104/pfm_run2_domain_wall_explorer_20261104_134512.npz", 2, 2, {},
     "PFM: domain-wall explorer, 2026-11-04 run 2 (amplitude + phase) + its equivalent raster with both channels"),
]

EXAMPLES = ROOT / "tools" / "zenodo_examples"


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def preview(run, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from selfdriving_afm.reconstruction import bin_image
    b = run["bounds"]
    rows = [("path", run)] + ([("equivalent raster", run["raster"])] if run["raster"] is not None
                              and run["format"] != datasets.FORMAT_RASTER else [])
    fig, axs = plt.subplots(len(rows), 2, figsize=(9, 4.3 * len(rows)), squeeze=False)
    for i, (name, src) in enumerate(rows):
        x, y, z = (np.asarray(src[k]) for k in ("x", "y", "z"))
        k = max(1, len(x) // 40000)
        axs[i, 0].scatter(x[::k], y[::k], c=np.arange(len(x[::k])), s=0.3, cmap="viridis")
        axs[i, 0].set_aspect("equal"); axs[i, 0].set_xlim(b[0], b[1]); axs[i, 0].set_ylim(b[2], b[3])
        axs[i, 0].set_title(f"{name}: start (purple) -> end (yellow)", fontsize=9)
        B, _ = bin_image(np.column_stack([x, y]), z, 200, b)
        lo, hi = np.nanpercentile(B, [1, 99]) if np.isfinite(B).any() else (None, None)
        axs[i, 1].imshow(B, origin="lower", extent=b, cmap="copper", vmin=lo, vmax=hi)
        axs[i, 1].set_title(f"binned z ({run['z_unit']}), {np.isfinite(B).mean():.0%} of 200x200 px", fontsize=9)
        for a in axs[i]:
            a.set_xlabel("X drive (V)"); a.set_ylabel("Y drive (V)")
    fig.suptitle(f"{pathlib.Path(run['file']).name}\n{run['modality']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=70)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-previews", action="store_true")
    a = ap.parse_args(argv)
    src_root, out = pathlib.Path(a.data_root), pathlib.Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    rows = []
    for folder, rel, st, rst, extra, desc in DATASETS:
        src = src_root / rel
        if not src.exists():
            print("MISSING (skipped):", src)
            continue
        stem = src.stem[:-5] if src.stem.endswith("_lite") else src.stem
        dst = out / "data" / folder / f"{stem}_lite.npz"
        datasets.make_lite(src, dst, stride=st, raster_stride=rst, extra=extra)
        run = datasets.load(dst)
        sm = datasets.summary(run)
        if not a.no_previews:
            (out / "previews").mkdir(exist_ok=True)
            preview(run, out / "previews" / f"{stem}.png")
        rows.append(dict(file=str(dst.relative_to(out)).replace("\\", "/"), folder=folder, mode=run["modality"],
                         description=desc, samples=sm["samples"], stride=sm["stride"],
                         samples_original=sm["samples_original"], raster_samples=(sm["equivalent_raster"] or {}).get("samples", 0),
                         channels=" ".join(sm["channels"]), z_unit=run["z_unit"],
                         um_per_V=" ".join(f"{v:.4f}" for v in (run["um_per_V"] if run["um_per_V"] is not None else [])),
                         size_MB=round(dst.stat().st_size / 1e6, 2), sha256=sha256(dst), source=src.name))
        print(f"  {rows[-1]['file']}: {rows[-1]['size_MB']} MB, {sm['samples']:,} samples (1 in {sm['stride']})")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out / "manifest.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    if EXAMPLES.is_dir():
        (out / "examples").mkdir(exist_ok=True)
        for q in EXAMPLES.iterdir():
            if q.is_file():
                shutil.copy2(q, out / "examples" / q.name)
    shutil.copy2(ROOT / "selfdriving_afm" / "datasets.py", out / "examples" / "scout_datasets.py")
    shutil.copy2(ROOT / "tools" / "zenodo_README.md", out / "README.md")
    print(f"{len(rows)} datasets, {sum(r['size_MB'] for r in rows):.0f} MB ->", out)


if __name__ == "__main__":
    main()
