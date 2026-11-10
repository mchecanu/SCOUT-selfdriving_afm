"""Quick look at every SCOUT dataset -- needs only numpy + matplotlib.

    python examples/quickstart.py                 (from the data folder's top level)
    python examples/quickstart.py data/05_gp_path_edge_focus/<file>.npz

Prints what each file holds and saves <name>_quicklook.png next to this script: the path
(coloured start -> end), the measured pixels binned on a 200 x 200 grid, and the same for the
run's equivalent raster (a raster with the same number of measured pixels) when there is one.
Coordinates are drive volts; `scout_datasets.to_um` converts them to micrometres.
"""
import csv
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scout_datasets as sd  # noqa: E402  (a copy of selfdriving_afm/datasets.py)

TOP = HERE.parent


def binned(x, y, z, b, n=200):
    """Mean z per pixel of an n x n grid over b = (xmin, xmax, ymin, ymax); NaN = not measured."""
    ix = np.clip(((x - b[0]) / (b[1] - b[0]) * n).astype(int), 0, n - 1)
    iy = np.clip(((y - b[2]) / (b[3] - b[2]) * n).astype(int), 0, n - 1)
    ok = (x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3]) & np.isfinite(z)
    s = np.bincount(iy[ok] * n + ix[ok], z[ok], n * n)
    c = np.bincount(iy[ok] * n + ix[ok], None, n * n)
    with np.errstate(invalid="ignore"):
        return (s / c).reshape(n, n)


def quicklook(path):
    run = sd.load(path)
    print(json.dumps(sd.summary(run)))
    b = run["bounds"]
    rows = [("self-driving path", run)]
    if run["raster"] is not None and run["format"] != sd.FORMAT_RASTER:
        rows.append(("equivalent raster", run["raster"]))
    fig, axs = plt.subplots(len(rows), 2, figsize=(9, 4.3 * len(rows)), squeeze=False)
    for i, (name, s) in enumerate(rows):
        x, y, z = (np.asarray(s[k]) for k in ("x", "y", "z"))
        k = max(1, len(x) // 40000)
        axs[i, 0].scatter(x[::k], y[::k], c=np.arange(len(x[::k])), s=0.3, cmap="viridis")
        axs[i, 0].set(aspect="equal", xlim=b[:2], ylim=b[2:], title=f"{name} (start purple -> end yellow)")
        B = binned(x, y, z, b)
        lo, hi = np.nanpercentile(B, [1, 99])
        axs[i, 1].imshow(B, origin="lower", extent=b, cmap="copper", vmin=lo, vmax=hi)
        axs[i, 1].set_title(f"binned z ({run['z_unit']}): {np.isfinite(B).mean():.0%} of pixels measured")
    fig.suptitle(pathlib.Path(path).name, fontsize=9)
    fig.tight_layout()
    out = HERE / (pathlib.Path(path).stem + "_quicklook.png")
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print("  ->", out)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        with open(TOP / "manifest.csv", newline="", encoding="utf-8") as f:
            files = [str(TOP / r["file"]) for r in csv.DictReader(f)]
    for p in files:
        quicklook(p)
