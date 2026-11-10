"""Maps as in the paper -- with the SCOUT package (pip install selfdriving-afm, or the GitHub repo):
substrate plane removed (robust lower envelope), total-variation (TV) reconstruction, the run next
to its equivalent raster, axes in micrometres.

    python examples/reconstruct_with_scout.py data/07_hybrids/<file>.npz [--channel phase] [--method tv]

The same is available from an LLM client through the package's MCP server:
"reconstruct_scan(npz=..., method='tv', units='um')" (README "MCP server").
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from selfdriving_afm import datasets
from selfdriving_afm.config import LiveViewConfig
from selfdriving_afm.reconstruction import flattened_reconstruction, bin_image, inpaint_binned

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--channel", default="z", help="z, or a PFM channel: amplitude / phase")
ap.add_argument("--method", default="tv", help="tv, linear, compressed_sensing, fast, gp")
ap.add_argument("--pixels", type=int, default=200)
a = ap.parse_args()

run = datasets.load(a.npz)
b = run["bounds"]
cfg = LiveViewConfig()
cfg.reconstruction, cfg.n_image_pixels, cfg.tv_mu = a.method, a.pixels, 0.3
rows = [("SCOUT", run)] + ([("equivalent raster", run["raster"])] if run["raster"] is not None else [])
ext = b
if run["um_per_V"] is not None:
    ux, uy = datasets.to_um(run, b[:2], b[2:])
    ext = (ux[0], ux[1], uy[0], uy[1])
fig, axs = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.6), squeeze=False)
for ax, (name, s) in zip(axs[0], rows):
    z = s["z"] if a.channel == "z" else s.get("channels", {})[a.channel]
    pts = np.column_stack([s["x"], s["y"]])
    if a.channel == "z":
        binned, recon, plane = flattened_reconstruction(pts, np.asarray(z), b, cfg)
    else:
        binned, _ = bin_image(pts, np.asarray(z), a.pixels, b)
        recon = inpaint_binned(binned, b, cfg)
    im = ax.imshow(recon, origin="lower", extent=ext, cmap="copper")
    ax.set_title(f"{name}: {a.method} of {np.isfinite(binned).mean():.0%} measured px")
    ax.set_xlabel("x (um)" if ext is not b else "X (V)")
    fig.colorbar(im, ax=ax, label=f"{a.channel} ({run['z_unit'] if a.channel == 'z' else ''})")
fig.tight_layout()
out = pathlib.Path(a.npz).with_suffix("").name + f"_{a.method}_{a.channel}.png"
fig.savefig(out, dpi=100)
print("saved", out)
