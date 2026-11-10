"""Animation of ONE self-driving run on the ground truth: path (viridis, start -> end, tip in
red), TV map of what was measured so far (+ RMSE), and the waveform being executed.

    python run_gif.py RUN.npz --gt GT.npz --out FILE.gif [--frames 60] [--title "..."]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fig4_gp as F  # noqa: E402
import hybrid_fig as H  # noqa: E402

plt = F.plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--gt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=60); ap.add_argument("--title", default=None); ap.add_argument("--fps", type=int, default=5)
    a = ap.parse_args()
    M = H.load(a.run)
    b = M["area"]
    d = np.load(a.run, allow_pickle=True)
    # waveform labels in time order (history steps), to caption each frame
    seg = d["segment_lengths"]; kinds = d["kinds"]
    ends = np.cumsum(seg) / M["rate"]
    D = np.load(a.gt, allow_pickle=True)
    gx, gy, gz = (D[k].astype(float) for k in ("xs", "ys", "zs"))
    REF = F.tv(F.binned(gx, gy, H.level_own(gx, gy, gz, b), b), 1500)
    X0, Y0 = F.to_um(M, np.array(b[:2]), np.array(b[2:]))
    ext = [X0[0], X0[1], Y0[0], Y0[1]]; xl = (min(ext[:2]), max(ext[:2])); yl = (min(ext[2:]), max(ext[2:]))
    from matplotlib.animation import PillowWriter
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.8), layout="constrained")
    T = M["t"][-1]
    w = PillowWriter(fps=a.fps)
    sub = 20
    with w.saving(fig, a.out, dpi=70):
        for f in list(np.linspace(0.02, 1, a.frames)) + [1.0] * 8:
            tt = f * T
            u = max(2, min(int(np.searchsorted(M["t"], tt)), len(M["t"])))
            k = int(np.searchsorted(ends, tt))
            kind = str(kinds[min(k, len(kinds) - 1)])
            for ax in axs:
                ax.clear()
            axs[0].imshow(REF, origin="lower", extent=ext, cmap="gray", vmin=-15, vmax=220)
            ux, uy = F.to_um(M, M["x"][:u:sub], M["y"][:u:sub]); F.vpath(axs[0], ux, uy, len(M["x"][::sub]))
            axs[0].plot(ux[-1:], uy[-1:], "o", color="r", ms=7, zorder=5)
            axs[0].set_title(f"path at the measured positions -- {tt:.1f} s of scanning\nnow: {kind}", fontsize=10)
            rec = F.tv(F.binned(M["x"][:u], M["y"][:u], M["z"][:u], b), 400)
            axs[1].imshow(rec, origin="lower", extent=ext, cmap="copper", vmin=-15, vmax=135)
            axs[1].contour(np.linspace(*ext[:2], F.N), np.linspace(*ext[2:], F.N), REF, [F.OBJ_NM], colors="w", linewidths=0.4)
            axs[1].set_title(f"TV map of the samples so far: {F.rmse(rec, REF):.1f} nm vs ground truth", fontsize=10)
            for ax in axs:
                ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.set_xlabel("X (um)")
            axs[0].set_ylabel("Y (um)")
            fig.suptitle(a.title or M["name"], fontsize=12)
            w.grab_frame()
    plt.close(fig)
    print("animation:", a.out)


if __name__ == "__main__":
    main()
