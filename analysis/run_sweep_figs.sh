#!/bin/bash
# per-run Fig. 4 panels for the clean lambda sweep, vs the 256-line ground truth, then the summary
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
GT=../scan_data/ground_truth_256lines_20261109_105402.npz
RUNS="../scan_data/scan_rerun_lam000_pure_gp_smooth_240V_gp_path_20261109_122204.npz ../scan_data/scan_rerun_lam025_pure_gp_smooth_240V_gp_path_20261109_123005.npz ../scan_data/scan_sweep_lam050_pure_gp_smooth_240V_gp_path_20261109_113659.npz ../scan_data/scan_sweep_lam075_pure_gp_smooth_240V_gp_path_20261109_114446.npz ../scan_data/scan_sweep_lam100_pure_gp_smooth_240V_gp_path_20261109_115212.npz"
for r in $RUNS; do
  echo "=== $r"; python fig4_gp.py "$r" --dense $GT --out sweep_runs 2>&1 | grep -E "pixels:|figure:|Error|Traceback|rmse_final"
done
python sweep_fig.py --runs $RUNS --gt $GT --metrics sweep_runs --out sweep 2>&1 | grep -v -i warn
