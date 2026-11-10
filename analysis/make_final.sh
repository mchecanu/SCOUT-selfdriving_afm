#!/bin/bash
# final figures + animations (2026-11-09), all vs the 256-line ground truth
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
GT=../scan_data/ground_truth_256lines_20261109_105402.npz
D=../scan_data
mkdir -p final
python fig4_gp.py $D/scan_edgefocus_gamma2_lam100_pure_gp_smooth_240V_gp_path_20261109_143003.npz --dense $GT --out final/fig4_gamma2 --gif > final/log_gamma2.txt 2>&1 &
python fig4_gp.py $D/scan_sweep_lam050_pure_gp_smooth_240V_gp_path_20261109_113659.npz --dense $GT --out final/fig4_lam050 --gif > final/log_lam050.txt 2>&1 &
python hybrid_fig.py --hybrid $D/scan_hybrid5fig3_lam000_gpL3_edgeL2_gp_edge_explorer_20261109_132517.npz --l2 $D/scan_L2_contour_edge_explorer_20261109_124234.npz \
   --gp $D/scan_sweep_lam050_pure_gp_smooth_240V_gp_path_20261109_113659.npz --gt $GT --out final/hybrid --gif > final/log_hybrid.txt 2>&1 &
wait
echo ALL FINAL FIGURES DONE
