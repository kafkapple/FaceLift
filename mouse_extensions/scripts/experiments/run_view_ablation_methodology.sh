#!/usr/bin/env bash
# Phase 1 methodology validation — sequential 6-run experiment on a single GPU.
# Created: 2026-04-08
# Purpose: Quantify information leakage from random_view_selection=true in view ablation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=7 bash mouse_extensions/scripts/experiments/run_view_ablation_methodology.sh
#
# Run inside tmux: tmux new -s fixview "CUDA_VISIBLE_DEVICES=7 bash mouse_extensions/scripts/experiments/run_view_ablation_methodology.sh"
#
# Failure policy: stop on first error (set -e). Resume by re-running — completed runs
# will skip because train_gslrm.py checks existing checkpoint_dir.

set -eo pipefail
# NOTE: do not use `set -u` — conda activate triggers unbound-variable errors
# (ADDR2LINE, AR, etc. in binutils activate.d scripts).

FACELIFT_DIR="/home/joon/dev/FaceLift"
LOG_DIR="/node_data/joon/logs/FaceLift/phase1_methodology/$(date +%y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

# Sequential order: core experiments first, sensitivity/variance last.
# If GPU needs to be reclaimed mid-run, later runs are the lowest priority to drop.
RUNS=(
    "2view_fixed_widest_s1"     # 1. 2v upper bound (177°)
    "2view_fixed_narrowest_s1"  # 2. 2v lower bound (57.8°)
    "2view_fixed_midrange_s1"   # 3. 2v mid (120°) — cherry-pick guard
    "4view_fixed_s1"            # 4. 4v default (0,1,2,3)
    "4view_fixed_alt_s1"        # 5. 4v alternate (2,3,4,5) — subset sensitivity
    "2view_fixed_widest_s2"     # 6. widest seed=1337 — variance estimate
    "2view_random_rerun_s1"     # 7. 2v random matched control (lr=1e-7, same infra)
    "4view_random_rerun_s1"     # 8. 4v random matched control (lr=1e-7, same infra)
)

cd "$FACELIFT_DIR"
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

echo "======================================================================"
echo "Phase 1 Methodology Validation — Sequential Runner"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Log dir: $LOG_DIR"
echo "Total runs: ${#RUNS[@]}"
echo "Start: $(date)"
echo "======================================================================"

for i in "${!RUNS[@]}"; do
    NAME="${RUNS[$i]}"
    IDX=$((i + 1))
    LOG_FILE="$LOG_DIR/${IDX}_${NAME}.log"

    echo ""
    echo "------ [$IDX/${#RUNS[@]}] $NAME ------"
    echo "Log: $LOG_FILE"
    echo "Start: $(date)"

    python train_gslrm.py \
        -b configs/mouse/uniform/base_uniform_v2.yaml \
        -e "configs/mouse/uniform/${NAME}.yaml" \
        2>&1 | tee "$LOG_FILE"

    RC=${PIPESTATUS[0]}
    if [ "$RC" -ne 0 ]; then
        echo "FAILED: $NAME (exit $RC). Aborting sequence." >&2
        exit "$RC"
    fi
    echo "Done: $NAME @ $(date)"
done

echo ""
echo "======================================================================"
echo "All runs complete. End: $(date)"
echo "Checkpoints: /node_data/joon/checkpoints/FaceLift/gslrm/uniform_v2/"
echo "  2view_fixed_widest_s1/ _s2/"
echo "  2view_fixed_narrowest_s1/"
echo "  2view_fixed_midrange_s1/"
echo "  4view_fixed_s1/ _alt_s1/"
echo "======================================================================"
