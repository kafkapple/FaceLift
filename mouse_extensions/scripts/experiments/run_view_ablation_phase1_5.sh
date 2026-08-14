#!/usr/bin/env bash
# Phase 1.5 methodology validation follow-up — matched controls + variance fix.
# Created: 2026-04-08
# Addresses Research Analyst P0 findings from post-launch audit of Phase 1:
#   1. Random baseline needs matched DataLoader config (num_workers=4, prefetch=2)
#   2. Latent bug `dataset.num_input_views default=1` affected old baselines
#   3. Seed asymmetry: narrowest needs 2 seeds for variance estimate
#
# PREREQUISITE: Phase 1 runner (run_view_ablation_methodology.sh) must be complete.
# Check: tmux has-session -t fixview → should fail (session gone)
#        /tmp/fixview_methodology_*/6_2view_fixed_widest_s2.log → exists with "Done"
#
# Usage:
#   tmux new -d -s fixview15 "CUDA_VISIBLE_DEVICES=7 bash mouse_extensions/scripts/experiments/run_view_ablation_phase1_5.sh"

set -eo pipefail

FACELIFT_DIR="/home/joon/dev/FaceLift"
LOG_DIR="/tmp/fixview_phase1_5_$(date +%y%m%d_%H%M)"
mkdir -p "$LOG_DIR"

RUNS=(
    "2view_random_rerun_s1"       # matched 2v random control (num_workers=4, no latent bug)
    "4view_random_rerun_s1"       # matched 4v random control
    "2view_fixed_narrowest_s2"    # narrowest 2nd seed (power for noisy condition)
)

cd "$FACELIFT_DIR"
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

echo "======================================================================"
echo "Phase 1.5 Methodology Validation Follow-up — Sequential"
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
echo "Phase 1.5 complete. End: $(date)"
echo "======================================================================"
