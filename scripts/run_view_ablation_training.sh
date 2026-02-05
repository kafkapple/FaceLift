#!/bin/bash
# =============================================================================
# View Ablation: Training (각 조건별 GS-LRM 재학습)
# =============================================================================
# 사용법: ./scripts/run_view_ablation_training.sh [2|3|5|6|all]

set -e
cd /home/joon/dev/FaceLift

source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

PYTHON=/home/joon/anaconda3/envs/facelift/bin/python

run_training() {
    local NUM_VIEWS=$1
    local GPU=$2
    local CONFIG=configs/experiments/view_ablation/E0_1_${NUM_VIEWS}view.yaml
    local LOG=logs/view_ablation_${NUM_VIEWS}view.log
    
    echo "=== Training ${NUM_VIEWS}-view model on GPU $GPU ==="
    echo "Config: $CONFIG"
    echo "Log: $LOG"
    
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_gslrm.py \
        -d M5t2 \
        -e view_ablation/E0_1_${NUM_VIEWS}view \
        > $LOG 2>&1
}

case "$1" in
    2) run_training 2 5 ;;
    3) run_training 3 6 ;;
    5) run_training 5 7 ;;
    6) run_training 6 5 ;;
    all)
        echo "Training 2, 3, 5, 6 view models in parallel..."
        echo "Note: 4-view baseline already exists (M5t2_E0_1_facelift)"
        run_training 2 5 &
        run_training 3 6 &
        run_training 5 7 &
        wait
        run_training 6 5 &
        wait
        echo "All training jobs done!"
        ;;
    *)
        echo "Usage: $0 [2|3|5|6|all]"
        echo "  4-view baseline: M5t2_E0_1_facelift (already trained)"
        exit 1
        ;;
esac
