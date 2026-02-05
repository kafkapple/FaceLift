#!/bin/bash
# =============================================================================
# View Ablation: Inference-time (기존 4-view 모델로 빠르게 탐색)
# =============================================================================
# 사용법: ./scripts/run_view_ablation_inference.sh [2|3|5|6|all]

set -e
cd /home/joon/dev/FaceLift

PYTHON=/home/joon/anaconda3/envs/facelift/bin/python
DATA_DIR=/home/joon/data/preprocessed/FaceLift_mouse/M5
SPLIT=$DATA_DIR/data_mouse_t2_test.txt
NUM_FRAMES=50
MODEL=M5t2
OUTPUT_BASE=outputs/eval/view_ablation_inference

run_ablation() {
    local NUM_VIEWS=$1
    local GPU=$2
    local OUTPUT_DIR=${OUTPUT_BASE}/${NUM_VIEWS}view
    
    echo "=== Running ${NUM_VIEWS}-view ablation on GPU $GPU ==="
    
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
        --data_dir $DATA_DIR \
        --split $SPLIT \
        --num_frames $NUM_FRAMES \
        --model $MODEL \
        --num_input_views $NUM_VIEWS \
        --skip_preprocess --prefer_ema \
        --output_dir $OUTPUT_DIR
}

case "$1" in
    2) run_ablation 2 5 ;;
    3) run_ablation 3 6 ;;
    5) run_ablation 5 7 ;;
    6) run_ablation 6 5 ;;
    all)
        echo "Running 2, 3, 5, 6 view ablations in parallel (GPU 5,6,7)..."
        run_ablation 2 5 &
        run_ablation 3 6 &
        run_ablation 5 7 &
        wait
        run_ablation 6 5 &
        wait
        echo "All inference ablations done!"
        ;;
    *)
        echo "Usage: $0 [2|3|5|6|all]"
        echo "  4-view baseline: already in h1_diagnosis_M5t2/h1b_gslrm_test"
        exit 1
        ;;
esac
