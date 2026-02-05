#!/bin/bash
# =============================================================================
# H1 Experiments: E2E vs GS-LRM × Train vs Test
# =============================================================================
# Purpose: Diagnose whether quality degradation comes from:
#   - MVDiffusion (compare E2E vs GS-LRM)
#   - Test set generalization (compare Train vs Test)
#
# Usage: ./scripts/run_h1_experiments.sh [experiment_id]
#   experiment_id: h1a, h1b, h1c, h1d, or 'all'
# =============================================================================

set -e
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift

# Paths
GSLRM_CKPT="/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt"
MVDIFF_CKPT="/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000"
TRAIN_LIST="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt"
TEST_LIST="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt"
OUTPUT_BASE="/node_data/joon/outputs/FaceLift/eval/H1"
NUM_SAMPLES=50

run_h1a() {
    echo "=== H1a: GS-LRM + Train ==="
    python -m mouse_extensions.scripts.evaluate_test \
        --checkpoint $GSLRM_CKPT \
        --data_list $TRAIN_LIST \
        --num_samples $NUM_SAMPLES \
        --output_dir ${OUTPUT_BASE}/h1a_gslrm_train \
        --save_images
}

run_h1b() {
    echo "=== H1b: GS-LRM + Test ==="
    python -m mouse_extensions.scripts.evaluate_test \
        --checkpoint $GSLRM_CKPT \
        --data_list $TEST_LIST \
        --num_samples $NUM_SAMPLES \
        --output_dir ${OUTPUT_BASE}/h1b_gslrm_test \
        --save_images
}

run_h1c() {
    echo "=== H1c: E2E + Train ==="
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --mvdiff_checkpoint $MVDIFF_CKPT \
        --gslrm_checkpoint $GSLRM_CKPT \
        --data_list $TRAIN_LIST \
        --num_samples $NUM_SAMPLES \
        --output_dir ${OUTPUT_BASE}/h1c_e2e_train \
        --save_images --compute_metrics
}

run_h1d() {
    echo "=== H1d: E2E + Test ==="
    python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --mvdiff_checkpoint $MVDIFF_CKPT \
        --gslrm_checkpoint $GSLRM_CKPT \
        --data_list $TEST_LIST \
        --num_samples $NUM_SAMPLES \
        --output_dir ${OUTPUT_BASE}/h1d_e2e_test \
        --save_images --compute_metrics
}

case "${1:-all}" in
    h1a) run_h1a ;;
    h1b) run_h1b ;;
    h1c) run_h1c ;;
    h1d) run_h1d ;;
    all)
        run_h1a
        run_h1b
        run_h1c
        run_h1d
        echo "=== All H1 experiments complete ==="
        ;;
    *) echo "Usage: $0 [h1a|h1b|h1c|h1d|all]" ;;
esac
