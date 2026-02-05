#!/bin/bash
# =============================================================================
# H1 Experiments: Parallel on 4 GPUs (Fixed Arguments)
# =============================================================================
set -e
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift

GSLRM_CKPT="M5t2_E0_1_facelift"
MVDIFF_CKPT="/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000"
DATA_DIR="/home/joon/data/preprocessed/FaceLift_mouse/M5"
TRAIN_SPLIT="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt"
TEST_SPLIT="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt"
OUTPUT_BASE="/node_data/joon/outputs/FaceLift/eval/H1"
NUM_FRAMES=50

mkdir -p $OUTPUT_BASE

echo "Starting H1 experiments in parallel (4 GPUs)..."
echo "Output: $OUTPUT_BASE"

# H1a: GPU 4 - GS-LRM + Train (6-view GT input)
echo "H1a: GS-LRM + Train"
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR \
    --split $TRAIN_SPLIT \
    --num_frames $NUM_FRAMES \
    --model M5t2 \
    --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess \
    --output_dir ${OUTPUT_BASE}/h1a_gslrm_train \
    2>&1 | tee ${OUTPUT_BASE}/h1a.log &
PID_A=$!

# H1b: GPU 5 - GS-LRM + Test (6-view GT input)
echo "H1b: GS-LRM + Test"
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR \
    --split $TEST_SPLIT \
    --num_frames $NUM_FRAMES \
    --model M5t2 \
    --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess \
    --output_dir ${OUTPUT_BASE}/h1b_gslrm_test \
    2>&1 | tee ${OUTPUT_BASE}/h1b.log &
PID_B=$!

# H1c: GPU 6 - E2E + Train (1-view input, MVDiffusion generates 6)
echo "H1c: E2E + Train"
CUDA_VISIBLE_DEVICES=6 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR \
    --split $TRAIN_SPLIT \
    --num_frames $NUM_FRAMES \
    --input_view_idx 0 \
    --model M5t2 \
    --mvdiffusion_checkpoint $MVDIFF_CKPT \
    --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess \
    --prefer_ema \
    --output_dir ${OUTPUT_BASE}/h1c_e2e_train \
    2>&1 | tee ${OUTPUT_BASE}/h1c.log &
PID_C=$!

# H1d: GPU 7 - E2E + Test (1-view input, MVDiffusion generates 6)
echo "H1d: E2E + Test"
CUDA_VISIBLE_DEVICES=7 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR \
    --split $TEST_SPLIT \
    --num_frames $NUM_FRAMES \
    --input_view_idx 0 \
    --model M5t2 \
    --mvdiffusion_checkpoint $MVDIFF_CKPT \
    --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess \
    --prefer_ema \
    --output_dir ${OUTPUT_BASE}/h1d_e2e_test \
    2>&1 | tee ${OUTPUT_BASE}/h1d.log &
PID_D=$!

echo "PIDs: H1a=$PID_A, H1b=$PID_B, H1c=$PID_C, H1d=$PID_D"
echo "Waiting for all experiments..."

wait $PID_A $PID_B $PID_C $PID_D

echo "=== All H1 experiments complete ==="
echo "Results in: $OUTPUT_BASE"

# Summary
echo ""
echo "=== H1 Summary ==="
for exp in h1a h1b h1c h1d; do
    if [ -f ${OUTPUT_BASE}/${exp}_*/metrics.json ]; then
        echo "${exp}: $(cat ${OUTPUT_BASE}/${exp}_*/metrics.json 2>/dev/null | head -1)"
    fi
done
