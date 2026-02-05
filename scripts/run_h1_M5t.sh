#!/bin/bash
# H1 Diagnosis Experiments for M5t dataset (temporal 1:1:1)

cd /home/joon/dev/FaceLift

PYTHON="/home/joon/anaconda3/envs/facelift/bin/python"
GSLRM_CKPT="M5t_E0_1_facelift"
MVDIFF_CKPT="/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t/checkpoint-8000"
DATA_DIR="/home/joon/data/preprocessed/FaceLift_mouse/M5"
TRAIN_SPLIT="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_train.txt"
TEST_SPLIT="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt"
OUTPUT_BASE="/home/joon/dev/FaceLift/outputs/eval/h1_diagnosis_M5t"

mkdir -p $OUTPUT_BASE

echo "=== H1 M5t Experiments ==="

# GPU 5: h1a → h1b (GS-LRM only, sequential)
(
echo "[GPU5] h1a (GS-LRM Train)..."
CUDA_VISIBLE_DEVICES=5 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $TRAIN_SPLIT --num_frames 50 \
    --input_view_idx 0 --model M5t \
    --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/h1a_gslrm_train > $OUTPUT_BASE/h1a.log 2>&1

echo "[GPU5] h1b (GS-LRM Test)..."
CUDA_VISIBLE_DEVICES=5 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $TEST_SPLIT --num_frames 50 \
    --input_view_idx 0 --model M5t \
    --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/h1b_gslrm_test > $OUTPUT_BASE/h1b.log 2>&1
echo "[GPU5] Done."
) &

# GPU 6: h1c (E2E Train)
(
echo "[GPU6] h1c (E2E Train)..."
CUDA_VISIBLE_DEVICES=6 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $TRAIN_SPLIT --num_frames 50 \
    --input_view_idx 0 --model M5t \
    --mvdiffusion_checkpoint $MVDIFF_CKPT --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/h1c_e2e_train > $OUTPUT_BASE/h1c.log 2>&1
echo "[GPU6] Done."
) &

# GPU 7: h1d (E2E Test)
(
echo "[GPU7] h1d (E2E Test)..."
CUDA_VISIBLE_DEVICES=7 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $TEST_SPLIT --num_frames 50 \
    --input_view_idx 0 --model M5t \
    --mvdiffusion_checkpoint $MVDIFF_CKPT --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/h1d_e2e_test > $OUTPUT_BASE/h1d.log 2>&1
echo "[GPU7] Done."
) &

echo "Started: GPU5(h1a→h1b), GPU6(h1c), GPU7(h1d)"
