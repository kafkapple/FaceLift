#!/bin/bash
# H1 View Ablation: E2E Test with different input views
# Dataset: M5t2 (temporal 80:10:10)

cd /home/joon/dev/FaceLift

GSLRM_CKPT="M5t2_E0_1_facelift"
MVDIFF_CKPT="/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000"
DATA_DIR="/home/joon/data/preprocessed/FaceLift_mouse/M5"
SPLIT="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt"
OUTPUT_BASE="/home/joon/dev/FaceLift/outputs/eval/h1_view_ablation_M5t2"

mkdir -p $OUTPUT_BASE

# GPU 5: views 1, 2 (sequential)
(
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $SPLIT --num_frames 50 \
    --input_view_idx 1 --model M5t2 \
    --mvdiffusion_checkpoint $MVDIFF_CKPT --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/view_1 > $OUTPUT_BASE/view_1.log 2>&1

CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $SPLIT --num_frames 50 \
    --input_view_idx 2 --model M5t2 \
    --mvdiffusion_checkpoint $MVDIFF_CKPT --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/view_2 > $OUTPUT_BASE/view_2.log 2>&1
) &

# GPU 6: views 3, 4 (sequential)
(
CUDA_VISIBLE_DEVICES=6 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $SPLIT --num_frames 50 \
    --input_view_idx 3 --model M5t2 \
    --mvdiffusion_checkpoint $MVDIFF_CKPT --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/view_3 > $OUTPUT_BASE/view_3.log 2>&1

CUDA_VISIBLE_DEVICES=6 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $SPLIT --num_frames 50 \
    --input_view_idx 4 --model M5t2 \
    --mvdiffusion_checkpoint $MVDIFF_CKPT --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/view_4 > $OUTPUT_BASE/view_4.log 2>&1
) &

# GPU 7: view 5
(
CUDA_VISIBLE_DEVICES=7 python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir $DATA_DIR --split $SPLIT --num_frames 50 \
    --input_view_idx 5 --model M5t2 \
    --mvdiffusion_checkpoint $MVDIFF_CKPT --gslrm_checkpoint $GSLRM_CKPT \
    --skip_preprocess --prefer_ema \
    --output_dir $OUTPUT_BASE/view_5 > $OUTPUT_BASE/view_5.log 2>&1
) &

echo "Started M5t2 view ablation (views 1-5) on GPUs 5,6,7"
echo "Output: $OUTPUT_BASE"
