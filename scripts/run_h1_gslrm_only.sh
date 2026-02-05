#!/bin/bash
# H1 GS-LRM Only Experiments (Corrected)
# h1a, h1b for both M5t and M5t2

cd /home/joon/dev/FaceLift

PYTHON="/home/joon/anaconda3/envs/facelift/bin/python"

echo "=== H1 GS-LRM Only Experiments (Corrected) ==="

# M5t2 experiments (GPU 5)
(
echo "[GPU5] M5t2 h1a (GS-LRM Train)..."
CUDA_VISIBLE_DEVICES=5 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --num_frames 50 \
    --model M5t2 \
    --skip_preprocess --prefer_ema \
    --output_dir outputs/eval/h1_diagnosis_M5t2/h1a_gslrm_train_v2 > outputs/eval/h1_diagnosis_M5t2/h1a_v2.log 2>&1

echo "[GPU5] M5t2 h1b (GS-LRM Test)..."
CUDA_VISIBLE_DEVICES=5 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --num_frames 50 \
    --model M5t2 \
    --skip_preprocess --prefer_ema \
    --output_dir outputs/eval/h1_diagnosis_M5t2/h1b_gslrm_test_v2 > outputs/eval/h1_diagnosis_M5t2/h1b_v2.log 2>&1

echo "[GPU5] M5t2 Done."
) &

# M5t experiments (GPU 6)
(
echo "[GPU6] M5t h1a (GS-LRM Train)..."
CUDA_VISIBLE_DEVICES=6 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_train.txt \
    --num_frames 50 \
    --model M5t \
    --skip_preprocess --prefer_ema \
    --output_dir outputs/eval/h1_diagnosis_M5t/h1a_gslrm_train_v2 > outputs/eval/h1_diagnosis_M5t/h1a_v2.log 2>&1

echo "[GPU6] M5t h1b (GS-LRM Test)..."
CUDA_VISIBLE_DEVICES=6 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --num_frames 50 \
    --model M5t \
    --skip_preprocess --prefer_ema \
    --output_dir outputs/eval/h1_diagnosis_M5t/h1b_gslrm_test_v2 > outputs/eval/h1_diagnosis_M5t/h1b_v2.log 2>&1

echo "[GPU6] M5t Done."
) &

wait
echo "All H1 GS-LRM Only experiments complete."
