#!/bin/bash
set -e
cd /home/joon/dev/FaceLift
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift
export CUDA_VISIBLE_DEVICES=7

echo "[$(date)] Step 1/3: E2E inference with checkpoint-20000"
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_20k_cosine/checkpoint-20000 \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/phase3_e2e/E1_cosine_20k \
    --no_turntable --no_mesh

echo "[$(date)] Step 2/3: Standard metrics"
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir outputs/phase3_e2e/E1_cosine_20k \
    --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt

echo "[$(date)] Step 3/3: Fair comparison"
python -m mouse_extensions.scripts.eval.fair_comparison evaluate_fl \
    --render_dir outputs/phase3_e2e/E1_cosine_20k/samples \
    --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --output experiments/comparison/tier/e1_cosine_20k_fair.json \
    --vis_every 30

echo "[$(date)] ALL DONE"
