#!/bin/bash
# M5t2 vs M5t2_cfgr Comparison
# 각 체크포인트로 test set 10개 샘플 추론 후 비교

set -e
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift

export CUDA_VISIBLE_DEVICES=4
SPLIT="/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt"
GSLRM_CKPT="/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt"
NUM_SAMPLES=10

echo "=== M5t2 (5000 steps, sparse attention) ==="
python -m mouse_extensions.scripts.inference.run_e2e_inference \
  --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --split $SPLIT \
  --num_frames $NUM_SAMPLES \
  --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2/checkpoint-5000 \
  --gslrm_checkpoint $GSLRM_CKPT \
  --output_dir outputs/mvdiff_comparison/M5t2_5000 \
  --input_view_idx 0

echo ""
echo "=== M5t2_cfgr (10000 steps, full attention) ==="
python -m mouse_extensions.scripts.inference.run_e2e_inference \
  --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --split $SPLIT \
  --num_frames $NUM_SAMPLES \
  --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000 \
  --gslrm_checkpoint $GSLRM_CKPT \
  --output_dir outputs/mvdiff_comparison/M5t2_cfgr_10000 \
  --input_view_idx 0

echo ""
echo "=== Comparison Complete ==="
echo "Results in: outputs/mvdiff_comparison/"
