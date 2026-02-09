#!/bin/bash
# Verify rotation direction + smooth trajectory patches
# Usage: bash scripts/verify_turntable.sh [GPU_ID]
# Results: outputs/verify_turntable/

set -e
GPU=6
CKPT_DIR=/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift
OUT_DIR=outputs/verify_turntable
LOG=/tmp/verify_turntable.log

source ~/.bashrc.local
conda activate facelift
export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $OUT_DIR

echo "=== Step 1/3: Training (1 step for vis) ===" | tee $LOG

# Run 1 extra step to trigger vis_every=100 at step 9301
/home/joon/anaconda3/envs/facelift/bin/torchrun     --standalone --nproc_per_node=1     train_gslrm.py -d M5t2 -e E0_1_facelift     --set training.schedule.max_fwdbwd_passes 9302     --set validation.val_every 100     2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Step 2/3: Copy training results ===" | tee -a $LOG
LATEST=$(ls -td ${CKPT_DIR}/iter_* | head -1)
echo "Latest checkpoint dir: $LATEST" | tee -a $LOG
cp ${LATEST}/turntable_*.mp4 ${LATEST}/turntable_*.jpg $OUT_DIR/ 2>/dev/null || true
ls -la $OUT_DIR/ | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Step 3/3: Inference (standalone) ===" | tee -a $LOG
python mouse_extensions/scripts/inference/render_from_checkpoint.py     --checkpoint ${CKPT_DIR}/ckpt_0000000000009200.pt     --config configs/base/gslrm_mouse.yaml     --data_path ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt     --output_dir ${OUT_DIR}/inference     --mode turntable     --num_samples 1     2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Done! Results in $OUT_DIR ===" | tee -a $LOG
ls -la $OUT_DIR/ $OUT_DIR/inference/ 2>/dev/null | tee -a $LOG
