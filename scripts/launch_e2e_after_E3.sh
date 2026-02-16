#!/bin/bash
# Auto-launch E2E eval after E3 (pose extrinsic+add 10K) completes
# Wait for E3 → E2E inference on GPU 7 → metrics → fair comparison
set -e

E3_PID=1538036
CKPT_DIR="/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_pose_extrinsic_add"
GSLRM_CKPT="/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt"
OUTPUT_DIR="outputs/phase3_e2e/E3_pose_extrinsic_add"
FAIR_OUTPUT="experiments/comparison/tier/e3_pose_fair.json"

cd /home/joon/dev/FaceLift
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

echo "[$(date)] Waiting for E3 (PID $E3_PID) to finish..."

while true; do
    if ! kill -0 $E3_PID 2>/dev/null; then
        echo "[$(date)] E3 process exited."
        break
    fi
    # Check latest checkpoint for progress
    LATEST=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 | grep -oP '\d+$')
    echo "[$(date)] E3 progress: checkpoint-${LATEST:-?}/10000"
    sleep 300
done

echo "[$(date)] E3 completed. Finding last checkpoint..."
LAST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$LAST_CKPT" ]; then
    echo "[$(date)] ERROR: No checkpoint found in $CKPT_DIR"
    exit 1
fi
echo "[$(date)] Using: $LAST_CKPT"

# E2E inference on GPU 7
echo "[$(date)] Step 1/3: E2E inference..."
export CUDA_VISIBLE_DEVICES=7

python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --input_view_idx 0 \
    --skip_preprocess \
    --model M5t \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint "$GSLRM_CKPT" \
    --mvdiffusion_checkpoint "$LAST_CKPT" \
    --mvdiffusion_base checkpoints/mvdiffusion/pipeckpts \
    --prompt_embed_path /home/joon/dev/FaceLift/mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
    --prefer_ema \
    --output_dir "$OUTPUT_DIR" \
    --no_turntable --no_mesh \
    > logs/phase3_e2e_E3_pose.log 2>&1

echo "[$(date)] Step 2/3: Standard metrics..."
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir "$OUTPUT_DIR" \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --skip_input_view 0 \
    > logs/phase3_e2e_E3_pose_metrics.log 2>&1

echo "[$(date)] Step 3/3: Fair comparison metrics..."
python -m mouse_extensions.scripts.eval.fair_comparison evaluate_fl \
    --render_dir "${OUTPUT_DIR}/samples" \
    --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --output "$FAIR_OUTPUT" \
    --vis_every 30 \
    > logs/phase3_e2e_E3_fair.log 2>&1

echo ""
echo "=========================================="
echo "  E3 Pose E2E Results"
echo "=========================================="
echo "  Checkpoint: $LAST_CKPT"
echo "  Output:     $OUTPUT_DIR"
echo "  Fair JSON:  $FAIR_OUTPUT"
echo ""
echo "--- Standard Metrics ---"
cat logs/phase3_e2e_E3_pose_metrics.log | tail -20
echo ""
echo "--- Fair Metrics ---"
python3 -c "
import json
with open('$FAIR_OUTPUT') as f:
    d = json.load(f)
o = d['overall']
print('PSNR_gt: %.2f' % o['psnr_gt_masked']['mean'])
print('PSNR_int: %.2f' % o['psnr_intersection']['mean'])
print('IoU: %.3f' % o['iou']['mean'])
print('Coverage: %.1f%%' % (o['coverage']['mean']*100))
# Per-view for proximity bias check
pv = d.get('per_view', {})
for v in range(1,6):
    k = 'view_%d' % v
    if k in pv:
        print('  view_%d PSNR_gt: %.2f' % (v, pv[k]['psnr_gt_masked']['mean']))
"
echo "=========================================="
echo "[$(date)] Done!"
