#!/bin/bash
set -e
cd /home/joon/dev/FaceLift
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

E1_PID=1253076
CKPT_DIR="/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_20k_cosine"
GSLRM_CKPT="/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt"
OUTPUT_DIR="outputs/phase3_e2e/E1_cosine_20k"
FAIR_JSON="experiments/comparison/tier/e1_cosine_20k_fair.json"

echo "[$(date)] Waiting for E1 (PID $E1_PID) to finish..."

# Wait for E1 training
while kill -0 $E1_PID 2>/dev/null; do
    LATEST=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1 | xargs basename 2>/dev/null || echo "none")
    echo "[$(date)] E1 progress: ${LATEST}/20000"
    sleep 300
done

echo "[$(date)] E1 process exited."

# Find last checkpoint
LAST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$LAST_CKPT" ]; then
    echo "[$(date)] ERROR: No checkpoint found!"
    exit 1
fi
echo "[$(date)] Using: $LAST_CKPT"

# Step 1: E2E inference on GPU 7
echo "[$(date)] Step 1/3: E2E inference..."
export CUDA_VISIBLE_DEVICES=7
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --mvdiff_checkpoint ${LAST_CKPT} \
    --gslrm_checkpoint ${GSLRM_CKPT} \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir ${OUTPUT_DIR} \
    --no_turntable --no_mesh

# Step 2: Standard metrics
echo "[$(date)] Step 2/3: Standard metrics..."
python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir ${OUTPUT_DIR} \
    --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt

# Step 3: Fair comparison
echo "[$(date)] Step 3/3: Fair comparison metrics..."
python -m mouse_extensions.scripts.eval.fair_comparison evaluate_fl \
    --render_dir ${OUTPUT_DIR}/samples \
    --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --output ${FAIR_JSON} \
    --vis_every 30

echo ""
echo "=========================================="
echo "  E1 Cosine 20K E2E Results"
echo "=========================================="
echo "  Checkpoint: ${LAST_CKPT}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Fair JSON:  ${FAIR_JSON}"
echo ""

# Print standard metrics
echo "--- Standard Metrics ---"
python3 -c "
import json, glob
files = sorted(glob.glob('${OUTPUT_DIR}/metrics_*.json'))
if files:
    d = json.load(open(files[-1]))
    for mode in ['full_white', 'fg_only', 'alpha_weighted']:
        if mode in d:
            m = d[mode]
            print(f'  {mode:25s} {m.get(\"psnr\",0):8.2f} {m.get(\"ssim\",0):8.4f} {m.get(\"lpips\",0):8.4f} {m.get(\"l1\",0):8.4f} {m.get(\"iou\",0):8.3f}')
    if 'per_view' in d:
        print()
        for vk in sorted(d['per_view'].keys()):
            vd = d['per_view'][vk]
            fw = vd.get('full_white', {})
            fg = vd.get('fg_only', {})
            print(f'  {vk} PSNR_fw={fw.get(\"psnr\",0):.2f}, PSNR_fg={fg.get(\"psnr\",0):.2f}')
"

echo ""
echo "--- Fair Metrics ---"
python3 -c "
import json
d = json.load(open('${FAIR_JSON}'))
o = d['overall']
print(f'PSNR_gt: {o[\"psnr_gt_masked\"][\"mean\"]:.2f}')
print(f'PSNR_int: {o[\"psnr_intersection\"][\"mean\"]:.2f}')
print(f'IoU: {o[\"iou\"][\"mean\"]:.3f}')
print(f'Coverage: {o[\"coverage\"][\"mean\"]*100:.1f}%')
for v in sorted(d.get('per_view',{}).keys()):
    vd = d['per_view'][v]
    print(f'  {v} PSNR_gt: {vd[\"psnr_gt_masked\"][\"mean\"]:.2f}')
"
echo "=========================================="
echo "[$(date)] Done!"
