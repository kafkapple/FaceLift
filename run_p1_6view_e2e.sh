#!/bin/bash
# P1: GS-LRM 6v + MVDiff E2E — Full test set (360 frames)
# Tests hypothesis: using all 6 MVDiff views improves E2E quality
set -e

cd /home/joon/dev/FaceLift
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

GPU="${CUDA_VISIBLE_DEVICES:-7}"
export CUDA_VISIBLE_DEVICES=$GPU

OUTPUT_DIR="outputs/P1_6view_e2e"
GT_DIR="/home/joon/data/preprocessed/FaceLift_mouse/M5"
FAIR_JSON="experiments/comparison/tier/p1_6view_e2e_fair.json"

echo "P1: GS-LRM 6v + MVDiff E2E"
echo "=========================="
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Full inference (skip existing dry-run frames)
echo "[$(date)] Step 1: E2E inference (360 test frames)"
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir "$GT_DIR" \
    --split data_mouse_t2_test.txt \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000 \
    --gslrm_checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_6view_v2/best_psnr.pt \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --num_input_views 6 \
    --input_view_idx 0 \
    --output_dir "$OUTPUT_DIR" \
    --no_turntable --no_mesh --no_metrics \
    --prefer_ema --skip_preprocess

echo "[$(date)] Step 1 complete"
NRENDER=$(ls "$OUTPUT_DIR/samples/" 2>/dev/null | wc -l)
echo "  Rendered: $NRENDER frames"
echo ""

# Step 2: Fair comparison metrics
echo "[$(date)] Step 2: Fair comparison metrics"
python -m mouse_extensions.scripts.eval.fair_comparison evaluate_fl \
    --render_dir "$OUTPUT_DIR/samples" \
    --gt_dir "$GT_DIR" \
    --output "$FAIR_JSON" \
    --vis_every 60

echo "[$(date)] Step 2 complete"
echo "  Saved: $FAIR_JSON"
echo ""

# Step 3: Quick summary
echo "[$(date)] Step 3: Results summary"
python3 -c "
import json
with open('$FAIR_JSON') as f:
    d = json.load(f)
o = d.get('overall', {})
def gm(v):
    return v.get('mean', v) if isinstance(v, dict) else v
print('P1: GS-LRM 6v + MVDiff E2E Results')
print('=' * 50)
print(f\"  PSNR_gt_masked:  {gm(o.get('psnr_gt_masked', 0)):.2f} dB\")
print(f\"  PSNR_intersection: {gm(o.get('psnr_intersection', 0)):.2f} dB\")
print(f\"  IoU:             {gm(o.get('iou', 0)):.3f}\")
print(f\"  Coverage:        {gm(o.get('coverage', 0))*100:.1f}%\")
print()
# Comparison
print('Comparison with 4v E2E (E2 resume):')
print(f\"  PSNR_gt: {gm(o.get('psnr_gt_masked', 0)):.2f} vs 8.20 (delta: {gm(o.get('psnr_gt_masked', 0))-8.20:+.2f})\")
print(f\"  IoU:     {gm(o.get('iou', 0)):.3f} vs 0.521 (delta: {gm(o.get('iou', 0))-0.521:+.3f})\")
print(f\"  PSNR_int: {gm(o.get('psnr_intersection', 0)):.2f} vs 15.63 (delta: {gm(o.get('psnr_intersection', 0))-15.63:+.2f})\")
"

echo ""
echo "[$(date)] P1 ALL DONE"
