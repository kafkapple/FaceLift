#!/bin/bash
# BehaviorSplatter Visualization Batch Runner
# ============================================
# Generates all visualization outputs in one run.
#
# Usage:
#   cd /home/joon/dev/FaceLift
#   CUDA_VISIBLE_DEVICES=5 bash mouse_extensions/behavior/run_all_visualizations.sh
#
# Outputs:
#   outputs/sdannce_poc/
#   ├── cinematic_white/        # Cinematic demo (white BG, keypoints off)
#   ├── cinematic_white_kp/     # Cinematic demo (white BG, keypoints on)
#   ├── cinematic_black/        # Cinematic demo (black BG)
#   ├── multiview_filter/       # N-threshold sweep comparison (existing)
#   ├── multiview_filter_video/ # N>=2 temporal 6-view grid (existing)
#   └── novel_grid_filtered/    # Novel elevation grid (existing)
#
# Estimated time: ~2-3 hours total (GPU dependent)

set -e
OUTBASE="outputs/sdannce_poc"

echo "========================================"
echo "BehaviorSplatter Visualization Suite"
echo "========================================"

# --- 1. Default cinematic (white BG, no keypoints) ---
echo ""
echo "[1/4] Cinematic — white BG, no keypoints"
python -m mouse_extensions.behavior.cinematic_sequence \
    --config mouse_extensions/behavior/cinematic_default.yaml \
    --output-dir "${OUTBASE}/cinematic_white"

# --- 2. Cinematic with keypoint overlay ---
echo ""
echo "[2/4] Cinematic — white BG, with keypoints"
python -c "
import yaml
with open('mouse_extensions/behavior/cinematic_default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['global']['keypoint_overlay'] = True
with open('/tmp/cinematic_kp.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
python -m mouse_extensions.behavior.cinematic_sequence \
    --config /tmp/cinematic_kp.yaml \
    --output-dir "${OUTBASE}/cinematic_white_kp"

# --- 3. Cinematic with black BG ---
echo ""
echo "[3/4] Cinematic — black BG, no keypoints"
python -c "
import yaml
with open('mouse_extensions/behavior/cinematic_default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['global']['bg_color'] = [0.0, 0.0, 0.0]
with open('/tmp/cinematic_black.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
python -m mouse_extensions.behavior.cinematic_sequence \
    --config /tmp/cinematic_black.yaml \
    --output-dir "${OUTBASE}/cinematic_black"

# --- 4. N>=2 temporal 6-view grid video (if not already generated) ---
if [ ! -f "${OUTBASE}/multiview_filter_video/video_all_filtered_white_grid.mp4" ]; then
    echo ""
    echo "[4/4] Multi-view filter temporal 6-view grid"
    python -m mouse_extensions.behavior.multiview_visibility_filter \
        --mode video \
        --frame-range 195:255 \
        --n-filter 2 \
        --views 0 1 2 3 4 5 \
        --parts face tail torso \
        --output-dir "${OUTBASE}/multiview_filter_video" \
        --fps 10
else
    echo ""
    echo "[4/4] Multi-view filter videos already exist, skipping"
fi

echo ""
echo "========================================"
echo "All visualizations complete!"
echo "Outputs: ${OUTBASE}/"
echo "========================================"
echo ""
echo "To re-run individual configs:"
echo "  python -m mouse_extensions.behavior.cinematic_sequence \\"
echo "      --config mouse_extensions/behavior/cinematic_default.yaml \\"
echo "      --output-dir outputs/sdannce_poc/my_output"
echo ""
echo "Config options (edit YAML or override via CLI):"
echo "  global.fps: 15              # frames per second"
echo "  global.bg_color: [1,1,1]    # white BG (default) or [0,0,0] black"
echo "  global.n_filter: 2          # multi-view visibility threshold"
echo "  global.keypoint_overlay: true/false"
echo "  global.mask_border_color: null or [R,G,B]"
echo "  frame_range: '195:315'      # temporal frame range"
