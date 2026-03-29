#!/bin/bash
# run_cinematic_v6_all.sh — Generate all cinematic v6 variants
#
# Variants:
#   demo_v6_A: controls=ON, extreme views=ON  (+ dual noctrl output demo_v6_B)
#   demo_v6_C: controls=OFF, extreme views=OFF
#
# Usage:
#   bash mouse_extensions/scripts/run_cinematic_v6_all.sh [GPU_ID]
#
# Run 1 generates A (with controls) + B (no controls, zero re-inference) via --dual-output-dir
# Run 2 generates C (clean standard)
# Use --use-cache to share segment caches across runs where possible.

set -e
GPU=${1:-5}
BASE_DIR="outputs/viz/cinematic/mouse"
CFG_DIR="configs/mouse/cinematic"

echo "============================================================"
echo "  Cinematic v6 — All Variants  (GPU $GPU)"
echo "============================================================"

# === Run 1: v6_A (controls) + v6_B (no controls, free via dual output) ===
echo ""
echo "[1/2] Generating v6_A (controls ON, extreme ON) + v6_B (no controls)"
CUDA_VISIBLE_DEVICES=$GPU python -m mouse_extensions.behavior.cinematic_sequence \
    --config "$CFG_DIR/cinematic_demo_v6_A.yaml" \
    --output-dir "$BASE_DIR/demo_v6_A" \
    --use-cache \
    --dual-output-dir "$BASE_DIR/demo_v6_B"

echo ""
echo "[1/2] DONE: demo_v6_A + demo_v6_B"

# === Run 2: v6_C (clean standard) ===
echo ""
echo "[2/2] Generating v6_C (controls OFF, extreme OFF)"
CUDA_VISIBLE_DEVICES=$GPU python -m mouse_extensions.behavior.cinematic_sequence \
    --config "$CFG_DIR/cinematic_demo_v6_C.yaml" \
    --output-dir "$BASE_DIR/demo_v6_C" \
    --use-cache

echo ""
echo "[2/2] DONE: demo_v6_C"

echo ""
echo "============================================================"
echo "  All variants complete!"
echo "  Outputs:"
echo "    $BASE_DIR/demo_v6_A/cinematic_demo.mp4  (controls + extreme)"
echo "    $BASE_DIR/demo_v6_B/cinematic_demo.mp4  (no controls + extreme)"
echo "    $BASE_DIR/demo_v6_C/cinematic_demo.mp4  (clean standard)"
echo ""
echo "  Download:"
echo "    scp gpu03:~/dev/FaceLift/$BASE_DIR/demo_v6_{A,B,C}/cinematic_demo.mp4 ~/Downloads/"
echo "============================================================"
