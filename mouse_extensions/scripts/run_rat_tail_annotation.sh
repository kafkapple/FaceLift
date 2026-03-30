#!/bin/bash
# Rat Tail Annotation Pipeline — POC & Full
#
# Usage:
#   bash run_rat_tail_annotation.sh poc     # Phase 1: 3cam × 5kf (검증)
#   bash run_rat_tail_annotation.sh full    # Phase 2: 6cam × 18kf (전체)
#   bash run_rat_tail_annotation.sh viewer  # Launch annotation viewer
#   bash run_rat_tail_annotation.sh propagate-poc   # Propagate after POC annotation
#   bash run_rat_tail_annotation.sh propagate-full  # Propagate after Full annotation
#
# Prerequisites:
#   conda activate sdannce
#   cd /home/joon/dev/sdannce-poc

set -e

SESSION="/home/joon/data/sdannce/rat/dataverse/SCN2A_WK1_2022_09_16_M1"
SETUP_SCRIPT="/home/joon/dev/FaceLift/mouse_extensions/scripts/setup_tail_annotation.py"
GPU="${RAT_GPU:-7}"  # override: RAT_GPU=5 bash run_rat_tail_annotation.sh ...
PORT="${RAT_PORT:-8770}"

case "${1:-help}" in

  poc)
    echo "=== Phase 1: POC — 3cam × 5 keyframes ==="
    echo ""

    # Step 1: Generate kp-guided masks (if not already done)
    if [ ! -d "${SESSION}/sam2_masks_test/Camera1" ]; then
      echo "[1/2] Generating kp-guided masks (3cam, 50 frames)..."
      CUDA_VISIBLE_DEVICES=$GPU python segmentation/kp_sam2_lone.py \
        --session_dir "$SESSION" \
        --output "${SESSION}/sam2_masks_test" \
        --cameras 1,2,3 \
        --start 0 --end 2500 --step 50 \
        --gpu 0
    else
      echo "[1/2] KP masks already exist — skipping"
    fi

    # Step 2: Setup annotation keyframes
    echo "[2/2] Setting up annotation keyframes..."
    python "$SETUP_SCRIPT" \
      --session "$SESSION" \
      --kp-masks sam2_masks_test \
      --cameras 1,2,3 \
      --keyframes 0,500,1000,1500,2000

    echo ""
    echo "✅ POC setup complete. Run: bash run_rat_tail_annotation.sh viewer"
    ;;

  full)
    echo "=== Phase 2: Full — 6cam × 18 keyframes ==="
    echo ""

    # Step 1: Generate kp-guided masks for all 6 cameras
    if [ ! -d "${SESSION}/sam2_masks_rat2/Camera1" ]; then
      echo "[1/2] Generating kp-guided masks (6cam, step=30, ~70 min)..."
      CUDA_VISIBLE_DEVICES=$GPU python segmentation/kp_sam2_lone.py \
        --session_dir "$SESSION" \
        --output "${SESSION}/sam2_masks_rat2" \
        --cameras 1,2,3,4,5,6 \
        --start 0 --end 89100 --step 30 \
        --gpu 0
    else
      echo "[1/2] KP masks already exist — skipping"
    fi

    # Step 2: Setup annotation keyframes
    echo "[2/2] Setting up annotation keyframes..."
    python "$SETUP_SCRIPT" \
      --session "$SESSION" \
      --kp-masks sam2_masks_rat2 \
      --cameras 1,2,3,4,5,6 \
      --keyframes 0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000,75000,80000,85000

    echo ""
    echo "✅ Full setup complete. Run: bash run_rat_tail_annotation.sh viewer"
    ;;

  viewer)
    echo "=== Launching Annotation Viewer (GPU $GPU, port $PORT) ==="
    echo ""
    echo "SSH tunnel (run on local Mac):"
    echo "  ssh -L ${PORT}:localhost:${PORT} gpu03"
    echo "Browser: http://localhost:${PORT}"
    echo ""
    CUDA_VISIBLE_DEVICES=$GPU python viewers/mask_annotator.py \
      --session "$SESSION" \
      --port "$PORT"
    ;;

  propagate-poc)
    echo "=== Propagate POC (6cam, sparse step=50) ==="
    for cam in 1 2 3 4 5 6; do
      echo "--- Camera $cam ---"
      CUDA_VISIBLE_DEVICES=$GPU python segmentation/sam2_propagate.py \
        --session "$SESSION" \
        --cameras "$cam" \
        --mode sparse --step 50 \
        --model base_plus \
        --output "${SESSION}/masks/Camera${cam}/propagated/poc"
    done
    echo ""
    echo "✅ POC propagation complete. Check overlay videos in masks/Camera{1-6}/propagated/poc/"
    ;;

  propagate-full)
    echo "=== Propagate Full (6cam, sparse step=30) ==="
    for cam in 1 2 3 4 5 6; do
      echo "--- Camera $cam ---"
      CUDA_VISIBLE_DEVICES=$GPU python segmentation/sam2_propagate.py \
        --session "$SESSION" \
        --cameras "$cam" \
        --mode sparse --step 30 \
        --model base_plus \
        --output "${SESSION}/masks/Camera${cam}/propagated/full"
    done
    echo ""
    echo "✅ Full propagation complete."
    ;;

  *)
    echo "Rat Tail Annotation Pipeline"
    echo ""
    echo "Usage: bash run_rat_tail_annotation.sh <command>"
    echo ""
    echo "Commands:"
    echo "  poc              Phase 1: 3cam × 5kf setup (검증용)"
    echo "  full             Phase 2: 6cam × 18kf setup (전체)"
    echo "  viewer           Launch annotation viewer"
    echo "  propagate-poc    Propagate after POC annotation"
    echo "  propagate-full   Propagate after Full annotation"
    echo ""
    echo "Environment:"
    echo "  RAT_GPU=7        GPU index (default: 7)"
    echo "  RAT_PORT=8770    Viewer port (default: 8770)"
    echo ""
    echo "Typical workflow:"
    echo "  1. bash run_rat_tail_annotation.sh poc"
    echo "  2. bash run_rat_tail_annotation.sh viewer"
    echo "  3. (annotate tail in browser)"
    echo "  4. bash run_rat_tail_annotation.sh propagate-poc"
    echo "  5. (verify quality)"
    echo "  6. bash run_rat_tail_annotation.sh full"
    echo "  7. bash run_rat_tail_annotation.sh viewer"
    echo "  8. (annotate tail in browser)"
    echo "  9. bash run_rat_tail_annotation.sh propagate-full"
    ;;

esac
