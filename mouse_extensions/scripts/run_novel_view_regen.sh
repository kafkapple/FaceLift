#!/bin/bash
# run_novel_view_regen.sh — Regenerate novel view dataset at 512px with new MAMMAL mesh
#
# Phase 1 (GPU): GS-LRM renders at novel cameras (α=0.3 checkpoint)
# Phase 2 (CPU/EGL): MAMMAL mesh renders at same cameras (production_3600_slerp)
# Phase 3 (CPU): Rebuild DiFix Type 2 pairs + comparison samples
#
# Usage:
#   bash mouse_extensions/scripts/run_novel_view_regen.sh [GPU_ID]

set -e
GPU=${1:-5}
PYTHON=/home/joon/anaconda3/envs/facelift/bin/python
MAMMAL_PYTHON=/home/joon/anaconda3/envs/mammal_stable/bin/python
BASE_DIR=/home/joon/dev/FaceLift
OUTPUT=$BASE_DIR/outputs/datasets/novel_view

echo "============================================================"
echo "  Novel View Dataset Regeneration (512px, α=0.3, new mesh)"
echo "  GPU: $GPU"
echo "============================================================"

# === Phase 1: GS-LRM novel view renders ===
echo ""
echo "[Phase 1/3] GS-LRM novel view renders (4 views × 3600 frames @ 512px)"
echo "  Checkpoint: M5t2_6view_alpha03_v3 (IoU=0.913)"
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase gslrm \
    --frame_range 0 3600 \
    --force

echo ""
echo "[Phase 1/3] DONE: tier0_raw/ regenerated"

# === Phase 2: MAMMAL mesh renders ===
echo ""
echo "[Phase 2/3] MAMMAL mesh renders (production_3600_slerp, UV textured)"
echo "  Mesh: production_3600_slerp/obj_textured/"
PYOPENGL_PLATFORM=egl $MAMMAL_PYTHON -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase mammal \
    --frame_range 0 3600 \
    --force

echo ""
echo "[Phase 2/3] DONE: pseudo_gt/ regenerated"

# === Phase 3: Rebuild DiFix pairs ===
echo ""
echo "[Phase 3/3] Rebuilding DiFix Type 2 pairs..."
$PYTHON -m mouse_extensions.scripts.eval.build_difix_dataset \
    --src $OUTPUT \
    --dst $BASE_DIR/outputs/datasets/difix_pairs \
    --types 2 \
    --use-symlinks

echo ""
echo "============================================================"
echo "  Regeneration Complete!"
echo ""
echo "  Outputs:"
echo "    tier0_raw/{bottom,top,front_low,side_low}/  (GS-LRM 512px)"
echo "    pseudo_gt/{bottom,top,front_low,side_low}/  (MAMMAL 512px)"
echo "    difix_pairs/type2_novel_view/               (14,400 pairs)"
echo ""
echo "  Compare old vs new:"
echo "    Old (384px): $OUTPUT/mouse_m5t2/_archive_384px/"
echo "    New (512px): $OUTPUT/mouse_m5t2/tier0_raw/"
echo "============================================================"
