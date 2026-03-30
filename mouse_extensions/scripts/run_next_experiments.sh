#!/bin/bash
# run_next_experiments.sh — Launch 4D Deformation + DiFix Type 2 in parallel
#
# GPU4: Deformation Phase 0 (Gaussian cache generation)
# GPU5: DiFix Type 2 pair build + zero-shot evaluation
#
# Usage:
#   bash mouse_extensions/scripts/run_next_experiments.sh
#
# Prerequisites:
#   - configs/deformation/default.yaml: checkpoint=alpha03_v3, sh_degree=0 (DONE)
#   - tier0_raw + pseudo_gt data exists (VERIFIED: 4x3600 images)
#   - GPU4 and GPU5 free

set -e
PYTHON=/home/joon/anaconda3/envs/facelift/bin/python
MAMMAL_PYTHON=/home/joon/anaconda3/envs/mammal_stable/bin/python
BASE_DIR=/home/joon/dev/FaceLift

echo "============================================================"
echo "  Next Experiments — Parallel Launch (GPU4 + GPU5)"
echo "============================================================"

# === GPU4: 4D Deformation Phase 0 (Gaussian cache) ===
echo ""
echo "[GPU4] Launching Deformation Phase 0 (Gaussian cache generation)..."
tmux new-session -d -s deform_phase0 "\
cd $BASE_DIR && \
CUDA_VISIBLE_DEVICES=4 $PYTHON -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml \
    --precompute_cache \
    2>&1 | tee outputs/experiments/mouse/deform_phase0.log; \
echo 'EXIT_CODE='\$?"

echo "  tmux session: deform_phase0"
echo "  Log: outputs/experiments/mouse/deform_phase0.log"

# === GPU5: DiFix Type 2 pair build + zero-shot ===
echo ""
echo "[GPU5] Launching DiFix Type 2 zero-shot evaluation..."
tmux new-session -d -s difix_type2 "\
cd $BASE_DIR && \
echo '[1/2] Building Type 2 pairs...' && \
$PYTHON -m mouse_extensions.scripts.eval.build_difix_dataset \
    --types 2 \
    --use-symlinks \
    2>&1 | tee outputs/experiments/mouse/difix_type2_build.log && \
echo '[2/2] Running DiFix zero-shot on bottom view...' && \
CUDA_VISIBLE_DEVICES=5 $PYTHON -m mouse_extensions.scripts.eval.difix_zero_shot \
    --input_dir outputs/datasets/novel_view/mouse_m5t2/tier0_raw/bottom \
    --output_dir outputs/difix_zero_shot/type2_bottom_v2 \
    --n_samples 30 \
    2>&1 | tee -a outputs/experiments/mouse/difix_type2_zeroshot.log; \
echo 'EXIT_CODE='\$?"

echo "  tmux session: difix_type2"
echo "  Log: outputs/experiments/mouse/difix_type2_*.log"

echo ""
echo "============================================================"
echo "  Monitor:"
echo "    tmux attach -t deform_phase0   # GPU4 deformation"
echo "    tmux attach -t difix_type2     # GPU5 DiFix"
echo ""
echo "  Quick check:"
echo "    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -i 4,5"
echo "============================================================"
