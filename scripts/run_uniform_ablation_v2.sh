#!/bin/bash
# ==============================================================================
# Uniform View Ablation Experiment v2
# ==============================================================================
# Run all view ablation experiments with identical settings
# Created: 2026-02-06
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# GPU assignment (A6000 only: 4,5,6,7 on gpu03)
declare -A GPU_MAP=(
    [1]=4
    [2]=5
    [3]=6
    [5]=7
    [6]=4  # Reuse GPU 4 (run after 1-view completes)
)

# Conda setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift

echo "=== Uniform View Ablation v2 ==="
echo "Base config: configs/mouse/uniform/base_uniform_v2.yaml"
echo "Max steps: 10000"
echo ""

for views in 1 2 3 5; do
    GPU_ID=${GPU_MAP[$views]}
    echo "Starting ${views}-view on GPU ${GPU_ID}..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID torchrun \
        --standalone --nproc_per_node=1 \
        train_gslrm.py \
        --config configs/mouse/uniform/base_uniform_v2.yaml \
        --config configs/mouse/uniform/${views}view_v2.yaml \
        > logs/uniform_v2_${views}view.log 2>&1 &
    
    echo "  PID: $!"
    sleep 5  # Stagger starts
done

echo ""
echo "All experiments started. Monitor with:"
echo "  tail -f logs/uniform_v2_*.log"
echo "  watch -n 5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'"
echo ""
echo "Note: 6-view will start after 1-view completes (same GPU)"
