#!/bin/bash
# =============================================================================
# Mouse MVDiffusion Training Script
# =============================================================================
# Usage:
#   On gpu05: CUDA_VISIBLE_DEVICES=1 bash scripts/train_mouse_mvdiffusion.sh
#   On local (RTX 3060): CUDA_VISIBLE_DEVICES=0 bash scripts/train_mouse_mvdiffusion.sh --local
# =============================================================================

set -e

# Check for local mode
LOCAL_MODE=false
if [[ "$1" == "--local" ]]; then
    LOCAL_MODE=true
    echo "[INFO] Running in local (low memory) mode"
fi

# Change to project root
cd "$(dirname "$0")/.."

# Activate conda environment (skip if already activated)
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate facelift_rtx3060
fi
echo "[INFO] Using conda environment: $CONDA_DEFAULT_ENV"

# Check GPU
echo "[INFO] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Verify data exists
if [[ ! -f "data_mouse_local/data_mouse_train.txt" ]]; then
    echo "[ERROR] Training data not found: data_mouse_local/data_mouse_train.txt"
    echo "        Please ensure the data is available."
    exit 1
fi

TRAIN_SAMPLES=$(wc -l < data_mouse_local/data_mouse_train.txt)
VAL_SAMPLES=$(wc -l < data_mouse_local/data_mouse_val.txt)
echo "[INFO] Training samples: $TRAIN_SAMPLES, Validation samples: $VAL_SAMPLES"

# Create output directory
mkdir -p checkpoints/mvdiffusion/mouse_local
mkdir -p checkpoints/mvdiffusion/mouse_local/val

# Select config based on mode
if [[ "$LOCAL_MODE" == "true" ]]; then
    CONFIG="configs/mouse_mvdiffusion_lowmem.yaml"
    echo "[INFO] Using low memory config: $CONFIG"

    # Create lowmem config if it doesn't exist
    if [[ ! -f "$CONFIG" ]]; then
        echo "[INFO] Creating low memory config..."
        cp configs/mouse_mvdiffusion.yaml "$CONFIG"
        # Modify for low memory
        sed -i 's/train_batch_size: 4/train_batch_size: 1/' "$CONFIG"
        sed -i 's/gradient_accumulation_steps: 4/gradient_accumulation_steps: 8/' "$CONFIG"
        sed -i 's/max_train_steps: 30000/max_train_steps: 10000/' "$CONFIG"
        sed -i 's/checkpointing_steps: 1000/checkpointing_steps: 500/' "$CONFIG"
        sed -i 's/validation_steps: 500/validation_steps: 250/' "$CONFIG"
        sed -i 's/dataloader_num_workers: 8/dataloader_num_workers: 4/' "$CONFIG"
    fi
else
    CONFIG="configs/mouse_mvdiffusion.yaml"
    echo "[INFO] Using full config: $CONFIG"
fi

# Run training
echo "[INFO] Starting MVDiffusion training..."
echo "[INFO] Config: $CONFIG"
echo "[INFO] Output: checkpoints/mvdiffusion/mouse_local/"

accelerate launch \
    --mixed_precision=fp16 \
    --num_processes=1 \
    train_diffusion.py \
    --config "$CONFIG"

echo "[INFO] Training completed!"
echo "[INFO] Checkpoints saved to: checkpoints/mvdiffusion/mouse_local/"
