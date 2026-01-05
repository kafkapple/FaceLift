#!/bin/bash
# =============================================================================
# Mouse GS-LRM Training Script
# =============================================================================
# Usage:
#   On gpu05: CUDA_VISIBLE_DEVICES=1 bash scripts/train_mouse_gslrm.sh
#   On local (RTX 3060): CUDA_VISIBLE_DEVICES=0 bash scripts/train_mouse_gslrm.sh --local
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

# Select config based on mode
if [[ "$LOCAL_MODE" == "true" ]]; then
    CONFIG="configs/mouse_gslrm_lowmem.yaml"
    echo "[INFO] Using low memory config: $CONFIG"
else
    CONFIG="configs/mouse_gslrm_full.yaml"
    echo "[INFO] Using full config: $CONFIG"
fi

# Verify data exists
TRAIN_PATH=$(grep "dataset_path:" "$CONFIG" | head -1 | awk '{print $2}' | tr -d '"')
if [[ ! -f "$TRAIN_PATH" ]]; then
    echo "[ERROR] Training data not found: $TRAIN_PATH"
    echo "        Please generate synthetic data first:"
    echo "        python scripts/generate_synthetic_data.py"
    exit 1
fi

TRAIN_SAMPLES=$(wc -l < "$TRAIN_PATH")
echo "[INFO] Training samples: $TRAIN_SAMPLES"

# Create output directories
CKPT_DIR=$(grep "checkpoint_dir:" "$CONFIG" | awk '{print $2}' | tr -d '"')
mkdir -p "$CKPT_DIR"

VAL_DIR=$(grep -A1 "^validation:" "$CONFIG" | grep "output_dir:" | awk '{print $2}' | tr -d '"')
mkdir -p "$VAL_DIR"

# Run training
echo "[INFO] Starting GS-LRM training..."
echo "[INFO] Config: $CONFIG"
echo "[INFO] Checkpoints: $CKPT_DIR"

python train_gslrm.py --config "$CONFIG"

echo "[INFO] Training completed!"
echo "[INFO] Checkpoints saved to: $CKPT_DIR"
