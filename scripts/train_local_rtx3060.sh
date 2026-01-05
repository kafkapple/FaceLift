#!/bin/bash
# Local RTX 3060 Training Script
# Run from FaceLift root directory

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

# Copy data list files if not exist
if [ ! -f "data_mouse_uniform/data_mouse_train.txt" ]; then
    echo "Copying data list files..."
    # Update paths from data_mouse_centered to data_mouse_uniform
    sed 's|data_mouse_centered|data_mouse_uniform|g' \
        data_mouse_centered/data_mouse_train.txt > data_mouse_uniform/data_mouse_train.txt
    sed 's|data_mouse_centered|data_mouse_uniform|g' \
        data_mouse_centered/data_mouse_val.txt > data_mouse_uniform/data_mouse_val.txt
    echo "Data list files created."
fi

# Verify GPU
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Set wandb to online mode
export WANDB_MODE=online

# Run training
echo ""
echo "=== Starting Training ==="
echo "Config: configs/mouse_gslrm_local_rtx3060.yaml"
echo ""

python train_gslrm.py --config configs/mouse_gslrm_local_rtx3060.yaml
