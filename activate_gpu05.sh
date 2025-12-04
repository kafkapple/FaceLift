#!/bin/bash
# =============================================================================
# GPU05 Environment Activation Script for Mouse-FaceLift
# =============================================================================
#
# This script sets up the necessary environment variables for running
# Mouse-FaceLift on gpu05 server.
#
# Usage:
#   source activate_gpu05.sh
#   # Then you can run training/inference scripts
#
# Note: This script must be sourced, not executed directly.
# =============================================================================

# Check if being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced, not executed directly."
    echo "Usage: source activate_gpu05.sh"
    exit 1
fi

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

# Set CUDA environment (CUDA 11.8)
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set GCC version for any compilation
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

# Verify setup
echo "=============================================="
echo "Mouse-FaceLift Environment Activated (gpu05)"
echo "=============================================="
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep release | cut -d, -f1 | cut -d' ' -f5)"
echo "GCC version: $(gcc --version 2>/dev/null | head -1)"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "GPU available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo ""
echo "Ready to run:"
echo "  python scripts/process_mouse_data.py --help"
echo "  python train_mouse.py --config configs/mouse_config.yaml --overfit 10"
echo "=============================================="
