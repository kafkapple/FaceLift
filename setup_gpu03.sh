#!/usr/bin/env bash
# =============================================================================
# Mouse-FaceLift Environment Setup for gpu03
# =============================================================================
#
# Target: gpu03 server
#   - Ubuntu 24.04 LTS
#   - 4x RTX PRO 6000 Blackwell (98GB each)
#   - 4x RTX A6000 (49GB each)
#   - CUDA 13.0 / Driver 580.65
#   - gcc 13.3.0
#
# Usage:
#   bash setup_gpu03.sh [ENV_NAME]
#
# Date: 2025-01-05
# =============================================================================

set -eo pipefail
IFS=$'\n\t'

# Configuration
ENV_NAME="${1:-facelift}"
PYTHON_VERSION="3.11"
CUDA_PATH="/usr/local/cuda-13.0"

echo "============================================="
echo "Mouse-FaceLift gpu03 Setup"
echo "============================================="
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "CUDA path: ${CUDA_PATH}"
echo ""

# =============================================================================
# Step 1: Install Miniconda (if not present)
# =============================================================================

if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    
    curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
    bash "${MINICONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
    rm "${MINICONDA_INSTALLER}"
    
    # Initialize conda
    "${HOME}/miniconda3/bin/conda" init bash
    
    echo ""
    echo "Miniconda installed. Please run:"
    echo "  source ~/.bashrc"
    echo "  bash setup_gpu03.sh"
    exit 0
fi

echo "conda found: $(conda --version)"

# Initialize conda
eval "$(conda shell.bash hook)"

# =============================================================================
# Step 2: System Checks
# =============================================================================

echo ""
echo "Checking system requirements..."

# Check GPU
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "GPUs:"
echo "${GPU_INFO}" | while read line; do echo "  $line"; done

# Check CUDA
if [ -f "${CUDA_PATH}/bin/nvcc" ]; then
    CUDA_VERSION=$(${CUDA_PATH}/bin/nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)
    echo "CUDA: ${CUDA_VERSION}"
else
    echo "Error: CUDA not found at ${CUDA_PATH}"
    exit 1
fi

# Check gcc
echo "gcc: $(gcc --version | head -1)"

# =============================================================================
# Step 3: Create Conda Environment
# =============================================================================

echo ""
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists!"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Aborting."
        exit 1
    fi
fi

echo "Creating conda environment: ${ENV_NAME}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

conda activate "${ENV_NAME}"
echo "Activated: ${CONDA_DEFAULT_ENV}"

# =============================================================================
# Step 4: Install PyTorch (CUDA 12.4 - best compatibility)
# =============================================================================

echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Installing PyTorch with CUDA 12.4..."
# Note: CUDA 13.0 driver is backward compatible with CUDA 12.4 toolkit
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Verify
echo ""
echo "Verifying PyTorch..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# =============================================================================
# Step 5: Install Core Dependencies
# =============================================================================

echo ""
echo "Installing core dependencies..."

pip install packaging typing-extensions

# AI/ML libraries
echo "Installing AI/ML libraries..."
pip install transformers diffusers[torch] huggingface-hub accelerate

# xformers (memory efficient attention)
echo "Installing xformers..."
pip install xformers

# Computer vision
echo "Installing computer vision libraries..."
pip install Pillow opencv-python scikit-image lpips
pip install facenet-pytorch --no-deps
pip install rembg

# Scientific computing
echo "Installing scientific computing..."
pip install numpy matplotlib scikit-learn einops jaxtyping pytorch-msssim scipy

# Utilities
echo "Installing utilities..."
pip install easydict pyyaml termcolor plyfile tqdm gradio pandas rich
pip install wandb --only-binary=:all:
pip install videoio ffmpeg-python

# =============================================================================
# Step 6: Install diff-gaussian-rasterization
# =============================================================================

echo ""
echo "Installing diff-gaussian-rasterization..."
echo "This requires CUDA toolkit and may take a few minutes..."

# Set CUDA environment
export CUDA_HOME="${CUDA_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Build and install
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization --no-build-isolation

# =============================================================================
# Step 7: Environment Variables
# =============================================================================

echo ""
echo "Setting up environment variables..."

CONDA_ENV_PATH="$(conda info --base)/envs/${ENV_NAME}"
if [[ -d "$CONDA_ENV_PATH" ]]; then
    ACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/activate.d"
    DEACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/deactivate.d"
    mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

    cat > "$ACTIVATE_DIR/env_vars.sh" << 'ACTIVATE_EOF'
#!/bin/bash
export _OLD_CUDA_HOME=$CUDA_HOME
export _OLD_PATH=$PATH
export _OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Multi-GPU optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACTIVATE_EOF

    cat > "$DEACTIVATE_DIR/env_vars.sh" << 'DEACTIVATE_EOF'
#!/bin/bash
export CUDA_HOME=$_OLD_CUDA_HOME
export PATH=$_OLD_PATH
export LD_LIBRARY_PATH=$_OLD_LD_LIBRARY_PATH
unset _OLD_CUDA_HOME _OLD_PATH _OLD_LD_LIBRARY_PATH
unset NCCL_DEBUG PYTORCH_CUDA_ALLOC_CONF
DEACTIVATE_EOF

    chmod +x "$ACTIVATE_DIR/env_vars.sh" "$DEACTIVATE_DIR/env_vars.sh"
    echo "Environment variables configured."
fi

# =============================================================================
# Step 8: Verification
# =============================================================================

echo ""
echo "============================================="
echo "Verifying installation..."
echo "============================================="

python << 'VERIFY_EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(min(torch.cuda.device_count(), 4)):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.0f}GB)")

packages = ['transformers', 'diffusers', 'xformers', 'accelerate', 'lpips', 'wandb']
print("\nPackage verification:")
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f"  {pkg}: {ver}")
    except ImportError as e:
        print(f"  {pkg}: FAILED ({e})")

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    print("  diff-gaussian-rasterization: OK")
except ImportError as e:
    print(f"  diff-gaussian-rasterization: FAILED ({e})")

print("\nAll checks passed!")
VERIFY_EOF

# =============================================================================
# Step 9: Save Info
# =============================================================================

echo ""
echo "Saving environment info..."
{
    echo "Environment: ${ENV_NAME}"
    echo "Created: $(date)"
    echo "Host: $(hostname)"
    echo "Python: ${PYTHON_VERSION}"
    echo "CUDA: ${CUDA_PATH}"
    echo ""
    echo "Installed packages:"
    pip list
} > conda_env_gpu03_info.txt

# =============================================================================
# Complete
# =============================================================================

echo ""
echo "============================================="
echo "Setup Complete!"
echo "============================================="
echo ""
echo "To activate:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Multi-GPU training:"
echo "  torchrun --nproc_per_node 4 train_gslrm.py --config configs/mouse_gslrm_pixel_based_v2.yaml"
echo ""
echo "Single GPU training:"
echo "  python train_gslrm.py --config configs/mouse_gslrm_pixel_based_v2.yaml"
echo ""
