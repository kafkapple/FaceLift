#!/usr/bin/env bash
# =============================================================================
# Mouse-FaceLift Environment Setup for gpu03
# =============================================================================
#
# Target: gpu03 server
#   - Ubuntu 24.04 LTS
#   - RTX A6000 (49GB) - devices 4-7
#   - CUDA 13.0 / Driver 580.65
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
echo ""

# =============================================================================
# Step 1: Find or Install Conda
# =============================================================================

CONDA_EXE=""

# Priority 1: Check anaconda3 in home
if [ -f "${HOME}/anaconda3/bin/conda" ]; then
    CONDA_EXE="${HOME}/anaconda3/bin/conda"
    echo "Found anaconda3: ${CONDA_EXE}"
# Priority 2: Check miniconda3 in home
elif [ -f "${HOME}/miniconda3/bin/conda" ]; then
    CONDA_EXE="${HOME}/miniconda3/bin/conda"
    echo "Found miniconda3: ${CONDA_EXE}"
# Priority 3: Check if conda is in PATH
elif command -v conda &> /dev/null; then
    CONDA_EXE=$(which conda)
    echo "Found conda in PATH: ${CONDA_EXE}"
fi

# If no conda found, install miniconda
if [ -z "${CONDA_EXE}" ]; then
    echo "No conda found. Installing Miniconda..."
    
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
    
    curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
    bash "${MINICONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
    rm "${MINICONDA_INSTALLER}"
    
    CONDA_EXE="${HOME}/miniconda3/bin/conda"
    "${CONDA_EXE}" init bash
    
    echo ""
    echo "Miniconda installed. Please run:"
    echo "  source ~/.bashrc"
    echo "  bash setup_gpu03.sh"
    exit 0
fi

echo "conda version: $(${CONDA_EXE} --version)"

# Initialize conda for this shell
CONDA_BASE=$(dirname $(dirname ${CONDA_EXE}))
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# =============================================================================
# Step 2: System Checks
# =============================================================================

echo ""
echo "Checking system requirements..."

# Check GPU
echo "GPUs (using A6000 devices 4-7):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | grep -E "^[4-7],"

# Check CUDA
if [ -f "${CUDA_PATH}/bin/nvcc" ]; then
    echo "CUDA: $(${CUDA_PATH}/bin/nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
else
    echo "Warning: CUDA not found at ${CUDA_PATH}"
fi

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
# Step 4: Install PyTorch
# =============================================================================

echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Verify
echo ""
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

# =============================================================================
# Step 5: Install Dependencies
# =============================================================================

echo ""
echo "Installing dependencies..."

pip install packaging typing-extensions

# AI/ML
pip install transformers diffusers[torch] huggingface-hub accelerate
pip install xformers

# Computer vision
pip install Pillow opencv-python scikit-image lpips
pip install facenet-pytorch --no-deps
pip install rembg

# Scientific
pip install numpy matplotlib scikit-learn einops jaxtyping pytorch-msssim scipy

# Utilities
pip install easydict pyyaml termcolor plyfile tqdm gradio pandas rich
pip install wandb --only-binary=:all:
pip install videoio ffmpeg-python

# =============================================================================
# Step 6: Install diff-gaussian-rasterization
# =============================================================================

echo ""
echo "Installing diff-gaussian-rasterization..."

export CUDA_HOME="$CONDA_PREFIX"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install CUDA toolkit via conda-forge (matches PyTorch cu124)
conda install -c conda-forge cuda-nvcc=12.4 cuda-cudart-dev=12.4 -y

# Build diff-gaussian-rasterization
export CUDA_HOME="$CONDA_PREFIX"
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization --no-build-isolation

# =============================================================================
# Step 7: Environment Variables
# =============================================================================

echo ""
echo "Setting up environment variables..."

CONDA_ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"
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

# Use RTX A6000 only (devices 4-7)
export CUDA_VISIBLE_DEVICES=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACTIVATE_EOF

    cat > "$DEACTIVATE_DIR/env_vars.sh" << 'DEACTIVATE_EOF'
#!/bin/bash
export CUDA_HOME=$_OLD_CUDA_HOME
export PATH=$_OLD_PATH
export LD_LIBRARY_PATH=$_OLD_LD_LIBRARY_PATH
unset _OLD_CUDA_HOME _OLD_PATH _OLD_LD_LIBRARY_PATH
unset PYTORCH_CUDA_ALLOC_CONF
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
print(f"Python: {sys.version.split()[0]}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

packages = ['transformers', 'diffusers', 'xformers', 'lpips', 'wandb']
print("\nPackages:")
for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"  {pkg}: OK")
    except:
        print(f"  {pkg}: FAILED")

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    print("  diff-gaussian-rasterization: OK")
except:
    print("  diff-gaussian-rasterization: FAILED")
VERIFY_EOF

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
echo "Training:"
echo "  python train_gslrm.py --config configs/mouse_gslrm_pixel_based_v2.yaml"
echo ""
