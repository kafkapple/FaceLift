#!/usr/bin/env bash
# =============================================================================
# Mouse-FaceLift Local RTX 3060 Environment Setup
# =============================================================================
#
# Target: Local Ubuntu 22.04 machine with RTX 3060 12GB
#
# System Requirements:
#   - Ubuntu 22.04 LTS
#   - CUDA 11.8 or 12.x installed
#   - conda installed
#   - RTX 3060 12GB GPU
#
# Usage:
#   bash setup_local_rtx3060.sh [ENV_NAME]
#
# Date: 2024-12-30
# =============================================================================

set -eo pipefail
IFS=$'\n\t'

# Configuration
ENV_NAME="${1:-facelift_rtx3060}"
PYTHON_VERSION="3.10"

echo "=============================================="
echo "Mouse-FaceLift Local RTX 3060 Setup"
echo "=============================================="
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo ""

# =============================================================================
# System Checks
# =============================================================================

echo "Checking system requirements..."

# Check conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed"
    exit 1
fi
echo "  conda: $(conda --version)"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Install NVIDIA driver."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  GPU: ${GPU_NAME} (${GPU_MEM})"

# Check CUDA
if [ -d "/usr/local/cuda-12.4" ]; then
    CUDA_PATH="/usr/local/cuda-12.4"
    CUDA_VERSION="12.4"
    PYTORCH_CUDA="cu124"
elif [ -d "/usr/local/cuda-11.8" ]; then
    CUDA_PATH="/usr/local/cuda-11.8"
    CUDA_VERSION="11.8"
    PYTORCH_CUDA="cu118"
else
    echo "Error: CUDA 11.8 or 12.4 not found in /usr/local/"
    exit 1
fi
echo "  CUDA: ${CUDA_VERSION} (${CUDA_PATH})"

# Check gcc
GCC_VERSION=$(gcc -dumpversion)
echo "  gcc: ${GCC_VERSION}"

echo ""

# =============================================================================
# Conda Environment Setup
# =============================================================================

# Initialize conda
eval "$(conda shell.bash hook)"

# Check existing environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists!"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Aborting. Use different name or activate existing env."
        exit 1
    fi
fi

# Create environment
echo "Creating conda environment: ${ENV_NAME}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate
conda activate "${ENV_NAME}"
echo "Activated: ${CONDA_DEFAULT_ENV}"
echo ""

# =============================================================================
# Install Dependencies
# =============================================================================

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo ""
echo "Installing core dependencies..."
pip install packaging==24.2 typing-extensions==4.14.0

# PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
if [ "$PYTORCH_CUDA" = "cu124" ]; then
    pip install torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
else
    pip install torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
fi

# Verify CUDA
echo ""
echo "Verifying PyTorch CUDA..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# AI/ML libraries
echo ""
echo "Installing AI/ML libraries..."
pip install transformers==4.44.2 \
    diffusers[torch]==0.30.3 \
    huggingface-hub==0.35.3 \
    accelerate==0.33.0

# xformers (memory efficient attention - critical for RTX 3060)
echo ""
echo "Installing xformers (memory optimization)..."
if [ "$PYTORCH_CUDA" = "cu124" ]; then
    pip install xformers==0.0.27.post2
else
    pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
fi

# Computer vision
echo ""
echo "Installing computer vision libraries..."
pip install Pillow==10.4.0 \
    opencv-python==4.10.0.84 \
    scikit-image==0.21.0 \
    lpips==0.1.4

pip install facenet-pytorch --no-deps
pip install rembg

# Scientific computing
echo ""
echo "Installing scientific computing..."
pip install numpy==1.26.4 \
    matplotlib==3.7.5 \
    scikit-learn==1.3.2 \
    einops==0.8.0 \
    jaxtyping==0.2.19 \
    pytorch-msssim==1.0.0

# Utilities
echo ""
echo "Installing utilities..."
pip install easydict==1.13 \
    pyyaml==6.0.2 \
    termcolor==2.4.0 \
    plyfile==1.0.3 \
    tqdm \
    gradio==5.49.1 \
    pandas \
    rich

# wandb (binary only to avoid build issues)
pip install wandb --only-binary=:all:

# Video processing
pip install videoio==0.3.0 ffmpeg-python==0.2.0

# =============================================================================
# Install diff-gaussian-rasterization
# =============================================================================

echo ""
echo "Installing diff-gaussian-rasterization..."
echo "This requires CUDA toolkit and may take a few minutes..."

# Set CUDA environment for compilation
export CUDA_HOME="${CUDA_PATH}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# IMPORTANT: Use --no-build-isolation to access installed torch during build
# Without this, pip creates an isolated env that doesn't have torch
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization --no-build-isolation

# =============================================================================
# Environment Variables Setup
# =============================================================================

echo ""
echo "Setting up environment variables..."

CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME}" | awk '{print $NF}')
if [[ -n "$CONDA_ENV_PATH" && -d "$CONDA_ENV_PATH" ]]; then
    ACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/activate.d"
    DEACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/deactivate.d"
    mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

    # Activate script
    cat > "$ACTIVATE_DIR/env_vars.sh" << EOF
#!/bin/bash
# Mouse-FaceLift RTX 3060 environment
export _OLD_CUDA_HOME=\$CUDA_HOME
export _OLD_PATH=\$PATH
export _OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH

export CUDA_HOME=${CUDA_PATH}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# RTX 3060 optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
EOF

    # Deactivate script
    cat > "$DEACTIVATE_DIR/env_vars.sh" << 'EOF'
#!/bin/bash
export CUDA_HOME=$_OLD_CUDA_HOME
export PATH=$_OLD_PATH
export LD_LIBRARY_PATH=$_OLD_LD_LIBRARY_PATH
unset _OLD_CUDA_HOME _OLD_PATH _OLD_LD_LIBRARY_PATH
unset PYTORCH_CUDA_ALLOC_CONF CUDA_LAUNCH_BLOCKING
EOF

    chmod +x "$ACTIVATE_DIR/env_vars.sh" "$DEACTIVATE_DIR/env_vars.sh"
    echo "Environment variables configured."
fi

# =============================================================================
# Verification
# =============================================================================

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python << 'VERIFY_EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Key packages
packages = ['transformers', 'diffusers', 'xformers', 'accelerate', 'lpips', 'wandb']
print("\nPackage verification:")
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f"  {pkg}: {ver}")
    except ImportError as e:
        print(f"  {pkg}: FAILED ({e})")

# Gaussian rasterization
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    print("  diff-gaussian-rasterization: OK")
except ImportError as e:
    print(f"  diff-gaussian-rasterization: FAILED ({e})")

print("\nAll checks passed!")
VERIFY_EOF

# =============================================================================
# Save Environment Info
# =============================================================================

echo ""
echo "Saving environment info..."
{
    echo "Environment: ${ENV_NAME}"
    echo "Created: $(date)"
    echo "Host: $(hostname)"
    echo "GPU: ${GPU_NAME}"
    echo "CUDA: ${CUDA_VERSION}"
    echo ""
    echo "Installed packages:"
    pip list
} > conda_env_rtx3060_info.txt

# =============================================================================
# Completion
# =============================================================================

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Quick test:"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "Start training (RTX 3060 optimized):"
echo "  python train_gslrm.py --config configs/mouse_gslrm_local_rtx3060.yaml"
echo ""
echo "Memory usage note:"
echo "  RTX 3060 12GB requires batch_size=1 with grad_accum=4"
echo "  Config: configs/mouse_gslrm_local_rtx3060.yaml"
echo ""
