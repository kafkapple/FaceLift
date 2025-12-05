#!/usr/bin/env bash
# =============================================================================
# Mouse-FaceLift Conda Environment Setup Script
# =============================================================================
#
# This script creates a new conda environment for Mouse-FaceLift training.
# Designed for both local development and gpu05 server.
#
# Usage:
#   bash setup_mouse_env.sh [ENV_NAME]
#
# Arguments:
#   ENV_NAME: Name of the conda environment (default: mouse_facelift)
#
# Prerequisites:
#   - conda installed and initialized
#   - CUDA 12.4 compatible GPU (for gpu05)
#
# Author: Claude Code (AI-assisted)
# Date: 2024-12-04
# =============================================================================

set -eo pipefail
# Note: Removed -u flag because conda activate scripts may use unbound variables
IFS=$'\n\t'

# Configuration
ENV_NAME="${1:-mouse_facelift}"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.4"

echo "=============================================="
echo "Mouse-FaceLift Environment Setup"
echo "=============================================="
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "CUDA version: ${CUDA_VERSION}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install miniconda or anaconda first"
    exit 1
fi

# Initialize conda for script
eval "$(conda shell.bash hook)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Warning: Environment '${ENV_NAME}' already exists!"
    read -p "Do you want to remove and recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Aborting. Please use a different environment name or activate the existing one."
        exit 1
    fi
fi

# Create new conda environment
echo ""
echo "Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Activate environment
echo "Activating environment..."
conda activate "${ENV_NAME}"

# Verify activation
if [[ "${CONDA_DEFAULT_ENV}" != "${ENV_NAME}" ]]; then
    echo "Error: Failed to activate environment"
    exit 1
fi

echo "Environment activated: ${CONDA_DEFAULT_ENV}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# =============================================================================
# Install Core Dependencies
# =============================================================================

echo ""
echo "Installing core dependencies..."

# Core packaging
pip install packaging==24.2 typing-extensions==4.14.0

# PyTorch with CUDA support (CUDA 12.4)
echo ""
echo "Installing PyTorch with CUDA ${CUDA_VERSION} support..."
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall

# Verify CUDA availability
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# =============================================================================
# Install AI/ML Libraries
# =============================================================================

echo ""
echo "Installing AI/ML libraries..."

pip install transformers==4.44.2 \
    diffusers[torch]==0.30.3 \
    huggingface-hub==0.35.3 \
    xformers==0.0.27.post2 \
    accelerate==0.33.0

# =============================================================================
# Install Computer Vision & Image Processing
# =============================================================================

echo ""
echo "Installing computer vision libraries..."

pip install Pillow==10.4.0 \
    opencv-python==4.10.0.84 \
    scikit-image==0.21.0 \
    lpips==0.1.4

# Face detection (without deps to avoid conflicts)
pip install facenet-pytorch --no-deps

# Background removal
pip install rembg

# =============================================================================
# Install Scientific Computing
# =============================================================================

echo ""
echo "Installing scientific computing libraries..."

pip install numpy==1.26.4 \
    matplotlib==3.7.5 \
    scikit-learn==1.3.2 \
    einops==0.8.0 \
    jaxtyping==0.2.19 \
    pytorch-msssim==1.0.0

# =============================================================================
# Install Utilities & Configuration
# =============================================================================

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

# wandb requires Go to build from source, use binary wheel
pip install wandb --only-binary=:all:

# Video processing
pip install videoio==0.3.0 ffmpeg-python==0.2.0

# =============================================================================
# Install Gaussian Splatting Rasterizer
# =============================================================================

echo ""
echo "Installing diff-gaussian-rasterization..."
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization

# =============================================================================
# Verify Installation
# =============================================================================

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python << 'EOF'
import sys
print(f"Python: {sys.version}")

# Core
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Check key packages
packages = [
    'transformers', 'diffusers', 'xformers', 'accelerate',
    'PIL', 'cv2', 'numpy', 'einops', 'lpips', 'wandb', 'yaml', 'easydict'
]

print("\nPackage verification:")
for pkg in packages:
    try:
        if pkg == 'PIL':
            from PIL import Image
            print(f"  {pkg}: OK")
        elif pkg == 'cv2':
            import cv2
            print(f"  {pkg}: {cv2.__version__}")
        elif pkg == 'yaml':
            import yaml
            print(f"  {pkg}: OK")
        else:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'OK')
            print(f"  {pkg}: {ver}")
    except ImportError as e:
        print(f"  {pkg}: FAILED ({e})")

# Try importing diff-gaussian-rasterization
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    print("  diff-gaussian-rasterization: OK")
except ImportError as e:
    print(f"  diff-gaussian-rasterization: FAILED ({e})")

print("\nSetup complete!")
EOF

# =============================================================================
# Setup Permanent Environment Variables (for gpu05)
# =============================================================================

# Detect conda env path
CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME} " | awk '{print $NF}')
if [[ -z "$CONDA_ENV_PATH" ]]; then
    CONDA_ENV_PATH=$(conda info --envs | grep "^${ENV_NAME}$" -A1 | tail -1 | awk '{print $1}')
fi

if [[ -n "$CONDA_ENV_PATH" && -d "$CONDA_ENV_PATH" ]]; then
    echo ""
    echo "Setting up permanent environment variables..."

    ACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/activate.d"
    DEACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/deactivate.d"

    mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

    # Create activate script
    cat > "$ACTIVATE_DIR/env_vars.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Save original values
export _OLD_CUDA_HOME=$CUDA_HOME
export _OLD_PATH=$PATH
export _OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export _OLD_CC=$CC
export _OLD_CXX=$CXX

# Set Mouse-FaceLift environment
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
ACTIVATE_EOF

    # Create deactivate script
    cat > "$DEACTIVATE_DIR/env_vars.sh" << 'DEACTIVATE_EOF'
#!/bin/bash
# Restore original values
export CUDA_HOME=$_OLD_CUDA_HOME
export PATH=$_OLD_PATH
export LD_LIBRARY_PATH=$_OLD_LD_LIBRARY_PATH
export CC=$_OLD_CC
export CXX=$_OLD_CXX

# Cleanup
unset _OLD_CUDA_HOME _OLD_PATH _OLD_LD_LIBRARY_PATH _OLD_CC _OLD_CXX
DEACTIVATE_EOF

    chmod +x "$ACTIVATE_DIR/env_vars.sh" "$DEACTIVATE_DIR/env_vars.sh"
    echo "Environment variables configured at: $ACTIVATE_DIR/env_vars.sh"
else
    echo "Warning: Could not find conda environment path for ${ENV_NAME}"
fi

# =============================================================================
# Post-installation Notes
# =============================================================================

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify GPU access:"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate ${ENV_NAME}"
echo "  2. Process mouse data: python scripts/process_mouse_data.py --help"
echo "  3. Run overfitting test: python train_mouse.py --config configs/mouse_config.yaml --overfit 10"
echo "  4. Full training: torchrun --nproc_per_node 4 train_mouse.py --config configs/mouse_config.yaml"
echo ""

# Save environment info
echo "Saving environment info to conda_env_info.txt..."
{
    echo "Environment: ${ENV_NAME}"
    echo "Created: $(date)"
    echo "Python: ${PYTHON_VERSION}"
    echo "CUDA: ${CUDA_VERSION}"
    echo ""
    echo "Installed packages:"
    pip list
} > conda_env_info.txt

echo "Done!"
