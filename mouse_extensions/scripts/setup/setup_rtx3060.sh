#!/usr/bin/env bash
# =============================================================================
# Mouse-FaceLift Environment Setup for RTX 3060 (12GB VRAM)
# =============================================================================
#
# Target: Consumer GPU workstation
#   - NVIDIA RTX 3060 (12GB) - Ampere GA106, CC 8.6
#   - Also works on: RTX 3070/3080/3090, RTX 4060/4070/4080/4090
#
# Usage:
#   bash setup_rtx3060.sh [ENV_NAME]
#
# Created: 2026-02-07
# =============================================================================

set -eo pipefail

ENV_NAME="${1:-facelift}"
PYTHON_VERSION="3.11"

echo "============================================="
echo "Mouse-FaceLift RTX 3060 Setup"
echo "============================================="
echo "Environment: ${ENV_NAME}"
echo "Python: ${PYTHON_VERSION}"
echo ""

# --- Find conda ---
CONDA_EXE=""
for path in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda"; do
    [ -f "$path" ] && CONDA_EXE="$path" && break
done
command -v conda &>/dev/null && CONDA_EXE=$(which conda)

if [ -z "$CONDA_EXE" ]; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

CONDA_BASE=$(dirname $(dirname ${CONDA_EXE}))
source "${CONDA_BASE}/etc/profile.d/conda.sh"
echo "conda: $(conda --version)"

# --- System checks ---
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. CUDA driver may not be installed."
fi

# --- Create environment ---
echo ""
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' exists."
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborting."; exit 1; }
    conda env remove -n "${ENV_NAME}" -y
fi

echo "Creating ${ENV_NAME}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"

# --- PyTorch ---
echo ""
echo "Installing PyTorch (CUDA 12.4)..."
pip install --upgrade pip
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# --- Dependencies ---
echo ""
echo "Installing dependencies..."
pip install packaging typing-extensions

# AI/ML
pip install transformers diffusers[torch] huggingface-hub accelerate
pip install xformers

# Vision
pip install Pillow opencv-python scikit-image lpips
pip install facenet-pytorch --no-deps
pip install rembg

# Scientific
pip install numpy matplotlib scikit-learn einops jaxtyping pytorch-msssim scipy

# Utilities
pip install easydict pyyaml termcolor plyfile tqdm gradio pandas rich
pip install wandb --only-binary=:all:
pip install videoio ffmpeg-python

# --- diff-gaussian-rasterization ---
echo ""
echo "Installing diff-gaussian-rasterization..."
conda install -c conda-forge cuda-nvcc=12.4 cuda-cudart-dev=12.4 -y
export CUDA_HOME="$CONDA_PREFIX"
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization --no-build-isolation

# --- Environment variables ---
echo ""
echo "Configuring environment variables..."
CONDA_ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"
ACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_ENV_PATH/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/env_vars.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# RTX 3060 optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use GPU 0 by default (single GPU)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi
ACTIVATE_EOF

cat > "$DEACTIVATE_DIR/env_vars.sh" << 'DEACTIVATE_EOF'
#!/bin/bash
unset PYTORCH_CUDA_ALLOC_CONF
DEACTIVATE_EOF

chmod +x "$ACTIVATE_DIR/env_vars.sh" "$DEACTIVATE_DIR/env_vars.sh"

# --- Verify ---
echo ""
echo "============================================="
echo "Verification"
echo "============================================="
python << 'PYEOF'
import sys
print(f"Python: {sys.version.split()[0]}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"VRAM: {mem:.1f} GB")

for pkg in ['transformers', 'diffusers', 'xformers', 'lpips', 'wandb', 'accelerate']:
    try:
        __import__(pkg)
        print(f"  {pkg}: OK")
    except:
        print(f"  {pkg}: FAILED")

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings
    print("  diff-gaussian-rasterization: OK")
except:
    print("  diff-gaussian-rasterization: FAILED")
PYEOF

echo ""
echo "============================================="
echo "Setup Complete!"
echo "============================================="
echo ""
echo "Activate: conda activate ${ENV_NAME}"
echo ""
echo "GS-LRM:  python train_gslrm.py -b configs/mouse/uniform/base_uniform_v2.yaml -e configs/mouse/rtx3060/gslrm_3060.yaml"
echo "MVDiff:  cd mvdiffusion && accelerate launch --config_file 1gpu.yaml train_diffusion.py --config ../configs/mouse/rtx3060/mvdiffusion_3060.yaml"
