#!/bin/bash
# LGM Setup Script for Mouse-FaceLift Integration
# Run this on gpu05 to install LGM as a backup 3D reconstruction method

set -e

echo "=============================================="
echo "LGM Setup for Mouse-FaceLift"
echo "=============================================="

# Navigate to project directory
cd /home/joon/FaceLift

# Clone LGM repository
if [ ! -d "LGM" ]; then
    echo "Cloning LGM repository..."
    git clone https://github.com/3DTopia/LGM.git
else
    echo "LGM directory exists, skipping clone"
fi

cd LGM

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

# Install diff-gaussian-rasterization (required for LGM)
if [ ! -d "diff-gaussian-rasterization" ]; then
    echo "Installing diff-gaussian-rasterization..."
    git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
    pip install ./diff-gaussian-rasterization
else
    echo "diff-gaussian-rasterization exists, checking if installed..."
    pip show diff-gaussian-rasterization > /dev/null 2>&1 || pip install ./diff-gaussian-rasterization
fi

# Install nvdiffrast for mesh extraction
echo "Installing nvdiffrast..."
pip install git+https://github.com/NVlabs/nvdiffrast 2>/dev/null || echo "nvdiffrast may already be installed"

# Install other requirements
echo "Installing LGM requirements..."
pip install kiui plyfile trimesh pygltflib 2>/dev/null || true

# Download pretrained weights
mkdir -p pretrained
if [ ! -f "pretrained/model_fp16.safetensors" ]; then
    echo "Downloading LGM pretrained weights..."
    wget -q https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16.safetensors -O pretrained/model_fp16.safetensors
    echo "Downloaded: pretrained/model_fp16.safetensors"
else
    echo "Pretrained weights already exist"
fi

echo ""
echo "=============================================="
echo "LGM Setup Complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  # Generate 4-view images first with MVDiffusion"
echo "  python scripts/inference_with_lgm.py --input_image <image> --mvdiffusion_checkpoint <ckpt> --skip_lgm"
echo ""
echo "  # Then run LGM inference"
echo "  cd LGM && python infer.py big --resume pretrained/model_fp16.safetensors --test_path <4views_dir>"
