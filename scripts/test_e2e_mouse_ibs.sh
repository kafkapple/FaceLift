#!/bin/bash
# Test E2E pipeline with mouse_ibs samples
# Usage: ./scripts/test_e2e_mouse_ibs.sh [--with-sam]

set -e
cd /home/joon/dev/FaceLift

# Paths
SAMPLE_DIR="/home/joon/data/preprocessed/samples/mouse_ibs"
OUTPUT_DIR="outputs/test_mouse_ibs_$(date +%y%m%d_%H%M)"
MVDIFF_CKPT="checkpoints/mvdiffusion/mouse_M5t/checkpoint-1500"
GSLRM_CKPT="checkpoints/gslrm/M5t_E0_1_facelift/best_psnr.pt"
GSLRM_CONFIG="configs/base/gslrm_mouse.yaml"

# Optional SAM
SAM_CKPT=""
if [[ "$1" == "--with-sam" ]]; then
    SAM_CKPT="checkpoints/sam/sam_vit_h.pth"
    if [[ ! -f "$SAM_CKPT" ]]; then
        echo "SAM checkpoint not found. Download with:"
        echo "  mkdir -p checkpoints/sam"
        echo "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O $SAM_CKPT"
        exit 1
    fi
fi

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift

echo "=== Test E2E Pipeline with mouse_ibs samples ==="
echo "  Samples: $SAMPLE_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  MVDiffusion: $MVDIFF_CKPT"
echo "  GS-LRM: $GSLRM_CKPT"
echo "  SAM: ${SAM_CKPT:-'(none - fallback resize mode)'}"
echo ""

mkdir -p "$OUTPUT_DIR"

# Process each camera view
for CAM in cam1 cam2 cam3 cam4; do
    INPUT_IMAGE="$SAMPLE_DIR/${CAM}.png"
    
    if [[ ! -f "$INPUT_IMAGE" ]]; then
        echo "Skip $CAM: file not found"
        continue
    fi
    
    echo "=== Processing $CAM ==="
    
    CMD="python -m mouse_extensions.scripts.inference.run_e2e_inference \
        --input_image $INPUT_IMAGE \
        --mvdiffusion_checkpoint $MVDIFF_CKPT \
        --gslrm_checkpoint $GSLRM_CKPT \
        --gslrm_config $GSLRM_CONFIG \
        --output_dir $OUTPUT_DIR \
        --save_preprocess_steps \
        --turntable_views 60"
    
    if [[ -n "$SAM_CKPT" ]]; then
        CMD="$CMD --sam_checkpoint $SAM_CKPT"
    fi
    
    echo "Running: $CMD"
    eval $CMD
    
    echo ""
done

echo "=== Done! ==="
echo "Results: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
