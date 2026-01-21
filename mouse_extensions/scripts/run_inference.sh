#!/bin/bash
# run_inference.sh - Quick inference wrapper for Mouse-FaceLift
#
# Usage:
#   ./run_inference.sh <input>                    # Single sample
#   ./run_inference.sh <input> --use_zero123pp   # Single image with Zero123++
#
# Examples:
#   ./run_inference.sh data/D6-3/sample_000100
#   ./run_inference.sh my_mouse.png --use_zero123pp

set -e

# Default settings
CHECKPOINT="${CHECKPOINT:-checkpoints/gslrm/mouse/}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/inference}"
TIMESTAMP=$(date +%y%m%d_%H%M%S)

# Parse input
INPUT="$1"
shift || true

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <input> [options]"
    echo ""
    echo "Input can be:"
    echo "  - Sample directory with 6 views (e.g., data/D6-3/sample_000100)"
    echo "  - Single image file (e.g., mouse.png) with --use_zero123pp"
    echo ""
    echo "Options:"
    echo "  --use_zero123pp    Use Zero123++ for single image (required for .png/.jpg)"
    echo "  --checkpoint DIR   GSLRM checkpoint directory"
    echo "  --output_dir DIR   Output directory"
    echo ""
    echo "Environment variables:"
    echo "  CHECKPOINT         Default: checkpoints/gslrm/mouse/"
    echo "  OUTPUT_BASE        Default: outputs/inference"
    exit 1
fi

# Determine input type
if [ -d "$INPUT" ]; then
    # Directory input (6-view sample)
    SAMPLE_NAME=$(basename "$INPUT")
    OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${SAMPLE_NAME}"

    echo "=== Mouse-FaceLift Inference ==="
    echo "Input: $INPUT (6-view sample)"
    echo "Output: $OUTPUT_DIR"
    echo ""

    python inference_mouse.py \
        --sample_dir "$INPUT" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --save_turntable \
        --save_mesh \
        "$@"
else
    # File input (single image)
    IMAGE_NAME=$(basename "$INPUT" | sed 's/\.[^.]*$//')
    OUTPUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${IMAGE_NAME}"

    echo "=== Mouse-FaceLift Inference ==="
    echo "Input: $INPUT (single image)"
    echo "Output: $OUTPUT_DIR"
    echo "Note: Requires --use_zero123pp for single image input"
    echo ""

    python inference_mouse.py \
        --input_image "$INPUT" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --save_turntable \
        --save_mesh \
        "$@"
fi

echo ""
echo "=== Done ==="
echo "Outputs saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
ls -la "$OUTPUT_DIR" 2>/dev/null || echo "(output directory not found)"
