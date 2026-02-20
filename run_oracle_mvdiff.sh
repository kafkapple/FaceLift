#!/bin/bash
# A2: Oracle MVDiff - GS-LRM Sensitivity Analysis
# Runs GS-LRM inference on hybrid GT/MVDiff datasets
#
# Prerequisites:
#   1. prepare_oracle_mvdiff.py prepare (creates hybrid datasets)
#   2. This script (runs GS-LRM inference + fair metrics)
#
# Usage:
#   # Dry run (5 frames, level 1 only)
#   bash run_oracle_mvdiff.sh --dry-run
#
#   # Full run (all levels)
#   bash run_oracle_mvdiff.sh
#
#   # Single level
#   bash run_oracle_mvdiff.sh --level 2

set -e

cd /home/joon/dev/FaceLift
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

# Config
GSLRM_CKPT="/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt"
GSLRM_CONFIG="configs/base/gslrm_mouse.yaml"
GT_DIR="/home/joon/data/preprocessed/FaceLift_mouse/M5"
OUTPUT_BASE="outputs/analysis/oracle_mvdiff"
SPLIT="data_mouse_t2_test.txt"
GPU="${CUDA_VISIBLE_DEVICES:-7}"

# Parse args
DRY_RUN=false
LEVEL_FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --level) LEVEL_FILTER="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU

echo "A2: Oracle MVDiff - GS-LRM Sensitivity Analysis"
echo "================================================"
echo "GPU: $GPU"
echo "GS-LRM checkpoint: $GSLRM_CKPT"
echo "Output base: $OUTPUT_BASE"
echo "Dry run: $DRY_RUN"
echo ""

# Determine which levels to run
if [ -n "$LEVEL_FILTER" ]; then
    LEVELS=($LEVEL_FILTER)
else
    LEVELS=(1 2 3 4)
fi

# Step 0: Verify checkpoint and hybrid data
echo "[$(date)] Step 0: Verification"
if [ ! -f "$GSLRM_CKPT" ]; then
    echo "ERROR: GS-LRM checkpoint not found: $GSLRM_CKPT"
    exit 1
fi

for level in "${LEVELS[@]}"; do
    DATA_DIR="$OUTPUT_BASE/hybrid_level_$level"
    if [ ! -d "$DATA_DIR" ]; then
        echo "ERROR: Hybrid dataset not found: $DATA_DIR"
        echo "Run: python prepare_oracle_mvdiff.py prepare ... first"
        exit 1
    fi
    NFRAMES=$(ls "$DATA_DIR" | grep -E '^[0-9]+$' | wc -l)
    echo "  Level $level: $NFRAMES frames found in $DATA_DIR"
done
echo ""

# Step 1: GS-LRM inference for each level
for level in "${LEVELS[@]}"; do
    DATA_DIR="$OUTPUT_BASE/hybrid_level_$level"
    OUT_DIR="$OUTPUT_BASE/inference_level_$level"

    if [ -d "$OUT_DIR/samples" ]; then
        NRENDER=$(ls "$OUT_DIR/samples/" 2>/dev/null | wc -l)
        if [ "$NRENDER" -ge 300 ]; then
            echo "[$(date)] Level $level: SKIP inference ($NRENDER renders exist)"
            continue
        fi
    fi

    echo "[$(date)] Level $level: GS-LRM inference"
    echo "  Data: $DATA_DIR"
    echo "  Output: $OUT_DIR"

    if [ "$DRY_RUN" = true ]; then
        # Dry run: 5 frames only
        python -m mouse_extensions.scripts.inference.run_e2e_inference \
            --data_dir "$DATA_DIR" \
            --split "$SPLIT" \
            --gslrm_checkpoint "$GSLRM_CKPT" \
            --gslrm_config "$GSLRM_CONFIG" \
            --output_dir "$OUT_DIR" \
            --no_turntable --no_mesh --no_metrics \
            --prefer_ema --skip_preprocess \
            --num_frames 5
        echo "  Dry run complete. Check $OUT_DIR/samples/"
        # Verify output
        ls "$OUT_DIR/samples/" 2>/dev/null | head -5
        break  # Only level 1 for dry run
    else
        python -m mouse_extensions.scripts.inference.run_e2e_inference \
            --data_dir "$DATA_DIR" \
            --split "$SPLIT" \
            --gslrm_checkpoint "$GSLRM_CKPT" \
            --gslrm_config "$GSLRM_CONFIG" \
            --output_dir "$OUT_DIR" \
            --no_turntable --no_mesh --no_metrics \
            --prefer_ema --skip_preprocess
    fi
    echo "[$(date)] Level $level: inference complete"
    echo ""
done

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Dry run finished. Verify output, then run without --dry-run."
    exit 0
fi

# Step 2: Fair metrics for each level
echo ""
echo "[$(date)] Step 2: Fair comparison metrics"
for level in "${LEVELS[@]}"; do
    OUT_DIR="$OUTPUT_BASE/inference_level_$level"
    FAIR_JSON="$OUTPUT_BASE/hybrid_level_${level}_fair.json"

    if [ -f "$FAIR_JSON" ]; then
        echo "  Level $level: SKIP metrics (exists: $FAIR_JSON)"
        continue
    fi

    RENDER_DIR="$OUT_DIR/samples"
    if [ ! -d "$RENDER_DIR" ]; then
        echo "  Level $level: SKIP (no renders)"
        continue
    fi

    echo "  Level $level: computing fair metrics..."
    python -m mouse_extensions.scripts.eval.fair_comparison evaluate_fl \
        --render_dir "$RENDER_DIR" \
        --gt_dir "$GT_DIR" \
        --output "$FAIR_JSON" \
        --vis_every 60
    echo "  Saved: $FAIR_JSON"
done

# Step 3: Generate report
echo ""
echo "[$(date)] Step 3: Metrics summary"
python -m mouse_extensions.scripts.eval.prepare_oracle_mvdiff metrics \
    --gt_dir "$GT_DIR" \
    --output_base "$OUTPUT_BASE" \
    2>/dev/null || \
python prepare_oracle_mvdiff.py metrics \
    --gt_dir "$GT_DIR" \
    --output_base "$OUTPUT_BASE" \
    2>/dev/null || \
echo "  (Run metrics manually: python prepare_oracle_mvdiff.py metrics ...)"

echo ""
echo "[$(date)] ALL DONE"
echo "Results: $OUTPUT_BASE/"
