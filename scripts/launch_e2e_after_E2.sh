#!/bin/bash
# Auto-launch E2E eval after E2 (resume 20K) completes
# GPU 7 → wait → E2E eval on GPU 7
set -e

E2_MAIN_PID=1251813
LOG=/home/joon/dev/FaceLift/logs/mvdiff_M5t2_randref_20k_resume.log
CKPT_DIR="/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_randref_sparse"
GSLRM_CKPT="/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt"
OUTPUT_DIR="outputs/phase3_e2e/E2_resume_20k"

cd /home/joon/dev/FaceLift
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

echo "[$(date)] Waiting for E2 (PID $E2_MAIN_PID) to finish..."

# Poll log file for completion (more robust than kill -0)
while true; do
    # Check if process is still alive
    if ! kill -0 $E2_MAIN_PID 2>/dev/null; then
        echo "[$(date)] E2 process exited."
        break
    fi

    # Also check log for 20000/20000 completion
    if grep -q "20000/20000" "$LOG" 2>/dev/null; then
        echo "[$(date)] E2 reached 20000 steps in log."
        sleep 30  # Wait for final checkpoint save
        break
    fi

    # Progress report every 5 min
    STEP=$(grep -oP '\d+/20000' "$LOG" 2>/dev/null | tail -1)
    echo "[$(date)] E2 progress: $STEP"
    sleep 300
done

echo "[$(date)] E2 training completed. Finding last checkpoint..."

# Find the last checkpoint
LAST_CKPT=$(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
echo "[$(date)] Using checkpoint: $LAST_CKPT"

# Run E2E inference on GPU 7
echo "[$(date)] Launching E2E eval..."
export CUDA_VISIBLE_DEVICES=7

python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split data_mouse_t2_test.txt \
    --input_view_idx 0 \
    --skip_preprocess \
    --model M5t \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --gslrm_checkpoint "$GSLRM_CKPT" \
    --mvdiffusion_checkpoint "$LAST_CKPT" \
    --mvdiffusion_base checkpoints/mvdiffusion/pipeckpts \
    --prompt_embed_path /home/joon/dev/FaceLift/mvdiffusion/data/mouse_prompt_embeds_6view_1024 \
    --prefer_ema \
    --output_dir "$OUTPUT_DIR" \
    > logs/phase3_e2e_E2_resume.log 2>&1

echo "[$(date)] E2E inference done. Computing metrics..."

python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
    --output_dir "$OUTPUT_DIR" \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --skip_input_view 0 \
    > logs/phase3_e2e_E2_resume_metrics.log 2>&1

echo "[$(date)] All done! Check:"
echo "  Inference: logs/phase3_e2e_E2_resume.log"
echo "  Metrics:   logs/phase3_e2e_E2_resume_metrics.log"
echo "  Output:    $OUTPUT_DIR"

cat logs/phase3_e2e_E2_resume_metrics.log
