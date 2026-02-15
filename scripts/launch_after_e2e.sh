#!/bin/bash
# Wait for E2E eval to finish, compute metrics, then launch E3
set -e

E2E_PID=1438460
LOG_DIR=/home/joon/dev/FaceLift/logs

echo "[$(date)] Waiting for E2E eval (PID $E2E_PID) to finish..."

# Wait for E2E eval process to finish
while kill -0 $E2E_PID 2>/dev/null; do
    sleep 60
    PROGRESS=$(grep 'Processing:' $LOG_DIR/h6_v3_e2e_alpha10.log 2>/dev/null | tail -1 | grep -oP '\d+/360' || echo '?/?')
    echo "[$(date)] E2E eval: $PROGRESS"
done

echo "[$(date)] E2E eval finished. Computing metrics..."

# Compute E2E metrics
cd /home/joon/dev/FaceLift
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

export CUDA_VISIBLE_DEVICES=6
python -m mouse_extensions.scripts.eval.compute_e2e_metrics     --output_dir outputs/h6_v3_e2e/alpha10_v3     --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5     --skip_input_view 0     > $LOG_DIR/h6_v3_e2e_alpha10_metrics.log 2>&1

echo "[$(date)] Metrics computed. Launching E3 (extrinsic+add) on GPU 6..."

# Launch E3 pose extrinsic+add
export CUDA_VISIBLE_DEVICES=6
PYTHONUNBUFFERED=1 nohup accelerate launch     --config_file configs/accelerate/1gpu.yaml     train_diffusion.py --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_pose_extrinsic_add.yaml     > $LOG_DIR/mvdiff_M5t2_pose_extrinsic_add.log 2>&1 &

E3_PID=$!
echo "[$(date)] E3 launched with PID $E3_PID"
echo "E3_PID=$E3_PID" > $LOG_DIR/e3_pid.txt
