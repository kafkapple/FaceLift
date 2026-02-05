#!/bin/bash
# Temporal training with increased file limit

export CUDA_VISIBLE_DEVICES=6
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || echo 'Warning: Could not increase file limit'
echo "File limit: $(ulimit -n)"

cd /home/joon/dev/FaceLift
/home/joon/anaconda3/envs/facelift/bin/python train_gslrm.py --config configs/mouse/temporal_M5t2_train.yaml
