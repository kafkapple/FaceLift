#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
source /home/joon/anaconda3/etc/profile.d/conda.sh
conda activate facelift

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

cd /home/joon/dev/FaceLift
python train_diffusion.py     --config configs/mvdiffusion/mouse_mvdiffusion_M5.yaml
