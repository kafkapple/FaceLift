#!/bin/bash
# Verify TurntableRenderer through ACTUAL train and val code paths
# Usage: bash scripts/verify_train_val.sh [GPU_ID] [CHECKPOINT]

set -e
GPU=${1:-6}
CKPT=${2:-/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_0000000000010080.pt}
OUT_DIR=outputs/verify_turntable
PYTHON=/home/joon/anaconda3/envs/facelift/bin/python

export CUDA_VISIBLE_DEVICES=$GPU
export PATH="/home/joon/anaconda3/envs/facelift/bin:$PATH"
export PYTHONPATH=/home/joon/dev/FaceLift:$PYTHONPATH

cd /home/joon/dev/FaceLift
mkdir -p $OUT_DIR/{train,val}

echo "=== Verify Train + Val Paths ==="
echo "GPU: $GPU"
echo "Checkpoint: $CKPT"
echo "Output: $OUT_DIR"
echo ""

$PYTHON -c "
import torch, numpy as np, os, sys

# Load model
from gslrm.model.gslrm import GSLRM
from train_gslrm import load_modular_config
from omegaconf import OmegaConf

config = load_modular_config(dataset='M5t2', experiment='E0_1_facelift')

def expand_tilde(d):
    if isinstance(d, dict):
        return {k: expand_tilde(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [expand_tilde(v) for v in d]
    elif isinstance(d, str) and d.startswith('~'):
        return os.path.expanduser(d)
    return d
config = OmegaConf.create(expand_tilde(OmegaConf.to_container(config, resolve=True)))

model = GSLRM(config)
state = torch.load('$CKPT', map_location='cpu')
sd = state.get('module', state.get('model', state))
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()
print('Model loaded')

# Load val data
from mouse_extensions.data.mouse_dataset import MouseViewDataset
config.training.dataset.random_view_selection = False
config.training.dataset.maximize_view_overlap = False
val_dataset = MouseViewDataset(config, split='val')

sample = val_dataset[0]
if sample is None:
    print('ERROR: No valid sample found')
    sys.exit(1)
batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
         for k, v in sample.items()}
uid = sample.get('uid', 'test')
print(f'Loaded sample uid={uid}')

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    results = model(batch)
print('Forward done')

# ─── Path 1: TRAIN visualization (save_visualization_outputs) ───
print()
print('=== Path 1/2: Train (save_visualization_outputs) ===')
train_out = '$OUT_DIR/train'
os.makedirs(train_out, exist_ok=True)
model.save_visualization_outputs(train_out, results, batch, save_all_items=True)
print('Train path done.')
train_files = os.listdir(train_out)
print(f'  Files: {train_files}')
for f in sorted(train_files):
    fpath = os.path.join(train_out, f)
    if os.path.isfile(fpath):
        sz = os.path.getsize(fpath)
        print(f'  {f}: {sz/1024:.1f}K')

# ─── Path 2: VAL visualization (save_validation_results) ───
print()
print('=== Path 2/2: Val (save_validation_results) ===')
val_out = '$OUT_DIR/val'
os.makedirs(val_out, exist_ok=True)
metrics = model.save_validation_results(
    val_out, results, batch, val_dataset, save_visualizations=True
)
print('Val path done.')
print(f'  Metrics: PSNR={metrics.get(\"psnr\", \"N/A\"):.4f}, SSIM={metrics.get(\"ssim\", \"N/A\"):.4f}')

# List val outputs (may be in subdirectory)
for root, dirs, files in os.walk(val_out):
    for f in sorted(files):
        fpath = os.path.join(root, f)
        sz = os.path.getsize(fpath)
        rel = os.path.relpath(fpath, val_out)
        print(f'  {rel}: {sz/1024:.1f}K')
"

echo ""
echo "=== Done! ==="
echo "Results:"
echo "  Train: ${OUT_DIR}/train/"
echo "  Val:   ${OUT_DIR}/val/"
echo ""
echo "Copy to Mac:"
echo "  scp -r gpu03:~/dev/FaceLift/${OUT_DIR}/{train,val} ./verify_turntable/"
