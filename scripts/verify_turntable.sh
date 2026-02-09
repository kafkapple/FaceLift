#!/bin/bash
# Verify turntable visualization (rotation direction + smooth trajectory)
# Usage: bash scripts/verify_turntable.sh [GPU_ID] [CHECKPOINT]
#
# Runs 3 tests sequentially:
#   1. Training path: 1 step from scratch → triggers save_current_results
#   2. Validation path: 1 validation run from checkpoint
#   3. Inference path: render_from_checkpoint standalone
#
# Results: outputs/verify_turntable/

set -e
GPU=${1:-6}
CKPT=${2:-/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_0000000000009200.pt}
OUT_DIR=outputs/verify_turntable
PYTHON=/home/joon/anaconda3/envs/facelift/bin/python
TORCHRUN=/home/joon/anaconda3/envs/facelift/bin/torchrun
VAL_DATA=/home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt

export CUDA_VISIBLE_DEVICES=$GPU
export PATH="/home/joon/anaconda3/envs/facelift/bin:$PATH"

cd /home/joon/dev/FaceLift
mkdir -p $OUT_DIR/{train,val,inference}

echo "=== Verify Turntable Visualization ==="
echo "GPU: $GPU"
echo "Checkpoint: $CKPT"
echo "Output: $OUT_DIR"
echo ""

# ─────────────────────────────────────────────
# Step 1: Inference (standalone, fastest)
# ─────────────────────────────────────────────
echo "=== Step 1/2: Inference (render_from_checkpoint) ==="
$PYTHON mouse_extensions/scripts/inference/render_from_checkpoint.py \
    --checkpoint "$CKPT" \
    --config configs/base/gslrm_mouse.yaml \
    --data_path "$VAL_DATA" \
    --output_dir ${OUT_DIR}/inference \
    --mode turntable \
    --num_samples 2
echo "Inference results:"
ls -la ${OUT_DIR}/inference/

# ─────────────────────────────────────────────
# Step 2: TurntableRenderer direct test
#   Load checkpoint, render orbit + view_trajectory + grid
#   This tests the same code path as training/validation
# ─────────────────────────────────────────────
echo ""
echo "=== Step 2/2: TurntableRenderer direct test ==="
$PYTHON -c "
import torch, numpy as np, os, sys
from pathlib import Path

# Load model
from gslrm.model.gslrm import GSLRM
from train_gslrm import load_modular_config
from omegaconf import OmegaConf

config = load_modular_config(dataset='M5t2', experiment='E0_1_facelift')

# Expand ~ in paths
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
from gslrm.data.dataset import RandomViewDataset
val_dataset = RandomViewDataset(config, split='val')

# Try multiple samples until one loads
sample = None
for idx in range(min(50, len(val_dataset))):
    try:
        sample = val_dataset[idx]
        if sample is not None:
            break
    except Exception:
        continue

if sample is None:
    print('ERROR: No valid sample found')
    sys.exit(1)

batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
uid = sample.get('uid', 'test')
print(f'Loaded sample uid={uid}')

with torch.no_grad():
    results = model(batch)
print('Forward done')

# TurntableRenderer
from mouse_extensions.visualization.turntable_renderer import TurntableRenderer, TurntableVideoConfig
cfg = TurntableVideoConfig.from_config(config)
print(f'Config: view_smooth={cfg.view_smooth}, rotation_direction={cfg.rotation_direction}')

gaussians = results.gaussians[0]
dataset_c2ws = batch.get('target_RT', None)
if dataset_c2ws is not None:
    dataset_c2ws = dataset_c2ws[0].cpu().numpy()
dataset_fxfycxcy = batch.get('target_fxfycxcy', None)
if dataset_fxfycxcy is not None:
    dataset_fxfycxcy = dataset_fxfycxcy[0].cpu().numpy()

target_images = batch.get('target_image', None)
if target_images is not None:
    target_images = target_images[0]

out = '$OUT_DIR/renderer'
os.makedirs(out, exist_ok=True)

renderer = TurntableRenderer(cfg)
result = renderer.render_all(
    gaussians=gaussians,
    dataset_c2ws=dataset_c2ws,
    dataset_fxfycxcy=dataset_fxfycxcy,
    output_dir=out,
    uid=str(uid),
    rendering_resolution=384,
    original_resolution=384,
    target_images=target_images,
    input_indices=list(range(6)),
    view_indices=list(range(6)),
)
print(f'Generated:')
for k, v in result.items():
    print(f'  {k}: {v}')
"
echo "Renderer results:"
ls -la ${OUT_DIR}/renderer/

echo ""
echo "=== All done! ==="
echo "Results:"
echo "  Inference:  ${OUT_DIR}/inference/"
echo "  Renderer:   ${OUT_DIR}/renderer/"
echo ""
echo "Copy to Mac:"
echo "  scp -r gpu03:~/dev/FaceLift/${OUT_DIR}/ ."
