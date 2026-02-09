#!/bin/bash
# Verify turntable + temporal visualization
# Usage: bash scripts/verify_turntable.sh [GPU_ID] [CHECKPOINT] [NUM_FRAMES]
#
# Tests:
#   Step 1: Inference (render_from_checkpoint, orbit only)
#   Step 2: TurntableRenderer (orbit + view_traj + grid, single frame)
#   Step 3: TemporalVideoRenderer (time_fixed + time_rotating + time_grid_6view)
#
# Results: outputs/verify_turntable/{inference,renderer,temporal}/

set -e
GPU=${1:-6}
CKPT=${2:-/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_0000000000010080.pt}
NUM_FRAMES=${3:-10}
OUT_DIR=outputs/verify_turntable
PYTHON=/home/joon/anaconda3/envs/facelift/bin/python

export CUDA_VISIBLE_DEVICES=$GPU
export PATH="/home/joon/anaconda3/envs/facelift/bin:$PATH"
export PYTHONPATH=/home/joon/dev/FaceLift:$PYTHONPATH

cd /home/joon/dev/FaceLift
mkdir -p $OUT_DIR/{inference,renderer,temporal}

echo "=== Verify Turntable + Temporal Visualization ==="
echo "GPU: $GPU"
echo "Checkpoint: $CKPT"
echo "Temporal frames: $NUM_FRAMES"
echo "Output: $OUT_DIR"
echo ""

# ─────────────────────────────────────────────
# Step 1: Inference (standalone, orbit only)
# ─────────────────────────────────────────────
echo "=== Step 1/3: Inference (render_from_checkpoint) ==="
$PYTHON mouse_extensions/scripts/inference/render_from_checkpoint.py \
    --checkpoint "$CKPT" \
    --config configs/base/gslrm_mouse.yaml \
    --data_path /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt \
    --output_dir ${OUT_DIR}/inference \
    --mode turntable \
    --num_samples 2
echo "Step 1 done."
ls -la ${OUT_DIR}/inference/

# ─────────────────────────────────────────────
# Step 2+3: TurntableRenderer + TemporalVideoRenderer
# ─────────────────────────────────────────────
echo ""
echo "=== Step 2/3: TurntableRenderer (single-frame) ==="
echo "=== Step 3/3: TemporalVideoRenderer (multi-frame) ==="
$PYTHON -c "
import torch, numpy as np, os, sys
from pathlib import Path

NUM_FRAMES = $NUM_FRAMES

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

# Load multiple val samples (temporal frames)
from mouse_extensions.data.mouse_dataset import MouseViewDataset
config.training.dataset.random_view_selection = False
config.training.dataset.maximize_view_overlap = False
val_dataset = MouseViewDataset(config, split='val')
n_samples = min(NUM_FRAMES, len(val_dataset))
print(f'Loading {n_samples} samples for temporal test...')

all_turntables = []
first_result = None
first_batch = None

for idx in range(n_samples):
    sample = val_dataset[idx]
    if sample is None:
        continue
    batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
             for k, v in sample.items()}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        results = model(batch)

    gaussians = results.gaussians[0]

    # For first sample: full TurntableRenderer test (Step 2)
    if first_result is None:
        first_result = results
        first_batch = batch

    # Render orbit for temporal (Step 3)
    from mouse_extensions.visualization.turntable_renderer import TurntableRenderer, TurntableVideoConfig
    cfg = TurntableVideoConfig.from_config(config)
    renderer = TurntableRenderer(cfg)
    orbit_frames = renderer._render_orbit(gaussians, None, 384)
    all_turntables.append(orbit_frames)
    print(f'  [{idx+1}/{n_samples}] orbit: {orbit_frames.shape}')

# ─── Step 2: TurntableRenderer (first sample) ───
print()
print('=== Step 2 Results: TurntableRenderer ===')
gaussians = first_result.gaussians[0]
dataset_c2ws = first_batch.get('target_RT', first_batch.get('c2w', None))
if dataset_c2ws is not None:
    dataset_c2ws = dataset_c2ws[0].cpu().numpy()
dataset_fxfycxcy = first_batch.get('target_fxfycxcy', first_batch.get('fxfycxcy', None))
if dataset_fxfycxcy is not None:
    dataset_fxfycxcy = dataset_fxfycxcy[0].cpu().numpy()
target_images = first_batch.get('target_image', first_batch.get('image', None))
if target_images is not None:
    target_images = target_images[0]

out_renderer = '$OUT_DIR/renderer'
os.makedirs(out_renderer, exist_ok=True)
result = renderer.render_all(
    gaussians=gaussians,
    dataset_c2ws=dataset_c2ws,
    dataset_fxfycxcy=dataset_fxfycxcy,
    output_dir=out_renderer,
    uid='test',
    rendering_resolution=384,
    original_resolution=384,
    target_images=target_images,
    input_indices=list(range(6)),
    view_indices=list(range(6)),
)
for k, v in result.items():
    print(f'  {k}: {v}')

# ─── Step 3: TemporalVideoRenderer ───
print()
print('=== Step 3 Results: TemporalVideoRenderer ===')
print(f'Temporal frames: {len(all_turntables)}')
from mouse_extensions.visualization.turntable_renderer import TemporalVideoRenderer
temporal = TemporalVideoRenderer(cfg)
out_temporal = '$OUT_DIR/temporal'
os.makedirs(out_temporal, exist_ok=True)
temporal_result = temporal.render_temporal(
    all_turntables=all_turntables,
    output_dir=out_temporal,
    fixed_angles=[0],
)
for k, v in temporal_result.items():
    print(f'  {k}: {v}')
"

echo ""
echo "=== All done! ==="
echo "Results:"
echo "  Inference:  ${OUT_DIR}/inference/"
echo "  Renderer:   ${OUT_DIR}/renderer/"
echo "  Temporal:   ${OUT_DIR}/temporal/"
echo ""
echo "Copy to Mac:"
echo "  scp -r gpu03:~/dev/FaceLift/${OUT_DIR}/ ./verify_turntable"
