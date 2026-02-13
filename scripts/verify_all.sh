#!/bin/bash
# Verify TurntableRenderer through all 3 paths: Train, Val, Inference
# Usage: bash scripts/verify_all.sh [GPU_ID] [CHECKPOINT] [NUM_TEMPORAL_FRAMES]

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
mkdir -p $OUT_DIR/{train,val,inference,temporal}

echo "=== Verify All Paths (Train + Val + Inference + Temporal) ==="
echo "GPU: $GPU | Checkpoint: $CKPT"
echo "Temporal frames: $NUM_FRAMES | Output: $OUT_DIR"
echo ""

# ─────────────────────────────────────────────
# Step 1: Inference (render_from_checkpoint, standalone)
# Uses TurntableRenderer with TurntableVideoConfig defaults (orbit_views=120, fps=30)
# ─────────────────────────────────────────────
echo "=== Step 1/4: Inference (render_from_checkpoint via TurntableRenderer) ==="
$PYTHON mouse_extensions/scripts/inference/render_from_checkpoint.py \
    --checkpoint "$CKPT" \
    --config configs/base/gslrm_mouse.yaml \
    --data_path /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt \
    --output_dir ${OUT_DIR}/inference \
    --mode turntable \
    --num_samples 2
echo "Step 1 done."
echo "Inference outputs (per-sample subdirs):"
find ${OUT_DIR}/inference -type f | sort | while read f; do
    echo "  $(basename $(dirname $f))/$(basename $f): $(du -h "$f" | cut -f1)"
done
echo ""

# ─────────────────────────────────────────────
# Steps 2-4: Train + Val + Temporal (single model load)
# ─────────────────────────────────────────────
$PYTHON -c "
import torch, numpy as np, os, sys
from pathlib import Path

NUM_FRAMES = $NUM_FRAMES
ORBIT_VIEWS = 240  # 2x default for longer video
VIEW_NUM_FRAMES = 288  # 2x default (6 cams x hold=24 x 2)

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

# Override turntable config for longer videos
if 'visualization' not in config:
    config.visualization = {}
if 'turntable' not in config.visualization:
    config.visualization.turntable = {}
config.visualization.turntable.orbit_views = ORBIT_VIEWS
config.visualization.turntable.view_hold_frames = 30  # 2x default(15)
config.visualization.turntable.view_num_frames = VIEW_NUM_FRAMES

model = GSLRM(config)
state = torch.load('$CKPT', map_location='cpu')
sd = state.get('module', state.get('model', state))
model.load_state_dict(sd, strict=False)
model = model.cuda().eval()
print('Model loaded')
print(f'  orbit_views={ORBIT_VIEWS}, view_num_frames={VIEW_NUM_FRAMES}')

# Load val data
from mouse_extensions.data.mouse_dataset import MouseViewDataset
config.training.dataset.random_view_selection = False
config.training.dataset.maximize_view_overlap = False
val_dataset = MouseViewDataset(config, split='val')

# ─── Step 2: Train path (save_visualization_outputs) ───
print()
print('=== Step 2/4: Train (save_visualization_outputs) ===')
sample = val_dataset[0]
if sample is None:
    print('ERROR: No valid sample found')
    sys.exit(1)
batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
         for k, v in sample.items()}

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    results = model(batch)
print('Forward done')

train_out = '$OUT_DIR/train'
os.makedirs(train_out, exist_ok=True)
model.save_visualization_outputs(train_out, results, batch, save_all_items=True)
print('Train path done:')
for f in sorted(os.listdir(train_out)):
    fpath = os.path.join(train_out, f)
    if os.path.isfile(fpath):
        print(f'  {f}: {os.path.getsize(fpath)/1024:.1f}K')

# ─── Step 3: Val path (save_validation_results) ───
print()
print('=== Step 3/4: Val (save_validation_results) ===')
val_out = '$OUT_DIR/val'
os.makedirs(val_out, exist_ok=True)
metrics = model.save_validation_results(
    val_out, results, batch, val_dataset, save_visualizations=True
)
print(f'Val path done: PSNR={metrics.get(\"psnr\", 0):.4f}, SSIM={metrics.get(\"ssim\", 0):.4f}')
for root, dirs, files in os.walk(val_out):
    for f in sorted(files):
        fpath = os.path.join(root, f)
        rel = os.path.relpath(fpath, val_out)
        print(f'  {rel}: {os.path.getsize(fpath)/1024:.1f}K')

# ─── Step 4: Temporal (TemporalVideoRenderer, multi-frame) ───
print()
print('=== Step 4/4: Temporal (TemporalVideoRenderer) ===')
from mouse_extensions.visualization.turntable_renderer import (
    TurntableRenderer, TurntableVideoConfig, TemporalVideoRenderer
)
cfg = TurntableVideoConfig.from_config(config)
cfg.orbit_views = ORBIT_VIEWS
renderer = TurntableRenderer(cfg)

all_turntables = []
n_samples = min(NUM_FRAMES, len(val_dataset))
print(f'Rendering {n_samples} temporal frames (orbit_views={ORBIT_VIEWS})...')

for idx in range(n_samples):
    sample = val_dataset[idx]
    if sample is None:
        continue
    b = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
         for k, v in sample.items()}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        res = model(b)

    gaussians = res.gaussians[0]
    orbit_frames = renderer._render_orbit(gaussians, None, 384)
    all_turntables.append(orbit_frames)
    print(f'  [{idx+1}/{n_samples}] orbit: {orbit_frames.shape}')

temporal = TemporalVideoRenderer(cfg)
temporal_out = '$OUT_DIR/temporal'
os.makedirs(temporal_out, exist_ok=True)
temporal_result = temporal.render_temporal(
    all_turntables=all_turntables,
    output_dir=temporal_out,
    fixed_angles=[0],
)
print('Temporal done:')
for k, v in temporal_result.items():
    if os.path.isfile(str(v)):
        print(f'  {k}: {v} ({os.path.getsize(v)/1024:.1f}K)')
    else:
        print(f'  {k}: {v}')
"

echo ""
echo "=== Consistency Check ==="
echo "--- Filename conventions (expect turntable_orbit_*, turntable_view_*, turntable_*.jpg) ---"
find ${OUT_DIR} -name "turntable_*" -type f | sort | while read f; do
    relpath=$(echo "$f" | sed "s|${OUT_DIR}/||")
    echo "  $relpath ($(du -h "$f" | cut -f1))"
done
echo ""

echo "--- Frame count check (orbit videos should have consistent frame count) ---"
for vid in $(find ${OUT_DIR} -name "turntable_orbit_*.mp4" -type f | sort); do
    frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$vid" 2>/dev/null || echo "?")
    relpath=$(echo "$vid" | sed "s|${OUT_DIR}/||")
    echo "  $relpath: $frames frames"
done
echo ""

echo "=== All done! ==="
echo "Results:"
echo "  Inference:  ${OUT_DIR}/inference/"
echo "  Train:      ${OUT_DIR}/train/"
echo "  Val:        ${OUT_DIR}/val/"
echo "  Temporal:   ${OUT_DIR}/temporal/"
