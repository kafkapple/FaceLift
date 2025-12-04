# Mouse-FaceLift Usage Guide

---
date: 2024-12-04
context_name: "2_Research"
tags: [ai-assisted, mouse-reconstruction, multi-view, 3d-reconstruction]
project: FaceLift
status: in-progress
generator: ai-assisted
generator_tool: claude-code
---

## Overview

Mouse-FaceLift adapts the FaceLift 3D reconstruction pipeline for mouse multi-view data.
This guide covers environment setup, data preprocessing, training, and inference.

## Quick Start

```bash
# 1. Create conda environment
bash setup_mouse_env.sh mouse_facelift

# 2. Activate environment
conda activate mouse_facelift

# 3. Process mouse video data
python scripts/process_mouse_data.py \
    --video_dir /path/to/videos \
    --meta_dir /path/to/masks \
    --output_dir data_mouse \
    --num_samples 2000

# 4. Run overfitting test (verify setup)
python train_mouse.py --config configs/mouse_config.yaml --overfit 10

# 5. Full training (multi-GPU)
torchrun --nproc_per_node 4 train_mouse.py --config configs/mouse_config.yaml
```

## Detailed Instructions

### 1. Environment Setup

#### Local Development
```bash
cd /home/joon/dev/FaceLift
bash setup_mouse_env.sh mouse_facelift
conda activate mouse_facelift
```

#### GPU Server (gpu05)
```bash
ssh gpu05
cd /home/joon/FaceLift
bash setup_mouse_env.sh mouse_facelift
conda activate mouse_facelift
```

### 2. Data Preprocessing

The preprocessing script converts multi-view mouse videos into FaceLift training format.

#### Input Data Structure
```
/home/joon/data/markerless_mouse/
├── videos_undist/          # 6 synchronized videos
│   ├── 0.mp4
│   ├── 1.mp4
│   └── ...
└── simpleclick_undist/     # Mask videos (optional)
    ├── 0.mp4
    └── ...

/home/joon/data/markerless_mouse_1_nerf/
├── new_cam.pkl             # Camera calibration
└── transforms.json         # NeRF-style cameras (alternative)
```

#### Run Preprocessing
```bash
python scripts/process_mouse_data.py \
    --video_dir /home/joon/data/markerless_mouse \
    --meta_dir /home/joon/data/markerless_mouse_1_nerf \
    --output_dir data_mouse \
    --num_samples 2000 \
    --image_size 512 \
    --num_views 6
```

#### Output Structure
```
data_mouse/
├── data_mouse_train.txt    # Training sample paths
├── data_mouse_val.txt      # Validation sample paths
├── sample_000000/
│   ├── images/
│   │   ├── cam_000.png     # 512x512 RGBA
│   │   ├── cam_001.png
│   │   └── ...
│   └── opencv_cameras.json # Camera parameters
└── sample_000001/
    └── ...
```

### 3. Training

#### Configuration
Edit `configs/mouse_config.yaml` to adjust:
- Dataset paths
- Batch size
- Learning rate
- Number of views
- Checkpointing frequency

#### Overfitting Test (Recommended First Step)
```bash
# Use 10 samples to verify code works
python train_mouse.py --config configs/mouse_config.yaml --overfit 10
```

Expected behavior:
- Loss should drop to near-zero
- Input images should be perfectly reconstructed
- This validates the pipeline before full training

#### Single GPU Training
```bash
python train_mouse.py --config configs/mouse_config.yaml
```

#### Multi-GPU Training
```bash
# 4 GPUs on single node
torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id ${RANDOM} --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    train_mouse.py --config configs/mouse_config.yaml
```

#### Resume from Checkpoint
```bash
python train_mouse.py --config configs/mouse_config.yaml \
    --load checkpoints/gslrm/mouse/
```

### 4. Inference

#### Single Image
```bash
python inference_mouse.py \
    --input_image path/to/mouse.png \
    --output_dir outputs/ \
    --checkpoint checkpoints/gslrm/mouse/ \
    --save_video
```

#### Directory of Images
```bash
python inference_mouse.py \
    --input_dir examples/mouse/ \
    --output_dir outputs/mouse/ \
    --checkpoint checkpoints/gslrm/mouse/ \
    --num_views 6
```

### 5. Monitoring

#### Weights & Biases
Training automatically logs to W&B if configured:
```yaml
# In configs/mouse_config.yaml
training:
  logging:
    wandb:
      project: "mouse_facelift"
      exp_name: "mouse_6view"
      offline: false
```

View at: https://wandb.ai/

#### TensorBoard (Alternative)
```bash
# Not implemented by default, but can be added
```

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config
training:
  dataloader:
    batch_size_per_gpu: 1  # Default is 2
```

### Slow Data Loading
```yaml
# Adjust workers
training:
  dataloader:
    num_workers: 8  # Increase if CPU bound
    prefetch_factor: 32
```

### Training Not Converging
1. Run overfitting test first
2. Check data visualization in `checkpoints/*/data_examples/`
3. Verify camera parameters are correct
4. Try lower learning rate

### Camera Coordinate Issues
```python
# Debug camera visualization
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cameras(cameras):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cam in cameras:
        c2w = np.linalg.inv(np.array(cam["w2c"]))
        pos = c2w[:3, 3]
        forward = c2w[:3, 2]

        ax.scatter(*pos, c='b', s=100)
        ax.quiver(*pos, *forward, length=0.3, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
```

## Git Workflow

### Sync between local and gpu05

#### On Local (after making changes)
```bash
cd /home/joon/dev/FaceLift
git add -A
git commit -m "feat(mouse): add preprocessing pipeline"
git push
```

#### On gpu05 (before training)
```bash
ssh gpu05
cd /home/joon/FaceLift
git pull
conda activate mouse_facelift
```

#### After training on gpu05
```bash
# On gpu05
git add checkpoints/ outputs/
git commit -m "chore: add training checkpoints"
git push

# On local
cd /home/joon/dev/FaceLift
git pull
```

## File Reference

| File | Purpose |
|------|---------|
| `setup_mouse_env.sh` | Conda environment setup |
| `scripts/process_mouse_data.py` | Video → FaceLift format |
| `gslrm/data/mouse_dataset.py` | PyTorch Dataset class |
| `configs/mouse_config.yaml` | Training configuration |
| `train_mouse.py` | Training script |
| `inference_mouse.py` | Inference script |

## Next Steps

1. [ ] Verify camera coordinate conversion
2. [ ] Run overfitting test
3. [ ] Full training on gpu05
4. [ ] Evaluate on held-out frames
5. [ ] Iterate on hyperparameters
