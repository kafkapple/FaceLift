# FaceLift Mouse - Project Structure

## Overview

This is a fork of [weijielyu/FaceLift](https://github.com/weijielyu/FaceLift) adapted for mouse 3D reconstruction.

## Directory Structure

```
FaceLift/
├── configs/
│   ├── gslrm.yaml                 # Original pretrained config
│   ├── gslrm_pretrain_256.yaml    # 256 resolution
│   ├── gslrm_pretrain_512.yaml    # 512 resolution
│   ├── mvdiffusion.yaml           # MVDiffusion config
│   └── mouse/
│       ├── gslrm_recommended.yaml # Mouse training (recommended)
│       └── gslrm_deterministic.yaml
│
├── scripts/
│   ├── convert_markerless_unified.py  # Preprocessing (v5/v10/v11)
│   ├── generate_temporal_split.py     # Train/Val split
│   ├── generate_mouse_prompt_embeds.py
│   ├── generate_prompt_embeds.py
│   ├── download_checkpoints.py
│   └── download_weights.py
│
├── gslrm/
│   ├── data/
│   │   ├── dataset.py             # Original dataset
│   │   └── mouse_dataset.py       # Mouse dataset (added)
│   └── model/
│       ├── gslrm.py               # Main model + loss
│       ├── gaussians_renderer.py  # GS rendering
│       └── utils_*.py             # Utilities
│
├── tests/                         # Test scripts
├── docs/                          # Documentation
├── _archive/                      # Archived files
│
├── train_gslrm.py                 # Training script
├── inference_mouse.py             # Mouse inference
├── inference.py                   # Original inference
├── train_diffusion.py             # MVDiffusion training
├── gradio_app.py                  # Demo app
├── setup_env.sh                   # Original setup
└── setup_gpu03.sh                 # GPU03 setup
```

## Key Files

### Training
- `train_gslrm.py`: Main training script with DDP support
- `configs/mouse/gslrm_recommended.yaml`: Recommended config

### Preprocessing
- `scripts/convert_markerless_unified.py`: Unified preprocessing (v5/v10/v11)
  - v5: Object centered, cx=256 fixed (deprecated)
  - v10: Object centered, cx/cy corrected
  - v11: PP centered, cx=cy=256 exact (recommended)

### Data
- `gslrm/data/mouse_dataset.py`: Mouse dataset with auto mask generation

### Inference
- `inference_mouse.py`: Complete mouse 3D reconstruction pipeline

## Quick Start

```bash
# 1. Preprocessing
python scripts/convert_markerless_unified.py \
    --version v11 \
    --input_dir /path/to/markerless_mouse \
    --output_dir /path/to/output

# 2. Generate split
python scripts/generate_temporal_split.py \
    --data_dir /path/to/output

# 3. Training
torchrun --nproc_per_node=1 train_gslrm.py \
    --config configs/mouse/gslrm_recommended.yaml
```

## Changes from Original

| Component | Original | Mouse Fork |
|-----------|----------|------------|
| Dataset | Objaverse/Human | Mouse markerless |
| Config | gslrm.yaml | mouse/gslrm_recommended.yaml |
| Preprocessing | None | convert_markerless_unified.py |
| Mask | RGBA alpha | Auto-generated (threshold) |
| Background | Transparent | White (loss weighted) |

## Archived Components

- `scripts/_archive/lgm/`: LGM experiments (unused)
- `scripts/_archive/legacy/`: Old preprocessing scripts
- `scripts/_archive/visualization/`: Debug visualizations
- `tests/`: Test scripts
