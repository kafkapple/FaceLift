# Mouse Inference Preprocessing

SAM-based preprocessing for arbitrary mouse images to match M5 training format.

## Overview

Analogous to human face preprocessing:
- **Human**: MTCNN (face detection) → rembg (background) → crop_face (alignment)
- **Mouse**: SAM (segmentation) → white background → center + coverage alignment

## M5 Training Format

| Parameter | Value |
|-----------|-------|
| Resolution | 512×512 |
| fx, fy | 549 |
| cx, cy | 256 (image center) |
| Background | White (255, 255, 255) |

## Installation

```bash
# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download SAM checkpoint
mkdir -p checkpoints/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O checkpoints/sam/sam_vit_h.pth
```

## Usage

### Python API

```python
from mouse_extensions.inference.preprocessing import MouseInferencePreprocessor

# With SAM (full preprocessing)
preprocessor = MouseInferencePreprocessor(
    sam_checkpoint="checkpoints/sam/sam_vit_h.pth"
)
result = preprocessor.preprocess("raw_mouse_photo.jpg")
# result.image: 512×512 RGB, white background, mouse centered

# Without SAM (auto-detection + fallback resize)
preprocessor = MouseInferencePreprocessor()
result = preprocessor.preprocess("input.jpg")
# If image is 512×512 with white corners: passes through unchanged
# Otherwise: resizes to 512×512 (fallback mode)
```

### CLI

```bash
# Full E2E with preprocessing
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image raw_mouse.jpg \
    --sam_checkpoint checkpoints/sam/sam_vit_h.pth \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/

# Skip preprocessing (for already M5-format images)
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --input_image m5_format_image.png \
    --skip_preprocess \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse_M5/checkpoint-3500 \
    --gslrm_checkpoint M5t_E0_1_facelift \
    --gslrm_config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/
```

### Debugging

```python
# Visualize preprocessing steps
preprocessor = MouseInferencePreprocessor(sam_checkpoint="...")
steps = preprocessor.visualize_steps("input.jpg", output_dir="debug/")
# Saves: 01_original.png, 02_detection.png, 03_bg_removed.png, 04_normalized.png
```

## Auto-Detection

The preprocessor auto-detects if an image is already in M5 format:
1. Resolution is 512×512 (±10px)
2. At least 3 corners are white (>250)
3. Center has content (not empty)

If auto-detected, preprocessing is skipped. Override with `force=True`.

## Fallback Mode

When SAM is unavailable or detection fails:
- `fallback_mode="resize"` (default): Simple 512×512 resize
- `fallback_mode="error"`: Raise RuntimeError

## Configuration

```python
from mouse_extensions.inference.preprocessing import MousePreprocessConfig

config = MousePreprocessConfig(
    target_resolution=512,
    target_coverage=0.06,  # Used for coverage normalization
    target_cx=256.0,
    target_cy=256.0,
    bg_color=(255, 255, 255),
    min_scale=0.3,
    max_scale=3.0,
    sam_model_type="vit_h",
    fallback_mode="resize",
)
```
