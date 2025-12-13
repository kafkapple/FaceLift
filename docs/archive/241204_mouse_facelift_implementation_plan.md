# Mouse-FaceLift Implementation Plan

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

Adapt the FaceLift model (originally for human synthetic single-view 3D reconstruction) to perform single-view to multi-view 3D reconstruction for mice using a custom 6-view video dataset.

## Reference Materials

- **Original Paper**: [FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads](https://arxiv.org/html/2412.17812v2)
- **Original Repo**: `/home/joon/dev/FaceLift`
- **Reference Code (6-view)**: `/home/joon/MAMMAL_mouse`

## Data Paths (Server: ssh gpu05)

| Type | Path |
|------|------|
| Base Code | `/home/joon/FaceLift` |
| Reference Code | `/home/joon/MAMMAL_mouse` |
| Raw Video Data | `/home/joon/data/markerless_mouse` (2 mice, 6 views each) |
| Meta Data (Masks/Keypoints) | `/home/joon/data/markerless_mouse_1_nerf` |

## Architecture Analysis

### FaceLift Pipeline
1. **MVDiffusion**: Single image → 6 multi-view images (512x512 RGBA)
2. **GS-LRM**: 6 views → 3D Gaussian Splats representation

### Key Data Format (FaceLift)

**Camera JSON Structure** (`opencv_cameras.json`):
```json
{
  "id": "sample_000",
  "frames": [
    {
      "w": 512, "h": 512,
      "fx": 548.99, "fy": 548.99,
      "cx": 256.0, "cy": 256.0,
      "w2c": [[4x4 matrix]],
      "file_path": "images/cam_000.png"
    }
  ]
}
```

**Image Format**:
- Size: 512x512
- Format: RGBA (with alpha channel for background removal)
- Background: White/Black/Gray (configurable)

### MAMMAL_mouse Multi-view Processing

**Camera Format** (`new_cam.pkl`):
```python
{
    'R': (3, 3),  # Rotation matrix
    'T': (3, 1),  # Translation vector
    'K': (3, 3)   # Intrinsic matrix
}
```

**View Synchronization**: Frame-indexed access across 6 VideoCapture objects.

## Implementation Steps

### Step 1: Data Pipeline (process_mouse_data.py)

**Tasks**:
1. Parse 6-view videos from raw data path
2. Extract synchronized frames (every Nth frame for ~2,000-5,000 samples)
3. Apply masks from meta data path
4. Convert to FaceLift format (JSON + images)

**Camera Coordinate Conversion**:
```python
# MAMMAL format: K @ (R @ X + T)
# FaceLift format: w2c = [R|T] (4x4 matrix)

def convert_mammal_to_facelift(R, T, K):
    """Convert MAMMAL camera params to FaceLift w2c format"""
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    return {
        "w2c": w2c.tolist(),
        "fx": fx, "fy": fy,
        "cx": cx, "cy": cy
    }
```

### Step 2: Model Configuration (mouse_config.yaml)

**Key Modifications**:
- Input views: 1 (single image input)
- Target views: 5 (reconstruction supervision)
- Total views: 6
- Background: Use mask from meta data
- No face detection/cropping (not applicable to mice)

### Step 3: Training Strategy

**Option A: Finetune from FaceLift**
- Load pretrained weights
- Freeze early layers (image tokenizer)
- Train deformation layers with higher LR

**Option B: Train from Scratch**
- Use GS-LRM pretrained on Objaverse (generic 3D objects)
- Train entire network on mouse data

**Recommendation**: Start with Option B (Objaverse pretrain) due to large domain gap between human faces and mice.

### Step 4: Validation Pipeline

**Overfitting Test** (Critical):
1. Use 10 samples only
2. Train until reconstruction loss is near-zero
3. Verify input image is perfectly reconstructed
4. This validates code correctness before full training

## File Structure

```
FaceLift/
├── scripts/
│   └── process_mouse_data.py      # Data preprocessing
├── gslrm/
│   └── data/
│       └── mouse_dataset.py       # Mouse-specific dataset
├── configs/
│   └── mouse_config.yaml          # Training config
├── train_mouse.py                  # Training script
├── inference_mouse.py              # Inference demo
└── docs/
    └── reports/
        └── 241204_mouse_facelift_implementation_plan.md
```

## Expected Challenges

1. **Camera Coordinate System**: Converting between MAMMAL (OpenCV) and FaceLift coordinate conventions
2. **Domain Gap**: Human face priors may not help for mouse reconstruction
3. **Data Scale**: Real video dataset is much smaller than synthetic training data
4. **Texture Complexity**: Mouse fur texture vs synthetic face texture

## Validation Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Visual Inspection**: Novel view synthesis quality

## Next Steps

1. [ ] Set up conda environment on gpu05
2. [ ] Run data preprocessing script
3. [ ] Verify camera conversion with visualization
4. [ ] Run overfitting test on 10 samples
5. [ ] Full training run
6. [ ] Evaluate on held-out test frames
