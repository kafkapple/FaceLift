# FaceLift Mouse Preprocessing Registry

> Last Updated: 2026-01-24 | Version: 3.1

---

## Table of Contents
1. [Overview](#overview)
2. [Version Summary](#version-summary)
3. [Paradigm Details](#paradigm-details)
4. [Recommended Workflows](#recommended-workflows)
5. [Technical Reference](#technical-reference)

---

## Overview

### Evolution Path
```
D1-D4 (Deprecated)     D6 (Geometry)     D7 (PP-Shift)     D8 (Precision)     D9 (Native)     D10 (Proposed)
     │                      │                  │                 │                 │                │
     ▼                      ▼                  ▼                 ▼                 ▼                ▼
Per-view center     Accurate PP        PP=256 shift      Homography        Full-res        Up-aligned
+ PP=256 forced     Variable cx,cy     fx=549 norm       + skew            1152x1024       + Adaptive zoom
     │                      │                  │                 │                 │                │
     ▼                      ▼                  ▼                 ▼                 ▼                ▼
   ❌ 5-13° ray err   ✅ 0° ray err    ✅ ~0° ray err    ✅ ~0° ray err    ✅ 0° ray err    ✅ 0° ray err
```

### Key Metrics Comparison

| Dataset | Ray Error | Pretrained Compat | Mouse Size | Memory | Status |
|---------|-----------|-------------------|------------|--------|--------|
| D1-D4 | 5-13° | ✅ | Medium | 1x | Deprecated |
| D6-3 | 0° | ⚠️ Variable PP | Medium | 1x | Active |
| **D7.1** | ~0° | ✅ fx=549, PP=256 | Medium | 1x | **Recommended** |
| **D8** | ~0° | ✅ fx=549, PP=256 | Medium | 1x | **Precision** |
| D8.1 | ~0° | ⚠️ PP varies | Large (1.3x) | 1x | Active |
| D9_norm | 0° | ⚠️ fx=1632 | Large | 4.5x | Experimental |
| **D10** | ~0° | ✅ fx=549, PP=256 | Medium | 1x | **Proposed** |
| D10.1 | ~0° | ✅ fx=549, PP=256 | Large (adaptive) | 1x | Proposed |

---

## Version Summary

### Active Versions

| Version | Paradigm | Transform | PP Method | Key Feature |
|---------|----------|-----------|-----------|-------------|
| **D7.1** | pp_centered_shift | Affine (individual) | shift_to_256 | Stable baseline |
| D7.2 | pp_centered_shift | Affine (average) | shift_to_256 | Isotropic scaling |
| **D8** | precision_homography | Homography | shift_to_256 | **+Skew correction** |
| D8.1 | precision_homography | Homography | shift_to_256 | +1.3x zoom |
| D9 | native | None | original | Full resolution |
| D9_norm | native | None | original | +Trans normalized |
| **D10** | up_aligned_zoom | Homography | shift_to_256 | **+Up alignment** |
| D10.1 | up_aligned_zoom | Homography | shift_to_256 | +Adaptive zoom |
| D10.2 | up_aligned_zoom | Homography | shift_to_256 | +Camera Y fallback |

### Deprecated Versions

| Version | Reason | Alternative |
|---------|--------|-------------|
| D1 | Per-view center → cross-view inconsistency | D7.1 |
| D4 | PP=256 forced → 5-13° ray error | D8 |
| D7 | fy=549 forced → slight distortion | D7.1 |

---

## Paradigm Details

### 1. PP-Centered Shift (D7.x)

**Principle**: Shift image so that Principal Point lands at center (256, 256)

```
Original Image          After PP-Shift
┌─────────────────┐    ┌─────────────────┐
│        ⊙        │    │                 │
│   (cx=380,      │ →  │        ⊙        │
│    cy=290)      │    │   (cx=256,      │
│                 │    │    cy=256)      │
└─────────────────┘    └─────────────────┘
```

**Intrinsics Update**:
```python
shift_x = 256 - cx_original
shift_y = 256 - cy_original
# Image shifted by (shift_x, shift_y)
# cx_new = 256, cy_new = 256
# fx, fy unchanged (but normalized to 549)
```

### 2. Precision Homography (D8.x)

**Principle**: D7.1 + Skew correction via Homography matrix

```
Affine (D7.1):           Homography (D8):
[sx  0  tx]              [h11 h12 h13]
[ 0 sy  ty]              [h21 h22 h23]
[ 0  0   1]              [h31 h32   1]

6 DoF                    8 DoF (includes skew)
```

**When Skew Matters**:
- Camera not perfectly aligned
- Lens distortion residuals
- Object at edge of frame

### 3. Native Resolution (D9.x)

**Principle**: No image transformation, preserve original geometry 100%

```
Original: 1152 × 1024
↓
No transform (D9) or resize (D9_resized)
↓
fx = 1632 (original) → needs trans_norm = fx/203 ≈ 8.0
```

**Trade-offs**:
| Aspect | D9/D9_norm | D9_resized |
|--------|------------|------------|
| Geometry | 100% accurate | ~0° error |
| Memory | 4.5x | 1x |
| Detail | Maximum | Some loss |
| Pretrained | ⚠️ Different distribution | ✅ Matches |

### 4. Up-Aligned + Zoom (D10.x) [PROPOSED]

**Principle**: D8 + World coordinate alignment + Optional adaptive zoom

```
vertical_lines.npz
up = [0.91, -0.41, 0.05]
        │
        ▼
R_align = rotation_matrix([0,0,1] → -up)
        │
        ▼
Cameras aligned to gravity
        │
        ▼
Optional: Adaptive zoom based on bbox
```

**Benefits**:
1. **Turntable consistency**: Mouse always "stands up" correctly
2. **Cross-dataset compatibility**: Aligned coordinate system
3. **Larger mouse**: Adaptive zoom maximizes frame usage

---

## Recommended Workflows

### Production (Stable)
```bash
# Dataset: D7.1 or D8
# Experiment: E2_gt_alpha
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha
```

### Precision Experiments
```bash
# Dataset: D8 (homography + skew)
# For comparison against D7.1
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha
```

### High-Resolution (Experimental)
```bash
# Dataset: D9_norm (1152x1024)
# Requires A6000+ GPU
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D9_norm -e E2_gt_alpha_native
```

### Proposed: Up-Aligned (D10)
```bash
# Step 1: Generate D10 dataset (up-alignment)
cd /home/joon/dev/FaceLift
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D10

# Step 2: Generate D10.1 dataset (up-alignment + adaptive zoom)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D10_1

# Step 3: Run experiments
CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D10 -e E2_gt_alpha

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D10_1 -e E2_gt_alpha
```

### View Ablation Experiments
```bash
# 3-view input (robustness test)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha_3v

# 5-view input (maximum quality)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha_5v
```

---

## Technical Reference

### Camera Parameters

| Dataset | fx | fy | cx | cy | trans_norm |
|---------|----|----|----|----|------------|
| D7.1 | 549.0 | 549.0 | 256 | 256 | 2.7 |
| D8 | 548.99 | 548.99 | 256 | 256 | 2.7 |
| D8.1 | 713.69 | 713.69 | 256 | 256 | 2.7 |
| D9 | 1632 | 1632 | 576 | 512 | 246 (raw) |
| D9_norm | 1632 | 1632 | 576 | 512 | 8.0 |

### Ray Direction Formula

```python
# Correct ray direction calculation
d_cam = np.array([
    (u - cx) / fx,
    (v - cy) / fy,
    1.0
])
d_world = R.T @ d_cam
d_world = d_world / np.linalg.norm(d_world)
```

### PP Error Impact

```
PP_error = |cx_actual - cx_used|
ray_angle_error = arctan(PP_error / fx)

Example:
  PP_error = 37 px, fx = 549
  ray_angle_error = arctan(37/549) ≈ 3.9°
```

### Up Direction (vertical_lines.npz)

```python
# Load up direction
up_raw = np.load("vertical_lines.npz")["up"]
# [0.91152574, -0.40814659, 0.05037057]

# Apply for auto_orient
up = -up_raw  # Sign flip for coordinate convention
R_align = rotation_matrix_between([0, 0, 1], up)
```

---

## Data Locations

| Dataset | Path | Status |
|---------|------|--------|
| D7_1 | `/home/joon/data/preprocessed/FaceLift_mouse/D7_1/` | ✅ Ready |
| D8 | `/home/joon/data/preprocessed/FaceLift_mouse/D8/` | ✅ Ready |
| D8_1 | `/home/joon/data/preprocessed/FaceLift_mouse/D8_1/` | ✅ Ready |
| D9 | `/home/joon/data/preprocessed/FaceLift_mouse/D9/` | ✅ Ready |
| D9_norm | `/home/joon/data/preprocessed/FaceLift_mouse/D9_norm/` | ✅ Ready |
| D10 | `/home/joon/data/preprocessed/FaceLift_mouse/D10/` | ⚠️ Needs preprocessing |
| D10_1 | `/home/joon/data/preprocessed/FaceLift_mouse/D10_1/` | ⚠️ Needs preprocessing |

---

## Code Locations

| Component | Path |
|-----------|------|
| Presets | `mouse_extensions/preprocessing/presets.py` |
| Preprocessor | `mouse_extensions/preprocessing/preprocess.py` |
| Center Estimation | `mouse_extensions/preprocessing/center_estimation.py` |
| Camera Normalizer | `mouse_extensions/preprocessing/camera_normalizer.py` |
| Dataset Configs | `configs/datasets/` |
| Experiment Configs | `configs/experiments/` |

---

*FaceLift Mouse | Reference Documentation*
