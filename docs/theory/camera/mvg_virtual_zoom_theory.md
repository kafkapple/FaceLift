# MVG Theory: Virtual Zoom for Small Object Enhancement
# Generated: 2026-01-21

## 1. Problem Statement

**Goal**: Make the mouse appear larger in preprocessed images while maintaining
geometric accuracy for 3D reconstruction.

**Constraints**:
- Camera intrinsics must remain geometrically consistent
- No distortion or information loss
- Must work across all 6 views simultaneously

## 2. MVG Background: Image Formation

### 2.1 Pinhole Camera Model

A 3D point X projects to image point x:

```
x = K @ [R | t] @ X

where K = [[fx,  skew, cx],
           [0,   fy,   cy],
           [0,   0,    1 ]]
```

### 2.2 Key Relationship

**Focal length determines FOV**:

```
FOV = 2 * arctan(image_size / (2 * focal_length))
```

For 512x512 image with fx=549: FOV = 50 deg
For 512x512 image with fx=1098: FOV = 26.2 deg (zoomed in 2x)

## 3. Methods for Virtual Zoom

### 3.1 Method A: Crop + Resize (Recommended)

**Principle**: Cropping is equivalent to increasing focal length

**Process**:
1. Crop image from WxH to W'xH' centered at (crop_cx, crop_cy)
2. Resize cropped region to target size (512x512)
3. Update intrinsics:

```python
fx_new = fx * (target_size / crop_size)
fy_new = fy * (target_size / crop_size)
cx_new = (cx - crop_x) * (target_size / crop_size)
cy_new = (cy - crop_y) * (target_size / crop_size)
```

**Geometric accuracy**: PERFECT (no approximation)

**Example** (1.45x zoom):
- Original: 512x512, fx=549, cx=256, cy=256
- Crop: 354x354 centered on mouse at (244, 295)
- After resize to 512x512:
  - fx_new = 549 * (512/354) = 794
  - scale = 512/354 = 1.446

### 3.2 Method B: Virtual Camera Repositioning

**Principle**: Move camera closer while adjusting focal length

**Process**:
1. Scale translation: T_new = T * (1 / zoom_factor)
2. Adjust focal length: fx_new = fx * zoom_factor
3. Image remains unchanged

**Geometric accuracy**:
- PERFECT for planar scenes
- APPROXIMATE for 3D scenes (depth changes)

**Math**:
For point at depth Z, projection is x = fx * X/Z + cx
If we move camera closer by factor k:
- New depth Z' = Z/k
- New projection x' = fx * X/(Z/k) + cx = k * fx * X/Z + cx

This is equivalent to scaling focal length by k.

**Limitation**: Works exactly only if all points are at same depth.
For objects with depth variation (like a mouse), introduces small parallax errors.

### 3.3 Method C: Homographic Zoom (Planar Approximation)

**Principle**: Apply homography that simulates zoom

```
H_zoom = [[s, 0, cx*(1-s)],
          [0, s, cy*(1-s)],
          [0, 0, 1       ]]

where s = zoom_factor
```

**Geometric accuracy**:
- Only exact for planar scenes
- Introduces distortion for 3D objects

**Not recommended** for 3D reconstruction.

## 4. Optimal Strategy for Mouse Data

### 4.1 Recommended: Method A (Crop + Resize)

**Why**:
1. Geometrically exact (no approximation)
2. Works with 3D objects of any depth
3. Simple implementation
4. Consistent with GS-LRM architecture

### 4.2 Implementation Steps

1. **Find global crop region**:
   - For each frame, get mouse bounding box from mask
   - Find the tightest crop that contains mouse in ALL views
   - Add padding (20-30% margin)

2. **Apply crop per view**:
   - Crop region may differ per view (due to 3D geometry)
   - OR use same crop offset for all views (simpler, small loss)

3. **Update intrinsics**:

```python
scale = target_size / crop_size  # e.g., 512/354 = 1.45
fx_new = fx_orig * scale
fy_new = fy_orig * scale
cx_new = (cx_orig - crop_x) * scale
cy_new = (cy_orig - crop_y) * scale
```

4. **Verify GS-LRM compatibility**:
   - GS-LRM expects fx approx 549
   - If fx_new >> 549, need additional normalization
   - Solution: Apply camera distance scaling inversely

### 4.3 Camera Normalization with Zoom

**Problem**: After zoom, fx_new = 794 (not 549)

**Solution**: Keep larger effective focal length
- GS-LRM reads fxfycxcy from data
- NOT hardcoded to 549
- Can handle varying focal lengths

**Verification needed**: Check GS-LRM behavior with fx >> 549

### 4.4 Alternative: Zoom + Distance Compensation

To maintain fx approx 549 while zooming:

```
zoom_factor = 1.45
fx_after_crop = 549 * 1.45 = 796

To normalize back to fx=549:
distance_scale = 549 / 796 = 0.69
T_new = T_orig * 0.69  # Move camera closer
```

**Effect**: Mouse appears larger, fx=549, camera closer
**Warning**: May violate GS-LRM distance assumptions (d=2.7)

## 5. Practical Zoom Limits

### 5.1 Current Statistics

| Metric | Current D7_1 | After 1.45x Zoom |
|--------|--------------|------------------|
| Mouse width | 147 px (29%) | 213 px (42%) |
| Mouse height | 162 px (32%) | 235 px (46%) |
| Error amplification | 3.3x | 2.3x |
| fx | 549 | 796 |
| FOV | 50 deg | 35 deg |

### 5.2 Maximum Practical Zoom

- Largest mouse: 222x295 pixels
- With 20% padding: 266x354 -> crop size 354
- Maximum zoom: 512/354 = **1.45x**

Higher zoom risks cropping the mouse in some frames.

### 5.3 Recommended Zoom: 1.3x (Conservative)

- Crop size: 394x394
- fx_new = 549 * 1.3 = 714
- FOV: 39.5 deg
- Mouse size: 38-41% of image
- Error amplification: 2.5x

## 6. Implementation Plan (D9)

```python
def preprocess_D9_zoom(image, mask, K, R, T, zoom=1.3, target_size=512):
    """
    D9 preprocessing: D8 precision + virtual zoom

    Steps:
    1. Find mouse center from mask
    2. Calculate crop region
    3. Crop image
    4. Resize to target_size
    5. Update intrinsics
    """
    import numpy as np
    import cv2

    # 1. Find mouse center
    ys, xs = np.where(mask > 127)
    mouse_cx = (xs.min() + xs.max()) / 2
    mouse_cy = (ys.min() + ys.max()) / 2

    # 2. Calculate crop size
    crop_size = int(target_size / zoom)
    crop_x = int(mouse_cx - crop_size / 2)
    crop_y = int(mouse_cy - crop_size / 2)

    # Clamp to valid range
    crop_x = max(0, min(crop_x, image.shape[1] - crop_size))
    crop_y = max(0, min(crop_y, image.shape[0] - crop_size))

    # 3. Crop
    cropped = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

    # 4. Resize
    resized = cv2.resize(cropped, (target_size, target_size))

    # 5. Update intrinsics
    scale = target_size / crop_size
    fx_new = K[0,0] * scale
    fy_new = K[1,1] * scale
    cx_new = (K[0,2] - crop_x) * scale
    cy_new = (K[1,2] - crop_y) * scale

    K_new = np.array([
        [fx_new, K[0,1]*scale, cx_new],
        [0,      fy_new,       cy_new],
        [0,      0,            1     ]
    ])

    return resized, K_new, R, T  # R, T unchanged!
```

## 7. Summary

| Method | Geometric Accuracy | Implementation | Recommendation |
|--------|-------------------|----------------|----------------|
| Crop + Resize | Exact | Simple | RECOMMENDED |
| Virtual Reposition | Approximate | Medium | For planar only |
| Homographic Zoom | Distorted | Complex | Not recommended |

**Final Recommendation**:
- D8: Precision preprocessing (skew correction, exact values)
- D9: D8 + Virtual zoom (1.3x via crop+resize)

---
*MVG Theory Reference: Hartley & Zisserman, "Multiple View Geometry in Computer Vision"*
