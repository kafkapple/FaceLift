# Intrinsics Scaling for Multi-Resolution Rendering

## 1. Problem Statement

When rendering at a different resolution than the original image, camera intrinsics must be scaled accordingly. Failure to do so causes:
- **Incorrect FOV** (Field of View)
- **Object clipping** or appearing too small/large
- **Misaligned projections**

---

## 2. Camera Intrinsics Model

### Pinhole Camera Model

The projection from 3D world coordinates $(X, Y, Z)$ to 2D image coordinates $(u, v)$ is:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}
$$

Where:
- $f_x, f_y$: Focal lengths in pixels
- $c_x, c_y$: Principal point (optical center) in pixels
- $(u, v)$: Image coordinates in pixels

### Key Insight

**All intrinsic parameters are in pixel units**, tied to the original image resolution.

---

## 3. Resolution Scaling Formula

When scaling from resolution $W_{orig} \times H_{orig}$ to $W_{new} \times H_{new}$:

$$
s = \frac{W_{new}}{W_{orig}} = \frac{H_{new}}{H_{orig}} \quad \text{(assuming square scaling)}
$$

The scaled intrinsics become:

$$
\begin{aligned}
f_x' &= s \cdot f_x \\
f_y' &= s \cdot f_y \\
c_x' &= s \cdot c_x \\
c_y' &= s \cdot c_y
\end{aligned}
$$

### Matrix Form

$$
K' = \begin{bmatrix} s & 0 & 0 \\ 0 & s & 0 \\ 0 & 0 & 1 \end{bmatrix} K = \begin{bmatrix} s \cdot f_x & 0 & s \cdot c_x \\ 0 & s \cdot f_y & s \cdot c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

---

## 4. Bug Analysis

### Affected Function

`get_turntable_with_dataset_views()` in `gaussians_renderer.py`

### Bug Description

```python
# BEFORE (BUG):
# dataset_fxfycxcy: original resolution (e.g., 512)
# turntable_fxfycxcy: rendering resolution (e.g., 384)
fxfycxcy = np.concatenate([dataset_fxfycxcy, turntable_fxfycxcy], axis=0)
# ↑ Scale mismatch! Dataset views render with wrong FOV
```

### Visual Symptoms

| Symptom | Cause |
|---------|-------|
| Object clipped/cut off | FOV too narrow (intrinsics too large for rendering resolution) |
| Object too small | FOV too wide (intrinsics too small) |
| White background only | Object completely outside view frustum |

### Numerical Example

| Parameter | Original (512) | Rendering (384) | Scale Factor |
|-----------|---------------|-----------------|--------------|
| Resolution | 512 | 384 | 0.75 |
| $f_x$ | 549 | 411.75 | 0.75 |
| $f_y$ | 549 | 411.75 | 0.75 |
| $c_x$ | 256 | 192 | 0.75 |
| $c_y$ | 256 | 192 | 0.75 |

---

## 5. Fix Implementation

### Code Change

```python
def get_turntable_with_dataset_views(
    dataset_c2ws: np.ndarray,
    dataset_fxfycxcy: np.ndarray,
    ...
    original_resolution: int = None,  # NEW PARAMETER
):
    # Scale dataset intrinsics if rendering at different resolution
    scaled_dataset_fxfycxcy = dataset_fxfycxcy.copy()
    if original_resolution is not None and original_resolution != w:
        scale = w / original_resolution
        scaled_dataset_fxfycxcy = scaled_dataset_fxfycxcy * scale
    
    # Now combine with matching scales
    fxfycxcy = np.concatenate([scaled_dataset_fxfycxcy, turntable_fxfycxcy], axis=0)
```

---

## 6. Related Functions

| Function | Scaling Status |
|----------|---------------|
| `render_dataset_views()` | ✅ Has `original_resolution` param |
| `get_turntable_with_dataset_views()` | ❌ **Fixed in this commit** |
| `render_turntable()` | ✅ Generates own intrinsics |

---

## 7. References

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Multiple View Geometry in Computer Vision, Hartley & Zisserman](https://www.robots.ox.ac.uk/~vgg/hzbook/)

---

*Created: 2026-01-26 | FaceLift Mouse Project*
