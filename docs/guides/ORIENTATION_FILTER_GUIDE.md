# Orientation-Aware Gaussian Filter Guide

> **Navigation**: [← Visualization Guide](VISUALIZATION_GUIDE.md) | [← H8 Analysis](../hypotheses/H8_opacity_anisotropy_analysis.md)
> **Module**: `mouse_extensions/model/orientation_filter.py`
> **Status**: Tested, ready for integration

---

## 1. Problem

Novel bottom views show "thin white line" artifacts from flat Gaussians seen edge-on.

- 93.8% of Gaussians are flat (max/min scaling ratio ≥ 30) — **this is normal**
- The problem subset: flat Gaussians whose **thinnest axis lies in the XY plane** (z_alignment < 0.15)
- These create lines when viewed from below (camera looking up along +Z)

## 2. Filter Algorithm

```python
from mouse_extensions.model.orientation_filter import suppress_z_aligned_flat

# After standard filtering
gaussians.apply_all_filters(opacity_thres=0.04, ...)

# Apply orientation filter (optional, safe)
stats = suppress_z_aligned_flat(
    gaussians,
    ratio_thresh=30.0,     # Min anisotropy ratio to consider
    z_align_thresh=0.15,   # Max |dot(thin_axis, Z)| — lower = more XY-aligned
    opacity_ceil=0.4,      # Protect high-opacity surfaces
    attenuation=0.0,       # 0.0 = full removal, 0.1 = 90% opacity reduction
    mode="prune",          # "prune" or "attenuate"
)
print(f"Suppressed {stats['n_suppressed']} / {stats['n_total']}")
```

### How It Works

```
For each Gaussian:
  1. Check anisotropy: max_scale / min_scale > ratio_thresh?
  2. Convert quaternion [w,x,y,z] → rotation matrix R (3×3)
  3. Find thinnest axis direction in world space: R[:, argmin(scale)]
  4. Compute z_alignment = |dot(thin_axis, [0,0,1])|
     - Low z_alignment → thin axis in XY plane → edge-on from bottom = LINE
     - High z_alignment → thin axis in Z → face-on from bottom = circle (OK)
  5. Only suppress if: flat AND low_z_align AND opacity < ceiling
```

### Critical: Z-Alignment Direction

| z_alignment | Meaning | Bottom View | Filter? |
|:-----------:|---------|:-----------:|:-------:|
| **< 0.15** | Thin axis ≈ XY plane | **Edge-on = LINE** | ✅ Yes |
| 0.15 - 0.85 | Diagonal orientation | Moderate | No |
| **> 0.85** | Thin axis ≈ Z | Face-on = circle | No |

## 3. Parameters

| Parameter | Default | Range | Effect |
|-----------|:-------:|:-----:|--------|
| `ratio_thresh` | 30.0 | 10-100 | Lower = more aggressive (catches less-flat Gaussians) |
| `z_align_thresh` | 0.15 | 0.05-0.30 | Higher = more aggressive (wider cone around XY) |
| `opacity_ceil` | 0.4 | 0.1-1.0 | Higher = more aggressive (also suppresses bright Gaussians) |
| `attenuation` | 0.0 | 0.0-0.5 | 0.0 = delete, 0.1 = reduce to 10% opacity |

### Tuning Guide

- **Start conservative**: ratio=30, z_align=0.15, opacity_ceil=0.4
- **If artifacts persist**: raise z_align_thresh to 0.20-0.25
- **If surfaces degrade**: lower opacity_ceil to 0.2 or raise ratio_thresh to 50

## 4. Test Results (2026-03-26)

| Setting | Avg Suppressed | % of Total | Effect |
|---------|:-------------:|:----------:|--------|
| v1 (z_align > 0.85, **BUG**) | 125 | 0.9% | No visible change |
| **v2 (z_align < 0.15, corrected)** | **1,974** | **15.6%** | Visible artifact reduction |

## 5. Diagnostic Scripts

```bash
# Test filter with before/after comparison images
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.test_orientation_filter \
    --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --n-frames 5 --z-align-thresh 0.15 --attenuation 0.0

# Multi-view grid video comparison (4 angles × before/after)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.filter_grid_comparison \
    --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --frame-range 3240:3280 --z-align-thresh 0.15 --attenuation 0.0
```

## 6. Integration Points

| Script | Integration | Status |
|--------|-------------|:------:|
| `gslrm_pipeline.py` | Add after `apply_all_filters()` | Planned |
| `cinematic_sequence.py` | Add before rendering | Planned |
| `turntable_renderer.py` | Add to render loop | Planned |
| `ablation_comparison.py` | Optional post-filter | Planned |

## Related

- ↑ [[VISUALIZATION_GUIDE]] — Parent guide
- ↔ [[../hypotheses/H8_opacity_anisotropy_analysis]] — Analysis report
- ↔ `mouse_extensions/scripts/eval/opacity_analysis.py` — Distribution analysis

---

*Orientation Filter Guide v1.0 | 2026-03-26*
