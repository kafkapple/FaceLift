# Multi-View Object Center Estimation

## Overview

This module addresses a critical bug in D1/D2 preprocessing where per-view 2D center
estimation caused cross-view inconsistency, leading to severe ghosting artifacts.

**Key Insight (2026-01-17)**:
> Per-view 2D center estimation is fundamentally flawed for multi-view 3D reconstruction.
> Each view's centroid/bbox differs due to projection, causing the mouse to appear at
> different pixel positions across views → ray intersection failure → ghosting.

## The Problem

### Previous Approaches (Broken)

| Method | Issue |
|--------|-------|
| **D1** (PP-centered crop) | Per-view PP shift → different shifts per view |
| **D2** (Bbox center) | Per-view bbox center → different crop regions |
| **v12/v13** (Mask centroid) | Per-view 2D centroid → different offsets |

## Correct Approach

1. Estimate a single **3D center** that is consistent across all views
2. **Back-project** this 3D center to each view to get 2D crop centers
3. Apply the same **crop size** to all views

## Implementation

### Module: `mouse_extensions/preprocessing/center_estimation.py`

### Available Methods

| Method | Ray Error | Confidence | Speed | Recommended |
|--------|-----------|------------|-------|-------------|
| **Triangulation** | 2.72 ± 0.80 | 0.76 | Fast | ✅ Yes |
| **Visual Hull** | 2.72 ± 0.80 | 1.00 | Slow | For robust cases |
| **Global Average** | 34.41 ± 6.22 | ~0.00 | Fastest | ❌ No |

### Usage

```python
from mouse_extensions.preprocessing.center_estimation import (
    CenterEstimator, compare_all_methods
)

# Initialize
estimator = CenterEstimator(cameras, method='triangulation')

# Estimate for single frame
result = estimator.estimate(masks)
# result.center_3d: (3,) 3D center in world coordinates
# result.centers_2d: (N_views, 2) projected centers per view
```

## Temporal Analysis Results

**Mouse Movement Statistics** (10 frames, step=200):

| Metric | Mean | Max |
|--------|------|-----|
| Frame-to-frame movement | 48.87 mm | 95.98 mm |
| Equivalent pixels | 99.4 px | 195.2 px |
| Deviation from global center | - | 169 px |

**Conclusion**: Mouse moves significantly → **Per-frame center estimation required**

## pose-splatter Reference

pose-splatter uses the same approach:
1. Pre-computes `center_rotation.npz` with per-frame centers
2. Each frame loads its specific `p_3d = self.centers[idx]`
3. Uses this per-frame 3D center for shape carving

See: `/home/joon/dev/pose-splatter/src/preprocessing/center_estimator.py`

## Validation

### Run Validation Script
```bash
python mouse_extensions/scripts/validate_center_comprehensive.py \
    --data_dir /home/joon/data/markerless_mouse_1_nerf \
    --output_dir mouse_extensions/reports/center_validation \
    --num_samples 10
```

### Output Files
- `metrics.json`: Quantitative results
- `report.html`: Visual report with embedded images
- `frame_*.png`: Per-frame visualizations
- `temporal_variation.png`: 3D center trajectory over time

## Changelog

- **2026-01-17**: Initial implementation
- **2026-01-17**: Validation on 10 samples, triangulation recommended
- **2026-01-17**: Confirmed mouse moves significantly (max 195 px) → per-frame required
