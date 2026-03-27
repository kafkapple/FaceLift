# H8: Opacity & Anisotropy Distribution Analysis

> **Navigation**: [← Index](../INDEX.md) | [Experiment Registry](../experiments/EXPERIMENT_REGISTRY.md)
> **Status**: Findings confirmed, fix proposed (orientation-aware filter)
> **Date**: 2026-03-26

---

## 1. Hypothesis

**H8a**: Opacity distribution is bimodal → mixture-model-based thresholding possible.
**H8b**: Thin white Gaussian artifacts in novel bottom views are caused by high-anisotropy Gaussians.

## 2. Method

- `opacity_analysis.py`: 20 test frames (M5t2 test set 3240-3599), GS-LRM inference
- Compared 6-view α=0.0 (`base_uniform_v2_6view_v2`) vs 4-view α=0.3 (`base_uniform_v2_4view_alpha03_v3`)
- Tracked opacity at each `apply_all_filters` stage
- Computed scaling anisotropy (max/min ratio) distribution

## 3. Results

### 3.1 Opacity Distribution (H8a: REJECTED — Unimodal)

| Stage | N Gaussians | Mean | Median | Notes |
|-------|:-----------:|:----:|:------:|-------|
| Raw (pre-filter) | 20,971,560 | 0.008 | 0.001 | 98.8% < 0.1 |
| After opacity prune (>0.04) | 330,605 | 0.380 | 0.275 | 98.4% removed |
| After scaling prune (<0.1) | 330,560 | 0.380 | 0.275 | -45 only |
| After floater crop | 330,542 | 0.380 | 0.275 | -18 only |
| **After all filters** | **265,300** | **0.317** | **0.243** | **1.3% survival** |

**Conclusion**: Right-skewed unimodal. Simple threshold sufficient; mixture model unnecessary.

### 3.2 Filter Effectiveness

| Filter | Removed | % of Total Filtering | Verdict |
|--------|:-------:|:-------------------:|---------|
| Opacity prune | ~20.6M | **98.4%** | Dominant |
| Scaling prune | 45 | 0.0002% | Negligible |
| Floater crop | 18 | 0.00009% | Negligible |
| BBox crop | ~65K | 1.6% | Secondary |

### 3.3 Alpha Supervision Effect

| Checkpoint | Raw Mean Opacity | Filtered Mean | Filtered N |
|-----------|:----------------:|:-------------:|:----------:|
| 6v α=0.0 | 0.0076 | 0.317 | 265,300 |
| 4v α=0.3 | **0.0049** | **0.281** | 265,849 |

α=0.3 reduces raw mean opacity by 35% → pushes spurious Gaussians toward 0.

### 3.4 Anisotropy Distribution (H8b: PARTIALLY CONFIRMED)

| Metric | Value |
|--------|:-----:|
| Isotropic (ratio < 3) | **1.6%** |
| Flat (ratio ≥ 30) | **93.8%** |
| Median ratio | **6335** |
| Scale Z distribution | Near-zero (disc/plate shape) |

**Critical finding**: 93.8% flat is **NORMAL** for surface representation.
The problem is NOT anisotropy itself, but **flat Gaussians whose min-scale axis aligns with world Z** → visible edge-on from bottom views.

## 4. Root Cause Analysis (MoA Audit, 3-model consensus)

```
Thin white lines in novel bottom view
  ← Flat Gaussians seen edge-on from below
    ← min-scale axis ≈ world Z (ground normal)
      ← Model represents top-down surfaces as horizontal discs
        ← Training rig has 6 cameras in top-down configuration
```

**NOT caused by**: anisotropy in general, opacity threshold, scaling threshold.

## 5. Recommended Fix: Orientation-Aware Filter

### 5.1 Algorithm

```python
# For each Gaussian:
# 1. Compute rotation matrix from quaternion [w,x,y,z]
# 2. Find min-scale axis direction in world space
# 3. alignment = |dot(thin_axis, [0,0,1])|
# 4. is_artifact = (ratio > 30) AND (alignment > 0.85) AND (opacity < 0.4)
# 5. opacity[is_artifact] *= 0.1  (attenuate, not delete)
```

### 5.2 Why This Works

| Filter Criterion | % Gaussians Hit | Purpose |
|-----------------|:--------------:|---------|
| ratio > 30 only | 93.8% | Too broad |
| + Z-alignment > 0.85 | ~15-25% est. | Targets horizontal discs |
| + opacity < 0.4 | **~5-10% est.** | Excludes real surfaces |

### 5.3 Implementation

- **Module**: `mouse_extensions/model/orientation_filter.py`
- **Integration**: Optional post-filter after `apply_all_filters`
- **Does NOT modify** `gslrm/` core code

## 6. Artifacts

| File | Location |
|------|----------|
| opacity_histogram.png | `outputs/analysis/mouse/opacity/` |
| opacity_by_filter.png | `outputs/analysis/mouse/opacity/` |
| scaling_anisotropy.png | `outputs/analysis/mouse/opacity/` |
| multi_checkpoint_opacity.png | `outputs/analysis/mouse/opacity/` |
| stats_*.json | `outputs/analysis/mouse/opacity/` |

## 7. Related

- [[../experiments/EXPERIMENT_REGISTRY]] — H6 alpha ablation
- [[../experiments/CHECKPOINT_INVENTORY_260326]] — 체크포인트 현황 + P0-P3 우선순위
- `mouse_extensions/scripts/eval/opacity_analysis.py` — analysis script
- `mouse_extensions/model/orientation_filter.py` — filter implementation
- Obsidian: `docs/analysis/PRUNING_ABLATION_DESIGN.md` — Pruning 실험 설계 근거 (SSOT)

---

*H8 Opacity & Anisotropy Analysis v1.0 | 2026-03-26 | MoA+Audit 3-model consensus*
