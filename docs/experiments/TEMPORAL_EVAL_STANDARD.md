# Temporal Evaluation Standard

All temporal consistency experiments MUST follow this standard for reproducibility and fair comparison.

---

## 1. Frame Convention

### Sparse Sampling (per-frame quality + static novel view comparison)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frame IDs** | `3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3599` | 10 frames from test split, ~40-frame spacing |
| **Split** | Test (3240-3599) | Never seen during training |
| **Count** | 10 | Sufficient for statistical metrics |

### Dense Sequence (temporal consistency analysis)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frame Range** | `3300-3320` | 21 consecutive frames within test split |
| **Spacing** | 1 frame (no skip) | Required for temporal metrics (tOF, TLPIPS) |
| **Purpose** | Temporal flickering, smoothing comparison | Consecutive frames reveal jitter |

### Why These Frames?

- **Test split only** (3240-3599): ensures evaluation integrity
- **3300-3320 dense region**: overlaps with sparse samples at 3300/3320 for cross-validation
- **Previous renders**: baseline/alpha03 used 3300-3580 (sparse), alpha05/alpha10 used 0-2000 (training set) — **non-comparable, deprecated**

## 2. Novel Views

| View | Elevation | Azimuth | Purpose |
|------|-----------|---------|---------|
| `bottom` | -70° | 0° | Ventral surface, alpha artifact sensitive |
| `top` | +70° | 0° | Dorsal surface, body shape |
| `front_low` | -30° | 0° | Face/head detail |
| `side_low` | -30° | 90° | Profile, tail/limb articulation |

**Resolution**: 512×512 (matches GS-LRM training resolution)
**Render radius**: 2.7 (FaceLift pretrained standard)

## 3. Experiment Settings

| ID | Label | Views | α weight | Checkpoint |
|----|-------|:-----:|:--------:|------------|
| `baseline_6v` | Baseline (α=0) | 6 | 0.0 | `uniform_v2/6view/best_psnr.pt` |
| `alpha03` | α=0.3 | 4 | 0.3 | `uniform_v2/4view_alpha03_v3/best_psnr.pt` |
| `alpha05` | α=0.5 | 4 | 0.5 | `uniform_v2/4view_alpha05_v3/best_psnr.pt` |
| `alpha10` | α=1.0 | 4 | 1.0 | `uniform_v2/4view_alpha10_v3/best_psnr.pt` |

**Checkpoint root**: `/node_data/joon/checkpoints/FaceLift/gslrm/`

## 4. Temporal Smoothing Methods

| Method | Type | Parameters | Implementation |
|--------|------|------------|----------------|
| **Original** | None | — | Raw GS-LRM output |
| **EMA** | 2D post-process | α ∈ {0.1, 0.3, 0.5, 0.7} | `evaluation/temporal_smoothing.py` |
| **DeformV2** | 3D Gaussian | blend ∈ {0.3, 0.5, 0.7} | `model/deformation/temporal_deform_inference.py` |
| **NN-Match** | 3D Gaussian | k=1, interp=lerp/slerp | `scripts/eval/temporal_comparison.py` |
| **OptFlow** | 2D post-process | Farneback, blend α | `evaluation/temporal_smoothing.py` |

## 5. Metrics

### Metric Protocol (IMPORTANT)

> **Object-centric 평가 표준: Masked Foreground 메트릭 사용.**
> Full-image (white BG) PSNR은 배경 95%가 완벽 → 부풀려진 수치. 논문에서는 참고용으로만 병기.
> Reference: 3DGS (Kerbl 2023), GS-LRM (Zhang 2024), One-2-3-45 모두 masked foreground 채택.

### Per-Frame Quality (higher is better unless noted)

| Metric | Scope | Formula | Tool |
|--------|-------|---------|------|
| **PSNR_gt_masked** | **FG only** (primary) | PSNR on GT mask pixels | `eval/fair_comparison.py` |
| **SSIM_gt_masked** | **FG bbox** (primary) | SSIM on GT mask bbox+10px | `eval/fair_comparison.py` |
| **LPIPS** | FG region | AlexNet perceptual | `evaluation/metrics.py` |
| **IoU** | Silhouette | mask intersection/union | `eval/fair_comparison.py` |
| *PSNR_full* | *Full image (참고)* | *20·log₁₀(1/√MSE)* | *evaluation/metrics.py* |

### Temporal Stability (lower is better)

| Metric | Formula | Tool |
|--------|---------|------|
| **tOF** | std(‖OF(t,t+1)‖) over sequence | `evaluation/temporal_smoothing.py` |
| **TLPIPS** | mean(LPIPS(t, t+1)) over sequence | `evaluation/temporal_smoothing.py` |
| **FF-SSIM-var** | var(SSIM(t, t+1)) over sequence | `evaluation/temporal_smoothing.py` |
| **Flicker Rate** | frames with ΔI > threshold / total | `evaluation/temporal_smoothing.py` |

### Gaussian-Level (from dense sequence, lower is better)

| Metric | Formula | Tool |
|--------|---------|------|
| **Position Jitter** | mean(‖Δxyz_matched‖) | `evaluation/temporal_metrics.py` |
| **Scale Variance** | var(scale_matched) over time | `evaluation/temporal_metrics.py` |
| **Persistence Ratio** | matched_N / total_N (higher=better) | `scripts/eval/temporal_comparison.py` |

## 6. Output Directories

```
outputs/datasets/temporal_eval/
├── {experiment_id}/
│   ├── sparse/                    # 10 sparse frames
│   │   └── {view}/{frame_id}.png
│   ├── dense/                     # 21 consecutive frames (3300-3320)
│   │   └── {view}/{frame_id}.png
│   └── gaussians/                 # Raw Gaussian NPZ (dense only)
│       └── {frame_id}.npz
├── smoothed/
│   ├── ema_{alpha}/               # EMA results
│   ├── deform_{blend}/            # Deformation results
│   ├── nn_match/                  # NN matching results
│   └── optflow_{alpha}/           # Optical flow results
└── metrics/
    ├── per_frame_quality.json     # PSNR/SSIM/LPIPS per frame
    ├── temporal_stability.json    # tOF/TLPIPS/FF-SSIM-var
    └── comparison_table.md        # Summary table
```

## 7. Grid Visualization Convention

### Per-View Comparison
- **Layout**: 1 row × N columns (one per experiment/method)
- **Labels**: Top of each column, font ≥20pt
- **Frame ID**: Bottom-right corner overlay

### Combined Grid
- **Rows**: Views (bottom, top, front_low, side_low)
- **Columns**: Experiments or methods
- **Row labels**: Left side
- **Column labels**: Top

### Temporal Strip
- **Rows**: Methods (Original, EMA, Deform, etc.)
- **Columns**: Consecutive frames (3300, 3301, ..., 3310)
- **Purpose**: Visual temporal smoothness comparison

---

## Quick Reference

```python
# Standard frames
SPARSE_FRAMES = [3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3599]
DENSE_RANGE = range(3300, 3321)  # 3300-3320 inclusive
VIEWS = ["bottom", "top", "front_low", "side_low"]
EXPERIMENTS = ["baseline_6v", "alpha03", "alpha05", "alpha10"]
RESOLUTION = 512
RENDER_RADIUS = 2.7
```

---

*Created: 2026-03-21 | SSOT for temporal evaluation*
