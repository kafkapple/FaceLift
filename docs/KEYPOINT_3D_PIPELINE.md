# 3D Keypoint Multi-View Analysis Pipeline

> Multi-view triangulation analysis + FL vs PS comparison framework.
> **Created**: 2026-03-04 | **Version**: v1.0

---

## Overview

This pipeline provides tools for:
1. **Triangulation Analysis** — DANNCE 2D detections → DLT → 3D → MPJPE vs MAMMAL GT
2. **Multi-View Triangulation** — GS-LRM novel views → synthetic 2D → triangulate → accuracy
3. **FL vs PS Comparison** — GT 6-view grid visualization with keypoint overlay

## Architecture

```
mouse_extensions/
├── analysis/
│   ├── __init__.py
│   └── triangulation_analysis.py    # Core module (Phase 1)
├── scripts/
│   ├── multiview_triangulation_eval.py  # GS-LRM novel view eval (Phase 2)
│   └── render_comparison_grid.py        # FL vs PS comparison (Phase 3)
```

---

## Section 9: Triangulation Analysis (DANNCE 2D → DLT → MPJPE)

### Why

DANNCE provides 2D keypoint detections per camera view. To evaluate 3D accuracy,
we triangulate these 2D detections using DLT and compare against MAMMAL 3D GT.
This establishes a baseline for how many views are needed for accurate 3D reconstruction.

### How

**DLT (Direct Linear Transform)**: For each keypoint, construct the linear system
`A @ X = 0` from projection matrices and 2D points, solve via SVD.

- Confidence-weighted: Higher confidence detections contribute more
- Per-joint: 22 keypoints triangulated independently
- Batch processing: Efficient over multiple frames

### Module: `triangulation_analysis.py`

| Function | Purpose |
|----------|---------|
| `load_dannce_2d(data_dir)` | Load (6, 18000, 22, 3) DANNCE detections |
| `load_mammal_3d(npz_path)` | Load (3600, 22, 3) MAMMAL GT |
| `load_raw_cameras(cam_pkl)` | Load 6 camera params (K, R, t) |
| `build_projection_matrix(K, R, t)` | P = K @ [R \| t] (3×4) |
| `triangulate_dlt(pts_2d, P, conf)` | DLT for single point |
| `triangulate_batch(pts_2d, P)` | DLT for all 22 joints |
| `compute_mpjpe(pred, gt)` | Mean Per-Joint Position Error |
| `compute_pa_mpjpe(pred, gt)` | Procrustes-Aligned MPJPE |
| `run_gt_view_experiment(...)` | Full experiment: N views → MPJPE |

### Usage

```bash
python -m mouse_extensions.analysis.triangulation_analysis \
    --data-dir ~/data/raw/markerless_mouse_1_nerf \
    --cam-pkl ~/data/raw/markerless_mouse_1_nerf/new_cam.pkl \
    --mammal-3d /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
    --output-dir outputs/triangulation_analysis \
    --view-counts 2 3 4 5 6
```

### Expected Results

- 6-view: MPJPE ~few mm (real DANNCE noise)
- Monotonic decrease: More views → lower MPJPE
- View selection: Uniform spacing ≥ confidence-based for well-calibrated cameras

---

## Section 10: Multi-View Triangulation (GS-LRM Novel Views)

### Why

GS-LRM can generate novel views from any camera pose. By rendering N views
(more than the original 6), we can potentially achieve better triangulation accuracy.
This section quantifies the tradeoff between view count and noise.

### How

1. GS-LRM `predict()` → 3D Gaussians (one forward pass)
2. `get_turntable_cameras(num_views=N)` → N uniformly spaced cameras
3. Project MAMMAL 3D GT → 2D on each camera
4. Add Gaussian noise σ = {0, 1, 2, 5} px to simulate detection error
5. Triangulate → compare with GT

### Key Insight

By comparing DANNCE's real 6-view MPJPE (Section 9) with synthetic N-view results
at various noise levels, we can estimate DANNCE's effective detection noise σ.

### Usage

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.multiview_triangulation_eval \
    --config configs/mouse/base_uniform_v2.yaml \
    --checkpoint checkpoints/gslrm/6view_v2/best_psnr.pt \
    --data-dir ~/data/preprocessed/FaceLift_mouse/M5t2 \
    --mammal-3d /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
    --output-dir outputs/triangulation_eval \
    --view-counts 6 9 12 24 \
    --noise-levels 0 1 2 5 \
    --save-views
```

### Outputs

| File | Description |
|------|-------------|
| `multiview_results.json` | view_count × noise_level → MPJPE table |
| `accuracy_vs_views.png` | Line chart (x=views, y=MPJPE, lines=noise) |
| `representative_views.png` | N=24 renders + keypoint overlay |

---

## Section 11: FL vs PS Comparison

### Why

FaceLift and Pose-Splatter both reconstruct 3D from multi-view input.
Since PS only supports fixed camera indices (no arbitrary c2w), we compare
on the GT 6-camera views where both can render.

### How

For each test frame:
- **GT**: Load original 6-view images
- **FL**: Load GS-LRM renders (or render live from Gaussians)
- **PS**: Load pre-rendered images (from `dump_ps_renders.py`)
- Overlay MAMMAL 3D keypoints projected to each view

### Usage

```bash
# Image grids
python -m mouse_extensions.scripts.render_comparison_grid \
    --gt-dir ~/data/preprocessed/FaceLift_mouse/M5t2 \
    --fl-dir outputs/tier_comparison/gslrm_6view_test/samples \
    --ps-dir /tmp/ps_renders \
    --mammal-3d /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
    --cam-pkl ~/data/raw/markerless_mouse_1_nerf/new_cam.pkl \
    --output-dir outputs/comparison_grids

# H.264 video
python -m mouse_extensions.scripts.render_comparison_grid \
    --gt-dir ... --fl-dir ... --ps-dir ... \
    --mammal-3d ... --cam-pkl ... \
    --output-dir outputs/comparison_grids \
    --video --fps 20
```

### Grid Layout

```
       View 0  View 1  View 2  View 3  View 4  View 5
GT     [img]   [img]   [img]   [img]   [img]   [img]
FL     [img]   [img]   [img]   [img]   [img]   [img]
PS     [img]   [img]   [img]   [img]   [img]   [img]

+ Colored keypoint overlay (22 joints + skeleton)
```

---

## Data Sources

| Data | Path | Format |
|------|------|--------|
| DANNCE 2D | `~/data/raw/markerless_mouse_1_nerf/keypoints2d_undist/result_view_{0-5}.pkl` | (18000, 22, 3) |
| MAMMAL 3D | `/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz` | (3600, 22, 3) |
| Raw Cameras | `~/data/raw/markerless_mouse_1_nerf/new_cam.pkl` | 6 × {K, R, t} |
| M5t2 Images | `~/data/preprocessed/FaceLift_mouse/M5t2/{frame}/images/cam_{v:03d}.png` | RGBA |

### Frame Mapping

- DANNCE: 18000 frames at 100fps
- M5: 3600 frames at 20fps
- Mapping: `dannce_frame = m5_frame × 5`
- Test set: M5 frames 3240-3599

---

## Dependencies

- `numpy`, `scipy` (triangulation, Procrustes)
- `matplotlib` (plotting)
- `PIL` (image grid composition)
- `ffmpeg` (H.264 video encoding)
- `torch`, GS-LRM (novel view rendering, Phase 2 only)

---

## Section 12: Multi-View Saturation Analysis

### Why

How many views are truly needed for accurate 3D triangulation? Beyond a certain
point, adding more views yields diminishing returns. This section quantifies
the saturation point and compares real vs virtual camera geometry.

### Three Experiments

#### Exp 1: Pure Turntable (3-192 views)
All cameras are virtual turntable cameras (elevation=20°, radius=2.7, 384px).
Noise σ = {0.5, 1, 2, 3, 5, 10} px applied uniformly.

#### Exp 2: Real 6-cam Baseline
Same noise levels applied to the 6 physical cameras (1152×1024).

#### Exp 3: Hybrid (6 Real + N Virtual)
Start with 6 real cameras, add 0-186 virtual turntable cameras.
Tests the practical scenario of augmenting existing hardware with GS-LRM renders.

### Key Results (σ=5px)

| Config | Views | MPJPE (mm) | vs Real 6-cam |
|--------|:-----:|:----------:|:-------------:|
| Real 6-cam only | 6 | **0.82** | baseline |
| Turntable 6v | 6 | 3.62 | 4.4x worse |
| Turntable 24v | 24 | 1.52 | 1.9x worse |
| Turntable 72v | 72 | 0.85 | ≈match |
| Turntable 192v | 192 | 0.53 | 36% better |
| Hybrid 6r+6t | 12 | 0.61 | 25% better |
| Hybrid 6r+18t | 24 | 0.49 | 40% better |
| Hybrid 6r+90t | 96 | 0.42 | 49% better |
| Hybrid 6r+186t | 192 | 0.40 | 51% better |
| **DANNCE real** | **6** | **6.15** | **7.5x worse** |

### Key Findings

1. **Theoretical scaling**: MPJPE ∝ σ/√N with ~15-20% geometry bonus
   - Log-log slope ≈ -0.55 (vs -0.50 theoretical)
   - Empirical always beats 1/√N due to parallax improvement

2. **Camera quality matters more than quantity**:
   - Real 6 cameras (0.82mm) ≈ 72 turntable cameras (0.85mm)
   - Resolution (1152×1024 vs 384) + 3D angular diversity → 4.4x advantage

3. **Hybrid is efficient**: Adding virtual views to real cameras saturates fast
   - 6r+6t (12 total): 25% improvement
   - 6r+18t (24 total): 40% improvement
   - 6r+90t+ (96+): ~50% improvement plateau

4. **Saturation point** (turntable, σ=5px):
   - 12-16 views: 80% of maximum improvement
   - 24 views: diminishing returns onset (<0.05 mm/cam)
   - 48+ views: <0.015 mm/cam marginal value

5. **DANNCE gap analysis**:
   - DANNCE real (6.15mm) vs Oracle σ=5px (0.82mm) → 5.33mm gap
   - Gap = calibration error + body model mismatch + detection bias
   - Implies DANNCE effective noise ≫ σ=5px (closer to σ=40px or systematic)

### Literature Context

- DLT triangulation error bound: O(σ/√N) (Hartley & Zisserman, 2003)
- Animal pose standard: 3-6 cameras (DANNCE, Anipose, DeepLabCut)
- No prior work systematically studies N>6 for animal pose triangulation
- This analysis establishes **geometric upper bounds** for virtual view augmentation

### Module: `saturation_visualizer.py`

```bash
python -m mouse_extensions.analysis.saturation_visualizer \
    --sat-json outputs/triangulation_saturation/saturation_analysis.json \
    --hybrid-json outputs/triangulation_saturation/hybrid_results.json \
    --output-dir outputs/triangulation_saturation/plots
```

### Outputs

| File | Description |
|------|-------------|
| `saturation_analysis.json` | Turntable + Real 6-cam results |
| `hybrid_results.json` | Hybrid (6 Real + N Virtual) results |
| `plots/01_saturation_curve.png` | MPJPE vs views (linear + log-log) |
| `plots/02_theoretical_fit.png` | Empirical vs 1/√N theory |
| `plots/03_real_vs_virtual.png` | Real camera superiority analysis |
| `plots/04_diminishing_returns.png` | Marginal value per camera |
| `plots/05_hybrid_comparison.png` | Turntable vs Hybrid comparison |
| `plots/06_comprehensive_dashboard.png` | 6-panel combined dashboard |

---

*v1.1 | 2026-03-04 | Saturation analysis added*
