# FaceLift Outputs Directory Inventory

**Date**: 2026-03-05
**Location**: `/home/joon/dev/FaceLift/outputs/`
**Total Subdirectories**: 13
**Key Metrics**: Full phase3_e2e inventory with run configurations and benchmark results

---

## Directory Structure Overview

```
outputs/
├── phase3_e2e/              # Primary E2E experiment results (5 checkpoints)
├── _archive_renders/        # Archived E2E baseline results (6 experiments)
├── tier_comparison/         # Input view count ablation (1-6 views)
├── triangulation/           # Pose estimation & 3D reconstruction analysis
├── analysis/                # Post-hoc analysis directories
├── visualizations/          # Cross-comparison visualizations
├── reports/                 # Markdown & JSON analysis reports
├── verify_turntable/        # Ground truth turntable rendering validation
├── dataset_verification/    # Data consistency checks
├── camera_follow_v5/        # Camera trajectory visualization
├── keypoint_viz/            # 2D keypoint overlay comparisons
└── *.log, .gitignore        # Inference logs & git config
```

---

## 1. phase3_e2e/ - Primary E2E Experiment Results

**Purpose**: End-to-end MVDiffusion + GS-LRM inference on M5t2 test set (360 frames)
**Data**: `/home/joon/data/preprocessed/FaceLift_mouse/M5` (split: `data_mouse_t2_test.txt`)
**Model Config**: `M5t` with `M5t2_E0_1_facelift/best_psnr.pt` GS-LRM checkpoint
**Inference Params**: `input_view_idx=0` (GT first view), `num_steps=50`, `guidance_scale=3.0`

### 1.1 H3_resume_pose

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | H3_resume_pose |
| **Timestamp** | 2026-02-24T13:59:12 |
| **MVDiffusion Checkpoint** | `/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_H3_resume_pose/checkpoint-10000` |
| **Pose Integration** | None (no pose_config_yaml) |
| **Output Artifacts** | `samples/`, `metrics.json`, `metrics_v2.json`, MP4 videos, grid image |
| **Key Metrics** | PSNR_fg=8.85±2.60 dB, IoU=0.569±0.145, SSIM=0.968±0.011 |
| **Artifacts Generated** | turntable_grid.jpg, time_*.mp4 (3 variants) |

**Metrics Extract (overall)**:
```json
{
  "psnr_full_white": {"mean": 21.953, "std": 2.629},
  "psnr_fg_only": {"mean": 8.854, "std": 2.601},
  "ssim_full_white": {"mean": 0.9684, "std": 0.0105},
  "silhouette_iou": {"mean": 0.5687, "std": 0.1448},
  "lpips_full_white": {"mean": 0.0546, "std": 0.0225}
}
```

### 1.2 H4b_extended_step8000

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | H4b_extended_step8000 |
| **Timestamp** | 2026-02-27T09:51:48 |
| **MVDiffusion Checkpoint** | `/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_H4b_extended/checkpoint-8000` |
| **Pose Integration** | None (`pose_config_yaml=null`, `pose_weights=null`) |
| **Outputs** | `samples/`, `metrics.json`, no videos/mesh (flags: `no_turntable=true`, `no_mesh=true`) |
| **Status** | Intermediate checkpoint; metrics available but no visualization |

### 1.3 H4b_step20000_e2e ⭐ CRITICAL CHECKPOINT

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | H4b_step20000_e2e |
| **Timestamp** | 2026-03-03T22:01:00 |
| **MVDiffusion Checkpoint** | `/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_H4b_extended/checkpoint-20000` |
| **Pose Integration** | **YES** — Plucker ray conditioning with spatial tokens |
| **Pose Config** | `configs/mvdiffusion/mouse_mvdiffusion_M5t2_H4b_extended.yaml` |
| **Pose Weights** | `/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_H4b_extended/checkpoint-20000/pose_injector.pt` |
| **Outputs** | `samples/` only (no metrics, videos, or mesh) |
| **Note** | Latest checkpoint with pose support; inference completed but metrics NOT computed |

### 1.4 H6a_v2_plucker_step9000

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | H6a_v2_plucker_step9000 |
| **Timestamp** | 2026-02-26T21:55:52 |
| **MVDiffusion Checkpoint** | `/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_H6a_v2_plucker_trainable/checkpoint-9000` |
| **Pose Integration** | None (baseline Plucker without pose loss) |
| **Outputs** | `samples/`, `metrics.json` |
| **Purpose** | Plucker ray embedding baseline (val PSNR=25.57@1K) |

### 1.5 H6a_v2_plucker_step9000_with_pose

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | H6a_v2_plucker_step9000_with_pose |
| **Timestamp** | 2026-02-27T01:16:37 |
| **MVDiffusion Checkpoint** | `/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_H6a_v2_plucker_trainable/checkpoint-9000` |
| **Pose Integration** | **YES** — Spatial token pose conditioning |
| **Pose Config** | `configs/mvdiffusion/mouse_mvdiffusion_M5t2_H6a_v2_plucker_trainable.yaml` |
| **Pose Weights** | `null` (frozen pretrained encoder, no learnable pose injector) |
| **Outputs** | `samples/`, `metrics.json` |
| **Purpose** | Plucker + spatial token pose conditioning (frozen encoder variant) |

---

## 2. _archive_renders/ - Archived Baselines & Early Experiments

**Purpose**: Historical E2E baseline results and early ablation experiments
**Status**: Superseded by phase3_e2e experiments; retained for reference

### 2.1 E1_cosine_20k

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | E1_cosine_20k |
| **MVDiffusion Checkpoint** | `mouse_M5t2_E1_cosine_lr/checkpoint-20000` |
| **Training Schedule** | Cosine annealing LR |
| **Outputs** | `samples/`, `metrics_v2.json` (no metrics.json) |
| **Note** | Early training experiment; metrics only in v2 format |

### 2.2 E2_resume_20k ⭐ BASELINE E2E

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | E2_resume_20k |
| **MVDiffusion Checkpoint** | `mouse_M5t2_E2_resume_linear_decay/checkpoint-20000` |
| **Training Schedule** | Linear decay LR, resumed from E1 |
| **Outputs** | `samples/`, `metrics.json`, `metrics_v2.json`, MP4 videos, grid image |
| **Key Metrics** | PSNR_fg=8.20±3.22 dB, IoU=0.521±0.188 |
| **Purpose** | Standard baseline for E2E comparison |

**Metrics Extract (overall)**:
```json
{
  "psnr_full_white": {"mean": 21.362, "std": 3.065},
  "psnr_fg_only": {"mean": 8.198, "std": 3.219},
  "ssim_full_white": {"mean": 0.9657, "std": 0.0140},
  "silhouette_iou": {"mean": 0.5206, "std": 0.1878},
  "lpips_full_white": {"mean": 0.0628, "std": 0.0330}
}
```

### 2.3 E3_pose_extrinsic_add

| Attribute | Value |
|-----------|-------|
| **Experiment ID** | E3_pose_extrinsic_add |
| **Pose Integration** | Pose MLP + additive injection |
| **Outputs** | `samples/`, `metrics.json`, `metrics_v2.json` |
| **Note** | Early pose experiment (additive method); superseded by H4b/H6a variants |

### 2.4 P1_6view_e2e, P1_BL_6view_e2e, P1_E1_6view_e2e

| Experiment | MVDiffusion Checkpoint | Pose | Outputs |
|-----------|----------------------|------|---------|
| P1_6view_e2e | `mouse_M5t2/checkpoint-8000` | None | `samples/`, `run_config.json` |
| P1_BL_6view_e2e | Baseline config | N/A | `samples/`, `run_config.json` |
| P1_E1_6view_e2e | E1 variant | None | `samples/`, `run_config.json` |

**Note**: These early experiments have only `run_config.json` and `samples/`; metrics and videos not generated.

---

## 3. tier_comparison/ - Input View Count Ablation

**Purpose**: Investigate GS-LRM performance scaling with input view count (1–6 views)
**Baseline**: `base_uniform_v2_*view_v2/best_psnr.pt` checkpoints per view count
**MVDiffusion**: All using `mouse_M5t/checkpoint-8000` (non-M5t2 variant)

| Directory | Num Views | GS-LRM Checkpoint | Metrics | Notes |
|-----------|-----------|-------------------|---------|-------|
| gslrm_1view_test | 1 | base_uniform_v2_1view_v2 | ✓ metrics.json | Input view=1 condition |
| gslrm_2view_test | 2 | base_uniform_v2_2view_v2 | ✗ | No metrics computed |
| gslrm_3view_test | 3 | base_uniform_v2_3view_v2 | ✗ | No metrics computed |
| gslrm_4view_test | 4 | base_uniform_v2_4view_v2 | ✗ | No metrics computed |
| gslrm_5view_test | 5 | base_uniform_v2_5view_v2 | ✗ | No metrics computed |
| gslrm_6view_test | 6 | base_uniform_v2_6view_v2 | ✓ metrics.json | 6-view bound |

**Key Finding**: gslrm_1view_test baseline only available checkpoint with metrics for single-view condition.

---

## 4. triangulation/ - Pose Estimation & Multi-View Reconstruction

**Purpose**: Validate 2D→3D keypoint triangulation and multi-view pose consistency

### 4.1 Structure

```
triangulation/
├── neural_detection/          # Neural network pose predictions
│   └── renders/               # Rendered output directory
├── oracle_gt_views/           # GT pose triangulation (6-view consensus)
│   ├── gt_view_results.json   # Triangulation accuracy metrics
│   └── accuracy_vs_views.png  # Plot: accuracy vs. view count
├── oracle_multiview/          # Multi-view stereo reconstruction
│   ├── multiview_results.json
│   └── accuracy_vs_views.png
├── oracle_saturation/         # Noise saturation analysis
│   ├── saturation_analysis.json
│   ├── hybrid_results.json    # Hybrid oracle+neural blend
│   ├── extended_results.json
│   ├── hybrid_log.txt
│   └── plots/
├── visualizations/            # 5 diagnostic PNG plots
│   ├── 01_heatmap.png
│   ├── 02_lines_with_gt.png
│   ├── 03_per_joint_6v.png
│   ├── 04_noise_reduction.png
│   └── 05_camera_layouts.png
└── README.md                  # Detailed triangulation methodology
```

### 4.2 Key Files

| File | Contents | Purpose |
|------|----------|---------|
| `oracle_gt_views/gt_view_results.json` | Accuracy metrics vs. view subset | Ground truth triangulation baseline |
| `oracle_multiview/multiview_results.json` | Multi-view stereo metrics | Cross-validation with camera geometry |
| `oracle_saturation/saturation_analysis.json` | Noise robustness curves | Input noise tolerance analysis |
| `oracle_saturation/hybrid_results.json` | Oracle + neural blend metrics | Hybrid approach validation |

---

## 5. verify_turntable/ - Ground Truth Turntable Rendering

**Purpose**: Render GT Gaussian splats on turntable motion for validation
**Data**: M5t2 subset frames rendered with GS-LRM checkpoints

### 5.1 Structure

```
verify_turntable/
├── train/                 # Training set GT renders
│   ├── input_0.jpg       # Input view
│   ├── gaussians_0.ply   # GS-LRM output
│   ├── turntable_0.jpg   # Single turntable frame
│   └── turntable_orbit_0.mp4
├── val/                   # Validation set (sparse)
│   └── 00000000/
├── temporal/              # Temporal consistency across frames
│   ├── time_fixed.mp4
│   ├── time_grid_6view.mp4
│   ├── time_rotating.mp4
│   └── turntable_grid.jpg
└── inference/             # Inference-mode GT renders
    └── sample_0000, sample_0001
```

### 5.2 Key Outputs

| File | Type | Purpose |
|------|------|---------|
| `temporal/time_*.mp4` | Video (variable fps) | Temporal coherence check |
| `temporal/turntable_grid.jpg` | Image grid | Multi-frame turntable montage |
| `train/*.ply` | 3D mesh (PLY format) | Gaussian splat geometry |

---

## 6. dataset_verification/ - Data Consistency Checks

**Purpose**: Validate preprocessed data integrity and consistency

### 6.1 Structure

```
dataset_verification/
├── dataset_consistency.json     # Quantitative consistency metrics
├── dataset_consistency_report.md # Markdown report
└── image_comparison/            # Visual validation folder
```

### 6.2 Key Metrics

| Metric | Source |
|--------|--------|
| Frame count, resolution validation | `dataset_consistency.json` |
| Camera parameter consistency | `dataset_consistency.json` |
| RGB/alpha channel checks | `image_comparison/` (visual) |

---

## 7. triangulation/README.md - Detailed Pose Documentation

**Contents**: Full triangulation methodology, multi-view geometry, noise analysis
**Location**: `/home/joon/dev/FaceLift/outputs/triangulation/README.md`

---

## 8. visualizations/ - Cross-Comparison Plots

### 8.1 cross_view_comparison/

**Purpose**: Compare FaceLift outputs across 4 different model variants

| File Pattern | Content |
|--------------|---------|
| `cross_view_*.png` | Full-resolution grid comparisons |
| `row_XXXXXX_v*.png` | Per-frame row crops (v1=baseline, v3=latest) |
| `cross_view_full_grid.png` | Master comparison grid |

**Frames Shown**: 003240, 003320, 003400 (representative test set frames)

### 8.2 fl_vs_ps_comparison/

**Purpose**: FaceLift vs. Pose-Splatter fairness comparison

| File | Content |
|------|---------|
| `comparison_frame_XXXXXX.png` | Side-by-side frame comparison |
| `fl_vs_ps_multi_frame.png` | Multi-frame montage |

**Frames**: 003240, 003300, 003360, 003420, 003480, 003540, 003599 (temporal diversity)

---

## 9. reports/ - Analysis & Summary Reports

### 9.1 Primary Reports

| File | Purpose |
|------|---------|
| **UNIFIED_ANALYSIS_REPORT.md** | Master summary report (all experiments) |
| **h1_diagnosis_report.md** | H1 (GS-LRM only) diagnosis & comparison |
| **mvdiff_checkpoint_comparison.md** | MVDiffusion checkpoint ablation summary |
| **view_ablation_report.md** | Tier comparison (1–6 views) analysis |

### 9.2 JSON Data Files

| File | Contents |
|------|----------|
| `report_summary.json` | Aggregated metrics across all experiments |
| `h1_diagnosis_report.json` | H1 baseline structured metrics |
| `view_ablation_report.json` | View ablation quantitative results |
| `h1_diagnosis_comparison.json` | H1 vs. other variants comparison |

### 9.3 Images Subfolder

```
reports/images/
├── view_ablation_*.png     # Tier comparison visualizations
├── mvdiff_checkpoint_*.png # Checkpoint comparison plots
└── ...
```

---

## 10. camera_follow_v5/ - Camera Trajectory Visualization

**Purpose**: Novel camera motion visualization (face-following trajectory)
**Data**: Computed trajectory for mouse face region across test set

| File | Type | Purpose |
|------|------|---------|
| `camera_follow_face_clean.mp4` | Video | Trajectory only (no overlay) |
| `camera_follow_face_overlay.mp4` | Video | GT rendered + trajectory overlay |
| `camera_follow_face_sidebyside.mp4` | Video | Side-by-side (clean + overlay) |
| `representative_sidebyside.png` | Image | Single frame comparison |
| `trajectory.npz` | NumPy array | Raw (x, y, z) coordinates |

**FPS/Duration**: 10 fps (matches inference output)

---

## 11. keypoint_viz/ - 2D Pose Overlay Comparisons

**Purpose**: Visualize 2D keypoint predictions vs. ground truth
**Coverage**: 5 representative test frames (003240, 003280, 003320, 003360, 003400)

### 11.1 File Pattern per Frame

```
frame_XXXXXX_6cam_overlay.png      # 6-camera keypoint overlay
frame_XXXXXX_gt_vs_pred.png        # GT (green) vs. Predicted (red) side-by-side
frame_XXXXXX_labeled_detail.png    # Labeled joint names + accuracy scores
```

### 11.2 Typical Contents

- **6cam_overlay**: Multi-view grid showing 2D keypoint detections on each camera angle
- **gt_vs_pred**: Pixel-space comparison (GT green, pred red, overlap yellow)
- **labeled_detail**: Per-joint accuracy metrics (AP, PCK@0.05) text overlays

---

## 12. analysis/oracle_mvdiff/

**Purpose**: Post-hoc analysis of MVDiffusion behavior with oracle inputs

**Status**: Directory exists; detailed structure TBD (requires deepening)

---

## 13. Auxiliary Files

### 13.1 Log Files (outputs/ root)

```
gslrm_*view_inference.log  (113-115 KB)  # Inference logs (2, 3, 5 views)
gslrm_*view_fair.log       (749-836 B)   # Fairness evaluation snippets
```

### 13.2 Configuration

- `.gitignore`: Standard Python/torch output exclusions
- No other config files in outputs/ root

---

## Summary Table: phase3_e2e Checkpoint Comparison

| Checkpoint | Date | MVDiff Step | Pose? | PSNR_fg | IoU | SSIM | Videos | Mesh | Status |
|-----------|------|-------------|-------|---------|-----|------|--------|------|--------|
| H3_resume_pose | 2/24 | 10K | ✗ | 8.85 | 0.569 | 0.968 | ✓ | ✓ | ✓ Complete |
| H4b_extended_step8000 | 2/27 | 8K | ✗ | ? | ? | ? | ✗ | ✗ | Partial |
| **H4b_step20000_e2e** | 3/3 | 20K | ✓ Plucker+Token | — | — | — | ✗ | ✗ | ⚠️ Inference only |
| H6a_v2_plucker_step9000 | 2/26 | 9K (Plucker BL) | ✗ | ? | ? | ? | ✗ | ✗ | Partial |
| H6a_v2_plucker_step9000_with_pose | 2/27 | 9K (Plucker) | ✓ Spatial Token (frozen) | ? | ? | ? | ✗ | ✗ | Partial |

**Legend**:
- **✓ Complete**: Metrics, videos, and mesh all generated
- **Partial**: Metrics and samples only (no video/mesh)
- **⚠️ Inference only**: Samples generated; metrics NOT computed
- **?**: Metrics not extracted in inventory (see metrics.json for details)

---

## File Count Summary

| Directory | File Type | Approx. Count | Size |
|-----------|-----------|---------------|------|
| phase3_e2e/*/samples/ | PNG frames | ~3,600 (360 frames × 10 views) | ~10 GB |
| _archive_renders/*/samples/ | PNG frames | ~4,320 (6 exps × 360 frames × ~20 views) | ~12 GB |
| tier_comparison/*/samples/ | PNG frames | ~3,600 (6 tiers × 360 frames × ~16 views) | ~10 GB |
| Other (metrics, videos, reports) | JSON, MP4, MD, PNG | ~150 | ~500 MB |
| **TOTAL** | — | **~11,000+ frames** | **~32+ GB** |

---

## Key Checkpoints for README Documentation

### Best E2E Results
- **H3_resume_pose** (Feb 24): Most complete results with videos and metrics
- **E2_resume_20k** (archived): Baseline E2E standard

### Latest Pose Integration
- **H4b_step20000_e2e** (Mar 3): Newest checkpoint with Plucker ray + spatial token pose, but **metrics NOT computed yet**

### Fairness Comparisons
- **tier_comparison/**: Full view count ablation (1–6 views)
- **visualizations/fl_vs_ps_comparison/**: FaceLift vs. Pose-Splatter side-by-side

### Validation
- **triangulation/**: Complete 3D pose validation with oracle and multi-view consensus
- **verify_turntable/**: GT rendering consistency checks

---

*Created 2026-03-05 | SSH-gathered from gpu03:/home/joon/dev/FaceLift/outputs/*
