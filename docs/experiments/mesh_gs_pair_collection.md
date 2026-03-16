# Mesh-GS Pair Collection Pipeline v2.0

> **Navigation**: [← INDEX](../INDEX.md) | [EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)
> **Purpose**: Novel view dataset collection — GS-LRM + MAMMAL mesh pairs, tier-based structure
> **Updated**: 2026-03-12 | **Version**: v2.0

---

## 1. Overview

3-tier dataset pipeline for NeurIPS 2026 Evaluations & Datasets Track:

| Tier | Source | Content | Benchmark Use |
|:----:|--------|---------|:-------------:|
| **Tier 0** | GS-LRM (6v GT input) | Raw novel view renders | Baseline |
| **Tier 1** | OpenCV cleanup | Artifact-removed renders | Primary GT |
| **Tier 2** | AI enhancement (Nano Banana/Gemini) | Qualitative improvement | Qualitative only |
| **Pseudo-GT** | MAMMAL mesh fitting | Template-based renders | Reference (not absolute GT) |

## 2. Why

GS-LRM은 6-view GT 입력 시 PSNR 23.84 dB로 우수하지만, novel view에서는 Gaussian artifact가 발생.
MAMMAL mesh는 동일 카메라에서 pseudo-GT를 제공하여 artifact 정량화 가능.
3-tier 구조로 raw → cleaned → enhanced 진행, 각 단계의 기여를 독립 측정.

## 3. Directory Structure (v2.0)

```
outputs/novel_view_dataset/
├── manifest.json                    # Dataset-level metadata
├── cameras/
│   └── novel_views.json             # Global novel view camera params (same all frames)
├── mouse_m5t2/                      # Species + dataset
│   ├── tier0_raw/                   # Raw GS-LRM novel view renders
│   │   ├── bottom/                  # -70° elevation
│   │   │   ├── 00000.png
│   │   │   └── ...
│   │   ├── top/                     # +70° elevation
│   │   ├── front_low/               # -30° elevation, 0° azimuth
│   │   └── side_low/                # -30° elevation, 90° azimuth
│   ├── pseudo_gt/                   # MAMMAL mesh renders at novel cameras
│   │   └── (same view structure)
│   ├── tier1_cleaned/               # Stage 1: OpenCV artifact removal (TODO)
│   ├── tier2_enhanced/              # Stage 2: AI enhancement (TODO)
│   ├── artifact_masks/              # Binary artifact masks (TODO)
│   ├── gt_views/                    # GS-LRM renders at GT camera poses
│   │   ├── cam_000/ ... cam_005/
│   ├── gt_rgb/                      # Original GT camera images
│   │   └── (same cam structure)
│   └── metadata/                    # Per-frame metadata JSON
│       ├── 00000.json ... NNNNN.json
├── splits/
│   ├── train.json                   # Frame 0-2879
│   ├── val.json                     # Frame 2880-3239
│   └── test.json                    # Frame 3240-3599
└── visualizations/                  # Comparison grids, videos
```

**Key design**: view-first organization (`tier0_raw/bottom/00000.png`) — easy to glob all frames for one view type.

## 4. Novel View Cameras

All frames share identical 4 novel view cameras:

| View | Elevation | Azimuth | Purpose |
|------|:---------:|:-------:|---------|
| `bottom` | -70° | 0° | Ventral view (belly, paws) |
| `top` | +70° | 0° | Dorsal view (back, ears) |
| `front_low` | -30° | 0° | Low frontal (face, whiskers) |
| `side_low` | -30° | 90° | Low lateral (profile, gait) |

- **Intrinsics**: fx=fy=411.75 @ 384×384 (scaled from GT 548.99@512)
- **Convention**: OpenCV (X-right, Y-down, Z-forward)
- **Radius**: 2.7 (turntable distance)

## 5. Coordinate Transform

```python
point_fl = (point_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE
# M5_SCENE_CENTER = [59.672, 51.517, 107.099]  (mm)
# M5_DISTANCE_SCALE = 2.7 / 307.785 ≈ 0.008781
```

Frame mapping: M5 frame idx N → MAMMAL video frame N×5 (100fps→20fps)

> **UV texture**: Template mesh vertex expansion (14,522→~15,399) handled via face-level mapping.
> **✅ Resolved**: See [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]]

## 6. Usage

### Migration from PoC v0

```bash
# On gpu03, facelift env
python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode migrate \
    --poc_dir outputs/poc_mesh_gs_pairs_v0_archive
```

### Fresh Generation

```bash
# Phase 1: GS-LRM renders (facelift env, needs GPU)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase gslrm \
    --frame_range 0 150

# Phase 2: MAMMAL pseudo-GT (mammal_stable env, EGL)
conda activate mammal_stable
PYOPENGL_PLATFORM=egl python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase mammal \
    --frame_range 0 150

# Rebuild manifest
python -m mouse_extensions.scripts.novel_view.collect_dataset --mode manifest

# Visualize comparison grids
python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode visualize --frames 0 100 137
```

## 7. Metadata Schema

### Per-Frame (`metadata/NNNNN.json`)

```json
{
  "frame_idx": 137,
  "mammal_video_frame": 685,
  "split": "train",
  "species": "mouse",
  "dataset": "M5t2",
  "resolution": 384,
  "data_available": {
    "tier0_raw": true, "pseudo_gt": true,
    "gt_views": true, "gt_rgb": true,
    "tier1_cleaned": false, "tier2_enhanced": false,
    "artifact_masks": false
  },
  "gt_cameras": { "cam_000": { "c2w": [...], "fxfycxcy": [...] }, ... }
}
```

### Dataset Manifest (`manifest.json`)

Total frames, availability counts, pipeline versions, split boundaries, coordinate transform constants.

## 8. Data Scale

| Component | Count |
|-----------|:-----:|
| Total frames (M5t2) | 3,600 |
| GT camera views | 6 per frame = 21,600 |
| Novel views | 4 per frame = 14,400 |
| **Total pairs** | **36,000** |

## 9. M5t2 Splits

| Split | Frame Range | Count | Ratio |
|-------|:-----------:|:-----:|:-----:|
| Train | 0 ~ 2,879 | 2,880 | 80% |
| Val | 2,880 ~ 3,239 | 360 | 10% |
| Test | 3,240 ~ 3,599 | 360 | 10% |

## 10. History

| Version | Date | Change |
|:-------:|------|--------|
| v2.0 | 2026-03-12 | Tier-based structure, per-frame metadata, migration pipeline, view-first org |
| v1.0 | 2026-03-11 | PoC flat structure (poc_mesh_gs_pairs.py) |

## 11. Related Documents

- ↑ [[../INDEX]] — Document hub
- ↔ [[PHASE2_NOVEL_VIEW_ROADMAP]] — Phase 2 roadmap
- ↔ [[FL_vs_PS_comparison]] — FaceLift vs Pose-Splatter comparison
- ↔ [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] — Coordinate transforms
- ↔ [[../../mouse_extensions/docs/DATASET_FRAME_INDEXING]] — Frame indexing & data specs
- ↔ [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] — UV texture bug fix
- ↔ [[DIFIX_TRAINING_STRATEGY]] — DiFix 3D+ training strategy (uses this dataset as source)

---

*FaceLift | Mesh-GS Pair Collection Pipeline | 2026-03-12*
