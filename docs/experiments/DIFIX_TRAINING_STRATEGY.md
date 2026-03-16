# DiFix 3D+ Training Strategy for GS-LRM Artifact Removal

> **Navigation**: [<- INDEX](../INDEX.md) | [PHASE2_NOVEL_VIEW_ROADMAP](PHASE2_NOVEL_VIEW_ROADMAP.md) | [mesh_gs_pair_collection](mesh_gs_pair_collection.md)
> **Version**: v1.0 | **Created**: 2026-03-12 | **Status**: PLANNING
> **Source**: 3-model deliberation (Claude/Gemini/GPT-4o) consensus

---

## 1. Overview

GS-LRM novel view renders suffer from Gaussian artifacts (floaters, spikes, white splats).
**DiFix 3D+** (CVPR 2025) is a diffusion-based 3D-aware image restoration model that can
remove such artifacts when trained on (degraded, clean) image pairs.

### Three Pair Types

| Type | Input (Degraded) | Target (Clean) | Domain Gap | Scale |
|:----:|-------------------|-----------------|:----------:|:-----:|
| **Type 1** | N-view GS-LRM render @ GT camera | Original GT RGB | Low (same camera) | 3600 x 6 x 5 = 108K |
| **Type 2** | 6-view GS-LRM render @ novel camera | MAMMAL mesh render | Medium (pseudo-GT) | 3600 x 4 = 14.4K |
| **Type 3** | N-view GS-LRM render @ GT camera | 6-view GS-LRM render @ same camera | Minimal (same model) | 3600 x 6 x 5 = 108K |

- **Type 1**: Real GT supervision, but only at GT camera poses (no novel views)
- **Type 2**: Novel view pairs, but MAMMAL pseudo-GT has template fitting limitations
- **Type 3**: Self-supervised within GS-LRM; degradation = fewer input views

## 2. Training Strategy: 2.5-Stage Curriculum

```
Stage 0: Zero-Shot Baseline
  └── DiFix 3D+ pretrained weights → evaluate on our data without fine-tuning
      └── Establishes lower bound

Stage 1: Self-Supervised Foundation (Type 3)
  ├── Input: N-view (1,2,3,4,5) GS-LRM renders @ GT cameras
  ├── Target: 6-view GS-LRM renders @ same GT cameras
  ├── Why: Minimal domain gap, large scale (108K pairs), teaches artifact patterns
  ├── Curriculum: Start with 1-view (max artifacts) → add 5-view (subtle artifacts)
  └── Expected: Model learns GS-LRM-specific artifact removal

Stage 1.5: Novel View Bridge (+Type 2)
  ├── Mix Type 3 (80%) + Type 2 (20%) pairs
  ├── Type 2: 6-view novel renders → MAMMAL pseudo-GT
  ├── Why: Bridges GT-camera-only → novel-view generalization
  ├── Caution: MAMMAL pseudo-GT has template fitting noise
  └── Expected: Artifact removal generalizes to novel viewpoints

Stage 2: GT Fine-Tune (+Type 1)
  ├── Mix Type 1 (50%) + Type 3 (30%) + Type 2 (20%)
  ├── Type 1: N-view renders → real GT RGB
  ├── Why: Real GT signal for photorealistic output quality
  ├── Lower LR, shorter schedule (fine-tune only)
  └── Expected: Best quality at GT cameras, improved novel views
```

### Why This Order?

1. **Stage 1 first**: Type 3 has zero domain gap (same renderer). Model learns
   what GS-LRM artifacts look like without confounding factors.
2. **Stage 1.5 bridge**: Type 2 introduces viewpoint diversity. MAMMAL pseudo-GT
   is imperfect but provides geometric guidance for novel views.
3. **Stage 2 last**: Type 1 provides real GT but only at GT cameras. Fine-tuning
   on this prevents quality regression while keeping novel view generalization.

## 3. Data Pair Specification

### Type 1: GT View Pairs (N-view render <-> GT RGB)

```
Input:  outputs/datasets/novel_view/mouse_m5t2/ablation_{N}view/cam_NNN/NNNNN.png
Target: outputs/datasets/novel_view/mouse_m5t2/gt_rgb/cam_NNN/NNNNN.png

N = 1, 2, 3, 4, 5 (view count)
cam = cam_000 ... cam_005 (6 GT cameras)
frame = 00000 ... 03599 (3600 frames)
```

- **Degradation gradient**: 1-view (PSNR ~10.5 dB, severe) -> 5-view (PSNR ~22.2 dB, subtle)
- **Resolution**: 384x384
- **Pairs**: 3600 x 6 x 5 = 108,000

### Type 2: Novel View Pairs (6-view novel render <-> MAMMAL pseudo-GT)

```
Input:  outputs/datasets/novel_view/mouse_m5t2/tier0_raw/{view}/NNNNN.png
Target: outputs/datasets/novel_view/mouse_m5t2/pseudo_gt/{view}/NNNNN.png

view = bottom, top, front_low, side_low
frame = 00000 ... 03599
```

- **Domain gap**: GS-LRM (Gaussian splatting) vs MAMMAL (mesh template with UV texture)
- **MAMMAL limitations**: Template fitting error, limited surface detail, texture quality
- **Pairs**: 3600 x 4 = 14,400

### Type 3: View Ablation Pairs (N-view render <-> 6-view render)

```
Input:  outputs/datasets/novel_view/mouse_m5t2/ablation_{N}view/cam_NNN/NNNNN.png
Target: outputs/datasets/novel_view/mouse_m5t2/gt_views/cam_NNN/NNNNN.png

N = 1, 2, 3, 4, 5
cam = cam_000 ... cam_005
frame = 00000 ... 03599
```

- **Domain gap**: Zero (same model, same renderer, different input views)
- **Degradation level known**: N-view count directly controls artifact severity
- **Pairs**: 3600 x 6 x 5 = 108,000

### Domain Gap Analysis

| Pair Type | Spatial Gap | Appearance Gap | Scale | Training Signal Quality |
|:---------:|:----------:|:--------------:|:-----:|:----------------------:|
| Type 1 | None (same camera) | Low (render vs photo) | 108K | Best (real GT) |
| Type 2 | Novel viewpoint | High (GS vs mesh) | 14.4K | Medium (pseudo-GT) |
| Type 3 | None (same camera) | Zero (same renderer) | 108K | Good (self-supervised) |

## 4. Inference Pipeline

### Per-View DiFix Application

```
For each novel view camera C:
  1. GS-LRM render → raw image I_raw(C)
  2. DiFix inference → cleaned image I_clean(C)
  3. Optional: confidence/uncertainty map for downstream filtering

Multi-view consistency (post-processing):
  - Per-view DiFix may introduce inter-view inconsistency
  - Mitigation 1: Shared latent conditioning across views (DiFix 3D+ native)
  - Mitigation 2: Epipolar consistency check → re-render inconsistent regions
  - Mitigation 3: 3DGS re-fitting from cleaned multi-view → consistent novel views
```

### Integration with Existing Pipeline

```
[6-view GT input]
    → MVDiffusion (if E2E) or direct (GT input)
    → GS-LRM → 3D Gaussians
    → Render at novel cameras → raw renders (Tier 0)
    → DiFix 3D+ → cleaned renders (Tier 1+)
    → Evaluate: raw vs cleaned vs MAMMAL pseudo-GT
```

## 5. Folder Structure

### Source Data (outputs/datasets/novel_view/)

```
outputs/datasets/novel_view/
├── cameras/, splits/, manifest.json
└── mouse_m5t2/
    ├── tier0_raw/{view}/           # 6v GS-LRM novel renders (14,400)
    ├── pseudo_gt/{view}/           # MAMMAL mesh renders (14,400)
    ├── gt_views/cam_NNN/           # 6v GS-LRM @ GT cameras (21,600)
    ├── gt_rgb/cam_NNN/             # Original GT images (21,600)
    ├── ablation_{1-5}view/cam_NNN/ # N-view renders (108,000)
    └── metadata/
```

### DiFix Training Pairs (outputs/datasets/difix_pairs/)

```
outputs/datasets/difix_pairs/
├── manifest.json                   # Unified manifest with pair_type, degradation_level
├── type1_gt_view/                  # N-view render <-> GT RGB
│   ├── {frame}_{cam}_{N}v/
│   │   ├── input.png → symlink to ablation_{N}view/cam/frame.png
│   │   └── target.png → symlink to gt_rgb/cam/frame.png
│   └── ...
├── type2_novel_view/               # 6v novel render <-> MAMMAL pseudo-GT
│   ├── {frame}_{view}/
│   │   ├── input.png → symlink to tier0_raw/view/frame.png
│   │   └── target.png → symlink to pseudo_gt/view/frame.png
│   └── ...
└── type3_view_ablation/            # N-view render <-> 6-view render
    ├── {frame}_{cam}_{N}v/
    │   ├── input.png → symlink to ablation_{N}view/cam/frame.png
    │   └── target.png → symlink to gt_views/cam/frame.png
    └── ...
```

## 6. Compute & Storage Estimates

### Rendering

| Phase | GPU | Wall Time | Output |
|-------|:---:|:---------:|:------:|
| 6-view GS-LRM (3600 frames) | 1x A6000 | ~30 min | tier0_raw + gt_views + gt_rgb (8.3 GB) |
| MAMMAL pseudo-GT (3600 frames) | 1x A6000 | ~18 min | pseudo_gt (2.1 GB) |
| View ablation (5 x 3600 frames) | 3x A6000 | ~50 min | ablation_{1-5}view (15.6 GB) |
| **Total rendering** | | **~1.5 hr** | **~26 GB** |

### DiFix Pair Building

| Phase | Resource | Time | Output |
|-------|:--------:|:----:|:------:|
| Symlink-based pair build | CPU | ~10 min | ~230K symlinks (~negligible disk) |
| Manifest generation | CPU | ~2 min | manifest.json |

### DiFix Training (Estimated)

| Stage | Pairs | Epochs | Est. Time (1x A6000) |
|-------|:-----:|:------:|:--------------------:|
| Stage 1 (Type 3) | 108K | 20 | ~8 hr |
| Stage 1.5 (+Type 2) | 122K | 10 | ~5 hr |
| Stage 2 (+Type 1) | 230K | 5 | ~4 hr |
| **Total training** | | | **~17 hr** |

### Storage Summary

| Component | Size |
|-----------|:----:|
| Raw renders (novel_view/) | ~26 GB |
| DiFix pairs (symlinks) | ~1 MB |
| DiFix model checkpoints | ~2 GB |
| **Total** | **~28 GB** |

## 7. Related Documents

- <- [[../INDEX]] -- Document hub
- <-> [[PHASE2_NOVEL_VIEW_ROADMAP]] -- Phase 2 roadmap (novel view enhancement pipeline)
- <-> [[mesh_gs_pair_collection]] -- Tier-based dataset pipeline
- <-> [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] -- Coordinate transforms
- <-> [[STAGE1_REPLACEMENT_CANDIDATES]] -- Stage 1 alternatives (DiFix 3D+ is a candidate)

---

*FaceLift | DiFix 3D+ Training Strategy | 2026-03-12*
