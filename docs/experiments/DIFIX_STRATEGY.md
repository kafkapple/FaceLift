# DiFix Artifact Removal Strategy

> [← INDEX](../INDEX.md) | [PHASE2](PHASE2_NOVEL_VIEW_ROADMAP.md)
> **Version**: v2.0 | Consolidated: 2026-03-31
> **Status**: PoC failed (mode collapse on Type 3 + Gram loss explosion). Type 2 MAMMAL re-attempt MoA approved (3/3 consensus: "Do NOT retry" was over-generalization).

---

## §1. Original Strategy [DEPRECATED]

> Source: `DIFIX_TRAINING_STRATEGY.md` (v1.0, 2026-03-12)
> 3-model deliberation (Claude/Gemini/GPT-4o) consensus

### 1.1 Overview

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

### 1.2 Training Strategy: 2.5-Stage Curriculum

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

#### Why This Order?

1. **Stage 1 first**: Type 3 has zero domain gap (same renderer). Model learns
   what GS-LRM artifacts look like without confounding factors.
2. **Stage 1.5 bridge**: Type 2 introduces viewpoint diversity. MAMMAL pseudo-GT
   is imperfect but provides geometric guidance for novel views.
3. **Stage 2 last**: Type 1 provides real GT but only at GT cameras. Fine-tuning
   on this prevents quality regression while keeping novel view generalization.

### 1.3 Data Pair Specification

#### Type 1: GT View Pairs (N-view render <-> GT RGB)

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

#### Type 2: Novel View Pairs (6-view novel render <-> MAMMAL pseudo-GT)

```
Input:  outputs/datasets/novel_view/mouse_m5t2/tier0_raw/{view}/NNNNN.png
Target: outputs/datasets/novel_view/mouse_m5t2/pseudo_gt/{view}/NNNNN.png

view = bottom, top, front_low, side_low
frame = 00000 ... 03599
```

- **Domain gap**: GS-LRM (Gaussian splatting) vs MAMMAL (mesh template with UV texture)
- **MAMMAL limitations**: Template fitting error, limited surface detail, texture quality
- **Pairs**: 3600 x 4 = 14,400

#### Type 3: View Ablation Pairs (N-view render <-> 6-view render)

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

#### Domain Gap Analysis

| Pair Type | Spatial Gap | Appearance Gap | Scale | Training Signal Quality |
|:---------:|:----------:|:--------------:|:-----:|:----------------------:|
| Type 1 | None (same camera) | Low (render vs photo) | 108K | Best (real GT) |
| Type 2 | Novel viewpoint | High (GS vs mesh) | 14.4K | Medium (pseudo-GT) |
| Type 3 | None (same camera) | Zero (same renderer) | 108K | Good (self-supervised) |

### 1.4 Inference Pipeline

#### Per-View DiFix Application

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

#### Integration with Existing Pipeline

```
[6-view GT input]
    → MVDiffusion (if E2E) or direct (GT input)
    → GS-LRM → 3D Gaussians
    → Render at novel cameras → raw renders (Tier 0)
    → DiFix 3D+ → cleaned renders (Tier 1+)
    → Evaluate: raw vs cleaned vs MAMMAL pseudo-GT
```

### 1.5 Folder Structure

#### Source Data (outputs/datasets/novel_view/)

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

#### DiFix Training Pairs (outputs/datasets/difix_pairs/)

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

### 1.6 Compute & Storage Estimates

#### Rendering

| Phase | GPU | Wall Time | Output |
|-------|:---:|:---------:|:------:|
| 6-view GS-LRM (3600 frames) | 1x A6000 | ~30 min | tier0_raw + gt_views + gt_rgb (8.3 GB) |
| MAMMAL pseudo-GT (3600 frames) | 1x A6000 | ~18 min | pseudo_gt (2.1 GB) |
| View ablation (5 x 3600 frames) | 3x A6000 | ~50 min | ablation_{1-5}view (15.6 GB) |
| **Total rendering** | | **~1.5 hr** | **~26 GB** |

#### DiFix Pair Building

| Phase | Resource | Time | Output |
|-------|:--------:|:----:|:------:|
| Symlink-based pair build | CPU | ~10 min | ~230K symlinks (~negligible disk) |
| Manifest generation | CPU | ~2 min | manifest.json |

#### DiFix Training (Estimated)

| Stage | Pairs | Epochs | Est. Time (1x A6000) |
|-------|:-----:|:------:|:--------------------:|
| Stage 1 (Type 3) | 108K | 20 | ~8 hr |
| Stage 1.5 (+Type 2) | 122K | 10 | ~5 hr |
| Stage 2 (+Type 1) | 230K | 5 | ~4 hr |
| **Total training** | | | **~17 hr** |

#### Storage Summary

| Component | Size |
|-----------|:----:|
| Raw renders (novel_view/) | ~26 GB |
| DiFix pairs (symlinks) | ~1 MB |
| DiFix model checkpoints | ~2 GB |
| **Total** | **~28 GB** |

---

## §2. Type 2 Re-attempt (260329)

> Source: `DIFIX_TYPE2_MAMMAL_PLAN.md` (2026-03-29)
> MoA Audit: "Do NOT retry" conclusion was over-generalization (3/3 consensus)

### 2.1 Previous Failure Analysis

| Item | Previous PoC (Failed) | This Plan |
|------|----------------|----------|
| Data | Type 3 only (zero gap) | **Type 2** (MAMMAL mesh pseudo-GT) |
| Loss | LPIPS + L2 + **Gram** (unstable) | LPIPS + L2 (Gram removed) |
| Steps | 2000 intended → 9500 bug | 2000 strict (early stop) |
| Target | GS-LRM 6v (same model) | **MAMMAL mesh render** (3D geometry) |
| Evaluation | 2D image quality only | **3D-aware** (novel view re-render) |

### Why Retry?

1. Previous failure was Type 3 (zero domain gap) → MLP learned identity function
2. Gram loss explosion → loss design problem (not fundamental limitation)
3. Training bug (9500 steps) → uncontrolled experiment
4. **Type 2 (MAMMAL pseudo-GT) untested** — real 3D geometry supervision signal

### 2.2 Data Pair Generation

#### Bottom View Camera

```python
# collect_dataset.py
NOVEL_VIEWS = {
    "bottom": {"elevation": -70.0, "azimuth": 0.0},
}
# radius=2.7, fx=fy=411.75 @ 512x512
```

#### Step 1: GS-LRM Novel View Renders (Input)

```bash
ssh gpu03
cd /home/joon/dev/FaceLift
conda activate facelift

# Check if already exists
ls outputs/datasets/novel_view/mouse_m5t2/tier0_raw/bottom/ 2>/dev/null | wc -l

# If < 3600, generate:
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase gslrm \
    --frame_range 0 3600 \
    --views bottom
# Expected: ~30min (6v checkpoint inference)
```

#### Step 2: MAMMAL Mesh Renders (Target)

```bash
# Requires mammal_stable env for pyrender
conda activate mammal_stable
PYOPENGL_PLATFORM=egl python -m mouse_extensions.scripts.novel_view.collect_dataset \
    --mode generate --phase mammal \
    --frame_range 0 3600 \
    --views bottom
# Expected: ~18min
```

#### Step 3: Build DiFix Pairs

```bash
conda activate facelift
python -m mouse_extensions.scripts.eval.build_difix_dataset \
    --types 2 \
    --views bottom \
    --use-symlinks
# Output: difix_pairs/type2_novel_view/{frame}_{view}/input.png + target.png
# Expected pairs: 3,600 (bottom only)
```

### 2.3 Training Plan

#### Stage 0: Zero-Shot Baseline

```bash
python -m mouse_extensions.scripts.eval.difix_zero_shot \
    --input_dir outputs/datasets/novel_view/mouse_m5t2/tier0_raw/bottom \
    --output_dir outputs/difix_zero_shot/type2_bottom \
    --n_samples 20
# 1-2 hours, pretrained DiFix weights
```

#### Stage 1: Type 2 Fine-Tuning

```bash
python -m mouse_extensions.scripts.eval.train_difix \
    --pairs_dir difix_pairs/type2_novel_view \
    --loss lpips+l2 \
    --max_steps 2000 \
    --eval_every 200 \
    --output_dir outputs/difix_training/type2_bottom_v1
# Gram loss removed (previous explosion cause)
```

#### Stage 2: Mixed Training (Optional)

Type 3 80% + Type 2 20% mix (curriculum Stage 1.5):
```bash
python -m mouse_extensions.scripts.eval.train_difix \
    --pairs_dir difix_pairs/type2_novel_view:difix_pairs/type3_view_ablation \
    --mix_ratio 0.2:0.8 \
    --loss lpips+l2 \
    --max_steps 5000
```

### 2.4 Evaluation Protocol

#### Style Transfer Risk Prevention (Gemini audit warning)

Type 2 risks learning GS-LRM look → MAMMAL look style transfer only.

**Evaluation approach**:
1. ❌ 2D image direct comparison (PSNR vs MAMMAL render) — rewards style transfer
2. ✅ **3D-aware evaluation**:
   - DiFix refined image → novel view consistency check
   - Multi-angle DiFix application → multi-view consistency measurement
   - FG-PSNR (foreground masked, GT view) — vs actual GT

#### Metrics

| Metric | Description | Target |
|--------|------|--------|
| FG-PSNR (GT view) | Foreground PSNR vs real GT | > 20.01 dB (α=0.3 baseline) |
| LPIPS (GT view) | Perceptual quality | < baseline |
| Spike count | Needle artifact quantification | Decrease |
| Visual | Side-by-side comparison | Improvement confirmed |

### 2.5 Resource Requirements

| Item | GPU | Time | Notes |
|------|:---:|:----:|------|
| GS-LRM novel renders | 1 | ~30min | One-time |
| MAMMAL mesh renders | 0 (CPU/EGL) | ~18min | One-time |
| DiFix zero-shot | 1 | ~2h | 20 samples |
| DiFix Type 2 training | 1 | ~4-6h | 2000 steps |
| Evaluation | 1 | ~1h | FG-PSNR + visual |

**Total: ~1 day (1 GPU)**

### 2.6 Risk Assessment

| Risk | Severity | Mitigation |
|------|:--------:|------------|
| Style transfer (MAMMAL look) | High | 3D-aware evaluation, GT view cross-validation |
| Mode collapse (recurrence) | Medium | Gram loss removed, 2000 step strict |
| Resolution mismatch (384 vs 512) | Low | Verify 512 before generation |
| MAMMAL fitting quality | Medium | Keyframes only (900 frames) |

---

## Related Documents

- ← [[../INDEX]] — Document hub
- ↔ [[PHASE2_NOVEL_VIEW_ROADMAP]] — Phase 2 roadmap (novel view enhancement pipeline)
- ↔ [[mesh_gs_pair_collection]] — Tier-based dataset pipeline
- ↔ [[../../mouse_extensions/docs/COORDINATE_SYSTEMS]] — Coordinate transforms
- ↔ [[STAGE1_REPLACEMENT_CANDIDATES]] — Stage 1 alternatives (DiFix 3D+ is a candidate)

---

*FaceLift | DiFix Strategy (Consolidated) | v2.0 | 2026-03-31*
*Merged from: DIFIX_TRAINING_STRATEGY.md (2026-03-12) + DIFIX_TYPE2_MAMMAL_PLAN.md (2026-03-29)*
