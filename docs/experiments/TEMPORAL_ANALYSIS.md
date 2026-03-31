# Temporal Consistency Analysis

> [← INDEX](../INDEX.md) | [DEFORMATION_ROADMAP](DEFORMATION_ROADMAP.md)
> **Version**: v1.0 | Consolidated: 2026-03-31

---

## §1. Evaluation Standard (Reference Spec)

> Source: `TEMPORAL_EVAL_STANDARD.md` (2026-03-21)
> All temporal consistency experiments MUST follow this standard for reproducibility and fair comparison.

### 1.1 Frame Convention

#### Sparse Sampling (per-frame quality + static novel view comparison)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frame IDs** | `3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3599` | 10 frames from test split, ~40-frame spacing |
| **Split** | Test (3240-3599) | Never seen during training |
| **Count** | 10 | Sufficient for statistical metrics |

#### Dense Sequence (temporal consistency analysis)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Frame Range** | `3300-3320` | 21 consecutive frames within test split |
| **Spacing** | 1 frame (no skip) | Required for temporal metrics (tOF, TLPIPS) |
| **Purpose** | Temporal flickering, smoothing comparison | Consecutive frames reveal jitter |

#### Why These Frames?

- **Test split only** (3240-3599): ensures evaluation integrity
- **3300-3320 dense region**: overlaps with sparse samples at 3300/3320 for cross-validation
- **Previous renders**: baseline/alpha03 used 3300-3580 (sparse), alpha05/alpha10 used 0-2000 (training set) — **non-comparable, deprecated**

### 1.2 Novel Views

| View | Elevation | Azimuth | Purpose |
|------|-----------|---------|---------|
| `bottom` | -70° | 0° | Ventral surface, alpha artifact sensitive |
| `top` | +70° | 0° | Dorsal surface, body shape |
| `front_low` | -30° | 0° | Face/head detail |
| `side_low` | -30° | 90° | Profile, tail/limb articulation |

**Resolution**: 512×512 (matches GS-LRM training resolution)
**Render radius**: 2.7 (FaceLift pretrained standard)

### 1.3 Experiment Settings

| ID | Label | Views | α weight | Checkpoint |
|----|-------|:-----:|:--------:|------------|
| `baseline_6v` | Baseline (α=0) | 6 | 0.0 | `uniform_v2/6view/best_psnr.pt` |
| `alpha03` | α=0.3 | 4 | 0.3 | `uniform_v2/4view_alpha03_v3/best_psnr.pt` |
| `alpha05` | α=0.5 | 4 | 0.5 | `uniform_v2/4view_alpha05_v3/best_psnr.pt` |
| `alpha10` | α=1.0 | 4 | 1.0 | `uniform_v2/4view_alpha10_v3/best_psnr.pt` |

**Checkpoint root**: `/node_data/joon/checkpoints/FaceLift/gslrm/`

### 1.4 Temporal Smoothing Methods

| Method | Type | Parameters | Implementation |
|--------|------|------------|----------------|
| **Original** | None | — | Raw GS-LRM output |
| **EMA** | 2D post-process | α ∈ {0.1, 0.3, 0.5, 0.7} | `evaluation/temporal_smoothing.py` |
| **DeformV2** | 3D Gaussian | blend ∈ {0.3, 0.5, 0.7} | `model/deformation/temporal_deform_inference.py` |
| **NN-Match** | 3D Gaussian | k=1, interp=lerp/slerp | `scripts/eval/temporal_comparison.py` |
| **OptFlow** | 2D post-process | Farneback, blend α | `evaluation/temporal_smoothing.py` |

### 1.5 Metrics

#### Metric Protocol (IMPORTANT)

> **Object-centric evaluation standard: Masked Foreground metrics.**
> Full-image (white BG) PSNR inflated by 95% perfect background. Paper uses as reference only.
> Reference: 3DGS (Kerbl 2023), GS-LRM (Zhang 2024), One-2-3-45 all adopt masked foreground.

#### Per-Frame Quality (higher is better unless noted)

| Metric | Scope | Formula | Tool |
|--------|-------|---------|------|
| **PSNR_gt_masked** | **FG only** (primary) | PSNR on GT mask pixels | `eval/fair_comparison.py` |
| **SSIM_gt_masked** | **FG bbox** (primary) | SSIM on GT mask bbox+10px | `eval/fair_comparison.py` |
| **LPIPS** | FG region | AlexNet perceptual | `evaluation/metrics.py` |
| **IoU** | Silhouette | mask intersection/union | `eval/fair_comparison.py` |
| *PSNR_full* | *Full image (reference)* | *20·log₁₀(1/√MSE)* | *evaluation/metrics.py* |

#### Temporal Stability (lower is better)

| Metric | Formula | Tool |
|--------|---------|------|
| **tOF** | std(‖OF(t,t+1)‖) over sequence | `evaluation/temporal_smoothing.py` |
| **TLPIPS** | mean(LPIPS(t, t+1)) over sequence | `evaluation/temporal_smoothing.py` |
| **FF-SSIM-var** | var(SSIM(t, t+1)) over sequence | `evaluation/temporal_smoothing.py` |
| **Flicker Rate** | frames with ΔI > threshold / total | `evaluation/temporal_smoothing.py` |

#### Gaussian-Level (from dense sequence, lower is better)

| Metric | Formula | Tool |
|--------|---------|------|
| **Position Jitter** | mean(‖Δxyz_matched‖) | `evaluation/temporal_metrics.py` |
| **Scale Variance** | var(scale_matched) over time | `evaluation/temporal_metrics.py` |
| **Persistence Ratio** | matched_N / total_N (higher=better) | `scripts/eval/temporal_comparison.py` |

### 1.6 Output Directories

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

### 1.7 Grid Visualization Convention

#### Per-View Comparison
- **Layout**: 1 row × N columns (one per experiment/method)
- **Labels**: Top of each column, font ≥20pt
- **Frame ID**: Bottom-right corner overlay

#### Combined Grid
- **Rows**: Views (bottom, top, front_low, side_low)
- **Columns**: Experiments or methods
- **Row labels**: Left side
- **Column labels**: Top

#### Temporal Strip
- **Rows**: Methods (Original, EMA, Deform, etc.)
- **Columns**: Consecutive frames (3300, 3301, ..., 3310)
- **Purpose**: Visual temporal smoothness comparison

### Quick Reference

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

## §2. Consistency Study (Analysis & Results)

> Source: `TEMPORAL_CONSISTENCY_STUDY.md` (2026-03-21)
> Phase 1 complete, Phase 2-3 planned

### 2.1 Background

GS-LRM produces per-frame independent Gaussian sets, causing temporal flickering (frame-to-frame jitter) in novel view rendering. The original FaceLift paper mitigated this for faces with an 8-layer MLP deformation network, but our mouse scenario faces:
- Large non-rigid motion (running, grooming, rearing)
- No per-Gaussian identity guarantee (count/order varies per frame)
- Deformation checkpoint exists but comparison renders not yet performed

### 2.2 Phase 1 Results: 2D Post-Processing

#### Methods Tested

| Method | Description | Parameters |
|--------|-------------|------------|
| **Original** | Raw GS-LRM output | — |
| **EMA** | Exponential Moving Average on rendered images | α ∈ {0.1, 0.3, 0.5} |
| **OptFlow** | Farneback optical flow warping + blending | α ∈ {0.3, 0.5} |

#### tOF Results (mean across 4 views, lower = more stable)

| Method | Baseline | α=0.3 | α=0.5 | α=1.0 | Reduction |
|--------|:--------:|:-----:|:-----:|:-----:|:---------:|
| Original | 0.2052 | 0.1972 | 0.1956 | 0.1967 | — |
| **EMA α=0.1** | **0.0230** | **0.0251** | **0.0250** | **0.0250** | **~87%** |
| EMA α=0.3 | 0.0634 | 0.0700 | 0.0701 | 0.0702 | ~65% |
| EMA α=0.5 | 0.1336 | 0.1385 | 0.1387 | 0.1378 | ~30% |
| OptFlow α=0.3 | 0.1835 | 0.1802 | 0.1790 | 0.1789 | ~9% |
| OptFlow α=0.5 | 0.1927 | 0.1977 | 0.1962 | 0.1970 | ~0% |

#### Key Findings

1. **EMA α=0.1 dominant** — tOF 87% reduction. However, motion blur trade-off
2. **OptFlow ineffective** — Farneback flow warping barely reduces flickering (flow estimation itself affected by jitter)
3. **Alpha loss weight has no effect on temporal stability** — all 4 settings show similar tOF (±3%)
4. **Bottom view most unstable** — tOF=0.31 (side_low=0.25, front_low=0.15, top=0.10)
   - Ventral surface: legs, belly exposed → more flickering

#### Per-View tOF (Original, Baseline)

| View | tOF | Characteristics |
|------|:---:|------|
| bottom | 0.3143 | Ventral, legs/belly exposed |
| side_low | 0.2540 | Profile, tail/legs |
| front_low | 0.1478 | Face/head |
| top | 0.1049 | Dorsal, relatively stable |

### 2.3 Phase 1.5 Results: Window-Based Methods (2026-03-21)

Window-based methods tested to address EMA motion blur.

| Method | tOF↓ | Reduction | Motion Blur | Notes |
|--------|:----:|:------:|:-----------:|------|
| **Median w=5** | 0.139 | 56% | Minimal | Best balance |
| SavGol w=5 | 0.221 | 30% | Very Low | Edge preservation |
| EMA α=0.3 | 0.080 | 75% | Medium | |
| Bilateral w=5 σ=25 | 0.307 | 2% | None | ❌ Ineffective |

#### Visual Evaluation Conclusion

> **⚠️ All 2D temporal smoothing methods degrade visual quality vs original.**
> - Median/SavGol: flickering reduced but detail loss, "smeared" feel
> - EMA: motion blur ghosting
> - Bilateral: negligible effect
> - **Conclusion: 2D post-processing cannot improve temporal consistency. Keep originals.**
> - Fundamental solution requires 3D-level (deformation, scene flow, or architecture change).

**Status: On hold**. 2D temporal smoothing exploration terminated. Using original renders.

### 2.4 Phase 2: DeformationV2 (Planned)

#### Module Status

| Component | Path | Status |
|-----------|------|--------|
| DeformationNetworkV2 | `mouse_extensions/model/deformation/deformation_network.py` | ✅ Implemented |
| TemporalDeformInference | `model/deformation/temporal_deform_inference.py` | ✅ Implemented |
| Checkpoint | `/node_data/joon/checkpoints/FaceLift/deformation/default/checkpoint_010000.pt` | ✅ Exists |
| Gaussian NPZ cache | `outputs/datasets/temporal_eval/*/gaussians/` | ❌ Not generated |

#### Execution Plan

1. **Gaussian NPZ extraction**: modify `collect_dataset.py` → save per-frame Gaussian params
2. **NN Matching**: `cKDTree(xyz_t).query(xyz_t1)` → resolve N mismatch
3. **Deformation application**: `TemporalDeformInference.process_sequence()`, blend_alpha sweep
4. **Re-rendering**: deformed Gaussians → novel view renders → tOF comparison

#### FaceLift Appendix 3.5 — Deformation Module Analysis

##### GS-LRM Fundamental Structure

GS-LRM is a **per-pixel Gaussian** generation model:
- Each pixel position in input image outputs a 14-param Gaussian (xyz, rotation, scale, opacity, SH)
- Each frame produces Gaussian set **independently** → **no Gaussian identity**
- Count and order differ per frame → fundamental difficulty for scene flow computation

##### Paper's Canonical Deformation Approach

```
[Canonical Frame]
     ↓
  GS-LRM → Canonical Gaussians (fixed set)
     ↓
[Target Frame t]
     ↓
  8-layer MLP(canonical_xyz, target_features) → Δxyz, Δopacity, Δscale
     ↓
  Deformed Gaussians = Canonical + Δ  (tracking temporal deformation of same Gaussians)
```

Key: **predicts deformation from canonical Gaussians** so Gaussian identity is maintained.
- Loss: photometric (render vs GT) + ARAP (local rigidity preservation)
- Effective for faces: consistent topology, small motion → single canonical frame sufficient

##### Limitations for Mouse (3-model deliberation result)

| Problem | Description |
|------|------|
| **Fixed Topology** | Canonical frame lacks Gaussians for occluded parts (belly, inner legs) → cannot represent when exposed |
| **Large Motion** | Running/grooming/rearing → deformation too far from canonical → MLP capacity exceeded |
| **Self-Occlusion** | Tail wrapping, leg folding → topology changes, Gaussian birth/death needed but canonical is fixed |
| **Feed-Forward Constraint** | Must work feed-forward without per-scene optimization → generalization difficulty |

##### Alternative Approaches (Literature Survey)

| Method | Core | Mouse Suitability |
|------|------|:-----------:|
| **Multi-Canonical** | Fuse Gaussians from 5-10 keyframes → wider coverage | ⭐⭐⭐ |
| **Sliding-Window Canonical** | Adjacent frame canonical → handles large motion | ⭐⭐⭐ |
| **4DGS (per-scene optim)** | Whole sequence optimization for canonical + deform | ⭐⭐ (not feed-forward) |
| **SC-GS** | Sparse control points + implicit deformation | ⭐⭐ |
| **Recurrent Gaussian Gen** | Current frame references previous frame Gaussians | ⭐⭐⭐ (future direction) |

##### Minimum Viable Experiment

1. **Select neutral pose canonical** — standing pose, maximum body part exposure frame
2. **Apply deformation with existing checkpoint** (10K steps, blend_alpha=0.3-0.7)
3. **NN matching** for N mismatch: `cKDTree(canonical_xyz).query(target_xyz, k=1)`
4. Render comparison: original vs deformed (tOF + visual)
5. **Expected**: partial improvement in small motion segments, stretching artifacts in large motion

#### Feasibility Assessment (from /deliberate, updated 2026-03-21)

| Factor | Face (paper) | Mouse (ours) | Mitigation |
|--------|:-----------:|:-----------:|------|
| Topology consistency | High | Low | Multi-canonical needed |
| Inter-frame motion | Small | Large | Sliding window or recurrent |
| Self-occlusion | Rare | Frequent | Gaussian birth/death needed |
| Feed-forward constraint | N/A (per-scene) | Required | Generalized canonical learning |
| Expected benefit | High | **Low-Medium** | Verify with minimum experiment |

**Conclusion**: Single canonical approach has fundamental limitations for mouse. Multi-canonical or recurrent approach needed, but under current NeurIPS timeline **DiFix (artifact removal) prioritized**, deformation classified as future work.

### 2.5 Phase 3: Advanced Methods (Survey)

| Method | Type | Effort | Expected Effect | Status |
|--------|------|:------:|:---------------:|--------|
| 3D Scene Flow | 3D | Very High | High | Not implemented, next phase |
| Test-Time Optim | 3D | High | High | Not implemented |
| Neural Temporal Embed | 3D | Very High | Highest | Architecture change required |
| NN-Match + Interp | 3D | Medium | Medium | With Phase 2 |

#### Optical Flow Limitation Analysis

Why Farneback optical flow was ineffective:
- **Flow estimation itself contaminated by jitter**: GS-LRM flickering → noisy flow → noisy warp
- **Occlusion/disocclusion**: frequent in mouse motion → warp artifacts
- **3D motion → 2D projection**: parallax in novel views not capturable by 2D flow

**Alternative**: RAFT (learned) optical flow may be more robust than Farneback, but core limitation (3D→2D) remains.

### 2.6 Output Locations

| Type | Path |
|------|------|
| Unified renders | `outputs/datasets/temporal_eval/{experiment}/` |
| Temporal strips | `outputs/report/temporal_comparison/strips/` |
| Method grids | `outputs/report/temporal_comparison/grids/` |
| Metrics JSON | `outputs/report/temporal_comparison/metrics/temporal_comparison.json` |
| Comparison table | `outputs/report/temporal_comparison/metrics/comparison_table.md` |

### 2.7 Conclusion & Next Steps

1. ~~EMA α=0.1 is simple and effective baseline~~ → **Visual evaluation shows quality degradation, keep originals**
2. **Alpha loss has no effect on temporal stability** — alpha loss contribution limited to novel view artifact reduction
3. **3D-level smoothing (DeformV2, Scene Flow) needed** — 2D post-processing limitations confirmed
4. **Bottom view is key challenge** — most flickering, future improvement target
5. **Deformation module review needed** — canonical Gaussian approach applicability for mouse requires analysis

---

## Related

- ↑ [[INDEX]] (Document hub)
- ↔ [[DEFORMATION_ROADMAP]] (Deformation V2→V3 strategy)
- ↔ [[COMMANDS]] (Experiment commands)
- ↓ `mouse_extensions/evaluation/temporal_smoothing.py` (Implementation)
- ↓ `mouse_extensions/scripts/eval/temporal_comparison.py` (Comparison pipeline)

---

*FaceLift | Temporal Consistency Analysis (Consolidated) | v1.0 | 2026-03-31*
*Merged from: TEMPORAL_EVAL_STANDARD.md (2026-03-21) + TEMPORAL_CONSISTENCY_STUDY.md (2026-03-21)*
