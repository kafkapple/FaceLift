# FaceLift: Hypothesis Testing Status & Experiment Roadmap

> Version 3.0 | 2026-03-12 | SSOT — 전체 가설 통합, 6v>5v 오류 수정
> **⚠️ This is the authoritative hypothesis document.** `RESEARCH_HYPOTHESES.md` → archived (이 문서로 통합)

---

## 1. Completed Hypothesis Tests

### Early Hypotheses (Completed)

| # | Hypothesis | Result | Reference |
|---|-----------|--------|-----------|
| **HP** | Preprocessing geometry settings (PP, translation norm, intrinsics) | ✅ Batch Uniform + PP=256 + fx=549 | [[../../datasets/PREPROCESSING_REGISTRY]] |
| **H0** | PoC: GS-LRM fine-tuning on mouse data | ✅ Train PSNR ~27 | — |
| **H1-orig** | Temporal split prevents data leakage | ✅ Confirmed | — |
| **H2-orig** | Data amount: diversity > repetition | ✅ M5t2 (2880×6ep) > M5t (1198×20ep) | — |
| **H3-orig** | E2E bottleneck = MVDiffusion | ✅ MVDiff always bottleneck (13-17 dB gap) | [[mvdiffusion_bottleneck_analysis]] |
| **H3-bis** | Same-test revalidation | ✅ GS-LRM 20.92 vs E2E 7.91 = -13.8 dB | — |

### Main Hypothesis Tests

| # | Hypothesis | Method | Result | Conclusion |
|---|-----------|--------|--------|-----------|
| **H1** | GS-LRM > PS (same input) | Tier A: 6v GT → fair metrics | **+7.13 dB** (6v, same camera M5) | **Confirmed** — superior backbone |
| **H2** | MVDiff training strategy can break E2E ceiling | Phase 3: E1 cosine, E2 resume, E3 pose | E1/E2/E3 all 7.9-8.4 PSNR_gt | **Rejected** — architectural limit |
| **H3** | Stage 2 improvement transfers to E2E | E5: alpha regularization (0.3) | Val -1.0 dB, E2E transfer 0% | **Rejected** — not the bottleneck |
| **H4** | Shallow pose conditioning helps MVDiff | E3: extrinsic camera + additive | E3 ≈ E2 within noise | **Rejected** — too shallow |
| **H5** | Full attention > sparse attention | cfgr: sparse=false | cfgr < baseline in all metrics | **Rejected** — sparse is better |
| **H6** | 6v GS-LRM improves E2E over 4v | P1: 6v GS-LRM + MVDiff E2E | 8.44 vs 8.20 (+0.24 dB) | **Marginal** — MVDiff quality limits gains |
| **H7** | More input views always better (GT) | View ablation 1v→6v | **6v (23.84) > 5v (22.16) — monotonic** | **Confirmed** — 6v optimal |
| **H8** | Fewer MVDiff views → better per-view quality | 3-view E2E + 17-paper lit review | 3v novel=16.26 (-7.7 vs 6v); min viable=4v | **Rejected** — fewer views ≠ better quality |
| **H9** | Dense feature temporal advantage over sparse | HY_Concat vs SP_RawPCA temporal metrics | SP_MAE (same sparse input) > HY_Concat on ALL temporal metrics | **Rejected** — temporal advantage = representation method, not dense input. See Obsidian [[260317_Core_Hypothesis_Dense_Temporal_Stability]] §9 |

## 2. Analysis Phase Findings

| # | Analysis | Key Finding | Confidence |
|---|---------|-------------|:----------:|
| **A1** | MVDiff quality diagnostic | Sil IoU=0.582, 86% of E2E loss in MVDiff. **Shape error dominant** | High |
| **A2** | Oracle MVDiff (GS-LRM sensitivity) | **Threshold effect**: v4,5 unused (0 dB), v3 = -5.2 dB cliff | High |
| **A3** | Camera mismatch (M5 vs fj5_ds2) | HFOV 50° vs 35° → pixel-wise cross-model comparison invalid | **Critical** |
| **A4** | **View ablation curve** | 6v > 5v > ... > 1v (monotonic). Diminishing returns after 4v. Sharp cliff 2v→1v | **Corrected** |

---

## 3. Complete View Ablation Results (Tier C) ⭐ NEW

> GS-LRM GT inputs → test set (360 frames × 5 views, fair_comparison.py)

| Views | PSNR_gt | IoU | PSNR_int | Cov | Δ from prev |
|:-----:|:-------:|:---:|:--------:|:---:|:-----------:|
| **1v** | 10.47 | 0.028 | 10.47 | 1.000 | - |
| **2v** | 15.95 | 0.858 | 17.91 | 0.963 | **+5.48** |
| **3v** | 18.56 | 0.899 | 19.54 | 0.985 | +2.61 |
| **4v** | 20.66 | 0.926 | 21.29 | 0.993 | +2.10 |
| **5v** | 22.16 | 0.942 | 22.56 | 0.997 | +1.50 |
| **6v** | **23.84** | **0.954** | **24.02** | **0.998** | **+1.68** |

### Key Observations

1. **6v is optimal** on test set (23.84 dB) — **monotonic increase confirmed** (1v→6v)
2. **Diminishing returns**: 1→2v (+5.48), 2→3v (+2.61), 3→4v (+2.10), 4→5v (+1.50), 5→6v (+1.68)
3. **No saturation**: Unlike prior R1 (non-uniform) results, uniform v2 shows consistent improvement
4. **Sharp cliff at 1v**: IoU 0.028 = model essentially fails with single view
5. **2v → viable**: IoU 0.858, PSNR 15.95 — already competitive with PS

### Bottleneck Structure (Updated 2026-03-12)

```
GS-LRM 6v GT:  23.84 dB, IoU=0.954    ← Upper bound (test set, fair eval)
GS-LRM 5v GT:  22.16 dB, IoU=0.942
GS-LRM 4v GT:  20.66 dB, IoU=0.926
                    │
                    │  MVDiff: -15.40 dB loss (from 6v baseline)
                    │    - Shape: IoU 0.954 → 0.495 (dominant, 86%)
                    │    - Color: PSNR_int 24.02 → ~18.6 (secondary, 14%)
                    ▼
P1 6v E2E:       8.44 dB, IoU=0.495    ← Best E2E
E2 4v E2E:       8.20 dB, IoU=0.521    ← Previous best E2E
```

---

## 4. E2E Results (all evaluated with fair_comparison.py)

| Experiment | GS-LRM views | PSNR_gt | IoU | Notes |
|-----------|:------------:|:-------:|:---:|-------|
| **P1: 6v E2E (E2 MVDiff)** | 6 | **8.44** | 0.495 | Best E2E overall |
| P1: 6v E2E (baseline MVDiff) | 6 | 8.11 | 0.501 | |
| P1: 6v E2E (E1 MVDiff) | 6 | 8.04 | 0.511 | |
| E2 resume 20K | 4 | 8.20 | 0.521 | Best 4v E2E |
| E1 cosine 20K | 4 | 7.90 | 0.528 | Best color |
| E3 pose 10K | 4 | 8.10 | 0.523 | Shallow pose cond. |
| Baseline 5K | 4 | 7.93 | 0.474 | Original |
| cfgr (full attn) | 4 | 7.75 | 0.491 | Worst |

**Key insight**: All E2E variants converge at 7.9-8.4 regardless of GS-LRM views or MVDiff training. MVDiff view quality is the hard ceiling.

---

## 5. Cross-Model Comparison Status

### Tier A: Fair Quantitative (same #input views, GT)

| Model | Data | Views | PSNR_gt | IoU | Status |
|-------|------|:-----:|:-------:|:---:|--------|
| **GS-LRM 6v GT** | **M5** | **6** | **23.84** | **0.954** | **Done** ⭐ |
| GS-LRM 5v GT | M5 | 5 | 22.16 | 0.942 | Done |
| GS-LRM 4v GT | M5 | 4 | 20.66 | 0.926 | Done |
| PS (fj5_ds2) | fj5_ds2 | 5+1 | 16.80 | 0.827 | Done (different camera) |
| **PS (M5)** | **M5** | **5+1** | **?** | **?** | **Training (joon)** |

> **⚠️ Camera mismatch**: GS-LRM 5v 22.16 vs PS 16.80 (+5.36 dB) comparison is across different camera spaces (M5 HFOV=50° vs fj5_ds2 HFOV≈35°). **Option A** (PS M5 training) will enable fair same-camera comparison.

### Tier B: E2E vs PS

> Currently INVALID for quantitative comparison due to camera mismatch. Will become valid after Option A completes.

### Tier C: Pipeline Bottleneck

| Stage | Input | PSNR_gt | IoU | Drop from 5v |
|-------|:-----:|:-------:|:---:|:------------:|
| **GS-LRM 6v GT** | **6 GT** | **23.84** | **0.954** | **baseline** |
| GS-LRM 5v GT | 5 GT | 22.16 | 0.942 | -1.68 |
| GS-LRM 4v GT | 4 GT | 20.66 | 0.926 | -3.18 |
| GS-LRM 3v GT | 3 GT | 18.56 | 0.899 | -5.28 |
| GS-LRM 2v GT | 2 GT | 15.95 | 0.858 | -7.89 |
| GS-LRM 1v GT | 1 GT | 10.47 | 0.028 | -13.37 |
| **P1 6v E2E** | **6 MVDiff** | **8.44** | **0.495** | **-15.40** |
| E2 4v E2E | 4 MVDiff | 8.20 | 0.521 | -15.64 |

---

## 6. Current & Next Experiments

### In Progress

| Location | Experiment | Status | ETA |
|----------|-----------|--------|-----|
| joon | **Option A**: PS M5 training (`m5_baseline_gs`, epoch 1/50) | Training | ~28h |

### Completed Today (2026-02-19)

| Experiment | Result |
|-----------|--------|
| GS-LRM 5v GT inference + fair eval | PSNR_gt=22.16, IoU=0.942 |
| GS-LRM 2v GT inference + fair eval | PSNR_gt=15.95, IoU=0.858 |
| GS-LRM 3v GT inference + fair eval | PSNR_gt=18.56, IoU=0.899 |

### Next Queue

| # | Experiment | Purpose | Depends on | Cost |
|---|-----------|---------|------------|------|
| **Q1** | **DA1 E2E eval** | Domain adaptation → E2E PSNR | DA1 fine-tune | 1h |
| **Q2** | **H3 E2E eval** | Pose conditioning → E2E effect | H3 training | 1h |
| **Q3** | **H_Split: 1:1:1 split** | Split ratio fairness test | DA1+H3 done | ~24h |
| **Q4** | **Cross-species Rat7M** | Multi-species generalization | H_Split | ~3 days |

### Fairness & Generalization Experiments (Priority Tier 2)

| # | Hypothesis | Approach | Impact | Cost | Ref |
|---|-----------|----------|:------:|------|-----|
| **H_Split** | 8:1:1 split biases FL | Run FL+PS on 1:1:1 M5t split | **High** (fairness) | ~24h | FL_vs_PS §2.5 |
| **H_Rat** | FL generalizes to rat | Rat7M preprocessing + FL fine-tune + PS comparison | **High** (narrative) | ~3 days | FL_vs_PS §2.6 |

### Architecture Changes (MVDiff improvement, Priority Tier 3)

> All Phase 3 training strategies exhausted (H2 rejected). Only architecture-level changes can break the 7.9-8.4 ceiling.

| # | Hypothesis | Approach | Impact | Cost | Ref |
|---|-----------|----------|:------:|------|-----|
| **P2** | Silhouette-guided generation | Add sil L1/BCE loss to MVDiff | **High** | 1-2 days | Shape error = 86% of gap |
| **P3** | Deep camera pose conditioning | Inject extrinsics into RMA layers | **High** | 1-2 days | E3 failed (too shallow) |
| **P4** | Virtual camera interpolation | Generate at 60° → warp to actual | **Medium** | 2-3 days | Satisfy RMA assumption |
| P5 | View consistency enhancement | Cross-view geometric loss | Medium | 2-3 days | Inter-view coherence |

**Recommended order**: P2 → P3 → P4 (see §8 for rationale)

---

## 7. E3: Shallow Pose Conditioning (Detailed Record)

> **Status**: Completed, **Rejected** (no improvement)

| Item | Detail |
|------|--------|
| **Setup** | Camera extrinsic matrices (3×4) → linear projection → added to CLIP embedding |
| **Architecture** | Additive injection at UNet cross-attention input level |
| **Checkpoint** | `mouse_M5t2_E3_pose/checkpoint-10000` |
| **Result** | PSNR_gt=8.10, IoU=0.523 (vs E2 baseline: 8.20, 0.521) |
| **Per-view** | No improvement in any individual view; v4,v5 still IoU < 0.45 |
| **Why failed** | RMA structurally assumes 60° uniform spacing. Additive pose signal cannot override hardcoded attention patterns |
| **Lesson** | Need **multiplicative** or **layer-internal** injection (modifying Q/K/V or positional encoding) |

---

## 8. Architecture Change Strategy (MVDiff Improvement)

### Why P2 (Silhouette) first?

- A1: Shape error = 86% of E2E bottleneck (IoU: 0.942 → 0.582)
- Minimal architecture change: add silhouette loss term to existing training
- Alternative: 2-stage predict-then-generate (more complex but cleaner)
- **Success criterion**: MVDiff Sil IoU > 0.7 → E2E PSNR_gt > 10

### Why P3 (Deep Pose) after P2?

- E3 lesson: Shallow (additive) = no effect
- RMA assumes uniform 60° → mouse cameras are NON-uniform
- Need: Camera-conditioned positional encoding, per-head distance weighting
- Requires modifying `mvdiffusion/models/unet_mv2d_condition.py`

### Why P4 (Virtual Camera) is alternative to P3?

- Instead of changing RMA, satisfy its assumption
- Generate at 60° uniform positions → warp/interpolate to actual
- Trade-off: Avoids RMA modification but adds post-processing

### Decision Matrix

| Factor | P2 (Silhouette) | P3 (Deep Pose) | P4 (Virtual Cam) |
|--------|:---------------:|:--------------:|:-----------------:|
| Attacks shape error | Direct | Indirect | Indirect |
| Code complexity | Low | Medium | High |
| Training time | ~1 day | ~1 day | ~2 days |
| Risk | Low | Medium | Medium |
| **Recommended** | **1st** | **2nd** | **3rd (or skip)** |

---

## 9. Execution Timeline

```
[DONE - 260219]
  ✅ View ablation 1v-6v complete
  ✅ 5v optimal discovery (22.16 > 21.02)

[IN PROGRESS]
  joon: PS M5 training (epoch 1/50, ~28 hours remaining)

[+28 hours] PS M5 training done
  → Fair eval → Tier A: GS-LRM 5v vs PS M5 (FAIR)
  → Tier B: E2E vs PS M5 (FAIR)

[Next] Architecture changes:
  P2 → Silhouette supervision (1-2 days)
  P3 → Deep pose conditioning (1-2 days)
  P4 → Virtual camera interpolation (if P2/P3 insufficient)
```

---

## Appendix: Checkpoint Reference

| Views | Checkpoint Path | Val PSNR | PSNR_fg (fair) |
|:-----:|----------------|:--------:|:--------------:|
| 1v | `base_uniform_v2_1view_v2/best_psnr.pt` | 11.08 | 10.47 |
| 2v | `base_uniform_v2_2view_v2/best_psnr.pt` | 17.75 | 15.95 |
| 3v | `base_uniform_v2_3view_v2/best_psnr.pt` | 20.01 | 18.56 |
| 4v | `base_uniform_v2_4view_v2/best_psnr.pt` | **21.71** | 20.66 |
| 5v | `base_uniform_v2_5view_v2/best_psnr.pt` | 23.02 | 22.16 |
| 6v | `base_uniform_v2_6view_v2/best_psnr.pt` | **24.49** | 23.84 |
| E0_1 | `M5t2_E0_1_facelift/best_psnr.pt` | **22.34** | — |

> **Note**: Val PSNR = checkpoint best validation PSNR. PSNR_fg = fair eval foreground-masked PSNR (test set).
> E0_1은 논문 원본 config (`E0_1_facelift.yaml`), 나머지는 `base_uniform_v2.yaml` 기반.
> E0_1 (22.34)과 4v (21.71)의 차이는 config 차이에 기인하며 직접 비교 불가.

All at: `/node_data/joon/checkpoints/FaceLift/gslrm/`

---

---

## MVDiffusion Architecture Improvements

> Consolidated from `mvdiff_improvement_roadmap.md` (2026-02-24)

### Problem: Non-Uniform Camera Placement

M5 camera azimuths: `[0°, 22.5°, 36°, 73°, 88°, 151°]` — highly non-uniform with a **208.6° gap** (151° → 360°). MVDiffusion (SD2.1-UnCLIP + Era3D RMA) assumes **60° uniform spacing** from Objaverse pretraining. Per-view analysis confirms angular distance correlates with quality degradation (View 5 at 151° = lowest fg_PSNR 6.77).

### E3 Failure Analysis: Why Shallow Pose Conditioning Failed

E3 attempted additive injection of camera extrinsics into MVDiff conditioning:

```
E3 approach:  pose_signal = MLP(flatten(R, T))
              conditioning = view_embed + pose_signal  (additive)
```

**Why it failed**: RMA's cross-view attention weights encode **structural** 60°-uniform relationships learned from Objaverse. An additive signal at the conditioning level cannot override attention patterns that are **internal** to the transformer layers. The information pathway is too shallow — pose data never reaches where spatial relationships are computed.

**Lesson**: Need **attention-level** intervention (modifying Q/K/V, positional encoding, or attention bias), not surface-level signal addition.

| | E3 (failed) | Proposed Deep Conditioning |
|--|:-----------:|:--------------------------:|
| Integration | Additive to conditioning | Attention bias / relative pose encoding |
| Scope modified | Conditioning MLP only | **Cross-view attention structure** |
| RMA 60° assumption | Preserved | **Removed** (pose-dependent) |
| Pretrained compatibility | Compatible | Incompatible (structural change) |

### Strategy A: Camera Pose Conditioning (Architecture Modification)

**Goal**: Make MVDiff work with arbitrary camera placement by modifying RMA.

**Approach A — Pose-Aware Attention Bias** (minimal modification):

```python
# Current RMA: attention based on learned view-index weights
attention = softmax(Q @ K^T / sqrt(d))

# Proposed: explicit geometric bias from relative camera poses
relative_pose = compute_relative_pose(cam_i, cam_j)
pose_bias = pose_mlp(relative_pose)       # [N_views, N_views, N_heads]
attention = softmax(Q @ K^T / sqrt(d) + pose_bias)
```

**Approach B — Sinusoidal Camera Encoding** (NeRF-style): Replace discrete view-index embeddings with continuous positional encoding of camera extrinsics (R, T → 12-dim → frequency encoding → MLP).

**Approach C — Epipolar Attention** (full replacement): Replace RMA entirely with epipolar-line-based attention. Most fundamental but highest implementation cost.

**Related work**: Zero123++ (relative camera transforms), SV3D (camera trajectory conditioning), MVDream (explicit camera control).

### Strategy B: Virtual Camera Interpolation (Data Transformation)

**Goal**: Satisfy RMA's 60° assumption by generating uniform-view training data.

**Pipeline**:
1. GS-LRM GT 6v → 3D Gaussian reconstruction
2. Render from uniform virtual cameras: `[0°, 60°, 120°, 180°, 240°, 300°]`
3. Retrain MVDiffusion on uniform-view data (pretrained weights reusable)
4. Retrain GS-LRM for uniform-view input

**Critical constraint**: The 208.6° gap renders quality at ~256° (gap center) uncertain — all input views are far from this region. GS-LRM GT 6v achieves ~24 dB near input views but ~16.6 dB at far views. Poor virtual-view quality would propagate errors into MVDiff training.

### Comparison Matrix

| Criterion | Pose Conditioning | Virtual Camera | E3 (failed) |
|-----------|:-----------------:|:--------------:|:-----------:|
| Architecture change | **Major** | None | None |
| Pretrained reuse | Limited | **Yes** | Yes |
| Implementation effort | High | **Medium** | Low |
| Expected impact | **High** | Medium | None |
| Generalization | **High** | Low | None |
| Paper contribution | **High** | Medium | — |
| Timeline | Weeks | **Days** | Done |

**Recommended path**: Virtual Camera (short-term feasibility test) → Camera Pose Conditioning (long-term fundamental solution). These correspond to P4 and P3 in the experiment queue (§6).

---

*FaceLift Hypothesis Roadmap v3.0 | 2026-03-12 | 6v>5v 수정, 조기 가설 통합, SSOT 확립*
