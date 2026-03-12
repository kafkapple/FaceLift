# MVDiffusion Bottleneck Analysis

> **Terminology Note**: This document uses 'MVDiff' as shorthand for 'multi-view diffusion' (FaceLift Stage 1).
> The Stage 1 model is **SD2.1-UnCLIP + Era3D RMA**, NOT the 'MVDiffusion' paper by Tang et al.
> Code folder `mvdiffusion/` is an inherited name from the Era3D codebase.


> SSOT for MVDiffusion experiments, transfer rate analysis, and improvement hypotheses.
> Created: 2026-02-22 | Updated: 2026-02-22 | Version: v1.1

## 1. Executive Summary

**Problem**: GS-LRM achieves **23.84 dB** PSNR_fg with GT 6-view input, but E2E (MVDiff→GS-LRM) only achieves 8.44 dB. This **15.4 dB gap** is the critical bottleneck.

**Root Cause**: Not MVDiffusion quality alone (val PSNR improved from 24→27.7 dB with minimal E2E gain), but the **distribution mismatch** between GT views (GS-LRM training) and MVDiff-generated views (E2E inference).

**Transfer Rate**: ~14% — a 3.7 dB MVDiff improvement yields only 0.5 dB E2E improvement.

---

## 2. Complete Experiment Matrix

### 2.1 E2E Pipeline Results (Fair Eval, Test-Only)

| # | Experiment | MVDiff Changes | MVDiff Val PSNR | E2E PSNR_fg | E2E IoU | E2E Coverage |
|---|-----------|---------------|:---:|:---:|:---:|:---:|
| 0 | baseline_360f | — | ~24.0 | 7.93 | 0.474 | — |
| 1 | cfgr | full attention (no sparse) | ? | 7.75 | 0.491 | — |
| 2 | E1 (cosine 20K) | cosine LR, 20K steps | 26.93 | 7.90-8.04 | 0.528 | — |
| 3 | **E2 (resume 20K)** | randref + LR decay, 20K | **27.70** | **8.44** | 0.495 | 0.750 |
| 4 | E3 (pose) | extrinsic pose + cosine | ? | 8.10 | 0.523 | — |
| 5 | P1 (E2 pipeline) | E2 ckpt in full pipeline | **8.44** | 0.495 | 0.750 |
| 6 | P1_BL (baseline) | baseline ckpt-5000 | 8.11 | 0.501 | 0.740 |
| 7 | P1_E1 (cosine) | E1 ckpt in pipeline | 8.04 | 0.511 | 0.728 |

**Key Observation**: All E2E variants fall in 7.75–8.44 PSNR_fg range (0.69 dB total spread).

### 2.2 GS-LRM Upper Bound (GT Input)

| Views | PSNR_fg | IoU | Notes |
|:-----:|:---:|:---:|-------|
| 1v | 10.47 | 0.028 | Degenerate |
| 2v | 15.95 | 0.858 | Phase transition |
| 3v | 18.56 | 0.899 | |
| 4v | 20.66 | 0.926 | |
| 5v | 22.16 | 0.942 | |
| 6v | 23.84 | 0.954 | Upper bound |

---

## 3. What Was Tried (11 Configs, 9 Checkpoints)

### Architecture
| Approach | Result | Verdict |
|----------|--------|---------|
| Sparse vs full attention (cfgr) | Full attention worse (-0.18 dB) | Sparse better |
| Random reference view (P0→E2) | **Best MVDiff** (+3.7 dB val) | Significant |
| Pose conditioning (E3) | Marginal (+0.17 dB E2E) | Minor |
| 3-view / 4-view generation | Untested as E2E | — |

### Training Strategy
| Approach | Result | Verdict |
|----------|--------|---------|
| Cosine LR (E1) | 26.93 val PSNR, 8.04 E2E | Moderate |
| Resume + LR decay (E2) | **27.70 val PSNR, 8.44 E2E** | Best so far |
| Longer training (20K vs 10K) | Improves val, minimal E2E | Diminishing returns |
| No augmentation | Untested as E2E | — |

---

## 4. Transfer Rate Analysis

```
MVDiff improvement (val PSNR): +3.7 dB (24.0 → 27.70)
E2E improvement (PSNR_fg):     +0.51 dB (7.93 → 8.44)
Transfer rate:                  0.51 / 3.7 = 13.8%
```

**Why so low?** Three hypotheses:

### H_T1: Distribution Mismatch (Primary)
GS-LRM trained on GT views → expects clean, geometrically consistent multi-view images. MVDiff generates views with:
- View-inconsistent fine details
- Slight geometric errors (pose drift)
- Different noise characteristics

**Evidence**: GT 4v gives 20.66, MVDiff 6v gives 8.44. Even MVDiff's "6 views" are less useful than GT's 4 views.

### H_T2: Coverage Bottleneck
E2E coverage ~75% vs GT ~100%. Generated views don't fully cover the object, leading to incomplete Gaussians.

**Evidence**: P1_E1 has best IoU (0.511) but lowest PSNR_fg (8.04), suggesting coverage-quality tradeoff.

### H_T3: Geometric Inconsistency
Multi-view consistency of MVDiff is insufficient for GS-LRM's implicit multi-view stereo.

**Evidence**: Sparse attention (designed for cross-view consistency) outperforms full attention.

---

## 5. Hypothesis → Experiment Mapping

### 5.1 Three Transfer Rate Hypotheses

| ID | Hypothesis | Evidence | Priority | Experiment |
|:--:|-----------|----------|:--------:|-----------|
| **H_T1** | **Distribution Mismatch** — GS-LRM expects GT-style inputs but receives MVDiff-style | GT 4v (20.66) >> MVDiff 6v (8.44); +3.7dB MVDiff → only +0.51dB E2E | **P0** | **DA1** (domain adaptation) |
| H_T2 | Coverage Bottleneck — MVDiff views don't fully cover object (~75%) | P1_E1: best IoU (0.511) but lowest PSNR_fg (8.04) | P2 | Implicit in DA1 eval |
| H_T3 | Geometric Inconsistency — MVDiff multi-view consistency insufficient | Sparse attn > full attn for cross-view consistency | P2 | H3 (pose conditioning) |

### 5.2 Active Experiments

#### DA1: GS-LRM Domain Adaptation ★ PRIMARY (Tests H_T1) — CONFIRMED ✅

**Rationale**: Instead of improving MVDiff quality (diminishing returns at 14%), adapt GS-LRM to handle MVDiff-style inputs directly.

**Result**: E2E PSNR_fg = **10.08** (+1.64 dB over baseline 8.44). **H_T1 confirmed.**

**Pipeline**:
| Step | Task | Status | GPU |
|:----:|------|:------:|:---:|
| 1 | Generate MVDiff E2 views for 2880 train frames | ✅ Complete | 6 |
| 2 | Auto-launch GS-LRM fine-tune (`launch_da1_finetune.sh`) | ✅ Complete | 6 |
| 3 | Fine-tune GS-LRM (lr=1e-6, 7500 steps, resume from GT best) | ✅ Complete | 6 |
| 4 | E2E evaluation with DA1 checkpoint | ✅ Complete | 6 |

**Config**: `configs/mouse/uniform/domain_adapt_E2_v1.yaml`
**Detailed docs**: `docs/experiments/domain_adaptation_DA1.md`

**Implication**: Domain adaptation is **3.2× more efficient** than Stage 1 optimization (DA1 +1.64 vs Stage 1 best +0.51 per unit effort). Next: DA2 (mixed GT+MVDiff training) to prevent catastrophic forgetting.

#### H3: E2 + Pose Conditioning (Tests H_T3, negative control for H_T1)

**Rationale**: If MVDiff quality improvement (E2) combined with pose conditioning (E3) still barely moves E2E, this further confirms that the bottleneck is NOT in MVDiff quality (supporting H_T1).

- **Config**: `mouse_mvdiffusion_M5t2_H3_resume_pose.yaml`
- **Status**: GPU 5, training early phase / 10K steps
- **ETA**: ~28h
- **Expected**: Marginal E2E improvement (0.1-0.3 dB), serving as negative control

### 5.3 Untested Approaches (Future, ordered by priority)

| Priority | Approach | Tests | Rationale |
|:---:|---------|:-----:|-----------|
| P1 | DA2: Mixed training (GT + MVDiff 50/50) | H_T1 | Prevent catastrophic forgetting while adapting |
| P2 | Guidance scale sweep (inference-only) | — | Optimal gs for E2E may differ from image quality |
| P2 | DA3: Progressive domain adaptation | H_T1 | Gradual GT→MVDiff transition |
| P3 | Joint MVDiff + GS-LRM fine-tuning | H_T1+T3 | End-to-end gradient flow |
| P3 | Cosine + randref combined (new config) | — | E1 + E2 strengths |
| P4 | Different diffusion scheduler (DDIM) | — | 50 steps may not be optimal |

---

## 6. Checkpoints & Paths

### MVDiffusion Checkpoints (gpu03)
| Experiment | Path | Size |
|-----------|------|:---:|
| Baseline | `mouse_M5t2/checkpoint-5000` | 14G |
| E2 (best) | `mouse_M5t2_randref_sparse/checkpoint-20000` | 14G |
| E1 | `mouse_M5t2_20k_cosine/checkpoint-20000` | 14G |
| E3 | `mouse_M5t2_pose_extrinsic_add/checkpoint-10000` | 14G |
| H3 | `mouse_M5t2_H3_resume_pose/` (training) | — |

### Fair Eval JSONs
| File | Experiment |
|------|-----------|
| `baseline_360f_fair.json` | Original baseline |
| `e1_cosine_20k_fair.json` | E1 cosine LR |
| `e2_resume_20k_fair.json` | E2 resume (current best) |
| `e3_pose_fair.json` | E3 pose conditioning |
| `p1_6view_e2e_fair.json` | P1 pipeline (E2 ckpt) |
| `p1_bl_6view_e2e_fair.json` | P1 pipeline (baseline) |
| `p1_e1_6view_e2e_fair.json` | P1 pipeline (E1 ckpt) |

All in: `gpu03:experiments/comparison/tier/`

---

## 7. All Experiments Status (as of 2026-02-22)

### Running

| GPU | Experiment | Category | Progress | ETA |
|:---:|-----------|----------|:--------:|:---:|
| gpu03:4 | HP M5_5 (center+norm) | GS-LRM preprocessing ablation | Training, best_psnr.pt saved | ~12h |
| gpu03:5 | H3 MVDiff (E2+Pose) | MVDiff improvement (H_T3 test) | Training early/10K steps | ~28h |
| gpu03:6 | DA1 datagen | Domain adaptation (H_T1 test) | 218/2880 frames | ~3.5h |
| gpu03:7 | HP M5_4 (center only) | GS-LRM preprocessing ablation | step 6000+, best_psnr.pt | ~8h |
| joon:0 | PS M5 5v (holdout4) | Pose-Splatter comparison | epoch 7, training | ~18h |

### Queued

| Experiment | Trigger | GPU |
|-----------|---------|:---:|
| DA1 GS-LRM fine-tune | DA1 datagen complete → auto-launch | gpu03:6 |
| PS M5 4v | PS M5 5v complete → manual launch | joon:0 |
| DA1 E2E eval | DA1 fine-tune complete → manual | gpu03:6 |
| H3 E2E eval | H3 training complete → manual | gpu03:5 |

### Completed

| Experiment | Result | Conclusion |
|-----------|--------|-----------|
| FL GS-LRM 1-6v | 10.47–23.84 PSNR_fg | View ablation done, 2v=phase transition |
| FL E2E (BL/E1/E2/E3/P1) | 7.75–8.44 PSNR_fg | E2 best, all within 0.69 dB |
| PS M5 6v | 13.78 PSNR_fg | FL 6v >> PS by +9.62 dB |
| H6 Alpha mask (0.3/0.5/1.0) | Monotonic decline | Rejected: baseline optimal |
| H7 SSIM weight (0.3/0.5/1.0) | 0.5/1.0 collapse | Rejected: 0.1 optimal |
| HP M0 (raw data) | Diverged | Preprocessing required |

---

## 8. Key Conclusions

1. **MVDiff quality improvement has diminishing returns for E2E** (~14% transfer rate)
2. **Distribution adaptation (DA1) is the highest-priority experiment** — directly tests H_T1
3. **H3 serves as negative control** — if E2E barely improves, confirms H_T1 over H_T3
4. **E2 (randref + resume) is the best MVDiff checkpoint** (val 27.70, E2E 8.44)
5. **Sparse attention > full attention** for multi-view consistency
6. **Random reference view was the single most impactful change** for MVDiff quality

---

## 9. Document Cross-References

| Document | Path | Content |
|----------|------|---------|
| This document | `docs/experiments/mvdiffusion_bottleneck_analysis.md` | Transfer rate, hypotheses, experiment matrix |
| DA1 detailed | `docs/experiments/domain_adaptation_DA1.md` | DA1 pipeline, configs, success criteria |
| Eval protocol | `docs/experiments/evaluation_protocol_v1.md` | 9-experiment fair eval protocol |
| Fair eval JSONs | `experiments/comparison/tier/*.json` | All quantitative results |
| Report system | `mouse_extensions/scripts/report/` | Automated HTML report generation |

---

## 10. Navigation

| Link | Document |
|------|----------|
| ← Hub | [[INDEX]] |
| ← FL vs PS | [[FL_vs_PS_comparison]] |
| → DA1 details | [[domain_adaptation_DA1]] |
| → Eval protocol | [[evaluation_protocol_v1]] |
| → Hypothesis roadmap | [[hypothesis_roadmap]] |
| → MVDiff roadmap | [[hypothesis_roadmap]] (consolidated) |
| → Pipeline arch | [[theory/PIPELINE_ARCHITECTURE]] |

---

*MVDiffusion Bottleneck Analysis v1.1 | 2026-02-22*
