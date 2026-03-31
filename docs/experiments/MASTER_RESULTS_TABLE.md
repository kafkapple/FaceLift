# Master Results Table

> **Version**: v1.0 | **Created**: 2026-03-23 | **Status**: ACTIVE (논문 Table SSOT)
> **Navigation**: [← INDEX](../INDEX.md) | [UNIFIED_ABLATION_REPORT](UNIFIED_ABLATION_REPORT.md)
> **Last validated**: 2026-03-23 S26 (all numbers from fair eval JSONs on gpu03)

---

## 0. Evaluation Protocol

### Metric Definitions

| Metric | Abbreviation | Definition | Range | Use |
|:---|:---:|:---|:---:|:---|
| **PSNR_gt** | `psnr_gt_masked` | MSE on GT foreground pixels (α > 0.5) | 7–24 dB | **Primary** — fair cross-model comparison |
| **PSNR_int** | `psnr_intersection` | MSE on GT ∩ Pred intersection pixels | 10–24 dB | Diagnostic — color quality given correct silhouette |
| **PSNR_wh** | `psnr_full_white` | Full-image MSE on white-BG composite | 20–37 dB | ⚠️ Literature only — **inflated by background** |
| **IoU** | `iou` | Silhouette overlap (GT mask vs pred mask, threshold 0.98) | 0–1 | Geometry accuracy |
| **SSIM** | `ssim_gt_masked` | Structural similarity on GT foreground | 0–1 | Perceptual quality |
| **L1** | `l1_gt_masked` | Mean absolute error on GT foreground | 0–1 | Alternative to PSNR |
| **Coverage** | `coverage` | Fraction of GT foreground covered by prediction | 0–1 | Completeness |
| **LPIPS** | — | Learned perceptual similarity (VGG) | 0–1 | Perceptual quality (training only) |

**Source code**: `mouse_extensions/scripts/eval/fair_comparison.py`

### ⚠️ PSNR_wh vs PSNR_gt

| | PSNR_wh | PSNR_gt |
|:---|:---|:---|
| Background | Included (white-on-white) | Excluded (GT mask only) |
| Inflation | ~+10-13 dB from background | None |
| 6v baseline example | 34.00 dB | 23.84 dB |
| Cross-model comparison | ❌ Misleading | ✅ Fair |
| Paper standard | LGM, PoseSplatter use this | This project's SSOT |

**Rule**: All cross-model comparisons use PSNR_gt. PSNR_wh is reported only for literature compatibility.

### Data Split

| Split | Frames | Frame Range | % | Use |
|:---|:---:|:---|:---:|:---|
| Train | 2,880 | 000000–002879 | 80% | Model training |
| Val | 360 | 002880–003239 | 10% | Hyperparameter tuning |
| **Test** | **360** | **003240–003599** | **10%** | **All reported metrics** |

- **Evaluated views**: 5 per frame (views 1-5; view 0 excluded as E2E input view)
- **Total samples**: 1,800 per configuration (360 × 5), except where noted
- ⚠️ **View 0 inclusion**: Some archive JSONs include view 0 (n=2160, 6 views). This document uses **5-view evaluation** (views 1-5, n=1800) as the standard. Including view 0 gives different PSNR (e.g., 6v: 23.40 with view 0 vs 23.84 without).
- **Dataset**: M5t2 — C57BL/6 mouse, freely moving in transparent cage
  - 6 overhead cameras (5 × top-down + 1 × side, elevation range ±9.6°)
  - 100 fps, 3,600 frames (36 seconds), RGB 512×512
  - Mouse occupies ~2.5% of image area (small, dark object on white background)
  - Preprocessing: SAM2 masks, white-BG compositing, intrinsics fx=549

### Render Resolution

| Setting | Resolution | Note |
|:---|:---:|:---|
| GS-LRM training | 512 × 512 | |
| GS-LRM inference | **512 × 512** | Fixed 2026-03-22 (was 384, bug) |
| MVDiffusion | 512 × 512 | |
| All metrics in this doc | 512 × 512 | Pre-fix 384 data excluded |

---

## 1. GS-LRM: View Count Ablation (H4)

**Controlled**: base_uniform_v2 config, M5t2, L2 + Perceptual loss, GT input images
**Varied**: `num_input_views` (1–6)

| Views | PSNR_gt ↑ | ± std | IoU ↑ | ± std | PSNR_int ↑ | SSIM ↑ | Coverage | Δ PSNR | n |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 10.47 | 3.99 | 0.028 | 0.008 | 10.47 | 0.828 | 1.000† | — | 1800 |
| 2 | 15.95 | 4.37 | 0.858 | 0.065 | 17.91 | 0.866 | 0.963 | +5.48 | 1800 |
| 3 | 18.56 | 4.04 | 0.899 | 0.057 | 19.54 | 0.897 | 0.985 | +2.61 | 1800 |
| 4 | 20.66 | 3.78 | 0.926 | 0.044 | 21.29 | 0.928 | 0.993 | +2.10 | 1800 |
| 5 | 22.16 | — | 0.942 | — | 22.56 | 0.947 | 0.997 | +1.50 | 1800 |
| **6** | **23.84** | **1.68** | **0.954** | **0.009** | **24.02** | **0.963** | **0.999** | **+1.68** | **1800** |

> † 1v Coverage=1.000 with IoU=0.028 indicates a **degenerate solution**: model predicts nearly the entire image as foreground. This is a failure case, not a partial success.
>
> Note: 5v std missing (archive JSON format difference). Val PSNR from training logs differs: 6v=24.49, 4v=21.50 (different split + metric). All view ablation values use 5-view evaluation (views 1-5, n=1800).

**Checkpoints**: `/node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_{n}view_v2/best_psnr.pt`

---

## 2. GS-LRM: Alpha Loss Ablation (H6)

**Controlled**: M5t2, L2 + Perceptual loss, GT input images
**Varied**: `alpha_loss_weight` (0.0, 0.3, 0.5, 1.0) at 4-view and 6-view

### 2a. 6-View + Alpha (Fair Eval)

| α | PSNR_gt ↑ | ± std | IoU ↑ | ± std | PSNR_int | SSIM | L1 ↓ | Coverage | PSNR_wh | n |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | **23.84** | 1.68 | 0.954 | 0.009 | **24.02** | **0.963** | 0.032 | **0.999** | 34.00 | 1800 |
| 0.3 | 23.29 | 1.87 | **0.956** | 0.009 | 23.60 | 0.961 | **0.030** | 0.998 | **34.10** | 1800 |
| 0.5 | 23.00 | 1.87 | 0.953 | 0.009 | 23.28 | 0.959 | 0.031 | 0.998 | 34.07 | 1800 |
| 1.0 | 22.55 | 1.86 | 0.949 | 0.010 | 22.83 | 0.957 | 0.033 | 0.998 | 33.82 | 1800 |

**Checkpoints**: baseline = `base_uniform_v2_6view_v2/best_psnr.pt`, alpha = `M5t2_6view_alpha{03,05,10}_v3/ckpt_0000000000015840.pt`

> ⚠️ **Checkpoint asymmetry‡**: Baseline uses `best_psnr.pt` (val-selected, optimistic), alpha variants use fixed-step checkpoint (15840 steps). This systematically favors the baseline. Direct comparison is **indicative**, not strictly controlled. For publication, either re-evaluate baseline at step 15840, or re-train alpha variants with val-based selection.

### 2b. 4-View + Alpha (Fair Eval)

| α | PSNR_gt ↑ | ± std | IoU ↑ | ± std | PSNR_int | SSIM | L1 ↓ | Coverage | n |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.0** | **20.66** | 3.78 | **0.926** | 0.044 | **21.29** | **0.928** | **0.058** | 0.993 | 1800 |
| 0.3 | 19.88 | 3.40 | 0.914 | 0.051 | 20.50 | 0.923 | 0.062 | 0.992 | 1800 |
| 0.5 | 19.72 | 3.39 | 0.912 | 0.049 | 20.37 | 0.922 | 0.063 | 0.992 | 1800 |
| 1.0 | 19.49 | 3.26 | 0.908 | 0.046 | 20.14 | 0.920 | 0.064 | 0.992 | 1800 |

**Checkpoints**: `base_uniform_v2_4view_alpha{03,05,10}_v3/best_psnr.pt`

### 2c. 4-View Supplementary: Val PSNR + Perceptual Metrics

| α | Val PSNR | LPIPS ↓ | SSIM_val ↑ | Alpha IoU ↑ |
|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 21.82 | 0.0429 | 0.9473 | — |
| 0.5 | 21.20 | 0.0204 | 0.9725 | 0.9451 |
| 1.0 | 20.84 | **0.0147** | **0.9742** | **0.9562** |

> Different metric: Val PSNR ≠ Test PSNR_gt (different split + internal computation).

### 2d. Statistical Significance (Paired t-test, per-frame n=360)

| Comparison | PSNR diff | p-value | IoU diff | p-value | Cohen's d |
|:---|:---:|:---:|:---:|:---:|:---:|
| 6v: α=0 vs α=0.3 | **-0.55** | 2.8e-129 *** | **+0.002** | 7.6e-78 *** | -2.03 |
| 6v: α=0 vs α=0.5 | -0.84 | 4.0e-176 *** | -0.000 | 1.3e-03 ** | -2.89 |
| 6v: α=0 vs α=1.0 | -1.29 | 8.5e-208 *** | -0.005 | 1.2e-110 *** | -3.61 |
| 4v: α=0 vs α=0.3 | -0.78 | 2.0e-60 *** | -0.012 | 1.8e-69 *** | — |
| 4v: α=0 vs α=0.5 | -0.93 | 2.0e-77 *** | -0.014 | 6.6e-84 *** | — |
| 4v: α=0 vs α=1.0 | -1.17 | 1.9e-97 *** | -0.017 | 1.5e-101 *** | — |

> ⚠️ Autocorrelation: 360 frames from continuous video (step=5). Effective n < 360. P-values may be optimistic. Cohen's d is a more robust effect size measure.

---

## 3. Gaussian Quality Metrics (3D + Rendered Alpha)

**Proxy metrics** — no GT for novel views. Measure Gaussian shape/opacity properties.

### 3a. 3D Gaussian Space (5-frame average)

| Model | N Gaussians | Aniso (mean) ↓ | Aniso (p95) ↓ | Entropy (bottom) ↓ | Sparsity (bottom) ↓ | Mean Opacity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 6v α=0 | 884,738 | **71,833** | **328,469** | 2.97 | 0.0111 | 0.0065 |
| 6v α=0.3 | 884,738 | 109,140 | 453,318 | **0.73** | **0.0047** | **0.0034** |
| 6v α=0.5 | 884,738 | 94,577 | 384,434 | 0.76 | 0.0047 | 0.0034 |
| 6v α=1.0 | 884,738 | 79,880 | 314,498 | 0.80 | 0.0045 | 0.0035 |
| 4v α=0 | 589,826 | 67,941 | 306,851 | 3.37 | 0.0146 | 0.0074 |
| 4v α=0.5 | 589,826 | 68,258 | 295,234 | 1.14 | 0.0041 | 0.0049 |
| 4v α=1.0 | 589,826 | **61,922** | **258,628** | **1.00** | **0.0039** | **0.0047** |

**Key finding: 4v/6v anisotropy reversal**
- 4v: alpha ↓ anisotropy (-14% at α=1.0) — geometric correction
- 6v: alpha ↑ anisotropy (+52% at α=0.3) — extreme shapes at low opacity
- Both: alpha ↓ entropy/sparsity/opacity — visibility suppression

### 3b. Novel View FG Ratio / Edge Density (6-View, 512×512)

| Model | FG Ratio ↓ | Edge Density ↓ |
|:---|:---:|:---:|
| 6v α=0 | 0.0329 | 38.36 |
| 6v α=0.3 | 0.0318 | 37.76 |
| 6v α=0.5 | 0.0316 | 37.62 |
| 6v α=1.0 | 0.0314 | 37.93 |

---

## 4. E2E Pipeline (MVDiffusion + GS-LRM)

**Controlled**: M5t2 test set, GS-LRM 4v baseline (except p1 variants = 6v)
**Varied**: MVDiffusion training strategy

| Strategy | PSNR_gt ↑ | IoU ↑ | PSNR_int ↑ | SSIM | Coverage | Key Change | n |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| baseline 5K | 7.93 | 0.474 | 13.70 | 0.758 | 0.740 | Default | 1800 |
| cfgr (full attn) | 7.75 | 0.491 | 15.75 | 0.756 | 0.672 | Full attention | 1800 |
| e1 cosine 11K | 7.90 | 0.522 | 16.12 | 0.761 | 0.704 | Cosine LR (11K) | 1800 |
| e1 cosine 20K | 7.90 | **0.528** | **16.15** | 0.762 | 0.705 | Cosine LR (20K) | 1800 |
| **e2 resume 20K** | **8.20** | 0.521 | 15.63 | 0.761 | **0.715** | **Resume + random ref** | 1800 |
| e3 pose 10K | 8.10 | 0.523 | 15.88 | 0.762 | 0.716 | + pose conditioning | 1800 |
| p1 6v e2e | 8.44 | 0.495 | 14.65 | 0.761 | 0.750 | 6v GS-LRM | 1800 |
| p1_bl 6v e2e | 8.11 | 0.501 | 14.66 | 0.759 | 0.740 | 6v baseline | 1800 |
| p1_e1 6v e2e | 8.04 | 0.511 | 14.89 | — | — | 6v + cosine | 1800 |

**Convergence**: All strategies → PSNR_gt 7.75–8.44 (range 0.69 dB). **Saturated**.

### Bottleneck Decomposition

```
GS-LRM 6v (GT input):  PSNR_gt = 23.84,  IoU = 0.954  ← Upper bound
                                ↓
          MVDiff quality loss: -15.64 dB, IoU drop: -0.436
          ├── Shape loss (86%): IoU 0.954 → ~0.5
          └── Color loss (14%): PSNR_int 24.02 → 15.63
                                ↓
E2E best:               PSNR_gt = 8.20,   IoU = 0.521  ← Current ceiling
```

### Transfer Rate

| Source | Val Δ | E2E Δ | Transfer |
|:---|:---:|:---:|:---:|
| MVDiff strategy | +3.59 dB | +0.27 dB | ~7.5% |
| GS-LRM views (4→6) | +2.15 dB | 0 dB | **0%** |
| Cosine LR (PSNR_int) | — | +0.52 dB | ~15% |

---

## 5. Cross-Model Comparison (FaceLift vs Pose-Splatter)

| Model | PSNR_gt ↑ | IoU ↑ | Type |
|:---|:---:|:---:|:---|
| **GS-LRM 6v (GT)** | **23.84** | **0.954** | GS-LRM only (upper bound) |
| GS-LRM 4v (GT) | 20.66 | 0.926 | GS-LRM only |
| Pose-Splatter (M5) | 13.78 | 0.846 | Baseline |
| E2E best (e2_resume) | 8.20 | 0.521 | Full pipeline (n=1800) |
| E2E baseline (5K) | 7.93 | 0.474 | Full pipeline (n=1800) |

**GS-LRM 6v > PS by +10.06 dB** (GT input). But E2E is capped at ~8 dB by MVDiffusion.

---

## 6. Optimal Configuration Summary

### GS-LRM (Stage 2) — Recommended Settings

| Parameter | Value | Evidence |
|:---|:---|:---|
| `num_input_views` | **6** | §1: +1.68 dB over 5v, monotonic increase |
| `alpha_loss_weight` | **0.3** (trade-off) or **0.0** (max PSNR) | §2a: only setting with IoU↑ + artifact↓ |
| Resolution | **512 × 512** | Match training resolution |
| Loss | L2(1.0) + Perceptual(0.5) | opacity_reg harmful (-1.0 dB) |
| Convergence | ~8K steps | patience=10, val_every=200 |
| Checkpoint | `best_psnr.pt` (val-selected) | |

### MVDiffusion (Stage 1) — Recommended Settings

| Parameter | Value | Evidence |
|:---|:---|:---|
| Strategy | **e2 resume** (sparse + random ref) | §4: best PSNR_gt (8.20) |
| LR | piecewise 5e-5 (sqrt-scaled) | §4: 1e-4 diverges |
| Steps | 20K | Marginal gain 11K→20K |
| Attention | **Sparse** | §4: full attention worse |

### E2E — Current Ceiling

| Metric | Best Achievable | Limiting Factor |
|:---|:---:|:---|
| PSNR_gt | ~8.2 dB | MVDiffusion silhouette (IoU ~0.5) |
| IoU | ~0.53 | MVDiffusion (86% of gap) |
| PSNR_int | ~16.2 dB | Color quality (secondary) |

---

## 7. Checkpoint Registry

| Model | Path | Best Metric |
|:---|:---|:---|
| GS-LRM 6v (best) | `base_uniform_v2_6view_v2/best_psnr.pt` | PSNR_gt=23.84 |
| GS-LRM 6v α=0.3 | `M5t2_6view_alpha03_v3/ckpt_0000000000015840.pt` | PSNR_gt=23.29, IoU=0.956 |
| GS-LRM 4v | `base_uniform_v2_4view_v2/best_psnr.pt` | PSNR_gt=20.66 |
| MVDiff best | `mouse_M5t2_randref_sparse/checkpoint-20000` | E2E PSNR_gt=8.20 |
| MVDiff cosine | `mouse_M5t2_20k_cosine/checkpoint-20000` | E2E PSNR_int=16.15 |

All checkpoints at: `/node_data/joon/checkpoints/FaceLift/gslrm/` or `.../mvdiffusion/`

---

## 8. Data Source Files (gpu03)

| Data | Path |
|:---|:---|
| View ablation fair eval | `experiments/comparison/tier/gslrm_{n}view_fair.json` |
| 6v alpha fair eval | `experiments/comparison/alpha/6view_alpha{03,05,10}_v3_fair.json` |
| 4v alpha fair eval | `experiments/comparison/alpha/4view_alpha{03,05,10}_v3_fair.json` |
| 6v alpha PSNR_wh | `outputs/report/6v_alpha_comparison_512/metrics/6v_alpha_comparison_512.json` |
| Gaussian quality (4v) | `outputs/report/phase1_2_reconstruction/gaussian_quality_metrics/gaussian_quality_comparison.json` |
| Gaussian quality (6v) | `outputs/reports/gaussian_quality_6v_alpha/gaussian_quality_comparison.json` |
| E2E strategies | `experiments/comparison/tier/*_fair.json` |
| Per-frame (for t-test) | `experiments/comparison/alpha/*_perframe.json` |
| FL vs PS | `experiments/comparison/fair/fair_comparison_merged.json` |

---

## Related Documents

| Document | Content |
|:---|:---|
| ↑ [[../INDEX]] | Document hub |
| ↔ [[UNIFIED_ABLATION_REPORT]] | Ablation analysis + interpretation + audit trail |
| ↔ comprehensive_analysis_report _(archived)_ | H1-H8 hypothesis testing |
| ↔ [[ALPHA_LOSS_NOVEL_VIEW_ANALYSIS]] | Alpha loss deep-dive + artifact mechanism |
| ↔ [[EXPERIMENT_REGISTRY]] | Individual experiment configs |
| ↔ [[evaluation_protocol_v1]] | Eval protocol specification |

---

*Master Results Table | v1.0 | 2026-03-23*
*All values from fair eval protocol on M5t2 test set (512×512) unless noted*
