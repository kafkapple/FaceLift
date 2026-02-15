# FaceLift vs Pose-Splatter Comparison Experiment

> **Status**: Active | **Created**: 2026-02-15
> **Location**: `docs/experiments/FL_vs_PS_comparison.md`

---

## Overview

Two feed-forward 3D Gaussian Splatting models evaluated on the same multi-view mouse dataset:

| | FaceLift | Pose-Splatter |
|--|----------|---------------|
| **Paper** | Lyu et al., ICCV 2025 | Goffinet et al., NeurIPS 2025 |
| **Architecture** | SD2.1-UnCLIP MVDiffusion + GSLRM | Shape Carving + Stacked U-Net (3x) + gsplat |
| **Inference** | Feed-forward (two-stage) | Feed-forward (~30ms/frame) |
| **Dataset** | Multi-view mouse (6 cam, M5) | Mouse/Rat/Finch (6 cam, 30fps) |

FaceLift is the **primary model** (our focus); Pose-Splatter is the baseline comparison.

---

## Metric Protocol

### Key Difference

| Protocol | FaceLift | Pose-Splatter |
|----------|----------|---------------|
| **Background** | White-BG composite + full-image | Masked (foreground-only) |
| **PSNR** | `MSE(white_pred, white_gt)` full-image | `MSE(fg_pred, fg_gt)` masked |
| **SSIM** | skimage on white-BG composite | torchmetrics on masked region |
| **L1** | `sum\|pred-gt\| / (3 * sum(mask))` | Same formula |
| **IoU** | Silhouette: alpha > 0.5 | Binary mask > 0.5 |

**Comparable metrics**: L1 (masked) and IoU use the same formula.
**Non-comparable**: PSNR and SSIM differ in protocol. White-BG PSNR is inflated by easy background pixels.

### Unified Metrics (compare_with_baseline.py)

```
PSNR_full_white  — Literature standard (white-BG composite)
SSIM_full_white  — Literature standard
LPIPS_full_white — Perceptual (FaceLift only)
Mask_IoU         — Silhouette agreement
L1_masked        — Foreground error (most fair)
```

---

## Current Results

### FaceLift: h5_e2e/cfgr_ckpt10000 (gpu03)

| Metric | Mean |
|--------|------|
| PSNR (full_white) | 20.81 |
| SSIM | 0.9648 |
| IoU | 0.4911 |
| L1 (masked) | 0.3194 |

### Pose-Splatter: facelift_compare_5cam (joon)

| Metric | Overall | Holdout (view 5) |
|--------|---------|-----------------|
| PSNR (masked) | 24.68 | 24.94 |
| SSIM (masked) | 0.963 | 0.966 |
| IoU | 0.829 | 0.826 |
| L1 | 0.097 | 0.111 |

**Config**: 6 views, holdout=[5], image_downsample=2 (576x512), frame_jump=5

### Interpretation

- PS shows higher IoU (0.829 vs 0.491) — FL silhouette prediction needs improvement
- PS shows lower L1 (0.097 vs 0.319) — FL foreground color accuracy lower
- PSNR not directly comparable (different protocols)
- FL is still in early training stages; PS is a mature baseline

---

## How to Run

```bash
# On gpu03 (FaceLift repo root):

# Load pre-computed metrics from both models
python -m mouse_extensions.scripts.eval.compare_with_baseline \
    --facelift_metrics outputs/h5_e2e/cfgr_ckpt10000/metrics_v2.json \
    --baseline_metrics /path/to/pose-splatter/paper_standard_evaluation.json \
    --output_dir experiments/comparison/FL_vs_PS/

# Or recompute FaceLift metrics
python -m mouse_extensions.scripts.eval.compare_with_baseline \
    --facelift_dir outputs/h5_e2e/cfgr_ckpt10000 \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --baseline_metrics /path/to/pose-splatter/paper_standard_evaluation.json \
    --output_dir experiments/comparison/FL_vs_PS/

# Quick comparison with manual values
python -m mouse_extensions.scripts.eval.compare_with_baseline \
    --facelift_metrics outputs/h5_e2e/cfgr_ckpt10000/metrics_v2.json \
    --baseline_values '{"psnr": 24.68, "ssim": 0.963, "iou": 0.829, "l1": 0.097}' \
    --output_dir experiments/comparison/FL_vs_PS/
```

---

## Data Synchronization

PS results on `joon` server need to be copied to `gpu03` for comparison:

```bash
# Option A: Copy PS metrics JSON only (lightweight)
scp joon:/home/joon/dev/pose-splatter/output/facelift_compare_5cam/latest/paper_standard_evaluation.json \
    gpu03:/home/joon/dev/FaceLift/baselines/pose_splatter/

# Option B: Copy full PS output (rendered images for qualitative comparison)
ssh joon "tar czf /tmp/ps_results.tar.gz \
    -C /home/joon/dev/pose-splatter \
    output/facelift_compare_5cam/latest/paper_standard_evaluation.json \
    output/facelift_compare_5cam/latest/paper_eval_images/"
scp joon:/tmp/ps_results.tar.gz gpu03:/tmp/
ssh gpu03 "mkdir -p /home/joon/dev/FaceLift/baselines/pose_splatter && \
    tar xzf /tmp/ps_results.tar.gz -C /home/joon/dev/FaceLift/baselines/pose_splatter/"
```

---

## Related Files

| File | Location | Description |
|------|----------|-------------|
| `compare_with_baseline.py` | `mouse_extensions/scripts/eval/` | Unified comparison script |
| `unified_eval_config.yaml` | `mouse_extensions/scripts/eval/` | Config for comparison |
| `compute_e2e_metrics.py` | `mouse_extensions/scripts/eval/` | FaceLift E2E metrics v2.0 |
| `metrics.py` | `mouse_extensions/evaluation/` | MetricsComputer class |
| `POSE_SPLATTER_GUIDE.md` | `docs/guides/` | PS comparison protocol |
| `METRICS_PROTOCOL.md` | `docs/theory/` | Metric theory & protocol |

---

*Created: 2026-02-15 | FaceLift vs Pose-Splatter Unified Evaluation*
