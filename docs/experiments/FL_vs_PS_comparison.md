# FaceLift vs Pose-Splatter Comparison Experiment

> **Status**: Active | **Created**: 2026-02-15 | **Updated**: 2026-02-15
> **Location**: `docs/experiments/FL_vs_PS_comparison.md`

---

## Overview

Two feed-forward 3D Gaussian Splatting models evaluated on the same multi-view mouse dataset:

| | FaceLift | Pose-Splatter |
|--|----------|---------------|
| **Paper** | Lyu et al., ICCV 2025 | Goffinet et al., NeurIPS 2025 |
| **Architecture** | SD2.1-UnCLIP MVDiffusion + GSLRM | Shape Carving + Stacked U-Net (3x) + gsplat |
| **Inference** | Feed-forward (two-stage) | Feed-forward (~30ms/frame) |
| **Resolution** | **512x512** | 576x512 (ds=2 from 1152x1024) |
| **Dataset** | M5t2 (mouse, 6 cam, temporal 80:10:10) | markerless_mouse_1_nerf (6 cam) |

FaceLift is the **primary model**; Pose-Splatter is the baseline comparison.

---

## Current Best Checkpoints (M5t2)

### FaceLift (gpu03)

| Component | Checkpoint | Location | Performance |
|-----------|-----------|----------|-------------|
| **MVDiffusion** | `checkpoint-5000` (sparse attention) | `checkpoints/mvdiffusion/mouse_M5t2/` | Best MVDiff: PSNR_wh=21.29 |
| **GS-LRM** | `best_psnr.pt` (step 8001) | `checkpoints/gslrm/M5t2_E0_1_facelift/` | Val PSNR=22.34 |
| **E2E combo** | MVDiff ckpt-5000 + GS-LRM best | `outputs/h5_e2e/baseline_ckpt5000/` | PSNR_wh=20.81, IoU=0.491 |

**In-progress** (may improve):
- E1: `mouse_M5t2_20k_cosine` (extended 20K, cosine LR) — GPU 4
- E2: `mouse_M5t2_randref_20k_resume` (P0 resume, LR decay) — GPU 7

### Pose-Splatter (joon)

| Experiment | Config | Location |
|-----------|--------|----------|
| **facelift_compare_5cam** | ds=2, 3DGS, frame_jump=5 | `output/facelift_compare_5cam/latest/` |

---

## Metric Protocol Alignment

### The Problem

| Aspect | FaceLift | Pose-Splatter |
|--------|----------|---------------|
| **Background** | White-BG composite (`rgb*alpha + (1-alpha)`) | Raw RGB (no composite) |
| **PSNR** | `psnr_full_white` = full-image on white-BG | `masked_psnr` = foreground pixels only |
| **SSIM** | `ssim_full_white` = full-image skimage | `masked_ssim` = torchmetrics on FG |
| **L1** | `masked_l1 = sum\|pred-gt\|/(3*sum(mask))` | Same formula |
| **IoU** | `silhouette_iou` (alpha > 0.5) | Same formula |

### Aligned Comparison (Recommended)

For **fair comparison**, use these matched pairs:

| Metric | FaceLift key | PS key | Protocol | Comparable? |
|--------|-------------|--------|----------|-------------|
| **PSNR (FG)** | `psnr_fg_only` | `psnr` (masked) | Foreground-only MSE | **Yes** (minor edge diff) |
| **PSNR (white)** | `psnr_full_white` | _(not computed)_ | White-BG full-image | FL only |
| **L1** | `masked_l1` | `l1` | Same formula | **Yes** |
| **IoU** | `silhouette_iou` | `iou` | Same formula | **Yes** |
| **SSIM** | `ssim_full_white` | `ssim` (masked) | Different protocols | **No** |

**Edge pixel difference**: FL composites to white-BG then masks (semi-transparent edges get white-blended). PS masks raw RGB. For alpha > 0.5 boundary pixels, this causes minor discrepancy. Interior pixels are identical.

---

## Current Results (Protocol-Aligned)

### Comparable Metrics (same formula)

| Metric | FaceLift (E2E best) | Pose-Splatter | Gap |
|--------|--------------------|--------------|----|
| **PSNR (FG-only)** | **7.75** | **24.68** | -16.93 dB |
| **L1 (masked)** | 0.319 | **0.097** | +0.222 |
| **IoU** | 0.491 | **0.829** | -0.338 |

### Protocol-Specific Metrics

| Metric | FaceLift | PS | Notes |
|--------|----------|-----|-------|
| PSNR (full_white) | 20.81 | N/A | BG inflates PSNR |
| SSIM (full_white) | 0.965 | N/A | Not comparable to PS masked SSIM |
| SSIM (masked) | N/A | 0.963 | |

### Interpretation

FaceLift's foreground PSNR (7.75 dB) is drastically lower than PS (24.68 dB):
- **IoU 0.491**: FL silhouette covers only ~half of GT foreground
- **L1 0.319**: FL foreground color error 3x higher than PS
- FL is still in early E2E pipeline maturation; PS is an optimized per-scene model
- FL's `psnr_full_white` (20.81) is misleadingly inflated by easy background

### Per-View Breakdown (FaceLift full_white, PS masked)

| View | FL (full_white) | PS (masked) | PS holdout? |
|------|----------------|-------------|-------------|
| view_0 | N/A (input) | 24.85 | |
| view_1 | 23.16 | 25.22 | |
| view_2 | 19.52 | 24.08 | |
| view_3 | 20.32 | 24.27 | |
| view_4 | 20.49 | 24.68 | |
| view_5 | 20.55 | 24.94 | Yes |

---

## Experiment Configurations

### FaceLift E2E (h5_e2e/baseline_ckpt5000)

```yaml
# MVDiffusion: sparse attention, M5t2
mvdiffusion_checkpoint: checkpoints/mvdiffusion/mouse_M5t2/checkpoint-5000
n_views: 6
img_wh: 512
reference_view_idx: 0
sparse_mv_attention: true
background_color: white

# GS-LRM: 4-view input, 512x512
gslrm_checkpoint: checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt
image_size: 512
num_views: 6
num_input_views: 4
```

### Pose-Splatter (facelift_compare_5cam)

```json
{
  "image_width": 1152, "image_height": 1024,
  "image_downsample": 2,
  "holdout_views": [5],
  "train_views": [0, 1, 2, 3, 4],
  "frame_jump": 5,
  "gaussian_mode": "3d",
  "grid_size": 112,
  "split_ratios": [0.8, 0.1, 0.1]
}
```

---

## How to Run

```bash
# On gpu03 (FaceLift repo root):

# Protocol-aligned comparison (uses psnr_fg_only for FaceLift)
python -m mouse_extensions.scripts.eval.compare_with_baseline \
    --facelift_metrics outputs/h5_e2e/baseline_ckpt5000/metrics_v2.json \
    --baseline_metrics baselines/pose_splatter/paper_standard_evaluation.json \
    --output_dir experiments/comparison/FL_vs_PS/

# Recompute FaceLift metrics (if needed)
python -m mouse_extensions.scripts.eval.compare_with_baseline \
    --facelift_dir outputs/h5_e2e/baseline_ckpt5000 \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --baseline_metrics baselines/pose_splatter/paper_standard_evaluation.json \
    --output_dir experiments/comparison/FL_vs_PS/
```

---

## Data Sync (joon → gpu03)

```bash
# Copy PS metrics JSON only
scp joon:/home/joon/dev/pose-splatter/output/facelift_compare_5cam/latest/paper_standard_evaluation.json \
    gpu03:/home/joon/dev/FaceLift/baselines/pose_splatter/
```

---

## Related Files

| File | Server | Description |
|------|--------|-------------|
| `compare_with_baseline.py` | gpu03: `mouse_extensions/scripts/eval/` | Comparison script |
| `unified_eval_config.yaml` | gpu03: `mouse_extensions/scripts/eval/` | Config |
| `compute_e2e_metrics.py` | gpu03: `mouse_extensions/scripts/eval/` | FL metrics v2.0 (all protocols) |
| `metrics.py` | gpu03: `mouse_extensions/evaluation/` | MetricsComputer class |
| `METRICS_PROTOCOL.md` | gpu03: `docs/theory/` | Metric theory |
| `POSE_SPLATTER_GUIDE.md` | gpu03: `docs/guides/` | PS comparison guide |
| `paper_standard_evaluation.json` | joon: `output/facelift_compare_5cam/latest/` | PS metrics |
| `image_metrics.py` | joon: `src/modules/core/metrics/` | PS metrics implementation |
| `evaluation.md` | joon: `docs/practical/` | PS eval guide + FL comparison |

---

## Next Steps

1. **Wait for E1/E2 MVDiffusion training** to complete for potentially better checkpoints
2. **Re-evaluate with best combo** once E1/E2 finish
3. **Add SSIM alignment**: Compute white-BG SSIM for PS renders, or masked SSIM for FL
4. **Resolution matching**: Consider center-crop PS 576x512 → 512x512 for exact resolution match
5. **Side-by-side rendering**: GT / FL / PS comparison grid + orbit video

---

*Created: 2026-02-15 | FaceLift vs Pose-Splatter Unified Evaluation*
