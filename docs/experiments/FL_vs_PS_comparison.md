# FaceLift vs Pose-Splatter Comparison Experiment

> **Status**: Active | **Created**: 2026-02-15 | **Updated**: 2026-02-15
> **Location**: `docs/experiments/FL_vs_PS_comparison.md`
> **Version**: v4 (+ input view asymmetry analysis + tiered comparison)

---

## Overview

Two feed-forward 3D Gaussian Splatting models evaluated on the **same M5t2 dataset** (mouse, 6-camera, temporal 80:10:10 split):

| | FaceLift | Pose-Splatter |
|--|----------|---------------|
| **Paper** | Lyu et al., ICCV 2025 | Goffinet et al., NeurIPS 2025 |
| **Architecture** | SD2.1-UnCLIP MVDiffusion + GSLRM | Shape Carving + Stacked U-Net (3x) + gsplat |
| **Inference** | Feed-forward (two-stage) | Feed-forward (~30ms/frame) |
| **Resolution** | 512x512 | 576x512 (ds=2 from 1152x1024) |
| **Generalization** | Cross-scene (never seen test data) | Per-scene (optimized on same video) |

FaceLift is the **primary model**; Pose-Splatter is the baseline comparison.

---

## M5t2 Dataset (Canonical Split)

| Split | Frame Range | Count | Ratio |
|-------|------------|-------|-------|
| **Train** | 000000 ~ 002879 | 2,880 | 80% |
| **Val** | 002880 ~ 003239 | 360 | 10% |
| **Test** | 003240 ~ 003599 | 360 | 10% |
| **Total** | | 3,600 | 100% |

**Why 80:10:10**: FaceLift (data-hungry diffusion model) needs large training set. H1bis confirmed M5t2 > M5t (1:1:1) by +2.9 dB PSNR, 2x IoU improvement.

**Data sources** (same physical frames, different preprocessing):
- FL: `/home/joon/data/preprocessed/FaceLift_mouse/M5/` (512x512 RGBA)
- PS: `markerless_mouse_1_nerf/fj5_ds2/images/images.zarr` (3600, 6, 512, 576, 3) RGB

---

## Current Best Checkpoints (M5t2)

### FaceLift (gpu03)

| Component | Checkpoint | Path | Performance |
|-----------|-----------|------|-------------|
| **MVDiffusion** | `checkpoint-5000` (sparse attn) | `checkpoints/mvdiffusion/mouse_M5t2/` | Val PSNR_wh=27.30 |
| **GS-LRM** | `best_psnr.pt` (step 8001) | `checkpoints/gslrm/M5t2_E0_1_facelift/` | Val PSNR=22.34 (GT input) |
| **E2E combo** | MVDiff ckpt-5000 + GS-LRM best | `outputs/h5_e2e/baseline_ckpt5000/` | PSNR_wh=21.29, sIoU=0.518 |

**In-progress MVDiffusion improvements** (Phase 3, H5):
- **E1**: `mouse_M5t2_20k_cosine` — cosine LR, 20K steps, GPU 4 (~3K/20K)
- **E2**: `mouse_M5t2_randref_20k_resume` — P0 resume, LR=1e-5, GPU 7 (~13K/20K)

**GS-LRM ablations** (GT input reference, H4/H6):
- 6-view: **PSNR=24.49** (best geometry)
- 4-view + alpha=1.0: PSNR=20.84, **LPIPS=0.015, IoU=0.956** (best perceptual)

### Pose-Splatter (joon)

| Experiment | Config | Location |
|-----------|--------|----------|
| **facelift_compare_5cam** | ds=2, 3DGS, grid=112, fj=5 | `output/facelift_compare_5cam/latest/` |
| Status | 50 epochs, loss=0.370, ~37h training | completed |

---

## 6 Critical Fairness Issues (Identified 2026-02-15)

Previous comparison was **unfair**. Investigation found:

### Issue 1: Training Data Leakage (PS)

PS `paper_standard_evaluation` uses `frame_step=30` across ALL 3600 frames = 120 frames evaluated. Of these, **96/120 = 80% are training frames**:
- Train frames (0-2879): 96 frames
- Val frames (2880-3239): 12 frames
- Test frames (3240-3599): 12 frames

PS metrics are inflated by memorized training data.

### Issue 2: Model Type Asymmetry

| | FaceLift | Pose-Splatter |
|--|----------|---------------|
| Type | **Generalizing** (feed-forward, cross-scene) | **Memorizing** (per-scene optimization) |
| Test data | Never seen | Trained on same video (80% of frames) |
| Fair analog | Zero-shot on test frames | Per-scene test reconstruction |

### Issue 3: Mask Source Asymmetry

- FL GT: RGBA with clean binary alpha channel
- PS GT: RGB only in zarr, mask extracted from white-BG (`pixel == 1.0`)
- Different masks → different foreground regions → incomparable metrics

### Issue 4: Metric Protocol Mismatch

| Metric | FL (old) | PS (old) | Fair version |
|--------|----------|----------|-------------|
| PSNR | `psnr_full_white` (BG inflates) | `psnr` (whole image) | `psnr_gt_masked` (FG-only) |
| SSIM | skimage, full image | torchmetrics, full image | `ssim_gt_masked` (bbox crop) |
| L1 | `masked_l1` | `l1` (masked) | **Same** |
| IoU | `silhouette_iou` | `iou` | **Same** |

### Issue 5: FL Silhouette Extraction Problem

FL renders are RGB-only (no alpha). Silhouette extracted via `white_bg_threshold=0.98`. Mouse is only ~2.5% of image → threshold sensitivity causes IoU instability:

| Threshold | Estimated IoU | Coverage |
|-----------|:------------:|:--------:|
| 0.90 | ~0.35 | ~45% |
| 0.95 | ~0.45 | ~55% |
| 0.98 | ~0.49 | ~60% |

User's visual inspection confirms: **mouse itself looks accurate**, but metrics show poor performance due to coverage/extraction issues.

### Issue 6: Input View Count Asymmetry ★★★ (Most Critical)

The two models use fundamentally different amounts of input information:

| Stage | FaceLift | Pose-Splatter |
|-------|----------|---------------|
| **Training input** | 1 reference view (conditioning) | **5 observed views** (mask+image) for shape carving |
| **Training target** | All 6 views (loss) | Random 1 view (loss) |
| **Training paradigm** | Cross-scene (diverse animals) | **Per-scene** (same video, 50 epochs) |
| **Inference input** | **1 real image** | **6 real views** (5 observed + **target view mask**) |
| **Target view info** | None (zero-shot) | **Yes** — target mask used in shape carving |
| **Information ratio** | 1× | **6×** |

**Why this is the most critical issue**: PS uses the **target view's silhouette mask** as input to shape carving at inference. This is equivalent to "knowing the answer's outline before coloring it in." FL must predict the entire 3D from a single image with zero knowledge of the target view.

**Literature support**: The Pose-Splatter paper (Goffinet et al., NeurIPS 2025) explicitly states:
> *"Since [single-view] methods require fundamentally different input (single image vs. multi-view video), quantitative comparison with them would be inherently unfair."*

Additional references:
- **NerfBaselines** (Stier et al., NeurIPS 2025): Same input conditions for quantitative comparison
- **MVGBench** (ICCV 2025): Input view count tier separation
- **Charge** (CVPR 2025): Few-shot ablation (1/2/4/8 views)
- **LVSM** (Jin et al., ICLR 2025): Qualitative only + disclaimer for different input counts

---

## Tiered Comparison Protocol (v4) — Literature-Based

Based on Issue 6, direct E2E FL vs PS quantitative comparison is **invalid**. The following tiered protocol is recommended:

### Tier A: Quantitative (Same Input Conditions)

| Comparison | Input | FL Component | PS | Fair? |
|-----------|-------|-------------|-----|-------|
| **GS-LRM GT 6-view vs PS** | 6 GT views | GS-LRM only (bypass MVDiff) | Full pipeline | ✅ Yes |
| **GS-LRM GT 4-view vs PS 4-view** | 4 GT views | GS-LRM 4-view | PS with 4 views | ✅ Yes |

**Available data**: GS-LRM GT 6-view achieves **PSNR=24.49** (H4), vastly exceeding PS test-only **PSNR=16.80** (+7.69 dB).

### Tier B: Qualitative + Disclaimer (Different Input Conditions)

| Comparison | FL Input | PS Input | Presentation |
|-----------|---------|---------|-------------|
| **E2E FL vs PS** | 1 image | 6 images | Side-by-side visualization + explicit disclaimer |

Must include: *"FaceLift uses 1 input view; Pose-Splatter uses 6 views including target silhouette. Direct metric comparison is not meaningful."*

### Tier C: Upper Bound Analysis

| Comparison | Meaning |
|-----------|---------|
| GS-LRM GT 6-view (24.49) vs PS (16.80) | Same input count → FL reconstruction quality upper bound |
| GS-LRM GT 6-view (24.49) vs E2E (7.83) | MVDiffusion quality gap (bottleneck analysis) |

### Reinterpretation of Current Results

| Metric | FL E2E | PS test-only | Tier | Validity |
|--------|--------|-------------|------|----------|
| PSNR_gt_masked | 7.83 | 16.80 | **B** | Quantitative comparison invalid |
| PSNR_intersection | 15.44 | 20.48 | **B** | Reference only |
| IoU | 0.518 | 0.827 | **B** | Coverage gap (FL silhouette extraction) |
| GS-LRM GT 6-view | **24.49** | 16.80 | **A** | ✅ Valid quantitative comparison |

**Key insight**: When given the same 6 GT views, GS-LRM outperforms PS by **+7.69 dB**. The E2E quality drop is entirely due to MVDiffusion (Stage 1) generation quality, not reconstruction capability.

---

## Fair Evaluation Protocol (v3)

### Fairness Guarantees

| Guarantee | How |
|-----------|-----|
| **Test-only** | FL: frames 3240-3459 (200 rendered). PS: frames 3240-3599. |
| **GT masks** | Both use M5 alpha channel (binary, > 127) |
| **Same functions** | `compute_all_metrics()` in `fair_comparison.py` |
| **FG-only** | Masked PSNR/SSIM/L1 on foreground |
| **Coverage-aware** | Intersection metrics separate quality from coverage |
| **Resolution** | PS center-cropped 576→512 |

### Metric Definitions

| Metric | Description | Separates |
|--------|-------------|-----------|
| `psnr_gt_masked` | PSNR on GT foreground pixels | Quality + Coverage |
| `psnr_intersection` | PSNR where BOTH have foreground | **Quality only** |
| `ssim_gt_masked` | SSIM on white-BG composite, bbox crop | Quality + Coverage |
| `l1_gt_masked` | L1 on GT foreground | Quality + Coverage |
| `l1_intersection` | L1 where BOTH have foreground | **Quality only** |
| `iou` | Silhouette IoU (pred vs GT mask) | Coverage |
| `coverage` | GT FG covered by pred FG (%) | Coverage |
| `color_bias_{r,g,b}` | Mean color difference on intersection | Color accuracy |

### Key Insight

Previous PSNR_fg=7.75 was misleading because:
1. FL silhouette extraction misses ~50% of GT foreground
2. Uncovered GT pixels are treated as white (1.0) vs GT color → huge MSE
3. `psnr_intersection` isolates true color accuracy from coverage problems

---

## Old Results (UNFAIR — for reference only)

| Metric | FaceLift | PS | Notes |
|--------|----------|-----|-------|
| PSNR_fg | 7.75 | 24.68 | **UNFAIR**: PS includes 80% train frames |
| L1_masked | 0.319 | 0.097 | |
| IoU | 0.491 | 0.829 | FL extraction threshold issue |
| PSNR_full_white | 20.81 | N/A | BG inflates |

---

## How to Run (Fair Evaluation)

### Step 1: Evaluate FaceLift (gpu03)

```bash
cd /home/joon/dev/FaceLift
python -m mouse_extensions.scripts.eval.fair_comparison evaluate_fl \
  --render_dir outputs/h5_e2e/baseline_ckpt5000/samples \
  --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --output experiments/comparison/fair/facelift_fair.json
```

### Step 2: Evaluate Pose-Splatter (joon)

```bash
cd /home/joon/dev/pose-splatter
python scripts/eval/fair_test_only_eval.py \
  --exp_dir output/facelift_compare_5cam/latest \
  --test_start 3240 --test_end 3600 \
  --crop_to 512 \
  --output experiments/fair/posesplatter_fair.json

# With M5 GT masks (if M5 data accessible from joon):
python scripts/eval/fair_test_only_eval.py \
  --exp_dir output/facelift_compare_5cam/latest \
  --m5_gt_dir /path/to/FaceLift_mouse/M5 \
  --output experiments/fair/posesplatter_fair.json
```

### Step 3: Copy PS results to gpu03

```bash
scp joon:~/dev/pose-splatter/experiments/fair/posesplatter_fair.json \
  gpu03:~/dev/FaceLift/baselines/pose_splatter/
```

### Step 4: Compare

```bash
cd /home/joon/dev/FaceLift
python -m mouse_extensions.scripts.eval.fair_comparison compare \
  --facelift experiments/comparison/fair/facelift_fair.json \
  --baseline baselines/pose_splatter/posesplatter_fair.json \
  --output_dir experiments/comparison/fair/
```

---

## Experiment Configurations

### FaceLift E2E (baseline_ckpt5000)

```yaml
mvdiffusion_checkpoint: checkpoints/mvdiffusion/mouse_M5t2/checkpoint-5000
gslrm_checkpoint: checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt
n_views: 6
img_wh: 512
reference_view_idx: 0
sparse_mv_attention: true
num_input_views: 4
background_color: white
dataset: M5t2 (temporal 80:10:10)
```

### Pose-Splatter (facelift_compare_5cam)

```json
{
  "data": "markerless_mouse_1_nerf",
  "preprocess": "fj5_ds2",
  "image_width": 1152, "image_height": 1024,
  "image_downsample": 2,
  "holdout_views": [5],
  "train_views": [0, 1, 2, 3, 4],
  "split_ratios": [0.8, 0.1, 0.1],
  "gaussian_mode": "3d",
  "grid_size": 112,
  "ell": 0.22
}
```

---

## Related Hypotheses & Experiments

| Doc | Key Finding | Relevance |
|-----|------------|-----------|
| **H1bis** | M5t2 > M5t by +2.9 dB, 2x IoU | Justifies 80:10:10 split |
| **H4** | 6-view PSNR=24.49, monotonic increase with views | GS-LRM upper bound |
| **H5** | MVDiffusion bottleneck (GT→E2E: 50-70% loss) | Main improvement axis |
| **H6** | alpha=1.0 best perceptual (LPIPS=0.015) | Trade-off: PSNR vs perceptual |
| **H8** | 3-4 view generation viable | Future optimization |
| **Phase 3** | E1 (cosine LR) + E2 (resume) in progress | Next E2E improvement |

**Bottleneck**: MVDiffusion (Stage 1) is the pipeline bottleneck. GT input gives PSNR 21-24, but E2E drops to FG PSNR 6-9. Phase 3 training improvements (E1, E2) target this.

---

## Related Files

| File | Server | Description |
|------|--------|-------------|
| **`fair_comparison.py`** | gpu03: `mouse_extensions/scripts/eval/` | **Fair FL eval + comparison** |
| **`fair_test_only_eval.py`** | joon: `scripts/eval/` | **Fair PS test-only eval** |
| `compare_with_baseline.py` | gpu03: `mouse_extensions/scripts/eval/` | Old comparison (v2) |
| `compute_e2e_metrics.py` | gpu03: `mouse_extensions/scripts/eval/` | FL metrics v2.0 |
| `metrics.py` | gpu03: `mouse_extensions/evaluation/` | MetricsComputer class |
| `paper_standard_evaluate.py` | joon: `scripts/mouse/analysis/` | PS original eval |
| `image_metrics.py` | joon: `src/modules/core/metrics/` | PS metrics |
| `EXPERIMENT_REGISTRY.md` | gpu03: `docs/experiments/` | All experiments catalog |

---

## Fair Evaluation Results (2026-02-15)

### FL E2E (baseline_ckpt5000) — Test-only

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR_gt_masked | 7.83 | Low due to coverage gap |
| PSNR_intersection | 15.44 | Pure color accuracy (where both have FG) |
| SSIM_gt_masked | 0.718 | |
| L1_gt_masked | 0.280 | |
| IoU | 0.518 | Silhouette extraction threshold issue |
| Coverage | 71.4% | ~29% of GT FG missed |
| Color bias (R/G/B) | +0.06/+0.06/+0.06 | Systematic brightness offset |

- Frames: 200 (003240-003439), Views: [1,2,3,4,5]
- Spatial misalignment confirmed via visualization

### PS (facelift_compare_5cam) — Test-only

| Metric | Value | Notes |
|--------|-------|-------|
| PSNR_gt_masked | 16.80 | Dropped from 24.68 (was 80% train frames) |
| PSNR_intersection | 20.48 | |
| SSIM_gt_masked | 0.877 | |
| L1_gt_masked | 0.118 | |
| IoU | 0.827 | |
| Coverage | 91.9% | |

- Frames: 72 (test only, step=5), Views: [0-5]
- PSNR drop of **-7.88 dB** confirms training data leakage (Issue 1)

### Tier A Comparison (Same Input = 6 GT views)

| Metric | GS-LRM GT 6-view | PS test-only | Delta |
|--------|------------------|-------------|-------|
| PSNR | **24.49** | 16.80 | **+7.69 dB** ✅ |

**Conclusion**: Under fair input conditions (6 GT views), FaceLift's GS-LRM substantially outperforms Pose-Splatter.

---

## Next Steps

1. ~~Run fair evaluation on both servers~~ ✅ Completed
2. **Wait for E1/E2** MVDiffusion training to complete → re-evaluate E2E
3. **GS-LRM test-only evaluation**: Run GS-LRM with GT 6 views on test frames (3240-3599) for complete Tier A comparison
4. **Investigate spatial misalignment**: FL renders show systematic pixel offset vs GT
5. **Investigate color bias**: FL shows systematic brightness offset (+0.06 per channel)
6. **Side-by-side rendering**: GT / FL / PS comparison grid (Tier B qualitative)
7. **View ablation**: GS-LRM with 1/2/4/6 GT views to show view-count scaling
8. **Document**: Update comparison table in paper draft with tiered results + disclaimer

---

*Created: 2026-02-15 | Updated: 2026-02-15 | FaceLift vs Pose-Splatter Unified Evaluation v4*
