# FL vs PS: Metric & Data Consistency Analysis

> Created: 2026-02-19 | Version: 2.1
> Purpose: Code-level verification of evaluation fairness between FaceLift and Pose-Splatter
>
> **NOTE**: 핵심 분석 결과(카메라 미스매치, B-3 무효화, Option A)는 **[[FL_vs_PS_comparison]] v9.0**에 통합되었습니다. 이 문서는 코드 레벨 검증 상세를 유지합니다.

---

## 1. Executive Summary

FaceLift (FL) and Pose-Splatter (PS) use **identical metric functions** but differ in **GT data source**, **mask definition**, and **resolution**. These differences cause a **2.17x foreground ratio gap** (FL=2.33%, PS=5.04%), making direct PSNR/IoU comparison unreliable.

**Solution**: Unified evaluation pipeline — dump PS renders to disk, evaluate both models against the same M5 RGBA GT using FL's `fair_comparison.py`.

---

## 2. Metric Functions: Verified Identical

### 2.1 Code Comparison

| Function | FL (`fair_comparison.py`) | PS (`fair_test_only_eval.py`) | Match |
|----------|--------------------------|-------------------------------|:-----:|
| `compute_masked_psnr` | `fg = pred[mask>0.5]; mse = mean((fg_pred-fg_gt)^2)` | Identical | YES |
| `compute_masked_ssim` | White-BG composite → bbox crop → `skimage.ssim` | Identical | YES |
| `compute_masked_l1` | `sum(abs(pred-gt)*mask) / (3*sum(mask))` | Identical | YES |
| `compute_iou` | `(pred>0.5 & gt>0.5).sum / (pred>0.5 | gt>0.5).sum` | Identical | YES |
| `extract_foreground_mask` | `any(render < 0.98, axis=2)` | Identical | YES |
| `compute_all_metrics` | All sub-metrics + multi-threshold IoU | Identical | YES |

**Verification method**: Line-by-line diff of function bodies. Both files contain explicit comments: "MUST be identical to" the other file.

### 2.2 Metric Definitions

| Metric | Formula | Range | Higher = Better |
|--------|---------|-------|:---------------:|
| `psnr_gt_masked` | `-10*log10(MSE)` on GT FG pixels | 0-100 dB | YES |
| `psnr_intersection` | Same, but on pred∩GT FG pixels | 0-100 dB | YES |
| `ssim_gt_masked` | SSIM on white-BG composite, bbox crop | 0-1 | YES |
| `l1_gt_masked` | `sum(abs)/3*sum(mask)` on GT FG | 0-1 | NO |
| `iou` | Binary mask IoU | 0-1 | YES |
| `coverage` | GT FG pixels covered by pred FG | 0-1 | YES |
| `pred_precision` | Pred FG pixels overlapping GT FG | 0-1 | YES |

---

## 3. GT Data Source: CRITICAL DIFFERENCE

### 3.1 FL: M5 RGBA PNGs

```
Path: /home/joon/data/preprocessed/FaceLift_mouse/M5/{frame_id}/images/cam_{view:03d}.png
Format: 512×512, RGBA, uint8
Mask: alpha > 127 → binary FG mask
gt_fg_ratio: ~2.33% (small mouse, large BG)
```

- Original preprocessed data from FaceLift pipeline
- Alpha channel = ground truth foreground segmentation
- Native 512×512 resolution

### 3.2 PS: Zarr Datastore

```
Path: data/preprocessed/markerless_mouse_1_nerf/fj5_ds2/images/images.zarr
Format: (3600, 6, 512, 576, 3), uint8
Mask: white-BG extraction (pixel == [255,255,255] → BG) or optional M5 alpha
gt_fg_ratio: ~5.04% (wider FOV, center crop includes more mouse)
```

- Preprocessed from raw data with `fj5_ds2` pipeline
- **No alpha channel** — 3-channel RGB only
- Resolution 512(H)×576(W), center-cropped to 512×512 for FL alignment
- Different preprocessing pipeline than M5

### 3.3 Impact Analysis

| Factor | FL (M5) | PS (zarr) | Impact |
|--------|---------|-----------|--------|
| **Resolution** | 512×512 | 576×512→crop 512 | Slight framing difference |
| **Color space** | RGBA | RGB | Alpha channel absent in PS |
| **FG definition** | Alpha > 127 | White-BG extraction | **2.17x FG ratio gap** |
| **gt_fg_ratio** | 2.33% | 5.04% | PS has larger FG region |
| **Pixel values** | Original RGBA render | Separate preprocessing | May differ for same frame |

### 3.4 Why This Matters

Even with identical metric functions, **different GT images produce different PSNR/IoU values** for the same render:

1. **PSNR_gt_masked**: Computed on GT FG pixels. If GT FG is larger (PS), more pixels are evaluated → different result even for identical renders.
2. **IoU**: Depends on GT mask definition. PS white-BG mask (5.04%) is 2.17x larger than FL alpha mask (2.33%) → PS IoU artificially inflated.
3. **Coverage**: GT FG size directly determines denominator → PS's larger FG makes coverage appear lower.

**Conclusion**: Current cross-model PSNR/IoU numbers are NOT directly comparable.

---

## 4. Mask Definition Analysis

### 4.1 FL: GT Alpha Channel

```python
gt_raw = Image.open(gt_path)  # RGBA
gt_mask = (gt_raw[:, :, 3] > 127).astype(np.float32)
```

- **Source**: Ground truth alpha from rendering pipeline
- **Threshold**: 127 (mid-point of 0-255)
- **Characteristics**: Tight, precise silhouette matching render alpha
- **gt_fg_ratio**: ~2.33% of 512×512 = ~6,100 pixels

### 4.2 PS: White-Background Extraction (Default)

```python
# From zarr (3-channel RGB)
masks_tensor = torch.where(
    images_tensor[..., 0] == 1.0,  # pixel == white
    torch.tensor(0.0),  # BG
    torch.tensor(1.0)   # FG
)
```

- **Source**: RGB image, white = BG
- **Issue**: Any non-pure-white pixel counts as FG → includes semi-transparent edges, shadows
- **gt_fg_ratio**: ~5.04% of 512×512 = ~13,200 pixels

### 4.3 PS: Optional M5 GT Alpha (--m5_gt_dir)

```python
m5_img = Image.open(m5_path)  # RGBA
mask_gt_np_m5 = (m5_img[:, :, 3] > 127).astype(np.float32)
# Resize if resolution mismatch
```

PS eval supports `--m5_gt_dir` option, but this requires resolution matching (576→512) and was not used in the default evaluation.

### 4.4 Mask Comparison (Same Frame, Same View)

| Mask Type | FG Pixels | FG Ratio | Relative |
|-----------|-----------|----------|----------|
| M5 alpha > 127 | ~6,100 | 2.33% | 1.0x (baseline) |
| White-BG extraction | ~13,200 | 5.04% | 2.17x |
| IoU between masks | — | — | ~0.24 |

The two mask definitions only overlap by ~24% (IoU), meaning they define substantially different foreground regions.

---

## 5. Resolution & Framing

### 5.1 FL Pipeline

```
Raw render → 512×512 RGBA PNG (M5 preprocessing)
Eval input → 512×512 (native, no crop)
```

### 5.2 PS Pipeline

```
Raw image → 1152×1024 (original)
Preprocessed → 576×512 (ds=2)
Eval center crop → 512×512 (remove 32px each side)
```

### 5.3 Framing Difference

The 576→512 center crop in PS means the PS evaluation window is slightly narrower than FL's native 512×512. However, since the mouse is centered in frame, this primarily removes background and has minimal impact on FG metrics.

---

## 6. Pred Mask Extraction

Both models use the same function for pred mask extraction:

```python
def extract_foreground_mask(render, threshold=0.98):
    return np.any(render < threshold, axis=2).astype(np.float32)
```

- FL renders: GS-LRM outputs on white background → works correctly
- PS renders: Model outputs RGB + alpha separately → need white-BG composite first

For unified evaluation, PS renders must be composited onto white BG before saving:
```python
render_white = rgb * alpha + (1 - alpha) * 1.0  # white background
```

---

## 7. Solution: Unified Evaluation Pipeline

### 7.1 Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PS Model    │────→│ dump_ps_renders  │────→│ PS renders      │
│  (joon)      │     │ (512×512 PNG)    │     │ (FL format)     │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  FL renders  │────→│ fair_comparison  │────→│ Unified Results │
│  (gpu03)     │     │ .py evaluate_fl  │     │ (same GT,       │
└─────────────┘     │ (M5 RGBA GT)     │     │  same mask,     │
                     └──────────────────┘     │  same metrics)  │
                                              └─────────────────┘
```

### 7.2 Steps

1. **Dump PS renders** (`dump_ps_renders.py` on joon):
   - Load PS model from `output/facelift_compare_5cam/latest`
   - Render test frames 3240-3599, views 1-5
   - Composite onto white BG: `rgb * alpha + (1-alpha)`
   - Center crop 576→512
   - Save as `{frame_id}/render_view_{view:02d}.png` (FL-compatible format)

2. **Evaluate PS renders** (same `fair_comparison.py evaluate_fl` used for FL):
   - `--render_dir ps_renders/samples/`
   - `--gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5`
   - Output: `posesplatter_unified_fair.json`

3. **Evaluate FL renders** (already done):
   - Existing results: `facelift_fair.json` etc.
   - Already uses M5 GT alpha masks

4. **Compare**: `fair_comparison.py compare` with both JSONs

### 7.3 What This Guarantees

| Guarantee | Before | After |
|-----------|--------|-------|
| Same GT images | FL=M5 PNG, PS=zarr | Both = M5 RGBA PNG |
| Same GT mask | FL=alpha>127, PS=white-BG | Both = alpha > 127 |
| Same metric functions | Identical (verified) | Identical (same script) |
| Same resolution | FL=512, PS=576→crop | Both = 512×512 |
| Same pred mask extraction | Identical (verified) | Identical (same script) |

---

## 8. Unified Evaluation Results

> **SUPERSEDED (v8.1)**: B-3 unified results were invalidated by camera parameter mismatch discovery (M5 HFOV=50° vs fj5_ds2 HFOV≈35°). The PSNR gap was primarily from camera geometry, not GT source difference.
>
> **Current status**: Option A (PS retrained on M5 data) resolved this. See `FL_vs_PS_comparison.md` §3 (Tier A) for valid same-camera results.


## 9. GT Source Artifact Analysis

> **Partially superseded**: The -6.96 dB GT source artifact analysis remains valid as a code-level finding (FL RGBA vs PS zarr produce different RGB values for identical images). However, the practical impact is moot since Option A now uses identical data sources for both models.
>
> For the original analysis details, see git history (v2.0, 2026-02-19).


## 10. File Inventory

### Evaluation Scripts

| Script | Server | Purpose |
|--------|--------|---------|
| `fair_comparison.py` | gpu03: `mouse_extensions/scripts/eval/` | FL eval + comparison |
| `fair_test_only_eval.py` | joon: `scripts/eval/` | PS eval (old, zarr-based) |
| `dump_ps_renders.py` | joon: `scripts/eval/` | **NEW**: Dump PS renders to disk |
| `run_unified_eval.sh` | joon | **NEW**: Orchestrate dump + unified eval |

### Result JSONs

| File | Server | Content |
|------|--------|---------|
| `facelift_fair.json` | gpu03: `experiments/comparison/fair/` | FL E2E (best, E2 resume) |
| `posesplatter_fair.json` | gpu03: `baselines/pose_splatter/` | PS (old, zarr GT) |
| `posesplatter_unified_fair.json` | joon: `experiments/unified/` | **NEW**: PS (M5 GT, unified) |
| Various `*_fair.json` | gpu03: `experiments/comparison/tier/` | Tier A/C experiments |

---

## 11. Conclusion

### What We Confirmed
1. **Metric functions are identical** between FL and PS evaluation scripts (verified line-by-line)
2. **GT data source is the dominant confound**: -6.96 dB (83% of total drop) from GT RGB change alone
3. **Mask definition is secondary**: -1.38 dB (17% of total drop) from mask change
4. **Previous 8.5 dB gap was an artifact**: Unified eval shows FL ≈ PS on PSNR_gt (8.20 vs 8.37)

### What Remains Ambiguous
- **No neutral GT exists**: Both M5 and zarr are native to one model — any comparison inherently favors one side
- **PSNR_intersection most reliable**: FL 15.63 vs PS 12.42 (+3.21 dB FL advantage, partially robust to GT format bias)
- **IoU comparison unreliable across GT formats**: 0.824 → 0.247 is entirely mask-definition-driven

### Recommendation for Publication
Use **Tier A** (same input, same GT) as the primary comparison: GS-LRM > PS by +4.22 dB. Tier B comparisons should be presented with full disclosure of GT format bias and the B-1→B-2→B-3 decomposition.

---

*Metric Consistency Analysis v2.0 | 2026-02-19*
