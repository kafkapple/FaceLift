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

## 8. Unified Evaluation Results (ACTUAL)

### 8.1 B-1 → B-2 → B-3 PSNR Decomposition

| Step | GT Source | Mask Source | PSNR_gt | IoU | Change |
|------|-----------|-------------|:-------:|:---:|--------|
| **B-1** | zarr (PS native) | white-BG (5.04%) | 16.71 | 0.824 | baseline |
| **B-2** | zarr (PS native) | **M5 alpha (2.33%)** | 15.33 | 0.317 | **-1.38 dB** (mask only) |
| **B-3** | **M5 RGBA** | M5 alpha (2.33%) | 8.37 | 0.247 | **-6.96 dB** (GT RGB change) |

**Attribution**:
- Mask change contribution: -1.38 dB (**17%** of total 8.34 dB drop)
- GT RGB change contribution: -6.96 dB (**83%** of total drop)

### 8.2 Unified Results (B-3)

| Metric | FL E2E (E2) | PS (unified) | Delta | Winner |
|--------|:---:|:---:|:---:|:---:|
| PSNR_gt_masked | 8.20 | **8.37** | +0.17 | ~Tie |
| PSNR_intersection | **15.63** | 12.42 | **+3.21** | **FL** |
| IoU | **0.518** | 0.247 | **+0.271** | **FL** |
| Coverage | 71.5% | 70.7% | +0.8% | ~Tie |
| gt_fg_ratio | **2.33%** | **2.33%** | **0%** | **Unified** |

### 8.3 Key Insight: "Previous 8.5 dB Gap Was an Artifact"

The B-1 gap (PS 16.71 vs FL 8.20 = -8.51 dB) was **not a real quality difference**. It was dominated by:
1. GT source mismatch (different pixel values for same scene)
2. Mask definition mismatch (2.17x FG ratio gap)

With unified evaluation (B-3), the gap collapses to **+0.17 dB** (statistical tie on PSNR_gt).

---

## 9. GT Source Artifact: Root Cause Analysis

### 9.1 Why -6.96 dB from GT RGB Change Alone?

PS was optimized to reconstruct **fj5_ds2 zarr** images. B-3 evaluates PS renders against **M5 RGBA PNG** images. Same physical scene, same frame, same view — but different preprocessing pipelines:

```
Same raw data → fj5_ds2 pipeline → zarr (576×512, RGB)  → PS training GT
Same raw data → M5 pipeline      → PNG (512×512, RGBA)  → FL training GT, B-3 eval GT
```

Specific pixel-level differences:
1. **Resolution/framing**: zarr 576×512 vs M5 512×512 → center crop coordinates differ
2. **Color processing**: Different normalization, white balance in each pipeline
3. **Alpha compositing**: M5 has precise alpha for FG/BG; zarr uses RGB-only (BG=white)
4. **Anti-aliasing**: Sub-pixel edge treatment differs between pipelines

PS renders match zarr GT well (PSNR_gt=16.71) but mismatch M5 GT (PSNR_gt=8.37) because they were never trained/optimized for M5 pixel values.

### 9.2 Common Misconception: "PS Has Disadvantages"

**FALSE**. PS has multiple structural advantages:

| Factor | FaceLift | Pose-Splatter | Advantage |
|--------|----------|---------------|:---------:|
| **Model type** | Feed-forward (generalization) | Per-scene optimization | **PS** |
| **Input views** | 1 image | 6 GT views | **PS** |
| **Native resolution** | 512×512 | 576×512 (wider FOV) | **PS** |
| **Split** | Train 80%, Test 10% | Same 80:10:10 | **Equal** |
| **Training target** | General priors → M5 finetune | Directly optimized on M5 cameras | **PS** |

PS's poor B-3 results are **NOT** because PS is worse. They are because B-3 evaluates against M5 GT, which is "foreign" to PS (trained on zarr GT).

### 9.3 IoU=0.247 Explanation

PS silhouette is ~2.6× larger than M5 alpha mask:
- PS pred_fg ≈ ~6.0% (trained to match zarr white-BG mask ~5.04%)
- M5 GT_fg = 2.33% (tighter alpha-based mask)
- The mouse IS in the correct position (coverage=70.7%), but PS mask is bloated relative to M5 alpha
- This is an **evaluation protocol mismatch**, not a model failure

### 9.4 Fairness Spectrum

| Evaluation | Favors | Reason |
|------------|--------|--------|
| **B-1** (zarr GT, white-BG mask) | **PS** | PS's own training GT and mask |
| **B-3** (M5 GT, M5 alpha mask) | **FL** | FL's own training GT and mask |
| **Ideal** | Neither | Would require neutral 3rd-party GT |

B-3 is **more methodologically rigorous** than B-1 (same GT/mask/script for both models), but has an inherent FL-favoring bias because M5 is FL's native GT format.

### 9.5 Implications for Publication

Any cross-model comparison must disclose:
1. Each model's training GT differs in preprocessing pipeline
2. Unified eval uses one model's GT format (M5), which inherently favors that model
3. PSNR_intersection is the most robust metric (evaluates only overlapping FG pixels, partially mitigates GT framing differences)
4. IoU/coverage comparisons are **heavily affected** by mask definition alignment

---

## 9. File Inventory

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

*FaceLift vs Pose-Splatter | Metric Consistency Analysis v2.0 | 2026-02-19*
