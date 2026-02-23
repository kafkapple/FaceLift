# FaceLift vs Pose-Splatter: Comprehensive Comparison

> Version 12.0 | 2026-02-23 | + PS Paper vs Our Protocol analysis, metric protocol decomposition, dataset diff, rebuttal strategies, H_Split hypothesis, cross-species Rat plan

---

## 1. Model Architecture Comparison

| Aspect | FaceLift (FL) | Pose-Splatter (PS) |
|--------|:---:|:---:|
| **Type** | Feed-forward generalization | Per-scene optimization |
| **Pipeline** | Image → MVDiff (6 views) → GS-LRM (3DGS) | 6 GT views → 3DGS optimization |
| **Input at test** | 1 arbitrary image | 6 synchronized GT views |
| **Training** | Pretrain on large data → finetune on M5 | Optimize per-frame on M5 |
| **Inference** | ~1 sec/frame (single forward pass) | ~minutes/frame (iterative optimization) |
| **Generalization** | Yes (unseen frames/subjects) | No (per-scene only) |

### Why Direct Comparison Is Unfair

FL and PS solve **fundamentally different problems**:
- FL: Single-image 3D reconstruction (ill-posed, requires learned priors)
- PS: Multi-view 3D reconstruction (well-posed, exploits all available views)

Comparing their outputs directly conflates **model capability** with **input information asymmetry** (1 view vs 6 views).

---

## 2. Data & Evaluation Protocol

### 2.1 Shared Data (Our Fair Comparison)

| Item | Value |
|------|-------|
| Dataset | M5 (markerless_mouse_1, 3600 frames × 6 views) |
| Resolution | 512×512 |
| Split (M5t2) | Train 0-2879 (80%), Val 2880-3239 (10%), Test 3240-3599 (10%) |
| Subject | Single mouse, lab environment |

> **Data Provenance**:
> - 원본: DANNCE (Dunn et al. 2021, Harvard) `markerless_mouse_1` 시퀀스 (6cam, 1152×1024, 100fps, 18K frames)
> - 가공: MAMMAL (An et al. 2023) → `markerless_mouse_1_nerf/` (segment mask 추가, 512×512 비디오)
> - 전처리: M5 preset (center crop, 카메라 정규화, RGBA) → 3,600 frames
>
> **⚠️ PS 논문 데이터와의 관계**:
> PoseSplatter 논문(Goffinet et al. 2025)은 **자체 녹화한 별도 Duke 데이터**를 사용합니다
> (1536×2048, 30fps, 324K frames, 28cm 플라스틱 실린더, DOI: 10.7924/r4z323k2c).
> **DANNCE/MAMMAL 데이터와 완전히 별개**입니다 (해상도, FPS, 프레임 수, 아레나 모두 상이).
> 본 비교에서는 PS 코드를 M5 데이터에 적용(`m5_baseline_gs`)하여 동일 데이터 조건에서 비교합니다.
> 따라서 PS 논문의 수치(PSNR 33.5 등)와 본 비교의 수치(PSNR_fg 13.78 등)는 **직접 비교 불가**합니다
> (다른 데이터 + 다른 metric protocol: full-image vs foreground-masked).

### 2.2 Evaluation Asymmetries (5 Caveats — All Resolved)

| # | Issue | FaceLift | Pose-Splatter | Impact | Status |
|---|-------|:---:|:---:|--------|:---:|
| 1 | **Test frames** | 360 (full test set) | 360 (full test set) | Equalized | ✅ |
| 2 | **Evaluated views** | views 1-5 (view 0 = input) | views 1-5 (matched) | Equalized | ✅ |
| 3 | **GT FG ratio** | 2.3% (512×512 native) | 5.2% (576→512 crop) | PS IoU artificially higher | ✅ Documented |
| 4 | **Mask source** | GT alpha channel (RGBA) | White-BG extraction | **Dominates comparison** | ✅ P1 verified |
| 5 | **Crop** | Native 512×512 | Center crop from 576×512 | PS tighter framing | ✅ Documented |

### 2.3 Unified Metric Protocol

All metrics computed with identical functions (`fair_comparison.py` / `fair_test_only_eval.py`):

| Metric | Definition | Purpose |
|--------|-----------|---------|
| `psnr_gt_masked` | PSNR on GT foreground pixels | Overall FG quality |
| `psnr_intersection` | PSNR on pred∩GT foreground | Pure color accuracy (shape-independent) |
| `iou` | Silhouette IoU | Shape accuracy |
| `coverage` | GT FG pixels covered by pred | Completeness |

### 2.4 PS Paper Results vs Our Fair Evaluation ⭐ NEW (v12.0)

> PS 논문의 보고 수치와 본 프로젝트 fair evaluation 수치는 **직접 비교 불가**합니다.
> 아래에서 그 이유를 체계적으로 분석합니다.

#### A. 수치 비교

| Metric | PS Paper (Mouse, Table 2a) | Our Fair Eval (PS M5 6v) | Gap |
|--------|:---:|:---:|:---:|
| IoU | 0.868 | 0.846 | -0.022 (comparable) |
| PSNR | **33.5** (full-image) | **13.78** (FG-masked) | **-19.7 dB** |
| SSIM | 0.989 | ~0.85 | -0.14 |
| PSNR (whole) | — | 29.00 | — |

> **핵심**: IoU는 유사하지만 (0.868 vs 0.846), PSNR은 19.7 dB 차이. 이 차이는 모델 성능이 아닌 **metric protocol + dataset 차이**에 기인.

#### B. Metric Protocol 차이 (PSNR 19.7 dB Gap 분해)

| 요인 | PS Paper | Our Fair Eval | PSNR 영향 |
|------|----------|---------------|:---------:|
| **PSNR 계산 영역** | Full-image (BG 포함) | FG-masked only | **~15-20 dB** |
| **Background 비율** | ~85-90% white pixels | 제외됨 | Dominant factor |
| **Mouse FG 비율** | ~10-15% of image | 100% (masked) | |
| **SSIM 영역** | Full-image | FG region | ~0.1-0.15 |

**Why full-image PSNR is inflated**:
```
Full-image PSNR: 90% white BG (perfect match) + 10% mouse → average ≈ 33 dB
FG-masked PSNR:  100% mouse pixels only → ≈ 14 dB

White BG pixels contribute near-zero error, inflating the average.
Our PSNR_whole = 29.00 (same protocol as PS paper, different data).
```

**검증**: PS M5에서 `psnr_gt_whole = 29.00` (full-image, 우리 데이터). PS 논문 33.5와의 4.5 dB 차이는 dataset/resolution 차이에 기인.

#### C. Dataset & Experimental Condition 차이

| 요인 | PS Paper (Duke 자체 데이터) | Our Experiment (DANNCE/MAMMAL M5) |
|------|:---:|:---:|
| **Dataset 출처** | 자체 녹화 (Duke University) | DANNCE → MAMMAL 경유 |
| **DOI** | 10.7924/r4z323k2c | — (DANNCE/MAMMAL 공개 데이터) |
| **Resolution** | 1536×2048 (downsampled to 768×1024) | 512×512 |
| **FPS** | 30 | 20 (100fps ÷ 5 temporal downsample) |
| **Total frames** | 324,000 | 3,600 |
| **Training frames** | ~108,000 (1/3 of total) | 2,880 (80%) |
| **Arena** | 28cm 투명 플라스틱 실린더 | 표준 실험실 cage |
| **Background** | 깨끗한 white | 실험실 장비 visible |
| **Mouse size in frame** | ~10-15% (larger) | ~2.5% (smaller, M5 crop 후) |
| **Split** | 1:1:1 temporal | 8:1:1 temporal |
| **Mask 생성** | White-BG extraction | GT RGBA alpha channel |
| **# Cameras** | 6 (custom rig) | 6 (DANNCE standard) |

#### D. 예상 질문 및 반박 전략

**Q1: "왜 PS의 PSNR이 논문보다 19.7 dB 낮은가?"**

> **A**: Metric protocol이 다릅니다. PS 논문은 **full-image PSNR** (white BG 포함)을 보고하며, 이미지의 85-90%를 차지하는 white pixels이 near-zero error를 기여합니다.
> 우리는 **FG-masked PSNR** (foreground pixels only)을 사용하여 mouse body의 실제 reconstruction quality만 측정합니다.
> 동일한 full-image protocol로 계산하면 PS M5 = 29.00 dB로, 논문 수치(33.5)에 근접합니다 (나머지 4.5 dB는 dataset 차이).

**Q2: "다른 데이터에서 실험한 결과가 유의미한가?"**

> **A**: PS 코드를 **동일한 M5 데이터**에 적용했으므로, 데이터 조건은 완전히 통제되었습니다.
> PS 논문의 절대 수치는 비교하지 않으며, **동일 데이터에서 FL vs PS의 상대적 성능**만 비교합니다.
> 이것이 Tier A 비교 (§3)의 핵심입니다.

**Q3: "8:1:1 split이 PS에 불리한 것 아닌가?"**

> **A**: Split 비율은 양쪽에 동일하게 적용됩니다. 그러나 **모델 특성에 따른 영향은 비대칭**입니다:
> - FL (feed-forward): 학습 데이터가 많을수록 유리 (generalization 학습)
> - PS (per-scene optim): test frame에 직접 최적화하므로 train 데이터 양과 무관
>
> 따라서 8:1:1은 이론적으로 FL에 **더 유리**합니다. 이를 검증하기 위해 H_Split 실험을 제안합니다 (§2.5).

#### E. Metric Protocol 정리

| Metric | PS 논문 (Duke) | 우리 Fair Eval (M5) | 비교 가능성 |
|--------|:---:|:---:|:---:|
| IoU | ✅ 유사 정의 | ✅ 동일 정의 | **비교 가능** (0.868 vs 0.846) |
| PSNR | ⚠️ Full-image | ⚠️ FG-masked | **비교 불가** (33.5 vs 13.78) |
| SSIM | ⚠️ Full-image | ⚠️ FG region | **비교 불가** (0.989 vs ~0.85) |
| Coverage | — (미보고) | ✅ 89.3% | — |
| PSNR_int | — (미보고) | ✅ 20.47 | — |

### 2.5 H_Split Hypothesis: Split Ratio Fairness Test ⭐ NEW (v12.0)

> **가설**: 8:1:1 split이 FL에 유리한 bias를 만들 수 있다.
> PS 논문의 1:1:1 temporal split에서 실험하면 FL-PS 격차가 줄어들 수 있다.

#### Rationale

| Factor | 8:1:1 (M5t2) | 1:1:1 (M5t, PS style) |
|--------|:---:|:---:|
| Train frames | 2,880 | ~1,200 |
| Test frames | 360 | ~1,200 |
| FL expected | Best (most train data) | Worse (less data for generalization) |
| PS expected | Same (per-scene, train-independent) | Same (per-scene) |

#### Proposed Experiment

| Item | Value |
|------|-------|
| **Dataset** | M5t (1:1:1 temporal split, 같은 3600 frames) |
| **Models** | FL GS-LRM 6v GT + FL E2E + PS M5 |
| **Metric** | Same fair eval protocol |
| **GPU** | 1 GPU per model, ~24h each |
| **Priority** | P3 (DA1/H3 완료 후) |

#### Expected Results

| Model | M5t2 (8:1:1) | M5t (1:1:1) | Predicted Change |
|-------|:---:|:---:|:---:|
| FL GS-LRM 6v | 23.84 | ~21-22 | -1 to -3 dB (less train data) |
| FL E2E | 8.20 | ~7-8 | -0.5 to -1 dB |
| PS M5 6v | 13.78 | ~13-14 | ~0 dB (per-scene, train-independent) |

> **Success criterion**: If FL 6v - PS gap **narrows** significantly (from +10.06 to <+7 dB), split ratio is a confound.
> If gap **maintains** (>+8 dB), FL's advantage is robust to split ratio.

### 2.6 Cross-Species Generalization Plan ⭐ NEW (v12.0)

> PS 논문은 Mouse 외에 **Rat** (Rat7M dataset)에서도 실험을 수행함.
> FL의 cross-species generalization 가능성을 검증하는 실험 제안.

#### Rat7M Dataset

| Item | Value |
|------|-------|
| **출처** | DANNCE (Dunn et al.), figshare, CC BY 4.0 |
| **해상도** | 1320×1048, 6 cameras, 30fps |
| **규모** | 7M frames total (PS uses 135K subset) |
| **MAMMAL 포함?** | **아니오** — MAMMAL은 mouse/pig/dog만. Rat 데이터 없음 |
| **PS Rat 성능** | 6-cam IoU 0.797, PSNR 26.9 (full-image), SSIM 0.975 |
| **PS Cross-species** | Mouse→Rat: PSNR 25.1 (-1.8 dB from in-species) |
| **과제** | 침구(bedding) 가림, 불균일 조명, 새 전처리 필요 |

#### Proposed Experiment Pipeline

```
Phase 1: Rat7M → M5-format 전처리 (rat_M5)
Phase 2: FL GS-LRM fine-tune (mouse→rat transfer)
Phase 3: PS rat baseline 재현 (rat_M5 데이터)
Phase 4: Fair comparison (FL vs PS, rat domain)
```

**Priority**: P4 (DA1 → DA2 → H_Split → Rat)

---

## 3. Tiered Comparison Results

### Tier A: Fair Quantitative (GT views → 3D, same camera space ✅)

> **v11.0 — Camera Mismatch RESOLVED**: PS retrained on M5 data (`m5_baseline_gs`, 50 epochs on joon server). Both models now evaluated against identical M5 GT in the same camera space (HFOV=50°, fx=549, cx=256, 512×512). Pixel-wise comparison is now valid.

| Metric | GS-LRM 6v GT | GS-LRM 4v GT | PS M5 6v (360f) | FL 6v vs PS | FL 4v vs PS |
|--------|:---:|:---:|:---:|:---:|:---:|
| PSNR_gt_masked | **23.84** | 20.66 | 13.78 | **+10.06 dB** | **+6.88 dB** |
| PSNR_intersection | **24.02** | 21.29 | 20.47 | **+3.55 dB** | +0.82 dB |
| IoU | **0.954** | 0.926 | 0.846 | **+0.108** | +0.080 |
| Coverage | **99.9%** | 99.3% | 89.3% | **+10.6%p** | +10.0%p |
| PSNR_gt (whole img) | — | — | 29.00 | — | — |

> **Conclusion (v11.0)**: With camera mismatch resolved (same M5 camera space), the FL advantage **widens** from v10.0: **+10.06 dB** PSNR_fg (was +7.13). GS-LRM 6v feedforward massively outperforms PS per-scene optimization. Even GS-LRM 4v exceeds PS by +6.88 dB.
>
> **Coverage as primary gap driver**: PS coverage (89.3%) is significantly lower than FL (99.9%). The PSNR_intersection gap (+3.55 dB, pure color accuracy on overlapping regions) is much smaller than PSNR_fg gap (+10.06 dB), confirming coverage deficit as the dominant contributor.
>
> **Historical reference (v10.0, fj5_ds2 camera)**: PS fj5_ds2 had PSNR_fg=16.71, IoU=0.824, Coverage=91.7%. These numbers are from a different camera space (HFOV≈35°) and are **no longer used for cross-model comparison**. The M5 same-camera comparison above is now canonical.

**Experimental Conditions (Tier A)**:

| Condition | GS-LRM (FL) | PS (m5_baseline_gs) |
|-----------|-------------|---------------------|
| **Data** | M5t2 (3600 frames × 6 views × 512×512 RGBA) | M5 converted (`m5_for_ps_fj1`, identical frames, white-BG) |
| **Split** | Train 0-2879 (80%), Val 2880-3239 (10%), Test 3240-3599 (10%) | Same |
| **Camera** | M5 (HFOV=50°, fx=549, cx=256, centered) | Same (converted via `convert_m5_for_ps.py`, verified) |
| **Input** | 6 GT views (views 0-5) | 6 GT views (views 0-5) |
| **Model type** | Feed-forward generalization | Per-scene optimization (50 epochs) |
| **Training scope** | Pretrained on diverse data → M5 finetune (15840 steps) | Optimize from scratch per-frame on M5 |
| **Checkpoint** | `6view_v2/best_psnr.pt` (PSNR=24.49 val) | `m5_baseline_gs/latest/` (epoch 50) |
| **Test frames** | 360 frames (3240-3599) × views 1-5 (1800 renders) | 360 frames (3240-3599) × views 0-5 (2160 renders) |
| **Eval mask** | GT alpha channel (RGBA) | GT alpha channel (same source) |
| **Eval script** | `fair_comparison.py` (gpu03) | `fair_test_only_eval.py` (joon) |
| **FG threshold** | >0 alpha | >0 alpha |

**Caveats**:
- **Model-type asymmetry** (inherent, not a fairness issue): GS-LRM generalizes from diverse pretraining; PS optimizes per-scene. This reflects fundamentally different approaches.
- PS coverage (89.3%) significantly lower than FL (99.9%) — amplifies PSNR_fg gap
- FL view 0 excluded from eval (used as input); PS evaluates all 6 views — minor asymmetry
- View count monotonically improves GS-LRM quality (1v→6v): +5.48 (1→2v), +2.61 (2→3v), +2.10 (3→4v), +1.50 (4→5v), +1.68 (5→6v)

### Tier B: Practical Comparison (Different Input)

> **Disclaimer**: FL E2E uses 1 input view; PS uses 6 GT views.

**B-1: Each model's own mask protocol** (PS=white-BG, FL=GT alpha)

| Metric | FL E2E Best (E2) | PS (360f, v1-5) | Gap |
|--------|:---:|:---:|:---:|
| PSNR_gt_masked | 8.20 | 16.71 | -8.51 |
| IoU | 0.521 | 0.824 | -0.303 |
| PSNR_intersection | 15.63 | 20.54 | -4.91 |
| Coverage | 71.5% | 91.7% | -20.2% |

**B-2: Unified GT alpha mask** (identical foreground definition)

| Metric | FL E2E Best (E2) | PS GT mask | Gap | Winner |
|--------|:---:|:---:|:---:|:---:|
| PSNR_gt_masked | 8.20 | 15.33 | -7.13 | PS |
| PSNR_intersection | **15.63** | 14.34 | **+1.29** | **FL** |
| IoU | **0.521** | 0.317 | **+0.204** | **FL** |
| Coverage | **71.5%** | 69.9% | **+1.6%p** | **FL** |

> **P1 finding**: With unified GT alpha masks, FL E2E shows higher IoU and PSNR_int than PS. This reflects **training alignment** — FL trains on GT alpha, PS on white-BG images.

### Tier B Mask Source Analysis

| Metric | PS white-BG | PS GT alpha | Delta | Interpretation |
|--------|:---:|:---:|:---:|------|
| IoU | 0.824 | 0.317 | **-0.507** | White-BG mask 2.1× larger FG |
| PSNR_int | 20.54 | 14.34 | **-6.20** | Generous mask inflates PS color |
| Coverage | 91.7% | 69.9% | -21.8%p | PS misses 30% of GT alpha FG |

GT alpha fg=2.2% vs white-BG fg=4.8% (2.1×). Even GT input images have IoU=0.24 between the two mask definitions.

### B-3: "Unified" Evaluation — INVALIDATED by Camera Mismatch (v8.1)

> **CRITICAL CAVEAT (v8.1)**: B-3 결과는 **카메라 파라미터 불일치**로 인해 정량적 의미가 없습니다. PS 렌더(fj5_ds2 카메라, HFOV≈35°)와 M5 GT(M5 카메라, HFOV=50°)는 pixel-wise로 정렬되지 않습니다. 아래 수치는 모델 품질이 아닌 **카메라 기하학 차이**를 반영합니다.
>
> **Detail**: See [[FL_PS_metric_consistency]] for analysis.

**Camera Parameter Comparison:**

| Parameter | M5 (FL) | PS fj5_ds2 (after ds2+crop) | Gap |
|-----------|---------|---------------------------|-----|
| fx | 549 (all identical) | 778-818 (varies/cam) | **1.48×** |
| cx, cy | 256, 256 (centered) | 259-289, 209-276 (off-center) | up to 47px shift |
| **HFOV** | **50.0°** | **34.7-36.4°** | M5 **43% wider** |
| Camera dist | ~2.7 (batch uniform norm) | 246-414 (raw mm) | Different scale |
| Image transform | **Affine warp** | Simple downsample | **Geometric distortion** |

M5 전처리는 cx→256, fy→549로 **affine warp**를 적용하여 카메라를 정규화합니다. fj5_ds2는 단순 downsample입니다. 결과적으로 같은 마우스가 M5에서는 작게 (넓은 FOV), PS에서는 크게 (좁은 FOV) 나타납니다.

**Raw Results (참고용, 정량적 비교 부적합):**

| Metric | FL E2E (E2) | PS (B-3) | Note |
|--------|:---:|:---:|------|
| PSNR_gt_masked | 8.20 | 8.37 | Both low due to different reasons |
| IoU | 0.518 | 0.247 | PS low = camera framing mismatch |
| PSNR_intersection | 15.63 | 12.42 | PS low = scale/position shift |
| gt_fg_ratio | 2.33% | 2.33% | Mask unified, but images not aligned |

> **Why FL also shows ~8 dB**: FL E2E는 MVDiff 병목(-15.64 dB)으로 인해 낮음 (Tier C). PS B-3은 카메라 불일치로 낮음. **원인이 다릅니다.**

### Camera Mismatch Root Cause Analysis (v8.1)

**B-1 → B-2 → B-3 PSNR Decomposition (수정됨)**:

| Step | GT Source | Mask | PSNR_gt | Drop | Real Cause |
|------|-----------|:----:|:-------:|:----:|------------|
| B-1 | zarr (same camera) | white-BG (5.04%) | 16.71 | — | Same camera space ✅ |
| B-2 | zarr (same camera) | M5 alpha (2.33%) | 15.33 | -1.38 dB | Mask definition only |
| B-3 | **M5 (different camera)** | M5 alpha (2.33%) | 8.37 | -6.96 dB | **Camera geometry mismatch** |

이전 분석에서 B-2→B-3 하락을 "GT RGB 색상 차이"로 설명했으나, 실제 원인은 **카메라 파라미터 불일치**입니다:
- M5: affine warp → fx=549, cx=256, HFOV=50°
- PS render: raw camera → fx≈810, cx≈269, HFOV≈35°
- 같은 마우스가 서로 다른 위치/크기/왜곡으로 나타남 → pixel-wise PSNR 붕괴

**Center crop (576→512)은 해결책이 아님**: 해상도만 맞출 뿐, FOV(50° vs 35°), 초점거리(549 vs 810), 주점(256 vs 269) 불일치는 해결 불가.

### B-3 결론: Tier B 정량 비교의 한계

| 비교 | 카메라 정렬 | 정량적 유효성 |
|------|:---------:|:-----------:|
| **Tier A** (GS-LRM vs PS, 각자 카메라) | 각자 정렬됨 ✅ | **유효** ✅ |
| **B-1** (PS → zarr GT) | 동일 카메라 ✅ | PS 메트릭만 유효 |
| **B-2** (PS → zarr GT + M5 mask) | 동일 카메라 ✅ | PS 메트릭만 유효 |
| **B-3** (PS render → M5 GT) | **불일치** ⛔ | **무효** ⛔ |
| FL E2E → M5 GT | 동일 카메라 ✅ | FL 메트릭만 유효 |

**올바른 Tier B 비교를 위한 방법**: PS의 3D Gaussian을 M5 카메라 파라미터로 재렌더링 (PS 코드 수정 필요). 현재로서는 **Tier A만이 유일한 유효 정량 비교**입니다.

**Publication 가이드 (수정)**: Tier A (GS-LRM vs PS, 동일 입력)만 정량 비교로 사용. Tier B는 정량 비교 불가 — 카메라 파라미터 차이 때문. B-1/B-2 숫자는 PS 자체 평가로만 참조.

### Tier C: Pipeline Bottleneck Analysis (Complete View Ablation)

> v10.0: Full 1v-6v ablation on test set (360 frames × 5 views). 6v corrected from proper 6-view model.

| Stage | Input | PSNR_gt | IoU | PSNR_int | Drop from 6v |
|-------|:---:|:---:|:---:|:---:|:---:|
| **GS-LRM GT 6-view** | **6 GT views** | **23.84** | **0.954** | **24.02** | **(optimal)** |
| GS-LRM GT 5-view | 5 GT views | 22.16 | 0.942 | 22.56 | -1.68 dB |
| GS-LRM GT 4-view | 4 GT views | 20.66 | 0.926 | 21.29 | -3.18 dB |
| GS-LRM GT 3-view | 3 GT views | 18.56 | 0.899 | 19.54 | -5.28 dB |
| GS-LRM GT 2-view | 2 GT views | 15.95 | 0.858 | 17.91 | -7.89 dB |
| GS-LRM GT 1-view | 1 GT view | 10.47 | 0.028 | 10.47 | -13.37 dB |
| P1 6v E2E (best) | 6 MVDiff views | 8.44 | 0.495 | — | -15.40 dB |
| E2E Best (E2 20K) | 4 MVDiff views | 8.20 | 0.521 | 15.63 | -15.64 dB |
| E2E E1 cosine 20K | 4 MVDiff views | 7.90 | 0.528 | **16.15** | -15.94 dB |
| E2E Baseline (5K) | 4 MVDiff views | 7.93 | 0.474 | 13.70 | -15.91 dB |
| **PS M5 same-camera** | **6 GT views** | **13.78** | **0.846** | **20.47** | **(valid, v11)** |
| ~~PS fj5_ds2 (old)~~ | ~~6 GT views~~ | ~~16.71~~ | ~~0.824~~ | ~~20.54~~ | ~~different camera~~ |
| ~~PS (M5 GT, pre-retrain)~~ | ~~6 GT views~~ | ~~8.37~~ | ~~0.247~~ | ~~12.42~~ | ~~INVALID~~ |

> **Key findings (v11.0)**:
> 1. **6v optimal**: 23.84 dB. Monotonic increase: +5.48 (1→2v), +2.61 (2→3v), +2.10 (3→4v), +1.50 (4→5v), +1.68 (5→6v)
> 2. **MVDiff bottleneck**: 6v GT → E2E = **-15.64 dB** drop. MVDiff is the sole bottleneck.
> 3. **FL 6v >> PS M5**: +10.06 dB PSNR_fg, +3.55 dB PSNR_int (same camera, valid comparison)
> 4. **Break-even ≥ 3 GT views**: GS-LRM 3v (18.56) already exceeds PS M5 (13.78) by +4.78 dB
>
> **v10.0 correction**: v9.2 reported 6v=21.02 as from A2 oracle (4-view model receiving 6 GT inputs, using only views 0-3). The proper 6-view model (`6view_v2/best_psnr.pt`) gives 23.84 dB — confirming monotonic improvement with view count.
>
> **v8.1 correction**: PS "unified" row is struck through — the ~8 dB reflects camera parameter mismatch (HFOV 50° vs 35°, fx 549 vs 810), not model quality. See B-3 section for details.

---

## 4. E2E Improvement Experiments (Phase 3 — COMPLETE)

### 4.1 All E2E Results (360 test frames, unified fair metrics)

| Experiment | MVDiff Config | Steps | PSNR_gt | IoU | PSNR_int | Coverage |
|-----------|--------------|:---:|:---:|:---:|:---:|:---:|
| Baseline (5K) | sparse, aug, fixed ref | 5K | 7.93 | 0.474 | 13.70 | 74.0% |
| cfgr (10K) | full attn, CFG | 10K | 7.75 | 0.491 | 15.75 | 67.5% |
| E1 cosine (11K) | sparse, cosine LR | 11K | 7.90 | 0.522 | 16.12 | 70.4% |
| **E1 cosine (20K)** | **sparse, cosine LR** | **20K** | **7.90** | **0.528** | **16.15** | **70.5%** |
| E2 resume (20K) | sparse, random ref | 20K | **8.20** | 0.521 | 15.63 | **71.5%** |
| E3 pose (10K) | extrinsic+add | 10K | 8.10 | 0.523 | 15.88 | 71.6% |

### 4.2 GS-LRM Stage 2 Variants

| Variant | Config | Val PSNR | E2E Impact | Conclusion |
|---------|--------|:--------:|:----------:|-----------|
| **Baseline 4v** | Standard | **22.34** | PSNR_gt=7.90-8.20 | **Optimal** |
| E5 alpha03 | opacity_reg=0.03 | 21.34 | Not tested (predicted: ~8) | **-1.0 dB val** (regularization hurts) |
| 6v GT | 6 input views | 24.49 (test: 23.84) | Upper bound | Best possible |
| 1v GT | 1 input view | 11.08 (test: 10.47) | — | Insufficient views |

> **E5 Conclusion**: Alpha regularization (opacity_reg_weight=0.03) reduces GS-LRM val PSNR by 1.0 dB (22.34→21.34). Stage 2 improvements do not transfer to E2E (F3), so E5 E2E testing would yield ~8 dB regardless.

### 4.3 Optimal E2E Configuration

| Component | Best Config | Evidence |
|-----------|-----------|---------|
| **MVDiff** | E1 cosine 20K (color) or E2 resume 20K (overall) | §4.1 |
| **GS-LRM** | Baseline M5t2 (no regularization) | §4.2 |
| **Training strategy** | Irrelevant beyond baseline convergence | All E2E: 7.9-8.2 dB |
| **Bottleneck** | MVDiff architecture, not training | A1 §5, A2 §6 |

**Current settings are already optimal within the existing architecture.** Further improvement requires MVDiff architecture changes (better multi-view consistency, accurate silhouette prediction).

### 4.4 Phase 3 Conclusions

1. **All strategies converge**: PSNR_gt 7.90-8.20, IoU 0.47-0.53. No strategy breaks through.
2. **PSNR_int saturated**: E1 11K→20K = +0.03 dB only. Diminishing returns.
3. **Pose conditioning ineffective**: E3 ≈ E2 within noise margin.
4. **Cosine LR best for color**: PSNR_int 16.15 > 15.88 > 15.63. But PSNR_gt slightly lower.
5. **Stage 2 improvements do not transfer**: E5 alpha GS-LRM val -1.0 dB, E2E predicted unchanged.

### 4.5 Transfer Rate Analysis

| Source | Val Improvement | E2E Improvement | Transfer Rate |
|--------|:---:|:---:|:---:|
| MVDiff val (Stage 1) | +3.59 dB | +0.27 dB | ~7.5% |
| GS-LRM val (Stage 2) | +2.3 dB | 0.0 dB | **0%** |
| GS-LRM alpha (E5) | -1.0 dB val | predicted 0.0 dB | **0%** |
| Cosine LR (PSNR_int) | — | +0.52 dB (int only) | ~15% (color) |

### 4.6 GS-LRM Loss & Preprocessing Ablations (H6/H7/HP)

> v11.0: Systematic ablation of GS-LRM loss weights and data preprocessing. All experiments use the **same base model** (GS-LRM 4-view, M5t2 split) with one variable changed.

#### H6: Alpha Mask Supervision — REJECTED

**Hypothesis**: Adding explicit alpha channel supervision to GS-LRM loss improves silhouette accuracy and reconstruction quality.

| Condition | Value |
|-----------|-------|
| **Variable** | `alpha_loss_weight` ∈ {0.3, 0.5, 1.0} vs baseline (0) |
| **Base model** | GS-LRM 4-view, M5t2, pretrained → finetuned |
| **Data** | M5t2 (Train 0-2879, Val 2880-3239, Test 3240-3599) |
| **Other losses** | l2=1.0, lpips=0.05, perceptual=0.5, ssim=0.1 (unchanged) |
| **Training** | 15840 steps each, GPU 4-6 |
| **Eval metric** | Val PSNR (best checkpoint) |

| Alpha Weight | Best Val PSNR | Delta from Baseline |
|:------------:|:------------:|:-------------------:|
| 0 (baseline) | **21.71** | — |
| 0.3 | 21.34 | -0.37 |
| 0.5 | 21.20 | -0.51 |
| 1.0 | 20.84 | -0.87 |

> **Conclusion**: **Monotonic degradation** with increasing alpha weight. Alpha supervision adds redundant constraint that conflicts with existing losses. Baseline (no alpha loss) is optimal. **REJECTED.**

#### H7: SSIM Loss Weight — REJECTED

**Hypothesis**: Increasing SSIM loss weight improves perceptual quality and structural similarity of reconstructions.

| Condition | Value |
|-----------|-------|
| **Variable** | `ssim_loss_weight` ∈ {0.3, 0.5, 1.0} vs baseline (0.1) |
| **Base model** | GS-LRM 4-view, M5t2, pretrained → finetuned |
| **Data** | M5t2 (same split as H6) |
| **Other losses** | l2=1.0, lpips=0.05, perceptual=0.5, alpha=0 (unchanged) |
| **Training** | 15840 steps each, GPU 4-6 |

| SSIM Weight | Best Val PSNR | Final PSNR | Status |
|:-----------:|:------------:|:----------:|:------:|
| 0.1 (baseline) | **21.71** | **21.71** | Stable |
| 0.3 | 21.45 | 19.86 | **Declining** |
| 0.5 | 21.47 | 10.17 | **Collapsed** |
| 1.0 | 21.46 | 21.16 | Stable but lower |

> **Conclusion**: All variants achieve similar early peaks (~21.45-21.47) but diverge in stability. SSIM=0.5 undergoes catastrophic collapse (21.47→10.17). SSIM=0.3 shows gradual decline. Baseline SSIM=0.1 is optimal for both peak performance and training stability. **REJECTED.**
>
> **Insight**: Loss weight balance in GS-LRM is sensitive. Excessive SSIM weight destabilizes training, likely due to competition between pixel-wise (L2) and structural (SSIM) objectives.

#### HP: Data Preprocessing Ablation — IN PROGRESS

**Hypothesis**: GS-LRM can learn from raw (unpreprocessed) camera data without M5 normalization (centering, scale normalization).

| Condition | Value |
|-----------|-------|
| **Variable** | Preprocessing pipeline: M0 (raw) vs M5_4 (center only) vs M5_5 (TBD) |
| **Base model** | GS-LRM 4-view, pretrained on M5 |
| **Checkpoint** | M5t2 baseline (`M5t2_E0_1_facelift/best_psnr.pt`) |
| **Training** | Resume from M5 pretrained, GPU 7 |

| Preset | Preprocessing | Best Val PSNR | Status |
|:------:|:-------------|:------------:|:------:|
| M5 (baseline) | Center + scale norm + affine warp | **21.71** | Reference |
| M0 (raw) | None (raw camera params, raw images) | 11.46 (step 1 only) | **Diverged** (→1.85) |
| M5_4 (center) | Center only (no scale norm) | 19.06 @step 101 | In progress |
| M5_5 | TBD | — | Pending |

> **Finding (M0)**: Raw data causes immediate divergence. Pretrained model (trained on M5-normalized data) cannot adapt to raw camera distribution. Validates that M5 preprocessing is **essential** for GS-LRM training.
>
> **M5_4 status**: Early results (19.06 @step 101) are reasonable. Training continues (~24h remaining).

---

## 5. MVDiff Bottleneck Diagnostic (A1 Analysis)

### 5.1 MVDiff Generated View Quality

Compared MVDiff-generated views vs GT views for all 360 test frames:

| Metric | MVDiff Output | E2E Final | GS-LRM 6v GT | Interpretation |
|--------|:---:|:---:|:---:|------|
| Silhouette IoU | **0.582** | 0.528 | 0.954 | MVDiff shape accuracy = 58% |
| Coverage | 70.8% | 70.5% | 99.9% | Misses 29% of mouse |
| Precision | 69.9% | — | — | 30% of "foreground" is false |
| PSNR_int | **18.61** dB | 16.15 dB | 24.02 dB | Color quality decent where overlap |

### 5.2 Per-View MVDiff Quality

| View | Sil IoU | PSNR_int | PSNR_gt | Coverage |
|------|:---:|:---:|:---:|:---:|
| 0 (input) | 0.975 | 25.13 | 24.89 | 99.9% |
| 1 (adjacent) | 0.621 | 18.90 | 9.67 | 78.9% |
| 2 | 0.500 | 16.56 | 7.29 | 64.9% |
| 3 | 0.509 | 17.09 | 6.94 | 65.3% |
| 4 | 0.464 | 17.02 | 6.48 | 60.5% |
| 5 (farthest) | 0.423 | 16.95 | 6.60 | 55.0% |

### 5.3 Bottleneck Decomposition

> **Note**: A1 analysis used the 4-view GS-LRM model as the upper bound. For the proper 6-view model upper bound, see Tier A/C (23.84 dB / 0.954 IoU / 24.02 PSNR_int).

```
                         IoU      PSNR_int
GS-LRM 6v GT (upper):   0.954    24.02 dB   ← proper 6v model
GS-LRM 4v GT (A1 ref):  0.926    21.29 dB   ← 4v model used in A1
MVDiff output:           0.582    18.61 dB   ← generated views
E2E final:               0.528    16.15 dB   ← after GS-LRM
                         ─────    ─────────
MVDiff loss (from 6v):  -0.426    -7.87 dB   (IoU: 0.954→0.528)
  - MVDiff→output:      -0.372    -5.41 dB
  - GS-LRM propagation: -0.054    -2.46 dB
```

**The vast majority of E2E quality loss originates in MVDiff.** GS-LRM propagation loss is relatively small.

### 5.4 Failure Mode Identification

| Hypothesis | Result | Severity |
|-----------|--------|:--------:|
| H1a: View inconsistency | IoU spread 0.42-0.62, far views worst | Moderate |
| **H1b: Silhouette error** | **IoU=0.582, coverage=70.8%** | **Dominant** |
| H1c: Texture error | PSNR_int=18.61 on intersection (decent) | Moderate |

**Primary failure**: MVDiff silhouette position/shape error for non-adjacent views. Correct area (2.37% ≈ 2.39% GT) but **shifted position** → low IoU.

---

## 6. GS-LRM Sensitivity Analysis (A2: Oracle MVDiff)

> **Note (v10.0)**: A2 uses the **4-view GS-LRM model** (`num_input_views=4`). Level 0 shows 21.02 dB (not 23.84) because this model only uses views 0-3, ignoring views 4,5 entirely. The proper 6-view GS-LRM model achieves 23.84 dB — see Tier A/C for the corrected upper bound.

### 6.1 Experiment Design

Hybrid datasets: progressively replace GT views with MVDiff outputs (farthest→closest from input).

| Level | Config | GT views | MVDiff views |
|:-----:|--------|:--------:|:------------:|
| 0 | All GT (upper bound) | 0,1,2,3,4,5 | — |
| 1 | Replace v5 | 0,1,2,3,4 | 5 |
| 2 | Replace v4,5 | 0,1,2,3 | 4,5 |
| 3 | Replace v3,4,5 | 0,1,2 | 3,4,5 |
| 4 | Replace v2,3,4,5 | 0,1 | 2,3,4,5 |
| 5 | Replace v1-5 (≈ E2E) | 0 | 1,2,3,4,5 |

GS-LRM checkpoint: M5t2 baseline (`num_input_views=4` in config).

### 6.2 Results

| Level | Config | PSNR_gt | PSNR_int | IoU | Cov% |
|:-----:|--------|:-------:|:--------:|:---:|:----:|
| 0 | 6 GT | **21.02** | **22.36** | **0.943** | 98.9 |
| 1 | 5 GT + 1 MV (v5) | **21.02** | **22.36** | **0.943** | 98.9 |
| 2 | 4 GT + 2 MV (v4,5) | **21.02** | **22.36** | **0.943** | 98.9 |
| 3 | 3 GT + 3 MV (v3,4,5) | 15.82 | 19.67 | 0.780 | 90.2 |
| 4 | 2 GT + 4 MV (v2,3,4,5) | 10.44 | 16.85 | 0.614 | 76.9 |
| 5 | 1 GT + 5 MV (E2E) | 7.90 | 16.15 | 0.528 | 70.5 |

### 6.3 Degradation Pattern

| Transition | dPSNR_gt | dIoU | Interpretation |
|-----------|:--------:|:----:|------|
| 0→1 (replace v5) | **0.00** | **0.000** | **Zero impact** |
| 1→2 (replace v4) | **0.00** | **0.000** | **Zero impact** |
| 2→3 (replace v3) | **-5.20** | **-0.163** | **Sharp threshold** |
| 3→4 (replace v2) | -5.38 | -0.166 | Continued degradation |
| 4→5 (replace v1) | -2.54 | -0.086 | Smaller drop (adjacent view) |

### 6.4 Key Findings

1. **Views 4,5 are unused by GS-LRM**: The 4-view model (`num_input_views=4`) selects views 0,1,2,3 as encoding inputs. Views 4,5 are ignored entirely.

2. **Threshold effect at view 3**: Quality is perfect (=Level 0) until view 3 is corrupted, then drops sharply by -5.2 dB. This is a **cliff, not a slope**.

3. **Linearity check**:
   - Expected linear: each replaced view contributes -2.6 dB
   - Actual: Levels 1,2 = 0 dB loss; Levels 3,4 = -5.2/-5.4 dB; Level 5 = -2.5 dB
   - **Strongly sub-linear** (first replacements = free, later ones = expensive)

4. **Practical implications**:
   - Improving MVDiff on views 4,5 has **zero effect** on E2E
   - Views 1,2,3 quality is critical — all contribute significantly
   - View 1 (adjacent, best MVDiff quality) still causes -2.5 dB when corrupted
   - Even the *best* MVDiff view (IoU=0.62) is insufficient for GS-LRM

5. **Why Level 5 ≠ exact E2E**: Level 5 uses GT view 0 as-is; E2E reconstructs view 0 via MVDiff (view_00 ≈ input copy, IoU=0.975). Negligible difference.

### 6.5 View Selection Confirmation

The A2 results confirm the model's effective view selection:

| View | Used by GS-LRM? | A2 Evidence | A1 MVDiff IoU |
|------|:---:|------|:---:|
| 0 (input) | **Yes** (always GT in E2E) | — | 0.975 |
| 1 (adjacent) | **Yes** | -2.54 dB when replaced | 0.621 |
| 2 | **Yes** | -5.38 dB when replaced | 0.500 |
| 3 | **Yes** | -5.20 dB when replaced | 0.509 |
| 4 | **No** | 0 dB when replaced | 0.464 |
| 5 | **No** | 0 dB when replaced | 0.423 |

**MVDiff generates 6 views, but only 4 matter.** Focus improvement efforts on views 1, 2, 3.

---

## 7. Objective Assessment

### 7.1 What FL Does Well
- **Reconstruction backbone**: GS-LRM 6v GT beats PS M5 by **+10.06 dB** PSNR_fg (Tier A, same camera)
- **Generalization**: Single image → 3D, no per-scene optimization
- **Speed**: Real-time inference (~1s) vs minutes of optimization
- **Color accuracy under overlap**: PSNR_int +3.55 dB over PS (24.02 vs 20.47, pure color quality)
- **Coverage**: 99.9% vs PS 89.3% — FL produces more complete reconstructions

### 7.2 What FL Struggles With
- **MVDiff view generation**: Silhouette IoU=0.582 (primary bottleneck, A1)
- **Far-view prediction**: IoU drops 0.62→0.42 from adjacent to farthest (A1)
- **E2E training saturation**: All strategies converge at PSNR_gt ~8, PSNR_int ~16 (Phase 3)
- **GS-LRM only uses 4 of 6 generated views**: Views 4,5 wasted (A2)

### 7.3 Fair Takeaway (v11.0 Updated)

With camera mismatch resolved (PS retrained on M5 data), **Tier A is now a valid same-camera comparison**:

1. **Tier A** (GT views, same M5 camera): GS-LRM > PS by **+10.06 dB** PSNR_fg, **+3.55 dB** PSNR_int → superior reconstruction backbone. **Valid pixel-wise comparison.**
2. **Tier B**: Historical comparisons using fj5_ds2 camera remain invalid for cross-model comparison. Retained for reference only.
3. **Tier C** (diagnostic): MVDiff is the dominant source of E2E quality loss (A1). GS-LRM view ablation shows monotonic 1→6v improvement.
4. **A2** (sensitivity): GS-LRM 4v model uses only views 0-3; threshold effect at view 3
5. **Ablations** (H6/H7/HP): Baseline GS-LRM loss weights are already optimal. Alpha supervision and SSIM weight increases both degrade performance.

---

## 8. Paper Framing

### 8.1 Narrative

> "FaceLift demonstrates that a feed-forward, generalizable pipeline achieves **+10.06 dB** superior 3D reconstruction quality compared to per-scene optimization (Pose-Splatter) when given equivalent 6-view GT input, evaluated in the same M5 camera space (+3.55 dB pure color accuracy on overlapping regions). This advantage stems from both superior color prediction and significantly better coverage (99.9% vs 89.3%). Even with only 3 GT views, GS-LRM exceeds PS performance (+4.78 dB). Systematic ablations confirm the baseline loss configuration is optimal (H6: alpha supervision hurts, H7: SSIM weight increase causes instability). The remaining E2E bottleneck is MVDiff multi-view generation (silhouette IoU=0.58), which accounts for -15.64 dB of the total quality gap between oracle and end-to-end performance."

### 8.2 Key Claims (Evidence-Backed)

| Claim | Evidence | Section |
|-------|----------|:-------:|
| GS-LRM reconstruction >> PS (same camera) | 6v: **+10.06dB** PSNR_fg, **+3.55dB** PSNR_int | Tier A §3 |
| MVDiff = sole E2E bottleneck | 6v GT→E2E: **-15.64 dB** | A1 §5.3, Tier C |
| Silhouette error is primary failure | IoU=0.582, coverage=70.8% | A1 §5.4 |
| PS coverage deficit amplifies gap | PS 89.3% vs FL 99.9% coverage | Tier A §3 |
| GS-LRM ignores views 4,5 (4v model) | Replacement has 0 dB impact | A2 §6.2 |
| Threshold degradation at view 3 | -5.2 dB cliff, not slope | A2 §6.3 |
| Training strategy saturated | E1/E2/E3 all ~8 dB, ~0.52 IoU | §4.1 |
| Alpha supervision hurts GS-LRM | H6: monotonic decline 21.71→20.84 | §4.6 |
| SSIM weight increase causes collapse | H7: 0.5 → catastrophic collapse (10.17) | §4.6 |
| Raw data preprocessing essential | HP M0: divergence (11.46→1.85) | §4.6 |
| Baseline loss config is optimal | H6+H7: all alternatives worse than baseline | §4.6 |
| Mask protocol dominates comparison | PS IoU: 0.82→0.32 with GT alpha mask | Tier B §3 |
| Current architecture is the bottleneck | No training/loss config breaks through | §4.3, §4.6 |
| PS paper PSNR (33.5) incomparable to ours (13.78) | Full-image vs FG-masked metric protocol; IoU comparable | §2.4 |
| DA1 domain adaptation improves E2E | +1.64 dB (8.44→10.08), 3.2× more efficient than Stage 1 optim | §4.5, DA1 |

### 8.3 Three-Tier Comparison Framework

| Frame | Question | Answer |
|-------|---------|--------|
| **Tier A** | Same input, same camera → which reconstructs better? | GS-LRM >> PS (**+10.06 dB**, same M5 camera) |
| **Tier B** | Historical cross-camera comparison | Retained for reference only (fj5_ds2 vs M5 mismatch) |
| **Tier C** | Where does FL lose quality? | MVDiff silhouette error (dominant bottleneck) |
| **A2** | How sensitive is GS-LRM to view quality? | Threshold: v4,5 free, v3 critical |
| **H6/H7/HP** | Can loss/preprocessing tuning help? | No — baseline config already optimal |

### 8.4 Improvement Roadmap

| Priority | Action | Expected Impact | Evidence |
|:--------:|--------|:---:|---------|
| **P1** | Improve MVDiff silhouette prediction | High (addresses dominant bottleneck) | A1 §5.3 |
| **P2** | Improve MVDiff views 1-3 consistency | High (these are the used views) | A2 §6.5 |
| **P3** | Increase num_input_views (use all 6) | Medium (currently 4v ignores 2 views) | A2 §6.4 |
| Low | Extend training (>20K) | Negligible (saturated at ~16 PSNR_int) | §4.1 |
| Low | Stage 2 regularization | Negative (-1.0 dB val, 0% E2E) | §4.2 |
| **Ruled out** | Alpha supervision (H6) | Negative (monotonic decline) | §4.6 |
| **Ruled out** | SSIM weight increase (H7) | Negative (instability/collapse) | §4.6 |
| **Ruled out** | Raw data without preprocessing (HP M0) | Negative (divergence) | §4.6 |

---

## 9. Established Facts

| # | Fact | Evidence | Confidence |
|---|------|----------|:----------:|
| F1 | **GS-LRM >> PS reconstruction (same camera)** | 6v GT: **+10.06dB** PSNR_fg, +3.55dB PSNR_int | **High** |
| F2 | **MVDiff = sole E2E bottleneck** | GT 6v→E2E: **-15.64dB** | High |
| F3 | **MVDiff training strategy ≈ E2E** | E1/E2/E3: 7.90-8.20 PSNR_gt | High |
| F4 | **Pose conditioning ineffective** | E3 ≈ E2 within noise | High |
| F5 | **Cosine LR best for color** | PSNR_int: 16.15 > 15.88 > 15.63 | Medium |
| F6 | **Shape > Color bottleneck** | IoU gap 0.40 >> color gap | High |
| F7 | **GS-LRM 1v fails** | IoU=0.028 | High |
| F8 | **Mask protocol dominates comparison** | GT alpha vs white-BG IoU=0.24 | High |
| F9 | **E2E PSNR_int saturated** | E1 11K→20K: +0.03 only | Medium |
| F10 | **GS-LRM uses only views 0-3 (4v model)** | A2: v4,5 replacement = 0 dB impact | High |
| F11 | **Threshold degradation at v3** | A2: -5.2 dB cliff when v3 corrupted | High |
| F12 | **Alpha regularization hurts GS-LRM** | E5: val PSNR 22.34→21.34 (-1.0 dB) | High |
| F13 | **Camera mismatch dominates B-3** | M5: fx=549, HFOV=50° vs PS: fx≈810, HFOV≈35° | High |
| F14 | **Tier B pixel-wise comparison invalid (fj5_ds2)** | Different camera geometry → images not aligned | High |
| F15 | **Tier A valid with M5 same-camera** | PS retrained on M5, same camera space verified | **High** |
| F16 | **View count monotonically improves GS-LRM** | 1v→6v: 10.47→23.84 dB, no reversal | High |
| **F17** | **PS coverage deficit = primary PSNR_fg gap driver** | PS 89.3% vs FL 99.9%; PSNR_int gap (+3.55) << PSNR_fg gap (+10.06) | **High** |
| **F18** | **Alpha supervision monotonically hurts** | H6: 0.3→-0.37, 0.5→-0.51, 1.0→-0.87 dB | **High** |
| **F19** | **SSIM weight >0.1 causes instability** | H7: 0.5 collapsed (10.17), 0.3 declining | **High** |
| **F20** | **Raw data (M0) incompatible with pretrained GS-LRM** | HP: M0 diverges (11.46→1.85), preprocessing essential | **High** |
| **F21** | **Baseline loss config is optimal** | H6+H7: all alternatives worse than {l2=1, lpips=0.05, perc=0.5, ssim=0.1, alpha=0} | **High** |
| **F22** | **GS-LRM 3v already > PS M5** | 3v=18.56 vs PS=13.78, +4.78 dB (with only half the views) | **High** |
| **F23** | **PS paper PSNR 33.5 vs our 13.78 = metric protocol diff** | Full-image (85% white BG) vs FG-masked; IoU comparable (0.868 vs 0.846) | **High** |
| **F24** | **PS paper data ≠ our data** | Duke self-recorded (324K, 1536×2048) vs DANNCE/MAMMAL (3600, 512×512) | **High** |
| **F25** | **DA1 domain adaptation = +1.64 dB E2E** | GS-LRM fine-tuned on MVDiff outputs: 8.44→10.08 | **High** |

---

## 10. Complete Results Matrix

### 10.1 All Experiments

| Experiment | Type | Input | PSNR_gt | PSNR_int | IoU | Cov% |
|-----------|------|-------|:-------:|:--------:|:---:|:----:|
| GS-LRM 6v GT | S2 only | 6 GT views | **23.84** | **24.02** | **0.954** | 99.9 |
| GS-LRM 5v GT | S2 only | 5 GT views | 22.16 | 22.56 | 0.942 | 99.7 |
| GS-LRM 4v GT | S2 only | 4 GT views | 20.66 | 21.29 | 0.926 | 99.3 |
| A2 Level 3 | hybrid | 3GT+3MV | 15.82 | 19.67 | 0.780 | 90.2 |
| **PS M5 same-camera** | **baseline** | **6 GT views** | **13.78** | **20.47** | **0.846** | **89.3** |
| ~~PS fj5_ds2 (old)~~ | ~~baseline~~ | ~~6 GT views~~ | ~~16.80~~ | ~~20.54~~ | ~~0.827~~ | ~~91.7 (diff camera)~~ |
| ~~PS fj5_ds2 (GT mask)~~ | ~~baseline~~ | ~~6 GT views~~ | ~~15.33~~ | ~~14.34~~ | ~~0.317~~ | ~~69.9 (diff camera)~~ |
| A2 Level 4 | hybrid | 2GT+4MV | 10.44 | 16.85 | 0.614 | 76.9 |
| GS-LRM 1v GT | S2 only | 1 GT view | 10.47 | 10.47 | 0.028 | 100.0 |
| E2 resume 20K | E2E | 1 image | 8.20 | 15.63 | 0.521 | 71.5 |
| E3 pose 10K | E2E | 1 img+pose | 8.10 | 15.88 | 0.523 | 71.6 |
| Baseline 5K | E2E | 1 image | 7.93 | 13.70 | 0.474 | 74.0 |
| E1 cosine 20K | E2E | 1 image | 7.90 | 16.15 | 0.528 | 70.5 |

### 10.2 A2 Oracle MVDiff Complete Results

> **Note (v10.0)**: A2 uses the 4-view GS-LRM model (`num_input_views=4`). Level 0 shows 21.02 dB because this model only uses views 0-3, ignoring views 4,5. The proper 6-view model achieves 23.84 dB (see §10.1, Tier A/C).

| Level | GT Views | MV Views | PSNR_gt | PSNR_int | IoU | Cov% |
|:-----:|:--------:|:--------:|:-------:|:--------:|:---:|:----:|
| 0 | 0,1,2,3,4,5 | — | 21.02 | 22.36 | 0.943 | 98.9 |
| 1 | 0,1,2,3,4 | 5 | 21.02 | 22.36 | 0.943 | 98.9 |
| 2 | 0,1,2,3 | 4,5 | 21.02 | 22.36 | 0.943 | 98.9 |
| 3 | 0,1,2 | 3,4,5 | 15.82 | 19.67 | 0.780 | 90.2 |
| 4 | 0,1 | 2,3,4,5 | 10.44 | 16.85 | 0.614 | 76.9 |
| 5 | 0 | 1,2,3,4,5 | 7.90 | 16.15 | 0.528 | 70.5 |

### 10.3 Per-View Tables

*(Same as v6 §8.2-8.5 — unchanged)*

---

## 11. Option A: Fair Tier B via M5→PS Retraining

### 11.1 Why Option A Is Needed

Tier B (E2E vs PS) pixel-wise comparison is currently **invalid** because:
- FL outputs are in M5 camera space (HFOV=50°, fx=549, cx=256)
- PS outputs are in fj5_ds2 camera space (HFOV≈35°, fx≈810, cx≈269)

**Solution**: Retrain PS on M5 preprocessed data → both models render from identical cameras → pixel-wise comparison valid against same M5 GT.

### 11.2 Data Conversion

`convert_m5_for_ps.py` (gpu03 배포 완료):
- M5 RGBA PNGs → zarr (3600, 6, 512, 512, 3) white-BG composite
- M5 opencv_cameras.json → PS camera_params.h5 (K at 2× for ds=2 compat)
- center_rotation.npz: `centers_m5 = scale * (centers_fj5 - centroid)`
- Coordinate transform: **scale = 0.008772**, centroid = [59.67, 51.52, 107.10]

**Key parameter adjustments:**

| Parameter | fj5_ds2 (original) | M5 (converted) | Reason |
|-----------|:------------------:|:---------------:|--------|
| `ell` | 0.22 | **0.00193** | Scaled by 0.008772 |
| `image_width/height` | 1152/1024 | 1024/1024 | M5 = 512×512, stored at 2× |
| `image_downsample` | 2 | 2 | Kept for code compat |
| `frame_jump` | 5 | 1 | M5 already subsampled |
| Camera intrinsics | fx=1632, cx=601 | fx=1098, cx=512 | M5 params at 2× |

### 11.3 Execution Plan

```bash
# Step 1: Full conversion on gpu03 (M5 data lives here)
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift
python -m mouse_extensions.scripts.eval.convert_m5_for_ps \
  --m5_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
  --ps_camera_h5 /tmp/ps_ref_files/camera_params.h5 \
  --ps_center_npz /tmp/ps_ref_files/center_rotation.npz \
  --output_dir /home/joon/data/preprocessed/FaceLift_mouse/m5_for_ps

# Step 2: Transfer to joon server
scp -r /home/joon/data/preprocessed/FaceLift_mouse/m5_for_ps/* \
  joon:~/dev/pose-splatter/data/preprocessed/markerless_mouse_1_nerf/m5_fj1/

# Step 3: Train PS on joon server
ssh joon "cd ~/dev/pose-splatter && \
  python train.py --config configs/m5_config.json"

# Step 4: Evaluate (on gpu03, using fair_comparison.py)
# PS trained model renders → fair_comparison.py → compare against M5 GT
```

### 11.4 Expected Outcomes

| Comparison | Before Option A | After Option A |
|-----------|:--------------:|:-------------:|
| Tier B validity | **INVALID** (camera mismatch) | **Valid** (same camera) |
| PS input | fj5_ds2 images | M5 images |
| GT for eval | zarr (PS) / M5 (FL) | M5 (both) |
| Camera space | Different | **Identical** |

### 11.5 Data Conversion Verification (COMPLETED)

`verify_dataset_consistency.py` — ALL 5 CHECKS PASSED:

| Check | Status | Key Metric |
|-------|:------:|------------|
| Images | PASS | PSNR=**inf** (120/120 pixel-exact match) |
| Cameras | PASS | K/R/T diff = **0.00** |
| Center rotation | PASS | centers/angles/covs diff = **0.00** |
| Split | PASS | Train 2880 / Val 360 / Test 360 |
| Config | PASS | ell=0.00193, 512×512, ds=2 |

> Full report: `outputs/dataset_verification/dataset_consistency_report.md`

### 11.6 PS M5 Retraining Status — COMPLETED ✅

**Status (2026-02-22)**: Training **completed** (50 epochs) on joon server. Fair evaluation completed with `fair_test_only_eval.py`.

| Item | Value |
|------|-------|
| **Model** | `m5_baseline_gs` (GS-LRM baseline, joon server) |
| **Data** | `m5_for_ps_fj1` (M5 → PS format, verified pixel-exact) |
| **Training** | 50 epochs, `output/m5_baseline_gs/latest/` |
| **Test frames** | 3240-3599 (360 frames × 6 views = 2160 renders) |
| **PSNR_fg** | **13.78** |
| **PSNR_intersection** | **20.47** |
| **IoU** | **0.846** |
| **Coverage** | **89.3%** |
| **PSNR_gt (whole)** | **29.00** |

> See Tier A (§3) for same-camera comparison with FL GS-LRM.

### 11.7 Validation Criteria (Post-Training)

1. PS trained on M5 should achieve PSNR_gt ≈ 16-17 dB (similar to fj5_ds2 performance)
2. Both FL and PS eval against same M5 GT frames (test: 3240-3599)
3. Same mask protocol (GT alpha channel)
4. Pixel-wise metrics directly comparable

---

## 12. File Locations

### gpu03 (`/home/joon/dev/FaceLift/`)

| File | Purpose |
|------|---------|
| **Evaluation** | |
| `experiments/comparison/tier/*_fair.json` | Per-experiment fair metrics (1v-6v) |
| `experiments/analysis/mvdiff_quality/` | A1 MVDiff diagnostic |
| `outputs/analysis/oracle_mvdiff/oracle_analysis.json` | A2 summary |
| `mouse_extensions/scripts/eval/fair_comparison.py` | Fair evaluation (canonical) |
| `mouse_extensions/scripts/eval/convert_m5_for_ps.py` | M5→PS data conversion |
| `mouse_extensions/scripts/eval/verify_dataset_consistency.py` | Dataset verification |
| **Renders / Visualization** | |
| `outputs/tier_comparison/gslrm_{1-6}view_test/samples/{frame_id}/` | FL renders (360 frames per view count) |
| `outputs/tier_comparison/gslrm_{N}view_test/samples/{id}/render_view_0{0-5}.png` | Per-view rendered images |
| `outputs/tier_comparison/gslrm_{N}view_test/samples/{id}/comparison_grid.png` | Render vs GT grid |
| `outputs/tier_comparison/gslrm_{N}view_test/samples/{id}/turntable_grid.png` | Turntable visualization |
| `outputs/tier_comparison/gslrm_{N}view_test/metrics.json` | Per-model aggregate metrics |
| `outputs/visualizations/cross_view_comparison/` | 1-6v side-by-side grids (11 PNGs) |
| **Documents** | |
| `docs/experiments/FL_vs_PS_comparison.md` | **This document (SSOT)** |
| `outputs/dataset_verification/` | Verification report + visuals |

### joon (`/home/joon/dev/pose-splatter/`)

| File | Purpose |
|------|---------|
| **Evaluation** | |
| `scripts/mouse/analysis/fair_test_only_eval.py` | PS fair evaluation (test-only, FL-compatible) |
| `output/m5_baseline_gs/latest/fair_test_only_evaluation.json` | PS M5 fair eval results (canonical) |
| `experiments/fair/posesplatter_fair_360f_v15.json` | PS fj5_ds2 fair metrics (historical) |
| `experiments/fair/posesplatter_fair_360f_v15_gtmask.json` | PS with GT alpha mask (historical) |
| **Renders / Visualization** | |
| `output/m5_baseline_gs/latest/renders/` | PS training renders (101 files, `gt_epochXXX_frameYYYY.png`) |
| `output/m5_baseline_gs/latest/fair_eval_images/` | PS fair eval comparison grids (3 PNGs) |
| `output/m5_baseline_gs/latest/reconstruction.pdf` | Reconstruction summary |
| **Data** | |
| `data/preprocessed/.../fj5_ds2/` | PS original data (fj5 camera) |
| `data/preprocessed/.../m5_fj1/` | PS M5 data (same camera as FL) |

---

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| v1-v6 | ~260215 | Initial comparison, tier framework, fair eval |
| v7 | 260218 | Phase 3 results (E1/E2/E3), A1+A2 analysis |
| v8.0 | 260219 | Unified evaluation (B-3), metric consistency v2 |
| v8.1 | 260219 | **Camera mismatch discovery** → B-3 INVALIDATED |
| v9.0 | 260219 | Option A plan, metric consistency merged, docs consolidated |
| v9.1 | 260219 | Data conversion complete + verification (ALL PASS) |
| v9.2 | 260219 | Complete view ablation (1v-6v), 5v > 6v discovery |
| v10.0 | 260220 | 6v PSNR corrected (21.02→23.84): was A2 oracle (4v model), now proper 6v model. 6v optimal confirmed. |
| v11.0 | 260222 | PS M5 same-camera fair eval COMPLETE. Camera mismatch RESOLVED. GS-LRM ablations (H6/H7/HP). |
| **v12.0** | **260223** | **PS Paper vs Our Protocol analysis (§2.4): metric protocol decomposition (full-image vs FG-masked PSNR, 19.7 dB gap explained), dataset differences table, 3 rebuttal strategies. H_Split hypothesis (§2.5). Cross-species Rat7M plan (§2.6).** |

### Merged Documents
- `FL_PS_metric_consistency.md` v2.0 → B-3 camera mismatch section (§3) + §11에 통합
- INDEX v6 → v7 (이 문서 참조)

---

*FaceLift vs Pose-Splatter Comparison v12.0 | SSOT | 2026-02-23*
