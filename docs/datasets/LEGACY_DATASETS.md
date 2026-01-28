# Legacy Datasets Reference (D0-D10, M3 Series)

> **Consolidated from**: DATASET_MASTER_REFERENCE.md, M3_SERIES_SPEC.md, VERSION_SCHEMA.md, presets/
> **For current datasets, see**: M5_SERIES_SPEC.md, PREPROCESSING_REGISTRY.md
> **Created**: 2026-01-28

---

## Version Schema

### M-Series (권장)

| Alias | Preset | 변환 | PP | fx | 상태 |
|-------|--------|------|-----|-----|------|
| **M1** | D7.1 | Affine | 256 | 549 | ✅ 기준선 |
| **M2** | D8 | Homography | 256 | 549 | ✅ 정밀 |
| **M3_1** | M3_1 | Global Zoom | 256 | 549 | ✅ 검증됨 |
| **M3_2** | M3_2 | Per-sample | 256 | 549 | ✅ 검증됨 |
| **M5h_2** ★ | M5h_2 | Per-sample+Recenter | 256 | 549 | ⭐ **권장** |

### 분류 체계 (VERSION_HIERARCHY)

| 분류 | 프리셋 | 설명 |
|------|--------|------|
| **affine** | D7, D7.1, D7.2 | 기본 변환, skew 무시 |
| **homography** | D8, D8.1, D8.2 | skew 보정 포함 |
| **homography_zoom** | M3_1, M3_2 | homography + adaptive zoom |
| **geometry_broken** | D1, D4, D6-* | ⛔ PP 미보정, 사용 금지 |
| **recentered** | M5, M5h, M5h_1, M5h_2 | re-centered + uniform norm |
| **deprecated** | M3, M3_norm, D10.3 | fx/PP 버그 |

### Split 방식

| 접미사 | 의미 | 설명 |
|--------|------|------|
| **_t** | Temporal | 시간순 3분할 (train→val→test) |
| 없음 | Random | 무작위 샘플링 |

### Coverage 실측값

| 데이터셋 | Coverage | 비고 |
|----------|----------|------|
| D7_1, D8 (zoom 없음) | **~3%** | 원본 크기 |
| M3_1, M3_2 (zoom 있음) | **~6%** | 확대 후 |

> ⚠️ 이전 문서의 50%, 74%, 84% 수치는 검증되지 않음

---

## Dataset Master Reference (D0-D10)

### 핵심 원칙

#### 1. PP (Principal Point) 규칙

```
✅ 성공: zoom_center_mode = "image" → PP = 256 고정
❌ 실패: zoom_center_mode = "object" → PP 가변 → Ray Error → Ghosting
```

#### 2. 카메라 파라미터 (Pretrained 기준)

```
fx = 549
cx = cy = 256 (이미지 중앙)
translation_norm ≈ 2.7
```

#### 3. Coverage 효과

```
Coverage ↑ → PSNR ↑
- D7_1/D8: ~3% → PSNR ~20-21
- M3_1/M3_2: ~5% → PSNR TBD
- D3_normalized: ~6% → PSNR 27 (참고용)
```

### 성공/실패 원인 분석

#### zoom_center_mode 결정적 영향

| zoom_center_mode | PP | Ray Error | 결과 |
|------------------|-----|-----------|------|
| **"image"** (center-aligned) | 256 고정 | 0° | ✅ 성공 |
| **"object"** (object-centered) | 가변 | 11-17° | ❌ 실패 |

#### 검증된 사실

| 비교 | 결과 | 결론 |
|------|------|------|
| M3_2 vs M4 | PP 256 vs 가변 | PP 가변 → 실패 |
| D7_1 vs D8 | Affine vs Homography | 성능 차이 미미 (0.72 PSNR) |
| Coverage 3% vs 6% | D7_1 vs D3_norm | Coverage ↑ → PSNR ↑ |

### 실패/Deprecated 데이터셋 (사용 금지)

| Dataset | Ray Error | 실패 원인 |
|---------|-----------|-----------|
| M4 | 11.8° | Object-centered zoom → PP 가변 |
| M3_persample | 17.4° | zoom_center_mode: object |
| M3_norm | 15.0° | zoom_center_mode: object |
| D10.3 (M3) | ~8° | fx=739 버그 |
| D1/D4/D6-* | ~13° | PP 미보정 |

### 전처리 Evolution

```
v5 (원본) → 생쥐 위치/크기 불균일
    ↓
v2 (pixel_based) → 뷰별 독립 변환 → 복수 생쥐
    ↓
D1-D6 (PP 강제) → Ray 방향 오류
    ↓
D7/D8 (M1/M2) → PP 보정 + fx 정규화 → PSNR 20-21
    ↓
M3_1/M3_2 (Zoom 추가) → Coverage 향상
    ↓
M5 Series (Re-center) → ⭐ 현재 권장
```

---

## M3 Series Specification

### 버전 계층

| ID | 이름 | fx 정규화 | PP 정합 | Zoom | 상태 |
|----|------|-----------|---------|------|------|
| M3 | 원본 | X (739) | X 가변 | Global | Deprecated |
| M3_norm | fx 정규화 | O (549) | X 가변 | Global | Deprecated |
| M3_persample | Per-sample | O (549) | X 가변 | Per-sample | Deprecated |
| **M3_1** | Global MVG | O (549) | O (256) | Global | **Active** |
| **M3_2** | Per-sample MVG | O (549) | O (256) | Per-sample | **Active** |

### MVG 정합성 이론

**문제: Object-Centered Zoom**

1. 객체 중심 찾기 → 뷰마다 다른 2D 중심
2. 객체 중심 기준 crop → 뷰마다 다른 crop offset
3. PP (cx, cy) 가 뷰/샘플마다 다름
4. Ray direction error = arctan(PP_offset / fx)
5. 최대 16.15도 ray error → Ghosting artifact

**해결: Center-Aligned Zoom (M3_1, M3_2)**

1. 이미지 중심 (256, 256) 기준 crop
2. crop_x = (512 - crop_size) // 2 → 항상 동일
3. PP = 256 자동 보장 (모든 뷰, 모든 샘플)
4. Ray error = 0도 → Ghosting 제거

### 검증 결과

| 프리셋 | cx (mean ± std) | cy (mean ± std) | Ray Error | Val PSNR |
|--------|-----------------|-----------------|-----------|----------|
| M3_norm | 197.2 ± 46.4 | 136.7 ± 22.0 | 13.62° | 17.09 |
| M3_persample | 234.7 ± 32.1 | 212.0 ± 39.8 | 16.15° | 17.08 |
| **M3_1** | 256.0 ± 0.0 | 256.0 ± 0.0 | 0.00° | TBD |
| **M3_2** | 256.0 ± 0.0 | 256.0 ± 0.0 | 0.00° | TBD |

### Global vs Per-sample Zoom

**Global Zoom (M3_1)**:
```python
avg_coverage = mean([mask.sum() / (512*512) for mask in all_masks])
zoom = sqrt(target_coverage / avg_coverage)  # e.g., 1.347
# 모든 샘플에 동일 zoom 적용
```

**Per-sample Zoom (M3_2)**:
```python
for sample in samples:
    coverage = sample.mask.sum() / (512*512)
    zoom = sqrt(target_coverage / coverage)  # 샘플마다 다름
```

| 시나리오 | 권장 방식 | 이유 |
|----------|-----------|------|
| 가설 검증 | Global (M3_1) | 단순, 일관성 |
| 최종 성능 최적화 | Per-sample (M3_2) | 개별 최적화 |
| 새 모델 디버깅 | Global (M3_1) | 변수 최소화 |

### M3_1 vs M3_2 핵심 차이

| 설정 | M3_1 | M3_2 |
|------|------|------|
| zoom_scope | global | **per_sample** |
| 장점 | Clipping 없음 | 샘플별 최적화 |
| 단점 | 일부 샘플 저 coverage | 6.5% clipping |
| 권장 | 안전한 선택 | 최적 성능 |

---

## Preset Details

### D3_normalized

**Status**: Reference (최고 성능 기준, 재현 불가)

| 항목 | 값 |
|------|-----|
| **Val PSNR** | **27.09** |
| **Coverage** | **~6%** |
| **fx** | 549 |
| **PP (cx, cy)** | 256, 256 |
| **trans_norm** | 불균일 (std=0.52) |

**성공 요인**: Coverage ~6% (+6 PSNR) + PP=256 (+3 PSNR) + fx=549 (필수)

**재현 불가 요소**: 수동 전처리, preprocessing_info.json 없음. trans_norm 불균일 (카메라별 거리 다름).

**Data**: `/home/joon/data/preprocessed/FaceLift_mouse/D3_normalized/`

---

### D7.1 (M1) - Affine Baseline

**Status**: ✅ Active (기준선)

| 항목 | 값 |
|------|-----|
| **Transform** | Affine |
| **Scale Mode** | Individual |
| **fx** | 549 |
| **PP** | 256, 256 |
| **Coverage** | ~3% |
| **Val PSNR** | 20.93 |

```python
"D7.1": {
    "paradigm": "pp_centered_shift",
    "transform": "affine",
    "scale_mode": "individual",
    "pp_method": "shift_to_256",
    "skew_correction": False,
    "adaptive_zoom": False,
    "target_fx": 548.9937744140625,
    "output_size": 512,
}
```

**Commands**:
```bash
python -m mouse_extensions.preprocessing.preprocess --preset D7.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

---

### D8 (M2) - Homography with Skew Correction

**Status**: ✅ Active (정밀)

| 항목 | 값 |
|------|-----|
| **Transform** | Homography |
| **Skew Correction** | Yes |
| **fx** | 549 |
| **PP** | 256, 256 |
| **Coverage** | ~3% |
| **Val PSNR** | 20.21 |

```python
"D8": {
    "paradigm": "precision_homography",
    "transform": "homography",
    "scale_mode": "individual",
    "pp_method": "shift_to_256",
    "skew_correction": True,
    "adaptive_zoom": False,
    "target_fx": 548.9937744140625,
    "output_size": 512,
}
```

**Note**: Homography (8 DoF) vs Affine (6 DoF). D8 PSNR slightly lower (-0.72) than D7.1, possibly due to overfitting from extra degrees of freedom.

---

### M3 (D10.3) - H2 Verification Dataset

**Status**: ⚠️ H2 검증용 (일반 사용 비권장)

| 항목 | 값 |
|------|-----|
| **Transform** | Homography |
| **Zoom Scope** | Global |
| **fx** | **739** (미정규화) |
| **PP** | 가변 |
| **Coverage** | ~6% |
| **Ray Error** | 6.96° |

```python
"D10.3": {
    "paradigm": "precision_homography",
    "transform": "homography",
    "scale_mode": "individual",
    "pp_method": "shift_to_256",
    "skew_correction": True,
    "adaptive_zoom": True,
    "zoom_method": "coverage_based",
    "zoom_scope": "global",
    "zoom_center_mode": "object",
    "target_fg_coverage": 0.05,
    "zoom_range": [1.0, 2.5],
    "output_size": 512,
    "normalize_after_zoom": False,
}
```

**Purpose**: H2 hypothesis verification (fx normalization effect). M3 (fx=739) vs M3_norm (fx=549).

---

### M3_1 - Global Zoom + MVG-Correct

**Status**: ✅ Active

| 항목 | 값 |
|------|-----|
| **Transform** | Homography |
| **Zoom Scope** | Global (1.347 고정) |
| **Zoom Center** | Image (center-aligned) |
| **fx** | 549 |
| **PP** | 256, 256 (MVG-correct) |
| **Coverage** | ~6% |
| **Ray Error** | 0° |

```python
"M3_1": {
    "paradigm": "precision_homography",
    "transform": "homography",
    "scale_mode": "individual",
    "pp_method": "shift_to_256",
    "skew_correction": True,
    "adaptive_zoom": True,
    "zoom_method": "coverage_based",
    "zoom_scope": "global",
    "zoom_center_mode": "image",
    "target_fg_coverage": 0.05,
    "zoom_range": [1.0, 2.5],
    "target_fx": 548.9937744140625,
    "output_size": 512,
    "normalize_after_zoom": True,
}
```

---

### M3_2 - Per-Sample Zoom + MVG-Correct

**Status**: ✅ Active (M3 series 중 권장)

| 항목 | 값 |
|------|-----|
| **Transform** | Homography |
| **Zoom Scope** | Per-sample (가변) |
| **Zoom Center** | Image (center-aligned) |
| **fx** | 549 (재정규화) |
| **PP** | 256, 256 (MVG-correct) |
| **Coverage** | ~6% (샘플별 최적) |
| **Ray Error** | 0° |
| **Clipping** | 6.5% |

```python
"M3_2": {
    "paradigm": "precision_homography",
    "transform": "homography",
    "scale_mode": "individual",
    "pp_method": "shift_to_256",
    "skew_correction": True,
    "adaptive_zoom": True,
    "zoom_method": "coverage_based",
    "zoom_scope": "per_sample",
    "zoom_center_mode": "image",
    "target_fg_coverage": 0.05,
    "zoom_range": [1.0, 2.5],
    "target_fx": 548.9937744140625,
    "output_size": 512,
    "normalize_after_zoom": True,
}
```

---

## Experiment Commands

```bash
cd /home/joon/dev/FaceLift

# D7_1 (M1) baseline
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d D7_1 -e E1_2_alpha

# D8 (M2) homography
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d D8 -e E1_2_alpha

# M3_1 (global zoom)
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d M3_1 -e E1_2_alpha

# M3_2 (per-sample zoom, recommended)
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d M3_2 -e E1_2_alpha
```

---

*Legacy Datasets Reference v1.0 | Consolidated: 2026-01-28*
