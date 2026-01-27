# M3 시리즈 데이터셋 명세

> **최종 업데이트**: 2026-01-25
> **목적**: M3 계열 전처리 프리셋의 상세 명세 및 사용 가이드
> **관련 문서**: [[00_MoC_INDEX]] | [[PREPROCESSING_REGISTRY]] | [[PP_MVG_COMPREHENSIVE_ANALYSIS]]

---

## 1. M3 시리즈 개요

M3 시리즈는 **Homography 변환 + Adaptive Coverage Zoom**을 적용하는 전처리 방식입니다.

### 1.1 버전 계층

| ID | 이름 | fx 정규화 | PP 정합 | Zoom | 상태 |
|----|------|-----------|---------|------|------|
| M3 | 원본 | X (739) | X 가변 | Global | Deprecated |
| M3_norm | fx 정규화 | O (549) | X 가변 | Global | Deprecated |
| M3_persample | Per-sample | O (549) | X 가변 | Per-sample | Deprecated |
| **M3_1** | Global MVG | O (549) | O (256) | Global | **Active** |
| **M3_2** | Per-sample MVG | O (549) | O (256) | Per-sample | **Recommended** |

---

## 2. 프리셋 상세 명세

### 2.1 M3_1 (Global Zoom + MVG-Correct)

**핵심 특징**:
- Homography 변환 + skew correction
- **Global adaptive zoom**: 전체 데이터셋에서 동일 zoom 계수
- **Center-aligned crop**: 이미지 중심 기준 crop -> PP=256 자동 보장
- fx/fy 정규화: 549

**설정값**:
```python
"M3_1": {
    "paradigm": "precision_homography",
    "transform": "homography",
    "scale_mode": "individual",
    "pp_method": "shift_to_256",
    "skew_correction": True,
    "adaptive_zoom": True,
    "zoom_method": "coverage_based",
    "zoom_center_mode": "image",      # MVG-correct
    "target_fg_coverage": 0.05,
    "zoom_range": [1.0, 2.5],
    "target_fx": 548.9937744140625,
    "output_size": 512,
    "normalize_after_zoom": True,
}
```

### 2.2 M3_2 (Per-Sample Zoom + MVG-Correct) - Recommended

**핵심 특징**:
- Homography 변환 + skew correction
- **Per-sample adaptive zoom**: 각 샘플별 최적 zoom 계산
- **Center-aligned crop**: 이미지 중심 기준 crop -> PP=256 자동 보장
- fx/fy 정규화: 549

**설정값**:
```python
"M3_2": {
    "paradigm": "precision_homography",
    "transform": "homography",
    "scale_mode": "individual",
    "pp_method": "shift_to_256",
    "skew_correction": True,
    "adaptive_zoom": True,
    "zoom_method": "coverage_based",
    "zoom_scope": "per_sample",       # 샘플별 zoom
    "zoom_center_mode": "image",      # MVG-correct
    "target_fg_coverage": 0.05,
    "zoom_range": [1.0, 2.5],
    "target_fx": 548.9937744140625,
    "output_size": 512,
    "normalize_after_zoom": True,
}
```

---

## 3. MVG 정합성 이론

### 3.1 문제: Object-Centered Zoom

기존 M3_norm, M3_persample의 문제점:

1. 객체 중심 찾기 -> 뷰마다 다른 2D 중심
2. 객체 중심 기준 crop -> 뷰마다 다른 crop offset
3. PP (cx, cy) 가 뷰/샘플마다 다름
4. Ray direction error = arctan(PP_offset / fx)
5. 최대 16.15도 ray error -> Ghosting artifact

### 3.2 해결: Center-Aligned Zoom (M3_1, M3_2)

1. 이미지 중심 (256, 256) 기준 crop
2. crop_x = (512 - crop_size) // 2  # 항상 동일
3. PP = 256 자동 보장 (모든 뷰, 모든 샘플)
4. Ray error = 0도 -> Ghosting 제거

### 3.3 검증 결과

| 프리셋 | cx (mean +/- std) | cy (mean +/- std) | Ray Error | Val PSNR |
|--------|-------------------|-------------------|-----------|----------|
| M3_norm | 197.2 +/- 46.4 | 136.7 +/- 22.0 | 13.62도 | 17.09 |
| M3_persample | 234.7 +/- 32.1 | 212.0 +/- 39.8 | 16.15도 | 17.08 |
| **M3_1** | 256.0 +/- 0.0 | 256.0 +/- 0.0 | 0.00도 | TBD |
| **M3_2** | 256.0 +/- 0.0 | 256.0 +/- 0.0 | 0.00도 | TBD |

**참조**: [[PP_MVG_COMPREHENSIVE_ANALYSIS]]

---

## 4. 사용법

### 4.1 전처리 명령어

```bash
cd /home/joon/dev/FaceLift

# M3_1 (Global zoom)
/home/joon/anaconda3/envs/facelift/bin/python \
    -m mouse_extensions.preprocessing.preprocess \
    --preset M3_1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_1

# M3_2 (Per-sample zoom) - Recommended
/home/joon/anaconda3/envs/facelift/bin/python \
    -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2
```

### 4.2 검증 명령어

```bash
/home/joon/anaconda3/envs/facelift/bin/python \
    mouse_extensions/scripts/diagnostics/verify_pp_mvg_consistency.py \
    --datasets M3_1,M3_2 --verbose
```

### 4.3 학습 명령어

```bash
# M3_2 + E0_1_facelift (권장)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E0_1_facelift
```

---

## 5. Deprecated 프리셋

| ID | 문제점 | 대체 |
|----|--------|------|
| M3 | fx=739 (미정규화), PP 가변 | M3_1 |
| M3_norm | PP 가변 (ray error 13.62도) | M3_1 |
| M3_persample | PP 가변 (ray error 16.15도) | M3_2 |

---

## 6. 관련 문서

| 문서 | 내용 |
|------|------|
| [[00_MoC_INDEX]] | 문서 허브 |
| [[PP_MVG_COMPREHENSIVE_ANALYSIS]] | PP 정합성 종합 분석 |
| [[PP_FIX_MVG_THEORY]] | MVG 이론 및 해결책 |
| [[PREPROCESSING_REGISTRY]] | 전체 프리셋 목록 |
| [[EXPERIMENT_REGISTRY]] | 실험 설정 |
| [[TRAIN_VAL_GAP_ANALYSIS]] | Gap 분석 |

---

*M3 Series Specification v1.0 | 2026-01-25*

---

## 5. Global vs Per-sample Zoom 상세

### 5.1 동작 비교

**Global Zoom (M3, M3_1)**:
```python
# 전체 데이터셋에서 평균 coverage 계산
avg_coverage = mean([mask.sum() / (512*512) for mask in all_masks])
zoom = sqrt(target_coverage / avg_coverage)  # e.g., 1.347

# 모든 샘플에 동일 zoom 적용
for sample in samples:
    zoomed = apply_zoom(sample, zoom=1.347)  # 고정
```

**Per-sample Zoom (M3_2)**:
```python
# 각 샘플별로 coverage 계산 후 zoom 결정
for sample in samples:
    coverage = sample.mask.sum() / (512*512)
    zoom = sqrt(target_coverage / coverage)  # 샘플마다 다름
    zoomed = apply_zoom(sample, zoom=zoom)
```

### 5.2 fx 결과 비교

| 방식 | Zoom 범위 | fx 결과 | 특징 |
|------|-----------|---------|------|
| Global | 고정 (1.347) | 549 (일관) | 단순, 안정적 |
| Per-sample | 가변 (1.0~2.5) | 549 (재정규화) | 최적화, 복잡 |

### 5.3 권장 사용 시나리오

| 시나리오 | 권장 방식 | 이유 |
|----------|-----------|------|
| **H1, H2, H3 검증** | Global (M3_1) | 단순, 일관성 |
| **최종 성능 최적화** | Per-sample (M3_2) | 개별 최적화 |
| **새 모델 디버깅** | Global (M3_1) | 변수 최소화 |

---

## 6. 실험 가설 매핑

| 가설 | 검증 데이터셋 | 비교 대상 | 목적 |
|------|---------------|-----------|------|
| **H1: Coverage 효과** | M3_norm | D7_1 | Coverage ~3% vs ~6% |
| **H2: fx 정규화 효과** | M3 vs M3_norm | - | fx=739 vs fx=549 |
| **H3: PP 정합성 효과** | M3_1, M3_2 | M3_norm | PP=256 vs 가변 |

### 실험 우선순위

```
P0: M3_1 또는 M3_2 + E0_1_facelift
    → D3_normalized 재현 시도 (Coverage↑ + PP=256 + fx=549)

P1: M3 + E0_1_facelift
    → H2 검증 (fx=739 상태)

P2: M3_norm + E1_2_alpha
    → 이미 진행중

P3: View 수 실험 (5v)
    → P0 완료 후
```

---

## 7. 관련 문서

- [[VERSION_SCHEMA]] - 버전 체계, Coverage 비교
- [[PP_FX_MVG_ANALYSIS]] - PP/fx 이론
- [[PREPROCESSING_REGISTRY]] - 버그 이력
- [[00_MoC_INDEX]] - 문서 허브

---

*M3_SERIES_SPEC v1.2 | 2026-01-26 | Global vs Per-sample 추가*

---

## See Also

> Parent: [[INDEX]] > Datasets

- [[PREPROCESSING_REGISTRY]] - 전처리 SSOT
- [[PP_FX_MVG_ANALYSIS]] - PP/MVG 이론

### Archived M3 Analysis Documents
- `_archive/analysis/M3_INTEGRATION_RATIONALE.md`
- `_archive/analysis/M3_VERIFICATION_REPORT.md`

---

*Last updated: 2026-01-27*
