# Mouse 3D Reconstruction 전처리 가이드 v2.0

> Created: 2026-01-24 | Author: Claude Code

## 버전 체계 개편

### 기존 체계 (D-series) → 신규 체계 (M-series)

| 기존 | 신규 | 설명 |
|------|------|------|
| D1-D6 | - | DEPRECATED (cross-view inconsistency) |
| D7_1 | **M1** | 기하학적 정확성 기준선 |
| D8 | **M2** | Precision homography |
| - | **M3** ★ | Object-Centered Zoom (신규) |

### M-series 정의

**M1 (Geometric Baseline)**:
- D7_1 동일 (pp_centered_shift + individual affine)
- Ray Error: ~0°
- FG Coverage: 2-3%
- 용도: 기하학적 정확성 검증

**M2 (Precision)**:
- D8 동일 (precision_homography + skew correction)
- Ray Error: ~0°
- FG Coverage: 2-3%
- 용도: 최대 정밀도 필요시

**M3 (Optimal)** ★ 권장:
- M1 기반 + Object-Centered Zoom
- Ray Error: ~0°
- FG Coverage: **5%+**
- Center Offset: **<15px**
- 용도: 최적 성능 (Ghosting 해결)

---

## M3 처리 파이프라인

### 이론적 근거

**문제**: 작은 object → Plucker ray 분포 불균형 → View 구분 실패 → Ghosting

**해결**: Object-Centered Zoom으로 FG coverage 증가

### 파이프라인 단계

```
Raw Data
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 1: 3D Object Center Estimation      │
│   - 6 views mask → 2D centroids          │
│   - DLT triangulation → center_3d        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 2: Adaptive Zoom Calculation        │
│   - current_coverage ≈ 2-3%              │
│   - target_coverage = 5%                 │
│   - zoom = sqrt(target / current)        │
│   - zoom = clip(zoom, 1.0, 2.5)          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 3: Object-Centered Crop             │
│   - center_3d → back-project to 2D       │
│   - crop around 2D center                │
│   - apply zoom                           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 4: Camera Normalization (M1 기반)   │
│   - PP shift to 256                      │
│   - fx, fy → 549 normalization           │
│   - translation → 2.7 normalization      │
└─────────────────────────────────────────┘
    │
    ▼
Preprocessed Data (M3)
```

### 핵심 수학

**Intrinsics 업데이트**:
```python
# After crop (offset: crop_x, crop_y)
cx' = cx - crop_x
cy' = cy - crop_y

# After zoom
fx'' = fx' * zoom
fy'' = fy' * zoom
cx'' = cx' * zoom
cy'' = cy' * zoom

# After normalization (target fx=549)
scale = 549.0 / fx''
fx_final = 549.0
fy_final = fy'' * scale
cx_final = 256.0  # shift to center
cy_final = 256.0
trans_final = trans * scale
```

**Ray Accuracy 검증**:
```
ray_error = arctan(PP_error / fx)
          = arctan(0 / 549)  # PP shifted to 256
          = 0°
```

---

## 비교 실험 결과 예측

| 설정 | Dataset | FG Coverage | Center | PSNR (예상) |
|------|---------|-------------|--------|-------------|
| Baseline | M1 | 2.16% | 101px | ~20 |
| Precision | M2 | 2.0% | ~100px | ~19-20 |
| **Optimal** | **M3** | **5%+** | **<15px** | **24-26** |

---

## 사용법

### 전처리 실행

```bash
# M1 (Geometric Baseline)
python -m mouse_extensions.preprocessing.preprocess_m \
    --preset M1 \
    --input-dir /path/to/raw \
    --output-dir /path/to/M1

# M3 (Optimal - Object-Centered)
python -m mouse_extensions.preprocessing.preprocess_m \
    --preset M3 \
    --input-dir /path/to/raw \
    --output-dir /path/to/M3
```

### 설정 파일

```yaml
# configs/preprocessing/M3.yaml
preset: M3
paradigm: object_centered_zoom

# Object centering
center_method: triangulation
center_tolerance_px: 15

# Adaptive zoom
target_fg_coverage: 0.05
zoom_range: [1.0, 2.5]

# Camera normalization
target_fx: 549.0
target_cx: 256.0
target_cy: 256.0
target_distance: 2.7

# Transform
transform: affine  # M1 base
pp_method: shift_to_256
```

---

## 핵심 참고문헌

1. LRM (Hong, 2024): Object centering 전처리
2. GS-LRM (Zhang, 2024): Plucker ray encoding
3. Pose Splatter (Goffinet, 2025): Normalized masked loss
4. LGM (3DTopia, 2024): Alpha supervision

---

*v2.0 | 2026-01-24*
