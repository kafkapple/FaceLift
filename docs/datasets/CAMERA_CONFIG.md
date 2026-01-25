# Camera Configuration

> **Navigation**: [← Index](./00_INDEX.md) | [MoC](../00_MoC_INDEX.md)
> **SSOT**: 6카메라 배치 및 View 선택 가이드

---

## 1. 카메라 배치 (6 Views)

### 1.1 위치 및 각도

| View | Position (x, y, z) | Azimuth | Elevation | Z-Score |
|------|-------------------|---------|-----------|---------|
| 0 | (-2.19, -1.42, 0.70) | -147° | +14.9° | -0.56 |
| 1 | (+2.10, +1.41, 0.95) | +34° | +20.6° | +0.19 |
| 2 | (+0.18, +2.64, 0.53) | +86° | +11.3° | -1.04 |
| 3 | (+2.60, -0.52, 0.50) | -11° | +10.7° | -1.11 |
| 4 | (-1.95, +1.42, 1.21) | +144° | +26.5° | +0.98 |
| **5** | (+1.01, -2.09, 1.38) | -64° | **+30.8°** | **+1.54** ⚠️ |

### 1.2 통계

| 항목 | 값 |
|------|-----|
| **평균 Elevation** | 19.2° |
| **Elevation 범위** | 10.7° ~ 30.8° |
| **Outlier** | View 5 (+11.6° 이탈, z-score=1.54) |

### 1.3 Top View 배치

```
              +Y
               │
        4      │      2
         •     │     •
               │
    ─────0─────┼─────1───── +X
         •     │     •
               │
        5      │      3
         •     │     •
               │
              -Y

주: 숫자는 카메라 ID, 원점은 피사체 위치
```

---

## 2. View 선택 순서

### 2.1 권장 순서

**Input Views (모델 입력)**: `0 → 4 → 2 → 1 → 3`

| 순서 | View | 이유 |
|------|------|------|
| 1 | 0 | 기준 뷰 (전면) |
| 2 | 4 | 0과 대각선 (최대 베이스라인) |
| 3 | 2 | Y축 정면 |
| 4 | 1 | 0과 반대측 |
| 5 | 3 | X축 정면 |

**Excluded**: View 5 (높은 elevation, outlier)

### 2.2 View 수별 조합

| 뷰 수 | 조합 | 용도 |
|-------|------|------|
| 4 views | 0, 4, 2, 1 | 기본 실험 |
| 5 views | 0, 4, 2, 1, 3 | **권장** |
| 6 views | 0, 4, 2, 1, 3, 5 | 최대 커버리지 |

### 2.3 View 5 주의사항

| 문제 | 설명 |
|------|------|
| **높은 Elevation** | 30.8° (평균 대비 +11.6°) |
| **Top-down 편향** | 다른 뷰와 기하학적 차이 큼 |
| **Z-score** | 1.54 (outlier 경계) |

**권장**: 5 views 사용 (View 5 제외)

---

## 3. 카메라 파라미터

### 3.1 Intrinsics (정규화 후)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **fx, fy** | 549 | 정규화된 focal length |
| **cx, cy** | 256 | Principal point (이미지 중심) |
| **skew** | 0 | D8에서 보정됨 |

### 3.2 Extrinsics (정규화 후)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **trans_norm** | 2.7 | 카메라 거리 (정규화) |
| **Rotation** | 다양 | 뷰마다 다름 |

### 3.3 원본 vs 정규화

| 항목 | 원본 | 정규화 후 |
|------|------|-----------|
| fx | 844 | 549 |
| trans_norm | 237 | 2.7 |
| 비율 (fx/trans) | 3.56 | 203 |

---

## 4. Config 설정

### 4.1 Dataset Config 예시

```yaml
# configs/datasets/M3_2.yaml
dataset:
  train:
    num_input_views: 5
    views: [0, 4, 2, 1, 3]  # 권장 순서

  validation:
    num_input_views: 5
    views: [0, 4, 2, 1, 3]
```

### 4.2 View Selection Mode

| 모드 | 설명 | 용도 |
|------|------|------|
| **fixed** | 고정 뷰 순서 | 기본 |
| **random** | 무작위 선택 | 데이터 증강 |
| **sequential** | 순차 선택 | 특수 용도 |

---

## 5. Triangulation 정확도

### 5.1 방식별 비교

| 방식 | Ray Error | 권장 |
|------|-----------|------|
| **DLT Triangulation** | 2.72 ± 0.80 mm | ✅ |
| Visual Hull | 2.72 ± 0.80 mm | ⚪ (느림) |
| Global Average | 34.41 ± 6.22 mm | ❌ |

### 5.2 Center Estimation

```python
from mouse_extensions.preprocessing.center_estimation import CenterEstimator

estimator = CenterEstimator(cameras, method='triangulation')
result = estimator.estimate(masks)
# result.center_3d → 모든 뷰에 back-project
```

---

## 6. 관련 문서

- [[RAW_DATA]] - 원본 데이터 출처
- [[VERSION_SCHEMA]] - 데이터셋 버전
- [[../theory/PP_FX_MVG_ANALYSIS]] - MVG 이론

---

*Camera Configuration v1.0 | 2026-01-26*
