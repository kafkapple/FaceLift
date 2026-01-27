# Dataset Version Schema

> **최종 업데이트**: 2026-01-28
> **SSOT**: 데이터셋 버전 체계 및 분류

---

## M-Series (권장)

| Alias | Preset | 변환 | PP | fx | 상태 |
|-------|--------|------|-----|-----|------|
| **M1** | D7.1 | Affine | 256 | 549 | ✅ 기준선 |
| **M2** | D8 | Homography | 256 | 549 | ✅ 정밀 |
| **M3_1** | M3_1 | Global Zoom | 256 | 549 | ✅ 검증됨 |
| **M5h_2** ★ | M5h_2 | Per-sample+Recenter | 256 | 549 | ⭐ **권장** |
| **M3_2** | M3_2 | Per-sample | 256 | 549 | ✅ 검증됨 |

**M3 상세**: [[M3_SERIES_SPEC]] (config, 이론, 검증 결과)

---

## 분류 체계 (VERSION_HIERARCHY)

| 분류 | 프리셋 | 설명 |
|------|--------|------|
| **affine** | D7, D7.1, D7.2 | 기본 변환, skew 무시 |
| **homography** | D8, D8.1, D8.2 | skew 보정 포함 |
| **homography_zoom** | M3_1, M3_2 | homography + adaptive zoom |
| **geometry_broken** | D1, D4, D6-* | ⛔ PP 미보정, 사용 금지 |
| **recentered** | M5, M5h, M5h_1, M5h_2 | re-centered + uniform norm |
| **deprecated** | M3, M3_norm, D10.3 | fx/PP 버그 |

---


## M5 Series: Re-centered + Uniform Normalization (2026-01-28)

| Alias | Base | Zoom | Re-center | 상태 |
|-------|------|------|-----------|------|
| **M5** | Affine | 없음 | Yes | Active |
| **M5h** | Homography | 없음 | Yes | Active |
| **M5h_1** | Homography | Global [1.0,1.8] | Yes | Active |
| **M5h_2** ★ | Homography | Per-sample [1.0,1.5] | Yes | **권장** |

**핵심 변경**:
- Per-view distance norm → Uniform distance norm (baseline 비율 보존)
- Camera centroid → 원점 (GS-LRM pretrained 호환)

**상세**: [[M5_SERIES_SPEC]]

---

## Split 방식

| 접미사 | 의미 | 설명 |
|--------|------|------|
| **_t** | Temporal | 시간순 3분할 (train→val→test) |
| 없음 | Random | 무작위 샘플링 |

---

## Coverage 실측값

| 데이터셋 | Coverage | 비고 |
|----------|----------|------|
| D7_1, D8 (zoom 없음) | **~3%** | 원본 크기 |
| M3_1, M3_2 (zoom 있음) | **~6%** | 확대 후 |

> ⚠️ 이전 문서의 50%, 74%, 84% 수치는 검증되지 않음

---

## 관련 문서

- [[M3_SERIES_SPEC]] - M3 상세 (config, 이론)
- [[PREPROCESSING_REGISTRY]] - 프리셋 정의
- [[EXPERIMENT_RESULTS]] - 실험 결과

---

*VERSION_SCHEMA v2.0 | 2026-01-28 | 중복 제거, 링크 구조화*
