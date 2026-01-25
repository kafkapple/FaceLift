# Dataset Version Schema

> **최종 업데이트**: 2026-01-26
> **관련**: [[00_MoC_INDEX]], [[PREPROCESSING_REGISTRY]], [[M3_SERIES_SPEC]]

---

## 핵심 개념

| 접미사 | 의미 | 설명 |
|--------|------|------|
| **_t** | **Temporal Split** | 시간순 3분할 (train→val→test). Data leak 방지 |
| 없음 | Random Split | 전 구간에서 무작위 샘플링. 오버핏 위험 있으나 학습 촉진 |

---

## M-Series (권장)

| Alias | Preset | 변환 | PP | fx | 상태 |
|-------|--------|------|-----|-----|------|
| **M1** | D7.1 | Affine | 256 | 549 | ✅ 기준선 |
| **M2** | D8 | Homography | 256 | 549 | ✅ 정밀 |
| **M3** | D10.3 | Homo+Zoom | 가변 | 739 | ⚠️ fx 버그 |
| **M3_1** | M3_1 | Global Zoom | 256 | 549 | ✅ 검증됨 |
| **M3_2** | M3_2 | Per-sample | 256 | 549 | ✅ 검증됨 ⭐ |

---

## D7 계열 관계도

```
D7 (기본: random split, fx_only scale)
├── D7_1: individual scale (별도 scale_x, scale_y)
│   └── D7_1_t: D7_1 + temporal split (★ 권장)
├── D7_2: average scale (동일 scale_x = scale_y)
├── D7_5: optimal scale
│   └── D7_5b: object-aware optimal
└── D7_t: D7 + temporal split
```

---

## Split 방식 비교

| 방식 | Train | Val | Test | 특징 |
|------|-------|-----|------|------|
| **Random** | 전구간 무작위 | 전구간 무작위 | - | Data leak 위험, 학습 촉진 |
| **Temporal (_t)** | 0-60% 시간 | 60-80% 시간 | 80-100% 시간 | Pose-splatter 호환, 공정 평가 |

---

## VERSION_HIERARCHY 분류

| 분류 | 프리셋 | 설명 |
|------|--------|------|
| **affine** | D7, D7.1, D7.2 | 기본 변환, skew 무시 |
| **homography** | D8, D8.1, D8.2 | skew 보정 포함 |
| **homography_zoom** | D10.3, M3_1, M3_2 | homography + adaptive zoom |
| **geometry_broken** | D1, D4, D6-* | ⛔ PP 미보정, 사용 금지 |
| **experimental** | D10, D10.1, D10.2 | up_alignment 문제 있음 |

---

## 상세 문서

- [[PREPROCESSING_REGISTRY]] - 프리셋 정의, 버그 이력
- [[theory/PP_FX_MVG_ANALYSIS]] - PP/fx 이론
- [[M3_SERIES_SPEC]] - M3 시리즈 상세

---

*VERSION_SCHEMA v1.0 | 2026-01-26*
