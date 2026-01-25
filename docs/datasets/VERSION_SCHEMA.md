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

---

## Coverage 비교

| 데이터셋 | fx | PP (cx) | Coverage | Val PSNR | 비고 |
|----------|-----|---------|----------|----------|------|
| **D3_normalized** | 549 | 256 고정 | **84.3%** | **27.09** | ⭐ 최고 성능 |
| D7_1 | 549 | 256 고정 | 50.5% | 20.93 | 안정 기준선 |
| D8 | 549 | 256 고정 | 50.5% | 20.21 | 정밀 기하학 |
| M3 | **739** | 가변 | 78.5% | ~17 | H2 검증용 |
| M3_norm | 549 | **가변** | 78.5% | ~17 | PP 불일치 |
| M3_persample | 549 | 256 고정 | **50.5%** | - | D7_1과 유사 |

### 핵심 발견

```
D3_normalized ≠ M3_persample (Coverage 차이\!)

D3_normalized: Coverage 84.3%, PP=256
M3_persample:  Coverage 50.5%, PP=256  ← D7_1과 같음\!

결론: D3_normalized 재현에는 높은 Coverage + PP=256 + fx=549 필요
```

---

## Global vs Per-sample Zoom

### 비교 표

| 방식 | fx | 이미지 | 장점 | 단점 |
|------|-----|--------|------|------|
| **Global zoom** | 모든 샘플 동일 (739→549) | 동일 비율 확대 | 단순, 일관성 | 개별 최적화 안됨 |
| **Per-sample zoom** | 샘플마다 다름 (600~900) | 개별 최적 확대 | 각 샘플 최적 | 복잡, fx 가변 |

### M3 시리즈 적용

| 데이터셋 | Zoom 방식 | Zoom 값 | fx 결과 | 목적 |
|----------|-----------|---------|---------|------|
| **M3** | Global | 1.347 고정 | 739 (미정규화) | H2 검증 (fx≠549) |
| **M3_norm** | Global | 1.347 고정 | 549 (재정규화) | H1, H3 검증 |
| **M3_1** | Global | 1.347 고정 | 549 | Center-aligned |
| **M3_2** | Per-sample | 가변 | 549 (재정규화) | 최적 Coverage |

### 핵심 질문 정리

| 질문 | 답변 |
|------|------|
| M3는 버그인가? | ❌ 아님, H2 검증용 유효 데이터 |
| 고정 zoom 문제? | ⚪ 문제없음 (global zoom 의도된 설계) |
| Per-sample이 더 좋은가? | 이론적 Yes, 실제로는 검증 필요 |
| fx 가변 상관없나? | pretrained 호환성에 영향 가능 (검증 필요) |

### 실험 우선순위

```
Phase 1: Global zoom 기반 검증 (현재)
├── M3 (fx=739) → H2 검증
├── M3_norm (fx=549) → H1, H3 검증
└── M3_1, M3_2 (fx=549, PP=256) → MVG-correct

Phase 2: Per-sample zoom 검토 (선택적)
└── Phase 1 결과가 아쉬우면 시도
```

---

## Center-Aligned Zoom 주의사항

### Clipping 위험

```
원본 (512×512):         Center-aligned Zoom (1.5x):
┌─────────────────┐     ┌─────────────────┐
│                 │     │   ┌─────────┐   │
│            🐭  │  →  │   │         │   │  ← 가장자리 생쥐
│                 │     │   │    🐭? │   │  ← CLIPPED\!
│                 │     │   └─────────┘   │
└─────────────────┘     └─────────────────┘
```

### 발생 조건

1. 생쥐가 이미지 가장자리에 있을 때
2. 높은 zoom factor (1.5x+) 적용 시

### 현재 상태

- M3_1, M3_2: `zoom_center_mode: "image"` (center-aligned)
- Clipping 방지 로직: **없음** (주의 필요)

---

*VERSION_SCHEMA v1.1 | 2026-01-26 | Coverage, Zoom 비교 추가*
