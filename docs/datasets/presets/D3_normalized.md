# D3_normalized - Reference (Highest PSNR)

> **Navigation**: [← Dataset Hub](../00_INDEX.md) | [Results](../EXPERIMENT_RESULTS.md)
> **Category**: pp_centered_shift
> **Status**: 📊 Reference (최고 성능 기준)

---

## 1. 개요

D3_normalized은 **PSNR 27.09**로 현재까지 최고 성능을 달성한 참조 데이터셋입니다.

| 항목 | 값 |
|------|-----|
| **Val PSNR** | **27.09** ⭐ |
| **Coverage** | **84.3%** (최고) |
| **fx** | 549 |
| **PP (cx, cy)** | 256, 256 |
| **trans_norm** | 불균일 (std=0.52) |

---

## 2. 성공 요인 분석

```
D3_normalized PSNR 27.09 =
    Coverage 84.3% (높음)
  + PP=256 (정확)
  + fx=549 (정규화)
```

### 요인별 기여도

| 요인 | 기여 | 근거 |
|------|------|------|
| **Coverage 84%** | +6 PSNR | D7_1(50%)→D3(84%) = 20.9→27.1 |
| **PP=256** | +3 PSNR | M3_norm(가변)→M3_1(256) |
| **fx=549** | 필수 | M3(739) → 학습 실패 |

---

## 3. 다른 데이터셋과 비교

| 데이터셋 | Coverage | PP | fx | Val PSNR | 차이 |
|----------|----------|-----|-----|----------|------|
| **D3_normalized** | **84.3%** | 256 | 549 | **27.09** | - |
| D7_1 | 50.5% | 256 | 549 | 20.93 | -6.16 |
| D8 | 50.5% | 256 | 549 | 20.21 | -6.88 |
| M3_norm | 78.5% | 가변 | 549 | 17.09 | -10.00 |
| M3_persample | 50.5% | 256 | 549 | 17.08 | -10.01 |

---

## 4. M3_2와의 비교

M3_2는 D3_normalized 재현을 목표로 합니다.

| 요소 | D3_normalized | M3_2 | 일치 |
|------|---------------|------|------|
| Coverage | 84.3% | 78%+ | ≈ (약간 낮음) |
| PP | 256 | 256 | ✅ |
| fx | 549 | 549 | ✅ |
| trans_norm | 불균일 (std=0.52) | 균일 (std=0) | 다름 |

### trans_norm 상세

| 데이터셋 | Camera 0 | Camera 1 | ... | std |
|----------|----------|----------|-----|-----|
| D3_normalized | 2.06 | 2.79 | ... | **0.52** |
| M3_2 | 2.70 | 2.70 | ... | **0.00** |

**차이점**: D3_normalized은 카메라별 거리가 불균일. M3_2는 완전 정규화.

---

## 5. 재현 실험 계획

### 목표

M3_2로 D3_normalized 수준 (PSNR 27+) 달성

### 실험 조합

| Priority | Dataset | Experiment | 목표 |
|----------|---------|------------|------|
| **P0** | M3_2 | E1_2_alpha | 27+ 목표 ⭐ |
| P1 | M3_1 | E1_2_alpha | Global zoom 비교 |
| P2 | M3_2 | E0_1_facelift | 원본 설정 비교 |

### 예상 시나리오

| 결과 | 해석 | 다음 단계 |
|------|------|-----------|
| PSNR ≥ 25 | 성공 | View 수 실험 |
| PSNR 20-24 | 부분 성공 | Coverage 조절 |
| PSNR < 20 | 실패 | trans_norm 불균일 적용 검토 |

---

## 6. 가설 검증 역할

D3_normalized은 다음 가설의 **목표 기준**입니다:

| 가설 | 내용 | D3_normalized 역할 |
|------|------|-------------------|
| **H1** | Coverage↑ → PSNR↑ | 84% Coverage = 27 PSNR |
| **H3** | PP=256 필수 | PP=256 = 성공 |

---

## 7. 데이터 위치

```
/home/joon/data/preprocessed/FaceLift_mouse/D3_normalized/
├── train/
│   ├── 000000/
│   │   ├── images/
│   │   ├── masks/
│   │   └── opencv_cameras.json
│   └── ...
└── val/
```

---

## 8. 주의사항

### 재현 불가 요소

D3_normalized의 일부 특성은 의도적 재현이 어려움:

1. **trans_norm 불균일**: 정규화 방식 차이
2. **Coverage 84.3%**: 특정 zoom 설정 필요

### 권장 접근

D3_normalized을 "정확히 복제"하기보다, **핵심 요소 (Coverage↑, PP=256, fx=549)**를 M3_2에서 달성하는 것이 목표.

---

## 9. 관련 문서

| 문서 | 내용 |
|------|------|
| [[../00_INDEX]] | Dataset Hub |
| [[../EXPERIMENT_RESULTS]] | 실험 결과 비교 |
| [[../HYPOTHESIS_VERIFICATION]] | 가설 검증 |
| [[M3_2]] | 재현 목표 데이터셋 |
| [[D7_1]] | Affine 기준선 |

---

*D3_normalized Reference v1.0 | 2026-01-26*
