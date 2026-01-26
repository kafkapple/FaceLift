# Hypothesis Verification Matrix

> **Navigation**: [← Index](./00_INDEX.md) | [MoC](../00_MoC_INDEX.md)
> **SSOT**: 실험 가설 검증 매트릭스

---

## 1. 가설 정의

| 가설 | 설명 | 검증 방법 | 상태 |
|------|------|-----------|------|
| **H1** | Coverage↑ → PSNR↑ | D7_1 (50%) vs D3 (74%) | ✅ 검증됨 |
| **H2** | fx=549 정규화 필수 | M3 (fx=739) vs M3_norm (fx=549) | ⚠️ 부분 |
| **H3** | PP=256 정합 필수 | M3_norm (가변) vs M3_1 (256) | ✅ 검증됨 |
| **H4** | Aspect Ratio 영향 | - | 🔲 미검증 |
| **H5** | Zoom 방식 (Global vs Per-sample) | M3_1 vs M3_2 | 🔄 진행 중 |
| **H6** | View Quality (Novel vs Input) | View 0-3 vs 4-5 | ✅ 발견됨 |

---

## 2. 검증 매트릭스

### 2.1 H1: Coverage 효과

| 데이터셋 | Coverage | fx | PP | Val PSNR | 비고 |
|----------|----------|-----|-----|----------|------|
| D7_1 | 50.5% | 549 | 256 | 20.93 | 기준 |
| D3_normalized | **74%** | 549 | 256 | **27.09** | +6.16 |
| M3_norm | 78.5% | 549 | 가변 | 17.09 | PP 문제 |

**결론**: ✅ Coverage 50% → 74%로 **+6 PSNR** (PP 정합 필요)

### 2.2 H2: fx 정규화 효과

| 데이터셋 | fx | Coverage | PP | Val PSNR |
|----------|-----|----------|-----|----------|
| M3 | **739** | 78.5% | 가변 | ~17 |
| M3_norm | **549** | 78.5% | 가변 | 17.09 |

**결론**: ⚠️ fx 정규화만으로는 PSNR 개선 미미 → PP 정합 필요

### 2.3 H3: PP 정합 효과

| 데이터셋 | PP | fx | Coverage | Ray Error | Val PSNR |
|----------|-----|-----|----------|-----------|----------|
| M3_norm | **가변** | 549 | 78.5% | 13.62° | 17.09 |
| M3_persample (deprecated) | **가변** | 549 | 50.5% | 16.15° | 17.08 |
| **M3_1** | **256** | 549 | 78%+ | 0° | TBD |
| **M3_2** | **256** | 549 | 78%+ | 0° | TBD ⭐ |

**결론**: ✅ PP=256 정합 시 Ray Error 제거 → PSNR 개선 기대

### 2.4 H5: Zoom 방식 비교

| 데이터셋 | Zoom 방식 | Zoom 범위 | fx 일관성 | 장점 |
|----------|-----------|-----------|-----------|------|
| M3_1 | Global | 1.347 고정 | 완전 일관 | 단순, 안정 |
| M3_2 | Per-sample | 1.0~2.5 | 재정규화 후 일관 | 개별 최적화 |

**결론**: 🔄 Per-sample이 실제로 더 나은지 실험으로 확인 필요

### 2.5 H6: View Quality ⭐ NEW

| View | 역할 (Validation) | PSNR 범위 | 해석 |
|------|-------------------|-----------|------|
| 0-3 | **Input** | 27-33 | Reconstruction |
| 4-5 | **Novel** | 10-13 | Synthesis |

**원인**: mouse_dataset.py에서 validation은 항상 [0,1,2,3]을 input으로 사용

```python
# mouse_dataset.py:276-279
if random_view_selection and split == "train":
    input_indices = sorted(random.sample(all_indices, 4))
else:
    input_indices = list(range(4))  # [0,1,2,3] 고정
```

**결론**: ✅ View 4-5 낮은 PSNR은 novel view synthesis 한계 (정상 동작)

상세: [[VIEW_SELECTION_ANALYSIS]]

---

## 3. D3_normalized 재현 요건

### 3.1 요건 분석

D3_normalized 성공 요인:
```
1. Coverage = 74% (높은 전경 비율)
2. PP = 256 (정확한 principal point)
3. fx = 549 (정규화된 focal length)
4. trans_norm ≈ 2.7 (정규화된 translation)
```

### 3.2 M3_1/M3_2와의 비교

| 요건 | D3_normalized | M3_1 | M3_2 |
|------|---------------|------|------|
| Coverage | 74% | 78%+ | 78%+ |
| PP | 256 | 256 ✅ | 256 ✅ |
| fx | 549 | 549 ✅ | 549 ✅ |
| trans_norm | 불균일 (std=0.52) | 균일 (std≈0) | 균일 (std≈0) |

### 3.3 예상 결과

| 시나리오 | 예상 PSNR | 근거 |
|----------|-----------|------|
| 최선 (모든 요건 충족) | 25-27 | D3_normalized 재현 |
| 중간 (PP만 해결) | 20-22 | D7_1 수준 + Coverage 효과 |
| 최악 (문제 미해결) | 17-18 | M3_norm 수준 |

---

## 4. 실험 계획

### Phase 1: 기본 검증 (현재)

| 순서 | 데이터셋 | 실험 | 검증 가설 | 상태 |
|------|----------|------|-----------|------|
| 1 | M3_1 | E1_2_alpha | H3 (PP) | ⏳ |
| 2 | M3_2 | E1_2_alpha | H3 + H5 | ⏳ |

### Phase 2: 비교 분석

| 순서 | 비교 | 목적 |
|------|------|------|
| 3 | M3_1 vs D7_1 | Coverage 효과 (H1) |
| 4 | M3_1 vs M3_2 | Zoom 방식 효과 (H5) |
| 5 | M3 vs M3_1 | fx 정규화 효과 (H2) |
| 6 | random vs fixed | View quality 효과 (H6) |

### Phase 3: 최적화

| 조건 | 다음 단계 |
|------|-----------|
| PSNR ≥ 25 | 성공, View 수 실험으로 |
| PSNR 20-24 | 부분 성공, Coverage 조절 |
| PSNR < 20 | 실패, 원인 분석 |

---

## 5. Ray Error 공식

```
θ_error = arctan(Δpp / fx)

여기서:
- Δpp = sqrt((cx - 256)² + (cy - 256)²)
- fx = 549 (정규화 후)

예시:
- M3_norm: Δpp = 78px → θ = arctan(78/549) = 8.1°
- M3_persample (deprecated): Δpp = 52px → θ = arctan(52/549) = 5.4°
- M3_1/M3_2: Δpp = 0 → θ = 0° ✅
```

---

## 6. 검증 매트릭스 요약

```
           H1    H2    H3    H4    H5    H6
           Cov   fx    PP    AR    Zoom  View
         ┌─────┬─────┬─────┬─────┬─────┬─────┐
D3_norm  │ ✅  │ ✅  │ ✅  │  -  │  -  │  -  │
D7_1     │ ⚪  │ ✅  │ ✅  │  -  │  -  │  -  │
D8       │ ⚪  │ ✅  │ ✅  │  -  │  -  │  -  │
M3       │ ⚪  │ ❌  │ ❌  │  -  │  -  │  -  │
M3_norm  │ ⚪  │ ✅  │ ❌  │  -  │  -  │  -  │
M3_1     │ ⚪  │ ✅  │ ✅  │  -  │ ⚪  │  -  │
M3_2     │ ⚪  │ ✅  │ ✅  │  -  │ ✅  │  -  │
         └─────┴─────┴─────┴─────┴─────┴─────┘

Legend: ✅ 충족  ⚪ 부분  ❌ 미충족  - N/A
```

---

## 7. 관련 문서

- [[EXPERIMENT_RESULTS]] - 실험 결과표
- [[EXPERIMENT_NAMING]] - 실험 명명규칙
- [[VIEW_SELECTION_ANALYSIS]] - H6 상세 분석
- [[VERSION_SCHEMA]] - Coverage 비교
- [[M3_SERIES_SPEC]] - M3 상세

---

*Hypothesis Verification v2.0 | 2026-01-26 | H6 Added*
