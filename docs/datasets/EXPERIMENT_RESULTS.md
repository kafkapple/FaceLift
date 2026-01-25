# Experiment Results Registry

> **Navigation**: [← Index](./00_INDEX.md) | [MoC](../00_MoC_INDEX.md)
> **SSOT**: 실험 결과 비교표

---

## 1. Validation PSNR 비교

| Dataset | Experiment | Val PSNR | Coverage | PP | fx | 비고 |
|---------|------------|----------|----------|-----|-----|------|
| **D3_normalized** | E0 | **27.09** | 74% | 256 | 549 | ⭐ 최고 성능 |
| D7_1 | E0_paper | 20.93 | 50% | 256 | 549 | 안정 기준선 |
| D8 | E0 | 20.21 | 50% | 256 | 549 | Homography |
| M3 | E0 | ~17 | 78% | 가변 | **739** | fx 미정규화 |
| M3_norm | E0 | 17.09 | 78% | **가변** | 549 | PP 불일치 |
| **M3_1** | E1_2_alpha | TBD | 78%+ | 256 | 549 | ✅ 검증 완료 |
| **M3_2** | E1_2_alpha | TBD | 78%+ | 256 | 549 | ✅ 검증 완료 ⭐ |

---

## 2. View별 PSNR 분석 ⭐ NEW

### Input vs Novel Views

| View | 역할 (Val) | PSNR 범위 | 해석 |
|------|-----------|-----------|------|
| 0-3 | Input | 27-33 | Reconstruction |
| 4-5 | Novel | 10-13 | Synthesis |

```
View Quality Distribution (Validation)

PSNR
 35 ┤
 30 ┤  ████████████████
 25 ┤  ████████████████
 20 ┤  ████████████████
 15 ┤  ████████████████
 10 ┤  ████████████████  ████████
  5 ┤                    ████████
    └──────────────────────────────
       V0  V1  V2  V3  V4  V5
       ──────────────  ────────
        Input Views    Novel Views
```

**핵심**: View 4-5 낮은 PSNR은 정상 (novel view synthesis 한계)

상세: [[VIEW_SELECTION_ANALYSIS]]

---

## 3. PP 정합 영향 분석

### PP-Aligned vs PP-Misaligned

| 그룹 | 데이터셋 | PP | 평균 PSNR |
|------|---------|-----|-----------|
| **PP-Aligned** | D3, D7_1, D8 | 256 | **~22** |
| PP-Misaligned | M3_* (norm/persample) | 가변 | **~17** |

**결론**: PP 미정합 → 약 5 PSNR 손실 (Ray error ~13°)

---

## 4. Coverage 영향 분석

### Coverage vs PSNR 상관관계

| Coverage | 대표 데이터셋 | PSNR | 증가분 |
|----------|--------------|------|--------|
| 50% | D7_1 | 20.93 | 기준 |
| 74% | D3_normalized | 27.09 | **+6.16** |
| 78% | M3_2 (예상) | 25+ | +4~5 |

### 회귀 분석

```
PSNR ≈ 0.25 × Coverage(%) + 8.5

예측:
- 50% → 21.0 (실제 20.93)
- 74% → 27.0 (실제 27.09)
- 78% → 28.0 (M3_2 목표)
```

---

## 5. 핵심 발견

### 5.1 D3_normalized 성능 요인

```
D3_normalized PSNR 27.09 = Coverage 74% + PP=256 + fx=549

분해:
- Coverage 효과: +6 PSNR (50% → 74%)
- PP 정합 효과: +5 PSNR (가변 → 256)
- fx 정규화: 필수 (739 → 549)
```

### 5.2 Translation Norm 비교

| 데이터셋 | trans_norm (per cam) | mean | std |
|----------|---------------------|------|-----|
| D3_normalized | [2.06, 2.79, 2.13, 3.23, 2.80, 3.44] | 2.74 | 0.52 |
| M3_2 | [2.70, 2.70, 2.70, 2.70, 2.70, 2.70] | 2.70 | **0.00** |

**차이점**: M3_2가 더 균일한 정규화

---

## 6. 실험 우선순위

### P0 (최우선) - D3_normalized 재현
| Dataset | Experiment | 목표 |
|---------|------------|------|
| **M3_2** | E1_2_alpha | Coverage↑ + PP=256 + fx=549 ⭐ |
| M3_1 | E1_2_alpha | Global zoom 대안 |

### P1 (기준선)
| Dataset | Experiment | 목표 |
|---------|------------|------|
| D7_1 | E1_2_alpha | 검증된 기준선 |

### P2 (가설 검증)
| Dataset | Experiment | 검증 가설 |
|---------|------------|-----------|
| M3_2 | E1_2_random | H6: View selection |
| M3_2 | E3_3_6v | Novel view 제거 |

---

## 7. 실험 명령어

### 7.1 M3_2 + E1_2_alpha (권장)

```bash
cd /home/joon/dev/FaceLift

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_alpha
```

### 7.2 H6 검증 (Random view)

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_random_baseline
```

### 7.3 D7_1 기준선

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_alpha
```

---

## 8. 결과 업데이트 로그

| 날짜 | 데이터셋 | 실험 | PSNR | 비고 |
|------|---------|------|------|------|
| 2026-01-26 | D3_normalized | - | 27.09 | 참조 기준 |
| 2026-01-26 | D7_1 | E1_1 | 20.93 | M1 baseline |
| 2026-01-26 | D8 | E1_1 | 20.21 | M2 baseline |
| - | M3_2 | E2_1 | TBD | 진행 예정 |

---

## 9. 관련 문서

- [[HYPOTHESIS_VERIFICATION]] - 가설 검증 매트릭스
- [[EXPERIMENT_NAMING]] - 실험 명명규칙
- [[VIEW_SELECTION_ANALYSIS]] - View 분석
- [[VERSION_SCHEMA]] - Coverage 비교
- [[M3_SERIES_SPEC]] - M3 상세

---

*Experiment Results v2.0 | 2026-01-26 | View Analysis Added*
