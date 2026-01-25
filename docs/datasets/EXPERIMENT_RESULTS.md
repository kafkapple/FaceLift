# Experiment Results Registry

> **Navigation**: [← Index](./00_INDEX.md) | [MoC](../00_MoC_INDEX.md)
> **SSOT**: 실험 결과 비교표

---

## 1. Validation PSNR 비교

| Dataset | Experiment | Val PSNR | Coverage | PP | fx | 비고 |
|---------|------------|----------|----------|-----|-----|------|
| **D3_normalized** | E0 | **27.09** | 84.3% | 256 | 549 | ⭐ 최고 성능 |
| D7_1 | E0_paper | 20.93 | 50.5% | 256 | 549 | 안정 기준선 |
| D8 | E0 | 20.21 | 50.5% | 256 | 549 | Homography |
| M3 | E0 | ~17 | 78.5% | 가변 | **739** | fx 미정규화 |
| M3_norm | E0 | 17.09 | 78.5% | **가변** | 549 | PP 불일치 |
| M3_persample | E0 | 17.08 | 50.5% | 256 | 549 | D7_1 동등 |
| **M3_1** | E1_2_gt_alpha | TBD | 78%+ | 256 | 549 | ✅ 검증 완료 |
| **M3_2** | E1_2_gt_alpha | TBD | 78%+ | 256 | 549 | ✅ 검증 완료 ⭐ |

---

## 2. 핵심 발견

### 2.1 D3_normalized 성능 요인

```
D3_normalized PSNR 27.09 = Coverage 84.3% + PP=256 + fx=549

분해:
- Coverage 효과: +6 PSNR (50% → 84%)
- PP 정합 효과: +2-3 PSNR (가변 → 256)
- fx 정규화: 필수 (739 → 549)
```

### 2.2 M3_persample ≠ D3_normalized

| 항목 | D3_normalized | M3_persample |
|------|---------------|--------------|
| Coverage | **84.3%** | 50.5% |
| PP | 256 | 256 |
| fx | 549 | 549 |
| PSNR | 27.09 | ~17 |

**결론**: Coverage 차이가 PSNR 10+ 차이의 주요 원인

### 2.3 Translation Norm 비교

| 데이터셋 | trans_norm (per cam) | mean | std |
|----------|---------------------|------|-----|
| D3_normalized | [2.06, 2.79, 2.13, 3.23, 2.80, 3.44] | 2.74 | 0.52 |
| M3_2 | [2.70, 2.70, 2.70, 2.70, 2.70, 2.70] | 2.70 | **0.00** |

**차이점**:
- D3_normalized: 카메라별 trans 불균일 (std=0.52)
- M3_2: 완전 정규화 (std≈0)

---

## 3. 실험 우선순위

### P0 (최우선) - D3_normalized 재현
| Dataset | Experiment | 목표 |
|---------|------------|------|
| **M3_2** | E1_2_gt_alpha | Coverage↑ + PP=256 + fx=549 ⭐ |
| M3_1 | E1_2_gt_alpha | Global zoom 대안 |

### P1 (기준선)
| Dataset | Experiment | 목표 |
|---------|------------|------|
| D7_1 | E1_2_gt_alpha | 검증된 기준선 |

### P2 (가설 검증)
| Dataset | Experiment | 검증 가설 |
|---------|------------|-----------|
| M3 | E0_1_facelift | H2: fx=739 효과 |
| M3_norm | E1_2_alpha | H3: PP 가변 효과 |
| M3_1 | E1_3_gt_alpha_lgm | Alpha 1.0 (LGM 설정) |

### P3 (확장)
| Dataset | Experiment | 목표 |
|---------|------------|------|
| M3_2 | E_5view | View 수 실험 |
| D8 | E1_2_gt_alpha | Homography 효과 |

---

## 4. Config 목록

### 4.1 Dataset Configs (`configs/datasets/`)

| Config | Alias | 특징 |
|--------|-------|------|
| v13.yaml | - | Legacy, 원본 |
| D1.yaml | - | geometry_broken |
| D7_1.yaml | M1 | ✅ 기준선 |
| D7_1_t.yaml | M1_t | Temporal split |
| D7_2.yaml | - | Average scale |
| D7_t.yaml | - | Temporal split |
| D8.yaml | M2 | Homography |
| D8_1.yaml | - | + 1.3x zoom |
| D4.yaml | - | geometry_broken |
| **M3_1.yaml** | M3_1 | Global zoom ✅ |
| **M3_2.yaml** | M3_2 | Per-sample ⭐ |

### 4.2 Experiment Configs (`configs/experiments/`)

| Config | 특징 | 권장 |
|--------|------|------|
| E0_1_facelift.yaml | 원본 논문 설정 | ⚪ |
| E1_2_gt_alpha.yaml | GT mask + alpha loss | ⭐ |
| E1_3_gt_alpha_lgm.yaml | Alpha 1.0 (LGM) | ⚪ |
| E2_1_alpha_only.yaml | Alpha mask만 | ⚠️ |

---

## 5. 실험 명령어

### 5.1 M3_2 + E1_2_gt_alpha (권장)

```bash
cd /home/joon/dev/FaceLift

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E1_2_gt_alpha
```

### 5.2 D7_1 기준선

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha
```

---

## 6. 관련 문서

- [[HYPOTHESIS_VERIFICATION]] - 가설 검증 매트릭스
- [[VERSION_SCHEMA]] - Coverage 비교
- [[M3_SERIES_SPEC]] - M3 상세
- [[../EXPERIMENT_REGISTRY]] - 전체 실험 목록

---

*Experiment Results v1.0 | 2026-01-26*
