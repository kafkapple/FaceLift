# Experiment Naming Convention

> **Navigation**: [← Index](./00_INDEX.md) | [Results](./EXPERIMENT_RESULTS.md)
> **SSOT**: 모든 실험 설정의 명명 규칙 및 체계

---

## 1. 명명 원칙

### Core Principle: "random=1"

```
기준선(baseline) = random view selection
→ 실험 번호 1번으로 지정
```

**이유**: Random view selection이 가장 일반적인 설정이므로 기준선으로 사용

### 명명 형식

```
E{시리즈}_{번호}_{설명}

예시:
- E1_1_paper_baseline  → E1 시리즈, 1번, 원본 논문 설정
- E2_1_gt_alpha        → E2 시리즈, 1번, GT mask + alpha loss
```

---

## 2. 실험 시리즈 계층

| 시리즈 | 주제 | 변수 | 설정 수 |
|--------|------|------|---------|
| **E1** | Paper Baseline | view selection | 2 |
| **E2** | Mask Mode | gt / alpha / none | 3 |
| **E3** | View Count | 4v / 5v / 6v | 3 |
| **E4** | Alpha Tuning | weight 조절 | 3 |
| **E5** | Loss Ablation | perceptual / bg | 3 |

---

## 3. 상세 설정

### E1: Paper Baseline

| Config | 설명 | 핵심 설정 |
|--------|------|-----------|
| **E1_1_paper_baseline** | 원본 논문 설정 | fixed view [0,1,2,3] |
| **E1_2_random_baseline** | Random view | random 4-view selection |

### E2: Mask Mode

| Config | 설명 | mask_mode | alpha_loss |
|--------|------|-----------|------------|
| **E2_1_gt_alpha** | GT mask + alpha | gt | 0.1 |
| **E2_2_alpha_only** | Alpha mask only | alpha | 0.1 |
| **E2_3_no_mask** | No masking | none | 0.0 |

**권장**: E2_1_gt_alpha (GT mask가 가장 안정적)

### E3: View Count

| Config | 설명 | input_views | target_views |
|--------|------|-------------|--------------|
| **E3_1_4v** | 4 view | 4 | 2 |
| **E3_2_5v** | 5 view | 5 | 1 |
| **E3_3_6v** | 6 view | 6 | 0 (self) |

### E4: Alpha Tuning

| Config | 설명 | alpha_loss_weight |
|--------|------|-------------------|
| **E4_1_alpha_005** | Low alpha | 0.05 |
| **E4_2_alpha_010** | Medium (기본) | 0.10 |
| **E4_3_alpha_020** | High alpha | 0.20 |

### E5: Loss Ablation

| Config | 설명 | perceptual | background |
|--------|------|------------|------------|
| **E5_1_no_perceptual** | No perceptual | 0.0 | 0.1 |
| **E5_2_no_background** | No background | 0.1 | 0.0 |
| **E5_3_both** | Both enabled | 0.1 | 0.1 |

---

## 4. 우선순위 체계

| Priority | 의미 | 실험 예시 |
|----------|------|-----------|
| **P0** | 즉시 실행 | M3_2 + E2_1_gt_alpha |
| **P1** | 높은 우선순위 | M3_1 + E2_1_gt_alpha |
| **P2** | 중간 | View count 비교 |
| **P3** | 낮음 | Alpha tuning |
| **P4** | 선택적 | Loss ablation |

---

## 5. 데이터셋 × 실험 매트릭스

### 권장 조합

| Dataset | 권장 실험 | 우선순위 | 목적 |
|---------|-----------|----------|------|
| **M3_2** | E2_1_gt_alpha | P0 | 27+ PSNR 목표 |
| **M3_1** | E2_1_gt_alpha | P1 | Global zoom 비교 |
| **D7_1** | E1_1_paper | P2 | Affine baseline |
| **D8** | E2_1_gt_alpha | P2 | Homography 검증 |

### 실행 명령어

```bash
cd /home/joon/dev/FaceLift

# P0: 권장 실험
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2 -e E2_1_gt_alpha

# P1: 대안 실험
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_1 -e E2_1_gt_alpha
```

---

## 6. 설정 파일 위치

```
configs/
├── datasets/
│   ├── D7_1.yaml
│   ├── D8.yaml
│   ├── M3_1.yaml
│   └── M3_2.yaml
└── experiments/
    ├── E1_1_paper_baseline.yaml
    ├── E1_2_random_baseline.yaml
    ├── E2_1_gt_alpha.yaml
    ├── E2_2_alpha_only.yaml
    ├── E2_3_no_mask.yaml
    └── ...
```

---

## 7. 관련 문서

- [[00_INDEX]] - Dataset Hub
- [[EXPERIMENT_RESULTS]] - 실험 결과 비교
- [[HYPOTHESIS_VERIFICATION]] - 가설 검증
- [[VIEW_SELECTION_ANALYSIS]] - View 선택 분석

---

*Experiment Naming Convention v1.0 | 2026-01-26*
