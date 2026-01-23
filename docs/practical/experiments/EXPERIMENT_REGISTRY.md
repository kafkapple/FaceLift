> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Practical](../) | [Experiments](./)

# FaceLift Experiment Registry

> **Version**: 2.0.0 (2026-01-24)
> **Single Source of Truth** for all experiment configurations

---

## Quick Reference

### 권장 실험 조합

| Priority | Dataset | Experiment | 목적 | 명령어 |
|----------|---------|------------|------|--------|
| **P0** | D7_1 | E6_1_gt_alpha_sup | 문헌 기반 마스크 ⭐ | `-d D7_1 -e D7_mask_E1_gt_alpha_sup` |
| **P1** | D7_1 | E1_1_paper_random | Paper baseline | `-d D7_1 -e E1_1_paper_random` |
| **P2** | D8 | E1_1_paper_random | Homography 검증 | `-d D8 -e E1_1_paper_random` |
| **P3** | v13 | E1_1_paper_random | Legacy 비교 | `-d v13 -e E1_1_paper_random` |
| **P4** | D4 | E1_1_paper_random | PP=256 강제 비교 | `-d D4 -e E1_1_paper_random` |

---

## 명명 규칙

```
{Dataset}_{Experiment}
   │          │
   │          └── E{시리즈}_{번호}_{설명}
   │
   └── D{버전} 또는 Legacy ID (v13, D1, D4)

예시: D7_1_E2_3 = D7.1 데이터셋 + E2 시리즈 3번 실험
```

---

## 실험 시리즈 개요

### E1: Paper Baseline (No Mask)

| Config | Views | Selection | Mask | 용도 |
|--------|-------|-----------|------|------|
| **E1_1_paper_random** | 4 | random | none | ★ 기본 baseline |
| E1_2_paper_fixed | 4 | fixed | none | Fixed view 비교 |
| E1_3_5v_paper_random | 5 | random | none | 5뷰 baseline |
| E1_4_5v_paper_fixed | 5 | fixed | none | 5뷰 fixed |

### E2: Mask Mode (GT vs Alpha)

| Config | Mask | Views | Selection | 용도 |
|--------|------|-------|-----------|------|
| **E2_1_gt_mask_random** | gt | 4 | random | ★ GT 기본 |
| E2_2_gt_mask | gt | 4 | fixed | GT fixed |
| E2_3_alpha_random | alpha | 4 | random | Alpha 기본 |
| E2_4_alpha_fixed | alpha | 4 | fixed | Alpha fixed |
| E2_5_5v_gt_random | gt | 5 | random | 5뷰 GT |
| E2_6_5v_gt_fixed | gt | 5 | fixed | 5뷰 GT fixed |

### E3: View Count (Alpha 기본)

| Config | Views | Selection | 용도 |
|--------|-------|-----------|------|
| E3_2_5v_alpha | 5 | fixed | ★ 5뷰 기준 |
| E3_3_4v_alpha | 4 | fixed | 4뷰 비교 |
| E3_4_4v_alpha_random | 4 | random | 4뷰 random |

### E4: Alpha Tuning

| Config | Threshold | Alpha Loss | Opacity Reg | 용도 |
|--------|-----------|------------|-------------|------|
| E4_1_alpha_basic | 0.5 | 0.0 | 0.0 | 기본 |
| **E4_2_alpha_conservative** | 0.7 | 0.3 | 0.0 | ★ 권장 |
| E4_3_alpha_aggressive | 0.5 | 0.5 | 0.01 | 공격적 |
| E4_4_4v_conservative | 0.7 | 0.3 | 0.0 | 4뷰 conservative |
| E4_5_4v_aggressive | 0.5 | 0.5 | 0.01 | 4뷰 aggressive |
| E4_6_aggressive_exclude_v5 | 0.5 | 0.5 | 0.01 | v5 제외 |
| E4_7_optimal_alpha | 0.7 | 0.5 | 0.02 | 최적화 시도 |

### E5: Loss Ablation / Threshold Variants

| Config | 핵심 변경 | 용도 |
|--------|-----------|------|
| E5_1_5v_alpha_random | 5v + random | Baseline |
| E5_2_alpha_loss_only | alpha_loss만 | Loss 분리 |
| E5_3_4v_alpha_loss | 4v + alpha_loss | 4뷰 loss |
| E5_4_rgb_mask_01 | RGB 마스크 0.1 | RGB 실험 |
| E5_5_rgb_mask_02 | RGB 마스크 0.2 | RGB 실험 |
| **E5_6_alpha_thresh_07** | threshold=0.7 | ★ 높은 threshold |
| E5_6b_alpha_thresh_07_random | 0.7 + random | Random 변형 |
| E5_7_optimal_alpha | 최적화 | 종합 |
| E5_8_alpha_thresh_08 | threshold=0.8 | 더 높은 threshold |
| E5_9_gt_mask | GT mask | GT 비교 |
| E5_10_no_mask | No mask | Baseline |

### E6: Literature-Based Mask (신규) ⭐

> **문헌 기반 설계**: LGM, Pose Splatter, Object-Centric 2DGS

| Config | Mode | Alpha Loss | BG Loss | 문헌 | Priority |
|--------|------|------------|---------|------|----------|
| D7_mask_E0_baseline | none | - | - | - | P4 |
| **D7_mask_E1_gt_alpha_sup** | gt | 0.1 MSE | - | LGM+Pose Splatter | **P0** ⭐ |
| D7_mask_E2_composite | composite | 0.05 MSE | - | Nerfstudio | P1 |
| D7_mask_E3_bg_penalty | none | 0.1 MSE | 0.5 | Obj-Centric 2DGS | P2 |
| D7_mask_E4_alpha_sup_only | none | 0.1 MSE | - | LGM | P3 |

**E6 권장 설정**:
```yaml
losses:
  mask_mode: gt              # GT mask로 RGB loss 제한
  normalize_by_mask: true    # Pose Splatter: 작은 전경 필수
  alpha_loss_weight: 0.1     # LGM: alpha supervision
  alpha_loss_type: mse       # BCE보다 안정적
```

→ 상세: [Mask_Experiment_Priority](./Mask_Experiment_Priority.md), [Mask_Literature_Review](../../theory/mask/Mask_Literature_Review.md)

### E_quick: 빠른 테스트

| Config | Steps | 용도 |
|--------|-------|------|
| E_quick_alpha | 500 | 빠른 검증 (~10분) |

---

## 가설 검증 매트릭스

### H1: Dataset 기하학 영향

| 비교 | 가설 | 지표 |
|------|------|------|
| D7_1 vs v13 | PP-shift > Legacy | PSNR, ghosting |
| D7_1 vs D4 | 정확한 PP > 강제 PP | ray error |
| D8 vs D7_1 | Homography > Affine | skew 보정 |

### H2: Mask 방식 영향

| 비교 | 가설 | 지표 |
|------|------|------|
| E2_1 vs E2_3 | GT mask > Alpha mask | mask_iou |
| E6_1 vs E2_1 | Alpha supervision 추가 효과 | shape convergence |

### H3: View Count 영향

| 비교 | 가설 | 지표 |
|------|------|------|
| E3_2 (5v) vs E3_3 (4v) | 더 많은 뷰 > | PSNR |

---

## Config 파일 위치

```
configs/
├── datasets/                    # 데이터셋 설정 (14개)
│   ├── v13.yaml                 # Legacy
│   ├── D4.yaml                  # PP=256 강제
│   ├── D7_1.yaml                # ★ 표준
│   ├── D8.yaml                  # ★★ Homography
│   └── ...
│
├── experiments/                 # 실험 설정 (33개)
│   ├── E1_*.yaml               # Paper baseline
│   ├── E2_*.yaml               # Mask mode
│   ├── E3_*.yaml               # View count
│   ├── E4_*.yaml               # Alpha tuning
│   ├── E5_*.yaml               # Loss ablation
│   └── E_quick_alpha.yaml      # Quick test
│
└── mouse/mask_exp/              # 문헌 기반 마스크 (5개)
    ├── D7_mask_E0_baseline.yaml
    ├── D7_mask_E1_gt_alpha_sup.yaml  # ★ P0
    ├── D7_mask_E2_composite.yaml
    ├── D7_mask_E3_bg_penalty.yaml
    └── D7_mask_E4_alpha_sup_only.yaml
```

---

## Related Documents

- [MOUSE_QUICK_REFERENCE](../MOUSE_QUICK_REFERENCE.md) - 명령어 빠른 참조
- [Mask_Experiment_Priority](./Mask_Experiment_Priority.md) - 마스크 실험 우선순위
- [Mask_Literature_Review](../../theory/mask/Mask_Literature_Review.md) - 마스크 문헌 조사
- [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) - 데이터셋 레지스트리

---

*Experiment Registry v2.0.0 | 2026-01-24*
