# Experiment Naming Convention v3.0

> Last Updated: 2026-01-23

## Overview

계층적 실험 명명 체계로, E번호가 마스크 모드 계열을, 세부 번호가 변형을 나타냅니다.

## Format

```
{Dataset}_E{MaskMode}_{SubNum}[_fixed]

예: D7_1_E2_1        = D7.1 + E2(gt+alpha) + 기본(5v, random)
    D7_1_E2_2        = D7.1 + E2 + 4v
    D7_1_E2_1_fixed  = D7.1 + E2 + 기본 + fixed view
```

## E 계열 정의 (Mask Mode)

| E# | mask_mode | normalize | α_loss | bg_loss | Reference |
|----|-----------|-----------|--------|---------|-----------|
| **E0** | none | false | 0.0 | 0.0 | GS-LRM Paper |
| **E1** | gt | true | 0.0 | 0.0 | Pose Splatter |
| **E2** ⭐ | gt | true | **0.1** | 0.0 | LGM + Pose Splatter |
| **E3** | none | false | 0.1 | 0.0 | LGM only |
| **E4** | none | false | 0.1 | **0.5** | Object-Centric 2DGS |
| **E5** | composite | false | 0.05 | 0.0 | Nerfstudio |

## SubNum 규칙

| SubNum | 의미 | num_input_views | random_view_selection |
|--------|------|-----------------|----------------------|
| **_1** | 기본 | 5 | true |
| **_2** | 4v | 4 | true |
| **_3** | 6v | 6 | true |
| **_4** | conservative | 5 | true (α=0.05) |
| **_5** | aggressive | 5 | true (α=0.2) |

## Suffix 규칙

| Suffix | 의미 |
|--------|------|
| (없음) | random view selection (기본) |
| **_fixed** | fixed view selection |

## Dataset ID

| Dataset | 설명 | 권장 |
|---------|------|------|
| **D7_1** | Individual scale, geometric correction | ⭐ 권장 |
| D7_2 | Average scale | - |
| D7_t | Temporal split | ablation |
| D8 | Homography transform | - |
| D9 | New preprocessing | testing |
| v13 | Legacy | - |

## Priority

| Priority | Config | 설명 |
|----------|--------|------|
| **P0** | D7_1_E2_1 | 권장 (GT mask + alpha supervision) |
| P1 | D7_1_E0_1 | Baseline 비교 |
| P2 | D7_1_E1_1 | GT only 비교 |
| P3 | D7_1_E2_2 | 4v ablation |
| P4 | D7_1_E4_1 | BG penalty |

## Examples

```bash
# P0: 권장 실험
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_1.yaml

# View ablation
CUDA_VISIBLE_DEVICES=4 ... --config configs/mouse/D7_1_E2_2.yaml  # 4v

# Fixed vs Random
CUDA_VISIBLE_DEVICES=6 ... --config configs/mouse/D7_1_E2_1_fixed.yaml
```

---

*Engram v1.0 | Schema v3.0*
