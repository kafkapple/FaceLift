# Experiment Naming Convention v3.1

> Last Updated: 2026-01-23

## Format

```
D{Dataset}_E{MaskMode}_{keyword}[_variant].yaml

예시:
  D7_1_E2_gt_alpha.yaml       = D7.1 + E2(gt+alpha) + 기본
  D7_1_E2_gt_alpha_4v.yaml    = D7.1 + E2 + 4뷰 ablation
  D7_1_E2_overfit_1v.yaml     = D7.1 + E2 + 1뷰 overfit
```

---

## Dataset ID

| ID | 설명 | 권장 |
|----|------|------|
| **D7_1** | Individual scale, geometric correction | ⭐ 권장 |
| D7_2 | Average scale | - |
| D8 | Homography transform | - |
| D9 | New preprocessing | testing |

---

## Experiment ID (E0-E5)

| E# | mask_mode | normalize | α_loss | bg_loss | 키워드 |
|----|-----------|-----------|--------|---------|--------|
| **E0** | none | false | 0.0 | 0.0 | baseline |
| **E1** | gt | true | 0.0 | 0.0 | gt |
| **E2** ⭐ | gt | true | 0.1 | 0.0 | gt_alpha |
| **E3** | none | false | 0.1 | 0.0 | alpha |
| **E4** | none | false | 0.1 | 0.5 | bg_penalty |
| **E5** | composite | false | 0.05 | 0.0 | composite |

---

## 키워드 규칙

| 키워드 | 의미 | 예시 |
|--------|------|------|
| baseline | 마스크/alpha 없음 | E0_baseline |
| gt | GT 마스크만 | E1_gt |
| gt_alpha | GT + alpha supervision | E2_gt_alpha |
| alpha | Alpha supervision만 | E3_alpha |
| bg_penalty | 배경 페널티 | E4_bg_penalty |
| composite | 배경 합성 | E5_composite |

---

## Variant (변형)

| Variant | 의미 | 예시 |
|---------|------|------|
| _4v | 4개 입력 뷰 | gt_alpha_4v |
| _6v | 6개 입력 뷰 | gt_alpha_6v |
| _fixed | 고정 뷰 선택 | gt_alpha_fixed |
| _overfit_1v | 1뷰 overfit | overfit_1v |

---

## 현재 사용 가능한 Configs

| Config | 설명 | Priority |
|--------|------|----------|
| **D7_1_E0_baseline** | No mask, no alpha | P2 |
| **D7_1_E1_gt** | GT mask only | P3 |
| **D7_1_E2_gt_alpha** | GT + alpha ⭐ | **P0** |
| **D7_1_E2_gt_alpha_4v** | 4v ablation | P4 |
| **D7_1_E2_overfit_1v** | 1v overfit test | P5 |
| **D7_1_E3_alpha** | Alpha only | P4 |
| **D7_1_E4_bg_penalty** | BG penalty | P3 |

---

## Quick Commands

```bash
# 권장 실험
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/D7_1_E2_gt_alpha.yaml

# Baseline 비교
CUDA_VISIBLE_DEVICES=4 torchrun ... --config configs/mouse/D7_1_E0_baseline.yaml

# 4v ablation
CUDA_VISIBLE_DEVICES=6 torchrun ... --config configs/mouse/D7_1_E2_gt_alpha_4v.yaml
```

---

## See Also

- [CONFIG_SCHEMA.md](../config/CONFIG_SCHEMA.md) - Config 구조 상세
- [configs/README.md](../../../configs/README.md) - Config 시스템 개요
