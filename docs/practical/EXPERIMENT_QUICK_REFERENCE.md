# Experiment Quick Reference

> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [Practical](./)
> **Updated**: 2026-01-24 | **Version**: v2.2

---

## Naming Convention

```
E{Category}_{Number}_{keywords}[_modifier]

Categories:
  E0 = Baseline (mask=none)
  E1 = GT Mask ⭐ (권장)
  E2 = Alpha Only
  E3 = Advanced

Modifiers:
  _3v, _5v    = View count (4v default)
  _fixed      = Fixed view order
  _overfit    = 1 sample test
```

---

## Quick Start Commands

### ⭐ 권장 실험 (Production)

```bash
# M3 + GT mask + Alpha loss
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3 -e E1_2_gt_alpha
```

### Baseline

```bash
# 논문 원본 재현
torchrun ... train_gslrm.py -d M3 -e E0_1_facelift
```

### View Ablation

```bash
# 3 views
torchrun ... train_gslrm.py -d M3 -e E1_2_gt_alpha_3v

# 5 views
torchrun ... train_gslrm.py -d M3 -e E1_2_gt_alpha_5v
```

### Overfit Test

```bash
# 1 sample sanity check
torchrun ... train_gslrm.py -d D7_1_overfit -e E1_2_gt_alpha_overfit
```

---

## Experiment Matrix

### E0: Baseline (mask=none)

| ID | 파일 | lr | view | 용도 |
|----|------|-----|------|------|
| E0_1 | E0_1_facelift.yaml | 1e-4 | random | 논문 재현 |
| E0_2 | E0_2_mouse.yaml | 1e-5 | random | Finetuning |

### E1: GT Mask ⭐

| ID | 파일 | mask | α loss | 용도 |
|----|------|------|--------|------|
| E1_1 | E1_1_gt.yaml | gt | 0.0 | Ablation |
| **E1_2** ⭐ | **E1_2_gt_alpha.yaml** | gt | **0.1** | **Production** |
| E1_3 | E1_3_gt_alpha_lgm.yaml | gt | 1.0 | LGM 재현 |

**E1_2 Variants:**
| Modifier | 파일 | 설명 |
|----------|------|------|
| (기본) | E1_2_gt_alpha.yaml | 4v, random |
| _3v | E1_2_gt_alpha_3v.yaml | 3 input views |
| _5v | E1_2_gt_alpha_5v.yaml | 5 input views |
| _fixed | E1_2_gt_alpha_fixed.yaml | Fixed view order |
| _overfit | E1_2_gt_alpha_overfit.yaml | 1 sample test |

### E2: Alpha Only

| ID | 파일 | mask | α loss | 상태 |
|----|------|------|--------|------|
| E2_1 | E2_1_alpha.yaml | none | 0.1 | ⚪ Experimental |
| E2_2 | E2_2_alpha_mask.yaml | alpha | 0.1 | ⚠️ **위험** |

> ⚠️ **E2_2 위험 근거**: mask_mode=alpha는 rendered_alpha를 마스크로 사용.
> 피드백 루프로 확장 가능. 문헌에서 사용 사례 없음.

### E3: Advanced

| ID | 파일 | Mode | α | bg | 용도 |
|----|------|------|---|-----|------|
| E3_1 | E3_1_composite.yaml | composite | 0.05 | - | Nerfstudio |
| E3_2 | E3_2_composite_strong.yaml | composite | 0.3 | - | Splatfacto-W |
| E3_3 | E3_3_bg.yaml | none | 0.1 | 0.5 | BG penalty |
| E3_4 | E3_4_bg_strong.yaml | none | 0.1 | 1.0 | BG strong |
| E3_5 | E3_5_combined.yaml | gt | 0.2 | 0.3 | Multi-lit |
| E3_6 | E3_6_clean.yaml | gt | 0.2 | 0.5 | +opacity_reg |

---

## Priority

| P | 실험 | 용도 | 명령어 |
|---|------|------|--------|
| ⭐ P0 | **E1_2_gt_alpha** | Production | `-e E1_2_gt_alpha` |
| ✅ P1 | E0_1, E1_1, E1_3 | Baseline/Ablation | |
| ⚪ P2 | E1_2_*v, E2_1, E3_* | Experimental | |
| ⚠️ P3 | E2_2_alpha_mask | 위험 | 안전장치 필수 |
| 🧪 P9 | E0_2, E1_2_overfit | Test/Debug | |

---

## Literature Mapping

| 문헌 | 해당 실험 | 핵심 설정 |
|------|----------|----------|
| **GS-LRM Paper** | E0_1_facelift | mask=none |
| **LGM (ECCV 2024)** | E1_3_gt_alpha_lgm | α=1.0, MSE |
| **Pose Splatter** | E1_2_gt_alpha | mask=gt, normalize |
| **Object-Centric 2DGS** | E3_3_bg | bg_loss=0.5 |
| **Splatfacto-W** | E3_2_composite_strong | composite, α=0.3 |

---

## Base Settings (Paper Original)

모든 실험은 다음 base 설정을 상속:

| 설정 | 값 | 출처 |
|------|-----|------|
| lr | 1e-4 | Paper |
| grad_clip_norm | 1.0 | Paper |
| random_view_selection | true | Paper |
| maximize_view_overlap | true | Paper |
| augmentation | false | Default |
| num_input_views | 4 | Paper |

---

*v2.2 | 2026-01-24*
