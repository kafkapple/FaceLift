# Mask Experiment Priority Analysis

> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Practical](../) | [Experiments](./)

**Date**: 2026-01-24
**Dataset**: D7_1

---

## Priority Matrix

| Priority | Config | Mode | α Loss | BG Loss | 근거 |
|----------|--------|------|--------|---------|------|
| **P0** ⭐ | E1_gt_alpha_sup | gt | 0.1 MSE | - | LGM + Pose Splatter |
| **P1** | E2_composite | composite | 0.05 MSE | - | Nerfstudio |
| **P2** | E3_bg_penalty | none | 0.1 MSE | 0.5 | Obj-Centric 2DGS |
| **P3** | E4_alpha_sup_only | none | 0.1 MSE | - | LGM only |
| P4 | E0_baseline | none | - | - | Baseline |

---

## Why P0 (GT + Alpha Supervision)?

### 이론적 근거
- **LGM**: "MSE alpha loss for faster convergence of the shape"
- **Pose Splatter**: Mouse/rat 데이터에서 검증된 normalized masked loss

### 실험적 근거
- Pose Splatter가 정확히 Mouse 데이터로 검증
- LGM이 feed-forward (GS-LRM과 유사) 아키텍처에서 검증

### 위험도
- 낮음: GT mask 사용 → spreading 문제 없음
- MSE alpha → gradient 안정적

---

## Experiment Configs

```
configs/mouse/mask_exp/
├── D7_mask_E0_baseline.yaml      # P4: No mask (baseline)
├── D7_mask_E1_gt_alpha_sup.yaml  # P0: GT + Alpha Sup (권장)
├── D7_mask_E2_composite.yaml     # P1: Composite
├── D7_mask_E3_bg_penalty.yaml    # P2: BG Penalty
└── D7_mask_E4_alpha_sup_only.yaml # P3: Alpha Sup Only
```

---

## Run Commands

```bash
cd /home/joon/dev/FaceLift

# P0: GT + Alpha Sup (최우선)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse/mask_exp/D7_mask_E1_gt_alpha_sup.yaml
```

---

## Expected Results

| Experiment | PSNR | mask_iou | 비고 |
|------------|------|----------|------|
| E0_baseline | 기준 | 기준 | 비교 대상 |
| **E1_gt_alpha_sup** | **+1~2dB** | **+0.1** | 최고 기대 |
| E2_composite | +0.5~1dB | +0.05 | 안정적 |
| E3_bg_penalty | 동등 | +0.05 | 경량화 효과 |
| E4_alpha_sup_only | 동등 | +0.05 | Shape만 개선 |

---

## Related Documents

- [Mask_Literature_Review](../../theory/mask/Mask_Literature_Review.md) - 문헌 조사
- [ALPHA_MASK_COMPLETE_GUIDE](../../theory/mask/ALPHA_MASK_COMPLETE_GUIDE.md) - 마스크 가이드
- [mask_losses.py](../../../mouse_extensions/model/mask_losses.py) - 구현 코드

---

*Mask Experiment Priority v1.0 | 2026-01-24*
