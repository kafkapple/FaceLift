# Experiment Registry v2.0

> **Updated**: 2026-01-24
> **Base**: Paper Original (lr=1e-4, grad_clip=1.0, random_view=true)

---

## Numbering System

```
E{Category}[_{Variant}][_{Modifier}]

Categories:
  E0 = Baseline (mask=none, 논문 원본)
  E1 = GT Mask 계열 ⭐
  E2 = Alpha Only (mask=none, α만)
  E3 = Advanced (composite, bg penalty)

Modifiers (optional):
  _3v, _5v     = View 수 (4v 기본, 생략)
  _fixed       = Fixed view order (vs random)
  _strong      = 강화 버전
  _overfit     = 1 sample overfit test
  _ft          = Finetuning (lr 조정)
```

---

## E0: Baseline (mask=none)

논문 원본 설정. 마스크 없이 전체 이미지로 학습.

| ID | Name | 설정 | 용도 |
|----|------|------|------|
| **E0** | paper_original | mask=none, random | 논문 재현 기준선 |
| E0_fixed | paper_fixed | random=false | Ablation |
| E0_ft | mouse_baseline | lr=1e-5 | Finetuning용 |

---

## E1: GT Mask 계열 ⭐ (권장)

GT 마스크로 loss 영역 제한. Alpha loss로 shape 수렴.

| ID | Name | mask | α loss | 용도 |
|----|------|------|--------|------|
| E1 | gt_only | gt | 0.0 | Ablation (α 효과 비교) |
| **E1_alpha** ⭐ | gt_alpha | gt | **0.1** | **Production 권장** |
| E1_alpha_lgm | lgm_full | gt | **1.0** | LGM 논문 재현 |

### E1 View Ablation
| ID | Views | 설정 |
|----|-------|------|
| E1_alpha_3v | 3 in / 3 out | Generalization 테스트 |
| E1_alpha | 4 in / 2 out | 기본 (생략 가능) |
| E1_alpha_5v | 5 in / 1 out | Maximum information |

### E1 Variants
| ID | 설정 | 용도 |
|----|------|------|
| E1_alpha_fixed | random=false | Reproducibility |
| E1_alpha_overfit | 1 sample, no aug | Sanity check |

---

## E2: Alpha Only (mask=none)

마스크 없이 alpha supervision만. 전체 이미지 학습.

| ID | Name | mask | α loss | 상태 |
|----|------|------|--------|------|
| E2 | alpha_only | none | 0.1 | ⚪ Experimental |
| E2_alpha_mask | alpha_optimized | **alpha** | 0.1 | ⚠️ **위험** |

> ⚠️ **E2_alpha_mask**: mask_mode=alpha 사용. 확장 위험. 안전장치(alpha_mask_safety) 필수.

---

## E3: Advanced

Composite, background penalty, 복합 설정.

### E3_composite: Composite Mode
| ID | Name | α loss | 용도 |
|----|------|--------|------|
| E3_composite | composite | 0.05 | Nerfstudio style |
| E3_composite_strong | composite_strong | **0.3** | Splatfacto-W |

### E3_bg: Background Penalty
| ID | Name | bg loss | 용도 |
|----|------|---------|------|
| E3_bg | bg_penalty | 0.5 | Object-Centric 2DGS |
| E3_bg_strong | bg_penalty_strong | **1.0** | 강화 |

### E3 Combined
| ID | Name | α | bg | 특징 |
|----|------|---|-----|------|
| E3_combined | combined | 0.2 | 0.3 | Multi-literature |
| E3_clean | clean_bg | 0.2 | 0.5 | +opacity_reg |

---

## Priority Summary

| Priority | 실험 | 용도 |
|----------|------|------|
| ⭐ **P0** | **E1_alpha** | Production (권장) |
| ✅ P1 | E0, E1, E1_alpha_lgm | Baseline, Ablation |
| ⚪ P2 | E1_alpha_3v/5v, E2, E3_* | Experimental |
| ⚠️ P3 | E2_alpha_mask | 위험 (안전장치 필수) |
| 🧪 P9 | E0_ft, E1_alpha_overfit | Test/Debug |

---

## File Mapping

| 현재 파일명 | 신규 ID | Priority |
|-------------|---------|----------|
| E0_paper_original | E0 | ✅ |
| E0_mouse_baseline | E0_ft | 🧪 |
| E1_gt | E1 | ✅ |
| **E2_gt_alpha** | **E1_alpha** ⭐ | ⭐ |
| E2_gt_alpha_3v | E1_alpha_3v | ⚪ |
| E2_gt_alpha_4v | E1_alpha | ⚪ |
| E2_gt_alpha_5v | E1_alpha_5v | ⚪ |
| E2_gt_alpha_fixed | E1_alpha_fixed | ⚪ |
| E2_gt_alpha_overfit | E1_alpha_overfit | 🧪 |
| E3_alpha | E2 | ⚪ |
| E4_bg_penalty | E3_bg | ⚪ |
| E5_composite | E3_composite | ⚪ |
| E5_composite_strong | E3_composite_strong | ⚪ |
| E6_lgm_full | E1_alpha_lgm | ✅ |
| E6_combined | E3_combined | ⚪ |
| E6_clean_bg | E3_clean | ⚪ |
| E6_bg_penalty_strong | E3_bg_strong | ⚪ |
| E7_alpha_optimized | E2_alpha_mask | ⚠️ |

---

## Quick Start

```bash
# 권장 실험 (M3 데이터셋)
torchrun ... train_gslrm.py -d M3 -e E2_gt_alpha        # = E1_alpha ⭐

# View Ablation
torchrun ... train_gslrm.py -d M3 -e E2_gt_alpha_3v     # = E1_alpha_3v
torchrun ... train_gslrm.py -d M3 -e E2_gt_alpha_5v     # = E1_alpha_5v

# Baseline
torchrun ... train_gslrm.py -d M3 -e E0_paper_original  # = E0

# Overfit Test
torchrun ... train_gslrm.py -d D7_1_overfit -e E2_gt_alpha_overfit
```

---

*FaceLift Experiment Registry v2.0 | 2026-01-24*
