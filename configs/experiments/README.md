# Experiment Registry

> **Last Updated**: 2026-01-24
> **Base**: Paper Original Settings (lr=1e-4, grad_clip=1.0)

---

## Numbering Convention

```
E{Category}_{Variant}_{Modifier}

Category (0-9):
  0 = Baseline (논문 재현)
  1 = GT Mask Only
  2 = GT + Alpha ⭐ (권장)
  3 = Alpha Only
  4 = BG Penalty
  5 = Composite
  6 = Combined/Advanced
  7 = Alpha Mask (⚠️ 위험)
  9 = Test/Debug

Variant (1-9): 세부 변형
Modifier: 추가 특성 (3v, fixed, overfit 등)
```

---

## Priority Classification

### ⭐ P0: 권장 (Production Ready)
| ID | Name | 설정 | 문헌 |
|----|------|------|------|
| **E2_1** | gt_alpha | mask=gt, α=0.1 | LGM + Pose Splatter |
| E0_1 | paper_original | mask=none | GS-LRM Paper |

### ✅ P1: 검증됨 (Validated)
| ID | Name | 설정 | 용도 |
|----|------|------|------|
| E1_1 | gt_only | mask=gt, α=0.0 | Ablation |
| E2_2 | gt_alpha_3v | 3 input views | View ablation |
| E2_3 | gt_alpha_4v | 4 input views | View ablation |
| E2_4 | gt_alpha_5v | 5 input views | View ablation |
| E6_1 | lgm_full | α=1.0 | LGM 원본 재현 |

### ⚪ P2: 실험적 (Experimental)
| ID | Name | 설정 | 목적 |
|----|------|------|------|
| E2_5 | gt_alpha_fixed | random=false | Fixed view order |
| E3_1 | alpha_only | mask=none, α=0.1 | Alpha supervision 효과 |
| E4_1 | bg_penalty | bg=0.5 | Object-Centric 2DGS |
| E5_1 | composite | composite mode | Nerfstudio style |
| E5_2 | composite_strong | α=0.3 | Splatfacto-W |
| E6_2 | combined | α=0.2, bg=0.3 | Multi-lit 복합 |
| E6_3 | clean_bg | +opacity_reg | Gaussian 정리 |
| E6_4 | bg_penalty_strong | bg=1.0 | 강화 bg penalty |

### ⚠️ P3: 위험 (Use with Caution)
| ID | Name | 설정 | 위험 |
|----|------|------|------|
| E7_1 | alpha_mask | mask=alpha | 확장 위험, 안전장치 필요 |

### 🧪 P9: 테스트/디버그
| ID | Name | 설정 | 목적 |
|----|------|------|------|
| E0_2 | mouse_baseline | lr=1e-5 | Finetuning 테스트 |
| E2_9 | overfit | 1 sample | Sanity check |

---

## File Mapping (현재 → 신규)

| 현재 파일 | 신규 ID | 상태 |
|-----------|---------|------|
| E0_paper_original.yaml | E0_1 | ✅ Active |
| E0_mouse_baseline.yaml | E0_2 | ✅ Active |
| E1_gt.yaml | E1_1 | ✅ Active |
| E2_gt_alpha.yaml | **E2_1** ⭐ | ✅ Active |
| E2_gt_alpha_3v.yaml | E2_2 | ✅ Active |
| E2_gt_alpha_4v.yaml | E2_3 | ✅ Active |
| E2_gt_alpha_5v.yaml | E2_4 | ✅ Active |
| E2_gt_alpha_fixed.yaml | E2_5 | ✅ Active |
| E2_gt_alpha_overfit.yaml | E2_9 | 🧪 Test |
| E3_alpha.yaml | E3_1 | ⚪ Experimental |
| E4_bg_penalty.yaml | E4_1 | ⚪ Experimental |
| E5_composite.yaml | E5_1 | ⚪ Experimental |
| E5_composite_strong.yaml | E5_2 | ⚪ Experimental |
| E6_lgm_full.yaml | E6_1 | ✅ Validated |
| E6_combined.yaml | E6_2 | ⚪ Experimental |
| E6_clean_bg.yaml | E6_3 | ⚪ Experimental |
| E6_bg_penalty_strong.yaml | E6_4 | ⚪ Experimental |
| E7_alpha_optimized.yaml | E7_1 | ⚠️ Risky |

---

## Quick Reference

```bash
# 권장 실험 (M3 데이터셋)
torchrun ... train_gslrm.py -d M3 -e E2_gt_alpha      # E2_1

# View Ablation
torchrun ... train_gslrm.py -d M3 -e E2_gt_alpha_3v   # E2_2
torchrun ... train_gslrm.py -d M3 -e E2_gt_alpha_5v   # E2_4

# Overfit Test
torchrun ... train_gslrm.py -d D7_1_overfit -e E2_gt_alpha_overfit  # E2_9
```

---

## Base Settings (Paper Original)

| 설정 | 값 | 출처 |
|------|-----|------|
| lr | **1e-4** | Paper |
| grad_clip_norm | **1.0** | Paper |
| random_view_selection | **true** | Paper |
| maximize_view_overlap | **true** | Paper |
| augmentation | **false** | Default |

---

*FaceLift Mouse Experiments | v2.0 | 2026-01-24*
