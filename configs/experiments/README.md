# Experiment Registry v2.2

> **Updated**: 2026-01-24
> **Naming**: E{Cat}_{Num}_{keywords}[_modifier]

---

## E0: Baseline (mask=none)

| ID | 파일명 | 설정 |
|----|--------|------|
| E0_1 | E0_1_paper.yaml | 논문 원본, random view |
| E0_2 | E0_2_finetune.yaml | lr=1e-5, finetuning용 |

---

## E1: GT Mask ⭐

| ID | 파일명 | mask | α | 용도 |
|----|--------|------|---|------|
| E1_1 | E1_1_gt.yaml | gt | 0.0 | GT only |
| **E1_2** ⭐ | **E1_2_gt_alpha.yaml** | gt | **0.1** | **권장** |
| E1_3 | E1_3_gt_alpha_lgm.yaml | gt | 1.0 | LGM 재현 |

### E1_2 Variants (View / Mode)
| ID | 파일명 | 설명 |
|----|--------|------|
| E1_2 | E1_2_gt_alpha.yaml | 4v 기본 |
| E1_2 | E1_2_gt_alpha_3v.yaml | 3 views |
| E1_2 | E1_2_gt_alpha_5v.yaml | 5 views |
| E1_2 | E1_2_gt_alpha_fixed.yaml | fixed view |
| E1_2 | E1_2_gt_alpha_overfit.yaml | 1 sample test |

---

## E2: Alpha Only (mask=none, α만)

| ID | 파일명 | mask | α | 상태 |
|----|--------|------|---|------|
| E2_1 | E2_1_alpha.yaml | none | 0.1 | ⚪ Experimental |
| E2_2 | E2_2_alpha_mask.yaml | alpha | 0.1 | ⚠️ 위험 |

---

## E3: Advanced

| ID | 파일명 | Mode | α | bg | 용도 |
|----|--------|------|---|-----|------|
| E3_1 | E3_1_composite.yaml | composite | 0.05 | - | Nerfstudio |
| E3_2 | E3_2_composite_strong.yaml | composite | 0.3 | - | Splatfacto-W |
| E3_3 | E3_3_bg.yaml | none | 0.1 | 0.5 | BG penalty |
| E3_4 | E3_4_bg_strong.yaml | none | 0.1 | 1.0 | BG strong |
| E3_5 | E3_5_combined.yaml | gt | 0.2 | 0.3 | Multi-lit |
| E3_6 | E3_6_clean.yaml | gt | 0.2 | 0.5 | +opacity_reg |

---

## File Rename Map

| 현재 | 신규 |
|------|------|
| E0_paper_original.yaml | E0_1_paper.yaml |
| E0_mouse_baseline.yaml | E0_2_finetune.yaml |
| E1_gt.yaml | E1_1_gt.yaml |
| E2_gt_alpha.yaml | **E1_2_gt_alpha.yaml** ⭐ |
| E2_gt_alpha_3v.yaml | E1_2_gt_alpha_3v.yaml |
| E2_gt_alpha_5v.yaml | E1_2_gt_alpha_5v.yaml |
| E2_gt_alpha_fixed.yaml | E1_2_gt_alpha_fixed.yaml |
| E2_gt_alpha_overfit.yaml | E1_2_gt_alpha_overfit.yaml |
| E3_alpha.yaml | E2_1_alpha.yaml |
| E4_bg_penalty.yaml | E3_3_bg.yaml |
| E5_composite.yaml | E3_1_composite.yaml |
| E5_composite_strong.yaml | E3_2_composite_strong.yaml |
| E6_lgm_full.yaml | E1_3_gt_alpha_lgm.yaml |
| E6_combined.yaml | E3_5_combined.yaml |
| E6_clean_bg.yaml | E3_6_clean.yaml |
| E6_bg_penalty_strong.yaml | E3_4_bg_strong.yaml |
| E7_alpha_optimized.yaml | E2_2_alpha_mask.yaml |
| E2_gt_alpha_4v.yaml | (삭제 - E1_2가 기본 4v) |

---

## Priority

| P | IDs | 설명 |
|---|-----|------|
| ⭐ P0 | **E1_2_gt_alpha** | Production |
| ✅ P1 | E0_1, E1_1, E1_3 | Baseline |
| ⚪ P2 | E1_2_*v, E2_1, E3_* | Experimental |
| ⚠️ P3 | E2_2_alpha_mask | 위험 |
| 🧪 P9 | E0_2, E1_2_overfit | Test |

---

*v2.2 | 2026-01-24*
