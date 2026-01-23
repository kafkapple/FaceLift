# Experiment Configurations

> Last Updated: 2026-01-24

## Quick Reference

| Experiment | mask_mode | alpha_loss | views | Priority | 설명 |
|------------|-----------|------------|-------|----------|------|
| **E2_gt_alpha** ⭐ | gt | 0.1 | 4 | **P0** | 권장 설정 |
| E0_none | none | 0.0 | 4 | P2 | Baseline |
| E1_gt | gt | 0.0 | 4 | P3 | GT mask only |
| E3_alpha | none | 0.1 | 4 | P4 | Alpha only |
| E4_bg_penalty | none | 0.1 + bg | 4 | P3 | Background penalty |

## View Ablation

| Experiment | Input Views | Holdout | 용도 |
|------------|-------------|---------|------|
| E2_gt_alpha_3v | 3 | 3 | 강건성 테스트 |
| E2_gt_alpha | 4 | 2 | 기본 (권장) |
| E2_gt_alpha_5v | 5 | 1 | 최대 정보 |
| E2_gt_alpha_overfit | 1 | 5 | 오버핏 테스트 |

## Special Variants

| Experiment | 특징 | 용도 |
|------------|------|------|
| E2_gt_alpha_fixed | random_view=false | 고정 뷰 순서 |
| E2_gt_alpha_native | batch=1, size=1024 | D9/D9_norm 전용 |

## Usage

```bash
# Modular mode (권장)
train_gslrm.py -d <DATASET> -e <EXPERIMENT>

# Example
train_gslrm.py -d D8 -e E2_gt_alpha
train_gslrm.py -d D8 -e E2_gt_alpha_3v
```
