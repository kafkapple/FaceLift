# Experiment Configurations (D7_1 Verified)

## Recommended Setup

**Dataset**: D7_1 (verified, geometrically correct)
**Experiment**: E2_gt_alpha (GT mask + alpha supervision)

## Quick Start

```bash
# Default 4-view experiment
python train_gslrm.py -d D7_1 -e E2_gt_alpha

# 3-view (stronger generalization)
python train_gslrm.py -d D7_1 -e E2_gt_alpha_3v

# 5-view (more context)
python train_gslrm.py -d D7_1 -e E2_gt_alpha_5v
```

## Experiment Matrix

| Config | mask_mode | alpha_loss | Views | Notes |
|--------|-----------|------------|-------|-------|
| **E2_gt_alpha** | gt | 0.1 | 4 | **Recommended** |
| E2_gt_alpha_3v | gt | 0.1 | 3 | Stronger generalization |
| E2_gt_alpha_5v | gt | 0.1 | 5 | More context |
| E2_gt_alpha_overfit | gt | 0.1 | 4 | Single sample test |
| E1_gt | gt | 0.0 | 4 | No alpha supervision |
| E0_none | none | 0.0 | 4 | No masking (baseline) |
| E3_alpha | alpha | 0.0 | 4 | **Not recommended** |
| E4_bg_penalty | none | 0.0 | 4 | Background penalty |
| E5_composite | composite | 0.0 | 4 | Nerfstudio style |
| E7_alpha_optimized | alpha | 0.1 | 4 | Threshold=0.6 |

## Mask Mode Reference

| Mode | Description | Status |
|------|-------------|--------|
| **gt** | Ground truth mask | Recommended |
| **none** | No masking | Baseline |
| alpha | Rendered alpha mask | Needs alpha_loss |
| composite | Background compositing | Experimental |
| rgb_pred | RGB distance from white | **DEPRECATED** |

## Archived

- `_archive/deprecated/`: E6_rgb_pred (deprecated)
- `_archive/d9_native/`: D9 native resolution (unverified)

---

*Updated: 2026-01-24 | D7_1 Verified*
