# Experiment Configurations (D7_1 Verified)

## Baseline Comparison

| Config | lr | grad_clip | random_view | Description |
|--------|-----|-----------|-------------|-------------|
| **E0_paper_original** | 1e-4 | 1.0 | true | Exact FaceLift paper settings |
| **E0_mouse_baseline** | 1e-5 | 5.0 | true | Mouse-adapted (finetuning) |

## Recommended Setup

**Dataset**: D7_1 (verified, geometrically correct)
**Experiment**: E2_gt_alpha (GT mask + alpha supervision)

## Quick Start

```bash
# Paper original baseline (lr=1e-4)
python train_gslrm.py -d D7_1 -e E0_paper_original

# Mouse-adapted baseline (lr=1e-5)
python train_gslrm.py -d D7_1 -e E0_mouse_baseline

# Recommended: GT mask + alpha supervision
python train_gslrm.py -d D7_1 -e E2_gt_alpha

# 3-view (stronger generalization)
python train_gslrm.py -d D7_1 -e E2_gt_alpha_3v
```

## Experiment Matrix

| Config | mask_mode | alpha_loss | Views | lr | Notes |
|--------|-----------|------------|-------|-----|-------|
| **E0_paper_original** | none | 0.0 | 4 | 1e-4 | Paper baseline |
| **E0_mouse_baseline** | none | 0.0 | 4 | 1e-5 | Mouse finetuning |
| **E2_gt_alpha** | gt | 0.1 | 4 | base | **Recommended** |
| E2_gt_alpha_3v | gt | 0.1 | 3 | base | Stronger generalization |
| E2_gt_alpha_5v | gt | 0.1 | 5 | base | More context |
| E2_gt_alpha_overfit | gt | 0.1 | 4 | base | Single sample test |
| E1_gt | gt | 0.0 | 4 | base | No alpha supervision |
| E3_alpha | alpha | 0.0 | 4 | base | **Not recommended** |
| E4_bg_penalty | none | 0.0 | 4 | base | Background penalty |
| E5_composite | composite | 0.0 | 4 | base | Nerfstudio style |
| E7_alpha_optimized | alpha | 0.1 | 4 | base | Threshold=0.6 |

## Key Settings Comparison

| Setting | Paper Original | Mouse Baseline | Current Base |
|---------|----------------|----------------|--------------|
| lr | **1e-4** | 1e-5 | 1e-6 |
| grad_clip_norm | **1.0** | 5.0 | 50.0 |
| num_input_views | 4 | 4 | 4 |
| random_view_selection | true | true | (unset) |
| maximize_view_overlap | true | false | false |

## Mask Mode Reference

| Mode | Description | Status |
|------|-------------|--------|
| **none** | No masking | Paper baseline |
| **gt** | Ground truth mask | Recommended |
| alpha | Rendered alpha mask | Needs alpha_loss |
| composite | Background compositing | Experimental |
| rgb_pred | RGB distance from white | **DEPRECATED** |

## Archived

- `_archive/E0_none.yaml`: Old baseline (incorrect lr)
- `_archive/deprecated/`: E6_rgb_pred (deprecated)
- `_archive/d9_native/`: D9 native resolution (unverified)

---

*Updated: 2026-01-24 | D7_1 Verified*
