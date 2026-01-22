# Experiment Configurations

## Naming Convention

```
E{series}_{number}_{description}.yaml
```

- **E1**: Paper baseline (4v random, 4v fixed, 5v)
- **E2**: GT mask experiments
- **E3**: Alpha mask variants
- **E4**: Alpha tuning (conservative, aggressive, optimal)
- **E5**: Mask mode comparison (num_input_views=4)

## E5 Series (Current Focus)

| Config | mask_mode | Views | Key Feature |
|--------|-----------|-------|-------------|
| E5_1_5v_alpha_random | alpha | 5 | Baseline 5v random |
| E5_2_alpha_loss_only | alpha | 5 | Alpha loss only |
| E5_3_4v_alpha_loss | alpha | 4 | 4v baseline |
| **E5_4_rgb_mask_01** | **rgb_pred** | **4** | RGB mask (threshold=0.1) |
| E5_5_rgb_mask_02 | rgb_pred | 4 | RGB mask (threshold=0.2) |
| **E5_6_alpha_thresh_07** | **alpha** | **4** | Alpha (threshold=0.7) |
| **E5_7_optimal_alpha** | **alpha** | **4** | Best alpha settings |
| E5_8_alpha_thresh_08 | alpha | 4 | Alpha (threshold=0.8) |
| E5_9_gt_mask | gt | 4 | GT mask reference |
| E5_10_no_mask | none | 4 | No mask baseline |

## Key Parameters

- **num_input_views**: 4 (paper setting) -> train=4, eval=2
- **num_views**: 6 (total cameras)
- **random_view_selection**: false (fixed view selection)

## Run Commands

```bash
cd /home/joon/dev/FaceLift

# Single experiment
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    --base configs/gslrm/base.yaml \
    --config configs/datasets/D7_1_t.yaml configs/experiments/E5_4_rgb_mask_01.yaml

# With logging
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    --base configs/gslrm/base.yaml \
    --config configs/datasets/D7_1_t.yaml configs/experiments/E5_4_rgb_mask_01.yaml \
    > logs/E5_4_rgb_mask_01.log 2>&1 &
```

## Recommended Experiments

1. **E5_4_rgb_mask_01**: RGB mask - stable, no alpha dependency
2. **E5_6_alpha_thresh_07**: Stricter alpha threshold
3. **E5_7_optimal_alpha**: Best consolidated settings

---
Updated: 2026-01-22
