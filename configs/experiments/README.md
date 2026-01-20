# Experiment Configurations Matrix

> **Updated**: 2026-01-21
> **Total**: 19 experiments

---

## Experiment Matrix

### Paper Baseline (No Mask)

| Views | Fixed | Random |
|-------|-------|--------|
| 4v | E1_2_paper_fixed | E1_1_paper_random |
| 5v | E1_4_5v_paper_fixed | E1_3_5v_paper_random |

### GT Mask

| Views | Fixed | Random |
|-------|-------|--------|
| 4v | E2_2_gt_mask | E2_3_gt_mask_random |
| 5v | E2_4_5v_gt_mask_fixed | E2_5_5v_gt_mask_random |

### Alpha Mask

| Views | Fixed | Random |
|-------|-------|--------|
| 4v | E3_3_4v_alpha | E3_4_4v_alpha_random |
| 5v | E3_2_5v_alpha | E5_1_5v_alpha_random |

### Alpha Loss

| Views | Fixed | Random |
|-------|-------|--------|
| 4v | E4_3_4v_alpha_loss | - |
| 5v | E4_2_5v_alpha_loss | - |

### Conservative (threshold=0.7, alpha_loss=0.3)

| Views | Fixed | Random |
|-------|-------|--------|
| 4v | E5_4_4v_alpha_conservative | - |
| 5v | E5_2_5v_alpha_conservative | - |

### Aggressive (threshold=0.5, alpha_loss=0.5, opacity_reg=0.01)

| Views | Fixed | Random |
|-------|-------|--------|
| 4v | E5_5_4v_alpha_aggressive | - |
| 5v | E5_3_5v_alpha_aggressive | - |

---

## Quick Run Commands

```bash
cd /home/joon/dev/FaceLift

# Format: -d {dataset} -e {experiment}
# Datasets: D7_1, D7_1_t, D7_t, v13, D1

# Paper baseline
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1_t -e E1_1_paper_random > logs/run.log 2>&1 &

# GT Mask
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1_t -e E2_2_gt_mask > logs/run.log 2>&1 &

# Alpha Mask  
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1_t -e E3_2_5v_alpha > logs/run.log 2>&1 &
```

---

## Full Experiment List

| # | Experiment | Views | Selection | Mask | Description |
|---|------------|-------|-----------|------|-------------|
| 1 | E1_1_paper_random | 4v | random | none | Paper baseline |
| 2 | E1_2_paper_fixed | 4v | fixed | none | Paper baseline |
| 3 | E1_3_5v_paper_random | 5v | random | none | Paper baseline |
| 4 | E1_4_5v_paper_fixed | 5v | fixed | none | Paper baseline |
| 5 | E2_2_gt_mask | 4v | fixed | gt | GT supervision |
| 6 | E2_3_gt_mask_random | 4v | random | gt | GT supervision |
| 7 | E2_4_5v_gt_mask_fixed | 5v | fixed | gt | GT supervision |
| 8 | E2_5_5v_gt_mask_random | 5v | random | gt | GT supervision |
| 9 | E2_3_alpha_mask | 4v | fixed | alpha | Alpha mask basic |
| 10 | E3_2_5v_alpha | 5v | fixed | alpha | Alpha mask 5v |
| 11 | E3_3_4v_alpha | 4v | fixed | alpha | Alpha mask 4v |
| 12 | E3_4_4v_alpha_random | 4v | random | alpha | Alpha mask random |
| 13 | E4_2_5v_alpha_loss | 5v | fixed | alpha | With alpha loss |
| 14 | E4_3_4v_alpha_loss | 4v | fixed | alpha | With alpha loss |
| 15 | E5_1_5v_alpha_random | 5v | random | alpha | Alpha random |
| 16 | E5_2_5v_alpha_conservative | 5v | fixed | alpha | Conservative |
| 17 | E5_3_5v_alpha_aggressive | 5v | fixed | alpha | Aggressive |
| 18 | E5_4_4v_alpha_conservative | 4v | fixed | alpha | Conservative |
| 19 | E5_5_4v_alpha_aggressive | 4v | fixed | alpha | Aggressive |

---

*Generated: 2026-01-21*
