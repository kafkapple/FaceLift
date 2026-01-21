# Experiment Configurations Matrix

> **Updated**: 2026-01-21
> **Naming Convention**: random=1 (paper default), fixed=2

---

## Naming Convention

```
E{Series}_{Number}_{description}

Series:
  E1: Paper Baseline (no mask)
  E2: Mask Mode (GT vs Alpha)
  E3: View Count (with alpha mask)
  E4: Alpha Tuning (threshold, loss, reg)
  E5: Loss Ablation

Number within series:
  - Lower number = random view (paper default)
  - Higher number = fixed view or variations
```

---

## E1: Paper Baseline (No Mask)

| # | Config | Views | Selection | Mask | Description |
|---|--------|-------|-----------|------|-------------|
| 1 | E1_1_paper_random | 4v | random | none | Paper baseline (default) |
| 2 | E1_2_paper_fixed | 4v | fixed | none | Fixed view control |
| 3 | E1_3_5v_paper_random | 5v | random | none | 5-view baseline |
| 4 | E1_4_5v_paper_fixed | 5v | fixed | none | 5-view fixed |

---

## E2: Mask Mode

| # | Config | Views | Selection | Mask | Description |
|---|--------|-------|-----------|------|-------------|
| 1 | E2_1_gt_mask_random | 4v | random | GT | GT mask baseline |
| 2 | E2_2_gt_mask | 4v | fixed | GT | GT mask fixed |
| 3 | E2_3_alpha_random | 4v | random | alpha | Alpha mask baseline |
| 4 | E2_4_alpha_fixed | 4v | fixed | alpha | Alpha mask fixed |
| 5 | E2_5_5v_gt_random | 5v | random | GT | 5-view GT random |
| 6 | E2_6_5v_gt_fixed | 5v | fixed | GT | 5-view GT fixed |

---

## E3: View Count (Alpha Mask)

| # | Config | Views | Selection | Mask | Description |
|---|--------|-------|-----------|------|-------------|
| 1 | E3_1_4v_alpha_random | 4v | random | alpha | 4-view alpha random |
| 2 | E3_2_5v_alpha | 5v | fixed | alpha | 5-view alpha (B2 baseline) |
| 3 | E3_3_4v_alpha | 4v | fixed | alpha | 4-view alpha fixed |
| 4 | E3_4_4v_alpha_random | 4v | random | alpha | Alias for E3_1 |

---

## E4: Alpha Tuning

| # | Config | Views | Threshold | Loss | Reg | Description |
|---|--------|-------|-----------|------|-----|-------------|
| 1 | E4_1_alpha_basic | 5v | 0.5 | 0.0 | - | Basic alpha (no tuning) |
| 2 | E4_2_alpha_conservative | 5v | 0.7 | 0.3 | - | Conservative (anti-bleeding) |
| 3 | E4_3_alpha_aggressive | 5v | 0.5 | 0.5 | 0.01 | Aggressive (shape priority) |
| 4 | E4_4_4v_conservative | 4v | 0.7 | 0.3 | - | 4-view conservative |
| 5 | E4_5_4v_aggressive | 4v | 0.5 | 0.5 | 0.01 | 4-view aggressive |

---

## E5: Loss Ablation

| # | Config | Views | Alpha Loss | Opacity Reg | Description |
|---|--------|-------|------------|-------------|-------------|
| 1 | E5_1_5v_alpha_random | 5v | 0.0 | - | Alpha random baseline |
| 2 | E5_2_alpha_loss_only | 5v | 0.1 | - | Alpha loss only |
| 3 | E5_3_opacity_reg_only | 5v | 0.0 | 0.01 | Opacity reg only |

---

## Baseline Definitions

| ID | Config | Dataset | Role |
|----|--------|---------|------|
| **B1** | E1_1_paper_random | D7_1 | All experiments baseline |
| **B2** | E3_2_5v_alpha | D7_1 | Mask experiments baseline |
| **B3** | E2_1_gt_mask_random | D7_1 | GT vs Alpha comparison |

---

## Quick Run Commands

```bash
cd /home/joon/dev/FaceLift

# Format: -d {dataset} -e {experiment}
# Datasets: D7_1, D7_1_t, D8, D8_1, v13, D1

# Paper baseline (B1)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_1_paper_random > logs/d7_1_e1_1.log 2>&1 &

# GT Mask baseline (B3)
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_1_gt_mask_random > logs/d7_1_e2_1.log 2>&1 &

# Alpha Mask baseline (B2)
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E3_2_5v_alpha > logs/d7_1_e3_2.log 2>&1 &

# D8 precision dataset
CUDA_VISIBLE_DEVICES=3 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E1_1_paper_random > logs/d8_e1_1.log 2>&1 &
```

---

*FaceLift Mouse Project | Updated: 2026-01-21*
