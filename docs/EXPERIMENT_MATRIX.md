# FaceLift Mouse Experiment Matrix

> Last Updated: 2026-01-21

## Quick Reference

### 데이터셋

| Dataset | Description | Status | Ray Error | Train/Val/Test |
|---------|-------------|--------|-----------|----------------|
| **D7_1** | Individual scale, geometric correct | ★ Recommended | ~0° | 2916/313/- |
| **D7_1_t** | D7_1 + temporal split | ★ Recommended | ~0° | 1095/1061/1077 |
| v13 | Legacy PP=256 forced | DEPRECATED | 5-13° | 4334/124/- |
| D1 | PP-centered crop | DEPRECATED | ~13° | 3604/380/- |

### 실험 그룹

| Group | Hypothesis | mask_mode | Description |
|-------|------------|-----------|-------------|
| **E1** | H1: Baseline | none | No mask supervision |
| **E2** | H2: GT Mask | gt | Ground truth mask supervision |
| **E3** | H3: Alpha Mask | alpha | Rendered alpha mask |
| **E4** | H4: Alpha Loss | alpha | + stronger alpha loss |
| **E5** | H5: Conservative | alpha | Higher threshold, stronger loss |

---

## Experiment Configurations

### E1: Baseline (No Mask)

| Experiment | Views | Random | mask_mode | alpha_loss |
|------------|-------|--------|-----------|------------|
| E1_1_paper_random | 4 | ✓ | none | 0.0 |
| E1_2_paper_fixed | 4 | ✗ | none | 0.0 |
| E1_3_5v_paper_random | 5 | ✓ | none | 0.0 |
| E1_4_5v_paper_fixed | 5 | ✗ | none | 0.0 |

### E2: GT Mask

| Experiment | Views | Random | mask_mode | alpha_loss |
|------------|-------|--------|-----------|------------|
| E2_2_gt_mask | 4 | ✗ | gt | 0.0 |
| E2_3_gt_mask_random | 4 | ✓ | gt | 0.0 |
| E2_4_5v_gt_mask_fixed | 5 | ✗ | gt | 0.0 |
| E2_5_5v_gt_mask_random | 5 | ✓ | gt | 0.0 |

### E3: Alpha Mask

| Experiment | Views | Random | mask_mode | alpha_threshold | alpha_loss |
|------------|-------|--------|-----------|-----------------|------------|
| E3_2_5v_alpha | 5 | ✗ | alpha | 0.5 | 0.1 |
| E3_3_4v_alpha | 4 | ✗ | alpha | 0.5 | 0.1 |
| E3_4_4v_alpha_random | 4 | ✓ | alpha | 0.5 | 0.1 |

### E4: Alpha Loss (Stronger)

| Experiment | Views | Random | alpha_threshold | alpha_loss | loss_type |
|------------|-------|--------|-----------------|------------|-----------|
| E4_2_5v_alpha_loss | 5 | ✗ | 0.5 | 0.3 | bce |
| E4_3_4v_alpha_loss | 4 | ✗ | 0.5 | 0.3 | bce |

### E5: Conservative Alpha

| Experiment | Views | Random | alpha_threshold | alpha_loss | Description |
|------------|-------|--------|-----------------|------------|-------------|
| E5_2_5v_alpha_conservative | 5 | ✗ | **0.7** | 0.3 | Anti-bleeding |
| E5_4_4v_alpha_conservative | 4 | ✗ | **0.7** | 0.3 | Anti-bleeding |

---

## Running Experiments

### Modular Config System

```bash
# Format: -d {dataset} -e {experiment}
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d D7_1 -e E1_1_paper_random

# Config files loaded:
# - configs/base/gslrm_mouse.yaml      (base)
# - configs/datasets/D7_1.yaml          (dataset)
# - configs/experiments/E1_1_paper_random.yaml  (experiment)
```

### GPU Assignment Example

```bash
# Baseline comparison
CUDA_VISIBLE_DEVICES=0 nohup torchrun ... -d D7_1 -e E1_1_paper_random > logs/d7_1_e1_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup torchrun ... -d D7_1 -e E1_2_paper_fixed > logs/d7_1_e1_2.log 2>&1 &

# Mask comparison
CUDA_VISIBLE_DEVICES=2 nohup torchrun ... -d D7_1 -e E2_2_gt_mask > logs/d7_1_e2_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup torchrun ... -d D7_1 -e E3_3_4v_alpha > logs/d7_1_e3_3.log 2>&1 &
```

---

## Monitoring

### WandB Dashboard
- **Config**: Run → Config tab shows all settings including mask_mode
- **Group**: Experiments grouped by dataset (D7_1, D7_1_t, etc.)

### Checkpoint Files
```bash
# View saved config
cat checkpoints/gslrm/D7_1_E1_1_paper_random/config.yaml | grep -A10 losses
```

### Debug Mode (optional)
```bash
# Enable verbose mask logging (logs once at start)
DEBUG_MASK=1 torchrun ... -d D7_1 -e E3_3_4v_alpha
```

---

## Comparison Groups

### Random vs Fixed View Selection
- E1_1 vs E1_2 (4v, no mask)
- E2_2 vs E2_3 (4v, gt mask)
- E3_3 vs E3_4 (4v, alpha)

### 4-View vs 5-View
- E1_1 vs E1_3 (no mask)
- E3_3 vs E3_2 (alpha)
- E5_4 vs E5_2 (conservative)

### Mask Mode Comparison
- E1_1 (none) vs E2_3 (gt) vs E3_4 (alpha)

### Alpha Threshold
- E3_3 (0.5) vs E5_4 (0.7)

---

*FaceLift Mouse | Experiment Documentation*
