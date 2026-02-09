# HP: Preprocessing Ablation Study

> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: ⏳ 실험 대기 | **Updated**: 2026-02-09

Created: 2026-02-09

## Motivation

HP (Preprocessing hypothesis) was verified **theoretically** (MVG analysis) but never
**empirically** with a training comparison. M0 (raw) data exists but was never trained.
This experiment provides the missing empirical evidence.

### Key Question

**Which preprocessing factors actually matter for GS-LRM reconstruction quality?**

Two independent factors to isolate:
1. **PP Centering** (principal point → 256): pretrained model compatibility
2. **Camera Normalization** (centroid→origin + uniform scale): geometric consistency

## Experimental Design

### 2×2+1 Factorial Design

```
              PP Centering
              ❌ (original)    ✅ (shift_to_256)
         ┌───────────────┬───────────────┐
  None   │  M0 (raw)     │  M5_4         │
         │  baseline      │  PP only      │
         ├───────────────┼───────────────┤
  Batch  │  (N/A)        │  M5 ⭐        │
  Uniform│               │  standard     │
         ├───────────────┼───────────────┤
  Per-   │  M0_n         │  M5_5         │
  view   │  norm only    │  PP+per-view  │
         └───────────────┴───────────────┘
```

### Conditions (4 experiments)

| Cond | Dataset | PP→256 | Recenter | Norm | Purpose |
|------|---------|--------|----------|------|---------|
| **1** | **M5** | ✅ | ✅ | Batch Uniform | Current standard (already trained) |
| **2** | **M0** | ❌ | ❌ | None | Raw baseline (no processing) |
| **3** | **M5_4** | ✅ | ❌ | None | Isolate PP centering effect |
| **4** | **M5_5** | ✅ | ❌ | Per-view | Test per-view norm (expected harmful) |

### Planned Comparisons

| Comparison | Isolates | Expected |
|------------|----------|----------|
| M0 vs M5_4 | PP centering alone | M5_4 >> M0 (PP compatibility) |
| M5_4 vs M5 | Batch Uniform norm | M5 > M5_4 (geometric consistency) |
| M5 vs M5_5 | Batch vs Per-view | M5 >> M5_5 (per-view breaks parallax) |
| M0 vs M5 | Total preprocessing | M5 >> M0 (combined effect) |

### Fixed Variables

| Variable | Value |
|----------|-------|
| Base config | `base_uniform_v2.yaml` |
| num_input_views | 4 (paper default) |
| max_fwdbwd_passes | 15000 |
| Seed | 42 |
| Split | 80:10:10 temporal (frames 0-2879 / 2880-3239 / 3240-3599) |
| GS-LRM pretrained | `ckpt_0000000000021125.pt` |
| Optimizer | AdamW, LR=1e-6, β=(0.9, 0.95) |

## Prerequisites

### 1. Generate Split Files

M0, M5_4, M5_5 need `data_mouse_t2_{train,val,test}.txt` matching M5 split:

```bash
for dataset in M0 M5_4 M5_5; do
    DATA_DIR=~/data/preprocessed/FaceLift_mouse/$dataset

    # Train: frames 0-2879
    seq -f "%06g" 0 2879 | while read f; do
        [ -d "$DATA_DIR/$f" ] && echo "$f"
    done > "$DATA_DIR/data_mouse_t2_train.txt"

    # Val: frames 2880-3239
    seq -f "%06g" 2880 3239 | while read f; do
        [ -d "$DATA_DIR/$f" ] && echo "$f"
    done > "$DATA_DIR/data_mouse_t2_val.txt"

    # Test: frames 3240-3599
    seq -f "%06g" 3240 3599 | while read f; do
        [ -d "$DATA_DIR/$f" ] && echo "$f"
    done > "$DATA_DIR/data_mouse_t2_test.txt"

    echo "$dataset: train=$(wc -l < $DATA_DIR/data_mouse_t2_train.txt) val=$(wc -l < $DATA_DIR/data_mouse_t2_val.txt) test=$(wc -l < $DATA_DIR/data_mouse_t2_test.txt)"
done
```

### 2. Training Configs

Create experiment configs pointing to each dataset:

- `configs/mouse/uniform/hp_M0.yaml`
- `configs/mouse/uniform/hp_M5_4.yaml`
- `configs/mouse/uniform/hp_M5_5.yaml`

(M5 = existing `4view_v2.yaml` or `baseline_v2.yaml`)

## Commands

### Condition 1: M5 (already trained)
Use existing: `base_uniform_v2_4view_v2` checkpoint

### Condition 2: M0 (raw baseline)
```bash
CUDA_VISIBLE_DEVICES=X python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/hp_M0.yaml \
  > logs/hp_M0.log 2>&1
```

### Condition 3: M5_4 (PP centering only)
```bash
CUDA_VISIBLE_DEVICES=X python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/hp_M5_4.yaml \
  > logs/hp_M5_4.log 2>&1
```

### Condition 4: M5_5 (PP + per-view norm)
```bash
CUDA_VISIBLE_DEVICES=X python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/hp_M5_5.yaml \
  > logs/hp_M5_5.log 2>&1
```

## Execution Plan

| Priority | When | GPU | Action |
|----------|------|-----|--------|
| 1 | After uniform_v2 3-view completes (~10h) | freed GPU | M0 training |
| 2 | After uniform_v2 5/6-view completes | freed GPU | M5_4 training |
| 3 | After above | freed GPU | M5_5 training |

Each training: ~15000 steps × 13s/step ≈ 54h on A6000

## Analysis Plan

### 1. Aggregate Comparison

| Condition | Dataset | PSNR | SSIM | IoU |
|-----------|---------|------|------|-----|
| 1 | M5 (standard) | (from 4view_v2) | | |
| 2 | M0 (raw) | | | |
| 3 | M5_4 (PP only) | | | |
| 4 | M5_5 (per-view) | | | |

### 2. Factor Contribution

| Factor | ∆PSNR | Significance |
|--------|-------|-------------|
| PP centering (M0→M5_4) | | |
| Batch Uniform (M5_4→M5) | | |
| Per-view harm (M5→M5_5) | | |
| Total (M0→M5) | | |

### 3. Per-view Breakdown

Check if preprocessing effects are uniform across views or view-dependent.

## Expected Outcomes

Based on theoretical analysis:
1. **PP centering is critical**: M5_4 >> M0 (pretrained expects cx=cy=256)
2. **Batch Uniform helps**: M5 > M5_4 (consistent geometry)
3. **Per-view hurts**: M5 >> M5_5 (parallax distortion)
4. **Combined >> Raw**: M5 >> M0

If PP centering alone recovers most quality (M5_4 ≈ M5), then Batch Uniform
is a refinement, not a necessity. If M5_4 << M5, then geometric normalization
is independently important.
