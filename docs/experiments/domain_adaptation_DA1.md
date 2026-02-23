# Domain Adaptation DA1: GS-LRM Fine-tuning on MVDiff-Generated Views

> SSOT for the DA1 experiment: rationale, pipeline, configs, and expected outcomes.
> Created: 2026-02-22 | Updated: 2026-02-22 | Version: v1.1

---

## 1. Problem Statement

### The Distribution Mismatch Gap

FaceLift E2E pipeline has a critical bottleneck:

```
GS-LRM with GT 6-view input:  PSNR_fg = 23.84, IoU = 0.954
GS-LRM with MVDiff 6v input:  PSNR_fg =  8.44, IoU = 0.495
                               ─────────────────────────────
                               Gap:       15.4 dB  (65% loss)
```

**Why?** GS-LRM was trained exclusively on GT (ground truth) multi-view images.
At E2E inference time, it receives MVDiff-generated views which have:

| Property | GT Views | MVDiff Views |
|----------|----------|-------------|
| Geometric consistency | Perfect | Approximate (view drift) |
| Fine detail | Clean, sharp | Diffusion artifacts |
| Noise characteristics | Sensor noise only | Diffusion noise residual |
| Coverage | Complete | ~75% (incomplete views) |
| Color distribution | Original | Slightly shifted |

This is a classic **domain shift** problem: the model was trained on distribution A (GT)
but deployed on distribution B (MVDiff-generated).

### Why Not Just Improve MVDiff?

We tried. MVDiff val PSNR improved from 24.0 to 27.70 dB (+3.7 dB), but E2E only
improved from 7.93 to 8.44 dB (+0.51 dB). **Transfer rate: ~14%**.

This means further MVDiff improvement has severely diminishing returns for E2E quality.
The bottleneck is not MVDiff quality alone, but GS-LRM's inability to handle
MVDiff-style inputs.

See: `docs/experiments/mvdiffusion_bottleneck_analysis.md` for full analysis.

---

## 2. Why This MVDiff Checkpoint? (E2 Selection Rationale)

11 MVDiffusion configs and 9 checkpoints were evaluated. E2 was selected as the
data source for DA1 because it is the best-performing MVDiff model.

### MVDiff Checkpoint Comparison

| Checkpoint | Key Changes | Val PSNR | E2E PSNR_fg | E2E IoU |
|-----------|------------|:--------:|:-----------:|:-------:|
| Baseline (ckpt-5000) | Original config | ~24.0 | 7.93 | 0.474 |
| cfgr (full attn) | Removed sparse attention | ? | 7.75 | 0.491 |
| E1 (cosine 20K) | Cosine LR schedule, 20K steps | 26.93 | 8.04 | 0.528 |
| **E2 (randref sparse 20K)** | **Random ref view + LR decay resume** | **27.70** | **8.44** | **0.495** |
| E3 (pose extrinsic) | Extrinsic pose conditioning + cosine | ? | 8.10 | 0.523 |

**E2 was chosen because**:
1. Highest MVDiff image quality (val PSNR 27.70, +3.7 dB over baseline)
2. Best E2E performance (PSNR_fg 8.44)
3. Random reference view = more diverse training signal for generation
4. Sparse attention preserved = better multi-view consistency than full attention

**Path**: `/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000`

### Inference Settings (same as E2E evaluation)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| DDPM steps | 50 | Standard for this pipeline |
| Guidance scale | 3.0 | Default; sweep untested (future P2) |
| Seed | 42 | Deterministic for reproducibility |
| Input view | GT cam_000 (front) | Same as E2E inference |
| Output views | 6 (cam_000–005) | Full 6-view set |
| Resolution | 512×512 | Native MVDiff resolution |

These settings ensure the DA1 training data matches the exact distribution that
GS-LRM will encounter during E2E inference.

---

## 3. DA1 Strategy: Fine-tune GS-LRM on MVDiff Outputs

### Core Idea

Instead of making MVDiff outputs look more like GT (hard, diminishing returns),
**teach GS-LRM to handle MVDiff outputs** (potentially much easier).

```
[Before DA1]
GT views ──→ GS-LRM (GT-trained) ──→ Good 3D (23.84 dB)
MVDiff views ──→ GS-LRM (GT-trained) ──→ Bad 3D (8.44 dB)

[After DA1]
GT views ──→ GS-LRM (DA-finetuned) ──→ Slightly worse 3D (~22-23 dB?)
MVDiff views ──→ GS-LRM (DA-finetuned) ──→ Better 3D (~10-12 dB?)
```

### Why This Should Work

1. **Domain adaptation is well-studied**: Fine-tuning on target domain data is the
   standard approach for distribution shift in deep learning.

2. **Low LR preserves GT knowledge**: Using lr=1e-6 (same as GT training but with
   `reset_training_state: true`) means the model adapts gradually without forgetting
   GT-learned 3D reconstruction priors.

3. **GT validation measures real performance**: We validate on GT images, not MVDiff,
   so we measure actual reconstruction quality improvement.

4. **Evidence from transfer rate**: The 14% transfer rate suggests the gap is not in
   MVDiff image quality (which improved significantly) but in GS-LRM's processing of
   non-GT inputs. DA directly addresses this.

---

## 3. Three-Step Pipeline

### Overview

```
Step 1: Generate MVDiff views          Step 2: Format dataset       Step 3: Fine-tune GS-LRM
─────────────────────────             ─────────────────────       ─────────────────────────
For each train frame (2880):          (Integrated in Step 1)      Resume from GT-trained ckpt
  input: GT view 0 image                                          Train on MVDiff views
  → MVDiff E2 inference                                           Validate on GT views
  → 6 generated views                                             7500 steps, lr=1e-6
  → save as GS-LRM format
```

### Step 1: MVDiff Data Generation

**Script**: `mouse_extensions/scripts/domain_adapt/generate_mvdiff_train_data.py`

**What it does**:
For each of the 2880 training frames, takes the GT view 0 (front view) as input,
runs MVDiff E2 (best checkpoint) inference to generate 6 multi-view images, and
saves them in GS-LRM's expected dataset format.

**Input**:
- MVDiff E2 checkpoint: `mouse_M5t2_randref_sparse/checkpoint-20000` (val PSNR 27.70)
- Train frame list: `M5/data_mouse_t2_train.txt` (2880 frames, indices 0-2879)
- GT data root: `~/data/preprocessed/FaceLift_mouse/M5/` (for camera JSON copying)

**Process per frame**:
```
GT frame dir (e.g., M5/000042/)
├── opencv_cameras.json     ← copy to output
└── images/
    └── cam_000.png         ← input to MVDiff (view 0)

                  MVDiff E2 inference
                  (50 steps, guidance_scale=3.0, seed=42)
                            │
                            ▼

M5_mvdiff/000042/
├── opencv_cameras.json     ← copied from GT (same cameras)
└── images/
    ├── cam_000.png         ← MVDiff generated view 0
    ├── cam_001.png         ← MVDiff generated view 1
    ├── cam_002.png         ← MVDiff generated view 2
    ├── cam_003.png         ← MVDiff generated view 3
    ├── cam_004.png         ← MVDiff generated view 4
    └── cam_005.png         ← MVDiff generated view 5
```

**Key design decisions**:
- **Camera JSON from GT**: MVDiff generates views for the same camera poses, so we reuse
  GT camera parameters. This ensures GS-LRM receives correct camera intrinsics/extrinsics.
- **RGB only (no alpha)**: MVDiff outputs RGB images, not RGBA. The DA config sets
  `remove_alpha: true` to handle this.
- **Deterministic (seed=42)**: Same seed for reproducibility.
- **Resume support**: `--batch-start`/`--batch-end` flags allow restarting from any point.
  Existing frames (with `cam_005.png`) are automatically skipped.

**Output**:
- `~/data/preprocessed/FaceLift_mouse/M5_mvdiff/` — 2880 frame directories
- `M5_mvdiff/data_mouse_t2_train_mvdiff.txt` — Frame list for GS-LRM training

**Resources**:
- GPU: 1x A6000 (48GB), uses ~10GB VRAM
- Speed: ~10 seconds/frame (50-step DDPM sampling)
- Total time: ~8 hours for 2880 frames

**Command**:
```bash
CUDA_VISIBLE_DEVICES=6 /home/joon/anaconda3/envs/facelift/bin/python \
  mouse_extensions/scripts/domain_adapt/generate_mvdiff_train_data.py \
  --mvdiff-checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000 \
  --data-txt ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
  --gt-dir ~/data/preprocessed/FaceLift_mouse/M5 \
  --output-dir ~/data/preprocessed/FaceLift_mouse/M5_mvdiff \
  --guidance-scale 3.0
```

### Step 2: Dataset Formatting (Integrated)

Step 1's script already outputs data in GS-LRM format:
- Same directory structure as GT dataset (`{frame}/opencv_cameras.json` + `images/cam_XXX.png`)
- Generates the `data_mouse_t2_train_mvdiff.txt` file listing all frame directories
- GS-LRM's dataloader can read it directly without any conversion

### Step 3: GS-LRM Fine-tuning

**Config**: `configs/mouse/uniform/domain_adapt_E2_v1.yaml`

**Key parameters vs GT baseline**:

| Parameter | GT Baseline (`base_uniform_v2`) | DA1 (`domain_adapt_E2_v1`) | Rationale |
|-----------|:------:|:------:|-----------|
| Dataset | `M5/data_mouse_t2_train.txt` (GT) | `M5_mvdiff/data_mouse_t2_train_mvdiff.txt` | MVDiff-generated views |
| LR | 1e-6 | 1e-6 | Same (conservative fine-tuning) |
| Max steps | 15,000 | 7,500 | Fewer (fine-tuning, not from scratch) |
| Warmup | 500 | 200 | Shorter warmup |
| Resume ckpt | None | `M5t2_E0_1_facelift/best_psnr.pt` | Pre-trained GT model |
| Reset LR | — | true | Fresh LR schedule |
| Reset state | — | true | Fresh optimizer state |
| remove_alpha | false | true | MVDiff outputs RGB |
| Validation | GT images | GT images | Real-world performance |
| Input views | 4 | 4 | Same architecture |

**Resume checkpoint**: `M5t2_E0_1_facelift/best_psnr.pt`
- This is the best GS-LRM model trained on GT data (PSNR 22.34 on 4-view input)
- We resume from this to preserve learned 3D reconstruction priors

**Validation strategy**: Crucially, validation uses **GT images** (not MVDiff).
This measures the model's actual reconstruction quality on clean inputs.
If DA1 succeeds, the model should generalize better to both GT and MVDiff inputs.

**Command** (to be run after Step 1 completes):
```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=6 /home/joon/anaconda3/envs/facelift/bin/python \
  train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/domain_adapt_E2_v1.yaml
```

**Expected resources**:
- GPU: 1x A6000 (48GB), ~22GB VRAM
- Training time: ~12-15 hours (7500 steps)

---

## 4. Expected Outcomes

### Success Criteria

| Metric | Current E2E (no DA) | Target (with DA) | Significance |
|--------|:---:|:---:|---|
| PSNR_fg | 8.44 | 10-12 | +1.5-3.5 dB would validate H_T1 |
| IoU | 0.495 | 0.6-0.7 | Better shape coverage |
| Coverage | 0.750 | 0.80+ | More complete reconstruction |

### Interpretation Guide

| Result | Interpretation | Next Step |
|--------|---------------|-----------|
| E2E PSNR_fg > 10.0 | **H_T1 confirmed**: Distribution mismatch was the primary bottleneck | DA2: Mixed training (GT + MVDiff), guidance scale sweep |
| E2E PSNR_fg 9-10 | **Partial confirmation**: DA helps but gap remains | Investigate geometric consistency (H_T3) |
| E2E PSNR_fg < 9.0 | **H_T1 insufficient**: Other factors dominate | Focus on MVDiff architecture changes |

### Risk: GT Performance Degradation

Fine-tuning on MVDiff data may degrade performance on GT inputs (catastrophic forgetting).
We monitor this via GT validation. Acceptable tradeoff:
- GT PSNR drops by <2 dB: acceptable (we need E2E, not GT-only)
- GT PSNR drops by >3 dB: too much — consider DA2 (mixed training)

---

## 5. Files & Paths

### Scripts
| File | Location (gpu03) | Purpose |
|------|-------------------|---------|
| Data generation | `mouse_extensions/scripts/domain_adapt/generate_mvdiff_train_data.py` | Step 1: MVDiff batch inference |
| GS-LRM training | `train_gslrm.py` (project root) | Step 3: Fine-tuning |

### Configs
| File | Location (gpu03) | Purpose |
|------|-------------------|---------|
| DA1 config | `configs/mouse/uniform/domain_adapt_E2_v1.yaml` | GS-LRM fine-tune settings |
| Base config | `configs/mouse/uniform/base_uniform_v2.yaml` | Base settings (inherited) |

### Data
| Path | Content | Size |
|------|---------|------|
| `~/data/preprocessed/FaceLift_mouse/M5/` | GT dataset (2880 train + 360 val + 360 test) | ~20GB |
| `~/data/preprocessed/FaceLift_mouse/M5_mvdiff/` | MVDiff-generated dataset (2880 train) | ~12GB (est.) |
| `M5_mvdiff/data_mouse_t2_train_mvdiff.txt` | Frame list for training | 2880 lines |

### Checkpoints
| Path | Content |
|------|---------|
| `.../mvdiffusion/mouse_M5t2_randref_sparse/checkpoint-20000` | MVDiff E2 (best, val 27.70) — used for data generation |
| `.../gslrm/M5t2_E0_1_facelift/best_psnr.pt` | GS-LRM GT-trained (PSNR 22.34) — resume checkpoint |
| `.../gslrm/domain_adapt_E2_v1/` | DA1 output checkpoint directory |

---

## 6. Evaluation Plan

After DA1 training completes:

### 6.1 Standard E2E Evaluation
Run the full FaceLift E2E pipeline with DA1's GS-LRM checkpoint:
1. MVDiff E2 generates 6 views from test frames (360 frames)
2. DA1 GS-LRM reconstructs 3D from generated views
3. Render novel views and compute fair eval metrics

### 6.2 GT Input Evaluation
Run GS-LRM with GT views to measure potential degradation:
- Compare DA1 vs original GT-trained on same GT inputs
- Acceptable if PSNR drop < 2 dB

### 6.3 Comparison Matrix

| Condition | GS-LRM Checkpoint | Input Views |
|-----------|-------------------|------------|
| Upper bound | GT-trained | GT 6v |
| DA1 on GT | DA1-finetuned | GT 6v |
| DA1 E2E | DA1-finetuned | MVDiff E2 6v |
| Baseline E2E | GT-trained | MVDiff E2 6v |

---

## 7. Future Directions (if DA1 succeeds)

### DA2: Mixed Training
Train GS-LRM on 50/50 mix of GT and MVDiff views.
Hypothesis: Prevents catastrophic forgetting while adapting to MVDiff distribution.

### DA3: Progressive Domain Adaptation
Start with GT views, gradually increase MVDiff proportion during training.
Hypothesis: Smoother adaptation, better retention of GT priors.

### DA4: Guidance Scale Sweep (inference-only)
Test different guidance scales (1.0, 2.0, 3.0, 5.0) for MVDiff generation.
Hypothesis: Optimal guidance scale for E2E may differ from optimal for image quality.

---

## 8. Navigation

| Link | Document |
|------|----------|
| ← Hub | [[INDEX]] |
| ← Bottleneck analysis | [[mvdiffusion_bottleneck_analysis]] (parent: hypothesis context) |
| ← FL vs PS | [[FL_vs_PS_comparison]] |
| → Training settings | [[FaceLift_training_optimal_settings]] (base config reference) |
| → Eval protocol | [[evaluation_protocol_v1]] (DA1 eval method) |
| → Preprocessing | [[datasets/PREPROCESSING_REGISTRY]] (M5 dataset specs) |

---

*Domain Adaptation DA1 Documentation v1.1 | Updated: 2026-02-22*
