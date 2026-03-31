# Deformation Roadmap (V2→V3)

> [← INDEX](../INDEX.md) | [TEMPORAL_ANALYSIS](TEMPORAL_ANALYSIS.md)
> **Version**: v1.0 | Consolidated: 2026-03-31
> **Note**: V2 param MSE approach abandoned (3/3 audit consensus — param MSE ≠ rendering quality). V3 rendering loss is the current active strategy.

---

## §1. V2: 4DGS Experiment Plan [Abandoned]

> Source: `DEFORMATION_4DGS_EXPERIMENT_PLAN.md` (2026-03-29)
> Based on FaceLift paper arXiv:2412.17812 Appendix 3.5

### 1.1 Objective

FaceLift deformable module for temporal consistency:
- Canonical Gaussians + 8-layer MLP deformation → per-frame Gaussian identity maintenance
- Implicit 3D scene flow learning (rendering supervision)

### 1.2 Paper Architecture (Target)

```
Frame t (canonical)     Frame t+1 (target pseudo-GT)
    |                        |
    v                        v (independent GS-LRM inference)
G_t [1,572,866 x 14]        G_{t+1} [1,572,866 x 14]
    |
    v 8-layer MLP
D(G_t) = [dx,dy,dz,dα,ds]
    |
    v Apply deformation
G'_{t+1} = G_t + D(G_t)
    |
    v Rendering (6-view)
rendered images
    |
    v Loss vs rendered(G_{t+1})
L = MSE + Perceptual
```

### 1.3 Implementation Status (at time of abandonment)

| Component | Status | File |
|-----------|:------:|------|
| DeformationNetwork V2 (8-layer MLP) | ✅ | `model/deformation/deformation_network.py` |
| DeformationTrainer V2 | ✅ | `model/deformation/deformation_trainer_v2.py` |
| ARAP + Velocity losses | ✅ | `model/deformation/temporal_regularization.py` |
| GS-LRM integration | ✅ | `model/deformation/gslrm_integration.py` |
| Pre-computed Gaussian cache | ✅ (code) | `model/deformation/gslrm_integration.py:GaussianCache` |
| **Trained checkpoint** | **❌** | Not executed |

### 1.4 Experiment Plan (Not Executed)

#### Phase 0: Gaussian Cache Generation (one-time)

```bash
ssh gpu03
cd /home/joon/dev/FaceLift
conda activate facelift

# GS-LRM full frame Gaussian inference + cache
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml \
    --precompute_cache

# Expected: ~2 hours (3600 frames × ~2s/frame)
# Output: /node_data/joon/checkpoints/FaceLift/deformation/default/gaussian_cache/frame_*.pt
```

**Config requirements (configs/deformation/default.yaml)**:
- `gslrm.checkpoint`: `/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt`
- `data.data_dir`: `/home/joon/data/preprocessed/FaceLift_mouse/M5`
- `data.frame_range`: `0:3600` (full) or `1500:1860` (cinematic range)

#### Phase 1: Paper-aligned Training (V2)

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.train_deform_v2 \
    --config configs/mouse/deform_v2.yaml

# Config (deform_v2.yaml):
#   num_layers: 8
#   hidden_dim: 256
#   use_positional_encoding: true
#   use_time_embedding: true
#   loss: param(1.0) + ARAP(0.1) + velocity(0.01)
#   epochs: 200
#   lr: 1e-4
#   batch_size: 10000 Gaussians
```

#### Phase 2: Rendering Supervision Addition (paper-aligned)

V2 used Gaussian parameter loss (MSE). Paper uses **rendering loss**.
→ Rendering loss implementation required:

```python
# Pseudo-code for rendering supervision
deformed_gaussians = canonical + mlp(canonical.xyz)
rendered = render_6views(deformed_gaussians, cameras)
target_rendered = render_6views(target_gaussians, cameras)
loss = MSE(rendered, target_rendered) + LPIPS(rendered, target_rendered)
```

#### Phase 3: Temporal Evaluation

```bash
# Generate deformed frames
python -m mouse_extensions.scripts.inference.run_temporal_deform \
    --checkpoint /path/to/deform_v2/best.pt \
    --frame_range 3300:3320 \
    --output_dir outputs/datasets/temporal_eval/deform_v2

# Compare with EMA
python -m mouse_extensions.scripts.eval.temporal_comparison \
    --data_root outputs/datasets/temporal_eval \
    --output_dir outputs/viz/comparison/mouse/deform_vs_ema
```

### 1.5 Key Design Decisions

#### Gaussian Identity Problem

- GS-LRM per-pixel → same count per frame (1,572,866) guaranteed
- Pixel-position identity (index N = same pixel position) ≠ semantic identity
- At 0.05s intervals: spatial ≈ semantic (small motion)
- **Fast motion segments**: explicit correspondence needed → future work

#### Frame Range Selection

| Range | Frames | Purpose |
|-------|:------:|------|
| `1500:1860` | 360 | Cinematic range (active behavior) |
| `3300:3320` | 21 | Dense temporal eval (standard) |
| `0:3600` | 3600 | Full training (max data) |

**Recommended**: Phase 1 uses `1500:1860` (360 frames, fast iteration). Expand to full on success.

#### Visibility Filtering

- Deformation applied to **all** Gaussians before filtering
- At render time: vis_mask applied (foreground only)
- Background Gaussians included in deformation training (consistency maintenance)

### 1.6 Success Criteria

| Metric | Baseline (Original) | Target (Deformed) | Best (EMA α=0.1) |
|--------|:-------------------:|:-----------------:|:-----------------:|
| tOF | 0.2052 | < 0.10 | 0.0230 |
| PSNR_gt | ~20 dB | >= 20 dB | ~19.5 dB (blur) |
| Visual | Flickering | Smooth + sharp | Smooth but blurry |

**Core goal**: Temporal consistency without EMA's blur.

### 1.7 Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pixel-identity != semantic | Deformation failure | Prioritize small motion segments |
| Rendering loss cost | Slow training | Parameter loss warmup → rendering loss fine-tune |
| Overfitting (360 frames) | Generalization failure | Val split, augmentation |

---

## §2. V3: FG-Aware Rendering Loss [Active]

> Source: `DEFORM_V3_FG_AWARE_PLAN.md` (2026-03-31)
> Method: 4× `/deliberate --moa --audit` (Deform strategy + RAFT + FG-only + termination judgment)
> Status: V2 stopped (epoch 26), 237GB cache deleted

### 2.1 Why V3 (V2 Failure Analysis)

#### V2 Structural Problems (3/3 audit consensus)

| Problem | Severity | Description |
|------|:--------:|------|
| **Param MSE ≠ rendering quality** | 🔴 Critical | Gaussian parameter distance does not directly correlate with visual quality |
| **97.5% BG training waste** | 🔴 Critical | Background Gaussians deformation=0 → MLP capacity waste, 237GB cache |
| **Pixel-identity approximate** | 🔴 Critical | Pair training without confirmed correspondence |
| **ARAP topology undefined** | 🟡 Major | k-NN method undefined for 1.57M unstructured points |

#### V2 → V3 Key Changes

| | V2 (Abandoned) | V3 (Planned) |
|---|---|---|
| **Loss** | param MSE (PRIMARY) | **Rendering loss** (PRIMARY) |
| **Data** | ALL 1.57M Gaussians | **FG ~39K + BG 5% sparse** |
| **Cache** | 237GB (85MB/frame) | **~8GB (~3MB/frame)** |
| **Correspondence** | Pixel-identity only | **RAFT + CoTracker hybrid** (Phase 2) |
| **ARAP** | Undefined topology | **k-NN k=8 on FG (world-space)** |
| **Training view** | N/A | **Lock loss (preservation)** |

### 2.2 FG-Aware Gaussian Cache

#### FG Identification Strategy

**Soft filtering** (hard filtering prohibited — audit 3/3 consensus):

```python
# Alpha mask → FG identification
# GS-LRM RGBA images: alpha channel = FG mask
# Pixel-aligned: alpha[u,v] > threshold → Gaussian at (u,v) is FG

# 3-tier classification:
#   FG core:    alpha > 0.8     → weight = 1.0 (full training)
#   FG boundary: 0.2 < alpha ≤ 0.8 → weight = alpha (soft contribution)
#   BG:          alpha ≤ 0.2    → 5% random sample, weight = 0.1
```

**Why soft, not hard**:
- Hard filtering loses boundary Gaussians (fur, whiskers)
- BG sparse sample provides "δ=0" negative signal → generalization maintenance
- Alpha-weighted loss handles boundary naturally

#### Cache Format

```python
# Per-frame cache (V3 format):
{
    "fg_xyz": tensor[N_fg, 3],        # FG core + boundary positions
    "fg_features": tensor[N_fg, 1, 3], # SH features
    "fg_scaling": tensor[N_fg, 3],
    "fg_rotation": tensor[N_fg, 4],
    "fg_opacity": tensor[N_fg, 1],
    "fg_weights": tensor[N_fg],        # alpha-based weights
    "fg_indices": tensor[N_fg],        # original pixel indices (for correspondence)
    "bg_sample_xyz": tensor[N_bg, 3],  # 5% BG sample
    "bg_sample_indices": tensor[N_bg],
}
# Estimated: ~3MB/frame (float16) × 2880 = ~8GB (vs 237GB)
```

#### Cache Generation Script

```bash
# Step 1: GS-LRM inference → full Gaussians + alpha masks
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.cache_fg_gaussians \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --output_dir /node_data/joon/checkpoints/FaceLift/deformation/v3/fg_cache \
    --fg_threshold 0.2 \
    --bg_sample_ratio 0.05 \
    --dtype float16
# Expected: ~2h, output ~8GB
```

### 2.3 V3 Loss Design

#### Loss Components

```
L_total = 1.0·L_render + 0.1·L_arap + 0.05·L_vel + 0.01·L_zero

Phase 2 additions (after RAFT integration):
  + 0.1·L_corr (correspondence)
  + 0.1·L_tOF (temporal consistency)
```

#### L_zero (Zero Regularization — BG 대체)

> Added: commit `45df0f1` (260331). Config: `deform_v3.yaml` `zero_weight: 0.01`

```python
def zero_regularization(model, n_samples=1000, device='cuda'):
    """Random positions → MLP output should be zero (no deformation).
    Replaces V2's 97.5% BG data with cheaper synthetic zero-targets."""
    random_xyz = torch.randn(n_samples, 3, device=device) * 2.0
    random_time = torch.rand(n_samples, 1, device=device)
    delta = model(random_xyz, random_time)
    return delta.pow(2).mean()
```

**Why**: V2는 BG Gaussians(97.5%)를 "deformation=0" 학습에 사용 → 237GB 캐시. L_zero는 랜덤 좌표에서 동일 signal을 0.01 weight로 제공, BG 데이터 없이 MLP의 zero-bias를 유지.

#### L_render (PRIMARY — largest change)

```python
def rendering_loss(gs_deformed, cameras, gt_images):
    """
    Differentiable rendering → image space loss.
    Stochastic: 2-3 random views per step (memory saving).
    """
    selected_cams = random.sample(cameras, k=min(3, len(cameras)))
    loss = 0
    for cam in selected_cams:
        rendered = differentiable_render(gs_deformed, cam)
        loss += 0.8 * L1(rendered, gt_images[cam])
        loss += 0.2 * (1 - SSIM(rendered, gt_images[cam]))
    return loss / len(selected_cams)
```

**Why rendering loss > param MSE**:
- PSNR is measured on rendered images → loss should optimize in rendering space
- Small param change can cause large visual change (and vice versa) → MSE is an inadequate proxy
- Differentiable GS rendering already exists in codebase (gsplat)

#### L_arap (FG-aware)

```python
def fg_arap_loss(g_t, g_deformed, fg_indices, k=8):
    """
    k-NN graph on FG Gaussians (world-space XYZ).
    Graph extends to BG but loss computed on FG only.
    """
    # Build k-NN on FG positions
    knn_graph = build_knn(g_t.xyz[fg_indices], k=k)

    # Edge length preservation
    for (i, j) in knn_graph.edges:
        e_before = g_t.xyz[i] - g_t.xyz[j]
        e_after = g_deformed.xyz[i] - g_deformed.xyz[j]
        loss += (e_after.norm() - e_before.norm()) ** 2

    return loss / len(knn_graph.edges)
```

#### Alpha-Weighted Training

```python
def weighted_loss(loss_per_gaussian, weights):
    """FG core=1.0, boundary=alpha, BG=0.1"""
    return (loss_per_gaussian * weights).sum() / weights.sum()
```

### 2.4 Architecture Changes

#### Network (Minimal Change)

V2's 8-layer MLP retained, input/output changes only:

```
V2: [xyz(3) + PE(60) + time(32) + features(15)] → [dx,dy,dz,dα,ds] (5)
V3: same input → [dx,dy,dz, dq0,dq1,dq2,dq3, ds0,ds1,ds2, dα] (11)
                   position    rotation(quat)    scale        opacity
```

**Why rotation delta added**: Audit flagged rotation omission (o3-mini). Mouse limb rotation is significant.

#### Variable Batch Size

FG Gaussian count varies per frame (~30K-50K):

```python
# Dynamic batching: pad to max in batch, mask out padding
def collate_fg_pairs(batch):
    max_fg = max(b.fg_xyz.shape[0] for b in batch)
    padded = [pad_to(b, max_fg) for b in batch]
    masks = [create_mask(b.fg_xyz.shape[0], max_fg) for b in batch]
    return padded, masks
```

### 2.5 Implementation Priority

| Step | Task | Time | Dependency |
|:----:|------|:----:|:------:|
| **S0** | `cache_fg_gaussians.py` script | 2h | — |
| **S1** | FG cache generation (2880 frames) | 2h GPU | S0 |
| **S2** | `train_deform_v3.py` — rendering loss + FG-aware | 4h | S0 |
| **S3** | 10-frame smoke test (convergence check) | 1h GPU | S1+S2 |
| **S4** | Full training (2880 frames, 100 epochs) | ~4h GPU | S3 pass |
| **S5** | tOF + PSNR measurement (temporal eval) | 1h | S4 |

**Total: ~2 days** (1 day implementation + 1 day training)

#### Go/No-Go Gates

| Gate | Criteria | On Failure |
|------|------|---------|
| S3 smoke test | Loss converges (10 frames, 20 epochs) | Loss design review |
| S5 tOF | tOF < V2 baseline (or < 0.15) | Add correspondence (RAFT) |
| S5 PSNR | PSNR_gt ≥ 20 dB maintained | Adjust rendering loss weight |

### 2.6 Compute Savings

| | V2 | V3 | Savings |
|---|---:|---:|:----:|
| Cache | 237GB | ~8GB | **97%** |
| Per-epoch time | ~24min | ~2min (est.) | **92%** |
| Total training (100ep) | ~40h | ~3-4h | **90%** |
| GPU memory | ~17GB | ~4GB (est.) | **76%** |

### 2.7 Migration from V2

#### Deleted (2026-03-31)

- ✅ V2 training stopped (epoch 26/100)
- ✅ V2 checkpoints deleted (15MB — flawed loss, no reuse value)
- ✅ V1 checkpoints deleted (46MB)
- ✅ 237GB gaussian_cache deleted
- ✅ temporal_inference_v2 deleted

#### Preserved

- ✅ `train_deform_v2.py` (code reference, preserved in git history)
- ✅ `configs/mouse/deform_v2.yaml` (V3 config base)
- ✅ `mouse_extensions/model/deformation.py` (MLP architecture reuse)

### 2.8 V3 → V3.5 (RAFT Phase)

V3 baseline established, then add RAFT correspondence:

```
V3.0: FG-aware + rendering loss (this document)
V3.5: + RAFT/CoTracker correspondence + L_corr + L_tOF
V4.0: + Hierarchical Transformer (long-term)
```

Details: Obsidian `theory/DEFORMATION_STRATEGY.md` §4-7

---

## Related

- ← [[../INDEX]] — Document hub
- ↔ [[TEMPORAL_ANALYSIS]] — Temporal consistency analysis & eval standard
- ↔ [[COMMANDS]] — Experiment commands
- → Obsidian `theory/DEFORMATION_STRATEGY.md` — Theory & long-term strategy

---

*FaceLift | Deformation Roadmap (Consolidated) | v1.0 | 2026-03-31*
*Merged from: DEFORMATION_4DGS_EXPERIMENT_PLAN.md (2026-03-29) + DEFORM_V3_FG_AWARE_PLAN.md (2026-03-31)*
