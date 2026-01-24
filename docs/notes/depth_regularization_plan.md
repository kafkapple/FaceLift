# Depth Regularization Research Plan (v70+)

## 1. Overview

### Goal
Use monocular depth estimation as pseudo ground truth to regularize Gaussian positions and reduce floater artifacts.

### Why Depth Regularization?

| Approach | Pros | Cons |
|----------|------|------|
| Opacity Reg | Simple, no extra data | Indirect constraint |
| **Depth Reg** | Direct 3D constraint | Requires depth model |

Depth regularization provides **direct geometric supervision**, complementing opacity regularization.

---

## 2. Monocular Depth Models Comparison

### 2.1 Recommended: Depth Anything v2

| Aspect | Details |
|--------|---------|
| **Paper** | Depth Anything V2 (2024) |

#### Future: VGGT Integration
- Multi-view depth + camera pose 동시 추정
- CVPR 2025 Best Paper
- 구현 계획: Phase 2 (DA V2 검증 후)
- Reference: https://github.com/facebookresearch/vggt
| **Link** | https://depth-anything-v2.github.io/ |
| **License** | Apache 2.0 |
| **Speed** | ~30 FPS on RTX 3090 |
| **Quality** | State-of-the-art on multiple benchmarks |
| **Models** | ViT-S (24M), ViT-B (97M), ViT-L (335M), ViT-G (1B) |

**Installation**:
```bash
pip install depth-anything-v2
# or
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

### 2.2 Alternative: Marigold

| Aspect | Details |
|--------|---------|
| **Paper** | Marigold: Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation (2024) |
| **Link** | https://marigoldmonodepth.github.io/ |
| **Quality** | Very high (diffusion-based) |
| **Speed** | Slower (~2-5 sec/image) |
| **Use case** | When quality > speed |

### 2.3 VGGT Consideration

| Aspect | Details |
|--------|---------|
| **Paper** | VGGT: Visual Geometry Grounded Transformer |
| **Capability** | Multi-task (depth, normal, camera) |
| **Pros** | Single model for multiple tasks |
| **Cons** | Not depth-specialized, may be less accurate |
| **Verdict** | ⚠️ Consider if already using VGGT for other tasks |

### 2.4 Comparison Table

| Model | Speed | Quality | Memory | Recommendation |
|-------|-------|---------|--------|----------------|
| Depth Anything v2-S | ⭐⭐⭐ | ⭐⭐ | Low | Quick experiments |
| Depth Anything v2-L | ⭐⭐ | ⭐⭐⭐ | Med | **Production** |
| Marigold | ⭐ | ⭐⭐⭐⭐ | High | High quality needs |
| MiDaS/DPT | ⭐⭐ | ⭐⭐ | Med | Legacy/baseline |
| VGGT | ⭐⭐ | ⭐⭐ | High | Multi-task scenarios |

---

## 3. Implementation Plan

### Phase 1: Pseudo Depth Generation Pipeline

```
Input Images → Depth Model → Pseudo GT Depth → Save to Dataset
```

**Tasks**:
1. [ ] Install Depth Anything v2
2. [ ] Create batch inference script
3. [ ] Generate depth for training/validation sets
4. [ ] Save as `.npy` or `.exr` alongside images

**Script Location**: `mouse_extensions/scripts/generate_pseudo_depth.py`

### Phase 2: Dataset Integration

```python
# In GSLRMDataset
def __getitem__(self, idx):
    ...
    # Load pseudo depth if available
    depth_path = image_path.replace(.png, _depth.npy)
    if os.path.exists(depth_path):
        pseudo_depth = np.load(depth_path)
    ...
```

**Tasks**:
1. [ ] Modify dataset to load pseudo depth
2. [ ] Handle missing depth gracefully
3. [ ] Normalize depth values (scale-invariant)

### Phase 3: Loss Integration

**Tasks**:
1. [ ] Connect DepthRegularizer to GSLRM forward pass
2. [ ] Add depth rendering to Gaussian renderer (if not present)
3. [ ] Implement scale-invariant depth loss

**Loss Options**:

```python
# Scale-invariant loss (recommended)
def scale_invariant_loss(pred, gt, mask):
    d = torch.log(pred) - torch.log(gt)
    d = d[mask]
    return torch.sqrt((d**2).mean() - 0.5 * (d.mean())**2)

# Gradient loss (edge-aware)
def gradient_loss(pred, gt):
    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
    gt_dx = gt[:, :, 1:] - gt[:, :, :-1]
    # ... similar for dy
    return (pred_dx - gt_dx).abs().mean() + (pred_dy - gt_dy).abs().mean()
```

---

## 4. Timeline

| Phase | Task | Estimated Effort |
|-------|------|------------------|
| **Phase 1** | Depth generation pipeline | 1-2 days |
| **Phase 2** | Dataset integration | 0.5 day |
| **Phase 3** | Loss integration | 0.5 day |
| **Testing** | v70 experiment | 1 day |

**Total**: ~3-4 days

---

## 5. Config Design (v70)

```yaml
# gslrm_v70_depth_reg.yaml
training:
  dataset:
    load_pseudo_depth: true
    pseudo_depth_suffix: "_depth.npy"
  losses:
    depth_reg_weight: 0.1
    depth_reg_type: "scale_invariant"  # scale_invariant | l1 | gradient
    depth_normalize: true
```

---

## 6. References

1. **Depth Anything v2** (Yang et al., 2024)
   - https://arxiv.org/abs/2406.09414
   - https://github.com/DepthAnything/Depth-Anything-V2

2. **Marigold** (Ke et al., 2024)
   - https://arxiv.org/abs/2312.02145
   - https://github.com/prs-eth/Marigold

3. **MonoGS** (Matsuki et al., 2024)
   - Gaussian Splatting with monocular depth
   - https://arxiv.org/abs/2405.13776

4. **DN-Splatter** (Turkulainen et al., 2024)
   - Depth and Normal priors for Gaussian Splatting
   - https://arxiv.org/abs/2403.17822

5. **VGGT** (Wang et al., 2025)
   - https://github.com/facebookresearch/vggt

---

## 7. Decision Points

### Q1: Which depth model?
**Recommendation**: Start with **Depth Anything v2-L** (best balance)

### Q2: Pre-compute vs on-the-fly?
**Recommendation**: **Pre-compute** (faster training, one-time cost)

### Q3: Loss type?
**Recommendation**: **Scale-invariant** (robust to scale ambiguity)

---

*Created: 2026-01-16 | Status: Planning*
