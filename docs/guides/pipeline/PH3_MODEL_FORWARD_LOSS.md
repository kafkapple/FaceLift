# Phase 3: Model Forward & Loss Computation

> **Navigation**: [← Hub](../PIPELINE_DEEP_DIVE.md) | [Prev: PH2](PH2_DATA_LOADING.md) | [Next: PH4 →](PH4_POSE_CONDITIONING.md)
>
> **핵심 파일**: `gslrm/model/gslrm.py`, `mouse_extensions/model/loss_extensions.py`

---

### 3.1 Forward Pass Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ File: gslrm/model/gslrm.py:1526-1800                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ batch = {image: [B,6,3,512,512], c2w: [B,6,4,4], ...}          │
│                              │                                  │
│   1. Data Split (:1545)      input 5 / target 6                 │
│   2. Plucker Coords (:1560)  [B, 5, 9, H, W] = RGB(3)+Plk(6)  │
│   3. Patch Tokenize (:1575)  [B, V×P, D], P=(512/16)²=1024     │
│   4. Transformer (:1595)     24 layers, 16 heads                │
│   5. Gaussian Gen (:1615)    xyz, features, scale, rot, opacity │
│   6. Render (:820-913)       [B, V, 3, H, W] + alphas           │
│   7. Loss (:343-530)         L2 + Perceptual + SSIM + BG + Alpha│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Loss Computation Details

**File**: `gslrm/model/gslrm.py:343-530`, `mouse_extensions/model/loss_extensions.py`

#### 3.2.1 Mask Computation (`loss_extensions.py:113-260`)

```python
# compute_mask_from_config() — Priority:
# mask_mode (explicit) > use_rendered_alpha_mask > use_predicted_mask > gt > none

# MaskType enum (L17-23):
#   NONE / GT / RGB_PRED / ALPHA

# mask_mode="alpha" (L159-215):
mask = (rendered_alpha > alpha_threshold).float()  # default threshold=0.5
# min_mask_ratio safety: adaptive threshold if coverage too low

# mask_mode="rgb_pred" (L220-232): DEPRECATED
# White-BG assumption, mouse data에서 IoU ~0.06 → 비효과적
```

#### 3.2.2 Alpha Loss (`loss_extensions.py:430-493`)

```python
# compute_alpha_loss(rendered_alpha, gt_alpha)
# Input: rendered_alpha [B*V, 1, H, W], gt_alpha [B*V, 1, H, W]
#
# Loss types:
#   "mse"   (L463):  F.mse_loss
#   "bce"   (L466):  ★ AMP 비호환 → autocast(enabled=False) 강제
#   "dice"  (L473):  1 - 2*intersection / (sum_pred + sum_gt)
#   "focal" (L480):  (1-p_t)^gamma * bce, fg/bg 불균형 대응
```

#### 3.2.3 Opacity Regularization (`loss_extensions.py:619-663`)

```python
# compute_opacity_regularization(opacity) — Floater artifact 감소
# Input: opacity [N] or [B, N], range [0, 1]
# ★ bf16 호환: opacity.float().clamp(1e-4, 1-1e-4)
#
# Types:
#   "entropy":   -p*log(p) - (1-p)*log(1-p)  (p=0,1 최소)
#   "l1_sparse": |mean - target_sparsity|
#   "l2_binary": min(p, 1-p)^2
```

#### 3.2.4 Mask IoU (`loss_extensions.py:264-315`)

```python
# compute_mask_iou(rendering, gt_mask) → scalar IoU
# intersection = (pred_mask * gt_mask).sum()
# union = ((pred_mask + gt_mask) > 0.5).float().sum()
# iou = intersection / union.clamp(min=1.0)
```

---

*← [PH2](PH2_DATA_LOADING.md) | [Hub](../PIPELINE_DEEP_DIVE.md) | [Next: PH4 →](PH4_POSE_CONDITIONING.md)*
