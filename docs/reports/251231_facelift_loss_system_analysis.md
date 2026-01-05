# FaceLift GS-LRM Loss System Analysis

**Date**: 2024-12-31
**Purpose**: FaceLift 원논문 Loss 구현 체계 정리 및 Mouse 적용 현황 분석

---

## 1. Loss Function Overview

### 1.1 Original FaceLift Paper Loss Components

| Loss Type | Weight (Paper) | Mouse Config | Description |
|-----------|----------------|--------------|-------------|
| **L2 (MSE)** | 1.0 | 1.0 | Pixel-wise reconstruction loss |
| **Perceptual** | 0.5 | 0.5 | VGG19-based feature matching |
| **LPIPS** | 0.0 | 0.0 | Learned perceptual similarity |
| **SSIM** | 0.0 | 0.0 | Structural similarity |
| **PixelAlign** | 0.0 | 0.0 | Ray-Gaussian alignment |
| **PointsDist** | 0.0 (warmup only) | 0.0 | Gaussian distance regularization |

### 1.2 Loss Formula

```
Total Loss = w_l2 * L2_loss
           + w_perceptual * Perceptual_loss
           + w_lpips * LPIPS_loss
           + w_ssim * SSIM_loss
           + w_pixelalign * PixelAlign_loss
           + w_pointsdist * PointsDist_loss
```

---

## 2. Individual Loss Implementations

### 2.1 L2 Loss (MSE)

**Location**: `gslrm/model/gslrm.py:404-425`

```python
def _compute_l2_loss(self, rendering, target, mask=None):
    if use_mask and mask is not None:
        mask_binary = (mask > 0.5).float()
        num_valid = mask_binary.sum().clamp(min=1.0)
        squared_error = (rendering - target) ** 2
        masked_error = squared_error * mask_binary
        return masked_error.sum() / (num_valid * 3)  # 3 for RGB
    else:
        return F.mse_loss(rendering, target)
```

**Input Range**:
- `rendering`: [0, 1] (clamped if `clamp_rendering: true`)
- `target`: [0, 1]

**Output**: Scalar loss value

### 2.2 Perceptual Loss (VGG19)

**Location**: `gslrm/model/utils_losses.py:221-344`

**Architecture**:
- Uses VGG19 layers: conv1, conv2, conv3, conv4, conv5 (after ReLU)
- Layer weights: [1.0, 1/2.6, 1/4.8, 1/3.7, 1/5.6, 10/1.5]

**Normalization**:
```python
# ImageNet mean subtraction
IMAGENET_MEAN = [123.6800, 116.7790, 103.9390]  # BGR order
real_img_normalized = real_img * 255.0 - imagenet_mean
```

**Input Range**: [0, 1] → internally scaled to [0, 255] - mean

### 2.3 LPIPS Loss

**Location**: `gslrm/model/gslrm.py:427-434`

```python
def _compute_lpips_loss(self, rendering, target):
    # LPIPS expects inputs in range [-1, 1]
    return self.lpips_loss_module(
        rendering * 2.0 - 1.0,
        target * 2.0 - 1.0
    ).mean()
```

**Input Range**: [0, 1] → transformed to [-1, 1]

### 2.4 SSIM Loss

**Location**: `gslrm/model/utils_losses.py:345-380`

```python
class SsimLoss(nn.Module):
    def __init__(self, data_range: float = 1.0):
        self.ssim_module = SSIM(
            win_size=11,
            win_sigma=1.5,
            data_range=self.data_range,
            size_average=True,
            channel=3,
        )

    def forward(self, x, y):
        ssim_value = self.ssim_module(x, y)
        ssim_value = torch.clamp(ssim_value, 0.0, 1.0)
        return 1.0 - ssim_value  # Convert similarity to loss
```

**Input Range**: [0, 1]
**Output Range**: [0, 1] (0 = perfect match)

### 2.5 PixelAlign Loss

**Location**: `gslrm/model/gslrm.py:464-483`

Purpose: Ensures pixel-aligned Gaussians lie along ray directions

```python
def _compute_pixelalign_loss(self, img_aligned_xyz, input, mask, b, v, h, w):
    xyz_vec = img_aligned_xyz - input.ray_o
    ortho_vec = xyz_vec - torch.sum(xyz_vec.detach() * input.ray_d, dim=2, keepdim=True) * input.ray_d
    return torch.mean(ortho_vec.norm(dim=2, p=2))
```

### 2.6 PointsDist Loss

**Location**: `gslrm/model/gslrm.py:485-504`

Purpose: Regularizes Gaussian positions to reasonable depth range

---

## 3. Value Ranges

### 3.1 Image Tensor Ranges

| Stage | Format | Range | Shape |
|-------|--------|-------|-------|
| Dataset Output | RGBA | [0, 1] | [B, V, 4, H, W] |
| Model Input (RGB) | RGB | [0, 1] | [B, V, 3, H, W] |
| Model Input (Posed) | RGB + Plucker | [-1, 1] for RGB, unbounded for Plucker | [B, V, 9, H, W] |
| Rendered Output | RGB | [0, 1] after clamp | [B, V, 3, H, W] |
| Target (GT) | RGBA | [0, 1] | [B, V, 4, H, W] |

### 3.2 Gaussian Parameter Ranges

| Parameter | Raw Range | After Activation | Description |
|-----------|-----------|------------------|-------------|
| xyz | unbounded | clipped to [-1, 1] if `clip_xyz: true` | 3D position |
| features (SH) | unbounded | unbounded | Spherical harmonics |
| scaling | unbounded | `exp(x - 2.3)`, clamped max=-1.20 | Log scale |
| rotation | unbounded | normalized quaternion | Orientation |
| opacity | unbounded | `sigmoid(x - 2.0)` | [0, 1] |

**Implementation** (`gslrm/model/gslrm.py:202-234`):
```python
def to_gs(self, gaussians):
    xyz, features, scaling, rotation, opacity = gaussians.split([3, sh_dim, 3, 4, 1], dim=2)

    # Scaling: exp(x - 2.3) clamped
    scaling = (scaling - 2.3).clamp(max=-1.20)

    # Opacity: sigmoid(x - 2.0)
    opacity = opacity - 2.0

    return xyz, features, scaling, rotation, opacity
```

### 3.3 Camera Parameter Ranges

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Camera Distance | 2.7 | FaceLift pretrained model expectation |
| fx, fy | ~700-800 | For 512x512 images |
| cx, cy | ~256 | Image center |

---

## 4. Loss Logging & Monitoring

### 4.1 Currently Logged Metrics

| Metric | Prefix | Description |
|--------|--------|-------------|
| `train/l2_loss` | train | L2 reconstruction loss |
| `train/psnr` | train | Peak Signal-to-Noise Ratio |
| `train/perceptual_loss` | train | VGG19 perceptual loss |
| `train/lpips_loss` | train | LPIPS (if weight > 0) |
| `train/ssim_loss` | train | SSIM loss (if weight > 0) |
| `train/gt_min/max/mean` | train | GT pixel value statistics |
| `train/pred_min/max/mean` | train | Predicted pixel value statistics |
| `val/psnr` | val | Validation PSNR |
| `val/ssim` | val | Validation SSIM |
| `val/lpips` | val | Validation LPIPS |

### 4.2 Diagnostic Statistics

Added in `_compute_all_losses()` for debugging:
```python
losses['gt_min'] = target.min()
losses['gt_max'] = target.max()
losses['gt_mean'] = target.mean()
losses['pred_min'] = rendering.min()
losses['pred_max'] = rendering.max()
losses['pred_mean'] = rendering.mean()
```

---

## 5. Mask Handling

### 5.1 Auto Mask Generation

**Location**: `gslrm/data/mouse_dataset.py:573-579`

```python
if self.auto_generate_mask and image_np.shape[2] == 3:
    threshold = self.mask_threshold / 255.0  # default: 250/255
    is_background = np.all(image_np > threshold, axis=2)
    alpha = (~is_background).astype(np.float32)
    image_np = np.concatenate([image_np, alpha[:, :, None]], axis=2)
```

### 5.2 Masked Loss Options

| Option | Config Key | Effect |
|--------|------------|--------|
| Masked L2 | `masked_l2_loss: true` | L2 only on foreground pixels |
| Masked SSIM | `masked_ssim_loss: true` | SSIM on masked regions |
| Masked PixelAlign | `masked_pixelalign_loss: true` | PixelAlign on foreground |

---

## 6. Warmup Strategies

### 6.1 L2 Warmup

**Config**: `l2_warmup_steps: 50`

During warmup (step < l2_warmup_steps):
- Perceptual loss weight → 0
- LPIPS loss weight → 0
- Only L2 loss active

### 6.2 PointsDist Warmup

**Config**: `warmup_pointsdist: false`

If enabled (step < 1000):
- L2 weight → 0
- Perceptual weight → 0
- PointsDist weight → 0.1
- `clip_xyz` disabled

---

## 7. Recommendations for Mouse Training

### 7.1 Current Working Configuration

```yaml
training:
  losses:
    l2_loss_weight: 1.0
    perceptual_loss_weight: 0.5
    lpips_loss_weight: 0.0      # Can cause gradient explosion
    ssim_loss_weight: 0.0       # Not used in original paper
    pixelalign_loss_weight: 0.0
    pointsdist_loss_weight: 0.0
    masked_l2_loss: false       # Enable if background dominates
```

### 7.2 Debugging Checklist

1. **Check value ranges**: GT and Pred should both be in [0, 1]
2. **Monitor PSNR**: Should increase over training (typical: 15-30 dB)
3. **Check mask coverage**: Foreground should be ~5-30% of image
4. **Verify camera normalization**: Distance should be ~2.7

---

## 8. File References

| Component | File | Lines |
|-----------|------|-------|
| LossComputer | `gslrm/model/gslrm.py` | 237-720 |
| PerceptualLoss | `gslrm/model/utils_losses.py` | 221-344 |
| SsimLoss | `gslrm/model/utils_losses.py` | 345-380 |
| MouseViewDataset | `gslrm/data/mouse_dataset.py` | 275-772 |
| Training Loop | `train_gslrm.py` | 592-999 |

---

## 9. New Features Added (2024-12-31)

### 9.1 Mask IoU Tracking

**Purpose**: Track foreground/background mask accuracy over training

**Location**: `gslrm/model/gslrm.py:377-419`

```python
def _compute_mask_iou(self, rendering, target, gt_mask):
    # Predicted mask from rendering (pixels far from white background)
    color_distance = (rendering - bg_tensor).abs().mean(dim=1, keepdim=True)
    pred_mask = (color_distance > bg_threshold).float()

    # IoU = intersection / union
    intersection = (pred_mask * gt_mask_binary).sum()
    union = ((pred_mask + gt_mask_binary) > 0.5).float().sum()
    return intersection / union.clamp(min=1.0)
```

**Config Options**:
```yaml
training:
  losses:
    mask_iou_threshold: 0.1  # Distance from background to be considered foreground
```

**Logged Metrics**:
- `train/mask_iou`: Training mask IoU
- `val/mask_iou`: Validation mask IoU

### 9.2 Vertical Colorbar Visualization

**Purpose**: Clear error scale visualization in GT vs Pred comparison images

**Location**: `gslrm/model/gslrm.py:684-794`

**Features**:
- Vertical colorbar on right side of error row
- Top (red): 0.3 (max error)
- Bottom (blue): 0.0 (no error)
- Yellow line: Mean error position
- Error statistics text in bottom-left

### 9.3 WandB Section Separation

**Purpose**: Separate images and metrics into distinct wandb UI sections

**Location**: `train_gslrm.py:780-887`

**New Key Structure**:
```
# Images (separate "images" section in wandb)
images/train/supervision_0
images/train/input_0
images/train/turntable_0
images/val/gt_vs_pred_0

# Metrics (separate "metrics" section)
metrics/iter
metrics/lr
metrics/grad_norm

# Loss metrics (under "train" section)
train/l2_loss
train/psnr
train/mask_iou
```

---

*Document generated by Claude Code*
