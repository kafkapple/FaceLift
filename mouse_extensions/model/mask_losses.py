"""
Mask Loss Module - Literature-Based Implementation

Based on:
- Pose Splatter (NeurIPS 2025): Normalized masked L1 for small foreground
- LGM (ECCV 2024): MSE alpha supervision for shape convergence
- GaussianObject (SIGGRAPH Asia 2024): BCE alpha supervision
- Object-Centric 2DGS (2025): Background opacity penalty
- Nerfstudio: Background compositing loss

Location: mouse_extensions/model/mask_losses.py
Author: FaceLift Mouse Extensions
Date: 2026-01-24
Version: 2.0
"""

import torch
import torch.nn.functional as F
from enum import Enum
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field


# Debug utilities
from mouse_extensions.utils.debug_breakpoints import debug_break, debug_inspect


# =============================================================================
# Enums and Configuration
# =============================================================================

class MaskMode(Enum):
    """
    Mask modes for loss computation based on literature.

    Literature References:
        - NONE: Original 3DGS, GS-LRM, FaceLift (no masking)
        - GT: Pose Splatter (NeurIPS 2025) - normalized masked loss
        - GT_ALPHA_SUP: LGM + Pose Splatter combination (RECOMMENDED)
        - ALPHA_SUP_ONLY: LGM (ECCV 2024) - alpha supervision only
        - BG_PENALTY: Object-Centric 2DGS (2025) - background loss
        - COMPOSITE: Nerfstudio - background compositing
    """
    NONE = "none"
    GT = "gt"
    GT_ALPHA_SUP = "gt_alpha_sup"
    ALPHA_SUP_ONLY = "alpha_sup_only"
    BG_PENALTY = "bg_penalty"
    COMPOSITE = "composite"

    # Deprecated (kept for backward compatibility)
    ALPHA = "alpha"  # DEPRECATED: No literature support
    RGB_PRED = "rgb_pred"  # DEPRECATED: Limited use


class AlphaLossType(Enum):
    """Alpha supervision loss types."""
    MSE = "mse"    # LGM style - stable gradients (RECOMMENDED)
    BCE = "bce"    # GaussianObject style - sharp boundaries
    DICE = "dice"  # IoU-like
    FOCAL = "focal"  # Imbalanced fg/bg


@dataclass
class MaskLossConfig:
    """
    Configuration for mask-based loss computation.

    Example YAML:
        training:
          losses:
            mask_mode: gt_alpha_sup
            alpha_loss_weight: 0.1
            alpha_loss_type: mse
            bg_loss_weight: 0.0
            normalize_by_mask: true
            background_color: 1.0
    """
    mode: MaskMode = MaskMode.NONE

    # Alpha supervision (LGM, GaussianObject)
    alpha_loss_weight: float = 0.0
    alpha_loss_type: AlphaLossType = AlphaLossType.MSE
    alpha_focal_gamma: float = 2.0

    # Background penalty (Object-Centric 2DGS)
    bg_loss_weight: float = 0.0

    # Pose Splatter normalization
    normalize_by_mask: bool = True

    # Background color for compositing
    background_color: float = 1.0  # white

    # Thresholds (for deprecated modes only)
    alpha_threshold: float = 0.5
    rgb_threshold: float = 0.1


# =============================================================================
# Core Loss Functions
# =============================================================================

def compute_masked_rgb_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "l2",
    normalize_by_mask: bool = True,
) -> torch.Tensor:
    """
    Compute masked RGB loss with optional normalization.

    Based on Pose Splatter (NeurIPS 2025):
        L_color = Σ |pred - gt| * mask / (3 * Σ mask)

    Normalization by mask pixel count prevents bias toward large objects.
    Critical for small foreground (e.g., mouse) vs large background.

    Args:
        pred: Predicted RGB [B, 3, H, W] or [B*V, 3, H, W]
        gt: Ground truth RGB [B, 3, H, W] or [B*V, 3, H, W]
        mask: Binary mask [B, 1, H, W] or [B*V, 1, H, W]
        loss_type: "l1" (Pose Splatter) or "l2" (MSE)
        normalize_by_mask: If True, divide by mask pixel count

    Returns:
        Scalar loss tensor

    Reference:
        Pose Splatter: "L1 loss normalized by mask pixels to avoid
        bias toward small objects"
    """
    debug_break("loss")  # BP3: compute_masked_rgb_loss entry
    # Expand mask to RGB channels
    if mask.shape[1] == 1 and pred.shape[1] == 3:
        mask_expanded = mask.expand_as(pred)
    else:
        mask_expanded = mask

    # Per-pixel loss
    if loss_type == "l1":
        pixel_loss = torch.abs(pred - gt)
    elif loss_type == "l2":
        pixel_loss = (pred - gt) ** 2
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Apply mask
    masked_loss = pixel_loss * mask_expanded

    if normalize_by_mask:
        # Pose Splatter normalization
        num_valid = mask_expanded.sum()
        if num_valid > 0:
            return masked_loss.sum() / (num_valid + 1e-6)
        else:
            return masked_loss.sum() * 0.0
    else:
        return masked_loss.mean()


def compute_alpha_supervision_loss(
    rendered_alpha: torch.Tensor,
    gt_mask: torch.Tensor,
    loss_type: AlphaLossType = AlphaLossType.MSE,
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """
    Compute alpha supervision loss.

    Based on:
        - LGM (ECCV 2024): L_α = MSE(rendered_α, GT_mask)
          "faster convergence of the shape"
        - GaussianObject: BCE(rendered_α, GT_mask)

    Args:
        rendered_alpha: Rendered alpha [B, 1, H, W]
        gt_mask: Ground truth mask [B, 1, H, W]
        loss_type: MSE (stable) or BCE (sharp)
        focal_gamma: Gamma for focal loss

    Returns:
        Scalar loss tensor
    """
    if rendered_alpha is None or gt_mask is None:
        return torch.tensor(0.0)

    # Shape alignment
    if rendered_alpha.shape != gt_mask.shape:
        gt_mask = F.interpolate(
            gt_mask, size=rendered_alpha.shape[-2:],
            mode='bilinear', align_corners=False
        )

    # Numerical stability
    rendered_alpha = rendered_alpha.clamp(1e-6, 1 - 1e-6)
    gt_mask = gt_mask.clamp(0, 1)

    if loss_type == AlphaLossType.MSE:
        return F.mse_loss(rendered_alpha, gt_mask)

    elif loss_type == AlphaLossType.BCE:
        with torch.amp.autocast(device_type='cuda', enabled=False):
            return F.binary_cross_entropy(
                rendered_alpha.float(), gt_mask.float()
            )

    elif loss_type == AlphaLossType.DICE:
        intersection = (rendered_alpha * gt_mask).sum()
        union = rendered_alpha.sum() + gt_mask.sum()
        return 1.0 - 2.0 * intersection / (union + 1e-6)

    elif loss_type == AlphaLossType.FOCAL:
        with torch.amp.autocast(device_type='cuda', enabled=False):
            bce = F.binary_cross_entropy(
                rendered_alpha.float(), gt_mask.float(), reduction='none'
            )
        p_t = rendered_alpha * gt_mask + (1 - rendered_alpha) * (1 - gt_mask)
        focal_weight = (1 - p_t) ** focal_gamma
        return (focal_weight * bce).mean()

    else:
        raise ValueError(f"Unknown alpha loss type: {loss_type}")


def compute_background_penalty_loss(
    rendered_alpha: torch.Tensor,
    gt_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute background opacity penalty.

    Based on Object-Centric 2DGS (2025):
        L_b = (1/(h*w)) * Σ[A_i * (1 - M_i)]

    Penalizes opacity in background regions.

    Benefits (from paper):
        - 96% model size reduction
        - 71% training speed improvement

    Args:
        rendered_alpha: Rendered alpha [B, 1, H, W]
        gt_mask: GT foreground mask [B, 1, H, W]

    Returns:
        Scalar loss tensor
    """
    if rendered_alpha is None or gt_mask is None:
        return torch.tensor(0.0)

    bg_mask = 1.0 - gt_mask
    bg_opacity = rendered_alpha * bg_mask
    return bg_opacity.mean()


def compute_composite_loss(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    rendered_alpha: torch.Tensor,
    gt_mask: torch.Tensor,
    bg_color: float = 1.0,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Compute background compositing loss.

    Based on Nerfstudio `blend_background_for_loss_computation`:
        pred_composite = pred_rgb * rendered_α + bg * (1 - rendered_α)
        gt_composite = gt_rgb * gt_mask + bg * (1 - gt_mask)
        loss = MSE(pred_composite, gt_composite)

    Provides implicit alpha supervision through RGB loss.

    Args:
        pred_rgb: Predicted RGB [B, 3, H, W]
        gt_rgb: Ground truth RGB [B, 3, H, W]
        rendered_alpha: Rendered alpha [B, 1, H, W]
        gt_mask: GT foreground mask [B, 1, H, W]
        bg_color: Background color (default: 1.0 = white)
        loss_type: "l1" or "l2"

    Returns:
        Scalar loss tensor
    """
    if rendered_alpha is None or gt_mask is None:
        return F.mse_loss(pred_rgb, gt_rgb)

    # Composite with respective masks
    pred_composite = pred_rgb * rendered_alpha + bg_color * (1 - rendered_alpha)
    gt_composite = gt_rgb * gt_mask + bg_color * (1 - gt_mask)

    if loss_type == "l1":
        return F.l1_loss(pred_composite, gt_composite)
    else:
        return F.mse_loss(pred_composite, gt_composite)


def compute_effective_rank_loss(
    scales: torch.Tensor,
    target_rank: float = 3.0,
    eps: float = 1e-8,
    log_scale: bool = True,
) -> torch.Tensor:
    """
    Effective Rank Regularization (ERR) to penalize pancake Gaussians.

    Encourages Gaussians to have higher effective rank (more isotropic)
    by maximizing the entropy of the normalized covariance eigenvalues.

    Effective rank = exp(H) where H = -sum(p_i * log(p_i)) and p_i = s_i^2 / sum(s_j^2).
    Range: [1.0, 3.0] where 1 = degenerate (needle/pancake), 3 = sphere.

    Implementation note:
        When scales are in log-space (log_s), the normalized eigenvalues become:
            p_i = exp(2*log_s_i) / sum_j(exp(2*log_s_j)) = softmax(2*log_s)_i
        This avoids numerical issues from explicit exp and is more stable.

    Reference:
        Roy & Bhattacharyya (2007), "Effective Rank" concept applied to
        3D Gaussian Splatting regularization.

    Args:
        scales: Gaussian scales [B, N, 3] or [N, 3].
        target_rank: Target effective rank (3.0 = sphere, default).
        eps: Small constant for numerical stability.
        log_scale: If True (default), treat input as log-scale values.
                   GS-LRM's to_gs() returns log-scales (pre-exp activation).
                   If False, treat as raw scale values.

    Returns:
        Scalar loss: mean(target_rank - effective_rank), >= 0.
    """
    if scales.numel() == 0:
        return torch.tensor(0.0, device=scales.device)

    # Flatten to [M, 3] for uniform handling
    scales_flat = scales.reshape(-1, 3)

    if log_scale:
        # Use log-sum-exp trick for numerical stability:
        # p_i = exp(2*log_s_i) / sum(exp(2*log_s_j)) = softmax(2*log_s)
        log_eigenvalues = 2.0 * scales_flat  # [M, 3]
        log_p = torch.nn.functional.log_softmax(log_eigenvalues, dim=-1)  # [M, 3]
        p = torch.exp(log_p)  # [M, 3]
    else:
        # Raw scales: eigenvalues = s^2, then normalize
        eigenvalues = scales_flat ** 2  # [M, 3]
        eigenvalues_sum = eigenvalues.sum(dim=-1, keepdim=True)  # [M, 1]
        p = eigenvalues / (eigenvalues_sum + eps)  # [M, 3]
        log_p = torch.log(p + eps)

    # Shannon entropy: H = -sum(p_i * log(p_i))
    entropy = -(p * log_p).sum(dim=-1)  # [M]

    # Effective rank = exp(H)
    effective_rank = torch.exp(entropy)  # [M], range [1, 3]

    # Loss: penalize low effective rank
    loss = (target_rank - effective_rank).clamp(min=0.0).mean()

    return loss


def compute_silhouette_iou_loss(
    rendered_alpha: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute silhouette IoU loss.

    Based on Pose Splatter (NeurIPS 2025):
        L_IoU = 1 - Σ(m̂·m) / Σ(m̂ + m - m̂·m)

    Args:
        rendered_alpha: Rendered alpha [B, 1, H, W]
        gt_mask: GT mask [B, 1, H, W]
        threshold: Binarization threshold

    Returns:
        1 - IoU
    """
    if rendered_alpha is None or gt_mask is None:
        return torch.tensor(0.0)

    pred_binary = (rendered_alpha > threshold).float()
    gt_binary = (gt_mask > threshold).float()

    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection

    iou = intersection / (union + 1e-6)
    return 1.0 - iou


# =============================================================================
# Unified Loss Computer
# =============================================================================

class MaskLossComputer:
    """
    Unified mask loss computation based on literature.

    Supports multiple modes:
        - NONE: No masking (Original 3DGS, GS-LRM)
        - GT: GT-masked RGB with normalization (Pose Splatter)
        - GT_ALPHA_SUP: GT-masked + alpha supervision (RECOMMENDED)
        - ALPHA_SUP_ONLY: Alpha supervision only (LGM)
        - BG_PENALTY: Background opacity penalty (Object-Centric 2DGS)
        - COMPOSITE: Background compositing (Nerfstudio)

    Usage:
        config = MaskLossConfig(
            mode=MaskMode.GT_ALPHA_SUP,
            alpha_loss_weight=0.1,
            alpha_loss_type=AlphaLossType.MSE,
        )
        computer = MaskLossComputer(config)

        losses = computer(pred_rgb, gt_rgb, gt_mask, rendered_alpha)
        total_loss = losses['total']
    """

    def __init__(self, config: Union[MaskLossConfig, dict]):
        if isinstance(config, dict):
            config = self._from_dict(config)
        self.config = config
        self._warn_deprecated()
        self._log_config()

    def _from_dict(self, d: dict) -> MaskLossConfig:
        """Create config from dictionary (YAML losses section)."""
        mode_str = d.get('mask_mode', 'none')
        mode_map = {
            'none': MaskMode.NONE,
            'gt': MaskMode.GT,
            'gt_alpha_sup': MaskMode.GT_ALPHA_SUP,
            'alpha_sup_only': MaskMode.ALPHA_SUP_ONLY,
            'bg_penalty': MaskMode.BG_PENALTY,
            'composite': MaskMode.COMPOSITE,
            'alpha': MaskMode.ALPHA,
            'rgb_pred': MaskMode.RGB_PRED,
        }
        mode = mode_map.get(mode_str, MaskMode.NONE)

        alpha_type_str = d.get('alpha_loss_type', 'mse')
        alpha_type_map = {
            'mse': AlphaLossType.MSE,
            'bce': AlphaLossType.BCE,
            'dice': AlphaLossType.DICE,
            'focal': AlphaLossType.FOCAL,
        }
        alpha_type = alpha_type_map.get(alpha_type_str, AlphaLossType.MSE)

        return MaskLossConfig(
            mode=mode,
            alpha_loss_weight=d.get('alpha_loss_weight', 0.0),
            alpha_loss_type=alpha_type,
            alpha_focal_gamma=d.get('alpha_focal_gamma', 2.0),
            bg_loss_weight=d.get('bg_loss_weight', 0.0),
            normalize_by_mask=d.get('normalize_by_mask', True),
            background_color=d.get('background_color', 1.0),
            alpha_threshold=d.get('alpha_mask_threshold', 0.5),
            rgb_threshold=d.get('pred_mask_threshold', 0.1),
        )

    def _warn_deprecated(self):
        """Warn about deprecated modes."""
        if self.config.mode == MaskMode.ALPHA:
            print(
                "[WARNING] mask_mode='alpha' is DEPRECATED. "
                "No literature support. Use 'gt' + alpha_loss_weight > 0."
            )
        elif self.config.mode == MaskMode.RGB_PRED:
            print(
                "[WARNING] mask_mode='rgb_pred' is DEPRECATED. "
                "Consider 'gt' or 'bg_penalty'."
            )

    def _log_config(self):
        """Log configuration."""
        mode = self.config.mode.value
        alpha_w = self.config.alpha_loss_weight
        bg_w = self.config.bg_loss_weight
        print(f"[MaskLossComputer] mode={mode}, alpha_w={alpha_w}, bg_w={bg_w}")

    def __call__(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_mask: Optional[torch.Tensor],
        rendered_alpha: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all mask-related losses.

        Args:
            pred_rgb: Predicted RGB [B, 3, H, W]
            gt_rgb: Ground truth RGB [B, 3, H, W]
            gt_mask: GT foreground mask [B, 1, H, W]
            rendered_alpha: Rendered alpha [B, 1, H, W]

        Returns:
            Dict with 'rgb', 'alpha', 'bg', 'iou', 'total' losses
        """
        device = pred_rgb.device
        losses = {
            'rgb': torch.tensor(0.0, device=device),
            'alpha': torch.tensor(0.0, device=device),
            'bg': torch.tensor(0.0, device=device),
            'iou': torch.tensor(0.0, device=device),
        }

        mode = self.config.mode
        cfg = self.config

        # =================================================================
        # RGB Loss
        # =================================================================

        if mode == MaskMode.NONE or mode == MaskMode.ALPHA_SUP_ONLY:
            losses['rgb'] = F.mse_loss(pred_rgb, gt_rgb)

        elif mode in [MaskMode.GT, MaskMode.GT_ALPHA_SUP]:
            if gt_mask is not None:
                losses['rgb'] = compute_masked_rgb_loss(
                    pred_rgb, gt_rgb, gt_mask,
                    loss_type="l2",
                    normalize_by_mask=cfg.normalize_by_mask,
                )
            else:
                losses['rgb'] = F.mse_loss(pred_rgb, gt_rgb)

        elif mode == MaskMode.BG_PENALTY:
            losses['rgb'] = F.mse_loss(pred_rgb, gt_rgb)

        elif mode == MaskMode.COMPOSITE:
            losses['rgb'] = compute_composite_loss(
                pred_rgb, gt_rgb, rendered_alpha, gt_mask,
                bg_color=cfg.background_color,
                loss_type="l2",
            )

        elif mode == MaskMode.ALPHA:
            # DEPRECATED
            if rendered_alpha is not None:
                alpha_mask = (rendered_alpha > cfg.alpha_threshold).float()
                losses['rgb'] = compute_masked_rgb_loss(
                    pred_rgb, gt_rgb, alpha_mask,
                    loss_type="l2",
                    normalize_by_mask=cfg.normalize_by_mask,
                )
            else:
                losses['rgb'] = F.mse_loss(pred_rgb, gt_rgb)

        elif mode == MaskMode.RGB_PRED:
            # DEPRECATED
            bg = cfg.background_color
            dist = (pred_rgb - bg).abs().mean(dim=1, keepdim=True)
            rgb_mask = (dist > cfg.rgb_threshold).float()
            losses['rgb'] = compute_masked_rgb_loss(
                pred_rgb, gt_rgb, rgb_mask,
                loss_type="l2",
                normalize_by_mask=cfg.normalize_by_mask,
            )

        # =================================================================
        # Alpha Supervision Loss (LGM, GaussianObject)
        # =================================================================

        if cfg.alpha_loss_weight > 0 and gt_mask is not None:
            alpha_loss = compute_alpha_supervision_loss(
                rendered_alpha, gt_mask,
                loss_type=cfg.alpha_loss_type,
                focal_gamma=cfg.alpha_focal_gamma,
            )
            losses['alpha'] = cfg.alpha_loss_weight * alpha_loss

        # =================================================================
        # Background Penalty (Object-Centric 2DGS)
        # =================================================================

        if cfg.bg_loss_weight > 0 and gt_mask is not None:
            bg_loss = compute_background_penalty_loss(rendered_alpha, gt_mask)
            losses['bg'] = cfg.bg_loss_weight * bg_loss

        # =================================================================
        # Total
        # =================================================================

        losses['total'] = losses['rgb'] + losses['alpha'] + losses['bg']

        return losses

    def get_mask_for_visualization(
        self,
        gt_mask: Optional[torch.Tensor],
        rendered_alpha: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Get mask used for loss computation (for visualization)."""
        mode = self.config.mode

        if mode in [MaskMode.GT, MaskMode.GT_ALPHA_SUP]:
            return gt_mask
        elif mode == MaskMode.ALPHA:
            if rendered_alpha is not None:
                return (rendered_alpha > self.config.alpha_threshold).float()
        elif mode == MaskMode.RGB_PRED:
            return None  # Computed from rendering
        else:
            return None

    def get_config_info(self) -> Dict[str, any]:
        """Get config info for logging."""
        return {
            'mask/mode': self.config.mode.value,
            'mask/alpha_weight': self.config.alpha_loss_weight,
            'mask/alpha_type': self.config.alpha_loss_type.value,
            'mask/bg_weight': self.config.bg_loss_weight,
            'mask/normalize': self.config.normalize_by_mask,
        }


# =============================================================================
# Factory and Helpers
# =============================================================================

def create_mask_loss_computer(losses_config: dict) -> MaskLossComputer:
    """
    Factory function to create MaskLossComputer from config.

    Args:
        losses_config: training.losses section from YAML

    Returns:
        MaskLossComputer instance
    """
    return MaskLossComputer(losses_config)


def get_recommended_config(scenario: str) -> MaskLossConfig:
    """
    Get recommended config for common scenarios.

    Args:
        scenario: One of:
            - "mouse": Small foreground, white background (FaceLift Mouse)
            - "stable": Most stable, no masking
            - "fast_shape": Fast shape convergence
            - "clean_boundary": Clean object boundaries

    Returns:
        Recommended MaskLossConfig
    """
    configs = {
        "mouse": MaskLossConfig(
            mode=MaskMode.GT_ALPHA_SUP,
            alpha_loss_weight=0.1,
            alpha_loss_type=AlphaLossType.MSE,
            normalize_by_mask=True,
        ),
        "stable": MaskLossConfig(
            mode=MaskMode.NONE,
        ),
        "fast_shape": MaskLossConfig(
            mode=MaskMode.ALPHA_SUP_ONLY,
            alpha_loss_weight=0.1,
            alpha_loss_type=AlphaLossType.MSE,
        ),
        "clean_boundary": MaskLossConfig(
            mode=MaskMode.BG_PENALTY,
            alpha_loss_weight=0.1,
            bg_loss_weight=0.5,
        ),
    }
    return configs.get(scenario, configs["stable"])


# =============================================================================
# Documentation
# =============================================================================

MASK_MODE_DOCS = """
# Mask Loss Module - Quick Reference

## Modes

| Mode | RGB Loss | Alpha Loss | Literature |
|------|----------|------------|------------|
| none | Full image | - | 3DGS, GS-LRM |
| gt | GT masked | - | Pose Splatter |
| gt_alpha_sup | GT masked | MSE/BCE | LGM + Pose (RECOMMENDED) |
| alpha_sup_only | Full image | MSE/BCE | LGM |
| bg_penalty | Full image | BG penalty | Obj-Centric 2DGS |
| composite | Composited | (implicit) | Nerfstudio |

## Config Example

```yaml
training:
  losses:
    mask_mode: gt_alpha_sup
    alpha_loss_weight: 0.1
    alpha_loss_type: mse
    normalize_by_mask: true
    bg_loss_weight: 0.0
```

## Usage

```python
from mouse_extensions.model.mask_losses import MaskLossComputer

computer = MaskLossComputer(config.training.losses)
losses = computer(pred_rgb, gt_rgb, gt_mask, rendered_alpha)
total_loss = losses['total']
```
"""

if __name__ == "__main__":
    print(MASK_MODE_DOCS)
