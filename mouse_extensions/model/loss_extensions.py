"""
Loss Extensions

Mask computation options and ghost metrics for loss calculation.
"""

import torch
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


class MaskType(Enum):
    """Mask types for loss computation."""
    NONE = "none"           # Full image loss (no mask)
    GT = "gt"               # Ground truth mask from alpha channel
    RGB_PRED = "rgb_pred"   # Predicted from RGB distance to background
    ALPHA = "alpha"         # Rendered alpha from Gaussian splatting


@dataclass
class MaskConfig:
    """Configuration for mask computation."""
    mask_type: MaskType = MaskType.NONE
    use_masked_loss: bool = False
    pred_mask_threshold: float = 0.1
    alpha_mask_threshold: float = 0.5
    background_color: float = 1.0  # white


def compute_mask_from_config(
    config,
    rendering: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    rendered_alpha: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], MaskType]:
    """
    Compute mask based on config settings.

    Args:
        config: Training config with losses section
        rendering: Rendered RGB [b*v, 3, h, w]
        gt_mask: Ground truth mask [b*v, 1, h, w] or None
        rendered_alpha: Rendered alpha [b*v, 1, h, w] or None

    Returns:
        tuple: (mask tensor or None, mask_type enum)
    """
    losses_config = config.training.losses

    use_pred_mask = losses_config.get("use_predicted_mask", False)
    use_alpha_mask = losses_config.get("use_rendered_alpha_mask", False)
    use_masked_loss = losses_config.get("masked_l2_loss", False)

    pred_threshold = losses_config.get("pred_mask_threshold", 0.1)
    alpha_threshold = losses_config.get("alpha_mask_threshold", 0.5)
    bg_color = 1.0  # white background

    # Priority: alpha > rgb_pred > gt > none
    if use_alpha_mask and rendered_alpha is not None:
        mask = (rendered_alpha > alpha_threshold).float()
        return mask, MaskType.ALPHA

    if use_pred_mask:
        color_distance = (rendering - bg_color).abs().mean(dim=1, keepdim=True)
        mask = (color_distance > pred_threshold).float()
        return mask, MaskType.RGB_PRED

    if use_masked_loss and gt_mask is not None:
        return gt_mask, MaskType.GT

    return None, MaskType.NONE


def compute_ghost_metrics(
    rendered_alpha: Optional[torch.Tensor],
    opacity: Optional[torch.Tensor],
) -> Dict[str, float]:
    """
    Compute ghosting-related metrics.

    Args:
        rendered_alpha: Rendered alpha [b, v, 1, h, w]
        opacity: Gaussian opacity tensor

    Returns:
        dict with ghost metrics
    """
    metrics = {}

    if rendered_alpha is not None:
        # Foreground coverage: fraction of pixels with alpha > 0.5
        metrics["ghost_fg_coverage"] = (rendered_alpha > 0.5).float().mean().item()
        # Alpha standard deviation: high value indicates inconsistent alpha
        metrics["ghost_alpha_std"] = rendered_alpha.std().item()

    if opacity is not None:
        # Opacity statistics
        metrics["ghost_opacity_std"] = opacity.std().item()
        metrics["ghost_opacity_mean"] = opacity.mean().item()

    return metrics


def compute_gaussians_usage(opacity: torch.Tensor, threshold: float = 0.05) -> float:
    """
    Compute fraction of Gaussians above opacity threshold.

    Args:
        opacity: Gaussian opacity tensor
        threshold: Minimum opacity to consider "active"

    Returns:
        float: Fraction of active Gaussians
    """
    return (opacity > threshold).float().mean().item()


class LossExtensions:
    """
    Helper class to extend original LossComputer.

    Usage:
        ext = LossExtensions(config)
        mask, mask_type = ext.get_mask(rendering, gt_mask, rendered_alpha)
        ghost_metrics = ext.get_ghost_metrics(rendered_alpha, opacity)
    """

    def __init__(self, config):
        self.config = config
        self._mask_type = MaskType.NONE
        self._alpha_threshold = config.training.losses.get("alpha_mask_threshold", 0.5)
        self._pred_threshold = config.training.losses.get("pred_mask_threshold", 0.1)

    def get_mask(
        self,
        rendering: torch.Tensor,
        gt_mask: Optional[torch.Tensor],
        rendered_alpha: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], MaskType]:
        """Get mask based on config."""
        mask, self._mask_type = compute_mask_from_config(
            self.config, rendering, gt_mask, rendered_alpha
        )
        return mask, self._mask_type

    def get_ghost_metrics(
        self,
        rendered_alpha: Optional[torch.Tensor],
        opacity: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Get ghost metrics."""
        return compute_ghost_metrics(rendered_alpha, opacity)

    @property
    def mask_type(self) -> str:
        """Get current mask type as string."""
        return self._mask_type.value

    @property
    def alpha_threshold(self) -> float:
        """Get alpha threshold."""
        return self._alpha_threshold

    def get_logging_info(self) -> Dict[str, any]:
        """Get info for WandB logging."""
        return {
            "config/mask_type": self._mask_type.value,
            "config/alpha_threshold": self._alpha_threshold,
            "config/pred_threshold": self._pred_threshold,
        }
