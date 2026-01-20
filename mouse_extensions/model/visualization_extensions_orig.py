"""
Visualization Extensions for Mouse FaceLift

Provides config-aware mask visualization that respects loss computation settings.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from enum import Enum


class VisualizationMaskType(Enum):
    """Mask types for visualization."""
    RGB = "rgb"       # RGB distance threshold (removebg style)
    ALPHA = "alpha"   # Rendered alpha threshold
    GT = "gt"         # Ground truth mask


def compute_visualization_mask(
    rendering: torch.Tensor,
    rendered_alpha: Optional[torch.Tensor],
    gt_mask: Optional[torch.Tensor],
    config,
    force_type: Optional[str] = None,
) -> Tuple[torch.Tensor, VisualizationMaskType]:
    """
    Compute mask for visualization based on config settings.
    
    Mirrors the logic in loss computation to ensure visualization 
    matches what the model is actually learning.
    
    Args:
        rendering: Rendered RGB [b, v, 3, h, w] or [b*v, 3, h, w]
        rendered_alpha: Rendered alpha [b, v, 1, h, w] or [b*v, 1, h, w] or None
        gt_mask: Ground truth mask or None
        config: Training config
        force_type: Force specific mask type ('rgb', 'alpha', 'gt')
        
    Returns:
        tuple: (mask tensor, mask_type enum)
    """
    losses_config = config.training.losses
    
    # Get threshold values from config
    rgb_threshold = losses_config.get("pred_mask_threshold", 0.1)
    alpha_threshold = losses_config.get("alpha_mask_threshold", 0.5)
    
    # Check config settings
    use_alpha_mask = losses_config.get("use_rendered_alpha_mask", False)
    use_pred_mask = losses_config.get("use_predicted_mask", False)
    
    # Force specific type if requested
    if force_type == "alpha" and rendered_alpha is not None:
        mask = (rendered_alpha > alpha_threshold).float()
        return mask, VisualizationMaskType.ALPHA
    elif force_type == "rgb":
        bg_color = 1.0
        color_distance = (rendering - bg_color).abs().mean(dim=-3, keepdim=True)
        mask = (color_distance > rgb_threshold).float()
        return mask, VisualizationMaskType.RGB
    elif force_type == "gt" and gt_mask is not None:
        return gt_mask, VisualizationMaskType.GT
    
    # Follow same priority as loss computation: alpha > rgb_pred > gt
    if use_alpha_mask and rendered_alpha is not None:
        mask = (rendered_alpha > alpha_threshold).float()
        return mask, VisualizationMaskType.ALPHA
    
    if use_pred_mask:
        bg_color = 1.0
        color_distance = (rendering - bg_color).abs().mean(dim=-3, keepdim=True)
        mask = (color_distance > rgb_threshold).float()
        return mask, VisualizationMaskType.RGB
    
    # Default: use RGB threshold (legacy behavior)
    bg_color = 1.0
    color_distance = (rendering - bg_color).abs().mean(dim=-3, keepdim=True)
    mask = (color_distance > rgb_threshold).float()
    return mask, VisualizationMaskType.RGB


def create_mask_overlay(
    image: torch.Tensor,
    mask: torch.Tensor,
    fg_color: Tuple[float, float, float] = (0.2, 0.8, 0.2),
    bg_color: Tuple[float, float, float] = (0.8, 0.2, 0.2),
    blend_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Create image with mask overlay.
    
    Args:
        image: RGB image [*, 3, h, w]
        mask: Binary mask [*, 1, h, w]
        fg_color: Foreground tint color (green)
        bg_color: Background tint color (red)
        blend_ratio: How much to blend mask color
        
    Returns:
        Blended image with mask overlay
    """
    device = image.device
    fg = torch.tensor(fg_color, device=device).view(*([1] * (image.dim() - 3)), 3, 1, 1)
    bg = torch.tensor(bg_color, device=device).view(*([1] * (image.dim() - 3)), 3, 1, 1)
    
    mask_rgb = mask.expand(*mask.shape[:-3], 3, *mask.shape[-2:])
    mask_overlay = mask_rgb * fg + (1 - mask_rgb) * bg
    
    return image * (1 - blend_ratio) + mask_overlay * blend_ratio


def create_threshold_comparison_grid(
    rendering: torch.Tensor,
    rendered_alpha: Optional[torch.Tensor],
    gt_mask: Optional[torch.Tensor],
    alpha_thresholds: list = [0.1, 0.3, 0.5, 0.7],
    rgb_thresholds: list = [0.05, 0.1, 0.2, 0.3],
) -> np.ndarray:
    """
    Create comparison grid showing masks at different thresholds.
    
    Args:
        rendering: Rendered RGB [v, 3, h, w]
        rendered_alpha: Rendered alpha [v, 1, h, w] or None
        gt_mask: Ground truth mask [v, 1, h, w] or None
        alpha_thresholds: List of alpha thresholds to visualize
        rgb_thresholds: List of RGB thresholds to visualize
        
    Returns:
        Numpy array visualization grid
    """
    device = rendering.device
    v, _, h, w = rendering.shape
    
    fg_color = (0.2, 0.8, 0.2)
    bg_color = (0.8, 0.2, 0.2)
    
    rows = []
    
    # Row 1: Alpha threshold comparison (first view only)
    if rendered_alpha is not None:
        alpha_row = []
        for thresh in alpha_thresholds:
            mask = (rendered_alpha[0:1] > thresh).float()
            overlay = create_mask_overlay(rendering[0:1], mask, fg_color, bg_color, 0.5)
            alpha_row.append(overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        rows.append(np.hstack(alpha_row))
    
    # Row 2: RGB threshold comparison
    rgb_row = []
    for thresh in rgb_thresholds:
        color_dist = (rendering[0:1] - 1.0).abs().mean(dim=1, keepdim=True)
        mask = (color_dist > thresh).float()
        overlay = create_mask_overlay(rendering[0:1], mask, fg_color, bg_color, 0.5)
        rgb_row.append(overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
    rows.append(np.hstack(rgb_row))
    
    # Row 3: GT vs Best comparisons (if GT available)
    if gt_mask is not None and rendered_alpha is not None:
        comparison_row = []
        
        # GT mask
        gt_overlay = create_mask_overlay(rendering[0:1], gt_mask[0:1], fg_color, bg_color, 0.5)
        comparison_row.append(gt_overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        # Alpha mask (0.5)
        alpha_mask = (rendered_alpha[0:1] > 0.5).float()
        alpha_overlay = create_mask_overlay(rendering[0:1], alpha_mask, fg_color, bg_color, 0.5)
        comparison_row.append(alpha_overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        # RGB mask (0.1)
        rgb_dist = (rendering[0:1] - 1.0).abs().mean(dim=1, keepdim=True)
        rgb_mask = (rgb_dist > 0.1).float()
        rgb_overlay = create_mask_overlay(rendering[0:1], rgb_mask, fg_color, bg_color, 0.5)
        comparison_row.append(rgb_overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        # Difference visualization (blue=alpha only, red=gt only, green=both)
        alpha_only = (alpha_mask > 0.5) & (gt_mask[0:1] < 0.5)
        gt_only = (gt_mask[0:1] > 0.5) & (alpha_mask < 0.5)
        both = (alpha_mask > 0.5) & (gt_mask[0:1] > 0.5)
        
        diff_img = torch.zeros(1, 3, h, w, device=device)
        diff_img[:, 0:1] = gt_only.float()      # Red = GT only
        diff_img[:, 2:3] = alpha_only.float()   # Blue = Alpha only
        diff_img[:, 1:2] = both.float() * 0.5   # Green = both
        comparison_row.append(diff_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        rows.append(np.hstack(comparison_row))
    
    # Stack all rows
    result = np.vstack(rows)
    return (result * 255).clip(0, 255).astype(np.uint8)


def get_visualization_config_info(config) -> Dict[str, any]:
    """
    Get visualization-relevant config for logging.
    
    Args:
        config: Training config
        
    Returns:
        Dict with config info for WandB logging
    """
    losses = config.training.losses
    return {
        "viz_config/use_alpha_mask": losses.get("use_rendered_alpha_mask", False),
        "viz_config/use_pred_mask": losses.get("use_predicted_mask", False),
        "viz_config/alpha_threshold": losses.get("alpha_mask_threshold", 0.5),
        "viz_config/rgb_threshold": losses.get("pred_mask_threshold", 0.1),
        "viz_config/masked_l2_loss": losses.get("masked_l2_loss", False),
    }
