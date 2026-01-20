"""
GSLRM Patches for Mouse Extensions

This module provides patches to integrate alpha loss and config-aware visualization
into the main GSLRM model with minimal changes to the original code.

Usage:
    1. Import this module in gslrm/model/gslrm.py
    2. Call patch functions at appropriate points
    
Or use apply_patches.py to automatically inject these changes.
"""

import torch
from typing import Dict, Optional, Tuple
from .loss_extensions import AlphaLossComputer, compute_alpha_loss
from .visualization_extensions import compute_visualization_mask, VisualizationMaskType


def compute_alpha_loss_if_enabled(
    config,
    rendered_alpha: torch.Tensor,
    gt_alpha: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, bool]:
    """
    Compute alpha loss if enabled in config.
    
    This should be called in _compute_all_losses and the result added to total loss.
    
    Args:
        config: Training config
        rendered_alpha: Rendered alpha [B*V, 1, H, W]
        gt_alpha: Ground truth alpha (from mask) [B*V, 1, H, W] or None
        
    Returns:
        tuple: (loss_value, is_enabled)
    """
    weight = config.training.losses.get("alpha_loss_weight", 0.0)
    
    if weight <= 0.0 or rendered_alpha is None or gt_alpha is None:
        device = rendered_alpha.device if rendered_alpha is not None else 'cpu'
        return torch.tensor(0.0, device=device), False
    
    loss_type = config.training.losses.get("alpha_loss_type", "bce")
    focal_gamma = config.training.losses.get("alpha_focal_gamma", 2.0)
    
    loss = compute_alpha_loss(rendered_alpha, gt_alpha, loss_type, focal_gamma)
    
    return weight * loss, True


def get_visualization_mask_for_display(
    config,
    rendering: torch.Tensor,
    rendered_alpha: Optional[torch.Tensor],
    gt_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, str]:
    """
    Get mask for visualization that matches loss computation settings.
    
    Use this instead of hardcoded `(color_distance > 0.1).float()` in visualization code.
    
    Args:
        config: Training config
        rendering: Rendered RGB [*, 3, H, W]
        rendered_alpha: Rendered alpha [*, 1, H, W] or None
        gt_mask: Ground truth mask or None
        
    Returns:
        tuple: (mask tensor, description string)
    """
    mask, mask_type = compute_visualization_mask(
        rendering, rendered_alpha, gt_mask, config
    )
    
    # Get threshold used
    if mask_type == VisualizationMaskType.ALPHA:
        thresh = config.training.losses.get("alpha_mask_threshold", 0.5)
        desc = f"Alpha > {thresh}"
    elif mask_type == VisualizationMaskType.RGB:
        thresh = config.training.losses.get("pred_mask_threshold", 0.1)
        desc = f"RGB > {thresh}"
    else:
        desc = "GT Mask"
    
    return mask, desc


def create_config_aware_visual(
    config,
    rendering: torch.Tensor,
    target: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    rendered_alpha: Optional[torch.Tensor],
    v: int,
) -> torch.Tensor:
    """
    Create visualization with config-aware mask display.
    
    This is a drop-in replacement for parts of _create_visual that handle mask overlay.
    
    Args:
        config: Training config
        rendering: Rendered images [B*V, 3, H, W]
        target: Target images [B*V, 3, H, W]
        gt_mask: Ground truth mask [B*V, 1, H, W] or None
        rendered_alpha: Rendered alpha [B*V, 1, H, W] or None
        v: Number of views
        
    Returns:
        Visualization tensor
    """
    from einops import rearrange
    import numpy as np
    
    b = target.size(0) // v
    h = target.size(2)
    device = target.device
    
    # Reshape to [b, v, c, h, w]
    target_bv = rearrange(target, "(b v) c h w -> b v c h w", v=v)
    rendering_bv = rearrange(rendering, "(b v) c h w -> b v c h w", v=v)
    
    # Get pred mask based on config
    pred_mask, mask_desc = get_visualization_mask_for_display(
        config, rendering_bv, 
        rearrange(rendered_alpha, "(b v) c h w -> b v c h w", v=v) if rendered_alpha is not None else None,
        rearrange(gt_mask, "(b v) c h w -> b v c h w", v=v) if gt_mask is not None else None,
    )
    
    # Create overlays
    fg_color = torch.tensor([0.2, 0.8, 0.2], device=device).view(1, 1, 3, 1, 1)
    bg_color = torch.tensor([0.8, 0.2, 0.2], device=device).view(1, 1, 3, 1, 1)
    
    if gt_mask is not None:
        gt_mask_bv = rearrange(gt_mask, "(b v) c h w -> b v c h w", v=v)
        gt_mask_rgb = gt_mask_bv.expand(-1, -1, 3, -1, -1)
        gt_mask_overlay = gt_mask_rgb * fg_color + (1 - gt_mask_rgb) * bg_color
        masked_target = target_bv * 0.7 + gt_mask_overlay * 0.3
    else:
        masked_target = target_bv
    
    pred_mask_rgb = pred_mask.expand(-1, -1, 3, -1, -1)
    pred_mask_overlay = pred_mask_rgb * fg_color + (1 - pred_mask_rgb) * bg_color
    masked_rendering = rendering_bv * 0.7 + pred_mask_overlay * 0.3
    
    # Error map
    error = (target_bv - rendering_bv).abs().mean(dim=2, keepdim=True)
    error_normalized = (error / 0.3).clamp(0, 1)
    error_r = error_normalized
    error_g = (1 - error_normalized * 2).clamp(0, 1)
    error_b = (1 - error_normalized)
    error_heatmap = torch.cat([error_r, error_g, error_b], dim=2)
    
    # Stack rows
    visual = torch.stack([
        target_bv,
        rendering_bv,
        masked_target,
        masked_rendering,
        error_heatmap
    ], dim=1)
    
    visual = rearrange(visual, "b rows v c h w -> (b rows h) (v w) c")
    
    return visual, mask_desc


# Logging helper for config info
def get_mask_config_for_logging(config) -> Dict[str, any]:
    """
    Get mask configuration summary for WandB logging.
    
    Call this at training start to log the mask configuration being used.
    """
    losses = config.training.losses
    
    use_alpha = losses.get("use_rendered_alpha_mask", False)
    use_pred = losses.get("use_predicted_mask", False)
    use_masked_loss = losses.get("masked_l2_loss", False)
    
    # Determine effective mask type
    if use_alpha:
        mask_type = "alpha"
        threshold = losses.get("alpha_mask_threshold", 0.5)
    elif use_pred:
        mask_type = "rgb_pred"
        threshold = losses.get("pred_mask_threshold", 0.1)
    elif use_masked_loss:
        mask_type = "gt"
        threshold = 0.5
    else:
        mask_type = "none"
        threshold = None
    
    return {
        "config/mask_type": mask_type,
        "config/mask_threshold": threshold,
        "config/use_alpha_mask": use_alpha,
        "config/use_pred_mask": use_pred,
        "config/masked_l2_loss": use_masked_loss,
        "config/alpha_loss_weight": losses.get("alpha_loss_weight", 0.0),
        "config/background_loss_weight": losses.get("background_loss_weight", 0.0),
    }
