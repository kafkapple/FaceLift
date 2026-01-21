"""
Visualization Extensions for Mouse FaceLift - v2.0

Provides modular visualization functions to reduce code duplication in gslrm.py.

Key functions:
- create_error_heatmap: Error visualization with configurable colormap
- create_training_visual: Full training visualization with mask overlays
- create_validation_visual: Validation visualization  
- compute_error_stats: Error statistics calculation

Usage:
    from mouse_extensions.model.visualization_extensions import (
        create_training_visual, VisualizationConfig
    )
    
    config = VisualizationConfig(mask_mode='alpha', alpha_threshold=0.5)
    visual = create_training_visual(target, rendering, rendered_alpha, config)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# Configuration
# =============================================================================

# Import MaskType from loss_extensions (single source of truth)
from .loss_extensions import MaskType

# Backward compatibility alias
VisualizationMaskType = MaskType


@dataclass
class VisualizationConfig:
    """Configuration for visualization functions."""
    mask_mode: str = "none"
    alpha_threshold: float = 0.5
    rgb_threshold: float = 0.1
    error_max: float = 0.3        # Clamp max for error visualization
    overlay_blend: float = 0.3    # Blend ratio for mask overlay
    
    # Colors (green for foreground, red for background)
    fg_color: Tuple[float, float, float] = (0.2, 0.8, 0.2)
    bg_color: Tuple[float, float, float] = (0.8, 0.2, 0.2)
    gray_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    
    @classmethod
    def from_training_config(cls, config) -> "VisualizationConfig":
        """Create VisualizationConfig from training config."""
        losses = config.training.losses
        return cls(
            mask_mode=losses.get("mask_mode", "none"),
            alpha_threshold=losses.get("alpha_mask_threshold", 0.5),
            rgb_threshold=losses.get("pred_mask_threshold", 0.1),
        )


# =============================================================================
# Helper Functions
# =============================================================================

def _get_color_tensor(
    color: Tuple[float, float, float],
    device: torch.device,
    ndim: int = 5
) -> torch.Tensor:
    """Create color tensor with proper shape for broadcasting."""
    shape = [1] * (ndim - 3) + [3, 1, 1]
    return torch.tensor(color, device=device).view(*shape)


def compute_pred_mask(
    rendering: torch.Tensor,
    rendered_alpha: Optional[torch.Tensor],
    config: VisualizationConfig
) -> torch.Tensor:
    """
    Compute predicted mask from rendering.
    
    Args:
        rendering: Rendered RGB [..., 3, H, W]
        rendered_alpha: Rendered alpha [..., 1, H, W] or None
        config: Visualization config
        
    Returns:
        Binary mask [..., 1, H, W]
    """
    if config.mask_mode == "alpha" and rendered_alpha is not None:
        return (rendered_alpha > config.alpha_threshold).float()
    else:
        # RGB-based detection (removebg style)
        color_distance = (rendering - 1.0).abs().mean(dim=-3, keepdim=True)
        return (color_distance > config.rgb_threshold).float()


def _add_camera_labels(
    visual_np: np.ndarray,
    view_indices: List[int],
    image_width: int,
    label_height: int = 20,
    font_size: int = 12
) -> np.ndarray:
    """
    Add camera index labels at the top of each column in the visualization.
    
    Args:
        visual_np: Visualization image (H, W, 3) uint8
        view_indices: List of camera indices for each column
        image_width: Width of each view image
        label_height: Height of label bar
        font_size: Font size for labels
        
    Returns:
        Labeled visualization (H+label_height, W, 3) uint8
    """
    h, w, c = visual_np.shape
    num_views = len(view_indices)
    
    # Create label bar
    label_bar = np.ones((label_height, w, c), dtype=np.uint8) * 40  # Dark gray background
    
    # Convert to PIL for text drawing
    label_img = Image.fromarray(label_bar)
    draw = ImageDraw.Draw(label_img)
    
    # Try to use a monospace font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw camera labels centered on each column
    for i, cam_idx in enumerate(view_indices):
        label_text = f"Cam {cam_idx}"
        
        # Calculate center position for this column
        col_center_x = i * image_width + image_width // 2
        
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = col_center_x - text_width // 2
        text_y = (label_height - text_height) // 2
        
        # Draw text in white
        draw.text((text_x, text_y), label_text, fill=(255, 255, 255), font=font)
    
    # Convert back to numpy and concatenate
    label_bar = np.array(label_img)
    labeled_visual = np.concatenate([label_bar, visual_np], axis=0)
    
    return labeled_visual



def create_mask_overlay(
    image: torch.Tensor,
    mask: torch.Tensor,
    config: VisualizationConfig,
    blend_ratio: Optional[float] = None
) -> torch.Tensor:
    """
    Create image with mask overlay.
    
    Args:
        image: RGB image [..., 3, H, W]
        mask: Binary mask [..., 1, H, W]
        config: Visualization config
        blend_ratio: Override blend ratio (default: config.overlay_blend)
        
    Returns:
        Blended image with mask overlay [..., 3, H, W]
    """
    device = image.device
    ndim = image.dim()
    
    fg = _get_color_tensor(config.fg_color, device, ndim)
    bg = _get_color_tensor(config.bg_color, device, ndim)
    
    mask_rgb = mask.expand(*mask.shape[:-3], 3, *mask.shape[-2:])
    mask_overlay = mask_rgb * fg + (1 - mask_rgb) * bg
    
    blend = blend_ratio if blend_ratio is not None else config.overlay_blend
    return image * (1 - blend) + mask_overlay * blend


def compute_error_stats(
    error: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute error statistics.
    
    Args:
        error: Error tensor [..., 1, H, W]
        mask: Optional foreground mask [..., 1, H, W]
        
    Returns:
        Dict with min, max, mean, and optionally fg_min, fg_max, fg_mean
    """
    stats = {
        'min': error.min().item(),
        'max': error.max().item(),
        'mean': error.mean().item()
    }
    
    if mask is not None:
        mask_flat = mask.view(-1)
        error_flat = error.view(-1)
        fg_errors = error_flat[mask_flat > 0.5]
        
        if fg_errors.numel() > 0:
            stats['fg_min'] = fg_errors.min().item()
            stats['fg_max'] = fg_errors.max().item()
            stats['fg_mean'] = fg_errors.mean().item()
        else:
            stats['fg_min'] = 0.0
            stats['fg_max'] = 0.0
            stats['fg_mean'] = 0.0
    else:
        stats['fg_min'] = stats['min']
        stats['fg_max'] = stats['max']
        stats['fg_mean'] = stats['mean']
    
    return stats


def create_error_heatmap(
    error: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    config: Optional[VisualizationConfig] = None,
    error_max: float = 0.3
) -> torch.Tensor:
    """
    Create error heatmap visualization.
    
    Blue (low) -> Green -> Yellow -> Red (high)
    
    Args:
        error: Error tensor [..., 1, H, W]
        mask: Optional mask [..., 1, H, W], background shown as gray
        config: Visualization config (optional)
        error_max: Maximum error for normalization
        
    Returns:
        RGB heatmap [..., 3, H, W]
    """
    device = error.device
    
    if config is not None:
        error_max = config.error_max
        gray_color = config.gray_color
    else:
        gray_color = (0.3, 0.3, 0.3)
    
    # Apply mask if provided
    if mask is not None:
        masked_error = error * mask
    else:
        masked_error = error
        mask = torch.ones_like(error)
    
    # Normalize to [0, 1]
    error_normalized = (masked_error / error_max).clamp(0, 1)
    
    # Create heatmap colors
    error_r = error_normalized.clamp(0, 1)
    error_g = (1 - error_normalized.abs() * 2).clamp(0, 1)
    error_b = (1 - error_normalized).clamp(0, 1)
    heatmap = torch.cat([error_r, error_g, error_b], dim=-3)
    
    # Gray background where mask=0
    gray = _get_color_tensor(gray_color, device, heatmap.dim())
    heatmap = heatmap * mask + gray * (1 - mask)
    
    return heatmap


# =============================================================================
# Main Visualization Functions
# =============================================================================

def create_training_visual(
    target: torch.Tensor,
    rendering: torch.Tensor,
    config: VisualizationConfig,
    gt_mask: Optional[torch.Tensor] = None,
    rendered_alpha: Optional[torch.Tensor] = None,
    num_views: int = 6,
    view_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Create training visualization.
    
    Output rows:
    - Row 1: GT images (all views)
    - Row 2: Rendered images (all views)
    - Row 3: GT + Mask overlay (if mask_mode != "none")
    - Row 4: Rendered + Pred Mask overlay (if mask_mode != "none")
    - Row 5: Error heatmap
    
    Args:
        target: GT images (B*V, C, H, W) where C=3 or 4 (RGB or RGBA)
        rendering: Rendered images (B*V, 3, H, W)
        config: Visualization config
        gt_mask: Optional GT mask (B*V, 1, H, W)
        rendered_alpha: Optional rendered alpha (B*V, 1, H, W)
        num_views: Number of views per batch
        
    Returns:
        Tuple of (visualization numpy array, error statistics dict)
    """
    from einops import rearrange
    
    device = target.device
    v = num_views
    h = target.size(2)
    
    # Extract GT RGB and mask from target
    if target.size(1) == 4:
        target_rgb = target[:, :3, :, :]
        if gt_mask is None:
            gt_mask = target[:, 3:4, :, :]
    else:
        target_rgb = target
    
    # Reshape to [B, V, C, H, W]
    target_bv = rearrange(target_rgb, "(b v) c h w -> b v c h w", v=v)
    rendering_bv = rearrange(rendering, "(b v) c h w -> b v c h w", v=v)
    
    # Compute error
    error_raw = (target_bv - rendering_bv).abs().mean(dim=2, keepdim=True)
    
    # Build visualization rows
    rows_list = [target_bv, rendering_bv]
    num_rows = 2
    
    show_mask = config.mask_mode != "none" and (gt_mask is not None or rendered_alpha is not None)
    
    if show_mask:
        # Process masks
        if gt_mask is not None:
            gt_mask_bv = rearrange(gt_mask, "(b v) c h w -> b v c h w", v=v)
        else:
            gt_mask_bv = None
        
        if rendered_alpha is not None:
            rendered_alpha_bv = rearrange(rendered_alpha, "(b v) c h w -> b v c h w", v=v)
        else:
            rendered_alpha_bv = None
        
        # Create mask overlays
        if gt_mask_bv is not None:
            masked_target = create_mask_overlay(target_bv, gt_mask_bv, config)
            rows_list.append(masked_target)
            num_rows += 1
        
        # Compute pred mask and create overlay
        pred_mask_bv = compute_pred_mask(rendering_bv, rendered_alpha_bv, config)
        masked_rendering = create_mask_overlay(rendering_bv, pred_mask_bv, config)
        rows_list.append(masked_rendering)
        num_rows += 1
        
        # Compute union mask for error heatmap
        if gt_mask_bv is not None:
            gt_binary = (gt_mask_bv > 0.5).float()
            pred_binary = (pred_mask_bv > 0.5).float()
            union_mask = ((gt_binary + pred_binary) > 0.5).float()
        else:
            union_mask = pred_mask_bv
        
        error_stats = compute_error_stats(error_raw, union_mask)
        error_heatmap = create_error_heatmap(error_raw, union_mask, config)
    else:
        error_stats = compute_error_stats(error_raw)
        error_heatmap = create_error_heatmap(error_raw, config=config)
    
    rows_list.append(error_heatmap)
    num_rows += 1
    
    # Stack and rearrange
    visual = torch.stack(rows_list, dim=1)
    visual = rearrange(visual, "b rows v c h w -> (b rows h) (v w) c")
    
    # Convert to numpy
    visual_np = (visual.detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    # Add camera labels if view_indices provided
    if view_indices is not None:
        image_width = target.size(3)  # W dimension
        visual_np = _add_camera_labels(visual_np, view_indices, image_width)
    
    return visual_np, error_stats


def create_validation_visual(
    target: torch.Tensor,
    rendering: torch.Tensor,
    config: VisualizationConfig,
    rendered_alpha: Optional[torch.Tensor] = None,
    max_views: int = 10,
    view_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Create validation visualization.
    
    Output rows (if mask_mode != "none" and has mask):
    - Row 1: GT images
    - Row 2: Rendered images  
    - Row 3: GT + Mask overlay
    - Row 4: Rendered + Pred Mask overlay
    - Row 5: Error heatmap
    
    Output rows (no mask):
    - Row 1: GT images
    - Row 2: Rendered images
    - Row 3: Error heatmap
    
    Args:
        target: GT images (V, C, H, W) where C=3 or 4
        rendering: Rendered images (V, 3, H, W)
        config: Visualization config
        rendered_alpha: Optional rendered alpha (V, 1, H, W)
        max_views: Maximum views to show (subsamples if exceeded)
        
    Returns:
        Tuple of (visualization numpy array, error statistics dict)
    """
    from einops import rearrange
    
    device = target.device
    num_views = target.size(0)
    
    # Subsample if too many views
    if num_views > max_views:
        step = num_views // max_views
        target = target[::step]
        rendering = rendering[::step]
        if rendered_alpha is not None:
            rendered_alpha = rendered_alpha[::step]
        num_views = target.size(0)
    
    # Extract GT RGB and mask
    if target.size(1) == 4:
        gt_rgb = target[:, :3, :, :]
        gt_mask = target[:, 3:4, :, :]
    else:
        gt_rgb = target
        gt_mask = None
    
    # Compute error
    error_raw = (gt_rgb - rendering).abs().mean(dim=1, keepdim=True)
    
    # Build rows
    rows_list = [gt_rgb, rendering]
    
    show_mask = config.mask_mode != "none" and gt_mask is not None
    
    if show_mask:
        # GT mask overlay
        masked_gt = create_mask_overlay(gt_rgb, gt_mask, config)
        rows_list.append(masked_gt)
        
        # Pred mask overlay
        pred_mask = compute_pred_mask(rendering, rendered_alpha, config)
        masked_rendering = create_mask_overlay(rendering, pred_mask, config)
        rows_list.append(masked_rendering)
        
        # Error with union mask
        gt_binary = (gt_mask > 0.5).float()
        pred_binary = (pred_mask > 0.5).float()
        union_mask = ((gt_binary + pred_binary) > 0.5).float()
        
        error_stats = compute_error_stats(error_raw, union_mask)
        error_heatmap = create_error_heatmap(error_raw, union_mask, config)
    else:
        error_stats = compute_error_stats(error_raw)
        error_heatmap = create_error_heatmap(error_raw, config=config)
    
    rows_list.append(error_heatmap)
    
    # Stack: [num_rows, V, C, H, W]
    visual = torch.stack(rows_list, dim=0)
    
    # Rearrange to image
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")
    
    # Convert to numpy
    visual_np = (visual.detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    
    # Add camera labels if view_indices provided
    if view_indices is not None:
        # Handle subsampling case
        if num_views < len(view_indices):
            step = len(view_indices) // num_views
            view_indices = [view_indices[i] for i in range(0, len(view_indices), step)][:num_views]
        image_width = target.size(3)  # W dimension after potential subsampling
        visual_np = _add_camera_labels(visual_np, view_indices, image_width)
    
    return visual_np, error_stats


# =============================================================================
# Threshold Comparison (kept from original)
# =============================================================================

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
        rendering: Rendered RGB [V, 3, H, W]
        rendered_alpha: Rendered alpha [V, 1, H, W] or None
        gt_mask: Ground truth mask [V, 1, H, W] or None
        alpha_thresholds: List of alpha thresholds to visualize
        rgb_thresholds: List of RGB thresholds to visualize
        
    Returns:
        Numpy array visualization grid
    """
    device = rendering.device
    v, _, h, w = rendering.shape
    
    config = VisualizationConfig()
    
    rows = []
    
    # Row 1: Alpha threshold comparison (first view only)
    if rendered_alpha is not None:
        alpha_row = []
        for thresh in alpha_thresholds:
            mask = (rendered_alpha[0:1] > thresh).float()
            overlay = create_mask_overlay(rendering[0:1], mask, config, blend_ratio=0.5)
            alpha_row.append(overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        rows.append(np.hstack(alpha_row))
    
    # Row 2: RGB threshold comparison
    rgb_row = []
    for thresh in rgb_thresholds:
        color_dist = (rendering[0:1] - 1.0).abs().mean(dim=1, keepdim=True)
        mask = (color_dist > thresh).float()
        overlay = create_mask_overlay(rendering[0:1], mask, config, blend_ratio=0.5)
        rgb_row.append(overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
    rows.append(np.hstack(rgb_row))
    
    # Row 3: GT vs Alpha vs RGB vs Diff
    if gt_mask is not None and rendered_alpha is not None:
        comparison_row = []
        
        # GT mask
        gt_overlay = create_mask_overlay(rendering[0:1], gt_mask[0:1], config, blend_ratio=0.5)
        comparison_row.append(gt_overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        # Alpha mask (0.5)
        alpha_mask = (rendered_alpha[0:1] > 0.5).float()
        alpha_overlay = create_mask_overlay(rendering[0:1], alpha_mask, config, blend_ratio=0.5)
        comparison_row.append(alpha_overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        # RGB mask (0.1)
        rgb_dist = (rendering[0:1] - 1.0).abs().mean(dim=1, keepdim=True)
        rgb_mask = (rgb_dist > 0.1).float()
        rgb_overlay = create_mask_overlay(rendering[0:1], rgb_mask, config, blend_ratio=0.5)
        comparison_row.append(rgb_overlay.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        # Diff visualization
        alpha_binary = (alpha_mask > 0.5)
        gt_binary = (gt_mask[0:1] > 0.5)
        
        alpha_only = alpha_binary & (~gt_binary)
        gt_only = gt_binary & (~alpha_binary)
        both = alpha_binary & gt_binary
        
        diff_img = torch.zeros(1, 3, h, w, device=device)
        diff_img[:, 0:1] = gt_only.float()      # Red = GT only
        diff_img[:, 2:3] = alpha_only.float()   # Blue = Alpha only
        diff_img[:, 1:2] = both.float() * 0.5   # Green = both
        comparison_row.append(diff_img.squeeze(0).permute(1, 2, 0).cpu().numpy())
        
        rows.append(np.hstack(comparison_row))
    
    if len(rows) == 0:
        return None
    
    # Ensure all rows have same width
    max_width = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_width:
            pad = np.ones((r.shape[0], max_width - r.shape[1], 3)) * 0.5
            r = np.hstack([r, pad])
        padded_rows.append(r)
    
    result = np.vstack(padded_rows)
    return (result * 255).clip(0, 255).astype(np.uint8)


# =============================================================================
# Legacy compatibility functions
# =============================================================================

def compute_visualization_mask(
    rendering: torch.Tensor,
    rendered_alpha: Optional[torch.Tensor],
    gt_mask: Optional[torch.Tensor],
    config,
    force_type: Optional[str] = None,
) -> Tuple[torch.Tensor, VisualizationMaskType]:
    """
    Legacy function for backward compatibility.
    
    Use compute_pred_mask() for new code.
    """
    viz_config = VisualizationConfig.from_training_config(config)
    
    if force_type == "alpha" and rendered_alpha is not None:
        mask = (rendered_alpha > viz_config.alpha_threshold).float()
        return mask, VisualizationMaskType.ALPHA
    elif force_type == "rgb":
        color_distance = (rendering - 1.0).abs().mean(dim=-3, keepdim=True)
        mask = (color_distance > viz_config.rgb_threshold).float()
        return mask, VisualizationMaskType.RGB
    elif force_type == "gt" and gt_mask is not None:
        return gt_mask, VisualizationMaskType.GT
    
    # Auto-detect based on config
    if viz_config.mask_mode == "alpha" and rendered_alpha is not None:
        mask = (rendered_alpha > viz_config.alpha_threshold).float()
        return mask, VisualizationMaskType.ALPHA
    
    # Default: RGB threshold
    color_distance = (rendering - 1.0).abs().mean(dim=-3, keepdim=True)
    mask = (color_distance > viz_config.rgb_threshold).float()
    return mask, VisualizationMaskType.RGB


def get_visualization_config_info(config) -> Dict[str, any]:
    """Get visualization-relevant config for logging."""
    losses = config.training.losses
    return {
        "viz_config/mask_mode": losses.get("mask_mode", "none"),
        "viz_config/alpha_threshold": losses.get("alpha_mask_threshold", 0.5),
        "viz_config/rgb_threshold": losses.get("pred_mask_threshold", 0.1),
        "viz_config/masked_l2_loss": losses.get("masked_l2_loss", False),
    }
