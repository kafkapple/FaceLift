"""
Ghost Gaussian Pruning Extensions

Provides utilities to identify and remove "ghost" Gaussians that extend 
beyond the actual object boundary.
"""

import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PruningConfig:
    """Configuration for Gaussian pruning."""
    opacity_threshold: float = 0.05      # Remove Gaussians with opacity below this
    alpha_threshold: float = 0.3         # Consider pixels with alpha > this as object
    boundary_margin: float = 0.1         # Margin around object boundary
    min_opacity_mean: float = 0.1        # Minimum mean opacity after pruning
    

def compute_ghost_gaussian_mask(
    gaussian_xyz: torch.Tensor,
    rendered_alpha: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    camera_params: Dict,
    config: PruningConfig,
) -> torch.Tensor:
    """
    Compute mask identifying ghost Gaussians.
    
    Ghost Gaussians are defined as:
    1. Low opacity (< threshold)
    2. Outside the object boundary defined by GT mask or rendered alpha
    3. Contributing to alpha "bleed" beyond the object
    
    Args:
        gaussian_xyz: Gaussian positions [N, 3]
        rendered_alpha: Rendered alpha map [H, W] or [B, H, W]
        gt_mask: Ground truth mask [H, W] or None
        camera_params: Camera parameters for projection
        config: Pruning configuration
        
    Returns:
        Boolean mask [N] where True = keep, False = prune
    """
    # This is a placeholder for actual implementation
    # Full implementation would require:
    # 1. Project Gaussians to 2D
    # 2. Check if projection falls within GT mask
    # 3. Check opacity values
    
    # For now, return all True (no pruning)
    return torch.ones(gaussian_xyz.shape[0], dtype=torch.bool, device=gaussian_xyz.device)


def compute_boundary_gaussians(
    rendered_alpha: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    alpha_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics about Gaussians near/beyond object boundary.
    
    Args:
        rendered_alpha: Rendered alpha [B, 1, H, W]
        gt_mask: Ground truth mask [B, 1, H, W] or None
        alpha_threshold: Threshold for object/background decision
        
    Returns:
        Dict with boundary metrics
    """
    metrics = {}
    
    if gt_mask is None:
        return metrics
    
    # Binary masks
    pred_fg = (rendered_alpha > alpha_threshold).float()
    gt_fg = (gt_mask > 0.5).float()
    
    # Compute areas
    pred_area = pred_fg.sum()
    gt_area = gt_fg.sum()
    
    # Ghost area: predicted foreground but GT background
    ghost_area = (pred_fg * (1 - gt_fg)).sum()
    
    # Missing area: GT foreground but predicted background
    missing_area = ((1 - pred_fg) * gt_fg).sum()
    
    # Overlap (IoU)
    intersection = (pred_fg * gt_fg).sum()
    union = ((pred_fg + gt_fg) > 0.5).float().sum()
    iou = intersection / union.clamp(min=1)
    
    metrics["boundary/ghost_area"] = ghost_area.item()
    metrics["boundary/ghost_ratio"] = (ghost_area / gt_area.clamp(min=1)).item()
    metrics["boundary/missing_area"] = missing_area.item()
    metrics["boundary/missing_ratio"] = (missing_area / gt_area.clamp(min=1)).item()
    metrics["boundary/iou"] = iou.item()
    metrics["boundary/pred_area_ratio"] = (pred_area / gt_area.clamp(min=1)).item()
    
    return metrics


class GhostGaussianRegularizer:
    """
    Regularization loss to penalize ghost Gaussians.
    
    This adds a loss term that penalizes:
    1. Gaussians with rendered alpha outside GT mask region
    2. Gaussians with very low opacity (likely floaters)
    
    Config options:
        training.losses.ghost_reg_weight: float (default 0.0, disabled)
        training.losses.ghost_alpha_threshold: float (default 0.3)
    """
    
    def __init__(self, config):
        self.config = config
        losses = config.training.losses
        
        self.weight = losses.get("ghost_reg_weight", 0.0)
        self.alpha_threshold = losses.get("ghost_alpha_threshold", 0.3)
        self.enabled = self.weight > 0.0
        
        if self.enabled:
            print(f"[GhostGaussianRegularizer] Enabled: weight={self.weight}")
    
    def __call__(
        self,
        rendered_alpha: torch.Tensor,
        gt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute ghost regularization loss.
        
        Args:
            rendered_alpha: Rendered alpha [B, V, 1, H, W]
            gt_mask: Ground truth mask [B, V, 1, H, W] or None
            
        Returns:
            Scalar loss tensor
        """
        if not self.enabled or gt_mask is None:
            device = rendered_alpha.device if rendered_alpha is not None else 'cpu'
            return torch.tensor(0.0, device=device)
        
        # Ghost area: rendered alpha where GT says background
        gt_bg = (gt_mask < 0.5).float()
        ghost_alpha = rendered_alpha * gt_bg
        
        # Penalize alpha values above threshold in background region
        ghost_penalty = torch.relu(ghost_alpha - self.alpha_threshold)
        
        return self.weight * ghost_penalty.mean()
    
    def get_metrics(
        self,
        rendered_alpha: torch.Tensor,
        gt_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """Get ghost-related metrics."""
        return compute_boundary_gaussians(rendered_alpha, gt_mask, self.alpha_threshold)
