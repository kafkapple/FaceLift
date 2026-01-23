"""
Loss Extensions

Mask computation options and ghost metrics for loss calculation.
"""

import torch
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


class MaskType(Enum):
    """Mask types for loss computation and visualization."""
    NONE = "none"           # Full image loss (no mask)
    GT = "gt"               # Ground truth mask from alpha channel
    RGB_PRED = "rgb_pred"   # Predicted from RGB distance to background
    RGB = "rgb"             # Alias for RGB_PRED (backward compat with visualization)
    ALPHA = "alpha"         # Rendered alpha from Gaussian splatting


class AlphaMaskSafetyMode(Enum):
    """
    Safety modes for alpha mask usage without alpha supervision.

    When mask_mode=alpha is used without alpha_loss_weight > 0,
    the mask can expand during training causing ghosting artifacts.

    Modes:
        WARN: Allow but print warning (default, backward compatible)
        STRICT: Raise error, require alpha_loss > 0
        FALLBACK_GT: Fall back to GT mask if available
        FALLBACK_NONE: Fall back to no masking
    """
    WARN = "warn"
    STRICT = "strict"
    FALLBACK_GT = "fallback_gt"
    FALLBACK_NONE = "fallback_none"


# Global flag to prevent repeated warnings
_ALPHA_MASK_WARNING_SHOWN = False


def reset_alpha_mask_warning():
    """Reset the alpha mask warning flag (useful for testing)."""
    global _ALPHA_MASK_WARNING_SHOWN
    _ALPHA_MASK_WARNING_SHOWN = False


def validate_mask_config(config) -> list:
    """
    Validate mask configuration and return list of warnings.

    Args:
        config: Training config with losses section

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []
    losses = config.training.losses

    mask_mode = losses.get("mask_mode", None)
    alpha_loss_weight = losses.get("alpha_loss_weight", 0.0)

    # Check alpha mask without supervision
    if mask_mode == "alpha" and alpha_loss_weight <= 0.0:
        safety = losses.get("alpha_mask_safety", "warn")
        if safety == "warn":
            warnings.append(
                "mask_mode=alpha without alpha_loss_weight may cause mask spreading. "
                "Recommended: set alpha_loss_weight > 0 or use mask_mode=gt"
            )

    # Check rgb_pred with low threshold
    if mask_mode == "rgb_pred":
        threshold = losses.get("pred_mask_threshold", 0.1)
        if threshold < 0.05:
            warnings.append(
                f"pred_mask_threshold={threshold} is very low, may include background noise"
            )

    # Check masked perceptual/ssim (non-original behavior)
    if losses.get("masked_perceptual_loss", False):
        warnings.append(
            "masked_perceptual_loss=True differs from original GS-LRM/FaceLift behavior"
        )
    if losses.get("masked_ssim_loss", False):
        warnings.append(
            "masked_ssim_loss=True differs from original GS-LRM/FaceLift behavior"
        )

    return warnings


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

    Priority: mask_mode (explicit) > use_rendered_alpha_mask > use_predicted_mask > gt > none
    
    Args:
        config: Training config with losses section
        rendering: Rendered RGB [b*v, 3, h, w]
        gt_mask: Ground truth mask [b*v, 1, h, w] or None
        rendered_alpha: Rendered alpha [b*v, 1, h, w] or None

    Returns:
        tuple: (mask tensor or None, mask_type enum)
    """
    losses_config = config.training.losses

    # NEW: Check explicit mask_mode first
    mask_mode = losses_config.get("mask_mode", None)
    
    if mask_mode is not None:
        # Debug: only log once per session (controlled by env var DEBUG_MASK)
        pass  # Use DEBUG_MASK=1 env var to enable debug logging
        # Explicit mask_mode takes priority
        alpha_threshold = losses_config.get("alpha_mask_threshold", 0.5)
        pred_threshold = losses_config.get("pred_mask_threshold", 0.1)
        min_mask_ratio = losses_config.get("min_mask_ratio", 0.0)
        bg_color = 1.0  # white background
        
        if mask_mode == "none":
            return None, MaskType.NONE
        
        elif mask_mode == "gt":
            if gt_mask is not None:
                return gt_mask, MaskType.GT
            else:
                print("[WARNING] mask_mode=gt but gt_mask is None, falling back to NONE")
                return None, MaskType.NONE
        
        elif mask_mode == "alpha":
            if rendered_alpha is not None:
                # Safety check: alpha mask without alpha supervision can cause spreading
                alpha_loss_weight = losses_config.get("alpha_loss_weight", 0.0)
                safety_mode_str = losses_config.get("alpha_mask_safety", "warn")

                try:
                    safety_mode = AlphaMaskSafetyMode(safety_mode_str)
                except ValueError:
                    safety_mode = AlphaMaskSafetyMode.WARN

                if alpha_loss_weight <= 0.0:
                    global _ALPHA_MASK_WARNING_SHOWN

                    if safety_mode == AlphaMaskSafetyMode.STRICT:
                        raise ValueError(
                            "mask_mode=alpha requires alpha_loss_weight > 0 to prevent mask spreading. "
                            "Set alpha_loss_weight > 0 or change alpha_mask_safety to 'warn'/'fallback_gt'/'fallback_none'."
                        )

                    elif safety_mode == AlphaMaskSafetyMode.FALLBACK_GT:
                        if gt_mask is not None:
                            if not _ALPHA_MASK_WARNING_SHOWN:
                                print("[WARNING] mask_mode=alpha without alpha_loss, falling back to GT mask")
                                _ALPHA_MASK_WARNING_SHOWN = True
                            return gt_mask, MaskType.GT
                        else:
                            if not _ALPHA_MASK_WARNING_SHOWN:
                                print("[WARNING] mask_mode=alpha without alpha_loss, GT unavailable, using NONE")
                                _ALPHA_MASK_WARNING_SHOWN = True
                            return None, MaskType.NONE

                    elif safety_mode == AlphaMaskSafetyMode.FALLBACK_NONE:
                        if not _ALPHA_MASK_WARNING_SHOWN:
                            print("[WARNING] mask_mode=alpha without alpha_loss, falling back to NONE")
                            _ALPHA_MASK_WARNING_SHOWN = True
                        return None, MaskType.NONE

                    else:  # WARN (default)
                        if not _ALPHA_MASK_WARNING_SHOWN:
                            print(
                                "[WARNING] mask_mode=alpha without alpha_loss_weight may cause mask spreading/ghosting. "
                                "Consider setting alpha_loss_weight > 0 or using mask_mode=gt for stability."
                            )
                            _ALPHA_MASK_WARNING_SHOWN = True

                # Proceed with alpha mask
                mask = (rendered_alpha > alpha_threshold).float()
                # Safety: ensure minimum mask ratio
                if min_mask_ratio > 0:
                    mask_ratio = mask.mean()
                    if mask_ratio < min_mask_ratio:
                        # Lower threshold to include more pixels
                        sorted_alpha = rendered_alpha.flatten().sort(descending=True).values
                        new_threshold = sorted_alpha[int(min_mask_ratio * len(sorted_alpha))].item()
                        mask = (rendered_alpha > new_threshold).float()
                return mask, MaskType.ALPHA
            else:
                print("[WARNING] mask_mode=alpha but rendered_alpha is None, falling back to NONE")
                return None, MaskType.NONE
        
        elif mask_mode == "rgb_pred":
            color_distance = (rendering - bg_color).abs().mean(dim=1, keepdim=True)
            mask = (color_distance > pred_threshold).float()
            return mask, MaskType.RGB_PRED
        
        else:
            print(f"[WARNING] Unknown mask_mode: {mask_mode}, falling back to NONE")
            return None, MaskType.NONE
    
    # Fallback: legacy config support (use_rendered_alpha_mask, etc.)
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


# =============================================================================
# Alpha Loss Implementation
# =============================================================================

class AlphaLossType(Enum):
    """Alpha loss types."""
    NONE = "none"
    BCE = "bce"           # Binary Cross Entropy
    MSE = "mse"           # Mean Squared Error
    DICE = "dice"         # Dice Loss
    FOCAL = "focal"       # Focal Loss (for imbalanced masks)


def compute_alpha_loss(
    rendered_alpha: torch.Tensor,
    gt_alpha: torch.Tensor,
    loss_type: str = "bce",
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """
    Compute alpha supervision loss.
    
    Directly supervises rendered alpha to match GT alpha.
    
    Args:
        rendered_alpha: Rendered alpha from Gaussians [B*V, 1, H, W], range [0, 1]
        gt_alpha: Ground truth alpha [B*V, 1, H, W], range [0, 1]
        loss_type: Loss type ('bce', 'mse', 'dice', 'focal')
        focal_gamma: Gamma for focal loss
        
    Returns:
        Scalar loss tensor
    """
    if rendered_alpha is None or gt_alpha is None:
        return torch.tensor(0.0, device=rendered_alpha.device if rendered_alpha is not None else 'cpu')
    
    # Ensure same shape
    if rendered_alpha.shape != gt_alpha.shape:
        gt_alpha = torch.nn.functional.interpolate(
            gt_alpha, size=rendered_alpha.shape[-2:], mode='bilinear', align_corners=False
        )
    
    # Clamp to valid range
    rendered_alpha = rendered_alpha.clamp(1e-7, 1 - 1e-7)
    gt_alpha = gt_alpha.clamp(0, 1)
    
    if loss_type == "mse":
        return torch.nn.functional.mse_loss(rendered_alpha, gt_alpha)
    
    elif loss_type == "bce":
        # Disable autocast for BCE (not AMP compatible)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            return torch.nn.functional.binary_cross_entropy(
                rendered_alpha.float(), gt_alpha.float()
            )
    
    elif loss_type == "dice":
        # Dice loss: 1 - 2*intersection / (sum_pred + sum_gt)
        intersection = (rendered_alpha * gt_alpha).sum()
        union = rendered_alpha.sum() + gt_alpha.sum()
        dice = 2.0 * intersection / union.clamp(min=1e-7)
        return 1.0 - dice
    
    elif loss_type == "focal":
        # Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        # Helps with class imbalance (lots of background)
        # Disable autocast for BCE (not AMP compatible)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            bce = torch.nn.functional.binary_cross_entropy(
                rendered_alpha.float(), gt_alpha.float(), reduction="none"
            )
        p_t = rendered_alpha * gt_alpha + (1 - rendered_alpha) * (1 - gt_alpha)
        focal_weight = (1 - p_t) ** focal_gamma
        return (focal_weight * bce).mean()
    
    else:
        raise ValueError(f"Unknown alpha loss type: {loss_type}")


def compute_alpha_metrics(
    rendered_alpha: torch.Tensor,
    gt_alpha: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute alpha-related metrics.
    
    Args:
        rendered_alpha: Rendered alpha [B*V, 1, H, W]
        gt_alpha: Ground truth alpha [B*V, 1, H, W]
        threshold: Binarization threshold
        
    Returns:
        Dict with metrics (iou, precision, recall, f1)
    """
    if rendered_alpha is None or gt_alpha is None:
        return {}
    
    pred_binary = (rendered_alpha > threshold).float()
    gt_binary = (gt_alpha > threshold).float()
    
    # IoU
    intersection = (pred_binary * gt_binary).sum()
    union = ((pred_binary + gt_binary) > 0).float().sum()
    iou = (intersection / union.clamp(min=1)).item()
    
    # Precision & Recall
    tp = intersection
    fp = (pred_binary * (1 - gt_binary)).sum()
    fn = ((1 - pred_binary) * gt_binary).sum()
    
    precision = (tp / (tp + fp).clamp(min=1)).item()
    recall = (tp / (tp + fn).clamp(min=1)).item()
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    
    return {
        "alpha_iou": iou,
        "alpha_precision": precision,
        "alpha_recall": recall,
        "alpha_f1": f1,
    }


class AlphaLossComputer:
    """
    Alpha loss computation module.

    Usage:
        alpha_loss_computer = AlphaLossComputer(config)
        loss = alpha_loss_computer(rendered_alpha, gt_alpha)
        metrics = alpha_loss_computer.get_metrics(rendered_alpha, gt_alpha)

    Config options:
        training.losses.alpha_loss_weight: float (default 0.0, disabled)
        training.losses.alpha_loss_type: str ('bce', 'mse', 'dice', 'focal')
        training.losses.alpha_focal_gamma: float (default 2.0)

    Recommended settings for mask_mode=alpha:
        alpha_loss_weight: 0.1 (or higher)
        alpha_loss_type: focal (for imbalanced fg/bg)
    """

    def __init__(self, config):
        self.config = config
        losses = config.training.losses

        self.weight = losses.get("alpha_loss_weight", 0.0)
        self.loss_type = losses.get("alpha_loss_type", "bce")
        self.focal_gamma = losses.get("alpha_focal_gamma", 2.0)
        self.enabled = self.weight > 0.0

        # Check for potential misconfiguration
        mask_mode = losses.get("mask_mode", None)
        self._using_alpha_mask_without_supervision = (
            mask_mode == "alpha" and not self.enabled
        )

        if self.enabled:
            print(f"[AlphaLossComputer] Enabled: weight={self.weight}, type={self.loss_type}")
        elif self._using_alpha_mask_without_supervision:
            # Warning is handled by compute_mask_from_config, but we track it here
            pass
    
    def __call__(
        self,
        rendered_alpha: torch.Tensor,
        gt_alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted alpha loss."""
        if not self.enabled:
            device = rendered_alpha.device if rendered_alpha is not None else 'cpu'
            return torch.tensor(0.0, device=device)
        
        loss = compute_alpha_loss(
            rendered_alpha, gt_alpha,
            loss_type=self.loss_type,
            focal_gamma=self.focal_gamma,
        )
        return self.weight * loss
    
    def get_metrics(
        self,
        rendered_alpha: torch.Tensor,
        gt_alpha: torch.Tensor,
    ) -> Dict[str, float]:
        """Get alpha metrics."""
        return compute_alpha_metrics(rendered_alpha, gt_alpha)
    
    def get_config_info(self) -> Dict[str, any]:
        """Get config for logging."""
        return {
            "alpha_loss/enabled": self.enabled,
            "alpha_loss/weight": self.weight,
            "alpha_loss/type": self.loss_type,
            "alpha_loss/unsafe_alpha_mask": self._using_alpha_mask_without_supervision,
        }


# =============================================================================
# Floater/Ghosting Artifact Mitigation
# =============================================================================

def compute_opacity_regularization(
    opacity: torch.Tensor,
    reg_type: str = "entropy",
    target_sparsity: float = 0.5,
) -> torch.Tensor:
    """
    Compute opacity regularization to reduce floater artifacts.
    
    Floaters arise when opacity gets stuck in local minima (mid-range values).
    This regularization encourages opacity to be either 0 or 1.
    
    Args:
        opacity: Gaussian opacity values [N] or [B, N] in range [0, 1]
        reg_type: Type of regularization
            - "entropy": Binary entropy loss (encourages 0 or 1)
            - "l1_sparse": L1 penalty for sparsity
            - "l2_binary": L2 distance to nearest binary value
        target_sparsity: Target fraction of opaque Gaussians (for l1_sparse)
    
    Returns:
        Scalar regularization loss
    
    Reference:
        StableGS: A Floater-Free Framework for 3D Gaussian Splatting
        https://arxiv.org/html/2503.18458
    """
    opacity = opacity.float().clamp(1e-4, 1 - 1e-4)  # float32 + wider range for bf16 compatibility
    
    if reg_type == "entropy":
        # Binary cross-entropy style: -p*log(p) - (1-p)*log(1-p)
        # Minimized when p=0 or p=1
        entropy = -opacity * torch.log(opacity) - (1 - opacity) * torch.log(1 - opacity)
        return entropy.mean()
    
    elif reg_type == "l1_sparse":
        # Encourage sparsity (fewer opaque Gaussians)
        return torch.abs(opacity.mean() - target_sparsity)
    
    elif reg_type == "l2_binary":
        # Distance to nearest binary value (0 or 1)
        dist_to_binary = torch.min(opacity, 1 - opacity)
        return (dist_to_binary ** 2).mean()
    
    else:
        raise ValueError(f"Unknown opacity reg type: {reg_type}")


class OpacityRegularizer:
    """
    Opacity regularization module for reducing floater artifacts.
    
    Config options:
        training.losses.opacity_reg_weight: Weight for opacity regularization (default: 0.0)
        training.losses.opacity_reg_type: "entropy", "l1_sparse", "l2_binary"
        training.losses.opacity_target_sparsity: Target sparsity for l1_sparse
    """
    
    def __init__(self, config):
        self.config = config
        losses_cfg = config.training.losses
        self.weight = losses_cfg.get("opacity_reg_weight", 0.0)
        self.reg_type = losses_cfg.get("opacity_reg_type", "entropy")
        self.target_sparsity = losses_cfg.get("opacity_target_sparsity", 0.5)
        self.enabled = self.weight > 0.0
    
    def __call__(self, opacity: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=opacity.device)
        
        return self.weight * compute_opacity_regularization(
            opacity, self.reg_type, self.target_sparsity
        )


def compute_depth_regularization(
    rendered_depth: torch.Tensor,
    pseudo_gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = "l1",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute depth regularization loss using pseudo ground truth depth.
    
    Depth priors help constrain Gaussians to plausible depths,
    reducing floaters that appear at incorrect distances.
    
    Args:
        rendered_depth: Depth from Gaussian rendering [B, 1, H, W]
        pseudo_gt_depth: Depth from monocular estimation [B, 1, H, W]
        mask: Optional foreground mask [B, 1, H, W]
        loss_type: "l1", "l2", or "ssim"
        normalize: Whether to normalize depths to same scale
    
    Returns:
        Scalar depth regularization loss
    
    Reference:
        DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization
        DepthRegularizedGS, SparseGS
    """
    if normalize:
        # Scale-invariant normalization
        rendered_mean = rendered_depth.mean()
        pseudo_mean = pseudo_gt_depth.mean()
        rendered_depth = rendered_depth / (rendered_mean + 1e-8)
        pseudo_gt_depth = pseudo_gt_depth / (pseudo_mean + 1e-8)
    
    if mask is not None:
        rendered_depth = rendered_depth * mask
        pseudo_gt_depth = pseudo_gt_depth * mask
        num_valid = mask.sum().clamp(min=1)
    else:
        num_valid = rendered_depth.numel()
    
    if loss_type == "l1":
        loss = torch.abs(rendered_depth - pseudo_gt_depth).sum() / num_valid
    elif loss_type == "l2":
        loss = ((rendered_depth - pseudo_gt_depth) ** 2).sum() / num_valid
    elif loss_type == "gradient":
        # Edge-aware depth loss (gradient matching)
        def gradient(x):
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            return dx, dy
        
        dx_r, dy_r = gradient(rendered_depth)
        dx_p, dy_p = gradient(pseudo_gt_depth)
        loss = (torch.abs(dx_r - dx_p).mean() + torch.abs(dy_r - dy_p).mean()) / 2
    else:
        raise ValueError(f"Unknown depth loss type: {loss_type}")
    
    return loss


class DepthRegularizer:
    """
    Depth regularization module using monocular depth estimation.
    
    Config options:
        training.losses.depth_reg_weight: Weight for depth regularization (default: 0.0)
        training.losses.depth_reg_type: "l1", "l2", "gradient"
        training.losses.depth_normalize: Whether to normalize depths
    
    Note: Requires rendered_depth from Gaussian renderer and 
          pseudo_gt_depth from monocular depth estimator (e.g., DPT, Marigold)
    """
    
    def __init__(self, config):
        self.config = config
        losses_cfg = config.training.losses
        self.weight = losses_cfg.get("depth_reg_weight", 0.0)
        self.loss_type = losses_cfg.get("depth_reg_type", "l1")
        self.normalize = losses_cfg.get("depth_normalize", True)
        self.enabled = self.weight > 0.0
    
    def __call__(
        self, 
        rendered_depth: torch.Tensor, 
        pseudo_gt_depth: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.tensor(0.0, device=rendered_depth.device)
        
        return self.weight * compute_depth_regularization(
            rendered_depth, pseudo_gt_depth, mask, self.loss_type, self.normalize
        )


# =============================================================================
# Turntable Rendering Utilities  
# =============================================================================

def get_turntable_config(config) -> dict:
    """
    Get turntable rendering configuration from config.
    
    Config options (under visualization section):
        visualization.turntable_elevation: Camera elevation in degrees (default: 20)
        visualization.turntable_radius: Camera distance (default: 2.7)
        visualization.turntable_num_views: Number of views (default: 64)
        visualization.turntable_resolution: Rendering resolution (default: 384)
        visualization.turntable_grid_cols: Grid columns for preview image (default: 8)
        visualization.turntable_grid_rows: Grid rows for preview image (default: 8)
        visualization.save_video: Whether to save MP4 video (default: True)
    
    Returns:
        dict with turntable settings
    """
    vis_cfg = config.get("visualization", {})
    num_views = vis_cfg.get("turntable_num_views", 64)
    grid_cols = vis_cfg.get("turntable_grid_cols", 8)
    grid_rows = vis_cfg.get("turntable_grid_rows", 8)
    
    return {
        "elevation": vis_cfg.get("turntable_elevation", 20),
        "radius": vis_cfg.get("turntable_radius", 2.7),
        "num_views": num_views,
        "resolution": vis_cfg.get("turntable_resolution", 384),
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
        "save_video": vis_cfg.get("save_video", True),
        "fps": vis_cfg.get("turntable_fps", 15),  # Default 15 for slower rotation
    }
