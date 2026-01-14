"""
Logging Utilities

Extended WandB logging for experiment tracking.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExperimentInfo:
    """Experiment configuration summary."""
    # Dataset
    dataset_version: str
    dataset_path: str
    # Views
    num_input_views: int
    total_views: int
    input_ratio: float
    random_view_selection: bool
    # Mask
    use_masked_loss: bool
    mask_source: str  # "none", "gt", "rgb_pred", "alpha"
    alpha_threshold: float
    pred_threshold: float
    # Training
    max_steps: int
    batch_size: int
    lr: float
    # Preprocessing
    normalize_cameras: bool
    target_camera_distance: float
    normalize_to_z_up: bool
    background_color: str


def get_experiment_info(config) -> Dict[str, Any]:
    """
    Extract experiment info from config for WandB logging.

    Args:
        config: Full training config

    Returns:
        dict: Experiment info suitable for wandb.init(config=...)
    """
    def extract_version(path: str) -> str:
        match = re.search(r"v(\d+)", path)
        return f"v{match.group(1)}" if match else "unknown"

    def extract_dataset_name(path: str) -> str:
        if "/" in path:
            return path.split("/")[-2]
        return "unknown"

    # Determine mask source
    losses = config.training.losses
    if losses.get("use_rendered_alpha_mask", False):
        mask_source = "alpha"
    elif losses.get("use_predicted_mask", False):
        mask_source = "rgb_pred"
    elif losses.get("masked_l2_loss", False):
        mask_source = "gt"
    else:
        mask_source = "none"

    # Mouse config (optional)
    mouse_config = getattr(config, "mouse", {})
    if not isinstance(mouse_config, dict):
        mouse_config = dict(mouse_config) if hasattr(mouse_config, "__iter__") else {}

    dataset = config.training.dataset

    return {
        # Dataset info
        "dataset_version": extract_version(dataset.dataset_path),
        "dataset_name": extract_dataset_name(dataset.dataset_path),
        # View configuration
        "num_input_views": dataset.num_input_views,
        "total_views": dataset.num_views,
        "input_ratio": dataset.num_input_views / dataset.num_views,
        "random_view_selection": dataset.get("random_view_selection", False),
        # Mask configuration
        "use_masked_loss": losses.get("masked_l2_loss", False),
        "mask_source": mask_source,
        "alpha_threshold": losses.get("alpha_mask_threshold", 0.5),
        "pred_threshold": losses.get("pred_mask_threshold", 0.1),
        # Training info
        "max_steps": config.training.schedule.max_fwdbwd_passes,
        "batch_size": config.training.dataloader.batch_size_per_gpu,
        "lr": config.training.optimizer.lr,
        # Preprocessing (mouse-specific)
        "normalize_cameras": mouse_config.get("normalize_cameras", False),
        "target_camera_distance": mouse_config.get("target_camera_distance", 0.0),
        "normalize_to_z_up": mouse_config.get("normalize_to_z_up", True),
        "background_color": dataset.get("background_color", "white"),
    }


def get_wandb_log_dict(
    loss_metrics: Any,
    step: int,
    lr: float,
    grad_norm: float,
    iter_time: float,
    epoch: int,
    include_ghost: bool = True,
) -> Dict[str, Any]:
    """
    Create WandB log dict from loss metrics.

    Args:
        loss_metrics: Loss metrics from model
        step: Current step
        lr: Learning rate
        grad_norm: Gradient norm
        iter_time: Iteration time
        epoch: Current epoch
        include_ghost: Include ghost metrics

    Returns:
        dict: Ready for wandb.log()
    """
    log_dict = {
        "train_meta/step": step,
        "train_meta/lr": lr,
        "train_meta/grad_norm": grad_norm,
        "train_meta/iter_time": iter_time,
        "train_meta/epoch": epoch,
    }

    # Primary metrics (always log)
    primary = ["loss", "l2_loss", "psnr", "mask_iou"]

    # Secondary metrics (log if non-zero)
    secondary = ["perceptual_loss", "ssim_loss", "lpips_loss", "background_loss"]

    # Auxiliary metrics
    auxiliary = ["gt_mean", "pred_mean", "mask_coverage", "gaussians_usage"]

    # Ghost metrics
    ghost = ["ghost_fg_coverage", "ghost_alpha_std", "ghost_opacity_std", "ghost_opacity_mean"]

    # Extract metrics from loss_metrics object
    for name in primary:
        if hasattr(loss_metrics, name):
            val = getattr(loss_metrics, name)
            if isinstance(val, (int, float)):
                log_dict[f"train/{name}"] = val
            elif hasattr(val, "item"):
                log_dict[f"train/{name}"] = val.item()

    for name in secondary:
        if hasattr(loss_metrics, name):
            val = getattr(loss_metrics, name)
            v = val.item() if hasattr(val, "item") else val
            if v != 0:
                log_dict[f"train/{name}"] = v

    for name in auxiliary:
        if hasattr(loss_metrics, name):
            val = getattr(loss_metrics, name)
            v = val.item() if hasattr(val, "item") else val
            log_dict[f"train_aux/{name}"] = v

    if include_ghost:
        for name in ghost:
            if hasattr(loss_metrics, name):
                val = getattr(loss_metrics, name)
                v = val.item() if hasattr(val, "item") else val
                short_name = name.replace("ghost_", "")
                log_dict[f"ghost/{short_name}"] = v

    return log_dict


def get_validation_log_dict(
    val_metrics: Dict[str, List[float]],
    step: int,
    total_steps: int,
) -> Dict[str, Any]:
    """
    Create WandB log dict for validation metrics.

    Args:
        val_metrics: dict with psnr, ssim, lpips, mask_iou lists
        step: Current step
        total_steps: Total training steps

    Returns:
        dict: Ready for wandb.log()
    """
    def safe_mean(lst):
        return sum(lst) / max(len(lst), 1) if lst else 0.0

    log_dict = {
        "val/psnr": safe_mean(val_metrics.get("psnr", [])),
        "val/ssim": safe_mean(val_metrics.get("ssim", [])),
        "val/lpips": safe_mean(val_metrics.get("lpips", [])),
        "val/mask_iou": safe_mean(val_metrics.get("mask_iou", [])),
        "val/ssim_loss": 1.0 - safe_mean(val_metrics.get("ssim", [])),
        # Meta info
        "meta/current_step": step,
        "meta/total_steps": total_steps,
        "meta/progress": step / total_steps if total_steps > 0 else 0,
    }

    # Per-view metrics
    if "per_view_psnr" in val_metrics:
        for i, (psnr, lpips, ssim) in enumerate(zip(
            val_metrics["per_view_psnr"],
            val_metrics.get("per_view_lpips", []),
            val_metrics.get("per_view_ssim", []),
        )):
            log_dict[f"val_view/view{i}_psnr"] = psnr
            if lpips:
                log_dict[f"val_view/view{i}_lpips"] = lpips
            if ssim:
                log_dict[f"val_view/view{i}_ssim"] = ssim

    return log_dict
