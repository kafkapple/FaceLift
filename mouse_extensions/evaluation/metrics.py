"""
Metrics computation module for FaceLift mouse experiments.

Centralizes all metric functions into a single class to prevent
the "missing parameter" bug pattern when adding new metrics.

Usage:
    from mouse_extensions.evaluation import MetricsComputer

    metrics = MetricsComputer()
    psnr = metrics.compute_psnr(gt, pred, mask=mask)
    l1 = metrics.compute_l1(gt, pred, mask=mask)

    # Or get all metrics at once
    results = metrics.compute_all(gt, pred, mask=mask)
"""

from typing import Dict, Optional, Any
import torch
from torch import Tensor
from jaxtyping import Float


class MetricsComputer:
    """
    Unified metrics computation class.

    Wraps all metric functions from gslrm.model.utils_metrics into a single object.
    This prevents the "3-place modification" bug when adding new metrics:
    - Before: import + function signature + call site
    - After: just add method to this class

    Benefits:
    - Single object to pass around (dependency injection friendly)
    - Easy to extend with new metrics
    - Lazy initialization of heavy resources (LPIPS model)
    - Consistent interface for all metrics
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize MetricsComputer.

        Args:
            device: Device for LPIPS model. If None, auto-detected on first use.
        """
        self._device = device
        self._lpips_model = None  # Lazy initialization

    @property
    def device(self) -> torch.device:
        """Get device, auto-detecting if not set."""
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    @property
    def lpips_model(self):
        """Lazy-load LPIPS model on first use."""
        if self._lpips_model is None:
            from gslrm.model.utils_metrics import get_lpips
            self._lpips_model = get_lpips(self.device)
        return self._lpips_model

    # =========================================================================
    # Core Metric Functions (delegating to gslrm.model.utils_metrics)
    # =========================================================================

    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
        mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
    ) -> Float[Tensor, " batch"]:
        """Compute PSNR between ground truth and predicted images."""
        from gslrm.model.utils_metrics import compute_psnr
        return compute_psnr(ground_truth, predicted, mask)

    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
        mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
    ) -> Float[Tensor, " batch"]:
        """Compute SSIM between ground truth and predicted images."""
        from gslrm.model.utils_metrics import compute_ssim
        return compute_ssim(ground_truth, predicted, mask)

    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
        mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
    ) -> Float[Tensor, " batch"]:
        """Compute LPIPS perceptual distance."""
        from gslrm.model.utils_metrics import compute_lpips
        return compute_lpips(ground_truth, predicted, mask)

    def compute_l1(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
        mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
        normalize_by_mask: bool = True,
    ) -> Float[Tensor, " batch"]:
        """
        Compute L1 distance (Pose Splatter style normalized masked L1).

        Args:
            normalize_by_mask: If True, normalize by mask area (Pose Splatter style)
        """
        from gslrm.model.utils_metrics import compute_l1
        return compute_l1(ground_truth, predicted, mask, normalize_by_mask)

    def compute_mask_iou(
        self,
        rendered: Float[Tensor, "batch channel height width"],
        gt_mask: Float[Tensor, "batch 1 height width"],
        bg_threshold: float = 0.1,
    ) -> Float[Tensor, " batch"]:
        """Compute IoU between rendered alpha and ground truth mask."""
        from gslrm.model.utils_metrics import compute_mask_iou
        return compute_mask_iou(rendered, gt_mask, bg_threshold)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def compute_all(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
        mask: Optional[Float[Tensor, "batch 1 height width"]] = None,
        include_iou: bool = True,
    ) -> Dict[str, Float[Tensor, " batch"]]:
        """
        Compute all metrics at once.

        Args:
            ground_truth: GT images [B, C, H, W]
            predicted: Predicted images [B, C, H, W]
            mask: Optional mask [B, 1, H, W]
            include_iou: Whether to compute mask IoU (requires mask)

        Returns:
            Dict with keys: psnr, ssim, lpips, l1, (mask_iou if include_iou)
        """
        results = {
            "psnr": self.compute_psnr(ground_truth, predicted, mask),
            "ssim": self.compute_ssim(ground_truth, predicted, mask),
            "lpips": self.compute_lpips(ground_truth, predicted, mask),
            "l1": self.compute_l1(ground_truth, predicted, mask),
        }

        if include_iou and mask is not None:
            results["mask_iou"] = self.compute_mask_iou(predicted, mask)

        return results

    def compute_per_view_metrics(
        self,
        target_image: Float[Tensor, "views 3 height width"],
        rendered: Float[Tensor, "views 3 height width"],
        gt_mask: Optional[Float[Tensor, "views 1 height width"]] = None,
    ) -> Dict[str, Any]:
        """
        Compute per-view metrics for validation.

        This is the main method used by ValidationRunner._compute_batch_metrics.

        Returns:
            Dict containing:
            - psnr, ssim, lpips, l1, mask_iou: scalar mean values
            - per_view_psnr, per_view_ssim, etc.: list of per-view values
        """
        per_view_psnr = self.compute_psnr(target_image, rendered, mask=gt_mask)
        per_view_ssim = self.compute_ssim(target_image, rendered, mask=gt_mask)
        per_view_lpips = self.compute_lpips(target_image, rendered, mask=gt_mask)
        per_view_l1 = self.compute_l1(target_image, rendered, mask=gt_mask, normalize_by_mask=True)

        mask_iou = 0.0
        if gt_mask is not None:
            per_view_iou = self.compute_mask_iou(rendered, gt_mask, bg_threshold=0.1)
            mask_iou = per_view_iou.mean().item()

        return {
            # Scalar means
            "psnr": per_view_psnr.mean().item(),
            "ssim": per_view_ssim.mean().item(),
            "lpips": per_view_lpips.mean().item(),
            "l1": per_view_l1.mean().item(),
            "mask_iou": mask_iou,
            # Per-view lists
            "per_view_psnr": per_view_psnr.cpu().tolist(),
            "per_view_ssim": per_view_ssim.cpu().tolist(),
            "per_view_lpips": per_view_lpips.cpu().tolist(),
            "per_view_l1": per_view_l1.cpu().tolist(),
        }


# Module-level singleton for convenience
_default_metrics_computer: Optional[MetricsComputer] = None


def get_metrics_computer(device: Optional[torch.device] = None) -> MetricsComputer:
    """Get or create the default MetricsComputer instance."""
    global _default_metrics_computer
    if _default_metrics_computer is None:
        _default_metrics_computer = MetricsComputer(device)
    return _default_metrics_computer
