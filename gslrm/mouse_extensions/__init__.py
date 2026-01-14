"""
Mouse Extensions for FaceLift/GS-LRM

Minimal modifications to original codebase for mouse data experiments.
"""

from .alpha_renderer import (
    render_opencv_cam_with_alpha,
    DeferredGaussianRenderWithAlpha,
)
from .loss_extensions import (
    compute_mask_from_config,
    compute_ghost_metrics,
    MaskType,
)
from .visualization import create_threshold_comparison
from .logging_utils import (
    get_experiment_info,
    get_wandb_log_dict,
)

__version__ = "1.0.0"
__all__ = [
    "render_opencv_cam_with_alpha",
    "DeferredGaussianRenderWithAlpha",
    "compute_mask_from_config",
    "compute_ghost_metrics",
    "MaskType",
    "create_threshold_comparison",
    "get_experiment_info",
    "get_wandb_log_dict",
]
