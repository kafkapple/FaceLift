"""
Model Training Extensions Module

Alpha rendering, mask computation, loss extensions,
and training visualization for mouse data experiments.
"""

# Loss and mask computation
from .loss_extensions import (
    compute_mask_from_config,
    compute_ghost_metrics,
    compute_gaussians_usage,
    MaskType,
    MaskConfig,
    LossExtensions,
)

# Visualization
from .visualization import (
    create_threshold_comparison,
    add_labels_to_visualization,
)

# Alpha rendering (optional - requires diff_gauss)
try:
    from .alpha_renderer import (
        render_opencv_cam_with_alpha,
        DeferredGaussianRenderWithAlpha,
        DIFF_GAUSS_AVAILABLE,
    )
except ImportError:
    DIFF_GAUSS_AVAILABLE = False
    render_opencv_cam_with_alpha = None
    DeferredGaussianRenderWithAlpha = None

__all__ = [
    # Loss
    "compute_mask_from_config",
    "compute_ghost_metrics",
    "compute_gaussians_usage",
    "MaskType",
    "MaskConfig",
    "LossExtensions",
    # Visualization
    "create_threshold_comparison",
    "add_labels_to_visualization",
    # Alpha rendering
    "render_opencv_cam_with_alpha",
    "DeferredGaussianRenderWithAlpha",
    "DIFF_GAUSS_AVAILABLE",
]
