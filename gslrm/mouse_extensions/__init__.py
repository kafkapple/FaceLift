"""
Mouse Extensions for FaceLift/GS-LRM

Minimal modifications to original codebase for mouse data experiments.

Modules:
- preprocessing: Camera normalization and coordinate transforms
- loss_extensions: Mask computation and ghost metrics
- visualization: Threshold comparison visualization
- logging_utils: WandB logging utilities
- alpha_renderer: Alpha rendering support (diff_gauss)
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
from .visualization import create_threshold_comparison

# Logging utilities
from .logging_utils import (
    get_experiment_info,
    get_wandb_log_dict,
    get_validation_log_dict,
)

# Preprocessing utilities
from .preprocessing import (
    pil_to_np,
    normalize_camera_distance,
    normalize_camera_distance_with_intrinsics,
    normalize_cameras_to_y_up,
    normalize_cameras_to_z_up,
    get_bg_color,
    preprocess_cameras,
    PreprocessingConfig,
    BG_COLORS,
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

__version__ = "1.1.0"
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
    # Logging
    "get_experiment_info",
    "get_wandb_log_dict",
    "get_validation_log_dict",
    # Preprocessing
    "pil_to_np",
    "normalize_camera_distance",
    "normalize_camera_distance_with_intrinsics",
    "normalize_cameras_to_y_up",
    "normalize_cameras_to_z_up",
    "get_bg_color",
    "preprocess_cameras",
    "PreprocessingConfig",
    "BG_COLORS",
    # Alpha rendering
    "render_opencv_cam_with_alpha",
    "DeferredGaussianRenderWithAlpha",
    "DIFF_GAUSS_AVAILABLE",
]
