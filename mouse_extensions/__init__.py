"""
Mouse Extensions for FaceLift/GS-LRM (v1.2.0)

Modular extensions organized by functionality:
- data/: Data preprocessing (camera normalization, backgrounds)
- model/: Model training (loss, rendering, visualization)
- utils/: Utilities (logging, experiment tracking)

Usage:
    # Direct import (recommended)
    from gslrm.mouse_extensions.data import preprocess_cameras
    from gslrm.mouse_extensions.model import compute_mask_from_config
    from gslrm.mouse_extensions.utils import get_experiment_info

    # Legacy import (backward compatible)
    from gslrm.mouse_extensions import preprocess_cameras
"""

__version__ = "1.2.0"

# =============================================================================
# Re-export from data/ module
# =============================================================================
from .data import (
    pil_to_np,
    normalize_camera_distance,
    normalize_camera_distance_with_intrinsics,
    normalize_cameras_to_y_up,
    normalize_cameras_to_z_up,
    get_bg_color,
    BG_COLORS,
    preprocess_cameras,
    PreprocessingConfig,
)

# =============================================================================
# Re-export from model/ module
# =============================================================================
from .model import (
    # Loss
    compute_mask_from_config,
    compute_ghost_metrics,
    compute_gaussians_usage,
    MaskType,
    MaskConfig,
    LossExtensions,
    # Visualization
    create_threshold_comparison,
    add_labels_to_visualization,
    # Alpha rendering
    DIFF_GAUSS_AVAILABLE,
)

# Conditional imports for alpha rendering
if DIFF_GAUSS_AVAILABLE:
    from .model import (
        render_opencv_cam_with_alpha,
        DeferredGaussianRenderWithAlpha,
    )

# =============================================================================
# Re-export from utils/ module
# =============================================================================
from .utils import (
    get_experiment_info,
    get_wandb_log_dict,
    get_validation_log_dict,
)

# =============================================================================
# All exports
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Data
    "pil_to_np",
    "normalize_camera_distance",
    "normalize_camera_distance_with_intrinsics",
    "normalize_cameras_to_y_up",
    "normalize_cameras_to_z_up",
    "get_bg_color",
    "BG_COLORS",
    "preprocess_cameras",
    "PreprocessingConfig",
    # Model - Loss
    "compute_mask_from_config",
    "compute_ghost_metrics",
    "compute_gaussians_usage",
    "MaskType",
    "MaskConfig",
    "LossExtensions",
    # Model - Visualization
    "create_threshold_comparison",
    "add_labels_to_visualization",
    # Model - Alpha rendering
    "DIFF_GAUSS_AVAILABLE",
    "render_opencv_cam_with_alpha",
    "DeferredGaussianRenderWithAlpha",
    # Utils
    "get_experiment_info",
    "get_wandb_log_dict",
    "get_validation_log_dict",
]
