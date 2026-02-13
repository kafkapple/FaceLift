"""
Model Training Extensions Module

Alpha rendering, mask computation, loss extensions,
and training visualization for mouse data experiments.
"""

# Loss and mask computation
from .loss_extensions import (
    compute_mask_iou,
    AlphaLossComputer,
    compute_alpha_loss,
    compute_alpha_metrics,
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
    "compute_mask_iou",
    "compute_mask_from_config",
    "compute_ghost_metrics",
    "compute_gaussians_usage",
    "MaskType",
    "MaskConfig",
    "LossExtensions",
    "AlphaLossComputer",
    "compute_alpha_loss",
    "compute_alpha_metrics",
    # Visualization
    "create_threshold_comparison",
    "add_labels_to_visualization",
    # Alpha rendering
    "render_opencv_cam_with_alpha",
    "DeferredGaussianRenderWithAlpha",
    "DIFF_GAUSS_AVAILABLE",
]

# GSLRM Patches (new)
from .gslrm_patches import (
    compute_alpha_loss_if_enabled,
    get_visualization_mask_for_display,
    create_config_aware_visual,
    get_mask_config_for_logging,
)

# Visualization Extensions (new)
from .visualization_extensions import (
    VisualizationConfig,
    create_training_visual,
    create_validation_visual,
    compute_pred_mask,
    compute_error_stats,
    create_error_heatmap,
    compute_visualization_mask,
    create_mask_overlay,
    create_threshold_comparison_grid,
    get_visualization_config_info,
    VisualizationMaskType,
)

# Ghost Gaussian Pruning (new)
from .gaussian_pruning import (
    compute_boundary_gaussians,
    GhostGaussianRegularizer,
    PruningConfig,
)

__all__ += [
    # GSLRM Patches
    "compute_alpha_loss_if_enabled",
    "get_visualization_mask_for_display",
    "create_config_aware_visual",
    "get_mask_config_for_logging",
    # Visualization Extensions
    "compute_visualization_mask",
    "create_mask_overlay",
    "create_threshold_comparison_grid",
    "get_visualization_config_info",
    "VisualizationMaskType",
    # Ghost Gaussian
    "compute_boundary_gaussians",
    "GhostGaussianRegularizer",
    "PruningConfig",
]

# Floater/Ghosting Artifact Mitigation (new)
from .loss_extensions import (
    OpacityRegularizer,
    DepthRegularizer,
    compute_opacity_regularization,
    compute_depth_regularization,
    get_turntable_config,
)

__all__ += [
    # Floater Mitigation
    "OpacityRegularizer",
    "DepthRegularizer",
    "compute_opacity_regularization",
    "compute_depth_regularization",
    "get_turntable_config",
]

# Pose Conditioning (camera-aware generation)
from .pose_conditioning import (
    CameraPoseConditioner,
    SphericalPoseEncoder,
    ExtrinsicPoseEncoder,
    PluckerRayEncoder,
    FourierEncoder,
    create_pose_conditioner_for_unet,
)
from .pose_conditioning_integration import (
    PoseConditioningInjector,
    load_m5_cameras,
    create_pose_injector_from_config,
)

__all__ += [
    # Pose Conditioning
    "CameraPoseConditioner",
    "SphericalPoseEncoder",
    "ExtrinsicPoseEncoder",
    "PluckerRayEncoder",
    "FourierEncoder",
    "create_pose_conditioner_for_unet",
    "PoseConditioningInjector",
    "load_m5_cameras",
    "create_pose_injector_from_config",
]

# MV-Adapter (future multi-view replacement - scaffolding)
from .mv_adapter import (
    MVAdapterConfig,
    MVAdapterModule,
    CameraGuider,
    create_mv_adapter_for_facelift,
)

__all__ += [
    # MV-Adapter
    "MVAdapterConfig",
    "MVAdapterModule",
    "CameraGuider",
    "create_mv_adapter_for_facelift",
]
