"""
Mouse Extensions Visualization Module

Provides:
- Alpha mask visualization
- Turntable configuration and trajectory generation
- Unified video generation for train/val
- Error annotation utilities
- Unified visualization (train/val consistent)
"""

from .alpha_visualization import (
    visualize_alpha_comparison,
    compute_alpha_metrics,
    should_visualize_alpha,
)

from .turntable_config import (
    MOUSE_CAMERA_ORDER,
    DEFAULT_TURNTABLE_CONFIG,
    get_camera_order,
    get_camera_pairs,
    interpolate_camera_extrinsics,
    create_turntable_trajectory,
    compute_camera_convergence_center,
    get_dynamic_camera_order,
    get_grid_row_labels,
    subsample_frames_for_grid,
    create_grid_from_video,
)

from .video_generator import (
    # create_dataset_views_video,  # REMOVED
    create_turntable_video,
    create_grid_image,
    VideoGeneratorContext,
    get_video_generator,
)

from .error_annotation import (
    add_error_scale_annotation,
    compute_pred_mask_for_visualization,
)

__all__ = [
    # Alpha visualization
    "visualize_alpha_comparison",
    "compute_alpha_metrics",
    "should_visualize_alpha",
    # Turntable config
    "MOUSE_CAMERA_ORDER",
    "DEFAULT_TURNTABLE_CONFIG",
    "get_camera_order",
    "get_camera_pairs",
    "interpolate_camera_extrinsics",
    "create_turntable_trajectory",
    "get_grid_row_labels",
    # Video generation (legacy)
    # "create_dataset_views_video",  # REMOVED
    "create_turntable_video",
    "create_grid_image",
    "VideoGeneratorContext",
    "get_video_generator",
    # Error annotation
    "add_error_scale_annotation",
    "compute_pred_mask_for_visualization",
    # Unified visualization (recommended)
    "TurntableConfig",
    "create_unified_visualizer",
]
