"""
Mouse Extensions Visualization Module

Provides:
- Alpha mask visualization
- Turntable configuration and trajectory generation
- TurntableRenderer: unified turntable video/grid generation (train/val/inference)
- TemporalVideoRenderer: multi-frame batch temporal videos
- Error annotation utilities
- Gaussian export (PLY, NPZ, Rerun .rrd)
- Inference visualization (comparison grids, multi-view turntable grids)
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
    add_angle_overlay_to_grid,
)

from .turntable_renderer import (
    TurntableVideoConfig,
    TurntableRenderer,
    TemporalVideoRenderer,
)

from .error_annotation import (
    add_error_scale_annotation,
    compute_pred_mask_for_visualization,
)

from .gaussian_export import (
    GaussianExporter,
    RerunExporter,
)

from .inference_viz import (
    save_comparison_grid,
    save_multiview_turntable_grid,
)


from .keypoint_overlay import (
    KeypointVisualizer,
    project_3d_to_2d,
    draw_keypoint_overlay,
    draw_bounding_box,
    create_legend,
    CameraFollowConfig,
    KeypointFollowCamera,
    SKELETON_BONES,
    JOINT_GROUPS,
    KEYPOINT_NAMES,
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
    "compute_camera_convergence_center",
    "get_dynamic_camera_order",
    "get_grid_row_labels",
    "subsample_frames_for_grid",
    "create_grid_from_video",
    "add_angle_overlay_to_grid",
    # Turntable renderer (unified)
    "TurntableVideoConfig",
    "TurntableRenderer",
    "TemporalVideoRenderer",
    # Error annotation
    "add_error_scale_annotation",
    "compute_pred_mask_for_visualization",
    # Gaussian export
    "GaussianExporter",
    "RerunExporter",
    # Inference visualization
    "save_comparison_grid",
    "save_multiview_turntable_grid",
    # Keypoint overlay
    "KeypointVisualizer",
    "project_3d_to_2d",
    "draw_keypoint_overlay",
    "draw_bounding_box",
    "create_legend",
    "SKELETON_BONES",
    "JOINT_GROUPS",
    "KEYPOINT_NAMES",
    "CameraFollowConfig",
    "KeypointFollowCamera",
]
