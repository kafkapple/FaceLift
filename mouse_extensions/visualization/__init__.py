"""Mouse Extensions Visualization Module.

Provides:
- Video I/O (videoio + cv2 fallback)
- Camera utilities (order, convergence, turntable generation)
- Grid utilities (grid creation, labels, input strip)
- TurntableRenderer / TemporalVideoRenderer (unified turntable videos)
- Alpha mask, error annotation, Gaussian export, inference viz
- Keypoint overlay and camera follow
- GS-LRM rendering re-exports (render_opencv_cam, render_turntable, GaussianModel)

Refactored 2026-03-23: turntable_config.py split into camera_utils, grid_utils, video_io.
Re-exports gslrm.model.gaussians_renderer public API to decouple mouse_extensions from
internal gslrm paths.
"""

from .alpha_visualization import (
    visualize_alpha_comparison,
    compute_alpha_metrics,
    should_visualize_alpha,
)

from .video_io import (
    imageseq2video,
    save_video,
)

from .camera_utils import (
    compute_camera_order_from_extrinsics,
    compute_camera_convergence_center,
    get_dynamic_camera_order,
    get_turntable_cameras,
    get_turntable_with_dataset_views,
)

from .grid_utils import (
    subsample_frames_for_grid,
    create_grid_from_video,
    add_angle_overlay_to_grid,
    add_row_labels_to_grid,
    add_left_row_labels,
    create_labeled_input_strip,
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
    JOINT_GROUPS,
    KEYPOINT_NAMES,
)

from gslrm.model.gaussians_renderer import (
    render_opencv_cam,
    render_turntable,
    render_dataset_trajectory,
    GaussianModel,
)

__all__ = [
    # Video I/O
    "imageseq2video",
    "save_video",
    # Camera utilities
    "compute_camera_order_from_extrinsics",
    "compute_camera_convergence_center",
    "get_dynamic_camera_order",
    "get_turntable_cameras",
    "get_turntable_with_dataset_views",
    # Grid utilities
    "subsample_frames_for_grid",
    "create_grid_from_video",
    "add_angle_overlay_to_grid",
    "add_row_labels_to_grid",
    "add_left_row_labels",
    "create_labeled_input_strip",
    # Turntable renderer
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
    "JOINT_GROUPS",
    "KEYPOINT_NAMES",
    "CameraFollowConfig",
    "KeypointFollowCamera",
    # GS-LRM rendering (re-exported from gslrm.model.gaussians_renderer)
    "render_opencv_cam",
    "render_turntable",
    "render_dataset_trajectory",
    "GaussianModel",
]
