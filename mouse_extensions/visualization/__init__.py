"""
Mouse Extensions Visualization Module

Provides:
- Alpha mask visualization
- Turntable configuration and trajectory generation
- Unified video generation for train/val
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
    get_grid_row_labels,
)

from .video_generator import (
    create_dataset_views_video,
    create_turntable_video,
    create_grid_image,
    VideoGeneratorContext,
    get_video_generator,
)

__all__ = [
    # Alpha visualization
    'visualize_alpha_comparison',
    'compute_alpha_metrics', 
    'should_visualize_alpha',
    # Turntable config
    'MOUSE_CAMERA_ORDER',
    'DEFAULT_TURNTABLE_CONFIG',
    'get_camera_order',
    'get_camera_pairs',
    'interpolate_camera_extrinsics',
    'create_turntable_trajectory',
    'get_grid_row_labels',
    # Video generation
    'create_dataset_views_video',
    'create_turntable_video',
    'create_grid_image',
    'VideoGeneratorContext',
    'get_video_generator',
]
