"""
Data Preprocessing Module

Camera normalization and coordinate system transformations
for adapting mouse capture data to FaceLift/GS-LRM format.
"""

from .preprocessing import (
    # Image utilities
    pil_to_np,
    # Camera normalization
    normalize_camera_distance,
    normalize_camera_distance_with_intrinsics,
    normalize_cameras_to_y_up,
    normalize_cameras_to_z_up,
    # Background color
    get_bg_color,
    BG_COLORS,
    # Pipeline
    preprocess_cameras,
    PreprocessingConfig,
)

__all__ = [
    "pil_to_np",
    "normalize_camera_distance",
    "normalize_camera_distance_with_intrinsics",
    "normalize_cameras_to_y_up",
    "normalize_cameras_to_z_up",
    "get_bg_color",
    "BG_COLORS",
    "preprocess_cameras",
    "PreprocessingConfig",
]
