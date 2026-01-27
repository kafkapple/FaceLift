"""
Mouse Extensions - Data Module

Custom datasets for mouse reconstruction.
"""

from .mouse_dataset import MouseViewDataset
from .preprocessing import (
    normalize_camera_distance,
    normalize_camera_distance_with_intrinsics,
    normalize_cameras_to_y_up,
    normalize_cameras_to_z_up,
    get_bg_color,
)

__all__ = [
    "MouseViewDataset",
    "normalize_camera_distance",
    "normalize_camera_distance_with_intrinsics",
    "normalize_cameras_to_y_up",
    "normalize_cameras_to_z_up",
    "get_bg_color",
]
