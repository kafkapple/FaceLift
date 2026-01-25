"""
Turntable Configuration Module

Defines camera order and turntable rendering settings for mouse data.
Physical camera layout: clockwise 360-degree arrangement.

Camera Order (clockwise from front):
    cam 0 (front) -> cam 4 -> cam 2 -> cam 1 -> cam 3 -> cam 5 -> (back to 0)

Author: Claude Code
Date: 2026-01-25
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation, Slerp


# Physical camera arrangement (clockwise from front view)
MOUSE_CAMERA_ORDER = [1, 3, 5, 0, 4, 2]

# Default turntable settings
DEFAULT_TURNTABLE_CONFIG = {
    "camera_order": MOUSE_CAMERA_ORDER,
    "fps": 15,  # Slower than default 30 (2x slower)
    "interpolation_steps": 6,  # Steps between each camera pair
    "grid_rows": 6,
    "grid_cols": 6,
    "add_row_labels": True,
    "use_camera_interpolation": True,
}


def get_camera_order() -> List[int]:
    """Get the physical camera order for 360-degree rotation."""
    return MOUSE_CAMERA_ORDER.copy()


def get_camera_pairs() -> List[Tuple[int, int]]:
    """
    Get camera pairs for interpolation.

    Returns:
        List of (from_cam, to_cam) tuples
    """
    order = MOUSE_CAMERA_ORDER
    pairs = []
    for i in range(len(order)):
        from_cam = order[i]
        to_cam = order[(i + 1) % len(order)]
        pairs.append((from_cam, to_cam))
    return pairs


def interpolate_camera_extrinsics(
    c2w_from: np.ndarray,
    c2w_to: np.ndarray,
    num_steps: int = 6
) -> List[np.ndarray]:
    """
    Interpolate between two camera extrinsics (c2w matrices).

    Uses SLERP for rotation and linear interpolation for translation.

    Args:
        c2w_from: Source camera-to-world matrix [4, 4]
        c2w_to: Target camera-to-world matrix [4, 4]
        num_steps: Number of interpolation steps (including start, excluding end)

    Returns:
        List of interpolated c2w matrices
    """
    # Extract rotation and translation
    R_from = c2w_from[:3, :3]
    t_from = c2w_from[:3, 3]

    R_to = c2w_to[:3, :3]
    t_to = c2w_to[:3, 3]

    # Convert rotations to scipy Rotation objects
    rot_from = Rotation.from_matrix(R_from)
    rot_to = Rotation.from_matrix(R_to)

    # Create SLERP interpolator
    key_times = [0, 1]
    key_rots = Rotation.from_matrix(np.stack([R_from, R_to]))
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    interpolated = []
    for i in range(num_steps):
        t = i / num_steps  # 0, 1/6, 2/6, ..., 5/6 (excludes 1.0 = next camera)

        # Interpolate rotation (SLERP)
        R_interp = slerp(t).as_matrix()

        # Interpolate translation (linear)
        t_interp = (1 - t) * t_from + t * t_to

        # Construct c2w matrix
        c2w_interp = np.eye(4)
        c2w_interp[:3, :3] = R_interp
        c2w_interp[:3, 3] = t_interp

        interpolated.append(c2w_interp)

    return interpolated


def create_turntable_trajectory(
    camera_c2ws: List[np.ndarray],
    camera_order: Optional[List[int]] = None,
    interpolation_steps: int = 6
) -> List[np.ndarray]:
    """
    Create full turntable trajectory with camera interpolation.

    Args:
        camera_c2ws: List of c2w matrices for each camera [num_cams, 4, 4]
        camera_order: Order to visit cameras (default: MOUSE_CAMERA_ORDER)
        interpolation_steps: Number of steps between each camera pair

    Returns:
        List of interpolated c2w matrices for full trajectory
    """
    if camera_order is None:
        camera_order = MOUSE_CAMERA_ORDER

    trajectory = []

    for i in range(len(camera_order)):
        from_idx = camera_order[i]
        to_idx = camera_order[(i + 1) % len(camera_order)]

        c2w_from = camera_c2ws[from_idx]
        c2w_to = camera_c2ws[to_idx]

        # Interpolate between this camera pair
        segment = interpolate_camera_extrinsics(c2w_from, c2w_to, interpolation_steps)
        trajectory.extend(segment)

    return trajectory


def get_grid_row_labels(camera_order: Optional[List[int]] = None) -> List[str]:
    """
    Get row labels for grid visualization.

    Returns:
        List of labels like "Cam 0 -> 4", "Cam 4 -> 2", etc.
    """
    if camera_order is None:
        camera_order = MOUSE_CAMERA_ORDER

    labels = []
    for i in range(len(camera_order)):
        from_cam = camera_order[i]
        to_cam = camera_order[(i + 1) % len(camera_order)]
        labels.append(f"Cam {from_cam} -> {to_cam}")

    return labels
