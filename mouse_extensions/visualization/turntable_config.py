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
MOUSE_CAMERA_ORDER = [0, 5, 3, 1, 2, 4]

# Default turntable settings
DEFAULT_TURNTABLE_CONFIG = {
    "camera_order": MOUSE_CAMERA_ORDER,
    "fps": 30,  # 2x faster turntable rotation
    "interpolation_steps": 6,  # Steps between each camera pair
    "grid_rows": 6,
    "grid_cols": 6,
    "video_views": 144,  # Smooth video (144 = 36*4 frames)
    "grid_views": 36,    # Grid image (6x6 = 36)
    "add_row_labels": True,
    "use_camera_interpolation": True,
    "save_video": True,  # Enable video by default for train/val
    "smooth_trajectory": True,  # Enable smooth camera interpolation
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


def add_left_row_labels(
    grid_image: np.ndarray,
    camera_order: Optional[List[int]] = None,
    grid_rows: int = 6,
    grid_cols: int = 6,
    label_width: int = 120,
    font_scale: float = 1.0,
    loop: bool = True
) -> np.ndarray:
    """
    Add row labels to the LEFT side of each row in the grid.
    
    Args:
        grid_image: [H, W, 3] uint8 image
        camera_order: e.g., [1, 3, 5, 0, 4, 2] for 360 deg traversal
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        label_width: Width of label bar in pixels
        font_scale: Font scale for labels
        loop: Whether camera order loops back to start
    
    Returns:
        [H, W + label_width, 3] image with labels on left
    """
    import cv2
    
    if camera_order is None:
        camera_order = MOUSE_CAMERA_ORDER
    
    # Extend camera order for looping
    full_order = camera_order + [camera_order[0]] if loop else camera_order
    
    h, w = grid_image.shape[:2]
    row_height = h // grid_rows
    
    # Calculate frames per row and segment info
    total_frames = grid_rows * grid_cols
    num_segments = len(camera_order)
    frames_per_segment = total_frames / num_segments
    
    # Generate row labels
    row_labels = []
    for row_idx in range(grid_rows):
        row_start_frame = row_idx * grid_cols
        row_end_frame = row_start_frame + grid_cols - 1
        
        # Find which segment this row spans
        start_seg = int(row_start_frame / frames_per_segment) if frames_per_segment > 0 else 0
        end_seg = int(row_end_frame / frames_per_segment) if frames_per_segment > 0 else 0
        
        start_seg = min(start_seg, num_segments - 1)
        end_seg = min(end_seg, num_segments - 1)
        
        from_cam = full_order[start_seg]
        to_cam = full_order[min(end_seg + 1, len(full_order) - 1)]
        row_labels.append(f"{from_cam}->{to_cam}")
    
    # Create new image with label bar on left
    result = np.zeros((h, w + label_width, 3), dtype=np.uint8)
    
    # Dark background for label column
    result[:, :label_width] = (40, 40, 40)
    
    # Copy original grid image
    result[:, label_width:] = grid_image
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thick = 2
    
    for row_idx in range(grid_rows):
        # Position of this row's label
        label_y = row_idx * row_height + row_height // 2
        
        # Add text (centered vertically in row)
        text = row_labels[row_idx]
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
        text_x = (label_width - tw) // 2
        text_y = label_y + th // 2
        
        # Draw text
        cv2.putText(result, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thick)
    
    return result


def subsample_frames_for_grid(
    all_frames: np.ndarray,
    video_views: int,
    grid_views: int = 36
) -> np.ndarray:
    """
    Subsample video frames for grid display.
    
    Args:
        all_frames: [H, V, W, C] array of all video frames
        video_views: Number of video frames (e.g., 144)
        grid_views: Target number of grid frames (e.g., 36)
    
    Returns:
        [H, grid_views, W, C] subsampled frames
    """
    if video_views > grid_views:
        # Take evenly spaced frames (144 -> 36 = every 4th frame)
        indices = np.linspace(0, video_views - 1, grid_views, dtype=int)
        return all_frames[:, indices, :, :]
    return all_frames


def create_grid_from_video(
    turntable_image: np.ndarray,
    video_views: int,
    grid_rows: int = 6,
    grid_cols: int = 6
) -> tuple:
    """
    Create grid image from turntable video frames.
    
    Args:
        turntable_image: [H, V*W, C] concatenated video frames
        video_views: Number of video frames
        grid_rows: Number of grid rows (default 6)
        grid_cols: Number of grid columns (default 6)
    
    Returns:
        (grid_image, all_frames, h_img) tuple
    """
    from einops import rearrange
    
    h_img = turntable_image.shape[0]
    w_per_view = turntable_image.shape[1] // video_views
    all_frames = turntable_image.reshape(h_img, video_views, w_per_view, 3)
    
    # Subsample for grid
    grid_views = grid_rows * grid_cols
    grid_frames = subsample_frames_for_grid(all_frames, video_views, grid_views)
    
    # Rearrange to grid layout
    grid_image = rearrange(grid_frames, "h (rows cols) w c -> (rows h) (cols w) c", 
                          rows=grid_rows, cols=grid_cols)
    
    return grid_image, all_frames, h_img
