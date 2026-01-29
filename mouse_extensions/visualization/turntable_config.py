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
MOUSE_CAMERA_ORDER = [0, 4, 2, 1, 3, 5]  # CCW order by actual azimuth

# Default turntable settings
DEFAULT_TURNTABLE_CONFIG = {
    "camera_order": MOUSE_CAMERA_ORDER,
    "fps": 15,  # Slower rotation (was 30)
    "interpolation_steps": 6,  # Steps between each camera pair
    "grid_rows": 6,
    "grid_cols": 6,
    "video_views": 144,  # Smooth video (144 = 36*4 frames)
    "grid_views": 36,    # Grid image (6x6 = 36)
    "add_row_labels": False,
    "add_angle_overlay": True,
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
    label_width: int = 240,
    font_scale: float = 2.0,
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
    font_thick = 4
    
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
    grid_views: int = 36,
    segments: list = None
) -> np.ndarray:
    """
    Subsample video frames for grid display.
    
    When segments are provided, only transition (non-hold) frames are used
    to avoid duplicates from hold periods in the grid.
    
    Args:
        all_frames: [H, V, W, C] array of all video frames
        video_views: Number of video frames (e.g., 144)
        grid_views: Target number of grid frames (e.g., 36)
        segments: Optional list of segment dicts with 'is_hold' and frame range info
    
    Returns:
        [H, grid_views, W, C] subsampled frames
    """
    if video_views <= grid_views:
        return all_frames
    
    if segments is not None:
        # Filter out hold frames, keep only transition frames
        # segments: list of (start_frame, end_frame, from_cam, to_cam, is_hold) tuples
        transition_indices = []
        for seg in segments:
            start, end, _, _, is_hold = seg
            if not is_hold:
                for idx in range(start, end):
                    if idx < video_views:
                        transition_indices.append(idx)
        
        if len(transition_indices) >= grid_views:
            # Uniformly subsample from transition frames only
            sub_idx = np.linspace(0, len(transition_indices) - 1, grid_views, dtype=int)
            indices = [transition_indices[i] for i in sub_idx]
        else:
            # Not enough transition frames, fall back to all frames
            indices = np.linspace(0, video_views - 1, grid_views, dtype=int).tolist()
    else:
        # No segment info: uniform subsample from all frames
        indices = np.linspace(0, video_views - 1, grid_views, dtype=int).tolist()
    
    return all_frames[:, indices, :, :]


def create_grid_from_video(
    turntable_image: np.ndarray,
    video_views: int,
    grid_rows: int = 6,
    grid_cols: int = 6,
    segments: list = None
) -> tuple:
    """
    Create grid image from turntable video frames.
    
    Args:
        turntable_image: [H, V*W, C] concatenated video frames
        video_views: Number of video frames
        grid_rows: Number of grid rows (default 6)
        grid_cols: Number of grid columns (default 6)
        segments: Optional segment info to exclude hold frames from grid
    
    Returns:
        (grid_image, all_frames, h_img) tuple
    """
    from einops import rearrange
    
    h_img = turntable_image.shape[0]
    w_per_view = turntable_image.shape[1] // video_views
    all_frames = turntable_image.reshape(h_img, video_views, w_per_view, 3)
    
    # Subsample for grid (exclude hold frames if segments provided)
    grid_views = grid_rows * grid_cols
    grid_frames = subsample_frames_for_grid(all_frames, video_views, grid_views, segments)
    
    # Rearrange to grid layout
    grid_image = rearrange(grid_frames, "h (rows cols) w c -> (rows h) (cols w) c", 
                          rows=grid_rows, cols=grid_cols)
    
    return grid_image, all_frames, h_img


# =============================================================================
# Angle-Proportional Frame Distribution
# =============================================================================

# Camera azimuth angles (degrees) - measured from physical setup
# Actual camera azimuths from dataset (computed from extrinsics)
MOUSE_CAMERA_AZIMUTHS = {
    0: -123.0,  # Back-left
    1: +56.0,   # Right-front
    2: +3.9,    # Front (almost center)
    3: +101.3,  # Right
    4: -54.0,   # Left
    5: +154.1,  # Back-right
}


def get_angle_proportional_frames(camera_order: List[int], total_frames: int = 180) -> List[int]:
    """
    Calculate frame counts proportional to angular gaps between cameras.
    
    This provides smoother visual rotation by allocating more frames to
    larger angular gaps between cameras.
    
    Args:
        camera_order: Camera traversal order (e.g., [0, 5, 3, 1, 2, 4])
        total_frames: Total number of frames to distribute
        
    Returns:
        List of frame counts for each segment (same length as camera_order)
        
    Example:
        >>> get_angle_proportional_frames([0, 5, 3, 1, 2, 4], total_frames=180)
        [25, 16, 16, 31, 17, 75]  # More frames for larger angular gaps
    """
    # Calculate angular gap for each segment
    angle_gaps = []
    for i in range(len(camera_order)):
        curr_cam = camera_order[i]
        next_cam = camera_order[(i + 1) % len(camera_order)]
        
        curr_az = MOUSE_CAMERA_AZIMUTHS[curr_cam]
        next_az = MOUSE_CAMERA_AZIMUTHS[next_cam]
        
        # Compute shortest angular difference
        diff = next_az - curr_az
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        angle_gaps.append(abs(diff))
    
    # Proportional distribution
    total_angle = sum(angle_gaps)
    frames_per_segment = [int(total_frames * gap / total_angle) for gap in angle_gaps]
    
    # Distribute rounding remainder
    remainder = total_frames - sum(frames_per_segment)
    for i in range(remainder):
        frames_per_segment[i % len(frames_per_segment)] += 1
    
    return frames_per_segment


def get_uniform_frames(camera_order: List[int], total_frames: int = 180) -> List[int]:
    """
    Calculate uniform frame counts (equal frames per segment).
    
    Args:
        camera_order: Camera traversal order
        total_frames: Total number of frames
        
    Returns:
        List of frame counts (all equal or nearly equal)
    """
    n_segments = len(camera_order)
    base_frames = total_frames // n_segments
    remainder = total_frames % n_segments
    
    frames_per_segment = [base_frames] * n_segments
    for i in range(remainder):
        frames_per_segment[i] += 1
    
    return frames_per_segment

# =============================================================================
# Dynamic Camera Order Computation
# =============================================================================

def compute_camera_order_from_extrinsics(c2ws: np.ndarray, direction: str = 'ccw') -> List[int]:
    """
    Compute camera traversal order from actual camera extrinsics.
    
    This replaces hardcoded MOUSE_CAMERA_ORDER with dynamic computation
    based on actual camera positions in the dataset.
    
    Args:
        c2ws: Camera-to-world matrices [num_cams, 4, 4]
        direction: 'ccw' for counter-clockwise, 'cw' for clockwise
        
    Returns:
        List of camera indices sorted by azimuth angle
        
    Example:
        >>> c2ws = batch['c2w'].cpu().numpy()
        >>> order = compute_camera_order_from_extrinsics(c2ws)
        >>> # Use order instead of MOUSE_CAMERA_ORDER
    """
    num_cams = c2ws.shape[0]
    
    # Extract camera positions (translation from c2w)
    positions = c2ws[:, :3, 3]  # [num_cams, 3]
    
    # Compute azimuth angles (angle from +Y axis in XY plane)
    # azimuth = atan2(x, y) for top-down view where +Y is forward
    azimuths = np.degrees(np.arctan2(positions[:, 0], positions[:, 1]))
    
    # Sort by azimuth (ascending = CCW from most negative to positive)
    sorted_indices = np.argsort(azimuths)
    
    if direction == 'cw':
        sorted_indices = sorted_indices[::-1]
    
    return sorted_indices.tolist()


def get_dynamic_camera_order(c2ws: np.ndarray, config: dict = None) -> List[int]:
    """
    Get camera order, preferring dynamic computation over hardcoded values.
    
    Args:
        c2ws: Camera-to-world matrices [num_cams, 4, 4]
        config: Optional config dict with turntable settings
        
    Returns:
        Camera order list
    """
    if config is not None:
        turntable_cfg = config.get('visualization', {}).get('turntable', {})
        
        # If explicit camera_order is specified and not 'auto', use it
        explicit_order = turntable_cfg.get('camera_order', None)
        if explicit_order is not None and explicit_order != 'auto':
            return explicit_order
        
        # Get direction preference
        direction = turntable_cfg.get('rotation_direction', 'ccw')
    else:
        direction = 'ccw'
    
    # Compute dynamically from extrinsics
    return compute_camera_order_from_extrinsics(c2ws, direction)


def compute_camera_convergence_center(c2ws: np.ndarray) -> np.ndarray:
    """
    Compute the convergence center from multiple camera c2w matrices.
    
    This finds the point that minimizes the sum of squared distances
    to all camera viewing rays (least squares intersection).
    
    Args:
        c2ws: [N, 4, 4] camera-to-world matrices
    
    Returns:
        center: [3,] the convergence center point
    """
    N = c2ws.shape[0]
    
    # Extract camera positions and viewing directions
    cam_positions = c2ws[:, :3, 3]  # [N, 3] - camera positions
    
    # Forward direction is -Z in camera space (OpenCV convention)
    forward_dirs = -c2ws[:, :3, 2]  # [N, 3] - viewing directions
    forward_dirs = forward_dirs / np.linalg.norm(forward_dirs, axis=1, keepdims=True)
    
    # Least squares intersection of rays
    # For each ray: P = origin + t * direction
    # Find point closest to all rays: A @ center = b
    # where A = sum(I - d*d^T), b = sum((I - d*d^T) @ origin)
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for i in range(N):
        origin = cam_positions[i]
        d = forward_dirs[i]
        I_minus_ddT = np.eye(3) - np.outer(d, d)
        A += I_minus_ddT
        b += I_minus_ddT @ origin
    
    # Solve for center
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to average camera position
        center = cam_positions.mean(axis=0)
    
    return center


def add_angle_overlay_to_grid(
    grid_image: np.ndarray,
    grid_rows: int = 6,
    grid_cols: int = 6,
    font_scale: float = 0.5,
    start_angle: float = 0.0,
) -> np.ndarray:
    """
    Add angle overlay (azimuth) to each cell in the grid.
    
    Args:
        grid_image: [H, W, 3] uint8 grid image
        grid_rows: Number of rows
        grid_cols: Number of columns
        font_scale: Font scale for overlay text
        start_angle: Starting angle (default 0)
    
    Returns:
        Grid image with angle overlays on bottom-left of each cell
    """
    import cv2
    
    result = grid_image.copy()
    h, w = grid_image.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thick = 1
    
    total_frames = grid_rows * grid_cols
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            frame_idx = row * grid_cols + col
            # Calculate angle for this frame (0 to 360 degrees)
            angle = start_angle + (frame_idx / total_frames) * 360
            angle = angle % 360  # Normalize to 0-360
            
            # Position in cell (bottom-left corner)
            x = col * cell_w + 5
            y = (row + 1) * cell_h - 5
            
            # Draw angle text
            text = str(int(angle))
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thick)
            
            # Background rectangle for readability
            cv2.rectangle(result, (x-2, y-th-2), (x+tw+2, y+2), (0, 0, 0), -1)
            cv2.putText(result, text, (x, y), font, font_scale, (255, 255, 255), font_thick)
    
    return result
