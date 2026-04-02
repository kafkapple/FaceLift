"""
Camera Normalization Module for FaceLift Compatibility

FaceLift pretrained model expects:
- fx = fy = 549.0
- Camera distance ~ 2.7 units from origin
- Object centered at origin
"""

import numpy as np
from typing import Dict, List, Tuple


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert world-to-camera to camera-to-world matrix."""
    return np.linalg.inv(w2c)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Convert camera-to-world to world-to-camera matrix."""
    return np.linalg.inv(c2w)


def get_camera_position(w2c: np.ndarray) -> np.ndarray:
    """Get camera position in world coordinates from w2c matrix."""
    c2w = w2c_to_c2w(w2c)
    return c2w[:3, 3]


def normalize_cameras(
    frames: List[Dict],
    target_fx: float = 549.0,
    target_distance: float = 2.7,
) -> Tuple[List[Dict], Dict]:
    """
    Normalize camera parameters to FaceLift convention.
    
    Steps:
    1. Normalize fx, fy to target_fx
    2. Scale camera translations to target_distance
    3. Record transform info for reference
    
    Args:
        frames: List of frame dicts with fx, fy, cx, cy, w2c
        target_fx: Target focal length (default 549.0)
        target_distance: Target camera distance from origin (default 2.7)
    
    Returns:
        (normalized_frames, transform_info)
    """
    if not frames:
        return frames, {}
    
    # Step 1: Compute current average fx and camera distance
    original_fx_list = [f["fx"] for f in frames]
    original_fy_list = [f["fy"] for f in frames]
    avg_original_fx = np.mean(original_fx_list)
    
    # Get camera positions and compute average distance from origin
    camera_positions = []
    for frame in frames:
        w2c = np.array(frame["w2c"])
        pos = get_camera_position(w2c)
        camera_positions.append(pos)
    
    camera_positions = np.array(camera_positions)
    distances = np.linalg.norm(camera_positions, axis=1)
    avg_distance = np.mean(distances)
    
    # Step 2: Compute scale factors
    fx_scale = target_fx / avg_original_fx
    distance_scale = target_distance / avg_distance if avg_distance > 0 else 1.0
    
    # Step 3: Apply normalization
    normalized_frames = []
    for i, frame in enumerate(frames):
        new_frame = frame.copy()
        
        # Normalize intrinsics
        new_frame["fx"] = target_fx
        new_frame["fy"] = frame["fy"] * fx_scale  # Scale fy proportionally
        # cx, cy remain unchanged (already at 256 if using FORCE_256)
        
        # Normalize extrinsics (translation only, preserve rotation)
        w2c = np.array(frame["w2c"])
        c2w = w2c_to_c2w(w2c)
        
        # Scale translation
        c2w[:3, 3] = c2w[:3, 3] * distance_scale
        
        # Convert back to w2c
        new_w2c = c2w_to_w2c(c2w)
        new_frame["w2c"] = new_w2c.tolist()
        
        normalized_frames.append(new_frame)
    
    # Record transform info
    transform_info = {
        "original_avg_fx": float(avg_original_fx),
        "original_avg_distance": float(avg_distance),
        "fx_scale": float(fx_scale),
        "distance_scale": float(distance_scale),
        "target_fx": target_fx,
        "target_distance": target_distance,
    }
    
    return normalized_frames, transform_info


def normalize_single_frame(
    frame: Dict,
    fx_scale: float,
    distance_scale: float,
    target_fx: float = 549.0,
) -> Dict:
    """Normalize a single frame using pre-computed scales."""
    new_frame = frame.copy()
    
    # Normalize intrinsics
    new_frame["fx"] = target_fx
    new_frame["fy"] = frame["fy"] * fx_scale
    
    # Normalize extrinsics
    w2c = np.array(frame["w2c"])
    c2w = w2c_to_c2w(w2c)
    c2w[:3, 3] = c2w[:3, 3] * distance_scale
    new_w2c = c2w_to_w2c(c2w)
    new_frame["w2c"] = new_w2c.tolist()
    
    return new_frame


def normalize_camera_distance(
    cameras_w2c: List[np.ndarray],
    target_distance: float = 2.7,
) -> Tuple[List[np.ndarray], float]:
    """Normalize camera distance without modifying intrinsics.

    Unlike normalize_cameras() which scales both intrinsics and distance,
    this function only scales the translation component of w2c matrices.
    Use when intrinsics are already correct (e.g., after PP-centering where
    cx=cy=256 is guaranteed by the crop step).

    Args:
        cameras_w2c: List of (4, 4) world-to-camera matrices.
        target_distance: Target average camera distance from origin.

    Returns:
        (normalized_w2c_list, spatial_scale) tuple.
    """
    distances = []
    for w2c in cameras_w2c:
        c2w = w2c_to_c2w(w2c)
        cam_pos = c2w[:3, 3]
        distances.append(np.linalg.norm(cam_pos))

    mean_dist = np.mean(distances)
    spatial_scale = target_distance / mean_dist if mean_dist > 0 else 1.0

    norm_w2c_list = []
    for w2c in cameras_w2c:
        w2c_norm = w2c.copy()
        w2c_norm[:3, 3] *= spatial_scale
        norm_w2c_list.append(w2c_norm)

    return norm_w2c_list, spatial_scale
