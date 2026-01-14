"""
Preprocessing Utilities for Mouse Data

Camera normalization and coordinate system transformations
for adapting mouse capture data to FaceLift/GS-LRM format.

Key functions:
- normalize_camera_distance: Scale cameras to fixed radius (2.7)
- normalize_camera_distance_with_intrinsics: Scale with fx/fy adjustment
- normalize_cameras_to_y_up: Align up direction to Y-axis
- normalize_cameras_to_z_up: Align up direction to Z-axis
- get_bg_color: Parse background color from config
"""

import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image


def pil_to_np(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy array, preserving RGBA alpha channel."""
    if pil_image.mode == "RGBA":
        r, g, b, a = pil_image.split()
        r, g, b, a = np.asarray(r), np.asarray(g), np.asarray(b), np.asarray(a)
        image = np.stack([r, g, b, a], axis=2)
    else:
        image = np.asarray(pil_image)
    return image


def normalize_camera_distance(
    c2w_matrices: np.ndarray, 
    target_distance: float = 2.7
) -> np.ndarray:
    """
    Normalize camera distances to a fixed radius from origin.

    The pretrained GS-LRM was trained with cameras at radius=2.7.
    This ensures consistent behavior when using different camera setups.

    Args:
        c2w_matrices: Camera-to-world matrices [N, 4, 4]
        target_distance: Target distance from origin (default: 2.7)

    Returns:
        Normalized c2w matrices [N, 4, 4] with uniform camera distance
    """
    normalized_c2ws = []

    for c2w in c2w_matrices:
        c2w_new = c2w.copy()
        cam_pos = c2w[:3, 3]
        current_distance = np.linalg.norm(cam_pos)

        if current_distance > 1e-6:
            scale = target_distance / current_distance
            c2w_new[:3, 3] = cam_pos * scale

        normalized_c2ws.append(c2w_new)

    return np.stack(normalized_c2ws, axis=0)


def normalize_camera_distance_with_intrinsics(
    c2w_matrices: np.ndarray, 
    fxfycxcy: np.ndarray,
    target_distance: float = 2.7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize camera distances AND adjust intrinsics accordingly.
    
    When camera distance changes, fx/fy must also change proportionally
    to maintain the same projected image size.
    
    Mathematical basis:
    - Perspective projection: pixel_x = fx * (X/Z) + cx
    - If distance doubles, object appears half size
    - To compensate: fx, fy must also double
    - Formula: new_fx = old_fx * (new_distance / old_distance)
    
    Args:
        c2w_matrices: Camera-to-world matrices [N, 4, 4]
        fxfycxcy: Intrinsic parameters [N, 4] as [fx, fy, cx, cy]
        target_distance: Target distance from origin (default: 2.7)
    
    Returns:
        Tuple of (normalized_c2ws, adjusted_intrinsics)
    """
    normalized_c2ws = []
    adjusted_intrinsics = []
    
    for i, c2w in enumerate(c2w_matrices):
        c2w_new = c2w.copy()
        intrinsics_new = fxfycxcy[i].copy()
        
        cam_pos = c2w[:3, 3]
        current_distance = np.linalg.norm(cam_pos)
        
        if current_distance > 1e-6:
            scale = target_distance / current_distance
            c2w_new[:3, 3] = cam_pos * scale
            intrinsics_new[0] *= scale  # fx
            intrinsics_new[1] *= scale  # fy
        
        normalized_c2ws.append(c2w_new)
        adjusted_intrinsics.append(intrinsics_new)
    
    return np.stack(normalized_c2ws, axis=0), np.stack(adjusted_intrinsics, axis=0)


def _compute_rotation_matrix(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """Compute rotation matrix from one vector to another using Rodrigues formula."""
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)
    
    rotation_axis = np.cross(from_vec, to_vec)
    axis_norm = np.linalg.norm(rotation_axis)
    
    if axis_norm < 1e-6:
        # Vectors are parallel
        if np.dot(from_vec, to_vec) < 0:
            # 180 degree rotation - find perpendicular axis
            if abs(from_vec[0]) < 0.9:
                perp = np.cross(from_vec, np.array([1, 0, 0]))
            else:
                perp = np.cross(from_vec, np.array([0, 1, 0]))
            perp = perp / np.linalg.norm(perp)
            # Rotation by pi around perpendicular axis
            K = np.array([
                [0, -perp[2], perp[1]],
                [perp[2], 0, -perp[0]],
                [-perp[1], perp[0], 0]
            ], dtype=np.float32)
            return -np.eye(3) + 2 * np.outer(perp, perp)
        else:
            return np.eye(3, dtype=np.float32)
    
    rotation_axis = rotation_axis / axis_norm
    cos_angle = np.clip(np.dot(from_vec, to_vec), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Rodrigues formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ], dtype=np.float32)
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def _estimate_up_direction(c2w_matrices: np.ndarray) -> np.ndarray:
    """Estimate up direction from camera positions using PCA."""
    positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])
    center = np.mean(positions, axis=0)
    centered = positions - center
    
    if len(c2w_matrices) >= 3:
        # Use PCA to find orbit plane normal
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        min_idx = np.argmin(eigenvalues.real)
        up_direction = eigenvectors[:, min_idx].real
    else:
        # Fallback to camera up vectors average
        avg_cam_up = np.mean([-c2w[:3, 1] for c2w in c2w_matrices], axis=0)
        up_direction = avg_cam_up / np.linalg.norm(avg_cam_up)
    
    # Ensure up direction aligns with camera up vectors
    avg_cam_up = np.mean([-c2w[:3, 1] for c2w in c2w_matrices], axis=0)
    if np.dot(up_direction, avg_cam_up) < 0:
        up_direction = -up_direction
    
    return up_direction / np.linalg.norm(up_direction)


def normalize_cameras_to_y_up(
    c2w_matrices: np.ndarray, 
    up_direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize camera poses so that the up direction aligns with Y-axis.

    Args:
        c2w_matrices: Camera-to-world matrices [N, 4, 4]
        up_direction: Actual "up" direction in world coordinates. 
                      If None, estimate from cameras.

    Returns:
        Normalized c2w matrices [N, 4, 4] with Y-up alignment
    """
    if up_direction is None:
        up_direction = _estimate_up_direction(c2w_matrices)
    
    up_direction = up_direction / np.linalg.norm(up_direction)
    target_up = np.array([0.0, 1.0, 0.0])
    
    R_align = _compute_rotation_matrix(up_direction, target_up)
    
    normalized_c2ws = []
    for c2w in c2w_matrices:
        c2w_new = c2w.copy()
        c2w_new[:3, :3] = R_align @ c2w[:3, :3]
        c2w_new[:3, 3] = R_align @ c2w[:3, 3]
        normalized_c2ws.append(c2w_new)
    
    return np.stack(normalized_c2ws, axis=0)


def normalize_cameras_to_z_up(
    c2w_matrices: np.ndarray, 
    up_direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Normalize camera poses so that the up direction aligns with Z-axis.

    Args:
        c2w_matrices: Camera-to-world matrices [N, 4, 4]
        up_direction: Actual "up" direction in world coordinates.
                      If None, estimate from cameras.

    Returns:
        Normalized c2w matrices [N, 4, 4] with Z-up alignment
    """
    if up_direction is None:
        up_direction = _estimate_up_direction(c2w_matrices)
    
    up_direction = up_direction / np.linalg.norm(up_direction)
    target_up = np.array([0.0, 0.0, 1.0])
    
    R_align = _compute_rotation_matrix(up_direction, target_up)
    
    normalized_c2ws = []
    for c2w in c2w_matrices:
        c2w_new = c2w.copy()
        c2w_new[:3, :3] = R_align @ c2w[:3, :3]
        c2w_new[:3, 3] = R_align @ c2w[:3, 3]
        normalized_c2ws.append(c2w_new)
    
    return np.stack(normalized_c2ws, axis=0)


# Background color constants
BG_COLORS = {
    'white': np.array([1.0, 1.0, 1.0], dtype=np.float32),
    'black': np.array([0.0, 0.0, 0.0], dtype=np.float32),
    'gray': np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


def get_bg_color(bg_color_config: Union[str, float]) -> torch.Tensor:
    """
    Generate background color tensor from configuration.
    
    Args:
        bg_color_config: One of:
            - 'white', 'black', 'gray': Predefined colors
            - 'random': Random RGB color
            - 'three_choices': Random choice from white/black/gray
            - float in [0, 1]: Grayscale value
    
    Returns:
        torch.Tensor of shape [3] with RGB values in [0, 1]
    """
    if isinstance(bg_color_config, str):
        if bg_color_config in BG_COLORS:
            bg_color = BG_COLORS[bg_color_config]
        elif bg_color_config == 'random':
            bg_color = np.random.rand(3).astype(np.float32)
        elif bg_color_config == 'three_choices':
            bg_color = random.choice(list(BG_COLORS.values()))
        else:
            raise ValueError(f"Unsupported background color: '{bg_color_config}'")
    elif isinstance(bg_color_config, (int, float)):
        if not 0 <= bg_color_config <= 1:
            raise ValueError(f"Background color must be in [0, 1], got {bg_color_config}")
        bg_color = np.array([bg_color_config] * 3, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported background color type: {type(bg_color_config)}")

    return torch.from_numpy(bg_color)


# Convenience class for preprocessing configuration
class PreprocessingConfig:
    """Configuration for camera preprocessing."""
    
    def __init__(
        self,
        normalize_cameras: bool = False,
        target_camera_distance: float = 2.7,
        normalize_to_z_up: bool = True,
        adjust_intrinsics: bool = True,
    ):
        self.normalize_cameras = normalize_cameras
        self.target_camera_distance = target_camera_distance
        self.normalize_to_z_up = normalize_to_z_up
        self.adjust_intrinsics = adjust_intrinsics
    
    @classmethod
    def from_config(cls, config) -> 'PreprocessingConfig':
        """Create from training config."""
        mouse_config = getattr(config, 'mouse', {})
        if not isinstance(mouse_config, dict):
            mouse_config = dict(mouse_config) if hasattr(mouse_config, '__iter__') else {}
        
        return cls(
            normalize_cameras=mouse_config.get('normalize_cameras', False),
            target_camera_distance=mouse_config.get('target_camera_distance', 2.7),
            normalize_to_z_up=mouse_config.get('normalize_to_z_up', True),
            adjust_intrinsics=mouse_config.get('adjust_intrinsics', True),
        )


def preprocess_cameras(
    c2w_matrices: np.ndarray,
    fxfycxcy: np.ndarray,
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply full preprocessing pipeline to cameras.
    
    Args:
        c2w_matrices: Camera-to-world matrices [N, 4, 4]
        fxfycxcy: Intrinsic parameters [N, 4]
        config: Preprocessing configuration
    
    Returns:
        Tuple of (processed_c2ws, processed_intrinsics)
    """
    processed_c2ws = c2w_matrices.copy()
    processed_intrinsics = fxfycxcy.copy()
    
    if not config.normalize_cameras:
        return processed_c2ws, processed_intrinsics
    
    # Step 1: Normalize up direction
    if config.normalize_to_z_up:
        processed_c2ws = normalize_cameras_to_z_up(processed_c2ws)
    else:
        processed_c2ws = normalize_cameras_to_y_up(processed_c2ws)
    
    # Step 2: Normalize camera distance
    if config.adjust_intrinsics:
        processed_c2ws, processed_intrinsics = normalize_camera_distance_with_intrinsics(
            processed_c2ws, processed_intrinsics, config.target_camera_distance
        )
    else:
        processed_c2ws = normalize_camera_distance(
            processed_c2ws, config.target_camera_distance
        )
    
    return processed_c2ws, processed_intrinsics
