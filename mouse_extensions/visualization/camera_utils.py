"""Camera utilities for visualization.

Provides camera order computation, convergence center calculation,
and turntable camera generation for multi-view visualization.

Consolidated from turntable_config.py and gaussians_renderer.py (2026-03-23).

Camera conventions (canonical spec: Obsidian COORDINATE_SYSTEMS.md):
  - All cameras use OpenCV convention: X-right, Y-down, Z-forward
  - World coordinate: Z-up
  - Turntable: azimuth from +X axis (cos->x, sin->y), CW in math = physical CCW
  - Camera order: azimuth from +Y axis (atan2(x,y)), ascending = CCW
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import torch

TrajectoryMode = Literal["turntable", "spiral", "figure8", "arc", "dataset_cameras"]


# ---------------------------------------------------------------------------
# Camera order computation (from turntable_config.py, 2026-03-23)
# ---------------------------------------------------------------------------

def compute_camera_order_from_extrinsics(
    c2ws: np.ndarray, direction: str = "ccw"
) -> List[int]:
    """Compute camera traversal order from actual camera extrinsics.

    Replaces hardcoded camera orders with dynamic computation based on
    actual camera positions in the dataset.

    Args:
        c2ws: Camera-to-world matrices [num_cams, 4, 4].
        direction: 'ccw' for counter-clockwise, 'cw' for clockwise.

    Returns:
        List of camera indices sorted by azimuth angle.
    """
    positions = c2ws[:, :3, 3]  # [num_cams, 3]
    # Azimuth = angle from +Y axis in XY plane (top-down view)
    azimuths = np.degrees(np.arctan2(positions[:, 0], positions[:, 1]))
    sorted_indices = np.argsort(azimuths)
    if direction == "cw":
        sorted_indices = sorted_indices[::-1]
    return sorted_indices.tolist()


def get_dynamic_camera_order(
    c2ws: np.ndarray, config: Optional[dict] = None
) -> List[int]:
    """Get camera order, preferring dynamic computation over hardcoded values.

    Args:
        c2ws: Camera-to-world matrices [num_cams, 4, 4].
        config: Optional config dict with visualization.turntable settings.

    Returns:
        Camera order list.
    """
    if config is not None:
        turntable_cfg = config.get("visualization", {}).get("turntable", {})
        explicit_order = turntable_cfg.get("camera_order", None)
        if explicit_order is not None and explicit_order != "auto":
            return explicit_order
        direction = turntable_cfg.get("rotation_direction", "ccw")
    else:
        direction = "ccw"
    return compute_camera_order_from_extrinsics(c2ws, direction)


# ---------------------------------------------------------------------------
# Convergence center (from turntable_config.py, 2026-03-23)
# ---------------------------------------------------------------------------

def compute_camera_convergence_center(c2ws: np.ndarray) -> np.ndarray:
    """Compute the convergence center from multiple camera c2w matrices.

    Finds the point that minimizes the sum of squared distances to all
    camera viewing rays (least-squares ray intersection).

    Args:
        c2ws: [N, 4, 4] camera-to-world matrices.

    Returns:
        center: [3,] the convergence center point.
    """
    N = c2ws.shape[0]
    cam_positions = c2ws[:, :3, 3]

    # Forward direction is -Z in camera space (OpenCV convention)
    forward_dirs = -c2ws[:, :3, 2]
    forward_dirs = forward_dirs / np.linalg.norm(forward_dirs, axis=1, keepdims=True)

    # Least-squares intersection: A @ center = b
    # where A = sum(I - d*d^T), b = sum((I - d*d^T) @ origin)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(N):
        d = forward_dirs[i]
        I_minus_ddT = np.eye(3) - np.outer(d, d)
        A += I_minus_ddT
        b += I_minus_ddT @ cam_positions[i]

    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        center = cam_positions.mean(axis=0)
    return center


# ---------------------------------------------------------------------------
# Turntable camera generation (from gaussians_renderer.py, 2026-03-23)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_turntable_cameras(
    hfov: float = 50,
    num_views: int = 8,
    w: int = 512,
    h: int = 512,
    radius: float = 2.7,
    elevation: float = 20,
    elevation_end: Optional[float] = None,
    trajectory_mode: TrajectoryMode = "turntable",
    up_vector: np.ndarray = np.array([0, 0, 1]),
    center: Optional[np.ndarray] = None,
    clockwise: bool = True,
    start_azimuth: float = 270.0,
):
    """Generate camera poses for visualization.

    Args:
        trajectory_mode: Camera path type
            - "turntable": Fixed elevation, 360 deg rotation (default)
            - "spiral": Elevation varies while rotating (1.5 rotations)
            - "figure8": Figure-8 pattern for diverse viewpoints
            - "arc": Single arc trajectory, varying elevation only
        elevation_end: End elevation for spiral/arc modes.
        center: 3D point for camera orbit center. If None, uses origin.
        clockwise: CW in math coords = physical CCW from above (default).
        start_azimuth: Starting azimuth in degrees. Default 270. Set to GT
            camera azimuth for seamless SLERP transition from GT view.

    Returns:
        (w, h, num_views, fxfycxcy, c2ws) where
        fxfycxcy: [num_views, 4], c2ws: [num_views, 4, 4].
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.asarray(center).flatten()[:3]

    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = (
        np.array([fx, fy, cx, cy]).reshape(1, 4).repeat(num_views, axis=0)
    )

    # Normalize start azimuth to [0, 360)
    sa = start_azimuth % 360

    # Generate azimuth and elevation based on trajectory mode
    if trajectory_mode == "turntable":
        if clockwise:
            azimuths = np.linspace(sa, sa - 360, num_views, endpoint=False)
        else:
            azimuths = np.linspace(sa, sa + 360, num_views, endpoint=False)
        elevations = np.ones(num_views) * elevation

    elif trajectory_mode == "spiral":
        azimuths = np.linspace(270, 630 + 360, num_views, endpoint=False)
        elev_end = elevation_end if elevation_end is not None else elevation + 40
        elevations = np.linspace(elevation, elev_end, num_views)

    elif trajectory_mode == "figure8":
        t = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        azimuths = 270 + 180 * np.sin(t)
        elev_end = elevation_end if elevation_end is not None else elevation + 40
        elev_amplitude = (elev_end - elevation) / 2
        elev_center = (elevation + elev_end) / 2
        elevations = elev_center + elev_amplitude * np.sin(2 * t)

    elif trajectory_mode == "arc":
        azimuths = np.ones(num_views) * 270
        elev_end = elevation_end if elevation_end is not None else 80
        elevations = np.linspace(elevation, elev_end, num_views)

    else:
        raise ValueError(f"Unknown trajectory mode: {trajectory_mode}")

    c2ws = []
    for elev, azim in zip(elevations, azimuths):
        elev, azim = np.deg2rad(elev), np.deg2rad(azim)
        z = radius * np.sin(elev)
        base = radius * np.cos(elev)
        x = base * np.cos(azim)
        y = base * np.sin(azim)
        cam_pos_rel = np.array([x, y, z])
        cam_pos = cam_pos_rel + center
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.stack((right, -up, forward), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)
    return w, h, num_views, fxfycxcy, c2ws


def get_turntable_with_dataset_views(
    dataset_c2ws: np.ndarray,
    dataset_fxfycxcy: np.ndarray,
    hfov: float = 50,
    num_turntable_views: int = 58,
    w: int = 384,
    h: int = 384,
    radius: float = 2.7,
    elevation: float = 20,
    original_resolution: Optional[int] = None,
):
    """Generate turntable cameras + exact dataset cameras.

    Dataset views are placed at positions 0..N-1, turntable views after.

    Args:
        original_resolution: If provided and different from w, dataset
            intrinsics will be scaled to match (w x h).

    Returns:
        (w, h, total_views, fxfycxcy, c2ws, dataset_view_indices).
    """
    num_dataset = dataset_c2ws.shape[0]
    total_views = num_dataset + num_turntable_views

    scaled_dataset_fxfycxcy = dataset_fxfycxcy.copy()
    if original_resolution is not None and original_resolution != w:
        scale = w / original_resolution
        scaled_dataset_fxfycxcy = scaled_dataset_fxfycxcy * scale

    _, _, _, turntable_fxfycxcy, turntable_c2ws = get_turntable_cameras(
        hfov=hfov, num_views=num_turntable_views, w=w, h=h,
        radius=radius, elevation=elevation, trajectory_mode="turntable",
    )

    fxfycxcy = np.concatenate([scaled_dataset_fxfycxcy, turntable_fxfycxcy], axis=0)
    c2ws = np.concatenate([dataset_c2ws, turntable_c2ws], axis=0)
    dataset_view_indices = list(range(num_dataset))

    return w, h, total_views, fxfycxcy, c2ws, dataset_view_indices
