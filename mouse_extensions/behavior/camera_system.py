"""Unified Camera System for BehaviorSplatter visualization.

Resolves the GT vs turntable camera convention mismatch:
  - GT cameras: from OpenCV calibration (w2c matrix), may NOT look at origin
  - Turntable cameras: always look at origin from spherical coordinates
  - Solution: re-orient GT camera to look at origin, preserving position

All cameras use OpenCV convention: X-right, Y-down, Z-forward.
World coordinate: Z-up.

Canonical spec: Obsidian docs/theory/COORDINATE_SYSTEMS.md -> "Camera Conventions".

Usage:
    from mouse_extensions.behavior.camera_system import CameraSystem

    cs = CameraSystem()
    c2w = cs.gt_camera_looking_at_origin(frame_dir, view_idx)
    orbit_c2ws = cs.turntable(n=60, elevation=20, start_c2w=c2w)
    interp = cs.interpolate(c2w_a, c2w_b, n_frames=15)
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_UP = np.array([0.0, 0.0, 1.0])
LOOK_AT_TARGET = np.array([0.0, 0.0, 0.0])
DEFAULT_RADIUS = 2.7
DEFAULT_HFOV = 50
DEFAULT_RESOLUTION = 512

# Anatomical constants: imported from SSOT (2026-03-23)
from mouse_extensions.constants import (  # noqa: E402
    MAMMAL_KP_COLORS, SKELETON_BONES, BODY_PARTS, BODY_PART_COLORS,
)


# ---------------------------------------------------------------------------
# Core camera functions
# ---------------------------------------------------------------------------

def create_opencv_c2w(
    position: np.ndarray,
    target: np.ndarray = LOOK_AT_TARGET,
    world_up: np.ndarray = WORLD_UP,
) -> np.ndarray:
    """Construct c2w matrix (OpenCV: X-right, Y-down, Z-forward) looking at target.

    This is the SINGLE function for camera matrix construction.
    Both GT cameras and turntable cameras should use this.
    """
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(world_up, dtype=np.float64)

    z_forward = target - position
    z_forward = z_forward / np.linalg.norm(z_forward)

    x_right = np.cross(world_up, z_forward)
    x_norm = np.linalg.norm(x_right)
    if x_norm < 1e-6:
        # Camera looking straight up/down — use fallback up
        x_right = np.cross(np.array([0, 1, 0]), z_forward)
        x_norm = np.linalg.norm(x_right)
    x_right = x_right / x_norm

    y_down = np.cross(z_forward, x_right)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_right
    c2w[:3, 1] = y_down
    c2w[:3, 2] = z_forward
    c2w[:3, 3] = position
    return c2w


def gt_camera_position(frame_dir: str, view_idx: int) -> np.ndarray:
    """Extract GT camera position from w2c matrix."""
    cam_path = str(Path(frame_dir) / "opencv_cameras.json")
    with open(cam_path) as f:
        cams = json.load(f)
    w2c = np.array(cams["frames"][view_idx]["w2c"], dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    return c2w[:3, 3]


def gt_camera_c2w_original(frame_dir: str, view_idx: int) -> np.ndarray:
    """Get GT camera's original c2w (may not look at origin)."""
    cam_path = str(Path(frame_dir) / "opencv_cameras.json")
    with open(cam_path) as f:
        cams = json.load(f)
    w2c = np.array(cams["frames"][view_idx]["w2c"], dtype=np.float64)
    return np.linalg.inv(w2c).astype(np.float32)


def gt_camera_looking_at_origin(frame_dir: str, view_idx: int) -> np.ndarray:
    """Get GT camera c2w, re-oriented to look at origin.

    This resolves the GT/turntable convention mismatch:
    same position as GT, but rotation constructed to look at [0,0,0].
    """
    pos = gt_camera_position(frame_dir, view_idx)
    return create_opencv_c2w(pos)


def gt_camera_fxfycxcy(frame_dir: str, view_idx: int) -> np.ndarray:
    """Get GT camera intrinsics as [fx, fy, cx, cy]."""
    cam_path = str(Path(frame_dir) / "opencv_cameras.json")
    with open(cam_path) as f:
        cams = json.load(f)
    cam = cams["frames"][view_idx]
    return np.array([cam["fx"], cam["fy"], cam["cx"], cam["cy"]], dtype=np.float32)


def spherical_to_position(azimuth_deg: float, elevation_deg: float,
                          radius: float) -> np.ndarray:
    """Convert spherical coordinates to 3D position (Z-up)."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = radius * np.cos(el) * np.sin(az)
    y = radius * np.cos(el) * np.cos(az)
    z = radius * np.sin(el)
    return np.array([x, y, z])


def position_to_spherical(position: np.ndarray) -> Tuple[float, float, float]:
    """Convert 3D position to (azimuth_deg, elevation_deg, radius)."""
    x, y, z = position
    radius = np.linalg.norm(position)
    elevation = np.degrees(np.arcsin(z / max(radius, 1e-8)))
    azimuth = np.degrees(np.arctan2(x, y))
    return azimuth, elevation, radius


# ---------------------------------------------------------------------------
# Camera trajectory generation
# ---------------------------------------------------------------------------

def turntable_cameras(
    n_frames: int,
    elevation: float = 20.0,
    radius: float = DEFAULT_RADIUS,
    hfov: float = DEFAULT_HFOV,
    resolution: int = DEFAULT_RESOLUTION,
    start_c2w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Generate turntable orbit cameras, optionally starting from a reference c2w.

    If start_c2w is provided, the orbit starts from the same azimuth as that camera.

    Returns:
        c2ws: (n_frames, 4, 4) camera-to-world matrices
        fxfycxcy: (n_frames, 4) intrinsics
        w, h: image dimensions
    """
    # Determine starting azimuth
    if start_c2w is not None:
        pos = start_c2w[:3, 3]
        start_az, _, _ = position_to_spherical(pos)
    else:
        start_az = 0.0

    azimuths = np.linspace(start_az, start_az + 360, n_frames, endpoint=False)

    fx = resolution / (2 * np.tan(np.radians(hfov) / 2.0))
    fxfycxcy = np.array([fx, fx, resolution / 2.0, resolution / 2.0], dtype=np.float32)
    fxfycxcy_arr = np.tile(fxfycxcy, (n_frames, 1))

    c2ws = np.zeros((n_frames, 4, 4), dtype=np.float32)
    for i, az in enumerate(azimuths):
        pos = spherical_to_position(az, elevation, radius)
        c2ws[i] = create_opencv_c2w(pos)

    return c2ws, fxfycxcy_arr, resolution, resolution


def elevation_arc_cameras(
    n_frames: int,
    start_elevation: float = -30.0,
    end_elevation: float = 80.0,
    azimuth: float = 0.0,
    radius: float = DEFAULT_RADIUS,
    hfov: float = DEFAULT_HFOV,
    resolution: int = DEFAULT_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Generate elevation arc cameras (fixed azimuth, varying elevation)."""
    elevations = np.linspace(start_elevation, end_elevation, n_frames)

    fx = resolution / (2 * np.tan(np.radians(hfov) / 2.0))
    fxfycxcy = np.array([fx, fx, resolution / 2.0, resolution / 2.0], dtype=np.float32)
    fxfycxcy_arr = np.tile(fxfycxcy, (n_frames, 1))

    c2ws = np.zeros((n_frames, 4, 4), dtype=np.float32)
    for i, el in enumerate(elevations):
        pos = spherical_to_position(azimuth, el, radius)
        c2ws[i] = create_opencv_c2w(pos)

    return c2ws, fxfycxcy_arr, resolution, resolution


def interpolate_cameras(
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    """SLERP rotation + linear translation interpolation.

    Returns (n_frames, 4, 4) array.
    """
    R_a = Rotation.from_matrix(c2w_a[:3, :3])
    R_b = Rotation.from_matrix(c2w_b[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([R_a, R_b]))

    ts = np.linspace(0, 1, n_frames)
    interp_R = slerp(ts).as_matrix()

    t_a, t_b = c2w_a[:3, 3], c2w_b[:3, 3]
    interp_t = (1 - ts[:, None]) * t_a + ts[:, None] * t_b

    out = np.zeros((n_frames, 4, 4), dtype=np.float32)
    out[:, :3, :3] = interp_R
    out[:, :3, 3] = interp_t
    out[:, 3, 3] = 1.0
    return out
