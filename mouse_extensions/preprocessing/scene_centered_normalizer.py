"""
Scene-Centered Camera Normalization (Wrapper)

Wrapper module for narrow-baseline rigs (e.g., sdannce rat) where:
- Camera centroid != scene center (cameras above/around, subject below/inside)
- Mouse-style camera-centroid recentering would push the scene far from origin

This module recenters cameras around an EXTERNALLY-PROVIDED scene center
(e.g., from com3d.mat for rat sdannce data), then applies uniform distance scaling.

Design principles:
- Open/Closed: does NOT modify existing _normalize_cameras_batch
- Backward compat: not used unless explicitly enabled in preset config
- Pure function: stateless, no side effects on input

Created: 2026-04-07 (audit-driven plan v2)
"""

import numpy as np
from typing import Dict, List, Optional, Sequence


def normalize_cameras_scene_centric(
    cam_params_list: List[Dict],
    scene_center: Sequence[float],
    target_distance: float = 2.7,
) -> List[Dict]:
    """Recenter cameras around an external scene center, then uniform scale.

    Unlike `UnifiedPreprocessor._normalize_cameras_batch` which recenters around
    the camera centroid (assumes mouse-style 360° rig with subject ≈ centroid),
    this function recenters around an externally-provided scene center.

    Use case: sdannce rat data where 6 cameras cluster on the ceiling and
    look down at the rat. Camera centroid is far from the rat (the actual
    scene of interest).

    Args:
        cam_params_list: List of camera parameter dicts with 'w2c' key
                         (4x4 world-to-camera matrices, list-of-lists or array).
        scene_center: 3D world coordinates of the scene center (rat position).
                      Typically from com3d.mat median.
        target_distance: Target mean camera-to-origin distance after scaling.
                         Default 2.7 (FaceLift convention).

    Returns:
        New list of camera parameter dicts with updated 'w2c' matrices.
        Original list is NOT modified.

    Math:
        1. Translate: cam_pos_new = cam_pos_old - scene_center
           (scene now at origin)
        2. Compute mean distance from new origin (= mean cam-to-scene distance)
        3. Uniform scale: cam_pos_final = cam_pos_new * (target_distance / mean_dist)
           (scaling preserves baseline ratios → preserves parallax structure)

    Properties:
        - Translation invariance: pairwise camera angles UNCHANGED (around scene center)
        - Uniform scale: relative geometry preserved (baselines scale together)
        - Result: scene at (0,0,0), cameras at mean distance target_distance
    """
    if not cam_params_list:
        return []

    scene_center = np.asarray(scene_center, dtype=np.float64)
    if scene_center.shape != (3,):
        raise ValueError(f"scene_center must be (3,), got {scene_center.shape}")

    # Step 1: Extract camera positions from w2c matrices
    positions = []
    for params in cam_params_list:
        w2c = np.asarray(params["w2c"], dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        positions.append(c2w[:3, 3].copy())
    positions = np.array(positions)  # (N, 3)

    # Step 2: Translate so scene center → origin
    centered = positions - scene_center  # (N, 3)

    # Step 3: Compute uniform scale factor
    distances = np.linalg.norm(centered, axis=1)  # (N,)
    mean_dist = float(distances.mean())
    if mean_dist < 1e-9:
        raise ValueError(
            f"Cameras are at the scene center (mean distance {mean_dist:.2e}). "
            "Cannot normalize."
        )
    scale = target_distance / mean_dist

    # Step 4: Apply uniform scale + write back
    final_positions = centered * scale

    new_params_list = []
    for i, params in enumerate(cam_params_list):
        new_params = dict(params)  # shallow copy (other keys unchanged)
        w2c = np.asarray(params["w2c"], dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = final_positions[i]
        new_w2c = np.linalg.inv(new_c2w)
        new_params["w2c"] = new_w2c.tolist()
        new_params_list.append(new_params)

    return new_params_list


def compute_scene_center_from_com3d(
    com3d_array: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """Compute a single scene center from per-frame center-of-mass positions.

    Args:
        com3d_array: (N, 3) per-frame 3D positions (e.g., rat COM from sdannce).
        method: 'median' (robust to outliers) or 'mean' (sensitive but unbiased).

    Returns:
        (3,) scene center in same coordinate system as input.
    """
    com3d_array = np.asarray(com3d_array, dtype=np.float64)
    if com3d_array.ndim != 2 or com3d_array.shape[1] != 3:
        raise ValueError(f"com3d_array must be (N, 3), got {com3d_array.shape}")

    if method == "median":
        return np.median(com3d_array, axis=0)
    elif method == "mean":
        return np.mean(com3d_array, axis=0)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'median' or 'mean'.")
