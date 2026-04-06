"""
Camera Geometry Utilities

Functions for computing scene center from camera configurations.
Used by mouse_dataset.py for recentering cameras to the convergence point.
"""

import numpy as np


def compute_convergence_point(c2w_matrices: np.ndarray) -> np.ndarray:
    """
    Compute the convergence point where camera viewing rays intersect.

    Uses least-squares to find the 3D point that minimizes the sum of
    squared distances to all camera viewing rays. This is the point that
    cameras are collectively "looking at".

    For origin-centered data (e.g., preprocessed mouse), returns ~[0,0,0].
    For offset data (e.g., s-DANNCE rat), returns the actual scene center.

    Mathematical basis:
        For ray (p_i, d_i), the projection matrix onto the ray complement is:
            P_i = I - d_i @ d_i^T
        The convergence point x minimizes sum_i |P_i(x - p_i)|^2
        Solution: x = (sum P_i)^{-1} @ (sum P_i @ p_i)

    Args:
        c2w_matrices: Camera-to-world matrices [N, 4, 4]

    Returns:
        Convergence point [3] in world coordinates.
        Falls back to camera centroid if rays are near-parallel (singular system).
    """
    cam_positions = c2w_matrices[:, :3, 3]  # [N, 3]
    # Forward direction = 3rd column of rotation (Z-axis in OpenCV c2w)
    forward_dirs = c2w_matrices[:, :3, 2]   # [N, 3]
    # Normalize directions
    norms = np.linalg.norm(forward_dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    forward_dirs = forward_dirs / norms

    # Build least-squares system: (sum P_i) x = sum P_i @ p_i
    # where P_i = I - d_i @ d_i^T
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I3 = np.eye(3)
    for i in range(len(cam_positions)):
        d = forward_dirs[i]
        P = I3 - np.outer(d, d)
        A += P
        b += P @ cam_positions[i]

    try:
        convergence = np.linalg.solve(A, b)
        # Sanity check: convergence should be within reasonable range of cameras
        cam_center = cam_positions.mean(axis=0)
        cam_spread = np.linalg.norm(cam_positions - cam_center, axis=1).max()
        conv_to_center = np.linalg.norm(convergence - cam_center)
        if conv_to_center > max(cam_spread, 1.0) * 20:
            return cam_center
        return convergence
    except np.linalg.LinAlgError:
        # Singular matrix = parallel rays, fall back to centroid
        return cam_positions.mean(axis=0)
