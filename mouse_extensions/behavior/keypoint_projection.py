"""Keypoint 3D→2D projection module.

Projects MAMMAL 3D keypoints to camera image space using proper camera
intrinsics and extrinsics (w2c). Reusable across all visualization modules.

Usage:
    from mouse_extensions.behavior.keypoint_projection import (
        project_keypoints, load_camera_params, KeypointProjector,
    )

    # Single frame
    uv = project_keypoints(kp_3d, w2c, fx, fy, cx, cy)

    # Batch (all 6 cameras)
    projector = KeypointProjector(camera_json_path)
    uv_all_cams = projector.project_frame(kp_3d)  # dict[cam_idx → (22, 2)]
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np


# MAMMAL skeleton connections and colors (consistent across all visualizations)
MAMMAL_BONES = [
    (2, 0), (2, 1),  # nose → ears
    (2, 3),           # nose → neck
    (3, 4),           # neck → body_middle
    (4, 5), (5, 6), (6, 7),  # spine → tail
    (3, 11), (11, 10), (10, 8), (8, 9),    # L arm
    (3, 15), (15, 14), (14, 12), (12, 13),  # R arm
    (4, 18), (18, 17), (17, 16),  # L leg
    (4, 21), (21, 20), (20, 19),  # R leg
]

BODY_PARTS = {
    "Head":    {"joints": [0, 1, 2, 3],         "color": "#FF6B6B"},
    "Spine":   {"joints": [4],                   "color": "#4ECDC4"},
    "Tail":    {"joints": [5, 6, 7],             "color": "#95E1D3"},
    "Front_L": {"joints": [8, 9, 10, 11],        "color": "#FFD93D"},
    "Front_R": {"joints": [12, 13, 14, 15],      "color": "#6BCB77"},
    "Hind_L":  {"joints": [16, 17, 18],          "color": "#4D96FF"},
    "Hind_R":  {"joints": [19, 20, 21],          "color": "#9B59B6"},
}

JOINT_COLORS = {}
for part_name, info in BODY_PARTS.items():
    for j in info["joints"]:
        JOINT_COLORS[j] = info["color"]

KEYPOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder", "L_foot", "L_knee",
    "L_hip", "R_foot", "R_knee", "R_hip",
]


# MAMMAL mm → GS-LRM normalized space transform
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785  # ≈ 0.008781


def mammal_to_gslrm(kp_mm: np.ndarray) -> np.ndarray:
    """Convert MAMMAL mm coordinates to GS-LRM normalized space.

    The camera w2c matrices in opencv_cameras.json expect GS-LRM normalized
    coordinates, NOT raw MAMMAL mm coordinates.
    """
    return (kp_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def project_keypoints(
    kp_3d: np.ndarray,
    w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_size: int = 512,
    from_mammal_mm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D keypoints to 2D pixel coordinates.

    Args:
        kp_3d: (K, 3) 3D keypoints in MAMMAL world space (mm) or GS-LRM normalized
        w2c: (4, 4) world-to-camera transform (expects GS-LRM normalized coords)
        fx, fy, cx, cy: camera intrinsics
        image_size: for boundary check
        from_mammal_mm: if True, convert MAMMAL mm → GS-LRM normalized first

    Returns:
        uv: (K, 2) pixel coordinates [u, v]
        visible: (K,) boolean mask — True if within image bounds
    """
    K = kp_3d.shape[0]

    # Convert coordinate system if needed
    if from_mammal_mm:
        kp_3d = mammal_to_gslrm(kp_3d)

    kp_h = np.hstack([kp_3d, np.ones((K, 1))])  # (K, 4)
    kp_cam = (w2c @ kp_h.T).T[:, :3]  # (K, 3)

    z = kp_cam[:, 2]
    z = np.clip(z, 1e-6, None)  # avoid div by zero

    u = fx * kp_cam[:, 0] / z + cx
    v = fy * kp_cam[:, 1] / z + cy

    uv = np.stack([u, v], axis=1)  # (K, 2)
    visible = (u >= 0) & (u < image_size) & (v >= 0) & (v < image_size) & (z > 0)

    return uv, visible


def load_camera_params(camera_json_path: str) -> list[dict]:
    """Load camera parameters from opencv_cameras.json.

    Returns:
        List of camera dicts with keys: w2c, fx, fy, cx, cy, w, h
    """
    with open(camera_json_path) as f:
        data = json.load(f)

    cameras = []
    for frame in data["frames"]:
        cameras.append({
            "w2c": np.array(frame["w2c"]),
            "fx": frame["fx"],
            "fy": frame["fy"],
            "cx": frame["cx"],
            "cy": frame["cy"],
            "w": frame.get("w", 512),
            "h": frame.get("h", 512),
        })
    return cameras


class KeypointProjector:
    """Reusable projector for a specific M5 frame's cameras."""

    def __init__(self, sample_dir: str):
        """Load cameras from a sample directory.

        Args:
            sample_dir: Path to M5/{frame_idx:06d}/ containing opencv_cameras.json
        """
        cam_json = Path(sample_dir) / "opencv_cameras.json"
        self.cameras = load_camera_params(str(cam_json))

    def project_frame(
        self, kp_3d: np.ndarray, cam_idx: Optional[int] = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Project keypoints to all (or specific) cameras.

        Args:
            kp_3d: (22, 3) 3D keypoints
            cam_idx: specific camera index, or None for all

        Returns:
            dict[cam_idx → (uv, visible)]
        """
        result = {}
        cams = [cam_idx] if cam_idx is not None else range(len(self.cameras))
        for ci in cams:
            cam = self.cameras[ci]
            uv, vis = project_keypoints(
                kp_3d, cam["w2c"], cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                image_size=cam["w"],
            )
            result[ci] = (uv, vis)
        return result


def draw_skeleton_on_image(
    image: np.ndarray,
    uv: np.ndarray,
    visible: np.ndarray,
    show_labels: bool = False,
    linewidth: float = 2.0,
    markersize: float = 5.0,
    alpha: float = 0.8,
) -> np.ndarray:
    """Draw MAMMAL skeleton overlay on an image using matplotlib.

    Args:
        image: (H, W, 3) RGB image
        uv: (22, 2) pixel coordinates
        visible: (22,) boolean visibility mask
        show_labels: annotate joint indices

    Returns:
        (H, W, 3) image with skeleton overlay
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = image.shape[:2]
    dpi = 80
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(image)

    # Draw bones
    for i, j in MAMMAL_BONES:
        if visible[i] and visible[j]:
            color = JOINT_COLORS.get(i, "#888")
            ax.plot([uv[i, 0], uv[j, 0]], [uv[i, 1], uv[j, 1]],
                    color=color, linewidth=linewidth, alpha=alpha)

    # Draw joints
    for idx in range(len(uv)):
        if visible[idx]:
            color = JOINT_COLORS.get(idx, "#888")
            ax.plot(uv[idx, 0], uv[idx, 1], "o", color=color,
                    markersize=markersize, markeredgecolor="white",
                    markeredgewidth=0.5)
            if show_labels:
                ax.annotate(f"{idx}:{KEYPOINT_NAMES[idx]}", (uv[idx, 0] + 3, uv[idx, 1] - 3),
                           fontsize=4, color=color, fontweight="bold")

    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    result = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return result
