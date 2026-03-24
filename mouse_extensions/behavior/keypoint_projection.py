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

from mouse_extensions.behavior.coordinate_utils import (
    mammal_to_gslrm,
    M5_SCENE_CENTER,
    M5_DISTANCE_SCALE,
)
from mouse_extensions.constants import (
    SKELETON_BONES as MAMMAL_BONES,
    MOUSE_KP_NAMES as KEYPOINT_NAMES,
    BODY_PARTS as _BODY_PARTS,
)


# Build visualization-friendly body parts with colors
_VIZ_COLORS = {
    "face": "#FF6B6B", "torso": "#4ECDC4", "tail": "#95E1D3",
    "left_paw": "#FFD93D", "right_paw": "#6BCB77",
    "left_hind": "#4D96FF", "right_hind": "#9B59B6",
}
BODY_PARTS = {
    name: {"joints": indices, "color": _VIZ_COLORS.get(name, "#AAAAAA")}
    for name, indices in _BODY_PARTS.items()
}

JOINT_COLORS = {}
for part_name, info in BODY_PARTS.items():
    for j in info["joints"]:
        JOINT_COLORS[j] = info["color"]



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
