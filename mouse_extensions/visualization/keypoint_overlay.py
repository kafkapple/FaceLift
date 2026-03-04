"""Keypoint overlay visualization for 3D-to-2D projection on rendered images.

Provides utilities to project MAMMAL 22 3D keypoints onto 2D images and draw
skeleton overlays with joint markers and bone connections.

Color scheme follows MAMMAL KEYPOINTS.md convention:
    Head (0-2): Yellow | Body (3-4): Magenta | Tail (5-7): Orange
    Left Front (8-11): Blue | Right Front (12-15): Green
    Left Hind (16-18): Cyan | Right Hind (19-21): Red
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# --- Constants ---

KEYPOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle",
    "tail_root", "tail_middle", "tail_end",
    "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip",
    "R_foot", "R_knee", "R_hip",
]

# Skeleton bones from MAMMAL KEYPOINTS.md (ground truth)
SKELETON_BONES = [
    # Head: ears → nose
    (0, 2), (1, 2),
    # Spine: nose → neck → body_middle → tail_root
    (2, 3), (3, 4), (4, 5),
    # Tail: tail_root → tail_middle → tail_end
    (5, 6), (6, 7),
    # Left front limb: shoulder→elbow→paw_end→paw, shoulder→neck
    (11, 3), (10, 11), (9, 10), (8, 9),
    # Right front limb
    (15, 3), (14, 15), (13, 14), (12, 13),
    # Left hind limb: hip→knee→foot, hip→tail_root
    (18, 5), (17, 18), (16, 17),
    # Right hind limb
    (21, 5), (20, 21), (19, 20),
]

# Body part grouping with BGR colors (OpenCV convention)
JOINT_GROUPS = {
    "head":        {"indices": [0, 1, 2],         "color": (0, 255, 255)},    # yellow
    "body":        {"indices": [3, 4],             "color": (255, 0, 255)},    # magenta
    "tail":        {"indices": [5, 6, 7],          "color": (0, 165, 255)},    # orange
    "left_front":  {"indices": [8, 9, 10, 11],     "color": (255, 0, 0)},      # blue
    "right_front": {"indices": [12, 13, 14, 15],   "color": (0, 255, 0)},      # green
    "left_hind":   {"indices": [16, 17, 18],        "color": (255, 255, 0)},    # cyan
    "right_hind":  {"indices": [19, 20, 21],        "color": (0, 0, 255)},      # red
}

# Precomputed index → color map
_JOINT_COLORS: Dict[int, Tuple[int, int, int]] = {}
for _info in JOINT_GROUPS.values():
    for _idx in _info["indices"]:
        _JOINT_COLORS[_idx] = _info["color"]


# --- Projection ---

def project_3d_to_2d(
    keypoints_3d: np.ndarray,
    w2c: np.ndarray,
    intrinsics: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D keypoints to 2D image coordinates.

    Args:
        keypoints_3d: (K, 3) world-space 3D keypoints
        w2c: (4, 4) world-to-camera transformation matrix
        intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'

    Returns:
        kp_2d: (K, 2) pixel coordinates (u, v)
        valid: (K,) boolean mask for points in front of camera (z > 0)
    """
    K = keypoints_3d.shape[0]

    # Homogeneous coordinates
    kp_homo = np.concatenate([keypoints_3d, np.ones((K, 1))], axis=1)  # (K, 4)

    # World to camera transform
    kp_cam = (w2c @ kp_homo.T).T  # (K, 4)

    # Depth validity check
    valid = kp_cam[:, 2] > 0.01

    # Perspective projection
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    z = np.clip(kp_cam[:, 2], 0.01, None)
    u = fx * kp_cam[:, 0] / z + cx
    v = fy * kp_cam[:, 1] / z + cy

    kp_2d = np.stack([u, v], axis=1)  # (K, 2)
    return kp_2d, valid


# --- Drawing ---

def _in_bounds(pt: Tuple[int, int], w: int, h: int, margin: int = 50) -> bool:
    """Check if a point is within image bounds (with margin)."""
    return -margin <= pt[0] < w + margin and -margin <= pt[1] < h + margin


def draw_keypoint_overlay(
    image: np.ndarray,
    kp_2d: np.ndarray,
    valid: np.ndarray,
    draw_labels: bool = False,
    draw_skeleton: bool = True,
    joint_radius: int = 4,
    bone_thickness: int = 2,
    label_font_scale: float = 0.3,
) -> np.ndarray:
    """Draw keypoint overlay on image.

    Args:
        image: (H, W, 3) BGR image (uint8)
        kp_2d: (22, 2) pixel coordinates
        valid: (22,) boolean mask
        draw_labels: whether to draw joint name labels
        draw_skeleton: whether to draw bone connections
        joint_radius: radius of joint circles
        bone_thickness: thickness of bone lines
        label_font_scale: font scale for labels

    Returns:
        Annotated image copy
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Draw skeleton bones first (behind joints)
    if draw_skeleton:
        for i, j in SKELETON_BONES:
            if not (valid[i] and valid[j]):
                continue
            p1 = tuple(kp_2d[i].astype(int))
            p2 = tuple(kp_2d[j].astype(int))
            if not (_in_bounds(p1, w, h) or _in_bounds(p2, w, h)):
                continue
            # Average color of connected joints
            c1 = np.array(_JOINT_COLORS.get(i, (255, 255, 255)))
            c2 = np.array(_JOINT_COLORS.get(j, (255, 255, 255)))
            color = tuple(((c1 + c2) / 2).astype(int).tolist())
            cv2.line(img, p1, p2, color, bone_thickness, cv2.LINE_AA)

    # Draw joints on top
    for idx in range(kp_2d.shape[0]):
        if not valid[idx]:
            continue
        pt = tuple(kp_2d[idx].astype(int))
        if not _in_bounds(pt, w, h):
            continue
        color = _JOINT_COLORS.get(idx, (255, 255, 255))
        cv2.circle(img, pt, joint_radius, color, -1, cv2.LINE_AA)
        cv2.circle(img, pt, joint_radius, (0, 0, 0), 1, cv2.LINE_AA)  # outline

        if draw_labels:
            # Show compact index number instead of full name
            cv2.putText(
                img, str(idx),
                (pt[0] + joint_radius + 1, pt[1] - joint_radius),
                cv2.FONT_HERSHEY_SIMPLEX, label_font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    return img


def draw_bounding_box(
    image: np.ndarray,
    kp_2d: np.ndarray,
    valid: np.ndarray,
    label: str = "",
    color: Tuple[int, int, int] = (0, 255, 0),
    padding: int = 10,
) -> np.ndarray:
    """Draw bounding box around valid keypoints."""
    img = image.copy()

    valid_kp = kp_2d[valid]
    if len(valid_kp) == 0:
        return img

    x_min = int(valid_kp[:, 0].min()) - padding
    y_min = int(valid_kp[:, 1].min()) - padding
    x_max = int(valid_kp[:, 0].max()) + padding
    y_max = int(valid_kp[:, 1].max()) + padding

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    if label:
        cv2.putText(
            img, label, (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    return img


# --- Visualizer class ---

@dataclass
class KeypointVisualizer:
    """Manages keypoint visualization settings and batch processing."""

    joint_radius: int = 4
    bone_thickness: int = 2
    draw_labels: bool = False
    draw_skeleton: bool = True
    label_font_scale: float = 0.3

    def overlay_on_image(
        self,
        image: np.ndarray,
        keypoints_3d: np.ndarray,
        w2c: np.ndarray,
        intrinsics: Dict[str, float],
    ) -> np.ndarray:
        """Project 3D keypoints and overlay on image.

        Args:
            image: (H, W, 3) BGR image
            keypoints_3d: (22, 3) world-space keypoints
            w2c: (4, 4) world-to-camera matrix
            intrinsics: dict with fx, fy, cx, cy

        Returns:
            Annotated image
        """
        kp_2d, valid = project_3d_to_2d(keypoints_3d, w2c, intrinsics)
        return draw_keypoint_overlay(
            image, kp_2d, valid,
            draw_labels=self.draw_labels,
            draw_skeleton=self.draw_skeleton,
            joint_radius=self.joint_radius,
            bone_thickness=self.bone_thickness,
            label_font_scale=self.label_font_scale,
        )

    def overlay_multiview(
        self,
        images: List[np.ndarray],
        keypoints_3d: np.ndarray,
        w2cs: List[np.ndarray],
        intrinsics_list: List[Dict[str, float]],
    ) -> List[np.ndarray]:
        """Overlay keypoints on multiple camera views."""
        return [
            self.overlay_on_image(img, keypoints_3d, w2c, intr)
            for img, w2c, intr in zip(images, w2cs, intrinsics_list)
        ]

    def create_comparison_row(
        self,
        gt_images: List[np.ndarray],
        pred_images: List[np.ndarray],
        keypoints_3d: np.ndarray,
        w2cs: List[np.ndarray],
        intrinsics_list: List[Dict[str, float]],
    ) -> np.ndarray:
        """Create GT vs Pred comparison with keypoint overlay.

        Returns:
            Stacked image: [GT_with_KP row | Pred_with_KP row]
        """
        gt_overlays = self.overlay_multiview(
            gt_images, keypoints_3d, w2cs, intrinsics_list,
        )
        pred_overlays = self.overlay_multiview(
            pred_images, keypoints_3d, w2cs, intrinsics_list,
        )

        gt_row = np.concatenate(gt_overlays, axis=1)
        pred_row = np.concatenate(pred_overlays, axis=1)

        return np.concatenate([gt_row, pred_row], axis=0)


# --- Camera Follow ---

def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, safe for near-zero magnitude."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0.0, 0.0, 1.0])


def compute_face_camera_c2w(
    kp_3d: np.ndarray,
    distance: float = 0.8,
) -> np.ndarray:
    """Compute c2w matrix for a camera looking down at the face from above.

    Camera is positioned along the face-plane normal (perpendicular to the
    triangle formed by L_ear, R_ear, nose). Nose direction points "up" in
    the rendered image.

    Args:
        kp_3d: (22, 3) keypoints in FaceLift normalized world
        distance: camera distance from face center

    Returns:
        c2w: (4, 4) camera-to-world matrix (OpenCV convention)
    """
    L_ear, R_ear, nose = kp_3d[0], kp_3d[1], kp_3d[2]
    face_center = (L_ear + R_ear + nose) / 3.0

    # Face plane vectors
    v_ear = R_ear - L_ear                     # ear-to-ear
    ear_mid = (L_ear + R_ear) / 2.0
    v_nose = nose - ear_mid                   # ear-center → nose

    # Face normal (perpendicular to face plane, pointing "out" from skull)
    face_normal = _normalize(np.cross(v_ear, v_nose))

    # Camera position: above face along normal
    cam_pos = face_center + distance * face_normal

    # Build c2w (OpenCV: z=forward into scene, y=down in image)
    forward = _normalize(face_center - cam_pos)   # z-axis: look direction
    # Use nose direction as "up" hint so nose appears "up" in image
    right = _normalize(np.cross(forward, v_nose))  # x-axis
    down = np.cross(forward, right)                # y-axis

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos

    return c2w


@dataclass
class CameraFollowConfig:
    """Configuration for camera-follow trajectory."""

    target: str = "face"          # "face" or "body"
    distance: float = 0.8         # camera distance from target
    smoothing_alpha: float = 0.3  # EMA smoothing (0=no smooth, 1=no memory)
    resolution: int = 512         # render resolution
    fx: float = 549.0             # focal length
    fy: float = 549.0
    cx: float = 256.0
    cy: float = 256.0


class KeypointFollowCamera:
    """Compute smoothed camera trajectories that follow keypoint motion.

    Produces per-frame c2w matrices and intrinsics suitable for
    render_opencv_cam.
    """

    def __init__(self, config: Optional[CameraFollowConfig] = None):
        self.config = config or CameraFollowConfig()
        self._prev_pos = None
        self._prev_target = None
        self._prev_up = None

    def reset(self):
        """Reset smoothing state for a new sequence."""
        self._prev_pos = None
        self._prev_target = None
        self._prev_up = None

    def compute_frame(self, kp_3d: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute camera c2w and intrinsics for one frame.

        Args:
            kp_3d: (22, 3) keypoints in FaceLift normalized world

        Returns:
            c2w: (4, 4) camera-to-world matrix
            intrinsics: dict with fx, fy, cx, cy
        """
        cfg = self.config

        if cfg.target == "face":
            raw_c2w = compute_face_camera_c2w(kp_3d, distance=cfg.distance)
        else:
            # Fallback: look at body center from above
            body_center = kp_3d[3:5].mean(axis=0)  # neck + body_middle
            neck = kp_3d[3]
            up_hint = _normalize(kp_3d[2] - neck)  # nose direction
            cam_pos = body_center + np.array([0, 0, cfg.distance])
            forward = _normalize(body_center - cam_pos)
            right = _normalize(np.cross(forward, up_hint))
            down = np.cross(forward, right)
            raw_c2w = np.eye(4)
            raw_c2w[:3, 0] = right
            raw_c2w[:3, 1] = down
            raw_c2w[:3, 2] = forward
            raw_c2w[:3, 3] = cam_pos

        # Apply temporal smoothing (EMA)
        pos = raw_c2w[:3, 3].copy()
        target = pos + raw_c2w[:3, 2] * cfg.distance  # look-at point
        up = -raw_c2w[:3, 1].copy()  # up = -down

        alpha = cfg.smoothing_alpha
        if self._prev_pos is not None:
            pos = alpha * pos + (1 - alpha) * self._prev_pos
            target = alpha * target + (1 - alpha) * self._prev_target
            up = alpha * up + (1 - alpha) * self._prev_up

        self._prev_pos = pos.copy()
        self._prev_target = target.copy()
        self._prev_up = up.copy()

        # Reconstruct c2w from smoothed values
        forward = _normalize(target - pos)
        right = _normalize(np.cross(forward, up))
        down = np.cross(forward, right)

        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = forward
        c2w[:3, 3] = pos

        intrinsics = {"fx": cfg.fx, "fy": cfg.fy, "cx": cfg.cx, "cy": cfg.cy}
        return c2w, intrinsics

    def compute_trajectory(
        self, kp_3d_sequence: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
        """Compute smoothed camera trajectory for a sequence of frames.

        Args:
            kp_3d_sequence: (T, 22, 3) keypoints over time

        Returns:
            c2ws: list of T (4, 4) c2w matrices
            intrinsics_list: list of T intrinsics dicts
        """
        self.reset()
        c2ws = []
        intrinsics_list = []
        for t in range(len(kp_3d_sequence)):
            c2w, intr = self.compute_frame(kp_3d_sequence[t])
            c2ws.append(c2w)
            intrinsics_list.append(intr)
        return c2ws, intrinsics_list


def create_legend(height: int = 512, width: int = 200) -> np.ndarray:
    """Create a color legend with index→name mapping for all 22 keypoints."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Title
    cv2.putText(img, "Keypoints", (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y = 32

    # Per-group: group name header + individual keypoint entries
    for group_name, info in JOINT_GROUPS.items():
        color = info["color"]
        # Group header with color bar
        cv2.rectangle(img, (5, y - 8), (width - 10, y + 2), color, -1)
        cv2.putText(img, group_name, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
        y += 16

        # Individual keypoints: "idx: name"
        for idx in info["indices"]:
            name = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kp{idx}"
            cv2.circle(img, (12, y - 3), 4, color, -1, cv2.LINE_AA)
            cv2.putText(img, f"{idx:2d}: {name}", (22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1, cv2.LINE_AA)
            y += 14
        y += 6  # spacing between groups

    return img
