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
    """Compute c2w for a camera looking at the face from the front.

    Camera is positioned along the nose→neck direction (extended outward
    from the nose), so it looks directly at the face from the front.
    The face-plane normal is used as the "up" hint so ears appear
    horizontally in the rendered image.

    Args:
        kp_3d: (22, 3) keypoints in FaceLift normalized world
        distance: camera distance from face center

    Returns:
        c2w: (4, 4) camera-to-world matrix (OpenCV convention)
    """
    nose, neck = kp_3d[2], kp_3d[3]
    L_ear, R_ear = kp_3d[0], kp_3d[1]
    face_center = (L_ear + R_ear + nose) / 3.0

    # Camera viewing direction: nose → neck (into the face)
    # Camera sits on the opposite side: extend outward from nose
    gaze_dir = _normalize(neck - nose)  # nose→neck = into the head
    cam_pos = face_center - distance * gaze_dir  # in front of face

    # Build c2w (OpenCV: z=forward into scene, y=down in image)
    forward = _normalize(face_center - cam_pos)  # z-axis: look direction

    # Up hint: face-plane normal (skull outward) keeps ears horizontal
    v_ear = R_ear - L_ear
    ear_mid = (L_ear + R_ear) / 2.0
    v_nose = nose - ear_mid
    face_normal = _normalize(np.cross(v_ear, v_nose))

    right = _normalize(np.cross(forward, face_normal))  # x-axis
    down = np.cross(forward, right)                      # y-axis

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos

    return c2w


def _build_c2w(cam_pos: np.ndarray, look_at: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build c2w matrix from position, look-at point, and up hint.

    Uses OpenCV convention: z=forward into scene, y=down in image.
    Robust: if up_hint is nearly parallel to forward, falls back to world Z-up.
    """
    forward = _normalize(look_at - cam_pos)

    # Check if up_hint is nearly parallel to forward
    if abs(np.dot(forward, _normalize(up_hint))) > 0.95:
        up_hint = np.array([0.0, 0.0, 1.0])  # world Z-up fallback
        if abs(np.dot(forward, up_hint)) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0])  # Y-up last resort

    right = _normalize(np.cross(forward, up_hint))
    down = np.cross(forward, right)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos
    return c2w


def _body_center(kp_3d: np.ndarray) -> np.ndarray:
    """Robust body center: average of spine keypoints (neck + body_middle)."""
    return kp_3d[3:5].mean(axis=0)


def _spine_direction(kp_3d: np.ndarray) -> np.ndarray:
    """Spine direction: body_middle → neck (head direction)."""
    return _normalize(kp_3d[3] - kp_3d[4])


# Scene scale constant — FaceLift mouse is normalized to this radius
_SCENE_RADIUS = 2.7


def compute_tail_base_camera_c2w(
    kp_3d: np.ndarray,
    distance: float = 1.5,
) -> np.ndarray:
    """Camera behind tail root, looking along the tail.

    Uses world Z-up for stable orientation.
    """
    tail_root = kp_3d[5]
    tail_mid = kp_3d[6]
    tail_dir = _normalize(tail_mid - tail_root)

    # Position behind tail, looking toward tail tip
    cam_pos = tail_root - distance * tail_dir
    return _build_c2w(cam_pos, tail_root, up_hint=np.array([0.0, 0.0, 1.0]))


def compute_paw_camera_c2w(
    kp_3d: np.ndarray,
    distance: float = 1.2,
    paw_side: str = "left_front",
) -> np.ndarray:
    """Camera looking at a paw from an outward-and-slightly-below angle.

    Positioned outside the body looking inward at the paw, with enough
    distance to keep the whole limb in frame.
    """
    paw_indices = {
        "left_front": (8, 9, 10),    # L_paw, L_paw_end, L_elbow
        "right_front": (12, 13, 14),  # R_paw, R_paw_end, R_elbow
        "left_hind": (16, 17, 18),    # L_foot, L_knee, L_hip
        "right_hind": (19, 20, 21),   # R_foot, R_knee, R_hip
    }
    if paw_side not in paw_indices:
        raise ValueError(f"Unknown paw_side: {paw_side}. Use: {list(paw_indices)}")

    paw_idx, paw_end_idx, joint_idx = paw_indices[paw_side]
    paw = kp_3d[paw_idx]
    joint = kp_3d[joint_idx]

    # Look-at: midpoint of limb (between paw and upper joint)
    look_at = (paw + joint) / 2.0

    # Camera direction: outward from body center + slightly below
    body_ctr = _body_center(kp_3d)
    outward = _normalize(look_at - body_ctr)
    # Add downward offset (world Z) for "below" angle
    cam_dir = _normalize(outward - 0.3 * np.array([0.0, 0.0, 1.0]))

    cam_pos = look_at + distance * cam_dir
    return _build_c2w(cam_pos, look_at, up_hint=np.array([0.0, 0.0, 1.0]))


def compute_generic_camera_c2w(
    kp_3d: np.ndarray,
    target_indices: List[int],
    look_from_indices: Optional[List[int]] = None,
    distance: float = 1.5,
    elevation_deg: float = 0.0,
) -> np.ndarray:
    """Generic keypoint-driven camera with world-up stability."""
    target_center = kp_3d[target_indices].mean(axis=0)
    origin = kp_3d[look_from_indices].mean(axis=0) if look_from_indices else kp_3d[4]

    approach_dir = _normalize(target_center - origin)

    if abs(elevation_deg) > 0.1:
        world_up = np.array([0.0, 0.0, 1.0])
        horiz = _normalize(np.cross(approach_dir, world_up))
        rad = np.radians(elevation_deg)
        approach_dir = _normalize(
            approach_dir * np.cos(rad) + np.cross(horiz, approach_dir) * np.sin(rad)
        )

    cam_pos = target_center - distance * approach_dir
    return _build_c2w(cam_pos, target_center, up_hint=np.array([0.0, 0.0, 1.0]))


# --- View Presets (body-relative direction, world-up stabilized) ---

def compute_preset_camera_c2w(
    kp_3d: np.ndarray,
    preset: str,
    distance: float = 2.5,
) -> np.ndarray:
    """Fixed-direction camera relative to the body center.

    All presets use world Z-up for stable orientation and sufficient distance
    (~scene radius) to keep the entire mouse in frame.

    Presets:
        top_down: directly above, looking down
        bottom_up: directly below, looking up
        frontal: in front of face, eye level
        lateral_left / lateral_right: side views
        posterior: behind the tail
    """
    center = _body_center(kp_3d)
    spine = _spine_direction(kp_3d)  # head direction (horizontal)
    world_up = np.array([0.0, 0.0, 1.0])
    # Lateral direction: perpendicular to spine in horizontal plane
    lateral = _normalize(np.cross(spine, world_up))

    if preset == "top_down":
        cam_pos = center + distance * world_up
        up_hint = spine  # head direction = "up" in top-down image
    elif preset == "bottom_up":
        cam_pos = center - distance * world_up
        up_hint = spine
    elif preset == "frontal":
        # In front of head, at eye level (slight upward angle)
        cam_pos = center + distance * spine + 0.3 * distance * world_up
        up_hint = world_up
    elif preset == "lateral_left":
        cam_pos = center + distance * lateral
        up_hint = world_up
    elif preset == "lateral_right":
        cam_pos = center - distance * lateral
        up_hint = world_up
    elif preset == "posterior":
        # Behind tail, slightly above
        cam_pos = center - distance * spine + 0.3 * distance * world_up
        up_hint = world_up
    else:
        raise ValueError(
            f"Unknown preset: {preset}. Use: top_down, bottom_up, "
            "frontal, lateral_left, lateral_right, posterior"
        )

    return _build_c2w(cam_pos, center, up_hint)


# --- Body Stabilization ---

def compute_body_stabilization_matrix(kp_3d: np.ndarray) -> np.ndarray:
    """Compute body-root transform for stabilized rendering.

    Returns a simple translation + rotation that places body_middle at
    the origin with spine along +Z and world-up preserved.
    Using only translation (position) for stabilization to avoid
    rotation artifacts from noisy keypoints.

    Returns:
        M_body: (4, 4) body-to-world transform (translation only)
    """
    M = np.eye(4)
    M[:3, 3] = kp_3d[4].copy()  # body_middle position
    return M


# --- Config & Follow Camera ---

CAMERA_TARGET_PRESETS = {
    "face": "Face frontal (nose→neck direction)",
    "body": "Body top-down (above body center)",
    "tail_base": "Tail base (looking along tail)",
    "left_front_paw": "Left front paw (below/side view)",
    "right_front_paw": "Right front paw (below/side view)",
    "left_hind_paw": "Left hind paw (below/side view)",
    "right_hind_paw": "Right hind paw (below/side view)",
    "top_down": "Top-down (above body, looking down)",
    "bottom_up": "Bottom-up (below body, looking up)",
    "frontal": "Frontal (in front, eye level)",
    "lateral_left": "Left lateral (side view)",
    "lateral_right": "Right lateral (side view)",
    "posterior": "Posterior (behind tail)",
}


@dataclass
class CameraFollowConfig:
    """Configuration for camera-follow trajectory.

    Targets:
        face, body, tail_base,
        left_front_paw, right_front_paw, left_hind_paw, right_hind_paw,
        top_down, bottom_up, frontal, lateral_left, lateral_right, posterior
    """

    target: str = "face"
    distance: float = 0.8         # camera distance from target
    smoothing_alpha: float = 0.3  # EMA smoothing (0=no smooth, 1=no memory)
    stabilize_body: bool = False  # fix body root, show only limb movement
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
        self._ref_body = None      # reference body offset for stabilization
        self._ref_rotation = None  # reference rotation for stabilization

    def reset(self):
        """Reset smoothing state for a new sequence."""
        self._prev_pos = None
        self._prev_target = None
        self._prev_up = None
        self._ref_body = None
        self._ref_rotation = None

    def compute_frame(self, kp_3d: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute camera c2w and intrinsics for one frame.

        Args:
            kp_3d: (22, 3) keypoints in FaceLift normalized world

        Returns:
            c2w: (4, 4) camera-to-world matrix
            intrinsics: dict with fx, fy, cx, cy
        """
        cfg = self.config

        # Dispatch to target-specific camera computation
        _PRESET_VIEWS = {
            "top_down", "bottom_up", "frontal",
            "lateral_left", "lateral_right", "posterior",
        }
        _PAW_MAP = {
            "left_front_paw": "left_front",
            "right_front_paw": "right_front",
            "left_hind_paw": "left_hind",
            "right_hind_paw": "right_hind",
        }

        if cfg.target == "face":
            raw_c2w = compute_face_camera_c2w(kp_3d, distance=cfg.distance)
        elif cfg.target == "tail_base":
            raw_c2w = compute_tail_base_camera_c2w(kp_3d, distance=cfg.distance)
        elif cfg.target in _PAW_MAP:
            raw_c2w = compute_paw_camera_c2w(
                kp_3d, distance=cfg.distance, paw_side=_PAW_MAP[cfg.target],
            )
        elif cfg.target in _PRESET_VIEWS:
            raw_c2w = compute_preset_camera_c2w(
                kp_3d, preset=cfg.target, distance=cfg.distance,
            )
        elif cfg.target == "body":
            # Legacy body target: top-down from above
            raw_c2w = compute_preset_camera_c2w(
                kp_3d, preset="top_down", distance=cfg.distance,
            )
        else:
            raise ValueError(
                f"Unknown camera target: {cfg.target}. "
                f"Available: {list(CAMERA_TARGET_PRESETS.keys())}"
            )

        # Body stabilization: camera follows body_middle position but
        # uses frame-0's body orientation for the viewing direction.
        # This keeps the mouse centered while showing limb/head movement
        # relative to a fixed body frame.
        if cfg.stabilize_body:
            body_pos = kp_3d[4].copy()  # current body_middle
            if self._ref_body is None:
                # Store frame-0's camera offset from body center
                self._ref_body = raw_c2w[:3, 3] - body_pos
                # Also store frame-0's rotation
                self._ref_rotation = raw_c2w[:3, :3].copy()
            # Keep frame-0's rotation, but translate to follow body_middle
            raw_c2w[:3, :3] = self._ref_rotation
            raw_c2w[:3, 3] = body_pos + self._ref_body

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
