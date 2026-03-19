"""Camera preset generators for comparison rendering.

Each preset produces a sequence of camera extrinsics (c2w matrices) and
intrinsics suitable for render_opencv_cam or render_turntable.

Supported presets:
    - orbit_360: full 360-degree turntable orbit
    - top_down: overhead bird's-eye view (static or slow rotation)
    - bottom_up: ventral view (static or slow rotation)
    - extrapolated_novel: viewpoints outside the training camera hull
    - body_follow: keypoint-driven camera that tracks a body part

All c2w matrices are in OpenCV convention (used by GS-LRM):
    +X right, +Y down, +Z forward (into scene).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class CameraPreset:
    """Base class for camera trajectory generation.

    Subclasses override `generate()` to produce (c2ws, fxfycxcy) arrays.

    Attributes:
        n_frames: number of output frames
        resolution: square render resolution
        radius: orbit radius (distance from center)
        center: orbit center in world coords (default: origin)
        fx, fy, cx, cy: intrinsics (GS-LRM defaults)
    """

    n_frames: int = 120
    resolution: int = 512
    radius: float = 2.7
    center: Optional[List[float]] = None
    fx: float = 549.0
    fy: float = 549.0
    cx: float = 256.0
    cy: float = 256.0

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate camera trajectory.

        Returns:
            c2ws: (n_frames, 4, 4) camera-to-world matrices
            fxfycxcy: (n_frames, 4) intrinsics per frame
        """
        raise NotImplementedError

    def _intrinsics_array(self) -> np.ndarray:
        """Return (n_frames, 4) intrinsics array."""
        intr = np.array([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)
        return np.tile(intr, (self.n_frames, 1))

    @staticmethod
    def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Compute c2w from eye position, target, and up vector (OpenCV)."""
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Gimbal lock: forward is collinear with up. Use fallback up.
            fallback_up = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            right = np.cross(forward, fallback_up)
            right_norm = np.linalg.norm(right)
        right = right / (right_norm + 1e-8)

        new_up = np.cross(right, forward)
        new_up = new_up / (np.linalg.norm(new_up) + 1e-8)

        # OpenCV convention: Z forward, Y down, X right
        # R columns = [right, -up, forward]
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = -new_up
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        return c2w

    def _get_center(self) -> np.ndarray:
        if self.center is not None:
            return np.array(self.center, dtype=np.float64)
        return np.zeros(3, dtype=np.float64)


# ---------------------------------------------------------------------------
# Orbit 360
# ---------------------------------------------------------------------------

@dataclass
class Orbit360Preset(CameraPreset):
    """Full 360-degree orbit around the scene center.

    Uses the same convention as turntable_renderer: clockwise in math
    coords = physical CCW from above.
    """

    elevation: float = 20.0      # degrees above horizontal
    clockwise: bool = True       # math CW = physical CCW
    start_azimuth: float = 270.0 # degrees, default matches get_turntable_cameras (-Y start)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        center = self._get_center()
        elev_rad = np.deg2rad(self.elevation)

        c2ws = []
        for i in range(self.n_frames):
            # Configurable start azimuth (default 270° = -Y direction)
            # clockwise=True → decreasing azimuth in math coords = physical CCW from above
            sweep = 360.0 * i / self.n_frames
            if self.clockwise:
                azim_deg = self.start_azimuth - sweep
            else:
                azim_deg = self.start_azimuth + sweep
            azim_rad = np.deg2rad(azim_deg)

            x = center[0] + self.radius * np.cos(azim_rad) * np.cos(elev_rad)
            y = center[1] + self.radius * np.sin(azim_rad) * np.cos(elev_rad)
            z = center[2] + self.radius * np.sin(elev_rad)
            eye = np.array([x, y, z])

            up = np.array([0.0, 0.0, 1.0])
            c2ws.append(self._look_at(eye, center, up))

        return np.stack(c2ws), self._intrinsics_array()


# ---------------------------------------------------------------------------
# Top-down / Bottom-up
# ---------------------------------------------------------------------------

@dataclass
class TopDownPreset(CameraPreset):
    """Bird's-eye view from above, with optional slow rotation."""

    height: float = 2.5        # distance above center
    rotate: bool = True        # slow rotation around vertical

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        center = self._get_center()

        c2ws = []
        for i in range(self.n_frames):
            if self.rotate:
                angle = 2.0 * np.pi * i / self.n_frames
            else:
                angle = 0.0

            eye = center + np.array([0.0, 0.0, self.height])
            # Rotate the "forward" direction around Z
            fwd_xy = np.array([np.cos(angle), np.sin(angle), 0.0])
            up = fwd_xy  # up in image = forward direction on ground

            c2ws.append(self._look_at(eye, center, up))

        return np.stack(c2ws), self._intrinsics_array()


@dataclass
class BottomUpPreset(CameraPreset):
    """Ventral view from below, with optional slow rotation."""

    depth: float = 2.5         # distance below center
    rotate: bool = True

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        center = self._get_center()

        c2ws = []
        for i in range(self.n_frames):
            if self.rotate:
                angle = 2.0 * np.pi * i / self.n_frames
            else:
                angle = 0.0

            eye = center + np.array([0.0, 0.0, -self.depth])
            fwd_xy = np.array([np.cos(angle), np.sin(angle), 0.0])
            up = -fwd_xy  # flip for looking upward

            c2ws.append(self._look_at(eye, center, up))

        return np.stack(c2ws), self._intrinsics_array()


# ---------------------------------------------------------------------------
# Extrapolated novel views
# ---------------------------------------------------------------------------

@dataclass
class ExtrapolatedNovelPreset(CameraPreset):
    """Novel views outside the training camera hull.

    Generates views at extreme elevations (high and low) and distances
    not seen during training, useful for evaluating generalization.
    """

    elevation_range: Tuple[float, float] = (-30.0, 60.0)  # degrees
    distance_range: Tuple[float, float] = (1.5, 4.0)      # radius range

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        center = self._get_center()
        up = np.array([0.0, 0.0, 1.0])

        c2ws = []
        for i in range(self.n_frames):
            t = i / max(self.n_frames - 1, 1)  # 0..1

            # Sweep azimuth full circle (match turntable: start at 270°, CW)
            azimuth = np.deg2rad(270.0 - 360.0 * t)

            # Sweep elevation from min to max
            elev_deg = self.elevation_range[0] + t * (
                self.elevation_range[1] - self.elevation_range[0]
            )
            elev_rad = np.deg2rad(elev_deg)

            # Sweep distance from near to far
            dist = self.distance_range[0] + t * (
                self.distance_range[1] - self.distance_range[0]
            )

            x = center[0] + dist * np.cos(azimuth) * np.cos(elev_rad)
            y = center[1] + dist * np.sin(azimuth) * np.cos(elev_rad)
            z = center[2] + dist * np.sin(elev_rad)
            eye = np.array([x, y, z])

            c2ws.append(self._look_at(eye, center, up))

        return np.stack(c2ws), self._intrinsics_array()


# ---------------------------------------------------------------------------
# Body-follow (keypoint-driven)
# ---------------------------------------------------------------------------

@dataclass
class BodyFollowPreset(CameraPreset):
    """Camera that tracks a body part using keypoint data.

    Requires keypoints_3d to be set before generate() is called.
    Falls back to KeypointFollowCamera from keypoint_overlay module.
    """

    keypoint_idx: int = 0          # keypoint index to follow
    offset: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.0])
    smoothing_alpha: float = 0.3
    distance: float = 0.8

    # Set externally before generate()
    keypoints_3d: Optional[np.ndarray] = field(default=None, repr=False)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.keypoints_3d is None:
            raise ValueError(
                "BodyFollowPreset requires keypoints_3d to be set. "
                "Provide (T, N_joints, 3) array before calling generate()."
            )

        from mouse_extensions.visualization.keypoint_overlay import (
            CameraFollowConfig,
            KeypointFollowCamera,
        )

        # Determine target name from keypoint index
        _KP_TO_TARGET = {
            0: "face",          # nose
            5: "tail_base",     # lower_back
            9: "right_front_paw",
            12: "left_front_paw",
            15: "right_hind_paw",
            18: "left_hind_paw",
        }
        target = _KP_TO_TARGET.get(self.keypoint_idx, "face")

        config = CameraFollowConfig(
            target=target,
            distance=self.distance,
            smoothing_alpha=self.smoothing_alpha,
            resolution=self.resolution,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )

        cam = KeypointFollowCamera(config)
        c2ws_list, intrinsics_list = cam.compute_trajectory(self.keypoints_3d)

        # Convert to arrays
        c2ws = np.stack(c2ws_list)  # (T, 4, 4)
        fxfycxcy = np.array([
            [d["fx"], d["fy"], d["cx"], d["cy"]] for d in intrinsics_list
        ])

        # n_frames may differ from keypoint count
        actual = min(self.n_frames, len(c2ws))
        return c2ws[:actual], fxfycxcy[:actual]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PRESET_REGISTRY = {
    "orbit_360": Orbit360Preset,
    "top_down": TopDownPreset,
    "bottom_up": BottomUpPreset,
    "extrapolated_novel": ExtrapolatedNovelPreset,
    "body_follow": BodyFollowPreset,
}


def make_preset(preset_type: str, **kwargs) -> CameraPreset:
    """Create a CameraPreset instance from type name and config kwargs.

    Args:
        preset_type: one of 'orbit_360', 'top_down', 'bottom_up',
                     'extrapolated_novel', 'body_follow'
        **kwargs: passed to the preset dataclass constructor

    Returns:
        CameraPreset instance ready for generate()

    Raises:
        ValueError: if preset_type is unknown
    """
    cls = _PRESET_REGISTRY.get(preset_type)
    if cls is None:
        available = ", ".join(sorted(_PRESET_REGISTRY.keys()))
        raise ValueError(f"Unknown preset type '{preset_type}'. Available: {available}")

    # Filter kwargs to only accepted fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}

    return cls(**filtered)
