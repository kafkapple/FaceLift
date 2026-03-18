"""Keypoint preprocessing pipeline for behavior clustering.

Standard pipeline order (literature-based):
1. Frame jump masking
2. Temporal smoothing (Savitzky-Golay or Kalman)
3. COM centering (per-frame)
4. Body-size normalization (optional, session median)
5. Egocentric alignment (optional, hip→nose = +X)

References:
- B-SOiD (Hsu 2021): center + session-median scale + Savgol
- Keypoint-MoSeq (Markowitz 2023): root centering + body length norm + Savgol + ego-align
- SUBTLE (Wiltschko 2020): torso centering + body length norm + Savgol + ego-align
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

# MAMMAL 22-joint indices
NOSE_IDX = 2
TAIL_ROOT_IDX = 5
L_HIP_IDX = 18
R_HIP_IDX = 21
NECK_IDX = 3

# Frame jumps in M5t2
FRAME_JUMP_INDICES = [1180, 2360, 3540]
FRAME_JUMP_MARGIN = 2


@dataclass
class PreprocessConfig:
    """Configuration for keypoint preprocessing."""

    # Frame masking
    mask_frame_jumps: bool = True
    jump_indices: list[int] = field(default_factory=lambda: FRAME_JUMP_INDICES)
    jump_margin: int = FRAME_JUMP_MARGIN

    # Temporal smoothing
    smooth: bool = True
    smooth_method: str = "savgol"  # "savgol", "kalman", "median", "none"
    savgol_window: int = 7
    savgol_poly: int = 3
    median_kernel: int = 5

    # Spatial preprocessing
    center: bool = True
    center_method: str = "com"  # "com" (all joints), "root" (hip midpoint)
    normalize_size: bool = True
    size_method: str = "session_median"  # "session_median", "per_frame", "fixed"
    fixed_size_mm: float = 72.0  # fallback if method="fixed"

    # Egocentric alignment
    egocentric: bool = False

    # Output scale (for clustering method compatibility)
    output_scale: str = "normalized"  # "normalized" (0-1.5), "mm" (original scale), "zscore"

    @property
    def name(self) -> str:
        """Short name for this config."""
        parts = []
        if self.center:
            parts.append("ctr")
        if self.normalize_size:
            parts.append("norm")
        if self.smooth:
            parts.append(f"sm_{self.smooth_method}")
        if self.egocentric:
            parts.append("ego")
        return "_".join(parts) or "raw"


# Preset configurations
PRESETS = {
    "raw": PreprocessConfig(
        mask_frame_jumps=True, smooth=False, center=False,
        normalize_size=False, egocentric=False,
    ),
    "centered": PreprocessConfig(
        smooth=False, center=True, normalize_size=False, egocentric=False,
    ),
    "centered_smooth": PreprocessConfig(
        smooth=True, center=True, normalize_size=False, egocentric=False,
    ),
    "bsoid_standard": PreprocessConfig(
        smooth=True, center=True, normalize_size=True,
        size_method="session_median", egocentric=False,
        output_scale="normalized",
    ),
    "moseq_standard": PreprocessConfig(
        smooth=True, center=True, center_method="root",
        normalize_size=True, egocentric=True,
    ),
    "full": PreprocessConfig(
        smooth=True, center=True, normalize_size=True, egocentric=True,
    ),
}


def preprocess(
    kp: np.ndarray,
    config: Optional[PreprocessConfig] = None,
    preset: Optional[str] = None,
) -> tuple[np.ndarray, dict]:
    """Apply preprocessing pipeline.

    Args:
        kp: (T, K, 3) raw 3D keypoints
        config: PreprocessConfig instance
        preset: Name of preset config (overrides config if given)

    Returns:
        (kp_processed, metadata) where metadata contains preprocessing info
    """
    if preset is not None:
        config = PRESETS[preset]
    if config is None:
        config = PRESETS["bsoid_standard"]

    meta = {"config": config.name, "original_shape": kp.shape}
    kp = kp.copy()

    # Step 0: Frame jump masking
    if config.mask_frame_jumps:
        kp, valid_mask = _mask_frame_jumps(kp, config.jump_indices, config.jump_margin)
        meta["frames_masked"] = int((~valid_mask).sum())
        meta["frames_after_mask"] = int(kp.shape[0])

    # Step 1: Temporal smoothing (BEFORE spatial preprocessing)
    if config.smooth:
        kp = _smooth(kp, config)
        meta["smoothing"] = config.smooth_method

    # Step 2: COM centering
    if config.center:
        kp, com_trajectory = _center(kp, config)
        meta["com_mean"] = com_trajectory.mean(axis=0).tolist()

    # Step 3: Body-size normalization
    if config.normalize_size:
        kp, scale_factor = _normalize_size(kp, config)
        meta["scale_factor_mm"] = float(scale_factor)

    # Step 4: Egocentric alignment
    if config.egocentric:
        kp = _align_egocentric(kp)
        meta["egocentric"] = True

    meta["output_range"] = [float(kp.min()), float(kp.max())]
    meta["output_std"] = float(kp.std())

    return kp, meta


def _mask_frame_jumps(
    kp: np.ndarray, indices: list[int], margin: int
) -> tuple[np.ndarray, np.ndarray]:
    T = kp.shape[0]
    valid = np.ones(T, dtype=bool)
    for idx in indices:
        s, e = max(0, idx - margin), min(T, idx + margin + 1)
        valid[s:e] = False
    return kp[valid], valid


def _smooth(kp: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    T, K, D = kp.shape

    if config.smooth_method == "savgol":
        for k in range(K):
            for d in range(D):
                kp[:, k, d] = savgol_filter(
                    kp[:, k, d], config.savgol_window, config.savgol_poly
                )

    elif config.smooth_method == "median":
        from scipy.ndimage import median_filter
        for k in range(K):
            for d in range(D):
                kp[:, k, d] = median_filter(kp[:, k, d], size=config.median_kernel)

    elif config.smooth_method == "kalman":
        kp = _kalman_smooth(kp)

    return kp


def _kalman_smooth(kp: np.ndarray) -> np.ndarray:
    """Simple constant-velocity Kalman filter per joint per axis.

    State: [position, velocity]
    Measurement: position only
    """
    T, K, D = kp.shape
    dt = 1.0  # frame interval (normalized)

    # State transition: x_new = x + v*dt, v_new = v
    A = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])  # observe position only

    # Noise covariances (tuned for ~20fps animal keypoints)
    Q = np.array([[0.1, 0], [0, 1.0]])  # process noise
    R = np.array([[5.0]])  # measurement noise (mm^2 for raw, adjust for normalized)

    kp_smooth = np.zeros_like(kp)

    for k in range(K):
        for d in range(D):
            z = kp[:, k, d]  # measurements

            # Initialize
            x = np.array([z[0], 0.0])  # [position, velocity]
            P = np.eye(2) * 10.0

            smoothed = np.zeros(T)
            for t in range(T):
                # Predict
                x_pred = A @ x
                P_pred = A @ P @ A.T + Q

                # Update
                y = z[t] - H @ x_pred  # innovation
                S = H @ P_pred @ H.T + R
                K_gain = P_pred @ H.T @ np.linalg.inv(S)
                x = x_pred + K_gain.flatten() * y.item()
                P = (np.eye(2) - K_gain @ H) @ P_pred

                smoothed[t] = x[0]

            kp_smooth[:, k, d] = smoothed

    return kp_smooth


def _center(
    kp: np.ndarray, config: PreprocessConfig
) -> tuple[np.ndarray, np.ndarray]:
    if config.center_method == "root":
        # Hip midpoint as root
        root = (kp[:, L_HIP_IDX] + kp[:, R_HIP_IDX]) / 2  # (T, 3)
        com = root[:, np.newaxis, :]  # (T, 1, 3)
    else:
        # All-joint center of mass
        com = kp.mean(axis=1, keepdims=True)  # (T, 1, 3)

    kp_centered = kp - com
    return kp_centered, com.squeeze(1)


def _normalize_size(
    kp: np.ndarray, config: PreprocessConfig
) -> tuple[np.ndarray, float]:
    nose_tail = np.linalg.norm(
        kp[:, NOSE_IDX] - kp[:, TAIL_ROOT_IDX], axis=1
    )

    if config.size_method == "session_median":
        scale = float(np.median(nose_tail))
    elif config.size_method == "per_frame":
        scale = nose_tail
    elif config.size_method == "fixed":
        scale = config.fixed_size_mm
    else:
        scale = float(np.median(nose_tail))

    scale = np.clip(scale, 10.0, None)  # avoid div-by-zero

    if isinstance(scale, np.ndarray):
        kp = kp / scale[:, np.newaxis, np.newaxis]
    else:
        kp = kp / scale

    return kp, float(np.median(scale)) if isinstance(scale, np.ndarray) else scale


def _align_egocentric(kp: np.ndarray) -> np.ndarray:
    """Rotate each frame so hip→nose = +X (yaw only)."""
    T = kp.shape[0]
    hip_mid = (kp[:, L_HIP_IDX] + kp[:, R_HIP_IDX]) / 2
    forward = kp[:, NOSE_IDX] - hip_mid
    forward_xy = forward[:, :2]
    norms = np.linalg.norm(forward_xy, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    forward_unit = forward_xy / norms

    kp_aligned = np.zeros_like(kp)
    for t in range(T):
        cos_a, sin_a = forward_unit[t]
        R = np.array([[cos_a, sin_a, 0], [-sin_a, cos_a, 0], [0, 0, 1]])
        kp_aligned[t] = kp[t] @ R.T

    return kp_aligned
