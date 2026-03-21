"""Temporal smoothing methods and 2D temporal consistency metrics.

Provides:
1. EMA (Exponential Moving Average) image smoothing
2. Optical flow warping-based smoothing
3. Temporal consistency metrics: tOF, TLPIPS, FF-SSIM-var, Flicker Rate

Usage:
    from mouse_extensions.evaluation.temporal_smoothing import (
        ema_smooth_sequence, optflow_smooth_sequence,
        compute_temporal_metrics, TemporalStabilityResult,
    )

See: docs/experiments/TEMPORAL_EVAL_STANDARD.md for metric definitions.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import lpips
    import torch
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ============================================================
# Data Structures
# ============================================================

@dataclass
class TemporalStabilityResult:
    """2D temporal stability metrics over an image sequence."""

    # Temporal Optical Flow consistency
    tof_mean: float          # mean optical flow magnitude between consecutive frames
    tof_std: float           # std of flow magnitudes (lower = more stable)

    # Temporal LPIPS (perceptual difference between consecutive frames)
    tlpips_mean: float       # mean LPIPS(t, t+1)
    tlpips_std: float

    # Frame-to-frame SSIM variance
    ff_ssim_mean: float      # mean SSIM(t, t+1)
    ff_ssim_var: float       # variance of SSIM(t, t+1) — lower = more consistent

    # Flicker rate
    flicker_rate: float      # fraction of frames with large intensity change
    flicker_threshold: float  # threshold used

    # Per-frame quality (if GT available)
    psnr_mean: Optional[float] = None
    psnr_std: Optional[float] = None
    ssim_quality_mean: Optional[float] = None

    # Raw per-pair data
    per_pair_tof: List[float] = field(default_factory=list)
    per_pair_tlpips: List[float] = field(default_factory=list)
    per_pair_ssim: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "tof/mean": self.tof_mean,
            "tof/std": self.tof_std,
            "tlpips/mean": self.tlpips_mean,
            "tlpips/std": self.tlpips_std,
            "ff_ssim/mean": self.ff_ssim_mean,
            "ff_ssim/var": self.ff_ssim_var,
            "flicker/rate": self.flicker_rate,
            "flicker/threshold": self.flicker_threshold,
            "psnr/mean": self.psnr_mean,
            "psnr/std": self.psnr_std,
        }

    def summary(self) -> str:
        lines = [
            "=== Temporal Stability Metrics ===",
            f"  tOF:       mean={self.tof_mean:.4f}  std={self.tof_std:.4f}",
            f"  TLPIPS:    mean={self.tlpips_mean:.4f}  std={self.tlpips_std:.4f}",
            f"  FF-SSIM:   mean={self.ff_ssim_mean:.4f}  var={self.ff_ssim_var:.6f}",
            f"  Flicker:   {self.flicker_rate:.3f} (threshold={self.flicker_threshold})",
        ]
        if self.psnr_mean is not None:
            lines.append(f"  PSNR:      mean={self.psnr_mean:.2f}  std={self.psnr_std:.2f}")
        return "\n".join(lines)


# ============================================================
# Smoothing Methods
# ============================================================

def ema_smooth_sequence(
    frames: Sequence[np.ndarray],
    alpha: float = 0.3,
) -> List[np.ndarray]:
    """Apply Exponential Moving Average smoothing to an image sequence.

    Formula: smoothed_t = alpha * frame_t + (1 - alpha) * smoothed_{t-1}
    alpha=1.0 → no smoothing (original), alpha→0 → heavy smoothing

    Args:
        frames: List of [H, W, 3] uint8 images
        alpha: EMA weight for current frame (0-1)

    Returns:
        List of smoothed [H, W, 3] uint8 images
    """
    if not frames:
        return []

    smoothed = []
    prev = frames[0].astype(np.float32)
    smoothed.append(frames[0])

    for frame in frames[1:]:
        curr = frame.astype(np.float32)
        prev = alpha * curr + (1.0 - alpha) * prev
        smoothed.append(np.clip(prev, 0, 255).astype(np.uint8))

    return smoothed


def temporal_median_sequence(
    frames: Sequence[np.ndarray],
    window: int = 3,
) -> List[np.ndarray]:
    """Apply temporal median filter to an image sequence.

    Non-linear filter: takes pixel-wise median over a sliding window.
    Rejects outlier frames (flicker) without blending → minimal motion blur.

    Args:
        frames: List of [H, W, 3] uint8 images
        window: Temporal window size (odd number, default 3)

    Returns:
        List of smoothed [H, W, 3] uint8 images
    """
    if not frames:
        return []

    n = len(frames)
    half = window // 2
    smoothed = []

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        block = np.stack(frames[lo:hi], axis=0)  # (w, H, W, 3)
        median = np.median(block, axis=0).astype(np.uint8)
        smoothed.append(median)

    return smoothed


def bilateral_temporal_sequence(
    frames: Sequence[np.ndarray],
    window: int = 5,
    sigma_intensity: float = 25.0,
    sigma_temporal: float = 1.5,
) -> List[np.ndarray]:
    """Apply bilateral temporal filter to an image sequence.

    Weights neighboring frames by both temporal distance AND intensity
    similarity. Preserves motion edges (low blur) while smoothing flicker.

    Args:
        frames: List of [H, W, 3] uint8 images
        window: Temporal window size
        sigma_intensity: Intensity similarity bandwidth (0-255 scale)
        sigma_temporal: Temporal distance bandwidth (in frames)

    Returns:
        List of smoothed [H, W, 3] uint8 images
    """
    if not frames:
        return []

    n = len(frames)
    half = window // 2
    smoothed = []

    for i in range(n):
        center = frames[i].astype(np.float32)
        weighted_sum = np.zeros_like(center)
        weight_sum = np.zeros(center.shape[:2], dtype=np.float32)

        for j in range(max(0, i - half), min(n, i + half + 1)):
            neighbor = frames[j].astype(np.float32)

            # Temporal weight (Gaussian)
            w_t = np.exp(-0.5 * ((j - i) / sigma_temporal) ** 2)

            # Intensity weight (per-pixel, based on RGB distance)
            diff = np.sqrt(np.sum((center - neighbor) ** 2, axis=2))
            w_i = np.exp(-0.5 * (diff / sigma_intensity) ** 2)

            w = w_t * w_i  # (H, W)
            weighted_sum += neighbor * w[:, :, np.newaxis]
            weight_sum += w

        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-8)
        result = weighted_sum / weight_sum[:, :, np.newaxis]
        smoothed.append(np.clip(result, 0, 255).astype(np.uint8))

    return smoothed


def savgol_temporal_sequence(
    frames: Sequence[np.ndarray],
    window: int = 5,
    poly_order: int = 2,
) -> List[np.ndarray]:
    """Apply Savitzky-Golay temporal filter to an image sequence.

    Fits local polynomial to each pixel's temporal signal.
    Preserves signal shape (edges, peaks) better than moving average.

    Args:
        frames: List of [H, W, 3] uint8 images
        window: Window size (must be odd, >= poly_order + 1)
        poly_order: Polynomial order for fitting

    Returns:
        List of smoothed [H, W, 3] uint8 images
    """
    from scipy.signal import savgol_filter

    if not frames:
        return []

    # Stack to (T, H, W, 3)
    stack = np.stack(frames, axis=0).astype(np.float32)
    t, h, w, c = stack.shape

    # Reshape to (T, H*W*3) for vectorized savgol
    flat = stack.reshape(t, -1)

    # Apply savgol along time axis
    if window > t:
        window = t if t % 2 == 1 else t - 1
    if window < poly_order + 1:
        return list(frames)

    filtered = savgol_filter(flat, window_length=window, polyorder=poly_order, axis=0)
    filtered = filtered.reshape(t, h, w, c)

    return [np.clip(filtered[i], 0, 255).astype(np.uint8) for i in range(t)]


def optflow_smooth_sequence(
    frames: Sequence[np.ndarray],
    alpha: float = 0.3,
) -> List[np.ndarray]:
    """Apply optical flow warping-based temporal smoothing.

    For each frame t: warp frame t-1 using optical flow to t,
    then blend: result = alpha * frame_t + (1-alpha) * warped_{t-1}

    Args:
        frames: List of [H, W, 3] uint8 images
        alpha: Blend weight for current frame

    Returns:
        List of smoothed [H, W, 3] uint8 images
    """
    if not frames:
        return []

    smoothed = [frames[0]]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    prev_result = frames[0].astype(np.float32)

    for frame in frames[1:]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Compute optical flow (prev -> curr)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Warp previous result to current frame coordinates
        h, w = frame.shape[:2]
        flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
        flow_map += flow
        warped = cv2.remap(
            prev_result.astype(np.float32),
            flow_map[:, :, 0], flow_map[:, :, 1],
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        # Blend
        result = alpha * frame.astype(np.float32) + (1.0 - alpha) * warped
        result = np.clip(result, 0, 255)
        smoothed.append(result.astype(np.uint8))

        prev_gray = curr_gray
        prev_result = result

    return smoothed


# ============================================================
# Temporal Consistency Metrics
# ============================================================

def _compute_optical_flow_magnitude(
    frame_a: np.ndarray, frame_b: np.ndarray,
) -> float:
    """Compute mean optical flow magnitude between two frames."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    return float(np.mean(mag))


def _compute_frame_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute SSIM between two RGB frames."""
    if not SKIMAGE_AVAILABLE:
        raise ImportError("skimage required for SSIM computation")
    return float(ssim(frame_a, frame_b, channel_axis=2, data_range=255))


def _compute_frame_lpips(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    lpips_fn: "lpips.LPIPS",
    device: str = "cuda",
) -> float:
    """Compute LPIPS between two RGB frames."""
    def to_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0
        return t.unsqueeze(0).to(device)

    with torch.no_grad():
        d = lpips_fn(to_tensor(frame_a), to_tensor(frame_b))
    return float(d.item())


def _compute_psnr(rendered: np.ndarray, gt: np.ndarray) -> float:
    """Compute PSNR between rendered and GT images."""
    mse = np.mean((rendered.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_temporal_metrics(
    frames: Sequence[np.ndarray],
    gt_frames: Optional[Sequence[np.ndarray]] = None,
    use_lpips: bool = True,
    flicker_threshold: float = 10.0,
    device: str = "cuda",
) -> TemporalStabilityResult:
    """Compute full temporal stability metrics for an image sequence.

    Args:
        frames: List of [H, W, 3] uint8 images (rendered or smoothed)
        gt_frames: Optional GT images for per-frame quality (PSNR)
        use_lpips: Whether to compute TLPIPS (requires lpips + torch)
        flicker_threshold: Mean pixel difference threshold for flicker detection
        device: Device for LPIPS computation

    Returns:
        TemporalStabilityResult with all metrics
    """
    n = len(frames)
    if n < 2:
        raise ValueError(f"Need at least 2 frames, got {n}")

    # Initialize LPIPS
    lpips_fn = None
    if use_lpips and LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()

    tof_values = []
    ssim_values = []
    lpips_values = []
    flicker_count = 0

    for i in range(n - 1):
        fa, fb = frames[i], frames[i + 1]

        # tOF
        tof_values.append(_compute_optical_flow_magnitude(fa, fb))

        # FF-SSIM
        if SKIMAGE_AVAILABLE:
            ssim_values.append(_compute_frame_ssim(fa, fb))

        # TLPIPS
        if lpips_fn is not None:
            lpips_values.append(_compute_frame_lpips(fa, fb, lpips_fn, device))

        # Flicker detection
        diff = np.mean(np.abs(fa.astype(float) - fb.astype(float)))
        if diff > flicker_threshold:
            flicker_count += 1

    # Per-frame quality against GT
    psnr_values = []
    ssim_quality = []
    if gt_frames is not None and len(gt_frames) == n:
        for rendered, gt in zip(frames, gt_frames):
            psnr_values.append(_compute_psnr(rendered, gt))
            if SKIMAGE_AVAILABLE:
                ssim_quality.append(_compute_frame_ssim(rendered, gt))

    tof_arr = np.array(tof_values)
    ssim_arr = np.array(ssim_values) if ssim_values else np.array([0.0])
    lpips_arr = np.array(lpips_values) if lpips_values else np.array([0.0])

    return TemporalStabilityResult(
        tof_mean=float(np.mean(tof_arr)),
        tof_std=float(np.std(tof_arr)),
        tlpips_mean=float(np.mean(lpips_arr)),
        tlpips_std=float(np.std(lpips_arr)),
        ff_ssim_mean=float(np.mean(ssim_arr)),
        ff_ssim_var=float(np.var(ssim_arr)),
        flicker_rate=float(flicker_count / max(n - 1, 1)),
        flicker_threshold=flicker_threshold,
        psnr_mean=float(np.mean(psnr_values)) if psnr_values else None,
        psnr_std=float(np.std(psnr_values)) if psnr_values else None,
        ssim_quality_mean=float(np.mean(ssim_quality)) if ssim_quality else None,
        per_pair_tof=tof_values,
        per_pair_tlpips=lpips_values,
        per_pair_ssim=ssim_values,
    )


# ============================================================
# I/O Utilities
# ============================================================

def load_frame_sequence(
    frame_dir: Union[str, Path],
    frame_ids: Optional[List[int]] = None,
    pattern: str = "{:05d}.png",
) -> List[np.ndarray]:
    """Load a sequence of frames from a directory.

    Args:
        frame_dir: Directory containing frame PNGs
        frame_ids: Specific frame IDs to load (sorted). If None, load all.
        pattern: Filename pattern (default: 5-digit zero-padded)

    Returns:
        List of [H, W, 3] uint8 RGB images
    """
    frame_dir = Path(frame_dir)
    if frame_ids is None:
        files = sorted(frame_dir.glob("*.png"))
        frame_ids = [int(f.stem) for f in files]

    frames = []
    for fid in sorted(frame_ids):
        path = frame_dir / pattern.format(fid)
        if not path.exists():
            raise FileNotFoundError(f"Frame not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to read: {path}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return frames


def save_frame_sequence(
    frames: Sequence[np.ndarray],
    output_dir: Union[str, Path],
    frame_ids: Optional[List[int]] = None,
    pattern: str = "{:05d}.png",
) -> None:
    """Save a sequence of frames to a directory.

    Args:
        frames: List of [H, W, 3] uint8 RGB images
        output_dir: Output directory
        frame_ids: Frame IDs for filenames. If None, use 0, 1, 2...
        pattern: Filename pattern
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if frame_ids is None:
        frame_ids = list(range(len(frames)))

    for fid, frame in zip(frame_ids, frames):
        path = output_dir / pattern.format(fid)
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
