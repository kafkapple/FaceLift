"""
Video encoding utilities for FaceLift Mouse extensions.

Provides functions for encoding image sequences to video files.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def encode_video_imageio(
    frames: Union[np.ndarray, List[np.ndarray]],
    output_path: Union[str, Path],
    fps: int = 6,
    codec: str = "libx264",
    quality: int = 23,
    pixel_format: str = "yuv420p",
) -> Path:
    """
    Encode frames to MP4 video using OpenCV (more reliable than imageio).

    Args:
        frames: Video frames [T, H, W, 3] uint8 or list of [H, W, 3]
        output_path: Output path (.mp4)
        fps: Frames per second (default 6 for 5-interval sampled data)
        codec: Video codec (ignored, uses mp4v)
        quality: CRF quality (ignored)
        pixel_format: Pixel format (ignored)

    Returns:
        Path to encoded video
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure output has .mp4 extension
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")

    # Convert to numpy array if list
    if isinstance(frames, list):
        frames = np.array(frames)
    
    # Get video dimensions
    T, H, W, C = frames.shape
    
    # Use mp4v codec (widely compatible)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()
    print(f"Video saved: {output_path} ({T} frames @ {fps}fps, {T/fps:.1f}s)")
    return output_path


def encode_video_cv2(
    frames: Union[np.ndarray, List[np.ndarray]],
    output_path: Union[str, Path],
    fps: int = 6,
    fourcc: str = "mp4v",
) -> Path:
    """
    Fallback video encoder using OpenCV.

    Args:
        frames: Video frames [T, H, W, 3] uint8 (RGB)
        output_path: Output path (.mp4)
        fps: Frames per second
        fourcc: FourCC codec code

    Returns:
        Path to encoded video
    """
    return encode_video_imageio(frames, output_path, fps)


def create_grid_image(
    frames: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Create a grid image from multiple frames.

    Args:
        frames: [N, H, W, 3] uint8 frames
        rows: Number of rows in grid
        cols: Number of columns in grid

    Returns:
        Grid image [rows*H, cols*W, 3]
    """
    from einops import rearrange

    N, H, W, C = frames.shape
    assert N <= rows * cols, f"Too many frames ({N}) for grid ({rows}x{cols}={rows*cols})"

    # Pad if necessary
    if N < rows * cols:
        padding = np.zeros((rows * cols - N, H, W, C), dtype=frames.dtype)
        frames = np.concatenate([frames, padding], axis=0)

    # Rearrange to grid
    grid = rearrange(frames, "(r c) h w ch -> (r h) (c w) ch", r=rows, c=cols)
    return grid


def create_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    labels: Optional[tuple] = None,
) -> np.ndarray:
    """
    Create side-by-side comparison of two image sequences.

    Args:
        left: [T, H, W, 3] or [H, W, 3] left images
        right: [T, H, W, 3] or [H, W, 3] right images
        labels: Optional (left_label, right_label) tuple

    Returns:
        Combined frames [T, H, 2*W, 3] or [H, 2*W, 3]
    """
    import cv2

    if left.ndim == 3:
        left = left[np.newaxis]
        right = right[np.newaxis]

    combined = np.concatenate([left, right], axis=2)  # Concat along width

    if labels:
        for i in range(len(combined)):
            cv2.putText(combined[i], labels[0], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined[i], labels[1], (left.shape[2] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return combined.squeeze() if combined.shape[0] == 1 else combined
