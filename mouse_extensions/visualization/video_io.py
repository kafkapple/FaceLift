"""Video I/O utilities for visualization output.

Provides video saving with videoio backend and cv2 fallback.
Consolidated from turntable_renderer.py and gaussians_renderer.py (2026-03-23).
"""

from __future__ import annotations

import cv2
import numpy as np


def imageseq2video(images: np.ndarray, filename: str, fps: int = 24) -> None:
    """Save image sequence to video using videoio (lossless-capable, fast).

    Args:
        images: (N, H, W, 3) uint8 or float32 array.
        filename: Output video path.
        fps: Frames per second.
    """
    import videoio

    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0
    videoio.videosave(filename, images, lossless=False, preset="medium", fps=fps)


def save_video(frames: np.ndarray, path: str, fps: int = 30) -> bool:
    """Save video frames with imageseq2video, falling back to cv2.

    Args:
        frames: (N, H, W, 3) uint8 or float32 array.
        path: Output video path.
        fps: Frames per second.

    Returns:
        True if saved successfully, False otherwise.
    """
    try:
        imageseq2video(frames, path, fps=fps)
        return True
    except Exception as exc:
        print(f"Warning: videoio failed ({type(exc).__name__}), trying cv2 fallback: {exc}")

    try:
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MPEG-4, reliable cross-platform fallback
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print(f"Warning: Could not open video writer for {path}")
            return False
        for frame in frames:
            if frame.dtype in (np.float32, np.float64):
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        return True
    except Exception as exc2:
        print(f"Warning: cv2 fallback also failed for {path}: {exc2}")
        return False
