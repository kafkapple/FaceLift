"""Grid composer for multi-experiment comparison visualization.

Takes rendered frame sequences from multiple experiments and composites
them into side-by-side comparison grids (images) and videos.

Supports:
    - Arbitrary grid layouts (rows x cols)
    - Per-experiment text labels
    - Video output (H.264 via ffmpeg) or image sequences
    - Optional per-frame metrics overlay (PSNR, IoU, etc.)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class GridComposer:
    """Compose rendered frames from multiple experiments into comparison grids.

    Usage:
        composer = GridComposer(
            layout=(1, 3),
            labels=["baseline", "alpha=0.5", "alpha=1.0"],
            resolution=512,
        )
        # frames: dict mapping experiment name -> (N, H, W, 3) uint8 array
        composer.compose_video(frames, output_path="comparison.mp4", fps=30)
        composer.compose_image(frames, frame_idx=0, output_path="frame0.png")
    """

    def __init__(
        self,
        layout: Tuple[int, int] = (1, 2),
        labels: Optional[List[str]] = None,
        resolution: int = 512,
        label_height: int = 32,
        padding: int = 4,
        bg_color: Tuple[int, int, int] = (30, 30, 30),
        label_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.6,
    ):
        """
        Args:
            layout: (rows, cols) grid layout
            labels: experiment labels (length = rows * cols)
            resolution: per-cell resolution (square)
            label_height: height of text label bar per cell
            padding: pixel padding between cells
            bg_color: background/padding color
            label_color: text color for labels
            font_scale: OpenCV font scale
        """
        self.rows, self.cols = layout
        self.n_cells = self.rows * self.cols
        self.labels = labels
        self.resolution = resolution
        self.label_height = label_height
        self.padding = padding
        self.bg_color = bg_color
        self.label_color = label_color
        self.font_scale = font_scale

    @property
    def cell_height(self) -> int:
        return self.resolution + self.label_height

    @property
    def grid_width(self) -> int:
        return self.cols * self.resolution + (self.cols + 1) * self.padding

    @property
    def grid_height(self) -> int:
        return self.rows * self.cell_height + (self.rows + 1) * self.padding

    def _make_label_bar(self, text: str) -> np.ndarray:
        """Create a text label bar for one cell."""
        bar = np.full(
            (self.label_height, self.resolution, 3),
            self.bg_color, dtype=np.uint8,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, self.font_scale, thickness)
        x = (self.resolution - tw) // 2
        y = (self.label_height + th) // 2
        cv2.putText(
            bar, text, (x, y), font, self.font_scale,
            self.label_color, thickness, cv2.LINE_AA,
        )
        return bar

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a single frame to target resolution."""
        h, w = frame.shape[:2]
        if h == self.resolution and w == self.resolution:
            return frame
        return cv2.resize(frame, (self.resolution, self.resolution))

    def compose_frame(
        self,
        frames: Dict[str, np.ndarray],
        frame_idx: int = 0,
        metrics: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Compose a single grid frame from multiple experiments.

        Args:
            frames: {experiment_name: (N, H, W, 3) or (H, W, 3)} uint8 arrays
            frame_idx: which frame to extract from sequences
            metrics: optional {experiment_name: "PSNR=23.5"} overlay text

        Returns:
            (grid_H, grid_W, 3) uint8 composite image
        """
        grid = np.full(
            (self.grid_height, self.grid_width, 3),
            self.bg_color, dtype=np.uint8,
        )

        experiment_names = list(frames.keys())

        for cell_idx in range(min(self.n_cells, len(experiment_names))):
            row = cell_idx // self.cols
            col = cell_idx % self.cols

            name = experiment_names[cell_idx]
            seq = frames[name]

            # Extract single frame
            if seq.ndim == 4:
                fi = min(frame_idx, seq.shape[0] - 1)
                img = seq[fi]
            else:
                img = seq  # already single frame

            img = self._resize_frame(img)

            # Label bar
            label_text = self.labels[cell_idx] if self.labels else name
            if metrics and name in metrics:
                label_text = f"{label_text}  |  {metrics[name]}"
            label_bar = self._make_label_bar(label_text)

            # Compute cell position
            y = self.padding + row * (self.cell_height + self.padding)
            x = self.padding + col * (self.resolution + self.padding)

            # Place label + image
            grid[y:y + self.label_height, x:x + self.resolution] = label_bar
            grid[y + self.label_height:y + self.cell_height, x:x + self.resolution] = img

        return grid

    def compose_image(
        self,
        frames: Dict[str, np.ndarray],
        frame_idx: int = 0,
        output_path: Optional[str] = None,
        metrics: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Compose and optionally save a single comparison image.

        Returns:
            Grid image as uint8 numpy array
        """
        grid = self.compose_frame(frames, frame_idx, metrics)

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # Convert RGB to BGR for cv2, or save via PIL
            Image.fromarray(grid).save(output_path, quality=95)

        return grid

    def compose_video(
        self,
        frames: Dict[str, np.ndarray],
        output_path: str,
        fps: int = 30,
        metrics_per_frame: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Compose and save a comparison video.

        Args:
            frames: {experiment_name: (N, H, W, 3)} uint8 frame sequences
            output_path: output .mp4 path
            fps: video framerate
            metrics_per_frame: optional per-frame metrics list

        Returns:
            output_path
        """
        # Determine frame count from shortest sequence
        experiment_names = list(frames.keys())
        seqs_4d = [seq for seq in frames.values() if seq.ndim == 4]
        if not seqs_4d:
            raise ValueError("compose_video requires at least one 4D frame sequence (N,H,W,3)")
        n_frames = min(seq.shape[0] for seq in seqs_4d)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try ffmpeg pipe for H.264
        h, w = self.grid_height, self.grid_width
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "23",
            output_path,
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )

            for i in range(n_frames):
                metrics = metrics_per_frame[i] if metrics_per_frame else None
                grid = self.compose_frame(frames, i, metrics)
                proc.stdin.write(grid.tobytes())

            proc.stdin.close()
            proc.wait()

            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.read().decode()[:500])

        except (FileNotFoundError, RuntimeError) as e:
            # Fallback: cv2 VideoWriter
            print(f"ffmpeg unavailable ({e}), falling back to cv2")
            self._compose_video_cv2(frames, output_path, fps, n_frames, metrics_per_frame)

        return output_path

    def _compose_video_cv2(
        self,
        frames: Dict[str, np.ndarray],
        output_path: str,
        fps: int,
        n_frames: int,
        metrics_per_frame: Optional[List[Dict[str, str]]],
    ) -> None:
        """Fallback video writer using cv2."""
        h, w = self.grid_height, self.grid_width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for i in range(n_frames):
            metrics = metrics_per_frame[i] if metrics_per_frame else None
            grid = self.compose_frame(frames, i, metrics)
            writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        writer.release()

    def compose_strip(
        self,
        frames: Dict[str, np.ndarray],
        frame_indices: Optional[List[int]] = None,
        n_samples: int = 8,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Create a temporal strip: rows=experiments, cols=time samples.

        Useful for showing temporal progression of each experiment.

        Args:
            frames: {experiment_name: (N, H, W, 3)}
            frame_indices: specific frames to sample (overrides n_samples)
            n_samples: number of evenly-spaced frames if indices not given
            output_path: optional save path

        Returns:
            Strip image as uint8 numpy array
        """
        experiment_names = list(frames.keys())
        n_exp = len(experiment_names)

        # Determine sample indices
        first_seq = frames[experiment_names[0]]
        total = first_seq.shape[0]
        if frame_indices is None:
            frame_indices = np.linspace(0, total - 1, n_samples, dtype=int).tolist()
        n_cols = len(frame_indices)

        # Thumbnail size
        thumb = self.resolution // 2
        label_w = 120
        pad = 2

        strip_w = label_w + n_cols * (thumb + pad) + pad
        strip_h = n_exp * (thumb + pad) + pad

        strip = np.full((strip_h, strip_w, 3), self.bg_color, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX

        for row, name in enumerate(experiment_names):
            seq = frames[name]
            y = pad + row * (thumb + pad)

            # Row label
            label = self.labels[row] if self.labels and row < len(self.labels) else name
            (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
            cv2.putText(
                strip, label,
                ((label_w - tw) // 2, y + (thumb + th) // 2),
                font, 0.4, self.label_color, 1, cv2.LINE_AA,
            )

            for col, fi in enumerate(frame_indices):
                fi = min(fi, seq.shape[0] - 1)
                img = cv2.resize(seq[fi], (thumb, thumb))
                x = label_w + pad + col * (thumb + pad)
                strip[y:y + thumb, x:x + thumb] = img

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(strip).save(output_path, quality=95)

        return strip
