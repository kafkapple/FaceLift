"""Turntable Renderer — SSOT for all turntable video generation.

Classes:
  - TurntableVideoConfig: Centralized defaults (rotation, speed, fps, etc.)
  - TurntableRenderer: Single-frame A-type videos (orbit, view_traj, grid)
  - TemporalVideoRenderer: Multi-frame B-type videos (time_fixed, time_rotating)

Output Files (A-type, single-frame):
  1. turntable_orbit_{uid}.mp4           — 360 deg synthetic orbit (CCW), fps=30
  2. turntable_orbit_with_input_{uid}.mp4 — orbit + labeled input strip
  3. turntable_view_with_input_{uid}.mp4  — 6-cam trajectory + hold + input strip
  4. turntable_{uid}.jpg                  — 6x6 grid image

Output Files (B-type, temporal batch):
  5. time_fixed.mp4 (+ _angle{N})        — fixed view, time variation
  6. time_rotating.mp4                    — time + rotation simultaneous
  7. turntable_grid.jpg                   — 6x6 grid of first frame

Dependencies (2026-03-23 refactored):
  - video_io.py: save_video (videoio + cv2 fallback)
  - camera_utils.py: get_dynamic_camera_order, compute_camera_convergence_center
  - grid_utils.py: grid creation, labels, input strip
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from einops import rearrange
from PIL import Image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TurntableVideoConfig:
    """Centralized turntable video configuration.

    All callers (train/val/inference) share these defaults.
    Override per-field via constructor kwargs or from_config().
    """

    # --- Orbit (360-degree synthetic rotation) ---
    orbit_views: int = 120
    orbit_fps: int = 30
    orbit_elevation: int = 20
    orbit_radius: float = 2.7

    # --- View trajectory (dataset-camera path) ---
    view_fps: int = 10
    view_hold_frames: int = 15       # ~1.5 s pause at each camera
    view_num_frames: int = 144       # total frames for smooth trajectory
    view_loop: bool = True
    view_show_overlay: bool = False  # "Cam X" text on frames
    view_fov_scale: float = 1.0     # FOV scaling (< 1.0 = zoom out)

    # --- Grid ---
    grid_rows: int = 6
    grid_cols: int = 6
    grid_add_row_labels: bool = True
    grid_label_position: str = "top"  # "top" or "left"
    grid_add_angle_overlay: bool = False

    # --- Input strip ---
    input_strip_height_ratio: float = 0.25

    # --- Output toggles ---
    save_orbit: bool = True
    save_orbit_with_input: bool = True
    save_view_with_input: bool = True
    save_grid: bool = True

    # --- Rotation direction (centralized) ---
    rotation_direction: str = "ccw"  # "ccw" (counter-clockwise) or "cw"
    view_smooth: bool = True  # Spline-based smooth view trajectory

    # --- Temporal batch ---
    temporal_fps: int = 10
    temporal_rotation_speed: float = 1.0

    @classmethod
    def from_config(cls, config) -> "TurntableVideoConfig":
        """Build from nested OmegaConf / dict config.

        Expected path: config.visualization.turntable.*
        """
        if config is None:
            return cls()
        cfg = config
        if hasattr(cfg, "get"):
            cfg = cfg.get("visualization", cfg)
            if hasattr(cfg, "get"):
                cfg = cfg.get("turntable", cfg)

        def _g(key: str, default):
            if hasattr(cfg, "get"):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        return cls(
            orbit_views=_g("orbit_views", cls.orbit_views),
            orbit_fps=_g("orbit_fps", _g("fps", cls.orbit_fps)),
            orbit_elevation=_g("elevation", cls.orbit_elevation),
            orbit_radius=_g("orbit_radius", _g("radius", cls.orbit_radius)),
            view_fps=_g("trajectory_fps", cls.view_fps),
            view_hold_frames=_g("hold_frames", cls.view_hold_frames),
            view_num_frames=_g("video_views", cls.view_num_frames),
            view_loop=_g("loop", cls.view_loop),
            view_show_overlay=_g("show_frame_overlay", cls.view_show_overlay),
            view_fov_scale=_g("trajectory_fov_scale", cls.view_fov_scale),
            grid_rows=_g("grid_rows", cls.grid_rows),
            grid_cols=_g("grid_cols", cls.grid_cols),
            grid_add_row_labels=_g("add_row_labels", cls.grid_add_row_labels),
            grid_label_position=_g("label_position", cls.grid_label_position),
            grid_add_angle_overlay=_g("add_angle_overlay", cls.grid_add_angle_overlay),
            save_orbit=_g("save_orbit_turntable", cls.save_orbit),
            save_orbit_with_input=_g("save_orbit_with_input", cls.save_orbit_with_input),
            save_view_with_input=_g("save_view_with_input", _g("save_video", cls.save_view_with_input)),
            save_grid=_g("save_grid", _g("save_video", cls.save_grid)),
            rotation_direction=_g("rotation_direction", cls.rotation_direction),
            view_smooth=_g("smooth_trajectory", cls.view_smooth),
            temporal_fps=_g("temporal_fps", cls.temporal_fps),
            temporal_rotation_speed=_g("temporal_rotation_speed", _g("rotation_speed", cls.temporal_rotation_speed)),
            input_strip_height_ratio=_g("input_strip_height_ratio", cls.input_strip_height_ratio),
        )


# ---------------------------------------------------------------------------
# Video I/O helper
# ---------------------------------------------------------------------------

from mouse_extensions.visualization.video_io import save_video as _save_video  # noqa: E402


# ---------------------------------------------------------------------------
# TurntableRenderer  (A-type: single-frame)
# ---------------------------------------------------------------------------

class TurntableRenderer:
    """Generates turntable videos & grid for a single reconstructed sample.

    Used by: training (gslrm.py), validation (validator.py), inference (gslrm_pipeline.py).
    """

    def __init__(self, config: Optional[TurntableVideoConfig] = None):
        self.cfg = config or TurntableVideoConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_all(
        self,
        gaussians,
        output_dir: str,
        uid: str,
        rendering_resolution: int,
        *,
        dataset_c2ws: Optional[np.ndarray] = None,
        dataset_fxfycxcy: Optional[np.ndarray] = None,
        original_resolution: Optional[int] = None,
        target_images=None,           # torch.Tensor [V, C, H, W]
        input_indices: Optional[List[int]] = None,
        view_indices: Optional[List[int]] = None,
        camera_order: Optional[List[int]] = None,
    ) -> Dict[str, str]:
        """Render all enabled turntable outputs and return {name: filepath}.

        Args:
            gaussians: GaussianModel for a single sample.
            output_dir: Directory to write output files.
            uid: Unique identifier for filenames.
            rendering_resolution: Square resolution for rendering.
            dataset_c2ws: [N, 4, 4] dataset camera poses (needed for view trajectory).
            dataset_fxfycxcy: [N, 4] dataset intrinsics (needed for view trajectory).
            original_resolution: Original input resolution (for intrinsics scaling).
            target_images: [V, C, H, W] tensor of all views (for input strip).
            input_indices: Which views are inputs (green border in strip).
            view_indices: Tensor-position -> camera-ID mapping.
            camera_order: CCW camera visitation order.
        """
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, str] = {}

        # Determine camera_order dynamically if not provided
        if camera_order is None and dataset_c2ws is not None:
            from mouse_extensions.visualization.camera_utils import get_dynamic_camera_order
            camera_order = get_dynamic_camera_order(dataset_c2ws)

        # Filter camera_order to valid indices
        if camera_order is not None and dataset_c2ws is not None:
            num_cams = dataset_c2ws.shape[0]
            camera_order = [c for c in camera_order if c < num_cams]

        # --- 1. View trajectory (dataset camera path) ---
        # Render first so grid can reuse these frames
        view_frames = None
        segments = None
        has_dataset_cameras = (dataset_c2ws is not None and dataset_fxfycxcy is not None)

        if has_dataset_cameras and (cfg.save_view_with_input or cfg.save_grid):
            view_frames, segments = self._render_view_trajectory(
                gaussians, dataset_c2ws, dataset_fxfycxcy,
                rendering_resolution, original_resolution, camera_order,
            )

        if cfg.save_view_with_input and view_frames is not None:
            strip = self._create_input_strip(
                target_images, camera_order, rendering_resolution,
                input_indices=input_indices, view_indices=view_indices,
            )
            path = os.path.join(output_dir, f"turntable_view_with_input_{uid}.mp4")
            if strip is not None:
                input_seq = np.tile(strip[None], (view_frames.shape[0], 1, 1, 1))
                combined = np.concatenate((view_frames, input_seq), axis=1)
                _save_video(combined, path, fps=cfg.view_fps)
            else:
                _save_video(view_frames, path, fps=cfg.view_fps)
            results["view_with_input"] = path

        # --- 2. Orbit turntable (360° synthetic) ---
        orbit_frames = None
        if cfg.save_orbit or cfg.save_orbit_with_input:
            orbit_frames = self._render_orbit(
                gaussians, dataset_c2ws, rendering_resolution,
            )

        if cfg.save_orbit and orbit_frames is not None:
            path = os.path.join(output_dir, f"turntable_orbit_{uid}.mp4")
            _save_video(orbit_frames, path, fps=cfg.orbit_fps)
            results["orbit"] = path

        if cfg.save_orbit_with_input and orbit_frames is not None:
            strip = self._create_input_strip(
                target_images, camera_order, rendering_resolution,
                input_indices=input_indices, view_indices=view_indices,
            )
            if strip is not None:
                path = os.path.join(output_dir, f"turntable_orbit_with_input_{uid}.mp4")
                input_seq = np.tile(strip[None], (orbit_frames.shape[0], 1, 1, 1))
                combined = np.concatenate((orbit_frames, input_seq), axis=1)
                _save_video(combined, path, fps=cfg.orbit_fps)
                results["orbit_with_input"] = path

        # --- 3. Grid image ---
        if cfg.save_grid:
            # Prefer view_trajectory frames (shows dataset cameras with labels)
            # Fall back to orbit frames
            if view_frames is not None:
                grid_image = rearrange(view_frames, "v h w c -> h (v w) c")
                grid_num = view_frames.shape[0]
                grid_seg = segments
            elif orbit_frames is not None:
                grid_image = rearrange(orbit_frames, "v h w c -> h (v w) c")
                grid_num = orbit_frames.shape[0]
                grid_seg = None
            else:
                grid_image = None
                grid_num = 0
                grid_seg = None

            if grid_image is not None:
                path = os.path.join(output_dir, f"turntable_{uid}.jpg")
                self._create_grid(
                    grid_image, grid_num, path,
                    segments=grid_seg, camera_order=camera_order,
                )
                results["grid"] = path

        return results

    # ------------------------------------------------------------------
    # Private rendering methods
    # ------------------------------------------------------------------

    def _render_orbit(
        self,
        gaussians,
        dataset_c2ws: Optional[np.ndarray],
        resolution: int,
    ) -> Optional[np.ndarray]:
        """Render 360° orbit and return [V, H, W, 3] uint8 array."""
        from mouse_extensions.visualization import render_turntable
        from mouse_extensions.visualization.camera_utils import compute_camera_convergence_center

        cfg = self.cfg

        # Compute orbit center
        if dataset_c2ws is not None:
            center = compute_camera_convergence_center(dataset_c2ws)
        else:
            # Fallback: Gaussian centroid
            center = gaussians._xyz.mean(dim=0).detach().cpu().numpy()

        try:
            # Physical CCW (from above) = CW in standard math XY coords
            # Camera layout uses atan2(x,y) convention (angle from +Y axis),
            # while orbit uses cos->x, sin->y (angle from +X axis).
            clockwise = (cfg.rotation_direction == "ccw")
            orbit_image = render_turntable(
                gaussians,
                rendering_resolution=resolution,
                num_views=cfg.orbit_views,
                elevation=cfg.orbit_elevation,
                radius=cfg.orbit_radius,
                trajectory_mode="turntable",
                center=center,
                clockwise=clockwise,
            )
            # orbit_image: [H, V*W, 3] -> [V, H, W, 3]
            h = orbit_image.shape[0]
            w = orbit_image.shape[1] // cfg.orbit_views
            frames = orbit_image.reshape(h, cfg.orbit_views, w, 3)
            frames = np.transpose(frames, (1, 0, 2, 3))
            return np.ascontiguousarray(frames)
        except Exception as exc:
            print(f"Warning: orbit turntable failed: {exc}")
            return None

    def _render_view_trajectory(
        self,
        gaussians,
        dataset_c2ws: np.ndarray,
        dataset_fxfycxcy: np.ndarray,
        rendering_resolution: int,
        original_resolution: Optional[int],
        camera_order: Optional[List[int]],
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """Render dataset-camera trajectory. Returns ([V,H,W,3], segments)."""
        from mouse_extensions.visualization import render_dataset_trajectory

        cfg = self.cfg

        # Apply FOV scaling
        scaled_fxfycxcy = dataset_fxfycxcy.copy()
        if cfg.view_fov_scale != 1.0:
            scaled_fxfycxcy[:, :2] *= cfg.view_fov_scale

        try:
            frames, segments = render_dataset_trajectory(
                gaussians,
                dataset_c2ws, scaled_fxfycxcy,
                rendering_resolution=rendering_resolution,
                num_views=cfg.view_num_frames,
                camera_order=camera_order,
                loop=cfg.view_loop,
                show_overlay=cfg.view_show_overlay,
                original_resolution=original_resolution,
                hold_frames=cfg.view_hold_frames,
                smooth=cfg.view_smooth,
            )
            return np.ascontiguousarray(frames), segments
        except Exception as exc:
            print(f"Warning: view trajectory failed: {exc}")
            return None, None

    def _create_input_strip(
        self,
        target_images,    # torch.Tensor [V, C, H, W] or None
        camera_order: Optional[List[int]],
        rendering_resolution: int,
        *,
        input_indices: Optional[List[int]] = None,
        view_indices: Optional[List[int]] = None,
    ) -> Optional[np.ndarray]:
        """Create labeled input strip. Returns [H_strip, W, 3] uint8 or None."""
        if target_images is None or camera_order is None:
            return None

        from mouse_extensions.visualization.grid_utils import create_labeled_input_strip

        strip_h = int(rendering_resolution * self.cfg.input_strip_height_ratio)
        try:
            return create_labeled_input_strip(
                target_images,
                camera_order=camera_order,
                target_h=strip_h,
                target_w=rendering_resolution,
                border=2,
                input_indices=input_indices,
                view_indices=view_indices,
            )
        except Exception as exc:
            print(f"Warning: input strip creation failed: {exc}")
            return None

    @staticmethod
    def _assemble_row(
        frames: np.ndarray,  # [N, H, W, 3]
        cols: int,
        padding: int,
    ) -> np.ndarray:
        """Arrange frames into a grid row(s). Returns [grid_H, grid_W, 3]."""
        n = frames.shape[0]
        h, w = frames.shape[1], frames.shape[2]
        rows = (n + cols - 1) // cols

        # Pad to fill grid
        total = rows * cols
        if n < total:
            pad = np.zeros((total - n, h, w, 3), dtype=frames.dtype)
            frames = np.concatenate([frames, pad], axis=0)

        grid_h = rows * h + (rows - 1) * padding
        grid_w = cols * w + (cols - 1) * padding
        grid = np.zeros((grid_h, grid_w, 3), dtype=frames.dtype)

        for i in range(total):
            r, c = i // cols, i % cols
            y = r * (h + padding)
            x = c * (w + padding)
            grid[y:y + h, x:x + w] = frames[i]

        return grid

    @staticmethod
    def _make_label_bar(
        width: int, height: int, text: str,
        bg_color: tuple = (40, 40, 40),
        text_color: tuple = (255, 255, 255),
    ) -> np.ndarray:
        """Create a text label bar."""
        bar = np.full((height, width, 3), bg_color, dtype=np.uint8)
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = (width - tw) // 2
            y = (height + th) // 2
            cv2.putText(bar, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
        except Exception:
            pass
        return bar

    def _create_grid(
        self,
        source_image: np.ndarray,   # [H, V*W, C]
        num_views: int,
        output_path: str,
        *,
        segments: Optional[list] = None,
        camera_order: Optional[List[int]] = None,
    ) -> None:
        """Create and save grid image (e.g. 6x6)."""
        from mouse_extensions.visualization.grid_utils import (
            create_grid_from_video,
            add_angle_overlay_to_grid,
            add_row_labels_to_grid,
            add_left_row_labels,
        )

        cfg = self.cfg
        grid, _all_frames, h_img = create_grid_from_video(
            source_image, num_views,
            cfg.grid_rows, cfg.grid_cols,
            segments=segments,
        )

        if cfg.grid_add_row_labels and camera_order is not None:
            if cfg.grid_label_position == "left":
                grid = add_left_row_labels(grid, camera_order, cfg.grid_rows, cfg.grid_cols, h_img)
            else:
                grid = add_row_labels_to_grid(grid, camera_order, cfg.grid_rows, cfg.grid_cols, h_img)

        if cfg.grid_add_angle_overlay:
            grid = add_angle_overlay_to_grid(grid, cfg.grid_rows, cfg.grid_cols)

        Image.fromarray(grid).save(output_path, quality=95)


# ---------------------------------------------------------------------------
# TemporalVideoRenderer  (B-type: multi-frame batch)
# ---------------------------------------------------------------------------

class TemporalVideoRenderer:
    """Generates temporal videos from multiple turntable sequences.

    Used by: batch inference (run_e2e_inference.py).
    """

    def __init__(self, config: Optional[TurntableVideoConfig] = None):
        self.cfg = config or TurntableVideoConfig()

    def render_temporal(
        self,
        all_turntables: List[np.ndarray],
        output_dir: str,
        *,
        fps: Optional[int] = None,
        fixed_angles: Optional[List[int]] = None,
        rotation_speed: Optional[float] = None,
        grid_views: int = 36,
    ) -> Dict[str, str]:
        """Generate temporal videos from list of turntable frame arrays.

        Args:
            all_turntables: List of [V, H, W, 3] uint8 arrays (one per time step).
            output_dir: Directory to write output files.
            fps: Output FPS (default: cfg.temporal_fps).
            fixed_angles: Angle indices for time_fixed videos (default: [0]).
            rotation_speed: Speed factor (default: cfg.temporal_rotation_speed).
            grid_views: Number of views for grid image.

        Returns:
            Dict mapping output name to file path.
        """
        cfg = self.cfg
        fps = fps or cfg.temporal_fps
        fixed_angles = fixed_angles if fixed_angles is not None else [0]
        rotation_speed = rotation_speed if rotation_speed is not None else cfg.temporal_rotation_speed

        if len(all_turntables) < 2:
            print(f"Need at least 2 turntables for temporal videos, found {len(all_turntables)}")
            return {}

        os.makedirs(output_dir, exist_ok=True)
        results: Dict[str, str] = {}

        T = len(all_turntables)
        V = all_turntables[0].shape[0]

        # Apply rotation speed adjustment
        if rotation_speed != 1.0:
            all_turntables = [self._interpolate_speed(t, rotation_speed) for t in all_turntables]
            V = all_turntables[0].shape[0]

        # --- time_fixed: fixed angle, time variation ---
        for angle_idx in fixed_angles:
            angle_idx = angle_idx % V
            frames = np.stack([t[angle_idx] for t in all_turntables])
            suffix = f"_angle{angle_idx}" if len(fixed_angles) > 1 else ""
            path = os.path.join(output_dir, f"time_fixed{suffix}.mp4")
            _save_video(frames, path, fps=fps)
            results[f"time_fixed{suffix}"] = path

        # --- time_rotating: rotation + time ---
        rotating = []
        for t_idx in range(T):
            angle = (t_idx * V // T) % V
            rotating.append(all_turntables[t_idx][angle])
        path = os.path.join(output_dir, "time_rotating.mp4")
        _save_video(np.stack(rotating), path, fps=fps)
        results["time_rotating"] = path

        # --- turntable_grid.jpg (first time-step) ---
        try:
            first = all_turntables[0]
            n_grid = min(grid_views, V)
            cols = 6
            rows = (n_grid + cols - 1) // cols
            total = rows * cols

            if V > n_grid:
                indices = np.linspace(0, V - 1, n_grid, dtype=int)
                arr = first[indices]
            else:
                arr = first
                n_grid = V

            if n_grid < total:
                pad = np.zeros_like(arr[0])
                arr = np.concatenate([arr, np.stack([pad] * (total - n_grid))])

            grid = rearrange(arr[:total], "(r c) h w ch -> (r h) (c w) ch", r=rows, c=cols)
            path = os.path.join(output_dir, "turntable_grid.jpg")
            Image.fromarray(grid).save(path, quality=95)
            results["grid"] = path
        except Exception as exc:
            print(f"Warning: temporal grid skipped: {exc}")

        # --- time_grid_6view: 6 viewpoints x temporal progression ---
        try:
            n_views = min(6, V)
            view_indices = [int(i * V / n_views) for i in range(n_views)]
            grid_cols = 3
            grid_rows = (n_views + grid_cols - 1) // grid_cols

            h, w = all_turntables[0].shape[1], all_turntables[0].shape[2]
            padding = 4
            grid_h = grid_rows * h + (grid_rows - 1) * padding
            grid_w = grid_cols * w + (grid_cols - 1) * padding

            grid_frames = []
            for t_idx in range(T):
                views = np.stack([all_turntables[t_idx][v] for v in view_indices])
                # Assemble 2x3 grid
                grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                for i in range(n_views):
                    r, c = i // grid_cols, i % grid_cols
                    y = r * (h + padding)
                    x = c * (w + padding)
                    grid[y:y + h, x:x + w] = views[i]
                grid_frames.append(grid)

            path = os.path.join(output_dir, "time_grid_6view.mp4")
            _save_video(np.stack(grid_frames), path, fps=fps)
            results["time_grid_6view"] = path
        except Exception as exc:
            print(f"Warning: time_grid_6view failed: {exc}")

        return results

    @staticmethod
    def _interpolate_speed(frames: np.ndarray, speed_factor: float) -> np.ndarray:
        """Adjust rotation speed by resampling frames."""
        if speed_factor == 1.0:
            return frames
        num_original = frames.shape[0]
        num_target = max(1, int(num_original / speed_factor))
        indices = np.linspace(0, num_original - 1, num_target)
        return np.stack([frames[int(np.round(idx))] for idx in indices])

    @staticmethod
    def load_turntable_videos(
        output_dir: str,
        video_pattern: str = "turntable_orbit_*.mp4",
    ) -> List[np.ndarray]:
        """Load turntable videos from samples/ subdirectory.

        Searches for video_pattern (glob), then falls back to legacy "turntable.mp4".
        Returns list of [V, H, W, 3] uint8 arrays sorted by sample directory.
        """
        output_path = Path(output_dir)
        samples_dir = output_path / "samples"
        if samples_dir.exists():
            sample_dirs = sorted(d for d in samples_dir.iterdir() if d.is_dir())
        else:
            sample_dirs = sorted(
                d for d in output_path.iterdir()
                if d.is_dir() and d.name != "samples"
            )

        all_turntables: List[np.ndarray] = []
        for sd in sample_dirs:
            # Try glob pattern first (handles uid in filename)
            matches = sorted(sd.glob(video_pattern))
            if not matches:
                matches = sorted(sd.glob("turntable.mp4"))  # legacy fallback
            if not matches:
                # Check cam_* subdirectories
                matches = sorted(sd.glob(f"cam_*/{video_pattern}"))
                if not matches:
                    matches = sorted(sd.glob("cam_*/turntable.mp4"))
            if not matches:
                continue

            video_path = matches[0]
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                all_turntables.append(np.stack(frames))

        return all_turntables
