# no-split: unified inference viz pipeline (load → infer → render → video), single entry point
"""RAT multi-frame inference visualization.

Runs GS-LRM inference on rat preprocessing versions and generates:
1. Per-frame turntable orbit MP4
2. Temporal video (fixed angle, time advancing) across N frames
3. Input 6-view + reconstruction comparison grid

Reuses existing mouse modules:
  - GSLRMInference (gslrm_pipeline.py)
  - render_gaussian_at_view (fl_gt_view_comparison.py)
  - get_turntable_cameras (camera_utils.py)
  - save_video (video_io.py)

Usage (gpu03):
    # Zero-shot (mouse pretrained → rat data)
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.viz_rat_inference \
        --checkpoint M5t2_6view_alpha03_v3 \
        --data-version gslrm_format_rat2_despilled_fxnorm \
        --num-frames 10 --frame-stride 30

    # After fine-tuning
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.viz_rat_inference \
        --checkpoint RAT2_despilled_rat_ft_v5 \
        --data-version gslrm_format_rat2_despilled_fxnorm \
        --num-frames 10
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.scripts.eval.fl_gt_view_comparison import render_gaussian_at_view
from mouse_extensions.visualization.camera_utils import get_turntable_cameras
from mouse_extensions.visualization.video_io import save_video


# Defaults
DEFAULT_DATA_ROOT = "/node_data/joon/data/preprocessed/FaceLift_rat"
DEFAULT_DATA_VERSION = "rat2_s1despill_s2fxnorm"
DEFAULT_CKPT_DIR = "/node_data/joon/checkpoints/FaceLift/gslrm"
DEFAULT_CONFIG = "/home/joon/dev/FaceLift/configs/base/gslrm_mouse.yaml"
DEFAULT_OUTPUT = "outputs/viz/rat_inference"


def _get_frame_dirs(data_dir: Path, num_frames: int, stride: int,
                    start: int = 0) -> List[Path]:
    """Get evenly-spaced frame directories."""
    all_dirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    selected = all_dirs[start::stride][:num_frames]
    return selected


def _make_input_strip(sample_dir: Path, target_h: int = 128) -> np.ndarray:
    """Create a horizontal strip of 6 input views, resized to target_h."""
    from PIL import Image
    imgs = []
    for i in range(6):
        img = np.array(Image.open(sample_dir / "images" / f"cam_{i:03d}.png"))
        rgb = img[:, :, :3]
        scale = target_h / rgb.shape[0]
        resized = cv2.resize(rgb, None, fx=scale, fy=scale)
        imgs.append(resized)
    return np.concatenate(imgs, axis=1)


def run_inference_viz(
    checkpoint: str,
    data_version: str,
    num_frames: int = 10,
    frame_stride: int = 30,
    frame_start: int = 0,
    orbit_views: int = 60,
    orbit_elevation: float = 20.0,
    render_size: int = 256,
    fps: int = 20,
    output_dir: Optional[str] = None,
    config_path: str = DEFAULT_CONFIG,
    device: str = "cuda",
):
    """Main inference visualization pipeline."""
    data_dir = Path(DEFAULT_DATA_ROOT) / data_version
    ckpt_path = Path(DEFAULT_CKPT_DIR) / checkpoint

    if output_dir is None:
        output_dir = f"{DEFAULT_OUTPUT}/{data_version}/{checkpoint}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {checkpoint}")
    pipeline = GSLRMInference(config_path, str(ckpt_path), device=device)

    # Get frame directories
    frame_dirs = _get_frame_dirs(data_dir, num_frames, frame_stride, frame_start)
    print(f"Processing {len(frame_dirs)} frames from {data_version}")

    # Pre-compute orbit cameras
    _, _, _, orbit_fxfycxcy, orbit_c2ws = get_turntable_cameras(
        num_views=orbit_views, w=render_size, h=render_size,
        elevation=orbit_elevation,
    )
    orbit_cams = [(orbit_c2ws[i], orbit_fxfycxcy[i]) for i in range(orbit_views)]

    # Storage for temporal video
    all_orbit_frames = []  # [T, V, H, W, 3]
    all_input_strips = []  # [T, H_strip, W_strip, 3]
    frame_labels = []

    for fi, fdir in enumerate(tqdm(frame_dirs, desc="Inference")):
        # Load data
        images, c2ws, fxfycxcys, index = load_sample_data(
            str(fdir), image_size=512, device=device,
        )

        # Inference — result.gaussians is a list of GaussianModel per batch
        with torch.no_grad():
            result = pipeline.predict(images, c2ws, fxfycxcys, index)
            gaussians = result.gaussians[0]

        # Render orbit
        orbit_renders = []
        for cam in orbit_cams:
            rendered = render_gaussian_at_view(
                gaussians, cam, resolution=render_size,
                bg_color=(1.0, 1.0, 1.0), device=device,
            )
            orbit_renders.append(rendered)
        orbit_array = np.stack(orbit_renders)  # [V, H, W, 3]
        all_orbit_frames.append(orbit_array)

        # Input strip
        input_strip = _make_input_strip(fdir, target_h=render_size)
        all_input_strips.append(input_strip)
        frame_labels.append(fdir.name)

    # === Save outputs ===

    # 1. Per-frame orbit MP4 (first and last frame only to save time)
    for idx in [0, len(frame_dirs) - 1]:
        orbit_path = out / f"orbit_{frame_labels[idx]}.mp4"
        save_video(all_orbit_frames[idx], str(orbit_path), fps=30)
        print(f"  Orbit: {orbit_path}")

    # 2. Temporal video — fixed front angle, time advancing
    front_idx = 0  # first orbit angle (front view)
    temporal_frames = []
    for t in range(len(all_orbit_frames)):
        render = all_orbit_frames[t][front_idx]  # [H, W, 3]
        # Add frame label
        labeled = render.copy()
        cv2.putText(labeled, f"F:{frame_labels[t]}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        temporal_frames.append(labeled)
    temporal_path = out / "temporal_front.mp4"
    save_video(np.stack(temporal_frames), str(temporal_path), fps=fps)
    print(f"  Temporal (front): {temporal_path}")

    # 3. Temporal video — rotating orbit + time advancing
    rotating_frames = []
    for t in range(len(all_orbit_frames)):
        angle_idx = (t * orbit_views // len(all_orbit_frames)) % orbit_views
        render = all_orbit_frames[t][angle_idx].copy()
        cv2.putText(render, f"F:{frame_labels[t]}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rotating_frames.append(render)
    rotating_path = out / "temporal_rotating.mp4"
    save_video(np.stack(rotating_frames), str(rotating_path), fps=fps)
    print(f"  Temporal (rotating): {rotating_path}")

    # 4. Comparison grid: input views (top) + 4 orbit angles (bottom)
    grid_angles = [0, orbit_views // 4, orbit_views // 2, 3 * orbit_views // 4]
    grid_frames = []
    for t in range(len(all_orbit_frames)):
        # Top: input strip resized to match width
        renders_row = np.concatenate(
            [all_orbit_frames[t][a] for a in grid_angles], axis=1
        )  # [H, 4*W, 3]
        # Resize input strip to match width
        strip = all_input_strips[t]
        target_w = renders_row.shape[1]
        strip_resized = cv2.resize(strip, (target_w, render_size))
        # Stack: input on top, renders on bottom
        combined = np.concatenate([strip_resized, renders_row], axis=0)
        # Add frame label
        cv2.putText(combined, f"Frame {frame_labels[t]}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        grid_frames.append(combined)

    grid_path = out / "input_vs_recon.mp4"
    save_video(np.stack(grid_frames), str(grid_path), fps=fps)
    print(f"  Grid (input vs recon): {grid_path}")

    print(f"\nDone. All outputs in {out}")


def main():
    parser = argparse.ArgumentParser(
        description="RAT multi-frame inference visualization"
    )
    parser.add_argument("--checkpoint", default="M5t2_6view_alpha03_v3",
                        help="Checkpoint name under gslrm/")
    parser.add_argument("--data-version", default=DEFAULT_DATA_VERSION,
                        help="Preprocessing version dirname")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=30,
                        help="Stride between frames (30 = every 30th dir)")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--orbit-views", type=int, default=60)
    parser.add_argument("--orbit-elevation", type=float, default=20.0)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_inference_viz(
        checkpoint=args.checkpoint,
        data_version=args.data_version,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        frame_start=args.frame_start,
        orbit_views=args.orbit_views,
        orbit_elevation=args.orbit_elevation,
        render_size=args.render_size,
        fps=args.fps,
        output_dir=args.output_dir,
        config_path=args.config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
