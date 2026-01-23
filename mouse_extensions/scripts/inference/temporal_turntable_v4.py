#!/usr/bin/env python3
"""
Temporal Turntable Video Generation v4

Fixes from v3:
- Fixed center across all frames (uses first frame center)
- Option to disable Gaussian filtering
- Saves both standard 360° turntable AND temporal video
- Smoother frame transitions
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from easydict import EasyDict as edict
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gslrm.model.gaussians_renderer import (
    render_turntable,
    imageseq2video,
    render_opencv_cam,
    GaussianModel,
)


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load GS-LRM model from checkpoint."""
    from gslrm.model.gslrm import GSLRM

    print(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = GSLRM(config)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    step = checkpoint.get("fwdbwd_pass_step", checkpoint.get("step", "unknown"))
    print(f"Model loaded. Step: {step}")
    return model, config


def load_single_sample(
    sample_dir: Path,
    device: str = "cuda",
    target_camera_distance: float = 2.7,
) -> edict:
    """Load a single sample with preprocessing."""
    import cv2
    from mouse_extensions.data import (
        normalize_camera_distance_with_intrinsics,
        normalize_cameras_to_z_up,
        get_bg_color,
    )

    cam_path = sample_dir / "opencv_cameras.json"
    with open(cam_path, "r") as f:
        cam_data = json.load(f)

    frames = cam_data["frames"]
    num_views = len(frames)

    images = []
    c2ws = []
    fxfycxcy_list = []

    bg_color = get_bg_color("white")

    for i, frame_info in enumerate(frames):
        img_path = sample_dir / "images" / f"cam_{i:03d}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        if img.shape[-1] == 4:
            img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Auto-generate mask from white background
            gray = np.mean(img_rgb, axis=-1)
            alpha = (gray < 250).astype(np.float32)[..., None]

        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_rgba = img_rgb * alpha + bg_color.numpy() * (1 - alpha)
        images.append(img_rgba)

        w2c = np.array(frame_info["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)

        fx, fy = frame_info["fx"], frame_info["fy"]
        cx, cy = frame_info["cx"], frame_info["cy"]
        fxfycxcy_list.append([fx, fy, cx, cy])

    images = np.stack(images, axis=0)
    c2ws = np.array(c2ws)
    fxfycxcy = np.array(fxfycxcy_list)

    # Normalize cameras
    c2ws = normalize_cameras_to_z_up(c2ws, up_direction=None)
    if target_camera_distance > 0:
        c2ws, fxfycxcy = normalize_camera_distance_with_intrinsics(
            c2ws, fxfycxcy, target_camera_distance
        )

    images = torch.from_numpy(images).float().to(device)
    images = rearrange(images, "v h w c -> v c h w")
    c2ws = torch.from_numpy(c2ws).float().to(device)
    fxfycxcy = torch.from_numpy(fxfycxcy).float().to(device)

    # Create index tensor
    index = torch.zeros(num_views, 3, dtype=torch.long, device=device)

    return edict({
        "image": images.unsqueeze(0),
        "c2w": c2ws.unsqueeze(0),
        "fxfycxcy": fxfycxcy.unsqueeze(0),
        "index": index.unsqueeze(0),
    })


@torch.no_grad()
def infer_gaussians(model, batch_data: edict):
    """Run inference and return Gaussians."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch_data, create_visual=False, split_data=False)
    return output.gaussians[0]


def render_standard_turntable(
    gaussians: GaussianModel,
    resolution: int = 384,
    num_views: int = 60,
    elevation: float = 20,
    radius: float = 2.7,
) -> np.ndarray:
    """Render standard 360° turntable (no filtering, no center adjustment)."""
    turntable_strip = render_turntable(
        gaussians,
        rendering_resolution=resolution,
        num_views=num_views,
        elevation=elevation,
        radius=radius,
        trajectory_mode="turntable"
    )
    # turntable_strip: [H, V*W, C]
    h = turntable_strip.shape[0]
    w_per_view = turntable_strip.shape[1] // num_views
    frames = turntable_strip.reshape(h, num_views, w_per_view, 3)
    frames = rearrange(frames, "h v w c -> v h w c")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Temporal Turntable v4")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=60, help="Views per turntable")
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=2.7)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="outputs/temporal_turntable")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model(args.checkpoint, args.config, args.device)

    data_dir = Path(args.data_dir)
    frame_indices = list(range(args.start_frame, args.end_frame, args.frame_step))
    
    print(f"Processing {len(frame_indices)} frames: {frame_indices[0]} to {frame_indices[-1]}")

    all_turntables = []
    
    for i, frame_idx in enumerate(tqdm(frame_indices, desc="Processing frames")):
        sample_dir = data_dir / f"{frame_idx:06d}"
        
        if not sample_dir.exists():
            print(f"Warning: {sample_dir} not found, skipping")
            continue
        
        # Load sample
        sample = load_single_sample(
            sample_dir, 
            device=args.device,
            target_camera_distance=args.radius
        )
        
        # Infer Gaussians
        gaussians = infer_gaussians(model, sample)
        
        # Render standard 360° turntable
        turntable_frames = render_standard_turntable(
            gaussians,
            resolution=args.resolution,
            num_views=args.num_views,
            elevation=args.elevation,
            radius=args.radius
        )
        
        all_turntables.append(turntable_frames)
        
        # Save individual turntable video for first frame
        if i == 0:
            imageseq2video(
                turntable_frames, 
                str(output_dir / "turntable_frame0.mp4"), 
                fps=args.fps
            )
            print(f"Saved: turntable_frame0.mp4")
        
        del sample, gaussians
        gc.collect()
        torch.cuda.empty_cache()

    if not all_turntables:
        print("No frames processed!")
        return

    all_turntables = np.stack(all_turntables)  # [T, V, H, W, C]
    T, V, H, W, C = all_turntables.shape
    print(f"Collected: {T} frames x {V} views")

    # === Save different video modes ===
    
    # 1. Single angle over time (fixed viewpoint, time changes)
    fixed_view = 0
    single_angle_frames = all_turntables[:, fixed_view]  # [T, H, W, C]
    imageseq2video(single_angle_frames, str(output_dir / "temporal_single_angle.mp4"), fps=args.fps)
    print(f"Saved: temporal_single_angle.mp4 (fixed view {fixed_view})")

    # 2. Rotating over time (view rotates as time progresses)
    rotating_frames = []
    for t in range(T):
        view_idx = int(t * V / T) % V
        rotating_frames.append(all_turntables[t, view_idx])
    rotating_frames = np.stack(rotating_frames)
    imageseq2video(rotating_frames, str(output_dir / "temporal_rotating.mp4"), fps=args.fps)
    print(f"Saved: temporal_rotating.mp4")

    # 3. Full turntable for each time (all views, all times)
    full_frames = all_turntables.reshape(T * V, H, W, C)
    imageseq2video(full_frames, str(output_dir / "temporal_full.mp4"), fps=args.fps)
    print(f"Saved: temporal_full.mp4 ({T}x{V}={T*V} frames)")

    # 4. Grid image (first frame turntable as grid)
    grid_rows, grid_cols = 6, 10
    if V == grid_rows * grid_cols:
        first_turntable = all_turntables[0]  # [V, H, W, C]
        grid = rearrange(first_turntable, "(r c) h w ch -> (r h) (c w) ch", r=grid_rows, c=grid_cols)
        from PIL import Image
        Image.fromarray(grid).save(str(output_dir / "turntable_grid.jpg"))
        print(f"Saved: turntable_grid.jpg ({grid_rows}x{grid_cols})")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
