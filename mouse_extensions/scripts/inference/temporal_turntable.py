#!/usr/bin/env python3
"""
Temporal Turntable Video Generation v3

Fixes:
- Actually filters Gaussians (removes background completely)
- Centers camera on high-opacity Gaussians (mouse center)
- Multiple filtering criteria: opacity + color
"""

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from easydict import EasyDict as edict
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mouse_extensions.data import (
    normalize_camera_distance_with_intrinsics,
    normalize_cameras_to_z_up,
    get_bg_color,
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
    print(f"Model loaded successfully. Step: {step}")
    return model, config


def load_single_sample(
    sample_dir: Path,
    sample_idx: int = 0,
    device: str = "cuda",
    target_camera_distance: float = 2.7,
    normalize_to_z_up: bool = True,
    auto_generate_mask: bool = True,
    mask_threshold: int = 250,
) -> edict:
    """Load a single sample with proper preprocessing."""
    import cv2

    cam_path = sample_dir / "opencv_cameras.json"
    with open(cam_path, "r") as f:
        cam_data = json.load(f)

    frames = cam_data["frames"]
    num_views = len(frames)

    images = []
    c2ws = []
    fxfycxcy_list = []

    bg_color = get_bg_color("white")
    threshold = mask_threshold / 255.0

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
            alpha = None

        img_np = img_rgb.astype(np.float32) / 255.0

        if auto_generate_mask and alpha is None:
            is_background = np.all(img_np > threshold, axis=2)
            alpha = (~is_background).astype(np.float32)[:, :, np.newaxis]

        if alpha is not None:
            img_np = np.concatenate([img_np, alpha], axis=2)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        images.append(img_tensor)

        w2c = np.array(frame_info["w2c"])
        c2w = np.linalg.inv(w2c)
        c2ws.append(c2w)

        fxfycxcy_list.append(np.array([
            frame_info["fx"], frame_info["fy"],
            frame_info["cx"], frame_info["cy"]
        ]))

    images = torch.stack(images)
    c2ws_np = np.array(c2ws)
    fxfycxcy_np = np.array(fxfycxcy_list)

    if normalize_to_z_up:
        c2ws_np = normalize_cameras_to_z_up(c2ws_np, up_direction=None)

    if target_camera_distance > 0:
        c2ws_np, fxfycxcy_np = normalize_camera_distance_with_intrinsics(
            c2ws_np, fxfycxcy_np, target_camera_distance
        )

    c2ws = torch.from_numpy(c2ws_np).float()
    fxfycxcy = torch.from_numpy(fxfycxcy_np).float()

    images = images.unsqueeze(0).to(device)
    c2ws = c2ws.unsqueeze(0).to(device)
    fxfycxcy = fxfycxcy.unsqueeze(0).to(device)

    camera_indices = torch.arange(num_views).unsqueeze(-1)
    scene_indices = torch.full((num_views, 1), sample_idx)
    index = torch.cat([camera_indices, scene_indices], dim=-1)
    index = index.unsqueeze(0).to(device)

    return edict(
        image=images,
        c2w=c2ws,
        fxfycxcy=fxfycxcy,
        index=index,
        bg_color=torch.tensor(bg_color, device=device).float(),
    )


def get_centered_turntable_cameras(
    center: np.ndarray,
    hfov: float = 50,
    num_views: int = 60,
    w: int = 384,
    h: int = 384,
    radius: float = 2.7,
    elevation: float = 15,
    up_vector: np.ndarray = np.array([0, 0, 1]),
):
    """Generate turntable cameras that look at a specific center point."""
    fx = w / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    fxfycxcy = np.array([fx, fy, cx, cy]).reshape(1, 4).repeat(num_views, axis=0)

    azimuths = np.linspace(0, 360, num_views, endpoint=False)

    c2ws = []
    for azim in azimuths:
        azim_rad = np.deg2rad(azim)
        elev_rad = np.deg2rad(elevation)

        z = radius * np.sin(elev_rad)
        base = radius * np.cos(elev_rad)
        x = base * np.cos(azim_rad)
        y = base * np.sin(azim_rad)

        cam_pos = center + np.array([x, y, z])

        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        R = np.stack((right, -up, forward), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)
        c2ws.append(c2w)

    c2ws = np.stack(c2ws, axis=0)
    return w, h, num_views, fxfycxcy, c2ws


def filter_gaussians(gaussians, opacity_threshold: float = 0.1, color_threshold: float = 0.95):
    """
    Create a filtered copy of gaussians, keeping only foreground (mouse).

    Filters by:
    1. Opacity > threshold
    2. Color not too white (brightness < color_threshold)
    """
    from gslrm.model.gaussians_renderer import GaussianModel

    device = gaussians._xyz.device

    # Get opacity values
    opacity = torch.sigmoid(gaussians._opacity).squeeze()

    # Get color brightness (using SH DC component)
    # _features_dc shape: [N, 1, 3]
    sh_dc = gaussians._features_dc.squeeze(1)  # [N, 3]
    # SH to RGB: color = SH_C0 * sh_dc + 0.5
    SH_C0 = 0.28209479177387814
    colors = SH_C0 * sh_dc + 0.5  # [N, 3]
    brightness = colors.mean(dim=1)  # [N]

    # Combined mask: high opacity AND not too bright (not white background)
    opacity_mask = opacity > opacity_threshold
    color_mask = brightness < color_threshold
    keep_mask = opacity_mask & color_mask

    n_total = gaussians._xyz.shape[0]
    n_keep = keep_mask.sum().item()
    n_opacity = opacity_mask.sum().item()
    n_color = color_mask.sum().item()

    print(f"    Filtering: {n_total} -> {n_keep} "
          f"(opacity>{opacity_threshold}: {n_opacity}, brightness<{color_threshold}: {n_color})")

    # Create new GaussianModel with filtered data
    sh_degree = gaussians.sh_degree if hasattr(gaussians, 'sh_degree') else 0
    scaling_modifier = gaussians.scaling_modifier if hasattr(gaussians, 'scaling_modifier') else None
    filtered = GaussianModel(sh_degree=sh_degree, scaling_modifier=scaling_modifier)

    # Copy filtered attributes
    filtered._xyz = gaussians._xyz[keep_mask].clone()
    filtered._features_dc = gaussians._features_dc[keep_mask].clone()
    if gaussians._features_rest is not None:
        filtered._features_rest = gaussians._features_rest[keep_mask].clone()
    filtered._scaling = gaussians._scaling[keep_mask].clone()
    filtered._rotation = gaussians._rotation[keep_mask].clone()
    filtered._opacity = gaussians._opacity[keep_mask].clone()

    return filtered


def render_filtered_turntable(
    gaussians,
    center: np.ndarray,
    rendering_resolution: int = 384,
    num_views: int = 60,
    elevation: float = 15.0,
    radius: float = 2.7,
    opacity_threshold: float = 0.1,
    color_threshold: float = 0.95,
) -> np.ndarray:
    """Render turntable with filtered Gaussians (background completely removed)."""
    from gslrm.model.gaussians_renderer import render_opencv_cam

    device = gaussians._xyz.device

    # Filter Gaussians (create new model without background)
    filtered_gaussians = filter_gaussians(
        gaussians,
        opacity_threshold=opacity_threshold,
        color_threshold=color_threshold
    )

    w, h, v, fxfycxcy, c2ws = get_centered_turntable_cameras(
        center=center,
        h=rendering_resolution,
        w=rendering_resolution,
        num_views=num_views,
        elevation=elevation,
        radius=radius,
    )

    fxfycxcy = torch.from_numpy(fxfycxcy).float().to(device)
    c2ws = torch.from_numpy(c2ws).float().to(device)

    # Render with WHITE background
    renderings = torch.zeros(v, 3, h, w, dtype=torch.float32, device=device)
    for j in range(v):
        render_result = render_opencv_cam(
            filtered_gaussians, h, w, c2ws[j], fxfycxcy[j],
            bg_color=torch.ones(3, device=device)  # White background
        )
        renderings[j] = render_result["render"]

    torch.cuda.empty_cache()
    renderings = renderings.detach().cpu().numpy()
    renderings = (renderings * 255).clip(0, 255).astype(np.uint8)
    renderings = rearrange(renderings, "v c h w -> v h w c")

    return renderings


@torch.no_grad()
def infer_and_render_turntable(
    model,
    batch_data: edict,
    num_views: int = 60,
    resolution: int = 384,
    elevation: float = 15.0,
    radius: float = 2.7,
    opacity_threshold: float = 0.1,
    color_threshold: float = 0.95,
) -> np.ndarray:
    """Run GS-LRM inference and render centered turntable with filtered Gaussians."""

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model(batch_data, create_visual=False, split_data=False)

    gaussians = output.gaussians[0]

    # Calculate center of high-opacity, non-white Gaussians
    xyz = gaussians._xyz
    opacity = torch.sigmoid(gaussians._opacity).squeeze()

    # Color brightness
    sh_dc = gaussians._features_dc.squeeze(1)
    SH_C0 = 0.28209479177387814
    colors = SH_C0 * sh_dc + 0.5
    brightness = colors.mean(dim=1)

    # Mask for foreground
    fg_mask = (opacity > opacity_threshold) & (brightness < color_threshold)

    if fg_mask.sum() > 0:
        center = xyz[fg_mask].mean(dim=0).cpu().numpy()
    else:
        center = xyz.mean(dim=0).cpu().numpy()

    print(f"  Total Gaussians: {xyz.shape[0]}, Foreground: {fg_mask.sum().item()}, "
          f"Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

    # Render centered turntable with filtered Gaussians
    turntable_frames = render_filtered_turntable(
        gaussians,
        center=center,
        rendering_resolution=resolution,
        num_views=num_views,
        elevation=elevation,
        radius=radius,
        opacity_threshold=opacity_threshold,
        color_threshold=color_threshold,
    )

    return turntable_frames


def assemble_temporal_video(
    all_turntables: List[np.ndarray],
    mode: str,
    fixed_angle: int = 0,
) -> np.ndarray:
    """Assemble turntable renders into video frames."""
    T = len(all_turntables)
    N, H, W, C = all_turntables[0].shape

    all_turntables = np.stack(all_turntables)

    if mode == "temporal_single_angle":
        frames = all_turntables[:, fixed_angle]

    elif mode == "rotating":
        frames = []
        for t in range(T):
            angle_idx = int(t * N / T) % N
            frames.append(all_turntables[t, angle_idx])
        frames = np.stack(frames)

    elif mode == "full_turntable":
        frames = all_turntables.reshape(T * N, H, W, C)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return frames


def process_frame(
    model,
    data_dir: Path,
    frame_idx: int,
    num_views: int,
    resolution: int,
    elevation: float,
    radius: float,
    device: str,
    target_camera_distance: float = 2.7,
    opacity_threshold: float = 0.1,
    color_threshold: float = 0.95,
) -> np.ndarray:
    """Process a single frame."""
    sample_dir = data_dir / f"{frame_idx:06d}"

    sample = load_single_sample(
        sample_dir,
        sample_idx=frame_idx,
        device=device,
        target_camera_distance=target_camera_distance,
        normalize_to_z_up=True,
        auto_generate_mask=True,
        mask_threshold=250,
    )

    turntable = infer_and_render_turntable(
        model, sample,
        num_views=num_views,
        resolution=resolution,
        elevation=elevation,
        radius=radius,
        opacity_threshold=opacity_threshold,
        color_threshold=color_threshold,
    )

    del sample
    gc.collect()
    torch.cuda.empty_cache()

    return turntable


def main():
    parser = argparse.ArgumentParser(description="Generate temporal turntable videos (v3)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--mode", type=str, default="rotating",
                       choices=["temporal_single_angle", "rotating", "full_turntable"])
    parser.add_argument("--num_views", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--elevation", type=float, default=15.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--opacity_threshold", type=float, default=0.5,
                       help="Filter Gaussians below this opacity")
    parser.add_argument("--color_threshold", type=float, default=0.5,
                       help="Filter Gaussians brighter than this (removes white background)")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="outputs/temporal_turntable")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint, args.config, device)

    mouse_config = config.get("mouse", {})
    target_camera_distance = mouse_config.get("target_camera_distance", 2.7)
    print(f"Target camera distance: {target_camera_distance}")
    print(f"Opacity threshold: {args.opacity_threshold}")
    print(f"Color threshold: {args.color_threshold}")

    frame_indices = list(range(args.start_frame, args.end_frame, args.frame_step))
    print(f"Processing {len(frame_indices)} frames: {frame_indices[0]} to {frame_indices[-1]}")

    all_turntables = []
    for frame_idx in tqdm(frame_indices, desc="Processing frames"):
        turntable = process_frame(
            model, data_dir, frame_idx,
            num_views=args.num_views,
            resolution=args.resolution,
            elevation=args.elevation,
            radius=args.radius,
            device=device,
            target_camera_distance=target_camera_distance,
            opacity_threshold=args.opacity_threshold,
            color_threshold=args.color_threshold,
        )
        all_turntables.append(turntable)

    print(f"Assembling video in mode: {args.mode}")
    video_frames = assemble_temporal_video(all_turntables, args.mode)

    from mouse_extensions.utils.video_utils import encode_video_imageio

    ckpt_name = Path(args.checkpoint).stem
    exp_name = Path(args.checkpoint).parent.name
    output_name = f"{exp_name}_{ckpt_name}_{args.mode}_f{args.start_frame}-{args.end_frame}_filtered"
    output_path = output_dir / f"{output_name}.mp4"

    encode_video_imageio(video_frames, output_path, fps=args.fps)

    metadata = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "data_dir": args.data_dir,
        "frames": frame_indices,
        "mode": args.mode,
        "num_views": args.num_views,
        "resolution": args.resolution,
        "elevation": args.elevation,
        "radius": args.radius,
        "opacity_threshold": args.opacity_threshold,
        "color_threshold": args.color_threshold,
        "fps": args.fps,
        "filtered": True,
    }
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
