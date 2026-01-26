#!/usr/bin/env python3
"""
Standalone rendering script for GS-LRM models.

Renders turntable videos and multi-view grids from trained checkpoints.

Usage Examples:
    # Turntable video at default elevation (20°)
    python render_from_checkpoint.py \
        --checkpoint checkpoints/gslrm/mouse_v63_v12_random/ckpt_best.pt \
        --config configs/mouse/gslrm_v63_v12_random.yaml \
        --data_path /home/joon/data/preprocessed/FaceLift_mouse/data_mouse_v12_centered/data_mouse_val.txt \
        --output_dir outputs/v63_renders \
        --mode turntable

    # Turntable at multiple elevations
    python render_from_checkpoint.py \
        --checkpoint checkpoints/gslrm/mouse_v63_v12_random/ckpt_best.pt \
        --config configs/mouse/gslrm_v63_v12_random.yaml \
        --data_path /path/to/val.txt \
        --output_dir outputs/v63_renders \
        --elevations 10 15 20 25 30 \
        --mode turntable

    # Multi-view grid at various elevations
    python render_from_checkpoint.py \
        --checkpoint checkpoints/gslrm/mouse_v63_v12_random/ckpt_best.pt \
        --config configs/mouse/gslrm_v63_v12_random.yaml \
        --data_path /path/to/val.txt \
        --output_dir outputs/v63_renders \
        --mode multiview \
        --elevations 10 20 30
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    from gslrm.model.gslrm import GSLRM
    
    config = OmegaConf.load(config_path)
    model = GSLRM(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    return model, config


def save_video(frames: list, output_path: str, fps: int = 30):
    """Save frames as mp4 video using temporary PNG files + ffmpeg."""
    import subprocess
    import tempfile
    import os
    from PIL import Image
    
    frames_np = []
    for f in frames:
        if isinstance(f, torch.Tensor):
            f = f.permute(1, 2, 0).cpu().numpy()
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)
        frames_np.append(f)
    
    if len(frames_np) == 0:
        return
    
    # Save frames to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames_np):
            img = Image.fromarray(frame)
            img.save(os.path.join(tmpdir, f"frame_{i:05d}.png"))
        
        # Use ffmpeg to encode video with H.264 (Mac compatible)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")


def render_turntable_video(
    model,
    sample_batch: dict,
    output_path: str,
    elevation: float = 20.0,
    radius: float = 2.7,
    num_views: int = 150,
    fps: int = 30,
    device: str = "cuda",
):
    """Render turntable video for a single sample."""
    from gslrm.model.gaussians_renderer import render_turntable
    from easydict import EasyDict as edict
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Move all tensors to device
        batch_data = edict()
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(device)
            else:
                batch_data[k] = v
        
        # Forward pass to get Gaussians
        results = model(batch_data, create_visual=False, split_data=True)
        
        gaussians = results.gaussians[0]
        render_res = batch_data.image.shape[-1]
        
        # Render turntable - returns concatenated image (H, V*W, 3)
        concat_img = render_turntable(
            gaussians,
            rendering_resolution=render_res,
            num_views=num_views,
            elevation=elevation,
            radius=radius,
        )
        
        # Split into individual frames
        # concat_img shape: (H, V*W, 3) where V=num_views, H=W=render_res
        h = render_res
        w = render_res
        frames = []
        for i in range(num_views):
            frame = concat_img[:, i*w:(i+1)*w, :]
            frames.append(frame)
        
        # Save video
        save_video(frames, output_path, fps=fps)
        
    return output_path



def render_multiview_grid(
    model,
    sample_batch: dict,
    output_path: str,
    elevations: list = [10, 20, 30],
    num_azimuth: int = 8,
    radius: float = 2.7,
    device: str = "cuda",
):
    """Render multi-view grid at various angles."""
    from gslrm.model.gaussians_renderer import render_turntable
    
    from easydict import EasyDict as edict
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Move all tensors to device
        batch_data = edict()
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(device)
            else:
                batch_data[k] = v
        
        # Forward pass to get Gaussians
        results = model(batch_data, create_visual=False, split_data=True)
        
        gaussians = results.gaussians[0]
        render_res = batch_data.image.shape[-1]
        
        all_rows = []
        
        for elev in elevations:
            # Render at this elevation - returns concatenated image (H, V*W, 3)
            concat_img = render_turntable(
                gaussians,
                rendering_resolution=render_res,
                num_views=num_azimuth,
                elevation=elev,
                radius=radius,
            )
            # concat_img is already (H, V*W, 3) - use directly as row
            all_rows.append(concat_img)
        
        # Concat all rows vertically
        grid_img = np.concatenate(all_rows, axis=0)
        # render_turntable already returns uint8 (0-255), no need to scale
        if grid_img.dtype != np.uint8:
            grid_img = (grid_img * 255).clip(0, 255).astype(np.uint8)
        
        Image.fromarray(grid_img).save(output_path)
        
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render from GS-LRM checkpoint")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to checkpoint")
    parser.add_argument("--config", "-cfg", required=True, help="Path to config YAML")
    parser.add_argument("--data_path", "-d", required=True, help="Path to data list file")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["turntable", "multiview", "both"], default="turntable")
    parser.add_argument("--elevations", nargs="+", type=float, default=[20.0],
                        help="Elevation angles in degrees")
    parser.add_argument("--radius", type=float, default=2.7, help="Camera radius")
    parser.add_argument("--num_views", type=int, default=150, help="Turntable frames")
    parser.add_argument("--num_azimuth", type=int, default=8, help="Azimuth samples for grid")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to render")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.config, args.device)
    
    # Create dataset
    from mouse_extensions.data.mouse_dataset import MouseViewDataset
    
    # Override validation dataset path if provided
    if args.data_path:
        config.validation.dataset_path = args.data_path
    
    # Use fixed view selection for reproducibility
    config.training.dataset.random_view_selection = False
    config.training.dataset.maximize_view_overlap = False
    
    dataset = MouseViewDataset(config, split="val")
    num_samples = min(args.num_samples, len(dataset))
    
    print(f"Rendering {num_samples} samples...")
    print(f"Mode: {args.mode}, Elevations: {args.elevations}")
    
    for idx in range(num_samples):
        sample = dataset[idx]
        sample_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                       for k, v in sample.items()}
        
        sample_name = f"sample_{idx:04d}"
        print(f"\n[{idx+1}/{num_samples}] Processing {sample_name}...")
        
        if args.mode in ["turntable", "both"]:
            for elev in args.elevations:
                output_path = os.path.join(
                    args.output_dir, 
                    f"{sample_name}_turntable_elev{int(elev)}.mp4"
                )
                print(f"  Rendering turntable at elevation {elev}°...")
                render_turntable_video(
                    model, sample_batch, output_path,
                    elevation=elev, radius=args.radius,
                    num_views=args.num_views, fps=args.fps,
                    device=args.device,
                )
                print(f"  Saved: {output_path}")
        
        if args.mode in ["multiview", "both"]:
            output_path = os.path.join(args.output_dir, f"{sample_name}_multiview.png")
            print(f"  Rendering multi-view grid...")
            render_multiview_grid(
                model, sample_batch, output_path,
                elevations=args.elevations,
                num_azimuth=args.num_azimuth,
                radius=args.radius,
                device=args.device,
            )
            print(f"  Saved: {output_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
