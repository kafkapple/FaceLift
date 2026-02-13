#!/usr/bin/env python3
"""
Standalone rendering script for GS-LRM models.

Renders turntable videos and multi-view grids from trained checkpoints.
Uses TurntableRenderer for consistent output across all paths (train/val/inference).

Usage Examples:
    # Turntable video at default elevation (20°)
    python render_from_checkpoint.py \
        --checkpoint checkpoints/gslrm/mouse_v63_v12_random/ckpt_best.pt \
        --config configs/mouse/gslrm_v63_v12_random.yaml \
        --data_path ~/data/preprocessed/FaceLift_mouse/data_mouse_v12_centered/data_mouse_val.txt \
        --output_dir outputs/v63_renders \
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


def render_multiview_grid(
    model,
    sample_batch: dict,
    output_path: str,
    elevations: list = [10, 20, 30],
    num_azimuth: int = 8,
    radius: float = 2.7,
    device: str = "cuda",
):
    """Render multi-view grid at various elevations.

    This is separate from TurntableRenderer because it renders at multiple
    elevation angles simultaneously (TurntableRenderer handles single-elevation orbit).
    """
    from gslrm.model.gaussians_renderer import render_turntable
    from easydict import EasyDict as edict

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        batch_data = edict()
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(device)
            else:
                batch_data[k] = v

        results = model(batch_data, create_visual=False, split_data=True)

        gaussians = results.gaussians[0]
        render_res = batch_data.image.shape[-1]

        all_rows = []

        for elev in elevations:
            concat_img = render_turntable(
                gaussians,
                rendering_resolution=render_res,
                num_views=num_azimuth,
                elevation=elev,
                radius=radius,
            )
            all_rows.append(concat_img)

        grid_img = np.concatenate(all_rows, axis=0)
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
    parser.add_argument("--radius", type=float, default=None,
                        help="Camera radius (default: from TurntableVideoConfig)")
    parser.add_argument("--num_views", type=int, default=None,
                        help="Turntable orbit frames (default: from TurntableVideoConfig)")
    parser.add_argument("--num_azimuth", type=int, default=8, help="Azimuth samples for grid")
    parser.add_argument("--fps", type=int, default=None,
                        help="Video FPS (default: from TurntableVideoConfig)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to render")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.config, args.device)

    # Build TurntableVideoConfig — start from config file, then CLI overrides
    from mouse_extensions.visualization.turntable_renderer import (
        TurntableRenderer, TurntableVideoConfig,
    )

    tt_cfg = TurntableVideoConfig.from_config(config)

    # CLI overrides
    if args.num_views is not None:
        tt_cfg.orbit_views = args.num_views
    if args.fps is not None:
        tt_cfg.orbit_fps = args.fps
    if args.radius is not None:
        tt_cfg.orbit_radius = args.radius
    if args.elevations and len(args.elevations) == 1:
        tt_cfg.orbit_elevation = int(args.elevations[0])

    # Inference-specific: no dataset cameras, so disable view trajectory
    tt_cfg.save_view_with_input = False
    tt_cfg.save_orbit_with_input = False

    renderer = TurntableRenderer(tt_cfg)

    print(f"TurntableVideoConfig: orbit_views={tt_cfg.orbit_views}, "
          f"orbit_fps={tt_cfg.orbit_fps}, orbit_elevation={tt_cfg.orbit_elevation}, "
          f"orbit_radius={tt_cfg.orbit_radius}")

    # Create dataset
    from mouse_extensions.data.mouse_dataset import MouseViewDataset

    if args.data_path:
        config.validation.dataset_path = args.data_path

    config.training.dataset.random_view_selection = False
    config.training.dataset.maximize_view_overlap = False

    dataset = MouseViewDataset(config, split="val")
    num_samples = min(args.num_samples, len(dataset))

    print(f"Rendering {num_samples} samples...")
    print(f"Mode: {args.mode}, Elevations: {args.elevations}")

    from easydict import EasyDict as edict

    for idx in range(num_samples):
        sample = dataset[idx]
        sample_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                       for k, v in sample.items()}

        sample_name = f"sample_{idx:04d}"
        sample_out = os.path.join(args.output_dir, sample_name)
        print(f"\n[{idx+1}/{num_samples}] Processing {sample_name}...")

        if args.mode in ["turntable", "both"]:
            # Forward pass to get Gaussians
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_data = edict()
                for k, v in sample_batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(args.device)
                    else:
                        batch_data[k] = v

                results = model(batch_data, create_visual=False, split_data=True)
                gaussians = results.gaussians[0]
                render_res = batch_data.image.shape[-1]

            # For multiple elevations, render each with a modified config
            for elev in args.elevations:
                tt_cfg.orbit_elevation = int(elev)
                elev_renderer = TurntableRenderer(tt_cfg)

                elev_suffix = f"_elev{int(elev)}" if len(args.elevations) > 1 else ""
                uid = f"{sample_name}{elev_suffix}"

                print(f"  Rendering turntable at elevation {elev}°...")
                out_results = elev_renderer.render_all(
                    gaussians=gaussians,
                    output_dir=sample_out,
                    uid=uid,
                    rendering_resolution=render_res,
                )
                for name, path in out_results.items():
                    print(f"  Saved {name}: {path}")

        if args.mode in ["multiview", "both"]:
            output_path = os.path.join(sample_out, f"{sample_name}_multiview.png")
            os.makedirs(sample_out, exist_ok=True)
            print(f"  Rendering multi-view grid...")
            render_multiview_grid(
                model, sample_batch, output_path,
                elevations=args.elevations,
                num_azimuth=args.num_azimuth,
                radius=tt_cfg.orbit_radius,
                device=args.device,
            )
            print(f"  Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
