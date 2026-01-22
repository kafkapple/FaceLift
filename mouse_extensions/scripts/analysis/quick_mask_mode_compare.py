#!/usr/bin/env python3
"""
Quick Mask Mode Comparison

Runs minimal training with different mask modes and generates
WandB-style visualization for comparison.

Usage:
    # Compare all mask modes (100 steps each)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.quick_mask_mode_compare \
        --config configs/mouse/D7_1_E1_1_paper_random.yaml \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
        --output_dir mask_mode_comparison \
        --steps 100

    # Test specific mask modes
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.quick_mask_mode_compare \
        --config configs/mouse/D7_1_E1_1_paper_random.yaml \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
        --output_dir mask_mode_comparison \
        --mask_modes none gt alpha \
        --steps 50

Output:
    {output_dir}/
    ├── mode_none/
    │   ├── ckpt_final.pt
    │   └── figures/gt_vs_pred_wandb.png
    ├── mode_gt/
    │   ├── ckpt_final.pt
    │   └── figures/gt_vs_pred_wandb.png
    ├── mode_alpha/
    │   └── ...
    └── comparison_grid.png  # Side-by-side comparison
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json
import copy

import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from omegaconf import OmegaConf


# Available mask modes
MASK_MODES = ["none", "gt", "alpha", "rgb_pred"]


def load_config(config_path: str):
    """Load and return config."""
    return OmegaConf.load(config_path)


def modify_config_for_quick_training(config, mask_mode: str, steps: int, output_dir: Path):
    """Modify config for quick training with specified mask mode."""
    cfg = OmegaConf.to_container(config, resolve=True)
    cfg = edict(cfg)

    # Set mask mode
    if not hasattr(cfg, 'training'):
        cfg.training = edict({})
    if not hasattr(cfg.training, 'losses'):
        cfg.training.losses = edict({})

    cfg.training.losses.mask_mode = mask_mode

    # Quick training settings
    cfg.training.max_steps = steps
    cfg.training.val_every = steps  # Validate only at end
    cfg.training.save_every = steps
    cfg.training.log_every = 10
    cfg.training.vis_every = steps

    # Disable wandb for quick experiments
    if hasattr(cfg, 'wandb'):
        cfg.wandb.enabled = False

    # Set output directory
    cfg.output_dir = str(output_dir)

    return cfg


def load_sample(data_dir: Path, sample_idx: int, device: str = "cuda"):
    """Load sample with images and camera parameters."""
    train_dir = data_dir / "train"
    if not train_dir.exists():
        train_dir = data_dir

    sample_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if sample_idx >= len(sample_dirs):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(sample_dirs)-1})")

    sample_dir = sample_dirs[sample_idx]
    img_dir = sample_dir / "images"
    camera_file = sample_dir / "opencv_cameras.json"

    # Load images (RGBA)
    img_files = sorted(img_dir.glob("*.png"))
    images = []
    gt_masks = []

    for img_file in img_files:
        img = Image.open(img_file)
        img_np = np.array(img).astype(np.float32) / 255.0

        if img_np.shape[2] == 4:
            rgb = img_np[:, :, :3]
            mask = img_np[:, :, 3:4]
        else:
            rgb = img_np
            mask = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.float32)

        images.append(torch.from_numpy(rgb).permute(2, 0, 1))
        gt_masks.append(torch.from_numpy(mask).permute(2, 0, 1))

    images = torch.stack(images).unsqueeze(0).to(device)
    gt_masks = torch.stack(gt_masks).unsqueeze(0).to(device)

    # Load camera parameters
    with open(camera_file, 'r') as f:
        cameras = json.load(f)

    fxfycxcy = []
    c2ws = []

    if "frames" in cameras:
        frames = cameras["frames"]
        for cam in frames:
            fxfycxcy.append([cam["fx"], cam["fy"], cam["cx"], cam["cy"]])
            w2c = np.array(cam["w2c"])
            c2w = np.linalg.inv(w2c)
            c2ws.append(c2w)
    else:
        for cam_key in sorted(cameras.keys()):
            cam = cameras[cam_key]
            if "K" in cam:
                K = np.array(cam["K"])
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            else:
                fx, fy = cam["fx"], cam["fy"]
                cx, cy = cam["cx"], cam["cy"]
            fxfycxcy.append([fx, fy, cx, cy])

            if "w2c" in cam:
                w2c = np.array(cam["w2c"])
                c2w = np.linalg.inv(w2c)
            else:
                R = np.array(cam["R"])
                t = np.array(cam["t"]).flatten()
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = t
                c2w = np.linalg.inv(w2c)
            c2ws.append(c2w)

    fxfycxcy = torch.tensor(fxfycxcy, dtype=torch.float32).unsqueeze(0).to(device)
    c2ws = torch.tensor(np.stack(c2ws), dtype=torch.float32).unsqueeze(0).to(device)

    num_views = images.shape[1]
    index = torch.stack([
        torch.arange(num_views).long(),
        torch.zeros(num_views).long(),
    ], dim=-1).unsqueeze(0).to(device)

    return {
        "images": images,
        "gt_masks": gt_masks,
        "fxfycxcy": fxfycxcy,
        "c2ws": c2ws,
        "index": index,
        "sample_dir": str(sample_dir),
    }


def run_quick_training(config, mask_mode: str, steps: int, output_dir: Path, device: str = "cuda"):
    """Run quick training with specified mask mode."""
    from gslrm.model.gslrm import GSLRM
    from torch.optim import AdamW

    mode_dir = output_dir / f"mode_{mask_mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    # Modify config
    cfg = modify_config_for_quick_training(config, mask_mode, steps, mode_dir)

    print(f"\n{'='*60}")
    print(f"Quick Training: mask_mode={mask_mode}, steps={steps}")
    print(f"{'='*60}")

    # Build model
    print("Building model...")
    model = GSLRM(cfg)
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.training.get('lr', 1e-4))

    # Data loader - simple version using first sample repeatedly
    data_dir = Path(cfg.dataset.data_dir)
    sample = load_sample(data_dir, 0, device)

    # Training loop
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        batch = edict({
            "image": sample["images"],
            "c2w": sample["c2ws"],
            "fxfycxcy": sample["fxfycxcy"],
            "index": sample["index"],
        })

        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
            result = model.forward(batch, create_visual=False, split_data=True)

            # Simple L2 loss
            rendered = result.render
            target = result.target.image

            if target.shape[2] == 4:
                target_rgb = target[:, :, :3]
                gt_mask = target[:, :, 3:4]
            else:
                target_rgb = target
                gt_mask = sample["gt_masks"]

            # Apply mask based on mode
            if mask_mode == "none":
                loss = ((rendered - target_rgb) ** 2).mean()
            elif mask_mode == "gt":
                mask = gt_mask
                loss = ((rendered - target_rgb) ** 2 * mask).sum() / (mask.sum() + 1e-6)
            elif mask_mode == "alpha":
                alpha = result.rendered_alpha if result.rendered_alpha is not None else torch.ones_like(gt_mask)
                mask = (alpha > 0.5).float()
                loss = ((rendered - target_rgb) ** 2 * mask).sum() / (mask.sum() + 1e-6)
            else:  # rgb_pred
                loss = ((rendered - target_rgb) ** 2).mean()

        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"  Step {step}/{steps}: loss={loss.item():.4f}")

    # Save checkpoint
    ckpt_path = mode_dir / "ckpt_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": steps,
        "mask_mode": mask_mode,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    return model, mode_dir


@torch.no_grad()
def generate_visualization(model, data, output_dir: Path, mask_mode: str, threshold: float = 0.5):
    """Generate WandB-style visualization."""
    from einops import rearrange

    model.eval()

    input_batch = edict({
        "image": data["images"],
        "c2w": data["c2ws"],
        "fxfycxcy": data["fxfycxcy"],
        "index": data["index"],
    })

    with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
        result = model.forward(input_batch, create_visual=True, split_data=True)

    rendered_rgb = result.render[0]  # [V, 3, H, W]
    rendered_alpha = result.rendered_alpha[0] if result.rendered_alpha is not None else torch.ones_like(data["gt_masks"][0])

    target = result.target.image
    if target.shape[2] == 4:
        gt_rgb = target[0, :, :3]
        gt_mask = target[0, :, 3:4]
    else:
        gt_rgb = target[0]
        gt_mask = data["gt_masks"][0]

    # Create visualization
    V, _, H, W = gt_rgb.shape

    def overlay_mask(rgb, mask, color):
        mask_3ch = mask.expand(-1, 3, -1, -1)
        overlay = rgb.clone()
        overlay = overlay * (1 - 0.4 * mask_3ch) + color.view(1, 3, 1, 1).to(rgb.device) * 0.4 * mask_3ch
        return overlay

    green = torch.tensor([0.0, 1.0, 0.0])
    blue = torch.tensor([0.0, 0.5, 1.0])

    row1 = gt_rgb  # GT RGB
    row2 = rendered_rgb  # Rendered RGB
    row3 = overlay_mask(gt_rgb, (gt_mask > 0.5).float(), green)  # GT + mask
    row4 = overlay_mask(rendered_rgb, (rendered_alpha > threshold).float(), blue)  # Rendered + alpha

    error = (gt_rgb - rendered_rgb).abs().mean(dim=1, keepdim=True)
    error_normalized = error / (error.max() + 1e-8)
    row5 = torch.zeros_like(gt_rgb)
    row5[:, 0:1] = error_normalized

    visual = torch.stack([row1, row2, row3, row4, row5], dim=0)
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")

    visual_np = (visual.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Add label
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(visual_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), f"mask_mode: {mask_mode}", fill=(255, 255, 0), font=font)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    save_path = fig_dir / "gt_vs_pred_wandb.png"
    img.save(save_path)
    print(f"Saved: {save_path}")

    return visual_np


def create_comparison_grid(output_dir: Path, mask_modes: List[str]):
    """Create side-by-side comparison of all mask modes."""
    images = []

    for mode in mask_modes:
        mode_path = output_dir / f"mode_{mode}" / "figures" / "gt_vs_pred_wandb.png"
        if mode_path.exists():
            img = Image.open(mode_path)
            images.append(np.array(img))

    if not images:
        print("No images found for comparison")
        return

    # Stack horizontally
    comparison = np.concatenate(images, axis=1)

    save_path = output_dir / "comparison_grid.png"
    Image.fromarray(comparison).save(save_path)
    print(f"\nComparison grid saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Quick Mask Mode Comparison")
    parser.add_argument("--config", type=str, required=True, help="Base config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--output_dir", type=str, default="mask_mode_comparison", help="Output directory")
    parser.add_argument("--mask_modes", nargs="+", default=["none", "gt", "alpha"], help="Mask modes to compare")
    parser.add_argument("--steps", type=int, default=100, help="Training steps per mode")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index for visualization")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--threshold", type=float, default=0.5, help="Alpha threshold")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"Quick Mask Mode Comparison")
    print(f"{'='*60}")
    print(f"Time: {timestamp}")
    print(f"Config: {args.config}")
    print(f"Mask Modes: {args.mask_modes}")
    print(f"Steps per mode: {args.steps}")
    print(f"{'='*60}")

    # Load base config
    config = load_config(args.config)

    # Override data_dir
    config.dataset.data_dir = args.data_dir

    # Load sample for visualization
    data = load_sample(Path(args.data_dir), args.sample_idx, args.device)

    # Run for each mask mode
    for mask_mode in args.mask_modes:
        if mask_mode not in MASK_MODES:
            print(f"Warning: Unknown mask mode '{mask_mode}', skipping")
            continue

        model, mode_dir = run_quick_training(
            config, mask_mode, args.steps, output_dir, args.device
        )

        generate_visualization(model, data, mode_dir, mask_mode, args.threshold)

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Create comparison grid
    create_comparison_grid(output_dir, args.mask_modes)

    print(f"\n{'='*60}")
    print("Comparison Complete!")
    print(f"{'='*60}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
