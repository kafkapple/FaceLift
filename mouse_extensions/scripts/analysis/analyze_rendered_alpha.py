#!/usr/bin/env python3
"""
Analyze Rendered Alpha from Checkpoint (WandB Style)

Loads a trained GS-LRM checkpoint, runs inference on sample data,
and creates WandB-style visualization with GT and pred comparison.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
        --checkpoint checkpoints/_temp/D7_1_E1_1_paper_random/ckpt_0000000000002300.pt \
        --config configs/mouse/D7_1_E1_1_paper_random.yaml \
        --data_dir ~/data/preprocessed/FaceLift_mouse/D7_1 \
        --output_dir alpha_analysis/D7_1_E1_1 \
        --sample_idx 0

Output Layout (WandB style):
    Row 1: GT RGB (6 views)
    Row 2: Rendered RGB (6 views)
    Row 3: GT + Mask overlay (green)
    Row 4: Rendered + Alpha Mask overlay (blue)
    Row 5: Error heatmap
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

import numpy as np
import torch
from PIL import Image
from easydict import EasyDict as edict
from einops import rearrange

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from omegaconf import OmegaConf


def load_model_and_config(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load trained GS-LRM model from checkpoint."""
    from gslrm.model.gslrm import GSLRM

    print(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("Building model...")
    model = GSLRM(config)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    step = checkpoint.get("step", "unknown")
    print(f"Model loaded. Step: {step}")

    return model, config, step


def load_sample(data_dir: Path, sample_idx: int, device: str = "cuda"):
    """Load sample with images and camera parameters."""
    train_dir = data_dir / "train"
    if not train_dir.exists():
        train_dir = data_dir

    sample_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if sample_idx >= len(sample_dirs):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(sample_dirs)-1})")

    sample_dir = sample_dirs[sample_idx]
    print(f"Loading sample: {sample_dir}")

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

    images = torch.stack(images).unsqueeze(0).to(device)  # [1, V, 3, H, W]
    gt_masks = torch.stack(gt_masks).unsqueeze(0).to(device)  # [1, V, 1, H, W]

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

    # Create index tensor
    num_views = images.shape[1]
    index = torch.stack([
        torch.arange(num_views).long(),
        torch.zeros(num_views).long(),
    ], dim=-1).unsqueeze(0).to(device)

    print(f"Images: {images.shape}, GT Masks: {gt_masks.shape}, Cameras: {fxfycxcy.shape}")

    return {
        "images": images,
        "gt_masks": gt_masks,
        "fxfycxcy": fxfycxcy,
        "c2ws": c2ws,
        "index": index,
        "sample_dir": str(sample_dir),
    }


@torch.no_grad()
def run_inference(model, data, config):
    """Run GS-LRM inference."""
    input_batch = edict({
        "image": data["images"],
        "c2w": data["c2ws"],
        "fxfycxcy": data["fxfycxcy"],
        "index": data["index"],
    })

    with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
        result = model.forward(input_batch, create_visual=True, split_data=True)

    return result


def create_wandb_style_visual(
    gt_images: torch.Tensor,      # [V, 3, H, W]
    gt_masks: torch.Tensor,       # [V, 1, H, W]
    rendered_images: torch.Tensor, # [V, 3, H, W]
    rendered_alpha: torch.Tensor,  # [V, 1, H, W]
    alpha_threshold: float = 0.5,
) -> np.ndarray:
    """
    Create WandB-style visualization.

    Layout:
        Row 1: GT RGB
        Row 2: Rendered RGB
        Row 3: GT + GT Mask overlay (green)
        Row 4: Rendered + Alpha Mask overlay (blue)
        Row 5: Difference heatmap
    """
    V, _, H, W = gt_images.shape

    # Helper: mask overlay
    def overlay_mask(rgb, mask, color):
        """Overlay mask on RGB with color."""
        mask_3ch = mask.expand(-1, 3, -1, -1)
        overlay = rgb.clone()
        overlay = overlay * (1 - 0.4 * mask_3ch) + color.view(1, 3, 1, 1).to(rgb.device) * 0.4 * mask_3ch
        return overlay

    # Colors
    green = torch.tensor([0.0, 1.0, 0.0])
    blue = torch.tensor([0.0, 0.5, 1.0])

    # Row 1: GT RGB (with alpha applied to remove background)
    gt_mask_3ch = gt_masks.expand(-1, 3, -1, -1)
    row1 = gt_images * gt_mask_3ch

    # Row 2: Rendered RGB
    row2 = rendered_images

    # Row 3: GT + GT Mask
    gt_mask_binary = (gt_masks > 0.5).float()
    row3 = overlay_mask(gt_images, gt_mask_binary, green)

    # Row 4: Rendered + Alpha Mask
    alpha_mask_binary = (rendered_alpha > alpha_threshold).float()
    row4 = overlay_mask(rendered_images, alpha_mask_binary, blue)

    # Row 5: Error heatmap (foreground only)
    gt_mask_binary = (gt_masks > 0.5).float()
    error = (gt_images - rendered_images).abs().mean(dim=1, keepdim=True)
    error = error * gt_mask_binary  # Only foreground error
    error_normalized = error / (error.max() + 1e-8)
    # Apply colormap (red=high error)
    row5 = torch.zeros_like(gt_images)
    row5[:, 0:1, :, :] = error_normalized  # Red channel

    # Stack rows
    visual = torch.stack([row1, row2, row3, row4, row5], dim=0)  # [5, V, 3, H, W]
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")

    visual_np = (visual.detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return visual_np


def compute_alpha_stats(alpha: torch.Tensor) -> Dict[str, float]:
    """Compute alpha statistics."""
    alpha_flat = alpha.flatten()
    return {
        "min": alpha_flat.min().item(),
        "max": alpha_flat.max().item(),
        "mean": alpha_flat.mean().item(),
        "std": alpha_flat.std().item(),
        "pct_above_0.5": (alpha_flat > 0.5).float().mean().item() * 100,
        "pct_above_0.7": (alpha_flat > 0.7).float().mean().item() * 100,
        "pct_above_0.9": (alpha_flat > 0.9).float().mean().item() * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Rendered Alpha (WandB Style)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="alpha_analysis")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print("=" * 60)
    print("Rendered Alpha Analysis (WandB Style)")
    print("=" * 60)

    # Load model
    model, config, step = load_model_and_config(args.checkpoint, args.config, args.device)

    # Load data
    data = load_sample(Path(args.data_dir), args.sample_idx, args.device)

    # Run inference
    print("\nRunning inference...")
    result = run_inference(model, data, config)

    # Get rendered output
    rendered_rgb = result.render  # [B, V, 3, H, W]
    rendered_alpha = result.rendered_alpha  # [B, V, 1, H, W]

    if rendered_rgb is None:
        print("ERROR: No rendered output!")
        return

    print(f"Rendered RGB: {rendered_rgb.shape}")
    print(f"Rendered Alpha: {rendered_alpha.shape if rendered_alpha is not None else 'None'}")

    # Get GT from target data (after split)
    target_rgb = result.target.image  # [B, V, C, H, W]
    print(f"Target shape: {target_rgb.shape}")

    # Extract GT RGB and mask
    if target_rgb.shape[2] == 4:
        gt_rgb = target_rgb[0, :, :3, :, :]
        gt_mask = target_rgb[0, :, 3:4, :, :]
    else:
        gt_rgb = target_rgb[0]
        gt_mask = data["gt_masks"][0]

    rendered_rgb_v = rendered_rgb[0]
    rendered_alpha_v = rendered_alpha[0] if rendered_alpha is not None else torch.ones_like(gt_mask)

    # Alpha statistics
    print("\nRendered Alpha Statistics:")
    stats = compute_alpha_stats(rendered_alpha_v)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    # GT mask statistics
    print("\nGT Mask Statistics:")
    gt_stats = compute_alpha_stats(gt_mask)
    for k, v in gt_stats.items():
        print(f"  {k}: {v:.4f}")

    # Create WandB-style visualization
    print("\nCreating visualization...")
    visual = create_wandb_style_visual(
        gt_rgb, gt_mask, rendered_rgb_v, rendered_alpha_v, args.threshold
    )

    save_path = output_dir / "figures" / "gt_vs_pred_wandb.png"
    Image.fromarray(visual).save(save_path)
    print(f"Saved: {save_path}")

    # Save report
    sample_dir_str = data['sample_dir']
    obs_msg = "Rendered alpha is saturated (mean > 0.9) - opacity regularization needed" if stats['mean'] > 0.9 else "Rendered alpha has reasonable distribution"

    report = f"""# Alpha Analysis Report (WandB Style)

**Generated**: {timestamp}
**Checkpoint**: `{args.checkpoint}`
**Step**: {step}
**Sample**: `{sample_dir_str}`

## Visualization Layout

- Row 1: GT RGB
- Row 2: Rendered RGB
- Row 3: GT + GT Mask overlay (green)
- Row 4: Rendered + Alpha Mask overlay (blue, threshold={args.threshold})
- Row 5: Error heatmap

![WandB Style](figures/gt_vs_pred_wandb.png)

## Rendered Alpha Statistics

| Metric | Value |
|--------|-------|
| Min | {stats['min']:.4f} |
| Max | {stats['max']:.4f} |
| Mean | {stats['mean']:.4f} |
| % > 0.5 | {stats['pct_above_0.5']:.1f}% |

## GT Mask Statistics

| Metric | Value |
|--------|-------|
| Min | {gt_stats['min']:.4f} |
| Max | {gt_stats['max']:.4f} |
| Mean | {gt_stats['mean']:.4f} |
| % > 0.5 | {gt_stats['pct_above_0.5']:.1f}% |

## Observation

{obs_msg}
"""

    with open(output_dir / "alpha_analysis_report.md", "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print("Analysis Complete!")
    print(f"{'=' * 60}")
    print(f"Report: {output_dir / 'alpha_analysis_report.md'}")


if __name__ == "__main__":
    main()
