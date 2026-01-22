#!/usr/bin/env python3
"""
Alpha Threshold Comparison Visualization

기존 체크포인트에서 렌더링된 alpha에 대해 threshold만 달리하여
foreground mask 시각화를 비교합니다.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
        --checkpoint checkpoints/_temp/D7_1_E1_1_paper_random/ckpt_0000000000002300.pt \
        --config configs/mouse/D7_1_E1_1_paper_random.yaml \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
        --output_dir alpha_threshold_comparison \
        --thresholds 0.3 0.5 0.7 0.9

Output:
    threshold_comparison.png - 각 threshold별 mask overlay 비교
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict as edict
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from omegaconf import OmegaConf


def load_model_and_config(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load trained GS-LRM model from checkpoint."""
    from gslrm.model.gslrm import GSLRM

    config = OmegaConf.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GSLRM(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    step = checkpoint.get("step", "unknown")
    return model, config, step


def load_sample(data_dir: Path, sample_idx: int, device: str = "cuda"):
    """Load sample with images and camera parameters."""
    train_dir = data_dir / "train"
    if not train_dir.exists():
        train_dir = data_dir

    sample_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    sample_dir = sample_dirs[sample_idx]

    img_dir = sample_dir / "images"
    camera_file = sample_dir / "opencv_cameras.json"

    img_files = sorted(img_dir.glob("*.png"))
    images, gt_masks = [], []

    for img_file in img_files:
        img = Image.open(img_file)
        img_np = np.array(img).astype(np.float32) / 255.0

        if img_np.shape[2] == 4:
            rgb, mask = img_np[:, :, :3], img_np[:, :, 3:4]
        else:
            rgb = img_np
            mask = np.ones((img_np.shape[0], img_np.shape[1], 1), dtype=np.float32)

        images.append(torch.from_numpy(rgb).permute(2, 0, 1))
        gt_masks.append(torch.from_numpy(mask).permute(2, 0, 1))

    images = torch.stack(images).unsqueeze(0).to(device)
    gt_masks = torch.stack(gt_masks).unsqueeze(0).to(device)

    with open(camera_file, 'r') as f:
        cameras = json.load(f)

    fxfycxcy, c2ws = [], []
    if "frames" in cameras:
        for cam in cameras["frames"]:
            fxfycxcy.append([cam["fx"], cam["fy"], cam["cx"], cam["cy"]])
            c2ws.append(np.linalg.inv(np.array(cam["w2c"])))
    else:
        for cam_key in sorted(cameras.keys()):
            cam = cameras[cam_key]
            if "K" in cam:
                K = np.array(cam["K"])
                fxfycxcy.append([K[0,0], K[1,1], K[0,2], K[1,2]])
            else:
                fxfycxcy.append([cam["fx"], cam["fy"], cam["cx"], cam["cy"]])
            if "w2c" in cam:
                c2ws.append(np.linalg.inv(np.array(cam["w2c"])))
            else:
                w2c = np.eye(4)
                w2c[:3, :3] = np.array(cam["R"])
                w2c[:3, 3] = np.array(cam["t"]).flatten()
                c2ws.append(np.linalg.inv(w2c))

    fxfycxcy = torch.tensor(fxfycxcy, dtype=torch.float32).unsqueeze(0).to(device)
    c2ws = torch.tensor(np.stack(c2ws), dtype=torch.float32).unsqueeze(0).to(device)

    num_views = images.shape[1]
    index = torch.stack([
        torch.arange(num_views).long(),
        torch.zeros(num_views).long(),
    ], dim=-1).unsqueeze(0).to(device)

    return {
        "images": images, "gt_masks": gt_masks,
        "fxfycxcy": fxfycxcy, "c2ws": c2ws, "index": index,
    }


@torch.no_grad()
def run_inference(model, data):
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


def create_threshold_comparison(
    gt_rgb: torch.Tensor,       # [V, 3, H, W]
    gt_mask: torch.Tensor,      # [V, 1, H, W]
    rendered_rgb: torch.Tensor, # [V, 3, H, W]
    rendered_alpha: torch.Tensor,  # [V, 1, H, W]
    thresholds: list,
) -> np.ndarray:
    """
    Create threshold comparison visualization.

    Layout:
        Row 0: GT RGB (6 views)
        Row 1: Rendered RGB (6 views)
        Row 2: GT + GT Mask (green)
        Row 3+: Rendered + Alpha Mask at each threshold
    """
    V, _, H, W = gt_rgb.shape
    device = gt_rgb.device

    def overlay_mask(rgb, mask, color):
        mask_3ch = mask.expand(-1, 3, -1, -1)
        overlay = rgb.clone()
        overlay = overlay * (1 - 0.5 * mask_3ch) + color.view(1, 3, 1, 1).to(device) * 0.5 * mask_3ch
        return overlay

    rows = []
    labels = []

    # Row 0: GT RGB
    # Apply alpha to remove background
    gt_rgb_masked = gt_rgb * gt_mask.expand(-1, 3, -1, -1)
    rows.append(gt_rgb_masked)
    labels.append("GT RGB")

    # Row 1: Rendered RGB
    rows.append(rendered_rgb)
    labels.append("Rendered RGB")

    # Row 2: GT + GT Mask (green)
    green = torch.tensor([0.0, 1.0, 0.0])
    gt_mask_binary = (gt_mask > 0.5).float()
    rows.append(overlay_mask(gt_rgb, gt_mask_binary, green))
    labels.append("GT + Mask (green)")

    # Rows 3+: Rendered + Alpha at each threshold
    colors = [
        torch.tensor([0.0, 0.5, 1.0]),   # blue
        torch.tensor([1.0, 0.5, 0.0]),   # orange
        torch.tensor([1.0, 0.0, 1.0]),   # magenta
        torch.tensor([0.0, 1.0, 1.0]),   # cyan
    ]

    for i, thresh in enumerate(thresholds):
        color = colors[i % len(colors)]
        alpha_mask = (rendered_alpha > thresh).float()
        coverage = alpha_mask.mean().item() * 100
        rows.append(overlay_mask(rendered_rgb, alpha_mask, color))
        labels.append(f"Alpha > {thresh} ({coverage:.1f}%)")

    # Stack all rows
    visual = torch.stack(rows, dim=0)  # [num_rows, V, 3, H, W]
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")
    visual_np = (visual.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Add labels
    img = Image.fromarray(visual_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        y_pos = i * H + 5
        # Draw text with background
        draw.rectangle([5, y_pos, 250, y_pos + 25], fill=(0, 0, 0, 180))
        draw.text((10, y_pos + 2), label, fill=(255, 255, 255), font=font)

    return np.array(img), labels


def main():
    parser = argparse.ArgumentParser(description="Alpha Threshold Comparison")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="alpha_threshold_comparison")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Alpha Threshold Comparison")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds: {args.thresholds}")

    # Load model
    model, config, step = load_model_and_config(args.checkpoint, args.config, args.device)
    print(f"Model loaded (step: {step})")

    # Load data
    data = load_sample(Path(args.data_dir), args.sample_idx, args.device)
    print(f"Sample loaded: {args.sample_idx}")

    # Run inference
    print("Running inference...")
    result = run_inference(model, data)

    rendered_rgb = result.render[0]
    rendered_alpha = result.rendered_alpha[0] if result.rendered_alpha is not None else torch.ones_like(data["gt_masks"][0])

    target = result.target.image
    if target.shape[2] == 4:
        gt_rgb = target[0, :, :3]
        gt_mask = target[0, :, 3:4]
    else:
        gt_rgb = target[0]
        gt_mask = data["gt_masks"][0]

    # Print alpha statistics
    print("\n--- Rendered Alpha Statistics ---")
    alpha_flat = rendered_alpha.flatten()
    print(f"  Min: {alpha_flat.min().item():.4f}")
    print(f"  Max: {alpha_flat.max().item():.4f}")
    print(f"  Mean: {alpha_flat.mean().item():.4f}")
    for thresh in args.thresholds:
        pct = (alpha_flat > thresh).float().mean().item() * 100
        print(f"  % > {thresh}: {pct:.1f}%")

    print("\n--- GT Mask Statistics ---")
    gt_flat = gt_mask.flatten()
    print(f"  Mean: {gt_flat.mean().item():.4f}")
    print(f"  % > 0.5: {(gt_flat > 0.5).float().mean().item() * 100:.1f}%")

    # Create visualization
    print("\nCreating visualization...")
    visual, labels = create_threshold_comparison(
        gt_rgb, gt_mask, rendered_rgb, rendered_alpha, args.thresholds
    )

    save_path = output_dir / "threshold_comparison.png"
    Image.fromarray(visual).save(save_path)
    print(f"Saved: {save_path}")

    # Save report
    report = f"""# Alpha Threshold Comparison

**Checkpoint**: `{args.checkpoint}`
**Step**: {step}
**Thresholds**: {args.thresholds}

## Alpha Statistics

| Metric | Value |
|--------|-------|
| Min | {alpha_flat.min().item():.4f} |
| Max | {alpha_flat.max().item():.4f} |
| Mean | {alpha_flat.mean().item():.4f} |
"""
    for thresh in args.thresholds:
        pct = (alpha_flat > thresh).float().mean().item() * 100
        report += f"| % > {thresh} | {pct:.1f}% |\n"

    report += f"""
## GT Mask Statistics

| Metric | Value |
|--------|-------|
| Mean | {gt_flat.mean().item():.4f} |
| % > 0.5 | {(gt_flat > 0.5).float().mean().item() * 100:.1f}% |

## Visualization

![Threshold Comparison](threshold_comparison.png)

**Rows:**
"""
    for i, label in enumerate(labels):
        report += f"- Row {i}: {label}\n"

    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    print(f"\nReport: {output_dir / 'report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
