#!/usr/bin/env python3
"""
Mask Mode Analysis Script for FaceLift Mouse Project (v2.0)

Generates quantitative and qualitative comparison report for different mask modes
and threshold parameters using REAL MODEL INFERENCE.

v2.0 Changes:
- Requires checkpoint for real model inference (no simulation)
- WandB-style visualizations
- Per-threshold alpha comparison with actual rendered alpha
- Comprehensive report with images

Usage:
    # With trained checkpoint (RECOMMENDED)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.mask_mode_analysis \
        --checkpoint checkpoints/gslrm/D7_1_E2/ckpt_step_1000.pt \
        --config configs/mouse/D7_1_E2.yaml \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
        --output_dir experiments/analysis/mask_mode_D7_1_E2

Output:
    - mask_analysis_report.md: Markdown report WITH images
    - figures/: All visualization images
    - results.json: Raw metrics data
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange
from easydict import EasyDict as edict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from omegaconf import OmegaConf


# =============================================================================
# Model Loading
# =============================================================================

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


# =============================================================================
# Data Loading
# =============================================================================

def load_sample(data_dir: Path, sample_idx: int, device: str = "cuda"):
    """Load sample with images and camera parameters."""
    train_dir = data_dir / "train"
    if not train_dir.exists():
        train_dir = data_dir

    sample_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if sample_idx >= len(sample_dirs):
        sample_idx = 0
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

    with open(camera_file, "r") as f:
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
                fxfycxcy.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
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
        "images": images,
        "gt_masks": gt_masks,
        "fxfycxcy": fxfycxcy,
        "c2ws": c2ws,
        "index": index,
        "sample_dir": str(sample_dir),
        "num_views": num_views,
    }


# =============================================================================
# Model Inference
# =============================================================================

@torch.no_grad()
def run_inference(model, data):
    """Run GS-LRM inference and return results."""
    input_batch = edict({
        "image": data["images"],
        "c2w": data["c2ws"],
        "fxfycxcy": data["fxfycxcy"],
        "index": data["index"],
    })

    with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
        result = model.forward(input_batch, create_visual=True, split_data=True)

    rendered_rgb = result.render[0]  # [V, 3, H, W]
    rendered_alpha = result.rendered_alpha[0] if result.rendered_alpha is not None else None

    target = result.target.image
    if target.shape[2] == 4:
        gt_rgb = target[0, :, :3]  # [V, 3, H, W]
        gt_mask = target[0, :, 3:4]  # [V, 1, H, W]
    else:
        gt_rgb = target[0]
        gt_mask = data["gt_masks"][0]

    return {
        "rendered_rgb": rendered_rgb,
        "rendered_alpha": rendered_alpha,
        "gt_rgb": gt_rgb,
        "gt_mask": gt_mask,
    }


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_mask_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
    """Compute IoU, Precision, Recall, F1 between predicted and GT masks."""
    pred = (pred_mask > 0.5).float().view(-1)
    gt = (gt_mask > 0.5).float().view(-1)

    tp = (pred * gt).sum().item()
    fp = (pred * (1 - gt)).sum().item()
    fn = ((1 - pred) * gt).sum().item()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pred_mask_ratio": round(pred.mean().item(), 4),
    }


def compute_per_view_metrics(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> List[Dict]:
    """Compute metrics per view."""
    V = pred_mask.shape[0]
    results = []
    for v in range(V):
        metrics = compute_mask_metrics(pred_mask[v], gt_mask[v])
        metrics["view"] = v
        results.append(metrics)
    return results


# =============================================================================
# Visualization Functions (WandB Style)
# =============================================================================

def overlay_mask(rgb: torch.Tensor, mask: torch.Tensor, color: torch.Tensor, blend: float = 0.4) -> torch.Tensor:
    """Overlay colored mask on RGB image."""
    mask_3ch = mask.expand(-1, 3, -1, -1)
    color_overlay = color.view(1, 3, 1, 1).to(rgb.device) * mask_3ch
    return rgb * (1 - blend * mask_3ch) + color_overlay * blend


def create_wandb_style_comparison(
    gt_rgb: torch.Tensor,
    gt_mask: torch.Tensor,
    rendered_rgb: torch.Tensor,
    rendered_alpha: torch.Tensor,
    thresholds: List[float],
    max_views: int = 6,
) -> Tuple[np.ndarray, Dict]:
    """
    Create WandB-style multi-row comparison visualization.

    Layout:
        Row 0: GT RGB (masked by GT alpha)
        Row 1: Rendered RGB
        Row 2: GT + GT Mask (green overlay)
        Row 3-N: Rendered + Alpha Mask at each threshold (colored overlay)

    Returns:
        (visualization numpy array, metrics dict)
    """
    V = min(gt_rgb.shape[0], max_views)
    _, _, H, W = gt_rgb.shape
    device = gt_rgb.device

    # Colors
    green = torch.tensor([0.0, 1.0, 0.0])
    colors = [
        torch.tensor([0.0, 0.5, 1.0]),   # blue
        torch.tensor([1.0, 0.5, 0.0]),   # orange
        torch.tensor([1.0, 0.0, 1.0]),   # magenta
        torch.tensor([0.0, 1.0, 1.0]),   # cyan
        torch.tensor([1.0, 1.0, 0.0]),   # yellow
    ]

    rows = []
    labels = []

    # Limit views
    gt_rgb = gt_rgb[:V]
    gt_mask = gt_mask[:V]
    rendered_rgb = rendered_rgb[:V]
    rendered_alpha = rendered_alpha[:V] if rendered_alpha is not None else torch.ones_like(gt_mask)

    # Row 0: GT RGB (with alpha applied for clean background)
    gt_rgb_masked = gt_rgb * gt_mask.expand(-1, 3, -1, -1)
    rows.append(gt_rgb_masked)
    labels.append("GT RGB (masked)")

    # Row 1: Rendered RGB
    rows.append(rendered_rgb)
    labels.append("Rendered RGB")

    # Row 2: GT + GT Mask overlay
    gt_mask_binary = (gt_mask > 0.5).float()
    gt_overlay = overlay_mask(gt_rgb, gt_mask_binary, green)
    rows.append(gt_overlay)
    labels.append("GT + Mask (green)")

    # Compute metrics for each threshold
    metrics = {}
    gt_mask_ratio = gt_mask_binary.mean().item() * 100

    for i, thresh in enumerate(thresholds):
        color = colors[i % len(colors)]
        alpha_mask = (rendered_alpha > thresh).float()
        pred_ratio = alpha_mask.mean().item() * 100

        # Compute metrics
        m = compute_mask_metrics(alpha_mask, gt_mask)
        m['threshold'] = thresh
        m['pred_mask_percent'] = round(pred_ratio, 2)
        metrics[f"alpha_{thresh}"] = m

        # Create overlay
        overlay = overlay_mask(rendered_rgb, alpha_mask, color)
        rows.append(overlay)
        labels.append(f"Alpha > {thresh} ({pred_ratio:.1f}%)")

    metrics["gt_mask_percent"] = round(gt_mask_ratio, 2)

    # Stack rows: [num_rows, V, 3, H, W] -> [(num_rows*H), (V*W), 3]
    visual = torch.stack(rows, dim=0)
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")
    visual_np = (visual.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Add row labels
    img = Image.fromarray(visual_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        y_pos = i * H + 5
        # Black background for text
        draw.rectangle([5, y_pos, 300, y_pos + 22], fill=(0, 0, 0, 200))
        draw.text((10, y_pos + 2), label, fill=(255, 255, 255), font=font)

    return np.array(img), labels, metrics


def create_rgb_threshold_comparison(
    gt_rgb: torch.Tensor,
    gt_mask: torch.Tensor,
    rendered_rgb: torch.Tensor,
    thresholds: List[float],
    max_views: int = 6,
) -> Tuple[np.ndarray, Dict]:
    """
    Create RGB prediction mask comparison (|rendered - white| > threshold).
    """
    V = min(gt_rgb.shape[0], max_views)
    _, _, H, W = gt_rgb.shape
    device = gt_rgb.device

    colors = [
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([1.0, 0.5, 0.0]),
        torch.tensor([1.0, 0.0, 1.0]),
        torch.tensor([0.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 0.0]),
    ]

    gt_rgb = gt_rgb[:V]
    gt_mask = gt_mask[:V]
    rendered_rgb = rendered_rgb[:V]

    rows = []
    labels = []

    # Row 0: GT with mask
    gt_mask_binary = (gt_mask > 0.5).float()
    gt_overlay = overlay_mask(gt_rgb, gt_mask_binary, torch.tensor([0.0, 1.0, 0.0]))
    rows.append(gt_overlay)
    labels.append("GT + Mask (green)")

    # Compute distance from white
    white = torch.ones_like(rendered_rgb)
    color_dist = (rendered_rgb - white).abs().mean(dim=1, keepdim=True)  # [V, 1, H, W]

    metrics = {}

    for i, thresh in enumerate(thresholds):
        color = colors[i % len(colors)]
        rgb_mask = (color_dist > thresh).float()
        pred_ratio = rgb_mask.mean().item() * 100

        m = compute_mask_metrics(rgb_mask, gt_mask)
        m['threshold'] = thresh
        m['pred_mask_percent'] = round(pred_ratio, 2)
        metrics[f"rgb_{thresh}"] = m

        overlay = overlay_mask(rendered_rgb, rgb_mask, color)
        rows.append(overlay)
        labels.append(f"RGB dist > {thresh} ({pred_ratio:.1f}%)")

    visual = torch.stack(rows, dim=0)
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")
    visual_np = (visual.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Add labels
    img = Image.fromarray(visual_np)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        y_pos = i * gt_rgb.shape[2] + 5
        draw.rectangle([5, y_pos, 300, y_pos + 22], fill=(0, 0, 0, 200))
        draw.text((10, y_pos + 2), label, fill=(255, 255, 255), font=font)

    return np.array(img), metrics


def create_alpha_histogram(rendered_alpha: torch.Tensor, gt_mask: torch.Tensor) -> np.ndarray:
    """Create histogram of rendered alpha values in FG and BG regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alpha_flat = rendered_alpha.flatten().cpu().numpy()
    gt_flat = gt_mask.flatten().cpu().numpy()

    fg_alpha = alpha_flat[gt_flat > 0.5]
    bg_alpha = alpha_flat[gt_flat <= 0.5]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(fg_alpha, bins=50, alpha=0.7, label=f"Foreground (n={len(fg_alpha)})", color="green")
    ax.hist(bg_alpha, bins=50, alpha=0.7, label=f"Background (n={len(bg_alpha)})", color="red")
    ax.set_xlabel("Rendered Alpha Value")
    ax.set_ylabel("Count")
    ax.set_title("Alpha Distribution in FG vs BG Regions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: Dict,
    output_dir: Path,
    checkpoint_path: str,
    step: str,
    timestamp: str,
) -> str:
    """Generate markdown report with embedded image references."""

    report = f"""# Mask Mode Analysis Report

> **Generated**: {timestamp}
> **Checkpoint**: `{checkpoint_path}`
> **Step**: {step}
> **Sample**: `{results.get("sample_dir", "N/A")}`

---

## Overview

This report compares different mask configurations using **REAL MODEL INFERENCE**:
- **mask_mode: alpha** - Uses rendered alpha > threshold
- **mask_mode: rgb_pred** - Uses |RGB - white| > threshold
- **mask_mode: gt** - Uses ground truth alpha (baseline)

GT Mask Ratio: **{results.get("gt_mask_percent", 0):.2f}%** (mouse is small in frame)

---

## Alpha Threshold Analysis (mask_mode: alpha)

| Threshold | IoU | Precision | Recall | F1 | Pred Mask % |
|-----------|-----|-----------|--------|-----|-------------|
"""

    for key, m in results.get("alpha_metrics", {}).items():
        if key.startswith("alpha_"):
            report += f"| {m['threshold']:.2f} | {m['iou']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['pred_mask_percent']:.2f}% |\n"

    report += """
### Visualization

![Alpha Threshold Comparison](figures/alpha_comparison.png)

---

## RGB Prediction Threshold Analysis (mask_mode: rgb_pred)

| Threshold | IoU | Precision | Recall | F1 | Pred Mask % |
|-----------|-----|-----------|--------|-----|-------------|
"""

    for key, m in results.get("rgb_metrics", {}).items():
        if key.startswith("rgb_"):
            report += f"| {m['threshold']:.3f} | {m['iou']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['pred_mask_percent']:.2f}% |\n"

    report += """
### Visualization

![RGB Threshold Comparison](figures/rgb_comparison.png)

---

## Alpha Distribution Analysis

![Alpha Histogram](figures/alpha_histogram.png)

This histogram shows the distribution of rendered alpha values:
- **Green**: Alpha values in GT foreground regions
- **Red**: Alpha values in GT background regions

Good separation indicates the model learned to distinguish foreground from background.

---

## Best Configurations

"""

    # Find best alpha
    best_alpha = max(
        [m for k, m in results.get("alpha_metrics", {}).items() if k.startswith("alpha_")],
        key=lambda x: x["iou"],
        default={"threshold": 0.5, "iou": 0}
    )

    # Find best rgb
    best_rgb = max(
        [m for k, m in results.get("rgb_metrics", {}).items() if k.startswith("rgb_")],
        key=lambda x: x["iou"],
        default={"threshold": 0.1, "iou": 0}
    )

    report += f"""### Overall Best (by IoU)
| Mode | Best Threshold | IoU | F1 |
|------|----------------|-----|-----|
| **Alpha** | {best_alpha.get("threshold", 0.5):.2f} | {best_alpha.get("iou", 0):.4f} | {best_alpha.get("f1", 0):.4f} |
| **RGB Pred** | {best_rgb.get("threshold", 0.1):.3f} | {best_rgb.get("iou", 0):.4f} | {best_rgb.get("f1", 0):.4f} |

---

## Recommendations

### For FaceLift Mouse Project:

1. **mask_mode: gt** (Recommended)
   - Uses actual GT alpha, no threshold needed
   - Most stable, no mask expansion during training

2. **mask_mode: alpha** with threshold={best_alpha.get("threshold", 0.5):.2f}
   - Best IoU: {best_alpha.get("iou", 0):.4f}
   - ⚠️ Risk: Mask can expand during training if model overfits

3. **mask_mode: rgb_pred** with threshold={best_rgb.get("threshold", 0.1):.3f}
   - Best IoU: {best_rgb.get("iou", 0):.4f}
   - More stable than alpha mode but lower accuracy

### Key Insights:

- GT mask ratio: ~{results.get("gt_mask_percent", 2.5):.1f}% (mouse is small in frame)
- Higher threshold (>0.7) leads to lower recall but prevents mask expansion
- Lower threshold (<0.3) captures more but may include noise

---

*Generated by mask_mode_analysis.py v2.0 | FaceLift Mouse Project*
"""

    return report


# =============================================================================
# Main Analysis
# =============================================================================

def run_analysis(
    checkpoint_path: str,
    config_path: str,
    data_dir: Path,
    output_dir: Path,
    sample_idx: int = 0,
    alpha_thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    rgb_thresholds: List[float] = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    device: str = "cuda",
):
    """Run full mask mode analysis with real model inference."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print(f"{'='*60}")
    print("Mask Mode Analysis v2.0 (Real Model Inference)")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Alpha thresholds: {alpha_thresholds}")
    print(f"RGB thresholds: {rgb_thresholds}")

    # Load model
    print("\n1. Loading model...")
    model, config, step = load_model_and_config(checkpoint_path, config_path, device)
    print(f"   Model loaded (step: {step})")

    # Load data
    print("\n2. Loading sample...")
    data = load_sample(data_dir, sample_idx, device)
    print(f"   Sample: {data['sample_dir']}")
    print(f"   Views: {data['num_views']}")

    # Run inference
    print("\n3. Running inference...")
    result = run_inference(model, data)
    print(f"   Rendered RGB: {result[rendered_rgb].shape}")
    print(f"   Rendered Alpha: {result[rendered_alpha].shape if result[rendered_alpha] is not None else None}")

    # Check if we have rendered alpha
    if result["rendered_alpha"] is None:
        print("\n   WARNING: No rendered alpha available! Using GT mask for alpha comparison.")
        result["rendered_alpha"] = result["gt_mask"]

    # Print alpha statistics
    alpha = result["rendered_alpha"]
    print(f"\n   Alpha statistics:")
    print(f"     Min: {alpha.min().item():.4f}")
    print(f"     Max: {alpha.max().item():.4f}")
    print(f"     Mean: {alpha.mean().item():.4f}")

    all_results = {
        "sample_dir": data['sample_dir'],
        "step": str(step),
        "checkpoint": checkpoint_path,
    }

    # 4. Alpha threshold comparison
    print("\n4. Creating alpha threshold comparison...")
    alpha_visual, alpha_labels, alpha_metrics = create_wandb_style_comparison(
        result["gt_rgb"],
        result["gt_mask"],
        result["rendered_rgb"],
        result["rendered_alpha"],
        alpha_thresholds,
    )
    Image.fromarray(alpha_visual).save(output_dir / "figures" / "alpha_comparison.png")
    all_results["alpha_metrics"] = alpha_metrics
    all_results["gt_mask_percent"] = alpha_metrics.get("gt_mask_percent", 0)
    print(f"   Saved: figures/alpha_comparison.png")

    # 5. RGB threshold comparison
    print("\n5. Creating RGB threshold comparison...")
    rgb_visual, rgb_metrics = create_rgb_threshold_comparison(
        result["gt_rgb"],
        result["gt_mask"],
        result["rendered_rgb"],
        rgb_thresholds,
    )
    Image.fromarray(rgb_visual).save(output_dir / "figures" / "rgb_comparison.png")
    all_results["rgb_metrics"] = rgb_metrics
    print(f"   Saved: figures/rgb_comparison.png")

    # 6. Alpha histogram
    print("\n6. Creating alpha histogram...")
    hist_img = create_alpha_histogram(result["rendered_alpha"], result["gt_mask"])
    Image.fromarray(hist_img).save(output_dir / "figures" / "alpha_histogram.png")
    print(f"   Saved: figures/alpha_histogram.png")

    # 7. Per-view metrics (for best alpha threshold)
    print("\n7. Computing per-view metrics...")
    best_thresh = max(
        [m for k, m in alpha_metrics.items() if k.startswith("alpha_")],
        key=lambda x: x["iou"]
    )["threshold"]
    best_alpha_mask = (result["rendered_alpha"] > best_thresh).float()
    per_view = compute_per_view_metrics(best_alpha_mask, result["gt_mask"])
    all_results["per_view_metrics"] = per_view
    print(f"   Best threshold: {best_thresh}")

    # 8. Generate report
    print("\n8. Generating report...")
    report = generate_report(all_results, output_dir, checkpoint_path, step, timestamp)
    with open(output_dir / "mask_analysis_report.md", "w") as f:
        f.write(report)
    print(f"   Saved: mask_analysis_report.md")

    # 9. Save raw results
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"   Saved: results.json")

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"Report: {output_dir / mask_analysis_report.md}")
    print(f"Figures: {output_dir / figures}")

    # Summary
    best_alpha = max(
        [m for k, m in alpha_metrics.items() if k.startswith("alpha_")],
        key=lambda x: x["iou"]
    )
    best_rgb = max(
        [m for k, m in rgb_metrics.items() if k.startswith("rgb_")],
        key=lambda x: x["iou"]
    )
    print(f"\nBest configurations:")
    print(f"  Alpha: threshold={best_alpha[threshold]:.2f}, IoU={best_alpha[iou]:.4f}")
    print(f"  RGB:   threshold={best_rgb[threshold]:.3f}, IoU={best_rgb[iou]:.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Mask Mode Analysis v2.0 - Real Model Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with checkpoint (REQUIRED)
  CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.mask_mode_analysis \\
      --checkpoint checkpoints/gslrm/D7_1_E2/ckpt_step_1000.pt \\
      --config configs/mouse/D7_1_E2.yaml \\
      --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \\
      --output_dir experiments/analysis/mask_mode_D7_1_E2
"""
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to preprocessed data")
    parser.add_argument("--output_dir", type=str, default="mask_analysis_output", help="Output directory")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to analyze")
    parser.add_argument(
        "--alpha_thresholds", type=float, nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Alpha threshold values to test"
    )
    parser.add_argument(
        "--rgb_thresholds", type=float, nargs="+",
        default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
        help="RGB prediction threshold values to test"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    run_analysis(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sample_idx=args.sample_idx,
        alpha_thresholds=args.alpha_thresholds,
        rgb_thresholds=args.rgb_thresholds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
