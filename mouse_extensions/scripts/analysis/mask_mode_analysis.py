#!/usr/bin/env python3
"""
Mask Mode Analysis Script for FaceLift Mouse Project

Generates quantitative and qualitative comparison report for different mask modes
and threshold parameters using real mouse data.

Usage:
    # Basic usage with D9 dataset
    python -m mouse_extensions.scripts.analysis.mask_mode_analysis \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D9

    # Custom thresholds
    python -m mouse_extensions.scripts.analysis.mask_mode_analysis \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
        --alpha_thresholds 0.3 0.5 0.7 \
        --rgb_thresholds 0.05 0.1 0.2

Output:
    - mask_analysis_report.md: Markdown report
    - figures/: Visualization images (WandB style)
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
from PIL import Image
from einops import rearrange

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mouse_extensions.model.visualization_extensions import (
    VisualizationConfig,
    create_mask_overlay,
    create_error_heatmap,
    compute_pred_mask,
    compute_error_stats,
    _add_camera_labels,
)


# =============================================================================
# Data Loading (Real Mouse Data Only)
# =============================================================================

def load_first_frame_data(data_dir: Path, view_indices: List[int] = None) -> Dict[str, torch.Tensor]:
    """
    Load first frame from preprocessed mouse dataset (all 6 views).

    Args:
        data_dir: Path to preprocessed dataset (e.g., D9/)
        view_indices: Optional specific view indices to load (default: all 6)

    Returns:
        Dict with:
            - images: [V, 3, H, W] RGB images
            - masks: [V, 1, H, W] GT masks (if available)
            - view_indices: List of camera indices
    """
    train_dir = data_dir / "train"
    if not train_dir.exists():
        train_dir = data_dir

    # Find first sample (supports multiple naming conventions)
    # Try: sample_*, numeric (000000), or any directory
    sample_dirs = sorted(train_dir.glob("sample_*"))
    if not sample_dirs:
        # Try numeric directories (D9 style: 000000, 000001, ...)
        sample_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not sample_dirs:
        # Fallback: any directory
        sample_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {train_dir}")

    sample_dir = sample_dirs[0]
    print(f"Loading from: {sample_dir}")

    img_dir = sample_dir / "images"
    mask_dir = sample_dir / "masks"

    # Get all image files
    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    # Determine view indices
    if view_indices is None:
        view_indices = list(range(len(img_files)))

    images = []
    masks = []
    actual_view_indices = []

    for idx in view_indices:
        if idx >= len(img_files):
            continue

        img_file = img_files[idx]
        actual_view_indices.append(idx)

        # Load image
        img = Image.open(img_file).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # [3, H, W]
        images.append(img_tensor)

        # Load mask if exists
        mask_file = mask_dir / img_file.name
        if mask_file.exists():
            mask = Image.open(mask_file).convert("L")
            mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]
            masks.append(mask_tensor)

    result = {
        "images": torch.stack(images),  # [V, 3, H, W]
        "view_indices": actual_view_indices,
        "sample_dir": str(sample_dir),
    }

    if masks:
        result["masks"] = torch.stack(masks)  # [V, 1, H, W]

    print(f"Loaded {len(images)} views, shape: {result['images'].shape}")
    if masks:
        print(f"GT masks available: {result['masks'].shape}")

    return result


# =============================================================================
# Quantitative Metrics
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
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fg_ratio": pred.mean().item(),
    }


# =============================================================================
# WandB-Style Visualization Functions
# =============================================================================

def create_mode_comparison_visual(
    images: torch.Tensor,
    gt_mask: torch.Tensor,
    alpha_threshold: float = 0.5,
    rgb_threshold: float = 0.1,
    view_indices: List[int] = None,
) -> Tuple[np.ndarray, Dict[str, Dict]]:
    """
    Create WandB-style multi-row visualization comparing mask modes.

    Output Layout:
        Row 1: GT Images (all views)
        Row 2: GT + GT Mask overlay
        Row 3: GT + Alpha Mask overlay (simulated from GT with noise)
        Row 4: GT + RGB Pred Mask overlay
        Row 5: Difference Heatmap (GT vs RGB Pred)

    Args:
        images: [V, 3, H, W] RGB images
        gt_mask: [V, 1, H, W] GT masks
        alpha_threshold: Threshold for alpha mode
        rgb_threshold: Threshold for rgb_pred mode
        view_indices: Camera indices for labeling

    Returns:
        Tuple of (visualization numpy array, metrics dict per mode)
    """
    device = images.device
    v, _, h, w = images.shape

    config_gt = VisualizationConfig(mask_mode="gt", overlay_blend=0.3)
    config_alpha = VisualizationConfig(
        mask_mode="alpha",
        alpha_threshold=alpha_threshold,
        overlay_blend=0.3
    )
    config_rgb = VisualizationConfig(
        mask_mode="rgb_pred",
        rgb_threshold=rgb_threshold,
        overlay_blend=0.3
    )

    # Simulate rendered_alpha from GT mask with noise (for testing without model)
    noise = torch.randn_like(gt_mask) * 0.15
    simulated_alpha = (gt_mask + noise).clamp(0, 1)

    # Compute masks
    alpha_mask = (simulated_alpha > alpha_threshold).float()
    rgb_mask = compute_pred_mask(images, None, config_rgb)  # RGB-based

    # Compute metrics
    metrics = {
        "alpha": compute_mask_metrics(alpha_mask, gt_mask),
        "rgb_pred": compute_mask_metrics(rgb_mask, gt_mask),
        "gt": {"iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
    }

    # Build visualization rows
    rows = []

    # Row 1: Original images
    rows.append(images)

    # Row 2: GT mask overlay
    gt_overlay = create_mask_overlay(images, gt_mask, config_gt)
    rows.append(gt_overlay)

    # Row 3: Alpha mask overlay
    alpha_overlay = create_mask_overlay(images, alpha_mask, config_alpha)
    rows.append(alpha_overlay)

    # Row 4: RGB pred mask overlay
    rgb_overlay = create_mask_overlay(images, rgb_mask, config_rgb)
    rows.append(rgb_overlay)

    # Row 5: Difference heatmap (GT vs RGB Pred)
    error_raw = (gt_mask.float() - rgb_mask.float()).abs()
    union_mask = ((gt_mask > 0.5).float() + (rgb_mask > 0.5).float() > 0.5).float()
    error_heatmap = create_error_heatmap(error_raw, union_mask, config_gt, error_max=1.0)
    rows.append(error_heatmap)

    # Stack rows: [num_rows, V, C, H, W]
    visual = torch.stack(rows, dim=0)

    # Rearrange to image: [num_rows * H, V * W, 3]
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")

    # Convert to numpy
    visual_np = (visual.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Add camera labels
    if view_indices is not None:
        visual_np = _add_camera_labels(visual_np, view_indices, w)

    return visual_np, metrics


def create_threshold_sweep_visual(
    images: torch.Tensor,
    gt_mask: torch.Tensor,
    mode: str = "alpha",
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    view_idx: int = 0,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Create threshold sweep visualization for a single view.

    Output Layout (for each threshold):
        Row 1: Mask at threshold
        Row 2: Overlay at threshold
        Row 3: Difference from GT

    Args:
        images: [V, 3, H, W] RGB images
        gt_mask: [V, 1, H, W] GT masks
        mode: "alpha" or "rgb_pred"
        thresholds: List of threshold values
        view_idx: Which view to visualize

    Returns:
        Tuple of (visualization numpy array, metrics list)
    """
    device = images.device

    # Extract single view
    img = images[view_idx:view_idx+1]  # [1, 3, H, W]
    gt = gt_mask[view_idx:view_idx+1]  # [1, 1, H, W]

    # Simulate alpha for testing
    noise = torch.randn_like(gt) * 0.15
    simulated_alpha = (gt + noise).clamp(0, 1)

    config = VisualizationConfig(overlay_blend=0.4)

    metrics_list = []
    columns = []

    # First column: GT reference
    gt_col = []
    gt_col.append(gt.expand(-1, 3, -1, -1))  # Mask as grayscale RGB
    gt_col.append(create_mask_overlay(img, gt, config))
    gt_col.append(torch.zeros_like(img))  # No diff for GT
    columns.append(torch.cat(gt_col, dim=2))  # Stack vertically

    # Threshold columns
    for thresh in thresholds:
        if mode == "alpha":
            pred = (simulated_alpha > thresh).float()
        else:
            color_dist = (img - 1.0).abs().mean(dim=1, keepdim=True)
            pred = (color_dist > thresh).float()

        # Compute metrics
        m = compute_mask_metrics(pred, gt)
        m["threshold"] = thresh
        metrics_list.append(m)

        col = []
        # Row 1: Mask
        col.append(pred.expand(-1, 3, -1, -1))
        # Row 2: Overlay
        col.append(create_mask_overlay(img, pred, config))
        # Row 3: Difference heatmap
        diff = (gt.float() - pred.float()).abs()
        diff_heat = create_error_heatmap(diff, error_max=1.0)
        col.append(diff_heat)

        columns.append(torch.cat(col, dim=2))

    # Stack columns horizontally
    visual = torch.cat(columns, dim=3)
    visual = visual.squeeze(0).permute(1, 2, 0)  # [H*3, W*(n+1), 3]

    visual_np = (visual.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    return visual_np, metrics_list


def create_all_views_threshold_grid(
    images: torch.Tensor,
    gt_mask: torch.Tensor,
    mode: str = "alpha",
    threshold: float = 0.5,
    view_indices: List[int] = None,
) -> np.ndarray:
    """
    Create grid showing all views at a specific threshold.

    Layout:
        Row 1: GT images
        Row 2: GT mask overlay
        Row 3: Predicted mask overlay
        Row 4: Difference heatmap
    """
    device = images.device
    v, _, h, w = images.shape

    config = VisualizationConfig(
        mask_mode=mode,
        alpha_threshold=threshold,
        rgb_threshold=threshold,
        overlay_blend=0.3
    )

    # Simulate alpha
    noise = torch.randn_like(gt_mask) * 0.15
    simulated_alpha = (gt_mask + noise).clamp(0, 1)

    # Compute predicted mask
    if mode == "alpha":
        pred_mask = (simulated_alpha > threshold).float()
    else:
        pred_mask = compute_pred_mask(images, None, config)

    rows = [
        images,
        create_mask_overlay(images, gt_mask, config),
        create_mask_overlay(images, pred_mask, config),
        create_error_heatmap(
            (gt_mask.float() - pred_mask.float()).abs(),
            error_max=1.0
        ),
    ]

    visual = torch.stack(rows, dim=0)
    visual = rearrange(visual, "rows v c h w -> (rows h) (v w) c")
    visual_np = (visual.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    if view_indices:
        visual_np = _add_camera_labels(visual_np, view_indices, w)

    return visual_np


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: Dict,
    output_dir: Path,
    data_dir: Path,
    timestamp: str,
) -> str:
    """Generate markdown report."""

    report = f"""# Mask Mode Analysis Report

**Generated**: {timestamp}
**Data Source**: `{data_dir}`
**Sample**: `{results.get('sample_dir', 'N/A')}`

---

## 1. Mode Comparison (All 6 Views)

### Quantitative Metrics vs GT Mask

| Mode | IoU | F1 | Precision | Recall | FG Ratio |
|------|-----|----|-----------| -------|----------|
"""

    for mode, m in results.get("mode_metrics", {}).items():
        report += f"| `{mode}` | {m['iou']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m.get('fg_ratio', 0):.3f} |\n"

    report += """
### Visual Comparison

![Mode Comparison](figures/mode_comparison.png)

**Rows (top to bottom)**:
1. GT Images
2. GT + GT Mask overlay (green=foreground, red=background)
3. GT + Alpha Mask overlay (simulated)
4. GT + RGB Pred Mask overlay
5. Difference Heatmap (GT vs RGB Pred)

---

## 2. Alpha Threshold Sweep

"""

    if "alpha_sweep" in results:
        report += "| Threshold | IoU | F1 | Precision | Recall |\n"
        report += "|-----------|-----|----|-----------| -------|\n"
        for m in results["alpha_sweep"]:
            report += f"| {m['threshold']:.2f} | {m['iou']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |\n"

        report += "\n![Alpha Sweep](figures/alpha_threshold_sweep.png)\n"
        report += "\n**Columns**: GT, then each threshold\n"
        report += "**Rows**: Mask, Overlay, Difference from GT\n"

    report += """
---

## 3. RGB Prediction Threshold Sweep

"""

    if "rgb_sweep" in results:
        report += "| Threshold | IoU | F1 | Precision | Recall |\n"
        report += "|-----------|-----|----|-----------| -------|\n"
        for m in results["rgb_sweep"]:
            report += f"| {m['threshold']:.2f} | {m['iou']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} |\n"

        report += "\n![RGB Sweep](figures/rgb_threshold_sweep.png)\n"

    report += f"""
---

## 4. Recommendations

Based on the analysis:

| Mode | Best Threshold | IoU |
|------|----------------|-----|
| Alpha | {results.get('best_alpha_threshold', 0.5):.2f} | {results.get('best_alpha_iou', 0):.4f} |
| RGB Pred | {results.get('best_rgb_threshold', 0.1):.2f} | {results.get('best_rgb_iou', 0):.4f} |

### Recommended Configuration

```yaml
training:
  losses:
    mask_mode: "alpha"
    alpha_mask_threshold: {results.get('best_alpha_threshold', 0.5):.2f}
    pred_mask_threshold: {results.get('best_rgb_threshold', 0.1):.2f}
    masked_l2_loss: true
```

---

*Generated by mask_mode_analysis.py | FaceLift Mouse Project*
"""

    return report


# =============================================================================
# Main
# =============================================================================

def run_analysis(
    data_dir: Path,
    output_dir: Path,
    alpha_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    rgb_thresholds: List[float] = [0.05, 0.1, 0.15, 0.2, 0.3],
):
    """Run full mask mode analysis on real mouse data."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print(f"{'='*60}")
    print("Mask Mode Analysis")
    print(f"{'='*60}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")

    # Load first frame (all 6 views)
    data = load_first_frame_data(data_dir)
    images = data["images"]
    view_indices = data["view_indices"]

    if "masks" not in data:
        raise ValueError("GT masks required for analysis. Dataset must have masks/ directory.")

    gt_mask = data["masks"]

    results = {
        "sample_dir": data["sample_dir"],
        "num_views": len(view_indices),
        "view_indices": view_indices,
    }

    # 1. Mode comparison (all views)
    print("\n1. Creating mode comparison...")
    visual, mode_metrics = create_mode_comparison_visual(
        images, gt_mask,
        alpha_threshold=0.5,
        rgb_threshold=0.1,
        view_indices=view_indices,
    )
    Image.fromarray(visual).save(output_dir / "figures" / "mode_comparison.png")
    results["mode_metrics"] = mode_metrics

    # 2. Alpha threshold sweep (single view for clarity)
    print("2. Creating alpha threshold sweep...")
    visual, alpha_metrics = create_threshold_sweep_visual(
        images, gt_mask,
        mode="alpha",
        thresholds=alpha_thresholds,
        view_idx=0,
    )
    Image.fromarray(visual).save(output_dir / "figures" / "alpha_threshold_sweep.png")
    results["alpha_sweep"] = alpha_metrics

    best_alpha = max(alpha_metrics, key=lambda x: x["iou"])
    results["best_alpha_threshold"] = best_alpha["threshold"]
    results["best_alpha_iou"] = best_alpha["iou"]

    # 3. RGB threshold sweep
    print("3. Creating RGB threshold sweep...")
    visual, rgb_metrics = create_threshold_sweep_visual(
        images, gt_mask,
        mode="rgb_pred",
        thresholds=rgb_thresholds,
        view_idx=0,
    )
    Image.fromarray(visual).save(output_dir / "figures" / "rgb_threshold_sweep.png")
    results["rgb_sweep"] = rgb_metrics

    best_rgb = max(rgb_metrics, key=lambda x: x["iou"])
    results["best_rgb_threshold"] = best_rgb["threshold"]
    results["best_rgb_iou"] = best_rgb["iou"]

    # 4. Best threshold grid (all views)
    print("4. Creating best threshold grids...")
    visual = create_all_views_threshold_grid(
        images, gt_mask,
        mode="alpha",
        threshold=best_alpha["threshold"],
        view_indices=view_indices,
    )
    Image.fromarray(visual).save(output_dir / "figures" / "best_alpha_all_views.png")

    visual = create_all_views_threshold_grid(
        images, gt_mask,
        mode="rgb_pred",
        threshold=best_rgb["threshold"],
        view_indices=view_indices,
    )
    Image.fromarray(visual).save(output_dir / "figures" / "best_rgb_all_views.png")

    # 5. Generate report
    print("5. Generating report...")
    report = generate_report(results, output_dir, data_dir, timestamp)

    with open(output_dir / "mask_analysis_report.md", "w") as f:
        f.write(report)

    # Save raw results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print(f"Report: {output_dir / 'mask_analysis_report.md'}")
    print(f"Figures: {output_dir / 'figures'}")
    print(f"\nBest thresholds:")
    print(f"  Alpha: {results['best_alpha_threshold']:.2f} (IoU={results['best_alpha_iou']:.4f})")
    print(f"  RGB:   {results['best_rgb_threshold']:.2f} (IoU={results['best_rgb_iou']:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Mask Mode Analysis for FaceLift Mouse Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze D9 dataset
  python -m mouse_extensions.scripts.analysis.mask_mode_analysis \\
      --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D9

  # Custom output and thresholds
  python -m mouse_extensions.scripts.analysis.mask_mode_analysis \\
      --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D4 \\
      --output_dir ./analysis_d4 \\
      --alpha_thresholds 0.3 0.5 0.7
"""
    )

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to preprocessed dataset directory (must have masks/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="mask_analysis_output",
        help="Output directory for report and figures"
    )
    parser.add_argument(
        "--alpha_thresholds", type=float, nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help="Alpha threshold values to test"
    )
    parser.add_argument(
        "--rgb_thresholds", type=float, nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.3],
        help="RGB prediction threshold values to test"
    )

    args = parser.parse_args()

    run_analysis(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        alpha_thresholds=args.alpha_thresholds,
        rgb_thresholds=args.rgb_thresholds,
    )


if __name__ == "__main__":
    main()
