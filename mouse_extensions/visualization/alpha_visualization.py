"""
Alpha Mask Visualization Module

Provides visualization for rendered alpha masks when alpha_loss_weight > 0.
Color scheme consistent with supervision visualization:
- Foreground (object): Green (0.2, 0.8, 0.2)
- Background: Red (0.8, 0.2, 0.2)

Author: Claude Code
Date: 2026-01-26 (Updated for color consistency)
"""

import numpy as np
import torch
from typing import Optional, Dict, Tuple
from PIL import Image

# Consistent color scheme with supervision visualization
FG_COLOR = (0.2, 0.8, 0.2)   # Green for foreground
BG_COLOR = (0.8, 0.2, 0.2)   # Red for background


def visualize_alpha_comparison(
    gt_mask: torch.Tensor,
    rendered_alpha: torch.Tensor,
    num_views: int = 6,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Create side-by-side comparison of GT mask and rendered alpha.

    Layout (4 rows x num_views cols):
    - Row 0: GT Mask (Green=FG, Red=BG)
    - Row 1: Rendered Alpha (continuous grayscale)
    - Row 2: Rendered Alpha > threshold (Green=FG, Red=BG)
    - Row 3: Difference (Green=TP, Red=FN, Blue=FP)

    Args:
        gt_mask: GT mask tensor [V, 1, H, W] or [B*V, 1, H, W]
        rendered_alpha: Rendered alpha tensor [V, 1, H, W] or [B*V, 1, H, W]
        num_views: Number of views
        threshold: Threshold for binary conversion

    Returns:
        Visualization image as numpy array [H*4, W*V, 3]
    """
    device = gt_mask.device

    v = num_views
    gt = gt_mask[:v].squeeze(1)  # [V, H, W]
    alpha = rendered_alpha[:v].squeeze(1)  # [V, H, W]

    h, w = gt.shape[1], gt.shape[2]

    # Color tensors
    fg = torch.tensor(FG_COLOR, device=device).view(1, 1, 1, 3)
    bg = torch.tensor(BG_COLOR, device=device).view(1, 1, 1, 3)

    # Row 0: GT Mask (Green FG, Red BG)
    gt_expanded = gt.unsqueeze(-1)  # [V, H, W, 1]
    gt_vis = gt_expanded * fg + (1 - gt_expanded) * bg  # [V, H, W, 3]

    # Row 1: Rendered Alpha (grayscale - keep as-is for continuous visualization)
    alpha_vis = alpha.unsqueeze(-1).expand(-1, -1, -1, 3)  # [V, H, W, 3]

    # Row 2: Rendered Alpha thresholded (Green FG, Red BG)
    alpha_binary = (alpha > threshold).float().unsqueeze(-1)  # [V, H, W, 1]
    alpha_binary_vis = alpha_binary * fg + (1 - alpha_binary) * bg

    # Row 3: Difference visualization (confusion matrix colors)
    gt_binary = (gt > threshold).float()
    pred_binary = (alpha > threshold).float()

    diff_vis = torch.zeros(v, h, w, 3, device=device)

    # True Positive (both 1): Green - correct foreground
    tp = (gt_binary * pred_binary).bool()
    diff_vis[tp] = torch.tensor(FG_COLOR, device=device)

    # True Negative (both 0): Dark red - correct background
    tn = ((1 - gt_binary) * (1 - pred_binary)).bool()
    diff_vis[tn] = torch.tensor((0.3, 0.1, 0.1), device=device)  # Dark red

    # False Negative (pred=0, gt=1): Bright Red - missed foreground
    fn = (gt_binary * (1 - pred_binary)).bool()
    diff_vis[fn] = torch.tensor((1.0, 0.2, 0.2), device=device)  # Bright red

    # False Positive (pred=1, gt=0): Blue - false foreground
    fp = ((1 - gt_binary) * pred_binary).bool()
    diff_vis[fp] = torch.tensor((0.2, 0.2, 1.0), device=device)  # Blue

    # Stack rows
    rows = torch.stack([gt_vis, alpha_vis, alpha_binary_vis, diff_vis], dim=0)  # [4, V, H, W, 3]

    # Rearrange to [4*H, V*W, 3]
    rows = rows.permute(0, 2, 1, 3, 4)  # [4, H, V, W, 3]
    rows = rows.reshape(4 * h, v * w, 3)

    # Convert to numpy
    vis_np = (rows.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    # Add labels
    vis_np = _add_row_labels_alpha(vis_np, h)

    return vis_np


def _add_row_labels_alpha(image: np.ndarray, row_height: int) -> np.ndarray:
    """Add row labels to alpha comparison image."""
    try:
        from PIL import ImageDraw, ImageFont

        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()

        labels = [
            "GT Mask",
            "Rendered Alpha", 
            "Alpha > 0.5",
            "Diff (G=TP, R=FN, B=FP)"
        ]

        for i, label in enumerate(labels):
            y = i * row_height + 8
            # Draw text with outline for visibility
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
                draw.text((8+dx, y+dy), label, fill=(0,0,0), font=font)
            draw.text((8, y), label, fill=(255,255,255), font=font)

        return np.array(pil_img)
    except Exception as e:
        print(f"Warning: Could not add labels: {e}")
        return image


def compute_alpha_metrics(
    gt_mask: torch.Tensor,
    rendered_alpha: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute alpha mask metrics (IoU, precision, recall)."""
    gt_binary = (gt_mask > threshold).float()
    pred_binary = (rendered_alpha > threshold).float()

    gt_flat = gt_binary.view(-1)
    pred_flat = pred_binary.view(-1)

    intersection = (gt_flat * pred_flat).sum()
    union = ((gt_flat + pred_flat) > 0).float().sum()

    iou = (intersection / (union + 1e-6)).item()

    tp = intersection
    fp = ((1 - gt_flat) * pred_flat).sum()
    fn = (gt_flat * (1 - pred_flat)).sum()

    precision = (tp / (tp + fp + 1e-6)).item()
    recall = (tp / (tp + fn + 1e-6)).item()
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "alpha_iou": iou,
        "alpha_precision": precision,
        "alpha_recall": recall,
        "alpha_f1": f1,
    }


def should_visualize_alpha(config) -> bool:
    """Check if alpha visualization should be enabled based on config."""
    try:
        alpha_weight = getattr(config.training.losses, "alpha_loss_weight", 0.0)
        return alpha_weight > 0
    except:
        return False
