"""
Visualization Extensions

Additional visualizations for mask comparison and threshold analysis.
Includes quantitative metrics (IoU) displayed on each panel.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple


def compute_mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """Compute IoU between predicted and ground truth masks."""
    pred_binary = (pred_mask > 0.5).float()
    gt_binary = (gt_mask > 0.5).float()
    
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum() - intersection
    
    if union < 1e-6:
        return 1.0 if intersection < 1e-6 else 0.0
    return (intersection / union).item()


def _add_text_to_image(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int] = (5, 20),
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Add text with background to image for visibility."""
    try:
        import cv2
        result = img.copy()
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(result, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
        
        # Draw text
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return result
    except ImportError:
        return img


def create_threshold_comparison(
    rendering: torch.Tensor,  # [b*v, 3, h, w]
    rendered_alpha: torch.Tensor,  # [b*v, 1, h, w]
    gt_mask: Optional[torch.Tensor],  # [b*v, 1, h, w]
    num_views: int,
    height: int,
    alpha_thresholds: List[float] = [0.1, 0.3, 0.5, 0.7],
    rgb_thresholds: List[float] = [0.05, 0.1, 0.2, 0.3],
) -> Optional[np.ndarray]:
    """
    Create threshold comparison visualization with quantitative metrics.

    Shows masks at different thresholds with IoU values displayed.
    - Row 1: Alpha thresholds with IoU
    - Row 2: RGB thresholds with IoU
    - Row 3: GT vs Alpha(0.5) vs RGB(0.1) vs Diff

    Args:
        rendering: Rendered RGB images [b*v, 3, h, w]
        rendered_alpha: Rendered alpha from Gaussians [b*v, 1, h, w]
        gt_mask: Ground truth mask (optional) [b*v, 1, h, w]
        num_views: Number of views (v)
        height: Image height
        alpha_thresholds: Thresholds for alpha mask
        rgb_thresholds: Thresholds for RGB mask

    Returns:
        np.ndarray: Visualization image (H, W, 3) uint8 with metrics
    """
    if rendered_alpha is None:
        return None

    device = rendering.device
    v = num_views

    # Use first batch item only
    render_sample = rendering[0:v]  # [v, 3, h, w]
    alpha_sample = rendered_alpha[0:v]  # [v, 1, h, w]
    gt_mask_sample = gt_mask[0:v] if gt_mask is not None else None

    # Colors for overlay
    fg_color = torch.tensor([0.2, 0.8, 0.2], device=device).view(1, 3, 1, 1)  # green
    bg_color_overlay = torch.tensor([0.8, 0.2, 0.2], device=device).view(1, 3, 1, 1)  # red

    rows = []
    img_width = render_sample.shape[-1]

    # Row 1: Alpha threshold masks (first view with different thresholds)
    alpha_row = []
    for thresh in alpha_thresholds:
        alpha_mask = (alpha_sample[0:1] > thresh).float()
        mask_rgb = alpha_mask.expand(-1, 3, -1, -1)
        overlay = render_sample[0:1] * 0.5 + mask_rgb * fg_color + (1 - mask_rgb) * bg_color_overlay * 0.5
        overlay_np = (overlay.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Compute IoU if GT available
        if gt_mask_sample is not None:
            iou = compute_mask_iou(alpha_mask, gt_mask_sample[0:1])
            label = f"Alpha>{thresh:.1f} IoU:{iou:.3f}"
        else:
            coverage = alpha_mask.float().mean().item()
            label = f"Alpha>{thresh:.1f} Cov:{coverage:.2f}"
        
        overlay_np = _add_text_to_image(overlay_np, label)
        alpha_row.append(overlay_np)
    
    alpha_row = np.hstack(alpha_row)
    rows.append(alpha_row)

    # Row 2: RGB threshold masks
    rgb_row = []
    color_dist = (render_sample[0:1] - 1.0).abs().mean(dim=1, keepdim=True)
    for thresh in rgb_thresholds:
        rgb_mask = (color_dist > thresh).float()
        mask_rgb = rgb_mask.expand(-1, 3, -1, -1)
        overlay = render_sample[0:1] * 0.5 + mask_rgb * fg_color + (1 - mask_rgb) * bg_color_overlay * 0.5
        overlay_np = (overlay.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Compute IoU if GT available
        if gt_mask_sample is not None:
            iou = compute_mask_iou(rgb_mask, gt_mask_sample[0:1])
            label = f"RGB>{thresh:.2f} IoU:{iou:.3f}"
        else:
            coverage = rgb_mask.float().mean().item()
            label = f"RGB>{thresh:.2f} Cov:{coverage:.2f}"
        
        overlay_np = _add_text_to_image(overlay_np, label)
        rgb_row.append(overlay_np)
    
    rgb_row = np.hstack(rgb_row)
    rows.append(rgb_row)

    # Row 3: GT vs Alpha(0.5) vs RGB(0.1) vs Diff
    if gt_mask_sample is not None:
        comparison_row = []

        # GT mask
        gt_binary = (gt_mask_sample[0:1] > 0.5).float()
        gt_overlay = _create_mask_overlay(render_sample[0:1], gt_binary, fg_color, bg_color_overlay)
        gt_overlay_np = (gt_overlay * 255).clip(0, 255).astype(np.uint8)
        gt_coverage = gt_binary.float().mean().item()
        gt_overlay_np = _add_text_to_image(gt_overlay_np, f"GT Cov:{gt_coverage:.3f}")
        comparison_row.append(gt_overlay_np)

        # Alpha mask (0.5)
        alpha_binary = (alpha_sample[0:1] > 0.5).float()
        alpha_overlay = _create_mask_overlay(render_sample[0:1], alpha_binary, fg_color, bg_color_overlay)
        alpha_overlay_np = (alpha_overlay * 255).clip(0, 255).astype(np.uint8)
        alpha_iou = compute_mask_iou(alpha_binary, gt_binary)
        alpha_overlay_np = _add_text_to_image(alpha_overlay_np, f"Alpha0.5 IoU:{alpha_iou:.3f}")
        comparison_row.append(alpha_overlay_np)

        # RGB mask (0.1)
        rgb_binary = (color_dist > 0.1).float()
        rgb_overlay = _create_mask_overlay(render_sample[0:1], rgb_binary, fg_color, bg_color_overlay)
        rgb_overlay_np = (rgb_overlay * 255).clip(0, 255).astype(np.uint8)
        rgb_iou = compute_mask_iou(rgb_binary, gt_binary)
        rgb_overlay_np = _add_text_to_image(rgb_overlay_np, f"RGB0.1 IoU:{rgb_iou:.3f}")
        comparison_row.append(rgb_overlay_np)

        # Diff: Alpha - GT
        diff_img = _create_diff_visualization(alpha_binary, gt_binary, height, img_width, device)
        diff_np = (diff_img * 255).clip(0, 255).astype(np.uint8)
        # Compute precision/recall
        tp = ((alpha_binary > 0.5) & (gt_binary > 0.5)).float().sum()
        fp = ((alpha_binary > 0.5) & (gt_binary < 0.5)).float().sum()
        fn = ((alpha_binary < 0.5) & (gt_binary > 0.5)).float().sum()
        precision = (tp / (tp + fp + 1e-6)).item()
        recall = (tp / (tp + fn + 1e-6)).item()
        diff_np = _add_text_to_image(diff_np, f"P:{precision:.2f} R:{recall:.2f}")
        comparison_row.append(diff_np)

        comparison_row = np.hstack(comparison_row)
        rows.append(comparison_row)

    # Ensure all rows have same width
    max_width = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_width:
            pad = np.ones((r.shape[0], max_width - r.shape[1], 3), dtype=np.uint8) * 128
            r = np.hstack([r, pad])
        padded_rows.append(r)

    result = np.vstack(padded_rows)
    return result


def _create_mask_overlay(
    render: torch.Tensor,
    mask: torch.Tensor,
    fg_color: torch.Tensor,
    bg_color: torch.Tensor,
) -> np.ndarray:
    """Create mask overlay visualization."""
    mask_rgb = mask.expand(-1, 3, -1, -1)
    overlay = render * 0.5 + mask_rgb * fg_color + (1 - mask_rgb) * bg_color * 0.5
    return overlay.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _create_diff_visualization(
    alpha_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> np.ndarray:
    """
    Create difference visualization.

    Colors:
    - Red: GT only (false negative)
    - Blue: Alpha only (false positive)
    - Green: Both agree (true positive)
    """
    alpha_only = (alpha_mask > 0.5) & (gt_mask < 0.5)
    gt_only = (gt_mask > 0.5) & (alpha_mask < 0.5)
    both = (alpha_mask > 0.5) & (gt_mask > 0.5)

    diff_img = torch.zeros(1, 3, height, width, device=device)
    diff_img[:, 0:1] = gt_only.float()  # Red = GT only (FN)
    diff_img[:, 2:3] = alpha_only.float()  # Blue = Alpha only (FP)
    diff_img[:, 1:2] = both.float() * 0.5  # Green = both (TP)

    return diff_img.squeeze(0).permute(1, 2, 0).cpu().numpy()


def add_labels_to_visualization(
    img: np.ndarray,
    labels: List[str],
    row_height: int,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Add text labels to visualization rows."""
    try:
        import cv2
        result = img.copy()
        for i, label in enumerate(labels):
            y_pos = i * row_height + 20
            cv2.putText(result, label, (5, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (255, 255, 255), 1)
        return result
    except ImportError:
        return img
