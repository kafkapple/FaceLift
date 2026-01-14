"""
Visualization Extensions

Additional visualizations for mask comparison and threshold analysis.
"""

import torch
import numpy as np
from typing import Optional, List


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
    Create threshold comparison visualization.

    Shows masks at different thresholds for both alpha and RGB methods.
    - Row 1: Alpha thresholds
    - Row 2: RGB thresholds
    - Row 3: GT vs Alpha(0.5) vs RGB(0.1) vs Diff

    Args:
        rendering: Rendered RGB images
        rendered_alpha: Rendered alpha from Gaussians
        gt_mask: Ground truth mask (optional)
        num_views: Number of views (v)
        height: Image height
        alpha_thresholds: Thresholds for alpha mask
        rgb_thresholds: Thresholds for RGB mask

    Returns:
        np.ndarray: Visualization image (H, W, 3) uint8
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
    bg_color = torch.tensor([0.8, 0.2, 0.2], device=device).view(1, 3, 1, 1)  # red

    rows = []

    # Row 1: Alpha threshold masks (first view with different thresholds)
    alpha_row = []
    for thresh in alpha_thresholds:
        alpha_mask = (alpha_sample[0:1] > thresh).float()
        mask_rgb = alpha_mask.expand(-1, 3, -1, -1)
        overlay = render_sample[0:1] * 0.5 + mask_rgb * fg_color + (1 - mask_rgb) * bg_color * 0.5
        overlay = overlay.squeeze(0).permute(1, 2, 0).cpu().numpy()
        alpha_row.append(overlay)
    alpha_row = np.hstack(alpha_row)
    rows.append(alpha_row)

    # Row 2: RGB threshold masks
    rgb_row = []
    color_dist = (render_sample[0:1] - 1.0).abs().mean(dim=1, keepdim=True)
    for thresh in rgb_thresholds:
        rgb_mask = (color_dist > thresh).float()
        mask_rgb = rgb_mask.expand(-1, 3, -1, -1)
        overlay = render_sample[0:1] * 0.5 + mask_rgb * fg_color + (1 - mask_rgb) * bg_color * 0.5
        overlay = overlay.squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb_row.append(overlay)
    rgb_row = np.hstack(rgb_row)
    rows.append(rgb_row)

    # Row 3: GT vs Alpha(0.5) vs RGB(0.1) vs Diff
    if gt_mask_sample is not None:
        comparison_row = []

        # GT mask
        gt_binary = (gt_mask_sample[0:1] > 0.5).float()
        gt_overlay = _create_mask_overlay(render_sample[0:1], gt_binary, fg_color, bg_color)
        comparison_row.append(gt_overlay)

        # Alpha mask (0.5)
        alpha_binary = (alpha_sample[0:1] > 0.5).float()
        alpha_overlay = _create_mask_overlay(render_sample[0:1], alpha_binary, fg_color, bg_color)
        comparison_row.append(alpha_overlay)

        # RGB mask (0.1)
        rgb_binary = (color_dist > 0.1).float()
        rgb_overlay = _create_mask_overlay(render_sample[0:1], rgb_binary, fg_color, bg_color)
        comparison_row.append(rgb_overlay)

        # Diff: Alpha - GT
        diff_img = _create_diff_visualization(alpha_binary, gt_binary, height, render_sample.shape[-1], device)
        comparison_row.append(diff_img)

        comparison_row = np.hstack(comparison_row)
        rows.append(comparison_row)

    # Ensure all rows have same width
    max_width = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_width:
            pad = np.ones((r.shape[0], max_width - r.shape[1], 3)) * 0.5
            r = np.hstack([r, pad])
        padded_rows.append(r)

    result = np.vstack(padded_rows)
    return (result * 255).clip(0, 255).astype(np.uint8)


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
    diff_img[:, 0:1] = gt_only.float()  # Red = GT only
    diff_img[:, 2:3] = alpha_only.float()  # Blue = Alpha only
    diff_img[:, 1:2] = both.float() * 0.5  # Green = both

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
