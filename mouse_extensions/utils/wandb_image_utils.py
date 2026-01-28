"""
WandB Image Utilities for MVDiffusion validation logging.

Provides helper functions for creating comparison grids with
input view borders and label bars.
"""

import numpy as np
import cv2 as cv
from typing import Optional, Tuple

import torch


def add_border(img_np: np.ndarray, color: Tuple[int, int, int], width: int = 4) -> np.ndarray:
    """Add colored border to image (H,W,C uint8)."""
    bordered = img_np.copy()
    bordered[:width, :] = color
    bordered[-width:, :] = color
    bordered[:, :width] = color
    bordered[:, -width:] = color
    return bordered


def make_label_bar(
    width: int,
    text: str,
    height: int = 22,
    font_scale: float = 0.5,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Create a label bar image with centered text."""
    bar = np.full((height, width, 3), bg_color, dtype=np.uint8)
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv.getTextSize(text, font, font_scale, 1)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv.putText(bar, text, (x, y), font, font_scale, text_color, 1, cv.LINE_AA)
    return bar


def views_to_row(
    views: torch.Tensor,
    ref_idx: int = 0,
    border_color: Tuple[int, int, int] = (0, 255, 0),
    border_width: int = 4,
) -> np.ndarray:
    """Convert [V, C, H, W] tensor to horizontal row with input view bordered.

    Args:
        views: Tensor of shape [V, C, H, W] in [0, 1] range
        ref_idx: Index of reference (input) view to highlight
        border_color: BGR color for input view border
        border_width: Border width in pixels

    Returns:
        np.ndarray: (H, V*W, 3) uint8 image
    """
    panels = []
    for v in range(views.shape[0]):
        vp = (views[v].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        if v == ref_idx:
            vp = add_border(vp, border_color, width=border_width)
        panels.append(vp)
    return np.concatenate(panels, axis=1)


def make_gt_vs_pred_comparison(
    gt_row: np.ndarray,
    pred_row: np.ndarray,
    ref_idx: int = 0,
    guidance_scale: float = 3.0,
) -> np.ndarray:
    """Create a 2-row GT vs Pred comparison image with label bars.

    Args:
        gt_row: (H, V*W, 3) uint8 ground truth row
        pred_row: (H, V*W, 3) uint8 prediction row
        ref_idx: Reference view index (shown in label)
        guidance_scale: CFG scale (shown in label)

    Returns:
        np.ndarray: (2*H + 2*label_h, V*W, 3) comparison image
    """
    row_w = gt_row.shape[1]
    gt_bar = make_label_bar(row_w, f"GT  (green border = input view {ref_idx})",
                            text_color=(200, 255, 200))
    pred_bar = make_label_bar(row_w, f"Pred  (cfg={guidance_scale:.1f})",
                              text_color=(200, 200, 255))
    return np.concatenate([gt_bar, gt_row, pred_bar, pred_row], axis=0)
