"""
Error visualization and annotation utilities.

Provides:
- Error scale annotation with colorbar
- Predicted mask computation for visualization
"""

from typing import Optional
import numpy as np
import cv2
import torch


def add_error_scale_annotation(
    visual_np: np.ndarray,
    error_stats: dict,
    row_height: int,
    num_rows: int,
) -> np.ndarray:
    """
    Add error scale range annotation with vertical colorbar to the visualization image.

    Adds:
    - Vertical colorbar on the right side of error row
    - Range labels at top (0.3/red) and bottom (0.0/blue)
    - Error statistics text

    Args:
        visual_np: Visualization image as numpy array [H, W, 3]
        error_stats: Dict with min, max, mean, fg_min, fg_max, fg_mean
        row_height: Height of each row in pixels
        num_rows: Total number of rows

    Returns:
        Annotated visualization image
    """
    # Error row is the last row
    error_row_start = (num_rows - 1) * row_height
    img_height, img_width = visual_np.shape[:2]

    # Create annotation text
    fg_min = error_stats.get("fg_min", error_stats.get("min", 0))
    fg_max = error_stats.get("fg_max", error_stats.get("max", 0.3))
    fg_mean = error_stats.get("fg_mean", error_stats.get("mean", 0.1))

    # --- Add Vertical Colorbar ---
    colorbar_width = 20
    colorbar_height = row_height - 40  # Leave margin for labels
    colorbar_x = img_width - colorbar_width - 50  # Right side with margin for labels
    colorbar_y = error_row_start + 20  # Top margin

    # Create gradient colorbar (top=red/high, bottom=blue/low)
    for i in range(colorbar_height):
        # Normalized value: 0 at bottom, 1 at top
        t = 1.0 - (i / colorbar_height)  # Invert: top=1, bottom=0

        # Same color mapping as error heatmap
        r = int(np.clip(t, 0, 1) * 255)
        g = int(np.clip(1 - abs(t) * 2, 0, 1) * 255)
        b = int(np.clip(1 - t, 0, 1) * 255)

        y_pos = colorbar_y + i
        if 0 <= y_pos < img_height:
            visual_np[y_pos, colorbar_x : colorbar_x + colorbar_width] = [r, g, b]

    # Add colorbar border
    cv2.rectangle(
        visual_np,
        (colorbar_x - 1, colorbar_y - 1),
        (colorbar_x + colorbar_width, colorbar_y + colorbar_height),
        (255, 255, 255),
        1,
    )

    # --- Add Range Labels ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (255, 255, 255)
    thickness = 1

    # Top label (0.3 = max error, red)
    top_label = "0.3"
    cv2.putText(
        visual_np,
        top_label,
        (colorbar_x + colorbar_width + 5, colorbar_y + 12),
        font,
        font_scale,
        font_color,
        thickness,
        cv2.LINE_AA,
    )

    # Bottom label (0.0 = min error, blue)
    bottom_label = "0.0"
    cv2.putText(
        visual_np,
        bottom_label,
        (colorbar_x + colorbar_width + 5, colorbar_y + colorbar_height - 2),
        font,
        font_scale,
        font_color,
        thickness,
        cv2.LINE_AA,
    )

    # Middle label (mean indicator line)
    if fg_mean > 0 and fg_mean < 0.3:
        mean_normalized = fg_mean / 0.3
        mean_y = colorbar_y + int((1 - mean_normalized) * colorbar_height)
        # Draw horizontal line at mean position
        cv2.line(
            visual_np,
            (colorbar_x - 5, mean_y),
            (colorbar_x + colorbar_width + 2, mean_y),
            (255, 255, 0),
            1,
        )  # Yellow line for mean
        mean_label = f"{fg_mean:.3f}"
        cv2.putText(
            visual_np,
            mean_label,
            (colorbar_x - 45, mean_y + 4),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    # --- Add Statistics Text (bottom-left) ---
    text_lines = [
        f"Error: [{fg_min:.4f}, {fg_max:.4f}]",
        f"Mean: {fg_mean:.4f}",
    ]

    bg_color = (0, 0, 0)
    line_height = 16
    x_start = 5
    y_start = error_row_start + 20

    for i, text in enumerate(text_lines):
        y = y_start + i * line_height
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            visual_np,
            (x_start - 2, y - text_h - 2),
            (x_start + text_w + 2, y + 4),
            bg_color,
            -1,
        )
        cv2.putText(
            visual_np,
            text,
            (x_start, y),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )

    return visual_np


def compute_pred_mask_for_visualization(
    rendering: torch.Tensor,
    rendered_alpha: Optional[torch.Tensor],
    config,
    channel_dim: int = 1,
) -> Optional[torch.Tensor]:
    """
    Compute predicted mask based on mask_mode configuration.

    Args:
        rendering: Rendered RGB tensor [..., 3, H, W] or [..., H, W, 3]
        rendered_alpha: Rendered alpha tensor [..., 1, H, W] or None
        config: Training config with losses.mask_mode
        channel_dim: Dimension of channel axis (default 1 for [B, 3, H, W])

    Returns:
        pred_mask: Binary mask tensor [..., 1, H, W] or None

    Mask Modes:
        - "none" / None: No masking, returns None
        - "alpha": Use rendered alpha with threshold
        - "rgb_pred": Detect foreground by color distance from white
        - "gt": Ground truth mask (handled externally, returns None here)
    """
    mask_mode = config.training.losses.get("mask_mode", None)

    # No masking
    if mask_mode is None or mask_mode == "none":
        return None

    # Alpha-based masking
    if mask_mode == "alpha":
        if rendered_alpha is None:
            # Fallback to rgb_pred if alpha not available
            mask_mode = "rgb_pred"
        else:
            alpha_threshold = config.training.losses.get("alpha_mask_threshold", 0.5)
            return (rendered_alpha > alpha_threshold).float()

    # RGB-based detection (for rgb_pred mode or alpha fallback)
    if mask_mode == "rgb_pred":
        pred_threshold = config.training.losses.get("pred_mask_threshold", 0.1)
        color_distance = (rendering - 1.0).abs().mean(dim=channel_dim, keepdim=True)
        return (color_distance > pred_threshold).float()

    # GT mask is provided externally
    if mask_mode == "gt":
        return None

    # Unknown mode - return None
    return None
