"""
Inference visualization utilities.
Creates GT vs Pred comparison grids and multi-elevation turntable grids.

Uses label bars from wandb_image_utils for consistent styling across
training (wandb) and inference visualizations.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from PIL import Image
from einops import rearrange

from mouse_extensions.utils.wandb_image_utils import make_label_bar


def _add_view_labels(row_np: np.ndarray, num_views: int, label_height: int = 18) -> np.ndarray:
    """Add per-view number labels below image row.

    Args:
        row_np: (H, V*W, 3) uint8 image row
        num_views: Number of views
        label_height: Height of label bar

    Returns:
        (H + label_height, V*W, 3) with view labels
    """
    h, total_w = row_np.shape[:2]
    view_w = total_w // num_views
    label_row = np.full((label_height, total_w, 3), 30, dtype=np.uint8)

    try:
        import cv2 as cv
        font = cv.FONT_HERSHEY_SIMPLEX
        for v in range(num_views):
            text = f"View {v}"
            (tw, th), _ = cv.getTextSize(text, font, 0.4, 1)
            x = v * view_w + (view_w - tw) // 2
            y = (label_height + th) // 2
            cv.putText(label_row, text, (x, y), font, 0.4, (200, 200, 200), 1, cv.LINE_AA)
    except ImportError:
        pass

    return np.concatenate([row_np, label_row], axis=0)


def save_comparison_grid(
    gt_images: torch.Tensor,
    pred_images: torch.Tensor,
    output_path: str,
    labels: bool = True,
    num_input_views: int = 0,
    step: int = 0,
) -> str:
    """
    Save GT (top) vs Pred (bottom) comparison grid with label bars.

    Args:
        gt_images: [B, V, C, H, W] or [V, C, H, W]
        pred_images: [V, C, H, W]
        output_path: Save path
        labels: Add row/view labels
        num_input_views: Number of input views (shown in label)
        step: Training/inference step (shown in label)

    Returns:
        Saved file path
    """
    if gt_images.dim() == 5:
        gt = gt_images[0].detach()
    else:
        gt = gt_images.detach()
    pred = pred_images.detach()

    nv = min(gt.size(0), pred.size(0))
    gt_row = rearrange(gt[:nv], "v c h w -> h (v w) c")
    pred_row = rearrange(pred[:nv], "v c h w -> h (v w) c")

    gt_np = (gt_row.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pred_np = (pred_row.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    if labels:
        row_w = gt_np.shape[1]
        input_info = f", {num_input_views} input" if num_input_views > 0 else ""
        step_info = f" | Step {step}" if step > 0 else ""

        gt_bar = make_label_bar(
            row_w,
            f"GT ({nv} views{input_info}){step_info}",
            text_color=(200, 255, 200),
        )
        pred_bar = make_label_bar(
            row_w,
            f"Pred ({nv} views){step_info}",
            text_color=(200, 200, 255),
        )
        # Add per-view labels
        gt_np = _add_view_labels(gt_np, nv)
        pred_np = _add_view_labels(pred_np, nv)

        comparison = np.concatenate([gt_bar, gt_np, pred_bar, pred_np], axis=0)
    else:
        comparison = np.concatenate([gt_np, pred_np], axis=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(comparison).save(output_path)
    return output_path


def save_multiview_turntable_grid(
    gaussians,
    output_path: str,
    elevations: List[float] = [0, 10, 20, 30],
    num_azimuth: int = 8,
    radius: float = 2.7,
    render_res: int = 512,
    gt_images: Optional[torch.Tensor] = None,
) -> str:
    """
    Save multi-elevation turntable grid with optional GT row.

    Args:
        gaussians: Gaussian model (from result.gaussians[0])
        output_path: Save path
        elevations: List of elevation angles
        num_azimuth: Views per elevation
        radius: Camera distance
        render_res: Render resolution
        gt_images: Optional GT images [B, V, C, H, W] to add as first row

    Returns:
        Saved file path
    """
    from mouse_extensions.visualization import render_turntable

    all_rows = []

    # Add GT row if provided
    if gt_images is not None:
        gt = gt_images[0].detach() if gt_images.dim() == 5 else gt_images.detach()
        nv = min(gt.size(0), num_azimuth)
        # Pad or truncate to match num_azimuth
        if nv < num_azimuth:
            pad = torch.zeros(num_azimuth - nv, *gt.shape[1:], device=gt.device)
            gt = torch.cat([gt[:nv], pad], dim=0)
        else:
            gt = gt[:num_azimuth]
        gt_row = rearrange(gt, "v c h w -> h (v w) c")
        gt_row = (gt_row.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        all_rows.append(gt_row)

    # Render turntable at each elevation
    for elev in elevations:
        concat_img = render_turntable(
            gaussians,
            rendering_resolution=render_res,
            num_views=num_azimuth,
            elevation=elev,
            radius=radius,
        )
        if concat_img.dtype != np.uint8:
            concat_img = (concat_img * 255).clip(0, 255).astype(np.uint8)
        all_rows.append(concat_img)

    grid = np.concatenate(all_rows, axis=0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    return output_path


def _make_label_col(h: int, w: int, text: str) -> torch.Tensor:
    """Create a label column [H, W, C] with text. Falls back to solid color."""
    try:
        from PIL import ImageDraw, ImageFont
        img = Image.new("RGB", (w, h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(255, 255, 255), font=font)
        return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
    except Exception:
        col = torch.zeros(h, w, 3)
        col[:, :, :] = 0.12  # dark gray
        return col
