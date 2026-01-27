"""
Inference visualization utilities.
Creates GT vs Pred comparison grids and multi-elevation turntable grids.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from PIL import Image
from einops import rearrange


def save_comparison_grid(
    gt_images: torch.Tensor,
    pred_images: torch.Tensor,
    output_path: str,
    labels: bool = True,
) -> str:
    """
    Save GT (top) vs Pred (bottom) comparison grid.

    Args:
        gt_images: [B, V, C, H, W] or [V, C, H, W]
        pred_images: [V, C, H, W]
        output_path: Save path
        labels: Add row labels

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

    if labels:
        # Add label column
        h = gt_row.size(0)
        w_label = 40
        gt_label = _make_label_col(h, w_label, "GT")
        pred_label = _make_label_col(h, w_label, "Pred")
        gt_row = torch.cat([gt_label.to(gt_row.device), gt_row], dim=1)
        pred_row = torch.cat([pred_label.to(pred_row.device), pred_row], dim=1)

    comparison = torch.cat([gt_row, pred_row], dim=0)
    comparison = (comparison.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

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
    from gslrm.model.gaussians_renderer import render_turntable

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
