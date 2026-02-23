"""Data loaders for metrics JSONs and images."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


ALL_METRICS = [
    "psnr_gt_masked",
    "psnr_intersection",
    "psnr_whole",
    "iou",
    "coverage",
    "ssim_gt_masked",
    "color_bias",
]


def load_fair_eval(json_path: str) -> dict:
    """Load fair eval JSON file."""
    with open(json_path) as f:
        return json.load(f)


def get_metrics(data: dict, view_key: str = None) -> dict:
    """Extract metrics from fair eval data.

    Args:
        data: Fair eval JSON data.
        view_key: If provided, extract from per_view[view_key].
                  If None, extract from overall.

    Returns:
        Dict of {metric_name: {mean, std}}.
    """
    if view_key:
        source = data.get("per_view", {}).get(view_key, {})
    else:
        source = data.get("overall", {})

    result = {}
    for m in ALL_METRICS:
        if m in source:
            val = source[m]
            if isinstance(val, dict):
                result[m] = {"mean": val.get("mean"), "std": val.get("std")}
            else:
                result[m] = {"mean": val, "std": None}
    return result


def load_image(path: str) -> np.ndarray | None:
    """Load image as RGB numpy array, compositing RGBA on white."""
    p = Path(path)
    if not p.exists():
        return None
    img = Image.open(p)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return np.array(bg)
    return np.array(img.convert("RGB"))


def load_render(render_dir: str, pattern: str,
                frame_id: int, view_id: int) -> np.ndarray | None:
    """Load a rendered image using pattern substitution."""
    fid = f"{frame_id:06d}"
    path = Path(render_dir) / pattern.format(fid=fid, vid=view_id)
    return load_image(str(path))


def load_gt(gt_dir: str, gt_pattern: str,
            frame_id: int, view_id: int) -> np.ndarray | None:
    """Load a GT image using pattern substitution."""
    fid = f"{frame_id:06d}"
    path = Path(gt_dir) / gt_pattern.format(fid=fid, vid=view_id)
    return load_image(str(path))
