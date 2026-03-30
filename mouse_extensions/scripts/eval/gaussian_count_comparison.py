"""Gaussian count comparison: FaceLift at different thresholds vs PoseSplatter.

Rows = conditions (GT, FL-all, FL-12K, FL-8.5K, PS, ...), Cols = [Render | Diff].

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.gaussian_count_comparison \
        --config configs/mouse/eval/gaussian_count_comparison.yaml \
        --output-dir outputs/viz/comparison/mouse/gaussian_count
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts
from mouse_extensions.scripts.eval.fl_gt_view_comparison import (
    add_label, get_novel_camera, render_gaussian_at_view, load_gt_image,
)


# ---------------------------------------------------------------------------
# Gaussian selection
# ---------------------------------------------------------------------------

def select_gaussians(
    gaussians, vis_mask: np.ndarray,
    method: str = "topk", k: int = 8500,
    opacity_thresh: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Select Gaussians by count or opacity threshold.

    Returns:
        (mask, count): boolean mask and number of selected Gaussians.
    """
    opacity = gaussians.get_opacity.squeeze(-1).detach().cpu().numpy()

    if method == "topk":
        # Among visible, take top-k by opacity
        vis_opacities = np.where(vis_mask, opacity, -1.0)
        topk_idx = np.argsort(-vis_opacities)[:k]
        mask = np.zeros(len(opacity), dtype=bool)
        mask[topk_idx] = True
        # Only keep those that were actually visible
        mask &= vis_mask
        return mask, int(mask.sum())

    elif method == "threshold":
        assert opacity_thresh is not None, "opacity_thresh required for threshold method"
        mask = vis_mask & (opacity > opacity_thresh)
        return mask, int(mask.sum())

    else:
        raise ValueError(f"Unknown method: {method}")


def build_condition_mask(
    gaussians, vis_mask: np.ndarray, condition: Dict,
) -> Tuple[np.ndarray, int, str]:
    """Build a selection mask from a single condition dict.

    Returns:
        (mask, count, label): boolean mask, count, display label.
    """
    name = condition["name"]
    max_gs = condition.get("max_gaussians")
    op_thresh = condition.get("opacity_thresh")

    if max_gs is not None:
        mask, cnt = select_gaussians(gaussians, vis_mask, method="topk", k=max_gs)
    elif op_thresh is not None:
        mask, cnt = select_gaussians(
            gaussians, vis_mask, method="threshold", opacity_thresh=op_thresh,
        )
    else:
        # No limit — use full vis_mask
        mask, cnt = vis_mask.copy(), int(vis_mask.sum())

    label = f"{name} ({cnt:,}G)"
    return mask, cnt, label


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------

def make_diff_image(render: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Absolute pixel difference, amplified for visibility (uint8)."""
    diff = np.abs(render.astype(np.float32) - gt.astype(np.float32))
    diff = np.clip(diff * 3.0, 0, 255).astype(np.uint8)  # 3x amplify
    return diff


def build_grid(
    rows: List[Tuple[str, np.ndarray]],
    gt_img: np.ndarray,
    cell_size: int,
    show_diff: bool = True,
) -> np.ndarray:
    """Build a vertical grid: each row = [label | render | diff-from-GT].

    Args:
        rows: list of (label, render_uint8) pairs.
        gt_img: GT reference image (uint8).
        cell_size: render resolution.
        show_diff: whether to append a diff column.
    """
    label_w = 180  # px for text label column
    n_cols = 2 if show_diff else 1
    grid_w = label_w + cell_size * n_cols
    grid_h = cell_size * len(rows)
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark background

    for i, (label, render) in enumerate(rows):
        y0 = i * cell_size
        # Label
        cv2.putText(
            grid, label, (6, y0 + cell_size // 2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA,
        )
        # Render
        x0 = label_w
        grid[y0:y0 + cell_size, x0:x0 + cell_size] = render
        # Diff
        if show_diff:
            diff = make_diff_image(render, gt_img)
            x1 = label_w + cell_size
            grid[y0:y0 + cell_size, x1:x1 + cell_size] = diff

    return grid


# ---------------------------------------------------------------------------
# PS image loading
# ---------------------------------------------------------------------------

def load_ps_render(ps_dir: str, frame_idx: int, view_idx: int,
                   resolution: int = 512) -> Optional[np.ndarray]:
    """Load pre-rendered PoseSplatter image.

    Tries common naming patterns: {frame:06d}_cam{view:03d}.png, etc.
    """
    ps_path = Path(ps_dir)
    candidates = [
        ps_path / f"{frame_idx:06d}" / f"cam_{view_idx:03d}.png",
        ps_path / f"{frame_idx:06d}_cam{view_idx:03d}.png",
        ps_path / f"frame_{frame_idx:06d}" / f"view_{view_idx}.png",
        ps_path / f"{frame_idx:06d}" / f"view_{view_idx}.png",
    ]
    for p in candidates:
        if p.exists():
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != resolution or img.shape[1] != resolution:
                img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
            return img
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def process_frame(
    model: GSLRMInference, cfg: Dict, frame_idx: int, view_idx: int,
    m5_dir: str, cell_size: int, device: str,
) -> Optional[np.ndarray]:
    """Render one frame×view grid across all conditions."""
    fd = Path(m5_dir) / f"{frame_idx:06d}"
    if not fd.exists():
        print(f"  [SKIP] frame dir {fd} not found")
        return None

    res = cfg["eval"].get("resolution", 512)

    # Infer Gaussians
    imgs, c2ws, fxfy, idx = load_sample_data(str(fd), image_size=res, device=device)
    result = model.predict(imgs, c2ws, fxfy, idx)
    gaussians = result.gaussians[0]

    # Visibility mask
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    vc = compute_visibility_counts(xyz, str(fd), n_views=6)
    vis_mask = vc >= 2

    # GT camera for this view
    from mouse_extensions.behavior.cinematic_sequence import gt_camera_c2w, gt_camera_fxfycxcy
    c2w_gt = gt_camera_c2w(str(fd), view_idx)
    fxfycxcy_gt = gt_camera_fxfycxcy(str(fd), view_idx)
    camera = (c2w_gt, fxfycxcy_gt)

    # GT image
    gt_img = load_gt_image(str(fd), view_idx, resolution=res)
    if gt_img.shape[0] != cell_size:
        gt_img = cv2.resize(gt_img, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)

    rows: List[Tuple[str, np.ndarray]] = []

    # GT row
    rows.append(("GT", gt_img))

    # FL conditions
    for cond in cfg["facelift"]["conditions"]:
        mask, cnt, label = build_condition_mask(gaussians, vis_mask, cond)
        render = render_gaussian_at_view(gaussians, camera, resolution=res, vis_mask=mask)
        if render.shape[0] != cell_size:
            render = cv2.resize(render, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
        rows.append((label, render))

    # PS row (if available)
    ps_dir = cfg.get("posesplatter", {}).get("renders_dir")
    if ps_dir:
        ps_img = load_ps_render(ps_dir, frame_idx, view_idx, resolution=res)
        if ps_img is not None:
            if ps_img.shape[0] != cell_size:
                ps_img = cv2.resize(ps_img, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
            rows.append(("PS (8.5K)", ps_img))

    # Cleanup
    del result, imgs, c2ws, fxfy, gaussians
    torch.cuda.empty_cache()

    return build_grid(rows, gt_img, cell_size, show_diff=True)


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian count comparison grid: FL conditions vs PS",
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output-dir", default="outputs/viz/comparison/mouse/gaussian_count")
    parser.add_argument("--cell-size", type=int, default=384, help="Per-cell render size in grid")
    parser.add_argument("--fps", type=int, default=15, help="MP4 frame rate")
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 generation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fl_cfg = cfg["facelift"]
    eval_cfg = cfg["eval"]
    frames = eval_cfg["frames"]
    views = eval_cfg.get("views", [0])
    m5_dir = fl_cfg["m5_dir"]

    # Load model
    model_cfg = fl_cfg.get("model_config", "configs/base/gslrm_mouse.yaml")
    print(f"Loading GS-LRM from {fl_cfg['checkpoint']} ...")
    model = GSLRMInference(model_cfg, fl_cfg["checkpoint"], device=args.device)

    # Process each frame × view
    video_writers: Dict[int, cv2.VideoWriter] = {}

    for fi in frames:
        for vi in views:
            print(f"[Frame {fi}, View {vi}]")
            grid = process_frame(
                model, cfg, fi, vi, m5_dir, args.cell_size, args.device,
            )
            if grid is None:
                continue

            # Save PNG
            png_path = out / f"f{fi:06d}_v{vi}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"  Saved {png_path}")

            # Accumulate video frames (one video per view)
            if not args.no_video:
                if vi not in video_writers:
                    h, w = grid.shape[:2]
                    vpath = out / f"comparison_v{vi}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writers[vi] = cv2.VideoWriter(str(vpath), fourcc, args.fps, (w, h))
                video_writers[vi].write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # Finalize videos
    for vi, writer in video_writers.items():
        writer.release()
        print(f"Saved video: {out / f'comparison_v{vi}.mp4'}")

    print(f"\nDone. Output: {out}")


if __name__ == "__main__":
    main()
