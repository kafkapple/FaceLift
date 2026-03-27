"""Multi-view grid comparison: before/after orientation filter.

Generates a grid video with 2 rows (Before/After) × 4 columns (view angles),
showing the effect of the corrected orientation filter across multiple frames.

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.filter_grid_comparison \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --output-dir outputs/analysis/mouse/filtering/filter_comparison \
        --frame-range 3240:3280
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.model.orientation_filter import suppress_z_aligned_flat
from mouse_extensions.scripts.eval.fl_gt_view_comparison import (
    add_label,
    get_novel_camera,
    render_gaussian_at_view,
)

FILTER_PARAMS = dict(
    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
)

# 4 representative angles including the artifact-prone bottom view
VIEW_ANGLES = [
    {"name": "Bottom (-80°)", "elev": -80, "azim": 0},
    {"name": "Side-low (-30°)", "elev": -30, "azim": 90},
    {"name": "Front (0°)", "elev": 0, "azim": 0},
    {"name": "Top (+60°)", "elev": 60, "azim": 0},
]


def main():
    parser = argparse.ArgumentParser(description="Filter grid comparison video")
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--checkpoint",
                        default="/node_data/joon/checkpoints/FaceLift/gslrm/"
                                "base_uniform_v2_6view_v2/best_psnr.pt")
    parser.add_argument("--frame-range", default="3240:3280")
    parser.add_argument("--cell-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/filtering/filter_comparison")
    # Filter settings
    parser.add_argument("--z-align-thresh", type=float, default=0.15)
    parser.add_argument("--attenuation", type=float, default=0.0,
                        help="0.0 = full removal for clear comparison")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = GSLRMInference(config_path=args.config, checkpoint_path=args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-compute cameras
    cams = {v["name"]: get_novel_camera(v["elev"], v["azim"], resolution=args.cell_size)
            for v in VIEW_ANGLES}

    n_views = len(VIEW_ANGLES)
    W_total = n_views * args.cell_size
    H_total = 2 * args.cell_size  # 2 rows: Before / After
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = str(out / "filter_comparison_grid.mp4")
    writer = cv2.VideoWriter(vid_path, fourcc, args.fps, (W_total, H_total))

    start, end = map(int, args.frame_range.split(":"))
    frame_indices = list(range(start, end))
    print(f"Rendering {len(frame_indices)} frames × 2 conditions × {n_views} views")
    print(f"Grid: {W_total}×{H_total} px (cell={args.cell_size})")

    for fi in frame_indices:
        fd = Path(args.m5_dir) / f"{fi:06d}"
        if not fd.exists():
            continue

        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

        rows = []
        for row_label, apply_filter in [("Before", False), ("After (low-Z filter)", True)]:
            result = model.predict(imgs, c2ws, fxfys, idx)
            gs = result.gaussians[0]
            gs.apply_all_filters(**FILTER_PARAMS)

            if apply_filter:
                stats = suppress_z_aligned_flat(
                    gs, z_align_thresh=args.z_align_thresh,
                    attenuation=args.attenuation, mode="prune" if args.attenuation == 0 else "attenuate",
                )

            cells = []
            for v in VIEW_ANGLES:
                cam = cams[v["name"]]
                try:
                    cell = render_gaussian_at_view(
                        gs, cam, resolution=args.cell_size, device=device)
                except Exception as e:
                    cell = np.full((args.cell_size, args.cell_size, 3), 80, dtype=np.uint8)
                cell = add_label(cell, v["name"], row_label)
                cells.append(cell)
            rows.append(np.hstack(cells))

        frame_grid = np.vstack(rows)
        cv2.putText(frame_grid, f"Frame {fi:04d}",
                     (8, H_total - 8), cv2.FONT_HERSHEY_SIMPLEX,
                     0.4, (255, 255, 0), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        # Save first and last frame as PNG for quick review
        if fi == frame_indices[0] or fi == frame_indices[-1]:
            png_path = out / f"grid_{fi:06d}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))
            print(f"  Saved {png_path}")

        if fi % 10 == 0:
            n_s = stats["n_suppressed"] if apply_filter else 0
            print(f"  frame {fi}/{end-1} (suppressed: {n_s})")

    writer.release()
    print(f"\nDone. Saved: {vid_path}")


if __name__ == "__main__":
    main()
