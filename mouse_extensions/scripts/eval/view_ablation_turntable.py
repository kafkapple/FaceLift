"""View ablation turntable comparison.

Renders a single frame's Gaussians from all 1v~6v models on a shared
360-degree turntable orbit, arranged as a 1×6 horizontal grid video.

Usage (run on gpu03):
    # PoC (60 turntable views, 256px)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_turntable \\
        --cell-size 256 --turntable-views 60 --fps 15

    # Full quality (120 views, 512px)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_turntable \\
        --cell-size 512 --turntable-views 120 --fps 30
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts
from mouse_extensions.inference.gslrm_pipeline import load_sample_data
from mouse_extensions.scripts.eval.ablation_comparison import (
    VIEW_ABLATION_EXPERIMENTS,
    _load_models,
)
from mouse_extensions.scripts.eval.fl_gt_view_comparison import (
    add_label,
    render_gaussian_at_view,
)
from mouse_extensions.visualization.camera_utils import get_turntable_cameras

_WHITE = (255, 255, 255)


def run_turntable_comparison(
    m5_dir: str,
    output_dir: str,
    frame_idx: int = 3240,
    cell_size: int = 512,
    turntable_views: int = 120,
    fps: int = 30,
    elevation: float = 20.0,
    n_filter: int = 2,
    device: str = "cuda",
) -> None:
    """Render turntable orbit for a single frame across all view models."""
    experiments = VIEW_ABLATION_EXPERIMENTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== View Ablation Turntable Comparison ===")
    print(f"Frame: {frame_idx}, Elevation: {elevation}°, Views: {turntable_views}")
    print(f"Loading 6 models...")
    models = _load_models(experiments, device)
    active = [(exp, m) for exp, m in zip(experiments, models) if m is not None]
    if not active:
        print("[ERROR] No valid checkpoints. Aborting.")
        return

    n_cols = len(active)

    # Load sample data
    fd = Path(m5_dir) / f"{frame_idx:06d}"
    if not fd.exists():
        print(f"[ERROR] Frame dir not found: {fd}")
        return
    imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

    # Pre-compute Gaussians for all models
    print("Pre-computing Gaussians for all models...")
    gaussians_list = []
    vis_masks = []
    for exp, model in active:
        n_in = exp["n_views"]
        result = model.predict(imgs[:, :n_in], c2ws[:, :n_in], fxfys[:, :n_in], idx)
        gs = result.gaussians[0]
        gaussians_list.append(gs)

        xyz = gs.get_xyz.detach().cpu().numpy()
        vc = compute_visibility_counts(xyz, str(fd), n_views=6)
        vis_masks.append(vc >= n_filter)
        print(f"  {exp['label']}: {xyz.shape[0]} Gaussians")

    # Generate turntable cameras
    w, h, n_cams, fxfycxcy, orbit_c2ws = get_turntable_cameras(
        num_views=turntable_views,
        w=cell_size,
        h=cell_size,
        elevation=elevation,
    )
    # fxfycxcy: [n_cams, 4], orbit_c2ws: [n_cams, 4, 4]

    # 2×3 layout for 6 models
    grid_cols = min(n_cols, 3)
    grid_rows = (n_cols + grid_cols - 1) // grid_cols
    W_total = grid_cols * cell_size
    H_total = grid_rows * cell_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(out / "view_ablation_turntable.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W_total, H_total))

    print(f"Rendering {turntable_views} orbit views × {n_cols} models  →  {out_path}")
    print(f"Grid: {W_total}×{H_total} px  (cell={cell_size})")

    first_frame_saved = False

    for vi in range(turntable_views):
        cam = (orbit_c2ws[vi], fxfycxcy[vi])
        cells = []

        for i, (exp, model) in enumerate(active):
            try:
                cell = render_gaussian_at_view(
                    gaussians_list[i], cam, resolution=cell_size,
                    device=device, vis_mask=vis_masks[i],
                )
            except Exception as e:
                print(f"  [warn] render failed {exp['label']} view {vi}: {e}")
                cell = np.full((cell_size, cell_size, 3), 80, dtype=np.uint8)

            cell = add_label(cell, f"{exp['n_views']}v", color=_WHITE, outline=True)
            cells.append(cell)

        # Arrange into 2×3 grid
        rows = []
        for r in range(grid_rows):
            row_cells = cells[r * grid_cols:(r + 1) * grid_cols]
            while len(row_cells) < grid_cols:
                row_cells.append(np.full((cell_size, cell_size, 3), 255, dtype=np.uint8))
            rows.append(np.hstack(row_cells))
        frame_grid = np.vstack(rows)
        writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        if not first_frame_saved:
            png_path = str(out / "turntable_first_frame.png")
            cv2.imwrite(png_path, cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))
            first_frame_saved = True

        if vi % 30 == 0:
            print(f"  orbit view {vi}/{turntable_views}")

    writer.release()
    print(f"\nDone. Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View ablation turntable comparison")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--output-dir",
                        default="outputs/viz/comparison/mouse/view_ablation_turntable")
    parser.add_argument("--frame-idx", type=int, default=3240)
    parser.add_argument("--cell-size", type=int, default=512)
    parser.add_argument("--turntable-views", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--n-filter", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_turntable_comparison(
        m5_dir=args.m5_dir,
        output_dir=args.output_dir,
        frame_idx=args.frame_idx,
        cell_size=args.cell_size,
        turntable_views=args.turntable_views,
        fps=args.fps,
        elevation=args.elevation,
        n_filter=args.n_filter,
        device=args.device,
    )


if __name__ == "__main__":
    main()
