"""View ablation 36-view grid image generator.

For a single temporal frame, renders each view-count model (1v~6v) from 36
turntable angles (6×6 sub-grid), then tiles them horizontally.

Output: one wide image per model, or a combined mega-grid.

Usage (run on gpu03):
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_grid_image \\
        --cell-size 128 --grid-views 36

    # Larger cells for detail
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_grid_image \\
        --cell-size 256 --grid-views 36
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

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


def run_grid_image(
    m5_dir: str,
    output_dir: str,
    frame_idx: int = 3240,
    cell_size: int = 128,
    grid_views: int = 36,
    elevation: float = 20.0,
    n_filter: int = 2,
    device: str = "cuda",
) -> None:
    """Render 36-view grid images for all view-count models."""
    experiments = VIEW_ABLATION_EXPERIMENTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== View Ablation 36-View Grid Image ===")
    print(f"Frame: {frame_idx}, Grid: {grid_views} views, Cell: {cell_size}px")
    print(f"Loading 6 models...")
    models = _load_models(experiments, device)
    active = [(exp, m) for exp, m in zip(experiments, models) if m is not None]
    if not active:
        print("[ERROR] No valid checkpoints. Aborting.")
        return

    # Load sample data
    fd = Path(m5_dir) / f"{frame_idx:06d}"
    if not fd.exists():
        print(f"[ERROR] Frame dir not found: {fd}")
        return
    imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

    # Generate turntable cameras
    w, h, n_cams, fxfycxcy, orbit_c2ws = get_turntable_cameras(
        num_views=grid_views,
        w=cell_size, h=cell_size,
        elevation=elevation,
    )

    grid_cols = int(math.ceil(math.sqrt(grid_views)))
    grid_rows = int(math.ceil(grid_views / grid_cols))

    model_grids = []

    for exp, model in active:
        n_in = exp["n_views"]
        print(f"  Rendering {exp['label']}...")

        result = model.predict(imgs[:, :n_in], c2ws[:, :n_in], fxfys[:, :n_in], idx)
        gaussians = result.gaussians[0]

        xyz = gaussians.get_xyz.detach().cpu().numpy()
        vc = compute_visibility_counts(xyz, str(fd), n_views=6)
        vis_mask = vc >= n_filter

        cells = []
        for vi in range(grid_views):
            cam = (orbit_c2ws[vi], fxfycxcy[vi])
            try:
                cell = render_gaussian_at_view(
                    gaussians, cam, resolution=cell_size,
                    device=device, vis_mask=vis_mask,
                )
            except Exception:
                cell = np.full((cell_size, cell_size, 3), 80, dtype=np.uint8)
            cells.append(cell)

        # Arrange into grid_rows × grid_cols
        rows = []
        for r in range(grid_rows):
            row_cells = []
            for c in range(grid_cols):
                idx_v = r * grid_cols + c
                if idx_v < len(cells):
                    row_cells.append(cells[idx_v])
                else:
                    row_cells.append(np.full((cell_size, cell_size, 3), 255, dtype=np.uint8))
            rows.append(np.hstack(row_cells))
        sub_grid = np.vstack(rows)

        # Add model label at top-left
        sub_grid = add_label(sub_grid, f"{n_in}v", color=_WHITE, outline=True)
        model_grids.append(sub_grid)

        # Save individual grid
        ind_path = str(out / f"grid_{n_in}v.png")
        cv2.imwrite(ind_path, cv2.cvtColor(sub_grid, cv2.COLOR_RGB2BGR))

    # Combined: 1×6 horizontal (all models side by side)
    combined = np.hstack(model_grids)
    combined_path = str(out / "grid_all_models_combined.png")
    cv2.imwrite(combined_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"  Combined: {combined.shape[1]}×{combined.shape[0]} px")

    # Also 2×3 layout for better aspect ratio
    top = np.hstack(model_grids[:3])
    bot = np.hstack(model_grids[3:])
    combined_2x3 = np.vstack([top, bot])
    combined_2x3_path = str(out / "grid_all_models_2x3.png")
    cv2.imwrite(combined_2x3_path, cv2.cvtColor(combined_2x3, cv2.COLOR_RGB2BGR))

    print(f"\nDone. Saved to: {out}")
    print(f"  Individual: grid_{{1..6}}v.png")
    print(f"  Combined 1×6: {combined_path}")
    print(f"  Combined 2×3: {combined_2x3_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View ablation 36-view grid image")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--output-dir",
                        default="outputs/viz/comparison/mouse/view_ablation_grid_image")
    parser.add_argument("--frame-idx", type=int, default=3240)
    parser.add_argument("--cell-size", type=int, default=128)
    parser.add_argument("--grid-views", type=int, default=36)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--n-filter", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_grid_image(
        m5_dir=args.m5_dir,
        output_dir=args.output_dir,
        frame_idx=args.frame_idx,
        cell_size=args.cell_size,
        grid_views=args.grid_views,
        elevation=args.elevation,
        n_filter=args.n_filter,
        device=args.device,
    )


if __name__ == "__main__":
    main()
