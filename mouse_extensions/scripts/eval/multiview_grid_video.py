"""Multi-view grid video — many camera angles shown simultaneously.

Renders N camera angles as a grid, with temporal frames advancing.
Shows reconstruction quality from many viewpoints at once.

Usage (run on gpu03):
    # 6v model, 24 angles, 4x6 grid
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.multiview_grid_video \
        --n-views 6 --grid-angles 24 --cell-size 128 --num-frames 60

    # All models side by side (6 videos)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.multiview_grid_video \
        --n-views all --grid-angles 24 --cell-size 128 --num-frames 60
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


def run_multiview_grid(
    m5_dir: str,
    output_dir: str,
    n_views_filter: str = "6",
    frame_start: int = 3240,
    num_frames: int = 60,
    grid_angles: int = 24,
    cell_size: int = 128,
    fps: int = 20,
    elevation: float = 20.0,
    n_filter: int = 2,
    device: str = "cuda",
) -> None:
    """Render multi-angle grid video for specified model(s)."""
    experiments = VIEW_ABLATION_EXPERIMENTS
    if n_views_filter != "all":
        target = int(n_views_filter)
        experiments = [e for e in experiments if e["n_views"] == target]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Multi-View Grid Video ===")
    print(f"Models: {[e['label'] for e in experiments]}")
    print(f"Grid: {grid_angles} angles, {cell_size}px cells, {num_frames} frames")

    models = _load_models(experiments, device)
    active = [(exp, m) for exp, m in zip(experiments, models) if m is not None]
    if not active:
        print("[ERROR] No valid checkpoints.")
        return

    # Pre-compute orbit cameras for the grid
    grid_cols = int(math.ceil(math.sqrt(grid_angles)))
    grid_rows = int(math.ceil(grid_angles / grid_cols))
    w_grid = grid_cols * cell_size
    h_grid = grid_rows * cell_size

    _, _, _, fxfycxcy, orbit_c2ws = get_turntable_cameras(
        num_views=grid_angles, w=cell_size, h=cell_size, elevation=elevation,
    )
    cams = [(orbit_c2ws[i], fxfycxcy[i]) for i in range(grid_angles)]

    for exp, model in active:
        n_in = exp["n_views"]
        tag = f"{n_in}v"
        out_path = str(out / f"multiview_{tag}_{grid_angles}angles.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w_grid, h_grid))

        print(f"\n  Rendering {tag}: {w_grid}x{h_grid}px, {num_frames} frames → {out_path}")
        first_saved = False

        for fi_off in range(num_frames):
            fi = frame_start + fi_off
            fd = Path(m5_dir) / f"{fi:06d}"
            if not fd.exists():
                continue

            imgs, c2ws_d, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)
            result = model.predict(
                imgs[:, :n_in], c2ws_d[:, :n_in], fxfys[:, :n_in], idx,
            )
            gaussians = result.gaussians[0]

            xyz = gaussians.get_xyz.detach().cpu().numpy()
            vc = compute_visibility_counts(xyz, str(fd), n_views=6)
            vis_mask = vc >= n_filter

            cells = []
            for cam in cams:
                try:
                    cell = render_gaussian_at_view(
                        gaussians, cam, resolution=cell_size,
                        device=device, vis_mask=vis_mask,
                    )
                except Exception:
                    cell = np.full((cell_size, cell_size, 3), 80, dtype=np.uint8)
                cells.append(cell)

            # Assemble grid
            rows = []
            for r in range(grid_rows):
                row_cells = cells[r * grid_cols:(r + 1) * grid_cols]
                while len(row_cells) < grid_cols:
                    row_cells.append(
                        np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
                    )
                rows.append(np.hstack(row_cells))
            frame_grid = np.vstack(rows)

            # Label
            frame_grid = add_label(frame_grid, tag, color=_WHITE, outline=True)

            writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

            if not first_saved:
                png = str(out / f"multiview_{tag}_{grid_angles}angles_first.png")
                cv2.imwrite(png, cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))
                first_saved = True

            if fi_off % 20 == 0:
                print(f"    frame {fi} ({fi_off}/{num_frames})")

        writer.release()
        print(f"  Done: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-view grid video")
    p.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    p.add_argument("--output-dir",
                   default="outputs/viz/comparison/mouse/multiview_grid")
    p.add_argument("--n-views", default="6", help="Model view count or 'all'")
    p.add_argument("--frame-start", type=int, default=3240)
    p.add_argument("--num-frames", type=int, default=60)
    p.add_argument("--grid-angles", type=int, default=24)
    p.add_argument("--cell-size", type=int, default=128)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--elevation", type=float, default=20.0)
    p.add_argument("--n-filter", type=int, default=2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run_multiview_grid(
        m5_dir=args.m5_dir,
        output_dir=args.output_dir,
        n_views_filter=args.n_views,
        frame_start=args.frame_start,
        num_frames=args.num_frames,
        grid_angles=args.grid_angles,
        cell_size=args.cell_size,
        fps=args.fps,
        elevation=args.elevation,
        n_filter=args.n_filter,
        device=args.device,
    )


if __name__ == "__main__":
    main()
