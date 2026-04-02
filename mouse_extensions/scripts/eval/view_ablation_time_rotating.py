"""View ablation time-rotating comparison.

Both time (frame index) and camera angle change simultaneously across the
video — each frame uses a different temporal sample AND a different orbit
angle. This highlights view-count differences more than static turntable.

Layout: 1×6 horizontal grid, one cell per view count (1v~6v).

Usage (run on gpu03):
    # PoC (30 frames, 256px)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_time_rotating \\
        --num-frames 30 --cell-size 256 --fps 10

    # Full quality (120 frames, 512px)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_time_rotating \\
        --num-frames 120 --cell-size 512 --fps 30
"""

import argparse
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


def run_time_rotating(
    m5_dir: str,
    output_dir: str,
    frame_start: int = 3240,
    num_frames: int = 120,
    cell_size: int = 512,
    fps: int = 30,
    elevation: float = 20.0,
    n_filter: int = 2,
    device: str = "cuda",
) -> None:
    """Render time-rotating comparison: frame AND camera change together."""
    experiments = VIEW_ABLATION_EXPERIMENTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== View Ablation Time-Rotating Comparison ===")
    print(f"Frames: {frame_start}~{frame_start + num_frames - 1}, Elevation: {elevation}°")
    print(f"Loading 6 models...")
    models = _load_models(experiments, device)
    active = [(exp, m) for exp, m in zip(experiments, models) if m is not None]
    if not active:
        print("[ERROR] No valid checkpoints. Aborting.")
        return

    n_cols = len(active)

    # Generate orbit cameras — one per frame, completing a full 360°
    w, h, n_cams, fxfycxcy, orbit_c2ws = get_turntable_cameras(
        num_views=num_frames,
        w=cell_size,
        h=cell_size,
        elevation=elevation,
    )

    W_total = n_cols * cell_size
    H_total = cell_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(out / "view_ablation_time_rotating.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W_total, H_total))

    print(f"Rendering {num_frames} frames × {n_cols} models  →  {out_path}")
    print(f"Grid: {W_total}×{H_total} px  (cell={cell_size})")

    first_frame_saved = False

    for fi_offset in range(num_frames):
        fi = frame_start + fi_offset
        fd = Path(m5_dir) / f"{fi:06d}"
        if not fd.exists():
            continue

        # Load data for this temporal frame
        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)
        cam = (orbit_c2ws[fi_offset], fxfycxcy[fi_offset])

        cells = []
        for exp, model in active:
            n_in = exp["n_views"]
            result = model.predict(imgs[:, :n_in], c2ws[:, :n_in], fxfys[:, :n_in], idx)
            gaussians = result.gaussians[0]

            xyz = gaussians.get_xyz.detach().cpu().numpy()
            vc = compute_visibility_counts(xyz, str(fd), n_views=6)
            vis_mask = vc >= n_filter

            try:
                cell = render_gaussian_at_view(
                    gaussians, cam, resolution=cell_size,
                    device=device, vis_mask=vis_mask,
                )
            except Exception as e:
                print(f"  [warn] render failed {exp['label']} frame {fi}: {e}")
                cell = np.full((cell_size, cell_size, 3), 80, dtype=np.uint8)

            cell = add_label(cell, f"{n_in}v", color=_WHITE, outline=True)
            cells.append(cell)

        frame_grid = np.hstack(cells)
        writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        if not first_frame_saved:
            png_path = str(out / "time_rotating_first_frame.png")
            cv2.imwrite(png_path, cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))
            first_frame_saved = True

        if fi_offset % 20 == 0:
            print(f"  frame {fi} ({fi_offset}/{num_frames})")

    writer.release()
    print(f"\nDone. Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View ablation time-rotating comparison")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--output-dir",
                        default="outputs/viz/comparison/mouse/view_ablation_time_rotating")
    parser.add_argument("--frame-start", type=int, default=3240)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--cell-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--elevation", type=float, default=20.0)
    parser.add_argument("--n-filter", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_time_rotating(
        m5_dir=args.m5_dir,
        output_dir=args.output_dir,
        frame_start=args.frame_start,
        num_frames=args.num_frames,
        cell_size=args.cell_size,
        fps=args.fps,
        elevation=args.elevation,
        n_filter=args.n_filter,
        device=args.device,
    )


if __name__ == "__main__":
    main()
