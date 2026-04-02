"""View ablation 2×6 grid generator.

Generates a 2-row × 6-column grid video/image:
  Row 1: GT camera viewpoint rendered from 1v~6v models
  Row 2: Novel view (default: bottom) rendered from 1v~6v models

Usage (run on gpu03):
    # PoC (10 frames, 256px)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_2x6 \\
        --frame-range 3240:3250 --cell-size 256 --fps 10

    # Full quality (120 frames, 512px)
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.view_ablation_2x6 \\
        --frame-range 3240:3360 --cell-size 512 --fps 30
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
    REPRESENTATIVE_VIEWS,
    add_label,
    get_novel_camera,
    render_gaussian_at_view,
)

_WHITE = (255, 255, 255)


def run_view2x6(
    m5_dir: str,
    output_dir: str,
    frame_range: str = "3240:3360",
    cell_size: int = 512,
    fps: int = 30,
    n_filter: int = 2,
    gt_view_idx: int = 0,
    novel_view: str = "bottom",
    device: str = "cuda",
    save_first_frame_png: bool = True,
) -> None:
    """Generate 2×6 grid: Row1=GT camera view, Row2=Novel view, Cols=1v~6v."""
    experiments = VIEW_ABLATION_EXPERIMENTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== View Ablation 2×6 Grid ===")
    print(f"Row 1: GT camera (cam_{gt_view_idx:03d})")
    print(f"Row 2: Novel view ({novel_view})")
    print(f"Loading 6 models...")
    models = _load_models(experiments, device)
    active = [(exp, m) for exp, m in zip(experiments, models) if m is not None]
    if not active:
        print("[ERROR] No valid checkpoints. Aborting.")
        return

    n_cols = len(active)
    vdef = REPRESENTATIVE_VIEWS[novel_view]
    novel_cam = get_novel_camera(vdef["elevation"], vdef["azimuth"], resolution=cell_size)

    from mouse_extensions.behavior.camera_system import (
        gt_camera_c2w_original,
        gt_camera_fxfycxcy,
    )

    W_total = n_cols * cell_size
    H_total = 2 * cell_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = str(out / "view_ablation_2x6.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W_total, H_total))

    start, end = map(int, frame_range.split(":"))
    frame_indices = list(range(start, end))
    print(f"Rendering {len(frame_indices)} frames × {n_cols} models × 2 views  →  {out_path}")
    print(f"Grid: {W_total}×{H_total} px  (cell={cell_size})")

    first_frame_saved = False

    for fi in frame_indices:
        fd = Path(m5_dir) / f"{fi:06d}"
        if not fd.exists():
            continue

        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

        gt_c2w = gt_camera_c2w_original(str(fd), gt_view_idx)
        gt_fxfy = gt_camera_fxfycxcy(str(fd), gt_view_idx)
        if cell_size != 512:
            gt_fxfy = gt_fxfy * (cell_size / 512.0)
        gt_cam = (gt_c2w, gt_fxfy)

        row1_cells = []
        row2_cells = []

        for exp, model in active:
            n_in = exp["n_views"]
            result = model.predict(imgs[:, :n_in], c2ws[:, :n_in], fxfys[:, :n_in], idx)
            gaussians = result.gaussians[0]

            xyz = gaussians.get_xyz.detach().cpu().numpy()
            vc = compute_visibility_counts(xyz, str(fd), n_views=6)
            vis_mask = vc >= n_filter

            # Row 1: GT camera viewpoint
            try:
                cell_gt = render_gaussian_at_view(
                    gaussians, gt_cam, resolution=cell_size, device=device, vis_mask=vis_mask,
                )
            except Exception as e:
                print(f"  [warn] GT render failed {exp['label']} frame {fi}: {e}")
                cell_gt = np.full((cell_size, cell_size, 3), 80, dtype=np.uint8)
            cell_gt = add_label(cell_gt, f"{n_in}v", color=_WHITE, outline=True)
            row1_cells.append(cell_gt)

            # Row 2: Novel view
            try:
                cell_novel = render_gaussian_at_view(
                    gaussians, novel_cam, resolution=cell_size, device=device, vis_mask=vis_mask,
                )
            except Exception as e:
                print(f"  [warn] Novel render failed {exp['label']} frame {fi}: {e}")
                cell_novel = np.full((cell_size, cell_size, 3), 80, dtype=np.uint8)
            cell_novel = add_label(cell_novel, f"{n_in}v", color=_WHITE, outline=True)
            row2_cells.append(cell_novel)

        # Row labels on leftmost cell only
        row1_cells[0] = add_label(
            row1_cells[0], f"{active[0][0]['n_views']}v",
            row_label=f"GT cam {gt_view_idx}", color=_WHITE, outline=True,
        )
        row2_cells[0] = add_label(
            row2_cells[0], f"{active[0][0]['n_views']}v",
            row_label=vdef["label"], color=_WHITE, outline=True,
        )

        frame_grid = np.vstack([np.hstack(row1_cells), np.hstack(row2_cells)])
        writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        if save_first_frame_png and not first_frame_saved:
            png_path = str(out / "view_ablation_2x6_first_frame.png")
            cv2.imwrite(png_path, cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))
            print(f"  Saved first frame: {png_path}")
            first_frame_saved = True

        if fi % 20 == 0:
            print(f"  frame {fi}/{end - 1}")

    writer.release()
    print(f"\nDone. Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View ablation 2×6 grid")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--output-dir", default="outputs/viz/comparison/mouse/view_ablation_2x6")
    parser.add_argument("--frame-range", default="3240:3360")
    parser.add_argument("--cell-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--n-filter", type=int, default=2)
    parser.add_argument("--gt-view-idx", type=int, default=0)
    parser.add_argument("--novel-view", default="bottom",
                        choices=list(REPRESENTATIVE_VIEWS.keys()))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_view2x6(
        m5_dir=args.m5_dir,
        output_dir=args.output_dir,
        frame_range=args.frame_range,
        cell_size=args.cell_size,
        fps=args.fps,
        n_filter=args.n_filter,
        gt_view_idx=args.gt_view_idx,
        novel_view=args.novel_view,
        device=args.device,
    )


if __name__ == "__main__":
    main()
