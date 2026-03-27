"""Resolution comparison: render same Gaussians at different resolutions.

Generates side-by-side grid images and video comparing 3 resolutions.
Same 3D Gaussians, same camera — only rendering resolution differs.

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.resolution_comparison \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --output-dir outputs/analysis/mouse/resolution_comparison \
        --frame-range 3240:3280
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.scripts.eval.fl_gt_view_comparison import (
    add_label,
    get_novel_camera,
    render_gaussian_at_view,
)

FILTER_PARAMS = dict(
    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
)

RESOLUTIONS = [384, 512, 768]

VIEW_ANGLES = [
    {"name": "Bottom (-80)", "elev": -80, "azim": 0},
    {"name": "Side (0)", "elev": 0, "azim": 90},
    {"name": "Top (+60)", "elev": 60, "azim": 0},
]


def main():
    parser = argparse.ArgumentParser(description="Resolution comparison grid")
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--checkpoint",
                        default="/node_data/joon/checkpoints/FaceLift/gslrm/"
                                "base_uniform_v2_6view_v2/best_psnr.pt")
    parser.add_argument("--frame-range", default="3240:3270")
    parser.add_argument("--display-size", type=int, default=384,
                        help="Each cell displayed at this size in the grid")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/resolution_comparison")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = GSLRMInference(config_path=args.config, checkpoint_path=args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_res = len(RESOLUTIONS)
    n_views = len(VIEW_ANGLES)
    ds = args.display_size

    # Grid: rows=resolutions, cols=views
    W_total = n_views * ds
    H_total = n_res * ds
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = str(out / "resolution_comparison.mp4")
    writer = cv2.VideoWriter(vid_path, fourcc, args.fps, (W_total, H_total))

    start, end = map(int, args.frame_range.split(":"))
    frame_indices = list(range(start, end))
    print(f"Rendering {len(frame_indices)} frames × {n_res} resolutions × {n_views} views")
    print(f"Resolutions: {RESOLUTIONS}, display cell: {ds}px")

    for fi in frame_indices:
        fd = Path(args.m5_dir) / f"{fi:06d}"
        if not fd.exists():
            continue

        # Input always at 512 (model native)
        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)
        result = model.predict(imgs, c2ws, fxfys, idx)
        gs = result.gaussians[0]
        gs.apply_all_filters(**FILTER_PARAMS)

        rows = []
        for res in RESOLUTIONS:
            cells = []
            for v in VIEW_ANGLES:
                cam = get_novel_camera(v["elev"], v["azim"], resolution=res)
                try:
                    cell = render_gaussian_at_view(gs, cam, resolution=res, device=device)
                except Exception as e:
                    cell = np.full((res, res, 3), 80, dtype=np.uint8)

                # Resize to display size for uniform grid
                if cell.shape[0] != ds:
                    cell = cv2.resize(cell, (ds, ds), interpolation=cv2.INTER_LANCZOS4)

                cell = add_label(cell, v["name"], f"{res}px")
                cells.append(cell)
            rows.append(np.hstack(cells))

        frame_grid = np.vstack(rows)
        cv2.putText(frame_grid, f"Frame {fi:04d}",
                    (8, H_total - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        # Save first frame as individual full-res images + grid PNG
        if fi == frame_indices[0]:
            png_path = out / f"grid_{fi:06d}.png"
            cv2.imwrite(str(png_path), cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))
            print(f"  Grid PNG: {png_path}")

            # Save individual full-res images for each resolution
            for res in RESOLUTIONS:
                res_dir = out / f"individual_{res}px"
                res_dir.mkdir(exist_ok=True)
                for v in VIEW_ANGLES:
                    cam = get_novel_camera(v["elev"], v["azim"], resolution=res)
                    img = render_gaussian_at_view(gs, cam, resolution=res, device=device)
                    fname = res_dir / f"{v['name'].replace(' ', '_').replace('(', '').replace(')', '')}_{fi:06d}.png"
                    cv2.imwrite(str(fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"  Individual {res}px images saved")

        if fi % 10 == 0:
            print(f"  frame {fi}/{end-1}")

    writer.release()
    print(f"\nDone. Saved: {vid_path}")


if __name__ == "__main__":
    main()
