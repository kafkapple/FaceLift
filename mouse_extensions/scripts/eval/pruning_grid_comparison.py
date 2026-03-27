"""Pruning E1-E5 multi-view grid video comparison.

Rows = 5 pruning conditions, Cols = 4 views (bottom/side/front/top).
Shows the visual effect of each filter combination.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m mouse_extensions.scripts.eval.pruning_grid_comparison \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --output-dir outputs/viz/comparison/mouse/pruning_grid_768
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts
from mouse_extensions.model.orientation_filter import suppress_z_aligned_flat
from mouse_extensions.scripts.eval.fl_gt_view_comparison import (
    add_label, get_novel_camera, render_gaussian_at_view,
)

CKPT_A00 = "/node_data/joon/checkpoints/FaceLift/gslrm/base_uniform_v2_6view_v2/best_psnr.pt"
CKPT_A03 = "/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_6view_alpha03_v3/best_psnr.pt"

EXPERIMENTS = [
    {"name": "E1: Opacity (α=0)", "ckpt": CKPT_A00, "vis": False, "orient": False},
    {"name": "E2: +Visibility",   "ckpt": CKPT_A00, "vis": True,  "orient": False},
    {"name": "E3: +Orient",       "ckpt": CKPT_A00, "vis": True,  "orient": True},
    {"name": "E4: α=0.3 only",    "ckpt": CKPT_A03, "vis": False, "orient": False},
    {"name": "E5: Full (α=0.3)",  "ckpt": CKPT_A03, "vis": True,  "orient": True},
]

VIEWS = [
    {"name": "Bottom (-80)", "elev": -80, "azim": 0},
    {"name": "Side (0)", "elev": 0, "azim": 90},
    {"name": "Front (0)", "elev": 0, "azim": 0},
    {"name": "Top (+60)", "elev": 60, "azim": 0},
]

FILTER_PARAMS = dict(
    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--frame-range", default="3240:3280")
    parser.add_argument("--cell-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output-dir", default="outputs/viz/comparison/mouse/pruning_grid_768")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    cs = args.cell_size

    # Pre-compute cameras
    cams = {v["name"]: get_novel_camera(v["elev"], v["azim"], resolution=cs) for v in VIEWS}

    n_exp, n_views = len(EXPERIMENTS), len(VIEWS)
    W_total = n_views * cs
    H_total = n_exp * cs
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_path = str(out / "pruning_grid.mp4")
    writer = cv2.VideoWriter(vid_path, fourcc, args.fps, (W_total, H_total))

    start, end = map(int, args.frame_range.split(":"))
    frame_indices = list(range(start, end))
    print(f"Rendering {len(frame_indices)} frames × {n_exp} conditions × {n_views} views")
    print(f"Grid: {W_total}×{H_total} px (cell={cs})")

    # Load both models
    models = {}
    for ckpt in set(e["ckpt"] for e in EXPERIMENTS):
        print(f"Loading {ckpt}...")
        m = GSLRMInference(config_path=args.config, checkpoint_path=ckpt)
        m.config.model.num_input_views = 6
        models[ckpt] = m

    for fi in frame_indices:
        fd = Path(args.m5_dir) / f"{fi:06d}"
        if not fd.exists():
            continue

        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)

        rows = []
        for exp in EXPERIMENTS:
            model = models[exp["ckpt"]]
            result = model.predict(imgs, c2ws, fxfys, idx)
            gs = result.gaussians[0]
            gs.apply_all_filters(**FILTER_PARAMS)

            if exp["vis"]:
                xyz = gs.get_xyz.detach().cpu().numpy()
                vc = compute_visibility_counts(xyz, str(fd), n_views=6)
                gs.filter(torch.from_numpy(vc >= 2).to(device))

            if exp["orient"]:
                suppress_z_aligned_flat(gs, z_align_thresh=0.15, attenuation=0.0, mode="prune")

            cells = []
            for v in VIEWS:
                cam = cams[v["name"]]
                try:
                    cell = render_gaussian_at_view(gs, cam, resolution=cs, device=device)
                except Exception:
                    cell = np.full((cs, cs, 3), 80, dtype=np.uint8)
                cell = add_label(cell, v["name"], exp["name"])
                cells.append(cell)
            rows.append(np.hstack(cells))

        frame_grid = np.vstack(rows)
        cv2.putText(frame_grid, f"Frame {fi:04d}",
                    (8, H_total - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        if fi == frame_indices[0]:
            cv2.imwrite(str(out / f"grid_{fi:06d}.png"),
                        cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR))

        if fi % 10 == 0:
            print(f"  frame {fi}/{end-1}")

    writer.release()
    print(f"Done: {vid_path}")


if __name__ == "__main__":
    main()
