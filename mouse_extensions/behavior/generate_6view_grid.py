"""6-View Grid Video Generator: GT views and GS-LRM renders side by side.

Generates temporal videos with all 6 camera views as 2x3 grid.
Supports keypoint overlay and side-by-side comparison (with/without KP).

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.generate_6view_grid \
        --frame-range 195:255 \
        --output-dir outputs/sdannce_poc/6view_grid
"""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from mouse_extensions.behavior.view_projected_filtering import (
    load_camera, load_keypoints_gslrm, project_points_to_2d,
)
from mouse_extensions.behavior.multiview_visibility_filter import (
    compute_visibility_counts, render_filtered_gaussians,
)
from mouse_extensions.behavior.cinematic_sequence import (
    load_gt, load_mask_overlay, overlay_keypoints, add_label,
)


def make_2x3_grid(images: List[np.ndarray]) -> np.ndarray:
    """Arrange 6 images into 2x3 grid."""
    assert len(images) == 6
    top = np.concatenate(images[:3], axis=1)
    bot = np.concatenate(images[3:], axis=1)
    return np.concatenate([top, bot], axis=0)


def project_kp(kp_3d, frame_dir, view_idx):
    cam = load_camera(str(Path(frame_dir) / "opencv_cameras.json"), view_idx)
    uv, valid = project_points_to_2d(
        kp_3d, np.array(cam["w2c"]), cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"])
    return uv, in_img


def main():
    parser = argparse.ArgumentParser(description="6-View Grid Video Generator")
    parser.add_argument("--frame-range", default="195:255")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5t")
    from mouse_extensions.paths import KP_22
    parser.add_argument("--kp-path", default=str(KP_22))
    parser.add_argument("--config-path", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt")
    parser.add_argument("--output-dir", default="outputs/sdannce_poc/6view_grid")
    parser.add_argument("--n-filter", type=int, default=2)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--with-kp", action="store_true", help="Overlay keypoints")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Generate side-by-side (no KP | with KP)")
    args = parser.parse_args()

    parts = args.frame_range.split(":")
    start, end = int(parts[0]), int(parts[1])
    step = int(parts[2]) if len(parts) > 2 else 1
    frames = list(range(start, end, step))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bg = (1.0, 1.0, 1.0)

    # --- GT 6-view grid ---
    print(f"GT 6-View Grid: {len(frames)} frames")
    gt_grids = []
    gt_kp_grids = []
    for fi in frames:
        fd = str(Path(args.m5_dir) / f"{fi:06d}")
        imgs = [load_gt(fd, v, bg) for v in range(6)]
        # Add view labels
        labeled = [add_label(img, f"View {v}") for v, img in enumerate(imgs)]
        gt_grids.append(make_2x3_grid(labeled))

        if args.with_kp or args.side_by_side:
            kp = load_keypoints_gslrm(args.kp_path, fi)
            kp_imgs = []
            for v in range(6):
                uv, valid = project_kp(kp, fd, v)
                kp_img = overlay_keypoints(imgs[v], uv, valid)
                kp_imgs.append(add_label(kp_img, f"View {v} + KP"))
            gt_kp_grids.append(make_2x3_grid(kp_imgs))

    _write_video(gt_grids, out_dir / "gt_6view_grid.mp4", args.fps)

    if gt_kp_grids:
        _write_video(gt_kp_grids, out_dir / "gt_6view_grid_kp.mp4", args.fps)

    if args.side_by_side and gt_kp_grids:
        sbs = [np.concatenate([a, b], axis=1) for a, b in zip(gt_grids, gt_kp_grids)]
        _write_video(sbs, out_dir / "gt_6view_side_by_side.mp4", args.fps)

    # --- GS-LRM Render 6-view grid ---
    print(f"\nGS-LRM Render 6-View Grid: {len(frames)} frames")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GSLRMInference(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    print("  Model loaded")

    render_grids = []
    render_kp_grids = []

    for i, fi in enumerate(frames):
        fd_path = Path(args.m5_dir) / f"{fi:06d}"
        if not fd_path.exists():
            continue

        images, c2ws, fxfycxcys, idx = load_sample_data(
            str(fd_path), image_size=512, device=device)
        result = model.predict(images, c2ws, fxfycxcys, idx)
        gaussians = result.gaussians[0]

        xyz = gaussians.get_xyz.detach().cpu().numpy()
        vc = compute_visibility_counts(xyz, str(fd_path), n_views=6)
        vis_mask = vc >= args.n_filter

        imgs = []
        for v in range(6):
            cam = load_camera(str(fd_path / "opencv_cameras.json"), v)
            img = render_filtered_gaussians(gaussians, vis_mask, cam, bg, device)
            imgs.append(img)

        labeled = [add_label(img, f"View {v} (N>={args.n_filter})") for v, img in enumerate(imgs)]
        render_grids.append(make_2x3_grid(labeled))

        if args.with_kp or args.side_by_side:
            kp = load_keypoints_gslrm(args.kp_path, fi)
            kp_imgs = []
            for v in range(6):
                uv, valid = project_kp(kp, str(fd_path), v)
                kp_img = overlay_keypoints(imgs[v], uv, valid)
                kp_imgs.append(add_label(kp_img, f"View {v} + KP"))
            render_kp_grids.append(make_2x3_grid(kp_imgs))

        del result, gaussians, images, c2ws, fxfycxcys
        torch.cuda.empty_cache()
        print(f"  f{fi} ({i+1}/{len(frames)}): {vis_mask.sum():,} Gaussians")

    _write_video(render_grids, out_dir / "render_6view_grid.mp4", args.fps)

    if render_kp_grids:
        _write_video(render_kp_grids, out_dir / "render_6view_grid_kp.mp4", args.fps)

    if args.side_by_side and render_kp_grids:
        sbs = [np.concatenate([a, b], axis=1) for a, b in zip(render_grids, render_kp_grids)]
        _write_video(sbs, out_dir / "render_6view_side_by_side.mp4", args.fps)

    print(f"\nAll outputs: {out_dir}")


def _write_video(frames, path, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Video: {path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
