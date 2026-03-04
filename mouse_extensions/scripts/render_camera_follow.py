#!/usr/bin/env python3
"""Render camera-follow video from GS-LRM with keypoint-driven camera trajectory.

For each frame: runs GS-LRM inference → computes face-follow camera pose from
MAMMAL 3D keypoints → renders Gaussian splat from follow-camera → saves frames
and compiles into video.

Generates video variants:
  Camera-follow (requires GS-LRM inference):
    - camera_follow_face_clean.mp4      (no overlay)
    - camera_follow_face_overlay.mp4    (keypoint skeleton overlay)
    - camera_follow_face_sidebyside.mp4 (clean | overlay side by side)

  Multi-view grid (GT images only, no inference needed):
    - multiview_grid_overlay.mp4        (2x3 grid, 6 camera views with keypoints)

Usage:
    # Camera-follow + grid (full)
    CUDA_VISIBLE_DEVICES=4 /home/joon/anaconda3/envs/facelift/bin/python3 \
        mouse_extensions/scripts/render_camera_follow.py \
        --checkpoint base_uniform_v2_6view_v2 \
        --keypoints_npz /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
        --data_txt ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
        --start_frame 3240 --num_frames 200 \
        --output_dir /home/joon/dev/FaceLift/outputs/camera_follow

    # Grid-only mode (fast, no GPU needed)
    python render_camera_follow.py --grid_only \
        --keypoints_npz ... --data_txt ... \
        --start_frame 3240 --num_frames 200 \
        --output_dir /home/joon/dev/FaceLift/outputs/camera_follow
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add FaceLift project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch

from mouse_extensions.visualization.keypoint_overlay import (
    CameraFollowConfig,
    KeypointFollowCamera,
    KeypointVisualizer,
    create_legend,
    project_3d_to_2d,
)

# M5 coordinate transform constants
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785


def transform_mammal_to_facelift(kp: np.ndarray) -> np.ndarray:
    return (kp - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def load_gslrm_pipeline(checkpoint_name: str, device: str = "cuda"):
    """Load GS-LRM inference pipeline (once for all frames)."""
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference

    ckpt_base = Path("checkpoints/gslrm")
    config_path = ckpt_base / checkpoint_name / "config.yaml"
    ckpt_dir = ckpt_base / checkpoint_name

    pipeline = GSLRMInference(
        config_path=str(config_path),
        checkpoint_path=str(ckpt_dir),
        device=device,
    )
    return pipeline


def run_inference_and_render(
    pipeline,
    sample_dir: str,
    c2w_follow: np.ndarray,
    intrinsics: Dict[str, float],
    device: str = "cuda",
) -> Optional[np.ndarray]:
    """Run GS-LRM inference and render from follow-camera.

    Returns:
        rendered: (H, W, 3) BGR image or None
    """
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data
    from gslrm.model.gaussians_renderer import render_opencv_cam

    images, c2ws, fxfycxcys, index = load_sample_data(sample_dir, device=device)
    result = pipeline.predict(images, c2ws, fxfycxcys, index)

    gaussians_raw = result.get("gaussians", None)
    if gaussians_raw is None:
        return None
    gaussians = gaussians_raw[0] if isinstance(gaussians_raw, list) else gaussians_raw

    c2w_tensor = torch.from_numpy(c2w_follow.astype(np.float32)).to(device)
    fxfycxcy = torch.tensor(
        [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]],
        dtype=torch.float32, device=device,
    )

    rendered = render_opencv_cam(
        gaussians,
        height=int(intrinsics.get("cy", 256) * 2),
        width=int(intrinsics.get("cx", 256) * 2),
        C2W=c2w_tensor,
        fxfycxcy=fxfycxcy,
    )

    render_img = rendered["render"]  # (C, H, W)
    render_np = (render_img.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)
    return render_bgr


def make_side_by_side(
    clean: np.ndarray,
    overlay: np.ndarray,
    legend: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stitch clean and overlay images side by side with optional legend.

    Args:
        clean: (H, W, 3) BGR image without overlay
        overlay: (H, W, 3) BGR image with keypoint overlay
        legend: optional (H, W_legend, 3) legend strip

    Returns:
        composite: (H, W_total, 3) side-by-side image
    """
    h = clean.shape[0]
    # Divider line (2px white)
    divider = np.ones((h, 2, 3), dtype=np.uint8) * 200
    parts = [clean, divider, overlay]
    if legend is not None:
        # Resize legend height to match
        legend_resized = cv2.resize(legend, (legend.shape[1], h))
        parts.extend([divider, legend_resized])
    return np.concatenate(parts, axis=1)


def load_gt_cameras_and_images(
    sample_dir: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict[str, float]]]:
    """Load GT camera views and parameters from a sample directory.

    Reads opencv_cameras.json (format: {"frames": [list of cam dicts]})
    and loads the 6 camera images from images/ subdirectory.

    Returns:
        images: list of 6 (H, W, 3) BGR images
        w2cs: list of 6 (4, 4) world-to-camera matrices
        intrinsics_list: list of 6 intrinsics dicts
    """
    cam_json = os.path.join(sample_dir, "opencv_cameras.json")
    with open(cam_json, "r") as f:
        cam_data = json.load(f)

    frames = cam_data["frames"]  # list of 6 camera dicts

    images = []
    w2cs = []
    intrinsics_list = []

    for cam_info in frames:
        # Load image (file_path is relative, e.g. "images/cam_000.png")
        file_path = cam_info.get("file_path", "")
        img_path = os.path.join(sample_dir, file_path)
        img = cv2.imread(img_path)
        images.append(img)

        # Camera extrinsics
        w2c = np.array(cam_info["w2c"], dtype=np.float64)  # (4, 4)
        w2cs.append(w2c)

        intrinsics_list.append({
            "fx": cam_info["fx"],
            "fy": cam_info["fy"],
            "cx": cam_info["cx"],
            "cy": cam_info["cy"],
        })

    return images, w2cs, intrinsics_list


def make_multiview_grid(
    images: List[np.ndarray],
    cam_labels: Optional[List[str]] = None,
    rows: int = 2,
    cols: int = 3,
    cell_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Arrange multiple images into a rows×cols grid.

    Args:
        images: list of (H, W, 3) BGR images
        cam_labels: optional text labels per cell (e.g. "cam_000")
        rows, cols: grid dimensions
        cell_size: (w, h) to resize each cell, or None to use first image size

    Returns:
        grid: (rows*h, cols*w, 3) BGR image
    """
    if cell_size is None:
        h, w = images[0].shape[:2]
    else:
        w, h = cell_size

    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx in range(min(len(images), rows * cols)):
        r, c = divmod(idx, cols)
        img = images[idx]
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        # Optional camera label
        if cam_labels and idx < len(cam_labels):
            cv2.putText(
                img, cam_labels[idx], (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    return grid


def frames_to_video(
    frame_dir: str,
    output_path: str,
    fps: int = 10,
    pattern: str = "frame_*.png",
):
    """Compile saved frames into MP4 video."""
    import glob
    frame_files = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    if not frame_files:
        print(f"  No frames found for {output_path}")
        return

    first = cv2.imread(frame_files[0])
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for ff in frame_files:
        img = cv2.imread(ff)
        writer.write(img)

    writer.release()
    print(f"  Video: {output_path} ({len(frame_files)} frames, {fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Camera-follow rendering with keypoints")
    parser.add_argument("--checkpoint", default="base_uniform_v2_6view_v2")
    parser.add_argument("--keypoints_npz", required=True)
    parser.add_argument("--data_txt", required=True)
    parser.add_argument("--start_frame", type=int, default=3240)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--distance", type=float, default=0.8,
                        help="Camera distance from face")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="EMA smoothing alpha (0=max smooth, 1=no smooth)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no_inference", action="store_true",
                        help="Skip camera-follow rendering, only save trajectory")
    parser.add_argument("--grid_only", action="store_true",
                        help="Only generate multi-view grid (no GS-LRM inference)")
    parser.add_argument("--no_grid", action="store_true",
                        help="Skip multi-view grid generation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # --- Load keypoints ---
    data = np.load(args.keypoints_npz, allow_pickle=True)
    kp_all = data["keypoints"]        # (3600, 22, 3) MAMMAL world
    frame_indices = data["frame_indices"]
    print(f"Loaded keypoints: {kp_all.shape}")

    # --- Load data file list ---
    data_txt = os.path.expanduser(args.data_txt)
    with open(data_txt, "r") as f:
        all_dirs = [l.strip().rstrip("/") for l in f if l.strip()]
    print(f"Data samples: {len(all_dirs)}")

    # Build lookup for quick dir matching
    dir_lookup = {}
    for sd in all_dirs:
        try:
            dir_lookup[int(os.path.basename(sd))] = sd
        except ValueError:
            pass

    # --- Build frame list ---
    m5_frames = list(range(args.start_frame, args.start_frame + args.num_frames))
    print(f"Frames: {m5_frames[0]}-{m5_frames[-1]} ({len(m5_frames)} total)")

    # --- Compute camera trajectory ---
    config = CameraFollowConfig(
        target="face",
        distance=args.distance,
        smoothing_alpha=args.smoothing,
    )
    cam_follow = KeypointFollowCamera(config)

    # Collect keypoints for trajectory (transform to FaceLift space)
    kp_sequence = []
    valid_frames = []
    valid_dirs = []

    for m5_idx in m5_frames:
        orig_frame = m5_idx * 5
        matches = np.where(frame_indices == orig_frame)[0]
        if len(matches) == 0:
            continue

        target_dir = dir_lookup.get(m5_idx)
        if target_dir is None:
            continue

        kp_raw = kp_all[matches[0]]  # (22, 3) MAMMAL world
        kp_fl = transform_mammal_to_facelift(kp_raw)
        kp_sequence.append(kp_fl)
        valid_frames.append(m5_idx)
        valid_dirs.append(target_dir)

    if not kp_sequence:
        print("No valid frames found!")
        return

    kp_seq_arr = np.stack(kp_sequence, axis=0)  # (T, 22, 3)
    c2ws, intrinsics_list = cam_follow.compute_trajectory(kp_seq_arr)
    print(f"Camera trajectory: {len(c2ws)} frames")

    # --- Setup output dirs ---
    os.makedirs(args.output_dir, exist_ok=True)

    viz = KeypointVisualizer(
        joint_radius=3, bone_thickness=1,
        draw_labels=True, draw_skeleton=True,
    )
    legend = create_legend(height=512, width=160)

    # ===== Camera-follow rendering =====
    if not args.grid_only:
        dirs = {}
        for name in ("clean", "overlay", "sidebyside"):
            d = os.path.join(args.output_dir, f"frames_{name}")
            os.makedirs(d, exist_ok=True)
            dirs[name] = d

        if not args.no_inference:
            print("Loading GS-LRM pipeline...")
            pipeline = load_gslrm_pipeline(args.checkpoint, args.device)

        n_total = len(valid_frames)
        for i, (m5_idx, sample_dir, c2w, intr, kp_fl) in enumerate(
            zip(valid_frames, valid_dirs, c2ws, intrinsics_list, kp_sequence)
        ):
            print(f"  [{i+1}/{n_total}] frame_{m5_idx:06d}", end="", flush=True)

            if args.no_inference:
                np.savez(
                    os.path.join(args.output_dir, f"frame_{m5_idx:06d}_camera.npz"),
                    c2w=c2w, **intr,
                )
                print("")
                continue

            rendered = run_inference_and_render(
                pipeline, sample_dir, c2w, intr, args.device,
            )

            if rendered is None:
                print("  [FAIL]")
                continue

            # Clean version
            clean_img = rendered.copy()

            # Overlay version
            w2c = np.linalg.inv(c2w)
            overlay_img = viz.overlay_on_image(rendered, kp_fl, w2c, intr)

            # Side-by-side version
            sbs_img = make_side_by_side(clean_img, overlay_img, legend)

            # Save all three
            fname = f"frame_{i:04d}.png"
            cv2.imwrite(os.path.join(dirs["clean"], fname), clean_img)
            cv2.imwrite(os.path.join(dirs["overlay"], fname), overlay_img)
            cv2.imwrite(os.path.join(dirs["sidebyside"], fname), sbs_img)

            print("  [OK]")

        # Compile camera-follow videos
        if not args.no_inference:
            print("Compiling camera-follow videos...")
            for name in ("clean", "overlay", "sidebyside"):
                video_path = os.path.join(
                    args.output_dir, f"camera_follow_face_{name}.mp4",
                )
                frames_to_video(dirs[name], video_path, fps=args.fps)

    # ===== Multi-view grid (GT 6-camera views + keypoint overlay) =====
    if not args.no_grid:
        print("Generating multi-view grid...")
        grid_dir = os.path.join(args.output_dir, "frames_grid")
        os.makedirs(grid_dir, exist_ok=True)

        cam_labels = [f"cam_{c:03d}" for c in range(6)]

        for i, (m5_idx, sample_dir, kp_fl) in enumerate(
            zip(valid_frames, valid_dirs, kp_sequence)
        ):
            if i % 50 == 0:
                print(f"  Grid [{i+1}/{len(valid_frames)}]")

            try:
                gt_imgs, gt_w2cs, gt_intrs = load_gt_cameras_and_images(sample_dir)
            except (FileNotFoundError, KeyError) as e:
                print(f"  Skip grid frame {m5_idx}: {e}")
                continue

            # Overlay keypoints on each camera view
            overlaid_views = []
            for img, w2c, intr in zip(gt_imgs, gt_w2cs, gt_intrs):
                if img is None:
                    overlaid_views.append(np.zeros((256, 256, 3), dtype=np.uint8))
                    continue
                ov = viz.overlay_on_image(img, kp_fl, w2c, intr)
                overlaid_views.append(ov)

            grid = make_multiview_grid(
                overlaid_views, cam_labels=cam_labels, rows=2, cols=3,
            )

            # Add frame info label
            cv2.putText(
                grid, f"M5:{m5_idx}", (grid.shape[1] - 100, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imwrite(os.path.join(grid_dir, f"frame_{i:04d}.png"), grid)

        # Compile grid video
        grid_video = os.path.join(args.output_dir, "multiview_grid_overlay.mp4")
        frames_to_video(grid_dir, grid_video, fps=args.fps)

    # --- Save trajectory summary ---
    traj_path = os.path.join(args.output_dir, "trajectory.npz")
    np.savez(
        traj_path,
        c2ws=np.stack(c2ws),
        frame_indices=np.array(valid_frames),
        config=str(config),
    )
    print(f"Trajectory saved: {traj_path}")
    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
