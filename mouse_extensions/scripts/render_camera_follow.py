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
    CAMERA_TARGET_PRESETS,
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


class StreamingVideoWriter:
    """Write frames directly to video without saving individual PNGs.

    Writes with OpenCV (mp4v) then converts to H.264 via ffmpeg for
    macOS QuickTime compatibility.
    """

    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self._tmp_path = output_path + ".tmp.mp4"
        self._writer = None
        self._count = 0

    def write(self, frame: np.ndarray):
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._tmp_path, fourcc, self.fps, (w, h),
            )
        self._writer.write(frame)
        self._count += 1

    def release(self):
        if self._writer is not None:
            self._writer.release()
            # Convert mp4v → H.264 for macOS compatibility
            import subprocess
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", self._tmp_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    self.output_path,
                ],
                capture_output=True,
            )
            if result.returncode == 0 and os.path.exists(self.output_path):
                os.remove(self._tmp_path)
            else:
                # Fallback: keep mp4v version
                os.rename(self._tmp_path, self.output_path)
                print(f"  Warning: ffmpeg conversion failed, using mp4v fallback")
            print(f"  Video: {self.output_path} ({self._count} frames, {self.fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Camera-follow rendering with keypoints")
    parser.add_argument("--checkpoint", default="base_uniform_v2_6view_v2")
    parser.add_argument("--keypoints_npz", required=True)
    parser.add_argument("--data_txt", required=True)
    parser.add_argument("--start_frame", type=int, default=3240)
    parser.add_argument("--num_frames", type=int, default=360)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", default="face",
                        choices=list(CAMERA_TARGET_PRESETS.keys()),
                        help="Camera target (default: face)")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Render multiple targets in batch (overrides --target)")
    parser.add_argument("--stabilize", action="store_true",
                        help="Enable body-stabilized rendering (fix spine axis)")
    parser.add_argument("--distance", type=float, default=0.8,
                        help="Camera distance from target")
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

    # Resolve target list
    if args.targets:
        for t in args.targets:
            if t not in CAMERA_TARGET_PRESETS:
                parser.error(f"Unknown target: {t}. Available: {list(CAMERA_TARGET_PRESETS.keys())}")
        target_list = args.targets
    else:
        target_list = [args.target]

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

    # --- Collect keypoints (shared across targets) ---
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

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    viz = KeypointVisualizer(
        joint_radius=3, bone_thickness=1,
        draw_labels=True, draw_skeleton=True,
    )
    legend = create_legend(height=512, width=160)

    # Load pipeline once (shared across targets)
    pipeline = None
    if not args.grid_only and not args.no_inference:
        print("Loading GS-LRM pipeline...")
        pipeline = load_gslrm_pipeline(args.checkpoint, args.device)

    # ===== Camera-follow rendering (per target) =====
    if not args.grid_only:
        stab_suffix = "_stabilized" if args.stabilize else ""
        print(f"\nTargets to render: {target_list}")
        print(f"Body stabilization: {'ON' if args.stabilize else 'OFF'}")

        for target_name in target_list:
            print(f"\n{'='*60}")
            print(f"Rendering target: {target_name} ({CAMERA_TARGET_PRESETS[target_name]})")
            print(f"{'='*60}")

            # Distance defaults per target type
            # Follow cameras: close-up (0.6~1.2), Preset: wide (2.5)
            dist = args.distance
            _DEFAULT_DISTANCES = {
                "face": 0.8,
                "tail_base": 1.0,
                "left_front_paw": 0.6, "right_front_paw": 0.6,
                "left_hind_paw": 0.6, "right_hind_paw": 0.6,
                "frontal": 1.0, "posterior": 1.0,
                "lateral_left": 1.2, "lateral_right": 1.2,
                "top_down": 2.5, "bottom_up": 2.5, "body": 2.5,
            }
            if args.distance == 0.8:  # user didn't override
                dist = _DEFAULT_DISTANCES.get(target_name, 1.0)

            config = CameraFollowConfig(
                target=target_name,
                distance=dist,
                smoothing_alpha=args.smoothing,
                stabilize_body=args.stabilize,
            )
            cam_follow = KeypointFollowCamera(config)
            c2ws, intrinsics_list = cam_follow.compute_trajectory(kp_seq_arr)
            print(f"  Camera trajectory: {len(c2ws)} frames")

            # Output subdirectory per target
            target_dir_out = os.path.join(args.output_dir, f"{target_name}{stab_suffix}")
            os.makedirs(target_dir_out, exist_ok=True)

            # Streaming video writers
            writers = {}
            for vname in ("clean", "overlay", "sidebyside"):
                path = os.path.join(
                    target_dir_out,
                    f"camera_follow_{target_name}{stab_suffix}_{vname}.mp4",
                )
                writers[vname] = StreamingVideoWriter(path, fps=args.fps)

            rep_frame_idx = len(valid_frames) // 2

            n_total = len(valid_frames)
            for i, (m5_idx, sample_dir, c2w, intr, kp_fl) in enumerate(
                zip(valid_frames, valid_dirs, c2ws, intrinsics_list, kp_sequence)
            ):
                if i % 50 == 0 or i == n_total - 1:
                    print(f"  [{i+1}/{n_total}] frame_{m5_idx:06d}", end="", flush=True)

                if args.no_inference:
                    np.savez(
                        os.path.join(target_dir_out, f"frame_{m5_idx:06d}_camera.npz"),
                        c2w=c2w, **intr,
                    )
                    if i % 50 == 0:
                        print("")
                    continue

                rendered = run_inference_and_render(
                    pipeline, sample_dir, c2w, intr, args.device,
                )

                if rendered is None:
                    if i % 50 == 0:
                        print("  [FAIL]")
                    continue

                clean_img = rendered.copy()
                w2c = np.linalg.inv(c2w)
                overlay_img = viz.overlay_on_image(rendered, kp_fl, w2c, intr)
                sbs_img = make_side_by_side(clean_img, overlay_img, legend)

                writers["clean"].write(clean_img)
                writers["overlay"].write(overlay_img)
                writers["sidebyside"].write(sbs_img)

                if i == rep_frame_idx:
                    cv2.imwrite(
                        os.path.join(target_dir_out, "representative_sidebyside.png"),
                        sbs_img,
                    )

                if i % 50 == 0:
                    print("  [OK]")

            # Finalize videos for this target
            if not args.no_inference:
                print(f"  Finalizing {target_name} videos...")
                for w in writers.values():
                    w.release()

            # Save trajectory
            np.savez(
                os.path.join(target_dir_out, "trajectory.npz"),
                c2ws=np.stack(c2ws),
                frame_indices=np.array(valid_frames),
                config=str(config),
            )

    # ===== Multi-view grid (GT 6-camera views + keypoint overlay) =====
    if not args.no_grid:
        print("Generating multi-view grid...")
        grid_writer = StreamingVideoWriter(
            os.path.join(args.output_dir, "multiview_grid_overlay.mp4"),
            fps=args.fps,
        )

        cam_labels = [f"cam_{c:03d}" for c in range(6)]
        grid_rep_idx = len(valid_frames) // 2

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

            grid_writer.write(grid)

            # Save one representative grid image
            if i == grid_rep_idx:
                cv2.imwrite(
                    os.path.join(args.output_dir, "representative_grid.png"),
                    grid,
                )

        grid_writer.release()

    print(f"\nDone. Output: {args.output_dir}")
    if not args.grid_only:
        print(f"Targets rendered: {target_list}")
        if args.stabilize:
            print("Body stabilization: ENABLED")


if __name__ == "__main__":
    main()
