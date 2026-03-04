#!/usr/bin/env python3
"""
FL vs PS Comparison Grid with Keypoint Overlay.

Extends the existing create_comparison_grid.py with:
  - 3D keypoint overlay (MAMMAL GT projected to each view)
  - H.264 video output (macOS compatible) across test frames
  - GS-LRM live rendering (not pre-rendered images)

Layout: 3 rows × 6 columns
  Row 1: GT images (6 cameras)
  Row 2: FaceLift GS-LRM renders (6 GT cameras)
  Row 3: Pose-Splatter renders (6 GT cameras, pre-rendered)

Usage:
    # Generate per-frame grids (image mode)
    python -m mouse_extensions.scripts.render_comparison_grid \
        --gt-dir ~/data/preprocessed/FaceLift_mouse/M5t2 \
        --fl-dir outputs/tier_comparison/gslrm_6view_test/samples \
        --ps-dir /tmp/ps_renders \
        --mammal-3d /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
        --cam-pkl ~/data/raw/markerless_mouse_1_nerf/new_cam.pkl \
        --output-dir outputs/comparison_grids

    # Generate H.264 video
    python -m mouse_extensions.scripts.render_comparison_grid \
        --gt-dir ~/data/preprocessed/FaceLift_mouse/M5t2 \
        --fl-dir outputs/tier_comparison/gslrm_6view_test/samples \
        --ps-dir /tmp/ps_renders \
        --mammal-3d ... --cam-pkl ... \
        --output-dir outputs/comparison_grids \
        --video --fps 20

Date: 2026-03-04
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Joint connection skeleton for mouse (22 keypoints)
MOUSE_SKELETON = [
    (0, 1), (1, 2), (2, 3),       # Spine: nose → head → neck → upper_back
    (3, 4), (4, 5),               # Spine continued: → mid_back → lower_back
    (5, 6),                        # Tail root
    (3, 7), (7, 8), (8, 9),       # Right forelimb
    (3, 10), (10, 11), (11, 12),  # Left forelimb
    (5, 13), (13, 14), (14, 15),  # Right hindlimb
    (5, 16), (16, 17), (17, 18),  # Left hindlimb
]

JOINT_COLORS = [
    (255, 0, 0),      # Nose
    (255, 50, 0),     # Head
    (255, 100, 0),    # Neck
    (255, 150, 0),    # Upper back
    (200, 200, 0),    # Mid back
    (100, 200, 0),    # Lower back
    (0, 200, 50),     # Tail root
    (0, 200, 200),    # RF shoulder
    (0, 100, 255),    # RF elbow
    (0, 50, 255),     # RF paw
    (100, 0, 255),    # LF shoulder
    (150, 0, 255),    # LF elbow
    (200, 0, 255),    # LF paw
    (255, 0, 200),    # RH hip
    (255, 0, 150),    # RH knee
    (255, 0, 100),    # RH paw
    (200, 50, 100),   # LH hip
    (200, 100, 100),  # LH knee
    (200, 150, 100),  # LH paw
]


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load image as RGB uint8 numpy array. Returns None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return np.array(bg)
    return np.array(img.convert("RGB"))


def draw_keypoints_on_image(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    img_size: int,
    draw_skeleton: bool = True,
    point_radius: int = 3,
    line_width: int = 1,
    alpha: float = 0.8,
) -> np.ndarray:
    """Draw 2D keypoints and skeleton on an image.

    Args:
        image: (H, W, 3) uint8 RGB
        keypoints_2d: (N_joints, 2) pixel coordinates
        img_size: Current image size (for bounds checking)
        draw_skeleton: Whether to draw bone connections
        point_radius: Keypoint circle radius
        line_width: Skeleton line width
        alpha: Overlay opacity

    Returns:
        Image with keypoints drawn
    """
    overlay = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(overlay)

    n_joints = len(keypoints_2d)

    # Draw skeleton lines first (behind points)
    if draw_skeleton:
        for j1, j2 in MOUSE_SKELETON:
            if j1 >= n_joints or j2 >= n_joints:
                continue
            x1, y1 = keypoints_2d[j1]
            x2, y2 = keypoints_2d[j2]
            # Check bounds
            if (0 <= x1 < img_size and 0 <= y1 < img_size
                    and 0 <= x2 < img_size and 0 <= y2 < img_size):
                color = tuple(
                    (np.array(JOINT_COLORS[j1 % len(JOINT_COLORS)])
                     + np.array(JOINT_COLORS[j2 % len(JOINT_COLORS)])) // 2
                )
                draw.line(
                    [(int(x1), int(y1)), (int(x2), int(y2))],
                    fill=color, width=line_width,
                )

    # Draw keypoint circles
    for j in range(n_joints):
        x, y = keypoints_2d[j]
        if 0 <= x < img_size and 0 <= y < img_size:
            color = JOINT_COLORS[j % len(JOINT_COLORS)]
            r = point_radius
            draw.ellipse(
                [int(x) - r, int(y) - r, int(x) + r, int(y) + r],
                fill=color, outline=(255, 255, 255), width=1,
            )

    # Blend with alpha
    result = np.array(overlay)
    blended = (alpha * result + (1 - alpha) * image).astype(np.uint8)
    return blended


def project_keypoints_to_view(
    keypoints_3d: np.ndarray,
    camera: dict,
    img_size: int,
    original_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Project 3D keypoints to 2D for a specific camera view.

    Args:
        keypoints_3d: (22, 3) world coordinates
        camera: Dict with K, R, t
        img_size: Target image size (for rescaling)
        original_size: Original camera resolution (w, h) for scaling intrinsics

    Returns:
        (22, 2) pixel coordinates in target resolution
    """
    from mouse_extensions.analysis.triangulation_analysis import project_3d_to_2d

    K = camera["K"].copy()

    # Scale intrinsics if needed
    if original_size is not None:
        scale_x = img_size / original_size[0]
        scale_y = img_size / original_size[1]
        K[0, 0] *= scale_x
        K[0, 2] *= scale_x
        K[1, 1] *= scale_y
        K[1, 2] *= scale_y

    return project_3d_to_2d(keypoints_3d, K, camera["R"], camera["t"])


def create_comparison_frame(
    gt_images: List[Optional[np.ndarray]],
    fl_images: List[Optional[np.ndarray]],
    ps_images: List[Optional[np.ndarray]],
    keypoints_2d_per_view: Optional[List[np.ndarray]] = None,
    frame_id: int = 0,
    img_size: int = 256,
    draw_kp: bool = True,
) -> np.ndarray:
    """Create a single 3×6 comparison grid frame.

    Args:
        gt_images: 6 GT view images (uint8 RGB)
        fl_images: 6 FL render images
        ps_images: 6 PS render images
        keypoints_2d_per_view: List of (22, 2) arrays per view (optional)
        frame_id: Frame number for annotation
        img_size: Per-image size in grid
        draw_kp: Whether to overlay keypoints

    Returns:
        Grid image as uint8 numpy array
    """
    num_views = 6
    num_rows = 3
    label_width = 80
    header_height = 30
    padding = 2

    grid_w = label_width + num_views * (img_size + padding) + padding
    grid_h = header_height + num_rows * (img_size + padding) + padding

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # Column headers
    for v in range(num_views):
        x = label_width + padding + v * (img_size + padding) + img_size // 2
        draw.text((x, header_height // 2), f"View {v}", fill=(0, 0, 0),
                  font=font_small, anchor="mm")

    # Row labels
    row_labels = ["GT", "FL", "PS"]
    row_colors = [(50, 50, 50), (0, 100, 200), (200, 50, 0)]
    all_images = [gt_images, fl_images, ps_images]

    for row_idx, (label, color, images) in enumerate(
            zip(row_labels, row_colors, all_images)):
        y_start = header_height + padding + row_idx * (img_size + padding)

        draw.text((label_width // 2, y_start + img_size // 2),
                  label, fill=color, font=font, anchor="mm")

        for v in range(num_views):
            x_start = label_width + padding + v * (img_size + padding)

            if images[v] is not None:
                img = Image.fromarray(images[v]).resize(
                    (img_size, img_size), Image.LANCZOS
                )
                img_np = np.array(img)

                # Overlay keypoints if available
                if draw_kp and keypoints_2d_per_view is not None:
                    # Scale keypoints to img_size
                    orig_h = images[v].shape[0]
                    orig_w = images[v].shape[1]
                    kp_scaled = keypoints_2d_per_view[v].copy()
                    kp_scaled[:, 0] *= img_size / orig_w
                    kp_scaled[:, 1] *= img_size / orig_h
                    img_np = draw_keypoints_on_image(img_np, kp_scaled, img_size)

                grid.paste(Image.fromarray(img_np), (x_start, y_start))
            else:
                draw.rectangle(
                    [x_start, y_start, x_start + img_size, y_start + img_size],
                    fill=(200, 200, 200), outline=(150, 150, 150),
                )
                draw.text(
                    (x_start + img_size // 2, y_start + img_size // 2),
                    "N/A", fill=(100, 100, 100), font=font_small, anchor="mm",
                )

    # Frame label
    draw.text((grid_w - 5, grid_h - 5), f"f{frame_id}", fill=(150, 150, 150),
              font=font_small, anchor="rb")

    return np.array(grid)


def frames_to_h264_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 20,
):
    """Write frames to H.264 MP4 video (macOS compatible).

    Args:
        frames: List of uint8 RGB numpy arrays (same size)
        output_path: Output .mp4 path
        fps: Frames per second
    """
    output_path = Path(output_path)
    h, w = frames[0].shape[:2]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        str(output_path),
    ]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    for frame in frames:
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        logger.error(f"ffmpeg failed: {stderr}")
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")

    logger.info(f"Video saved: {output_path} ({len(frames)} frames, {fps}fps)")


def load_gt_views(gt_dir: Path, frame_id: int) -> List[Optional[np.ndarray]]:
    """Load 6 GT view images for a frame."""
    fid = f"{frame_id:06d}"
    images = []
    for v in range(6):
        img_path = gt_dir / fid / "images" / f"cam_{v:03d}.png"
        images.append(load_image(img_path))
    return images


def load_fl_views(fl_dir: Path, frame_id: int) -> List[Optional[np.ndarray]]:
    """Load 6 FaceLift rendered views for a frame."""
    fid = f"{frame_id:06d}"
    images = []
    for v in range(6):
        # Try multiple naming conventions
        for pattern in [
            fl_dir / fid / f"render_view_{v:02d}.png",
            fl_dir / fid / f"cam_{v:03d}.png",
            fl_dir / fid / f"view_{v}.png",
        ]:
            if pattern.exists():
                images.append(load_image(pattern))
                break
        else:
            images.append(None)
    return images


def load_ps_views(ps_dir: Path, frame_id: int) -> List[Optional[np.ndarray]]:
    """Load 6 Pose-Splatter rendered views for a frame."""
    fid = f"{frame_id:06d}"
    images = []
    for v in range(6):
        for pattern in [
            ps_dir / fid / f"view_{v}.png",
            ps_dir / fid / f"cam_{v:03d}.png",
            ps_dir / f"frame_{fid}_view_{v}.png",
        ]:
            if pattern.exists():
                images.append(load_image(pattern))
                break
        else:
            images.append(None)
    return images


def main():
    parser = argparse.ArgumentParser(
        description="FL vs PS comparison grid with keypoint overlay"
    )
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="GT images dir (M5t2 preprocessed)")
    parser.add_argument("--fl-dir", type=str, required=True,
                        help="FL renders dir")
    parser.add_argument("--ps-dir", type=str, required=True,
                        help="PS renders dir")
    parser.add_argument("--mammal-3d", type=str, default=None,
                        help="MAMMAL 3D keypoints for overlay")
    parser.add_argument("--cam-pkl", type=str, default=None,
                        help="Camera params pkl for projection")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--frames", type=int, nargs="+",
                        default=None,
                        help="Specific frames (default: test set every 10)")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--video", action="store_true",
                        help="Generate H.264 video")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-keypoints", action="store_true",
                        help="Skip keypoint overlay")
    parser.add_argument("--original-cam-size", type=int, nargs=2,
                        default=[1152, 1024],
                        help="Original camera resolution (w h)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gt_dir = Path(args.gt_dir).expanduser()
    fl_dir = Path(args.fl_dir).expanduser()
    ps_dir = Path(args.ps_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine frames
    if args.frames:
        frame_ids = args.frames
    else:
        # Test set: 3240-3599, every 10 frames
        frame_ids = list(range(3240, 3600, 10))

    # Load keypoint data if available
    kp3d = None
    cameras = None
    if not args.no_keypoints and args.mammal_3d and args.cam_pkl:
        from mouse_extensions.analysis.triangulation_analysis import (
            load_mammal_3d, load_raw_cameras,
        )
        logger.info("Loading MAMMAL 3D keypoints and cameras...")
        kp3d = load_mammal_3d(Path(args.mammal_3d).expanduser())
        cameras = load_raw_cameras(Path(args.cam_pkl).expanduser())

    all_grid_frames = []

    for fi, frame_id in enumerate(frame_ids):
        if fi % 10 == 0:
            logger.info(f"Processing frame {frame_id} ({fi+1}/{len(frame_ids)})...")

        gt_imgs = load_gt_views(gt_dir, frame_id)
        fl_imgs = load_fl_views(fl_dir, frame_id)
        ps_imgs = load_ps_views(ps_dir, frame_id)

        # Project keypoints to each view
        kp_per_view = None
        if kp3d is not None and cameras is not None and frame_id < len(kp3d):
            kp_per_view = []
            for v in range(6):
                if v < len(cameras):
                    pts_2d = project_keypoints_to_view(
                        kp3d[frame_id],
                        cameras[v],
                        img_size=args.img_size,
                        original_size=tuple(args.original_cam_size),
                    )
                    kp_per_view.append(pts_2d)
                else:
                    kp_per_view.append(np.zeros((22, 2)))

        grid_frame = create_comparison_frame(
            gt_images=gt_imgs,
            fl_images=fl_imgs,
            ps_images=ps_imgs,
            keypoints_2d_per_view=kp_per_view,
            frame_id=frame_id,
            img_size=args.img_size,
            draw_kp=not args.no_keypoints,
        )

        all_grid_frames.append(grid_frame)

        # Save individual frame
        if not args.video:
            frame_path = output_dir / f"comparison_{frame_id:06d}.png"
            Image.fromarray(grid_frame).save(str(frame_path), quality=95)

    # Generate video
    if args.video and all_grid_frames:
        video_path = output_dir / "fl_vs_ps_comparison.mp4"
        frames_to_h264_video(all_grid_frames, video_path, fps=args.fps)

    logger.info(f"Done. {len(all_grid_frames)} frames processed → {output_dir}")


if __name__ == "__main__":
    main()
