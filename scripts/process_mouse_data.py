#!/usr/bin/env python3
"""
Mouse Video Data Preprocessing Script for FaceLift

This script processes multi-view mouse video data into the format required by FaceLift:
1. Extracts synchronized frames from 6-view videos
2. Applies background masks
3. Converts camera parameters to FaceLift format (opencv_cameras.json)
4. Saves images as 512x512 RGBA PNGs

Usage:
    python scripts/process_mouse_data.py \
        --video_dir /path/to/videos \
        --meta_dir /path/to/masks_and_cameras \
        --output_dir /path/to/output \
        --num_samples 2000 \
        --image_size 512

Author: Claude Code (AI-assisted)
Date: 2024-12-04
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_camera_params(meta_dir: str, num_views: int = 6) -> List[Dict]:
    """
    Load camera parameters from MAMMAL-style pickle or NeRF transforms.json.

    Args:
        meta_dir: Directory containing camera calibration files
        num_views: Number of camera views

    Returns:
        List of camera parameter dictionaries
    """
    cameras = []

    # Try loading from pickle (MAMMAL format)
    cam_pkl_path = os.path.join(meta_dir, "new_cam.pkl")
    if os.path.exists(cam_pkl_path):
        with open(cam_pkl_path, "rb") as f:
            cam_data = pickle.load(f)

        # Handle both list and dict formats
        if isinstance(cam_data, list):
            # MAMMAL format: list of camera dicts
            for i in range(min(num_views, len(cam_data))):
                cameras.append(cam_data[i])
        elif isinstance(cam_data, dict):
            # Dict format with integer or string keys
            for i in range(num_views):
                if i in cam_data:
                    cameras.append(cam_data[i])
                elif str(i) in cam_data:
                    cameras.append(cam_data[str(i)])

        if cameras:
            print(f"Loaded {len(cameras)} cameras from {cam_pkl_path}")
            # Debug: print camera info
            for i, cam in enumerate(cameras):
                R = np.array(cam.get("R", []))
                T = np.array(cam.get("T", [])).flatten()
                if R.size > 0 and T.size > 0:
                    cam_pos = -R.T @ T
                    dist = np.linalg.norm(cam_pos)
                    elev = np.degrees(np.arcsin(cam_pos[2] / dist)) if dist > 0 else 0
                    print(f"  Camera {i}: dist={dist:.1f}, elevation={elev:.1f}°")
            return cameras

    # Try loading from transforms.json (NeRF format)
    transforms_path = os.path.join(meta_dir, "transforms.json")
    if os.path.exists(transforms_path):
        with open(transforms_path, "r") as f:
            transforms = json.load(f)

        # Extract camera params from NeRF format
        fl_x = transforms.get("fl_x", transforms.get("camera_angle_x", None))
        fl_y = transforms.get("fl_y", fl_x)
        cx = transforms.get("cx", None)
        cy = transforms.get("cy", None)
        w = transforms.get("w", 512)
        h = transforms.get("h", 512)

        # If camera_angle_x is provided, compute focal length
        if "camera_angle_x" in transforms and fl_x is not None:
            fl_x = 0.5 * w / np.tan(0.5 * fl_x)
            fl_y = fl_x

        for frame in transforms.get("frames", [])[:num_views]:
            transform_matrix = np.array(frame["transform_matrix"])
            # NeRF uses c2w, we need to convert to R, T format
            c2w = transform_matrix
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3:]

            K = np.array([
                [fl_x, 0, cx if cx else w / 2],
                [0, fl_y, cy if cy else h / 2],
                [0, 0, 1]
            ])

            cameras.append({"R": R, "T": T, "K": K})

        if cameras:
            print(f"Loaded {len(cameras)} cameras from {transforms_path}")
            return cameras

    # Fallback: Generate default circular camera arrangement
    print("Warning: No camera calibration found, generating default cameras")
    cameras = generate_default_cameras(num_views)
    return cameras


def generate_default_cameras(
    num_views: int = 6,
    radius: float = 2.7,
    elevation_deg: float = 20.0,
    fov_deg: float = 50.0,
    image_size: int = 512
) -> List[Dict]:
    """
    Generate default camera parameters in a circular arrangement.

    Uses FaceLift-standard settings: radius=2.7, elevation=20°, FOV=50°.

    Args:
        num_views: Number of cameras
        radius: Distance from origin (FaceLift default: 2.7)
        elevation_deg: Camera elevation angle in degrees (FaceLift default: 20)
        fov_deg: Field of view in degrees (FaceLift default: 50)
        image_size: Image width/height

    Returns:
        List of camera dictionaries with R, T, K
    """
    cameras = []
    fx = fy = 0.5 * image_size / np.tan(0.5 * np.deg2rad(fov_deg))
    cx = cy = image_size / 2

    elevation_rad = np.deg2rad(elevation_deg)
    up_vector = np.array([0, 0, 1])

    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views

        # Camera position with elevation (matching FaceLift get_turntable_cameras)
        z = radius * np.sin(elevation_rad)
        base = radius * np.cos(elevation_rad)
        x = base * np.cos(azimuth)
        y = base * np.sin(azimuth)
        cam_pos = np.array([x, y, z])

        # Camera orientation (looking at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Rotation matrix (world to camera) - OpenCV convention
        R = np.stack([right, -up, forward], axis=0)
        T = -R @ cam_pos.reshape(3, 1)

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])

        cameras.append({"R": R, "T": T, "K": K})

    print(f"Generated {num_views} default cameras: radius={radius}, "
          f"elevation={elevation_deg}°, fov={fov_deg}°")

    return cameras


def convert_to_facelift_format(
    cameras: List[Dict],
    image_size: int = 512,
    target_distance: float = 2.7,
    target_fov_deg: float = 50.0
) -> Dict:
    """
    Convert MAMMAL-style cameras to FaceLift opencv_cameras.json format.

    IMPORTANT: FaceLift model was trained with normalized cameras:
    - Camera distance: ~2.7 units
    - FOV: ~50 degrees
    - fx ≈ 549 for 512px images

    MAMMAL cameras have distance 246-414 units, so we must normalize!

    Args:
        cameras: List of camera dicts with R, T, K
        image_size: Target image size
        target_distance: Target camera distance (FaceLift default: 2.7)
        target_fov_deg: Target field of view in degrees (FaceLift default: 50)

    Returns:
        FaceLift-style camera dict ready for JSON serialization
    """
    frames = []

    # First pass: compute average camera distance for normalization
    distances = []
    for cam in cameras:
        R = np.array(cam["R"])
        T = np.array(cam["T"]).flatten()
        # Camera position in world coords: -R^T @ T
        cam_pos = -R.T @ T
        dist = np.linalg.norm(cam_pos)
        distances.append(dist)

    avg_distance = np.mean(distances) if distances else 1.0
    scale_factor = target_distance / avg_distance if avg_distance > 0 else 1.0

    print(f"Camera normalization: avg_dist={avg_distance:.1f}, "
          f"target={target_distance}, scale={scale_factor:.6f}")

    # Target intrinsics for FaceLift
    target_fx = 0.5 * image_size / np.tan(0.5 * np.deg2rad(target_fov_deg))
    target_fy = target_fx
    target_cx = image_size / 2
    target_cy = image_size / 2

    for i, cam in enumerate(cameras):
        R = np.array(cam["R"])
        T = np.array(cam["T"]).flatten()

        # Normalize camera distance by scaling translation
        # w2c = [R | T], where T = -R @ cam_pos
        # Scaling cam_pos scales T proportionally
        T_normalized = T * scale_factor

        # Build w2c matrix (world-to-camera 4x4)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T_normalized

        # Use FaceLift-standard intrinsics for consistency
        # The model was trained with specific FOV, so we match that
        fx, fy = target_fx, target_fy
        cx, cy = target_cx, target_cy

        frame = {
            "w": image_size,
            "h": image_size,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "w2c": w2c.tolist(),
            "file_path": f"images/cam_{i:03d}.png",
            "view_id": i
        }
        frames.append(frame)

        if i == 0:
            # Debug: print first camera info after normalization
            cam_pos_norm = -R.T @ T_normalized
            dist_norm = np.linalg.norm(cam_pos_norm)
            elev = np.degrees(np.arcsin(cam_pos_norm[2] / dist_norm)) if dist_norm > 0 else 0
            print(f"  Camera 0 after norm: dist={dist_norm:.2f}, elev={elev:.1f}°, "
                  f"fx={fx:.1f}, fov={np.degrees(2*np.arctan(image_size/(2*fx))):.1f}°")

    return {"frames": frames}


def extract_video_frames(
    video_paths: List[str],
    frame_indices: List[int],
    mask_paths: Optional[List[str]] = None
) -> List[List[np.ndarray]]:
    """
    Extract synchronized frames from multiple video files.

    Args:
        video_paths: List of video file paths (one per view)
        frame_indices: List of frame indices to extract
        mask_paths: Optional list of mask video paths

    Returns:
        List of frame lists (one per sample, containing all views)
    """
    num_views = len(video_paths)

    # Open video captures
    caps = []
    mask_caps = []
    for i, vp in enumerate(video_paths):
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {vp}")
        caps.append(cap)

        if mask_paths and mask_paths[i]:
            mcap = cv2.VideoCapture(mask_paths[i])
            if mcap.isOpened():
                mask_caps.append(mcap)
            else:
                mask_caps.append(None)
        else:
            mask_caps.append(None)

    all_frames = []

    for frame_idx in tqdm(frame_indices, desc="Extracting frames"):
        views = []
        for v in range(num_views):
            # Seek to frame
            caps[v].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = caps[v].read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_idx} from view {v}")
                frame = np.zeros((512, 512, 3), dtype=np.uint8)

            # Read mask if available
            mask = None
            if mask_caps[v] is not None:
                mask_caps[v].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_m, mask_frame = mask_caps[v].read()
                if ret_m:
                    mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                    mask = (mask > 127).astype(np.uint8) * 255

            views.append((frame, mask))

        all_frames.append(views)

    # Release captures
    for cap in caps:
        cap.release()
    for mcap in mask_caps:
        if mcap is not None:
            mcap.release()

    return all_frames


def process_frame_with_mask(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    target_size: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Process a frame: apply mask, resize, convert to RGBA.

    Args:
        frame: BGR frame from OpenCV
        mask: Binary mask (255 = foreground, 0 = background)
        target_size: Output size (square)
        bg_color: Background color RGB

    Returns:
        RGBA image as numpy array (H, W, 4)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    # Create alpha channel from mask
    if mask is not None:
        # Resize mask to match frame if needed
        if mask.shape[:2] != frame_rgb.shape[:2]:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = mask
    else:
        alpha = np.ones((h, w), dtype=np.uint8) * 255

    # Compute center crop to make square
    if h != w:
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        frame_rgb = frame_rgb[start_h:start_h+min_dim, start_w:start_w+min_dim]
        alpha = alpha[start_h:start_h+min_dim, start_w:start_w+min_dim]

    # Resize to target size
    frame_rgb = cv2.resize(frame_rgb, (target_size, target_size),
                           interpolation=cv2.INTER_LANCZOS4)
    alpha = cv2.resize(alpha, (target_size, target_size),
                       interpolation=cv2.INTER_NEAREST)

    # Create RGBA image
    rgba = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame_rgb
    rgba[:, :, 3] = alpha

    return rgba


def find_video_files(video_dir: str) -> List[str]:
    """
    Find video files in directory, sorted by view index.

    Args:
        video_dir: Directory to search

    Returns:
        List of video file paths sorted by view number
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))

    # Sort by filename (assuming naming like 0.mp4, 1.mp4, etc.)
    video_files = sorted(video_files, key=lambda x: x.stem)

    return [str(f) for f in video_files]


def compute_sampling_indices(
    total_frames: int,
    num_samples: int,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    skip_start_percent: float = 0.05,
    skip_end_percent: float = 0.05
) -> List[int]:
    """
    Compute frame indices to sample for diverse poses.

    Args:
        total_frames: Total number of frames in video
        num_samples: Desired number of samples
        start_frame: Starting frame (inclusive)
        end_frame: Ending frame (exclusive), None for all
        skip_start_percent: Skip this percentage of frames at start
        skip_end_percent: Skip this percentage of frames at end

    Returns:
        List of frame indices to extract
    """
    if end_frame is None:
        end_frame = total_frames

    # Skip frames at beginning and end (often static setup)
    skip_start = int(total_frames * skip_start_percent)
    skip_end = int(total_frames * skip_end_percent)

    effective_start = max(start_frame, skip_start)
    effective_end = min(end_frame, total_frames - skip_end)
    effective_range = effective_end - effective_start

    if effective_range <= 0:
        effective_start = start_frame
        effective_end = end_frame
        effective_range = effective_end - effective_start

    if num_samples >= effective_range:
        # Return all frames in range
        return list(range(effective_start, effective_end))

    # Uniform sampling
    step = effective_range / num_samples
    indices = [int(effective_start + i * step) for i in range(num_samples)]

    return indices


def create_train_val_split(
    sample_dirs: List[str],
    val_ratio: float = 0.1,
    output_dir: str = "."
) -> Tuple[str, str]:
    """
    Create train/val split files.

    Args:
        sample_dirs: List of sample directory paths
        val_ratio: Ratio of samples for validation
        output_dir: Directory to save split files

    Returns:
        Tuple of (train_file_path, val_file_path)
    """
    np.random.seed(42)
    indices = np.random.permutation(len(sample_dirs))

    val_size = int(len(sample_dirs) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [sample_dirs[i] for i in train_indices]
    val_samples = [sample_dirs[i] for i in val_indices]

    train_file = os.path.join(output_dir, "data_mouse_train.txt")
    val_file = os.path.join(output_dir, "data_mouse_val.txt")

    with open(train_file, "w") as f:
        f.write("\n".join(train_samples))

    with open(val_file, "w") as f:
        f.write("\n".join(val_samples))

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(
        description="Process mouse video data for FaceLift training"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True,
        help="Directory containing multi-view videos"
    )
    parser.add_argument(
        "--meta_dir", type=str, required=True,
        help="Directory containing masks and camera parameters"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--num_samples", type=int, default=2000,
        help="Number of frames to sample (default: 2000)"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Output image size (default: 512)"
    )
    parser.add_argument(
        "--num_views", type=int, default=6,
        help="Number of camera views (default: 6)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--bg_color", type=str, default="white",
        choices=["white", "black", "gray"],
        help="Background color (default: white)"
    )
    parser.add_argument(
        "--target_distance", type=float, default=2.7,
        help="Target camera distance for normalization (FaceLift default: 2.7)"
    )
    parser.add_argument(
        "--target_fov", type=float, default=50.0,
        help="Target field of view in degrees (FaceLift default: 50)"
    )

    args = parser.parse_args()

    # Setup background color
    bg_colors = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128)
    }
    bg_color = bg_colors[args.bg_color]

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Find video files
    print(f"Searching for videos in: {args.video_dir}")
    video_files = find_video_files(os.path.join(args.video_dir, "videos_undist"))
    if not video_files:
        video_files = find_video_files(args.video_dir)

    if len(video_files) < args.num_views:
        print(f"Warning: Found only {len(video_files)} videos, expected {args.num_views}")
        print("Looking for individual camera folders...")
        video_files = []
        for i in range(args.num_views):
            pattern = os.path.join(args.video_dir, f"**/cam{i}*.mp4")
            matches = list(Path(args.video_dir).glob(f"**/{i}.mp4"))
            if matches:
                video_files.append(str(matches[0]))

    print(f"Found {len(video_files)} video files:")
    for vf in video_files:
        print(f"  - {vf}")

    # Find mask files (if available)
    mask_dir = os.path.join(args.meta_dir, "simpleclick_undist")
    mask_files = []
    if os.path.exists(mask_dir):
        mask_files = find_video_files(mask_dir)
        print(f"Found {len(mask_files)} mask videos")

    # Load camera parameters
    print("Loading camera parameters...")
    cameras = load_camera_params(args.meta_dir, args.num_views)
    if len(cameras) < args.num_views:
        print(f"Warning: Only {len(cameras)} cameras loaded, generating defaults")
        cameras = generate_default_cameras(args.num_views, image_size=args.image_size)

    # Convert to FaceLift format with distance normalization
    print(f"\nNormalizing cameras to FaceLift format...")
    print(f"  Target distance: {args.target_distance}")
    print(f"  Target FOV: {args.target_fov}°")
    facelift_cameras = convert_to_facelift_format(
        cameras,
        image_size=args.image_size,
        target_distance=args.target_distance,
        target_fov_deg=args.target_fov
    )

    # Get total frame count from first video
    if video_files:
        cap = cv2.VideoCapture(video_files[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Video info: {total_frames} frames @ {fps:.1f} fps")
    else:
        print("Error: No video files found!")
        return

    # Compute sampling indices
    frame_indices = compute_sampling_indices(
        total_frames, args.num_samples
    )
    print(f"Sampling {len(frame_indices)} frames from video")

    # Extract and process frames
    sample_dirs = []

    if video_files:
        print("Extracting frames from videos...")
        all_frames = extract_video_frames(
            video_files[:args.num_views],
            frame_indices,
            mask_files[:args.num_views] if mask_files else None
        )

        print("Processing and saving samples...")
        for sample_idx, views in enumerate(tqdm(all_frames, desc="Saving samples")):
            sample_name = f"sample_{sample_idx:06d}"
            sample_dir = os.path.join(args.output_dir, sample_name)
            images_dir = os.path.join(sample_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Process each view
            for view_idx, (frame, mask) in enumerate(views):
                rgba = process_frame_with_mask(
                    frame, mask, args.image_size, bg_color
                )

                # Save as PNG with alpha channel
                img = Image.fromarray(rgba, mode="RGBA")
                img_path = os.path.join(images_dir, f"cam_{view_idx:03d}.png")
                img.save(img_path)

            # Update camera file paths and save
            sample_cameras = facelift_cameras.copy()
            sample_cameras["id"] = sample_name

            cam_json_path = os.path.join(sample_dir, "opencv_cameras.json")
            with open(cam_json_path, "w") as f:
                json.dump(sample_cameras, f, indent=2)

            sample_dirs.append(sample_dir)

    # Create train/val split
    print("Creating train/val split...")
    train_file, val_file = create_train_val_split(
        sample_dirs, args.val_ratio, args.output_dir
    )

    print(f"\nProcessing complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Total samples: {len(sample_dirs)}")
    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")


if __name__ == "__main__":
    main()
