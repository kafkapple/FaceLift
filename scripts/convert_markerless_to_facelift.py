#!/usr/bin/env python3
"""
Convert markerless_mouse_1_nerf dataset to FaceLift format.

This script:
1. Extracts frames from 6 camera videos at specified intervals
2. Applies segmentation masks to remove background
3. Converts camera parameters (K, R, T) to FaceLift opencv_cameras.json format
4. Resizes images to 512x512
5. Applies center alignment

Usage:
    python scripts/convert_markerless_to_facelift.py \
        --input_dir /home/joon/data/markerless_mouse_1_nerf \
        --output_dir data_mouse_full \
        --frame_interval 5 \
        --target_size 512

Notes:
    - Jump frames at 5900, 11800, 17700 are automatically skipped
    - Camera parameters are converted from OpenCV format (K, R, T) to w2c matrix
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


# Jump frames to skip (sudden camera/subject jumps)
JUMP_FRAMES = {5900, 11800, 17700}


def load_camera_params(pkl_path: str) -> List[Dict]:
    """Load camera parameters from pickle file."""
    with open(pkl_path, 'rb') as f:
        cam_data = pickle.load(f)
    return cam_data


def convert_camera_to_facelift(
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    orig_size: Tuple[int, int],
    target_size: int = 512
) -> Dict:
    """
    Convert camera parameters to FaceLift format.

    Args:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix (world to camera)
        T: 3x1 translation vector
        orig_size: Original image size (width, height)
        target_size: Target image size

    Returns:
        FaceLift camera dict with w2c matrix
    """
    orig_w, orig_h = orig_size

    # Scale intrinsics to target size
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    fx = K[0, 0] * scale_x
    fy = K[1, 1] * scale_y
    cx = K[0, 2] * scale_x
    cy = K[1, 2] * scale_y

    # Build world-to-camera matrix (4x4)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T

    return {
        "w": target_size,
        "h": target_size,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w2c": w2c.tolist()
    }


def find_object_bbox(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: int = 250
) -> Optional[Tuple[int, int, int, int]]:
    """Find bounding box of object using mask or non-white pixels."""
    if mask is not None:
        binary_mask = mask > 127
    else:
        if len(image.shape) == 3:
            binary_mask = np.any(image < threshold, axis=2)
        else:
            binary_mask = image < threshold

    coords = np.where(binary_mask)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    return (x_min, y_min, x_max, y_max)


def apply_center_transform(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    scale: float,
    output_size: int = 512,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Apply center transformation to image."""
    x_min, y_min, x_max, y_max = bbox
    obj_center_x = (x_min + x_max) / 2
    obj_center_y = (y_min + y_max) / 2

    h, w = image.shape[:2]

    # Scale image
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 0 and new_h > 0:
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        return image

    # Calculate offset to center object
    target_center = output_size / 2
    scaled_center_x = obj_center_x * scale
    scaled_center_y = obj_center_y * scale

    offset_x = int(target_center - scaled_center_x)
    offset_y = int(target_center - scaled_center_y)

    # Create output image
    if len(image.shape) == 3:
        output = np.full((output_size, output_size, image.shape[2]),
                        background_color[:image.shape[2]], dtype=np.uint8)
    else:
        output = np.full((output_size, output_size), background_color[0], dtype=np.uint8)

    # Calculate paste region
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(new_w, output_size - offset_x)
    src_y2 = min(new_h, output_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]

    return output


def process_frame(
    frame_idx: int,
    video_caps: List[cv2.VideoCapture],
    mask_caps: List[cv2.VideoCapture],
    cam_params: List[Dict],
    output_dir: Path,
    target_size: int = 512,
    target_object_ratio: float = 0.6,
    orig_size: Tuple[int, int] = (1152, 1024)
) -> bool:
    """
    Process a single frame from all cameras.

    Args:
        frame_idx: Frame index to extract
        video_caps: List of video capture objects for each camera
        mask_caps: List of mask video capture objects
        cam_params: Camera parameters list
        output_dir: Output directory for this sample
        target_size: Target image size
        target_object_ratio: Target object size ratio for centering
        orig_size: Original video frame size (width, height)

    Returns:
        True if successful
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    masks = []
    bboxes = []
    max_obj_size = 0

    # First pass: read all frames and find max object size
    for cam_idx in range(6):
        # Seek to frame
        video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = video_caps[cam_idx].read()
        ret_mask, mask = mask_caps[cam_idx].read()

        if not ret or not ret_mask:
            return False

        # Convert mask to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Find bounding box
        bbox = find_object_bbox(frame, mask)
        if bbox is not None:
            obj_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            max_obj_size = max(max_obj_size, obj_size)

        frames.append(frame)
        masks.append(mask)
        bboxes.append(bbox)

    if max_obj_size == 0:
        return False

    # Calculate scale from max object size
    # Use original size for scale calculation, then resize
    target_obj_size = min(orig_size) * target_object_ratio
    ref_scale = target_obj_size / max_obj_size

    # Scale factor for final resize to target_size
    resize_scale = target_size / min(orig_size)
    final_scale = ref_scale * resize_scale

    # Second pass: process each view
    camera_frames = []
    for cam_idx in range(6):
        frame = frames[cam_idx]
        mask = masks[cam_idx]
        bbox = bboxes[cam_idx]

        if bbox is None:
            # No object detected, skip this sample
            return False

        # Apply mask (white background)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        frame_masked = np.where(mask_3ch > 127, frame_rgb, 255)

        # Apply center transform with final scale
        centered = apply_center_transform(
            frame_masked.astype(np.uint8),
            bbox,
            final_scale,
            target_size,
            background_color=(255, 255, 255)
        )

        # Save image
        img_pil = Image.fromarray(centered)
        img_pil.save(images_dir / f"cam_{cam_idx:03d}.png")

        # Prepare camera data
        cam = cam_params[cam_idx]
        cam_dict = convert_camera_to_facelift(
            cam['K'], cam['R'], cam['T'],
            orig_size, target_size
        )
        cam_dict["file_path"] = f"images/cam_{cam_idx:03d}.png"
        cam_dict["view_id"] = cam_idx
        camera_frames.append(cam_dict)

    # Save camera parameters
    cameras_json = {"frames": camera_frames}
    with open(output_dir / "opencv_cameras.json", 'w') as f:
        json.dump(cameras_json, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert markerless_mouse_1_nerf to FaceLift format"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input markerless_mouse_1_nerf directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for FaceLift format data"
    )
    parser.add_argument(
        "--frame_interval", type=int, default=5,
        help="Frame sampling interval (default: 5)"
    )
    parser.add_argument(
        "--target_size", type=int, default=512,
        help="Target image size (default: 512)"
    )
    parser.add_argument(
        "--target_ratio", type=float, default=0.6,
        help="Target object size ratio (default: 0.6)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--start_sample", type=int, default=0,
        help="Starting sample index for output naming"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    print("Loading camera parameters...")
    cam_params = load_camera_params(input_dir / "new_cam.pkl")
    print(f"Loaded {len(cam_params)} camera configurations")

    # Open video files
    print("Opening video files...")
    video_dir = input_dir / "videos_undist"
    mask_dir = input_dir / "simpleclick_undist"

    video_caps = []
    mask_caps = []

    for cam_idx in range(6):
        video_path = video_dir / f"{cam_idx}.mp4"
        mask_path = mask_dir / f"{cam_idx}.mp4"

        video_cap = cv2.VideoCapture(str(video_path))
        mask_cap = cv2.VideoCapture(str(mask_path))

        if not video_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        if not mask_cap.isOpened():
            raise RuntimeError(f"Failed to open mask video: {mask_path}")

        video_caps.append(video_cap)
        mask_caps.append(mask_cap)

    # Get video info
    total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(video_caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(video_caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {total_frames} frames, {orig_w}x{orig_h}")

    # Generate frame indices (excluding jump frames)
    frame_indices = []
    for frame_idx in range(0, total_frames, args.frame_interval):
        if frame_idx not in JUMP_FRAMES:
            frame_indices.append(frame_idx)

    if args.max_samples:
        frame_indices = frame_indices[:args.max_samples]

    print(f"Processing {len(frame_indices)} frames with interval {args.frame_interval}")
    print(f"Jump frames excluded: {JUMP_FRAMES}")

    # Process frames
    sample_paths = []
    sample_idx = args.start_sample

    for frame_idx in tqdm(frame_indices, desc="Converting"):
        sample_name = f"sample_{sample_idx:06d}"
        sample_dir = output_dir / sample_name

        success = process_frame(
            frame_idx,
            video_caps,
            mask_caps,
            cam_params,
            sample_dir,
            args.target_size,
            args.target_ratio,
            (orig_w, orig_h)
        )

        if success:
            sample_paths.append(str(sample_dir.absolute()))
            sample_idx += 1

    # Cleanup
    for cap in video_caps + mask_caps:
        cap.release()

    # Create train/val split files
    num_samples = len(sample_paths)
    num_val = min(100, max(10, int(num_samples * 0.05)))
    num_train = num_samples - num_val

    train_paths = sample_paths[:num_train]
    val_paths = sample_paths[num_train:]

    with open(output_dir / "data_mouse_train.txt", 'w') as f:
        f.write('\n'.join(train_paths))

    with open(output_dir / "data_mouse_val.txt", 'w') as f:
        f.write('\n'.join(val_paths))

    print(f"\nDone!")
    print(f"Total samples: {num_samples}")
    print(f"Train: {num_train}, Val: {num_val}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
