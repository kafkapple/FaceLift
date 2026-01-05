#!/usr/bin/env python3
"""
Center-align all views in mouse dataset for MVDiffusion training.

This script:
1. For each sample, computes center alignment parameters from View 0 (reference)
2. Applies the SAME transformation to ALL views (critical for consistency)
3. Saves centered images while preserving camera parameters

Key insight: Camera parameters don't need adjustment because:
- MVDiffusion learns with centered images
- GS-LRM uses fixed camera poses for MVDiffusion outputs
- The transformation is applied consistently across all views

Usage:
    python scripts/preprocess_center_align_all_views.py \
        --input_dir data_mouse \
        --output_dir data_mouse_centered \
        --num_samples 100  # For quick test
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def find_object_bbox(image: np.ndarray, alpha: Optional[np.ndarray] = None, threshold: int = 250) -> Optional[Tuple[int, int, int, int]]:
    """
    Find bounding box of object using alpha channel or non-white pixels.

    Args:
        image: RGB image as numpy array
        alpha: Optional alpha channel (if available, uses this for segmentation)
        threshold: Pixels with all channels > threshold are considered background

    Returns:
        (x_min, y_min, x_max, y_max) or None if no object found
    """
    # Prefer alpha channel if available
    if alpha is not None:
        mask = alpha > 0  # Non-transparent pixels
    else:
        # Fallback: Check if pixel is NOT white (background)
        if len(image.shape) == 3:
            mask = np.any(image < threshold, axis=2)
        else:
            mask = image < threshold

    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    return (x_min, y_min, x_max, y_max)


def compute_center_transform(
    bbox: Tuple[int, int, int, int],
    image_size: int = 512,
    target_object_ratio: float = 0.7,
) -> Dict:
    """
    Compute transformation parameters to center object.

    Args:
        bbox: Object bounding box (x_min, y_min, x_max, y_max)
        image_size: Target image size
        target_object_ratio: Target ratio of object size to image size

    Returns:
        Transform parameters dict
    """
    x_min, y_min, x_max, y_max = bbox

    # Object center and size
    obj_center_x = (x_min + x_max) / 2
    obj_center_y = (y_min + y_max) / 2
    obj_width = x_max - x_min
    obj_height = y_max - y_min
    obj_size = max(obj_width, obj_height)

    # Calculate scale to fit object
    target_size = image_size * target_object_ratio
    scale = target_size / obj_size if obj_size > 0 else 1.0

    # Target center (image center)
    target_center_x = image_size / 2
    target_center_y = image_size / 2

    return {
        "scale": scale,
        "obj_center": (obj_center_x, obj_center_y),
        "target_center": (target_center_x, target_center_y),
        "obj_size": obj_size,
        "bbox": bbox,
    }


def apply_center_transform(
    image: np.ndarray,
    transform: Dict,
    output_size: int = 512,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Apply center transformation to image.

    Args:
        image: Input image
        transform: Transform parameters from compute_center_transform
        output_size: Output image size
        background_color: Background fill color

    Returns:
        Centered image
    """
    scale = transform["scale"]
    obj_center_x, obj_center_y = transform["obj_center"]
    target_center_x, target_center_y = transform["target_center"]

    h, w = image.shape[:2]

    # Scale image
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w > 0 and new_h > 0:
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    else:
        scaled_image = image
        new_w, new_h = w, h
        scale = 1.0

    # Calculate offset to center object
    scaled_center_x = obj_center_x * scale
    scaled_center_y = obj_center_y * scale

    offset_x = int(target_center_x - scaled_center_x)
    offset_y = int(target_center_y - scaled_center_y)

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

    # Paste
    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]

    return output


def process_sample(
    input_dir: Path,
    output_dir: Path,
    num_views: int = 6,
    target_object_ratio: float = 0.7,
) -> bool:
    """
    Process a single sample: center-align each view independently.

    Key change: Scale is computed from the MAXIMUM object size across ALL views
    to guarantee no clipping in any view.

    Args:
        input_dir: Sample input directory
        output_dir: Sample output directory
        num_views: Number of views
        target_object_ratio: Target object size ratio

    Returns:
        True if successful
    """
    images_in = input_dir / "images"
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # First pass: collect all view data and find max object size
    view_data = []  # List of (img_path, rgb, alpha, bbox)
    max_obj_size = 0
    image_size = None

    for i in range(num_views):
        for pattern in [f"cam_{i:03d}.png", f"{i:02d}.png"]:
            img_path = images_in / pattern
            if img_path.exists():
                img = Image.open(img_path)
                if image_size is None:
                    image_size = img.size[0]

                if img.mode == "RGBA":
                    img_array = np.array(img)
                    rgb = img_array[:, :, :3]
                    alpha = img_array[:, :, 3]
                else:
                    rgb = np.array(img.convert("RGB"))
                    alpha = None

                bbox = find_object_bbox(rgb, alpha=alpha)
                if bbox is not None:
                    obj_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                    max_obj_size = max(max_obj_size, obj_size)

                view_data.append((i, img_path, rgb, alpha, bbox))
                break

    if not view_data or max_obj_size == 0:
        # No valid views found, just copy
        shutil.copytree(images_in, images_out, dirs_exist_ok=True)
        return True

    # Compute scale from MAX object size across all views (guarantees no clipping)
    target_size = image_size * target_object_ratio
    ref_scale = target_size / max_obj_size

    # Second pass: apply transformation to each view (reuse data from first pass)
    for i, img_path, rgb, alpha, bbox in view_data:
        if bbox is None:
            # No object, save original
            img = Image.open(img_path)
            img.save(images_out / f"cam_{i:03d}.png")
            continue

        # Compute transform for THIS view (use ref_scale from max size, center on this view's object)
        transform = {
            "scale": ref_scale,
            "obj_center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
            "target_center": (image_size / 2, image_size / 2),
        }

        # Handle RGBA
        if alpha is not None:
            # Apply transform to RGB
            centered_rgb = apply_center_transform(rgb, transform, image_size)

            # Apply transform to alpha
            centered_alpha = apply_center_transform(
                alpha, transform, image_size,
                background_color=(0, 0, 0)  # Transparent background
            )

            # Combine
            centered = np.zeros((centered_rgb.shape[0], centered_rgb.shape[1], 4), dtype=np.uint8)
            centered[:, :, :3] = centered_rgb
            centered[:, :, 3] = centered_alpha

            Image.fromarray(centered, mode="RGBA").save(images_out / f"cam_{i:03d}.png")
        else:
            centered = apply_center_transform(rgb, transform, image_size)
            Image.fromarray(centered).save(images_out / f"cam_{i:03d}.png")

    # Copy camera parameters (unchanged)
    camera_json = input_dir / "opencv_cameras.json"
    if camera_json.exists():
        shutil.copy(camera_json, output_dir / "opencv_cameras.json")

    return True


def main():
    parser = argparse.ArgumentParser(description="Center-align all views in mouse dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Input data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples (None=all)")
    parser.add_argument("--num_views", type=int, default=6, help="Number of views per sample")
    parser.add_argument("--target_ratio", type=float, default=0.7, help="Target object size ratio")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sample directories
    sample_dirs = sorted(input_dir.glob("sample_*"))
    if args.num_samples:
        sample_dirs = sample_dirs[:args.num_samples]

    print(f"Processing {len(sample_dirs)} samples")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target object ratio: {args.target_ratio}")

    # Process samples
    success_count = 0
    for sample_dir in tqdm(sample_dirs, desc="Center-aligning"):
        sample_name = sample_dir.name
        sample_output = output_dir / sample_name

        if process_sample(sample_dir, sample_output, args.num_views, args.target_ratio):
            success_count += 1

    print(f"\nDone! Processed {success_count}/{len(sample_dirs)} samples")

    # Copy train/val split files
    for split_file in ["data_mouse_train.txt", "data_mouse_val.txt"]:
        src = input_dir / split_file
        if src.exists():
            # Update paths in split file
            with open(src, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    # Replace input_dir with output_dir in paths
                    new_line = line.replace(str(input_dir), str(output_dir))
                    new_lines.append(new_line)

            dst = output_dir / split_file
            with open(dst, 'w') as f:
                f.write('\n'.join(new_lines))
            print(f"Created {dst}")


if __name__ == "__main__":
    main()
