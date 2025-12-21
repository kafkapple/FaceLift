#!/usr/bin/env python3
"""
Uniform Scale Preprocessing for Mouse-FaceLift

Problem:
  - FaceLift Human: all cameras at fixed distance 2.7, human at origin
  - Mouse: cameras at varying distances, mouse moves around in large space
  - Result: object appears at different sizes in each view (NOT correlated with camera distance!)

Solution (Image-based approach):
  - Measure actual object size in each image using alpha mask (bounding box)
  - Scale each image to make the object fill target_ratio (e.g., 60%) of the frame
  - This is a purely image-based approach, independent of camera parameters

Why NOT use camera distance?
  - Mouse is NOT at the origin (unlike FaceLift human data)
  - Camera-to-origin distance does NOT correlate with object appearance size
  - Only the actual image content tells us how big the object appears

Usage:
    python scripts/preprocess_uniform_scale.py \
        --input_dir data_mouse_centered \
        --output_dir data_mouse_uniform \
        --target_ratio 0.6
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


def get_object_bbox_from_alpha(alpha: np.ndarray, threshold: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box of object from alpha mask.

    Args:
        alpha: Alpha channel (H, W) with values 0-255
        threshold: Minimum alpha value to consider as object

    Returns:
        (x1, y1, x2, y2) bounding box, or None if no object found
    """
    mask = alpha > threshold
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return int(x1), int(y1), int(x2), int(y2)


def get_object_size_ratio(alpha: np.ndarray, threshold: int = 10) -> float:
    """
    Get the ratio of object area to image area using pixel count.

    This method counts actual pixels instead of using bounding box,
    which is more robust for objects with thin appendages like mouse tails.

    Args:
        alpha: Alpha channel (H, W) with values 0-255
        threshold: Minimum alpha value to consider as object

    Returns:
        Square root of (pixel_count / total_pixels) - represents "effective size"
        Using sqrt because we want to scale linearly with size, not area.
    """
    mask = alpha > threshold
    if not mask.any():
        return 0.0

    pixel_count = np.sum(mask)
    h, w = alpha.shape
    total_pixels = h * w

    # Use sqrt to convert area ratio to linear size ratio
    # This way, if we scale by 2x, the area becomes 4x but pixel_ratio becomes 2x
    area_ratio = pixel_count / total_pixels
    size_ratio = np.sqrt(area_ratio)

    return size_ratio


def scale_image_center(
    image: np.ndarray,
    scale: float,
    output_size: int = 512,
    background_color: Tuple[int, ...] = (255, 255, 255)
) -> np.ndarray:
    """
    Scale image while keeping center fixed.

    Args:
        image: Input image (RGB or RGBA)
        scale: Scale factor (>1 = zoom in, <1 = zoom out)
        output_size: Output image size
        background_color: Background fill color

    Returns:
        Scaled image with same dimensions
    """
    h, w = image.shape[:2]

    # New dimensions after scaling
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w <= 0 or new_h <= 0:
        return image

    # Resize image
    if len(image.shape) == 3:
        interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR

    scaled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Create output canvas
    n_channels = image.shape[2] if len(image.shape) == 3 else 1
    if len(image.shape) == 3:
        output = np.full((output_size, output_size, n_channels),
                         background_color[:n_channels], dtype=np.uint8)
    else:
        output = np.full((output_size, output_size), background_color[0], dtype=np.uint8)

    # Calculate paste position (center-aligned)
    scaled_center_x = new_w / 2
    scaled_center_y = new_h / 2

    # Offset to place scaled center at output center
    offset_x = int(output_size / 2 - scaled_center_x)
    offset_y = int(output_size / 2 - scaled_center_y)

    # Calculate crop/paste regions
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(new_w, output_size - offset_x)
    src_y2 = min(new_h, output_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Paste scaled image
    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]

    return output


def process_sample(
    input_dir: Path,
    output_dir: Path,
    target_ratio: float = 0.6,
    output_size: int = 512,
) -> Optional[Dict]:
    """
    Process a single sample: scale images based on actual object size in image.

    Args:
        input_dir: Sample input directory
        output_dir: Sample output directory
        target_ratio: Target object size ratio (0.6 = 60% of frame)
        output_size: Output image size

    Returns:
        Processing info dict, or None if failed
    """
    cameras_path = input_dir / "opencv_cameras.json"
    images_in = input_dir / "images"

    if not cameras_path.exists():
        return None

    if not images_in.exists():
        return None

    # Load camera data
    with open(cameras_path, 'r') as f:
        camera_data = json.load(f)

    frames = camera_data['frames']
    n_views = len(frames)

    # First pass: measure object size in each image
    view_info = []
    for i, frame in enumerate(frames):
        # Find image file
        img_path = None
        for pattern in [f"cam_{i:03d}.png", f"{i:02d}.png"]:
            candidate = images_in / pattern
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            view_info.append({
                'view_idx': i,
                'img_path': None,
                'current_ratio': 0.0,
                'scale_factor': 1.0,
            })
            continue

        # Load image and measure object size
        img = Image.open(img_path)

        if img.mode == 'RGBA':
            alpha = np.array(img)[:, :, 3]
            current_ratio = get_object_size_ratio(alpha)
        else:
            # If no alpha, assume object fills frame
            current_ratio = 0.8  # Conservative estimate

        # Calculate scale factor to achieve target_ratio
        if current_ratio > 0:
            scale_factor = target_ratio / current_ratio
        else:
            scale_factor = 1.0

        view_info.append({
            'view_idx': i,
            'img_path': img_path,
            'current_ratio': current_ratio,
            'scale_factor': scale_factor,
        })

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Second pass: apply scaling
    processed_frames = []
    for info in view_info:
        i = info['view_idx']
        img_path = info['img_path']
        scale = info['scale_factor']

        if img_path is None:
            continue

        # Load image
        img = Image.open(img_path)

        if img.mode == 'RGBA':
            img_array = np.array(img)
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]

            # Scale RGB and alpha separately
            scaled_rgb = scale_image_center(
                rgb, scale, output_size,
                background_color=(255, 255, 255)
            )
            scaled_alpha = scale_image_center(
                alpha, scale, output_size,
                background_color=(0,)  # Transparent background
            )

            # Combine
            output_img = np.zeros((output_size, output_size, 4), dtype=np.uint8)
            output_img[:, :, :3] = scaled_rgb
            output_img[:, :, 3] = scaled_alpha

            Image.fromarray(output_img, mode='RGBA').save(images_out / f"cam_{i:03d}.png")
        else:
            rgb = np.array(img.convert('RGB'))
            scaled_rgb = scale_image_center(
                rgb, scale, output_size,
                background_color=(255, 255, 255)
            )
            Image.fromarray(scaled_rgb).save(images_out / f"cam_{i:03d}.png")

        # Camera parameters remain UNCHANGED
        # The scaling is purely for visual consistency
        # The 3D geometry (w2c, intrinsics) should NOT be modified
        processed_frames.append(frames[i])

    # Save camera data (unchanged)
    output_camera_data = camera_data.copy()
    output_camera_data['frames'] = processed_frames
    output_camera_data['uniform_scale_info'] = {
        'method': 'image_based',
        'target_ratio': target_ratio,
        'original_ratios': [info['current_ratio'] for info in view_info],
        'scale_factors': [info['scale_factor'] for info in view_info],
    }

    with open(output_dir / "opencv_cameras.json", 'w') as f:
        json.dump(output_camera_data, f, indent=2)

    return {
        'n_views': n_views,
        'original_ratios': [info['current_ratio'] for info in view_info],
        'scales': [info['scale_factor'] for info in view_info],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply uniform scale based on image object size"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory with samples"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for uniformly scaled data"
    )
    parser.add_argument(
        "--target_ratio", type=float, default=0.6,
        help="Target object size ratio (default: 0.6 = 60%% of frame)"
    )
    parser.add_argument(
        "--output_size", type=int, default=512,
        help="Output image size (default: 512)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to process (None=all)"
    )
    # Keep --target_distance for backward compatibility but ignore it
    parser.add_argument(
        "--target_distance", type=float, default=2.7,
        help="[DEPRECATED] No longer used. Use --target_ratio instead."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sample directories
    sample_dirs = sorted(input_dir.glob("sample_*"))
    if args.num_samples:
        sample_dirs = sample_dirs[:args.num_samples]

    print("=" * 60)
    print("Uniform Scale Preprocessing (Image-Based)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target object ratio: {args.target_ratio:.1%}")
    print(f"Samples: {len(sample_dirs)}")
    print()

    # Collect statistics
    all_original_ratios = []
    all_scales = []
    successful = 0
    failed = 0

    for sample_dir in tqdm(sample_dirs, desc="Processing"):
        sample_name = sample_dir.name
        output_sample_dir = output_dir / sample_name

        try:
            result = process_sample(
                sample_dir, output_sample_dir,
                args.target_ratio, args.output_size
            )

            if result:
                all_original_ratios.extend(result['original_ratios'])
                all_scales.extend(result['scales'])
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            failed += 1

    # Copy data list files
    for list_file in input_dir.glob("*.txt"):
        with open(list_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            new_line = line.replace(input_dir.name, output_dir.name)
            new_lines.append(new_line)

        output_list_file = output_dir / list_file.name
        with open(output_list_file, 'w') as f:
            f.writelines(new_lines)

    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if all_original_ratios:
        valid_ratios = [r for r in all_original_ratios if r > 0]
        if valid_ratios:
            print()
            print("Original object size ratios:")
            print(f"  Min: {min(valid_ratios):.1%}")
            print(f"  Max: {max(valid_ratios):.1%}")
            print(f"  Mean: {np.mean(valid_ratios):.1%}")
            print()
            print("Scale factors applied:")
            valid_scales = [s for s in all_scales if 0.1 < s < 10]
            if valid_scales:
                print(f"  Min: {min(valid_scales):.3f}")
                print(f"  Max: {max(valid_scales):.3f}")
                print(f"  Mean: {np.mean(valid_scales):.3f}")

    # Verify one sample
    if successful > 0:
        sample = list(output_dir.glob("sample_*"))[0]
        with open(sample / "opencv_cameras.json", 'r') as f:
            data = json.load(f)

        print()
        print("=" * 60)
        print(f"Verification ({sample.name})")
        print("=" * 60)

        if 'uniform_scale_info' in data:
            info = data['uniform_scale_info']
            print(f"Method: {info['method']}")
            print(f"Target ratio: {info['target_ratio']:.1%}")
            print()
            print("View | Original | Scale | Target")
            print("-" * 40)
            for i, (orig, scale) in enumerate(zip(info['original_ratios'], info['scale_factors'])):
                target = orig * scale if orig > 0 else 0
                print(f"  {i}  |  {orig:.1%}   | {scale:.3f} |  {target:.1%}")


if __name__ == "__main__":
    main()
