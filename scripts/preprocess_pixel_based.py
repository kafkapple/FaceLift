#!/usr/bin/env python3
"""
Pixel-Based Preprocessing for Mouse-FaceLift

This is a unified preprocessing script that uses ONLY pixel-level information
for both centering and scaling. This is crucial for mouse data where:

1. Bounding box center is skewed by tail direction
2. Bounding box size varies with tail orientation (even for same body size)

Solution:
  - Centering: Use Center of Mass (CoM) instead of bounding box center
    CoM = sum(pixel_position * alpha) / sum(alpha)

  - Scaling: Use pixel count ratio instead of bounding box ratio
    size_ratio = sqrt(pixel_count / total_pixels)

Why Center of Mass?
  - Tail has fewer pixels than body -> less influence on center
  - More stable across different poses
  - Represents actual "mass center" of the object

Usage:
    python scripts/preprocess_pixel_based.py \
        --input_dir data_mouse \
        --output_dir data_mouse_pixel_based \
        --target_size_ratio 0.3 \
        --output_size 512
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_center_of_mass(alpha: np.ndarray, threshold: int = 10) -> Optional[Tuple[float, float]]:
    """
    Calculate center of mass from alpha mask.

    CoM = sum(position * weight) / sum(weight)
    where weight is the alpha value (or binary mask)

    Args:
        alpha: Alpha channel (H, W) with values 0-255
        threshold: Minimum alpha value to consider as object

    Returns:
        (cx, cy) center of mass in pixel coordinates, or None if no object
    """
    # Create binary mask
    mask = (alpha > threshold).astype(np.float32)

    if not mask.any():
        return None

    # Create coordinate grids
    h, w = alpha.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    # Calculate weighted sum
    total_mass = np.sum(mask)
    cx = np.sum(x_coords * mask) / total_mass
    cy = np.sum(y_coords * mask) / total_mass

    return float(cx), float(cy)


def get_pixel_size_ratio(alpha: np.ndarray, threshold: int = 10) -> float:
    """
    Get the size ratio of object using pixel count.

    Uses sqrt(pixel_count / total_pixels) because:
    - pixel_count is an area measure (2D)
    - We want linear size for scaling (1D)
    - sqrt converts area ratio to linear ratio

    Args:
        alpha: Alpha channel (H, W) with values 0-255
        threshold: Minimum alpha value to consider as object

    Returns:
        Square root of (pixel_count / total_pixels)
    """
    mask = alpha > threshold
    if not mask.any():
        return 0.0

    pixel_count = np.sum(mask)
    h, w = alpha.shape
    total_pixels = h * w

    area_ratio = pixel_count / total_pixels
    size_ratio = np.sqrt(area_ratio)

    return float(size_ratio)


def get_bbox_center(alpha: np.ndarray, threshold: int = 10) -> Optional[Tuple[float, float]]:
    """
    Get bounding box center (for comparison/statistics only).
    """
    mask = alpha > threshold
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return float(cx), float(cy)


def transform_image(
    image: np.ndarray,
    current_center: Tuple[float, float],
    scale: float,
    output_size: int = 512,
    background_color: Tuple[int, ...] = (255, 255, 255)
) -> np.ndarray:
    """
    Transform image: translate to center, then scale.

    Args:
        image: Input image (RGB or RGBA or grayscale)
        current_center: Current center of mass (cx, cy)
        scale: Scale factor to apply
        output_size: Output image size
        background_color: Background fill color

    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    cx, cy = current_center
    target_center = output_size / 2

    # Translation to move center of mass to image center
    tx = target_center - cx * scale
    ty = target_center - cy * scale

    # Combined transformation matrix: scale then translate
    # [scale, 0, tx]
    # [0, scale, ty]
    M = np.array([
        [scale, 0, tx],
        [0, scale, ty]
    ], dtype=np.float32)

    # Determine interpolation
    if scale > 1:
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_AREA

    # Create output canvas
    if len(image.shape) == 3:
        n_channels = image.shape[2]
        border_value = background_color[:n_channels]
    else:
        border_value = background_color[0]

    output = cv2.warpAffine(
        image, M, (output_size, output_size),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    return output


def process_sample(
    input_dir: Path,
    output_dir: Path,
    target_size_ratio: float = 0.3,
    output_size: int = 512,
) -> Optional[Dict]:
    """
    Process a single sample using pixel-based preprocessing.

    Args:
        input_dir: Sample input directory
        output_dir: Sample output directory
        target_size_ratio: Target object size ratio (0.3 = 30% of frame)
        output_size: Output image size

    Returns:
        Processing info dict, or None if failed
    """
    cameras_path = input_dir / "opencv_cameras.json"
    images_in = input_dir / "images"

    if not cameras_path.exists() or not images_in.exists():
        return None

    # Load camera data
    with open(cameras_path, 'r') as f:
        camera_data = json.load(f)

    frames = camera_data['frames']
    n_views = len(frames)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Process each view
    view_stats = []
    processed_frames = []

    for i, frame in enumerate(frames):
        # Find image file
        img_path = None
        for pattern in [f"cam_{i:03d}.png", f"{i:02d}.png"]:
            candidate = images_in / pattern
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            view_stats.append({
                'view_idx': i,
                'success': False,
                'reason': 'image_not_found'
            })
            continue

        # Load image
        img = Image.open(img_path)

        if img.mode != 'RGBA':
            view_stats.append({
                'view_idx': i,
                'success': False,
                'reason': 'no_alpha_channel'
            })
            continue

        img_array = np.array(img)
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3]

        # Calculate pixel-based metrics
        com = get_center_of_mass(alpha)
        bbox_center = get_bbox_center(alpha)
        size_ratio = get_pixel_size_ratio(alpha)

        if com is None or size_ratio == 0:
            view_stats.append({
                'view_idx': i,
                'success': False,
                'reason': 'empty_mask'
            })
            continue

        # Calculate scale factor
        scale = target_size_ratio / size_ratio

        # Transform RGB and alpha
        transformed_rgb = transform_image(
            rgb, com, scale, output_size,
            background_color=(255, 255, 255)
        )
        transformed_alpha = transform_image(
            alpha, com, scale, output_size,
            background_color=(0,)
        )

        # Combine and save
        output_img = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        output_img[:, :, :3] = transformed_rgb
        output_img[:, :, 3] = transformed_alpha

        Image.fromarray(output_img, mode='RGBA').save(images_out / f"cam_{i:03d}.png")

        # Verify output
        output_alpha = output_img[:, :, 3]
        output_com = get_center_of_mass(output_alpha)
        output_size_ratio = get_pixel_size_ratio(output_alpha)

        # Record stats
        view_stats.append({
            'view_idx': i,
            'success': True,
            'original_com': com,
            'original_bbox_center': bbox_center,
            'com_bbox_diff': np.sqrt((com[0] - bbox_center[0])**2 + (com[1] - bbox_center[1])**2) if bbox_center else 0,
            'original_size_ratio': size_ratio,
            'scale_factor': scale,
            'output_com': output_com,
            'output_size_ratio': output_size_ratio,
        })

        processed_frames.append(frames[i])

    # Save camera data (unchanged - we only do image-space transforms)
    output_camera_data = camera_data.copy()
    output_camera_data['frames'] = processed_frames
    output_camera_data['pixel_based_preprocessing'] = {
        'method': 'center_of_mass_and_pixel_scale',
        'target_size_ratio': target_size_ratio,
        'output_size': output_size,
        'view_stats': view_stats,
    }

    with open(output_dir / "opencv_cameras.json", 'w') as f:
        json.dump(output_camera_data, f, indent=2)

    return {
        'n_views': n_views,
        'n_processed': len(processed_frames),
        'view_stats': view_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pixel-based preprocessing (CoM centering + pixel scaling)"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory with samples"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--target_size_ratio", type=float, default=0.3,
        help="Target object size ratio (default: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--output_size", type=int, default=512,
        help="Output image size (default: 512)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to process (None=all)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find sample directories
    sample_dirs = sorted(input_dir.glob("sample_*"))
    if args.num_samples:
        sample_dirs = sample_dirs[:args.num_samples]

    print("=" * 70)
    print("Pixel-Based Preprocessing")
    print("  - Centering: Center of Mass (CoM)")
    print("  - Scaling: Pixel count ratio")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target size ratio: {args.target_size_ratio:.1%}")
    print(f"Output size: {args.output_size}")
    print(f"Samples: {len(sample_dirs)}")
    print()

    # Collect statistics
    all_original_sizes = []
    all_output_sizes = []
    all_com_bbox_diffs = []
    all_scales = []
    successful = 0
    failed = 0

    for sample_dir in tqdm(sample_dirs, desc="Processing"):
        sample_name = sample_dir.name
        output_sample_dir = output_dir / sample_name

        try:
            result = process_sample(
                sample_dir, output_sample_dir,
                args.target_size_ratio, args.output_size
            )

            if result and result['n_processed'] > 0:
                for stat in result['view_stats']:
                    if stat.get('success'):
                        all_original_sizes.append(stat['original_size_ratio'])
                        all_output_sizes.append(stat['output_size_ratio'])
                        all_com_bbox_diffs.append(stat['com_bbox_diff'])
                        all_scales.append(stat['scale_factor'])
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            failed += 1

    # Copy/update data list files
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
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successful samples: {successful}")
    print(f"Failed samples: {failed}")

    if all_original_sizes:
        print()
        print("Original Size Ratios:")
        print(f"  Mean: {np.mean(all_original_sizes):.1%}")
        print(f"  Std:  {np.std(all_original_sizes):.1%}")
        print(f"  CV:   {np.std(all_original_sizes)/np.mean(all_original_sizes)*100:.2f}%")
        print(f"  Min:  {np.min(all_original_sizes):.1%}")
        print(f"  Max:  {np.max(all_original_sizes):.1%}")

        print()
        print("Output Size Ratios (after scaling):")
        print(f"  Mean: {np.mean(all_output_sizes):.1%}")
        print(f"  Std:  {np.std(all_output_sizes):.1%}")
        print(f"  CV:   {np.std(all_output_sizes)/np.mean(all_output_sizes)*100:.2f}%" if np.mean(all_output_sizes) > 0 else "  CV:   N/A")
        print(f"  Target: {args.target_size_ratio:.1%}")

        print()
        print("Center of Mass vs Bbox Center Difference (pixels):")
        print(f"  Mean: {np.mean(all_com_bbox_diffs):.1f}")
        print(f"  Std:  {np.std(all_com_bbox_diffs):.1f}")
        print(f"  Max:  {np.max(all_com_bbox_diffs):.1f}")

        print()
        print("Scale Factors Applied:")
        print(f"  Mean: {np.mean(all_scales):.3f}")
        print(f"  Std:  {np.std(all_scales):.3f}")
        print(f"  Min:  {np.min(all_scales):.3f}")
        print(f"  Max:  {np.max(all_scales):.3f}")

    # Verify one sample
    if successful > 0:
        sample = list(output_dir.glob("sample_*"))[0]
        with open(sample / "opencv_cameras.json", 'r') as f:
            data = json.load(f)

        print()
        print("=" * 70)
        print(f"Verification ({sample.name})")
        print("=" * 70)

        if 'pixel_based_preprocessing' in data:
            info = data['pixel_based_preprocessing']
            print(f"Method: {info['method']}")
            print(f"Target size ratio: {info['target_size_ratio']:.1%}")
            print()
            print("View | Orig Size | Scale  | Out Size | CoM-Bbox Diff")
            print("-" * 55)
            for stat in info['view_stats'][:6]:  # Show first 6 views
                if stat.get('success'):
                    print(f"  {stat['view_idx']}  |   {stat['original_size_ratio']:.1%}   | {stat['scale_factor']:.3f} |   {stat['output_size_ratio']:.1%}   |   {stat['com_bbox_diff']:.1f}px")


if __name__ == "__main__":
    main()
