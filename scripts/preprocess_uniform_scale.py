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

Fit-in-Frame Scaling (v2):
  - Prevents clipping by calculating safe scale based on BBox position
  - Centers object using Center of Mass (CoM) before scaling
  - Ensures object stays within frame boundaries after scaling

Usage:
    python scripts/preprocess_uniform_scale.py \
        --input_dir data_mouse_centered \
        --output_dir data_mouse_uniform \
        --target_ratio 0.6 \
        --safe_margin 0.05
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


def get_center_of_mass(alpha: np.ndarray, threshold: int = 10) -> Optional[Tuple[float, float]]:
    """
    Calculate center of mass from alpha mask.

    CoM = sum(position * weight) / sum(weight)
    This is more stable than bbox center for asymmetric objects like mice with tails.

    Args:
        alpha: Alpha channel (H, W) with values 0-255
        threshold: Minimum alpha value to consider as object

    Returns:
        (cx, cy) center of mass in pixel coordinates, or None if no object
    """
    mask = (alpha > threshold).astype(np.float32)

    if not mask.any():
        return None

    h, w = alpha.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    total_mass = np.sum(mask)
    cx = np.sum(x_coords * mask) / total_mass
    cy = np.sum(y_coords * mask) / total_mass

    return float(cx), float(cy)


def calc_safe_scale(
    bbox: Tuple[int, int, int, int],
    com: Tuple[float, float],
    output_size: int,
    desired_scale: float,
    safe_margin: float = 0.05
) -> float:
    """
    Calculate the maximum safe scale that keeps BBox within frame after CoM-based centering.

    After scaling, the object will be centered at CoM. This function ensures that
    no part of the BBox exceeds the frame boundaries.

    Args:
        bbox: (x1, y1, x2, y2) current object bounding box
        com: (cx, cy) Center of Mass
        output_size: Output image size (e.g., 512)
        desired_scale: Target scale (target_ratio / current_ratio)
        safe_margin: Margin ratio to keep from edges (default: 5%)

    Returns:
        Safe scale that prevents clipping
    """
    x1, y1, x2, y2 = bbox
    cx, cy = com

    # Distance from CoM to each edge of BBox
    dist_left = cx - x1
    dist_right = x2 - cx
    dist_top = cy - y1
    dist_bottom = y2 - cy

    # After centering, CoM will be at output_size/2
    # Available space in each direction (with margin)
    half_size = (output_size / 2) * (1 - safe_margin)

    # Calculate max scale for each direction
    max_scales = []
    if dist_left > 0:
        max_scales.append(half_size / dist_left)
    if dist_right > 0:
        max_scales.append(half_size / dist_right)
    if dist_top > 0:
        max_scales.append(half_size / dist_top)
    if dist_bottom > 0:
        max_scales.append(half_size / dist_bottom)

    if not max_scales:
        return desired_scale

    # Use the most restrictive scale
    max_safe_scale = min(max_scales)

    return min(desired_scale, max_safe_scale)


def transform_image_com_based(
    image: np.ndarray,
    com: Tuple[float, float],
    scale: float,
    output_size: int = 512,
    background_color: Tuple[int, ...] = (255, 255, 255)
) -> np.ndarray:
    """
    Transform image: center at CoM, then scale.

    This ensures the object is centered at its center of mass after transformation,
    preventing asymmetric clipping.

    Args:
        image: Input image (RGB or RGBA or grayscale)
        com: Current center of mass (cx, cy)
        scale: Scale factor to apply
        output_size: Output image size
        background_color: Background fill color

    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    cx, cy = com
    target_center = output_size / 2

    # Translation to move CoM to image center after scaling
    tx = target_center - cx * scale
    ty = target_center - cy * scale

    # Combined transformation matrix: scale then translate
    M = np.array([
        [scale, 0, tx],
        [0, scale, ty]
    ], dtype=np.float32)

    # Choose interpolation based on scale direction
    if scale > 1:
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_AREA

    # Determine border value
    if len(image.shape) == 3:
        n_channels = image.shape[2]
        border_value = background_color[:n_channels]
    else:
        border_value = background_color[0] if isinstance(background_color, tuple) else background_color

    output = cv2.warpAffine(
        image, M, (output_size, output_size),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    return output


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
    safe_margin: float = 0.05,
    use_fit_in_frame: bool = True,
) -> Optional[Dict]:
    """
    Process a single sample: scale images based on actual object size in image.

    Args:
        input_dir: Sample input directory
        output_dir: Sample output directory
        target_ratio: Target object size ratio (0.6 = 60% of frame)
        output_size: Output image size
        safe_margin: Margin ratio to keep from edges (default: 5%)
        use_fit_in_frame: Use CoM-based centering with safe scale (default: True)

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

    # First pass: measure object size and calculate safe scales
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
                'desired_scale': 1.0,
                'safe_scale': 1.0,
                'com': None,
                'bbox': None,
            })
            continue

        # Load image and measure object size
        img = Image.open(img_path)

        if img.mode == 'RGBA':
            alpha = np.array(img)[:, :, 3]
            current_ratio = get_object_size_ratio(alpha)
            com = get_center_of_mass(alpha)
            bbox = get_object_bbox_from_alpha(alpha)
        else:
            # If no alpha, assume object fills frame
            current_ratio = 0.8
            com = (img.size[0] / 2, img.size[1] / 2)
            bbox = (0, 0, img.size[0] - 1, img.size[1] - 1)

        # Calculate desired scale factor
        if current_ratio > 0:
            desired_scale = target_ratio / current_ratio
        else:
            desired_scale = 1.0

        # Calculate safe scale (Fit-in-Frame)
        if use_fit_in_frame and com is not None and bbox is not None:
            safe_scale = calc_safe_scale(bbox, com, output_size, desired_scale, safe_margin)
        else:
            safe_scale = desired_scale

        view_info.append({
            'view_idx': i,
            'img_path': img_path,
            'current_ratio': current_ratio,
            'desired_scale': desired_scale,
            'safe_scale': safe_scale,
            'com': com,
            'bbox': bbox,
            'scale_limited': safe_scale < desired_scale,
        })

    # Calculate UNIFORM scale for the entire sample (min of all safe_scales)
    # This ensures 3D geometric consistency across all views
    valid_safe_scales = [info['safe_scale'] for info in view_info if info['img_path'] is not None]
    if valid_safe_scales:
        uniform_scale = min(valid_safe_scales)
    else:
        uniform_scale = 1.0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Second pass: apply UNIFORM scaling with CoM-based centering
    processed_frames = []
    for info in view_info:
        i = info['view_idx']
        img_path = info['img_path']
        scale = uniform_scale  # Use uniform scale for ALL views (3D consistency)
        com = info['com']

        if img_path is None:
            continue

        # Load image
        img = Image.open(img_path)

        if img.mode == 'RGBA':
            img_array = np.array(img)
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]

            if use_fit_in_frame and com is not None:
                # Use CoM-based transformation (prevents clipping)
                scaled_rgb = transform_image_com_based(
                    rgb, com, scale, output_size,
                    background_color=(255, 255, 255)
                )
                scaled_alpha = transform_image_com_based(
                    alpha, com, scale, output_size,
                    background_color=(0,)
                )
            else:
                # Legacy: center-based scaling
                scaled_rgb = scale_image_center(
                    rgb, scale, output_size,
                    background_color=(255, 255, 255)
                )
                scaled_alpha = scale_image_center(
                    alpha, scale, output_size,
                    background_color=(0,)
                )

            # Combine
            output_img = np.zeros((output_size, output_size, 4), dtype=np.uint8)
            output_img[:, :, :3] = scaled_rgb
            output_img[:, :, 3] = scaled_alpha

            Image.fromarray(output_img, mode='RGBA').save(images_out / f"cam_{i:03d}.png")
        else:
            rgb = np.array(img.convert('RGB'))
            if use_fit_in_frame and com is not None:
                scaled_rgb = transform_image_com_based(
                    rgb, com, scale, output_size,
                    background_color=(255, 255, 255)
                )
            else:
                scaled_rgb = scale_image_center(
                    rgb, scale, output_size,
                    background_color=(255, 255, 255)
                )
            Image.fromarray(scaled_rgb).save(images_out / f"cam_{i:03d}.png")

        # Update camera intrinsics to match the scaling transformation
        # This is CRITICAL for 3D consistency in FaceLift pipeline
        original_frame = frames[i].copy()

        # Get original intrinsics
        fx = original_frame['fx']
        fy = original_frame['fy']
        cx = original_frame['cx']
        cy = original_frame['cy']
        w = original_frame['w']
        h = original_frame['h']

        # Update intrinsics based on CoM-based transformation
        # The transformation: 1) scale around CoM, 2) translate CoM to center
        if com is not None:
            com_x, com_y = com
            # After transformation, CoM moves to image center
            # Principal point transforms accordingly
            # New principal point = (old - CoM) * scale + output_center
            new_cx = (cx - com_x) * scale + output_size / 2
            new_cy = (cy - com_y) * scale + output_size / 2
        else:
            # Fallback: scale around image center
            new_cx = (cx - w / 2) * scale + output_size / 2
            new_cy = (cy - h / 2) * scale + output_size / 2

        # Focal length scales linearly with image scale
        new_fx = fx * scale
        new_fy = fy * scale

        # Update frame with new intrinsics
        updated_frame = original_frame.copy()
        updated_frame['fx'] = new_fx
        updated_frame['fy'] = new_fy
        updated_frame['cx'] = new_cx
        updated_frame['cy'] = new_cy
        updated_frame['w'] = output_size
        updated_frame['h'] = output_size

        processed_frames.append(updated_frame)

    # Save camera data with processing info
    output_camera_data = camera_data.copy()
    output_camera_data['frames'] = processed_frames
    output_camera_data['uniform_scale_info'] = {
        'method': 'sample_uniform_scale',  # New: uniform scale across all views
        'target_ratio': target_ratio,
        'safe_margin': safe_margin,
        'uniform_scale': uniform_scale,  # Single scale applied to ALL views
        'original_ratios': [info['current_ratio'] for info in view_info],
        'per_view_safe_scales': [info['safe_scale'] for info in view_info],  # For reference
        'intrinsics_updated': True,  # Indicates fx, fy, cx, cy were scaled
    }

    with open(output_dir / "opencv_cameras.json", 'w') as f:
        json.dump(output_camera_data, f, indent=2)

    return {
        'n_views': n_views,
        'uniform_scale': uniform_scale,
        'original_ratios': [info['current_ratio'] for info in view_info],
        'per_view_safe_scales': [info['safe_scale'] for info in view_info],
        'intrinsics_updated': True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Apply uniform scale based on image object size (with Fit-in-Frame)"
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
    parser.add_argument(
        "--safe_margin", type=float, default=0.05,
        help="Margin ratio to keep from edges (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--no_fit_in_frame", action="store_true",
        help="Disable Fit-in-Frame scaling (use legacy center-based scaling)"
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

    use_fit_in_frame = not args.no_fit_in_frame

    # Find sample directories
    sample_dirs = sorted(input_dir.glob("sample_*"))
    if args.num_samples:
        sample_dirs = sample_dirs[:args.num_samples]

    print("=" * 60)
    print("Uniform Scale Preprocessing")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target object ratio: {args.target_ratio:.1%}")
    print(f"Safe margin: {args.safe_margin:.1%}")
    print(f"Fit-in-Frame: {'Enabled (CoM-based)' if use_fit_in_frame else 'Disabled (legacy)'}")
    print(f"Samples: {len(sample_dirs)}")
    print()

    # Collect statistics
    all_original_ratios = []
    all_desired_scales = []
    all_safe_scales = []
    all_scale_limited = []
    successful = 0
    failed = 0

    for sample_dir in tqdm(sample_dirs, desc="Processing"):
        sample_name = sample_dir.name
        output_sample_dir = output_dir / sample_name

        try:
            result = process_sample(
                sample_dir, output_sample_dir,
                args.target_ratio, args.output_size,
                args.safe_margin, use_fit_in_frame
            )

            if result:
                all_original_ratios.extend(result['original_ratios'])
                all_desired_scales.extend(result['desired_scales'])
                all_safe_scales.extend(result['safe_scales'])
                all_scale_limited.extend(result['scale_limited'])
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            import traceback
            traceback.print_exc()
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
            print("Scale factors (desired vs safe):")
            valid_desired = [s for s in all_desired_scales if 0.1 < s < 10]
            valid_safe = [s for s in all_safe_scales if 0.1 < s < 10]
            if valid_desired:
                print(f"  Desired - Min: {min(valid_desired):.3f}, Max: {max(valid_desired):.3f}, Mean: {np.mean(valid_desired):.3f}")
            if valid_safe:
                print(f"  Safe    - Min: {min(valid_safe):.3f}, Max: {max(valid_safe):.3f}, Mean: {np.mean(valid_safe):.3f}")

            limited_count = sum(all_scale_limited)
            total_count = len(all_scale_limited)
            print()
            print(f"Scale limited (to prevent clipping): {limited_count}/{total_count} ({100*limited_count/total_count:.1f}%)")

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
            print(f"Safe margin: {info.get('safe_margin', 'N/A')}")
            print(f"Scale limited count: {info.get('scale_limited_count', 'N/A')}")
            print()
            print("View | Original | Desired | Safe   | Result")
            print("-" * 55)
            for i, (orig, desired, safe) in enumerate(zip(
                info['original_ratios'],
                info.get('desired_scales', info.get('scale_factors', [])),
                info.get('safe_scales', info.get('scale_factors', []))
            )):
                result_ratio = orig * safe if orig > 0 else 0
                limited = "*" if safe < desired else " "
                print(f"  {i}  |  {orig:.1%}   | {desired:.3f}  | {safe:.3f}{limited} |  {result_ratio:.1%}")


if __name__ == "__main__":
    main()
