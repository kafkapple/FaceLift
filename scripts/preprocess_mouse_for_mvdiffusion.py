#!/usr/bin/env python3
"""
Preprocess mouse images for MVDiffusion inference.

Steps:
1. Remove background using rembg
2. Center-align the mouse in the image
3. Composite on white background
4. Save processed image

This preprocessing is essential for generating consistent multi-view images
because MVDiffusion was trained on centered, clean-background images.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    print("Warning: rembg not installed. Using threshold-based background removal.")


def remove_background_rembg(image: np.ndarray) -> np.ndarray:
    """
    Remove background using rembg.
    Returns RGBA image with transparent background.
    """
    if not HAS_REMBG:
        return remove_background_threshold(image)

    pil_image = Image.fromarray(image)
    result = rembg_remove(pil_image)
    return np.array(result)


def remove_background_threshold(
    image: np.ndarray,
    threshold: int = 50
) -> np.ndarray:
    """
    Fallback background removal using color threshold.
    Assumes dark background (low pixel values).
    """
    # Convert to grayscale for mask creation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Create mask: foreground is where pixels are brighter than threshold
    # For mouse on dark background, mouse is relatively bright
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Create RGBA output
    if len(image.shape) == 2:
        rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3:
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    else:
        rgba = image.copy()

    rgba[:, :, 3] = mask
    return rgba


def find_object_bbox(alpha_channel: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find bounding box of non-transparent pixels.
    Returns (x_min, y_min, x_max, y_max) or None.
    """
    # Find non-zero pixels in alpha channel
    coords = np.where(alpha_channel > 128)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    return (x_min, y_min, x_max, y_max)


def center_align_with_background_removal(
    image: np.ndarray,
    output_size: int = 512,
    target_object_ratio: float = 0.6,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    use_rembg: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Remove background, center-align object, and composite on white background.

    Args:
        image: Input RGB image
        output_size: Output image size (square)
        target_object_ratio: Target ratio of object size to image size
        background_color: Background color (default white)
        use_rembg: Use rembg for background removal

    Returns:
        (processed_image, metadata)
    """
    # Step 1: Remove background
    if use_rembg and HAS_REMBG:
        rgba = remove_background_rembg(image)
    else:
        rgba = remove_background_threshold(image)

    # Step 2: Find object bounding box
    alpha = rgba[:, :, 3]
    bbox = find_object_bbox(alpha)

    if bbox is None:
        # No object found, return original with white background
        output = np.full((output_size, output_size, 3), background_color, dtype=np.uint8)
        return output, {"error": "No object found"}

    x_min, y_min, x_max, y_max = bbox
    obj_width = x_max - x_min
    obj_height = y_max - y_min
    obj_size = max(obj_width, obj_height)

    # Step 3: Calculate scale to fit object with padding
    target_size = output_size * target_object_ratio
    scale = target_size / obj_size

    # Step 4: Scale image
    h, w = rgba.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    scaled_rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Update object center after scaling
    obj_center_x = (x_min + x_max) / 2 * scale
    obj_center_y = (y_min + y_max) / 2 * scale

    # Step 5: Create output with background color
    output = np.full((output_size, output_size, 3), background_color, dtype=np.uint8)

    # Step 6: Calculate paste position (center object)
    target_center = output_size / 2
    offset_x = int(target_center - obj_center_x)
    offset_y = int(target_center - obj_center_y)

    # Calculate source and destination regions
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(new_w, output_size - offset_x)
    src_y2 = min(new_h, output_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Step 7: Alpha composite
    src_region = scaled_rgba[src_y1:src_y2, src_x1:src_x2]
    dst_region = output[dst_y1:dst_y2, dst_x1:dst_x2]

    alpha_mask = src_region[:, :, 3:4].astype(float) / 255.0
    rgb_src = src_region[:, :, :3].astype(float)
    rgb_dst = dst_region.astype(float)

    blended = rgb_src * alpha_mask + rgb_dst * (1 - alpha_mask)
    output[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

    metadata = {
        "original_bbox": bbox,
        "scale": scale,
        "offset": (offset_x, offset_y),
        "obj_center": (obj_center_x / scale, obj_center_y / scale),
        "obj_size": obj_size
    }

    return output, metadata


def process_sample(
    sample_path: Path,
    output_path: Path,
    view_indices: list = None,
    output_size: int = 512,
    use_rembg: bool = True
) -> bool:
    """
    Process a single sample directory.

    Args:
        sample_path: Path to sample directory (containing images/)
        output_path: Output path for processed images
        view_indices: List of view indices to process (default: all 6)
        output_size: Output image size
        use_rembg: Use rembg for background removal

    Returns:
        True if successful
    """
    if view_indices is None:
        view_indices = list(range(6))

    images_dir = sample_path / "images"
    if not images_dir.exists():
        images_dir = sample_path  # Try sample_path directly

    output_images_dir = output_path / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    success = True

    for view_idx in view_indices:
        # Try different naming conventions
        for pattern in [f"cam_{view_idx:03d}.png", f"{view_idx:02d}.png"]:
            input_path = images_dir / pattern
            if input_path.exists():
                break
        else:
            print(f"  Warning: No image found for view {view_idx}")
            success = False
            continue

        # Load image
        image = np.array(Image.open(input_path).convert('RGB'))

        # Process
        processed, metadata = center_align_with_background_removal(
            image,
            output_size=output_size,
            target_object_ratio=0.6,
            use_rembg=use_rembg
        )

        # Save
        output_image_path = output_images_dir / f"{view_idx:02d}.png"
        Image.fromarray(processed).save(output_image_path)

    return success


def main():
    parser = argparse.ArgumentParser(description="Preprocess mouse images for MVDiffusion")

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input data directory (containing sample_* folders)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed images")
    parser.add_argument("--data_list", type=str, default=None,
                        help="Optional: file listing sample paths to process")
    parser.add_argument("--output_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--no_rembg", action="store_true",
                        help="Use threshold-based background removal instead of rembg")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample list
    if args.data_list and Path(args.data_list).exists():
        with open(args.data_list, 'r') as f:
            sample_paths = [Path(line.strip()) for line in f if line.strip()]
    else:
        sample_paths = sorted(input_dir.glob("sample_*"))

    if args.num_samples:
        sample_paths = sample_paths[:args.num_samples]

    print(f"\nProcessing {len(sample_paths)} samples")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Background removal: {'rembg' if not args.no_rembg and HAS_REMBG else 'threshold'}")
    print()

    processed = []

    for sample_path in tqdm(sample_paths, desc="Processing"):
        sample_name = sample_path.name
        output_sample_path = output_dir / sample_name

        success = process_sample(
            sample_path,
            output_sample_path,
            output_size=args.output_size,
            use_rembg=not args.no_rembg
        )

        if success:
            processed.append(str(output_sample_path))

    # Save processed sample list
    with open(output_dir / "processed_samples.txt", 'w') as f:
        f.write("\n".join(processed))

    print(f"\nDone! Processed {len(processed)} samples")
    print(f"Sample list saved to: {output_dir / 'processed_samples.txt'}")


if __name__ == "__main__":
    main()
