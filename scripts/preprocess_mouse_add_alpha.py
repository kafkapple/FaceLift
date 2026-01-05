#!/usr/bin/env python3
"""
Add alpha channel (mask) to mouse images.

This script processes existing mouse images to add proper segmentation masks.
IMPORTANT: This does NOT apply centering/scaling to preserve camera alignment.

For GS-LRM training with real 6-view data:
- Camera parameters must match image content
- Centering/scaling would break camera-image correspondence
- Only alpha channel is added for masked loss computation

Two methods available:
1. rembg: Use U2-Net based background removal (better quality)
2. threshold: Use simple white background thresholding (faster)

Usage:
    # Using rembg (recommended for quality)
    python scripts/preprocess_mouse_add_alpha.py \
        --input_dir data_mouse_local \
        --output_dir data_mouse_local_rgba \
        --method rembg

    # Using threshold (faster, for already white-background images)
    python scripts/preprocess_mouse_add_alpha.py \
        --input_dir data_mouse_local \
        --output_dir data_mouse_local_rgba \
        --method threshold \
        --threshold 250

Note: For MVDiffusion input preprocessing (centering/scaling needed),
use scripts/preprocess_mouse_for_mvdiffusion.py instead.
"""

import argparse
import os
import json
import shutil
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from rembg import remove as rembg_remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False
    print("Warning: rembg not installed. Install with: pip install rembg")


def add_alpha_rembg(image: np.ndarray) -> np.ndarray:
    """
    Add alpha channel using rembg (U2-Net based).
    Best quality, works on any background.
    """
    if not HAS_REMBG:
        raise RuntimeError("rembg not installed")

    pil_image = Image.fromarray(image)
    result = rembg_remove(pil_image)
    return np.array(result)


def add_alpha_threshold(
    image: np.ndarray,
    threshold: int = 250,
    background: str = "white"
) -> np.ndarray:
    """
    Add alpha channel using threshold-based detection.
    Fast, but requires uniform background.

    Args:
        image: RGB image [H, W, 3]
        threshold: Pixels > threshold (for white) or < threshold (for black) are background
        background: "white" or "black"
    """
    if image.shape[2] == 4:
        return image  # Already has alpha

    h, w = image.shape[:2]

    if background == "white":
        # White background: all RGB channels > threshold
        is_background = np.all(image > threshold, axis=2)
    else:
        # Black background: all RGB channels < threshold
        is_background = np.all(image < threshold, axis=2)

    alpha = np.where(is_background, 0, 255).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = alpha

    return rgba


def process_sample(
    input_sample_dir: Path,
    output_sample_dir: Path,
    method: str = "rembg",
    threshold: int = 250,
    background: str = "white"
) -> dict:
    """Process a single sample directory."""
    result = {"success": True, "sample": input_sample_dir.name, "images": 0}

    # Create output directory
    output_sample_dir.mkdir(parents=True, exist_ok=True)

    # Copy camera file
    camera_file = input_sample_dir / "opencv_cameras.json"
    if camera_file.exists():
        shutil.copy(camera_file, output_sample_dir / "opencv_cameras.json")

    # Process images
    images_dir = input_sample_dir / "images"
    output_images_dir = output_sample_dir / "images"
    output_images_dir.mkdir(exist_ok=True)

    if not images_dir.exists():
        result["success"] = False
        result["error"] = "No images directory"
        return result

    for img_path in sorted(images_dir.glob("*.png")):
        try:
            # Load image
            image = np.array(Image.open(img_path))

            # Skip if already has alpha
            if image.shape[2] == 4:
                rgba = image
            elif method == "rembg":
                rgba = add_alpha_rembg(image[:, :, :3])
            else:
                rgba = add_alpha_threshold(image[:, :, :3], threshold, background)

            # Save as PNG (preserves alpha)
            output_path = output_images_dir / img_path.name
            Image.fromarray(rgba).save(output_path)
            result["images"] += 1

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            break

    return result


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    method: str = "rembg",
    threshold: int = 250,
    background: str = "white",
    num_workers: int = 4
):
    """Process entire dataset."""
    # Find all sample directories
    sample_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith("sample_")
    ])

    print(f"Found {len(sample_dirs)} samples")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")

    # Create output directory first
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy data list files
    for list_file in input_dir.glob("data_mouse_*.txt"):
        # Update paths in list file
        with open(list_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip()
            if line:
                # Replace input dir with output dir
                new_path = line.replace(str(input_dir), str(output_dir))
                new_lines.append(new_path)

        output_list = output_dir / list_file.name
        with open(output_list, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Updated: {output_list}")

    # Process samples
    results = []
    if method == "rembg" and num_workers > 1:
        # Use single thread for rembg (GPU memory issues with parallel)
        num_workers = 1
        print("Note: Using single thread for rembg (GPU memory)")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for sample_dir in sample_dirs:
            output_sample = output_dir / sample_dir.name
            future = executor.submit(
                process_sample,
                sample_dir, output_sample,
                method, threshold, background
            )
            futures[future] = sample_dir.name

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)

            if not result["success"]:
                print(f"\nError in {result['sample']}: {result.get('error', 'Unknown')}")

    # Summary
    success = sum(1 for r in results if r["success"])
    total_images = sum(r["images"] for r in results)

    print(f"\nProcessed: {success}/{len(results)} samples")
    print(f"Total images: {total_images}")

    # Verify alpha channel
    print("\nVerifying alpha channel...")
    test_sample = output_dir / sample_dirs[0].name / "images"
    test_images = list(test_sample.glob("*.png"))
    if test_images:
        test_img = np.array(Image.open(test_images[0]))
        print(f"  Sample image shape: {test_img.shape}")
        if test_img.shape[2] == 4:
            fg_ratio = (test_img[:, :, 3] > 128).mean() * 100
            print(f"  Has alpha channel: Yes")
            print(f"  Foreground ratio: {fg_ratio:.1f}%")
        else:
            print(f"  Has alpha channel: No (ERROR)")


def main():
    parser = argparse.ArgumentParser(
        description='Add alpha channel to mouse images'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with sample_* folders')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--method', type=str, default='threshold',
                        choices=['rembg', 'threshold'],
                        help='Method for alpha generation')
    parser.add_argument('--threshold', type=int, default=250,
                        help='Threshold for white background detection')
    parser.add_argument('--background', type=str, default='white',
                        choices=['white', 'black'],
                        help='Background color for threshold method')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    if args.method == 'rembg' and not HAS_REMBG:
        print("Error: rembg not installed. Install with: pip install rembg")
        print("Or use --method threshold for fast threshold-based detection")
        return

    process_dataset(
        input_dir, output_dir,
        method=args.method,
        threshold=args.threshold,
        background=args.background,
        num_workers=args.num_workers
    )

    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print(f"1. Update config to use new data:")
    print(f"   dataset_path: \"{output_dir}/data_mouse_train.txt\"")
    print(f"\n2. Set remove_alpha: false to keep alpha channel")
    print(f"\n3. Run training:")
    print(f"   python train_gslrm.py --config configs/mouse_gslrm_local_rtx3060.yaml")


if __name__ == '__main__':
    main()
