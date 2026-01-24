#!/usr/bin/env python3
"""
Verify M3 preprocessing for clipping issues.

Checks:
1. Clipping rate: mask touching image boundary
2. FG coverage: actual foreground ratio
3. Center offset: mouse center vs image center
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def check_clipping(mask: np.ndarray, margin: int = 2) -> dict:
    """Check if mask touches image boundary."""
    h, w = mask.shape
    
    # Check each edge
    top = mask[:margin, :].any()
    bottom = mask[-margin:, :].any()
    left = mask[:, :margin].any()
    right = mask[:, -margin:].any()
    
    clipped = top or bottom or left or right
    
    return {
        "clipped": clipped,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def compute_metrics(mask: np.ndarray) -> dict:
    """Compute mask metrics."""
    h, w = mask.shape
    
    # FG coverage
    fg_pixels = (mask > 127).sum()
    coverage = fg_pixels / (h * w)
    
    # Center offset
    if fg_pixels > 0:
        ys, xs = np.where(mask > 127)
        com_x = xs.mean()
        com_y = ys.mean()
        center_offset = np.sqrt((com_x - w/2)**2 + (com_y - h/2)**2)
    else:
        center_offset = 0
    
    # BBox
    if fg_pixels > 0:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        bbox_ratio = max(y2-y1, x2-x1) / min(h, w)
    else:
        bbox_ratio = 0
    
    return {
        "coverage": coverage,
        "center_offset": center_offset,
        "bbox_ratio": bbox_ratio,
    }


def analyze_sample(sample_dir: Path) -> dict:
    """Analyze a single sample (all views)."""
    results = {
        "clipped_views": 0,
        "total_views": 0,
        "coverages": [],
        "center_offsets": [],
        "clip_details": [],
    }
    
    # Find all camera images
    images_dir = sample_dir / "images"
    if not images_dir.exists():
        return results
    
    for img_path in sorted(images_dir.glob("cam_*.png")):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] < 4:
            continue
        
        mask = img[:, :, 3]
        
        clip_result = check_clipping(mask)
        metrics = compute_metrics(mask)
        
        results["total_views"] += 1
        if clip_result["clipped"]:
            results["clipped_views"] += 1
            results["clip_details"].append({
                "view": img_path.stem,
                **clip_result
            })
        
        results["coverages"].append(metrics["coverage"])
        results["center_offsets"].append(metrics["center_offset"])
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Preprocessed dataset directory")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to check")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find samples
    samples_dir = data_dir / "samples"
    if samples_dir.exists():
        sample_dirs = sorted(samples_dir.iterdir())[:args.num_samples]
    else:
        # Try train/val structure
        sample_dirs = []
        for split in ["train", "val"]:
            split_dir = data_dir / split
            if split_dir.exists():
                sample_dirs.extend(sorted(split_dir.iterdir())[:args.num_samples // 2])
    
    if not sample_dirs:
        print(f"No samples found in {data_dir}")
        return
    
    print()
    print("=" * 60)
    print("M3 Preprocessing Verification")
    print("=" * 60)
    print(f"Dataset: {data_dir}")
    print(f"Samples to check: {len(sample_dirs)}")
    print()
    
    # Aggregate results
    total_clipped = 0
    total_views = 0
    all_coverages = []
    all_offsets = []
    clipped_samples = []
    
    for sample_dir in tqdm(sample_dirs, desc="Analyzing"):
        result = analyze_sample(sample_dir)
        
        total_clipped += result["clipped_views"]
        total_views += result["total_views"]
        all_coverages.extend(result["coverages"])
        all_offsets.extend(result["center_offsets"])
        
        if result["clipped_views"] > 0:
            clipped_samples.append({
                "sample": sample_dir.name,
                "clipped": result["clipped_views"],
                "total": result["total_views"],
            })
    
    # Report
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\n[Clipping Analysis]")
    print(f"  Clipped views: {total_clipped}/{total_views} ({100*total_clipped/max(total_views,1):.1f}%)")
    
    if clipped_samples:
        print("\n  Samples with clipping:")
        for s in clipped_samples[:10]:
            print(f"    - {s['sample']}: {s['clipped']}/{s['total']} views")
    
    print("\n[Coverage Analysis]")
    if all_coverages:
        print(f"  Mean: {100*np.mean(all_coverages):.2f}%")
        print(f"  Std:  {100*np.std(all_coverages):.2f}%")
        print(f"  Min:  {100*np.min(all_coverages):.2f}%")
        print(f"  Max:  {100*np.max(all_coverages):.2f}%")
    
    print("\n[Center Offset Analysis]")
    if all_offsets:
        print(f"  Mean: {np.mean(all_offsets):.1f} px")
        print(f"  Std:  {np.std(all_offsets):.1f} px")
        print(f"  Max:  {np.max(all_offsets):.1f} px")
    
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    clip_rate = total_clipped / max(total_views, 1)
    mean_coverage = np.mean(all_coverages) if all_coverages else 0
    
    if clip_rate < 0.01 and mean_coverage >= 0.03:
        print("PASS: Low clipping, adequate coverage")
    elif clip_rate < 0.05:
        print("WARNING: Minor clipping detected")
    else:
        print("FAIL: Significant clipping or coverage issues")


if __name__ == "__main__":
    main()
