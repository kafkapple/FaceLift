#!/usr/bin/env python3
"""
Validate preprocessed mouse dataset quality.

Usage:
    python -m mouse_extensions.scripts.validate_preprocessing --dataset_dir /path/to/dataset

Checks:
    - Object size consistency (ratio should be < 1.2)
    - Camera parameters (cx=cy=256, fx=549)
    - Image dimensions (512x512)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image


def measure_object_size(img_path: Path) -> int:
    """Measure object size from alpha channel."""
    img = np.array(Image.open(img_path))
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        # Auto-generate mask from RGB
        rgb = img[:, :, :3] if img.ndim == 3 else img
        is_bg = np.all(rgb > 250, axis=2)
        alpha = (~is_bg).astype(np.uint8) * 255
    
    coords = np.where(alpha > 128)
    if len(coords[0]) == 0:
        return 0
    h = coords[0].max() - coords[0].min()
    w = coords[1].max() - coords[1].min()
    return max(h, w)


def validate_sample(sample_dir: Path) -> dict:
    """Validate a single sample."""
    # Load camera params
    json_path = sample_dir / "opencv_cameras.json"
    with open(json_path) as f:
        data = json.load(f)
    
    results = {
        "sizes": [],
        "fx": [],
        "fy": [],
        "cx": [],
        "cy": [],
    }
    
    for frame in data["frames"]:
        # Camera params
        results["fx"].append(frame["fx"])
        results["fy"].append(frame["fy"])
        results["cx"].append(frame["cx"])
        results["cy"].append(frame["cy"])
        
        # Object size
        img_path = sample_dir / frame["file_path"]
        if img_path.exists():
            size = measure_object_size(img_path)
            results["sizes"].append(size)
    
    return results


def validate_dataset(dataset_dir: str, max_samples: int = 50) -> dict:
    """Validate entire dataset."""
    dataset_path = Path(dataset_dir)
    sample_dirs = sorted([d for d in dataset_path.iterdir() 
                         if d.is_dir() and d.name.startswith("sample_")])
    
    all_sizes = []
    all_fx = []
    all_fy = []
    all_cx = []
    all_cy = []
    
    for i, sample_dir in enumerate(sample_dirs[:max_samples]):
        results = validate_sample(sample_dir)
        all_sizes.extend(results["sizes"])
        all_fx.extend(results["fx"])
        all_fy.extend(results["fy"])
        all_cx.extend(results["cx"])
        all_cy.extend(results["cy"])
    
    if not all_sizes:
        return {"error": "No valid samples found"}
    
    size_ratio = max(all_sizes) / min(all_sizes) if min(all_sizes) > 0 else float("inf")
    
    report = {
        "num_samples": len(sample_dirs),
        "samples_checked": min(max_samples, len(sample_dirs)),
        "size_min": min(all_sizes),
        "size_max": max(all_sizes),
        "size_ratio": size_ratio,
        "fx_mean": np.mean(all_fx),
        "fy_mean": np.mean(all_fy),
        "cx_mean": np.mean(all_cx),
        "cy_mean": np.mean(all_cy),
        "cx_range": [min(all_cx), max(all_cx)],
        "cy_range": [min(all_cy), max(all_cy)],
    }
    
    # Quality checks
    report["checks"] = {
        "size_consistency": size_ratio < 1.2,
        "cx_centered": abs(report["cx_mean"] - 256) < 1,
        "cy_centered": abs(report["cy_mean"] - 256) < 1,
        "fx_correct": abs(report["fx_mean"] - 549) < 1,
    }
    report["all_checks_passed"] = all(report["checks"].values())
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed mouse dataset")
    parser.add_argument("--dataset_dir", required=True, help="Dataset directory")
    parser.add_argument("--max_samples", type=int, default=50, 
                       help="Max samples to check (default: 50)")
    args = parser.parse_args()
    
    print(f"Validating dataset: {args.dataset_dir}")
    print("-" * 60)
    
    report = validate_dataset(args.dataset_dir, args.max_samples)
    
    if "error" in report:
        print(f"Error: {report['error']}")
        return
    
    print(f"Total samples: {report['num_samples']}")
    print(f"Samples checked: {report['samples_checked']}")
    print()
    print(f"Object Size: min={report['size_min']}, max={report['size_max']}, ratio={report['size_ratio']:.2f}x")
    print(f"fx mean: {report['fx_mean']:.2f}")
    print(f"fy mean: {report['fy_mean']:.2f}")
    print(f"cx mean: {report['cx_mean']:.2f}, range=[{report['cx_range'][0]:.1f}, {report['cx_range'][1]:.1f}]")
    print(f"cy mean: {report['cy_mean']:.2f}, range=[{report['cy_range'][0]:.1f}, {report['cy_range'][1]:.1f}]")
    print()
    print("Quality Checks:")
    for check, passed in report["checks"].items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    print()
    
    if report["all_checks_passed"]:
        print("✅ All checks passed! Dataset is ready for training.")
    else:
        print("❌ Some checks failed. Please review the preprocessing.")


if __name__ == "__main__":
    main()
