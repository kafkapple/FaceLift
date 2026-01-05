#!/usr/bin/env python3
"""
Preprocessing Quality Evaluation Script for Mouse-FaceLift

This script evaluates the quality of preprocessed data by analyzing:
1. Center of Mass (CoM) position - should be near image center
2. Pixel-based size ratio - should match target ratio with low variance
3. Cross-view consistency - CoM should be consistent across views

Usage:
    python scripts/evaluate_preprocessing.py \
        --data_dir data_mouse_pixel_based \
        --output_dir reports/preprocessing_quality \
        --num_samples 100

Output:
    - Markdown report with statistics and visualizations
    - JSON file with detailed metrics
    - Visualization images (center overlay, histograms)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_center_of_mass(alpha: np.ndarray, threshold: int = 10) -> Optional[Tuple[float, float]]:
    """
    Calculate center of mass from alpha mask.

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


def get_pixel_size_ratio(alpha: np.ndarray, threshold: int = 10) -> float:
    """
    Get the size ratio of object using pixel count.

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
    """Get bounding box center for comparison."""
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


def analyze_sample(sample_dir: Path, img_size: int = 512) -> Optional[Dict]:
    """
    Analyze a single sample's preprocessing quality.

    Returns:
        Dict with metrics for all views, or None if failed
    """
    images_dir = sample_dir / "images"
    if not images_dir.exists():
        return None

    view_metrics = []
    image_center = img_size / 2

    for img_path in sorted(images_dir.glob("cam_*.png")):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGBA':
                continue

            alpha = np.array(img)[:, :, 3]

            # Get metrics
            com = get_center_of_mass(alpha)
            bbox_center = get_bbox_center(alpha)
            size_ratio = get_pixel_size_ratio(alpha)

            if com is None:
                continue

            # Calculate center offset from image center
            com_offset_x = com[0] - image_center
            com_offset_y = com[1] - image_center
            com_offset_dist = np.sqrt(com_offset_x**2 + com_offset_y**2)

            # Normalized offset (as percentage of image size)
            com_offset_pct = com_offset_dist / img_size * 100

            view_metrics.append({
                'view_idx': int(img_path.stem.split('_')[1]),
                'com': com,
                'bbox_center': bbox_center,
                'size_ratio': size_ratio,
                'com_offset_x': com_offset_x,
                'com_offset_y': com_offset_y,
                'com_offset_dist': com_offset_dist,
                'com_offset_pct': com_offset_pct,
            })
        except Exception as e:
            continue

    if not view_metrics:
        return None

    # Cross-view statistics
    size_ratios = [m['size_ratio'] for m in view_metrics]
    com_offsets = [m['com_offset_dist'] for m in view_metrics]

    return {
        'sample_name': sample_dir.name,
        'n_views': len(view_metrics),
        'view_metrics': view_metrics,
        'size_ratio_mean': np.mean(size_ratios),
        'size_ratio_std': np.std(size_ratios),
        'size_ratio_cv': np.std(size_ratios) / np.mean(size_ratios) * 100 if np.mean(size_ratios) > 0 else 0,
        'com_offset_mean': np.mean(com_offsets),
        'com_offset_std': np.std(com_offsets),
        'com_offset_max': np.max(com_offsets),
    }


def create_visualization(all_metrics: List[Dict], output_dir: Path, dataset_name: str):
    """Create visualization plots for the preprocessing quality."""

    # Collect all view-level data
    all_size_ratios = []
    all_com_offsets = []
    all_com_x = []
    all_com_y = []

    for sample in all_metrics:
        for view in sample['view_metrics']:
            all_size_ratios.append(view['size_ratio'])
            all_com_offsets.append(view['com_offset_dist'])
            all_com_x.append(view['com'][0])
            all_com_y.append(view['com'][1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Preprocessing Quality Report: {dataset_name}', fontsize=14, fontweight='bold')

    # 1. Size Ratio Distribution
    ax1 = axes[0, 0]
    ax1.hist(all_size_ratios, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(all_size_ratios), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_size_ratios):.3f}')
    ax1.set_xlabel('Size Ratio (sqrt of pixel area ratio)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Size Ratio Distribution\nStd: {np.std(all_size_ratios):.4f}, CV: {np.std(all_size_ratios)/np.mean(all_size_ratios)*100:.2f}%')
    ax1.legend()

    # 2. CoM Offset Distribution
    ax2 = axes[0, 1]
    ax2.hist(all_com_offsets, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.mean(all_com_offsets), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_com_offsets):.1f}px')
    ax2.set_xlabel('CoM Offset from Image Center (pixels)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Center of Mass Offset Distribution\nMax: {np.max(all_com_offsets):.1f}px')
    ax2.legend()

    # 3. CoM Position Scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(all_com_x, all_com_y, alpha=0.3, s=5, c=all_size_ratios, cmap='viridis')
    ax3.axhline(256, color='red', linestyle='--', alpha=0.5)
    ax3.axvline(256, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlim(0, 512)
    ax3.set_ylim(512, 0)  # Flip Y for image coordinates
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Center of Mass Positions (colored by size ratio)')
    ax3.set_aspect('equal')
    plt.colorbar(scatter, ax=ax3, label='Size Ratio')

    # 4. Per-Sample Size Ratio Consistency
    ax4 = axes[1, 1]
    sample_means = [s['size_ratio_mean'] for s in all_metrics]
    sample_stds = [s['size_ratio_std'] for s in all_metrics]
    ax4.errorbar(range(len(sample_means)), sample_means, yerr=sample_stds,
                 fmt='o', markersize=2, alpha=0.5, capsize=0)
    ax4.axhline(np.mean(sample_means), color='red', linestyle='--',
                label=f'Overall Mean: {np.mean(sample_means):.3f}')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Size Ratio')
    ax4.set_title(f'Per-Sample Size Ratio (mean ± std)\nIntra-sample CV avg: {np.mean([s["size_ratio_cv"] for s in all_metrics]):.2f}%')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_quality_report.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Additional: Sample visualization with center markers
    create_sample_visualization(all_metrics, output_dir, dataset_name)


def create_sample_visualization(all_metrics: List[Dict], output_dir: Path, dataset_name: str):
    """Create visualization of sample images with center markers."""

    # Find samples with different quality levels
    sorted_by_offset = sorted(all_metrics, key=lambda x: x['com_offset_mean'])

    # Select samples: best, median, worst
    samples_to_show = [
        sorted_by_offset[0],  # Best (smallest offset)
        sorted_by_offset[len(sorted_by_offset)//2],  # Median
        sorted_by_offset[-1],  # Worst (largest offset)
    ]

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    fig.suptitle(f'Sample Visualizations: {dataset_name}\n(Red dot = CoM, Blue dot = Image Center)', fontsize=12)

    for row, sample in enumerate(samples_to_show):
        sample_dir = Path(sample['_data_dir']) / sample['sample_name'] / 'images'
        quality = ['Best', 'Median', 'Worst'][row]

        for col, view_metric in enumerate(sample['view_metrics'][:6]):
            ax = axes[row, col]

            img_path = sample_dir / f"cam_{view_metric['view_idx']:03d}.png"
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)

                # Mark CoM
                com = view_metric['com']
                ax.scatter([com[0]], [com[1]], c='red', s=50, marker='o', zorder=10)

                # Mark image center
                ax.scatter([256], [256], c='blue', s=50, marker='x', zorder=10)

                ax.set_title(f"View {view_metric['view_idx']}\nOffset: {view_metric['com_offset_dist']:.1f}px", fontsize=8)

            ax.axis('off')

            if col == 0:
                ax.set_ylabel(f'{quality}\n(avg offset: {sample["com_offset_mean"]:.1f}px)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(all_metrics: List[Dict], output_dir: Path, dataset_name: str, data_dir: str):
    """Generate markdown report with statistics."""

    # Aggregate statistics
    all_size_ratios = []
    all_com_offsets = []
    all_com_offset_pcts = []

    for sample in all_metrics:
        for view in sample['view_metrics']:
            all_size_ratios.append(view['size_ratio'])
            all_com_offsets.append(view['com_offset_dist'])
            all_com_offset_pcts.append(view['com_offset_pct'])

    # Quality grades
    size_cv = np.std(all_size_ratios) / np.mean(all_size_ratios) * 100
    com_offset_mean = np.mean(all_com_offsets)

    if size_cv < 1.0 and com_offset_mean < 5:
        quality_grade = "A (Excellent)"
    elif size_cv < 3.0 and com_offset_mean < 10:
        quality_grade = "B (Good)"
    elif size_cv < 5.0 and com_offset_mean < 20:
        quality_grade = "C (Acceptable)"
    else:
        quality_grade = "D (Needs Improvement)"

    report = f"""---
date: {datetime.now().strftime('%Y-%m-%d')}
dataset: {dataset_name}
generator: evaluate_preprocessing.py
---

# Preprocessing Quality Report: {dataset_name}

## Overview

| Metric | Value |
|--------|-------|
| Dataset Path | `{data_dir}` |
| Samples Analyzed | {len(all_metrics)} |
| Total Views | {len(all_size_ratios)} |
| Quality Grade | **{quality_grade}** |

## Size Ratio Statistics (Pixel-Based)

> Size ratio = sqrt(object_pixels / total_pixels)
> Target: Uniform across all views

| Metric | Value |
|--------|-------|
| Mean | {np.mean(all_size_ratios):.4f} ({np.mean(all_size_ratios)*100:.1f}%) |
| Std | {np.std(all_size_ratios):.4f} |
| CV (Coefficient of Variation) | **{size_cv:.2f}%** |
| Min | {np.min(all_size_ratios):.4f} ({np.min(all_size_ratios)*100:.1f}%) |
| Max | {np.max(all_size_ratios):.4f} ({np.max(all_size_ratios)*100:.1f}%) |
| Range | {np.max(all_size_ratios) - np.min(all_size_ratios):.4f} |

### Interpretation
- CV < 1%: Excellent uniformity
- CV 1-3%: Good uniformity
- CV 3-5%: Acceptable
- CV > 5%: Needs improvement

## Center of Mass (CoM) Statistics

> CoM offset = distance from image center to object center of mass
> Target: Near zero (object centered)

| Metric | Value |
|--------|-------|
| Mean Offset | **{com_offset_mean:.1f}px** ({np.mean(all_com_offset_pcts):.2f}% of image) |
| Std | {np.std(all_com_offsets):.1f}px |
| Max Offset | {np.max(all_com_offsets):.1f}px |
| 95th Percentile | {np.percentile(all_com_offsets, 95):.1f}px |
| 99th Percentile | {np.percentile(all_com_offsets, 99):.1f}px |

### Interpretation
- Mean < 5px: Excellent centering
- Mean 5-10px: Good centering
- Mean 10-20px: Acceptable
- Mean > 20px: Needs improvement

## Cross-View Consistency

| Metric | Value |
|--------|-------|
| Avg Intra-Sample Size CV | {np.mean([s['size_ratio_cv'] for s in all_metrics]):.2f}% |
| Avg Intra-Sample CoM Offset Std | {np.mean([s['com_offset_std'] for s in all_metrics]):.1f}px |

## Quality Distribution

### Size Ratio Histogram
![Size Ratio Distribution](./{dataset_name}_quality_report.png)

### Sample Visualizations
![Sample Visualizations](./{dataset_name}_sample_visualization.png)

## Recommendations

"""

    # Add recommendations based on metrics
    recommendations = []

    if size_cv > 3.0:
        recommendations.append("- **Size Uniformity**: Consider re-running preprocessing with stricter target ratio enforcement")

    if com_offset_mean > 10:
        recommendations.append("- **Centering**: CoM-based centering may need adjustment. Consider using Visual Hull-based 3D center")

    if np.max(all_com_offsets) > 50:
        recommendations.append(f"- **Outliers**: {sum(1 for x in all_com_offsets if x > 50)} views have offset > 50px. Inspect these samples manually")

    if not recommendations:
        recommendations.append("- Preprocessing quality is good. No immediate improvements needed.")

    report += "\n".join(recommendations)

    report += f"""

## Generated
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Script: `scripts/evaluate_preprocessing.py`
"""

    with open(output_dir / f'{dataset_name}_quality_report.md', 'w') as f:
        f.write(report)

    return quality_grade


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate preprocessing quality with detailed metrics and visualizations"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to preprocessed dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="reports/preprocessing_quality",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to analyze (None=all)"
    )
    parser.add_argument(
        "--img_size", type=int, default=512,
        help="Image size (default: 512)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = data_dir.name

    # Find sample directories
    sample_dirs = sorted(data_dir.glob("sample_*"))
    if args.num_samples:
        sample_dirs = sample_dirs[:args.num_samples]

    print("=" * 70)
    print("Preprocessing Quality Evaluation")
    print("=" * 70)
    print(f"Dataset: {data_dir}")
    print(f"Samples: {len(sample_dirs)}")
    print(f"Output: {output_dir}")
    print()

    # Analyze all samples
    all_metrics = []
    for sample_dir in tqdm(sample_dirs, desc="Analyzing"):
        metrics = analyze_sample(sample_dir, args.img_size)
        if metrics:
            metrics['_data_dir'] = str(data_dir)
            all_metrics.append(metrics)

    if not all_metrics:
        print("ERROR: No valid samples found!")
        return

    print(f"\nAnalyzed {len(all_metrics)} samples successfully")

    # Generate visualizations
    print("Generating visualizations...")
    create_visualization(all_metrics, output_dir, dataset_name)

    # Generate report
    print("Generating report...")
    quality_grade = generate_report(all_metrics, output_dir, dataset_name, str(data_dir))

    # Save detailed metrics as JSON
    json_metrics = {
        'dataset': dataset_name,
        'data_dir': str(data_dir),
        'n_samples': len(all_metrics),
        'summary': {
            'size_ratio_mean': float(np.mean([s['size_ratio_mean'] for s in all_metrics])),
            'size_ratio_std': float(np.std([v['size_ratio'] for s in all_metrics for v in s['view_metrics']])),
            'com_offset_mean': float(np.mean([s['com_offset_mean'] for s in all_metrics])),
            'quality_grade': quality_grade,
        },
        'samples': [{k: v for k, v in s.items() if k != '_data_dir'} for s in all_metrics]
    }

    with open(output_dir / f'{dataset_name}_metrics.json', 'w') as f:
        json.dump(json_metrics, f, indent=2)

    # Print summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Quality Grade: {quality_grade}")
    print()
    print("Size Ratio:")
    all_sizes = [v['size_ratio'] for s in all_metrics for v in s['view_metrics']]
    print(f"  Mean: {np.mean(all_sizes):.4f} ({np.mean(all_sizes)*100:.1f}%)")
    print(f"  Std:  {np.std(all_sizes):.4f}")
    print(f"  CV:   {np.std(all_sizes)/np.mean(all_sizes)*100:.2f}%")

    print()
    print("Center of Mass Offset:")
    all_offsets = [v['com_offset_dist'] for s in all_metrics for v in s['view_metrics']]
    print(f"  Mean: {np.mean(all_offsets):.1f}px")
    print(f"  Std:  {np.std(all_offsets):.1f}px")
    print(f"  Max:  {np.max(all_offsets):.1f}px")

    print()
    print(f"Reports saved to: {output_dir}/")
    print(f"  - {dataset_name}_quality_report.md")
    print(f"  - {dataset_name}_quality_report.png")
    print(f"  - {dataset_name}_sample_visualization.png")
    print(f"  - {dataset_name}_metrics.json")


if __name__ == "__main__":
    main()
