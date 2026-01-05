#!/usr/bin/env python3
"""
Compare All Preprocessing Methods - Comprehensive Report Generator

Evaluates all mouse preprocessing datasets and generates:
1. Quantitative comparison table (Size CV, CoM Offset, Grade)
2. Qualitative visual comparison grid
3. Per-sample visualization for worst/best cases

Usage:
    python scripts/compare_preprocessing_all.py \
        --data_dirs data_mouse data_mouse_centered data_mouse_pixel_based \
        --output_dir reports/preprocessing_comparison \
        --num_samples 50
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_center_of_mass(alpha: np.ndarray, threshold: int = 10) -> Optional[Tuple[float, float]]:
    """Calculate center of mass from alpha mask."""
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
    """Get size ratio using sqrt(pixel_count / total_pixels)."""
    mask = alpha > threshold
    if not mask.any():
        return 0.0
    pixel_count = np.sum(mask)
    h, w = alpha.shape
    area_ratio = pixel_count / (h * w)
    return float(np.sqrt(area_ratio))


def analyze_sample(sample_dir: Path) -> Optional[Dict]:
    """Analyze a single sample and return metrics."""
    images_dir = sample_dir / "images"
    if not images_dir.exists():
        return None

    size_ratios = []
    com_offsets = []

    for img_path in sorted(images_dir.glob("*.png")):
        img = Image.open(img_path)
        if img.mode != 'RGBA':
            continue

        alpha = np.array(img)[:, :, 3]
        h, w = alpha.shape
        center = w / 2, h / 2

        com = get_center_of_mass(alpha)
        size_ratio = get_pixel_size_ratio(alpha)

        if com and size_ratio > 0:
            offset = np.sqrt((com[0] - center[0])**2 + (com[1] - center[1])**2)
            size_ratios.append(size_ratio)
            com_offsets.append(offset)

    if not size_ratios:
        return None

    return {
        'size_ratio_mean': np.mean(size_ratios),
        'size_ratio_std': np.std(size_ratios),
        'com_offset_mean': np.mean(com_offsets),
        'com_offset_max': np.max(com_offsets),
        'n_views': len(size_ratios),
    }


def analyze_dataset(data_dir: Path, num_samples: int = 50) -> Dict:
    """Analyze entire dataset and return aggregated metrics."""
    sample_dirs = sorted(data_dir.glob("sample_*"))[:num_samples]

    all_size_ratios = []
    all_com_offsets = []
    sample_metrics = []

    for sample_dir in tqdm(sample_dirs, desc=f"Analyzing {data_dir.name}", leave=False):
        result = analyze_sample(sample_dir)
        if result:
            all_size_ratios.append(result['size_ratio_mean'])
            all_com_offsets.append(result['com_offset_mean'])
            sample_metrics.append({
                'name': sample_dir.name,
                **result
            })

    if not all_size_ratios:
        return {'error': 'No valid samples'}

    size_mean = np.mean(all_size_ratios)
    size_std = np.std(all_size_ratios)
    size_cv = (size_std / size_mean * 100) if size_mean > 0 else 0

    offset_mean = np.mean(all_com_offsets)
    offset_std = np.std(all_com_offsets)
    offset_max = np.max(all_com_offsets)

    # Grade calculation
    if size_cv < 1 and offset_mean < 5:
        grade = 'A'
    elif size_cv < 3 and offset_mean < 10:
        grade = 'B'
    elif size_cv < 5 and offset_mean < 20:
        grade = 'C'
    else:
        grade = 'D'

    return {
        'name': data_dir.name,
        'n_samples': len(sample_metrics),
        'size_ratio_mean': size_mean,
        'size_ratio_std': size_std,
        'size_cv': size_cv,
        'com_offset_mean': offset_mean,
        'com_offset_std': offset_std,
        'com_offset_max': offset_max,
        'grade': grade,
        'sample_metrics': sample_metrics,
    }


def load_sample_image(sample_dir: Path, view_idx: int = 0) -> Optional[np.ndarray]:
    """Load a sample image from a sample directory."""
    images_dir = sample_dir / "images"
    if not images_dir.exists():
        return None

    patterns = [f"cam_{view_idx:03d}.png", f"{view_idx:02d}.png"]
    for pattern in patterns:
        img_path = images_dir / pattern
        if img_path.exists():
            return np.array(Image.open(img_path))

    # Try first available
    images = sorted(images_dir.glob("*.png"))
    if images:
        return np.array(Image.open(images[view_idx % len(images)]))
    return None


def create_comparison_figure(
    datasets: List[Dict],
    output_dir: Path,
    sample_idx: int = 0,
    view_idx: int = 0,
) -> None:
    """Create side-by-side comparison figure."""
    n_datasets = len(datasets)

    fig, axes = plt.subplots(2, n_datasets, figsize=(4 * n_datasets, 8))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)

    for i, dataset in enumerate(datasets):
        # Top row: Sample image
        sample_dirs = sorted(Path(dataset['path']).glob("sample_*"))
        if sample_idx < len(sample_dirs):
            img = load_sample_image(sample_dirs[sample_idx], view_idx)
            if img is not None:
                if img.shape[-1] == 4:  # RGBA
                    # Composite on white
                    rgb = img[:, :, :3].astype(float)
                    alpha = img[:, :, 3:4].astype(float) / 255
                    white = np.ones_like(rgb) * 255
                    composite = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
                    axes[0, i].imshow(composite)
                else:
                    axes[0, i].imshow(img)

        axes[0, i].set_title(f"{dataset['name']}\nGrade: {dataset['grade']}", fontsize=10)
        axes[0, i].axis('off')

        # Add crosshair at center
        if img is not None:
            h, w = img.shape[:2]
            axes[0, i].axhline(h/2, color='red', linewidth=0.5, alpha=0.5)
            axes[0, i].axvline(w/2, color='red', linewidth=0.5, alpha=0.5)

        # Bottom row: Metrics bar chart
        metrics = [
            ('Size CV', dataset['size_cv'], 5, 'lower'),
            ('CoM Offset', dataset['com_offset_mean'], 50, 'lower'),
        ]

        labels = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        max_vals = [m[2] for m in metrics]
        normalized = [v / m for v, m in zip(values, max_vals)]

        colors = []
        for m_label, v, m, direction in metrics:
            ratio = v / m
            if direction == 'lower':
                if ratio < 0.2:
                    colors.append('green')
                elif ratio < 0.5:
                    colors.append('yellow')
                else:
                    colors.append('red')
            else:
                if ratio > 0.8:
                    colors.append('green')
                elif ratio > 0.5:
                    colors.append('yellow')
                else:
                    colors.append('red')

        bars = axes[1, i].bar(labels, normalized, color=colors, alpha=0.7)
        axes[1, i].set_ylim(0, 1.2)
        axes[1, i].axhline(1.0, color='gray', linestyle='--', alpha=0.5)

        for bar, val in zip(bars, values):
            axes[1, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        axes[1, i].set_ylabel('Normalized Score' if i == 0 else '')
        axes[1, i].set_title(f"Size CV: {dataset['size_cv']:.2f}%\nCoM Offset: {dataset['com_offset_mean']:.1f}px",
                            fontsize=9)

    plt.suptitle(f"Preprocessing Comparison (Sample {sample_idx}, View {view_idx})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_grid.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_table(datasets: List[Dict], output_dir: Path) -> str:
    """Create markdown summary table."""
    lines = [
        "# Preprocessing Methods Comparison Report",
        "",
        f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary Table",
        "",
        "| Dataset | Samples | Size Mean | Size CV | CoM Offset | Grade |",
        "|---------|---------|-----------|---------|------------|-------|",
    ]

    # Sort by grade
    grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    sorted_datasets = sorted(datasets, key=lambda x: grade_order.get(x.get('grade', 'D'), 4))

    for d in sorted_datasets:
        if 'error' in d:
            lines.append(f"| {d['name']} | - | - | - | - | ERROR |")
        else:
            grade_emoji = {'A': '', 'B': '', 'C': '', 'D': ''}[d['grade']]
            lines.append(
                f"| {d['name']} | {d['n_samples']} | "
                f"{d['size_ratio_mean']:.1%} | {d['size_cv']:.2f}% | "
                f"{d['com_offset_mean']:.1f}px | {d['grade']} {grade_emoji} |"
            )

    lines.extend([
        "",
        "## Grading Criteria",
        "",
        "| Grade | Size CV | CoM Offset | Description |",
        "|-------|---------|------------|-------------|",
        "| A | < 1% | < 5px | Excellent - Ready for training |",
        "| B | < 3% | < 10px | Good - Minor improvements possible |",
        "| C | < 5% | < 20px | Fair - Needs improvement |",
        "| D | >= 5% | >= 20px | Poor - Requires reprocessing |",
        "",
    ])

    # Add detailed analysis
    lines.extend([
        "## Detailed Analysis",
        "",
    ])

    for d in sorted_datasets:
        if 'error' in d:
            continue

        lines.extend([
            f"### {d['name']}",
            "",
            f"- **Grade**: {d['grade']}",
            f"- **Samples Analyzed**: {d['n_samples']}",
            f"- **Size Ratio**:",
            f"  - Mean: {d['size_ratio_mean']:.1%}",
            f"  - Std: {d['size_ratio_std']:.2%}",
            f"  - CV: {d['size_cv']:.2f}%",
            f"- **Center of Mass Offset**:",
            f"  - Mean: {d['com_offset_mean']:.1f}px",
            f"  - Std: {d['com_offset_std']:.1f}px",
            f"  - Max: {d['com_offset_max']:.1f}px",
            "",
        ])

    # Recommendations
    lines.extend([
        "## Recommendations",
        "",
    ])

    best = sorted_datasets[0] if sorted_datasets else None
    if best and best.get('grade') in ['A', 'B']:
        lines.append(f"**Recommended Dataset**: `{best['name']}` (Grade {best['grade']})")
        lines.append("")
        lines.append(f"This dataset has the best combination of size uniformity (CV {best['size_cv']:.2f}%) "
                    f"and centering accuracy (offset {best['com_offset_mean']:.1f}px).")
    else:
        lines.append("**Warning**: No dataset achieved Grade A or B.")
        lines.append("Consider applying pixel-based preprocessing with CoM centering.")

    lines.append("")

    report = "\n".join(lines)

    with open(output_dir / "comparison_report.md", 'w') as f:
        f.write(report)

    return report


def create_multi_sample_comparison(
    datasets: List[Dict],
    output_dir: Path,
    num_samples: int = 5,
    num_views: int = 3,
) -> None:
    """Create multi-sample visual comparison."""
    n_datasets = len(datasets)

    fig, axes = plt.subplots(num_samples, n_datasets,
                             figsize=(3 * n_datasets, 3 * num_samples))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, dataset in enumerate(datasets):
        sample_dirs = sorted(Path(dataset['path']).glob("sample_*"))

        for j in range(num_samples):
            if j < len(sample_dirs):
                img = load_sample_image(sample_dirs[j], view_idx=0)
                if img is not None:
                    if img.shape[-1] == 4:
                        rgb = img[:, :, :3].astype(float)
                        alpha = img[:, :, 3:4].astype(float) / 255
                        white = np.ones_like(rgb) * 255
                        composite = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
                        axes[j, i].imshow(composite)
                    else:
                        axes[j, i].imshow(img)

                    # Add center crosshair
                    h, w = img.shape[:2]
                    axes[j, i].axhline(h/2, color='red', linewidth=0.3, alpha=0.5)
                    axes[j, i].axvline(w/2, color='red', linewidth=0.3, alpha=0.5)

            axes[j, i].axis('off')

            if j == 0:
                axes[j, i].set_title(f"{dataset['name']}\n({dataset['grade']})", fontsize=8)

    plt.suptitle("Multi-Sample Comparison (View 0)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "multi_sample_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_bar_chart_comparison(datasets: List[Dict], output_dir: Path) -> None:
    """Create bar chart comparing all datasets."""
    valid_datasets = [d for d in datasets if 'error' not in d]

    if not valid_datasets:
        return

    names = [d['name'].replace('data_mouse_', '').replace('data_mouse', 'original')
             for d in valid_datasets]
    size_cvs = [d['size_cv'] for d in valid_datasets]
    com_offsets = [d['com_offset_mean'] for d in valid_datasets]
    grades = [d['grade'] for d in valid_datasets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Size CV
    colors = ['green' if cv < 1 else 'yellow' if cv < 3 else 'orange' if cv < 5 else 'red'
              for cv in size_cvs]
    bars1 = ax1.bar(names, size_cvs, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(1, color='green', linestyle='--', alpha=0.5, label='Grade A threshold')
    ax1.axhline(3, color='yellow', linestyle='--', alpha=0.5, label='Grade B threshold')
    ax1.axhline(5, color='orange', linestyle='--', alpha=0.5, label='Grade C threshold')
    ax1.set_ylabel('Size CV (%)')
    ax1.set_title('Size Uniformity (Lower is Better)')
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend(loc='upper right', fontsize=8)

    for bar, cv, grade in zip(bars1, size_cvs, grades):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{cv:.2f}%\n({grade})', ha='center', va='bottom', fontsize=8)

    # CoM Offset
    colors = ['green' if offset < 5 else 'yellow' if offset < 10 else 'orange' if offset < 20 else 'red'
              for offset in com_offsets]
    bars2 = ax2.bar(names, com_offsets, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(5, color='green', linestyle='--', alpha=0.5, label='Grade A threshold')
    ax2.axhline(10, color='yellow', linestyle='--', alpha=0.5, label='Grade B threshold')
    ax2.axhline(20, color='orange', linestyle='--', alpha=0.5, label='Grade C threshold')
    ax2.set_ylabel('CoM Offset (pixels)')
    ax2.set_title('Centering Accuracy (Lower is Better)')
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=8)

    for bar, offset, grade in zip(bars2, com_offsets, grades):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{offset:.1f}px\n({grade})', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Preprocessing Quality Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "bar_chart_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare all preprocessing methods")
    parser.add_argument("--data_dirs", nargs='+', required=True,
                        help="List of data directories to compare")
    parser.add_argument("--output_dir", type=str, default="reports/preprocessing_comparison",
                        help="Output directory for reports")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to analyze per dataset")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Preprocessing Comparison Tool")
    print("=" * 70)
    print(f"Datasets: {len(args.data_dirs)}")
    print(f"Samples per dataset: {args.num_samples}")
    print(f"Output: {output_dir}")
    print()

    # Analyze all datasets
    all_results = []
    for data_dir in args.data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: {data_dir} does not exist, skipping")
            continue

        print(f"Analyzing {data_dir}...")
        result = analyze_dataset(data_path, args.num_samples)
        result['path'] = str(data_path)
        all_results.append(result)

    if not all_results:
        print("No valid datasets found!")
        return

    # Generate reports
    print("\nGenerating reports...")

    # 1. Summary table
    report = create_summary_table(all_results, output_dir)
    print("\n" + "=" * 70)
    print(report)

    # 2. Bar chart comparison
    create_bar_chart_comparison(all_results, output_dir)
    print(f"Saved: {output_dir}/bar_chart_comparison.png")

    # 3. Visual comparison grid
    create_comparison_figure(all_results, output_dir, sample_idx=0, view_idx=0)
    print(f"Saved: {output_dir}/comparison_grid.png")

    # 4. Multi-sample comparison
    create_multi_sample_comparison(all_results, output_dir, num_samples=5)
    print(f"Saved: {output_dir}/multi_sample_comparison.png")

    # 5. Save JSON metrics
    metrics_output = []
    for r in all_results:
        r_copy = r.copy()
        r_copy.pop('sample_metrics', None)  # Remove large nested data
        metrics_output.append(r_copy)

    with open(output_dir / "comparison_metrics.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Saved: {output_dir}/comparison_metrics.json")

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
