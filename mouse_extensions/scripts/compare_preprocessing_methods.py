#!/usr/bin/env python3
"""
Preprocessing Methods Comparison Report Generator
==================================================

Generates comprehensive comparison reports for different image preprocessing methods.

Features:
- Quantitative metrics table (fx, PP, coverage, samples)
- 6-view qualitative image comparison
- Per-method statistics summary
- Markdown report output

Usage:
    python compare_preprocessing_methods.py \
        --base-dir /home/joon/data/preprocessed/FaceLift_mouse \
        --datasets D6-1 D7 D7_5 D7_5b \
        --output-dir ./reports \
        --sample-idx 100

Created: 2026-01-19
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


@dataclass
class DatasetMetrics:
    """Metrics for a single preprocessing method."""
    name: str
    method: str = "unknown"
    description: str = ""
    num_train: int = 0
    num_val: int = 0
    num_views: int = 6

    # Per-view intrinsics
    fx_per_view: List[float] = field(default_factory=list)
    cx_per_view: List[float] = field(default_factory=list)
    cy_per_view: List[float] = field(default_factory=list)
    coverage_per_view: List[float] = field(default_factory=list)
    scale_per_view: List[float] = field(default_factory=list)

    @property
    def fx_range(self) -> Tuple[float, float]:
        if not self.fx_per_view:
            return (0, 0)
        return (min(self.fx_per_view), max(self.fx_per_view))

    @property
    def fx_fixed(self) -> bool:
        if not self.fx_per_view:
            return True
        return max(self.fx_per_view) - min(self.fx_per_view) < 1.0

    @property
    def pp_fixed(self) -> bool:
        if not self.cx_per_view or not self.cy_per_view:
            return True
        cx_range = max(self.cx_per_view) - min(self.cx_per_view)
        cy_range = max(self.cy_per_view) - min(self.cy_per_view)
        return cx_range < 1.0 and cy_range < 1.0

    @property
    def pp_at_256(self) -> bool:
        if not self.cx_per_view or not self.cy_per_view:
            return False
        return all(abs(cx - 256) < 1.0 for cx in self.cx_per_view) and \
               all(abs(cy - 256) < 1.0 for cy in self.cy_per_view)

    @property
    def avg_coverage(self) -> float:
        if not self.coverage_per_view:
            return 0.0
        return sum(self.coverage_per_view) / len(self.coverage_per_view)


def load_dataset_metrics(base_dir: Path, dataset_name: str) -> Optional[DatasetMetrics]:
    """Load metrics for a dataset from its first sample."""
    dataset_dir = base_dir / dataset_name

    if not dataset_dir.exists():
        print(f"  Warning: Dataset {dataset_name} not found")
        return None

    metrics = DatasetMetrics(name=dataset_name)

    # Count samples
    train_file = dataset_dir / "data_mouse_train.txt"
    val_file = dataset_dir / "data_mouse_val.txt"

    if train_file.exists():
        with open(train_file) as f:
            lines = [l.strip() for l in f if l.strip()]
            metrics.num_train = len(lines)
            first_sample = lines[0] if lines else None

    if val_file.exists():
        with open(val_file) as f:
            metrics.num_val = len([l for l in f if l.strip()])

    # Load camera info from first sample
    if first_sample:
        camera_file = Path(first_sample) / "opencv_cameras.json"
        if camera_file.exists():
            with open(camera_file) as f:
                data = json.load(f)

            # Get preprocessing info
            preprocess = data.get("_preprocessing", {})
            metrics.method = preprocess.get("method", "unknown")
            metrics.description = preprocess.get("description", "")

            # Get per-view metrics
            frames = data.get("frames", [])
            metrics.num_views = len(frames)

            for frame in frames:
                metrics.fx_per_view.append(frame.get("fx", 0))
                metrics.cx_per_view.append(frame.get("cx", 256))
                metrics.cy_per_view.append(frame.get("cy", 256))

                transform = frame.get("_transform", {})
                metrics.coverage_per_view.append(transform.get("coverage", 0))
                metrics.scale_per_view.append(transform.get("scale", 0))

    return metrics


def load_sample_images(
    base_dir: Path,
    dataset_name: str,
    sample_idx: int = 0,
    split: str = "train"
) -> Optional[List[np.ndarray]]:
    """Load 6-view images for a specific sample."""
    dataset_dir = base_dir / dataset_name
    list_file = dataset_dir / f"data_mouse_{split}.txt"

    if not list_file.exists():
        return None

    with open(list_file) as f:
        samples = [l.strip() for l in f if l.strip()]

    if sample_idx >= len(samples):
        sample_idx = 0

    sample_dir = Path(samples[sample_idx])
    images = []

    # Try different image naming conventions
    for view_idx in range(6):
        img_path = None
        for pattern in [
            sample_dir / "images" / f"cam_{view_idx:03d}.png",
            sample_dir / f"cam_{view_idx:03d}.png",
            sample_dir / f"{view_idx}.png",
        ]:
            if pattern.exists():
                img_path = pattern
                break

        if img_path:
            img = np.array(Image.open(img_path))
            images.append(img)
        else:
            images.append(np.zeros((512, 512, 4), dtype=np.uint8))

    return images


def create_comparison_figure(
    all_images: Dict[str, List[np.ndarray]],
    all_metrics: Dict[str, DatasetMetrics],
    output_path: Path,
    sample_idx: int
):
    """Create side-by-side comparison figure."""
    n_datasets = len(all_images)
    n_views = 6

    fig, axes = plt.subplots(n_datasets, n_views, figsize=(18, 3 * n_datasets))
    fig.suptitle(f"Preprocessing Methods Comparison (Sample {sample_idx})", fontsize=14, fontweight='bold')

    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (name, images) in enumerate(all_images.items()):
        metrics = all_metrics.get(name)

        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]

            # Display image (convert RGBA to RGB if needed)
            if img.shape[-1] == 4:
                # Create white background + foreground composite
                rgb = img[:, :, :3].astype(float) / 255
                alpha = img[:, :, 3:4].astype(float) / 255
                white_bg = np.ones_like(rgb)
                composite = rgb * alpha + white_bg * (1 - alpha)
                ax.imshow(composite)
            else:
                ax.imshow(img)

            ax.axis('off')

            # Add column title (view number) for first row only
            if row_idx == 0:
                ax.set_title(f"View {col_idx}", fontsize=10)

            # Add row label with key metrics
            if col_idx == 0 and metrics:
                fx_str = f"{metrics.fx_per_view[0]:.0f}" if metrics.fx_fixed else f"{metrics.fx_range[0]:.0f}-{metrics.fx_range[1]:.0f}"
                pp_str = "PP=256" if metrics.pp_at_256 else f"PP varies"
                label = f"{name}\nfx={fx_str}\n{pp_str}"
                ax.text(-0.15, 0.5, label, transform=ax.transAxes, fontsize=9,
                       verticalalignment='center', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

            # Add per-view fx annotation
            if metrics and col_idx < len(metrics.fx_per_view):
                fx = metrics.fx_per_view[col_idx]
                coverage = metrics.coverage_per_view[col_idx] if col_idx < len(metrics.coverage_per_view) else 0
                ax.text(0.02, 0.98, f"fx={fx:.0f}\ncov={coverage*100:.0f}%",
                       transform=ax.transAxes, fontsize=7, color='red',
                       verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison figure: {output_path}")


def create_coverage_diagram(
    all_metrics: Dict[str, DatasetMetrics],
    output_path: Path
):
    """Create coverage comparison diagram."""
    n_datasets = len(all_metrics)

    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 4))
    fig.suptitle("Image Coverage Comparison (512×512 Output)", fontsize=12, fontweight='bold')

    if n_datasets == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, all_metrics.items()):
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_aspect('equal')
        ax.set_title(f"{name}\nAvg Coverage: {metrics.avg_coverage*100:.1f}%", fontsize=10)

        # Draw output frame
        ax.add_patch(patches.Rectangle((0, 0), 512, 512,
                     linewidth=2, edgecolor='black', facecolor='lightgray'))

        # Draw PP marker
        ax.plot(256, 256, 'r+', markersize=20, markeredgewidth=2, label='PP (256,256)')

        # Estimate coverage area (simplified visualization)
        avg_scale = sum(metrics.scale_per_view) / len(metrics.scale_per_view) if metrics.scale_per_view else 0.4
        orig_w, orig_h = 1152 * avg_scale, 1024 * avg_scale

        # Center image area
        x_start = (512 - orig_w) / 2
        y_start = (512 - orig_h) / 2

        ax.add_patch(patches.Rectangle((x_start, y_start), orig_w, orig_h,
                     linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5,
                     label='Scaled Original'))

        ax.legend(loc='upper right', fontsize=8)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved coverage diagram: {output_path}")


def generate_markdown_report(
    all_metrics: Dict[str, DatasetMetrics],
    output_path: Path,
    image_paths: Dict[str, Path]
):
    """Generate comprehensive markdown report."""
    report = []
    report.append(f"# Preprocessing Methods Comparison Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary table
    report.append("## 1. Summary Table\n")
    report.append("| Dataset | Method | Samples | fx | PP | Coverage |")
    report.append("|---------|--------|---------|-----|-----|----------|")

    for name, m in all_metrics.items():
        fx_str = f"{m.fx_range[0]:.0f}" if m.fx_fixed else f"{m.fx_range[0]:.0f}-{m.fx_range[1]:.0f}"
        pp_str = "(256,256)" if m.pp_at_256 else "varies"
        cov_str = f"{m.avg_coverage*100:.1f}%" if m.coverage_per_view else "N/A"
        report.append(f"| **{name}** | {m.method} | {m.num_train + m.num_val} | {fx_str} | {pp_str} | {cov_str} |")

    # Detailed per-view metrics
    report.append("\n## 2. Per-View Metrics\n")

    for name, m in all_metrics.items():
        report.append(f"### {name}\n")
        report.append(f"- **Method**: `{m.method}`")
        report.append(f"- **Description**: {m.description or 'N/A'}")
        report.append(f"- **Train/Val**: {m.num_train}/{m.num_val}\n")

        if m.fx_per_view:
            report.append("| View | fx | cx | cy | Scale | Coverage |")
            report.append("|------|-----|-----|-----|-------|----------|")

            for i in range(m.num_views):
                fx = m.fx_per_view[i] if i < len(m.fx_per_view) else 0
                cx = m.cx_per_view[i] if i < len(m.cx_per_view) else 256
                cy = m.cy_per_view[i] if i < len(m.cy_per_view) else 256
                scale = m.scale_per_view[i] if i < len(m.scale_per_view) else 0
                coverage = m.coverage_per_view[i] if i < len(m.coverage_per_view) else 0
                report.append(f"| {i} | {fx:.1f} | {cx:.1f} | {cy:.1f} | {scale:.4f} | {coverage*100:.1f}% |")
        report.append("")

    # Method comparison analysis
    report.append("## 3. Method Comparison Analysis\n")

    report.append("### Key Differences\n")
    report.append("| Aspect | " + " | ".join(all_metrics.keys()) + " |")
    report.append("|--------|" + "|".join(["---"] * len(all_metrics)) + "|")

    # fx consistency
    row = "| **fx Fixed** |"
    for m in all_metrics.values():
        row += f" {'Yes' if m.fx_fixed else 'No'} |"
    report.append(row)

    # PP at 256
    row = "| **PP at (256,256)** |"
    for m in all_metrics.values():
        row += f" {'Yes' if m.pp_at_256 else 'No'} |"
    report.append(row)

    # Coverage
    row = "| **Avg Coverage** |"
    for m in all_metrics.values():
        cov = f"{m.avg_coverage*100:.1f}%" if m.coverage_per_view else "N/A"
        row += f" {cov} |"
    report.append(row)

    # Trade-offs
    report.append("\n### Trade-off Summary\n")
    report.append("| Method | Advantage | Disadvantage |")
    report.append("|--------|-----------|--------------|")

    for name, m in all_metrics.items():
        if "pp_centered" in m.method.lower() or "D7_PP" in m.method:
            adv = "PP mathematically correct, matches pretrained model"
            disadv = "Object may be off-center, ~56% coverage"
        elif "object_aware" in m.method.lower() or "D7_5b" in name:
            # Check D7_5b BEFORE D7_5 since D7_5 in name would match both
            adv = "Maximum object resolution, 100% object coverage"
            disadv = "fx varies per view (680-793)"
        elif "optimal_coverage" in m.method.lower() or "D7_5" in name:
            adv = "Full image preserved, 72-87% coverage"
            disadv = "fx varies per view (641-706)"
        elif "D6" in name:
            adv = "Object centered in frame, higher fx (~555)"
            disadv = "PP varies per view, geometric inconsistency"
        else:
            adv = "N/A"
            disadv = "N/A"
        report.append(f"| **{name}** | {adv} | {disadv} |")

    # Image references
    report.append("\n## 4. Visual Comparison\n")
    for name, path in image_paths.items():
        report.append(f"### {name}\n")
        report.append(f"![{name}]({path.name})\n")

    # Recommendations
    report.append("## 5. Recommendations\n")
    report.append("""
| Use Case | Recommended Dataset | Reason |
|----------|---------------------|--------|
| **FaceLift pretrained compatibility** | D7 | fx=549, PP=(256,256) matches exactly |
| **Maximum image quality** | D7_5 | Full image preserved, no cropping |
| **Maximum object detail** | D7_5b | Object-aware zoom, highest resolution |
| **Legacy/Debugging** | D6-1 | Object-centered for visual inspection |
""")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"  Saved markdown report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate preprocessing methods comparison report"
    )
    parser.add_argument('--base-dir', required=True,
                        help='Base directory containing preprocessed datasets')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Dataset names to compare (e.g., D6-1 D7 D7_5 D7_5b)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for report and figures')
    parser.add_argument('--sample-idx', type=int, default=100,
                        help='Sample index to use for visual comparison')
    parser.add_argument('--report-name', type=str, default=None,
                        help='Report name prefix (default: auto-generated)')

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report name
    if args.report_name:
        report_prefix = args.report_name
    else:
        report_prefix = f"preprocessing_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*60}")
    print("Preprocessing Methods Comparison Report Generator")
    print(f"{'='*60}")
    print(f"Base dir: {base_dir}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Sample idx: {args.sample_idx}")
    print(f"Output: {output_dir}")
    print()

    # Load metrics
    print("Loading dataset metrics...")
    all_metrics = {}
    for dataset_name in args.datasets:
        print(f"  Loading {dataset_name}...")
        metrics = load_dataset_metrics(base_dir, dataset_name)
        if metrics:
            all_metrics[dataset_name] = metrics

    if not all_metrics:
        print("Error: No valid datasets found!")
        return

    # Load sample images
    print("\nLoading sample images...")
    all_images = {}
    for dataset_name in args.datasets:
        print(f"  Loading {dataset_name} sample {args.sample_idx}...")
        images = load_sample_images(base_dir, dataset_name, args.sample_idx)
        if images:
            all_images[dataset_name] = images

    # Generate figures
    print("\nGenerating figures...")
    image_paths = {}

    # 6-view comparison
    comparison_path = output_dir / f"{report_prefix}_6view_comparison.png"
    create_comparison_figure(all_images, all_metrics, comparison_path, args.sample_idx)
    image_paths["6-View Comparison"] = comparison_path

    # Coverage diagram
    coverage_path = output_dir / f"{report_prefix}_coverage.png"
    create_coverage_diagram(all_metrics, coverage_path)
    image_paths["Coverage Diagram"] = coverage_path

    # Generate markdown report
    print("\nGenerating markdown report...")
    report_path = output_dir / f"{report_prefix}_report.md"
    generate_markdown_report(all_metrics, report_path, image_paths)

    print(f"\n{'='*60}")
    print("Report generation complete!")
    print(f"{'='*60}")
    print(f"Output files:")
    print(f"  - Report: {report_path}")
    for name, path in image_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
