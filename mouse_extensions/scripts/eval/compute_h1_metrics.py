#!/usr/bin/env python3
"""
Compute metrics for H1 diagnosis experiments.

Usage:
    python -m mouse_extensions.scripts.eval.compute_h1_metrics --dataset M5t2
    python -m mouse_extensions.scripts.eval.compute_h1_metrics --dataset M5t --all-views
"""

import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mouse_extensions.evaluation import (
    MetricsComputer,
    ReportGenerator,
    generate_h1_comparison_report,
)


def main():
    parser = argparse.ArgumentParser(description="Compute H1 experiment metrics")
    parser.add_argument(
        "--dataset",
        type=str,
        default="M5t2",
        choices=["M5t2", "M5t"],
        help="Dataset name (determines paths)",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=None,
        help="Override experiments directory",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for reports",
    )
    parser.add_argument(
        "--all-views",
        action="store_true",
        help="Compare all 6 views (default: only view 0)",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip LPIPS computation (faster)",
    )
    args = parser.parse_args()

    # Set default paths based on dataset
    facelift_root = Path("/home/joon/dev/FaceLift")
    data_root = Path("/home/joon/data/preprocessed/FaceLift_mouse")

    if args.experiments_dir:
        experiments_dir = Path(args.experiments_dir)
    else:
        experiments_dir = facelift_root / "outputs" / "eval" / f"h1_diagnosis_{args.dataset}"

    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = data_root / "M5"  # Both M5t and M5t2 use M5 dataset

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = experiments_dir / "reports"

    print(f"=== H1 Metrics Computation ===")
    print(f"Dataset: {args.dataset}")
    print(f"Experiments dir: {experiments_dir}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_dir}")
    print(f"Compare all views: {args.all_views}")
    print(f"Compute LPIPS: {not args.no_lpips}")
    print()

    # Generate reports
    reports = generate_h1_comparison_report(
        experiments_dir=experiments_dir,
        dataset_root=dataset_root,
        output_dir=output_dir,
        dataset_name=args.dataset,
        compare_all_views=args.all_views,
        compute_lpips=not args.no_lpips,
    )

    print("\n=== Summary ===")
    for name, path in reports.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
