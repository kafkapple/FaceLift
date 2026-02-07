#!/usr/bin/env python3
"""
Compute metrics for E2E inference outputs vs GT.

Compares render_view_{00-05}.png against GT cam_{000-005}.png
for each sample in the output directory.

Usage:
    # Single experiment
    python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
        --output_dir outputs/h5_e2e/cfgr_ckpt10000 \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5

    # Compare two experiments
    python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
        --output_dir outputs/h5_e2e/cfgr_ckpt10000 outputs/h5_e2e/baseline_ckpt5000 \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5

    # Input view only (view 0 = MVDiffusion input, skip from eval)
    python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
        --output_dir outputs/h5_e2e/cfgr_ckpt10000 \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --skip_input_view 0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mouse_extensions.evaluation.metrics import MetricsComputer, MetricResult


def compute_e2e_metrics(
    output_dir: str,
    data_dir: str,
    views: list = None,
    skip_input_view: int = None,
    device: str = "cuda",
    no_lpips: bool = False,
) -> dict:
    """
    Compute PSNR/SSIM/LPIPS for E2E inference outputs.

    Args:
        output_dir: E2E output directory (contains samples/ subdirectory)
        data_dir: GT dataset root (M5)
        views: List of view indices to evaluate (default: all 6)
        skip_input_view: Skip this view index (input to MVDiffusion)
        device: Computation device
        no_lpips: Skip LPIPS computation

    Returns:
        Dict with per-view and overall metrics
    """
    output_path = Path(output_dir)
    data_path = Path(data_dir)
    samples_dir = output_path / "samples"

    if not samples_dir.exists():
        print(f"ERROR: No samples/ directory in {output_dir}")
        return {}

    # Determine views to evaluate
    if views is None:
        views = list(range(6))
    if skip_input_view is not None and skip_input_view in views:
        views = [v for v in views if v != skip_input_view]

    computer = MetricsComputer(device=device, compute_lpips=not no_lpips)

    # Find all sample directories
    sample_dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
    )

    # E2E outputs have cam_000/ subdirectory
    # Structure: samples/{sample_id}/cam_000/render_view_{vv}.png
    per_view_results = {v: [] for v in views}
    all_results = []
    errors = []

    print(f"Evaluating {len(sample_dirs)} samples, views: {views}")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        gt_images_dir = data_path / sample_id / "images"

        if not gt_images_dir.exists():
            errors.append(f"GT not found: {sample_id}")
            continue

        # Find the cam subdirectory (cam_000 for input_view_idx=0)
        cam_subdirs = [d for d in sample_dir.iterdir() if d.is_dir() and d.name.startswith("cam_")]
        if not cam_subdirs:
            # Renders might be directly in sample dir
            render_dir = sample_dir
        else:
            render_dir = cam_subdirs[0]  # Use first cam subdir

        for view_idx in views:
            render_path = render_dir / f"render_view_{view_idx:02d}.png"
            gt_path = gt_images_dir / f"cam_{view_idx:03d}.png"

            if not render_path.exists():
                continue
            if not gt_path.exists():
                continue

            result = computer.compute(
                render_path, gt_path,
                sample_id=f"{sample_id}_v{view_idx}",
            )
            result.metadata["view_idx"] = view_idx
            result.metadata["sample_id"] = sample_id

            per_view_results[view_idx].append(result)
            all_results.append(result)

    # Aggregate
    overall = computer.aggregate(all_results)
    per_view_agg = {}
    for v in views:
        if per_view_results[v]:
            agg = computer.aggregate(per_view_results[v])
            per_view_agg[f"view_{v}"] = agg.to_dict()

    summary = {
        "experiment": output_path.name,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(sample_dirs),
        "n_evaluated": len(set(r.metadata["sample_id"] for r in all_results)),
        "views_evaluated": views,
        "overall": overall.to_dict(),
        "per_view": per_view_agg,
    }

    if errors:
        summary["errors"] = errors[:10]  # Limit error list

    return summary


def print_summary(summary: dict, label: str = ""):
    """Pretty print metrics summary."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    o = summary["overall"]
    n = summary["n_evaluated"]
    print(f"\n  Samples: {n}")
    print(f"  Views: {summary['views_evaluated']}")
    print(f"\n  Overall:")
    print(f"    PSNR:  {o['psnr']['mean']:.4f} ± {o['psnr']['std']:.4f}")
    print(f"    SSIM:  {o['ssim']['mean']:.4f} ± {o['ssim']['std']:.4f}")
    if "lpips" in o:
        print(f"    LPIPS: {o['lpips']['mean']:.4f} ± {o['lpips']['std']:.4f}")

    print(f"\n  Per-view:")
    for vk, vv in sorted(summary["per_view"].items()):
        lpips_str = f", LPIPS={vv['lpips']['mean']:.4f}" if "lpips" in vv else ""
        print(f"    {vk}: PSNR={vv['psnr']['mean']:.4f}, SSIM={vv['ssim']['mean']:.4f}{lpips_str}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute E2E inference metrics")
    parser.add_argument(
        "--output_dir", type=str, nargs="+", required=True,
        help="E2E output directories (1 or more for comparison)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="GT dataset root (e.g. M5)",
    )
    parser.add_argument(
        "--views", type=int, nargs="+", default=None,
        help="View indices to evaluate (default: all 6)",
    )
    parser.add_argument(
        "--skip_input_view", type=int, default=None,
        help="Skip input view from evaluation (e.g. 0 for MVDiffusion input)",
    )
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = []
    for out_dir in args.output_dir:
        print(f"\nProcessing: {out_dir}")
        summary = compute_e2e_metrics(
            output_dir=out_dir,
            data_dir=args.data_dir,
            views=args.views,
            skip_input_view=args.skip_input_view,
            device=args.device,
            no_lpips=args.no_lpips,
        )
        results.append(summary)

        # Save per-experiment results
        metrics_path = Path(out_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {metrics_path}")

        print_summary(summary, label=Path(out_dir).name)

    # Comparison table
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        print(f"\n  {'Experiment':<30} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'N':>5}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
        for r in results:
            o = r["overall"]
            lpips_str = f"{o['lpips']['mean']:.4f}" if "lpips" in o else "N/A"
            print(f"  {r['experiment']:<30} {o['psnr']['mean']:>8.4f} {o['ssim']['mean']:>8.4f} {lpips_str:>8} {r['n_evaluated']:>5}")
        print()


if __name__ == "__main__":
    main()
