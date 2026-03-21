#!/usr/bin/env python3
"""
unified_eval.py — Unified Evaluation Pipeline for GS-LRM Experiments
=====================================================================

Single entry point for ALL evaluation metrics:
  1. Fair comparison (PSNR, SSIM, L1, IoU) — from fair_comparison.py
  2. Artifact metrics (CAS, FAS, EFS, OAS) — from artifact_metrics.py
  3. Gaussian quality (anisotropy, opacity, scale) — from gaussian_quality_metrics.py

Usage:
    # Evaluate a single experiment
    python -m mouse_extensions.scripts.eval.unified_eval run \
        --name 6view_alpha03_v3 \
        --render_dir outputs/datasets/alpha_comparison/6view_alpha03_v3 \
        --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --views 1 2 3 4 5 \
        --output experiments/comparison/alpha/6view_alpha03_v3_unified.json

    # Compare multiple experiments
    python -m mouse_extensions.scripts.eval.unified_eval compare \
        --results experiments/comparison/alpha/*_unified.json \
        --output experiments/comparison/alpha/comparison_table.md

    # Evaluate with artifact metrics only (no GT comparison)
    python -m mouse_extensions.scripts.eval.unified_eval run \
        --name 6view_alpha03_v3 \
        --render_dir outputs/datasets/alpha_comparison/6view_alpha03_v3 \
        --metrics artifact \
        --output experiments/comparison/alpha/6view_alpha03_v3_artifact.json

Author: FaceLift Mouse Extensions
Date: 2026-03-21
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ============================================================
# Metric Runners (thin wrappers around existing modules)
# ============================================================

def run_fair_comparison(
    render_dir: str,
    gt_dir: str,
    views: List[int] = None,
    max_frames: Optional[int] = None,
) -> Dict:
    """Run fair comparison metrics (PSNR, SSIM, L1, IoU).

    Reuses compute functions from fair_comparison.py.
    """
    if views is None:
        views = [1, 2, 3, 4, 5]

    from mouse_extensions.scripts.eval.fair_comparison import (
        compute_all_metrics, extract_foreground_mask,
    )

    samples_dir = Path(render_dir) / "samples"
    if not samples_dir.exists():
        samples_dir = Path(render_dir)

    frames = sorted([d for d in os.listdir(samples_dir) if d.isdigit()])
    if max_frames:
        frames = frames[:max_frames]

    # Accumulate per-metric lists
    metric_lists = {}

    for frame_id in frames:
        frame_dir = samples_dir / frame_id
        gt_frame_dir = Path(gt_dir) / frame_id

        for view_idx in views:
            render_path = frame_dir / f"render_view_{view_idx:02d}.png"
            gt_path = gt_frame_dir / "images" / f"cam_{view_idx:03d}.png"

            if not render_path.exists() or not gt_path.exists():
                continue

            pred = np.array(Image.open(render_path).convert("RGB")).astype(np.float32) / 255.0
            gt_img = np.array(Image.open(gt_path))

            # Extract GT mask from alpha channel
            if gt_img.shape[-1] == 4:
                mask = (gt_img[:, :, 3] > 127).astype(np.float32)
                gt_rgb = gt_img[:, :, :3].astype(np.float32) / 255.0
            else:
                gt_rgb = gt_img.astype(np.float32) / 255.0
                mask = (gt_rgb.mean(axis=-1) < 0.95).astype(np.float32)

            if mask.sum() < 100:
                continue

            m = compute_all_metrics(pred, gt_rgb, mask)
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    metric_lists.setdefault(k, []).append(v)

    def stat(arr):
        a = np.array(arr)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}

    return {k: stat(v) for k, v in metric_lists.items()}


def run_artifact_metrics(
    render_dir: str,
    views: List[int] = None,
    max_frames: int = 50,
) -> Dict:
    """Run artifact detection metrics (CAS, FAS, EFS, OAS).

    Reuses compute_artifact_metrics from artifact_metrics.py.
    """
    if views is None:
        views = [1, 2, 3, 4, 5]

    from mouse_extensions.scripts.eval.artifact_metrics import compute_artifact_metrics

    samples_dir = Path(render_dir) / "samples"
    if not samples_dir.exists():
        samples_dir = Path(render_dir)

    frames = sorted([d for d in os.listdir(samples_dir) if d.isdigit()])[:max_frames]

    per_view = {}
    all_metrics = []

    for view_idx in views:
        view_metrics = []
        for frame_id in frames:
            render_path = samples_dir / frame_id / f"render_view_{view_idx:02d}.png"
            alpha_path = samples_dir / frame_id / f"render_alpha_{view_idx:02d}.png"

            if not render_path.exists():
                continue

            rgb = np.array(Image.open(render_path).convert("RGB")).astype(np.float32) / 255.0
            alpha = None
            if alpha_path.exists():
                alpha = np.array(Image.open(alpha_path).convert("L")).astype(np.float32) / 255.0

            m = compute_artifact_metrics(rgb, alpha)
            view_metrics.append(m)
            all_metrics.append(m)

        if view_metrics:
            per_view[f"view_{view_idx}"] = {
                "cas": float(np.mean([m.cas for m in view_metrics])),
                "fas": float(np.mean([m.fas for m in view_metrics])),
                "efs": float(np.mean([m.efs for m in view_metrics])),
                "oas": float(np.mean([m.oas for m in view_metrics])),
                "n_frames": len(view_metrics),
            }

    overall = {}
    if all_metrics:
        overall = {
            "cas": float(np.mean([m.cas for m in all_metrics])),
            "fas": float(np.mean([m.fas for m in all_metrics])),
            "efs": float(np.mean([m.efs for m in all_metrics])),
            "oas": float(np.mean([m.oas for m in all_metrics])),
            "n_frames": len(all_metrics),
        }

    return {"per_view": per_view, "overall": overall}


# ============================================================
# Unified Runner
# ============================================================

METRIC_GROUPS = {
    "fair": run_fair_comparison,
    "artifact": run_artifact_metrics,
}


def run_evaluation(
    name: str,
    render_dir: str,
    gt_dir: Optional[str] = None,
    views: List[int] = None,
    metrics: List[str] = None,
    max_frames: Optional[int] = None,
) -> Dict:
    """Run selected metric groups and aggregate results."""
    if views is None:
        views = [1, 2, 3, 4, 5]
    if metrics is None:
        metrics = ["fair", "artifact"] if gt_dir else ["artifact"]

    result = {
        "experiment": name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "render_dir": render_dir,
            "gt_dir": gt_dir,
            "views": views,
            "metrics_computed": metrics,
        },
        "metrics": {},
    }

    for metric_name in metrics:
        if metric_name not in METRIC_GROUPS:
            print(f"  Unknown metric group: {metric_name}, skipping")
            continue

        print(f"  Computing {metric_name}...")
        if metric_name == "fair":
            if not gt_dir:
                print(f"    SKIP fair (no gt_dir)")
                continue
            result["metrics"]["fair"] = run_fair_comparison(
                render_dir, gt_dir, views, max_frames
            )
        elif metric_name == "artifact":
            result["metrics"]["artifact"] = run_artifact_metrics(
                render_dir, views, max_frames or 50
            )

    return result


# ============================================================
# Comparison Table Generator
# ============================================================

def compare_experiments(result_files: List[str], output_path: Optional[str] = None):
    """Generate comparison table from multiple unified eval JSONs."""
    results = []
    for f in sorted(result_files):
        with open(f) as fp:
            results.append(json.load(fp))

    # Print table
    print("\n" + "=" * 100)
    print("UNIFIED EXPERIMENT COMPARISON (5-view, novel views only)")
    print("=" * 100)

    # Fair metrics
    has_fair = any("fair" in r.get("metrics", {}) for r in results)
    if has_fair:
        print(f"\n{'Experiment':<25} {'PSNR':>8} {'SSIM':>8} {'L1':>8} {'IoU':>8}")
        print("-" * 60)
        for r in results:
            name = r["experiment"]
            fair = r.get("metrics", {}).get("fair", {})
            if fair:
                psnr = fair["psnr_gt_masked"]["mean"]
                ssim = fair["ssim_gt_masked"]["mean"]
                l1 = fair["l1_gt_masked"]["mean"]
                iou = fair["iou"]["mean"]
                print(f"{name:<25} {psnr:8.2f} {ssim:8.4f} {l1:8.4f} {iou:8.3f}")

    # Artifact metrics
    has_art = any("artifact" in r.get("metrics", {}) for r in results)
    if has_art:
        print(f"\n{'Experiment':<25} {'CAS':>10} {'FAS%':>10} {'EFS%':>10} {'OAS%':>10}")
        print("-" * 70)
        for r in results:
            name = r["experiment"]
            art = r.get("metrics", {}).get("artifact", {}).get("overall", {})
            if art:
                print(f"{name:<25} {art['cas']:10.4f} {art['fas']*100:9.4f}% "
                      f"{art['efs']*100:9.4f}% {art['oas']*100:9.4f}%")

    # Save markdown
    if output_path:
        with open(output_path, "w") as f:
            f.write("# Experiment Comparison\n\n")
            f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"> Protocol: 5-view (novel views 1-5, input view 0 excluded)\n\n")

            if has_fair:
                f.write("## Image Quality (Fair Eval)\n\n")
                f.write("| Experiment | PSNR | SSIM | L1 | IoU |\n")
                f.write("|:-----------|:----:|:----:|:--:|:---:|\n")
                for r in results:
                    fair = r.get("metrics", {}).get("fair", {})
                    if fair:
                        name = r["experiment"]
                        f.write(f"| {name} | {fair['psnr_gt_masked']['mean']:.2f} "
                                f"| {fair['ssim_gt_masked']['mean']:.4f} "
                                f"| {fair['l1_gt_masked']['mean']:.4f} "
                                f"| {fair['iou']['mean']:.3f} |\n")

            if has_art:
                f.write("\n## Artifact Quality (lower = better)\n\n")
                f.write("| Experiment | CAS | FAS% | EFS% | OAS% |\n")
                f.write("|:-----------|:---:|:----:|:----:|:----:|\n")
                for r in results:
                    art = r.get("metrics", {}).get("artifact", {}).get("overall", {})
                    if art:
                        name = r["experiment"]
                        f.write(f"| {name} | {art['cas']:.4f} "
                                f"| {art['fas']*100:.4f}% "
                                f"| {art['efs']*100:.4f}% "
                                f"| {art['oas']*100:.4f}% |\n")

        print(f"\nSaved: {output_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified GS-LRM Evaluation Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Run command
    run_p = subparsers.add_parser("run", help="Evaluate a single experiment")
    run_p.add_argument("--name", required=True, help="Experiment name")
    run_p.add_argument("--render_dir", required=True, help="Render output directory")
    run_p.add_argument("--gt_dir", default=None, help="GT data directory")
    run_p.add_argument("--views", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                       help="View indices to evaluate (default: 1-5, excluding input view 0)")
    run_p.add_argument("--metrics", nargs="+", default=None,
                       choices=["fair", "artifact"],
                       help="Metric groups (default: all available)")
    run_p.add_argument("--max_frames", type=int, default=None,
                       help="Max frames for fair eval (default: all)")
    run_p.add_argument("--output", required=True, help="Output JSON path")

    # Compare command
    cmp_p = subparsers.add_parser("compare", help="Compare experiments")
    cmp_p.add_argument("--results", nargs="+", required=True,
                       help="Unified eval JSON files (supports glob)")
    cmp_p.add_argument("--output", default=None, help="Output markdown path")

    args = parser.parse_args()

    if args.command == "run":
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        print(f"Evaluating: {args.name}")
        result = run_evaluation(
            name=args.name,
            render_dir=args.render_dir,
            gt_dir=args.gt_dir,
            views=args.views,
            metrics=args.metrics,
            max_frames=args.max_frames,
        )
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {args.output}")

        # Print summary
        fair = result.get("metrics", {}).get("fair", {})
        if fair:
            print(f"  PSNR={fair['psnr_gt_masked']['mean']:.2f} "
                  f"SSIM={fair['ssim_gt_masked']['mean']:.4f} "
                  f"IoU={fair['iou']['mean']:.3f}")
        art = result.get("metrics", {}).get("artifact", {}).get("overall", {})
        if art:
            print(f"  CAS={art['cas']:.4f} FAS={art['fas']*100:.4f}%")

    elif args.command == "compare":
        # Expand globs
        files = []
        for pattern in args.results:
            files.extend(glob.glob(pattern))
        if not files:
            print("No result files found")
            return
        compare_experiments(files, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
