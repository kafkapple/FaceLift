#!/usr/bin/env python3
"""
Batch Visualization Script for FaceLift Experiments.

Automatically finds best checkpoints and generates visualizations:
- Rerun 3D Gaussian visualization (.rrd files)
- Summary images
- Turntable videos

Usage:
    # Visualize all D7 experiments
    python batch_visualize.py --exp-pattern "D7_*" --output-dir viz_output

    # Visualize specific experiment's best iteration
    python batch_visualize.py --exp-dir experiments/validation/D7_E4_5v_alpha

    # Only generate rerun files
    python batch_visualize.py --exp-pattern "D7_*" --rerun-only

    # Specify number of samples per experiment
    python batch_visualize.py --exp-pattern "D7_*" --num-samples 5

Created: 2026-01-19
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv


def find_best_iteration(exp_dir: Path) -> Optional[Path]:
    """
    Find the best iteration based on validation metrics.

    Looks for summary.csv or finds the latest iteration with highest PSNR.
    """
    summary_file = exp_dir / "summary.csv"

    if summary_file.exists():
        # Parse summary.csv to find best iteration
        best_psnr = -1
        best_iter = None

        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                psnr = float(row.get('psnr', row.get('avg_psnr', 0)))
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_iter = row.get('iteration', row.get('iter', None))

        if best_iter:
            iter_dir = exp_dir / f"iter_{int(best_iter):08d}"
            if iter_dir.exists():
                return iter_dir

    # Fallback: find latest iteration directory
    iter_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")])

    if iter_dirs:
        # Find best by checking metrics.txt in each
        best_psnr = -1
        best_dir = iter_dirs[-1]  # Default to latest

        for iter_dir in iter_dirs:
            # Check any sample's metrics
            sample_dirs = [d for d in iter_dir.iterdir() if d.is_dir()]
            if sample_dirs:
                metrics_file = sample_dirs[0] / "metrics.txt"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        for line in f:
                            if 'psnr' in line.lower():
                                try:
                                    psnr = float(line.split(':')[1].strip())
                                    if psnr > best_psnr:
                                        best_psnr = psnr
                                        best_dir = iter_dir
                                except:
                                    pass

        return best_dir

    return None


def find_gaussian_files(iter_dir: Path, num_samples: int = 5) -> List[Path]:
    """Find Gaussian PLY files in iteration directory."""
    ply_files = []

    sample_dirs = sorted([d for d in iter_dir.iterdir() if d.is_dir()])

    for sample_dir in sample_dirs[:num_samples]:
        ply_file = sample_dir / "gaussians.ply"
        if ply_file.exists():
            ply_files.append(ply_file)

    return ply_files


def generate_rerun_visualization(
    ply_files: List[Path],
    output_path: Path,
    animation: bool = False
) -> bool:
    """Generate Rerun visualization from PLY files."""
    script_dir = Path(__file__).parent
    rerun_script = script_dir / "visualize_gaussian_rerun.py"

    if not rerun_script.exists():
        print(f"Warning: Rerun script not found at {rerun_script}")
        return False

    try:
        cmd = ["python", str(rerun_script)]
        cmd.extend([str(f) for f in ply_files])
        cmd.extend(["--save", "--output", str(output_path)])

        if animation and len(ply_files) > 1:
            cmd.append("--sequence")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  Generated: {output_path}")
            return True
        else:
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error generating rerun: {e}")
        return False


def copy_visualization_assets(
    iter_dir: Path,
    output_dir: Path,
    num_samples: int = 5
) -> Dict[str, List[Path]]:
    """Copy visualization assets to output directory."""
    copied = {
        "images": [],
        "videos": [],
        "metrics": []
    }

    sample_dirs = sorted([d for d in iter_dir.iterdir() if d.is_dir()])

    for sample_dir in sample_dirs[:num_samples]:
        sample_id = sample_dir.name
        sample_output = output_dir / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img_name in ["gt_vs_pred.png", "input.png"]:
            src = sample_dir / img_name
            if src.exists():
                dst = sample_output / img_name
                shutil.copy(src, dst)
                copied["images"].append(dst)

        # Copy videos
        for vid_name in ["turntable.mp4", "turntable_with_input.mp4"]:
            src = sample_dir / vid_name
            if src.exists():
                dst = sample_output / vid_name
                shutil.copy(src, dst)
                copied["videos"].append(dst)

        # Copy metrics
        for metrics_name in ["metrics.txt", "perview_metrics.txt"]:
            src = sample_dir / metrics_name
            if src.exists():
                dst = sample_output / metrics_name
                shutil.copy(src, dst)
                copied["metrics"].append(dst)

    return copied


def process_experiment(
    exp_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
    rerun_only: bool = False,
    skip_existing: bool = True
) -> Dict:
    """Process single experiment directory."""
    exp_name = exp_dir.name
    exp_output = output_dir / exp_name

    print(f"\n{'='*60}")
    print(f"Processing: {exp_name}")
    print(f"{'='*60}")

    result = {
        "experiment": exp_name,
        "best_iteration": None,
        "num_samples": 0,
        "rerun_generated": False,
        "assets_copied": {}
    }

    # Find best iteration
    best_iter = find_best_iteration(exp_dir)
    if not best_iter:
        print(f"  No iterations found in {exp_dir}")
        return result

    result["best_iteration"] = best_iter.name
    print(f"  Best iteration: {best_iter.name}")

    # Find Gaussian files
    ply_files = find_gaussian_files(best_iter, num_samples)
    result["num_samples"] = len(ply_files)
    print(f"  Found {len(ply_files)} Gaussian files")

    if not ply_files:
        print(f"  No Gaussian files found")
        return result

    # Create output directory
    exp_output.mkdir(parents=True, exist_ok=True)

    # Generate Rerun visualization
    rerun_output = exp_output / f"{exp_name}_gaussians.rrd"
    if skip_existing and rerun_output.exists():
        print(f"  Rerun file exists, skipping: {rerun_output}")
        result["rerun_generated"] = True
    else:
        result["rerun_generated"] = generate_rerun_visualization(
            ply_files, rerun_output, animation=(len(ply_files) > 1)
        )

    # Copy other assets
    if not rerun_only:
        result["assets_copied"] = copy_visualization_assets(
            best_iter, exp_output, num_samples
        )
        print(f"  Copied: {len(result['assets_copied']['images'])} images, "
              f"{len(result['assets_copied']['videos'])} videos")

    # Save experiment info
    info_file = exp_output / "experiment_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            "experiment": exp_name,
            "source_dir": str(exp_dir),
            "best_iteration": result["best_iteration"],
            "num_samples": result["num_samples"]
        }, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch visualization for FaceLift experiments")
    parser.add_argument("--exp-dir", type=str, help="Single experiment directory")
    parser.add_argument("--exp-pattern", type=str, default="D7_*",
                        help="Experiment name pattern (glob)")
    parser.add_argument("--val-dir", type=str, default="experiments/validation",
                        help="Validation directory")
    parser.add_argument("--output-dir", type=str, default="viz_output",
                        help="Output directory")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples per experiment")
    parser.add_argument("--rerun-only", action="store_true",
                        help="Only generate Rerun files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output already exists")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find experiment directories
    if args.exp_dir:
        exp_dirs = [Path(args.exp_dir)]
    else:
        val_dir = Path(args.val_dir)
        if not val_dir.exists():
            print(f"Error: Validation directory not found: {val_dir}")
            return

        from glob import glob
        exp_dirs = sorted([Path(d) for d in glob(str(val_dir / args.exp_pattern)) if Path(d).is_dir()])

    if not exp_dirs:
        print(f"No experiment directories found matching pattern: {args.exp_pattern}")
        return

    print(f"Found {len(exp_dirs)} experiments to process")

    # Process each experiment
    results = []
    for exp_dir in exp_dirs:
        result = process_experiment(
            exp_dir,
            output_dir,
            num_samples=args.num_samples,
            rerun_only=args.rerun_only,
            skip_existing=args.skip_existing
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(results)}")
    print(f"Rerun files generated: {sum(1 for r in results if r['rerun_generated'])}")
    print(f"Output directory: {output_dir}")

    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
