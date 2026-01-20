#!/usr/bin/env python3
"""
FaceLift Mouse Camera Diagnostics

This script analyzes the FaceLift mouse dataset to identify camera parameter
issues that cause ghosting artifacts.

Key findings from v13 dataset analysis:
- Dataset uses v5 preprocessing which has a KNOWN BUG
- cx, cy are set to 256 (image center) but actual principal points differ
- The _transform.scaled_cx/scaled_cy show the REAL principal points
- Ray direction errors of 10-20 degrees are causing ghosting

Usage:
    python diagnose_facelift_cameras.py --dataset_path /path/to/data_mouse_train.txt

Author: AI Research Assistant
Date: 2026-01-17
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class ViewDiagnostics:
    """Diagnostics for a single view"""
    view_id: int
    reported_cx: float  # cx in JSON (usually 256)
    reported_cy: float  # cy in JSON (usually 256)
    actual_cx: float    # _transform.scaled_cx (actual PP)
    actual_cy: float    # _transform.scaled_cy (actual PP)
    offset_x: float     # actual - reported
    offset_y: float
    offset_magnitude: float
    fx: float
    fy: float
    camera_distance: float
    ray_angle_error_deg: float


@dataclass
class SampleDiagnostics:
    """Diagnostics for a complete sample"""
    sample_path: str
    preprocessing_version: str
    views: List[ViewDiagnostics]
    max_pp_offset: float
    mean_pp_offset: float
    max_ray_error: float
    mean_ray_error: float
    distance_std: float
    has_critical_issue: bool
    warnings: List[str]


def compute_camera_distance(w2c: np.ndarray) -> float:
    """
    Compute camera distance from world origin.

    w2c is world-to-camera matrix.
    Camera position in world = -R^T @ t where R = w2c[:3,:3], t = w2c[:3,3]
    """
    R = np.array(w2c[:3, :3])
    t = np.array(w2c[:3, 3])
    cam_pos = -R.T @ t
    return float(np.linalg.norm(cam_pos))


def analyze_sample(sample_path: str) -> Optional[SampleDiagnostics]:
    """Analyze a single sample's camera parameters"""

    camera_json_path = os.path.join(sample_path, "opencv_cameras.json")

    if not os.path.exists(camera_json_path):
        return None

    with open(camera_json_path, 'r') as f:
        data = json.load(f)

    # Get preprocessing info
    preprocessing = data.get("_preprocessing", {})
    version = preprocessing.get("version", "unknown")

    views = []
    warnings_list = []

    for frame in data.get("frames", []):
        view_id = frame.get("view_id", len(views))

        # Reported (used by model) values
        reported_cx = frame.get("cx", 256.0)
        reported_cy = frame.get("cy", 256.0)
        fx = frame.get("fx", 549.0)
        fy = frame.get("fy", 549.0)

        # Actual principal point (from transform info)
        transform = frame.get("_transform", {})

        # Check if PP correction was applied (D1 preprocessing)
        pp_correction_method = transform.get("pp_correction_method", None)
        if pp_correction_method == "crop":
            # D1: Image was cropped so actual PP is now at image center
            # The cx/cy in JSON (256) is now mathematically correct
            actual_cx = reported_cx  # PP is now correct
            actual_cy = reported_cy
            # Update version detection
            if version == "unknown":
                version = "D1_pp_centered"
        else:
            # Legacy (v12/v13): actual PP stored in scaled_cx/scaled_cy
            actual_cx = transform.get("scaled_cx", reported_cx)
            actual_cy = transform.get("scaled_cy", reported_cy)

        # Calculate offset
        offset_x = actual_cx - reported_cx
        offset_y = actual_cy - reported_cy
        offset_magnitude = np.sqrt(offset_x**2 + offset_y**2)

        # Calculate ray angle error
        # tan(angle) = offset / focal_length
        ray_angle_error = np.degrees(np.arctan(offset_magnitude / fx)) if fx > 0 else 0

        # Calculate camera distance
        w2c = np.array(frame.get("w2c", np.eye(4)))
        camera_distance = compute_camera_distance(w2c)

        views.append(ViewDiagnostics(
            view_id=view_id,
            reported_cx=reported_cx,
            reported_cy=reported_cy,
            actual_cx=actual_cx,
            actual_cy=actual_cy,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_magnitude=offset_magnitude,
            fx=fx,
            fy=fy,
            camera_distance=camera_distance,
            ray_angle_error_deg=ray_angle_error
        ))

    if not views:
        return None

    # Aggregate statistics
    offsets = [v.offset_magnitude for v in views]
    ray_errors = [v.ray_angle_error_deg for v in views]
    distances = [v.camera_distance for v in views]

    max_pp_offset = max(offsets)
    mean_pp_offset = np.mean(offsets)
    max_ray_error = max(ray_errors)
    mean_ray_error = np.mean(ray_errors)
    distance_std = np.std(distances)

    # Determine if there's a critical issue
    has_critical_issue = max_pp_offset > 50 or max_ray_error > 5

    # Generate warnings
    if "v5" in version.lower() or "deprecated" in version.lower():
        warnings_list.append(f"Uses deprecated preprocessing: {version}")

    if max_pp_offset > 50:
        warnings_list.append(f"CRITICAL: Principal point offset up to {max_pp_offset:.1f}px")
    elif max_pp_offset > 20:
        warnings_list.append(f"High principal point offset: {max_pp_offset:.1f}px")

    if max_ray_error > 10:
        warnings_list.append(f"CRITICAL: Ray angle error up to {max_ray_error:.1f}°")
    elif max_ray_error > 5:
        warnings_list.append(f"Significant ray angle error: {max_ray_error:.1f}°")

    return SampleDiagnostics(
        sample_path=sample_path,
        preprocessing_version=version,
        views=views,
        max_pp_offset=max_pp_offset,
        mean_pp_offset=mean_pp_offset,
        max_ray_error=max_ray_error,
        mean_ray_error=mean_ray_error,
        distance_std=distance_std,
        has_critical_issue=has_critical_issue,
        warnings=warnings_list
    )


def analyze_dataset(
    dataset_path: str,
    max_samples: int = 100,
    verbose: bool = True
) -> Dict:
    """Analyze entire dataset"""

    # Load sample paths
    with open(dataset_path, 'r') as f:
        sample_paths = [line.strip() for line in f if line.strip()]

    if verbose:
        print(f"Analyzing {min(len(sample_paths), max_samples)} samples from {dataset_path}")

    results = []
    preprocessing_versions = defaultdict(int)

    for i, sample_path in enumerate(sample_paths[:max_samples]):
        if verbose and i % 50 == 0:
            print(f"  Processing {i+1}/{min(len(sample_paths), max_samples)}...")

        diag = analyze_sample(sample_path)
        if diag:
            results.append(diag)
            preprocessing_versions[diag.preprocessing_version] += 1

    if not results:
        return {"error": "No valid samples found"}

    # Aggregate statistics
    all_pp_offsets = [r.max_pp_offset for r in results]
    all_ray_errors = [r.max_ray_error for r in results]
    critical_count = sum(1 for r in results if r.has_critical_issue)

    return {
        "total_samples_analyzed": len(results),
        "preprocessing_versions": dict(preprocessing_versions),
        "principal_point_analysis": {
            "mean_max_offset_px": np.mean(all_pp_offsets),
            "overall_max_offset_px": np.max(all_pp_offsets),
            "std_max_offset_px": np.std(all_pp_offsets),
            "samples_with_offset_gt_20px": sum(1 for x in all_pp_offsets if x > 20),
            "samples_with_offset_gt_50px": sum(1 for x in all_pp_offsets if x > 50),
        },
        "ray_angle_error_analysis": {
            "mean_max_error_deg": np.mean(all_ray_errors),
            "overall_max_error_deg": np.max(all_ray_errors),
            "std_max_error_deg": np.std(all_ray_errors),
            "samples_with_error_gt_5deg": sum(1 for x in all_ray_errors if x > 5),
            "samples_with_error_gt_10deg": sum(1 for x in all_ray_errors if x > 10),
        },
        "critical_issues": {
            "count": critical_count,
            "percentage": critical_count / len(results) * 100,
        },
        "sample_details": [
            {
                "path": r.sample_path,
                "max_pp_offset": r.max_pp_offset,
                "max_ray_error": r.max_ray_error,
                "warnings": r.warnings
            }
            for r in results[:10]  # First 10 samples
        ]
    }


def print_report(analysis: Dict):
    """Print formatted diagnostic report"""

    print("\n" + "="*80)
    print("FACELIFT MOUSE CAMERA DIAGNOSTIC REPORT")
    print("="*80)

    print(f"\nSamples analyzed: {analysis['total_samples_analyzed']}")

    print("\n[Preprocessing Versions]")
    for version, count in analysis['preprocessing_versions'].items():
        print(f"  - {version}: {count} samples")

    pp = analysis['principal_point_analysis']
    print("\n[Principal Point Analysis]")
    print(f"  Mean max offset:     {pp['mean_max_offset_px']:.1f} px")
    print(f"  Overall max offset:  {pp['overall_max_offset_px']:.1f} px")
    print(f"  Samples > 20px:      {pp['samples_with_offset_gt_20px']}")
    print(f"  Samples > 50px:      {pp['samples_with_offset_gt_50px']}")

    ray = analysis['ray_angle_error_analysis']
    print("\n[Ray Angle Error Analysis]")
    print(f"  Mean max error:      {ray['mean_max_error_deg']:.2f}°")
    print(f"  Overall max error:   {ray['overall_max_error_deg']:.2f}°")
    print(f"  Samples > 5°:        {ray['samples_with_error_gt_5deg']}")
    print(f"  Samples > 10°:       {ray['samples_with_error_gt_10deg']}")

    crit = analysis['critical_issues']
    print("\n[Critical Issues]")
    if crit['count'] > 0:
        print(f"  ⚠️  {crit['count']} samples ({crit['percentage']:.1f}%) have critical issues!")
    else:
        print(f"  ✅ No critical issues detected")

    print("\n[Sample Details (first 10)]")
    for sample in analysis['sample_details'][:5]:
        print(f"  {os.path.basename(sample['path'])}:")
        print(f"    PP offset: {sample['max_pp_offset']:.1f}px, Ray error: {sample['max_ray_error']:.2f}°")
        for warn in sample['warnings']:
            print(f"    ⚠️  {warn}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if pp['samples_with_offset_gt_50px'] > 0:
        print("""
⚠️  CRITICAL: Principal point misalignment detected!

ROOT CAUSE:
  The dataset uses v5 preprocessing which has a known bug where cx,cy are
  set to 256 (image center) even though the actual principal points differ.

  This causes rays to be back-projected in wrong directions, leading to
  3D reconstruction errors (ghosting artifacts).

RECOMMENDED SOLUTIONS (in order of preference):

1. RE-PREPROCESS WITH v11 (BEST)
   - v11 uses Principal Point centered approach
   - Results in mathematically correct cx=cy=256
   - No runtime overhead

   Command:
   python preprocess_mouse_data.py --method v11 --input_dir /path/to/raw --output_dir /path/to/v14

2. APPLY RUNTIME CORRECTION (WORKAROUND)
   - Use 'crop_to_pp' correction during data loading
   - Reads _transform.scaled_cx/scaled_cy from JSON
   - Crops images so actual PP becomes centered

   Integration:
   from solutions.principal_point_correction import PrincipalPointCorrector
   corrector = PrincipalPointCorrector(method="crop_to_pp")

3. TRAIN WITH CORRECTED INTRINSICS (HACK)
   - Feed actual cx,cy instead of 256
   - May require architecture changes for varying cx,cy
   - Not recommended for GS-LRM which assumes fixed intrinsics
""")

    print("\n" + "="*80)


def visualize_analysis(analysis: Dict, output_dir: str):
    """Generate visualization plots"""

    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get per-view data from first sample with full details
    # For proper visualization, we'd need to reload sample data

    pp = analysis['principal_point_analysis']
    ray = analysis['ray_angle_error_analysis']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Principal point offset histogram (simulated from summary stats)
    ax1 = axes[0]
    ax1.bar(['Mean', 'Max'], [pp['mean_max_offset_px'], pp['overall_max_offset_px']])
    ax1.axhline(20, color='orange', linestyle='--', label='Warning (20px)')
    ax1.axhline(50, color='r', linestyle='--', label='Critical (50px)')
    ax1.set_ylabel('Principal Point Offset (pixels)')
    ax1.set_title('Principal Point Misalignment')
    ax1.legend()

    # Ray angle error
    ax2 = axes[1]
    ax2.bar(['Mean', 'Max'], [ray['mean_max_error_deg'], ray['overall_max_error_deg']])
    ax2.axhline(5, color='orange', linestyle='--', label='Warning (5°)')
    ax2.axhline(10, color='r', linestyle='--', label='Critical (10°)')
    ax2.set_ylabel('Ray Angle Error (degrees)')
    ax2.set_title('Ray Direction Error from PP Misalignment')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'facelift_camera_diagnostics.png'), dpi=150)
    plt.close()

    print(f"Visualization saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose FaceLift mouse dataset camera parameters'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to dataset file (data_mouse_train.txt)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./facelift_diagnostics',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--max_samples', type=int, default=100,
        help='Maximum samples to analyze'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--json_output', type=str, default=None,
        help='Path for JSON output'
    )

    args = parser.parse_args()

    # Run analysis
    analysis = analyze_dataset(
        args.dataset_path,
        max_samples=args.max_samples,
        verbose=True
    )

    # Print report
    print_report(analysis)

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = args.json_output or os.path.join(args.output_dir, 'camera_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nJSON report saved to: {json_path}")

    # Visualize
    if args.visualize:
        visualize_analysis(analysis, args.output_dir)


if __name__ == '__main__':
    main()
