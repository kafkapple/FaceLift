#!/usr/bin/env python3
"""
Camera Validation Diagnostic Script for FaceLift Mouse GS-LRM

This script validates camera parameters across multi-view datasets to identify
potential causes of ghosting artifacts.

Checks performed:
1. Principal point offset from image center
2. Scale consistency (fx/dist ratio across views)
3. Camera distance distribution
4. Focal length consistency
5. Epipolar constraint validation (optional)

Usage:
    python validate_cameras.py --dataset_path /path/to/data.txt --output_dir ./diagnostics_output

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
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

# Try to import optional visualization libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Visualization will be skipped.")


@dataclass
class CameraParams:
    """Camera parameters for a single view"""
    view_id: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: np.ndarray  # 4x4 camera-to-world matrix
    image_size: Tuple[int, int]  # (H, W)

    @property
    def camera_position(self) -> np.ndarray:
        """Extract camera position from c2w matrix"""
        return self.c2w[:3, 3]

    @property
    def camera_direction(self) -> np.ndarray:
        """Extract camera viewing direction (negative z-axis)"""
        return -self.c2w[:3, 2]

    @property
    def distance_from_origin(self) -> float:
        """Distance from camera to world origin"""
        return float(np.linalg.norm(self.camera_position))

    @property
    def principal_point_offset(self) -> Tuple[float, float]:
        """Offset of principal point from image center"""
        H, W = self.image_size
        offset_x = self.cx - W / 2
        offset_y = self.cy - H / 2
        return (offset_x, offset_y)

    @property
    def principal_point_offset_norm(self) -> float:
        """Normalized offset magnitude (pixels)"""
        ox, oy = self.principal_point_offset
        return np.sqrt(ox**2 + oy**2)


@dataclass
class DiagnosticResult:
    """Results from camera validation"""
    sample_id: str
    num_views: int

    # Principal point analysis
    pp_offsets: List[float]  # offset from center per view
    pp_offset_mean: float
    pp_offset_std: float
    pp_offset_max: float
    pp_warning: bool  # True if offset > threshold

    # Scale consistency
    fx_values: List[float]
    fy_values: List[float]
    distances: List[float]
    fx_dist_ratios: List[float]
    scale_ratio_std: float
    scale_warning: bool

    # Distance distribution
    distance_mean: float
    distance_std: float
    distance_min: float
    distance_max: float
    distance_range_ratio: float  # max/min

    # Focal length consistency
    fx_std: float
    fy_std: float
    aspect_ratios: List[float]  # fy/fx per view
    aspect_ratio_std: float

    # Overall assessment
    overall_score: float  # 0-100
    warnings: List[str]
    recommendations: List[str]


class CameraValidator:
    """Validates camera parameters for multi-view consistency"""

    # Thresholds for warnings
    PP_OFFSET_THRESHOLD = 10.0  # pixels
    SCALE_RATIO_STD_THRESHOLD = 0.05  # 5%
    DISTANCE_RANGE_THRESHOLD = 2.0  # max/min ratio
    ASPECT_RATIO_STD_THRESHOLD = 0.01  # 1%

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []

    def validate_sample(
        self,
        sample_id: str,
        cameras: List[CameraParams]
    ) -> DiagnosticResult:
        """Validate camera parameters for a single sample"""

        num_views = len(cameras)
        warnings_list = []
        recommendations = []

        # 1. Principal Point Analysis
        pp_offsets = [cam.principal_point_offset_norm for cam in cameras]
        pp_offset_mean = np.mean(pp_offsets)
        pp_offset_std = np.std(pp_offsets)
        pp_offset_max = np.max(pp_offsets)
        pp_warning = pp_offset_max > self.PP_OFFSET_THRESHOLD

        if pp_warning:
            warnings_list.append(
                f"Principal point offset exceeds {self.PP_OFFSET_THRESHOLD}px "
                f"(max: {pp_offset_max:.1f}px)"
            )
            recommendations.append(
                "Consider re-centering images based on principal point, or "
                "verify camera calibration accuracy."
            )

        # 2. Scale Consistency (fx/distance ratio)
        fx_values = [cam.fx for cam in cameras]
        fy_values = [cam.fy for cam in cameras]
        distances = [cam.distance_from_origin for cam in cameras]

        # Avoid division by zero
        fx_dist_ratios = [
            fx / dist if dist > 0 else float('inf')
            for fx, dist in zip(fx_values, distances)
        ]
        scale_ratio_std = np.std(fx_dist_ratios) / np.mean(fx_dist_ratios) if np.mean(fx_dist_ratios) > 0 else 0
        scale_warning = scale_ratio_std > self.SCALE_RATIO_STD_THRESHOLD

        if scale_warning:
            warnings_list.append(
                f"Scale ratio (fx/dist) varies significantly across views "
                f"(relative std: {scale_ratio_std*100:.1f}%)"
            )
            recommendations.append(
                "Apply per-sample camera normalization to ensure consistent "
                "projected object size across views."
            )

        # 3. Distance Distribution
        distance_mean = np.mean(distances)
        distance_std = np.std(distances)
        distance_min = np.min(distances)
        distance_max = np.max(distances)
        distance_range_ratio = distance_max / distance_min if distance_min > 0 else float('inf')

        if distance_range_ratio > self.DISTANCE_RANGE_THRESHOLD:
            warnings_list.append(
                f"Large distance variation between views "
                f"(range ratio: {distance_range_ratio:.2f}x)"
            )
            recommendations.append(
                "Consider distance normalization in preprocessing to reduce "
                "scale ambiguity."
            )

        # 4. Focal Length & Aspect Ratio Consistency
        fx_std = np.std(fx_values)
        fy_std = np.std(fy_values)
        aspect_ratios = [fy / fx if fx > 0 else 1.0 for fx, fy in zip(fx_values, fy_values)]
        aspect_ratio_std = np.std(aspect_ratios)

        if aspect_ratio_std > self.ASPECT_RATIO_STD_THRESHOLD:
            warnings_list.append(
                f"Aspect ratio (fy/fx) varies across views "
                f"(std: {aspect_ratio_std:.4f})"
            )
            recommendations.append(
                "Standardize focal lengths across views to fy = fx for consistency."
            )

        # 5. Calculate Overall Score
        # Score based on how many checks pass
        score = 100.0
        if pp_warning:
            score -= 25
        if scale_warning:
            score -= 30
        if distance_range_ratio > self.DISTANCE_RANGE_THRESHOLD:
            score -= 20
        if aspect_ratio_std > self.ASPECT_RATIO_STD_THRESHOLD:
            score -= 10
        score = max(0, score)

        result = DiagnosticResult(
            sample_id=sample_id,
            num_views=num_views,
            pp_offsets=pp_offsets,
            pp_offset_mean=pp_offset_mean,
            pp_offset_std=pp_offset_std,
            pp_offset_max=pp_offset_max,
            pp_warning=pp_warning,
            fx_values=fx_values,
            fy_values=fy_values,
            distances=distances,
            fx_dist_ratios=fx_dist_ratios,
            scale_ratio_std=scale_ratio_std,
            scale_warning=scale_warning,
            distance_mean=distance_mean,
            distance_std=distance_std,
            distance_min=distance_min,
            distance_max=distance_max,
            distance_range_ratio=distance_range_ratio,
            fx_std=fx_std,
            fy_std=fy_std,
            aspect_ratios=aspect_ratios,
            aspect_ratio_std=aspect_ratio_std,
            overall_score=score,
            warnings=warnings_list,
            recommendations=recommendations
        )

        self.results.append(result)
        return result

    def compute_epipolar_error(
        self,
        cam1: CameraParams,
        cam2: CameraParams,
        correspondences: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute epipolar constraint error between two views.

        If correspondences not provided, uses image corners as test points.

        Args:
            cam1, cam2: Camera parameters
            correspondences: (N, 4) array of [x1, y1, x2, y2] point pairs

        Returns:
            Mean epipolar distance error in pixels
        """
        # Compute fundamental matrix
        K1 = np.array([
            [cam1.fx, 0, cam1.cx],
            [0, cam1.fy, cam1.cy],
            [0, 0, 1]
        ])
        K2 = np.array([
            [cam2.fx, 0, cam2.cx],
            [0, cam2.fy, cam2.cy],
            [0, 0, 1]
        ])

        # Relative pose
        R1 = cam1.c2w[:3, :3]
        t1 = cam1.c2w[:3, 3]
        R2 = cam2.c2w[:3, :3]
        t2 = cam2.c2w[:3, 3]

        # Relative rotation and translation
        R = R2.T @ R1
        t = R2.T @ (t1 - t2)

        # Skew-symmetric matrix of t
        tx = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0]
        ])

        # Essential matrix
        E = tx @ R

        # Fundamental matrix
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

        # If no correspondences provided, use corners
        if correspondences is None:
            H1, W1 = cam1.image_size
            # Create test points at corners and center
            pts1 = np.array([
                [0, 0],
                [W1, 0],
                [0, H1],
                [W1, H1],
                [W1/2, H1/2]
            ])
            # For testing, assume points project to similar locations
            pts2 = pts1.copy()
            correspondences = np.hstack([pts1, pts2])

        # Compute epipolar distances
        errors = []
        for corr in correspondences:
            x1 = np.array([corr[0], corr[1], 1])
            x2 = np.array([corr[2], corr[3], 1])

            # Epipolar line in image 2
            l2 = F @ x1

            # Distance from x2 to line l2
            dist = abs(x2 @ l2) / np.sqrt(l2[0]**2 + l2[1]**2)
            errors.append(dist)

        return np.mean(errors)

    def generate_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive diagnostic report"""

        if not self.results:
            return {"error": "No validation results available"}

        # Aggregate statistics
        all_pp_offsets = [r.pp_offset_max for r in self.results]
        all_scale_stds = [r.scale_ratio_std for r in self.results]
        all_scores = [r.overall_score for r in self.results]

        report = {
            "summary": {
                "total_samples": len(self.results),
                "mean_score": np.mean(all_scores),
                "min_score": np.min(all_scores),
                "samples_with_warnings": sum(1 for r in self.results if r.warnings),
                "pp_warning_count": sum(1 for r in self.results if r.pp_warning),
                "scale_warning_count": sum(1 for r in self.results if r.scale_warning),
            },
            "principal_point": {
                "mean_offset": np.mean(all_pp_offsets),
                "max_offset": np.max(all_pp_offsets),
                "std_offset": np.std(all_pp_offsets),
                "threshold": self.PP_OFFSET_THRESHOLD,
            },
            "scale_consistency": {
                "mean_ratio_std": np.mean(all_scale_stds),
                "max_ratio_std": np.max(all_scale_stds),
                "threshold": self.SCALE_RATIO_STD_THRESHOLD,
            },
            "recommendations": self._aggregate_recommendations(),
            "detailed_results": [self._result_to_dict(r) for r in self.results[:10]],  # First 10 samples
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            if self.verbose:
                print(f"Report saved to: {output_path}")

        return report

    def _result_to_dict(self, result: DiagnosticResult) -> Dict:
        """Convert result to JSON-serializable dict"""
        d = asdict(result)
        # Convert numpy arrays to lists
        for key in ['pp_offsets', 'fx_values', 'fy_values', 'distances',
                    'fx_dist_ratios', 'aspect_ratios']:
            if key in d:
                d[key] = [float(v) for v in d[key]]
        return d

    def _aggregate_recommendations(self) -> List[str]:
        """Aggregate unique recommendations from all results"""
        all_recs = set()
        for r in self.results:
            all_recs.update(r.recommendations)
        return list(all_recs)

    def visualize_diagnostics(self, output_dir: str):
        """Generate diagnostic visualizations"""
        if not HAS_MATPLOTLIB:
            print("Skipping visualization (matplotlib not available)")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Principal Point Offset Distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        all_pp_offsets = [r.pp_offset_max for r in self.results]
        axes[0, 0].hist(all_pp_offsets, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.PP_OFFSET_THRESHOLD, color='r', linestyle='--',
                          label=f'Threshold ({self.PP_OFFSET_THRESHOLD}px)')
        axes[0, 0].set_xlabel('Max PP Offset (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Principal Point Offset Distribution')
        axes[0, 0].legend()

        # 2. Scale Ratio Std Distribution
        all_scale_stds = [r.scale_ratio_std * 100 for r in self.results]
        axes[0, 1].hist(all_scale_stds, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(self.SCALE_RATIO_STD_THRESHOLD * 100, color='r', linestyle='--',
                          label=f'Threshold ({self.SCALE_RATIO_STD_THRESHOLD*100}%)')
        axes[0, 1].set_xlabel('Scale Ratio Std (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Scale Consistency Distribution')
        axes[0, 1].legend()

        # 3. Distance Range Distribution
        all_dist_ranges = [r.distance_range_ratio for r in self.results]
        axes[1, 0].hist(all_dist_ranges, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(self.DISTANCE_RANGE_THRESHOLD, color='r', linestyle='--',
                          label=f'Threshold ({self.DISTANCE_RANGE_THRESHOLD}x)')
        axes[1, 0].set_xlabel('Distance Range Ratio (max/min)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Camera Distance Variation')
        axes[1, 0].legend()

        # 4. Overall Score Distribution
        all_scores = [r.overall_score for r in self.results]
        axes[1, 1].hist(all_scores, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(70, color='orange', linestyle='--', label='Warning (70)')
        axes[1, 1].axvline(50, color='r', linestyle='--', label='Critical (50)')
        axes[1, 1].set_xlabel('Overall Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Validation Score Distribution')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'camera_diagnostics_overview.png'), dpi=150)
        plt.close()

        # 5. Per-view analysis for first sample
        if self.results:
            r = self.results[0]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            views = list(range(len(r.fx_values)))

            axes[0].bar(views, r.pp_offsets)
            axes[0].axhline(self.PP_OFFSET_THRESHOLD, color='r', linestyle='--')
            axes[0].set_xlabel('View')
            axes[0].set_ylabel('PP Offset (px)')
            axes[0].set_title(f'Principal Point Offset - Sample: {r.sample_id}')

            axes[1].bar(views, r.distances)
            axes[1].set_xlabel('View')
            axes[1].set_ylabel('Distance')
            axes[1].set_title('Camera Distance per View')

            axes[2].bar(views, r.fx_dist_ratios)
            axes[2].axhline(np.mean(r.fx_dist_ratios), color='g', linestyle='--',
                          label=f'Mean: {np.mean(r.fx_dist_ratios):.1f}')
            axes[2].set_xlabel('View')
            axes[2].set_ylabel('fx/dist ratio')
            axes[2].set_title('Scale Ratio per View')
            axes[2].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'per_view_analysis_sample0.png'), dpi=150)
            plt.close()

        if self.verbose:
            print(f"Visualizations saved to: {output_dir}")


def parse_dataset_file(dataset_path: str) -> List[Dict]:
    """
    Parse dataset file to extract camera parameters.

    Expected format (one line per sample):
    - JSON format with camera parameters, or
    - Path to NPZ files containing camera data

    This is a template - adjust based on actual data format.
    """
    samples = []

    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            # Try JSON format first
            if line.startswith('{'):
                data = json.loads(line)
                samples.append(data)
            else:
                # Assume it's a path to data file (NPZ or similar)
                # This needs to be customized based on actual format
                samples.append({
                    'path': line,
                    'index': idx
                })
        except json.JSONDecodeError:
            # Treat as path
            samples.append({
                'path': line,
                'index': idx
            })

    return samples


def load_camera_params_from_npz(npz_path: str, num_views: int = 6) -> List[CameraParams]:
    """
    Load camera parameters from NPZ file.

    Expected keys in NPZ:
    - 'c2w' or 'cameras': (N, 4, 4) camera-to-world matrices
    - 'intrinsics' or 'K': (N, 3, 3) or (N, 4) intrinsic matrices
    - 'image_size': (H, W) or (N, 2)

    Adjust based on actual data format.
    """
    data = np.load(npz_path, allow_pickle=True)

    cameras = []

    # Try different key names
    c2w_key = 'c2w' if 'c2w' in data else 'cameras' if 'cameras' in data else None
    K_key = 'intrinsics' if 'intrinsics' in data else 'K' if 'K' in data else 'fxfycxcy' if 'fxfycxcy' in data else None

    if c2w_key is None:
        raise KeyError(f"Cannot find c2w matrix. Available keys: {list(data.keys())}")

    c2w_matrices = data[c2w_key]

    # Handle intrinsics
    if K_key == 'fxfycxcy':
        # Format: (N, 4) with [fx, fy, cx, cy]
        fxfycxcy = data[K_key]
        for i in range(min(num_views, len(c2w_matrices))):
            fx, fy, cx, cy = fxfycxcy[i]
            cameras.append(CameraParams(
                view_id=i,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                c2w=c2w_matrices[i],
                image_size=(512, 512)  # Default, adjust as needed
            ))
    elif K_key:
        K_matrices = data[K_key]
        for i in range(min(num_views, len(c2w_matrices))):
            if K_matrices.shape[-1] == 3:
                # (N, 3, 3) format
                fx = K_matrices[i, 0, 0]
                fy = K_matrices[i, 1, 1]
                cx = K_matrices[i, 0, 2]
                cy = K_matrices[i, 1, 2]
            else:
                # (N, 4) format [fx, fy, cx, cy]
                fx, fy, cx, cy = K_matrices[i]

            cameras.append(CameraParams(
                view_id=i,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                c2w=c2w_matrices[i],
                image_size=(512, 512)
            ))
    else:
        # No intrinsics found, use defaults
        for i in range(min(num_views, len(c2w_matrices))):
            cameras.append(CameraParams(
                view_id=i,
                fx=549.0,  # FaceLift default
                fy=549.0,
                cx=256.0,
                cy=256.0,
                c2w=c2w_matrices[i],
                image_size=(512, 512)
            ))

    return cameras


def main():
    parser = argparse.ArgumentParser(
        description='Validate camera parameters for multi-view consistency'
    )
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='Path to dataset file (txt with sample paths or JSON)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./camera_diagnostics',
        help='Output directory for reports and visualizations'
    )
    parser.add_argument(
        '--num_views', type=int, default=6,
        help='Number of views per sample'
    )
    parser.add_argument(
        '--max_samples', type=int, default=100,
        help='Maximum number of samples to validate'
    )
    parser.add_argument(
        '--pp_threshold', type=float, default=10.0,
        help='Principal point offset warning threshold (pixels)'
    )
    parser.add_argument(
        '--scale_threshold', type=float, default=0.05,
        help='Scale ratio std warning threshold (relative)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize validator
    validator = CameraValidator(verbose=args.verbose)
    validator.PP_OFFSET_THRESHOLD = args.pp_threshold
    validator.SCALE_RATIO_STD_THRESHOLD = args.scale_threshold

    # Parse dataset
    if args.verbose:
        print(f"Loading dataset from: {args.dataset_path}")

    try:
        samples = parse_dataset_file(args.dataset_path)
    except Exception as e:
        print(f"Error parsing dataset: {e}")
        print("Please check the dataset format and adjust parse_dataset_file() function.")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(samples)} samples")

    # Validate samples
    validated = 0
    errors = 0

    for i, sample in enumerate(samples[:args.max_samples]):
        if args.verbose and i % 10 == 0:
            print(f"Processing sample {i+1}/{min(len(samples), args.max_samples)}...")

        try:
            # Load camera parameters
            if 'path' in sample:
                sample_path = sample['path']
                # Check if it's an NPZ file or directory
                if sample_path.endswith('.npz'):
                    cameras = load_camera_params_from_npz(sample_path, args.num_views)
                elif os.path.isdir(sample_path):
                    # Try to find NPZ in directory
                    npz_files = list(Path(sample_path).glob('*.npz'))
                    if npz_files:
                        cameras = load_camera_params_from_npz(str(npz_files[0]), args.num_views)
                    else:
                        # Try to load individual camera files
                        raise NotImplementedError(
                            "Directory format not implemented. "
                            "Please adjust load function for your data format."
                        )
                else:
                    raise ValueError(f"Unknown sample format: {sample_path}")

                sample_id = os.path.basename(sample_path)
            else:
                # Direct camera data in sample dict
                cameras = []
                for v in range(args.num_views):
                    cam_data = sample.get(f'view_{v}', sample.get('cameras', [{}])[v])
                    cameras.append(CameraParams(
                        view_id=v,
                        fx=cam_data.get('fx', 549),
                        fy=cam_data.get('fy', 549),
                        cx=cam_data.get('cx', 256),
                        cy=cam_data.get('cy', 256),
                        c2w=np.array(cam_data.get('c2w', np.eye(4))),
                        image_size=tuple(cam_data.get('image_size', (512, 512)))
                    ))
                sample_id = sample.get('id', f'sample_{i}')

            # Validate
            result = validator.validate_sample(sample_id, cameras)
            validated += 1

            if args.verbose and result.warnings:
                print(f"  {sample_id}: Score={result.overall_score:.0f}, "
                      f"Warnings: {len(result.warnings)}")

        except Exception as e:
            errors += 1
            if args.verbose:
                print(f"  Error processing sample {i}: {e}")

    # Generate report
    report_path = os.path.join(args.output_dir, 'camera_validation_report.json')
    report = validator.generate_report(report_path)

    # Generate visualizations
    if args.visualize:
        validator.visualize_diagnostics(args.output_dir)

    # Print summary
    print("\n" + "="*60)
    print("CAMERA VALIDATION SUMMARY")
    print("="*60)
    print(f"Samples processed: {validated}")
    print(f"Errors: {errors}")
    print(f"Mean validation score: {report['summary']['mean_score']:.1f}/100")
    print(f"Samples with warnings: {report['summary']['samples_with_warnings']}")
    print(f"  - PP offset warnings: {report['summary']['pp_warning_count']}")
    print(f"  - Scale consistency warnings: {report['summary']['scale_warning_count']}")
    print("\nPrincipal Point Analysis:")
    print(f"  Mean offset: {report['principal_point']['mean_offset']:.2f} px")
    print(f"  Max offset: {report['principal_point']['max_offset']:.2f} px")
    print("\nScale Consistency Analysis:")
    print(f"  Mean ratio std: {report['scale_consistency']['mean_ratio_std']*100:.2f}%")
    print(f"  Max ratio std: {report['scale_consistency']['max_ratio_std']*100:.2f}%")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    print("="*60)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == '__main__':
    main()
