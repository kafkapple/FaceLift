#!/usr/bin/env python3
"""
Center Estimation Validation Script

Validates and compares the three center estimation methods:
1. Triangulation
2. Visual Hull  
3. Global Average

Usage:
    python validate_center_estimation.py --data_dir /path/to/data --num_samples 10
    
Output:
    - Per-method accuracy metrics
    - Cross-view consistency measurements
    - Visualization of results

Author: AI Research Assistant
Date: 2026-01-17
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mouse_extensions.preprocessing.center_estimation import (
    CenterEstimator,
    CenterMethod,
    CenterEstimationResult,
    compare_all_methods,
)


def load_raw_data(
    raw_dir: Path,
    frame_idx: int,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Load masks and cameras for a specific frame from raw data.
    
    Returns:
        masks: List of (H, W) binary masks
        cameras: List of camera dicts with K, w2c
    """
    # Load camera parameters
    cam_file = raw_dir / "new_cam.pkl"
    with open(cam_file, 'rb') as f:
        cam_params = pickle.load(f)
    
    masks = []
    cameras = []
    
    # Load masks from video files
    mask_dir = raw_dir / "simpleclick_undist"
    
    for view_idx in range(6):  # 6 views
        # Load mask from video
        video_path = mask_dir / f"{view_idx}.mp4"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masks.append(mask)
            else:
                print(f"Warning: Could not read frame {frame_idx} from view {view_idx}")
                masks.append(np.zeros((576, 512), dtype=np.uint8))
        else:
            print(f"Warning: Mask video not found: {video_path}")
            masks.append(np.zeros((576, 512), dtype=np.uint8))
        
        # Extract camera parameters
        cam = cam_params[view_idx]
        K = cam['K']
        R = cam['R']
        T = cam['T']
        
        # Build w2c matrix
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.flatten()
        
        cameras.append({
            'K': K,
            'w2c': w2c.tolist(),
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
        })
    
    return masks, cameras


def load_preprocessed_data(
    data_dir: Path,
    sample_idx: int,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Load masks and cameras from preprocessed dataset (D1, D2, v13).
    """
    # Find sample directory
    train_dir = data_dir / "train"
    if not train_dir.exists():
        train_dir = data_dir
    
    sample_dirs = sorted(train_dir.glob("*"))
    if sample_idx >= len(sample_dirs):
        raise ValueError(f"Sample index {sample_idx} out of range")
    
    sample_dir = sample_dirs[sample_idx]
    
    # Load cameras.json
    cam_file = sample_dir / "cameras.json"
    with open(cam_file, 'r') as f:
        cameras_data = json.load(f)
    
    # Load masks and cameras
    masks = []
    cameras = []
    
    for view_key in sorted(cameras_data.keys()):
        cam = cameras_data[view_key]
        
        # Load mask
        mask_path = sample_dir / f"{view_key}.png"
        if mask_path.exists():
            # Read image and create mask from alpha or as grayscale
            img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if img.shape[-1] == 4:  # RGBA
                mask = img[:, :, 3]  # Alpha channel
            else:
                mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = (mask > 128).astype(np.uint8) * 255
        else:
            # Create mask from image (assume white background)
            img_path = sample_dir / f"{view_key}.png"
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = (gray < 250).astype(np.uint8) * 255
        
        masks.append(mask)
        
        # Extract camera params
        cameras.append({
            'K': np.array([
                [cam['fx'], 0, cam['cx']],
                [0, cam['fy'], cam['cy']],
                [0, 0, 1]
            ]),
            'w2c': np.array(cam['w2c']),
            'fx': cam['fx'],
            'fy': cam['fy'],
            'cx': cam['cx'],
            'cy': cam['cy'],
        })
    
    return masks, cameras


def compute_cross_view_consistency(
    centers_2d: np.ndarray,
    center_3d: np.ndarray,
    cameras: List[Dict],
) -> Dict:
    """
    Measure how consistently the 3D center projects across views.
    
    Returns metrics for cross-view consistency.
    """
    # Reproject 3D center to all views
    reprojected = []
    for i, cam in enumerate(cameras):
        K = np.array(cam['K']) if isinstance(cam['K'], list) else cam['K']
        w2c = np.array(cam['w2c']) if isinstance(cam['w2c'], list) else cam['w2c']
        
        # Project
        point_hom = np.append(center_3d, 1.0)
        point_cam = w2c @ point_hom
        point_img = K @ point_cam[:3]
        u = point_img[0] / point_img[2]
        v = point_img[1] / point_img[2]
        reprojected.append([u, v])
    
    reprojected = np.array(reprojected)
    
    # Compute consistency metrics
    reprojection_errors = np.linalg.norm(reprojected - centers_2d, axis=1)
    
    return {
        'mean_reprojection_error': float(reprojection_errors.mean()),
        'max_reprojection_error': float(reprojection_errors.max()),
        'std_reprojection_error': float(reprojection_errors.std()),
        'per_view_errors': reprojection_errors.tolist(),
    }


def validate_single_sample(
    masks: List[np.ndarray],
    cameras: List[Dict],
    verbose: bool = False,
) -> Dict:
    """
    Validate all three methods on a single sample.
    """
    masks_array = np.array(masks)
    
    # Compare all methods
    results = compare_all_methods(masks_array, cameras)
    
    # Analyze results
    analysis = {}
    for method_name, result in results.items():
        consistency = compute_cross_view_consistency(
            result.centers_2d,
            result.center_3d,
            cameras,
        )
        
        analysis[method_name] = {
            'center_3d': result.center_3d.tolist(),
            'confidence': result.confidence,
            'ray_convergence_error': result.ray_convergence_error,
            **consistency,
        }
        
        if verbose:
            print(f"\n{method_name.upper()}:")
            print(f"  3D Center: {result.center_3d}")
            print(f"  Confidence: {result.confidence:.4f}")
            print(f"  Ray Error: {result.ray_convergence_error:.4f}")
            print(f"  Reproj Error: {consistency['mean_reprojection_error']:.2f}px")
    
    return analysis


def validate_dataset(
    data_dir: Path,
    num_samples: int = 10,
    data_type: str = 'preprocessed',  # 'raw' or 'preprocessed'
    output_file: Optional[Path] = None,
) -> Dict:
    """
    Validate center estimation on multiple samples.
    """
    all_results = []
    
    for i in tqdm(range(num_samples), desc="Validating samples"):
        try:
            if data_type == 'raw':
                # Sample random frames
                frame_idx = i * 100  # Sample every 100 frames
                masks, cameras = load_raw_data(data_dir, frame_idx)
            else:
                masks, cameras = load_preprocessed_data(data_dir, i)
            
            result = validate_single_sample(masks, cameras)
            result['sample_idx'] = i
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Aggregate statistics
    summary = aggregate_results(all_results)
    
    # Save results
    if output_file:
        output = {
            'samples': all_results,
            'summary': summary,
        }
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    return summary


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate validation results across samples."""
    methods = ['triangulation', 'visual_hull', 'global_average']
    summary = {}
    
    for method in methods:
        method_results = [r[method] for r in results if method in r]
        
        if not method_results:
            continue
        
        ray_errors = [r['ray_convergence_error'] for r in method_results]
        reproj_errors = [r['mean_reprojection_error'] for r in method_results]
        confidences = [r['confidence'] for r in method_results]
        
        summary[method] = {
            'ray_convergence_error': {
                'mean': float(np.mean(ray_errors)),
                'std': float(np.std(ray_errors)),
                'min': float(np.min(ray_errors)),
                'max': float(np.max(ray_errors)),
            },
            'reprojection_error': {
                'mean': float(np.mean(reproj_errors)),
                'std': float(np.std(reproj_errors)),
                'min': float(np.min(reproj_errors)),
                'max': float(np.max(reproj_errors)),
            },
            'confidence': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
            },
            'num_samples': len(method_results),
        }
    
    return summary


def print_summary(summary: Dict):
    """Print formatted summary."""
    print("\n" + "="*70)
    print("CENTER ESTIMATION VALIDATION SUMMARY")
    print("="*70)
    
    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Method", "Ray Error", "Reproj Error", "Confidence"
    ))
    print("-"*70)
    
    for method in ['triangulation', 'visual_hull', 'global_average']:
        if method not in summary:
            continue
        s = summary[method]
        print("{:<20} {:>12.4f}±{:<5.2f} {:>10.2f}±{:<4.1f}px {:>10.4f}".format(
            method,
            s['ray_convergence_error']['mean'],
            s['ray_convergence_error']['std'],
            s['reprojection_error']['mean'],
            s['reprojection_error']['std'],
            s['confidence']['mean'],
        ))
    
    print("\n" + "="*70)
    
    # Recommendation
    best_method = min(summary.keys(), 
                      key=lambda m: summary[m]['ray_convergence_error']['mean'])
    print(f"\n✅ RECOMMENDED: {best_method.upper()}")
    print(f"   Lowest ray convergence error: {summary[best_method]['ray_convergence_error']['mean']:.4f}")


def visualize_comparison(
    masks: List[np.ndarray],
    cameras: List[Dict],
    output_path: Optional[Path] = None,
):
    """
    Visualize center estimation comparison across methods.
    """
    import matplotlib.pyplot as plt
    
    masks_array = np.array(masks)
    results = compare_all_methods(masks_array, cameras)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = {
        'triangulation': 'red',
        'visual_hull': 'blue', 
        'global_average': 'green',
    }
    
    # Show first 6 views
    for view_idx in range(min(6, len(masks))):
        ax = axes[view_idx // 3, view_idx % 3]
        ax.imshow(masks[view_idx], cmap='gray')
        ax.set_title(f'View {view_idx}')
        
        # Plot centers from each method
        for method_name, result in results.items():
            cx, cy = result.centers_2d[view_idx]
            ax.scatter(cx, cy, c=colors[method_name], s=100, 
                      marker='x', linewidths=2, label=method_name)
        
        if view_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Center Estimation Method Comparison', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Validate center estimation methods"
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to data directory (raw or preprocessed)'
    )
    parser.add_argument(
        '--data_type', type=str, default='raw',
        choices=['raw', 'preprocessed'],
        help='Type of data to load'
    )
    parser.add_argument(
        '--num_samples', type=int, default=10,
        help='Number of samples to validate'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate visualization'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed results'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_file = Path(args.output) if args.output else None
    
    # Run validation
    summary = validate_dataset(
        data_dir,
        num_samples=args.num_samples,
        data_type=args.data_type,
        output_file=output_file,
    )
    
    # Print summary
    print_summary(summary)
    
    # Visualization
    if args.visualize:
        if args.data_type == 'raw':
            masks, cameras = load_raw_data(data_dir, 0)
        else:
            masks, cameras = load_preprocessed_data(data_dir, 0)
        
        vis_path = output_file.with_suffix('.png') if output_file else None
        visualize_comparison(masks, cameras, vis_path)


if __name__ == '__main__':
    main()
