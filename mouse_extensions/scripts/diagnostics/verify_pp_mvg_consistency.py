#!/usr/bin/env python3
"""
PP (Principal Point) MVG Consistency Verification Script

This script verifies that all preprocessed datasets maintain MVG (Multi-View Geometry)
consistency by checking:
1. PP distribution (should be 256±0 for consistency)
2. Ray direction error (should be < 1° for accurate reconstruction)

Usage:
    python verify_pp_mvg_consistency.py [--datasets D1,D2,...] [--verbose]

Author: Claude Code
Date: 2026-01-25
Related: PP_FIX_MVG_THEORY.md, COMPREHENSIVE_ANALYSIS_260125.md
"""

import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Constants
DEFAULT_BASE_PATH = '" + str(Path.home()) + "/data/preprocessed/FaceLift_mouse'
TARGET_PP = 256.0
TARGET_FX = 548.9937744140625
MAX_ACCEPTABLE_RAY_ERROR = 1.0  # degrees

# Known datasets
KNOWN_DATASETS = [
    'D3_normalized',
    'D7_1', 
    'D8',
    'M3_norm',
    'M3_persample',
    'M3_1',
    'M3_2',  # Expected after fix
]


def compute_ray_error_degrees(pp_offset: float, fx: float = TARGET_FX) -> float:
    """Compute ray direction error in degrees from PP offset.
    
    Ray error ≈ arctan(PP_offset / fx)
    """
    return np.degrees(np.arctan(abs(pp_offset) / fx))


def analyze_dataset_pp(dataset_path: Path, max_samples: int = 100) -> Optional[Dict]:
    """Analyze PP distribution in a dataset.
    
    Returns:
        Dict with cx, cy statistics and ray error, or None if dataset not found
    """
    if not dataset_path.exists():
        return None
    
    cx_values = []
    cy_values = []
    fx_values = []
    
    # Try train folder first, then samples
    for subfolder in ['train', 'samples', '']:
        search_path = dataset_path / subfolder if subfolder else dataset_path
        pattern = '**/opencv_cameras.json'
        
        files = list(search_path.glob(pattern))[:max_samples]
        if files:
            break
    
    if not files:
        return None
    
    for cam_file in files:
        try:
            with open(cam_file) as f:
                data = json.load(f)
            
            for frame in data.get('frames', []):
                cx = frame.get('cx', frame.get('principal_point_x'))
                cy = frame.get('cy', frame.get('principal_point_y'))
                fx = frame.get('fx', frame.get('focal_length_x'))
                
                if cx is not None and cy is not None:
                    cx_values.append(cx)
                    cy_values.append(cy)
                if fx is not None:
                    fx_values.append(fx)
        except (json.JSONDecodeError, KeyError):
            continue
    
    if not cx_values:
        return None
    
    cx_arr = np.array(cx_values)
    cy_arr = np.array(cy_values)
    
    # Compute statistics
    cx_mean, cx_std = cx_arr.mean(), cx_arr.std()
    cy_mean, cy_std = cy_arr.mean(), cy_arr.std()
    
    # PP offset from target (256)
    cx_offset = abs(cx_mean - TARGET_PP)
    cy_offset = abs(cy_mean - TARGET_PP)
    max_pp_offset = max(cx_offset, cy_offset)
    
    # Maximum individual offset
    max_individual_offset = max(
        abs(cx_arr - TARGET_PP).max(),
        abs(cy_arr - TARGET_PP).max()
    )
    
    # Ray error
    avg_fx = np.array(fx_values).mean() if fx_values else TARGET_FX
    ray_error_mean = compute_ray_error_degrees(max_pp_offset, avg_fx)
    ray_error_max = compute_ray_error_degrees(max_individual_offset, avg_fx)
    
    return {
        'n_samples': len(files),
        'n_frames': len(cx_values),
        'cx_mean': cx_mean,
        'cx_std': cx_std,
        'cx_min': cx_arr.min(),
        'cx_max': cx_arr.max(),
        'cy_mean': cy_mean,
        'cy_std': cy_std,
        'cy_min': cy_arr.min(),
        'cy_max': cy_arr.max(),
        'fx_mean': avg_fx,
        'pp_offset_mean': max_pp_offset,
        'pp_offset_max': max_individual_offset,
        'ray_error_mean_deg': ray_error_mean,
        'ray_error_max_deg': ray_error_max,
        'mvg_consistent': ray_error_max < MAX_ACCEPTABLE_RAY_ERROR,
    }


def print_summary_table(results: Dict[str, Dict], verbose: bool = False):
    """Print formatted summary table."""
    print()
    print('=' * 90)
    print('PP (Principal Point) MVG Consistency Verification Report')
    print('=' * 90)
    print(f'Target PP: {TARGET_PP}, Target fx: {TARGET_FX:.2f}')
    print(f'Acceptable ray error: < {MAX_ACCEPTABLE_RAY_ERROR}°')
    print('-' * 90)
    print(f'{"Dataset":<25} {"cx (mean±std)":<18} {"cy (mean±std)":<18} {"Ray Err (max)":<15} {"Status":<10}')
    print('-' * 90)
    
    for name, stats in sorted(results.items()):
        if stats is None:
            print(f'{name:<25} {"NOT FOUND":<18}')
            continue
        
        cx_str = f"{stats['cx_mean']:.1f}±{stats['cx_std']:.1f}"
        cy_str = f"{stats['cy_mean']:.1f}±{stats['cy_std']:.1f}"
        ray_str = f"{stats['ray_error_max_deg']:.2f}°"
        status = '✅ OK' if stats['mvg_consistent'] else '❌ BAD'
        
        print(f'{name:<25} {cx_str:<18} {cy_str:<18} {ray_str:<15} {status:<10}')
    
    print('=' * 90)
    
    # Summary
    ok_count = sum(1 for s in results.values() if s and s['mvg_consistent'])
    bad_count = sum(1 for s in results.values() if s and not s['mvg_consistent'])
    missing = sum(1 for s in results.values() if s is None)
    
    print(f'Summary: {ok_count} OK, {bad_count} BAD, {missing} missing')
    print()
    
    if verbose:
        print('Detailed Statistics:')
        print('-' * 90)
        for name, stats in sorted(results.items()):
            if stats is None:
                continue
            print(f'\n{name}:')
            print(f'  Samples: {stats["n_samples"]}, Frames: {stats["n_frames"]}')
            print(f'  cx: {stats["cx_mean"]:.2f} ± {stats["cx_std"]:.2f} (range: {stats["cx_min"]:.1f} - {stats["cx_max"]:.1f})')
            print(f'  cy: {stats["cy_mean"]:.2f} ± {stats["cy_std"]:.2f} (range: {stats["cy_min"]:.1f} - {stats["cy_max"]:.1f})')
            print(f'  fx: {stats["fx_mean"]:.2f}')
            print(f'  PP offset: mean={stats["pp_offset_mean"]:.1f}px, max={stats["pp_offset_max"]:.1f}px')
            print(f'  Ray error: mean={stats["ray_error_mean_deg"]:.2f}°, max={stats["ray_error_max_deg"]:.2f}°')


def main():
    parser = argparse.ArgumentParser(description='Verify PP MVG consistency across datasets')
    parser.add_argument('--datasets', type=str, help='Comma-separated dataset names')
    parser.add_argument('--base-path', type=str, default=DEFAULT_BASE_PATH, help='Base path for datasets')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples to analyze per dataset')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed statistics')
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets = KNOWN_DATASETS
    
    results = {}
    for name in datasets:
        dataset_path = base_path / name
        print(f'Analyzing {name}...', file=sys.stderr)
        results[name] = analyze_dataset_pp(dataset_path, args.max_samples)
    
    print_summary_table(results, args.verbose)
    
    # Return exit code based on results
    bad_datasets = [n for n, s in results.items() if s and not s['mvg_consistent']]
    if bad_datasets:
        print(f'\n⚠️  WARNING: {len(bad_datasets)} dataset(s) have MVG consistency issues: {bad_datasets}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
