#!/usr/bin/env python3
"""
D7 Preprocessing Verification Script
=====================================

Validates:
1. Sample count (train/val split)
2. Camera intrinsics (fx, fy, cx, cy)
3. PP distribution
4. Ray direction accuracy
5. Distance normalization
6. Aspect ratio preservation

Output: Markdown report in reports/ folder

Created: 2026-01-20
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np


def load_dataset_cameras(dataset_path: str, max_samples: int = 100) -> List[Dict]:
    """Load camera parameters from multiple samples"""
    dataset_path = Path(dataset_path)
    cameras_list = []

    for split in ['train', 'val']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        samples = sorted(os.listdir(split_path))[:max_samples]
        for sample in samples:
            cam_file = split_path / sample / 'opencv_cameras.json'
            if cam_file.exists():
                with open(cam_file) as f:
                    data = json.load(f)
                    if 'frames' in data:
                        for frame in data['frames']:
                            frame['_sample'] = sample
                            frame['_split'] = split
                        cameras_list.extend(data['frames'])
                    else:
                        cameras_list.append(data)

    return cameras_list


def compute_ray_direction_error(camera: Dict) -> float:
    """
    Compute ray direction error caused by intrinsics mismatch.

    If fx_actual != fx_recorded, the ray direction will be wrong.
    Error angle = arctan(delta_f / f)
    """
    if '_original' not in camera or '_transform' not in camera:
        return 0.0

    orig = camera['_original']
    transform = camera['_transform']

    # D7 uses scale based on fx only
    if 'scale' in transform:
        scale = transform['scale']
    elif 'scale_x' in transform:
        scale = transform['scale_x']  # D7.1
    elif 'scale_avg' in transform:
        scale = transform['scale_avg']  # D7.2
    else:
        return 0.0

    # Expected fy after scaling
    expected_fy = orig['fy'] * scale

    # Recorded fy
    recorded_fy = camera['fy']

    # Error in focal length
    delta_fy = abs(expected_fy - recorded_fy)

    # Convert to angle error (at image edge, 256 pixels from center)
    if recorded_fy > 0:
        angle_error_rad = np.arctan(256 * delta_fy / (recorded_fy * recorded_fy))
        return np.degrees(angle_error_rad)

    return 0.0


def analyze_cameras(cameras: List[Dict]) -> Dict:
    """Analyze camera parameters"""
    results = {
        'count': len(cameras),
        'fx': [], 'fy': [], 'cx': [], 'cy': [],
        'distance': [],
        'aspect_ratio_original': [],
        'ray_errors': [],
        'scale_x': [], 'scale_y': [],
    }

    for cam in cameras:
        results['fx'].append(cam.get('fx', 0))
        results['fy'].append(cam.get('fy', 0))
        results['cx'].append(cam.get('cx', 0))
        results['cy'].append(cam.get('cy', 0))

        if '_original' in cam:
            orig = cam['_original']
            if 'distance' in orig:
                results['distance'].append(orig['distance'])
            if 'fx' in orig and 'fy' in orig and orig['fx'] > 0:
                results['aspect_ratio_original'].append(orig['fy'] / orig['fx'])

        if '_transform' in cam:
            t = cam['_transform']
            if 'scale_x' in t:
                results['scale_x'].append(t['scale_x'])
            if 'scale_y' in t:
                results['scale_y'].append(t['scale_y'])
            elif 'scale' in t:
                results['scale_x'].append(t['scale'])
                results['scale_y'].append(t['scale'])

        ray_err = compute_ray_direction_error(cam)
        results['ray_errors'].append(ray_err)

    return results


def generate_report(
    dataset_name: str,
    dataset_path: str,
    analysis: Dict,
    output_path: str
) -> str:
    """Generate markdown verification report"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Statistics helper
    def stats(arr):
        if not arr:
            return "N/A"
        arr = np.array(arr)
        return f"mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}"

    report = f"""# {dataset_name} Preprocessing Verification Report

> Generated: {now}
> Dataset: {dataset_path}

---

## 1. Sample Count

| Metric | Value |
|--------|-------|
| Total camera views analyzed | {analysis['count']} |

---

## 2. Intrinsics Analysis

### 2.1 Focal Length (fx, fy)

| Parameter | Statistics |
|-----------|------------|
| **fx** | {stats(analysis['fx'])} |
| **fy** | {stats(analysis['fy'])} |

**Expected (FaceLift)**: fx = fy = 549.0

"""

    # Check fx, fy accuracy
    fx_arr = np.array(analysis['fx'])
    fy_arr = np.array(analysis['fy'])

    fx_error = np.abs(fx_arr - 549.0).mean() if len(fx_arr) > 0 else 0
    fy_error = np.abs(fy_arr - 549.0).mean() if len(fy_arr) > 0 else 0

    report += f"""
**Accuracy Check**:
- fx error from 549: {fx_error:.4f} (mean absolute)
- fy error from 549: {fy_error:.4f} (mean absolute)
- fx == fy (all views)? {np.allclose(fx_arr, fy_arr) if len(fx_arr) > 0 else 'N/A'}

"""

    if analysis['aspect_ratio_original']:
        ar = np.array(analysis['aspect_ratio_original'])
        report += f"""
### 2.2 Original Aspect Ratio (fy/fx)

| Statistics |
|------------|
| {stats(analysis['aspect_ratio_original'])} |

**Note**: Aspect ratio != 1.0 indicates non-square pixels in original cameras.
- If aspect ratio > 1.0: fy > fx (vertical field of view smaller)
- Current D7 uses fx-only scale, which introduces ~{(ar.mean()-1)*100:.2f}% error in fy.

"""

    report += f"""
### 2.3 Principal Point (cx, cy)

| Parameter | Statistics |
|-----------|------------|
| **cx** | {stats(analysis['cx'])} |
| **cy** | {stats(analysis['cy'])} |

**Expected (FaceLift)**: cx = cy = 256.0

"""

    cx_arr = np.array(analysis['cx'])
    cy_arr = np.array(analysis['cy'])

    cx_error = np.abs(cx_arr - 256.0).mean() if len(cx_arr) > 0 else 0
    cy_error = np.abs(cy_arr - 256.0).mean() if len(cy_arr) > 0 else 0

    report += f"""
**Accuracy Check**:
- cx error from 256: {cx_error:.4f}
- cy error from 256: {cy_error:.4f}
- PP exactly at center? {cx_error < 0.01 and cy_error < 0.01}

---

## 3. Ray Direction Error Analysis

Ray direction error occurs when recorded intrinsics don't match actual image transform.

| Statistics |
|------------|
| {stats(analysis['ray_errors'])} |

"""

    ray_arr = np.array(analysis['ray_errors'])
    if len(ray_arr) > 0:
        report += f"""
**Interpretation**:
- Max ray error: {ray_arr.max():.4f} degrees
- Mean ray error: {ray_arr.mean():.4f} degrees
- Risk level: {'LOW' if ray_arr.max() < 0.5 else 'MEDIUM' if ray_arr.max() < 2 else 'HIGH'}

"""

    if analysis['distance']:
        report += f"""
---

## 4. Distance Normalization

Original camera distances (before normalization):

| Statistics |
|------------|
| {stats(analysis['distance'])} |

**Target distance**: 2.7

"""

    if analysis['scale_x'] and analysis['scale_y']:
        report += f"""
---

## 5. Scale Factor Analysis

| Parameter | Statistics |
|-----------|------------|
| **scale_x** | {stats(analysis['scale_x'])} |
| **scale_y** | {stats(analysis['scale_y'])} |

"""
        sx = np.array(analysis['scale_x'])
        sy = np.array(analysis['scale_y'])
        if len(sx) > 0 and len(sy) > 0:
            scale_diff = np.abs(sx - sy).mean()
            report += f"""
**Scale difference (scale_x vs scale_y)**:
- Mean absolute difference: {scale_diff:.6f}
- Percentage difference: {scale_diff / sx.mean() * 100:.4f}%

"""

    report += f"""
---

## 6. Issues Summary

"""

    issues = []

    # Check fx/fy accuracy
    if fy_error > 0.01:
        issues.append(f"fy not exactly 549: error = {fy_error:.4f}")

    # Check PP
    if cx_error > 0.01 or cy_error > 0.01:
        issues.append(f"PP not exactly (256,256): cx_err={cx_error:.4f}, cy_err={cy_error:.4f}")

    # Check ray error
    if len(ray_arr) > 0 and ray_arr.max() > 0.5:
        issues.append(f"Ray direction error: max={ray_arr.max():.4f} degrees")

    # Check aspect ratio
    if analysis['aspect_ratio_original']:
        ar = np.array(analysis['aspect_ratio_original'])
        if not np.allclose(ar, 1.0, atol=0.001):
            issues.append(f"Non-square pixels in original: aspect ratio = {ar.mean():.4f}")

    if issues:
        for issue in issues:
            report += f"- :warning: {issue}\n"
    else:
        report += "- :white_check_mark: No critical issues found\n"

    report += f"""
---

## 7. Recommendations

"""

    if fy_error > 0.01 and analysis['aspect_ratio_original']:
        ar = np.array(analysis['aspect_ratio_original'])
        report += f"""
### fy Normalization Issue

**Problem**: D7 uses `scale = target_fx / orig_fx` for both x and y.
Original cameras have aspect ratio {ar.mean():.4f} (fy/fx), causing ~{(ar.mean()-1)*100:.2f}% fy error.

**Recommended Fix Options**:

1. **D7.1 (Individual Scale)**:
   - `scale_x = target_fx / orig_fx`
   - `scale_y = target_fy / orig_fy`
   - Pro: Geometrically correct
   - Con: Non-isotropic scaling may cause minor artifacts

2. **D7.2 (Average Scale)**:
   - `scale = (scale_x + scale_y) / 2`
   - Pro: Isotropic scaling, balanced error
   - Con: Neither fx nor fy exactly 549

"""

    report += f"""
---

*Report generated by verify_D7_preprocessing.py*
"""

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify D7 preprocessing")
    parser.add_argument('--dataset', required=True, help='Dataset path (e.g., /path/to/D7)')
    parser.add_argument('--name', default='D7', help='Dataset name for report')
    parser.add_argument('--output', required=True, help='Output report path')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples to analyze')

    args = parser.parse_args()

    print(f"Loading cameras from {args.dataset}...")
    cameras = load_dataset_cameras(args.dataset, args.max_samples)
    print(f"Loaded {len(cameras)} camera views")

    print("Analyzing camera parameters...")
    analysis = analyze_cameras(cameras)

    print(f"Generating report to {args.output}...")
    report = generate_report(args.name, args.dataset, analysis, args.output)

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
