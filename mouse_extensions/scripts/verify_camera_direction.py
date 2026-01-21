#!/usr/bin/env python3
"""
Camera Direction Verification
==============================

Verify that camera directions (rotation matrices) are preserved
between original MAMMAL data and preprocessed FaceLift data.

Metrics:
1. Rotation matrix difference (Frobenius norm)
2. Forward direction angle difference
3. Up direction angle difference
4. Quaternion difference
"""

import numpy as np
import pickle
import json
import os
from scipy.spatial.transform import Rotation


def load_original_cameras(pkl_path):
    """Load original MAMMAL cameras."""
    with open(pkl_path, 'rb') as f:
        cameras = pickle.load(f)

    results = []
    for i, cam in enumerate(cameras):
        R = np.array(cam['R'])
        T = np.array(cam['T']).flatten()

        # Camera position
        pos = -R.T @ T

        # Forward direction (camera looks along +Z in camera frame, which is R.T @ [0,0,1] in world)
        forward = R.T @ np.array([0, 0, 1])
        forward = forward / np.linalg.norm(forward)

        # Up direction (Y axis of camera in world)
        up = R.T @ np.array([0, 1, 0])
        up = up / np.linalg.norm(up)

        # Right direction (X axis)
        right = R.T @ np.array([1, 0, 0])
        right = right / np.linalg.norm(right)

        results.append({
            'view_id': i,
            'R': R,
            'T': T,
            'position': pos,
            'forward': forward,
            'up': up,
            'right': right,
            'distance': np.linalg.norm(pos)
        })

    return results


def load_preprocessed_cameras(json_path):
    """Load preprocessed FaceLift cameras."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = []
    for frame in data['frames']:
        w2c = np.array(frame['w2c'])
        c2w = np.linalg.inv(w2c)

        R = w2c[:3, :3]
        T = w2c[:3, 3]
        pos = c2w[:3, 3]

        # Forward direction
        forward = -c2w[:3, 2]
        forward = forward / np.linalg.norm(forward)

        # Up direction
        up = c2w[:3, 1]
        up = up / np.linalg.norm(up)

        # Right direction
        right = c2w[:3, 0]
        right = right / np.linalg.norm(right)

        results.append({
            'view_id': frame.get('view_id', len(results)),
            'R': R,
            'T': T,
            'position': pos,
            'forward': forward,
            'up': up,
            'right': right,
            'distance': np.linalg.norm(pos)
        })

    return results


def angle_between_vectors(v1, v2):
    """Compute angle between two vectors in degrees."""
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def rotation_difference(R1, R2):
    """Compute rotation difference metrics."""
    # Frobenius norm
    frob_norm = np.linalg.norm(R1 - R2, 'fro')

    # Geodesic distance (rotation angle)
    R_diff = R1 @ R2.T
    trace = np.trace(R_diff)
    trace_clipped = np.clip((trace - 1) / 2, -1.0, 1.0)
    geodesic_angle = np.degrees(np.arccos(trace_clipped))

    return frob_norm, geodesic_angle


def verify_cameras(orig_cams, proc_cams):
    """Verify camera direction consistency."""

    print('=' * 70)
    print('CAMERA DIRECTION VERIFICATION')
    print('=' * 70)
    print()

    results = []

    for i in range(len(orig_cams)):
        orig = orig_cams[i]
        proc = proc_cams[i]

        # Rotation difference
        R_frob, R_geodesic = rotation_difference(orig['R'], proc['R'])

        # Direction angle differences
        forward_angle = angle_between_vectors(orig['forward'], proc['forward'])
        up_angle = angle_between_vectors(orig['up'], proc['up'])
        right_angle = angle_between_vectors(orig['right'], proc['right'])

        # Position direction (normalized)
        orig_pos_dir = orig['position'] / np.linalg.norm(orig['position'])
        proc_pos_dir = proc['position'] / np.linalg.norm(proc['position'])
        pos_angle = angle_between_vectors(orig_pos_dir, proc_pos_dir)

        # Distance ratio
        dist_ratio = proc['distance'] / orig['distance']

        results.append({
            'camera': i,
            'R_frob': R_frob,
            'R_geodesic': R_geodesic,
            'forward_angle': forward_angle,
            'up_angle': up_angle,
            'right_angle': right_angle,
            'pos_angle': pos_angle,
            'dist_ratio': dist_ratio,
            'orig_dist': orig['distance'],
            'proc_dist': proc['distance']
        })

    return results


def print_results(results):
    """Print verification results."""

    print('Per-Camera Analysis:')
    print('-' * 70)
    print('{:8} {:>12} {:>12} {:>12} {:>12} {:>12}'.format(
        'Camera', 'R_geodesic', 'Forward', 'Up', 'Right', 'Pos Dir'))
    print('{:8} {:>12} {:>12} {:>12} {:>12} {:>12}'.format(
        '', '(deg)', '(deg)', '(deg)', '(deg)', '(deg)'))
    print('-' * 70)

    for r in results:
        print('{:8} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}'.format(
            'Cam {}'.format(r['camera']),
            r['R_geodesic'],
            r['forward_angle'],
            r['up_angle'],
            r['right_angle'],
            r['pos_angle']
        ))

    print('-' * 70)

    # Summary statistics
    R_geodesic_vals = [r['R_geodesic'] for r in results]
    forward_vals = [r['forward_angle'] for r in results]
    up_vals = [r['up_angle'] for r in results]

    print()
    print('Summary Statistics:')
    print('-' * 70)
    print('Rotation Matrix Geodesic Distance:')
    print('  Mean: {:.6f} deg'.format(np.mean(R_geodesic_vals)))
    print('  Max:  {:.6f} deg'.format(np.max(R_geodesic_vals)))
    print('  Std:  {:.6f} deg'.format(np.std(R_geodesic_vals)))

    print()
    print('Forward Direction Angle:')
    print('  Mean: {:.6f} deg'.format(np.mean(forward_vals)))
    print('  Max:  {:.6f} deg'.format(np.max(forward_vals)))

    print()
    print('Up Direction Angle:')
    print('  Mean: {:.6f} deg'.format(np.mean(up_vals)))
    print('  Max:  {:.6f} deg'.format(np.max(up_vals)))

    # Verification verdict
    print()
    print('=' * 70)
    print('VERIFICATION RESULT')
    print('=' * 70)

    max_angle = max(max(R_geodesic_vals), max(forward_vals), max(up_vals))

    if max_angle < 0.001:
        print('Status: PERFECT MATCH')
        print('All camera directions are IDENTICAL (error < 0.001 deg)')
    elif max_angle < 0.1:
        print('Status: EXCELLENT MATCH')
        print('Camera directions match within 0.1 degree')
    elif max_angle < 1.0:
        print('Status: GOOD MATCH')
        print('Camera directions match within 1 degree')
    else:
        print('Status: WARNING - SIGNIFICANT DIFFERENCE')
        print('Maximum angle difference: {:.4f} deg'.format(max_angle))

    print()

    # Distance comparison
    print('Distance Scaling:')
    print('-' * 70)
    for r in results:
        print('Cam {}: {:.2f} mm -> {:.4f} (ratio: {:.6f})'.format(
            r['camera'], r['orig_dist'], r['proc_dist'], r['dist_ratio']))

    avg_ratio = np.mean([r['dist_ratio'] for r in results])
    print()
    print('Average scale factor: {:.6f}'.format(avg_ratio))

    return results


def print_detailed_matrices(orig_cams, proc_cams):
    """Print detailed matrix comparison."""

    print()
    print('=' * 70)
    print('DETAILED MATRIX COMPARISON (Camera 0)')
    print('=' * 70)

    print()
    print('Original R matrix:')
    for row in orig_cams[0]['R']:
        print('  [{:>12.8f} {:>12.8f} {:>12.8f}]'.format(*row))

    print()
    print('Preprocessed R matrix:')
    for row in proc_cams[0]['R']:
        print('  [{:>12.8f} {:>12.8f} {:>12.8f}]'.format(*row))

    print()
    print('Difference (R_orig - R_proc):')
    diff = orig_cams[0]['R'] - proc_cams[0]['R']
    for row in diff:
        print('  [{:>12.8f} {:>12.8f} {:>12.8f}]'.format(*row))

    print()
    print('Forward direction comparison:')
    print('  Original:     [{:>10.6f} {:>10.6f} {:>10.6f}]'.format(*orig_cams[0]['forward']))
    print('  Preprocessed: [{:>10.6f} {:>10.6f} {:>10.6f}]'.format(*proc_cams[0]['forward']))

    print()
    print('Up direction comparison:')
    print('  Original:     [{:>10.6f} {:>10.6f} {:>10.6f}]'.format(*orig_cams[0]['up']))
    print('  Preprocessed: [{:>10.6f} {:>10.6f} {:>10.6f}]'.format(*proc_cams[0]['up']))


def save_report(results, output_path):
    """Save verification report as JSON."""

    report = {
        'summary': {
            'R_geodesic_mean': float(np.mean([r['R_geodesic'] for r in results])),
            'R_geodesic_max': float(np.max([r['R_geodesic'] for r in results])),
            'forward_angle_mean': float(np.mean([r['forward_angle'] for r in results])),
            'forward_angle_max': float(np.max([r['forward_angle'] for r in results])),
            'up_angle_mean': float(np.mean([r['up_angle'] for r in results])),
            'up_angle_max': float(np.max([r['up_angle'] for r in results])),
            'avg_scale_factor': float(np.mean([r['dist_ratio'] for r in results])),
            'status': 'MATCH' if max([r['R_geodesic'] for r in results]) < 0.1 else 'MISMATCH'
        },
        'per_camera': results
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('Report saved: {}'.format(output_path))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify camera direction consistency')
    parser.add_argument('--original', type=str,
                       default='/home/joon/data/markerless_mouse_1_nerf/new_cam.pkl',
                       help='Path to original new_cam.pkl')
    parser.add_argument('--preprocessed', type=str,
                       default='/home/joon/data/preprocessed/FaceLift_mouse/D7_1/train/000000/opencv_cameras.json',
                       help='Path to preprocessed opencv_cameras.json')
    parser.add_argument('--output', type=str,
                       default='/home/joon/dev/FaceLift/mouse_extensions/reports/camera_verification_report.json',
                       help='Output report path')
    args = parser.parse_args()

    print('Loading original cameras: {}'.format(args.original))
    orig_cams = load_original_cameras(args.original)

    print('Loading preprocessed cameras: {}'.format(args.preprocessed))
    proc_cams = load_preprocessed_cameras(args.preprocessed)

    print()
    results = verify_cameras(orig_cams, proc_cams)
    print_results(results)
    print_detailed_matrices(orig_cams, proc_cams)

    # Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_report(results, args.output)


if __name__ == '__main__':
    main()
