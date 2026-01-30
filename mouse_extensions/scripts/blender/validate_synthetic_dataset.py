#!/usr/bin/env python3
"""
Validate synthetic dataset for GS-LRM compatibility.

Usage:
    python validate_synthetic_dataset.py /path/to/dataset

Checks:
    1. Image files exist and have correct dimensions
    2. Camera parameters are valid (intrinsics, extrinsics)
    3. Camera arrangement is correct (distance, angles)
    4. GS-LRM compatibility (fx=549, cx=cy=256, etc.)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import sys


def check_images(dataset_dir: Path) -> bool:
    """Check image files."""
    print("\n[1] Checking images...")
    
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        print(f"  ✗ Images directory not found: {images_dir}")
        return False
    
    image_files = sorted(images_dir.glob("cam_*.png"))
    if len(image_files) == 0:
        print(f"  ✗ No cam_*.png files found")
        return False
    
    print(f"  Found {len(image_files)} images")
    
    # Check each image
    resolutions = []
    for img_path in image_files:
        try:
            img = Image.open(img_path)
            resolutions.append(img.size)
            
            # Check RGBA
            if img.mode != "RGBA":
                print(f"  ⚠️ {img_path.name}: mode={img.mode} (expected RGBA)")
        except Exception as e:
            print(f"  ✗ Failed to open {img_path.name}: {e}")
            return False
    
    # Check resolution consistency
    if len(set(resolutions)) > 1:
        print(f"  ✗ Inconsistent resolutions: {set(resolutions)}")
        return False
    
    w, h = resolutions[0]
    if w != h:
        print(f"  ⚠️ Non-square images: {w}x{h}")
    
    print(f"  ✓ All images valid: {w}x{h} RGBA")
    return True


def check_camera_params(dataset_dir: Path) -> dict:
    """Check camera parameters."""
    print("\n[2] Checking camera parameters...")
    
    json_path = dataset_dir / "opencv_cameras.json"
    if not json_path.exists():
        print(f"  ✗ Camera file not found: {json_path}")
        return None
    
    with open(json_path) as f:
        data = json.load(f)
    
    frames = data.get("frames", [])
    if len(frames) == 0:
        print(f"  ✗ No frames in camera file")
        return None
    
    print(f"  Found {len(frames)} camera frames")
    
    # Check intrinsics
    fx_vals = [f["fx"] for f in frames]
    fy_vals = [f["fy"] for f in frames]
    cx_vals = [f["cx"] for f in frames]
    cy_vals = [f["cy"] for f in frames]
    
    fx_mean, fx_std = np.mean(fx_vals), np.std(fx_vals)
    fy_mean, fy_std = np.mean(fy_vals), np.std(fy_vals)
    cx_mean, cx_std = np.mean(cx_vals), np.std(cx_vals)
    cy_mean, cy_std = np.mean(cy_vals), np.std(cy_vals)
    
    print(f"  Intrinsics:")
    print(f"    fx = {fx_mean:.2f} ± {fx_std:.4f}")
    print(f"    fy = {fy_mean:.2f} ± {fy_std:.4f}")
    print(f"    cx = {cx_mean:.2f} ± {cx_std:.4f}")
    print(f"    cy = {cy_mean:.2f} ± {cy_std:.4f}")
    
    # Check GS-LRM compatibility
    print(f"\n  GS-LRM Compatibility:")
    
    # fx ≈ 549
    if abs(fx_mean - 549) < 10:
        print(f"    ✓ fx ≈ 549 (pretrained compatible)")
    else:
        print(f"    ⚠️ fx = {fx_mean:.1f} (pretrained expects ~549)")
    
    # fx = fy
    if abs(fx_mean - fy_mean) < 1:
        print(f"    ✓ fx = fy (square pixels)")
    else:
        print(f"    ✗ fx ≠ fy ({fx_mean:.1f} vs {fy_mean:.1f})")
    
    # cx = cy = resolution/2
    resolution = frames[0].get("w", 512)
    expected_center = resolution / 2
    if abs(cx_mean - expected_center) < 1 and abs(cy_mean - expected_center) < 1:
        print(f"    ✓ cx = cy = {expected_center} (centered)")
    else:
        print(f"    ⚠️ PP offset: cx={cx_mean:.1f}, cy={cy_mean:.1f} (expected {expected_center})")
    
    return data


def check_extrinsics(data: dict) -> bool:
    """Check extrinsic matrices."""
    print("\n[3] Checking extrinsics...")
    
    frames = data["frames"]
    
    distances = []
    for i, f in enumerate(frames):
        w2c = np.array(f["w2c"])
        
        # Check shape
        if w2c.shape != (4, 4):
            print(f"  ✗ Frame {i}: w2c shape {w2c.shape}")
            return False
        
        # Check rotation orthogonality
        R = w2c[:3, :3]
        RRT = R @ R.T
        if not np.allclose(RRT, np.eye(3), atol=0.01):
            print(f"  ✗ Frame {i}: rotation not orthogonal")
            det = np.linalg.det(R)
            print(f"    det(R) = {det:.4f}")
            return False
        
        # Compute camera position
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        dist = np.linalg.norm(cam_pos)
        distances.append(dist)
    
    dist_mean = np.mean(distances)
    dist_std = np.std(distances)
    
    print(f"  Camera distances: {dist_mean:.3f} ± {dist_std:.4f}")
    
    # Check GS-LRM compatibility (distance ≈ 2.7)
    if abs(dist_mean - 2.7) < 0.1:
        print(f"  ✓ Distance ≈ 2.7 (pretrained compatible)")
    else:
        print(f"  ⚠️ Distance = {dist_mean:.2f} (pretrained expects ~2.7)")
    
    # Check consistency
    if dist_std > 0.01:
        print(f"  ⚠️ Camera distances vary: std={dist_std:.4f}")
    else:
        print(f"  ✓ All cameras at same distance")
    
    print(f"  ✓ All {len(frames)} extrinsic matrices valid")
    return True


def check_view_coverage(data: dict) -> bool:
    """Check angular coverage of views."""
    print("\n[4] Checking view coverage...")
    
    frames = data["frames"]
    
    # Extract azimuths
    if "azimuth_deg" in frames[0]:
        azimuths = [f["azimuth_deg"] for f in frames]
    else:
        # Compute from camera positions
        azimuths = []
        for f in frames:
            c2w = np.linalg.inv(np.array(f["w2c"]))
            cam_pos = c2w[:3, 3]
            az = np.degrees(np.arctan2(cam_pos[0], cam_pos[1]))
            if az < 0:
                az += 360
            azimuths.append(az)
    
    azimuth_strs = [f"{a:.0f}°" for a in azimuths]
    print(f"  Azimuths: {azimuth_strs}")
    
    # Check spacing
    azimuths_sorted = sorted(azimuths)
    gaps = []
    for i in range(len(azimuths_sorted)):
        next_idx = (i + 1) % len(azimuths_sorted)
        gap = azimuths_sorted[next_idx] - azimuths_sorted[i]
        if gap < 0:
            gap += 360
        gaps.append(gap)
    
    expected_gap = 360 / len(frames)
    gap_std = np.std(gaps)
    
    if gap_std < 5:
        print(f"  ✓ Even angular spacing: ~{expected_gap:.0f}° gaps")
    else:
        gap_strs = [f"{g:.0f}°" for g in gaps]
        print(f"  ⚠️ Uneven spacing: gaps={gap_strs}")
    
    # Check 360° coverage
    total_coverage = sum(gaps)
    if abs(total_coverage - 360) < 1:
        print(f"  ✓ Full 360° coverage")
    else:
        print(f"  ⚠️ Coverage: {total_coverage:.0f}° (expected 360°)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic dataset")
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings")
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    
    print("=" * 60)
    print("Synthetic Dataset Validation")
    print("=" * 60)
    print(f"Dataset: {dataset_dir}")
    
    if not dataset_dir.exists():
        print(f"\n✗ Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Run checks
    ok = True
    ok &= check_images(dataset_dir)
    
    data = check_camera_params(dataset_dir)
    ok &= (data is not None)
    
    if data:
        ok &= check_extrinsics(data)
        ok &= check_view_coverage(data)
    
    print("\n" + "=" * 60)
    if ok:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print("=" * 60)
    
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
