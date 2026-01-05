#!/usr/bin/env python3
"""
Normalize mouse camera poses to match FaceLift's expected camera convention.

FaceLift expects:
- Object at origin (0, 0, 0)
- Cameras at fixed distance 2.7 from origin
- Cameras on a sphere looking at origin
- All cameras at same elevation angle

This script transforms mouse camera poses to match this convention.
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert world-to-camera to camera-to-world matrix."""
    return np.linalg.inv(w2c)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Convert camera-to-world to world-to-camera matrix."""
    return np.linalg.inv(c2w)


def get_camera_position(w2c: np.ndarray) -> np.ndarray:
    """Get camera position in world coordinates from w2c matrix."""
    c2w = w2c_to_c2w(w2c)
    return c2w[:3, 3]


def compute_scene_center(frames: List[Dict]) -> np.ndarray:
    """
    Compute the scene center as the point closest to all camera viewing rays.
    Uses least-squares to find the point that minimizes distance to all rays.
    """
    # Collect ray origins and directions
    ray_origins = []
    ray_directions = []

    for frame in frames:
        w2c = np.array(frame['w2c'])
        c2w = w2c_to_c2w(w2c)

        # Camera position (ray origin)
        cam_pos = c2w[:3, 3]

        # Camera viewing direction (negative Z in camera space, transformed to world)
        # In OpenCV convention, camera looks along +Z
        view_dir = c2w[:3, 2]  # Third column is Z axis direction
        view_dir = view_dir / np.linalg.norm(view_dir)

        ray_origins.append(cam_pos)
        ray_directions.append(view_dir)

    ray_origins = np.array(ray_origins)
    ray_directions = np.array(ray_directions)

    # Solve for the point closest to all rays using least squares
    # For each ray: point = origin + t * direction
    # We want to find P that minimizes sum of squared distances to all rays

    n_cameras = len(ray_origins)

    # Build the system: sum of projection matrices
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(n_cameras):
        d = ray_directions[i]
        o = ray_origins[i]

        # Projection matrix onto the plane perpendicular to ray
        I = np.eye(3)
        P = I - np.outer(d, d)

        A += P
        b += P @ o

    # Solve A @ center = b
    try:
        center = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: use centroid of camera positions
        center = ray_origins.mean(axis=0)

    return center


def normalize_cameras(
    frames: List[Dict],
    target_distance: float = 2.7,
    center_object: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Normalize camera poses to FaceLift convention.

    Steps:
    1. Find scene center (object location)
    2. Translate all cameras so object is at origin
    3. Scale camera distances to target_distance

    Args:
        frames: List of frame dictionaries with 'w2c' matrices
        target_distance: Target camera distance from origin (FaceLift uses 2.7)
        center_object: Whether to center the object at origin

    Returns:
        Tuple of (normalized_frames, transform_info)
    """
    # Step 1: Compute scene center
    scene_center = compute_scene_center(frames)

    # Get original camera positions and distances
    original_positions = []
    original_distances = []
    for frame in frames:
        w2c = np.array(frame['w2c'])
        pos = get_camera_position(w2c)
        dist = np.linalg.norm(pos - scene_center)
        original_positions.append(pos)
        original_distances.append(dist)

    avg_distance = np.mean(original_distances)
    scale_factor = target_distance / avg_distance

    # Step 2 & 3: Transform cameras
    normalized_frames = []

    for i, frame in enumerate(frames):
        w2c = np.array(frame['w2c'])
        c2w = w2c_to_c2w(w2c)

        # Get current camera position relative to scene center
        cam_pos = c2w[:3, 3]
        relative_pos = cam_pos - scene_center

        # Scale to target distance
        new_pos = relative_pos * scale_factor

        # Create new c2w matrix
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = new_pos

        # Convert back to w2c
        new_w2c = c2w_to_w2c(new_c2w)

        # Create new frame
        new_frame = frame.copy()
        new_frame['w2c'] = new_w2c.tolist()
        normalized_frames.append(new_frame)

    transform_info = {
        'scene_center': scene_center.tolist(),
        'original_avg_distance': float(avg_distance),
        'target_distance': target_distance,
        'scale_factor': float(scale_factor),
    }

    return normalized_frames, transform_info


def process_sample(
    input_dir: Path,
    output_dir: Path,
    target_distance: float = 2.7
) -> bool:
    """Process a single sample directory."""
    cameras_path = input_dir / "opencv_cameras.json"

    if not cameras_path.exists():
        print(f"Warning: No cameras file in {input_dir}")
        return False

    # Load cameras
    with open(cameras_path, 'r') as f:
        camera_data = json.load(f)

    # Normalize cameras
    normalized_frames, transform_info = normalize_cameras(
        camera_data['frames'],
        target_distance=target_distance
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    input_images_dir = input_dir / "images"
    output_images_dir = output_dir / "images"

    if input_images_dir.exists():
        if output_images_dir.exists():
            shutil.rmtree(output_images_dir)
        shutil.copytree(input_images_dir, output_images_dir)

    # Save normalized cameras
    output_camera_data = camera_data.copy()
    output_camera_data['frames'] = normalized_frames
    output_camera_data['normalization_info'] = transform_info

    with open(output_dir / "opencv_cameras.json", 'w') as f:
        json.dump(output_camera_data, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Normalize camera poses to FaceLift convention"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory with samples"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for normalized data"
    )
    parser.add_argument(
        "--target_distance", type=float, default=2.7,
        help="Target camera distance from origin (default: 2.7)"
    )
    parser.add_argument(
        "--sample_prefix", type=str, default="sample_",
        help="Prefix for sample directories"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Find all sample directories
    sample_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and d.name.startswith(args.sample_prefix)
    ])

    print(f"Found {len(sample_dirs)} samples")
    print(f"Target camera distance: {args.target_distance}")

    # Process samples
    successful = 0
    failed = 0

    for sample_dir in tqdm(sample_dirs, desc="Normalizing cameras"):
        sample_name = sample_dir.name
        output_sample_dir = output_dir / sample_name

        try:
            if process_sample(sample_dir, output_sample_dir, args.target_distance):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            failed += 1

    # Copy data lists if they exist
    for list_file in input_dir.glob("*.txt"):
        # Update paths in the list file
        with open(list_file, 'r') as f:
            lines = f.readlines()

        # Replace input_dir name with output_dir name
        new_lines = []
        for line in lines:
            new_line = line.replace(input_dir.name, output_dir.name)
            new_lines.append(new_line)

        output_list_file = output_dir / list_file.name
        with open(output_list_file, 'w') as f:
            f.writelines(new_lines)

    print(f"\n=== Camera Normalization Complete ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")

    # Verify one sample
    if successful > 0:
        sample = list(output_dir.glob(f"{args.sample_prefix}*"))[0]
        with open(sample / "opencv_cameras.json", 'r') as f:
            data = json.load(f)

        print(f"\n=== Verification ({sample.name}) ===")
        for i, frame in enumerate(data['frames'][:3]):
            w2c = np.array(frame['w2c'])
            c2w = np.linalg.inv(w2c)
            pos = c2w[:3, 3]
            dist = np.linalg.norm(pos)
            print(f"View {i}: pos={pos.round(3)}, dist={dist:.3f}")

        if 'normalization_info' in data:
            info = data['normalization_info']
            print(f"\nTransform info:")
            print(f"  Scene center: {np.array(info['scene_center']).round(3)}")
            print(f"  Original avg dist: {info['original_avg_distance']:.3f}")
            print(f"  Scale factor: {info['scale_factor']:.3f}")


if __name__ == "__main__":
    main()
