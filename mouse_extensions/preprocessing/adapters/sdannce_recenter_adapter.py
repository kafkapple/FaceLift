"""
SDANNCE Scene-Centric Recenter Adapter

Takes existing rat preprocessed data (rat2_s1despill_s2fxnorm/) which was processed
through sdannce_to_gslrm.py (despill + fxnorm + per-camera distance scaling),
and applies scene-center recentering to match mouse pipeline conventions.

Why this exists (audit-driven plan v2 Phase A.2):
- Existing rat preprocessing missed the mouse-style scene recentering step
- Mouse cameras: centered around scene origin → model assumption satisfied
- Rat cameras: clustered far from scene origin → model assumption violated
- This adapter fixes ONLY the recentering issue, leaving images and FX as-is
- Test the "centering matters" hypothesis without redoing full preprocessing

Pipeline:
    Input:  /node_data/joon/data/preprocessed/FaceLift_rat/rat2_s1despill_s2fxnorm/
    Action: For each frame, recenter cameras using rat com3d as scene center
    Output: /node_data/joon/data/preprocessed/FaceLift_rat/rat2_v8_recentered/

Honest framing (per audit):
- This is a CONTROL EXPERIMENT to disambiguate "preprocessing" vs "narrow baseline"
- Pairwise camera angles are unchanged (recentering is angle-invariant)
- If v8 ≈ v6: narrow baseline is the dominant issue, pivot to Phase D
- If v8 >> v6: centering was a major factor, continue refining

Created: 2026-04-07 (Plan v2 Phase A.2)
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio

from mouse_extensions.preprocessing.scene_centered_normalizer import (
    normalize_cameras_scene_centric,
    compute_scene_center_from_com3d,
)


# === Conversion factor (sdannce mm → FaceLift units) ===
# From earlier analysis: rat data was scaled by ~0.00234 units/mm
# Verify: cam1 raw (-8, 1039, 403)mm → converted (-0.02, 2.44, 0.94) units
# scale = 2.44 / 1039 ≈ 0.00235
SDANNCE_MM_TO_UNITS = 0.00234  # Approximate, computed from actual conversion


def load_com3d_in_units(com3d_mat_path: str, scale_factor: float = SDANNCE_MM_TO_UNITS) -> np.ndarray:
    """Load com3d.mat and convert to FaceLift units.

    Args:
        com3d_mat_path: Path to sdannce com3d.mat file
        scale_factor: mm → FaceLift units conversion (default measured value)

    Returns:
        (N, 3) array of rat 3D positions in FaceLift units
    """
    mat = sio.loadmat(com3d_mat_path)
    com3d_mm = np.asarray(mat["com"], dtype=np.float64)  # (N, 3) in mm
    if com3d_mm.shape[1] != 3:
        raise ValueError(f"com3d shape {com3d_mm.shape}, expected (N, 3)")
    return com3d_mm * scale_factor


def recenter_camera_json(cameras_json: dict, scene_center: np.ndarray, target_distance: float = 2.7) -> dict:
    """Apply scene-centric recentering to a single opencv_cameras.json content.

    Args:
        cameras_json: Loaded JSON dict (with "frames" key containing camera list)
        scene_center: (3,) scene center in FaceLift units
        target_distance: Target mean cam-to-origin distance after recenter

    Returns:
        Modified cameras_json (deep copy with updated w2c)
    """
    new_json = {"frames": [], "_preprocessing": cameras_json.get("_preprocessing", {})}

    # Convert frames list → cam_params_list format expected by wrapper
    cam_params_list = []
    for frame in cameras_json["frames"]:
        cam_params_list.append({
            "w2c": frame["w2c"],
            "fx": frame["fx"],
            "fy": frame["fy"],
            "cx": frame["cx"],
            "cy": frame["cy"],
            "w": frame["w"],
            "h": frame["h"],
            "file_path": frame["file_path"],
            "view_id": frame["view_id"],
        })

    # Apply wrapper
    recentered = normalize_cameras_scene_centric(
        cam_params_list,
        scene_center=scene_center,
        target_distance=target_distance,
    )

    # Build new frames list (preserve all original keys, only w2c updated)
    for orig_frame, new_params in zip(cameras_json["frames"], recentered):
        new_frame = dict(orig_frame)
        new_frame["w2c"] = new_params["w2c"]
        new_json["frames"].append(new_frame)

    # Mark preprocessing version
    new_json["_preprocessing"] = {
        **new_json.get("_preprocessing", {}),
        "scene_centered": True,
        "scene_center": list(map(float, scene_center)),
        "target_distance": target_distance,
        "adapter_version": "v8_recenter_v1",
    }

    return new_json


def adapt_dataset(
    input_dir: str,
    output_dir: str,
    com3d_mat_path: str,
    target_distance: float = 2.7,
    method: str = "median",
    scale_factor: float = SDANNCE_MM_TO_UNITS,
    copy_images: bool = True,
    overwrite: bool = False,
) -> dict:
    """Apply scene-centric recentering to an entire rat dataset.

    Args:
        input_dir: Path to existing rat preprocessed dataset (rat2_s1despill_s2fxnorm/)
        output_dir: Path to write recentered dataset (will be created)
        com3d_mat_path: Path to sdannce com3d.mat (provides scene center reference)
        target_distance: Target mean cam-to-origin distance
        method: 'median' or 'mean' for scene center computation
        scale_factor: mm → units conversion (default: measured 0.00234)
        copy_images: If True, copy/symlink image files. If False, only write JSON.
        overwrite: If True, allow output_dir to exist (will overwrite JSONs)

    Returns:
        Stats dict with conversion summary
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dir not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output dir exists: {output_path}. Use --overwrite.")

    # Load com3d → compute scene center
    com3d_units = load_com3d_in_units(com3d_mat_path, scale_factor=scale_factor)
    scene_center = compute_scene_center_from_com3d(com3d_units, method=method)
    print(f"[Adapter] Scene center ({method}): "
          f"({scene_center[0]:+.4f}, {scene_center[1]:+.4f}, {scene_center[2]:+.4f}) units")
    print(f"[Adapter] Equivalent in mm: "
          f"({scene_center[0]/scale_factor:+.0f}, {scene_center[1]/scale_factor:+.0f}, "
          f"{scene_center[2]/scale_factor:+.0f}) mm")

    output_path.mkdir(parents=True, exist_ok=True)

    # Process each frame dir
    frame_dirs = sorted([p for p in input_path.iterdir() if p.is_dir() and p.name.isdigit()])
    print(f"[Adapter] Found {len(frame_dirs)} frame directories")

    n_processed = 0
    n_skipped = 0
    for frame_dir in frame_dirs:
        out_frame_dir = output_path / frame_dir.name
        out_frame_dir.mkdir(exist_ok=True)

        cam_json_in = frame_dir / "opencv_cameras.json"
        cam_json_out = out_frame_dir / "opencv_cameras.json"

        if not cam_json_in.exists():
            n_skipped += 1
            continue

        with open(cam_json_in) as f:
            cameras_json = json.load(f)

        new_json = recenter_camera_json(cameras_json, scene_center, target_distance)

        with open(cam_json_out, "w") as f:
            json.dump(new_json, f, indent=2)

        # Symlink or copy images
        if copy_images:
            in_images = frame_dir / "images"
            out_images = out_frame_dir / "images"
            if in_images.exists() and not out_images.exists():
                # Symlink to save space (images are unchanged)
                out_images.symlink_to(in_images.absolute())

        n_processed += 1
        if n_processed % 500 == 0:
            print(f"[Adapter] Processed {n_processed}/{len(frame_dirs)} frames")

    # Copy split files (train/val/test .txt) if present
    for split_file in input_path.glob("data_*.txt"):
        out_split = output_path / split_file.name
        # Update paths in split file to point to new output location
        with open(split_file) as f:
            paths = [line.strip() for line in f if line.strip()]
        new_paths = []
        for p in paths:
            # Replace input_path prefix with output_path
            if p.startswith(str(input_path)):
                new_p = str(output_path) + p[len(str(input_path)):]
            else:
                # Just replace dir name
                new_p = p.replace(input_path.name, output_path.name)
            new_paths.append(new_p)
        with open(out_split, "w") as f:
            f.write("\n".join(new_paths) + "\n")
        print(f"[Adapter] Updated split file: {split_file.name}")

    stats = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "scene_center": list(map(float, scene_center)),
        "target_distance": target_distance,
        "n_frames_processed": n_processed,
        "n_frames_skipped": n_skipped,
        "method": method,
        "scale_factor": scale_factor,
    }

    # Save stats
    with open(output_path / "_adapter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[Adapter] Done. Processed {n_processed}, skipped {n_skipped}")
    print(f"[Adapter] Stats saved to {output_path / '_adapter_stats.json'}")

    return stats


def verify_recentering(output_dir: str, sample_frame: Optional[str] = None) -> None:
    """Verify the recentered dataset has scene at origin and correct distances."""
    output_path = Path(output_dir)
    if sample_frame is None:
        # Pick first frame
        frames = sorted([p for p in output_path.iterdir() if p.is_dir() and p.name.isdigit()])
        if not frames:
            raise FileNotFoundError("No frame dirs in output")
        sample_path = frames[0] / "opencv_cameras.json"
    else:
        sample_path = output_path / sample_frame / "opencv_cameras.json"

    with open(sample_path) as f:
        data = json.load(f)

    positions = []
    for frame in data["frames"]:
        w2c = np.asarray(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        positions.append(c2w[:3, 3])
    positions = np.array(positions)

    centroid = positions.mean(axis=0)
    distances = np.linalg.norm(positions, axis=1)

    # Pairwise angles around origin (= scene center after recentering)
    vn = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    angles = []
    for i in range(len(vn)):
        for j in range(i+1, len(vn)):
            angles.append(np.degrees(np.arccos(np.clip(np.dot(vn[i], vn[j]), -1, 1))))

    print(f"\n[Verify] Sample: {sample_path.parent.name}")
    print(f"  Camera centroid:    ({centroid[0]:+.4f}, {centroid[1]:+.4f}, {centroid[2]:+.4f})")
    print(f"  Cam→origin distances: min={distances.min():.4f}, max={distances.max():.4f}, mean={distances.mean():.4f}")
    print(f"  Pairwise angles around origin: min={min(angles):.1f}°, max={max(angles):.1f}°, mean={np.mean(angles):.1f}°")
    print(f"  Recenter metadata: {data.get('_preprocessing', {})}")


def main():
    parser = argparse.ArgumentParser(description="Apply scene-centric recentering to rat dataset")
    parser.add_argument("--input_dir", required=True, help="Input rat preprocessed dataset")
    parser.add_argument("--output_dir", required=True, help="Output recentered dataset")
    parser.add_argument("--com3d", required=True, help="sdannce com3d.mat path")
    parser.add_argument("--target_distance", type=float, default=2.7)
    parser.add_argument("--method", choices=["median", "mean"], default="median")
    parser.add_argument("--scale_factor", type=float, default=SDANNCE_MM_TO_UNITS,
                        help="mm → FaceLift units conversion factor")
    parser.add_argument("--no_images", action="store_true", help="Skip image symlink (JSON only)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing output, no conversion")
    args = parser.parse_args()

    if args.verify_only:
        verify_recentering(args.output_dir)
        return

    stats = adapt_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        com3d_mat_path=args.com3d,
        target_distance=args.target_distance,
        method=args.method,
        scale_factor=args.scale_factor,
        copy_images=not args.no_images,
        overwrite=args.overwrite,
    )

    verify_recentering(args.output_dir)


if __name__ == "__main__":
    main()
