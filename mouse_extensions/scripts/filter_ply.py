#!/usr/bin/env python3
"""
Filter PLY files by opacity threshold — post-hoc Gaussian filtering.

Terminology (GAUSSIAN_FILTERING_THEORY.md):
  "Pruning" = training-time adaptive density control (Kerbl 2023)
  "Filtering" = post-hoc opacity thresholding (THIS script)

Creates a parallel directory of filtered PLYs alongside originals.
Preserves all Gaussian properties, just removes low-opacity entries.

Usage:
    python -m mouse_extensions.scripts.filter_ply \
        --source /node_data/joon/data/shared/FaceLift_mouse_6view/gaussians/ply_a0.3 \
        --output /node_data/joon/data/shared/FaceLift_mouse_6view/gaussians/pruned/a0.3_t0.2 \
        --opacity_threshold 0.2

Terminology (see docs/theory/GAUSSIAN_FILTERING_THEORY.md):
  - "Pruning" = training-time adaptive density control (Kerbl 2023)
  - "Filtering" = post-hoc opacity thresholding (this script)
  We use "prune" in filename for user familiarity, but method is filtering.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def prune_single_ply(src_path: Path, dst_path: Path, threshold: float) -> dict:
    """Filter a single PLY by opacity threshold."""
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(src_path))
    vertex = ply["vertex"]
    n_original = len(vertex.data)

    # Get opacity — check if logit or sigmoid'd
    opacity_raw = vertex["opacity"]
    # GS-LRM stores logit (pre-sigmoid). Check range to determine.
    if opacity_raw.min() < -1 or opacity_raw.max() > 1.5:
        # Logit space — apply sigmoid
        opacity = sigmoid(opacity_raw)
    else:
        # Already in [0,1] range
        opacity = opacity_raw

    # Filter
    mask = opacity > threshold
    n_kept = mask.sum()

    # Create filtered vertex data
    filtered_data = vertex.data[mask]
    filtered_element = PlyElement.describe(filtered_data, "vertex")

    # Write
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([filtered_element]).write(str(dst_path))

    return {
        "n_original": n_original,
        "n_kept": int(n_kept),
        "n_removed": int(n_original - n_kept),
        "ratio_kept": float(n_kept / n_original),
        "src_size_mb": src_path.stat().st_size / 1024 ** 2,
        "dst_size_mb": dst_path.stat().st_size / 1024 ** 2,
    }


def prune_directory(src_dir: Path, dst_dir: Path, threshold: float) -> list:
    """Prune all PLY files in a directory (preserving train/test/val structure)."""
    results = []

    for split in ["train", "val", "test"]:
        split_src = src_dir / split
        split_dst = dst_dir / split
        if not split_src.exists():
            continue

        ply_files = sorted(split_src.glob("*.ply"))
        logger.info(f"Processing {split}: {len(ply_files)} files")

        for ply_path in tqdm(ply_files, desc=f"Pruning {split}"):
            dst_path = split_dst / ply_path.name
            try:
                stats = prune_single_ply(ply_path, dst_path, threshold)
                stats["split"] = split
                stats["frame"] = ply_path.stem
                results.append(stats)
            except Exception as e:
                logger.error(f"Error processing {ply_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prune PLY files by opacity threshold")
    parser.add_argument("--source", type=str, required=True, help="Source PLY directory (e.g., ply_a0.3)")
    parser.add_argument("--output", type=str, required=True, help="Output directory (e.g., pruned/a0.3_t0.2)")
    parser.add_argument("--opacity_threshold", type=float, default=0.2, help="Opacity threshold (sigmoid space)")
    args = parser.parse_args()

    src_dir = Path(args.source)
    dst_dir = Path(args.output)

    logger.info(f"Source: {src_dir}")
    logger.info(f"Output: {dst_dir}")
    logger.info(f"Threshold: sigmoid(opacity) > {args.opacity_threshold}")

    results = prune_directory(src_dir, dst_dir, args.opacity_threshold)

    if not results:
        logger.error("No files processed!")
        return

    # Summary
    arr = np.array([(r["n_original"], r["n_kept"], r["src_size_mb"], r["dst_size_mb"]) for r in results])
    logger.info(
        f"\n{'='*60}\n"
        f"Pruning complete: {len(results)} files\n"
        f"Threshold: sigmoid(opacity) > {args.opacity_threshold}\n"
        f"Gaussians: {arr[:,0].mean():.0f} → {arr[:,1].mean():.0f} "
        f"(kept {arr[:,1].sum()/arr[:,0].sum()*100:.1f}%)\n"
        f"File size: {arr[:,2].mean():.1f}MB → {arr[:,3].mean():.1f}MB "
        f"({arr[:,3].sum()/arr[:,2].sum()*100:.1f}%)\n"
        f"Total: {arr[:,2].sum()/1024:.1f}GB → {arr[:,3].sum()/1024:.1f}GB\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
