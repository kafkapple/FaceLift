#!/usr/bin/env python3
"""
Temporal Split Preprocessing for Mouse Dataset
===============================================

Creates train/val/test splits using PoseSplatter-style temporal separation.
This script processes existing preprocessed datasets (D7, D7_5, D7_5b) and
creates new file lists with temporal splits.

Dataset Versioning:
- D7   → D7_t   (temporal split version)
- D7_5 → D7_5_t (temporal split version)
- D7_5b → D7_5b_t (temporal split version)

Split Ratios (PoseSplatter style):
- Train: first 34% (frames 0 ~ 1224)
- Val:   middle 33% (frames 1224 ~ 2412)
- Test:  last 33% (frames 2412 ~ 3600)

Created: 2026-01-19
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

from temporal_split_utils import split_frames, SplitResult


@dataclass
class TemporalSplitConfig:
    """Configuration for temporal split preprocessing."""
    train_ratio: float = 0.34
    val_ratio: float = 0.33
    test_ratio: float = 0.33


def load_sample_paths(list_file: Path) -> List[str]:
    """Load sample paths from a file list."""
    with open(list_file) as f:
        return [line.strip() for line in f if line.strip()]


def extract_frame_index(sample_path: str, dataset_dir: Path) -> int:
    """
    Extract original frame index from a sample.

    Reads opencv_cameras.json to get the original_frame_idx.
    """
    camera_file = Path(sample_path) / "opencv_cameras.json"
    if camera_file.exists():
        with open(camera_file) as f:
            data = json.load(f)
            preprocess = data.get("_preprocessing", {})
            return preprocess.get("original_frame_idx", -1)
    return -1


def create_temporal_split(
    source_dir: Path,
    output_dir: Path,
    config: TemporalSplitConfig
) -> Tuple[int, int, int]:
    """
    Create temporal split from existing preprocessed dataset.

    Args:
        source_dir: Path to source dataset (e.g., D7)
        output_dir: Path to output dataset (e.g., D7_t)
        config: Split configuration

    Returns:
        Tuple of (train_count, val_count, test_count)
    """
    print(f"\n{'='*60}")
    print(f"Creating Temporal Split Dataset")
    print(f"{'='*60}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")

    # Load all samples from train and val
    all_samples = []

    train_file = source_dir / "data_mouse_train.txt"
    val_file = source_dir / "data_mouse_val.txt"

    if train_file.exists():
        all_samples.extend(load_sample_paths(train_file))
    if val_file.exists():
        all_samples.extend(load_sample_paths(val_file))

    print(f"Total samples: {len(all_samples)}")

    # Extract frame indices for each sample
    print("\nExtracting frame indices...")
    sample_frame_map = {}
    frame_indices = []

    for sample_path in all_samples:
        frame_idx = extract_frame_index(sample_path, source_dir)
        if frame_idx >= 0:
            sample_frame_map[frame_idx] = sample_path
            frame_indices.append(frame_idx)

    print(f"Valid samples with frame index: {len(frame_indices)}")
    print(f"Frame range: {min(frame_indices)} - {max(frame_indices)}")

    # Apply temporal split
    print("\nApplying temporal split...")
    split_result = split_frames(
        frame_indices,
        method="temporal",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )

    print(split_result.summary())
    print(f"\nTrain frame range: {min(split_result.train_indices)} - {max(split_result.train_indices)}")
    print(f"Val frame range:   {min(split_result.val_indices)} - {max(split_result.val_indices)}")
    print(f"Test frame range:  {min(split_result.test_indices)} - {max(split_result.test_indices)}")

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks and file lists
    train_paths = []
    val_paths = []
    test_paths = []

    for split_name, indices, path_list in [
        ("train", split_result.train_indices, train_paths),
        ("val", split_result.val_indices, val_paths),
        ("test", split_result.test_indices, test_paths)
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for i, frame_idx in enumerate(sorted(indices)):
            source_path = Path(sample_frame_map[frame_idx])
            # Create new sample directory name based on index in this split
            new_sample_id = f"{i:06d}"
            dest_path = split_dir / new_sample_id

            # Create symlink to original sample
            if not dest_path.exists():
                dest_path.symlink_to(source_path.resolve())

            path_list.append(str(dest_path) + "/")

    # Write file lists
    for split_name, paths in [
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths)
    ]:
        list_file = output_dir / f"data_mouse_{split_name}.txt"
        with open(list_file, "w") as f:
            f.write("\n".join(sorted(paths)))
        print(f"Saved {split_name}: {len(paths)} samples -> {list_file}")

    # Write split info
    split_info = {
        "method": "temporal",
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "train_count": len(train_paths),
        "val_count": len(val_paths),
        "test_count": len(test_paths),
        "train_frame_range": [min(split_result.train_indices), max(split_result.train_indices)],
        "val_frame_range": [min(split_result.val_indices), max(split_result.val_indices)],
        "test_frame_range": [min(split_result.test_indices), max(split_result.test_indices)],
        "source_dataset": str(source_dir),
        "description": "PoseSplatter-style temporal split for generalization evaluation"
    }

    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return len(train_paths), len(val_paths), len(test_paths)


def process_all_datasets(
    base_dir: Path,
    datasets: List[str],
    config: TemporalSplitConfig
):
    """Process multiple datasets to create temporal split versions."""

    results = {}

    for dataset_name in datasets:
        source_dir = base_dir / dataset_name
        output_name = f"{dataset_name}_t"  # e.g., D7 -> D7_t
        output_dir = base_dir / output_name

        if not source_dir.exists():
            print(f"\nSkipping {dataset_name}: not found")
            continue

        train_n, val_n, test_n = create_temporal_split(
            source_dir, output_dir, config
        )

        results[output_name] = {
            "train": train_n,
            "val": val_n,
            "test": test_n
        }

    # Print summary
    print(f"\n{'='*60}")
    print("Temporal Split Summary")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)

    for name, counts in results.items():
        total = counts["train"] + counts["val"] + counts["test"]
        print(f"{name:<15} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8} {total:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Create temporal split versions of mouse datasets"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/joon/data/preprocessed/FaceLift_mouse",
        help="Base directory containing preprocessed datasets"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["D7", "D7_5", "D7_5b"],
        help="Dataset names to process (e.g., D7 D7_5 D7_5b)"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.34,
        help="Training set ratio (default: 0.34)"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.33,
        help="Validation set ratio (default: 0.33)"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.33,
        help="Test set ratio (default: 0.33)"
    )

    args = parser.parse_args()

    config = TemporalSplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    process_all_datasets(
        base_dir=Path(args.base_dir),
        datasets=args.datasets,
        config=config
    )


if __name__ == "__main__":
    main()
