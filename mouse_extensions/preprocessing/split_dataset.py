#!/usr/bin/env python3
"""
Generate Temporal Stratified Train/Val Split

Divides time-series data into temporal strata (chunks) and samples
from each stratum for both train and val sets, ensuring similar
temporal distribution in both splits.

Usage:
    python generate_temporal_split.py --data_dir /path/to/data --val_ratio 0.1 --num_strata 10
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple


def get_sample_dirs(data_dir: Path) -> List[Path]:
    """Get all sample directories sorted by name (which reflects temporal order)."""
    samples = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("sample_")
    ])
    return samples


def temporal_stratified_split(
    samples: List[Path],
    val_ratio: float = 0.1,
    num_strata: int = 10,
    seed: int = 42
) -> Tuple[List[Path], List[Path]]:
    """
    Split samples using temporal stratification.
    
    Args:
        samples: List of sample paths in temporal order
        val_ratio: Fraction of samples for validation
        num_strata: Number of temporal chunks to divide data into
        seed: Random seed for reproducibility
    
    Returns:
        (train_samples, val_samples)
    """
    random.seed(seed)
    
    n_samples = len(samples)
    stratum_size = n_samples // num_strata
    
    train_samples = []
    val_samples = []
    
    for i in range(num_strata):
        start_idx = i * stratum_size
        if i == num_strata - 1:
            # Last stratum gets remaining samples
            end_idx = n_samples
        else:
            end_idx = (i + 1) * stratum_size
        
        stratum = samples[start_idx:end_idx]
        
        # Shuffle within stratum
        stratum_shuffled = stratum.copy()
        random.shuffle(stratum_shuffled)
        
        # Split stratum
        n_val = max(1, int(len(stratum) * val_ratio))
        val_samples.extend(stratum_shuffled[:n_val])
        train_samples.extend(stratum_shuffled[n_val:])
    
    # Shuffle final lists
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    return train_samples, val_samples


def write_split_file(samples: List[Path], output_path: Path):
    """Write sample paths to a text file."""
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(str(sample) + "\n")
    print(f"Written {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Temporal Stratified Split")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--num_strata", type=int, default=10, help="Number of temporal strata")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    # Get samples
    samples = get_sample_dirs(data_dir)
    print(f"Found {len(samples)} samples")
    
    if len(samples) == 0:
        print("No samples found!")
        return
    
    # Split
    train_samples, val_samples = temporal_stratified_split(
        samples,
        val_ratio=args.val_ratio,
        num_strata=args.num_strata,
        seed=args.seed
    )
    
    print(f"\nSplit Results:")
    print(f"  Train: {len(train_samples)} samples ({100*len(train_samples)/len(samples):.1f}%)")
    print(f"  Val:   {len(val_samples)} samples ({100*len(val_samples)/len(samples):.1f}%)")
    
    # Verify temporal distribution
    print(f"\nTemporal Distribution Check:")
    train_indices = sorted([int(s.name.split("_")[1]) for s in train_samples])
    val_indices = sorted([int(s.name.split("_")[1]) for s in val_samples])
    
    print(f"  Train range: {min(train_indices)} - {max(train_indices)}")
    print(f"  Val range:   {min(val_indices)} - {max(val_indices)}")
    
    # Write split files
    write_split_file(train_samples, data_dir / "data_mouse_train.txt")
    write_split_file(val_samples, data_dir / "data_mouse_val.txt")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
