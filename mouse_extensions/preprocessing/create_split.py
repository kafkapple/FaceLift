#!/usr/bin/env python3
"""
Generate train/val split text files from a preprocessed dataset.

Usage:
    python -m mouse_extensions.preprocessing.create_split \
        --data-dir /path/to/M3 \
        --val-ratio 0.1 \
        --seed 42 \
        --output-suffix ""  # creates train.txt, val.txt
    
    # Different split
    python -m mouse_extensions.preprocessing.create_split \
        --data-dir /path/to/M3 \
        --val-ratio 0.2 \
        --seed 123 \
        --output-suffix "_80_20"  # creates train_80_20.txt, val_80_20.txt
"""

import argparse
import random
from pathlib import Path


def create_split(data_dir: Path, val_ratio: float = 0.1, seed: int = 42, suffix: str = ""):
    """Generate split text files from samples/ folder."""
    
    samples_dir = data_dir / "samples"
    if not samples_dir.exists():
        # Fallback: try to find all sample directories in data_dir root
        samples_dir = data_dir
    
    # Find all sample directories
    sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not sample_dirs:
        raise ValueError(f"No sample directories found in {samples_dir}")
    
    print(f"Found {len(sample_dirs)} samples")
    
    # Shuffle with seed
    random.seed(seed)
    indices = list(range(len(sample_dirs)))
    random.shuffle(indices)
    
    # Split
    num_val = int(len(sample_dirs) * val_ratio)
    val_indices = set(indices[:num_val])
    
    train_paths = []
    val_paths = []
    
    for i, sample_dir in enumerate(sample_dirs):
        path = str(sample_dir.absolute()) + "/"
        if i in val_indices:
            val_paths.append(path)
        else:
            train_paths.append(path)
    
    # Write text files
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    train_file = splits_dir / f"train{suffix}.txt"
    val_file = splits_dir / f"val{suffix}.txt"
    
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_paths))
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_paths))
    
    # Also create symlinks in root for convenience
    root_train = data_dir / f"data_mouse_train{suffix}.txt"
    root_val = data_dir / f"data_mouse_val{suffix}.txt"
    
    # Remove old symlinks if exist
    root_train.unlink(missing_ok=True)
    root_val.unlink(missing_ok=True)
    
    # Create new symlinks
    root_train.symlink_to(train_file.relative_to(data_dir))
    root_val.symlink_to(val_file.relative_to(data_dir))
    
    print(f"Created: {train_file} ({len(train_paths)} samples)")
    print(f"Created: {val_file} ({len(val_paths)} samples)")
    print(f"Symlinks: {root_train.name}, {root_val.name}")
    
    return train_paths, val_paths


def main():
    parser = argparse.ArgumentParser(description="Generate train/val split text files")
    parser.add_argument("--data-dir", type=Path, required=True, help="Preprocessed data directory")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--suffix", type=str, default="", help="Output file suffix (e.g., '_80_20')")
    
    args = parser.parse_args()
    create_split(args.data_dir, args.val_ratio, args.seed, args.suffix)


if __name__ == "__main__":
    main()
