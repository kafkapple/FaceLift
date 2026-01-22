#!/usr/bin/env python3
"""
Generate Train/Val Split from Existing Preprocessed Data

Allows regenerating splits without re-preprocessing.
Same preprocessing + different splits = multiple dataset variants.

Usage:
    # Default 90/10 split
    python -m mouse_extensions.preprocessing.generate_split \
        --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9

    # Custom split ratio + auto-update config
    python -m mouse_extensions.preprocessing.generate_split \
        --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
        --val-ratio 0.2 \
        --update-config D9

    # Different random seed for different split
    python -m mouse_extensions.preprocessing.generate_split \
        --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
        --val-ratio 0.1 \
        --seed 123 \
        --update-config D9
"""

import argparse
from pathlib import Path
import numpy as np
import re


def update_dataset_config(config_name: str, data_dir: Path, num_train: int, num_val: int):
    """Update dataset config with new paths and stats."""
    
    # Find config file
    base_dir = Path(__file__).parent.parent.parent  # FaceLift root
    config_path = base_dir / "configs" / "datasets" / f"{config_name}.yaml"
    
    if not config_path.exists():
        print(f"WARNING: Config not found: {config_path}")
        return False
    
    # Read current config
    with open(config_path) as f:
        content = f.read()
    
    train_path = data_dir / "data_mouse_train.txt"
    val_path = data_dir / "data_mouse_val.txt"
    
    # Update dataset_path in training section
    content = re.sub(
        r'(training:\s*\n\s*dataset:\s*\n\s*dataset_path:\s*).*',
        f'\\1{train_path}',
        content
    )
    
    # Update dataset_path in validation section
    content = re.sub(
        r'(validation:\s*\n\s*dataset_path:\s*).*',
        f'\\1{val_path}',
        content
    )
    
    # Update stats
    content = re.sub(
        r'(_stats:\s*\n\s*num_train:\s*).*',
        f'\\g<1>{num_train}',
        content
    )
    content = re.sub(
        r'(num_val:\s*).*',
        f'\\g<1>{num_val}',
        content
    )
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(content)
    
    print(f"Updated config: {config_path}")
    print(f"  training.dataset.dataset_path: {train_path}")
    print(f"  validation.dataset_path: {val_path}")
    
    return True


def generate_split(data_dir: Path, val_ratio: float = 0.1, seed: int = 42, update_config: str = None):
    """Generate train/val split from existing data."""
    
    # Find all sample directories
    sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not sample_dirs:
        # Check if data_mouse_all.txt exists
        all_file = data_dir / "data_mouse_all.txt"
        if all_file.exists():
            with open(all_file) as f:
                all_paths = [line.strip() for line in f if line.strip()]
        else:
            print(f"ERROR: No sample directories or data_mouse_all.txt found in {data_dir}")
            return
    else:
        all_paths = [str(d) + '/' for d in sample_dirs]
    
    print(f"Found {len(all_paths)} samples")
    
    # Generate split
    np.random.seed(seed)
    indices = list(range(len(all_paths)))
    np.random.shuffle(indices)
    
    num_val = int(len(all_paths) * val_ratio)
    val_indices = set(indices[:num_val])
    
    train_paths = [all_paths[i] for i in range(len(all_paths)) if i not in val_indices]
    val_paths = [all_paths[i] for i in range(len(all_paths)) if i in val_indices]
    
    # Write split files
    with open(data_dir / "data_mouse_train.txt", 'w') as f:
        f.write('\n'.join(sorted(train_paths)))
    
    with open(data_dir / "data_mouse_val.txt", 'w') as f:
        f.write('\n'.join(sorted(val_paths)))
    
    # Update all file if not exists
    all_file = data_dir / "data_mouse_all.txt"
    if not all_file.exists():
        with open(all_file, 'w') as f:
            f.write('\n'.join(sorted(all_paths)))
    
    print(f"\nSplit generated (seed={seed}, val_ratio={val_ratio}):")
    print(f"  Train: {len(train_paths)}")
    print(f"  Val: {len(val_paths)}")
    print(f"  Files: data_mouse_train.txt, data_mouse_val.txt")
    
    # Update config if requested
    if update_config:
        print()
        update_dataset_config(update_config, data_dir, len(train_paths), len(val_paths))


def main():
    parser = argparse.ArgumentParser(description="Generate Train/Val Split")
    parser.add_argument('--data-dir', '-d', type=str, required=True, help="Preprocessed data directory")
    parser.add_argument('--val-ratio', '-v', type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument('--seed', '-s', type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument('--update-config', '-c', type=str, default=None, 
                        help="Dataset config name to update (e.g., D9)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return
    
    generate_split(data_dir, args.val_ratio, args.seed, args.update_config)


if __name__ == "__main__":
    main()
