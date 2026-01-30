#!/usr/bin/env python3
"""
Split Generator for FaceLift Mouse Data
========================================

Generates train/val/test splits compatible with Pose Splatter evaluation protocol.

Pose Splatter Protocol (NeurIPS 2025):
- Temporal consecutive 1/3 splits (train:val:test = 1:1:1)
- 324K frames, 6cam, 30fps, 4x spatial / 5x temporal downsample
- Metrics: IoU, L1, PSNR, SSIM

Usage:
    python -m mouse_extensions.preprocessing.split_generator \
        --dataset_dir /path/to/M5 \
        --strategy temporal \
        --ratios 0.333 0.333 0.334

Created: 2026-01-29
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Literal
import random


def generate_split(
    dataset_dir: str,
    strategy: Literal["temporal", "random"] = "temporal",
    ratios: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    output_prefix: str = "data_mouse",
    seed: int = 42,
    holdout_views: List[int] = None,
) -> Dict[str, List[str]]:
    """
    Generate train/val/test splits for FaceLift mouse data.
    
    Args:
        dataset_dir: Path to preprocessed dataset (e.g., M5/)
        strategy: "temporal" (consecutive blocks) or "random" (shuffled)
        ratios: (train, val, test) ratios, must sum to 1.0
        output_prefix: Prefix for output files
        seed: Random seed for reproducibility
        holdout_views: Views to hold out for NVS evaluation (e.g., [5])
        
    Returns:
        Dictionary with split info and file paths
    """
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.exists(), f"Dataset directory not found: {dataset_dir}"
    
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"
    
    # Collect all sample directories (numeric names)
    samples = sorted([
        d.name for d in dataset_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ], key=int)  # Sort numerically for correct temporal order
    
    total = len(samples)
    print(f"Total samples: {total}")
    
    if strategy == "temporal":
        # Contiguous temporal blocks (Pose Splatter style)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        n_test = total - n_train - n_val
        
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
        
    elif strategy == "random":
        # Random shuffle with fixed seed
        random.seed(seed)
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        
        train_samples = sorted(shuffled[:n_train], key=int)
        val_samples = sorted(shuffled[n_train:n_train + n_val], key=int)
        test_samples = sorted(shuffled[n_train + n_val:], key=int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Print split info
    print(f"\nSplit ({strategy}, ratios={ratios}):")
    print(f"  Train: {len(train_samples)} samples ({train_samples[0]} - {train_samples[-1]})")
    print(f"  Val:   {len(val_samples)} samples ({val_samples[0]} - {val_samples[-1]})")
    print(f"  Test:  {len(test_samples)} samples ({test_samples[0]} - {test_samples[-1]})")
    
    # Write split files with absolute paths
    output_files = {}
    
    # Check if files exist (unless --force)
    for split_name in ["train", "val", "test"]:
        out_path = dataset_dir / f"{output_prefix}_{split_name}.txt"
        if out_path.exists():
            print(f"WARNING: {out_path} already exists!")
            # This is a library function, so we just warn and continue
            # CLI will handle --force flag
            
    for split_name, split_data in [("train", train_samples),
                                    ("val", val_samples),
                                    ("test", test_samples)]:
        out_path = dataset_dir / f"{output_prefix}_{split_name}.txt"
        
        # Backup if exists
        if out_path.exists():
            backup_path = dataset_dir / "splits_backup" / f"{output_prefix}_{split_name}_backup.txt"
            backup_path.parent.mkdir(exist_ok=True)
            import shutil
            shutil.copy(out_path, backup_path)
            print(f"Backed up existing {out_path} -> {backup_path}")
        
        with open(out_path, "w") as f:
            for s in split_data:
                f.write(f"{dataset_dir}/{s}/\n")
        output_files[split_name] = str(out_path)
        print(f"Wrote {out_path} ({len(split_data)} samples)")
    
    # Write split.json (Pose Splatter compatible format)
    split_info = {
        "strategy": strategy,
        "ratios": list(ratios),
        "seed": seed,
        "total_samples": total,
        "splits": {
            "train": {
                "count": len(train_samples),
                "range": [train_samples[0], train_samples[-1]] if train_samples else [],
                "samples": train_samples,
            },
            "val": {
                "count": len(val_samples),
                "range": [val_samples[0], val_samples[-1]] if val_samples else [],
                "samples": val_samples,
            },
            "test": {
                "count": len(test_samples),
                "range": [test_samples[0], test_samples[-1]] if test_samples else [],
                "samples": test_samples,
            },
        },
        "files": output_files,
    }
    
    # Add holdout view info if specified
    if holdout_views:
        split_info["views"] = {
            "observed": [i for i in range(6) if i not in holdout_views],
            "holdout": holdout_views,
        }
    
    split_json_path = dataset_dir / "split.json"
    with open(split_json_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nWrote {split_json_path}")
    
    return split_info


def generate_pose_splatter_split(dataset_dir: str) -> Dict[str, List[str]]:
    """
    Generate Pose Splatter-compatible 1:1:1 temporal split.
    
    This is a convenience function for exact Pose Splatter reproduction.
    """
    return generate_split(
        dataset_dir=dataset_dir,
        strategy="temporal",
        ratios=(1/3, 1/3, 1/3),
        output_prefix="data_mouse",
        seed=42,
        holdout_views=[5],  # Pose Splatter: 5 observed, 1 holdout
    )


def create_temporal_variant(
    source_dir: str,
    variant_name: str = None,
    ratios: Tuple[float, float, float] = (1/3, 1/3, 1/3),
    holdout_views: List[int] = None,
    strategy: str = "temporal",
    seed: int = 42,
) -> Dict[str, any]:

    """
    Create a symlink-based temporal split variant of an existing dataset.

    This is the recommended way to create Pose Splatter-compatible datasets
    without duplicating data.

    Args:
        source_dir: Path to source dataset (e.g., M0/)
        variant_name: Name for variant (default: source + 't', e.g., M0 → M0t)
        ratios: Train/val/test ratios (default: 1:1:1)
        holdout_views: Views to hold out for NVS evaluation

    Returns:
        Dictionary with variant info

    Example:
        # Creates M0t/ with symlinks to M0/ and temporal 1:1:1 split
        create_temporal_variant("/path/to/M0")
    """
    source_dir = Path(source_dir)
    assert source_dir.exists(), f"Source directory not found: {source_dir}"

    # Determine variant name and directory
    if variant_name is None:
        variant_name = source_dir.name + "t"
    variant_dir = source_dir.parent / variant_name

    if variant_dir.exists():
        print(f"⚠️  Variant directory already exists: {variant_dir}")
        print("    Use --force to overwrite or choose a different name")
        return None

    # Collect sample directories from source
    samples = sorted([
        d.name for d in source_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ], key=int)

    total = len(samples)
    print(f"Source: {source_dir}")
    print(f"Variant: {variant_dir}")
    print(f"Total samples: {total}")

    # Create variant directory
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Create symlinks for all sample directories
    print(f"Creating {total} symlinks...")
    for sample in samples:
        src = source_dir / sample
        dst = variant_dir / sample
        dst.symlink_to(src)

    # Generate temporal split
    # Generate split based on strategy
    train_ratio, val_ratio, test_ratio = ratios
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    if strategy == "temporal":
        # Contiguous temporal blocks (Pose Splatter style)
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
    elif strategy == "random":
        # Random shuffle with fixed seed
        import random as rnd
        rnd.seed(seed)
        shuffled = samples.copy()
        rnd.shuffle(shuffled)
        train_samples = sorted(shuffled[:n_train], key=int)
        val_samples = sorted(shuffled[n_train:n_train + n_val], key=int)
        test_samples = sorted(shuffled[n_train + n_val:], key=int)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"Split ({strategy}): train={n_train}, val={n_val}, test={n_test}")
    for split_name, split_samples in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
        ("all", samples),
    ]:
        out_path = variant_dir / f"data_mouse_{split_name}.txt"
        with open(out_path, "w") as f:
            for s in split_samples:
                f.write(f"{variant_dir}/{s}/\n")
        output_files[split_name] = str(out_path)
        print(f"  Wrote {out_path.name} ({len(split_samples)} samples)")

    # Create split.json with metadata
    split_info = {
        "source": str(source_dir),
        "variant": str(variant_dir),
        "strategy": strategy,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {
            "train": {
                "count": len(train_samples),
                "range": [train_samples[0], train_samples[-1]] if train_samples else [],
            },
            "val": {
                "count": len(val_samples),
                "range": [val_samples[0], val_samples[-1]] if val_samples else [],
            },
            "test": {
                "count": len(test_samples),
                "range": [test_samples[0], test_samples[-1]] if test_samples else [],
            },
        },
        "files": output_files,
        "symlinked": True,
    }

    if holdout_views:
        split_info["views"] = {
            "observed": [i for i in range(6) if i not in holdout_views],
            "holdout": holdout_views,
        }

    split_json_path = variant_dir / "split.json"
    with open(split_json_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Wrote split.json")

    print(f"\n✅ Created {strategy} split variant: {variant_dir}")
    return split_info

def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test splits for FaceLift mouse data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pose Splatter style (1:1:1 temporal)
    python -m mouse_extensions.preprocessing.split_generator \\
        --dataset_dir /path/to/preprocessed/M5 \\
        --strategy temporal --ratios 0.333 0.333 0.334
    
    # Traditional 80/10/10 random split
    python -m mouse_extensions.preprocessing.split_generator \\
        --dataset_dir /path/to/preprocessed/M5 \\
        --strategy random --ratios 0.8 0.1 0.1
        
    # With holdout view for NVS evaluation
    python -m mouse_extensions.preprocessing.split_generator \\
        --dataset_dir /path/to/preprocessed/M5 \\
        --strategy temporal --ratios 0.333 0.333 0.334 \\
        --holdout_views 5
""")
    
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to preprocessed dataset directory")
    parser.add_argument("--strategy", type=str, default="temporal",
                        choices=["temporal", "random"],
                        help="Split strategy (default: temporal)")
    parser.add_argument("--ratios", type=float, nargs=3, 
                        default=[1/3, 1/3, 1/3],
                        help="Train/val/test ratios (default: 0.333 0.333 0.334)")
    parser.add_argument("--output_prefix", type=str, default="data_mouse",
                        help="Prefix for output files (default: data_mouse)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--holdout_views", type=int, nargs="*", default=None,
                        help="Views to hold out for NVS evaluation")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing split files (default: error if exists)")
    parser.add_argument("--backup", action="store_true", default=True,
                        help="Backup existing split files before overwriting (default: True)")

    parser.add_argument("--create-variant", type=str, metavar="NAME",
                        help="Create symlink-based temporal variant (e.g., M0t)")
    
    args = parser.parse_args()
    

    # Handle --create-variant option
    if args.create_variant:
        create_temporal_variant(
            source_dir=args.dataset_dir,
            variant_name=args.create_variant,
            ratios=tuple(args.ratios),
            holdout_views=args.holdout_views,
        )
        return

    generate_split(
        dataset_dir=args.dataset_dir,
        strategy=args.strategy,
        ratios=tuple(args.ratios),
        output_prefix=args.output_prefix,
        seed=args.seed,
        holdout_views=args.holdout_views,
    )


if __name__ == "__main__":
    main()
