#!/usr/bin/env python3
"""
Flexible Split Manager for FaceLift datasets.

Supports multiple split strategies:
- random: Simple random shuffle
- temporal: Time-based sequential split (train=early, val=middle, test=late)
- temporal_stratified: Sample from each time stratum (Pose Splatter style)

Usage:
    python -m mouse_extensions.preprocessing.split_manager \
        --data-dir /path/to/M3 \
        --config configs/splits/temporal_stratified.yaml
    
    python -m mouse_extensions.preprocessing.split_manager \
        --data-dir /path/to/M3 \
        --preset pose_splatter
"""

import argparse
import random
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class SplitConfig:
    name: str
    version_suffix: str
    strategy: str
    seed: Optional[int]
    splits: Dict[str, float]  # train, val, test ratios
    num_strata: int = 10
    exclude_frames: List[int] = None
    
    @classmethod
    def from_yaml(cls, path: Path) -> "SplitConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            version_suffix=data["version_suffix"],
            strategy=data["strategy"],
            seed=data.get("seed"),
            splits=data["splits"],
            num_strata=data.get("num_strata", 10),
            exclude_frames=data.get("exclude_frames", []),
        )


# Built-in presets
PRESETS = {
    "random": {
        "name": "random",
        "version_suffix": "r",
        "strategy": "random",
        "seed": 42,
        "splits": {"train": 0.9, "val": 0.1},
    },
    "pose_splatter": {
        "name": "temporal_stratified",
        "version_suffix": "s",
        "strategy": "temporal_stratified",
        "seed": 42,
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "num_strata": 10,
        "exclude_frames": [5900, 11800, 17700],
    },
    "temporal_3way": {
        "name": "temporal",
        "version_suffix": "t",
        "strategy": "temporal",
        "seed": None,
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        "exclude_frames": [5900, 11800, 17700],
    },
}


def get_sample_dirs(data_dir: Path) -> List[Path]:
    """Find all sample directories."""
    samples_dir = data_dir / "samples"
    if samples_dir.exists():
        dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    else:
        # Fallback: look in root
        dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    return dirs


def split_random(samples: List[Path], config: SplitConfig) -> Dict[str, List[Path]]:
    """Random shuffle split."""
    if config.seed is not None:
        random.seed(config.seed)
    
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    result = {}
    start = 0
    for split_name, ratio in config.splits.items():
        count = int(len(samples) * ratio)
        if split_name == list(config.splits.keys())[-1]:
            # Last split gets remainder
            split_indices = indices[start:]
        else:
            split_indices = indices[start:start + count]
        result[split_name] = [samples[i] for i in split_indices]
        start += count
    
    return result


def split_temporal(samples: List[Path], config: SplitConfig) -> Dict[str, List[Path]]:
    """Sequential temporal split (no shuffle)."""
    # Samples are already sorted by frame index
    result = {}
    start = 0
    for split_name, ratio in config.splits.items():
        count = int(len(samples) * ratio)
        if split_name == list(config.splits.keys())[-1]:
            result[split_name] = samples[start:]
        else:
            result[split_name] = samples[start:start + count]
        start += count
    
    return result


def split_temporal_stratified(samples: List[Path], config: SplitConfig) -> Dict[str, List[Path]]:
    """Temporal stratified split - sample from each time stratum."""
    num_strata = config.num_strata
    if config.seed is not None:
        random.seed(config.seed)
    
    # Divide into strata
    strata = []
    stratum_size = len(samples) // num_strata
    for i in range(num_strata):
        start = i * stratum_size
        end = start + stratum_size if i < num_strata - 1 else len(samples)
        strata.append(list(range(start, end)))
    
    result = {name: [] for name in config.splits.keys()}
    
    for stratum_indices in strata:
        random.shuffle(stratum_indices)
        start = 0
        for split_name, ratio in config.splits.items():
            count = max(1, int(len(stratum_indices) * ratio))
            if split_name == list(config.splits.keys())[-1]:
                split_indices = stratum_indices[start:]
            else:
                split_indices = stratum_indices[start:start + count]
            result[split_name].extend([samples[i] for i in split_indices])
            start += count
    
    return result


SPLIT_FUNCTIONS = {
    "random": split_random,
    "temporal": split_temporal,
    "temporal_stratified": split_temporal_stratified,
}


def generate_splits(data_dir: Path, config: SplitConfig) -> Dict[str, List[str]]:
    """Generate splits according to config."""
    samples = get_sample_dirs(data_dir)
    
    if not samples:
        raise ValueError(f"No samples found in {data_dir}")
    
    print(f"Found {len(samples)} samples")
    print(f"Strategy: {config.strategy}")
    print(f"Splits: {config.splits}")
    
    split_func = SPLIT_FUNCTIONS.get(config.strategy)
    if split_func is None:
        raise ValueError(f"Unknown strategy: {config.strategy}")
    
    splits = split_func(samples, config)
    
    # Convert to absolute paths with trailing slash
    result = {}
    for name, paths in splits.items():
        result[name] = [str(p.absolute()) + "/" for p in paths]
    
    return result


def save_splits(data_dir: Path, splits: Dict[str, List[str]], config: SplitConfig):
    """Save split text files."""
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    suffix = f"_{config.version_suffix}" if config.version_suffix else ""
    
    for name, paths in splits.items():
        # Save to splits/ directory
        split_file = splits_dir / f"{name}{suffix}.txt"
        with open(split_file, "w") as f:
            f.write("\n".join(paths))
        print(f"Created: {split_file} ({len(paths)} samples)")
        
        # Create symlink in root
        root_link = data_dir / f"data_mouse_{name}{suffix}.txt"
        root_link.unlink(missing_ok=True)
        root_link.symlink_to(split_file.relative_to(data_dir))
    
    # Save split config
    config_file = splits_dir / f"config{suffix}.yaml"
    with open(config_file, "w") as f:
        yaml.dump({
            "name": config.name,
            "version_suffix": config.version_suffix,
            "strategy": config.strategy,
            "seed": config.seed,
            "splits": config.splits,
            "num_strata": config.num_strata,
            "counts": {k: len(v) for k, v in splits.items()},
        }, f, default_flow_style=False)
    print(f"Config saved: {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Flexible Split Manager")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--config", type=Path, help="Split config YAML file")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use built-in preset")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available presets:")
        for name, cfg in PRESETS.items():
            print(f"  {name}: {cfg['strategy']} - {cfg['splits']}")  # Fixed f-string
        return
    
    if args.config:
        config = SplitConfig.from_yaml(args.config)
    elif args.preset:
        config = SplitConfig(**PRESETS[args.preset])
    else:
        parser.error("Either --config or --preset is required")
    
    splits = generate_splits(args.data_dir, config)
    save_splits(args.data_dir, splits, config)
    
    print(f"\nDataset version: {args.data_dir.name}.{config.version_suffix}")


if __name__ == "__main__":
    main()
