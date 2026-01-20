#!/usr/bin/env python3
"""
Temporal Split Utilities for Mouse Dataset
==========================================

PoseSplatter-style temporal split for time-series video data.
Prevents data leakage by strictly separating train/val/test by time.

Split Methods:
- 'random': Random shuffle (FaceLift default, for interpolation evaluation)
- 'temporal': Chronological split (PoseSplatter style, for generalization evaluation)

Created: 2026-01-19
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple
import numpy as np


class SplitMethod(Enum):
    """Available split methods."""
    RANDOM = "random"
    TEMPORAL = "temporal"


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    method: SplitMethod = SplitMethod.TEMPORAL
    train_ratio: float = 0.34  # ~1/3
    val_ratio: float = 0.33    # ~1/3
    test_ratio: float = 0.33   # ~1/3
    random_seed: int = 42

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class SplitResult:
    """Result of dataset splitting."""
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    config: SplitConfig

    @property
    def train_set(self) -> Set[int]:
        return set(self.train_indices)

    @property
    def val_set(self) -> Set[int]:
        return set(self.val_indices)

    @property
    def test_set(self) -> Set[int]:
        return set(self.test_indices)

    def get_split(self, index: int) -> str:
        """Get split name for a given index."""
        if index in self.train_set:
            return "train"
        elif index in self.val_set:
            return "val"
        elif index in self.test_set:
            return "test"
        else:
            raise ValueError(f"Index {index} not in any split")

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Split Summary ({self.config.method.value}):\n"
            f"  Train: {len(self.train_indices)} samples "
            f"({len(self.train_indices) / (len(self.train_indices) + len(self.val_indices) + len(self.test_indices)) * 100:.1f}%)\n"
            f"  Val:   {len(self.val_indices)} samples "
            f"({len(self.val_indices) / (len(self.train_indices) + len(self.val_indices) + len(self.test_indices)) * 100:.1f}%)\n"
            f"  Test:  {len(self.test_indices)} samples "
            f"({len(self.test_indices) / (len(self.train_indices) + len(self.val_indices) + len(self.test_indices)) * 100:.1f}%)"
        )


def split_temporal(
    frame_indices: List[int],
    config: SplitConfig
) -> SplitResult:
    """
    Split frames temporally (chronologically).

    PoseSplatter style: first 1/3 train, middle 1/3 val, last 1/3 test.
    This ensures train/val/test are from completely different time periods,
    preventing data leakage from temporally adjacent frames.

    Args:
        frame_indices: List of frame indices (assumed to be in temporal order)
        config: Split configuration

    Returns:
        SplitResult with train/val/test indices
    """
    n = len(frame_indices)

    # Ensure indices are sorted (temporal order)
    sorted_indices = sorted(frame_indices)

    # Calculate split points
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    train_indices = sorted_indices[:train_end]
    val_indices = sorted_indices[train_end:val_end]
    test_indices = sorted_indices[val_end:]

    return SplitResult(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=config
    )


def split_random(
    frame_indices: List[int],
    config: SplitConfig
) -> SplitResult:
    """
    Split frames randomly (FaceLift default style).

    Warning: For temporal data, this can cause data leakage as
    temporally adjacent frames may end up in different splits.

    Args:
        frame_indices: List of frame indices
        config: Split configuration

    Returns:
        SplitResult with train/val/test indices
    """
    n = len(frame_indices)

    np.random.seed(config.random_seed)
    shuffled = frame_indices.copy()
    np.random.shuffle(shuffled)

    # Calculate split points
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))

    train_indices = sorted(shuffled[:train_end])
    val_indices = sorted(shuffled[train_end:val_end])
    test_indices = sorted(shuffled[val_end:])

    return SplitResult(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=config
    )


def split_frames(
    frame_indices: List[int],
    method: str = "temporal",
    train_ratio: float = 0.34,
    val_ratio: float = 0.33,
    test_ratio: float = 0.33,
    random_seed: int = 42
) -> SplitResult:
    """
    Main entry point for splitting frames.

    Args:
        frame_indices: List of frame indices
        method: 'temporal' or 'random'
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed (only used for 'random' method)

    Returns:
        SplitResult with train/val/test indices

    Example:
        >>> indices = list(range(3600))
        >>> result = split_frames(indices, method='temporal')
        >>> print(result.summary())
        Split Summary (temporal):
          Train: 1224 samples (34.0%)
          Val:   1188 samples (33.0%)
          Test:  1188 samples (33.0%)
    """
    config = SplitConfig(
        method=SplitMethod(method),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    if config.method == SplitMethod.TEMPORAL:
        return split_temporal(frame_indices, config)
    else:
        return split_random(frame_indices, config)


def get_legacy_split(
    frame_indices: List[int],
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Set[int], Set[int]]:
    """
    Legacy FaceLift split (train/val only, no test set).

    For backward compatibility with existing code.

    Args:
        frame_indices: List of frame indices
        val_ratio: Fraction for validation (default 0.1)
        random_seed: Random seed

    Returns:
        (train_set, val_set) tuple
    """
    result = split_frames(
        frame_indices,
        method="random",
        train_ratio=1.0 - val_ratio,
        val_ratio=val_ratio,
        test_ratio=0.0,
        random_seed=random_seed
    )
    return result.train_set, result.val_set


# Comparison visualization
def visualize_split_comparison(frame_indices: List[int], output_path: str = None):
    """
    Visualize the difference between random and temporal splits.

    Args:
        frame_indices: List of frame indices
        output_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    random_result = split_frames(frame_indices, method="random")
    temporal_result = split_frames(frame_indices, method="temporal")

    fig, axes = plt.subplots(2, 1, figsize=(14, 4))

    for ax, (name, result) in zip(axes, [
        ("Random Split (FaceLift)", random_result),
        ("Temporal Split (PoseSplatter)", temporal_result)
    ]):
        # Create color array
        colors = []
        for idx in sorted(frame_indices):
            if idx in result.train_set:
                colors.append('blue')
            elif idx in result.val_set:
                colors.append('orange')
            else:
                colors.append('green')

        ax.scatter(sorted(frame_indices), [1] * len(frame_indices),
                   c=colors, s=2, alpha=0.5)
        ax.set_xlim(min(frame_indices), max(frame_indices))
        ax.set_ylim(0.5, 1.5)
        ax.set_yticks([])
        ax.set_title(f"{name} - Train: {len(result.train_indices)}, "
                     f"Val: {len(result.val_indices)}, Test: {len(result.test_indices)}")
        ax.set_xlabel("Frame Index (Time)")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Train'),
            Patch(facecolor='orange', label='Val'),
            Patch(facecolor='green', label='Test')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Temporal Split Utilities Demo")
    print("=" * 60)

    # Simulate mouse dataset (~3600 frames after interval=5)
    frame_indices = list(range(0, 18000, 5))  # Every 5th frame
    print(f"\nTotal frames: {len(frame_indices)}")

    # Compare splits
    print("\n--- Random Split (FaceLift default) ---")
    random_result = split_frames(frame_indices, method="random")
    print(random_result.summary())
    print(f"Train frame range: {min(random_result.train_indices)} - {max(random_result.train_indices)}")
    print(f"Val frame range:   {min(random_result.val_indices)} - {max(random_result.val_indices)}")
    print(f"Test frame range:  {min(random_result.test_indices)} - {max(random_result.test_indices)}")

    print("\n--- Temporal Split (PoseSplatter style) ---")
    temporal_result = split_frames(frame_indices, method="temporal")
    print(temporal_result.summary())
    print(f"Train frame range: {min(temporal_result.train_indices)} - {max(temporal_result.train_indices)}")
    print(f"Val frame range:   {min(temporal_result.val_indices)} - {max(temporal_result.val_indices)}")
    print(f"Test frame range:  {min(temporal_result.test_indices)} - {max(temporal_result.test_indices)}")

    # Key difference
    print("\n--- Key Difference ---")
    print("Random: Train/Val/Test are MIXED across all time periods")
    print("Temporal: Train=early, Val=middle, Test=late (STRICT separation)")
