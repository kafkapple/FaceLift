#!/usr/bin/env python3
"""
Solution 1: Per-Sample Camera Normalization

This module provides functions to normalize camera parameters on a per-sample basis,
ensuring consistent scale across all views within a sample.

The key insight is that in multi-view reconstruction, scale ambiguity can cause
"ghosting" artifacts where objects from different views don't align properly.
By normalizing camera positions to a consistent scale, we ensure that the
projected object size is consistent across views.

Integration:
    This module can be integrated into:
    1. Dataset preprocessing (offline normalization)
    2. DataLoader (online normalization during training)
    3. Model forward pass (just-in-time normalization)

Author: AI Research Assistant
Date: 2026-01-17
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union, Literal
from dataclasses import dataclass


@dataclass
class NormalizationResult:
    """Result of camera normalization"""
    c2w_normalized: torch.Tensor  # (N, 4, 4) normalized c2w matrices
    scale_factor: float  # Applied scale factor
    centroid: torch.Tensor  # (3,) centroid used for centering
    original_distances: torch.Tensor  # (N,) original distances
    normalized_distances: torch.Tensor  # (N,) distances after normalization


def normalize_cameras_per_sample(
    c2w: torch.Tensor,
    intrinsics: Optional[torch.Tensor] = None,
    method: Literal["centroid", "reference", "mean_distance", "target_distance"] = "centroid",
    target_distance: float = 2.7,
    reference_view: int = 0,
    return_scale: bool = False
) -> Union[torch.Tensor, NormalizationResult]:
    """
    Normalize camera positions for a single sample to ensure consistent scale.

    This function addresses scale ambiguity by normalizing all camera positions
    relative to a common reference (centroid, reference view, or target distance).

    Args:
        c2w: Camera-to-world matrices, shape (N, 4, 4) or (B, N, 4, 4)
        intrinsics: Optional intrinsics (N, 4) as [fx, fy, cx, cy]. Not modified.
        method: Normalization method
            - "centroid": Normalize relative to camera centroid
            - "reference": Normalize relative to reference view distance
            - "mean_distance": Normalize to mean camera distance
            - "target_distance": Scale all cameras to target distance
        target_distance: Target distance (used when method="target_distance")
        reference_view: Reference view index (used when method="reference")
        return_scale: If True, return NormalizationResult instead of just matrices

    Returns:
        If return_scale=False: Normalized c2w matrices (same shape as input)
        If return_scale=True: NormalizationResult with detailed information

    Example:
        >>> c2w = torch.randn(6, 4, 4)  # 6 views
        >>> c2w_norm = normalize_cameras_per_sample(c2w, method="centroid")
        >>> # Or with detailed output:
        >>> result = normalize_cameras_per_sample(c2w, return_scale=True)
        >>> print(f"Scale factor: {result.scale_factor}")
    """
    # Handle batched input
    if c2w.dim() == 4:
        # Batched: (B, N, 4, 4)
        batch_size = c2w.shape[0]
        results = []
        for b in range(batch_size):
            result = normalize_cameras_per_sample(
                c2w[b], intrinsics[b] if intrinsics is not None else None,
                method=method, target_distance=target_distance,
                reference_view=reference_view, return_scale=True
            )
            results.append(result)

        if return_scale:
            # Return list of results for batch
            return results
        else:
            return torch.stack([r.c2w_normalized for r in results])

    # Single sample: (N, 4, 4)
    assert c2w.dim() == 3 and c2w.shape[1:] == (4, 4), \
        f"Expected shape (N, 4, 4), got {c2w.shape}"

    num_views = c2w.shape[0]
    device = c2w.device
    dtype = c2w.dtype

    # Extract camera positions (translation component of c2w)
    positions = c2w[:, :3, 3].clone()  # (N, 3)

    # Calculate original distances from origin
    original_distances = torch.norm(positions, dim=1)  # (N,)

    # Compute centroid
    centroid = positions.mean(dim=0)  # (3,)

    # Normalize based on method
    if method == "centroid":
        # Center cameras around centroid, scale by mean distance from centroid
        centered_positions = positions - centroid
        distances_from_centroid = torch.norm(centered_positions, dim=1)
        scale_factor = distances_from_centroid.mean().item()

        if scale_factor > 1e-6:
            normalized_positions = centered_positions / scale_factor
        else:
            normalized_positions = centered_positions

    elif method == "reference":
        # Scale relative to reference view
        ref_distance = original_distances[reference_view].item()

        if ref_distance > 1e-6:
            scale_factor = ref_distance
            normalized_positions = positions / scale_factor
        else:
            scale_factor = 1.0
            normalized_positions = positions

    elif method == "mean_distance":
        # Scale to unit mean distance
        mean_distance = original_distances.mean().item()

        if mean_distance > 1e-6:
            scale_factor = mean_distance
            normalized_positions = positions / scale_factor
        else:
            scale_factor = 1.0
            normalized_positions = positions

    elif method == "target_distance":
        # Scale all cameras to have mean distance = target_distance
        mean_distance = original_distances.mean().item()

        if mean_distance > 1e-6:
            scale_factor = mean_distance / target_distance
            normalized_positions = positions / scale_factor
        else:
            scale_factor = 1.0
            normalized_positions = positions

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Construct normalized c2w matrices
    c2w_normalized = c2w.clone()
    c2w_normalized[:, :3, 3] = normalized_positions

    # Calculate normalized distances
    normalized_distances = torch.norm(normalized_positions, dim=1)

    if return_scale:
        return NormalizationResult(
            c2w_normalized=c2w_normalized,
            scale_factor=scale_factor,
            centroid=centroid,
            original_distances=original_distances,
            normalized_distances=normalized_distances
        )
    else:
        return c2w_normalized


def normalize_intrinsics_for_scale(
    intrinsics: torch.Tensor,
    scale_factor: float,
    normalize_focal: bool = True
) -> torch.Tensor:
    """
    Optionally adjust intrinsics to match camera scale normalization.

    When camera distances are scaled, the focal length can be adjusted
    to maintain the same projected object size.

    Note: In most cases, you should NOT adjust intrinsics after spatial
    normalization, as the goal is to normalize the 3D geometry, not the
    projection. This function is provided for completeness.

    Args:
        intrinsics: (N, 4) as [fx, fy, cx, cy] or (N, 3, 3) matrix
        scale_factor: Scale factor applied to camera positions
        normalize_focal: If True, scale focal length inversely

    Returns:
        Adjusted intrinsics (same shape as input)
    """
    if not normalize_focal:
        return intrinsics

    intrinsics_out = intrinsics.clone()

    if intrinsics.dim() == 2 and intrinsics.shape[1] == 4:
        # (N, 4) format: [fx, fy, cx, cy]
        intrinsics_out[:, 0] = intrinsics[:, 0] / scale_factor  # fx
        intrinsics_out[:, 1] = intrinsics[:, 1] / scale_factor  # fy
    elif intrinsics.dim() == 3 and intrinsics.shape[1:] == (3, 3):
        # (N, 3, 3) matrix format
        intrinsics_out[:, 0, 0] = intrinsics[:, 0, 0] / scale_factor  # fx
        intrinsics_out[:, 1, 1] = intrinsics[:, 1, 1] / scale_factor  # fy
    else:
        raise ValueError(f"Unknown intrinsics format: {intrinsics.shape}")

    return intrinsics_out


def compute_scale_consistency_metric(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """
    Compute scale consistency metric across views.

    The metric measures how consistent the "fx / distance" ratio is across views.
    A low standard deviation indicates good scale consistency.

    Args:
        c2w: Camera-to-world matrices (N, 4, 4)
        intrinsics: Intrinsics (N, 4) as [fx, fy, cx, cy]

    Returns:
        Tuple of:
            - relative_std: Standard deviation of fx/dist ratios divided by mean
            - ratios: Individual fx/dist ratios per view (N,)
    """
    positions = c2w[:, :3, 3]
    distances = torch.norm(positions, dim=1)  # (N,)

    fx = intrinsics[:, 0]  # (N,)

    # Avoid division by zero
    distances = torch.clamp(distances, min=1e-6)

    ratios = fx / distances  # (N,)

    mean_ratio = ratios.mean()
    std_ratio = ratios.std()

    relative_std = (std_ratio / mean_ratio).item() if mean_ratio > 1e-6 else 0.0

    return relative_std, ratios


class CameraNormalizer:
    """
    Camera normalizer that can be used as a transform in DataLoader.

    Example usage in dataset:
        >>> normalizer = CameraNormalizer(method="target_distance", target=2.7)
        >>> # In dataset __getitem__:
        >>> c2w_normalized = normalizer(c2w)
    """

    def __init__(
        self,
        method: str = "target_distance",
        target_distance: float = 2.7,
        reference_view: int = 0,
        normalize_intrinsics: bool = False
    ):
        """
        Initialize camera normalizer.

        Args:
            method: Normalization method (centroid, reference, mean_distance, target_distance)
            target_distance: Target distance for target_distance method
            reference_view: Reference view for reference method
            normalize_intrinsics: Whether to also normalize intrinsics (usually False)
        """
        self.method = method
        self.target_distance = target_distance
        self.reference_view = reference_view
        self.normalize_intrinsics = normalize_intrinsics

    def __call__(
        self,
        c2w: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Normalize camera parameters.

        Args:
            c2w: Camera-to-world matrices (N, 4, 4) or (B, N, 4, 4)
            intrinsics: Optional intrinsics (N, 4) or (B, N, 4)

        Returns:
            Normalized c2w, or tuple (c2w, intrinsics) if intrinsics provided
        """
        result = normalize_cameras_per_sample(
            c2w,
            intrinsics=intrinsics,
            method=self.method,
            target_distance=self.target_distance,
            reference_view=self.reference_view,
            return_scale=True
        )

        if isinstance(result, list):
            # Batched
            c2w_norm = torch.stack([r.c2w_normalized for r in result])
            if intrinsics is not None and self.normalize_intrinsics:
                int_norm = torch.stack([
                    normalize_intrinsics_for_scale(intrinsics[i], r.scale_factor)
                    for i, r in enumerate(result)
                ])
                return c2w_norm, int_norm
            return c2w_norm
        else:
            # Single sample
            if intrinsics is not None and self.normalize_intrinsics:
                int_norm = normalize_intrinsics_for_scale(intrinsics, result.scale_factor)
                return result.c2w_normalized, int_norm
            return result.c2w_normalized


# Integration helper for GS-LRM dataset
def integrate_with_mouse_dataset(dataset_class):
    """
    Decorator to add camera normalization to existing dataset class.

    Usage:
        @integrate_with_mouse_dataset
        class MouseViewDataset:
            ...

    Or manually:
        MouseViewDataset = integrate_with_mouse_dataset(MouseViewDataset)
    """
    original_getitem = dataset_class.__getitem__

    def new_getitem(self, idx):
        sample = original_getitem(self, idx)

        # Check if normalization is enabled
        if hasattr(self, 'camera_normalizer') and self.camera_normalizer is not None:
            if 'c2w' in sample:
                c2w = sample['c2w']
                if isinstance(c2w, np.ndarray):
                    c2w = torch.from_numpy(c2w)

                c2w_norm = self.camera_normalizer(c2w)

                if isinstance(c2w_norm, np.ndarray):
                    sample['c2w'] = c2w_norm
                else:
                    sample['c2w'] = c2w_norm.numpy() if isinstance(sample['c2w'], np.ndarray) else c2w_norm

        return sample

    dataset_class.__getitem__ = new_getitem
    return dataset_class


# Example config integration
CAMERA_NORMALIZATION_CONFIG = """
# Add to training config YAML:

training:
  dataset:
    # Existing settings...

    # Camera normalization settings (NEW)
    camera_normalization:
      enabled: true
      method: "target_distance"  # centroid | reference | mean_distance | target_distance
      target_distance: 2.7
      reference_view: 0
      normalize_intrinsics: false
"""


if __name__ == "__main__":
    # Test the normalization
    print("Testing camera normalization...")

    # Create test cameras with varying distances
    torch.manual_seed(42)
    num_views = 6

    # Simulate cameras at different distances
    distances = torch.tensor([2.0, 3.5, 2.8, 3.2, 2.5, 4.0])
    directions = torch.randn(num_views, 3)
    directions = directions / directions.norm(dim=1, keepdim=True)

    c2w = torch.eye(4).unsqueeze(0).repeat(num_views, 1, 1)
    c2w[:, :3, 3] = directions * distances.unsqueeze(1)

    intrinsics = torch.tensor([[549, 549, 256, 256]] * num_views, dtype=torch.float32)

    print(f"\nOriginal distances: {distances.tolist()}")
    print(f"Distance std: {distances.std().item():.3f}")

    # Test different methods
    for method in ["centroid", "mean_distance", "target_distance"]:
        result = normalize_cameras_per_sample(
            c2w, intrinsics, method=method, target_distance=2.7, return_scale=True
        )
        print(f"\nMethod: {method}")
        print(f"  Scale factor: {result.scale_factor:.3f}")
        print(f"  Normalized distances: {result.normalized_distances.tolist()}")
        print(f"  Distance std after: {result.normalized_distances.std().item():.3f}")

    # Test scale consistency metric
    rel_std, ratios = compute_scale_consistency_metric(c2w, intrinsics)
    print(f"\nScale consistency (before): relative std = {rel_std*100:.2f}%")

    c2w_norm = normalize_cameras_per_sample(c2w, method="target_distance", target_distance=2.7)
    rel_std_after, _ = compute_scale_consistency_metric(c2w_norm, intrinsics)
    print(f"Scale consistency (after): relative std = {rel_std_after*100:.2f}%")

    print("\nTest completed successfully!")
