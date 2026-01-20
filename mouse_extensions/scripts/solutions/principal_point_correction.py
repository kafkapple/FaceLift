#!/usr/bin/env python3
"""
Solution 2: Principal Point Verification & Correction

This module provides functions to verify and correct principal point (cx, cy)
alignment issues that can cause ghosting artifacts in multi-view reconstruction.

The principal point is where the optical axis intersects the image plane.
When cx, cy are not correctly calibrated, rays are back-projected in wrong
directions, causing 3D reconstruction errors.

Key issues addressed:
1. Principal point offset from image center
2. Inconsistent cx, cy across views
3. Image centering that doesn't update cx, cy correctly

Author: AI Research Assistant
Date: 2026-01-17
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class CenteringMethod(Enum):
    """Methods for image centering"""
    PRINCIPAL_POINT = "principal_point"  # Center based on cx, cy (recommended)
    OBJECT_CENTROID = "object_centroid"   # Center based on object mask
    IMAGE_CENTER = "image_center"          # Assume cx, cy at image center


@dataclass
class PrincipalPointDiagnostic:
    """Diagnostic results for principal point analysis"""
    view_id: int
    cx: float
    cy: float
    image_center_x: float
    image_center_y: float
    offset_x: float  # cx - image_center_x
    offset_y: float  # cy - image_center_y
    offset_magnitude: float  # sqrt(offset_x^2 + offset_y^2)
    ray_angle_error: float  # Estimated ray direction error in degrees
    severity: str  # "low", "medium", "high"


@dataclass
class CorrectionResult:
    """Result of principal point correction"""
    images: torch.Tensor  # Corrected images
    intrinsics: torch.Tensor  # Corrected intrinsics
    crop_offsets: torch.Tensor  # Applied crop offsets (N, 2)
    original_intrinsics: torch.Tensor
    method_used: str


def diagnose_principal_point(
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int] = (512, 512),
    focal_length: Optional[float] = None
) -> List[PrincipalPointDiagnostic]:
    """
    Diagnose principal point alignment issues.

    Args:
        intrinsics: (N, 4) as [fx, fy, cx, cy] or (N, 3, 3) matrix
        image_size: (H, W) image dimensions
        focal_length: If provided, compute ray angle error

    Returns:
        List of diagnostic results per view
    """
    H, W = image_size
    img_center_x = W / 2
    img_center_y = H / 2

    diagnostics = []

    # Handle different intrinsics formats
    if intrinsics.dim() == 2 and intrinsics.shape[1] == 4:
        # (N, 4) format
        num_views = intrinsics.shape[0]
        for i in range(num_views):
            fx, fy, cx, cy = intrinsics[i].tolist()
            fx_use = focal_length or fx

            offset_x = cx - img_center_x
            offset_y = cy - img_center_y
            offset_mag = np.sqrt(offset_x**2 + offset_y**2)

            # Estimate ray angle error (in degrees)
            # tan(angle) ≈ offset / focal_length
            if fx_use > 0:
                ray_error = np.degrees(np.arctan(offset_mag / fx_use))
            else:
                ray_error = 0.0

            # Determine severity
            if offset_mag < 5:
                severity = "low"
            elif offset_mag < 20:
                severity = "medium"
            else:
                severity = "high"

            diagnostics.append(PrincipalPointDiagnostic(
                view_id=i,
                cx=cx,
                cy=cy,
                image_center_x=img_center_x,
                image_center_y=img_center_y,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_magnitude=offset_mag,
                ray_angle_error=ray_error,
                severity=severity
            ))

    elif intrinsics.dim() == 3:
        # (N, 3, 3) matrix format
        num_views = intrinsics.shape[0]
        for i in range(num_views):
            K = intrinsics[i]
            fx = K[0, 0].item()
            fy = K[1, 1].item()
            cx = K[0, 2].item()
            cy = K[1, 2].item()
            fx_use = focal_length or fx

            offset_x = cx - img_center_x
            offset_y = cy - img_center_y
            offset_mag = np.sqrt(offset_x**2 + offset_y**2)

            if fx_use > 0:
                ray_error = np.degrees(np.arctan(offset_mag / fx_use))
            else:
                ray_error = 0.0

            if offset_mag < 5:
                severity = "low"
            elif offset_mag < 20:
                severity = "medium"
            else:
                severity = "high"

            diagnostics.append(PrincipalPointDiagnostic(
                view_id=i,
                cx=cx,
                cy=cy,
                image_center_x=img_center_x,
                image_center_y=img_center_y,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_magnitude=offset_mag,
                ray_angle_error=ray_error,
                severity=severity
            ))

    return diagnostics


def correct_principal_point(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    method: str = "center_to_image",
    target_size: Optional[Tuple[int, int]] = None
) -> CorrectionResult:
    """
    Correct principal point alignment.

    Methods:
    1. "center_to_image": Set cx, cy to image center (simple, may lose accuracy)
    2. "crop_to_pp": Crop image so that cx, cy becomes centered
    3. "pad_to_center": Pad image so that cx, cy becomes centered

    Args:
        images: (N, C, H, W) or (N, H, W, C) images
        intrinsics: (N, 4) as [fx, fy, cx, cy]
        method: Correction method
        target_size: Optional target output size after correction

    Returns:
        CorrectionResult with corrected images and intrinsics
    """
    # Detect image format
    if images.dim() == 4:
        if images.shape[1] in [1, 3, 4]:  # (N, C, H, W)
            channel_first = True
            N, C, H, W = images.shape
        else:  # (N, H, W, C)
            channel_first = False
            N, H, W, C = images.shape
            images = images.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)
    else:
        raise ValueError(f"Expected 4D tensor, got {images.dim()}D")

    target_size = target_size or (H, W)
    target_H, target_W = target_size

    original_intrinsics = intrinsics.clone()
    corrected_intrinsics = intrinsics.clone()
    crop_offsets = torch.zeros(N, 2)

    if method == "center_to_image":
        # Simply set cx, cy to image center
        # This is fast but may introduce geometric error
        corrected_intrinsics[:, 2] = target_W / 2  # cx
        corrected_intrinsics[:, 3] = target_H / 2  # cy

        # Resize images if needed
        if (H, W) != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )

        corrected_images = images

    elif method == "crop_to_pp":
        # Crop image so that principal point becomes centered
        # More accurate but may lose parts of the image
        corrected_images = []

        for i in range(N):
            fx, fy, cx, cy = intrinsics[i].tolist()

            # Calculate crop region
            # We want the crop center to be at (cx, cy)
            # And the output cx, cy to be at target_W/2, target_H/2

            crop_x1 = int(cx - target_W / 2)
            crop_y1 = int(cy - target_H / 2)

            # Handle boundary cases with padding
            pad_left = max(0, -crop_x1)
            pad_top = max(0, -crop_y1)
            pad_right = max(0, crop_x1 + target_W - W)
            pad_bottom = max(0, crop_y1 + target_H - H)

            # Adjust crop coordinates
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(W, crop_x1 + target_W - pad_left - pad_right)
            crop_y2 = min(H, crop_y1 + target_H - pad_top - pad_bottom)

            # Crop
            img_cropped = images[i, :, crop_y1:crop_y2, crop_x1:crop_x2]

            # Pad if necessary
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                img_cropped = torch.nn.functional.pad(
                    img_cropped,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=1.0  # White background
                )

            corrected_images.append(img_cropped)
            crop_offsets[i] = torch.tensor([crop_x1 - pad_left, crop_y1 - pad_top])

        corrected_images = torch.stack(corrected_images)

        # Update intrinsics - cx, cy are now centered
        corrected_intrinsics[:, 2] = target_W / 2
        corrected_intrinsics[:, 3] = target_H / 2

    elif method == "pad_to_center":
        # Pad image so that principal point becomes centered
        # Preserves all image content but increases size
        corrected_images = []

        for i in range(N):
            fx, fy, cx, cy = intrinsics[i].tolist()

            # Calculate padding needed
            # Current cx is at cx, we want it at target_W/2
            pad_left = int(max(0, target_W / 2 - cx))
            pad_right = int(max(0, cx - (W - target_W / 2)))
            pad_top = int(max(0, target_H / 2 - cy))
            pad_bottom = int(max(0, cy - (H - target_H / 2)))

            # Apply padding
            img_padded = torch.nn.functional.pad(
                images[i],
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=1.0
            )

            # Crop to target size from the center
            new_H, new_W = img_padded.shape[-2:]
            start_y = (new_H - target_H) // 2
            start_x = (new_W - target_W) // 2
            img_final = img_padded[:, start_y:start_y+target_H, start_x:start_x+target_W]

            corrected_images.append(img_final)
            crop_offsets[i] = torch.tensor([pad_left - start_x, pad_top - start_y])

        corrected_images = torch.stack(corrected_images)
        corrected_intrinsics[:, 2] = target_W / 2
        corrected_intrinsics[:, 3] = target_H / 2

    else:
        raise ValueError(f"Unknown correction method: {method}")

    # Convert back to original format if needed
    if not channel_first:
        corrected_images = corrected_images.permute(0, 2, 3, 1)

    return CorrectionResult(
        images=corrected_images,
        intrinsics=corrected_intrinsics,
        crop_offsets=crop_offsets,
        original_intrinsics=original_intrinsics,
        method_used=method
    )


def verify_principal_point_consistency(
    intrinsics_list: List[torch.Tensor],
    tolerance: float = 5.0
) -> Dict:
    """
    Verify that principal points are consistent across samples/views.

    Args:
        intrinsics_list: List of (N, 4) intrinsics tensors
        tolerance: Maximum allowed deviation in pixels

    Returns:
        Dictionary with verification results
    """
    all_cx = []
    all_cy = []

    for intrinsics in intrinsics_list:
        all_cx.extend(intrinsics[:, 2].tolist())
        all_cy.extend(intrinsics[:, 3].tolist())

    cx_mean = np.mean(all_cx)
    cy_mean = np.mean(all_cy)
    cx_std = np.std(all_cx)
    cy_std = np.std(all_cy)
    cx_range = max(all_cx) - min(all_cx)
    cy_range = max(all_cy) - min(all_cy)

    is_consistent = cx_std < tolerance and cy_std < tolerance

    return {
        "is_consistent": is_consistent,
        "cx_mean": cx_mean,
        "cy_mean": cy_mean,
        "cx_std": cx_std,
        "cy_std": cy_std,
        "cx_range": cx_range,
        "cy_range": cy_range,
        "num_samples": len(intrinsics_list),
        "total_views": len(all_cx),
        "recommendation": "No action needed" if is_consistent else
                         f"Principal point varies significantly (std > {tolerance}px). "
                         "Consider standardizing cx, cy or using 'crop_to_pp' correction."
    }


class PrincipalPointCorrector:
    """
    Transform class for applying principal point correction in DataLoader.

    Usage:
        >>> corrector = PrincipalPointCorrector(method="crop_to_pp")
        >>> # In dataset __getitem__:
        >>> images, intrinsics = corrector(images, intrinsics)
    """

    def __init__(
        self,
        method: str = "crop_to_pp",
        target_size: Tuple[int, int] = (512, 512),
        enabled: bool = True
    ):
        self.method = method
        self.target_size = target_size
        self.enabled = enabled

    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply principal point correction"""
        if not self.enabled:
            return images, intrinsics

        result = correct_principal_point(
            images, intrinsics,
            method=self.method,
            target_size=self.target_size
        )
        return result.images, result.intrinsics


def print_diagnostic_report(diagnostics: List[PrincipalPointDiagnostic]):
    """Print formatted diagnostic report"""
    print("\n" + "="*70)
    print("PRINCIPAL POINT DIAGNOSTIC REPORT")
    print("="*70)

    high_severity = [d for d in diagnostics if d.severity == "high"]
    medium_severity = [d for d in diagnostics if d.severity == "medium"]

    print(f"\nTotal views analyzed: {len(diagnostics)}")
    print(f"High severity issues: {len(high_severity)}")
    print(f"Medium severity issues: {len(medium_severity)}")

    print("\n" + "-"*70)
    print(f"{'View':>5} | {'cx':>8} | {'cy':>8} | {'Offset':>8} | {'Ray Error':>10} | {'Severity':>8}")
    print("-"*70)

    for d in diagnostics:
        print(f"{d.view_id:>5} | {d.cx:>8.1f} | {d.cy:>8.1f} | "
              f"{d.offset_magnitude:>8.1f}px | {d.ray_angle_error:>9.2f}° | {d.severity:>8}")

    print("-"*70)

    if high_severity:
        print("\n⚠️  HIGH SEVERITY ISSUES DETECTED!")
        print("   Principal point offset > 20px can significantly degrade reconstruction.")
        print("   Recommendation: Use 'crop_to_pp' correction method.")
    elif medium_severity:
        print("\n⚡ Medium severity issues detected.")
        print("   Consider applying principal point correction for better results.")
    else:
        print("\n✅ Principal points are well-aligned.")


# Integration with preprocessing pipeline
PREPROCESSING_INTEGRATION = """
# Integration with FaceLift preprocessing pipeline

1. In preprocessing script, after image resizing:

```python
from principal_point_correction import (
    diagnose_principal_point,
    correct_principal_point,
    print_diagnostic_report
)

# Diagnose current state
diagnostics = diagnose_principal_point(intrinsics, image_size=(512, 512))
print_diagnostic_report(diagnostics)

# Apply correction if needed
if any(d.severity == "high" for d in diagnostics):
    result = correct_principal_point(
        images, intrinsics,
        method="crop_to_pp",
        target_size=(512, 512)
    )
    images = result.images
    intrinsics = result.intrinsics
```

2. Or use as DataLoader transform:

```python
from principal_point_correction import PrincipalPointCorrector

class MyDataset(Dataset):
    def __init__(self, ...):
        self.pp_corrector = PrincipalPointCorrector(method="crop_to_pp")

    def __getitem__(self, idx):
        images, intrinsics = load_data(idx)
        images, intrinsics = self.pp_corrector(images, intrinsics)
        return images, intrinsics
```
"""


if __name__ == "__main__":
    print("Testing Principal Point Correction...")

    # Create test data
    torch.manual_seed(42)
    num_views = 6

    # Simulated intrinsics with varying principal points (problematic case)
    intrinsics = torch.tensor([
        [549, 549, 192, 230],   # View 0: significant offset
        [549, 549, 415, 280],   # View 1: large offset (v5 bug example)
        [549, 549, 365, 260],   # View 2: medium offset
        [549, 549, 256, 256],   # View 3: perfect center
        [549, 549, 245, 248],   # View 4: small offset
        [549, 549, 300, 220],   # View 5: medium offset
    ], dtype=torch.float32)

    # Create dummy images
    images = torch.rand(num_views, 3, 512, 512)

    # Run diagnostics
    diagnostics = diagnose_principal_point(intrinsics, image_size=(512, 512))
    print_diagnostic_report(diagnostics)

    # Apply correction
    print("\n\nApplying 'crop_to_pp' correction...")
    result = correct_principal_point(images, intrinsics, method="crop_to_pp")

    print(f"\nOriginal intrinsics (cx, cy):")
    for i in range(num_views):
        print(f"  View {i}: cx={intrinsics[i, 2]:.1f}, cy={intrinsics[i, 3]:.1f}")

    print(f"\nCorrected intrinsics (cx, cy):")
    for i in range(num_views):
        print(f"  View {i}: cx={result.intrinsics[i, 2]:.1f}, cy={result.intrinsics[i, 3]:.1f}")

    print(f"\nCorrected image shape: {result.images.shape}")
    print(f"Method used: {result.method_used}")

    # Verify correction
    print("\nVerifying correction...")
    diagnostics_after = diagnose_principal_point(result.intrinsics, image_size=(512, 512))
    high_after = sum(1 for d in diagnostics_after if d.severity == "high")
    print(f"High severity issues after correction: {high_after}")

    print("\nTest completed successfully!")
