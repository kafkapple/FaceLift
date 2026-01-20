#!/usr/bin/env python3
"""
Principal Point Correction Integration for FaceLift Mouse Dataset

This module provides functions to correct the cx,cy bug in v12/v13 datasets
during data loading (runtime correction).

Problem:
- v12/v13 datasets use object centering (centroid-based)
- After centering, cx=cy=256 is FORCED even though actual PP differs
- Actual PP values are stored in _transform.scaled_cx/scaled_cy
- This causes 11-13° ray direction errors → ghosting artifacts

Solution:
- Read actual PP from _transform metadata
- Either use actual PP values OR crop image to center actual PP

Integration:
- Add to mouse_dataset.py's __getitem__ method
- Or use as a config flag: `use_actual_pp: true`

Author: AI Research Assistant
Date: 2026-01-17
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from PIL import Image


def get_actual_principal_point(camera: Dict) -> Tuple[float, float]:
    """
    Extract actual principal point from camera JSON.

    The actual PP is stored in _transform.scaled_cx/scaled_cy after
    object centering preprocessing.

    Args:
        camera: Camera frame dict from opencv_cameras.json

    Returns:
        Tuple of (actual_cx, actual_cy)
    """
    transform = camera.get("_transform", {})
    actual_cx = transform.get("scaled_cx", camera.get("cx", 256.0))
    actual_cy = transform.get("scaled_cy", camera.get("cy", 256.0))
    return float(actual_cx), float(actual_cy)


def correct_intrinsics_to_actual_pp(
    camera: Dict,
    resize_ratio: float = 1.0
) -> np.ndarray:
    """
    Get corrected intrinsics using actual PP values.

    Args:
        camera: Camera frame dict
        resize_ratio: Resize ratio applied to image

    Returns:
        np.ndarray: [fx, fy, actual_cx, actual_cy] scaled by resize_ratio
    """
    fx = camera.get("fx", 549.0)
    fy = camera.get("fy", 549.0)
    actual_cx, actual_cy = get_actual_principal_point(camera)

    intrinsics = np.array([fx, fy, actual_cx, actual_cy], dtype=np.float32)
    intrinsics *= resize_ratio

    return intrinsics


def crop_image_to_pp(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    actual_cx: float,
    actual_cy: float,
    target_cx: float = 256.0,
    target_cy: float = 256.0,
    target_size: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Crop/shift image so that actual PP becomes target PP (center).

    This is geometrically equivalent to re-centering based on PP instead of object.

    IMPORTANT: For RGBA images, the alpha channel of padded areas is set to 0
    (transparent/background) to ensure correct mask handling.

    Args:
        image: Input image (PIL, numpy HWC, or torch CHW)
        actual_cx, actual_cy: Actual principal point position
        target_cx, target_cy: Target PP position (usually 256)
        target_size: Output image size
        bg_color: Background color for RGB padding (alpha is always 0 for bg)

    Returns:
        Cropped/padded image with same type as input
    """
    # Convert to numpy for processing
    input_type = type(image)
    if isinstance(image, Image.Image):
        img_np = np.array(image)
        is_pil = True
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
            # CHW format
            img_np = image.permute(1, 2, 0).numpy()
            was_chw = True
        else:
            img_np = image.numpy()
            was_chw = False
        is_pil = False
    else:
        img_np = image.copy()
        is_pil = False
        was_chw = False

    H, W = img_np.shape[:2]
    C = img_np.shape[2] if img_np.ndim == 3 else 1

    # Calculate shift needed
    shift_x = int(target_cx - actual_cx)
    shift_y = int(target_cy - actual_cy)

    # Create output canvas with proper background
    if img_np.ndim == 3:
        output = np.zeros((target_size, target_size, C), dtype=img_np.dtype)
        if C == 4:
            # RGBA: RGB=white (bg_color), Alpha=0 (transparent/background)
            output[:, :, 0] = bg_color[0] if len(bg_color) > 0 else 255
            output[:, :, 1] = bg_color[1] if len(bg_color) > 1 else 255
            output[:, :, 2] = bg_color[2] if len(bg_color) > 2 else 255
            output[:, :, 3] = 0  # Alpha=0 for background (CRITICAL for mask)
        elif C == 3:
            # RGB: just use bg_color
            output[:, :, 0] = bg_color[0] if len(bg_color) > 0 else 255
            output[:, :, 1] = bg_color[1] if len(bg_color) > 1 else 255
            output[:, :, 2] = bg_color[2] if len(bg_color) > 2 else 255
        else:
            output.fill(255)
    else:
        output = np.full((target_size, target_size), 255, dtype=img_np.dtype)

    # Calculate source and destination regions
    src_x1 = max(0, -shift_x)
    src_y1 = max(0, -shift_y)
    src_x2 = min(W, target_size - shift_x)
    src_y2 = min(H, target_size - shift_y)

    dst_x1 = max(0, shift_x)
    dst_y1 = max(0, shift_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Ensure valid regions
    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]

    # Convert back to original type
    if is_pil:
        return Image.fromarray(output)
    elif isinstance(image, torch.Tensor):
        output_tensor = torch.from_numpy(output)
        if was_chw:
            output_tensor = output_tensor.permute(2, 0, 1)
        return output_tensor
    else:
        return output


def apply_pp_correction(
    image: Union[Image.Image, np.ndarray],
    camera: Dict,
    method: str = "crop",
    resize_ratio: float = 1.0,
    target_size: int = 512,
    bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[Union[Image.Image, np.ndarray], np.ndarray]:
    """
    Apply principal point correction to image and intrinsics.

    Args:
        image: Input image
        camera: Camera frame dict from opencv_cameras.json
        method: Correction method
            - "none": No correction (original behavior)
            - "actual_pp": Use actual PP values in intrinsics
            - "crop": Crop image so actual PP becomes centered
        resize_ratio: Resize ratio applied to image
        target_size: Target image size
        bg_color: Background color for padding

    Returns:
        Tuple of (corrected_image, corrected_intrinsics)
    """
    actual_cx, actual_cy = get_actual_principal_point(camera)

    if method == "none":
        # Original behavior - use forced cx=cy=256
        intrinsics = np.array([
            camera.get("fx", 549.0),
            camera.get("fy", 549.0),
            camera.get("cx", 256.0),
            camera.get("cy", 256.0)
        ], dtype=np.float32) * resize_ratio
        return image, intrinsics

    elif method == "actual_pp":
        # Use actual PP values (varying cx,cy per view)
        intrinsics = correct_intrinsics_to_actual_pp(camera, resize_ratio)
        return image, intrinsics

    elif method == "crop":
        # Crop image so actual PP becomes centered
        # After this, cx=cy=256 is actually correct!
        corrected_image = crop_image_to_pp(
            image,
            actual_cx * resize_ratio,
            actual_cy * resize_ratio,
            target_cx=target_size / 2,
            target_cy=target_size / 2,
            target_size=target_size,
            bg_color=bg_color
        )
        # Now cx=cy=256 is mathematically correct
        intrinsics = np.array([
            camera.get("fx", 549.0),
            camera.get("fy", 549.0),
            target_size / 2,  # cx = 256
            target_size / 2   # cy = 256
        ], dtype=np.float32) * resize_ratio
        # Note: fx,fy scaled but cx,cy should be target_size/2 after scaling if already at center
        # Actually, since we cropped to center, cx,cy should be exactly center
        intrinsics[2] = target_size / 2 * resize_ratio  # Will be 256 if resize_ratio=1
        intrinsics[3] = target_size / 2 * resize_ratio
        return corrected_image, intrinsics

    else:
        raise ValueError(f"Unknown PP correction method: {method}")


# Integration code for mouse_dataset.py
INTEGRATION_PATCH = '''
# Add to mouse_dataset.py imports:
from mouse_extensions.scripts.solutions.pp_correction_integration import apply_pp_correction

# Add to MouseViewDataset.__init__:
self.pp_correction_method = mouse_config.get("pp_correction", "none")
# Options: "none" (default), "actual_pp", "crop"

# Replace intrinsics extraction in __getitem__ (around line 339):
# OLD:
#     intrinsics = np.array([
#         camera["fx"], camera["fy"], camera["cx"], camera["cy"]
#     ])
#     intrinsics *= resize_ratio
#
# NEW:
    image, intrinsics = apply_pp_correction(
        image,
        camera,
        method=self.pp_correction_method,
        resize_ratio=resize_ratio,
        target_size=target_size,
        bg_color=bg_color_255
    )
'''

# Config YAML example
CONFIG_EXAMPLE = '''
# Add to training config YAML:

mouse:
  # Principal Point correction method
  # Options:
  #   "none"      - No correction (original buggy behavior)
  #   "actual_pp" - Use actual PP from _transform (varying cx,cy)
  #   "crop"      - Crop image so actual PP becomes centered (recommended)
  pp_correction: "crop"
'''


if __name__ == "__main__":
    print("Testing PP Correction Integration...")

    # Create test camera data (simulating v13 bug)
    test_camera = {
        "fx": 549.0,
        "fy": 549.0,
        "cx": 256.0,  # Claimed PP (wrong!)
        "cy": 256.0,
        "_transform": {
            "scaled_cx": 184.3,  # Actual PP
            "scaled_cy": 150.6
        },
        "w2c": [[1,0,0,0], [0,1,0,0], [0,0,1,2.7], [0,0,0,1]]
    }

    # Create test image
    test_image = np.random.rand(512, 512, 3).astype(np.float32)

    print(f"\nTest camera:")
    print(f"  Claimed PP: ({test_camera['cx']}, {test_camera['cy']})")
    actual_cx, actual_cy = get_actual_principal_point(test_camera)
    print(f"  Actual PP: ({actual_cx}, {actual_cy})")
    print(f"  Offset: {np.sqrt((actual_cx-256)**2 + (actual_cy-256)**2):.1f}px")

    # Test different methods
    for method in ["none", "actual_pp", "crop"]:
        img, intrinsics = apply_pp_correction(
            test_image.copy(), test_camera, method=method
        )
        print(f"\nMethod '{method}':")
        print(f"  Intrinsics: fx={intrinsics[0]:.1f}, fy={intrinsics[1]:.1f}, "
              f"cx={intrinsics[2]:.1f}, cy={intrinsics[3]:.1f}")
        if method == "crop":
            print(f"  Image shape: {img.shape}")

    print("\nIntegration code:")
    print(INTEGRATION_PATCH)

    print("\nConfig example:")
    print(CONFIG_EXAMPLE)

    print("\nTest completed!")
