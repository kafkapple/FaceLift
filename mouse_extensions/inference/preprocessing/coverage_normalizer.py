"""Coverage normalization for mouse images.

Adjusts scale to match M5 training coverage (~6%).
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CoverageNormalizer:
    """Normalize image scale to match target coverage.
    
    Coverage = foreground_pixels / total_pixels
    
    For M5 training data, target coverage is ~6%.
    Scale is adjusted so output coverage matches target.
    """
    
    def __init__(
        self,
        target_coverage: float = 0.06,
        target_resolution: int = 512,
        min_scale: float = 0.3,
        max_scale: float = 3.0,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """Initialize normalizer.
        
        Args:
            target_coverage: Target foreground coverage (0-1).
            target_resolution: Target output resolution.
            min_scale: Minimum allowed scale factor.
            max_scale: Maximum allowed scale factor.
            bg_color: Background color for padding.
        """
        self.target_coverage = target_coverage
        self.target_resolution = target_resolution
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.bg_color = bg_color
        
    def compute_scale(self, mask: np.ndarray) -> float:
        """Compute scale factor to match target coverage.
        
        Args:
            mask: Binary mask (H, W), values 0 or 255.
            
        Returns:
            Scale factor. Values > 1 mean upscale, < 1 mean downscale.
        """
        h, w = mask.shape[:2]
        total_pixels = self.target_resolution ** 2
        
        # Current coverage
        foreground_pixels = (mask > 127).sum()
        current_coverage = foreground_pixels / (h * w)
        
        if current_coverage < 1e-6:
            logger.warning("Empty mask, using scale=1.0")
            return 1.0
            
        # Scale factor: area scales quadratically with linear scale
        # new_coverage = current_coverage * scale^2
        # scale = sqrt(target_coverage / current_coverage)
        scale = np.sqrt(self.target_coverage / current_coverage)
        
        # Clamp to safe range
        scale = np.clip(scale, self.min_scale, self.max_scale)
        
        logger.debug(
            f"Coverage: {current_coverage:.4f} -> {self.target_coverage:.4f}, "
            f"scale: {scale:.3f}"
        )
        
        return float(scale)
        
    def normalize(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        centroid: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], float]:
        """Normalize image to target coverage and center.
        
        Pipeline:
        1. Compute scale factor from mask coverage
        2. Scale image and mask
        3. Translate so centroid is at image center
        4. Crop/pad to target resolution
        
        Args:
            image: Input image (H, W, 3), RGB, uint8.
            mask: Binary mask (H, W), values 0 or 255.
            centroid: (x, y) centroid of foreground.
            
        Returns:
            Tuple of:
                - Normalized image (target_res, target_res, 3)
                - Normalized mask (target_res, target_res)
                - New centroid (should be near center)
                - Applied scale factor
        """
        h, w = image.shape[:2]
        res = self.target_resolution
        center = res / 2
        
        # Step 1: Compute scale
        scale = self.compute_scale(mask)
        
        # Step 2: Compute transform
        # Translation: scale * centroid -> center
        cx, cy = centroid
        tx = center - scale * cx
        ty = center - scale * cy
        
        # Affine matrix: scale + translate
        M = np.array([
            [scale, 0, tx],
            [0, scale, ty],
        ], dtype=np.float32)
        
        # Step 3: Apply transform
        bg = np.array(self.bg_color, dtype=np.uint8)
        norm_image = cv2.warpAffine(
            image, M, (res, res),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(bg.tolist()),
        )
        
        norm_mask = cv2.warpAffine(
            mask, M, (res, res),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        
        # Step 4: Verify new centroid
        ys, xs = np.where(norm_mask > 127)
        if len(xs) > 0:
            new_cx = float(xs.mean())
            new_cy = float(ys.mean())
        else:
            new_cx = center
            new_cy = center
            
        return norm_image, norm_mask, (new_cx, new_cy), scale
