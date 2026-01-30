"""Configuration for mouse inference preprocessing.

Defines M5 training statistics and preprocessing parameters.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MousePreprocessConfig:
    """Configuration for mouse inference preprocessing.
    
    M5 Training Statistics:
        - Resolution: 512×512
        - fx, fy: 549 (normalized focal length)
        - cx, cy: 256 (image center)
        - Coverage: ~6% (mouse area / total pixels)
        - Background: White (255, 255, 255)
    """
    
    # Target output specifications (from M5 training data)
    target_resolution: int = 512
    target_coverage: float = 0.06  # ~6% foreground coverage
    target_cx: float = 256.0
    target_cy: float = 256.0
    
    # Background color (RGB)
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Scale limits for safety (avoid extreme scaling)
    min_scale: float = 0.3
    max_scale: float = 3.0
    
    # SAM configuration
    sam_model_type: str = "vit_h"
    sam_points_per_side: int = 32
    sam_pred_iou_thresh: float = 0.88
    sam_stability_score_thresh: float = 0.95
    
    # Mask filtering
    min_area_ratio: float = 0.01  # Minimum 1% of image
    max_area_ratio: float = 0.90  # Maximum 90% of image
    
    # Fallback behavior when SAM fails
    fallback_mode: str = "resize"  # "resize" | "error"
    
    # Auto-detection thresholds
    auto_detect_coverage_range: Tuple[float, float] = (0.02, 0.15)
    auto_detect_resolution_tolerance: int = 10  # pixels
    
    def __post_init__(self):
        """Validate configuration."""
        if self.min_scale >= self.max_scale:
            raise ValueError("min_scale must be less than max_scale")
        if not (0 < self.target_coverage < 1):
            raise ValueError("target_coverage must be between 0 and 1")


# Default configuration instance
DEFAULT_CONFIG = MousePreprocessConfig()
