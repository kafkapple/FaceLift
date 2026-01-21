"""
Mouse Extensions - Preprocessing Module

This module handles preprocessing of markerless mouse data for FaceLift GS-LRM.

Usage:
    # Using unified preprocessor (D7+)
    from mouse_extensions.preprocessing import UnifiedPreprocessor, PreprocessConfig
    
    config = PreprocessConfig.from_preset("D8", input_dir=..., output_dir=...)
    preprocessor = UnifiedPreprocessor(config)
    preprocessor.run()

    # CLI
    python -m mouse_extensions.preprocessing.preprocess --preset D8 \\
        --input-dir /path/to/raw --output-dir /path/to/D8

Presets (D7+):
    - D7    : PP-centered shift, affine (fx_only scale)
    - D7.1  : PP-centered shift, affine (individual scale)
    - D7.2  : PP-centered shift, affine (average scale)
    - D8    : Precision homography, skew correction [RECOMMENDED]
    - D8.1  : D8 + 1.3x zoom for larger mouse

Legacy Presets (D1-D6):
    See archive/unified_preprocessor_d7_legacy.py

Key Components:
    - preprocess: Unified preprocessing for D7+ (single entry point)
    - presets: Preset configurations
    - center_estimation: Multi-view object center estimation
    - data_loader: Unified data loading
"""

from .preprocess import (
    UnifiedPreprocessor,
    PreprocessConfig,
    TransformType,
    ScaleMode,
)

from .presets import (
    PRESETS,
    get_preset,
    list_presets,
    get_recommended,
)

from .center_estimation import (
    CenterEstimator,
    CenterMethod,
    CenterEstimationResult,
    estimate_center_for_frame,
    compare_all_methods,
)

from .data_loader import DataLoader

__all__ = [
    # Unified Preprocessor (D7+)
    "UnifiedPreprocessor",
    "PreprocessConfig",
    "TransformType",
    "ScaleMode",
    # Presets
    "PRESETS",
    "get_preset",
    "list_presets",
    "get_recommended",
    # Center Estimation
    "CenterEstimator",
    "CenterMethod",
    "CenterEstimationResult",
    "estimate_center_for_frame",
    "compare_all_methods",
    # Data Loader
    "DataLoader",
]
