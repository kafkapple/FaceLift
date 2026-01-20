"""
Mouse Extensions - Preprocessing Module

This module handles preprocessing of markerless mouse data for FaceLift GS-LRM.

Usage:
    from mouse_extensions.preprocessing import UnifiedPreprocessor, PreprocessConfig
    from mouse_extensions.preprocessing.center_estimation import CenterEstimator
    
    # Using presets (recommended)
    config = PreprocessConfig.from_preset('D3', input_dir, output_dir)
    preprocessor = UnifiedPreprocessor(config)
    preprocessor.run()

Key Components:
    - unified_preprocessor: Consolidated preprocessing with presets (v13, D1, D2, D3)
    - center_estimation: Multi-view object center estimation
    - data_loader: Unified data loading for video/image sources
    - run_pipeline: YAML-based pipeline runner  
    - split_dataset: Train/val split generator

Presets:
    - v13: Legacy PP-bugged format (compatibility)
    - D1: PP-centered crop (cx=cy=256)
    - D2: Correct PP without shift
    - D3: Triangulation-based center with correct PP (RECOMMENDED)

Center Estimation (2026-01-17):
    Per-view 2D center estimation causes cross-view inconsistency (14.4px error).
    Use triangulation for unified 3D center (0px reprojection error).
"""

from .unified_preprocessor import (
    UnifiedPreprocessor,
    PreprocessConfig,
    CenterMethodType,
    PPMethodType,
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
    # Unified Preprocessor
    "UnifiedPreprocessor",
    "PreprocessConfig",
    "CenterMethodType",
    "PPMethodType",
    # Center Estimation
    "CenterEstimator",
    "CenterMethod",
    "CenterEstimationResult",
    "estimate_center_for_frame",
    "compare_all_methods",
    # Data Loader
    "DataLoader",
]
