"""
Comparison visualization module.

Config-driven multi-experiment comparison with novel-view rendering.

Components:
    - CameraPreset: generates camera extrinsics for various viewpoints
    - GridComposer: composites rendered frames into side-by-side grids/videos
    - compare.py: main entry point (CLI + programmatic API)
"""

from .camera_presets import CameraPreset, make_preset
from .grid_composer import GridComposer
from .compare import run_comparison, load_comparison_config

__all__ = [
    "CameraPreset",
    "make_preset",
    "GridComposer",
    "run_comparison",
    "load_comparison_config",
]
