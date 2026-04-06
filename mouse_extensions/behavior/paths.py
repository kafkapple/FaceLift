"""Output paths for behavior clustering experiments.

Imports critical paths from top-level SSOT (mouse_extensions.paths).
Defines behavior-specific output subdirectories.

Directory structure (v2 — see mouse_extensions/paths.py for full structure):
    outputs/features/clustering/         ← Legacy feature data (backward compat alias)
    │   ├── gaussians_raw/               ← Per-frame NPZ (1M Gaussians, float16)
    │   ├── covariance_n2/               ← Covariance features
    │   └── temporal/                    ← Temporal features
    outputs/reports/clustering/           ← Consolidated reports (topic-based)
    ├── data/                            ← Input data (keypoints copy)
    ├── results/                         ← JSON experiment results
    ├── visualizations/                  ← Plots, GIFs, images
    └── reports/                         ← Self-contained HTML reports

New canonical paths (use mouse_extensions.paths factory functions):
    outputs/features/{species}/{feature_type}/   ← Feature extraction
    outputs/analysis/{species}/{analysis_type}/  ← Exploratory analysis
    outputs/viz/{viz_type}/{species}/            ← Visualizations
"""
from pathlib import Path

from mouse_extensions.paths import (
    KP_22,
    M5_DATA,
    FEATURES_BASE,
    GAUSSIANS_RAW_DIR,
    GAUSSIAN_RAW_FEATURES,
    COV_N2_DIR,
    TEMPORAL_DIR,
    REPORT_BASE,
    OUTPUTS_DIR,
)

# Re-export for backward compatibility
FEATURES_DIR = FEATURES_BASE

# Report subdirectories
DATA_DIR = REPORT_BASE / "data"
RESULTS_DIR = REPORT_BASE / "results"
VIZ_DIR = REPORT_BASE / "visualizations"
REPORTS_DIR = REPORT_BASE / "reports"

# Result subdirs
SPARSE_ABLATION_DIR = RESULTS_DIR / "sparse_ablation"
DENSE_2D_DIR = RESULTS_DIR / "dense_2d"
GAUSSIAN_3D_DIR = RESULTS_DIR / "gaussian_3d"
GAUSSIAN_ABLATION_DIR = RESULTS_DIR / "gaussian_ablation"
MULTI_METHOD_DIR = RESULTS_DIR / "multi_method"
FULL_DATASET_DIR = RESULTS_DIR / "full_dataset"

# Visualization subdirs
UMAP_DIR = VIZ_DIR / "umap"
RGB_GIFS_DIR = VIZ_DIR / "rgb_gifs"
ETHOGRAMS_DIR = VIZ_DIR / "ethograms"
DEFORMATION_DIR = VIZ_DIR / "deformation"

# Data files
KEYPOINTS_PATH = DATA_DIR / "keypoints_22_3d.npz"

# Deprecated aliases (use top-level imports instead)
GPU03_KEYPOINTS = str(KP_22)
GPU03_M5_DATA = str(M5_DATA)


def ensure_dirs():
    """Create all output directories."""
    for d in [DATA_DIR, FEATURES_DIR, RESULTS_DIR, VIZ_DIR, REPORTS_DIR,
              GAUSSIANS_RAW_DIR, COV_N2_DIR, TEMPORAL_DIR,
              SPARSE_ABLATION_DIR, DENSE_2D_DIR, GAUSSIAN_3D_DIR,
              GAUSSIAN_ABLATION_DIR, MULTI_METHOD_DIR, FULL_DATASET_DIR,
              UMAP_DIR, RGB_GIFS_DIR, ETHOGRAMS_DIR, DEFORMATION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
