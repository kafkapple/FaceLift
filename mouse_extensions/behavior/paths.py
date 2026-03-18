"""Centralized output paths for behavior clustering experiments.

All modules should import paths from here for consistency.
SSOT for all output locations — never hardcode paths elsewhere.

Directory structure:
    outputs/report/clustering/           ← SSOT root
    ├── data/                            ← Input data (keypoints, raw Gaussians)
    │   └── gaussians_raw/               ← Per-frame raw Gaussian NPZ
    ├── features/                        ← Extracted feature files
    │   └── gaussians_raw/               ← Per-frame NPZ (1M Gaussians, float16)
    ├── results/                         ← JSON experiment results
    │   ├── sparse_ablation/
    │   ├── dense_2d/
    │   ├── gaussian_3d/
    │   ├── gaussian_ablation/           ← Gaussian count ablation
    │   ├── multi_method/
    │   └── full_dataset/
    ├── visualizations/                  ← Plots, GIFs, images
    │   ├── umap/
    │   ├── rgb_gifs/
    │   ├── ethograms/
    │   └── deformation/
    └── reports/                         ← Self-contained HTML reports
"""
from pathlib import Path

# Base output directory
REPORT_BASE = Path("outputs/report/clustering")

# Subdirectories
DATA_DIR = REPORT_BASE / "data"
FEATURES_DIR = REPORT_BASE / "features"
RESULTS_DIR = REPORT_BASE / "results"
VIZ_DIR = REPORT_BASE / "visualizations"
REPORTS_DIR = REPORT_BASE / "reports"

# Feature subdirs
GAUSSIANS_RAW_DIR = FEATURES_DIR / "gaussians_raw"

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

# gpu03 absolute paths (raw data, not in outputs/)
GPU03_KEYPOINTS = "/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz"
GPU03_M5_DATA = "/home/joon/data/preprocessed/FaceLift_mouse/M5"


def ensure_dirs():
    """Create all output directories."""
    for d in [DATA_DIR, FEATURES_DIR, RESULTS_DIR, VIZ_DIR, REPORTS_DIR,
              GAUSSIANS_RAW_DIR,
              SPARSE_ABLATION_DIR, DENSE_2D_DIR, GAUSSIAN_3D_DIR,
              GAUSSIAN_ABLATION_DIR, MULTI_METHOD_DIR, FULL_DATASET_DIR,
              UMAP_DIR, RGB_GIFS_DIR, ETHOGRAMS_DIR, DEFORMATION_DIR]:
        d.mkdir(parents=True, exist_ok=True)
