"""Centralized path configuration for mouse_extensions (SSOT).

All path constants should be imported from this module.
Supports environment variable overrides for flexibility across servers.

Environment Variables:
    FACELIFT_ROOT: Base directory of FaceLift project
    FACELIFT_DATA: Base directory for data storage
    FACELIFT_HOT:  NVMe tier root (fast, limited)
    FACELIFT_WARM: SSD tier root (medium speed)
    FACELIFT_COLD: NFS tier root (large, slow)
"""

import os
from pathlib import Path

# === Storage Tier Roots ===
HOT = Path(os.getenv("FACELIFT_HOT", "/node_data/joon"))       # NVMe
WARM = Path(os.getenv("FACELIFT_WARM", "/node_data_2/joon"))    # SSD
COLD = Path(os.getenv("FACELIFT_COLD", str(Path.home())))       # NFS

# === Project Root ===
FACELIFT_ROOT = Path(os.environ.get(
    "FACELIFT_ROOT",
    Path(__file__).parent.parent.resolve()
))

# === Data Root ===
DATA_ROOT = Path(os.environ.get(
    "FACELIFT_DATA",
    COLD / "data"
))

# === Preprocessed Datasets ===
PREPROCESSED_DIR = DATA_ROOT / "preprocessed" / "FaceLift_mouse"
RAW_DATA_DIR = DATA_ROOT / "raw"

# === Critical Data Assets (SSOT) ===
# Rule: register here if 2+ files share the path
KP_22 = COLD / "data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz"
M5_DATA = PREPROCESSED_DIR / "M5"
M5T2_DATA = PREPROCESSED_DIR / "M5t2"
RAT_DATA = FACELIFT_ROOT / "outputs/sdannce_rat_ft/gslrm_format"

# === Checkpoints ===
CKPT_DIR = HOT / "checkpoints/FaceLift/gslrm"
CHECKPOINTS_DIR = FACELIFT_ROOT / "checkpoints"
WANDB_DIR = WARM / "wandb_runs"

# === Project Directories ===
CONFIGS_DIR = FACELIFT_ROOT / "configs"
OUTPUTS_DIR = FACELIFT_ROOT / "outputs"
LOGS_DIR = FACELIFT_ROOT / "logs"

# === Feature Paths (moved from report/ to outputs/features/) ===
FEATURES_BASE = OUTPUTS_DIR / "features" / "clustering"
FEATURES_DIR = FEATURES_BASE
GAUSSIANS_RAW_DIR = FEATURES_BASE / "gaussians_raw"
GAUSSIAN_RAW_FEATURES = FEATURES_BASE / "gaussian_raw_features.npz"
COV_N2_DIR = FEATURES_BASE / "covariance_n2"
TEMPORAL_DIR = FEATURES_BASE / "temporal"

# === Report Paths (visualizations/reports only) ===
REPORT_BASE = OUTPUTS_DIR / "report" / "clustering"


# === Helpers ===

def get_dataset_dir(dataset_name: str) -> Path:
    """Get path to a preprocessed dataset."""
    return PREPROCESSED_DIR / dataset_name


def get_checkpoint_dir(experiment_name: str) -> Path:
    """Get path to experiment checkpoint directory."""
    return CHECKPOINTS_DIR / "gslrm" / experiment_name


def validate(*paths: Path) -> list[Path]:
    """Return list of missing paths. Empty = all OK."""
    return [p for p in paths if not p.exists()]


def check_critical():
    """Call before training/analysis. Warns on missing critical files."""
    missing = validate(KP_22, M5_DATA)
    if missing:
        import warnings
        for p in missing:
            warnings.warn(f"MISSING: {p}")
    return len(missing) == 0
