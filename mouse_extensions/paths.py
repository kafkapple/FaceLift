"""Centralized path configuration for mouse_extensions (SSOT).

All path constants should be imported from this module.
Supports environment variable overrides for flexibility across servers.

Environment Variables:
    FACELIFT_ROOT: Base directory of FaceLift project
    FACELIFT_DATA: Base directory for data storage
    FACELIFT_HOT:  NVMe tier root (fast, limited)
    FACELIFT_WARM: SSD tier root (medium speed)
    FACELIFT_COLD: NFS tier root (large, slow)

Output Directory Structure (v2 — 2026-03-23):
    outputs/
    ├── experiments/{species}/{experiment_id}/   # Training runs
    ├── eval/{species}/{experiment_id}/          # Quantitative evaluation
    ├── viz/{type}/{species}/{experiment_id}/    # Visualization media
    ├── features/{species}/{feature_type}/       # Extracted features
    ├── analysis/{species}/{analysis_type}/      # Exploratory analysis
    ├── datasets/{category}/                     # Generated/derived datasets
    ├── reports/{report_slug}/                   # Publication reports
    └── _archive/                                # Deprecated experiments
"""

import os
import warnings
from pathlib import Path
from typing import Literal

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
RAT_DATA = FACELIFT_ROOT / "outputs/datasets/fine_tune/rat/gslrm_format"

# === Checkpoints ===
CKPT_DIR = HOT / "checkpoints/FaceLift/gslrm"
CHECKPOINTS_DIR = FACELIFT_ROOT / "checkpoints"
WANDB_DIR = WARM / "wandb_runs"

# === Project Directories ===
CONFIGS_DIR = FACELIFT_ROOT / "configs"
OUTPUTS_DIR = FACELIFT_ROOT / "outputs"
LOGS_DIR = FACELIFT_ROOT / "logs"

# ============================================================================
# Output Structure v2 — Top-level directories
# ============================================================================

EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"   # Training runs + checkpoints
EVAL_DIR = OUTPUTS_DIR / "eval"                 # Quantitative evaluation
VIZ_ROOT = OUTPUTS_DIR / "viz"                  # Visualization media
FEATURES_ROOT = OUTPUTS_DIR / "features"        # Extracted features
ANALYSIS_ROOT = OUTPUTS_DIR / "analysis"        # Exploratory analysis
DATASETS_OUT_DIR = OUTPUTS_DIR / "datasets"     # Generated/derived datasets
REPORTS_OUT_DIR = OUTPUTS_DIR / "reports"        # Publication reports
ARCHIVE_DIR = OUTPUTS_DIR / "_archive"          # Deprecated experiments

# Valid species (extend as project grows)
VALID_SPECIES = {"mouse", "rat", "marmoset"}

# Valid visualization types
VALID_VIZ_TYPES = {"turntable", "cinematic", "bodypart", "comparison"}

# Valid feature types
VALID_FEATURE_TYPES = {"gaussian", "covariance", "temporal", "hlac"}

# Valid analysis types
VALID_ANALYSIS_TYPES = {
    "behavior_clustering", "bams", "neural_texture", "filtering",
    "opacity", "motion_analysis", "resolution_comparison",
}


# ============================================================================
# Backward-compatible aliases (DO NOT REMOVE — used by behavior/paths.py)
# These point to legacy locations; data will be symlinked during migration.
# For NEW code, use FEATURES_ROOT + get_feature_dir() instead.
# ============================================================================

FEATURES_BASE = OUTPUTS_DIR / "features" / "clustering"
FEATURES_DIR = FEATURES_BASE  # DEPRECATED: clustering-only. Use FEATURES_ROOT for new code.
GAUSSIANS_RAW_DIR = FEATURES_BASE / "gaussians_raw"
GAUSSIAN_RAW_FEATURES = FEATURES_BASE / "gaussian_raw_features.npz"
COV_N2_DIR = FEATURES_BASE / "covariance_n2"
TEMPORAL_DIR = FEATURES_BASE / "temporal"

REPORT_BASE = OUTPUTS_DIR / "report" / "clustering"


# ============================================================================
# Factory functions — species-aware path generation
# ============================================================================

def _validate_species(species: str) -> None:
    if species not in VALID_SPECIES:
        raise ValueError(
            f"Unknown species '{species}'. Valid: {VALID_SPECIES}. "
            "Add new species to VALID_SPECIES in paths.py."
        )


def get_experiment_dir(species: str, experiment_id: str) -> Path:
    """Get experiment output directory for training runs."""
    _validate_species(species)
    return EXPERIMENTS_DIR / species / experiment_id


def get_eval_dir(species: str, experiment_id: str) -> Path:
    """Get evaluation output directory."""
    _validate_species(species)
    return EVAL_DIR / species / experiment_id


def get_viz_dir(species: str, experiment_id: str, viz_type: str) -> Path:
    """Get visualization output directory.

    Args:
        viz_type: One of 'turntable', 'cinematic', 'bodypart', 'comparison'
    """
    _validate_species(species)
    if viz_type not in VALID_VIZ_TYPES:
        raise ValueError(f"Unknown viz_type '{viz_type}'. Valid: {VALID_VIZ_TYPES}")
    return VIZ_ROOT / viz_type / species / experiment_id


def get_feature_dir(species: str, feature_type: str) -> Path:
    """Get feature extraction output directory.

    Args:
        feature_type: One of 'gaussian', 'covariance', 'temporal', 'hlac'
    """
    _validate_species(species)
    if feature_type not in VALID_FEATURE_TYPES:
        raise ValueError(f"Unknown feature_type '{feature_type}'. Valid: {VALID_FEATURE_TYPES}")
    return FEATURES_ROOT / species / feature_type


def get_analysis_dir(species: str, analysis_type: str, run_id: str = "") -> Path:
    """Get analysis output directory.

    Args:
        analysis_type: One of 'behavior_clustering', 'bams', 'neural_texture', 'filtering'
        run_id: Optional run identifier (e.g., timestamp or descriptive name)
    """
    _validate_species(species)
    if analysis_type not in VALID_ANALYSIS_TYPES:
        raise ValueError(f"Unknown analysis_type '{analysis_type}'. Valid: {VALID_ANALYSIS_TYPES}")
    base = ANALYSIS_ROOT / species / analysis_type
    return base / run_id if run_id else base


def get_dataset_out_dir(category: str, species: str = "") -> Path:
    """Get generated dataset output directory.

    Args:
        category: 'novel_view', 'fine_tune', or 'generated'
        species: Optional species subdirectory
    """
    base = DATASETS_OUT_DIR / category
    if species:
        _validate_species(species)
        base = base / species
    return base


def get_report_dir(report_slug: str) -> Path:
    """Get report output directory."""
    return REPORTS_OUT_DIR / report_slug


# ============================================================================
# Existing helpers (preserved)
# ============================================================================

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
        for p in missing:
            warnings.warn(f"MISSING: {p}")
    return len(missing) == 0


def scaffold_output_dirs(species: str | None = None) -> None:
    """Create the full output directory scaffold.

    Args:
        species: If given, create dirs for this species only.
                 If None, create for all VALID_SPECIES.
    """
    species_list = [species] if species else sorted(VALID_SPECIES)
    for sp in species_list:
        _validate_species(sp)
        # experiments, eval
        (EXPERIMENTS_DIR / sp).mkdir(parents=True, exist_ok=True)
        (EVAL_DIR / sp).mkdir(parents=True, exist_ok=True)
        # viz types
        for vt in VALID_VIZ_TYPES:
            (VIZ_ROOT / vt / sp).mkdir(parents=True, exist_ok=True)
        # feature types
        for ft in VALID_FEATURE_TYPES:
            (FEATURES_ROOT / sp / ft).mkdir(parents=True, exist_ok=True)
        # analysis types
        for at in VALID_ANALYSIS_TYPES:
            (ANALYSIS_ROOT / sp / at).mkdir(parents=True, exist_ok=True)
        # datasets
        for cat in ("novel_view", "fine_tune", "generated"):
            (DATASETS_OUT_DIR / cat / sp).mkdir(parents=True, exist_ok=True)
    # Species-independent dirs
    REPORTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
