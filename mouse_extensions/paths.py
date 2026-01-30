"""Centralized path configuration for mouse_extensions.

All path constants should be imported from this module.
Supports environment variable overrides for flexibility across servers.

Environment Variables:
    FACELIFT_ROOT: Base directory of FaceLift project
    FACELIFT_DATA: Base directory for data storage
"""

import os
from pathlib import Path

# Project root (auto-detected or from environment)
FACELIFT_ROOT = Path(os.environ.get(
    "FACELIFT_ROOT",
    Path(__file__).parent.parent.resolve()
))

# Data root (configurable via environment)
DATA_ROOT = Path(os.environ.get(
    "FACELIFT_DATA",
    Path.home() / "data"
))

# Preprocessed datasets
PREPROCESSED_DIR = DATA_ROOT / "preprocessed" / "FaceLift_mouse"

# Raw data
RAW_DATA_DIR = DATA_ROOT / "raw"

# Checkpoints
CHECKPOINTS_DIR = FACELIFT_ROOT / "checkpoints"

# Configs
CONFIGS_DIR = FACELIFT_ROOT / "configs"

# Outputs
OUTPUTS_DIR = FACELIFT_ROOT / "outputs"

# Logs
LOGS_DIR = FACELIFT_ROOT / "logs"


def get_dataset_dir(dataset_name: str) -> Path:
    """Get path to a preprocessed dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'M5', 'M5h_2')
        
    Returns:
        Path to the dataset directory
    """
    return PREPROCESSED_DIR / dataset_name


def get_checkpoint_dir(experiment_name: str) -> Path:
    """Get path to experiment checkpoint directory.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Path to the checkpoint directory
    """
    return CHECKPOINTS_DIR / "gslrm" / experiment_name
