"""
Mouse Extensions for FaceLift GS-LRM

A modular extension package for adapting FaceLift to markerless mouse 3D reconstruction.

Architecture:
    gslrm/          - Original code (minimal modifications)
    mouse_extensions/ - All custom code (this package)
        data/       - Custom datasets (MouseViewDataset)
        model/      - Model extensions
        preprocessing/ - Data preprocessing
        validation/ - Validation utilities
        registry.py - Central registration
"""

__version__ = "1.1.0"

# Registry for datasets
from .registry import get_dataset_class, list_datasets, register_dataset

# Lazy imports to avoid circular dependencies
def get_model_extensions():
    from .model import (
        compute_mask_from_config,
        compute_ghost_metrics,
        compute_alpha_loss,
        compute_alpha_metrics,
        AlphaLossComputer,
    )
    return {
        "compute_mask_from_config": compute_mask_from_config,
        "compute_ghost_metrics": compute_ghost_metrics,
        "compute_alpha_loss": compute_alpha_loss,
        "compute_alpha_metrics": compute_alpha_metrics,
        "AlphaLossComputer": AlphaLossComputer,
    }


def get_dataset(name: str = "mouse"):
    """Get dataset class by name. Default: mouse."""
    return get_dataset_class(name)


__all__ = [
    "__version__",
    "get_model_extensions",
    "get_dataset",
    "get_dataset_class",
    "list_datasets",
    "register_dataset",
]
