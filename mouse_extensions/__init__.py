"""
Mouse Extensions for FaceLift GS-LRM

A modular extension package for adapting FaceLift to markerless mouse 3D reconstruction.
"""

__version__ = "1.0.0"

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

__all__ = ["__version__", "get_model_extensions"]
