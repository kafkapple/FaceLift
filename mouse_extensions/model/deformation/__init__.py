# Copyright 2026 FaceLift Mouse Extensions
# Deformation Network module for temporal Gaussian consistency

from .deformation_network import (
    DeformationNetwork,
    DeformationConfig,
    PositionalEncoding,
)
from .gaussian_params import GaussianParams
from .temporal_pipeline import (
    TemporalGaussianPipeline,
    TemporalConfig,
)
from .deformation_trainer import (
    DeformationTrainer,
    TrainerConfig,
)

__all__ = [
    # Core network
    "DeformationNetwork",
    "DeformationConfig",
    "PositionalEncoding",
    # Data structure
    "GaussianParams",
    # Pipeline
    "TemporalGaussianPipeline",
    "TemporalConfig",
    # Training
    "DeformationTrainer",
    "TrainerConfig",
]
