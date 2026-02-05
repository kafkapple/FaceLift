# Copyright 2026 FaceLift Mouse Extensions
# Deformation Network module for temporal Gaussian consistency

from .deformation_network import (
    # V1 (original - has drift problem)
    DeformationNetwork,
    DeformationConfig,
    PositionalEncoding,
    # V2 (fixed - dual-frame input)
    DeformationNetworkV2,
    DeformationConfigV2,
    TimeEmbedding,
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
from .deformation_trainer_v2 import (
    DeformationTrainerV2,
    TrainerConfigV2,
    ARAPLoss,
    VelocityLoss,
)
from .temporal_deform_inference import (
    TemporalDeformInference,
    InferenceConfig,
)
from .gslrm_integration import (
    GSLRMGaussianGenerator,
    GaussianCache,
)

__all__ = [
    # V1 Core network (deprecated - has drift)
    "DeformationNetwork",
    "DeformationConfig",
    "PositionalEncoding",
    # V2 Core network (recommended)
    "DeformationNetworkV2",
    "DeformationConfigV2",
    "TimeEmbedding",
    # Data structure
    "GaussianParams",
    # V1 Pipeline (deprecated)
    "TemporalGaussianPipeline",
    "TemporalConfig",
    # V1 Training (deprecated)
    "DeformationTrainer",
    "TrainerConfig",
    # V2 Training (recommended)
    "DeformationTrainerV2",
    "TrainerConfigV2",
    "ARAPLoss",
    "VelocityLoss",
    # V2 Inference (recommended)
    "TemporalDeformInference",
    "InferenceConfig",
    # GS-LRM integration
    "GSLRMGaussianGenerator",
    "GaussianCache",
]
