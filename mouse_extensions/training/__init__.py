"""Training extensions for Mouse-FaceLift.

Modules:
    TemporalTrainingConfig - Configuration for temporal training
    TemporalLossComputer - Computes ARAP, velocity, isometry losses
    create_temporal_dataloader - Dataloader for consecutive frames
"""

from mouse_extensions.training.temporal_trainer import (
    TemporalTrainingConfig,
    TemporalLossComputer,
    create_temporal_dataloader,
)

__all__ = [
    'TemporalTrainingConfig',
    'TemporalLossComputer',
    'create_temporal_dataloader',
]
