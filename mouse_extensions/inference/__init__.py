"""End-to-end inference pipeline for Mouse-FaceLift.

Modules:
    MVDiffusionInference  - Single image → 6-view generation
    GSLRMInference        - 6-view → 3D Gaussian reconstruction
    EndToEndPipeline      - Chain both pipelines (with optional preprocessing)
    TemporalE2EPipeline   - Multi-frame temporal inference with bilateral smoothing
    
Preprocessing (for raw images):
    MouseInferencePreprocessor - SAM-based mouse detection + alignment
    MousePreprocessConfig      - Configuration for preprocessing
"""

from mouse_extensions.inference.mvdiffusion_pipeline import MVDiffusionInference
from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
from mouse_extensions.inference.end_to_end import EndToEndPipeline
from mouse_extensions.inference.temporal_e2e import TemporalE2EPipeline


def get_preprocessor():
    """Get MouseInferencePreprocessor (lazy import)."""
    from mouse_extensions.inference.preprocessing import MouseInferencePreprocessor
    return MouseInferencePreprocessor


__all__ = [
    "MVDiffusionInference",
    "GSLRMInference",
    "EndToEndPipeline",
    "TemporalE2EPipeline",
    "get_preprocessor",
]
