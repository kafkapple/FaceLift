"""End-to-end inference pipeline for Mouse-FaceLift.

Modules:
    MVDiffusionInference  - Single image → 6-view generation
    GSLRMInference        - 6-view → 3D Gaussian reconstruction
    EndToEndPipeline      - Chain both pipelines
"""

from mouse_extensions.inference.mvdiffusion_pipeline import MVDiffusionInference
from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
from mouse_extensions.inference.end_to_end import EndToEndPipeline

__all__ = ["MVDiffusionInference", "GSLRMInference", "EndToEndPipeline"]
