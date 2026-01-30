"""Mouse inference preprocessing module.

Provides preprocessing pipeline for arbitrary input images to match M5 training format:
- SAM-based mouse detection
- Background removal (white background)
- Center alignment (centroid -> 256, 256)
- Coverage normalization (~6% target)

Usage:
    from mouse_extensions.inference.preprocessing import MouseInferencePreprocessor
    
    preprocessor = MouseInferencePreprocessor(sam_checkpoint="path/to/sam.pth")
    result = preprocessor.preprocess(image)  # {"image": np.ndarray, "mask": np.ndarray}
"""

from mouse_extensions.inference.preprocessing.mouse_inference_preprocessor import (
    MouseInferencePreprocessor,
)
from mouse_extensions.inference.preprocessing.config import MousePreprocessConfig

__all__ = ["MouseInferencePreprocessor", "MousePreprocessConfig"]
