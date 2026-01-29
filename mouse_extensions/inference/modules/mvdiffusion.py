"""MVDiffusion module: 1-view -> 6-view generation."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from mouse_extensions.inference.checkpoint_utils import find_mvdiffusion_checkpoint


class MVDiffusionModule:
    """MVDiffusion inference module."""

    def __init__(
        self,
        checkpoint: str,
        base_pipeline_path: str = "checkpoints/mvdiffusion/pipeckpts",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        prefer_ema: bool = True,
    ):
        self.device = device
        self.checkpoint_path = find_mvdiffusion_checkpoint(checkpoint)
        self.base_pipeline_path = base_pipeline_path
        self.dtype = dtype
        self.prefer_ema = prefer_ema
        self.pipeline = None

    def load(self):
        """Lazy load the model."""
        if self.pipeline is not None:
            return

        from mouse_extensions.inference.mvdiffusion_pipeline import MVDiffusionInference

        self.pipeline = MVDiffusionInference(
            checkpoint_path=self.checkpoint_path,
            base_pipeline_path=self.base_pipeline_path,
            device=self.device,
            dtype=self.dtype,
            prefer_ema=self.prefer_ema,
        )

    def generate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: int = 42,
    ) -> list[Image.Image]:
        """Generate 6 views from single image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            num_steps: Diffusion steps
            guidance_scale: CFG scale
            seed: Random seed
            
        Returns:
            List of 6 PIL Images
        """
        self.load()

        # Convert to PIL if needed
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        return self.pipeline.generate(
            image,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def unload(self):
        """Free GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
