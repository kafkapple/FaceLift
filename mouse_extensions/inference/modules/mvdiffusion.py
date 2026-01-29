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
        image_size: int = 512,
    ) -> list[Image.Image]:
        """Generate 6 views from single image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            num_steps: Diffusion steps
            guidance_scale: CFG scale
            seed: Random seed
            image_size: Output resolution
            
        Returns:
            List of 6 PIL Images
        """
        self.load()

        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Generate views - returns tensor [6, C, H, W] in [0, 1]
        views_tensor = self.pipeline.generate_views(
            input_image=image,
            image_size=image_size,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        # Convert tensor to list of PIL Images
        views = []
        for i in range(views_tensor.shape[0]):
            # [C, H, W] -> [H, W, C], scale to 0-255
            img_np = views_tensor[i].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            views.append(Image.fromarray(img_np))

        return views

    def unload(self):
        """Free GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache()
