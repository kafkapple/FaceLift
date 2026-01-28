"""MVDiffusion inference pipeline for multi-view generation.

Loads a fine-tuned MVDiffusion model and generates 6 views from a single input image.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# Camera configuration: 6 views evenly spaced in azimuth
MVDIFFUSION_AZIMUTHS = [0, 60, 120, 180, 240, 300]
MVDIFFUSION_ELEVATION = 0


class MVDiffusionInference:
    """MVDiffusion pipeline: single image → 6 multi-view images."""

    def __init__(
        self,
        checkpoint_path: str,
        base_pipeline_path: str = "checkpoints/mvdiffusion/pipeckpts",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        prefer_ema: bool = True,
    ):
        """Load MVDiffusion pipeline and replace UNet with trained weights.

        Args:
            checkpoint_path: Training checkpoint dir (contains unet/ subfolder).
            base_pipeline_path: Base StableUnCLIP pipeline path.
            device: Torch device.
            dtype: Model dtype (float16 recommended).
            prefer_ema: If True, use unet_ema/ weights when available.
        """
        from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import (
            StableUnCLIPImg2ImgPipeline,
        )
        from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

        checkpoint_path = Path(checkpoint_path)

        # Determine UNet path: prefer EMA if available
        unet_ema_path = checkpoint_path / "unet_ema"
        unet_path = checkpoint_path / "unet"
        if prefer_ema and unet_ema_path.exists():
            selected_unet_path = unet_ema_path
            print(f"Using EMA UNet from {unet_ema_path}")
        elif unet_path.exists():
            selected_unet_path = unet_path
        else:
            raise FileNotFoundError(
                f"No unet/ or unet_ema/ found in {checkpoint_path}"
            )

        # Load base pipeline
        print(f"Loading base MVDiffusion pipeline from {base_pipeline_path}...")
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            base_pipeline_path, torch_dtype=dtype
        )

        # Replace UNet with trained weights
        print(f"Loading trained UNet from {selected_unet_path}...")
        trained_unet = UNetMV2DConditionModel.from_pretrained(
            str(selected_unet_path), torch_dtype=dtype
        )
        self.pipe.unet = trained_unet
        self.pipe.to(device)

        # Enable memory optimization
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.device = device
        self.dtype = dtype
        self._prompt_embeds: Optional[torch.Tensor] = None

    def load_prompt_embeds(self, path: str) -> None:
        """Load pre-computed prompt embeddings.

        Args:
            path: Path to clr_embeds.pt file.
        """
        self._prompt_embeds = torch.load(path, weights_only=True).to(
            self.device, dtype=self.dtype
        )
        print(f"Loaded prompt embeddings: {self._prompt_embeds.shape}")

    def generate_views(
        self,
        input_image: str | Image.Image,
        image_size: int = 512,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: int = 42,
        prompt_embed_path: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate 6 multi-view images from a single input.

        Args:
            input_image: Path or PIL Image.
            image_size: Output resolution.
            num_steps: Diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed.
            prompt_embed_path: Path to prompt embeddings (optional, uses cached if loaded).

        Returns:
            Generated views tensor [6, C, H, W] in [0, 1].
        """
        # Load prompt embeddings if needed
        if prompt_embed_path and self._prompt_embeds is None:
            self.load_prompt_embeds(prompt_embed_path)

        if self._prompt_embeds is None:
            default_path = "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
            if Path(default_path).exists():
                self.load_prompt_embeds(default_path)
            else:
                print("Warning: No prompt embeddings, using zeros")
                self._prompt_embeds = torch.zeros(6, 77, 1024, device=self.device, dtype=self.dtype)

        # Load and preprocess image
        if isinstance(input_image, str):
            img = Image.open(input_image)
        else:
            img = input_image

        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.LANCZOS)

        # Replicate for 6 views
        input_tensor = TF.to_tensor(img).unsqueeze(0).repeat(6, 1, 1, 1)
        input_tensor = input_tensor.to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipe(
            image=input_tensor,
            prompt=[""] * 6,
            prompt_embeds=self._prompt_embeds,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pt",
        )

        return output.images  # [6, C, H, W]

    @staticmethod
    def compute_cameras(
        image_size: int = 512,
        device: str = "cuda",
        camera_distance: float = 2.7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fixed camera parameters for MVDiffusion 6-view layout.

        Returns:
            (c2w [6, 4, 4], fxfycxcy [6, 4])
        """
        fov = 50.0
        focal = image_size / (2 * np.tan(np.radians(fov / 2)))
        cx = cy = image_size / 2

        c2ws = []
        fxfycxcys = []

        for azim in MVDIFFUSION_AZIMUTHS:
            azim_rad = np.radians(azim)
            elev_rad = np.radians(MVDIFFUSION_ELEVATION)

            x = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
            y = camera_distance * np.sin(elev_rad)
            z = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)

            cam_pos = np.array([x, y, z])
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([1, 0, 0])
            right /= np.linalg.norm(right)
            up = np.cross(forward, right)

            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward  # OpenGL convention
            c2w[:3, 3] = cam_pos

            c2ws.append(c2w)
            fxfycxcys.append([focal, focal, cx, cy])

        c2ws = torch.from_numpy(np.array(c2ws)).float().to(device)
        fxfycxcys = torch.from_numpy(np.array(fxfycxcys)).float().to(device)

        return c2ws, fxfycxcys
