"""MVDiffusion inference pipeline for multi-view generation.

Loads a fine-tuned MVDiffusion model and generates 6 views from a single input image.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# Default camera configuration: M5 fixed 6-camera rig
# ---------------------------------------------------------------------------
# These are the actual calibrated camera poses from the M5 preprocessed
# dataset. All M5 frames share identical extrinsics (6 fixed cameras).
#
# Camera layout (non-uniform elevation, non-uniform azimuth):
#   Cam 0: elev=-34.5 deg, azim=-99.6 deg, dist=2.799
#   Cam 1: elev=+32.0 deg, azim=+81.6 deg, dist=2.739
#   Cam 2: elev=+80.6 deg, azim=-135.2 deg, dist=2.706  (near top-down)
#   Cam 3: elev=-23.3 deg, azim=+99.3 deg, dist=2.593
#   Cam 4: elev=+21.6 deg, azim=-83.1 deg, dist=2.757
#   Cam 5: elev=-75.6 deg, azim=+48.0 deg, dist=2.607   (near bottom-up)
#
# Intrinsics: fx=fy=549.0, cx=cy=256.0 (at 512x512)
_DEFAULT_CAMERA_JSON = Path(__file__).parent / "cameras" / "m5_cameras.json"


def _load_cameras_from_json(
    camera_json_path: str,
    image_size: int = 512,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load camera parameters from an opencv_cameras.json file.

    Args:
        camera_json_path: Path to opencv_cameras.json.
        image_size: Target image resolution (for intrinsics scaling).
        device: Torch device.

    Returns:
        (c2w [V, 4, 4], fxfycxcy [V, 4])
    """
    with open(camera_json_path, "r") as f:
        camera_data = json.load(f)

    frames = camera_data["frames"]
    c2ws = []
    fxfycxcys = []

    for frame in frames:
        w2c = np.array(frame["w2c"])
        c2ws.append(np.linalg.inv(w2c))

        scale = image_size / frame.get("w", image_size)
        fxfycxcys.append([
            frame["fx"] * scale,
            frame["fy"] * scale,
            frame["cx"] * scale,
            frame["cy"] * scale,
        ])

    c2ws_t = torch.from_numpy(np.array(c2ws)).float().to(device)
    fxfycxcys_t = torch.tensor(fxfycxcys, dtype=torch.float32).to(device)

    return c2ws_t, fxfycxcys_t


class MVDiffusionInference:
    """MVDiffusion pipeline: single image -> 6 multi-view images."""

    def __init__(
        self,
        checkpoint_path: str,
        base_pipeline_path: str = "checkpoints/mvdiffusion/pipeckpts",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        prefer_ema: bool = True,
        pose_config: Optional[dict] = None,
        pose_weights_path: Optional[str] = None,
    ):
        """Load MVDiffusion pipeline and replace UNet with trained weights.

        Args:
            checkpoint_path: Training checkpoint dir (contains unet/ subfolder).
            base_pipeline_path: Base StableUnCLIP pipeline path.
            device: Torch device.
            dtype: Model dtype (float16 recommended).
            prefer_ema: If True, use unet_ema/ weights when available.
            pose_config: Optional dict to configure pose conditioning injector.
            pose_weights_path: Optional path to trained pose encoder weights (.pt).
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
        # Sync pipeline num_views with UNet config
        if hasattr(trained_unet, 'config') and hasattr(trained_unet.config, 'num_views'):
            self.pipe.num_views = trained_unet.config.num_views
            print(f'  Pipeline num_views synced to {self.pipe.num_views}')
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

        # Pose conditioning injector (optional)
        self.pose_injector = None
        if pose_config and pose_config.get("enabled", False):
            from mouse_extensions.model.pose_conditioning_integration import (
                PoseConditioningInjector,
            )
            self.pose_injector = PoseConditioningInjector(
                method=pose_config.get("method", "extrinsic"),
                integration=pose_config.get("integration", "add"),
                embed_dim=pose_config.get("embed_dim", 1024),
                camera_json_path=pose_config.get(
                    "camera_json",
                    str(Path(__file__).parent / "cameras" / "m5_cameras.json")
                ),
                plucker_resolution=pose_config.get("plucker_resolution", 64),
                trainable=False,  # Always frozen for inference
                spatial_token_size=pose_config.get("spatial_token_size", 8),
            ).to(device)
            if pose_weights_path and Path(pose_weights_path).exists():
                state_dict = torch.load(
                    pose_weights_path, map_location=device, weights_only=True
                )
                self.pose_injector.load_state_dict(state_dict)
                print(f"Loaded trained pose encoder from {pose_weights_path}")
            else:
                print(f"[Pose] Using fresh encoder (no trained weights)")

    def load_prompt_embeds(self, path: str) -> None:
        """Load pre-computed prompt embeddings.

        Args:
            path: Path to clr_embeds.pt file or directory containing it.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "clr_embeds.pt"
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
            Generated views tensor [V, C, H, W] in [0, 1], V from prompt embeddings.
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

        # Determine number of views from prompt embeddings
        n_views = self._prompt_embeds.shape[0] if self._prompt_embeds is not None else 6

        # Replicate for n views
        input_tensor = TF.to_tensor(img).unsqueeze(0).repeat(n_views, 1, 1, 1)
        input_tensor = input_tensor.to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Inject pose conditioning into prompt embeddings if available
        active_prompt_embeds = self._prompt_embeds
        if self.pose_injector is not None:
            self.pose_injector.eval()
            inject_fn = (self.pose_injector.inject_spatial
                         if self.pose_injector.integration == 'spatial_token'
                         else self.pose_injector.inject)
            active_prompt_embeds = inject_fn(
                active_prompt_embeds,
                ref_view_idx=0,  # Fixed reference for inference
                n_views=n_views,
                batch_size=1,
            )

        output = self.pipe(
            image=input_tensor,
            prompt=[""] * n_views,
            prompt_embeds=active_prompt_embeds,
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
        camera_json: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return camera parameters for GS-LRM reconstruction.

        Default: loads the M5 fixed 6-camera rig from
        ``mouse_extensions/inference/cameras/m5_cameras.json``.
        Override with ``camera_json`` for other camera setups.

        Args:
            image_size: Image resolution (intrinsics scale with this).
            device: Torch device.
            camera_json: Path to opencv_cameras.json. If None, uses M5 default.

        Returns:
            (c2w [6, 4, 4], fxfycxcy [6, 4])
        """
        json_path = camera_json or str(_DEFAULT_CAMERA_JSON)
        return _load_cameras_from_json(json_path, image_size, device)
