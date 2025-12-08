#!/usr/bin/env python3
"""
Zero123++ Pipeline Wrapper for Mouse-FaceLift

Provides single-image to multi-view generation using Zero123++ v1.2
as an alternative to MVDiffusion for mouse data.

Zero123++ generates 6 views in a tiled 3x2 layout:
- Row 1: 30°, 90°, 150° azimuth at 20° elevation
- Row 2: 210°, 270°, 330° azimuth at -10° elevation

Usage:
    from mvdiffusion.pipelines.zero123pp_pipeline import Zero123PlusPipeline

    pipeline = Zero123PlusPipeline.from_pretrained()
    views = pipeline.generate_views(input_image)

References:
    - Paper: https://arxiv.org/abs/2310.15110
    - Model: https://huggingface.co/sudo-ai/zero123plus-v1.2
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline

# Zero123++ camera configuration (fixed poses)
# Azimuth angles in degrees (relative to input view)
ZERO123PP_AZIMUTHS = [30, 90, 150, 210, 270, 330]
# Elevation angles in degrees
ZERO123PP_ELEVATIONS = [20, 20, 20, -10, -10, -10]
# Image size for each view
ZERO123PP_VIEW_SIZE = 320  # Each view in the 3x2 grid


class Zero123PlusPipeline:
    """
    Wrapper for Zero123++ multi-view generation.

    Generates 6 consistent views from a single input image.
    """

    MODEL_ID = "sudo-ai/zero123plus-v1.2"

    def __init__(
        self,
        pipe: DiffusionPipeline,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the pipeline.

        Args:
            pipe: Loaded diffusers pipeline
            device: Device to run on
            dtype: Data type for inference
        """
        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        # Camera parameters for each view (approximate)
        self.camera_params = self._compute_camera_params()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None
    ) -> "Zero123PlusPipeline":
        """
        Load Zero123++ from HuggingFace.

        Args:
            model_id: HuggingFace model ID (default: sudo-ai/zero123plus-v1.2)
            device: Device to run on
            dtype: Data type
            cache_dir: Cache directory for downloaded weights

        Returns:
            Initialized pipeline
        """
        model_id = model_id or cls.MODEL_ID

        print(f"Loading Zero123++ from {model_id}...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        pipe.to(device)

        # Enable memory optimizations
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("  Enabled xformers attention")
            except Exception as e:
                print(f"  Could not enable xformers: {e}")

        print("Zero123++ loaded successfully")
        return cls(pipe, device, dtype)

    def _compute_camera_params(self) -> List[Dict]:
        """
        Compute approximate camera parameters for Zero123++ views.

        Returns:
            List of camera parameter dicts for each view
        """
        params = []

        # Assume unit distance from origin, looking at center
        for i, (azim, elev) in enumerate(zip(ZERO123PP_AZIMUTHS, ZERO123PP_ELEVATIONS)):
            azim_rad = np.radians(azim)
            elev_rad = np.radians(elev)

            # Camera position on unit sphere
            x = np.cos(elev_rad) * np.sin(azim_rad)
            y = np.sin(elev_rad)
            z = np.cos(elev_rad) * np.cos(azim_rad)

            # Camera-to-world matrix (look-at origin)
            cam_pos = np.array([x, y, z])
            forward = -cam_pos / np.linalg.norm(cam_pos)
            right = np.cross(np.array([0, 1, 0]), forward)
            right = right / np.linalg.norm(right)
            up = np.cross(forward, right)

            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward  # OpenGL convention
            c2w[:3, 3] = cam_pos

            # Approximate intrinsics (assuming ~50° FOV)
            focal = ZERO123PP_VIEW_SIZE / (2 * np.tan(np.radians(25)))
            cx = ZERO123PP_VIEW_SIZE / 2
            cy = ZERO123PP_VIEW_SIZE / 2

            params.append({
                'c2w': c2w,
                'w2c': np.linalg.inv(c2w),
                'fxfycxcy': [focal, focal, cx, cy],
                'azimuth': azim,
                'elevation': elev,
                'view_idx': i
            })

        return params

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        size: int = 320,
        bg_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """
        Preprocess input image for Zero123++.

        Args:
            image: Input image (PIL, numpy, or path)
            size: Output size (square)
            bg_color: Background color for RGBA images

        Returns:
            Preprocessed PIL image
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Handle RGBA
        if image.mode == 'RGBA':
            bg = Image.new('RGBA', image.size, (*bg_color, 255))
            image = Image.alpha_composite(bg, image).convert('RGB')
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize (center crop + resize)
        # Zero123++ expects square input
        w, h = image.size
        if w != h:
            # Center crop to square
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))

        image = image.resize((size, size), Image.LANCZOS)
        return image

    def extract_views(
        self,
        tiled_image: Image.Image,
        view_size: int = 320
    ) -> List[Image.Image]:
        """
        Extract individual views from Zero123++ tiled output.

        Zero123++ outputs a 3x2 grid of views (960x640 total).

        Args:
            tiled_image: Tiled output image (3 columns, 2 rows)
            view_size: Size of each view

        Returns:
            List of 6 PIL images
        """
        views = []

        # Grid: 3 columns, 2 rows
        for row in range(2):
            for col in range(3):
                left = col * view_size
                top = row * view_size
                right = left + view_size
                bottom = top + view_size

                view = tiled_image.crop((left, top, right, bottom))
                views.append(view)

        return views

    @torch.no_grad()
    def generate_views(
        self,
        image: Union[Image.Image, np.ndarray, str],
        num_inference_steps: int = 75,
        guidance_scale: float = 4.0,
        output_size: int = 512,
        seed: Optional[int] = None
    ) -> Tuple[List[Image.Image], List[Dict]]:
        """
        Generate 6 multi-view images from a single input.

        Args:
            image: Input image
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale
            output_size: Output view size (will resize from 320)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (list of 6 PIL images, list of camera parameters)
        """
        # Preprocess
        input_image = self.preprocess_image(image)

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate
        result = self.pipe(
            input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        # Extract tiled image
        tiled_output = result.images[0]

        # Extract individual views
        views = self.extract_views(tiled_output)

        # Resize to output size if needed
        if output_size != ZERO123PP_VIEW_SIZE:
            views = [v.resize((output_size, output_size), Image.LANCZOS) for v in views]

            # Update camera params for new size
            scale = output_size / ZERO123PP_VIEW_SIZE
            camera_params = []
            for p in self.camera_params:
                fx, fy, cx, cy = p['fxfycxcy']
                camera_params.append({
                    **p,
                    'fxfycxcy': [fx * scale, fy * scale, cx * scale, cy * scale]
                })
        else:
            camera_params = self.camera_params

        return views, camera_params

    def generate_for_gslrm(
        self,
        image: Union[Image.Image, np.ndarray, str],
        output_size: int = 512,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate views in format ready for GSLRM.

        Args:
            image: Input image
            output_size: Output size
            **kwargs: Additional args for generate_views

        Returns:
            Dict with 'image', 'c2w', 'fxfycxcy', 'index' tensors
        """
        views, camera_params = self.generate_views(image, output_size=output_size, **kwargs)

        # Convert to tensors
        images = []
        c2ws = []
        fxfycxcys = []

        for view, cam in zip(views, camera_params):
            # Image to tensor [C, H, W]
            img_np = np.array(view).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            images.append(img_tensor)

            c2ws.append(torch.from_numpy(cam['c2w']).float())
            fxfycxcys.append(torch.tensor(cam['fxfycxcy']).float())

        # Stack and add batch dimension
        images = torch.stack(images).unsqueeze(0)  # [1, V, C, H, W]
        c2ws = torch.stack(c2ws).unsqueeze(0)  # [1, V, 4, 4]
        fxfycxcys = torch.stack(fxfycxcys).unsqueeze(0)  # [1, V, 4]

        # Index tensor
        num_views = len(views)
        index = torch.stack([
            torch.zeros(num_views).long(),
            torch.arange(num_views).long()
        ], dim=-1).unsqueeze(0)  # [1, V, 2]

        return {
            'image': images.to(self.device),
            'c2w': c2ws.to(self.device),
            'fxfycxcy': fxfycxcys.to(self.device),
            'index': index.to(self.device)
        }

    def save_views(
        self,
        views: List[Image.Image],
        output_dir: str,
        prefix: str = "view"
    ) -> List[str]:
        """
        Save generated views to disk.

        Args:
            views: List of PIL images
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        for i, view in enumerate(views):
            path = os.path.join(output_dir, f"{prefix}_{i:02d}.png")
            view.save(path)
            paths.append(path)

        # Also save grid view
        grid_path = os.path.join(output_dir, f"{prefix}_grid.png")
        self._save_grid(views, grid_path)
        paths.append(grid_path)

        return paths

    def _save_grid(self, views: List[Image.Image], path: str):
        """Save views as a grid image."""
        if not views:
            return

        w, h = views[0].size
        grid = Image.new('RGB', (w * 3, h * 2))

        for i, view in enumerate(views[:6]):
            row = i // 3
            col = i % 3
            grid.paste(view, (col * w, row * h))

        grid.save(path)


def test_pipeline():
    """Test Zero123++ pipeline with a sample image."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="outputs/zero123pp_test/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load pipeline
    pipeline = Zero123PlusPipeline.from_pretrained()

    # Generate views
    print(f"Generating views from {args.image}...")
    views, cameras = pipeline.generate_views(
        args.image,
        seed=args.seed,
        output_size=512
    )

    # Save
    paths = pipeline.save_views(views, args.output)
    print(f"Saved {len(paths)} images to {args.output}")

    # Print camera info
    print("\nCamera parameters:")
    for i, cam in enumerate(cameras):
        print(f"  View {i}: azim={cam['azimuth']}°, elev={cam['elevation']}°")


if __name__ == "__main__":
    test_pipeline()
