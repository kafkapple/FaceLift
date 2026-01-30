"""End-to-end inference: single image -> 6 views -> 3D Gaussians.

Chains MVDiffusionInference and GSLRMInference into one pipeline.
Optionally includes SAM-based preprocessing for raw input images.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from mouse_extensions.inference.mvdiffusion_pipeline import MVDiffusionInference
from mouse_extensions.inference.gslrm_pipeline import (
    GSLRMInference,
    load_sample_data,
)
from mouse_extensions.inference.checkpoint_utils import find_mvdiffusion_checkpoint


class EndToEndPipeline:
    """MVDiffusion + GS-LRM end-to-end pipeline with optional preprocessing.
    
    For raw input images (not in M5 format):
        - SAM-based mouse detection
        - Background removal (white)
        - Center alignment + coverage normalization
    
    For preprocessed images (already M5 format):
        - Auto-detection skips preprocessing
        - Or use skip_preprocess=True explicitly
    """

    def __init__(
        self,
        gslrm_config: str,
        gslrm_checkpoint: str,
        mvdiffusion_checkpoint: Optional[str] = None,
        mvdiffusion_base: str = "checkpoints/mvdiffusion/pipeckpts",
        prompt_embed_path: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 512,
        prefer_ema: bool = True,
        camera_json: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
    ):
        """Initialize pipelines.

        Args:
            gslrm_config: GS-LRM YAML config path.
            gslrm_checkpoint: GS-LRM checkpoint path (dir or .pt).
            mvdiffusion_checkpoint: MVDiffusion checkpoint dir (None = views-only mode).
            mvdiffusion_base: Base pipeline for MVDiffusion.
            prompt_embed_path: Pre-computed prompt embeddings.
            device: Torch device.
            image_size: Image resolution.
            prefer_ema: Use EMA UNet weights if available.
            camera_json: Path to opencv_cameras.json for E2E camera params.
                         If None, uses M5 default cameras.
            sam_checkpoint: Path to SAM checkpoint for preprocessing.
                           If None, preprocessing uses simple resize fallback.
        """
        self.device = device
        self.image_size = image_size
        self.camera_json = camera_json

        # Always load GS-LRM
        print("=== Loading GS-LRM ===")
        self.gslrm = GSLRMInference(
            config_path=gslrm_config,
            checkpoint_path=gslrm_checkpoint,
            device=device,
            image_size=image_size,
        )

        # Optionally load MVDiffusion
        self.mvdiff: Optional[MVDiffusionInference] = None
        if mvdiffusion_checkpoint:
            print("=== Loading MVDiffusion ===")
            self.mvdiff = MVDiffusionInference(
                checkpoint_path=find_mvdiffusion_checkpoint(mvdiffusion_checkpoint),
                base_pipeline_path=mvdiffusion_base,
                device=device,
                prefer_ema=prefer_ema,
            )
            if prompt_embed_path:
                self.mvdiff.load_prompt_embeds(prompt_embed_path)
                
        # Optionally load preprocessor
        self.preprocessor = None
        if sam_checkpoint:
            print("=== Loading Preprocessor (SAM) ===")
            from mouse_extensions.inference.preprocessing import (
                MouseInferencePreprocessor,
            )
            self.preprocessor = MouseInferencePreprocessor(
                sam_checkpoint=sam_checkpoint,
                device=device,
            )
            print(f"  SAM available: {self.preprocessor.detector.is_available}")

    def run(
        self,
        input_image: str,
        output_dir: str,
        num_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: int = 42,
        save_turntable: bool = True,
        save_mesh: bool = True,
        turntable_views: int = 120,
        skip_preprocess: bool = False,
        save_preprocess_steps: bool = False,
    ) -> Path:
        """Run full pipeline: single image -> 6 views -> 3D -> save.

        Args:
            input_image: Path to single input image.
            output_dir: Output directory.
            num_steps: MVDiffusion diffusion steps.
            guidance_scale: MVDiffusion guidance scale.
            seed: Random seed.
            save_turntable: Generate turntable video.
            save_mesh: Save PLY.
            turntable_views: Number of turntable frames.
            skip_preprocess: If True, skip preprocessing (for M5-format images).
            save_preprocess_steps: If True, save visualization of preprocessing steps.

        Returns:
            Output path.
        """
        if self.mvdiff is None:
            raise RuntimeError(
                "MVDiffusion not loaded. Use run_from_views() for 6-view input."
            )

        sample_name = Path(input_image).stem
        out = Path(output_dir) / sample_name
        out.mkdir(parents=True, exist_ok=True)

        # Step 0: Preprocessing (optional)
        if not skip_preprocess and self.preprocessor is not None:
            print("[0/2] Preprocessing input image...")
            
            if save_preprocess_steps:
                preprocess_vis_dir = out / "preprocessing_steps"
                self.preprocessor.visualize_steps(input_image, preprocess_vis_dir)
                
            result = self.preprocessor.preprocess(input_image)
            
            if result.detection_used:
                print(f"  Detection used: scale={result.scale_applied:.2f}, "
                      f"centroid=({result.centroid[0]:.1f}, {result.centroid[1]:.1f})")
            else:
                print("  Using fallback (simple resize or already preprocessed)")
                
            # Save preprocessed image
            preprocessed_path = out / "preprocessed_input.png"
            Image.fromarray(result.image).save(preprocessed_path)
            print(f"  Saved preprocessed image to {preprocessed_path}")
            
            # Use preprocessed image for MVDiffusion
            mvdiff_input = preprocessed_path
        else:
            mvdiff_input = input_image

        # Step 1: Generate 6 views
        print(f"[1/2] Generating 6 views from {Path(mvdiff_input).name}...")
        views = self.mvdiff.generate_views(
            mvdiff_input,
            image_size=self.image_size,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )  # [6, C, H, W]

        # Save generated views
        views_dir = out / "generated_views"
        views_dir.mkdir(exist_ok=True)
        for i, v in enumerate(views):
            v_np = (v.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(v_np).save(views_dir / f"view_{i:02d}.png")
        print(f"  Saved views to {views_dir}")

        # Step 2: GS-LRM reconstruction
        # Use actual training camera parameters (from camera_json or M5 default)
        print("[2/2] Running GS-LRM reconstruction...")
        c2ws, fxfycxcys = MVDiffusionInference.compute_cameras(
            self.image_size, self.device, camera_json=self.camera_json
        )

        images = views.unsqueeze(0)  # [1, 6, C, H, W]
        c2ws = c2ws.unsqueeze(0)
        fxfycxcys = fxfycxcys.unsqueeze(0)

        # Index: (view_idx, scene_idx)
        index = torch.stack([
            torch.arange(6).long(),
            torch.zeros(6).long(),
        ], dim=-1).unsqueeze(0).to(self.device)

        result = self.gslrm.predict(images, c2ws, fxfycxcys, index)

        return self.gslrm.save_outputs(
            result,
            output_dir,
            sample_name,
            save_turntable=save_turntable,
            save_mesh=save_mesh,
            turntable_views=turntable_views,
            image_size=self.image_size,
        )

    def run_from_views(
        self,
        sample_dir: str,
        output_dir: str,
        save_turntable: bool = True,
        save_mesh: bool = True,
        turntable_views: int = 120,
    ) -> Path:
        """Run GS-LRM only from a 6-view sample directory.

        Uses camera parameters from the sample's opencv_cameras.json.
        No preprocessing needed (views are already generated).

        Args:
            sample_dir: Directory with images/ and opencv_cameras.json.
            output_dir: Output directory.
            save_turntable: Generate turntable video.
            save_mesh: Save PLY.
            turntable_views: Number of turntable frames.

        Returns:
            Output path.
        """
        sample_name = Path(sample_dir).name
        print(f"Running GS-LRM on {sample_name}...")

        images, c2ws, fxfycxcys, index = load_sample_data(
            sample_dir, self.image_size, self.device
        )
        result = self.gslrm.predict(images, c2ws, fxfycxcys, index)

        return self.gslrm.save_outputs(
            result,
            output_dir,
            sample_name,
            save_turntable=save_turntable,
            save_mesh=save_mesh,
            turntable_views=turntable_views,
            image_size=self.image_size,
        )
