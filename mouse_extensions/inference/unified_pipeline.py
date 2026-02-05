"""Unified inference pipeline combining MVDiffusion and GS-LRM."""

import gc
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from .modules.mvdiffusion import MVDiffusionModule
from .modules.gslrm import GSLRMModule
from .modules.renderer import RendererModule
from .modules.exporter import ExporterModule


class UnifiedPipeline:
    """Unified inference pipeline for FaceLift.
    
    Supports:
    - Single image -> 6-view -> 3D Gaussians
    - 6-view sample -> 3D Gaussians
    - Batch processing with video output
    """

    def __init__(self, config: Union[str, Path, DictConfig]):
        """Initialize pipeline from config.
        
        Args:
            config: Path to YAML config or OmegaConf object
        """
        if isinstance(config, (str, Path)):
            self.config = self._load_config(config)
        else:
            self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize modules (lazy loaded)
        self.mvdiffusion = None
        self.gslrm = None
        self.renderer = None
        self.exporter = None
        
        self._init_modules()

    def _load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """Load config with base config support."""
        config_path = Path(config_path)
        config = OmegaConf.load(config_path)
        
        # Handle _base_ inheritance
        if "_base_" in config:
            base_path = config_path.parent / config._base_
            base_config = OmegaConf.load(base_path)
            del config._base_
            config = OmegaConf.merge(base_config, config)
        
        return config

    def _init_modules(self):
        """Initialize module instances (not loaded yet)."""
        cfg = self.config
        
        # MVDiffusion (optional)
        mvd_cfg = cfg.pipeline.mvdiffusion
        use_mvd = self._should_use_mvdiffusion()
        if use_mvd:
            self.mvdiffusion = MVDiffusionModule(
                checkpoint=mvd_cfg.checkpoint,
                device=self.device,
            )
        
        # GS-LRM (always required)
        gslrm_cfg = cfg.pipeline.gslrm
        self.gslrm = GSLRMModule(
            checkpoint=gslrm_cfg.checkpoint,
            config=gslrm_cfg.config,
            device=self.device,
        )
        
        # Renderer
        render_cfg = cfg.rendering
        self.renderer = RendererModule(
            resolution=render_cfg.resolution,
            num_views=render_cfg.num_views,
            elevation=render_cfg.elevation,
            radius=render_cfg.radius,
        )
        
        # Exporter
        out_cfg = cfg.output
        self.exporter = ExporterModule(
            output_dir=out_cfg.dir,
            save_gaussian=out_cfg.gaussian,
            save_rerun=out_cfg.rerun,
            video_fps=out_cfg.video.fps,
        )

    def _should_use_mvdiffusion(self) -> bool:
        """Determine if MVDiffusion should be used."""
        cfg = self.config
        mvd_enabled = cfg.pipeline.mvdiffusion.enabled
        
        if mvd_enabled == "auto":
            # Auto: use if view_idx is set
            return cfg.input.view_idx is not None
        return mvd_enabled

    def run(self) -> dict:
        """Run the full pipeline based on config.
        
        Returns:
            Dictionary with output paths and statistics
        """
        mode = self.config.input.mode
        
        if mode == "single_image":
            return self._run_single_image()
        elif mode == "single_sample":
            return self._run_single_sample()
        elif mode == "batch":
            return self._run_batch()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _run_single_image(self) -> dict:
        """Process single external image."""
        image_path = self.config.input.image_path
        if not image_path:
            raise ValueError("input.image_path required for single_image mode")
        
        image = Image.open(image_path)
        return self._process_with_mvdiffusion(image, "single")

    def _run_single_sample(self) -> dict:
        """Process single sample directory."""
        sample_dir = Path(self.config.input.sample_dir)
        if not sample_dir:
            raise ValueError("input.sample_dir required for single_sample mode")
        
        view_idx = self.config.input.view_idx
        
        if view_idx is not None:
            # 1-view mode: use MVDiffusion
            image_path = sample_dir / "images" / f"cam_{view_idx:03d}.png"
            image = Image.open(image_path)
            return self._process_with_mvdiffusion(image, sample_dir.name)
        else:
            # 6-view mode: use GS-LRM directly
            return self._process_6view_sample(sample_dir)

    def _run_batch(self) -> dict:
        """Process batch of samples with video output."""
        data_dir = Path(self.config.input.data_dir)
        if not data_dir:
            raise ValueError("input.data_dir required for batch mode")
        
        # Find sample directories
        samples = self._find_samples(data_dir)
        
        # Apply frame range
        fr = self.config.input.frame_range
        if fr.start is not None or fr.end is not None:
            start = int(fr.start) if fr.start is not None else 0
            end = int(fr.end) if fr.end is not None else len(samples)
            samples = samples[start:end]
        
        if fr.step > 1:
            samples = samples[::fr.step]
        
        print(f"Processing {len(samples)} samples...")
        
        view_idx = self.config.input.view_idx
        use_mvd = view_idx is not None
        
        all_turntables = []
        all_inputs = []
        all_gaussians = []
        frame_indices = []
        
        for sample_dir in tqdm(samples, desc="Inference"):
            try:
                if use_mvd:
                    # 1-view mode
                    image_path = sample_dir / "images" / f"cam_{view_idx:03d}.png"
                    if not image_path.exists():
                        continue
                    image = Image.open(image_path)
                    result = self._process_with_mvdiffusion(
                        image, 
                        sample_dir.name,
                        save_per_sample=self.config.output.get("per_sample", True),
                    )
                else:
                    # 6-view mode
                    result = self._process_6view_sample(
                        sample_dir,
                        save_per_sample=self.config.output.get("per_sample", True),
                    )
                
                all_turntables.append(result["turntable_frames"])
                all_inputs.append(result.get("input_views"))
                all_gaussians.append(result["gaussians"])
                frame_indices.append(int(sample_dir.name))
                
            except Exception as e:
                print(f"Error processing {sample_dir}: {e}")
                continue
            
            # Clear cache periodically
            gc.collect()
            torch.cuda.empty_cache()
        
        if not all_turntables:
            return {"error": "No samples processed"}
        
        # Generate outputs
        return self._generate_batch_outputs(
            all_turntables,
            all_inputs,
            all_gaussians,
            frame_indices,
        )

    def _find_samples(self, data_dir: Path) -> list[Path]:
        """Find and sort sample directories.
        
        If config.input.data_list is set, filter samples by that list.
        """
        # Check for data_list filter
        data_list_path = self.config.input.get("data_list")
        valid_samples = None
        
        if data_list_path:
            from pathlib import Path as P
            list_path = P(data_list_path).expanduser()
            if list_path.exists():
                with open(list_path) as f:
                    # Parse format: /path/to/sample or sample_id
                    valid_samples = set()
                    for line in f:
                        line = line.strip()
                        if line:
                            # Extract sample ID from path
                            sample_id = P(line).name
                            valid_samples.add(sample_id)
                print(f"Using data_list filter: {len(valid_samples)} samples")
        
        samples = []
        for d in sorted(data_dir.iterdir()):
            if d.is_dir() and d.name.isdigit():
                if valid_samples is None or d.name in valid_samples:
                    samples.append(d)
        return samples

    def _process_with_mvdiffusion(
        self,
        image: Image.Image,
        name: str,
        save_per_sample: bool = True,
    ) -> dict:
        """Process single image through MVDiffusion + GS-LRM."""
        cfg = self.config.pipeline.mvdiffusion
        
        # Generate 6 views
        views = self.mvdiffusion.generate(
            image,
            num_steps=cfg.num_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed,
        )
        
        # Load camera params (from default or sample)
        c2ws, fxfycxcy = self._get_default_cameras()
        
        # Run GS-LRM
        output = self.gslrm.process_images(views, c2ws, fxfycxcy)
        gaussians = output.gaussians[0]
        
        # Render turntable
        turntable_frames = self.renderer.render_turntable(gaussians)
        
        # Convert PIL views to numpy for grid_6view
        import numpy as np
        views_np = np.stack([np.array(v) for v in views])
        
        result = {
            "gaussians": gaussians,
            "turntable_frames": turntable_frames,
            "generated_views": views,
            "input_views": views_np,  # For grid_6view (shows MVD output)
        }
        
        if save_per_sample:
            # Save outputs
            sample_dir = Path(self.config.output.dir) / name
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            exporter = ExporterModule(
                output_dir=sample_dir,
                save_gaussian=self.config.output.gaussian,
                save_rerun=self.config.output.rerun,
                video_fps=self.config.output.video.fps,
            )
            
            exporter.save_gaussians_ply(gaussians)
            exporter.save_grid(turntable_frames, "turntable_grid")
            
            # Save generated views
            views_dir = sample_dir / "generated_views"
            views_dir.mkdir(exist_ok=True)
            for i, view in enumerate(views):
                view.save(views_dir / f"view_{i:02d}.png")
        
        return result

    def _process_6view_sample(
        self,
        sample_dir: Path,
        save_per_sample: bool = True,
    ) -> dict:
        """Process 6-view sample through GS-LRM."""
        from mouse_extensions.inference.gslrm_pipeline import load_sample_data
        
        images, c2ws, fxfycxcy, index = load_sample_data(sample_dir, device=self.device)
        from easydict import EasyDict as edict
        sample = edict({"image": images, "c2w": c2ws, "fxfycxcy": fxfycxcy, "index": index})
        
        # Run GS-LRM
        output = self.gslrm.forward(sample)
        gaussians = output.gaussians[0]
        
        # Render turntable
        turntable_frames = self.renderer.render_turntable(gaussians)
        
        # Load input views for reference
        input_views = self._load_input_views(sample_dir)
        
        result = {
            "gaussians": gaussians,
            "turntable_frames": turntable_frames,
            "input_views": input_views,
        }
        
        if save_per_sample:
            out_dir = Path(self.config.output.dir) / sample_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            exporter = ExporterModule(
                output_dir=out_dir,
                save_gaussian=self.config.output.gaussian,
                save_rerun=self.config.output.rerun,
            )
            
            exporter.save_gaussians_ply(gaussians)
            exporter.save_grid(turntable_frames, "turntable_grid")
        
        return result

    def _load_input_views(self, sample_dir: Path) -> np.ndarray:
        """Load 6 input camera views."""
        import cv2
        
        views = []
        for i in range(6):
            img_path = sample_dir / "images" / f"cam_{i:03d}.png"
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                views.append(img)
        
        return np.stack(views) if views else None

    def _get_default_cameras(self, image_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Get camera parameters for MVDiffusion output from M5 default cameras."""
        from mouse_extensions.inference.mvdiffusion_pipeline import MVDiffusionInference
        
        c2ws, fxfycxcy = MVDiffusionInference.compute_cameras(
            image_size=image_size,
            device=self.device,
        )
        return c2ws.cpu().numpy(), fxfycxcy.cpu().numpy()

    def _generate_batch_outputs(
        self,
        all_turntables: list[np.ndarray],
        all_inputs: list[np.ndarray],
        all_gaussians: list[torch.Tensor],
        frame_indices: list[int],
    ) -> dict:
        """Generate combined outputs for batch processing."""
        cfg = self.config.output
        video_cfg = cfg.video
        
        T = len(all_turntables)
        V = all_turntables[0].shape[0]
        
        print(f"Generating outputs: T={T}, V={V}")
        
        outputs = {"frame_count": T, "view_count": V}
        
        # Videos
        if video_cfg.enabled:
            video_types = video_cfg.types
            
            if "turntable" in video_types:
                # First frame turntable
                # DISABLED: self.exporter.save_video(all_turntables[0], "turntable_first")
                # outputs["turntable_first"] = str(self.exporter.output_dir / "turntable_first.mp4")
                pass  # turntable_first disabled
            
            if "time_fixed" in video_types:
                # Fixed angle over time
                for angle_idx in self.config.rendering.fixed_angles:
                    angle_idx = angle_idx % V
                    fixed_frames = np.stack([t[angle_idx] for t in all_turntables])
                    suffix = f"_angle{angle_idx}" if len(self.config.rendering.fixed_angles) > 1 else ""
                    self.exporter.save_video(fixed_frames, f"time_fixed{suffix}")
            
            if "time_rotating" in video_types:
                # Rotating over time
                rotating = []
                for t in range(T):
                    angle = (t * V // T) % V
                    rotating.append(all_turntables[t][angle])
                self.exporter.save_video(np.stack(rotating), "time_rotating")
            
            if "full_all" in video_types:
                # All frames
                full = np.concatenate(all_turntables, axis=0)
                self.exporter.save_video(full, "full_all")
            
            if "grid_6view" in video_types and all_inputs[0] is not None:
                # Input 6-view grid video
                grid_frames = []
                for inputs in all_inputs:
                    if inputs is not None:
                        grid = self._make_grid(inputs, cols=3)
                        grid_frames.append(grid)
                if grid_frames:
                    self.exporter.save_video(np.stack(grid_frames), "grid_6view")
        
        # Grid images
        if cfg.grid.enabled:
            self.exporter.save_grid(
                all_turntables[0],
                "grid_first",
                cols=cfg.grid.cols,
            )
        
        # Gaussians
        if cfg.gaussian:
            gaussians_dir = self.exporter.output_dir / "gaussians"
            gaussians_dir.mkdir(exist_ok=True)
            for i, (gaussians, idx) in enumerate(zip(all_gaussians, frame_indices)):
                self.exporter.save_gaussians_ply(
                    gaussians,
                    gaussians_dir / f"frame_{idx:06d}.ply",
                )
        
        # Rerun
        if cfg.rerun:
            self.exporter.save_rerun_rrd(all_gaussians, frame_indices)
        
        outputs["output_dir"] = str(self.exporter.output_dir)
        return outputs

    def _make_grid(self, images: np.ndarray, cols: int = 3) -> np.ndarray:
        """Make grid from images."""
        n, h, w, c = images.shape
        rows = (n + cols - 1) // cols
        
        pad_count = rows * cols - n
        if pad_count > 0:
            padding = np.ones((pad_count, h, w, c), dtype=images.dtype) * 255
            images = np.concatenate([images, padding], axis=0)
        
        grid = images.reshape(rows, cols, h, w, c)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, c)
        return grid

    def cleanup(self):
        """Release GPU memory."""
        if self.mvdiffusion:
            self.mvdiffusion.unload()
        if self.gslrm:
            self.gslrm.unload()
        gc.collect()
        torch.cuda.empty_cache()
