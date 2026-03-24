"""E2E Temporal Pipeline: Multi-frame sequence → Temporally consistent 3D.

Pipeline:
    Input frames → GS-LRM (per-frame) → Bilateral Smoothing → Temporal Video

Key difference from V1 (autoregressive):
    - V1: Frame 0 Gaussian + cumulative deformation → drift/melting
    - V2: Per-frame Gaussian + post-hoc smoothing → bounded drift
"""

from pathlib import Path
from typing import Optional, List
import json

import numpy as np
import torch
from tqdm import tqdm


class TemporalE2EPipeline:
    """End-to-end temporal inference with bilateral smoothing."""

    def __init__(
        self,
        gslrm_config: str,
        gslrm_checkpoint: str,
        device: str = "cuda",
        image_size: int = 512,
    ):
        """Initialize GS-LRM for per-frame inference.
        
        Args:
            gslrm_config: Path to GS-LRM config YAML.
            gslrm_checkpoint: Path to GS-LRM checkpoint.
            device: Torch device.
            image_size: Image resolution.
        """
        self.device = device
        self.image_size = image_size
        
        self._gslrm = None
        self._gslrm_config = gslrm_config
        self._gslrm_checkpoint = gslrm_checkpoint
        
    @property
    def gslrm(self):
        """Lazy load GS-LRM model."""
        if self._gslrm is None:
            from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
            print("Loading GS-LRM model...")
            self._gslrm = GSLRMInference(
                config_path=self._gslrm_config,
                checkpoint_path=self._gslrm_checkpoint,
                device=self.device,
                image_size=self.image_size,
            )
        return self._gslrm
    
    def run_gslrm_temporal(
        self,
        data_dir: str,
        output_dir: str,
        frame_indices: Optional[List[int]] = None,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Run GS-LRM on temporal sequence, save per-frame Gaussians.
        
        Args:
            data_dir: Preprocessed dataset directory (with samples/).
            output_dir: Output directory for per-frame .pt files.
            frame_indices: Specific frame indices to process (None = all).
            max_frames: Maximum frames to process.
            
        Returns:
            Path to gaussian_cache directory.
        """
        from mouse_extensions.inference.gslrm_pipeline import load_sample_data
        
        data_path = Path(data_dir)
        samples_dir = data_path / "samples"
        
        if not samples_dir.exists():
            raise FileNotFoundError(f"samples/ not found in {data_dir}")
        
        sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
        
        if frame_indices is not None:
            sample_dirs = [sample_dirs[i] for i in frame_indices if i < len(sample_dirs)]
        elif max_frames is not None:
            sample_dirs = sample_dirs[:max_frames]
        
        output_path = Path(output_dir) / "gaussian_cache"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {len(sample_dirs)} frames...")
        
        for i, sample_dir in enumerate(tqdm(sample_dirs, desc="GS-LRM inference")):
            images, c2ws, fxfycxcys, index = load_sample_data(
                str(sample_dir), self.image_size, self.device
            )
            
            with torch.no_grad():
                result = self.gslrm.predict(images, c2ws, fxfycxcys, index)
            
            gaussian_params = result["gaussian_params"]
            
            save_data = {
                "xyz": gaussian_params.xyz.cpu(),
                "features": gaussian_params.features.cpu(),
                "scaling": gaussian_params.scaling.cpu(),
                "rotation": gaussian_params.rotation.cpu(),
                "opacity": gaussian_params.opacity.cpu(),
                "sample_name": sample_dir.name,
            }
            torch.save(save_data, output_path / f"frame_{i:06d}.pt")
            
            del images, c2ws, fxfycxcys, index, result
            torch.cuda.empty_cache()
        
        metadata = {
            "num_frames": len(sample_dirs),
            "data_dir": str(data_dir),
            "sample_names": [d.name for d in sample_dirs],
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(sample_dirs)} frames to {output_path}")
        return output_path
    
    def smooth_gaussians(
        self,
        gaussian_cache_dir: str,
        output_dir: str,
        window_size: int = 5,
        blend_alpha: float = 0.3,
        sigma_temporal: float = 1.0,
        sigma_spatial: float = 0.1,
    ) -> Path:
        """Apply bilateral smoothing to cached Gaussians."""
        from mouse_extensions.scripts.inference.run_temporal_v2_streaming import (
            bilateral_smooth_streaming
        )
        
        output_path = Path(output_dir) / f"smoothed_w{window_size}_a{blend_alpha}"
        
        bilateral_smooth_streaming(
            input_dir=gaussian_cache_dir,
            output_dir=str(output_path),
            window_size=window_size,
            blend_alpha=blend_alpha,
            sigma_temporal=sigma_temporal,
            sigma_spatial=sigma_spatial,
            device=self.device,
        )
        
        return output_path
    
    def render_comparison_video(
        self,
        original_cache_dir: str,
        smoothed_cache_dir: str,
        output_path: str,
        fps: int = 10,
        resolution: int = 512,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Render side-by-side comparison video using GS-LRM renderer."""
        import cv2
        from mouse_extensions.visualization import GaussianModel, render_turntable
        
        original_path = Path(original_cache_dir)
        smoothed_path = Path(smoothed_cache_dir)
        
        orig_files = sorted(original_path.glob("frame_*.pt"))
        smooth_files = sorted(smoothed_path.glob("frame_*.pt"))
        
        if max_frames:
            orig_files = orig_files[:max_frames]
            smooth_files = smooth_files[:max_frames]
        
        assert len(orig_files) == len(smooth_files), "Frame count mismatch"
        
        # Get frame size
        first_data = torch.load(orig_files[0], weights_only=True)
        gm_test = GaussianModel(sh_degree=0)
        gm_test._xyz = first_data["xyz"].to(self.device)
        gm_test._features_dc = first_data["features"][:, :1].to(self.device)
        gm_test._features_rest = first_data["features"][:, 1:].to(self.device)
        gm_test._scaling = first_data["scaling"].to(self.device)
        gm_test._rotation = first_data["rotation"].to(self.device)
        gm_test._opacity = first_data["opacity"].to(self.device)
        
        test_frame = render_turntable(
            gm_test, rendering_resolution=resolution, num_views=1, elevation=20, radius=2.7
        )
        h, w = test_frame.shape[:2]
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (w * 2, h))
        
        print(f"Rendering {len(orig_files)} frames...")
        
        for orig_file, smooth_file in tqdm(zip(orig_files, smooth_files), total=len(orig_files)):
            # Render original
            orig_data = torch.load(orig_file, weights_only=True)
            gm_orig = GaussianModel(sh_degree=0)
            gm_orig._xyz = orig_data["xyz"].to(self.device)
            gm_orig._features_dc = orig_data["features"][:, :1].to(self.device)
            gm_orig._features_rest = orig_data["features"][:, 1:].to(self.device)
            gm_orig._scaling = orig_data["scaling"].to(self.device)
            gm_orig._rotation = orig_data["rotation"].to(self.device)
            gm_orig._opacity = orig_data["opacity"].to(self.device)
            
            orig_frame = render_turntable(
                gm_orig, rendering_resolution=resolution, num_views=1, elevation=20, radius=2.7
            )
            
            # Render smoothed
            smooth_data = torch.load(smooth_file, weights_only=True)
            gm_smooth = GaussianModel(sh_degree=0)
            gm_smooth._xyz = smooth_data["xyz"].to(self.device)
            gm_smooth._features_dc = smooth_data["features"][:, :1].to(self.device)
            gm_smooth._features_rest = smooth_data["features"][:, 1:].to(self.device)
            gm_smooth._scaling = smooth_data["scaling"].to(self.device)
            gm_smooth._rotation = smooth_data["rotation"].to(self.device)
            gm_smooth._opacity = smooth_data["opacity"].to(self.device)
            
            smooth_frame = render_turntable(
                gm_smooth, rendering_resolution=resolution, num_views=1, elevation=20, radius=2.7
            )
            
            # Add labels
            orig_labeled = add_label(orig_frame, "Original (per-frame)")
            smooth_labeled = add_label(smooth_frame, "Smoothed (V2)")
            
            combined = np.concatenate([orig_labeled, smooth_labeled], axis=1)
            combined = np.ascontiguousarray(combined)
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            writer.write(combined_bgr)
            
            torch.cuda.empty_cache()
        
        writer.release()
        print(f"Saved video to {output_file}")
        return output_file
    
    def run_full_pipeline(
        self,
        data_dir: str,
        output_dir: str,
        window_size: int = 5,
        blend_alpha: float = 0.3,
        max_frames: Optional[int] = None,
        render_video: bool = True,
        skip_gslrm: bool = False,
        gaussian_cache_dir: Optional[str] = None,
    ) -> dict:
        """Run full E2E temporal pipeline.
        
        Args:
            data_dir: Preprocessed dataset directory.
            output_dir: Output directory.
            window_size: Temporal window for smoothing.
            blend_alpha: Smoothing blend factor.
            max_frames: Max frames to process.
            render_video: Whether to render comparison video.
            skip_gslrm: Skip GS-LRM inference, use existing cache.
            gaussian_cache_dir: Path to existing Gaussian cache (if skip_gslrm).
            
        Returns:
            Dictionary with output paths and statistics.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: GS-LRM per-frame inference
        if skip_gslrm and gaussian_cache_dir:
            print("\n=== Step 1: Using existing Gaussian cache ===")
            cache_dir = Path(gaussian_cache_dir)
        else:
            print("\n=== Step 1: GS-LRM Per-Frame Inference ===")
            cache_dir = self.run_gslrm_temporal(
                data_dir=data_dir,
                output_dir=str(output_path),
                max_frames=max_frames,
            )
        
        # Step 2: Bilateral smoothing
        print("\n=== Step 2: Bilateral Smoothing ===")
        smoothed_dir = self.smooth_gaussians(
            gaussian_cache_dir=str(cache_dir),
            output_dir=str(output_path),
            window_size=window_size,
            blend_alpha=blend_alpha,
        )
        
        results = {
            "gaussian_cache": str(cache_dir),
            "smoothed_cache": str(smoothed_dir),
        }
        
        # Step 3: Render comparison video
        if render_video:
            print("\n=== Step 3: Rendering Comparison Video ===")
            video_path = self.render_comparison_video(
                original_cache_dir=str(cache_dir),
                smoothed_cache_dir=str(smoothed_dir),
                output_path=str(output_path / "comparison.mp4"),
                max_frames=max_frames,
            )
            results["comparison_video"] = str(video_path)
        
        # Save summary
        summary_path = output_path / "pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Results saved to: {output_path}")
        
        return results


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Add text label to frame."""
    import cv2
    
    labeled = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    x, y = 10, 25
    cv2.rectangle(labeled, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(labeled, label, (x, y), font, font_scale, color, thickness)
    
    return labeled


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="E2E Temporal Pipeline")
    parser.add_argument("--data_dir", type=str, help="Preprocessed dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gslrm_config", type=str, required=True, help="GS-LRM config path")
    parser.add_argument("--gslrm_checkpoint", type=str, required=True, help="GS-LRM checkpoint path")
    parser.add_argument("--window_size", type=int, default=5, help="Smoothing window size")
    parser.add_argument("--blend_alpha", type=float, default=0.3, help="Smoothing blend factor")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no_video", action="store_true", help="Skip video rendering")
    parser.add_argument("--skip_gslrm", action="store_true", help="Skip GS-LRM, use existing cache")
    parser.add_argument("--gaussian_cache", type=str, help="Path to existing Gaussian cache")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    pipeline = TemporalE2EPipeline(
        gslrm_config=args.gslrm_config,
        gslrm_checkpoint=args.gslrm_checkpoint,
        device=args.device,
    )
    
    pipeline.run_full_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        blend_alpha=args.blend_alpha,
        max_frames=args.max_frames,
        render_video=not args.no_video,
        skip_gslrm=args.skip_gslrm,
        gaussian_cache_dir=args.gaussian_cache,
    )
