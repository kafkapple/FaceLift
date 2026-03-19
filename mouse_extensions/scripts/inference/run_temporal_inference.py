#!/usr/bin/env python3
"""Temporal Inference: Apply deformation network for video sequence consistency.

This script chains:
1. Frame sequence detection (directory/video/explicit)
2. GS-LRM inference for each frame -> Gaussians
3. Deformation network for temporal smoothing
4. Rendering and saving results

Usage:
    # From directory of frames
    python -m mouse_extensions.scripts.inference.run_temporal_inference \
        --input /path/to/frames/ \
        --deform_checkpoint /path/to/deformation/best.pt \
        --gslrm_config /path/to/gslrm/config.yaml \
        --gslrm_checkpoint /path/to/gslrm/best.pt \
        --output /path/to/output/

    # From video file
    python -m mouse_extensions.scripts.inference.run_temporal_inference \
        --video /path/to/video.mp4 \
        --frame_stride 1 \
        ...
"""

import argparse
import re
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal inference with deformation network")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, help="Directory containing frame images")
    input_group.add_argument("--video", type=str, help="Video file path")
    input_group.add_argument("--frames", type=str, nargs="+", help="Explicit list of frame paths")

    # Video options
    parser.add_argument("--frame_stride", type=int, default=1, help="Frame stride for video input")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to process")

    # Model paths
    parser.add_argument("--deform_checkpoint", type=str, required=True,
                        help="Deformation network checkpoint")
    parser.add_argument("--gslrm_config", type=str, required=True,
                        help="GS-LRM config YAML")
    parser.add_argument("--gslrm_checkpoint", type=str, required=True,
                        help="GS-LRM checkpoint")

    # Optional: use pre-computed cache
    parser.add_argument("--gaussian_cache", type=str, default=None,
                        help="Directory with pre-computed Gaussians (skip GS-LRM)")

    # Output
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--save_comparison", action="store_true",
                        help="Save before/after comparison")
    parser.add_argument("--turntable_views", type=int, default=60,
                        help="Number of turntable views per frame")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def detect_frame_sequence(input_path: str) -> List[Tuple[int, str]]:
    """Detect frame sequence from directory.

    Supports two formats:
    1. M5 dataset: directories named 000000/, 000001/, etc.
    2. Image files: frame_0001.png, 0001.png, etc.

    Returns list of (frame_index, sample_path) sorted by index.
    """
    input_dir = Path(input_path)
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_path}")

    frame_pattern = re.compile(r'(\d+)')
    indexed_items = []

    # Check for M5-style dataset (numbered directories with opencv_cameras.json)
    subdirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if subdirs:
        # M5 dataset format: directories like 000000/, 000001/, ...
        for d in subdirs:
            if (d / "opencv_cameras.json").exists():
                idx = int(d.name)
                indexed_items.append((idx, str(d)))

    if not indexed_items:
        # Fallback: image files in directory
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        files = []
        for pattern in patterns:
            files.extend(input_dir.glob(pattern))

        for f in files:
            matches = frame_pattern.findall(f.stem)
            if matches:
                idx = int(matches[-1])
                indexed_items.append((idx, str(f)))

    if not indexed_items:
        raise ValueError(f"No valid samples found in {input_path}")

    # Sort by index
    indexed_items.sort(key=lambda x: x[0])

    # Verify consecutive (warn if gaps)
    indices = [x[0] for x in indexed_items]
    if len(indices) > 1:
        expected = list(range(indices[0], indices[-1] + 1))
        if indices != expected:
            missing = set(expected) - set(indices)
            print(f"Warning: Missing frames: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")

    print(f"Detected {len(indexed_items)} samples (format: {'M5 dataset' if subdirs else 'image files'})")
    return indexed_items


def extract_video_frames(video_path: str, stride: int = 1, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Extract frames from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_idx = 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total // stride, desc="Extracting frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride == 0:
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            pbar.update(1)

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    pbar.close()

    print(f"Extracted {len(frames)} frames from video")
    return frames


class TemporalInferencePipeline:
    """Temporal inference pipeline combining GS-LRM and Deformation Network."""

    def __init__(
        self,
        gslrm_config: str,
        gslrm_checkpoint: str,
        deform_checkpoint: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        # Load GS-LRM
        print("Loading GS-LRM...")
        from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
        self.gslrm = GSLRMInference(
            config_path=gslrm_config,
            checkpoint_path=gslrm_checkpoint,
            device=device,
        )

        # Load Deformation Network
        print("Loading Deformation Network...")
        from mouse_extensions.model.deformation.deformation_network import (
            DeformationNetwork, DeformationConfig
        )
        from mouse_extensions.model.deformation.temporal_pipeline import (
            TemporalGaussianPipeline, TemporalConfig
        )
        from mouse_extensions.model.deformation.gaussian_params import GaussianParams

        checkpoint = torch.load(deform_checkpoint, map_location=self.device, weights_only=False)

        # Reconstruct config from checkpoint
        # Config can be TrainerConfig (has deform_config) or DeformationConfig directly
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            if hasattr(cfg, "deform_config"):
                # TrainerConfig object
                deform_cfg = cfg.deform_config
            elif isinstance(cfg, dict):
                deform_cfg = DeformationConfig(**cfg)
            else:
                deform_cfg = cfg  # Assume it's already DeformationConfig
        else:
            deform_cfg = DeformationConfig()  # Use defaults

        self.deform_net = DeformationNetwork(deform_cfg)
        self.deform_net.load_state_dict(checkpoint["model_state_dict"])
        self.deform_net.to(self.device)
        self.deform_net.eval()

        # Create temporal pipeline
        temporal_cfg = TemporalConfig(deform_config=deform_cfg)
        self.temporal_pipeline = TemporalGaussianPipeline(temporal_cfg)
        self.temporal_pipeline.deform_net = self.deform_net  # Share weights
        self.temporal_pipeline.to(self.device)
        self.temporal_pipeline.eval()

        self.GaussianParams = GaussianParams

        print(f"  Deformation: {self.deform_net}")
        print(f"  Loaded from step {checkpoint.get('global_step', 'unknown')}")

    def load_from_cache(self, cache_dir: str, frame_indices: List[int]) -> Tuple[List, List[int]]:
        """Load pre-computed Gaussians from cache.
        
        Returns:
            Tuple of (gaussians_list, valid_indices)
            Only loads frames that exist in cache.
        """
        from mouse_extensions.model.deformation.gslrm_integration import GaussianCache

        cache = GaussianCache(cache_dir=cache_dir)
        gaussians = []
        valid_indices = []
        
        # First pass: check which frames are available
        available = [idx for idx in frame_indices if cache.has(idx)]
        if len(available) < len(frame_indices):
            print(f"Warning: Only {len(available)}/{len(frame_indices)} frames found in cache")
            print(f"  Available range: {min(available)}-{max(available)}")

        for idx in tqdm(available, desc="Loading from cache"):
            gaussians.append(cache.get(idx))
            valid_indices.append(idx)

        return gaussians, valid_indices

    @torch.no_grad()
    def generate_gaussians(
        self,
        sample_paths: List[str],
        image_size: int = 512,
    ) -> List:
        """Generate Gaussians for each sample using GS-LRM.

        Args:
            sample_paths: List of sample directory paths (M5 format)
                          or image file paths.
            image_size: Target image resolution.

        Returns:
            List of GaussianParams.
        """
        from mouse_extensions.inference.gslrm_pipeline import load_sample_data

        gaussians = []

        for sample_path in tqdm(sample_paths, desc="Generating Gaussians"):
            # load_sample_data expects: sample_dir with images/ and opencv_cameras.json
            # For M5 dataset: sample_path is already the directory (e.g., .../M5/000000/)
            images, c2ws, fxfycxcys, index = load_sample_data(
                sample_path,
                image_size=image_size,
                device=str(self.device),
            )

            # Run GS-LRM
            result = self.gslrm.predict(images, c2ws, fxfycxcys, index)

            # Extract GaussianParams
            gm = result.gaussians[0]  # batch_size=1

            if gm._features_rest is not None:
                features = torch.cat([gm._features_dc, gm._features_rest], dim=1)
            else:
                features = gm._features_dc

            params = self.GaussianParams(
                xyz=gm._xyz,
                features=features,
                scaling=gm._scaling,
                rotation=gm._rotation,
                opacity=gm._opacity,
            )
            gaussians.append(params)

        return gaussians

    @torch.no_grad()
    def apply_temporal_smoothing_and_save(
        self,
        gaussians: List,
        output_dir: Path,
        save_comparison: bool = False,
        batch_size: int = 100,
    ) -> Tuple[int, int]:
        """Apply temporal smoothing and save immediately per batch.

        This avoids OOM by not accumulating all frames in memory.

        Args:
            gaussians: List of GaussianParams (one per frame)
            output_dir: Output directory
            save_comparison: If True, save original alongside smoothed
            batch_size: Process in batches to avoid OOM

        Returns:
            Tuple of (num_smoothed, num_original) saved
        """
        print(f"Processing {len(gaussians)} frames (batch_size={batch_size}, streaming save)...")

        smoothed_dir = output_dir / "smoothed"
        smoothed_dir.mkdir(parents=True, exist_ok=True)
        
        if save_comparison:
            original_dir = output_dir / "original"
            original_dir.mkdir(parents=True, exist_ok=True)

        num_smoothed = 0
        num_original = 0
        global_idx = 0
        
        for start_idx in tqdm(range(0, len(gaussians), batch_size), desc="Processing batches"):
            end_idx = min(start_idx + batch_size, len(gaussians))
            batch = gaussians[start_idx:end_idx]
            
            # Move batch to device
            batch_device = [g.to(self.device) for g in batch]
            
            # Apply temporal pipeline to batch
            smoothed_batch = self.temporal_pipeline(batch_device)
            
            # Save immediately (don't accumulate in memory)
            for i, (orig, smooth) in enumerate(zip(batch_device, smoothed_batch)):
                frame_idx = global_idx + i
                
                # Save smoothed
                save_path = smoothed_dir / f"smoothed_{frame_idx:06d}.pt"
                torch.save({
                    "xyz": smooth.xyz.cpu(),
                    "features": smooth.features.cpu(),
                    "scaling": smooth.scaling.cpu(),
                    "rotation": smooth.rotation.cpu(),
                    "opacity": smooth.opacity.cpu(),
                }, save_path)
                num_smoothed += 1
                
                # Save original if comparison mode
                if save_comparison:
                    save_path = original_dir / f"original_{frame_idx:06d}.pt"
                    torch.save({
                        "xyz": orig.xyz.cpu(),
                        "features": orig.features.cpu(),
                        "scaling": orig.scaling.cpu(),
                        "rotation": orig.rotation.cpu(),
                        "opacity": orig.opacity.cpu(),
                    }, save_path)
                    num_original += 1
            
            global_idx += len(batch)
            
            # Clear GPU memory
            del batch_device, smoothed_batch, batch
            torch.cuda.empty_cache()

        print(f"Saved {num_smoothed} smoothed frames to {smoothed_dir}")
        if save_comparison:
            print(f"Saved {num_original} original frames to {original_dir}")
            
        return num_smoothed, num_original
    
    @torch.no_grad()
    def apply_temporal_smoothing(
        self,
        gaussians: List,
        return_both: bool = False,
        batch_size: int = 100,
    ) -> List:
        """Apply deformation network for temporal smoothing (legacy, for small datasets)."""
        print(f"Applying temporal smoothing to {len(gaussians)} frames (batch_size={batch_size})...")

        all_smoothed = []
        all_original = [] if return_both else None
        
        for start_idx in tqdm(range(0, len(gaussians), batch_size), desc="Temporal smoothing"):
            end_idx = min(start_idx + batch_size, len(gaussians))
            batch = gaussians[start_idx:end_idx]
            batch_device = [g.to(self.device) for g in batch]
            smoothed_batch = self.temporal_pipeline(batch_device)
            smoothed_cpu = [s.to("cpu") for s in smoothed_batch]
            all_smoothed.extend(smoothed_cpu)
            
            if return_both:
                original_cpu = [g.to("cpu") for g in batch_device]
                all_original.extend(original_cpu)
            
            del batch_device, smoothed_batch
            torch.cuda.empty_cache()

        if return_both:
            return all_original, all_smoothed
        return all_smoothed

    def render_sequence(
        self,
        gaussians: List,
        output_dir: Path,
        prefix: str = "frame",
        turntable_views: int = 60,
    ):
        """Render Gaussian sequence to images/video."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Implement rendering using GS-LRM renderer
        # For now, save Gaussians as .pt files

        for i, g in enumerate(tqdm(gaussians, desc=f"Saving {prefix}")):
            save_path = output_dir / f"{prefix}_{i:04d}.pt"
            torch.save({
                "xyz": g.xyz.cpu(),
                "features": g.features.cpu(),
                "scaling": g.scaling.cpu(),
                "rotation": g.rotation.cpu(),
                "opacity": g.opacity.cpu(),
            }, save_path)

        print(f"Saved {len(gaussians)} frames to {output_dir}")


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine input samples
    if args.input:
        print(f"Scanning directory: {args.input}")
        indexed_samples = detect_frame_sequence(args.input)
        frame_indices = [x[0] for x in indexed_samples]
        sample_paths = [x[1] for x in indexed_samples]

    elif args.video:
        # Video input: not yet supported for M5-style inference
        # Would need to preprocess video to M5 format first
        raise NotImplementedError(
            "Video input requires M5-format preprocessing. "
            "Use mouse_extensions preprocessing tools first."
        )

    elif args.frames:
        # Explicit list of sample directories
        sample_paths = args.frames
        frame_indices = list(range(len(sample_paths)))

    if args.max_frames:
        sample_paths = sample_paths[:args.max_frames]
        frame_indices = frame_indices[:args.max_frames]

    print(f"Processing {len(sample_paths)} samples")

    # Initialize pipeline
    pipeline = TemporalInferencePipeline(
        gslrm_config=args.gslrm_config,
        gslrm_checkpoint=args.gslrm_checkpoint,
        deform_checkpoint=args.deform_checkpoint,
        device=args.device,
    )

    # Generate or load Gaussians
    if args.gaussian_cache:
        print(f"Loading from cache: {args.gaussian_cache}")
        gaussians, frame_indices = pipeline.load_from_cache(args.gaussian_cache, frame_indices)
        print(f"Loaded {len(gaussians)} frames from cache")
    else:
        gaussians = pipeline.generate_gaussians(sample_paths)

    # Apply temporal smoothing with streaming save (memory efficient)
    # Use streaming for large datasets (>500 frames)
    if len(gaussians) > 500:
        print(f"Using streaming save for {len(gaussians)} frames...")
        num_smoothed, num_original = pipeline.apply_temporal_smoothing_and_save(
            gaussians, 
            output_dir,
            save_comparison=args.save_comparison,
            batch_size=100,
        )
    else:
        # Small dataset: use legacy method
        if args.save_comparison:
            original, smoothed = pipeline.apply_temporal_smoothing(gaussians, return_both=True)
            pipeline.render_sequence(original, output_dir / "original", "original")
            pipeline.render_sequence(smoothed, output_dir / "smoothed", "smoothed")
        else:
            smoothed = pipeline.apply_temporal_smoothing(gaussians)
            pipeline.render_sequence(smoothed, output_dir / "smoothed", "smoothed")

    print(f"\n=== Done ===")
    print(f"Output: {output_dir}")


def render_gaussians_to_video(
    gaussian_files: List[str],
    output_path: str,
    resolution: int = 512,
    fps: int = 30,
    device: str = 'cuda',
):
    """Render saved Gaussian .pt files to video using GS-LRM renderer.
    
    Args:
        gaussian_files: List of .pt file paths
        output_path: Output video path (.mp4)
        resolution: Render resolution
        fps: Video FPS
        device: Device for rendering
    """
    from gslrm.model.gaussians_renderer import GaussianModel, render_turntable, imageseq2video
    import numpy as np
    
    print(f"Rendering {len(gaussian_files)} frames to video...")
    
    all_frames = []
    
    for pt_path in tqdm(gaussian_files, desc="Rendering"):
        data = torch.load(pt_path, weights_only=True)
        
        # Reconstruct GaussianModel
        gm = GaussianModel(sh_degree=2)
        gm._xyz = data['xyz'].to(device)
        gm._features_dc = data['features'].to(device)
        gm._features_rest = torch.zeros(data['xyz'].shape[0], 8, 3, device=device)
        gm._scaling = data['scaling'].to(device)
        gm._rotation = data['rotation'].to(device)
        gm._opacity = data['opacity'].to(device)
        
        # Render turntable (single view for video)
        vis = render_turntable(
            gm,
            rendering_resolution=resolution,
            num_views=1,  # Single front view
            elevation=20,
            radius=2.7,
        )
        
        # vis shape: [1, 3, H, W]
        frame = vis[0].permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        all_frames.append(frame)
    
    # Save video
    imageseq2video(all_frames, output_path, fps=fps)
    print(f"Saved video to {output_path}")


def create_comparison_video(
    original_dir: str,
    smoothed_dir: str,
    output_path: str,
    max_frames: Optional[int] = None,
    resolution: int = 512,
    fps: int = 30,
    device: str = 'cuda',
):
    """Create side-by-side comparison video."""
    import cv2
    from gslrm.model.gaussians_renderer import GaussianModel, render_turntable
    import numpy as np
    
    orig_files = sorted(Path(original_dir).glob('*.pt'))
    smooth_files = sorted(Path(smoothed_dir).glob('*.pt'))
    
    if max_frames:
        orig_files = orig_files[:max_frames]
        smooth_files = smooth_files[:max_frames]
    
    n_frames = min(len(orig_files), len(smooth_files))
    print(f"Creating comparison video for {n_frames} frames...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (resolution * 2, resolution))
    
    for i in tqdm(range(n_frames), desc="Rendering comparison"):
        # Load and render original
        orig_data = torch.load(orig_files[i], weights_only=True)
        gm_orig = GaussianModel(sh_degree=2)
        gm_orig._xyz = orig_data['xyz'].to(device)
        gm_orig._features_dc = orig_data['features'].to(device)
        gm_orig._features_rest = torch.zeros(orig_data['xyz'].shape[0], 8, 3, device=device)
        gm_orig._scaling = orig_data['scaling'].to(device)
        gm_orig._rotation = orig_data['rotation'].to(device)
        gm_orig._opacity = orig_data['opacity'].to(device)
        
        vis_orig = render_turntable(gm_orig, rendering_resolution=resolution, num_views=1, elevation=20, radius=2.7)
        orig_frame = vis_orig  # render_turntable returns numpy (h, w, c) uint8
        
        # Load and render smoothed
        smooth_data = torch.load(smooth_files[i], weights_only=True)
        gm_smooth = GaussianModel(sh_degree=2)
        gm_smooth._xyz = smooth_data['xyz'].to(device)
        gm_smooth._features_dc = smooth_data['features'].to(device)
        gm_smooth._features_rest = torch.zeros(smooth_data['xyz'].shape[0], 8, 3, device=device)
        gm_smooth._scaling = smooth_data['scaling'].to(device)
        gm_smooth._rotation = smooth_data['rotation'].to(device)
        gm_smooth._opacity = smooth_data['opacity'].to(device)
        
        vis_smooth = render_turntable(gm_smooth, rendering_resolution=resolution, num_views=1, elevation=20, radius=2.7)
        smooth_frame = vis_smooth  # render_turntable returns numpy (h, w, c) uint8
        
        # Combine side by side (already uint8 from render_turntable)
        comparison = np.ascontiguousarray(np.concatenate([orig_frame, smooth_frame], axis=1))
        
        # Add labels
        cv2.putText(comparison, f'[Original] GS-LRM Output (Frame {i})', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, f'[Deformed] Autoregressive (Frame {i})', (resolution + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        # Clear GPU memory
        del gm_orig, gm_smooth
        torch.cuda.empty_cache()
    
    out.release()
    print(f"Saved comparison video to {output_path}")


if __name__ == "__main__":
    main()
