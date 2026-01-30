"""
Unified Visualizer Module

Consolidates all visualization outputs for train/val/test/inference:
- Turntable videos with configurable speed
- Gaussian .ply and .npz files
- Rerun .rrd sequences for interactive 3D viewing

Reference: pose-splatter implementation at ~/dev/pose-splatter

Author: Claude Code
Date: 2026-01-29
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field

# Lazy imports for optional dependencies
RERUN_AVAILABLE = False
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    pass


@dataclass
class VisualizerConfig:
    """Configuration for unified visualizer."""
    # Video settings
    fps: int = 24
    rotation_speed_factor: float = 1.0  # 0.5 = half speed (2x frames)
    
    # Output toggles
    save_turntable_video: bool = True
    save_gaussian_ply: bool = True
    save_gaussian_npz: bool = True
    save_rerun_rrd: bool = True
    
    # Rerun settings
    rerun_point_size: float = 0.005
    rerun_recording_id: str = "facelift_gaussian"
    
    # Grid settings
    grid_rows: int = 6
    grid_cols: int = 10


class GaussianExporter:
    """Export Gaussians to various formats."""
    
    @staticmethod
    def to_npz(
        gaussian_model,
        output_path: str,
        filter_mask: Optional[np.ndarray] = None,
    ) -> str:
        """
        Export Gaussian to NPZ format (compatible with Rerun visualization).
        
        Args:
            gaussian_model: GaussianModel instance with _xyz, _features_dc, etc.
            output_path: Output .npz file path
            filter_mask: Optional boolean mask to filter Gaussians
            
        Returns:
            Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extract data from GaussianModel
        means = gaussian_model._xyz.detach().cpu().numpy()
        
        # Get colors from SH coefficients (DC component)
        f_dc = gaussian_model._features_dc.detach().cpu().numpy()
        # SH2RGB conversion: colors = 0.5 + C0 * f_dc where C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        colors = np.clip(0.5 + C0 * f_dc.squeeze(-1), 0, 1)
        
        # Get opacities (apply sigmoid if needed)
        opacities = gaussian_model.get_opacity.detach().cpu().numpy()
        
        # Get scales
        scales = gaussian_model.get_scaling.detach().cpu().numpy()
        
        # Get rotations (quaternions)
        quaternions = gaussian_model.get_rotation.detach().cpu().numpy()
        
        # Apply filter if provided
        if filter_mask is not None:
            means = means[filter_mask]
            colors = colors[filter_mask]
            opacities = opacities[filter_mask]
            scales = scales[filter_mask]
            quaternions = quaternions[filter_mask]
        
        # Save to NPZ
        np.savez(
            output_path,
            means=means,
            colors=colors,
            opacities=opacities,
            scales=scales,
            quaternions=quaternions,
        )
        
        return output_path
    
    @staticmethod
    def to_ply(
        gaussian_model,
        output_path: str,
        use_fp16: bool = False,
        enable_gs_viewer: bool = True,
        filter_mask: Optional[np.ndarray] = None,
    ) -> str:
        """
        Export Gaussian to PLY format using existing save_ply method.
        
        Args:
            gaussian_model: GaussianModel instance
            output_path: Output .ply file path
            use_fp16: Use float16 for smaller file size
            enable_gs_viewer: Enable GS viewer compatibility
            filter_mask: Optional boolean mask
            
        Returns:
            Output file path
        """
        gaussian_model.save_ply(
            output_path,
            use_fp16=use_fp16,
            enable_gs_viewer=enable_gs_viewer,
            filter_mask=filter_mask,
        )
        return output_path


class RerunExporter:
    """Export Gaussians to Rerun .rrd format."""
    
    @staticmethod
    def is_available() -> bool:
        """Check if Rerun SDK is available."""
        return RERUN_AVAILABLE
    
    @staticmethod
    def single_frame_to_rrd(
        npz_path: str,
        output_path: str,
        point_size: float = 0.005,
        recording_id: str = "facelift_gaussian",
    ) -> str:
        """
        Create .rrd file from single NPZ Gaussian file.
        
        Args:
            npz_path: Path to .npz file with Gaussian data
            output_path: Output .rrd file path
            point_size: Point size for visualization
            recording_id: Rerun recording identifier
            
        Returns:
            Output file path
        """
        if not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk not installed. Install with: pip install rerun-sdk")
        
        # Load NPZ data
        data = np.load(npz_path)
        means = data["means"]
        colors = data["colors"]
        opacities = data["opacities"]
        scales = data["scales"]
        
        num_gaussians = len(means)
        
        # Initialize Rerun for file output
        rr.init("FaceLift Gaussian Viewer", recording_id=recording_id)
        
        # Log colored point cloud
        rr.log(
            "gaussian/points",
            rr.Points3D(
                positions=means,
                colors=(colors * 255).astype(np.uint8),
                radii=np.full(num_gaussians, point_size),
            )
        )
        
        # Log opacity visualization
        opacity_colors = np.stack([opacities, opacities, opacities], axis=-1).squeeze()
        rr.log(
            "gaussian/opacity",
            rr.Points3D(
                positions=means,
                colors=(opacity_colors * 255).astype(np.uint8),
                radii=np.full(num_gaussians, point_size),
            )
        )
        
        # Log statistics
        stats_text = f"""
# FaceLift Gaussian Statistics

- **Total Gaussians**: {num_gaussians}
- **Opacity**: mean={opacities.mean():.4f}, std={opacities.std():.4f}
- **Scale**: mean={scales.mean():.6f}, std={scales.std():.6f}
- **Position X**: [{means[:, 0].min():.3f}, {means[:, 0].max():.3f}]
- **Position Y**: [{means[:, 1].min():.3f}, {means[:, 1].max():.3f}]
- **Position Z**: [{means[:, 2].min():.3f}, {means[:, 2].max():.3f}]
"""
        rr.log("stats", rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN))
        
        # Save to .rrd file
        rr.save(output_path)
        
        return output_path
    
    @staticmethod
    def sequence_to_rrd(
        npz_dir: str,
        output_path: str,
        point_size: float = 0.005,
        recording_id: str = "facelift_gaussian_sequence",
    ) -> str:
        """
        Create .rrd file from sequence of NPZ Gaussian files (with timeline).
        
        Args:
            npz_dir: Directory containing .npz files
            output_path: Output .rrd file path
            point_size: Point size for visualization
            recording_id: Rerun recording identifier
            
        Returns:
            Output file path
        """
        if not RERUN_AVAILABLE:
            raise ImportError("rerun-sdk not installed. Install with: pip install rerun-sdk")
        
        npz_files = sorted(Path(npz_dir).glob("*.npz"))
        
        if not npz_files:
            raise ValueError(f"No NPZ files found in {npz_dir}")
        
        # Initialize Rerun
        rr.init("FaceLift Gaussian Sequence", recording_id=recording_id)
        
        # Log each frame with timeline
        for frame_idx, npz_file in enumerate(npz_files):
            rr.set_time_sequence("frame", frame_idx)
            
            data = np.load(npz_file)
            means = data["means"]
            colors = data["colors"]
            opacities = data["opacities"]
            
            num_gaussians = len(means)
            
            # Log point cloud
            rr.log(
                "sequence/points",
                rr.Points3D(
                    positions=means,
                    colors=(colors * 255).astype(np.uint8),
                    radii=np.full(num_gaussians, point_size),
                )
            )
            
            # Log frame info
            frame_text = f"**Frame {frame_idx}**: {npz_file.name} ({num_gaussians} Gaussians)"
            rr.log("frame_info", rr.TextDocument(frame_text, media_type=rr.MediaType.MARKDOWN))
        
        # Save to .rrd file
        rr.save(output_path)
        
        return output_path


class UnifiedVisualizer:
    """
    Unified visualizer for FaceLift GS-LRM outputs.
    
    Handles:
    - Turntable videos with configurable rotation speed
    - Gaussian .ply and .npz exports
    - Rerun .rrd files for interactive 3D viewing
    """
    
    def __init__(
        self,
        output_dir: str,
        config: Optional[VisualizerConfig] = None,
    ):
        """
        Initialize unified visualizer.
        
        Args:
            output_dir: Base output directory
            config: Visualizer configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config or VisualizerConfig()
        
        # Create subdirectories
        self.video_dir = self.output_dir / "videos"
        self.gaussian_dir = self.output_dir / "gaussians"
        self.rerun_dir = self.output_dir / "rerun"
        
        for d in [self.video_dir, self.gaussian_dir, self.rerun_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def export_gaussian(
        self,
        gaussian_model,
        uid: str,
        filter_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Export Gaussian to all configured formats.
        
        Args:
            gaussian_model: GaussianModel instance
            uid: Unique identifier for the output
            filter_mask: Optional boolean mask
            
        Returns:
            Dict with paths: {"ply": path, "npz": path, "rrd": path}
        """
        paths = {}
        
        # Export to NPZ (needed for Rerun)
        if self.config.save_gaussian_npz or self.config.save_rerun_rrd:
            npz_path = str(self.gaussian_dir / f"{uid}.npz")
            GaussianExporter.to_npz(gaussian_model, npz_path, filter_mask)
            paths["npz"] = npz_path
        
        # Export to PLY
        if self.config.save_gaussian_ply:
            ply_path = str(self.gaussian_dir / f"{uid}.ply")
            GaussianExporter.to_ply(gaussian_model, ply_path, filter_mask=filter_mask)
            paths["ply"] = ply_path
        
        # Export to Rerun .rrd
        if self.config.save_rerun_rrd and RerunExporter.is_available():
            rrd_path = str(self.rerun_dir / f"{uid}.rrd")
            RerunExporter.single_frame_to_rrd(
                paths["npz"],
                rrd_path,
                point_size=self.config.rerun_point_size,
            )
            paths["rrd"] = rrd_path
        
        return paths
    
    def export_gaussian_sequence(
        self,
        gaussian_models: List,
        sequence_name: str,
        filter_masks: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, str]:
        """
        Export sequence of Gaussians for temporal visualization.
        
        Args:
            gaussian_models: List of GaussianModel instances
            sequence_name: Name for the sequence
            filter_masks: Optional list of boolean masks
            
        Returns:
            Dict with paths: {"npz_dir": path, "rrd": path}
        """
        paths = {}
        
        # Create sequence directory
        seq_dir = self.gaussian_dir / sequence_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        paths["npz_dir"] = str(seq_dir)
        
        # Export each frame
        for i, gaussian_model in enumerate(gaussian_models):
            mask = filter_masks[i] if filter_masks else None
            npz_path = str(seq_dir / f"frame_{i:06d}.npz")
            GaussianExporter.to_npz(gaussian_model, npz_path, mask)
        
        # Create Rerun .rrd sequence
        if self.config.save_rerun_rrd and RerunExporter.is_available():
            rrd_path = str(self.rerun_dir / f"{sequence_name}.rrd")
            RerunExporter.sequence_to_rrd(
                str(seq_dir),
                rrd_path,
                point_size=self.config.rerun_point_size,
            )
            paths["rrd"] = rrd_path
        
        return paths
    
    def create_turntable_video(
        self,
        turntable_frames: np.ndarray,
        uid: str,
        imageseq2video_fn,
        input_visualization: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Create turntable video with configurable rotation speed.
        
        Args:
            turntable_frames: [num_frames, H, W, 3] uint8 array
            uid: Unique identifier
            imageseq2video_fn: Function(frames, path, fps) to save video
            input_visualization: Optional input image to add as panel
            
        Returns:
            Dict with video paths
        """
        paths = {}
        
        if not self.config.save_turntable_video:
            return paths
        
        # Apply rotation speed factor (0.5 = half speed = double frames)
        if self.config.rotation_speed_factor != 1.0:
            # Interpolate to create more frames for slower rotation
            num_original = turntable_frames.shape[0]
            num_target = int(num_original / self.config.rotation_speed_factor)
            
            if num_target > num_original:
                # Interpolate frames
                indices = np.linspace(0, num_original - 1, num_target)
                new_frames = []
                for idx in indices:
                    lower = int(np.floor(idx))
                    upper = min(int(np.ceil(idx)), num_original - 1)
                    t = idx - lower
                    if lower == upper:
                        new_frames.append(turntable_frames[lower])
                    else:
                        blended = (1 - t) * turntable_frames[lower].astype(np.float32) + \
                                  t * turntable_frames[upper].astype(np.float32)
                        new_frames.append(blended.astype(np.uint8))
                turntable_frames = np.stack(new_frames)
        
        # Save main turntable video
        video_path = str(self.video_dir / f"turntable_{uid}.mp4")
        imageseq2video_fn(turntable_frames, video_path, fps=self.config.fps)
        paths["turntable"] = video_path
        
        # Create combined video with input if provided
        if input_visualization is not None:
            import cv2
            combined_frames = []
            input_h, input_w = input_visualization.shape[:2]
            
            for frame in turntable_frames:
                frame_h = frame.shape[0]
                if input_h != frame_h:
                    scale = frame_h / input_h
                    new_w = int(input_w * scale)
                    input_resized = cv2.resize(input_visualization, (new_w, frame_h))
                else:
                    input_resized = input_visualization
                
                combined = np.concatenate([input_resized, frame], axis=1)
                combined_frames.append(combined)
            
            combined_frames = np.stack(combined_frames)
            combined_path = str(self.video_dir / f"turntable_{uid}_with_input.mp4")
            imageseq2video_fn(combined_frames, combined_path, fps=self.config.fps)
            paths["turntable_with_input"] = combined_path
        
        return paths
    
    def create_grid_image(
        self,
        frames: np.ndarray,
        uid: str,
        add_row_labels_fn=None,
        camera_order: Optional[List[int]] = None,
    ) -> str:
        """
        Create grid image from turntable frames.
        
        Args:
            frames: [num_frames, H, W, 3] uint8
            uid: Unique identifier
            add_row_labels_fn: Optional function to add row labels
            camera_order: Camera order for labels
            
        Returns:
            Grid image path
        """
        from PIL import Image
        
        grid_rows = self.config.grid_rows
        grid_cols = self.config.grid_cols
        num_frames = frames.shape[0]
        
        # Limit frames to grid size
        max_frames = grid_rows * grid_cols
        if num_frames > max_frames:
            indices = np.linspace(0, num_frames - 1, max_frames).astype(int)
            frames = frames[indices]
        elif num_frames < max_frames:
            padding = max_frames - num_frames
            frames = np.concatenate([frames, np.repeat(frames[-1:], padding, axis=0)], axis=0)
        
        # Reshape to grid
        h, w = frames.shape[1:3]
        grid = frames.reshape(grid_rows, grid_cols, h, w, 3)
        grid = grid.transpose(0, 2, 1, 3, 4)
        grid = grid.reshape(grid_rows * h, grid_cols * w, 3)
        
        # Add row labels if function provided
        if add_row_labels_fn is not None and camera_order is not None:
            grid = add_row_labels_fn(
                grid, camera_order, grid_rows, grid_cols, h,
                label_height=55, loop=True
            )
        
        # Save
        grid_path = str(self.video_dir / f"grid_{uid}.jpg")
        Image.fromarray(grid).save(grid_path)
        
        return grid_path


def get_default_config() -> VisualizerConfig:
    """Get default visualizer configuration."""
    return VisualizerConfig()


def create_visualizer(
    output_dir: str,
    fps: int = 24,
    rotation_speed_factor: float = 0.5,  # Half speed by default
    save_gaussian_ply: bool = True,
    save_gaussian_npz: bool = True,
    save_rerun_rrd: bool = True,
) -> UnifiedVisualizer:
    """
    Factory function to create UnifiedVisualizer with custom settings.
    
    Args:
        output_dir: Output directory
        fps: Video frames per second
        rotation_speed_factor: 0.5 = half speed (recommended)
        save_gaussian_ply: Save .ply files
        save_gaussian_npz: Save .npz files
        save_rerun_rrd: Save Rerun .rrd files
        
    Returns:
        Configured UnifiedVisualizer instance
    """
    config = VisualizerConfig(
        fps=fps,
        rotation_speed_factor=rotation_speed_factor,
        save_gaussian_ply=save_gaussian_ply,
        save_gaussian_npz=save_gaussian_npz,
        save_rerun_rrd=save_rerun_rrd,
    )
    return UnifiedVisualizer(output_dir, config)
