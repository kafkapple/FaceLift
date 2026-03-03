"""
Gaussian Export Utilities

Extracted from unified_visualizer.py (_archive/).
Provides PLY/NPZ export and Rerun .rrd visualization.

Author: Claude Code
Date: 2026-01-29 (extracted 2026-03-03)
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
        colors = np.clip(0.5 + C0 * f_dc.squeeze(1), 0, 1)
        
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
            rr.set_time("frame", sequence=frame_idx)
            
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
