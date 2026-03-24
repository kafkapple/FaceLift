"""Exporter module: Save gaussians, videos, images."""

import os
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
from PIL import Image


def sh_to_rgb(features_dc: np.ndarray) -> np.ndarray:
    """Convert SH DC component to RGB color.
    
    SH DC = color * C0 where C0 = 0.28209479177387814
    So color = SH_DC / C0
    
    Args:
        features_dc: [..., 3] SH DC coefficients
        
    Returns:
        [..., 3] RGB colors in [0, 1]
    """
    C0 = 0.28209479177387814
    rgb = features_dc / C0
    # Apply sigmoid for proper color range
    rgb = 1 / (1 + np.exp(-rgb))
    return np.clip(rgb, 0, 1)


class ExporterModule:
    """Export module for various output formats."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        save_gaussian: bool = True,
        save_rerun: bool = True,
        video_fps: int = 24,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_gaussian = save_gaussian
        self.save_rerun = save_rerun
        self.video_fps = video_fps

    def save_gaussians_ply(
        self,
        gaussians,  # GaussianModel object
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        if not self.save_gaussian:
            return None
        path = Path(path) if path else self.output_dir / "gaussians.ply"
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(gaussians, 'save_ply'):
            gaussians.save_ply(str(path))
        else:
            self.save_gaussians_npz(gaussians, path.with_suffix('.npz'))
            return path.with_suffix('.npz')
        return path

    def save_gaussians_npz(
        self,
        gaussians,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        if not self.save_gaussian:
            return None
        path = Path(path) if path else self.output_dir / "gaussians.npz"
        path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(gaussians, '_xyz'):
            data = {
                'xyz': gaussians._xyz.detach().cpu().numpy(),
                'opacity': gaussians._opacity.detach().cpu().numpy(),
                'scaling': gaussians._scaling.detach().cpu().numpy(),
                'rotation': gaussians._rotation.detach().cpu().numpy(),
                'features_dc': gaussians._features_dc.detach().cpu().numpy(),
            }
            if gaussians._features_rest is not None:
                data['features_rest'] = gaussians._features_rest.detach().cpu().numpy()
        elif isinstance(gaussians, torch.Tensor):
            data = {'gaussians': gaussians.detach().cpu().numpy()}
        else:
            data = {'gaussians': np.array(gaussians)}

        np.savez_compressed(str(path), **data)
        return path

    def _extract_gaussian_data(self, gaussians) -> dict:
        """Extract positions, colors, opacity, and scaling from Gaussians.
        
        Args:
            gaussians: GaussianModel, dict, or tensor
            
        Returns:
            Dict with 'positions', 'colors' (uint8), 'opacity', 'scaling'
        """
        result = {'positions': None, 'colors': None, 'opacity': None, 'scaling': None}
        
        # GaussianModel object (GS-LRM style)
        if hasattr(gaussians, '_xyz'):
            result['positions'] = gaussians._xyz.detach().cpu().numpy()
            
            # Extract colors from SH DC coefficients
            if hasattr(gaussians, '_features_dc') and gaussians._features_dc is not None:
                features_dc = gaussians._features_dc.detach().cpu().numpy()
                if features_dc.ndim == 3:
                    features_dc = features_dc[:, 0, :]
                rgb = sh_to_rgb(features_dc)
                result['colors'] = (rgb * 255).clip(0, 255).astype(np.uint8)
            
            # Extract opacity (sigmoid activation)
            if hasattr(gaussians, '_opacity') and gaussians._opacity is not None:
                opacity = gaussians._opacity.detach().cpu().numpy()
                result['opacity'] = 1 / (1 + np.exp(-opacity.flatten()))
            
            # Extract scaling (exp activation, GS-LRM uses log scale)
            if hasattr(gaussians, '_scaling') and gaussians._scaling is not None:
                scaling_log = gaussians._scaling.detach().cpu().numpy()
                # scaling_log shape: [N, 3], apply exp to get actual scale
                result['scaling'] = np.exp(scaling_log)  # [N, 3]
                
        # Dict from GS-LRM forward
        elif isinstance(gaussians, dict):
            if 'xyz' in gaussians:
                result['positions'] = gaussians['xyz'].detach().cpu().numpy()
            elif 'means' in gaussians:
                result['positions'] = gaussians['means'].detach().cpu().numpy()
                
            if 'colors' in gaussians:
                colors = gaussians['colors'].detach().cpu().numpy()
                if colors.max() <= 1.0:
                    colors = (colors * 255).clip(0, 255)
                result['colors'] = colors.astype(np.uint8)
            elif 'features_dc' in gaussians:
                features_dc = gaussians['features_dc'].detach().cpu().numpy()
                if features_dc.ndim == 3:
                    features_dc = features_dc[:, 0, :]
                rgb = sh_to_rgb(features_dc)
                result['colors'] = (rgb * 255).clip(0, 255).astype(np.uint8)
                
            if 'opacity' in gaussians or 'opacities' in gaussians:
                key = 'opacity' if 'opacity' in gaussians else 'opacities'
                opacity = gaussians[key].detach().cpu().numpy().flatten()
                if opacity.max() > 1.0:  # logit space
                    opacity = 1 / (1 + np.exp(-opacity))
                result['opacity'] = opacity
                
            if 'scales' in gaussians or 'scaling' in gaussians:
                key = 'scales' if 'scales' in gaussians else 'scaling'
                scaling = gaussians[key].detach().cpu().numpy()
                # Check if already in linear scale or log scale
                if scaling.min() < 0:  # log scale
                    scaling = np.exp(scaling)
                result['scaling'] = scaling
                
        # Raw tensor [N, D] - assume xyz | colors | opacity | ...
        elif isinstance(gaussians, torch.Tensor):
            arr = gaussians.detach().cpu().numpy()
            if arr.ndim == 2:
                result['positions'] = arr[:, :3] if arr.shape[1] >= 3 else arr
                if arr.shape[1] >= 6:
                    colors = arr[:, 3:6]
                    if colors.max() <= 1.0:
                        colors = colors * 255
                    result['colors'] = colors.clip(0, 255).astype(np.uint8)
                    
        return result

    def save_rerun_rrd(
        self,
        gaussians_list: list,
        frame_indices: Optional[List[int]] = None,
        path: Optional[Union[str, Path]] = None,
        base_radius: float = 0.002,
        use_gaussian_scale: bool = True,
        scale_multiplier: float = 0.5,
        filter_by_opacity: bool = True,
        opacity_threshold: float = 0.1,
    ) -> Path:
        """Save temporal sequence as Rerun RRD file with colors and proper radii.
        
        Radii calculation (GS-LRM style):
            - If use_gaussian_scale=True: radius = mean(scaling) * scale_multiplier
            - Otherwise: radius = base_radius
            - Opacity modulates radius: final_radius *= (0.5 + opacity * 0.5)
        
        Args:
            gaussians_list: List of GaussianModel/dict objects per frame
            frame_indices: Frame indices for timeline
            path: Output path (default: output_dir/sequence.rrd)
            base_radius: Base radius when not using Gaussian scale
            use_gaussian_scale: Use Gaussian's actual 3D scale for radii
            scale_multiplier: Multiplier for Gaussian scale -> point radius
            filter_by_opacity: Only show gaussians with opacity > threshold
            opacity_threshold: Minimum opacity to display
            
        Returns:
            Path to saved file
        """
        if not self.save_rerun:
            return None

        try:
            import rerun as rr
        except ImportError:
            print("Rerun not installed, skipping RRD export")
            return None

        path = Path(path) if path else self.output_dir / "rerun" / "sequence.rrd"
        path.parent.mkdir(parents=True, exist_ok=True)

        rr.init("gaussian_sequence", spawn=False)
        rr.save(str(path))

        for i, gaussians in enumerate(gaussians_list):
            frame_idx = frame_indices[i] if frame_indices else i
            rr.set_time("frame", sequence=frame_idx)

            # Extract data with colors and scaling
            data = self._extract_gaussian_data(gaussians)
            
            if data['positions'] is None:
                continue
                
            positions = data['positions']
            colors = data['colors']
            opacity = data['opacity']
            scaling = data['scaling']  # [N, 3] actual 3D scale
            
            # Filter by opacity if available
            if filter_by_opacity and opacity is not None:
                mask = opacity > opacity_threshold
                positions = positions[mask]
                if colors is not None:
                    colors = colors[mask]
                if opacity is not None:
                    opacity = opacity[mask]
                if scaling is not None:
                    scaling = scaling[mask]
            
            # Calculate radii based on Gaussian scale (GS-LRM style)
            if use_gaussian_scale and scaling is not None:
                # Use mean of 3D scale as point radius
                radii = scaling.mean(axis=1) * scale_multiplier  # [N]
            else:
                radii = np.full(len(positions), base_radius)
            
            # Modulate by opacity for visibility
            if opacity is not None:
                radii = radii * (0.5 + opacity * 0.5)
            
            # Log with colors and proper radii
            if colors is not None:
                rr.log("gaussians/points", rr.Points3D(
                    positions,
                    colors=colors,
                    radii=radii,
                ))
            else:
                rr.log("gaussians/points", rr.Points3D(
                    positions,
                    radii=radii,
                ))

        print(f"RRD saved: {path}")
        return path

    def save_video(
        self,
        frames: np.ndarray,
        name: str,
        fps: Optional[int] = None,
    ) -> Path:
        from mouse_extensions.visualization import imageseq2video
        path = self.output_dir / f"{name}.mp4"
        imageseq2video(frames, str(path), fps=fps or self.video_fps)
        return path

    def save_image(
        self,
        image: Union[np.ndarray, Image.Image],
        name: str,
    ) -> Path:
        path = self.output_dir / f"{name}.jpg"
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(str(path), quality=95)
        return path

    def save_grid(
        self,
        images: np.ndarray,
        name: str,
        cols: int = 6,
        labels: Optional[List[str]] = None,
    ) -> Path:
        n, h, w, c = images.shape
        rows = (n + cols - 1) // cols
        pad_count = rows * cols - n
        if pad_count > 0:
            padding = np.zeros((pad_count, h, w, c), dtype=images.dtype)
            images = np.concatenate([images, padding], axis=0)
        grid = images.reshape(rows, cols, h, w, c)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, c)
        return self.save_image(grid, name)
