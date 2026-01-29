"""Exporter module: Save gaussians, videos, images."""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image


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
        """Save Gaussians as PLY file.
        
        Args:
            gaussians: GaussianModel object with save_ply method
            path: Output path (default: output_dir/gaussians.ply)
            
        Returns:
            Path to saved file
        """
        if not self.save_gaussian:
            return None

        path = Path(path) if path else self.output_dir / "gaussians.ply"
        path.parent.mkdir(parents=True, exist_ok=True)

        # GaussianModel has save_ply method
        if hasattr(gaussians, 'save_ply'):
            gaussians.save_ply(str(path))
        else:
            # Fallback: save as npz if not a GaussianModel
            self.save_gaussians_npz(gaussians, path.with_suffix('.npz'))
            return path.with_suffix('.npz')
            
        return path

    def save_gaussians_npz(
        self,
        gaussians,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save Gaussians as lightweight NPZ.
        
        Args:
            gaussians: GaussianModel or tensor
            path: Output path (default: output_dir/gaussians.npz)
            
        Returns:
            Path to saved file
        """
        if not self.save_gaussian:
            return None

        path = Path(path) if path else self.output_dir / "gaussians.npz"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract data from GaussianModel if available
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

    def save_rerun_rrd(
        self,
        gaussians_list: list,
        frame_indices: Optional[list[int]] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Save temporal sequence as Rerun RRD file.
        
        Args:
            gaussians_list: List of GaussianModel objects per frame
            frame_indices: Frame indices for timeline
            path: Output path (default: output_dir/sequence.rrd)
            
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
            rr.set_time_sequence("frame", frame_idx)

            # Extract positions from GaussianModel
            if hasattr(gaussians, '_xyz'):
                positions = gaussians._xyz.detach().cpu().numpy()
            elif isinstance(gaussians, torch.Tensor):
                positions = gaussians.detach().cpu().numpy()
                if positions.ndim == 2 and positions.shape[1] >= 3:
                    positions = positions[:, :3]
            else:
                continue

            rr.log("gaussians", rr.Points3D(positions))

        return path

    def save_video(
        self,
        frames: np.ndarray,
        name: str,
        fps: Optional[int] = None,
    ) -> Path:
        """Save frames as video.
        
        Args:
            frames: [T, H, W, C] uint8 array
            name: Video name (without extension)
            fps: Override FPS
            
        Returns:
            Path to saved video
        """
        from gslrm.model.gaussians_renderer import imageseq2video

        path = self.output_dir / f"{name}.mp4"
        imageseq2video(frames, str(path), fps=fps or self.video_fps)
        return path

    def save_image(
        self,
        image: Union[np.ndarray, Image.Image],
        name: str,
    ) -> Path:
        """Save image.
        
        Args:
            image: Image array or PIL Image
            name: Image name (without extension)
            
        Returns:
            Path to saved image
        """
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
        labels: Optional[list[str]] = None,
    ) -> Path:
        """Save images as grid.
        
        Args:
            images: [N, H, W, C] array
            name: Image name (without extension)
            cols: Number of columns
            labels: Optional labels for each image
            
        Returns:
            Path to saved image
        """
        n, h, w, c = images.shape
        rows = (n + cols - 1) // cols

        # Pad if needed
        pad_count = rows * cols - n
        if pad_count > 0:
            padding = np.zeros((pad_count, h, w, c), dtype=images.dtype)
            images = np.concatenate([images, padding], axis=0)

        # Reshape to grid
        grid = images.reshape(rows, cols, h, w, c)
        grid = grid.transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, c)

        return self.save_image(grid, name)
