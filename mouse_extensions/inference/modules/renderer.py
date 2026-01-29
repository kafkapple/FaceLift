"""Renderer module: Gaussians -> rendered images."""

from typing import Optional

import numpy as np
import torch


class RendererModule:
    """Gaussian rendering module."""

    def __init__(
        self,
        resolution: int = 384,
        num_views: int = 36,
        elevation: float = 20.0,
        radius: float = 2.7,
    ):
        self.resolution = resolution
        self.num_views = num_views
        self.elevation = elevation
        self.radius = radius

    def render_turntable(
        self,
        gaussians: torch.Tensor,
        num_views: Optional[int] = None,
        resolution: Optional[int] = None,
        elevation: Optional[float] = None,
        radius: Optional[float] = None,
    ) -> np.ndarray:
        """Render 360-degree turntable views.
        
        Args:
            gaussians: Gaussian parameters tensor
            num_views: Override number of views
            resolution: Override resolution
            elevation: Override elevation angle
            radius: Override camera radius
            
        Returns:
            Array of rendered frames [V, H, W, C]
        """
        from gslrm.model.gaussians_renderer import render_turntable

        nv = num_views or self.num_views
        res = resolution or self.resolution
        elev = elevation or self.elevation
        rad = radius or self.radius

        # render_turntable returns [H, V*W, C] strip
        strip = render_turntable(
            gaussians,
            rendering_resolution=res,
            num_views=nv,
            elevation=elev,
            radius=rad,
            trajectory_mode="turntable",
        )

        # Convert to [V, H, W, C]
        h = strip.shape[0]
        w = strip.shape[1] // nv
        frames = strip.reshape(h, nv, w, 3)
        frames = np.transpose(frames, (1, 0, 2, 3))

        return frames

    def render_from_cameras(
        self,
        gaussians: torch.Tensor,
        c2ws: np.ndarray,
        fxfycxcy: np.ndarray,
        resolution: Optional[int] = None,
    ) -> np.ndarray:
        """Render from specific camera poses.
        
        Args:
            gaussians: Gaussian parameters tensor
            c2ws: [N, 4, 4] camera-to-world matrices
            fxfycxcy: [N, 4] intrinsics
            resolution: Output resolution
            
        Returns:
            Array of rendered frames [N, H, W, C]
        """
        from gslrm.model.gaussians_renderer import render_from_cameras

        res = resolution or self.resolution

        frames = render_from_cameras(
            gaussians,
            c2ws,
            fxfycxcy,
            rendering_resolution=res,
        )

        return frames
