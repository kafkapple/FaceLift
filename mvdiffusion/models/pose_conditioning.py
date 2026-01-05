# =============================================================================
# Camera Pose Conditioning Module for MVDiffusion
# =============================================================================
# Purpose: Replace discrete view indices with continuous camera pose encoding
# Methods: Plücker Ray, Spherical Coordinates, Camera Extrinsic
# Reference: Zero123++, SV3D
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from typing import Optional, Tuple


class FourierEncoder(nn.Module):
    """
    Fourier feature encoding for continuous values.
    Maps scalar inputs to high-dimensional sinusoidal features.

    Reference: NeRF positional encoding
    """
    def __init__(self, num_freq_bands: int = 64, max_freq: float = 10.0):
        super().__init__()
        # Frequencies: 2^0, 2^1, ..., 2^(max_freq)
        freqs = 2.0 ** torch.linspace(0, max_freq, num_freq_bands)
        self.register_buffer('freqs', freqs)
        self.out_dim = num_freq_bands * 2  # sin + cos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [...] - input values (any shape)
        Returns:
            [..., num_freq_bands * 2] - Fourier features
        """
        # Expand to [..., num_freq_bands]
        x_proj = x.unsqueeze(-1) * self.freqs
        # Concat sin and cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SphericalPoseEncoder(nn.Module):
    """
    Encode camera pose using spherical coordinates (azimuth, elevation, distance).

    Best for: Camera arrangements on a sphere looking at origin.
    Pros: Simple, interpretable, rotation-invariant around vertical axis.
    Cons: Assumes camera looking at origin.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        num_freq_bands: int = 64,
        max_freq: float = 10.0,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.fourier = FourierEncoder(num_freq_bands, max_freq)

        # 3 coordinates (azimuth, elevation, distance) x fourier_dim
        in_features = 3 * self.fourier.out_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        azimuth: torch.Tensor,      # [B] or [B, N] in radians
        elevation: torch.Tensor,    # [B] or [B, N] in radians
        distance: torch.Tensor,     # [B] or [B, N]
    ) -> torch.Tensor:
        """
        Args:
            azimuth: Camera azimuth angle (radians, 0 = front)
            elevation: Camera elevation angle (radians, 0 = horizontal)
            distance: Camera-to-origin distance

        Returns:
            [B, embed_dim] or [B, N, embed_dim] - pose embeddings
        """
        az_enc = self.fourier(azimuth)
        el_enc = self.fourier(elevation)
        dist_enc = self.fourier(distance)

        features = torch.cat([az_enc, el_enc, dist_enc], dim=-1)
        return self.mlp(features)

    @staticmethod
    def from_camera_matrix(
        c2w: torch.Tensor,  # [B, 4, 4] or [B, N, 4, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract spherical coordinates from camera-to-world matrix.

        Assumes cameras looking at origin.
        """
        # Camera position is the translation component
        position = c2w[..., :3, 3]  # [B, 3] or [B, N, 3]

        x, y, z = position.unbind(-1)

        # Compute spherical coordinates
        distance = torch.sqrt(x**2 + y**2 + z**2)
        azimuth = torch.atan2(x, z)  # Angle in XZ plane from Z axis
        elevation = torch.asin(y / (distance + 1e-8))  # Angle from XZ plane

        return azimuth, elevation, distance


class ExtrinsicPoseEncoder(nn.Module):
    """
    Directly encode camera extrinsic parameters (rotation + translation).

    Best for: General camera configurations.
    Pros: No assumptions about camera arrangement.
    Cons: Less interpretable, may need more training data.
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        use_6d_rotation: bool = True,  # 6D rotation representation
    ):
        super().__init__()
        self.use_6d_rotation = use_6d_rotation

        # 6D rotation (first two columns) + 3D translation = 9D
        # or 9D rotation matrix + 3D translation = 12D
        in_features = 9 if use_6d_rotation else 12

        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        rotation: torch.Tensor,     # [B, 3, 3] or [B, N, 3, 3]
        translation: torch.Tensor,  # [B, 3] or [B, N, 3]
    ) -> torch.Tensor:
        """
        Args:
            rotation: 3x3 rotation matrix
            translation: 3D translation vector

        Returns:
            [B, embed_dim] or [B, N, embed_dim] - pose embeddings
        """
        if self.use_6d_rotation:
            # Use first two columns of rotation matrix (6D representation)
            rot_6d = rotation[..., :2].flatten(start_dim=-2)  # [B, 6] or [B, N, 6]
        else:
            rot_6d = rotation.flatten(start_dim=-2)  # [B, 9] or [B, N, 9]

        features = torch.cat([rot_6d, translation], dim=-1)
        return self.mlp(features)

    @staticmethod
    def from_camera_matrix(
        c2w: torch.Tensor,  # [B, 4, 4] or [B, N, 4, 4]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract rotation and translation from camera-to-world matrix."""
        rotation = c2w[..., :3, :3]
        translation = c2w[..., :3, 3]
        return rotation, translation


class PluckerRayEncoder(nn.Module):
    """
    Encode camera pose using Plücker coordinates for each pixel ray.

    Plücker coordinates: (direction, moment) where moment = origin × direction
    This provides a pixel-level geometric encoding of the camera.

    Best for: Dense 3D-aware conditioning.
    Pros: Most expressive, pixel-level geometry.
    Cons: Computationally heavier, needs ray generation.

    Reference: Zero123++, SV3D
    """
    def __init__(
        self,
        embed_dim: int = 320,  # Should match first UNet channel
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Plücker coordinates are 6D (direction 3D + moment 3D)
        self.conv = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1),
        )

    def forward(
        self,
        plucker: torch.Tensor,  # [B, 6, H, W]
    ) -> torch.Tensor:
        """
        Args:
            plucker: Plücker coordinates [direction (3D), moment (3D)]

        Returns:
            [B, embed_dim, H, W] - spatial pose features
        """
        return self.conv(plucker)

    @staticmethod
    def compute_plucker_coordinates(
        c2w: torch.Tensor,       # [B, 4, 4]
        intrinsics: torch.Tensor, # [B, 3, 3] or [B, 4]  (fx, fy, cx, cy)
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Compute Plücker coordinates for all pixels.

        Args:
            c2w: Camera-to-world transformation
            intrinsics: Camera intrinsics (matrix or [fx, fy, cx, cy])
            height, width: Image dimensions

        Returns:
            [B, 6, H, W] - Plücker coordinates for each pixel
        """
        B = c2w.shape[0]
        device = c2w.device

        # Parse intrinsics
        if intrinsics.shape[-1] == 4:
            fx, fy, cx, cy = intrinsics.unbind(-1)
        else:
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1]
            cx = intrinsics[:, 0, 2]
            cy = intrinsics[:, 1, 2]

        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing='ij'
        )
        x = x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        y = y.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]

        # Normalize pixel coordinates to camera space
        fx = fx.view(-1, 1, 1)
        fy = fy.view(-1, 1, 1)
        cx = cx.view(-1, 1, 1)
        cy = cy.view(-1, 1, 1)

        x_cam = (x - cx) / fx
        y_cam = (y - cy) / fy
        z_cam = torch.ones_like(x_cam)

        # Ray directions in camera space [B, H, W, 3]
        directions_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        directions_cam = F.normalize(directions_cam, dim=-1)

        # Transform to world space
        rotation = c2w[:, :3, :3]  # [B, 3, 3]
        translation = c2w[:, :3, 3]  # [B, 3]

        # [B, H, W, 3] @ [B, 3, 3].T -> [B, H, W, 3]
        directions_world = torch.einsum('bhwc,bdc->bhwd', directions_cam, rotation)

        # Camera origin in world space
        origin = translation.view(B, 1, 1, 3).expand(-1, height, width, -1)

        # Compute moment: m = origin × direction
        moment = torch.cross(origin, directions_world, dim=-1)

        # Concatenate direction and moment [B, H, W, 6]
        plucker = torch.cat([directions_world, moment], dim=-1)

        # Reshape to [B, 6, H, W]
        plucker = rearrange(plucker, 'b h w c -> b c h w')

        return plucker


class CameraPoseConditioner(nn.Module):
    """
    Unified Camera Pose Conditioning module for MVDiffusion.

    Replaces discrete view indices with continuous pose encoding.
    Supports multiple encoding methods.
    """
    def __init__(
        self,
        method: str = 'spherical',  # 'spherical', 'extrinsic', 'plucker'
        embed_dim: int = 1024,
        num_views: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.method = method
        self.embed_dim = embed_dim
        self.num_views = num_views

        if method == 'spherical':
            self.encoder = SphericalPoseEncoder(embed_dim=embed_dim, **kwargs)
        elif method == 'extrinsic':
            self.encoder = ExtrinsicPoseEncoder(embed_dim=embed_dim, **kwargs)
        elif method == 'plucker':
            self.encoder = PluckerRayEncoder(embed_dim=kwargs.get('spatial_dim', 320))
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(
        self,
        c2w: torch.Tensor,                    # [B, N, 4, 4] or [B, 4, 4]
        intrinsics: Optional[torch.Tensor] = None,  # For Plücker
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode camera poses.

        Returns shape depends on method:
        - spherical/extrinsic: [B, N, embed_dim] or [B, embed_dim]
        - plucker: [B*N, spatial_dim, H, W]
        """
        if self.method == 'spherical':
            azimuth, elevation, distance = SphericalPoseEncoder.from_camera_matrix(c2w)
            return self.encoder(azimuth, elevation, distance)

        elif self.method == 'extrinsic':
            rotation, translation = ExtrinsicPoseEncoder.from_camera_matrix(c2w)
            return self.encoder(rotation, translation)

        elif self.method == 'plucker':
            if intrinsics is None or height is None or width is None:
                raise ValueError("Plücker encoding requires intrinsics, height, width")

            # Handle batched views
            if c2w.dim() == 4:  # [B, N, 4, 4]
                B, N = c2w.shape[:2]
                c2w = rearrange(c2w, 'b n i j -> (b n) i j')
                if intrinsics.dim() == 3:  # [B, N, 4]
                    intrinsics = rearrange(intrinsics, 'b n c -> (b n) c')
                else:  # [B, 4] -> repeat for N views
                    intrinsics = repeat(intrinsics, 'b c -> (b n) c', n=N)

            plucker = PluckerRayEncoder.compute_plucker_coordinates(
                c2w, intrinsics, height, width
            )
            return self.encoder(plucker)


# =============================================================================
# Integration helper for MVDiffusion UNet
# =============================================================================

def create_pose_conditioner_for_unet(
    method: str = 'spherical',
    cross_attention_dim: int = 1024,
    num_views: int = 6,
) -> CameraPoseConditioner:
    """
    Factory function to create pose conditioner compatible with MVDiffusion UNet.

    The output embedding will be used in place of fixed view index embeddings.
    """
    return CameraPoseConditioner(
        method=method,
        embed_dim=cross_attention_dim,
        num_views=num_views,
    )


# =============================================================================
# Test utilities
# =============================================================================

def test_pose_encoders():
    """Test all pose encoding methods."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, N = 2, 6
    H, W = 64, 64

    # Create dummy camera matrices
    c2w = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    # Vary the translation for each view
    for i in range(N):
        angle = 2 * math.pi * i / N
        c2w[:, i, 0, 3] = 2.5 * math.sin(angle)
        c2w[:, i, 2, 3] = 2.5 * math.cos(angle)

    intrinsics = torch.tensor([512.0, 512.0, 256.0, 256.0], device=device)
    intrinsics = intrinsics.unsqueeze(0).expand(B, -1)

    print("Testing Camera Pose Conditioning modules...")
    print(f"Input: c2w shape = {c2w.shape}")

    # Test Spherical
    spherical = CameraPoseConditioner(method='spherical', embed_dim=1024).to(device)
    out_spherical = spherical(c2w)
    print(f"Spherical output shape: {out_spherical.shape}")  # [B, N, 1024]

    # Test Extrinsic
    extrinsic = CameraPoseConditioner(method='extrinsic', embed_dim=1024).to(device)
    out_extrinsic = extrinsic(c2w)
    print(f"Extrinsic output shape: {out_extrinsic.shape}")  # [B, N, 1024]

    # Test Plücker
    plucker = CameraPoseConditioner(method='plucker', spatial_dim=320).to(device)
    out_plucker = plucker(c2w, intrinsics, H, W)
    print(f"Plücker output shape: {out_plucker.shape}")  # [B*N, 320, H, W]

    print("All tests passed!")


if __name__ == '__main__':
    test_pose_encoders()
