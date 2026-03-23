"""Neural Mesh Texture: learns RGB appearance from UV coordinates.

Predicts per-pixel RGB from UV coordinates (+ optional keypoint pose),
supervised by GT camera images via non-differentiable rasterized UV maps.

Pipeline:
    1. Pre-compute UV maps: pyrender rasterizes MAMMAL mesh → per-pixel UV coords
    2. Train MLP: UV(2D) [+ pose(66D)] → RGB(3D), supervised by GT images
    3. Inference: query trained MLP at any UV coordinate → textured mesh render

Usage:
    from mouse_extensions.model.neural_texture import NeuralTextureMLP, FourierEncoder

    model = NeuralTextureMLP(use_pose=False)
    rgb = model(uv_coords)  # (B, N, 2) → (B, N, 3)
"""

import torch
import torch.nn as nn
import math


class FourierEncoder(nn.Module):
    """Positional encoding with Fourier features for UV coordinates.

    Maps 2D UV to higher-dimensional space for MLP to capture high-frequency
    texture patterns (fur detail, color variation).

    Output dim = in_dim * (2 * num_freqs + 1) if include_input else in_dim * 2 * num_freqs
    """

    def __init__(self, in_dim: int = 2, num_freqs: int = 8, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Frequency bands: 2^0, 2^1, ..., 2^(num_freqs-1)
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        d = self.in_dim * 2 * self.num_freqs
        if self.include_input:
            d += self.in_dim
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input coordinates with Fourier features.

        Args:
            x: (..., in_dim) input coordinates

        Returns:
            (..., out_dim) encoded features
        """
        parts = []
        if self.include_input:
            parts.append(x)

        # x: (..., D), freqs: (F,) → scaled: (..., D, F) → (..., D*F)
        scaled = x.unsqueeze(-1) * self.freqs * math.pi  # (..., D, F)
        parts.append(torch.sin(scaled).flatten(-2))
        parts.append(torch.cos(scaled).flatten(-2))

        return torch.cat(parts, dim=-1)


class NeuralTextureMLP(nn.Module):
    """MLP that predicts RGB from UV coordinates and optional pose.

    Architecture:
        UV (2D) → FourierEncoder → [concat pose_embed] → MLP → RGB (3D)

    Args:
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        num_freqs: Fourier encoding frequency bands
        use_pose: Whether to condition on keypoint pose
        pose_dim: Input pose dimension (22 keypoints × 3 = 66)
        pose_embed_dim: Pose embedding dimension after projection
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_freqs: int = 8,
        use_pose: bool = False,
        pose_dim: int = 66,
        pose_embed_dim: int = 64,
    ):
        super().__init__()
        self.use_pose = use_pose

        # UV encoder
        self.uv_encoder = FourierEncoder(in_dim=2, num_freqs=num_freqs)
        in_features = self.uv_encoder.out_dim

        # Optional pose encoder
        if use_pose:
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_dim, pose_embed_dim),
                nn.ReLU(),
                nn.Linear(pose_embed_dim, pose_embed_dim),
            )
            in_features += pose_embed_dim

        # MLP with skip connection at middle layer
        layers = []
        skip_layer = num_layers // 2
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
            elif i == skip_layer:
                layers.append(nn.Linear(hidden_dim + in_features, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.skip_layer = skip_layer

        self.layers = nn.ModuleList(
            [layers[i * 2] for i in range(num_layers)]
        )
        self.activations = nn.ModuleList(
            [layers[i * 2 + 1] for i in range(num_layers)]
        )

        # Output head
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(
        self, uv: torch.Tensor, pose: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict RGB from UV coordinates.

        Args:
            uv: (..., 2) UV coordinates in [0, 1]
            pose: (..., pose_dim) flattened keypoint positions (optional)

        Returns:
            (..., 3) RGB values in [0, 1]
        """
        # Encode UV
        feat = self.uv_encoder(uv)

        # Encode and concat pose
        if self.use_pose and pose is not None:
            # Broadcast pose to match UV spatial dims
            if pose.dim() < feat.dim():
                for _ in range(feat.dim() - pose.dim()):
                    pose = pose.unsqueeze(-2)
                pose = pose.expand(*feat.shape[:-1], -1)
            pose_feat = self.pose_encoder(pose)
            feat = torch.cat([feat, pose_feat], dim=-1)

        # MLP with skip connection
        h = feat
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            if i == self.skip_layer:
                h = torch.cat([h, feat], dim=-1)
            h = act(layer(h))

        return self.rgb_head(h)


def build_neural_texture(use_pose: bool = False, **kwargs) -> NeuralTextureMLP:
    """Factory function with sensible defaults."""
    defaults = dict(
        hidden_dim=256,
        num_layers=6,
        num_freqs=8,
        pose_dim=66,
        pose_embed_dim=64,
    )
    defaults.update(kwargs)
    return NeuralTextureMLP(use_pose=use_pose, **defaults)
