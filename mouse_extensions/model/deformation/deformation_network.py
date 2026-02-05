# Copyright 2026 FaceLift Mouse Extensions
# Deformation Network: 8-layer MLP for temporal Gaussian consistency
# Based on FaceLift paper Appendix 3.5 "Applying FaceLift on Videos"

"""
Deformation Network for predicting Gaussian parameter offsets between frames.

Architecture (from paper):
- 8-layer MLP
- Input: Gaussian 3D positions [N, 3]
- Output: Deformation parameters [N, output_dim]
  - Position offset: Δx, Δy, Δz
  - Opacity offset: Δα
  - Scale offset: Δs (isotropic) or Δsx, Δsy, Δsz (anisotropic)
"""

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeformationConfig:
    """Configuration for DeformationNetwork."""
    
    # Architecture
    input_dim: int = 3  # xyz positions
    hidden_dim: int = 256
    num_layers: int = 8
    activation: Literal["relu", "gelu", "silu"] = "relu"
    
    # Output options
    predict_position: bool = True   # Δxyz
    predict_opacity: bool = True    # Δα
    predict_scale: bool = True      # Δs
    anisotropic_scale: bool = False # Δsxyz vs Δs
    predict_rotation: bool = False  # Δq (quaternion) - optional
    
    # Regularization
    dropout: float = 0.0
    weight_decay: float = 0.0
    
    # Initialization
    zero_init_output: bool = True  # Start with identity deformation
    
    # Positional encoding (for fine-grained control)
    use_positional_encoding: bool = False
    pe_freq_bands: int = 10
    
    @property
    def output_dim(self) -> int:
        """Compute output dimension based on config."""
        dim = 0
        if self.predict_position:
            dim += 3
        if self.predict_opacity:
            dim += 1
        if self.predict_scale:
            dim += 3 if self.anisotropic_scale else 1
        if self.predict_rotation:
            dim += 4  # quaternion
        return dim
    
    @property
    def effective_input_dim(self) -> int:
        """Input dim after positional encoding."""
        if self.use_positional_encoding:
            return self.input_dim + self.input_dim * 2 * self.pe_freq_bands
        return self.input_dim


class PositionalEncoding(nn.Module):
    """Fourier positional encoding for fine-grained spatial control."""
    
    def __init__(self, input_dim: int = 3, freq_bands: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.freq_bands = freq_bands
        
        # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.linspace(0, freq_bands - 1, freq_bands)
        self.register_buffer("freqs", freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] positions
        Returns:
            encoded: [N, input_dim + input_dim * 2 * freq_bands]
        """
        # x: [N, D], freqs: [L]
        # x_expanded: [N, D, 1] * freqs: [L] -> [N, D, L]
        x_expanded = x.unsqueeze(-1) * self.freqs * 2 * torch.pi
        
        # sin and cos: [N, D, L] each
        sin_enc = torch.sin(x_expanded)
        cos_enc = torch.cos(x_expanded)
        
        # Flatten: [N, D * L * 2]
        encoded = torch.cat([
            sin_enc.flatten(start_dim=1),
            cos_enc.flatten(start_dim=1),
        ], dim=-1)
        
        # Concat with original: [N, D + D * L * 2]
        return torch.cat([x, encoded], dim=-1)


class DeformationNetwork(nn.Module):
    """
    8-layer MLP for Gaussian deformation prediction.
    
    Predicts per-Gaussian offsets for temporal consistency:
    G_{t+1} = G_t + D(G_t.positions)
    """
    
    def __init__(self, config: Optional[DeformationConfig] = None):
        super().__init__()
        
        self.config = config or DeformationConfig()
        cfg = self.config
        
        # Positional encoding (optional)
        if cfg.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                input_dim=cfg.input_dim,
                freq_bands=cfg.pe_freq_bands,
            )
        else:
            self.pos_encoder = None
        
        # Build MLP layers
        layers = []
        in_dim = cfg.effective_input_dim
        
        # First layer
        layers.append(nn.Linear(in_dim, cfg.hidden_dim))
        layers.append(self._get_activation(cfg.activation))
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        
        # Hidden layers (num_layers - 2 middle layers)
        for _ in range(cfg.num_layers - 2):
            layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            layers.append(self._get_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        
        # Output layer
        layers.append(nn.Linear(cfg.hidden_dim, cfg.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-initialize output layer for identity deformation at start
        if self.config.zero_init_output:
            last_linear = None
            for m in reversed(list(self.mlp.modules())):
                if isinstance(m, nn.Linear):
                    last_linear = m
                    break
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Predict deformation parameters from Gaussian positions.
        
        Args:
            positions: [N, 3] Gaussian center positions
            
        Returns:
            deformations: [N, output_dim] predicted offsets
                Layout depends on config:
                - [Δx, Δy, Δz] if predict_position
                - [Δα] if predict_opacity  
                - [Δs] or [Δsx, Δsy, Δsz] if predict_scale
        """
        if self.pos_encoder is not None:
            x = self.pos_encoder(positions)
        else:
            x = positions
        
        return self.mlp(x)
    
    def parse_output(self, deformations: torch.Tensor) -> dict:
        """
        Parse network output into named components.
        
        Args:
            deformations: [N, output_dim] raw network output
            
        Returns:
            dict with keys: 'position', 'opacity', 'scale', 'rotation'
        """
        cfg = self.config
        result = {}
        idx = 0
        
        if cfg.predict_position:
            result["position"] = deformations[:, idx:idx+3]
            idx += 3
        
        if cfg.predict_opacity:
            result["opacity"] = deformations[:, idx:idx+1]
            idx += 1
        
        if cfg.predict_scale:
            scale_dim = 3 if cfg.anisotropic_scale else 1
            result["scale"] = deformations[:, idx:idx+scale_dim]
            idx += scale_dim
        
        if cfg.predict_rotation:
            result["rotation"] = deformations[:, idx:idx+4]
            idx += 4
        
        return result
    
    def get_info(self) -> str:
        """Get info string for logging."""
        return repr(self)

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"DeformationNetwork("
            f"layers={cfg.num_layers}, "
            f"hidden={cfg.hidden_dim}, "
            f"output={cfg.output_dim}, "
            f"params={self.get_num_params():,}"
            f")"
        )


# ============================================================
# Unit Tests
# ============================================================

def _test_deformation_network():
    """Unit test for DeformationNetwork."""
    print("Testing DeformationNetwork...")
    
    # Test 1: Default config
    config = DeformationConfig()
    net = DeformationNetwork(config)
    print(f"  Created: {net}")
    
    # Test 2: Forward pass
    N = 1000  # Number of Gaussians
    positions = torch.randn(N, 3)
    output = net(positions)
    
    assert output.shape == (N, config.output_dim), f"Expected {(N, config.output_dim)}, got {output.shape}"
    print(f"  Forward: input {positions.shape} -> output {output.shape}")
    
    # Test 3: Zero initialization check
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6), "Output should be ~0 with zero init"
    print(f"  Zero init: output mean={output.mean().item():.6f}, std={output.std().item():.6f}")
    
    # Test 4: Parse output
    parsed = net.parse_output(output)
    assert "position" in parsed
    assert "opacity" in parsed
    assert "scale" in parsed
    print(f"  Parsed keys: {list(parsed.keys())}")
    
    # Test 5: With positional encoding
    config_pe = DeformationConfig(use_positional_encoding=True, pe_freq_bands=6)
    net_pe = DeformationNetwork(config_pe)
    output_pe = net_pe(positions)
    print(f"  With PE: effective_input_dim={config_pe.effective_input_dim}, output={output_pe.shape}")
    
    # Test 6: Gradient flow
    positions.requires_grad_(True)
    output = net(positions)
    loss = output.sum()
    loss.backward()
    assert positions.grad is not None, "Gradients should flow"
    print(f"  Gradient flow: ✓")
    
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    _test_deformation_network()


# ============================================================
# V2: Deformation Network with Per-Frame Information
# ============================================================

@dataclass
class DeformationConfigV2:
    """Configuration for DeformationNetworkV2 with per-frame information."""
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 8
    activation: Literal["relu", "gelu", "silu"] = "relu"
    
    # Input options
    use_positional_encoding: bool = True
    pe_freq_bands: int = 6
    use_time_embedding: bool = True
    time_embed_dim: int = 32
    
    # Output options
    predict_position: bool = True   # Δxyz
    predict_opacity: bool = True    # Δα
    predict_scale: bool = True      # Δs
    anisotropic_scale: bool = False
    predict_rotation: bool = False
    
    # Regularization
    dropout: float = 0.0
    
    # Initialization
    zero_init_output: bool = True
    
    @property
    def output_dim(self) -> int:
        dim = 0
        if self.predict_position:
            dim += 3
        if self.predict_opacity:
            dim += 1
        if self.predict_scale:
            dim += 3 if self.anisotropic_scale else 1
        if self.predict_rotation:
            dim += 4
        return dim
    
    @property
    def raw_input_dim(self) -> int:
        """Raw input dim before positional encoding."""
        # G_t.xyz (3) + G_{t+1}.xyz (3) = 6
        return 6
    
    @property
    def effective_input_dim(self) -> int:
        """Input dim after positional encoding + time embedding."""
        dim = self.raw_input_dim
        if self.use_positional_encoding:
            dim = dim + dim * 2 * self.pe_freq_bands
        if self.use_time_embedding:
            dim += self.time_embed_dim
        return dim


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for temporal information."""
    
    def __init__(self, embed_dim: int = 32, max_time: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Precompute frequency bands
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -torch.arange(half_dim) * (torch.log(torch.tensor(max_time)) / half_dim)
        )
        self.register_buffer("freqs", freqs)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [N] or [N, 1] time indices (can be float)
        Returns:
            embedding: [N, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [N, 1]
        
        # t: [N, 1], freqs: [D/2] -> [N, D/2]
        args = t * self.freqs.unsqueeze(0)
        
        # sin and cos: [N, D/2] each
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class DeformationNetworkV2(nn.Module):
    """
    V2 Deformation Network with per-frame information.
    
    Key differences from V1:
    - Input: G_t.xyz AND G_{t+1}.xyz (both frames)
    - Time embedding for temporal position
    - Predicts correction from G_t to better match G_{t+1}
    
    Usage:
        net = DeformationNetworkV2(config)
        deformation = net(G_t.xyz, G_t1.xyz, time_index)
        G_t1_corrected = G_t.apply_deformation(deformation)
    """
    
    def __init__(self, config: Optional[DeformationConfigV2] = None):
        super().__init__()
        
        self.config = config or DeformationConfigV2()
        cfg = self.config
        
        # Positional encoding for spatial features
        if cfg.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                input_dim=cfg.raw_input_dim,
                freq_bands=cfg.pe_freq_bands,
            )
        else:
            self.pos_encoder = None
        
        # Time embedding
        if cfg.use_time_embedding:
            self.time_embed = TimeEmbedding(embed_dim=cfg.time_embed_dim)
        else:
            self.time_embed = None
        
        # Build 8-layer MLP
        layers = []
        in_dim = cfg.effective_input_dim
        
        # First layer
        layers.append(nn.Linear(in_dim, cfg.hidden_dim))
        layers.append(self._get_activation(cfg.activation))
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        
        # Hidden layers (num_layers - 2)
        for _ in range(cfg.num_layers - 2):
            layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
            layers.append(self._get_activation(cfg.activation))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        
        # Output layer
        layers.append(nn.Linear(cfg.hidden_dim, cfg.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ReLU())
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-initialize output for identity deformation
        if self.config.zero_init_output:
            last_linear = None
            for m in reversed(list(self.mlp.modules())):
                if isinstance(m, nn.Linear):
                    last_linear = m
                    break
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
    
    def forward(
        self,
        xyz_t: torch.Tensor,
        xyz_t1: torch.Tensor,
        time_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict deformation using both frame positions.
        
        Args:
            xyz_t: [N, 3] Gaussian positions at time t
            xyz_t1: [N, 3] Gaussian positions at time t+1 (target reference)
            time_index: [N] or scalar, time index for embedding
            
        Returns:
            deformations: [N, output_dim] predicted offsets
        """
        N = xyz_t.shape[0]
        device = xyz_t.device
        
        # Concatenate both frame positions
        x = torch.cat([xyz_t, xyz_t1], dim=-1)  # [N, 6]
        
        # Apply positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        
        # Add time embedding
        if self.time_embed is not None:
            if time_index is None:
                time_index = torch.zeros(N, device=device)
            elif time_index.dim() == 0:
                time_index = time_index.expand(N)
            
            t_emb = self.time_embed(time_index)  # [N, time_embed_dim]
            x = torch.cat([x, t_emb], dim=-1)
        
        return self.mlp(x)
    
    def parse_output(self, deformations: torch.Tensor) -> dict:
        """Parse network output into named components."""
        cfg = self.config
        result = {}
        idx = 0
        
        if cfg.predict_position:
            result["position"] = deformations[:, idx:idx+3]
            idx += 3
        
        if cfg.predict_opacity:
            result["opacity"] = deformations[:, idx:idx+1]
            idx += 1
        
        if cfg.predict_scale:
            scale_dim = 3 if cfg.anisotropic_scale else 1
            result["scale"] = deformations[:, idx:idx+scale_dim]
            idx += scale_dim
        
        if cfg.predict_rotation:
            result["rotation"] = deformations[:, idx:idx+4]
            idx += 4
        
        return result
    
    def get_info(self) -> str:
        """Get info string for logging."""
        return repr(self)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"DeformationNetworkV2("
            f"layers={cfg.num_layers}, "
            f"hidden={cfg.hidden_dim}, "
            f"input={cfg.effective_input_dim}, "
            f"output={cfg.output_dim}, "
            f"params={self.get_num_params():,}"
            f")"
        )


def _test_deformation_network_v2():
    """Unit test for DeformationNetworkV2."""
    print("\nTesting DeformationNetworkV2...")
    
    # Test 1: Create network
    config = DeformationConfigV2()
    net = DeformationNetworkV2(config)
    print(f"  Created: {net}")
    print(f"  Input dim: {config.effective_input_dim} (raw={config.raw_input_dim})")
    
    # Test 2: Forward pass
    N = 1000
    xyz_t = torch.randn(N, 3)
    xyz_t1 = torch.randn(N, 3)
    time_idx = torch.ones(N) * 5
    
    output = net(xyz_t, xyz_t1, time_idx)
    assert output.shape == (N, config.output_dim), f"Expected {(N, config.output_dim)}, got {output.shape}"
    print(f"  Forward: ({N}, 3) + ({N}, 3) -> {output.shape}")
    
    # Test 3: Zero initialization
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6), "Output should be ~0"
    print(f"  Zero init: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
    
    # Test 4: Parse output
    parsed = net.parse_output(output)
    print(f"  Parsed: {list(parsed.keys())}")
    
    # Test 5: Gradient flow
    xyz_t.requires_grad_(True)
    xyz_t1.requires_grad_(True)
    output = net(xyz_t, xyz_t1, time_idx)
    loss = output.sum()
    loss.backward()
    assert xyz_t.grad is not None and xyz_t1.grad is not None, "Gradients should flow"
    print(f"  Gradient flow: ✓ (both inputs)")
    
    # Test 6: Without time embedding
    config_no_time = DeformationConfigV2(use_time_embedding=False)
    net_no_time = DeformationNetworkV2(config_no_time)
    output_no_time = net_no_time(xyz_t.detach(), xyz_t1.detach())
    print(f"  Without time: input_dim={config_no_time.effective_input_dim}")
    
    print("All V2 tests passed! ✓")
    return True


if __name__ == "__main__":
    _test_deformation_network()
    _test_deformation_network_v2()
