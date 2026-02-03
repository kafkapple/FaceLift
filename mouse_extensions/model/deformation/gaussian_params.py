# Copyright 2026 FaceLift Mouse Extensions
# GaussianParams: Data structure for Gaussian Splatting parameters

"""
GaussianParams: Structured container for 3D Gaussian parameters.

Compatible with GS-LRM output format:
- xyz: [N, 3] positions
- features: [N, (sh_degree+1)^2 * 3] SH coefficients
- scaling: [N, 3] log-scale values
- rotation: [N, 4] quaternions
- opacity: [N, 1] logit values

Note: scaling and opacity are in log/logit space (pre-activation).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GaussianParams:
    """
    Container for 3D Gaussian Splatting parameters.
    
    All tensors are expected to have shape [N, dim] or [B, N, dim].
    Values are stored in their raw (pre-activation) form to match GS-LRM.
    """
    
    xyz: torch.Tensor           # [N, 3] or [B, N, 3] - positions
    features: torch.Tensor      # [N, F] or [B, N, F] - SH coefficients
    scaling: torch.Tensor       # [N, 3] or [B, N, 3] - log-scale
    rotation: torch.Tensor      # [N, 4] or [B, N, 4] - quaternion
    opacity: torch.Tensor       # [N, 1] or [B, N, 1] - logit
    
    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians."""
        return self.xyz.shape[-2]
    
    @property
    def device(self) -> torch.device:
        return self.xyz.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.xyz.dtype
    
    @property
    def is_batched(self) -> bool:
        """Check if params have batch dimension."""
        return self.xyz.dim() == 3
    
    def clone(self) -> "GaussianParams":
        """Create a deep copy."""
        return GaussianParams(
            xyz=self.xyz.clone(),
            features=self.features.clone(),
            scaling=self.scaling.clone(),
            rotation=self.rotation.clone(),
            opacity=self.opacity.clone(),
        )
    
    def detach(self) -> "GaussianParams":
        """Detach from computation graph."""
        return GaussianParams(
            xyz=self.xyz.detach(),
            features=self.features.detach(),
            scaling=self.scaling.detach(),
            rotation=self.rotation.detach(),
            opacity=self.opacity.detach(),
        )
    
    def to(self, device: torch.device) -> "GaussianParams":
        """Move to device."""
        return GaussianParams(
            xyz=self.xyz.to(device),
            features=self.features.to(device),
            scaling=self.scaling.to(device),
            rotation=self.rotation.to(device),
            opacity=self.opacity.to(device),
        )
    
    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        sh_degree: int = 2,
    ) -> "GaussianParams":
        """
        Parse from concatenated tensor (GS-LRM output format).
        
        Args:
            tensor: [N, D] or [B, N, D] where D = 3 + F + 3 + 4 + 1
            sh_degree: SH degree (determines feature dim)
        
        Returns:
            GaussianParams instance
        """
        feature_dim = (sh_degree + 1) ** 2 * 3
        expected_dim = 3 + feature_dim + 3 + 4 + 1
        
        assert tensor.shape[-1] == expected_dim, (
            f"Expected dim {expected_dim}, got {tensor.shape[-1]}"
        )
        
        # Split tensor
        splits = [3, feature_dim, 3, 4, 1]
        xyz, features, scaling, rotation, opacity = tensor.split(splits, dim=-1)
        
        return cls(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )
    
    def to_tensor(self) -> torch.Tensor:
        """Concatenate to single tensor (GS-LRM format)."""
        return torch.cat([
            self.xyz,
            self.features,
            self.scaling,
            self.rotation,
            self.opacity,
        ], dim=-1)
    
    def apply_deformation(
        self,
        deformation: Dict[str, torch.Tensor],
        clamp_opacity: bool = True,
        clamp_scale: bool = True,
    ) -> "GaussianParams":
        """
        Apply deformation offsets to create new GaussianParams.
        
        Args:
            deformation: dict with keys 'position', 'opacity', 'scale'
                - position: [N, 3] xyz offset
                - opacity: [N, 1] logit offset
                - scale: [N, 1] or [N, 3] log-scale offset
            clamp_opacity: Clamp opacity to valid range
            clamp_scale: Clamp scale to valid range
        
        Returns:
            New GaussianParams with applied deformation
        """
        new_xyz = self.xyz
        new_opacity = self.opacity
        new_scaling = self.scaling
        
        # Position offset
        if "position" in deformation and deformation["position"] is not None:
            new_xyz = self.xyz + deformation["position"]
        
        # Opacity offset (in logit space)
        if "opacity" in deformation and deformation["opacity"] is not None:
            new_opacity = self.opacity + deformation["opacity"]
            if clamp_opacity:
                # Clamp logit to reasonable range (-10, 10)
                new_opacity = new_opacity.clamp(-10.0, 10.0)
        
        # Scale offset (in log space)
        if "scale" in deformation and deformation["scale"] is not None:
            scale_offset = deformation["scale"]
            # Handle isotropic vs anisotropic
            if scale_offset.shape[-1] == 1:
                scale_offset = scale_offset.expand_as(self.scaling)
            new_scaling = self.scaling + scale_offset
            if clamp_scale:
                # Clamp log-scale to reasonable range
                new_scaling = new_scaling.clamp(-10.0, 2.0)
        
        return GaussianParams(
            xyz=new_xyz,
            features=self.features,  # Features unchanged
            scaling=new_scaling,
            rotation=self.rotation,  # Rotation unchanged (or add if needed)
            opacity=new_opacity,
        )
    
    def get_activated_opacity(self) -> torch.Tensor:
        """Get opacity in [0, 1] range (sigmoid activated)."""
        return torch.sigmoid(self.opacity)
    
    def get_activated_scaling(self) -> torch.Tensor:
        """Get positive scaling (exp activated)."""
        return torch.exp(self.scaling)


# ============================================================
# Unit Tests
# ============================================================

def _test_gaussian_params():
    """Unit test for GaussianParams."""
    print("Testing GaussianParams...")
    
    N = 500
    sh_degree = 2
    feature_dim = (sh_degree + 1) ** 2 * 3  # 27
    
    # Test 1: Create from individual tensors
    params = GaussianParams(
        xyz=torch.randn(N, 3),
        features=torch.randn(N, feature_dim),
        scaling=torch.randn(N, 3),
        rotation=F.normalize(torch.randn(N, 4), dim=-1),
        opacity=torch.randn(N, 1),
    )
    print(f"  Created: {params.num_gaussians} Gaussians")
    
    # Test 2: Clone
    params2 = params.clone()
    params2.xyz[0] = 999
    assert not torch.equal(params.xyz[0], params2.xyz[0]), "Clone should be independent"
    print("  Clone: ✓")
    
    # Test 3: to_tensor and from_tensor round-trip
    tensor = params.to_tensor()
    expected_dim = 3 + feature_dim + 3 + 4 + 1
    assert tensor.shape == (N, expected_dim), f"Expected {(N, expected_dim)}, got {tensor.shape}"
    
    params_restored = GaussianParams.from_tensor(tensor, sh_degree=sh_degree)
    assert torch.allclose(params.xyz, params_restored.xyz), "Round-trip xyz mismatch"
    assert torch.allclose(params.opacity, params_restored.opacity), "Round-trip opacity mismatch"
    print(f"  Round-trip: tensor shape {tensor.shape} ✓")
    
    # Test 4: Apply deformation
    deformation = {
        "position": torch.randn(N, 3) * 0.1,
        "opacity": torch.randn(N, 1) * 0.1,
        "scale": torch.randn(N, 1) * 0.1,  # isotropic
    }
    params_deformed = params.apply_deformation(deformation)
    
    # Check position changed
    assert not torch.allclose(params.xyz, params_deformed.xyz), "Position should change"
    # Check features unchanged
    assert torch.allclose(params.features, params_deformed.features), "Features should be unchanged"
    print("  Apply deformation: ✓")
    
    # Test 5: Batched tensors
    B = 4
    batched_tensor = torch.randn(B, N, expected_dim)
    batched_params = GaussianParams.from_tensor(batched_tensor, sh_degree=sh_degree)
    assert batched_params.is_batched, "Should detect batch dimension"
    assert batched_params.num_gaussians == N, f"Expected {N} Gaussians"
    print(f"  Batched: {B} x {N} Gaussians ✓")
    
    # Test 6: Activations
    activated_opacity = params.get_activated_opacity()
    assert (activated_opacity >= 0).all() and (activated_opacity <= 1).all(), "Opacity should be in [0,1]"
    
    activated_scale = params.get_activated_scaling()
    assert (activated_scale > 0).all(), "Scale should be positive"
    print("  Activations: ✓")
    
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    _test_gaussian_params()
