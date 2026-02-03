# Copyright 2026 FaceLift Mouse Extensions
# Temporal Pipeline: Autoregressive Gaussian deformation for video consistency

"""
TemporalGaussianPipeline: Autoregressive generation for temporal consistency.

Pipeline (from FaceLift paper Appendix 3.5):
1. Generate initial Gaussians for each frame using GS-LRM
2. Select anchor frame (canonical Gaussians)
3. Propagate deformation autoregressively: G_t -> D(G_t) -> G'_{t+1}
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .deformation_network import DeformationNetwork, DeformationConfig
from .gaussian_params import GaussianParams


@dataclass
class TemporalConfig:
    """Configuration for temporal pipeline."""
    
    # Anchor frame selection
    anchor_frame_idx: int = 0  # First frame as canonical
    
    # Propagation direction
    bidirectional: bool = True  # Propagate both forward and backward
    
    # Deformation network config
    deform_config: DeformationConfig = None
    
    def __post_init__(self):
        if self.deform_config is None:
            self.deform_config = DeformationConfig()


class TemporalGaussianPipeline(nn.Module):
    """
    Autoregressive pipeline for temporal Gaussian consistency.
    
    Given a sequence of independently generated Gaussians,
    applies learned deformation to ensure temporal smoothness.
    """
    
    def __init__(self, config: Optional[TemporalConfig] = None):
        super().__init__()
        
        self.config = config or TemporalConfig()
        
        # Deformation network
        self.deform_net = DeformationNetwork(self.config.deform_config)
    
    def forward(
        self,
        gaussians_sequence: List[GaussianParams],
        return_intermediate: bool = False,
    ) -> List[GaussianParams]:
        """
        Apply autoregressive deformation to Gaussian sequence.
        
        Args:
            gaussians_sequence: List of T GaussianParams (independent per frame)
            return_intermediate: If True, also return deformation tensors
            
        Returns:
            List of T deformed GaussianParams with temporal consistency
        """
        T = len(gaussians_sequence)
        anchor_idx = min(self.config.anchor_frame_idx, T - 1)
        
        # Initialize output list
        deformed = [None] * T
        deformed[anchor_idx] = gaussians_sequence[anchor_idx].clone()
        
        intermediates = [] if return_intermediate else None
        
        # Forward propagation: anchor -> T-1
        for t in range(anchor_idx + 1, T):
            prev_G = deformed[t - 1]
            
            # Predict deformation
            deform_raw = self.deform_net(prev_G.xyz)
            deform_dict = self.deform_net.parse_output(deform_raw)
            
            # Apply deformation
            deformed[t] = prev_G.apply_deformation(deform_dict)
            
            if return_intermediate:
                intermediates.append({
                    "t": t,
                    "direction": "forward",
                    "deformation": deform_dict,
                })
        
        # Backward propagation: anchor -> 0
        if self.config.bidirectional:
            for t in range(anchor_idx - 1, -1, -1):
                next_G = deformed[t + 1]
                
                # Predict negative deformation (backward)
                deform_raw = self.deform_net(next_G.xyz)
                deform_dict = self.deform_net.parse_output(deform_raw)
                
                # Negate for backward direction
                for key in deform_dict:
                    if deform_dict[key] is not None:
                        deform_dict[key] = -deform_dict[key]
                
                deformed[t] = next_G.apply_deformation(deform_dict)
                
                if return_intermediate:
                    intermediates.append({
                        "t": t,
                        "direction": "backward",
                        "deformation": deform_dict,
                    })
        
        if return_intermediate:
            return deformed, intermediates
        return deformed
    
    def deform_single_step(
        self,
        gaussian: GaussianParams,
        forward: bool = True,
    ) -> Tuple[GaussianParams, dict]:
        """
        Single-step deformation for training.
        
        Args:
            gaussian: Input GaussianParams
            forward: If True, predict next frame; else previous
            
        Returns:
            (deformed_gaussian, deformation_dict)
        """
        deform_raw = self.deform_net(gaussian.xyz)
        deform_dict = self.deform_net.parse_output(deform_raw)
        
        if not forward:
            for key in deform_dict:
                if deform_dict[key] is not None:
                    deform_dict[key] = -deform_dict[key]
        
        deformed = gaussian.apply_deformation(deform_dict)
        return deformed, deform_dict
    
    def compute_temporal_consistency_loss(
        self,
        gaussians_sequence: List[GaussianParams],
    ) -> torch.Tensor:
        """
        Compute loss encouraging temporal smoothness.
        
        Computes difference between consecutive deformed Gaussians.
        """
        if len(gaussians_sequence) < 2:
            return torch.tensor(0.0, device=gaussians_sequence[0].device)
        
        deformed = self.forward(gaussians_sequence)
        
        loss = 0.0
        for t in range(len(deformed) - 1):
            # Position consistency
            pos_diff = (deformed[t].xyz - deformed[t + 1].xyz).pow(2).mean()
            loss = loss + pos_diff
        
        return loss / (len(deformed) - 1)


# ============================================================
# Unit Tests
# ============================================================

def _test_temporal_pipeline():
    """Unit test for TemporalGaussianPipeline."""
    print("Testing TemporalGaussianPipeline...")
    
    N = 100  # Gaussians per frame
    T = 5    # Number of frames
    sh_degree = 2
    feature_dim = (sh_degree + 1) ** 2 * 3
    
    # Create synthetic Gaussian sequence
    def make_gaussian():
        return GaussianParams(
            xyz=torch.randn(N, 3),
            features=torch.randn(N, feature_dim),
            scaling=torch.randn(N, 3),
            rotation=torch.nn.functional.normalize(torch.randn(N, 4), dim=-1),
            opacity=torch.randn(N, 1),
        )
    
    gaussians_seq = [make_gaussian() for _ in range(T)]
    print(f"  Created sequence: {T} frames, {N} Gaussians each")
    
    # Test 1: Create pipeline
    config = TemporalConfig(anchor_frame_idx=0)
    pipeline = TemporalGaussianPipeline(config)
    print(f"  Pipeline: {pipeline.deform_net}")
    
    # Test 2: Forward pass
    deformed = pipeline(gaussians_seq)
    assert len(deformed) == T, f"Expected {T} frames, got {len(deformed)}"
    assert all(d is not None for d in deformed), "All frames should be deformed"
    print(f"  Forward: {T} frames deformed")
    
    # Test 3: Anchor frame unchanged (with zero-init network)
    anchor_diff = (deformed[0].xyz - gaussians_seq[0].xyz).abs().max()
    assert anchor_diff < 1e-5, f"Anchor should be unchanged, diff={anchor_diff}"
    print(f"  Anchor preservation: ✓ (diff={anchor_diff:.2e})")
    
    # Test 4: With intermediate outputs
    deformed2, intermediates = pipeline(gaussians_seq, return_intermediate=True)
    assert len(intermediates) == T - 1, f"Expected {T-1} intermediates"
    print(f"  Intermediates: {len(intermediates)} deformation steps")
    
    # Test 5: Single step deformation
    G_single, deform_dict = pipeline.deform_single_step(gaussians_seq[0])
    assert "position" in deform_dict
    print(f"  Single step: position offset shape {deform_dict['position'].shape}")
    
    # Test 6: Temporal consistency loss
    loss = pipeline.compute_temporal_consistency_loss(gaussians_seq)
    assert loss.dim() == 0, "Loss should be scalar"
    print(f"  Temporal loss: {loss.item():.4f}")
    
    # Test 7: Gradient flow
    gaussians_seq[0].xyz.requires_grad_(True)
    deformed3 = pipeline(gaussians_seq)
    loss = sum(d.xyz.sum() for d in deformed3)
    loss.backward()
    assert gaussians_seq[0].xyz.grad is not None, "Gradients should flow"
    print("  Gradient flow: ✓")
    
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    _test_temporal_pipeline()
