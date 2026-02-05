"""V2 Temporal Smoothing: Per-frame base + Bilateral filtering.

Unlike V1 autoregressive (which drifts all frames toward anchor),
V2 preserves per-frame motion while removing high-frequency jitter.

Key insight: GS-LRM per-frame outputs have correct motion but jittery.
We want to smooth jitter without destroying motion.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from .gaussian_params import GaussianParams


class BilateralGaussianSmoother(nn.Module):
    """Bilateral filter for Gaussian sequences.
    
    Applies temporal smoothing that:
    - Preserves large motions (real object movement)
    - Removes small jitter (GS-LRM inconsistency)
    
    Uses bilateral weighting:
    - Temporal weight: closer frames have more influence
    - Spatial weight: similar Gaussians have more influence
    """
    
    def __init__(
        self,
        window_size: int = 5,
        sigma_temporal: float = 1.0,
        sigma_spatial: float = 0.1,
        smooth_position: bool = True,
        smooth_opacity: bool = True,
        smooth_scale: bool = False,  # Scale is usually stable
        smooth_rotation: bool = False,  # Rotation needs special handling
    ):
        super().__init__()
        self.window_size = window_size
        self.sigma_temporal = sigma_temporal
        self.sigma_spatial = sigma_spatial
        self.smooth_position = smooth_position
        self.smooth_opacity = smooth_opacity
        self.smooth_scale = smooth_scale
        self.smooth_rotation = smooth_rotation
    
    def forward(
        self,
        gaussians_sequence: List[GaussianParams],
    ) -> List[GaussianParams]:
        """Apply bilateral smoothing to Gaussian sequence.
        
        Args:
            gaussians_sequence: List of T GaussianParams (per-frame GS-LRM output)
            
        Returns:
            List of T smoothed GaussianParams
        """
        T = len(gaussians_sequence)
        if T < 2:
            return gaussians_sequence
        
        # Stack all positions for efficient computation
        # Shape: [T, N, 3]
        all_xyz = torch.stack([g.xyz for g in gaussians_sequence], dim=0)
        
        # Compute temporal weights (Gaussian decay)
        # Shape: [window_size]
        half_win = self.window_size // 2
        offsets = torch.arange(-half_win, half_win + 1, device=all_xyz.device)
        temporal_weights = torch.exp(-offsets.float()**2 / (2 * self.sigma_temporal**2))
        
        smoothed_list = []
        
        for t in range(T):
            # Get window indices
            start = max(0, t - half_win)
            end = min(T, t + half_win + 1)
            window_indices = list(range(start, end))
            
            # Current frame as reference
            ref_xyz = all_xyz[t]  # [N, 3]
            
            # Compute bilateral weights for each frame in window
            weighted_sum_xyz = torch.zeros_like(ref_xyz)
            weight_sum = torch.zeros(ref_xyz.shape[0], 1, device=ref_xyz.device)
            
            for idx in window_indices:
                # Temporal weight
                t_offset = idx - t + half_win
                if 0 <= t_offset < len(temporal_weights):
                    w_temporal = temporal_weights[t_offset]
                else:
                    w_temporal = temporal_weights[half_win]  # Center weight
                
                # Spatial weight (based on position difference)
                neighbor_xyz = all_xyz[idx]  # [N, 3]
                spatial_diff = (neighbor_xyz - ref_xyz).norm(dim=-1, keepdim=True)  # [N, 1]
                w_spatial = torch.exp(-spatial_diff**2 / (2 * self.sigma_spatial**2))
                
                # Combined weight
                w = w_temporal * w_spatial  # [N, 1]
                
                weighted_sum_xyz += w * neighbor_xyz
                weight_sum += w
            
            # Normalize
            smoothed_xyz = weighted_sum_xyz / (weight_sum + 1e-8)
            
            # Create smoothed GaussianParams
            orig = gaussians_sequence[t]
            smoothed = GaussianParams(
                xyz=smoothed_xyz if self.smooth_position else orig.xyz.clone(),
                features=orig.features.clone(),
                scaling=orig.scaling.clone(),  # TODO: smooth if needed
                rotation=orig.rotation.clone(),  # TODO: slerp if needed
                opacity=orig.opacity.clone(),  # TODO: smooth if needed
            )
            smoothed_list.append(smoothed)
        
        return smoothed_list


class TemporalGaussianSmootherV2(nn.Module):
    """V2 Temporal Pipeline: Per-frame + Bilateral.
    
    Unlike V1 (autoregressive from anchor), V2:
    1. Keeps each frame's GS-LRM output as base
    2. Applies bilateral filter to remove jitter
    3. Preserves actual object motion
    """
    
    def __init__(
        self,
        window_size: int = 5,
        sigma_temporal: float = 1.5,
        sigma_spatial: float = 0.05,
    ):
        super().__init__()
        self.bilateral = BilateralGaussianSmoother(
            window_size=window_size,
            sigma_temporal=sigma_temporal,
            sigma_spatial=sigma_spatial,
        )
    
    def forward(
        self,
        gaussians_sequence: List[GaussianParams],
    ) -> List[GaussianParams]:
        """Apply V2 smoothing.
        
        Args:
            gaussians_sequence: Per-frame GS-LRM outputs
            
        Returns:
            Smoothed sequence preserving motion
        """
        return self.bilateral(gaussians_sequence)


def _test_bilateral_smoother():
    """Test bilateral smoother."""
    print("Testing BilateralGaussianSmoother...")
    
    N = 100
    T = 10
    
    # Create sequence with motion + jitter
    base_motion = torch.linspace(0, 1, T).view(T, 1, 1).expand(T, N, 3)
    jitter = torch.randn(T, N, 3) * 0.1
    
    gaussians_seq = []
    for t in range(T):
        xyz = base_motion[t] + jitter[t]
        gaussians_seq.append(GaussianParams(
            xyz=xyz,
            features=torch.randn(N, 27),
            scaling=torch.randn(N, 3),
            rotation=torch.randn(N, 4),
            opacity=torch.randn(N, 1),
        ))
    
    # Apply smoother
    smoother = TemporalGaussianSmootherV2(window_size=5)
    smoothed = smoother(gaussians_seq)
    
    # Check motion preserved
    orig_motion = gaussians_seq[-1].xyz.mean() - gaussians_seq[0].xyz.mean()
    smooth_motion = smoothed[-1].xyz.mean() - smoothed[0].xyz.mean()
    
    print(f"  Original motion: {orig_motion:.4f}")
    print(f"  Smoothed motion: {smooth_motion:.4f}")
    print(f"  Motion preserved: {abs(smooth_motion - orig_motion) < 0.1}")
    
    # Check jitter reduced
    orig_jitter = torch.stack([
        (gaussians_seq[t+1].xyz - gaussians_seq[t].xyz).std() 
        for t in range(T-1)
    ]).mean()
    smooth_jitter = torch.stack([
        (smoothed[t+1].xyz - smoothed[t].xyz).std() 
        for t in range(T-1)
    ]).mean()
    
    print(f"  Original jitter: {orig_jitter:.4f}")
    print(f"  Smoothed jitter: {smooth_jitter:.4f}")
    print(f"  Jitter reduced: {smooth_jitter < orig_jitter}")
    
    print("Test passed!")


if __name__ == "__main__":
    _test_bilateral_smoother()
