# Copyright 2026 FaceLift Mouse Extensions
# Temporal Pipeline V2: Per-frame + Regularization (NOT autoregressive)
# Fixes the "melting" issue from V1

"""
TemporalPipelineV2: Per-frame Gaussians with temporal regularization.

Key difference from V1:
- V1: Autoregressive (anchor -> deform -> deform -> ...) = cumulative drift
- V2: Per-frame original + temporal regularization = preserves per-frame quality

Usage:
    pipeline = TemporalPipelineV2(config)
    
    # Inference: smooth per-frame Gaussians
    smoothed = pipeline.smooth_sequence(original_gaussians)
    
    # Training: add regularization loss
    loss = pipeline.compute_reg_loss(xyz_sequence)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_regularization import TemporalRegularization, TemporalRegConfig
from .gaussian_params import GaussianParams


@dataclass
class PipelineV2Config:
    """Configuration for TemporalPipelineV2."""
    
    # Regularization config
    reg_config: TemporalRegConfig = None
    
    # Smoothing parameters
    smoothing_method: str = "bilateral"  # "bilateral", "gaussian", "none"
    window_size: int = 5  # Temporal window for smoothing
    sigma_spatial: float = 0.1  # Spatial weight decay
    sigma_temporal: float = 1.0  # Temporal weight decay
    
    # Blending
    blend_alpha: float = 0.3  # 0 = original only, 1 = smoothed only
    
    def __post_init__(self):
        if self.reg_config is None:
            self.reg_config = TemporalRegConfig()


class TemporalPipelineV2(nn.Module):
    """
    Per-frame Gaussian pipeline with temporal regularization.
    
    Instead of autoregressive deformation, this pipeline:
    1. Keeps each frame's original Gaussians
    2. Applies temporal smoothing as post-processing
    3. Provides regularization losses for training
    """
    
    def __init__(self, config: Optional[PipelineV2Config] = None):
        super().__init__()
        
        self.config = config or PipelineV2Config()
        self.regularizer = TemporalRegularization(self.config.reg_config)
    
    def smooth_sequence(
        self,
        gaussians_sequence: List[GaussianParams],
        smooth_positions: bool = True,
        smooth_features: bool = False,
        smooth_opacity: bool = False,
    ) -> List[GaussianParams]:
        """
        Apply temporal smoothing to Gaussian sequence.
        
        Preserves per-frame structure while reducing temporal jitter.
        
        Args:
            gaussians_sequence: List of T GaussianParams
            smooth_positions: Smooth xyz positions
            smooth_features: Smooth appearance features
            smooth_opacity: Smooth opacity values
            
        Returns:
            List of T smoothed GaussianParams
        """
        cfg = self.config
        T = len(gaussians_sequence)
        
        if T < 2 or cfg.smoothing_method == "none":
            return [g.clone() for g in gaussians_sequence]
        
        smoothed = []
        half_window = cfg.window_size // 2
        
        for t in range(T):
            # Get window indices
            t_start = max(0, t - half_window)
            t_end = min(T, t + half_window + 1)
            
            # Clone current frame
            g_smoothed = gaussians_sequence[t].clone()
            
            if smooth_positions:
                g_smoothed.xyz = self._smooth_attribute(
                    [g.xyz for g in gaussians_sequence],
                    t, t_start, t_end,
                )
            
            if smooth_features:
                g_smoothed.features = self._smooth_attribute(
                    [g.features for g in gaussians_sequence],
                    t, t_start, t_end,
                )
            
            if smooth_opacity:
                g_smoothed.opacity = self._smooth_attribute(
                    [g.opacity for g in gaussians_sequence],
                    t, t_start, t_end,
                )
            
            smoothed.append(g_smoothed)
        
        return smoothed
    
    def _smooth_attribute(
        self,
        attr_sequence: List[torch.Tensor],
        t: int,
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        """
        Smooth a single attribute using bilateral filtering.
        
        Args:
            attr_sequence: List of [N, D] tensors
            t: Current frame index
            t_start, t_end: Window bounds
            
        Returns:
            Smoothed [N, D] tensor
        """
        cfg = self.config
        
        if cfg.smoothing_method == "gaussian":
            return self._gaussian_smooth(attr_sequence, t, t_start, t_end)
        elif cfg.smoothing_method == "bilateral":
            return self._bilateral_smooth(attr_sequence, t, t_start, t_end)
        else:
            return attr_sequence[t]
    
    def _gaussian_smooth(
        self,
        attr_sequence: List[torch.Tensor],
        t: int,
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        """Simple Gaussian temporal smoothing."""
        cfg = self.config
        
        weights = []
        values = []
        
        for ti in range(t_start, t_end):
            # Temporal weight
            dt = abs(ti - t)
            w = torch.exp(torch.tensor(-dt**2 / (2 * cfg.sigma_temporal**2)))
            weights.append(w)
            values.append(attr_sequence[ti])
        
        # Normalize weights
        weights = torch.stack(weights)
        weights = weights / weights.sum()
        
        # Weighted average
        result = sum(w * v for w, v in zip(weights, values))
        
        return result
    
    def _bilateral_smooth(
        self,
        attr_sequence: List[torch.Tensor],
        t: int,
        t_start: int,
        t_end: int,
    ) -> torch.Tensor:
        """
        Bilateral smoothing: considers both temporal distance and value similarity.
        
        Preserves edges/discontinuities while smoothing noise.
        """
        cfg = self.config
        current = attr_sequence[t]
        
        weighted_sum = torch.zeros_like(current)
        weight_sum = torch.zeros(current.shape[0], 1, device=current.device)
        
        for ti in range(t_start, t_end):
            # Temporal weight
            dt = abs(ti - t)
            w_temporal = torch.exp(torch.tensor(-dt**2 / (2 * cfg.sigma_temporal**2), device=current.device))
            
            # Spatial/value weight (per-Gaussian)
            diff = attr_sequence[ti] - current
            dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
            w_spatial = torch.exp(-dist_sq / (2 * cfg.sigma_spatial**2))
            
            # Combined weight
            w = w_temporal * w_spatial
            
            weighted_sum = weighted_sum + w * attr_sequence[ti]
            weight_sum = weight_sum + w
        
        # Normalize
        result = weighted_sum / (weight_sum + 1e-8)
        
        return result
    
    def blend_with_original(
        self,
        original: List[GaussianParams],
        smoothed: List[GaussianParams],
        alpha: Optional[float] = None,
    ) -> List[GaussianParams]:
        """
        Blend smoothed Gaussians with originals.
        
        result = (1 - alpha) * original + alpha * smoothed
        
        Args:
            original: Original per-frame Gaussians
            smoothed: Temporally smoothed Gaussians
            alpha: Blend factor (default from config)
            
        Returns:
            Blended Gaussians
        """
        alpha = alpha if alpha is not None else self.config.blend_alpha
        
        blended = []
        for orig, smooth in zip(original, smoothed):
            g = orig.clone()
            g.xyz = (1 - alpha) * orig.xyz + alpha * smooth.xyz
            # Keep other attributes from original (features, opacity, etc.)
            blended.append(g)
        
        return blended
    
    def compute_reg_loss(
        self,
        gaussians_sequence: List[GaussianParams],
    ) -> dict:
        """
        Compute regularization losses for training.
        
        Can be used to add temporal consistency to per-frame training.
        
        Args:
            gaussians_sequence: List of T GaussianParams
            
        Returns:
            dict with individual losses and total
        """
        xyz_sequence = [g.xyz for g in gaussians_sequence]
        return self.regularizer.compute_losses(xyz_sequence)
    
    def forward(
        self,
        gaussians_sequence: List[GaussianParams],
        return_losses: bool = False,
    ) -> Tuple[List[GaussianParams], dict]:
        """
        Full pipeline: smooth + optionally compute losses.
        
        Args:
            gaussians_sequence: List of T original GaussianParams
            return_losses: Whether to compute regularization losses
            
        Returns:
            smoothed: List of T smoothed GaussianParams
            losses: dict (empty if return_losses=False)
        """
        # Step 1: Smooth
        smoothed = self.smooth_sequence(gaussians_sequence)
        
        # Step 2: Optional blending
        if 0 < self.config.blend_alpha < 1:
            smoothed = self.blend_with_original(gaussians_sequence, smoothed)
        
        # Step 3: Optional losses (use smoothed for loss computation)
        losses = {}
        if return_losses:
            losses = self.compute_reg_loss(smoothed)
        
        return smoothed, losses


# ============================================================
# Comparison: V1 vs V2
# ============================================================

def compare_pipelines():
    """Compare V1 (autoregressive) vs V2 (per-frame + reg)."""
    from .temporal_pipeline import TemporalGaussianPipeline, TemporalConfig
    from .gaussian_params import GaussianParams
    
    print("=" * 60)
    print("Pipeline Comparison: V1 (Autoregressive) vs V2 (Per-frame)")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 500
    T = 10
    
    # Create sequence with slight per-frame noise
    xyz_base = torch.randn(N, 3, device=device)
    velocity = torch.randn(N, 3, device=device) * 0.05
    
    gaussians = []
    for t in range(T):
        xyz = xyz_base + velocity * t + torch.randn(N, 3, device=device) * 0.02
        g = GaussianParams(
            xyz=xyz,
            features=torch.randn(N, 27, device=device),
            scaling=torch.ones(N, 3, device=device) * 0.01,
            rotation=F.normalize(torch.randn(N, 4, device=device), dim=-1),
            opacity=torch.ones(N, 1, device=device) * 0.5,
        )
        gaussians.append(g)
    
    print(f"\nSequence: {T} frames, {N} Gaussians")
    
    # V1: Autoregressive
    print("\n--- V1: Autoregressive ---")
    v1 = TemporalGaussianPipeline(TemporalConfig())
    v1_out = v1(gaussians)
    
    # Check drift from original
    v1_drifts = []
    for t, (orig, out) in enumerate(zip(gaussians, v1_out)):
        drift = (orig.xyz - out.xyz).norm(dim=-1).mean().item()
        v1_drifts.append(drift)
        if t < 3 or t >= T - 2:
            print(f"  Frame {t}: drift = {drift:.4f}")
    print(f"  ..."  )
    print(f"  Avg drift: {sum(v1_drifts)/len(v1_drifts):.4f}")
    print(f"  Max drift: {max(v1_drifts):.4f} (frame {v1_drifts.index(max(v1_drifts))})")
    
    # V2: Per-frame + Reg
    print("\n--- V2: Per-frame + Regularization ---")
    v2 = TemporalPipelineV2(PipelineV2Config(blend_alpha=0.3))
    v2_out, losses = v2(gaussians, return_losses=True)
    
    # Check drift from original
    v2_drifts = []
    for t, (orig, out) in enumerate(zip(gaussians, v2_out)):
        drift = (orig.xyz - out.xyz).norm(dim=-1).mean().item()
        v2_drifts.append(drift)
        if t < 3 or t >= T - 2:
            print(f"  Frame {t}: drift = {drift:.4f}")
    print(f"  ..." )
    print(f"  Avg drift: {sum(v2_drifts)/len(v2_drifts):.4f}")
    print(f"  Max drift: {max(v2_drifts):.4f}")
    print(f"  Reg loss: {losses['total'].item():.6f}")
    
    # Summary
    print("\n--- Summary ---")
    print(f"V1 max drift: {max(v1_drifts):.4f} (cumulative error)")
    print(f"V2 max drift: {max(v2_drifts):.4f} (bounded by blend_alpha)")
    print(f"\nV2 preserves per-frame quality while adding temporal smoothness.")


if __name__ == "__main__":
    compare_pipelines()
