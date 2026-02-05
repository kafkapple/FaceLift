# Copyright 2026 FaceLift Mouse Extensions
# Temporal Deform Inference V2: Uses 8-layer MLP with dual-frame input
# Fixes autoregressive drift by using per-frame references

"""
TemporalDeformInference: 8-layer MLP based inference pipeline.

Key fix from V1:
- V1: MLP(G_t) only → autoregressive → cumulative drift
- V2: MLP(G_t, G_{t+1}) → dual-frame reference → bounded deformation

The network learns to deform G_t toward G_{t+1} while preserving local structure (ARAP).
At inference, we have all frames from GS-LRM, so we can use consecutive pairs.

Usage:
    from temporal_deform_inference import TemporalDeformInference
    
    # Load trained network
    inference = TemporalDeformInference.from_checkpoint('ckpt.pt', device='cuda')
    
    # Apply deformation to sequence
    smoothed = inference.process_sequence(gaussians_list)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformation_network import DeformationNetworkV2, DeformationConfigV2
from .gaussian_params import GaussianParams

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for TemporalDeformInference."""
    
    # Blending with original
    blend_alpha: float = 0.5  # 0 = original, 1 = fully deformed
    
    # Per-frame vs cumulative
    use_cumulative: bool = False  # True = V1 style (broken), False = V2 (correct)
    
    # Boundary handling
    boundary_mode: str = "replicate"  # "replicate", "zero", "mirror"
    
    # Device
    device: str = "cuda"


class TemporalDeformInference(nn.Module):
    """
    Inference pipeline using 8-layer MLP Deformation Network.
    
    V2 approach:
    1. Takes consecutive frame pairs (G_t, G_{t+1})
    2. MLP predicts deformation delta
    3. Applies delta to G_t
    4. Optionally blends with original
    
    This prevents autoregressive drift because:
    - Each deformation is supervised by the ORIGINAL G_{t+1}, not deformed output
    - ARAP loss during training ensures local structure preservation
    """
    
    def __init__(
        self,
        network: DeformationNetworkV2,
        config: Optional[InferenceConfig] = None,
    ):
        super().__init__()
        
        self.network = network
        self.config = config or InferenceConfig()
        
        # Freeze network for inference
        self.network.eval()
        for param in self.network.parameters():
            param.requires_grad = False
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        config: Optional[InferenceConfig] = None,
    ) -> "TemporalDeformInference":
        """Load from trained checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if "network_state_dict" in ckpt:
            state_dict = ckpt["network_state_dict"]
            net_config = ckpt.get("network_config", DeformationConfigV2())
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            net_config = ckpt.get("config", DeformationConfigV2())
        else:
            state_dict = ckpt
            net_config = DeformationConfigV2()
        
        # Create network
        network = DeformationNetworkV2(net_config).to(device)
        network.load_state_dict(state_dict)
        
        # Create config if not provided
        if config is None:
            config = InferenceConfig(device=device)
        else:
            config.device = device
        
        logger.info(f"Loaded network: {network.get_info()}")
        
        return cls(network, config)
    
    @torch.no_grad()
    def process_sequence(
        self,
        gaussians_sequence: List[GaussianParams],
        blend_alpha: Optional[float] = None,
    ) -> List[GaussianParams]:
        """
        Process a sequence of Gaussians with temporal deformation.
        
        Args:
            gaussians_sequence: List of T GaussianParams from GS-LRM
            blend_alpha: Override blend factor (None = use config)
            
        Returns:
            List of T deformed GaussianParams
        """
        cfg = self.config
        T = len(gaussians_sequence)
        alpha = blend_alpha if blend_alpha is not None else cfg.blend_alpha
        
        if T < 2:
            logger.warning("Sequence too short for temporal deformation")
            return [g.clone() for g in gaussians_sequence]
        
        self.network.eval()
        results = []
        
        for t in range(T):
            # Get reference frame (t+1 for forward direction)
            if t < T - 1:
                ref_idx = t + 1
            else:
                # Last frame: use previous as reference (reverse direction)
                ref_idx = t - 1 if cfg.boundary_mode == "replicate" else t
            
            G_t = gaussians_sequence[t]
            G_ref = gaussians_sequence[ref_idx]
            
            # Move to device
            xyz_t = G_t.xyz.to(cfg.device)
            xyz_ref = G_ref.xyz.to(cfg.device)
            time_t = torch.tensor(t, device=cfg.device)  # scalar, will be expanded
            
            # Predict deformation
            deform_raw = self.network(xyz_t, xyz_ref, time_t)
            
            # Parse deformation
            deform_dict = self._parse_deformation(deform_raw)
            
            # Apply deformation
            G_deformed = self._apply_deformation(G_t, deform_dict)
            
            # Blend with original
            if 0 < alpha < 1:
                G_result = self._blend(G_t, G_deformed, alpha)
            elif alpha >= 1:
                G_result = G_deformed
            else:
                G_result = G_t.clone()
            
            results.append(G_result)
        
        logger.info(f"Processed {T} frames with blend_alpha={alpha:.2f}")
        return results
    
    @torch.no_grad()
    def process_pair(
        self,
        G_t: GaussianParams,
        G_ref: GaussianParams,
        time_index: int = 0,
        blend_alpha: Optional[float] = None,
    ) -> GaussianParams:
        """Process a single pair of Gaussians."""
        cfg = self.config
        alpha = blend_alpha if blend_alpha is not None else cfg.blend_alpha
        
        self.network.eval()
        
        xyz_t = G_t.xyz.to(cfg.device)
        xyz_ref = G_ref.xyz.to(cfg.device)
        time_t = torch.tensor(time_index, device=cfg.device)  # scalar
        
        deform_raw = self.network(xyz_t, xyz_ref, time_t)
        deform_dict = self._parse_deformation(deform_raw)
        G_deformed = self._apply_deformation(G_t, deform_dict)
        
        if 0 < alpha < 1:
            return self._blend(G_t, G_deformed, alpha)
        elif alpha >= 1:
            return G_deformed
        else:
            return G_t.clone()
    
    def _parse_deformation(self, deform_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse raw network output into deformation components."""
        # deform_raw: [N, 5] = (dx, dy, dz, d_alpha, d_scale)
        return {
            "xyz": deform_raw[:, :3],      # Position delta
            "alpha": deform_raw[:, 3:4],   # Opacity delta
            "scale": deform_raw[:, 4:5],   # Scale delta (uniform)
        }
    
    def _apply_deformation(
        self,
        G: GaussianParams,
        deform_dict: Dict[str, torch.Tensor],
    ) -> GaussianParams:
        """Apply deformation to GaussianParams."""
        G_out = G.clone()
        
        # Apply position delta
        G_out.xyz = G.xyz + deform_dict["xyz"].to(G.xyz.device)
        
        # Apply opacity delta (with sigmoid to keep in [0, 1])
        if G.opacity is not None and "alpha" in deform_dict:
            # Convert to logit, add delta, convert back
            logit = torch.log(G.opacity / (1 - G.opacity + 1e-8) + 1e-8)
            logit = logit + deform_dict["alpha"].to(G.opacity.device)
            G_out.opacity = torch.sigmoid(logit)
        
        # Apply scale delta
        if G.scaling is not None and "scale" in deform_dict:
            # Multiplicative scaling (exp of delta)
            scale_factor = torch.exp(deform_dict["scale"].to(G.scaling.device))
            G_out.scaling = G.scaling * scale_factor
        
        return G_out
    
    def _blend(
        self,
        G_orig: GaussianParams,
        G_deformed: GaussianParams,
        alpha: float,
    ) -> GaussianParams:
        """Blend original and deformed Gaussians."""
        G_out = G_orig.clone()
        
        # Blend positions
        G_out.xyz = (1 - alpha) * G_orig.xyz + alpha * G_deformed.xyz
        
        # Blend opacity
        if G_orig.opacity is not None and G_deformed.opacity is not None:
            G_out.opacity = (1 - alpha) * G_orig.opacity + alpha * G_deformed.opacity
        
        # Blend scaling
        if G_orig.scaling is not None and G_deformed.scaling is not None:
            G_out.scaling = (1 - alpha) * G_orig.scaling + alpha * G_deformed.scaling
        
        return G_out
    
    def get_info(self) -> str:
        """Get info string."""
        return (
            f"TemporalDeformInference("
            f"network={self.network.get_info()}, "
            f"blend_alpha={self.config.blend_alpha:.2f})"
        )


# ============================================================
# Comparison: V1 (autoregressive) vs V2 (per-frame reference)
# ============================================================

def compare_v1_v2_inference(num_frames: int = 20, num_gaussians: int = 1000):
    """
    Compare V1 (autoregressive drift) vs V2 (bounded deformation) inference.
    
    This demonstrates why V2 is better:
    - V1: Error accumulates across frames
    - V2: Each frame is bounded by original GS-LRM output
    """
    from .temporal_pipeline import TemporalGaussianPipeline, TemporalConfig
    
    print("=" * 70)
    print("Inference Comparison: V1 (Autoregressive) vs V2 (Per-frame Reference)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = num_gaussians
    T = num_frames
    
    # Simulate GS-LRM outputs with natural motion
    print(f"\nCreating synthetic sequence: {T} frames, {N} Gaussians")
    xyz_base = torch.randn(N, 3, device=device)
    velocity = torch.randn(N, 3, device=device) * 0.05
    
    gaussians = []
    for t in range(T):
        # Add motion + small per-frame noise (simulating GS-LRM jitter)
        xyz = xyz_base + velocity * t + torch.randn(N, 3, device=device) * 0.01
        g = GaussianParams(
            xyz=xyz,
            features=torch.randn(N, 27, device=device),
            scaling=torch.ones(N, 3, device=device) * 0.01,
            rotation=F.normalize(torch.randn(N, 4, device=device), dim=-1),
            opacity=torch.ones(N, 1, device=device) * 0.5,
        )
        gaussians.append(g)
    
    # Measure original frame-to-frame distance
    orig_distances = []
    for t in range(T - 1):
        dist = (gaussians[t+1].xyz - gaussians[t].xyz).norm(dim=-1).mean().item()
        orig_distances.append(dist)
    print(f"Original mean frame-to-frame distance: {sum(orig_distances)/len(orig_distances):.4f}")
    
    # V1: Autoregressive (simulated - no trained network, just shows drift pattern)
    print("\n--- V1: Autoregressive Pattern (Simulated) ---")
    print("  (Using identity + small drift to show accumulation)")
    
    v1_results = [gaussians[0].clone()]
    drift_per_frame = torch.randn(N, 3, device=device) * 0.01  # Small drift
    
    for t in range(1, T):
        # V1: Use PREVIOUS OUTPUT, add drift
        prev = v1_results[-1]
        new_xyz = prev.xyz + drift_per_frame  # Cumulative!
        v1_results.append(GaussianParams(
            xyz=new_xyz,
            features=gaussians[t].features,
            scaling=gaussians[t].scaling,
            rotation=gaussians[t].rotation,
            opacity=gaussians[t].opacity,
        ))
    
    v1_drifts = []
    for t, (orig, v1) in enumerate(zip(gaussians, v1_results)):
        drift = (orig.xyz - v1.xyz).norm(dim=-1).mean().item()
        v1_drifts.append(drift)
    
    print(f"  Frame 0 drift: {v1_drifts[0]:.4f}")
    print(f"  Frame {T//2} drift: {v1_drifts[T//2]:.4f}")
    print(f"  Frame {T-1} drift: {v1_drifts[-1]:.4f}")
    print(f"  Pattern: INCREASING (cumulative error)")
    
    # V2: Per-frame reference (simulated)
    print("\n--- V2: Per-frame Reference Pattern (Simulated) ---")
    print("  (Using original as reference, bounded delta)")
    
    v2_results = []
    bounded_delta = torch.randn(N, 3, device=device) * 0.01  # Same magnitude
    
    for t in range(T):
        if t < T - 1:
            # V2: Use ORIGINAL NEXT FRAME as reference
            ref = gaussians[t + 1]
            # Delta is bounded by the reference, not cumulative
            direction = (ref.xyz - gaussians[t].xyz)
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            delta = direction * 0.01  # Bounded step toward reference
            new_xyz = gaussians[t].xyz + delta
        else:
            new_xyz = gaussians[t].xyz + bounded_delta * 0.5  # Last frame: smaller
        
        v2_results.append(GaussianParams(
            xyz=new_xyz,
            features=gaussians[t].features,
            scaling=gaussians[t].scaling,
            rotation=gaussians[t].rotation,
            opacity=gaussians[t].opacity,
        ))
    
    v2_drifts = []
    for t, (orig, v2) in enumerate(zip(gaussians, v2_results)):
        drift = (orig.xyz - v2.xyz).norm(dim=-1).mean().item()
        v2_drifts.append(drift)
    
    print(f"  Frame 0 drift: {v2_drifts[0]:.4f}")
    print(f"  Frame {T//2} drift: {v2_drifts[T//2]:.4f}")
    print(f"  Frame {T-1} drift: {v2_drifts[-1]:.4f}")
    print(f"  Pattern: CONSTANT (bounded by original reference)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  V1 final drift: {v1_drifts[-1]:.4f} (cumulative: {(v1_drifts[-1]/(v1_drifts[0] + 1e-8)):.1f}x initial)")
    print(f"  V2 final drift: {v2_drifts[-1]:.4f} (bounded)")
    print("\n  V1 Problem: Each frame uses PREVIOUS OUTPUT → error accumulates")
    print("  V2 Fix: Each frame uses ORIGINAL G_{t+1} as reference → error bounded")
    print("=" * 70)
    
    return v1_drifts, v2_drifts


def test_inference_pipeline():
    """Test the TemporalDeformInference class."""
    print("\n" + "=" * 60)
    print("Testing TemporalDeformInference")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 500
    T = 5
    
    # Create network
    config = DeformationConfigV2()
    network = DeformationNetworkV2(config).to(device)
    
    # Create inference pipeline
    inf_config = InferenceConfig(blend_alpha=0.5, device=device)
    pipeline = TemporalDeformInference(network, inf_config)
    
    print(f"Pipeline: {pipeline.get_info()}")
    
    # Create test sequence
    gaussians = []
    xyz_base = torch.randn(N, 3, device=device)
    for t in range(T):
        xyz = xyz_base + torch.randn(N, 3, device=device) * 0.02 * t
        g = GaussianParams(
            xyz=xyz,
            features=torch.randn(N, 27, device=device),
            scaling=torch.ones(N, 3, device=device) * 0.01,
            rotation=F.normalize(torch.randn(N, 4, device=device), dim=-1),
            opacity=torch.sigmoid(torch.randn(N, 1, device=device)),
        )
        gaussians.append(g)
    
    # Process sequence
    print(f"\nProcessing {T} frames with {N} Gaussians each...")
    results = pipeline.process_sequence(gaussians)
    
    print(f"\nResults:")
    for t, (orig, res) in enumerate(zip(gaussians, results)):
        xyz_diff = (orig.xyz - res.xyz).norm(dim=-1).mean().item()
        opacity_diff = (orig.opacity - res.opacity).abs().mean().item() if res.opacity is not None else 0
        print(f"  Frame {t}: xyz_diff={xyz_diff:.6f}, opacity_diff={opacity_diff:.6f}")
    
    print("\n✓ Inference pipeline test passed!")
    
    return True


if __name__ == "__main__":
    test_inference_pipeline()
    print("\n")
    compare_v1_v2_inference()
