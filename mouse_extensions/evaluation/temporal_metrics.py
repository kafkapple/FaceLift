"""Temporal consistency metrics for deformation evaluation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class TemporalMetricsResult:
    """Results from temporal metrics computation."""
    
    # Displacement statistics (deformation magnitude)
    mean_displacement: float      # Mean xyz displacement per Gaussian
    std_displacement: float       # Std of displacement
    max_displacement: float       # Max displacement
    
    # Temporal consistency (frame-to-frame smoothness)
    temporal_jitter_before: float  # Mean frame-to-frame xyz change (before)
    temporal_jitter_after: float   # Mean frame-to-frame xyz change (after)
    jitter_reduction: float        # Percentage reduction in jitter
    
    # Per-frame statistics
    per_frame_displacement: List[float]  # Displacement per frame
    per_frame_jitter_before: List[float]  # Jitter before per frame
    per_frame_jitter_after: List[float]   # Jitter after per frame
    
    # Opacity/scale changes
    mean_opacity_change: float
    mean_scale_change: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'displacement/mean': self.mean_displacement,
            'displacement/std': self.std_displacement,
            'displacement/max': self.max_displacement,
            'temporal/jitter_before': self.temporal_jitter_before,
            'temporal/jitter_after': self.temporal_jitter_after,
            'temporal/jitter_reduction_pct': self.jitter_reduction,
            'deformation/opacity_change': self.mean_opacity_change,
            'deformation/scale_change': self.mean_scale_change,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return f"""
=== Temporal Metrics Summary ===

Displacement (Deformation Magnitude):
  Mean: {self.mean_displacement:.6f}
  Std:  {self.std_displacement:.6f}
  Max:  {self.max_displacement:.6f}

Temporal Consistency (Jitter):
  Before: {self.temporal_jitter_before:.6f}
  After:  {self.temporal_jitter_after:.6f}
  Reduction: {self.jitter_reduction:.1f}%

Other Changes:
  Opacity: {self.mean_opacity_change:.6f}
  Scale:   {self.mean_scale_change:.6f}
"""


class TemporalMetrics:
    """Compute temporal consistency metrics."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def compute(
        self,
        original_gaussians: List[Dict],
        smoothed_gaussians: List[Dict],
    ) -> TemporalMetricsResult:
        """Compute all temporal metrics.
        
        Args:
            original_gaussians: List of dicts with xyz, opacity, scaling, etc.
            smoothed_gaussians: List of dicts with xyz, opacity, scaling, etc.
        
        Returns:
            TemporalMetricsResult with all computed metrics.
        """
        n_frames = len(original_gaussians)
        assert n_frames == len(smoothed_gaussians), "Frame count mismatch"
        
        # Per-frame displacement
        per_frame_displacement = []
        opacity_changes = []
        scale_changes = []
        
        for orig, smooth in zip(original_gaussians, smoothed_gaussians):
            orig_xyz = orig['xyz']
            smooth_xyz = smooth['xyz']
            
            # XYZ displacement
            disp = (smooth_xyz - orig_xyz).norm(dim=-1)
            per_frame_displacement.append(disp.mean().item())
            
            # Opacity change
            if 'opacity' in orig and 'opacity' in smooth:
                op_diff = (smooth['opacity'] - orig['opacity']).abs().mean()
                opacity_changes.append(op_diff.item())
            
            # Scale change (use first dimension if 3D)
            if 'scaling' in orig and 'scaling' in smooth:
                sc_diff = (smooth['scaling'] - orig['scaling']).abs().mean()
                scale_changes.append(sc_diff.item())
        
        # Temporal jitter (frame-to-frame xyz change)
        jitter_before = []
        jitter_after = []
        
        for i in range(n_frames - 1):
            # Before deformation
            orig_diff = (original_gaussians[i+1]['xyz'] - original_gaussians[i]['xyz']).norm(dim=-1)
            jitter_before.append(orig_diff.mean().item())
            
            # After deformation
            smooth_diff = (smoothed_gaussians[i+1]['xyz'] - smoothed_gaussians[i]['xyz']).norm(dim=-1)
            jitter_after.append(smooth_diff.mean().item())
        
        # Aggregate statistics
        mean_disp = np.mean(per_frame_displacement)
        std_disp = np.std(per_frame_displacement)
        max_disp = np.max(per_frame_displacement)
        
        mean_jitter_before = np.mean(jitter_before) if jitter_before else 0.0
        mean_jitter_after = np.mean(jitter_after) if jitter_after else 0.0
        
        # Jitter reduction percentage
        if mean_jitter_before > 0:
            jitter_reduction = (1 - mean_jitter_after / mean_jitter_before) * 100
        else:
            jitter_reduction = 0.0
        
        return TemporalMetricsResult(
            mean_displacement=mean_disp,
            std_displacement=std_disp,
            max_displacement=max_disp,
            temporal_jitter_before=mean_jitter_before,
            temporal_jitter_after=mean_jitter_after,
            jitter_reduction=jitter_reduction,
            per_frame_displacement=per_frame_displacement,
            per_frame_jitter_before=jitter_before,
            per_frame_jitter_after=jitter_after,
            mean_opacity_change=np.mean(opacity_changes) if opacity_changes else 0.0,
            mean_scale_change=np.mean(scale_changes) if scale_changes else 0.0,
        )
    
    def compute_from_files(
        self,
        original_dir: str,
        smoothed_dir: str,
        max_frames: Optional[int] = None,
    ) -> TemporalMetricsResult:
        """Compute metrics from saved .pt files."""
        from pathlib import Path
        
        orig_dir = Path(original_dir)
        smooth_dir = Path(smoothed_dir)
        
        orig_files = sorted(orig_dir.glob('*.pt'))
        smooth_files = sorted(smooth_dir.glob('*.pt'))
        
        if max_frames:
            orig_files = orig_files[:max_frames]
            smooth_files = smooth_files[:max_frames]
        
        print(f"Loading {len(orig_files)} original and {len(smooth_files)} smoothed files...")
        
        original_gaussians = [torch.load(f, weights_only=True) for f in orig_files]
        smoothed_gaussians = [torch.load(f, weights_only=True) for f in smooth_files]
        
        return self.compute(original_gaussians, smoothed_gaussians)
