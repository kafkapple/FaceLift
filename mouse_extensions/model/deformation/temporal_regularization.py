# Copyright 2026 FaceLift Mouse Extensions
# Temporal Regularization Losses for 4D Gaussian Consistency
# Based on SC-GS, Dynamic 3DGS, MotionGS papers

"""
Temporal Regularization Losses:
1. ARAP (As-Rigid-As-Possible): Local rigidity constraint
2. Velocity Smoothness: Temporal velocity continuity
3. Isometry: Distance preservation between Gaussians
4. Position Smoothness: Direct position regularization
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemporalRegConfig:
    """Configuration for temporal regularization."""
    
    # ARAP loss
    arap_weight: float = 0.1
    arap_k_neighbors: int = 8  # KNN for local neighborhood
    
    # Velocity smoothness
    velocity_weight: float = 0.01
    
    # Isometry (distance preservation)
    isometry_weight: float = 0.0  # Off by default (expensive)
    isometry_k_neighbors: int = 4
    
    # Position smoothness (simple L2)
    position_smooth_weight: float = 0.0
    
    # Acceleration consistency
    acceleration_weight: float = 0.0


def knn_points(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find K nearest neighbors.
    
    Args:
        query: [N, 3] query points
        reference: [M, 3] reference points
        k: number of neighbors
        
    Returns:
        distances: [N, K] squared distances
        indices: [N, K] neighbor indices
    """
    # Compute pairwise distances
    # query: [N, 1, 3], reference: [1, M, 3]
    diff = query.unsqueeze(1) - reference.unsqueeze(0)  # [N, M, 3]
    dist_sq = (diff ** 2).sum(dim=-1)  # [N, M]
    
    # Get top-k nearest
    distances, indices = torch.topk(dist_sq, k, dim=1, largest=False)
    
    return distances, indices


def compute_local_rotation_svd(
    source_neighbors: torch.Tensor,
    target_neighbors: torch.Tensor,
) -> torch.Tensor:
    """
    Compute optimal local rotation using SVD.
    
    Args:
        source_neighbors: [N, K, 3] neighbor positions at time t
        target_neighbors: [N, K, 3] neighbor positions at time t+1
        
    Returns:
        rotations: [N, 3, 3] optimal rotation matrices
    """
    # Center the neighborhoods
    source_centered = source_neighbors - source_neighbors.mean(dim=1, keepdim=True)
    target_centered = target_neighbors - target_neighbors.mean(dim=1, keepdim=True)
    
    # Compute covariance: H = source^T @ target
    H = torch.bmm(source_centered.transpose(1, 2), target_centered)  # [N, 3, 3]
    
    # SVD
    U, S, Vh = torch.linalg.svd(H)
    
    # R = V @ U^T
    R = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))
    
    # Handle reflection (det(R) = -1)
    det = torch.linalg.det(R)
    mask = det < 0
    if mask.any():
        Vh_fixed = Vh.clone()
        Vh_fixed[mask, -1, :] *= -1
        R[mask] = torch.bmm(
            Vh_fixed[mask].transpose(1, 2), 
            U[mask].transpose(1, 2)
        )
    
    return R


class TemporalRegularization(nn.Module):
    """
    Temporal regularization losses for Gaussian sequences.
    
    Encourages temporal consistency while preserving per-frame quality.
    """
    
    def __init__(self, config: Optional[TemporalRegConfig] = None):
        super().__init__()
        self.config = config or TemporalRegConfig()
        
        # Cache for KNN indices (reuse across frames)
        self._knn_cache = {}
    
    def arap_loss(
        self,
        xyz_t: torch.Tensor,
        xyz_t1: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        As-Rigid-As-Possible loss between consecutive frames.
        
        Encourages local neighborhoods to move rigidly.
        
        L_ARAP = Σᵢ Σⱼ∈N(i) ||p'ⱼ - p'ᵢ - R̂ᵢ(pⱼ - pᵢ)||²
        
        Args:
            xyz_t: [N, 3] positions at time t
            xyz_t1: [N, 3] positions at time t+1
            k: number of neighbors (default from config)
            
        Returns:
            loss: scalar ARAP loss
        """
        k = k or self.config.arap_k_neighbors
        N = xyz_t.shape[0]
        
        if N < k + 1:
            return torch.tensor(0.0, device=xyz_t.device)
        
        # Find KNN at time t
        _, indices = knn_points(xyz_t, xyz_t, k + 1)  # +1 to exclude self
        indices = indices[:, 1:]  # Remove self [N, K]
        
        # Gather neighbor positions
        # xyz_t: [N, 3], indices: [N, K]
        neighbors_t = xyz_t[indices]  # [N, K, 3]
        neighbors_t1 = xyz_t1[indices]  # [N, K, 3]
        
        # Compute relative positions
        rel_t = neighbors_t - xyz_t.unsqueeze(1)  # [N, K, 3]
        rel_t1 = neighbors_t1 - xyz_t1.unsqueeze(1)  # [N, K, 3]
        
        # Compute optimal local rotation via SVD
        R = compute_local_rotation_svd(rel_t, rel_t1)  # [N, 3, 3]
        
        # Apply rotation to source relatives
        rel_t_rotated = torch.bmm(rel_t, R.transpose(1, 2))  # [N, K, 3]
        
        # ARAP loss: ||rel_t1 - R @ rel_t||²
        loss = ((rel_t1 - rel_t_rotated) ** 2).sum(dim=-1).mean()
        
        return loss
    
    def velocity_smoothness_loss(
        self,
        xyz_t0: torch.Tensor,
        xyz_t1: torch.Tensor,
        xyz_t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Velocity smoothness loss for 3 consecutive frames.
        
        Encourages constant velocity (smooth motion).
        
        L_vel = ||v₁ - v₀||² where v = x_{t+1} - x_t
        
        Args:
            xyz_t0: [N, 3] positions at time t-1
            xyz_t1: [N, 3] positions at time t
            xyz_t2: [N, 3] positions at time t+1
            
        Returns:
            loss: scalar velocity smoothness loss
        """
        v0 = xyz_t1 - xyz_t0  # velocity at t-1 to t
        v1 = xyz_t2 - xyz_t1  # velocity at t to t+1
        
        loss = ((v1 - v0) ** 2).sum(dim=-1).mean()
        
        return loss
    
    def isometry_loss(
        self,
        xyz_t: torch.Tensor,
        xyz_t1: torch.Tensor,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Isometry (distance preservation) loss.
        
        Encourages pairwise distances to remain constant.
        
        L_iso = Σᵢⱼ ||d(pᵢ,pⱼ)_t - d(pᵢ,pⱼ)_{t+1}||²
        
        Args:
            xyz_t: [N, 3] positions at time t
            xyz_t1: [N, 3] positions at time t+1
            k: number of neighbors to consider
            
        Returns:
            loss: scalar isometry loss
        """
        k = k or self.config.isometry_k_neighbors
        N = xyz_t.shape[0]
        
        if N < k + 1:
            return torch.tensor(0.0, device=xyz_t.device)
        
        # Find KNN
        _, indices = knn_points(xyz_t, xyz_t, k + 1)
        indices = indices[:, 1:]
        
        # Compute distances at t
        neighbors_t = xyz_t[indices]  # [N, K, 3]
        dist_t = torch.norm(neighbors_t - xyz_t.unsqueeze(1), dim=-1)  # [N, K]
        
        # Compute distances at t+1
        neighbors_t1 = xyz_t1[indices]
        dist_t1 = torch.norm(neighbors_t1 - xyz_t1.unsqueeze(1), dim=-1)
        
        # Isometry loss
        loss = ((dist_t - dist_t1) ** 2).mean()
        
        return loss
    
    def position_smoothness_loss(
        self,
        xyz_t: torch.Tensor,
        xyz_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simple position smoothness (L2 displacement).
        
        Args:
            xyz_t: [N, 3] positions at time t
            xyz_t1: [N, 3] positions at time t+1
            
        Returns:
            loss: scalar L2 displacement loss
        """
        return ((xyz_t1 - xyz_t) ** 2).sum(dim=-1).mean()
    
    def acceleration_loss(
        self,
        xyz_t0: torch.Tensor,
        xyz_t1: torch.Tensor,
        xyz_t2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Acceleration consistency loss.
        
        Encourages smooth acceleration (constant jerk).
        
        L_acc = ||a₁ - a₀||² where a = v_{t+1} - v_t
        
        Args:
            xyz_t0, xyz_t1, xyz_t2: positions at 3 consecutive times
            
        Returns:
            loss: scalar acceleration loss
        """
        v0 = xyz_t1 - xyz_t0
        v1 = xyz_t2 - xyz_t1
        
        a0 = v1 - v0  # acceleration at t
        
        # For 3 frames, we can only compute one acceleration
        # Return L2 norm (encourages small acceleration)
        return (a0 ** 2).sum(dim=-1).mean()
    
    def compute_losses(
        self,
        xyz_sequence: List[torch.Tensor],
    ) -> dict:
        """
        Compute all temporal regularization losses for a sequence.
        
        Args:
            xyz_sequence: List of [N, 3] position tensors
            
        Returns:
            dict with individual losses and total
        """
        cfg = self.config
        T = len(xyz_sequence)
        device = xyz_sequence[0].device
        
        losses = {
            'arap': torch.tensor(0.0, device=device),
            'velocity': torch.tensor(0.0, device=device),
            'isometry': torch.tensor(0.0, device=device),
            'position_smooth': torch.tensor(0.0, device=device),
            'acceleration': torch.tensor(0.0, device=device),
        }
        
        # ARAP loss (pairs)
        if cfg.arap_weight > 0 and T >= 2:
            arap_sum = 0.0
            for t in range(T - 1):
                arap_sum = arap_sum + self.arap_loss(xyz_sequence[t], xyz_sequence[t + 1])
            losses['arap'] = arap_sum / (T - 1)
        
        # Velocity smoothness (triplets)
        if cfg.velocity_weight > 0 and T >= 3:
            vel_sum = 0.0
            for t in range(T - 2):
                vel_sum = vel_sum + self.velocity_smoothness_loss(
                    xyz_sequence[t], xyz_sequence[t + 1], xyz_sequence[t + 2]
                )
            losses['velocity'] = vel_sum / (T - 2)
        
        # Isometry loss (pairs)
        if cfg.isometry_weight > 0 and T >= 2:
            iso_sum = 0.0
            for t in range(T - 1):
                iso_sum = iso_sum + self.isometry_loss(xyz_sequence[t], xyz_sequence[t + 1])
            losses['isometry'] = iso_sum / (T - 1)
        
        # Position smoothness (pairs)
        if cfg.position_smooth_weight > 0 and T >= 2:
            pos_sum = 0.0
            for t in range(T - 1):
                pos_sum = pos_sum + self.position_smoothness_loss(
                    xyz_sequence[t], xyz_sequence[t + 1]
                )
            losses['position_smooth'] = pos_sum / (T - 1)
        
        # Acceleration loss (triplets)
        if cfg.acceleration_weight > 0 and T >= 3:
            acc_sum = 0.0
            for t in range(T - 2):
                acc_sum = acc_sum + self.acceleration_loss(
                    xyz_sequence[t], xyz_sequence[t + 1], xyz_sequence[t + 2]
                )
            losses['acceleration'] = acc_sum / (T - 2)
        
        # Weighted total
        total = (
            cfg.arap_weight * losses['arap'] +
            cfg.velocity_weight * losses['velocity'] +
            cfg.isometry_weight * losses['isometry'] +
            cfg.position_smooth_weight * losses['position_smooth'] +
            cfg.acceleration_weight * losses['acceleration']
        )
        losses['total'] = total
        
        return losses


# ============================================================
# Unit Tests
# ============================================================

def _test_temporal_regularization():
    """Unit test for TemporalRegularization."""
    print("Testing TemporalRegularization...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = 1000  # Number of Gaussians
    T = 5     # Number of frames
    
    # Create synthetic sequence with smooth motion
    xyz_base = torch.randn(N, 3, device=device)
    velocity = torch.randn(N, 3, device=device) * 0.01
    
    xyz_sequence = []
    for t in range(T):
        xyz_t = xyz_base + velocity * t + torch.randn(N, 3, device=device) * 0.001
        xyz_sequence.append(xyz_t)
    
    print(f"  Created sequence: {T} frames, {N} Gaussians")
    
    # Test 1: Create regularizer
    config = TemporalRegConfig(
        arap_weight=0.1,
        velocity_weight=0.01,
        isometry_weight=0.001,
    )
    reg = TemporalRegularization(config)
    print(f"  Config: ARAP={config.arap_weight}, vel={config.velocity_weight}")
    
    # Test 2: ARAP loss
    arap = reg.arap_loss(xyz_sequence[0], xyz_sequence[1])
    print(f"  ARAP loss: {arap.item():.6f}")
    
    # Test 3: Velocity smoothness
    vel = reg.velocity_smoothness_loss(xyz_sequence[0], xyz_sequence[1], xyz_sequence[2])
    print(f"  Velocity loss: {vel.item():.6f}")
    
    # Test 4: Isometry
    iso = reg.isometry_loss(xyz_sequence[0], xyz_sequence[1])
    print(f"  Isometry loss: {iso.item():.6f}")
    
    # Test 5: Full sequence
    losses = reg.compute_losses(xyz_sequence)
    print(f"  Total loss: {losses['total'].item():.6f}")
    for k, v in losses.items():
        if k != 'total':
            print(f"    {k}: {v.item():.6f}")
    
    # Test 6: Gradient flow
    xyz_sequence[0].requires_grad_(True)
    losses = reg.compute_losses(xyz_sequence)
    losses['total'].backward()
    assert xyz_sequence[0].grad is not None, "Gradients should flow"
    print("  Gradient flow: OK")
    
    # Test 7: Compare smooth vs noisy sequence
    xyz_noisy = [xyz + torch.randn_like(xyz) * 0.1 for xyz in xyz_sequence]
    losses_noisy = reg.compute_losses(xyz_noisy)
    print(f"  Smooth total: {losses['total'].item():.6f}")
    print(f"  Noisy total: {losses_noisy['total'].item():.6f}")
    assert losses_noisy['total'] > losses['total'], "Noisy should have higher loss"
    print("  Noise sensitivity: OK")
    
    print("All tests passed!")
    return True


if __name__ == "__main__":
    _test_temporal_regularization()
