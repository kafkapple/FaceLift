"""Efficient Temporal Regularization with sampled KNN."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TemporalRegConfig:
    arap_weight: float = 0.1
    velocity_weight: float = 0.01
    isometry_weight: float = 0.0
    arap_k_neighbors: int = 8
    max_points_for_knn: int = 4096  # Sample this many points for KNN


def sampled_knn(
    query: torch.Tensor,
    k: int,
    max_points: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KNN with random sampling for large point clouds.
    
    Args:
        query: [N, 3] points
        k: number of neighbors
        max_points: max points to use (sample if N > max_points)
        
    Returns:
        sampled_query: [M, 3] sampled points
        indices: [M, K] neighbor indices in sampled space
        sample_idx: [M] original indices of sampled points
    """
    N = query.shape[0]
    
    if N <= max_points:
        # Use all points
        sampled = query
        sample_idx = torch.arange(N, device=query.device)
    else:
        # Random sample
        sample_idx = torch.randperm(N, device=query.device)[:max_points]
        sampled = query[sample_idx]
    
    M = sampled.shape[0]
    
    # Compute pairwise distances for sampled points only
    # [M, M]
    diff = sampled.unsqueeze(1) - sampled.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1)
    
    # Get top-k nearest (excluding self)
    _, indices = torch.topk(dist_sq, k + 1, dim=1, largest=False)
    indices = indices[:, 1:]  # Exclude self
    
    return sampled, indices, sample_idx


class TemporalRegularizationEfficient(nn.Module):
    """Efficient temporal regularization for large point clouds."""
    
    def __init__(self, config: TemporalRegConfig):
        super().__init__()
        self.config = config
        
    def arap_loss(
        self,
        xyz_t: torch.Tensor,
        xyz_t1: torch.Tensor,
    ) -> torch.Tensor:
        """ARAP loss with sampling for efficiency."""
        k = self.config.arap_k_neighbors
        max_pts = self.config.max_points_for_knn
        
        # Sample and compute KNN
        sampled_t, knn_indices, sample_idx = sampled_knn(xyz_t, k, max_pts)
        sampled_t1 = xyz_t1[sample_idx]
        
        M = sampled_t.shape[0]
        
        # Get neighbor positions
        neighbors_t = sampled_t[knn_indices]   # [M, K, 3]
        neighbors_t1 = sampled_t1[knn_indices] # [M, K, 3]
        
        # Compute edge vectors
        edges_t = neighbors_t - sampled_t.unsqueeze(1)   # [M, K, 3]
        edges_t1 = neighbors_t1 - sampled_t1.unsqueeze(1) # [M, K, 3]
        
        # ARAP: edges should have same length (simplified without rotation)
        edge_len_t = edges_t.norm(dim=-1)   # [M, K]
        edge_len_t1 = edges_t1.norm(dim=-1) # [M, K]
        
        # Loss: difference in edge lengths
        loss = ((edge_len_t - edge_len_t1) ** 2).mean()
        
        return loss
    
    def velocity_smoothness_loss(
        self,
        xyz_t0: torch.Tensor,
        xyz_t1: torch.Tensor,
        xyz_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity smoothness: v_{t+1} should be similar to v_t."""
        # Sample same indices for all frames
        N = xyz_t0.shape[0]
        max_pts = self.config.max_points_for_knn
        
        if N > max_pts:
            idx = torch.randperm(N, device=xyz_t0.device)[:max_pts]
            xyz_t0 = xyz_t0[idx]
            xyz_t1 = xyz_t1[idx]
            xyz_t2 = xyz_t2[idx]
        
        v0 = xyz_t1 - xyz_t0
        v1 = xyz_t2 - xyz_t1
        
        return ((v1 - v0) ** 2).mean()
    
    def compute_losses(
        self,
        xyz_sequence: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute all temporal losses.
        
        NOTE: Accumulates losses as list and sums to preserve gradient flow.
        """
        cfg = self.config
        T = len(xyz_sequence)
        device = xyz_sequence[0].device
        
        # ARAP loss (accumulate as list to preserve gradients)
        arap_losses = []
        if cfg.arap_weight > 0 and T >= 2:
            for t in range(T - 1):
                arap_losses.append(self.arap_loss(xyz_sequence[t], xyz_sequence[t + 1]))
        
        # Velocity smoothness (accumulate as list)
        vel_losses = []
        if cfg.velocity_weight > 0 and T >= 3:
            for t in range(T - 2):
                vel_losses.append(self.velocity_smoothness_loss(
                    xyz_sequence[t], xyz_sequence[t + 1], xyz_sequence[t + 2]
                ))
        
        # Compute means (preserves gradients)
        if arap_losses:
            arap_mean = torch.stack(arap_losses).mean()
        else:
            arap_mean = torch.zeros(1, device=device, requires_grad=True).squeeze()
        
        if vel_losses:
            vel_mean = torch.stack(vel_losses).mean()
        else:
            vel_mean = torch.zeros(1, device=device, requires_grad=True).squeeze()
        
        # Total
        total = cfg.arap_weight * arap_mean + cfg.velocity_weight * vel_mean
        
        return {
            'arap': arap_mean,
            'velocity': vel_mean,
            'total': total,
        }


if __name__ == '__main__':
    print('Testing efficient temporal regularization...')
    
    config = TemporalRegConfig(
        arap_weight=0.1,
        velocity_weight=0.01,
        max_points_for_knn=2048,
    )
    
    reg = TemporalRegularizationEfficient(config).cuda()
    
    # Test with large point cloud
    N = 50000
    xyz_seq = [
        torch.randn(N, 3, device='cuda', requires_grad=True)
        for _ in range(3)
    ]
    
    losses = reg.compute_losses(xyz_seq)
    print(f'Total loss: {losses["total"].item():.6f}')
    print(f'ARAP: {losses["arap"].item():.6f}')
    print(f'Velocity: {losses["velocity"].item():.6f}')
    print(f'Requires grad: {losses["total"].requires_grad}')
    
    losses['total'].backward()
    grad_norm = xyz_seq[0].grad.norm().item()
    print(f'Gradient norm: {grad_norm:.4f}')
    print('✅ Test passed')
