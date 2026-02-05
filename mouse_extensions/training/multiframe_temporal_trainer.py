"""Multi-frame Temporal Training for GS-LRM.

Key insight: Load consecutive frames in the SAME batch to enable gradient flow
for temporal regularization.

Pipeline:
1. DataLoader yields (frame_t, frame_t+1) pairs
2. Forward both through GS-LRM → (G_t, G_{t+1})
3. Photometric loss: L_photo(G_t) + L_photo(G_{t+1})
4. Temporal loss: ARAP(G_t, G_{t+1}) + Velocity(G_t, G_{t+1})
5. Single backward with combined loss
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict


@dataclass
class MultiFrameTemporalConfig:
    """Configuration for multi-frame temporal training."""
    enabled: bool = True
    
    # Loss weights
    arap_weight: float = 0.1
    velocity_weight: float = 0.01
    
    # ARAP settings
    arap_k_neighbors: int = 8
    arap_max_points: int = 4096  # Sample for efficiency
    
    # Warmup
    warmup_steps: int = 500
    rampup_steps: int = 500


class MultiFrameTemporalLoss(nn.Module):
    """Compute temporal regularization between consecutive Gaussian frames.
    
    Both G_t and G_{t+1} must be in the same computation graph (same batch forward).
    """
    
    def __init__(self, config: MultiFrameTemporalConfig):
        super().__init__()
        self.config = config
        self._knn_cache = {}
    
    def forward(
        self,
        xyz_t: torch.Tensor,      # [N, 3] - frame t positions (with grad)
        xyz_t1: torch.Tensor,     # [N, 3] - frame t+1 positions (with grad)
        step: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute temporal losses between consecutive frames.
        
        Args:
            xyz_t: Gaussian positions at frame t
            xyz_t1: Gaussian positions at frame t+1
            step: Current training step
            
        Returns:
            Dict with 'loss', 'arap_loss', 'velocity_loss'
        """
        cfg = self.config
        device = xyz_t.device
        
        # Warmup check
        if step < cfg.warmup_steps:
            return {
                'loss': torch.tensor(0.0, device=device),
                'arap_loss': torch.tensor(0.0, device=device),
                'velocity_loss': torch.tensor(0.0, device=device),
            }
        
        # Rampup factor
        if cfg.rampup_steps > 0:
            steps_since_warmup = step - cfg.warmup_steps
            rampup = min(1.0, (steps_since_warmup + 1) / cfg.rampup_steps)
        else:
            rampup = 1.0
        
        # Sample points for efficiency
        N = xyz_t.shape[0]
        if N > cfg.arap_max_points:
            indices = torch.randperm(N, device=device)[:cfg.arap_max_points]
            xyz_t_sampled = xyz_t[indices]
            xyz_t1_sampled = xyz_t1[indices]
        else:
            xyz_t_sampled = xyz_t
            xyz_t1_sampled = xyz_t1
            indices = None
        
        # Compute KNN for ARAP (on frame t)
        knn_indices = self._compute_knn(xyz_t_sampled, cfg.arap_k_neighbors)
        
        # ARAP Loss: preserve edge lengths
        arap_loss = self._compute_arap_loss(
            xyz_t_sampled, xyz_t1_sampled, knn_indices
        )
        
        # Velocity smoothness: penalize large movements
        velocity_loss = self._compute_velocity_loss(xyz_t_sampled, xyz_t1_sampled)
        
        # Combine losses with rampup
        total_loss = rampup * (
            cfg.arap_weight * arap_loss + 
            cfg.velocity_weight * velocity_loss
        )
        
        return {
            'loss': total_loss,
            'arap_loss': arap_loss,
            'velocity_loss': velocity_loss,
            'rampup': rampup,
        }
    
    def _compute_knn(self, xyz: torch.Tensor, k: int) -> torch.Tensor:
        """Compute k-nearest neighbors."""
        # Simple brute-force for sampled points
        dists = torch.cdist(xyz, xyz)  # [N, N]
        _, indices = dists.topk(k + 1, dim=-1, largest=False)  # +1 for self
        return indices[:, 1:]  # Exclude self, shape [N, k]
    
    def _compute_arap_loss(
        self,
        xyz_t: torch.Tensor,
        xyz_t1: torch.Tensor,
        knn_indices: torch.Tensor,
    ) -> torch.Tensor:
        """ARAP loss: preserve local edge lengths across frames.
        
        L_ARAP = mean_i mean_j ||d(i,j)_t - d(i,j)_{t+1}||^2
        """
        # Gather neighbor positions
        # xyz_t: [N, 3], knn_indices: [N, k]
        neighbors_t = xyz_t[knn_indices]    # [N, k, 3]
        neighbors_t1 = xyz_t1[knn_indices]  # [N, k, 3]
        
        # Compute edge vectors
        edges_t = neighbors_t - xyz_t.unsqueeze(1)    # [N, k, 3]
        edges_t1 = neighbors_t1 - xyz_t1.unsqueeze(1)  # [N, k, 3]
        
        # Compute edge lengths
        lengths_t = edges_t.norm(dim=-1)    # [N, k]
        lengths_t1 = edges_t1.norm(dim=-1)  # [N, k]
        
        # Loss: preserve edge lengths
        arap_loss = F.mse_loss(lengths_t, lengths_t1)
        
        return arap_loss
    
    def _compute_velocity_loss(
        self,
        xyz_t: torch.Tensor,
        xyz_t1: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity smoothness: penalize large movements.
        
        L_vel = mean ||xyz_{t+1} - xyz_t||^2
        """
        velocity = xyz_t1 - xyz_t  # [N, 3]
        velocity_loss = velocity.pow(2).mean()
        
        return velocity_loss


class MultiFrameTemporalTrainer:
    """Wrapper that enables multi-frame temporal training.
    
    Usage:
        trainer = GSLRMTrainer(config)
        temporal_trainer = MultiFrameTemporalTrainer(trainer, temporal_config)
        
        for batch in dataloader:
            # batch contains consecutive frame pairs
            loss_dict = temporal_trainer.train_step(batch)
    """
    
    def __init__(
        self,
        base_trainer,  # GSLRMTrainer instance
        temporal_config: MultiFrameTemporalConfig,
    ):
        self.trainer = base_trainer
        self.config = temporal_config
        self.temporal_loss_fn = MultiFrameTemporalLoss(temporal_config)
        self.temporal_loss_fn.to(base_trainer.device)
        
        print(f"[MultiFrameTemporal] Initialized with:"
              f" arap={temporal_config.arap_weight},"
              f" velocity={temporal_config.velocity_weight},"
              f" warmup={temporal_config.warmup_steps}")
    
    def train_step_pair(
        self,
        batch_t: Dict,
        batch_t1: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Train step with consecutive frame pair.
        
        Args:
            batch_t: Batch for frame t
            batch_t1: Batch for frame t+1
            
        Returns:
            Combined loss dictionary
        """
        trainer = self.trainer
        step = trainer.fwdbwd_pass_step
        
        # Forward pass for both frames (in same computation graph)
        result_t = trainer.model(batch_t, create_visual=False)
        result_t1 = trainer.model(batch_t1, create_visual=False)
        
        # Get xyz with gradients
        xyz_t = result_t.gaussian_params_raw.xyz[0]   # [N, 3]
        xyz_t1 = result_t1.gaussian_params_raw.xyz[0]  # [N, 3]
        
        # Photometric losses (already computed in forward)
        photo_loss_t = result_t.loss_metrics.loss
        photo_loss_t1 = result_t1.loss_metrics.loss
        
        # Temporal regularization
        temporal_dict = self.temporal_loss_fn(xyz_t, xyz_t1, step)
        
        # Combine all losses
        grad_accum = trainer.config.training.runtime.grad_accum_steps
        total_loss = (
            (photo_loss_t + photo_loss_t1) / 2 +  # Average photo loss
            temporal_dict['loss']
        ) / grad_accum
        
        # Single backward
        trainer.scaler.scale(total_loss).backward()
        trainer.fwdbwd_pass_step += 1
        
        # Return metrics
        return {
            'loss': total_loss.item() * grad_accum,
            'photo_loss': (photo_loss_t + photo_loss_t1).item() / 2,
            'temporal_loss': temporal_dict['loss'].item(),
            'arap_loss': temporal_dict['arap_loss'].item(),
            'velocity_loss': temporal_dict['velocity_loss'].item(),
            'rampup': temporal_dict.get('rampup', 1.0),
        }


def create_multiframe_dataloader(
    base_dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """Create dataloader that yields consecutive frame pairs.
    
    Returns batches where each item is (batch_t, batch_t1).
    """
    from torch.utils.data import DataLoader, Dataset
    
    class ConsecutiveFrameDataset(Dataset):
        """Wraps base dataset to yield consecutive pairs."""
        
        def __init__(self, base):
            self.base = base
            # Assume samples are sorted by frame index
            self.length = len(base) - 1  # Pairs: (0,1), (1,2), ..., (N-2, N-1)
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            sample_t = self.base[idx]
            sample_t1 = self.base[idx + 1]
            return sample_t, sample_t1
    
    pair_dataset = ConsecutiveFrameDataset(base_dataset)
    
    def collate_pairs(batch):
        # batch is list of (sample_t, sample_t1) tuples
        batch_t = [b[0] for b in batch]
        batch_t1 = [b[1] for b in batch]
        
        # Use base collate for each
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch_t), default_collate(batch_t1)
    
    return DataLoader(
        pair_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
    )


# Test
def _test_multiframe_temporal():
    """Unit test for multi-frame temporal loss."""
    print("Testing MultiFrameTemporalLoss...")
    
    config = MultiFrameTemporalConfig(warmup_steps=0)
    loss_fn = MultiFrameTemporalLoss(config)
    
    N = 1000
    xyz_t = torch.randn(N, 3, requires_grad=True)
    xyz_t1 = xyz_t + torch.randn(N, 3) * 0.1  # Small movement
    xyz_t1.requires_grad_(True)
    
    result = loss_fn(xyz_t, xyz_t1, step=100)
    
    print(f"  Total loss: {result['loss'].item():.6f}")
    print(f"  ARAP loss: {result['arap_loss'].item():.6f}")
    print(f"  Velocity loss: {result['velocity_loss'].item():.6f}")
    
    # Check gradients flow
    result['loss'].backward()
    assert xyz_t.grad is not None, "Gradients should flow to xyz_t"
    assert xyz_t1.grad is not None, "Gradients should flow to xyz_t1"
    print(f"  Gradient flow: ✓")
    
    print("Test passed!")


if __name__ == "__main__":
    _test_multiframe_temporal()
