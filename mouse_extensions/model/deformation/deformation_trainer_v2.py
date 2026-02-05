# Copyright 2026 FaceLift Mouse Extensions
# DeformationTrainerV2: Training with per-frame information and proper losses

"""
V2 Deformation Trainer with:
- Both frame information (G_t, G_{t+1}) as input
- ARAP loss for local rigidity
- Velocity loss for smooth motion
- Photo loss (rendering-based) for quality
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .deformation_network import DeformationNetworkV2, DeformationConfigV2
from .gaussian_params import GaussianParams


@dataclass
class TrainerConfigV2:
    """Configuration for V2 deformation training."""
    
    # Model
    deform_config: DeformationConfigV2 = field(default_factory=DeformationConfigV2)
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_steps: int = 10000
    warmup_steps: int = 500
    
    # Loss weights
    photo_weight: float = 1.0      # L2 on Gaussian params
    arap_weight: float = 0.1       # Local rigidity
    velocity_weight: float = 0.01  # Smooth motion
    
    # ARAP settings
    arap_k_neighbors: int = 8
    arap_max_points: int = 4096    # Sample for OOM prevention
    
    # Logging
    log_every: int = 100
    save_every: int = 1000
    
    # Output
    output_dir: str = "outputs/deformation_v2"
    device: str = "cuda"


class ARAPLoss(nn.Module):
    """
    As-Rigid-As-Possible loss for local rigidity preservation.
    
    Ensures that local neighborhoods maintain their structure
    across consecutive frames.
    """
    
    def __init__(self, k_neighbors: int = 8, max_points: int = 4096):
        super().__init__()
        self.k = k_neighbors
        self.max_points = max_points
    
    def forward(
        self,
        xyz_t: torch.Tensor,
        xyz_t1_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ARAP loss between two point sets.
        
        Args:
            xyz_t: [N, 3] positions at time t
            xyz_t1_pred: [N, 3] predicted positions at time t+1
            
        Returns:
            loss: scalar ARAP loss
        """
        N = xyz_t.shape[0]
        device = xyz_t.device
        
        # Sample points for efficiency (OOM prevention)
        if N > self.max_points:
            indices = torch.randperm(N, device=device)[:self.max_points]
            xyz_t = xyz_t[indices]
            xyz_t1_pred = xyz_t1_pred[indices]
            N = self.max_points
        
        # Find K-NN neighbors
        # Use cdist for pairwise distances
        dists = torch.cdist(xyz_t, xyz_t)  # [N, N]
        _, knn_indices = dists.topk(self.k + 1, largest=False)  # [N, k+1]
        knn_indices = knn_indices[:, 1:]  # Exclude self, [N, k]
        
        # Compute edge vectors at time t
        # xyz_t: [N, 3], knn_indices: [N, k]
        neighbors_t = xyz_t[knn_indices]  # [N, k, 3]
        edges_t = neighbors_t - xyz_t.unsqueeze(1)  # [N, k, 3]
        edge_lengths_t = edges_t.norm(dim=-1)  # [N, k]
        
        # Compute edge vectors at time t+1 (predicted)
        neighbors_t1 = xyz_t1_pred[knn_indices]  # [N, k, 3]
        edges_t1 = neighbors_t1 - xyz_t1_pred.unsqueeze(1)  # [N, k, 3]
        edge_lengths_t1 = edges_t1.norm(dim=-1)  # [N, k]
        
        # ARAP: edge lengths should be preserved
        loss = F.mse_loss(edge_lengths_t1, edge_lengths_t)
        
        return loss


class VelocityLoss(nn.Module):
    """
    Velocity smoothness loss.
    
    Penalizes large or sudden changes in velocity.
    """
    
    def __init__(self, max_points: int = 4096):
        super().__init__()
        self.max_points = max_points
    
    def forward(
        self,
        xyz_t: torch.Tensor,
        xyz_t1_pred: torch.Tensor,
        xyz_t_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity loss.
        
        Args:
            xyz_t: [N, 3] positions at time t
            xyz_t1_pred: [N, 3] predicted positions at time t+1
            xyz_t_prev: [N, 3] positions at time t-1 (optional, for acceleration)
            
        Returns:
            loss: scalar velocity loss
        """
        N = xyz_t.shape[0]
        device = xyz_t.device
        
        # Sample for efficiency
        if N > self.max_points:
            indices = torch.randperm(N, device=device)[:self.max_points]
            xyz_t = xyz_t[indices]
            xyz_t1_pred = xyz_t1_pred[indices]
            if xyz_t_prev is not None:
                xyz_t_prev = xyz_t_prev[indices]
        
        # Velocity at t -> t+1
        velocity = xyz_t1_pred - xyz_t  # [N, 3]
        
        if xyz_t_prev is not None:
            # Acceleration (velocity change)
            velocity_prev = xyz_t - xyz_t_prev
            acceleration = velocity - velocity_prev
            loss = acceleration.pow(2).mean()
        else:
            # Just penalize large velocities
            loss = velocity.pow(2).mean()
        
        return loss


class DeformationTrainerV2:
    """
    V2 Trainer for deformation network.
    
    Key differences from V1:
    - Uses both G_t and G_{t+1} as input
    - ARAP + Velocity losses for temporal consistency
    - Proper gradient flow (no .detach() issues)
    """
    
    def __init__(self, config: TrainerConfigV2):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create deformation network V2
        self.deform_net = DeformationNetworkV2(config.deform_config)
        self.deform_net.to(self.device)
        
        # Loss modules
        self.arap_loss = ARAPLoss(
            k_neighbors=config.arap_k_neighbors,
            max_points=config.arap_max_points,
        )
        self.velocity_loss = VelocityLoss(max_points=config.arap_max_points)
        
        # Optimizer
        self.optimizer = AdamW(
            self.deform_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.01,
        )
        
        # State
        self.global_step = 0
        self.best_loss = float("inf")
        
        # Output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(
        self,
        G_t: GaussianParams,
        G_t1_target: GaussianParams,
        time_index: int = 0,
        G_t_prev: Optional[GaussianParams] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        V2 training step with per-frame information.
        
        Args:
            G_t: Gaussians at time t
            G_t1_target: Gaussians at time t+1 (target/pseudo GT)
            time_index: Frame index for time embedding
            G_t_prev: Gaussians at time t-1 (for velocity consistency)
            
        Returns:
            Dict with loss values
        """
        self.deform_net.train()
        
        # Move to device
        G_t = G_t.to(self.device)
        G_t1_target = G_t1_target.to(self.device)
        
        # Time index tensor
        N = G_t.xyz.shape[0]
        time_t = torch.full((N,), time_index, device=self.device, dtype=torch.float)
        
        # Predict deformation using BOTH frames
        deform_raw = self.deform_net(G_t.xyz, G_t1_target.xyz, time_t)
        deform_dict = self.deform_net.parse_output(deform_raw)
        
        # Apply deformation to G_t
        G_t1_pred = G_t.apply_deformation(deform_dict)
        
        # Compute losses
        losses = {}
        cfg = self.config
        
        # 1. Photo/Param loss: G_t1_pred should match G_t1_target
        param_loss = self._compute_param_loss(G_t1_pred, G_t1_target)
        losses["param_loss"] = param_loss
        
        # 2. ARAP loss: preserve local structure
        arap_loss = self.arap_loss(G_t.xyz, G_t1_pred.xyz)
        losses["arap_loss"] = arap_loss
        
        # 3. Velocity loss: smooth motion
        xyz_t_prev = G_t_prev.xyz.to(self.device) if G_t_prev is not None else None
        velocity_loss = self.velocity_loss(G_t.xyz, G_t1_pred.xyz, xyz_t_prev)
        losses["velocity_loss"] = velocity_loss
        
        # Total loss
        total_loss = (
            cfg.photo_weight * param_loss +
            cfg.arap_weight * arap_loss +
            cfg.velocity_weight * velocity_loss
        )
        losses["total_loss"] = total_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.deform_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return losses
    
    def _compute_param_loss(
        self,
        pred: GaussianParams,
        target: GaussianParams,
    ) -> torch.Tensor:
        """L2 loss on Gaussian parameters."""
        # Position loss (most important)
        pos_loss = F.mse_loss(pred.xyz, target.xyz)
        
        # Opacity loss
        opacity_loss = F.mse_loss(pred.opacity, target.opacity)
        
        # Scale loss
        scale_loss = F.mse_loss(pred.scaling, target.scaling)
        
        return pos_loss + 0.1 * opacity_loss + 0.1 * scale_loss
    
    def save_checkpoint(self, path: Optional[str] = None):
        if path is None:
            path = self.output_dir / f"checkpoint_{self.global_step:06d}.pt"
        
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.deform_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "best_loss": self.best_loss,
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.deform_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
    
    def get_lr(self) -> float:
        return self.scheduler.get_last_lr()[0]


# ============================================================
# Unit Tests
# ============================================================

def _test_losses():
    """Test ARAP and Velocity losses."""
    print("Testing losses...")
    
    N = 1000
    xyz_t = torch.randn(N, 3)
    xyz_t1 = xyz_t + torch.randn(N, 3) * 0.1  # Small deformation
    xyz_t_prev = xyz_t - torch.randn(N, 3) * 0.1
    
    # Test ARAP
    arap = ARAPLoss(k_neighbors=8, max_points=512)
    arap_val = arap(xyz_t, xyz_t1)
    print(f"  ARAP loss: {arap_val.item():.6f}")
    
    # Test Velocity
    vel = VelocityLoss(max_points=512)
    vel_val = vel(xyz_t, xyz_t1, xyz_t_prev)
    print(f"  Velocity loss: {vel_val.item():.6f}")
    
    # Test gradient flow
    xyz_t.requires_grad_(True)
    arap_val = arap(xyz_t, xyz_t1)
    arap_val.backward()
    assert xyz_t.grad is not None, "Gradient should flow"
    print("  Gradient flow: ✓")
    
    print("Loss tests passed! ✓")


def _test_trainer_v2():
    """Test DeformationTrainerV2."""
    import tempfile
    
    print("\nTesting DeformationTrainerV2...")
    
    N = 500
    sh_degree = 2
    feature_dim = (sh_degree + 1) ** 2 * 3
    
    def make_gaussian():
        return GaussianParams(
            xyz=torch.randn(N, 3),
            features=torch.randn(N, feature_dim),
            scaling=torch.randn(N, 3),
            rotation=F.normalize(torch.randn(N, 4), dim=-1),
            opacity=torch.randn(N, 1),
        )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = TrainerConfigV2(
            output_dir=tmpdir,
            device="cpu",
            max_steps=100,
        )
        trainer = DeformationTrainerV2(config)
        print(f"  Created trainer: {trainer.deform_net}")
        
        # Train step
        G_t = make_gaussian()
        G_t1 = make_gaussian()
        G_t_prev = make_gaussian()
        
        losses = trainer.train_step(G_t, G_t1, time_index=5, G_t_prev=G_t_prev)
        
        assert "total_loss" in losses
        assert "param_loss" in losses
        assert "arap_loss" in losses
        assert "velocity_loss" in losses
        
        print(f"  Train step losses:")
        for k, v in losses.items():
            print(f"    {k}: {v.item():.6f}")
        
        # Multiple steps
        for i in range(5):
            losses = trainer.train_step(make_gaussian(), make_gaussian(), i)
        
        print(f"  After 6 steps: global_step={trainer.global_step}")
        
        # Save/load
        trainer.save_checkpoint()
        trainer2 = DeformationTrainerV2(config)
        ckpt_path = Path(tmpdir) / f"checkpoint_{trainer.global_step:06d}.pt"
        trainer2.load_checkpoint(str(ckpt_path))
        assert trainer2.global_step == trainer.global_step
        print(f"  Checkpoint round-trip: ✓")
    
    print("Trainer V2 tests passed! ✓")


if __name__ == "__main__":
    _test_losses()
    _test_trainer_v2()
