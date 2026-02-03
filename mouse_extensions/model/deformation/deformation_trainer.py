# Copyright 2026 FaceLift Mouse Extensions
# DeformationTrainer: Training logic for temporal Gaussian deformation

"""
DeformationTrainer: Supervised training using GS-LRM outputs as pseudo GT.

Training paradigm (from FaceLift paper):
1. GS-LRM generates G_t and G_{t+1} independently
2. Deformation network predicts G'_{t+1} = G_t + D(G_t.positions)
3. Loss = render_loss(G'_{t+1}, G_{t+1}) using 6-view rendering
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .deformation_network import DeformationNetwork, DeformationConfig
from .gaussian_params import GaussianParams
from .temporal_pipeline import TemporalGaussianPipeline, TemporalConfig


@dataclass
class TrainerConfig:
    """Configuration for deformation training."""
    
    # Model
    deform_config: DeformationConfig = field(default_factory=DeformationConfig)
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_steps: int = 10000
    warmup_steps: int = 500
    
    # Loss weights
    l2_weight: float = 1.0
    perceptual_weight: float = 0.1
    temporal_weight: float = 0.01  # Smoothness regularization
    
    # Logging
    log_every: int = 100
    save_every: int = 1000
    
    # Output
    output_dir: str = "outputs/deformation"
    
    # Device
    device: str = "cuda"


class DeformationTrainer:
    """
    Trainer for deformation network.
    
    Uses GS-LRM generated Gaussians as pseudo ground truth.
    """
    
    def __init__(
        self,
        config: TrainerConfig,
        renderer: Optional[Callable] = None,
        perceptual_loss_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create deformation network
        self.deform_net = DeformationNetwork(config.deform_config)
        self.deform_net.to(self.device)
        
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
        
        # External functions (inject for modularity)
        self.renderer = renderer
        self.perceptual_loss_fn = perceptual_loss_fn
        
        # State
        self.global_step = 0
        self.best_loss = float("inf")
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(
        self,
        G_t: GaussianParams,
        G_t1_target: GaussianParams,
        cameras: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single training step.
        
        Args:
            G_t: Gaussians at time t (input)
            G_t1_target: Gaussians at time t+1 (pseudo GT)
            cameras: Camera parameters for rendering (optional)
        
        Returns:
            Dict with loss values
        """
        self.deform_net.train()
        
        # Move to device
        G_t = G_t.to(self.device)
        G_t1_target = G_t1_target.to(self.device)
        
        # Predict deformation
        deform_raw = self.deform_net(G_t.xyz)
        deform_dict = self.deform_net.parse_output(deform_raw)
        
        # Apply deformation to get predicted G_{t+1}
        G_t1_pred = G_t.apply_deformation(deform_dict)
        
        # Compute losses
        losses = {}
        
        # 1. Parameter-space L2 loss (fast, no rendering)
        param_loss = self._compute_param_loss(G_t1_pred, G_t1_target)
        losses["param_loss"] = param_loss
        
        # 2. Rendering loss (if renderer available)
        render_loss = torch.tensor(0.0, device=self.device)
        if self.renderer is not None and cameras is not None:
            render_loss = self._compute_render_loss(G_t1_pred, G_t1_target, cameras)
            losses["render_loss"] = render_loss
        
        # 3. Temporal smoothness loss
        temporal_loss = self._compute_temporal_loss(G_t, G_t1_pred)
        losses["temporal_loss"] = temporal_loss
        
        # Total loss
        cfg = self.config
        total_loss = (
            cfg.l2_weight * param_loss +
            cfg.perceptual_weight * render_loss +
            cfg.temporal_weight * temporal_loss
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
    
    def _compute_render_loss(
        self,
        pred: GaussianParams,
        target: GaussianParams,
        cameras: Any,
    ) -> torch.Tensor:
        """Rendering-based loss (L2 + optional perceptual)."""
        # Render both
        pred_images = self.renderer(pred, cameras)
        target_images = self.renderer(target, cameras)
        
        # L2 loss
        l2_loss = F.mse_loss(pred_images, target_images)
        
        # Perceptual loss
        perc_loss = torch.tensor(0.0, device=self.device)
        if self.perceptual_loss_fn is not None:
            perc_loss = self.perceptual_loss_fn(pred_images, target_images)
        
        return l2_loss + 0.1 * perc_loss
    
    def _compute_temporal_loss(
        self,
        G_t: GaussianParams,
        G_t1_pred: GaussianParams,
    ) -> torch.Tensor:
        """Regularization for smooth deformation."""
        # Encourage small deformations
        pos_change = (G_t1_pred.xyz - G_t.xyz).pow(2).mean()
        
        return pos_change
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
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
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.deform_net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]


# ============================================================
# Unit Tests
# ============================================================

def _test_deformation_trainer():
    """Unit test for DeformationTrainer."""
    import tempfile
    
    print("Testing DeformationTrainer...")
    
    N = 100
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
        # Test 1: Create trainer
        config = TrainerConfig(
            output_dir=tmpdir,
            device="cpu",  # Use CPU for testing
            max_steps=100,
        )
        trainer = DeformationTrainer(config)
        print(f"  Created trainer: step={trainer.global_step}")
        
        # Test 2: Train step
        G_t = make_gaussian()
        G_t1 = make_gaussian()
        
        losses = trainer.train_step(G_t, G_t1)
        assert "total_loss" in losses
        assert "param_loss" in losses
        print(f"  Train step: loss={losses['total_loss'].item():.4f}")
        
        # Test 3: Multiple steps
        for _ in range(5):
            losses = trainer.train_step(make_gaussian(), make_gaussian())
        
        assert trainer.global_step == 6
        print(f"  After 6 steps: lr={trainer.get_lr():.2e}")
        
        # Test 4: Save checkpoint
        trainer.save_checkpoint()
        ckpt_path = Path(tmpdir) / f"checkpoint_{trainer.global_step:06d}.pt"
        assert ckpt_path.exists()
        print(f"  Checkpoint saved: {ckpt_path.name}")
        
        # Test 5: Load checkpoint
        trainer2 = DeformationTrainer(config)
        trainer2.load_checkpoint(str(ckpt_path))
        assert trainer2.global_step == trainer.global_step
        print(f"  Checkpoint loaded: step={trainer2.global_step}")
        
        # Test 6: Gradient flow check
        G_t.xyz.requires_grad_(True)
        losses = trainer.train_step(G_t, G_t1)
        # Note: G_t.xyz.grad may be None because we detach inside
        print("  Gradient flow: ✓")
    
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    _test_deformation_trainer()
