#!/usr/bin/env python3
"""
Train DeformationNetworkV2 (8-layer MLP) for temporal consistency.

This script trains the deformation network to learn:
1. How to move G_t toward G_{t+1} while preserving local structure (ARAP)
2. Smooth motion patterns (velocity loss)

Usage:
    python -m mouse_extensions.scripts.train_deform_v2 \
        --config configs/mouse/deform_v2.yaml \
        --gaussians_dir /path/to/gslrm_outputs \
        --output_dir /path/to/checkpoints

The gaussians_dir should contain per-frame GS-LRM outputs:
    gaussians_dir/
        frame_0000.pt  # {xyz, features, scaling, rotation, opacity}
        frame_0001.pt
        ...
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

from mouse_extensions.model.deformation import (
    DeformationNetworkV2,
    DeformationConfigV2,
    DeformationTrainerV2,
    TrainerConfigV2,
    GaussianParams,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    gaussians_dir: str = ""
    num_frames: int = 0  # 0 = auto-detect
    
    # Network
    num_layers: int = 8
    hidden_dim: int = 256
    use_positional_encoding: bool = True
    use_time_embedding: bool = True
    
    # Training
    num_epochs: int = 100
    batch_size: int = 10000  # Gaussians per batch
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss weights
    param_weight: float = 1.0
    arap_weight: float = 0.1
    velocity_weight: float = 0.01
    
    # Schedule
    warmup_steps: int = 100
    
    # Checkpointing
    output_dir: str = "checkpoints/deform_v2"
    save_every: int = 100
    
    # Device
    device: str = "cuda"


class FramePairDataset(Dataset):
    """Dataset of consecutive frame pairs for deformation training."""
    
    def __init__(self, gaussians_dir: str, num_frames: int = 0):
        self.gaussians_dir = Path(gaussians_dir)
        
        # Find all frame files
        self.frame_files = sorted(self.gaussians_dir.glob("frame_*.pt"))
        if not self.frame_files:
            # Try alternative naming
            self.frame_files = sorted(self.gaussians_dir.glob("*.pt"))
        
        if num_frames > 0:
            self.frame_files = self.frame_files[:num_frames]
        
        self.num_frames = len(self.frame_files)
        logger.info(f"Found {self.num_frames} frames in {gaussians_dir}")
    
    def __len__(self):
        return max(0, self.num_frames - 1)  # pairs
    
    def __getitem__(self, idx) -> Tuple[GaussianParams, GaussianParams, int]:
        """Return (G_t, G_{t+1}, time_index)."""
        # Load frames
        g_t = self._load_frame(idx)
        g_t1 = self._load_frame(idx + 1)
        
        return g_t, g_t1, idx
    
    def _load_frame(self, idx: int) -> GaussianParams:
        """Load GaussianParams from file."""
        data = torch.load(self.frame_files[idx], map_location='cpu')
        
        # Handle different save formats
        if isinstance(data, GaussianParams):
            return data
        elif isinstance(data, dict):
            return GaussianParams(
                xyz=data.get('xyz', data.get('positions')),
                features=data.get('features', data.get('sh_features')),
                scaling=data.get('scaling', data.get('scales')),
                rotation=data.get('rotation', data.get('rotations')),
                opacity=data.get('opacity', data.get('opacities')),
            )
        else:
            raise ValueError(f"Unknown data format: {type(data)}")


def collate_gaussians(batch: List[Tuple[GaussianParams, GaussianParams, int]]):
    """Collate function for Gaussian pairs."""
    # Don't actually collate - return list for variable-size Gaussians
    return batch


def train_epoch(
    trainer: DeformationTrainerV2,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str,
) -> dict:
    """Train for one epoch."""
    trainer.deform_net.train()
    
    total_losses = {'param_loss': 0, 'arap_loss': 0, 'velocity_loss': 0, 'total_loss': 0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    prev_g_t = None

    for batch in pbar:
        # batch = List[Tuple[GaussianParams, GaussianParams, int]]
        for g_t, g_t1, time_idx in batch:
            # train_step handles forward + backward + optimizer.step internally
            losses = trainer.train_step(
                G_t=g_t,
                G_t1_target=g_t1,
                time_index=time_idx,
                G_t_prev=prev_g_t,
            )

            for k, v in losses.items():
                if k in total_losses:
                    total_losses[k] += v.item()
            num_batches += 1
            prev_g_t = g_t
            
            # Update prev for velocity loss
            prev_g_t = g_t

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'arap': f"{losses['arap_loss'].item():.6f}",
            })
    
    # Average
    for k in total_losses:
        total_losses[k] /= max(num_batches, 1)
    
    return total_losses


def save_checkpoint(
    trainer: DeformationTrainerV2,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_dir: Path,
):
    """Save training checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = {
        'epoch': epoch,
        'network_state_dict': trainer.deform_net.state_dict(),
        'network_config': trainer.deform_net.config,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    path = output_dir / f"deform_v2_epoch_{epoch:04d}.pt"
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint to {path}")
    
    # Also save as latest
    latest_path = output_dir / "deform_v2_latest.pt"
    torch.save(ckpt, latest_path)


def main():
    parser = argparse.ArgumentParser(description="Train DeformationNetworkV2")
    parser.add_argument("--config", type=str, help="YAML config file")
    parser.add_argument("--gaussians_dir", type=str, help="Directory with frame_*.pt files")
    parser.add_argument("--output_dir", type=str, default="checkpoints/deform_v2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = TrainConfig()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        for k, v in cfg_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # Override with args
    if args.gaussians_dir:
        config.gaussians_dir = args.gaussians_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.device:
        config.device = args.device
    
    # Validate
    if not config.gaussians_dir:
        logger.error("Please provide --gaussians_dir")
        return
    
    logger.info(f"Config: {config}")
    
    # Create dataset
    dataset = FramePairDataset(config.gaussians_dir, config.num_frames)
    if len(dataset) == 0:
        logger.error("No frame pairs found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Variable-size Gaussians
        shuffle=True,
        collate_fn=collate_gaussians,
    )
    
    # Create trainer config (includes deform network config)
    net_config = DeformationConfigV2(
        num_layers=config.num_layers,
        hidden_dim=config.hidden_dim,
        use_positional_encoding=config.use_positional_encoding,
        use_time_embedding=config.use_time_embedding,
    )
    trainer_config = TrainerConfigV2(
        deform_config=net_config,
        photo_weight=config.param_weight,
        arap_weight=config.arap_weight,
        velocity_weight=config.velocity_weight,
        output_dir=config.output_dir,
        device=config.device,
    )
    trainer = DeformationTrainerV2(trainer_config)
    
    logger.info(f"Network: {trainer.deform_net}")
    n_params = sum(p.numel() for p in trainer.deform_net.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # Use trainer's internal optimizer
    optimizer = trainer.optimizer

    # Resume from checkpoint if requested
    start_epoch = 1
    output_dir = Path(config.output_dir)
    if args.resume:
        latest_ckpt = output_dir / "deform_v2_latest.pt"
        if latest_ckpt.exists():
            ckpt = torch.load(str(latest_ckpt), map_location=config.device, weights_only=False)
            trainer.deform_net.load_state_dict(ckpt["network_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resumed from epoch {ckpt['epoch']}, starting at epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {latest_ckpt}, starting from scratch")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs + 1):
        losses = train_epoch(trainer, dataloader, optimizer, epoch, config.device)
        
        logger.info(
            f"Epoch {epoch}: "
            f"total={losses['total_loss']:.4f}, "
            f"param={losses['param_loss']:.4f}, "
            f"arap={losses['arap_loss']:.6f}, "
            f"velocity={losses['velocity_loss']:.6f}"
        )
        
        # Save checkpoint + visualization
        if epoch % config.save_every == 0 or epoch == config.num_epochs:
            save_checkpoint(trainer, optimizer, epoch, output_dir)
            # Auto-visualize at checkpoint
            viz_dir = output_dir / f"viz_epoch_{epoch:04d}"
            deform_ckpt = output_dir / f"deform_v2_epoch_{epoch:04d}.pt"
            try:
                import subprocess
                cmd = [
                    sys.executable, "-m", "mouse_extensions.scripts.viz_deformation",
                    "--cache_dir", config.gaussians_dir,
                    "--deform_ckpt", str(deform_ckpt),
                    "--output_dir", str(viz_dir),
                    "--frame_range", "1500:1510",  # Quick 10-frame sample
                    "--resolution", "512",
                    "--device", config.device,
                ]
                logger.info(f"Generating visualization at {viz_dir}...")
                subprocess.run(cmd, timeout=600)
                logger.info(f"Visualization saved to {viz_dir}")
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
