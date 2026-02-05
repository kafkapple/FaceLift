#!/usr/bin/env python3
"""Train GS-LRM with multi-frame temporal regularization.

Key difference from standard training:
- Loads consecutive frame pairs in same batch
- Computes photometric loss + temporal regularization in single forward-backward

Usage:
    python -m mouse_extensions.scripts.train_temporal_gslrm \
        --config configs/mouse/temporal_multiframe_M5t2.yaml \
        --output /node_data/joon/checkpoints/FaceLift/gslrm/temporal_v2
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import yaml
from omegaconf import OmegaConf, DictConfig
from easydict import EasyDict as edict
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def load_config_with_base(config_path: str) -> edict:
    """Load config with _base_ inheritance support."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Handle _base_ inheritance
    if '_base_' in cfg:
        base_path = cfg.pop('_base_')
        base_cfg = OmegaConf.load(base_path)
        override_cfg = OmegaConf.create(cfg)
        merged = OmegaConf.merge(base_cfg, override_cfg)
        cfg = OmegaConf.to_container(merged, resolve=True)
    
    return edict(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='FaceLift-Mouse')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def create_consecutive_pair_dataset(config):
    """Create dataset that yields consecutive frame pairs."""
    from mouse_extensions.data import MouseViewDataset
    from torch.utils.data import Dataset, DataLoader
    
    # Load base dataset using MouseViewDataset
    base_dataset = MouseViewDataset(config, split="train")
    print(f'[Dataset] Base dataset: {len(base_dataset)} samples')
    
    class ConsecutivePairDataset(Dataset):
        """Yields (sample_t, sample_t1) pairs."""
        def __init__(self, base):
            self.base = base
            # Filter to ensure consecutive samples exist
            self.valid_indices = list(range(len(base) - 1))
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            real_idx = self.valid_indices[idx]
            return self.base[real_idx], self.base[real_idx + 1]
    
    pair_dataset = ConsecutivePairDataset(base_dataset)
    print(f'[Dataset] {len(pair_dataset)} consecutive pairs')
    
    def collate_pairs(batch):
        batch_t = [b[0] for b in batch]
        batch_t1 = [b[1] for b in batch]
        
        # Stack each field
        def stack_field(items, key):
            vals = [item[key] for item in items]
            if isinstance(vals[0], torch.Tensor):
                return torch.stack(vals)
            return vals
        
        keys = batch_t[0].keys()
        collated_t = {k: stack_field(batch_t, k) for k in keys}
        collated_t1 = {k: stack_field(batch_t1, k) for k in keys}
        
        return edict(collated_t), edict(collated_t1)
    
    dataloader = DataLoader(
        pair_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_pairs,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


def main():
    args = parse_args()
    
    # Load config
    config = load_config_with_base(args.config)
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%y%m%d_%H%M')
        output_dir = Path(f'/node_data/joon/checkpoints/FaceLift/gslrm/temporal_v2_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[Output] {output_dir}')
    
    # Temporal config
    temporal_config = config.get('temporal', edict({
        'enabled': True,
        'arap_weight': 0.1,
        'velocity_weight': 0.01,
        'warmup_steps': 500,
        'rampup_steps': 500,
    }))
    
    print(f'[Temporal] arap={temporal_config.arap_weight}, vel={temporal_config.velocity_weight}')
    print(f'[Temporal] warmup={temporal_config.warmup_steps}, rampup={temporal_config.rampup_steps}')
    
    if args.dry_run:
        print('[DRY RUN] Config loaded successfully, exiting')
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')
    
    # AMP settings
    use_amp = config.training.runtime.get('use_amp', True)
    amp_dtype_str = config.training.runtime.get('amp_dtype', 'bfloat16')
    amp_dtype_mapping = {
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float16': torch.float16,
        'fp16': torch.float16,
    }
    amp_dtype = amp_dtype_mapping.get(amp_dtype_str, torch.bfloat16)
    print(f'[AMP] enabled={use_amp}, dtype={amp_dtype}')
    
    # Create model
    from gslrm.model.gslrm import GSLRM
    model = GSLRM(config).to(device)
    print('Renderer initialized')
    
    # Load checkpoint if specified
    if config.model.get('checkpoint_path'):
        ckpt = torch.load(config.model.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f'[Model] Loaded from {config.model.checkpoint_path}')
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.get('weight_decay', 0.01),
    )
    
    # Grad scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Create dataloader
    dataloader = create_consecutive_pair_dataset(config)
    
    # Create temporal loss
    from mouse_extensions.training.multiframe_temporal_trainer import (
        MultiFrameTemporalLoss, MultiFrameTemporalConfig
    )
    temporal_loss_fn = MultiFrameTemporalLoss(MultiFrameTemporalConfig(
        arap_weight=temporal_config.arap_weight,
        velocity_weight=temporal_config.velocity_weight,
        warmup_steps=temporal_config.warmup_steps,
        rampup_steps=temporal_config.rampup_steps,
    )).to(device)
    
    # WandB
    import wandb
    if args.wandb_name:
        wandb_name = args.wandb_name
    else:
        wandb_name = output_dir.name
    
    wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        config=dict(config),
        dir='/node_data/joon/wandb_logs',
    )
    
    # Training loop
    total_steps = config.training.total_steps
    grad_accum = config.training.get('grad_accum_steps', 1)
    
    model.train()
    step = 0
    epoch = 0
    
    print(f'[Training] Starting for {total_steps} steps')
    
    while step < total_steps:
        epoch += 1
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_t, batch_t1 in pbar:
            # Move to device
            batch_t = edict({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_t.items()})
            batch_t1 = edict({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_t1.items()})
            
            # Forward with autocast
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                # Forward both frames
                result_t = model(batch_t, create_visual=False)
                result_t1 = model(batch_t1, create_visual=False)
                
                # Photometric loss (average of both frames)
                photo_loss = (result_t.loss_metrics.loss + result_t1.loss_metrics.loss) / 2
                
                # Temporal regularization
                xyz_t = result_t.gaussian_params_raw.xyz[0]   # [N, 3]
                xyz_t1 = result_t1.gaussian_params_raw.xyz[0]  # [N, 3]
                
                temporal_dict = temporal_loss_fn(xyz_t, xyz_t1, step)
                temporal_loss = temporal_dict['loss']
                
                # Combined loss
                total_loss = (photo_loss + temporal_loss) / grad_accum
            
            # Backward with scaler
            scaler.scale(total_loss).backward()
            
            # Optimizer step
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Logging
            log_dict = {
                'train/loss': total_loss.item() * grad_accum,
                'train/photo_loss': photo_loss.item(),
                'train/temporal_loss': temporal_loss.item(),
                'train/arap_loss': temporal_dict['arap_loss'].item(),
                'train/velocity_loss': temporal_dict['velocity_loss'].item(),
                'train/rampup': temporal_dict.get('rampup', 1.0),
                'train/psnr': (result_t.loss_metrics.psnr + result_t1.loss_metrics.psnr) / 2,
            }
            wandb.log(log_dict, step=step)
            
            pbar.set_postfix({
                'loss': f'{total_loss.item() * grad_accum:.4f}',
                'photo': f'{photo_loss.item():.4f}',
                'temporal': f'{temporal_loss.item():.6f}',
            })
            
            step += 1
            
            # Checkpoint
            if step % config.training.save_every == 0:
                ckpt_path = output_dir / f'checkpoint_{step:06d}.pt'
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'config': dict(config),
                }, ckpt_path)
                print(f'\n[Checkpoint] Saved to {ckpt_path}')
            
            if step >= total_steps:
                break
    
    # Final checkpoint
    final_path = output_dir / 'final.pt'
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': dict(config),
    }, final_path)
    print(f'[Done] Saved final checkpoint to {final_path}')
    
    wandb.finish()


if __name__ == '__main__':
    main()
