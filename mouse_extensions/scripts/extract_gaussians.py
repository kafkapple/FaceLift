#!/usr/bin/env python3
"""
Extract per-frame Gaussians from trained GS-LRM for deformation network training.

Usage:
    python -m mouse_extensions.scripts.extract_gaussians \
        --checkpoint /path/to/gslrm_checkpoint.pt \
        --data_list /path/to/data_mouse_train.txt \
        --output_dir /path/to/gaussians_output \
        --num_frames 100

This creates:
    output_dir/
        frame_0000.pt  # {xyz, features, scaling, rotation, opacity}
        frame_0001.pt
        ...
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gslrm_model(checkpoint_path: str, device: str = 'cuda'):
    """Load GS-LRM model from checkpoint."""
    from gslrm.model.gslrm import GSLRM
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    if 'config' in ckpt:
        config = ckpt['config']
    else:
        # Default config
        from omegaconf import OmegaConf
        config = OmegaConf.create({
            'model': {
                'class_name': 'gslrm.model.gslrm.GSLRM',
                'num_input_views': 4,
                'num_views': 6,
                'clip_xyz': True,
            }
        })
    
    # Create model
    model = GSLRM(config.model)
    
    # Load weights
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded: {model.__class__.__name__}")
    return model, config


def create_dataloader(data_list_path: str, config, num_frames: int = 0):
    """Create dataloader for extracting Gaussians."""
    from gslrm.data.dataset import MultiViewDataset
    from torch.utils.data import DataLoader
    
    # Read data list
    with open(data_list_path) as f:
        data_paths = [line.strip() for line in f if line.strip()]
    
    if num_frames > 0:
        data_paths = data_paths[:num_frames]
    
    logger.info(f"Loading {len(data_paths)} samples from {data_list_path}")
    
    # Create dataset
    dataset = MultiViewDataset(
        data_paths,
        num_input_views=config.model.num_input_views,
        num_views=config.model.num_views,
        background_color='white',
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )
    
    return dataloader


def extract_gaussians(model, batch, device: str = 'cuda'):
    """Extract Gaussian parameters from model output."""
    from mouse_extensions.model.deformation import GaussianParams
    
    # Move batch to device
    images = batch['images'].to(device)
    cameras = batch['cameras']
    
    # Forward pass
    with torch.no_grad():
        output = model(images, cameras)
    
    # Extract Gaussian parameters
    gaussian_params = output['gaussian_params']  # [B, N, D]
    
    # Parse into components (model-specific)
    # Typical GS-LRM output: xyz(3) + features(27) + scaling(3) + rotation(4) + opacity(1)
    if hasattr(model, 'parse_gaussians'):
        parsed = model.parse_gaussians(gaussian_params)
    else:
        # Manual parsing based on typical format
        idx = 0
        xyz = gaussian_params[..., idx:idx+3]; idx += 3
        features = gaussian_params[..., idx:idx+27]; idx += 27  # SH features
        scaling = gaussian_params[..., idx:idx+3]; idx += 3
        rotation = gaussian_params[..., idx:idx+4]; idx += 4
        opacity = gaussian_params[..., idx:idx+1]; idx += 1
        
        parsed = {
            'xyz': xyz[0],  # Remove batch dim
            'features': features[0],
            'scaling': scaling[0],
            'rotation': rotation[0],
            'opacity': opacity[0],
        }
    
    return GaussianParams(**parsed)


def main():
    parser = argparse.ArgumentParser(description="Extract Gaussians from GS-LRM")
    parser.add_argument("--checkpoint", type=str, required=True, help="GS-LRM checkpoint")
    parser.add_argument("--data_list", type=str, required=True, help="Data list file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_frames", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, config = load_gslrm_model(args.checkpoint, args.device)
    
    # Create dataloader
    dataloader = create_dataloader(args.data_list, config, args.num_frames)
    
    # Extract Gaussians
    logger.info(f"Extracting Gaussians to {output_dir}")
    
    for idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
        try:
            gaussians = extract_gaussians(model, batch, args.device)
            
            # Save
            save_path = output_dir / f"frame_{idx:04d}.pt"
            torch.save({
                'xyz': gaussians.xyz.cpu(),
                'features': gaussians.features.cpu() if gaussians.features is not None else None,
                'scaling': gaussians.scaling.cpu() if gaussians.scaling is not None else None,
                'rotation': gaussians.rotation.cpu() if gaussians.rotation is not None else None,
                'opacity': gaussians.opacity.cpu() if gaussians.opacity is not None else None,
            }, save_path)
            
        except Exception as e:
            logger.error(f"Error processing frame {idx}: {e}")
            continue
    
    logger.info(f"Extracted {idx + 1} frames to {output_dir}")


if __name__ == "__main__":
    main()
