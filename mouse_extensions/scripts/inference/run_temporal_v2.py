#!/usr/bin/env python3
"""Temporal Inference V2: Per-frame + Regularization (no autoregressive drift)."""

import argparse
from pathlib import Path
from typing import List, Optional
import torch
from tqdm import tqdm

from mouse_extensions.model.deformation.temporal_pipeline_v2 import (
    TemporalPipelineV2, PipelineV2Config
)
from mouse_extensions.model.deformation.temporal_regularization import TemporalRegConfig
from mouse_extensions.model.deformation.gaussian_params import GaussianParams


def load_gaussians_from_cache(cache_dir: str, max_frames: Optional[int] = None) -> List[GaussianParams]:
    """Load pre-computed Gaussians from .pt files."""
    cache_path = Path(cache_dir)
    pt_files = sorted(cache_path.glob('*.pt'))
    
    if max_frames:
        pt_files = pt_files[:max_frames]
    
    print(f'Loading {len(pt_files)} frames from {cache_dir}')
    
    gaussians = []
    for pt_file in tqdm(pt_files, desc='Loading'):
        data = torch.load(pt_file, weights_only=True)
        g = GaussianParams(
            xyz=data['xyz'],
            features=data['features'],
            scaling=data['scaling'],
            rotation=data['rotation'],
            opacity=data['opacity'],
        )
        gaussians.append(g)
    
    return gaussians


def save_gaussians(gaussians: List[GaussianParams], output_dir: str):
    """Save Gaussians to .pt files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, g in enumerate(tqdm(gaussians, desc='Saving')):
        save_path = output_path / f'frame_{i:06d}.pt'
        torch.save({
            'xyz': g.xyz.cpu(),
            'features': g.features.cpu(),
            'scaling': g.scaling.cpu(),
            'rotation': g.rotation.cpu(),
            'opacity': g.opacity.cpu(),
        }, save_path)


def run_v2_inference(
    input_dir: str,
    output_dir: str,
    smoothing_method: str = 'bilateral',
    blend_alpha: float = 0.3,
    window_size: int = 5,
    max_frames: Optional[int] = None,
    device: str = 'cuda',
):
    """Run V2 temporal inference."""
    print('=' * 60)
    print('Temporal Inference V2: Per-frame + Regularization')
    print('=' * 60)
    print(f'  Input: {input_dir}')
    print(f'  Output: {output_dir}')
    print(f'  Method: {smoothing_method}')
    print(f'  Blend alpha: {blend_alpha}')
    print(f'  Window size: {window_size}')
    
    # Load Gaussians
    gaussians = load_gaussians_from_cache(input_dir, max_frames)
    
    # Move to device
    for g in gaussians:
        g.to(device)
    
    # Create V2 pipeline
    config = PipelineV2Config(
        smoothing_method=smoothing_method,
        blend_alpha=blend_alpha,
        window_size=window_size,
        reg_config=TemporalRegConfig(
            arap_weight=0.1,
            velocity_weight=0.01,
        ),
    )
    pipeline = TemporalPipelineV2(config)
    
    # Run smoothing
    print(f'\nApplying {smoothing_method} smoothing...')
    smoothed, losses = pipeline(gaussians, return_losses=True)
    
    # Print losses
    print(f'\nRegularization losses:')
    for k, v in losses.items():
        print(f'  {k}: {v.item():.6f}')
    
    # Compute drift statistics
    drifts = []
    for orig, smooth in zip(gaussians, smoothed):
        drift = (orig.xyz - smooth.xyz).norm(dim=-1).mean().item()
        drifts.append(drift)
    
    print(f'\nDrift statistics:')
    print(f'  Mean: {sum(drifts)/len(drifts):.4f}')
    print(f'  Max: {max(drifts):.4f}')
    print(f'  Min: {min(drifts):.4f}')
    
    # Save
    save_gaussians(smoothed, output_dir)
    
    # Save config
    import json
    config_path = Path(output_dir) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'smoothing_method': smoothing_method,
            'blend_alpha': blend_alpha,
            'window_size': window_size,
            'num_frames': len(smoothed),
            'losses': {k: v.item() for k, v in losses.items()},
            'drift_mean': sum(drifts)/len(drifts),
            'drift_max': max(drifts),
        }, f, indent=2)
    
    print(f'\nSaved {len(smoothed)} frames to {output_dir}')
    return smoothed, losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input cache directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--method', type=str, default='bilateral', choices=['bilateral', 'gaussian', 'none'])
    parser.add_argument('--blend_alpha', type=float, default=0.3)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    run_v2_inference(
        input_dir=args.input,
        output_dir=args.output,
        smoothing_method=args.method,
        blend_alpha=args.blend_alpha,
        window_size=args.window_size,
        max_frames=args.max_frames,
        device=args.device,
    )
