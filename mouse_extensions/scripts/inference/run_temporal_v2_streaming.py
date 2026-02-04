#!/usr/bin/env python3
"""Temporal Inference V2 (Streaming): Memory-efficient per-frame processing."""

import argparse
import json
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm

from mouse_extensions.model.deformation.gaussian_params import GaussianParams


def bilateral_smooth_streaming(
    input_dir: str,
    output_dir: str,
    window_size: int = 5,
    blend_alpha: float = 0.3,
    sigma_temporal: float = 1.0,
    sigma_spatial: float = 0.1,
    device: str = 'cuda',
):
    """
    Streaming bilateral smoothing - processes one frame at a time.
    Only keeps window_size frames in memory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pt_files = sorted(input_path.glob('*.pt'))
    T = len(pt_files)
    half_window = window_size // 2
    
    print(f'Processing {T} frames with window={window_size}, alpha={blend_alpha}')
    
    # Statistics
    drifts = []
    
    for t in tqdm(range(T), desc='Smoothing'):
        # Load current frame
        current_data = torch.load(pt_files[t], weights_only=True)
        current_xyz = current_data['xyz'].to(device)
        
        # Determine window
        t_start = max(0, t - half_window)
        t_end = min(T, t + half_window + 1)
        
        # Bilateral smoothing
        weighted_sum = torch.zeros_like(current_xyz)
        weight_sum = torch.zeros(current_xyz.shape[0], 1, device=device)
        
        for ti in range(t_start, t_end):
            # Load neighbor frame
            neighbor_data = torch.load(pt_files[ti], weights_only=True)
            neighbor_xyz = neighbor_data['xyz'].to(device)
            
            # Temporal weight
            dt = abs(ti - t)
            w_temporal = torch.exp(torch.tensor(-dt**2 / (2 * sigma_temporal**2), device=device))
            
            # Spatial/value weight
            diff = neighbor_xyz - current_xyz
            dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
            w_spatial = torch.exp(-dist_sq / (2 * sigma_spatial**2))
            
            # Combined weight
            w = w_temporal * w_spatial
            
            weighted_sum = weighted_sum + w * neighbor_xyz
            weight_sum = weight_sum + w
            
            del neighbor_data, neighbor_xyz
        
        # Normalize
        smoothed_xyz = weighted_sum / (weight_sum + 1e-8)
        
        # Blend with original
        final_xyz = (1 - blend_alpha) * current_xyz + blend_alpha * smoothed_xyz
        
        # Compute drift
        drift = (current_xyz - final_xyz).norm(dim=-1).mean().item()
        drifts.append(drift)
        
        # Save
        save_data = {
            'xyz': final_xyz.cpu(),
            'features': current_data['features'],
            'scaling': current_data['scaling'],
            'rotation': current_data['rotation'],
            'opacity': current_data['opacity'],
        }
        torch.save(save_data, output_path / f'frame_{t:06d}.pt')
        
        # Clear GPU
        del current_data, current_xyz, smoothed_xyz, final_xyz
        torch.cuda.empty_cache()
    
    # Save config
    config = {
        'method': 'bilateral_streaming',
        'window_size': window_size,
        'blend_alpha': blend_alpha,
        'sigma_temporal': sigma_temporal,
        'sigma_spatial': sigma_spatial,
        'num_frames': T,
        'drift_mean': sum(drifts) / len(drifts),
        'drift_max': max(drifts),
        'drift_min': min(drifts),
    }
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f'\nDrift statistics:')
    print(f'  Mean: {config["drift_mean"]:.4f}')
    print(f'  Max: {config["drift_max"]:.4f}')
    print(f'  Min: {config["drift_min"]:.4f}')
    print(f'\nSaved {T} frames to {output_dir}')
    
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--blend_alpha', type=float, default=0.3)
    parser.add_argument('--sigma_temporal', type=float, default=1.0)
    parser.add_argument('--sigma_spatial', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    bilateral_smooth_streaming(
        input_dir=args.input,
        output_dir=args.output,
        window_size=args.window_size,
        blend_alpha=args.blend_alpha,
        sigma_temporal=args.sigma_temporal,
        sigma_spatial=args.sigma_spatial,
        device=args.device,
    )
