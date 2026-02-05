#!/usr/bin/env python3
"""Compute PSNR/SSIM/LPIPS for H1 diagnosis experiments."""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Optional: LPIPS
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed, skipping LPIPS metric")


def load_image(path):
    """Load image as numpy array (H, W, C) in [0, 1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img) / 255.0


def compute_metrics(pred_path, gt_path, lpips_fn=None):
    """Compute PSNR, SSIM, and optionally LPIPS."""
    pred = load_image(pred_path)
    gt = load_image(gt_path)
    
    # Resize if needed
    if pred.shape != gt.shape:
        from PIL import Image
        gt_pil = Image.open(gt_path).convert('RGB')
        gt_pil = gt_pil.resize((pred.shape[1], pred.shape[0]), Image.LANCZOS)
        gt = np.array(gt_pil) / 255.0
    
    metrics = {
        'psnr': psnr(gt, pred, data_range=1.0),
        'ssim': ssim(gt, pred, data_range=1.0, channel_axis=2),
    }
    
    if lpips_fn is not None:
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            lpips_val = lpips_fn(pred_t, gt_t).item()
        metrics['lpips'] = lpips_val
    
    return metrics


def process_experiment(exp_dir, data_dir, split_file):
    """Process all samples in an experiment directory."""
    exp_dir = Path(exp_dir)
    samples_dir = exp_dir / 'samples'
    
    if not samples_dir.exists():
        print(f"No samples directory in {exp_dir}")
        return None
    
    # Load split to get GT paths
    with open(split_file, 'r') as f:
        sample_paths = [line.strip() for line in f if line.strip()]
    
    # Initialize LPIPS
    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net='alex')
    
    all_metrics = []
    sample_dirs = sorted(samples_dir.iterdir())
    
    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            continue
        
        sample_idx = int(sample_dir.name)
        if sample_idx >= len(sample_paths):
            continue
        
        gt_base = sample_paths[sample_idx]
        sample_metrics = {'sample': sample_dir.name, 'views': []}
        
        for view_idx in range(6):
            pred_path = sample_dir / f'render_view_{view_idx:02d}.png'
            gt_path = Path(gt_base) / f'image/{view_idx:03d}.png'
            
            if not pred_path.exists() or not gt_path.exists():
                continue
            
            metrics = compute_metrics(pred_path, gt_path, lpips_fn)
            metrics['view'] = view_idx
            sample_metrics['views'].append(metrics)
        
        if sample_metrics['views']:
            # Average across views
            sample_metrics['avg_psnr'] = np.mean([v['psnr'] for v in sample_metrics['views']])
            sample_metrics['avg_ssim'] = np.mean([v['ssim'] for v in sample_metrics['views']])
            if HAS_LPIPS:
                sample_metrics['avg_lpips'] = np.mean([v['lpips'] for v in sample_metrics['views']])
            all_metrics.append(sample_metrics)
    
    return all_metrics


def summarize_metrics(all_metrics):
    """Compute summary statistics."""
    psnrs = [m['avg_psnr'] for m in all_metrics]
    ssims = [m['avg_ssim'] for m in all_metrics]
    
    summary = {
        'num_samples': len(all_metrics),
        'psnr_mean': float(np.mean(psnrs)),
        'psnr_std': float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)),
        'ssim_std': float(np.std(ssims)),
    }
    
    if HAS_LPIPS and all_metrics and 'avg_lpips' in all_metrics[0]:
        lpips_vals = [m['avg_lpips'] for m in all_metrics]
        summary['lpips_mean'] = float(np.mean(lpips_vals))
        summary['lpips_std'] = float(np.std(lpips_vals))
    
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='Experiment output directory')
    parser.add_argument('--data_dir', default='/home/joon/data/preprocessed/FaceLift_mouse/M5')
    parser.add_argument('--split', required=True, help='Split file path')
    args = parser.parse_args()
    
    print(f"Processing {args.exp_dir}...")
    metrics = process_experiment(args.exp_dir, args.data_dir, args.split)
    
    if metrics:
        summary = summarize_metrics(metrics)
        print(f"\n=== Summary ===")
        print(f"Samples: {summary['num_samples']}")
        print(f"PSNR: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f}")
        print(f"SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
        if 'lpips_mean' in summary:
            print(f"LPIPS: {summary['lpips_mean']:.4f} ± {summary['lpips_std']:.4f}")
        
        # Save to JSON
        output_path = Path(args.exp_dir) / 'metrics_summary.json'
        with open(output_path, 'w') as f:
            json.dump({'summary': summary, 'per_sample': metrics}, f, indent=2)
        print(f"\nSaved to {output_path}")
    else:
        print("No metrics computed")
