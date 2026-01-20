#!/usr/bin/env python3
"""
Alpha Parameter Sweep - Quick Validation Script
Runs short training with various (threshold, alpha_loss) combinations
and compares mask visualizations.

Usage:
    python scripts/alpha_sweep.py --gpu 0 --steps 200 --dataset D7_1
    python scripts/alpha_sweep.py --gpu 0,1,2,3 --steps 200 --parallel
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from itertools import product
import json

# Parameter grid
THRESHOLDS = [0.3, 0.5, 0.7]
ALPHA_LOSSES = [0.0, 0.1, 0.3, 0.5]

def create_sweep_config(base_dir, threshold, alpha_loss, dataset="D7_1"):
    """Create a temporary config for this parameter combination."""
    config_name = f"sweep_t{threshold}_a{alpha_loss}".replace(".", "")
    config_path = Path(base_dir) / "configs" / "sweep" / f"{config_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_content = f"""# Auto-generated sweep config: threshold={threshold}, alpha_loss={alpha_loss}
_name: {config_name}

training:
  dataset:
    random_view_selection: false
    num_views: 6
    num_input_views: 5
  losses:
    masked_pixelalign_loss: true
    masked_l2_loss: true
    masked_ssim_loss: true
    background_loss_weight: 0.0
    mask_mode: alpha
    alpha_mask_threshold: {threshold}
    alpha_loss_weight: {alpha_loss}
    alpha_loss_type: bce
  schedule:
    max_fwdbwd_passes: {{steps}}  # Will be replaced
  checkpointing:
    checkpoint_dir: checkpoints/sweep/{config_name}
    checkpoint_every: 50
  logging:
    vis_every: 50
    wandb:
      project: FaceLift-Sweep
      group: alpha_sweep
      exp_name: {config_name}
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path, config_name

def run_sweep_experiment(base_dir, dataset, threshold, alpha_loss, gpu, steps):
    """Run a single sweep experiment."""
    config_path, config_name = create_sweep_config(base_dir, threshold, alpha_loss, dataset)
    
    # Update steps in config
    with open(config_path, 'r') as f:
        content = f.read()
    content = content.replace('{steps}', str(steps))
    with open(config_path, 'w') as f:
        f.write(content)
    
    log_path = Path(base_dir) / "logs" / "sweep" / f"{config_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = f"""cd {base_dir} && \
CUDA_VISIBLE_DEVICES={gpu} /home/joon/anaconda3/envs/facelift/bin/torchrun \
--standalone --nproc_per_node=1 train_gslrm.py \
-d {dataset} --config {config_path} \
2>&1 | tee {log_path}"""
    
    print(f"[GPU {gpu}] Starting: threshold={threshold}, alpha_loss={alpha_loss}")
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def run_sequential(base_dir, dataset, gpu, steps):
    """Run all combinations sequentially on one GPU."""
    results = []
    for threshold, alpha_loss in product(THRESHOLDS, ALPHA_LOSSES):
        print(f"\n{'='*60}")
        print(f"Running: threshold={threshold}, alpha_loss={alpha_loss}")
        print(f"{'='*60}")
        
        proc = run_sweep_experiment(base_dir, dataset, threshold, alpha_loss, gpu, steps)
        proc.wait()
        
        results.append({
            'threshold': threshold,
            'alpha_loss': alpha_loss,
            'gpu': gpu,
            'status': 'completed' if proc.returncode == 0 else 'failed'
        })
    
    return results

def run_parallel(base_dir, dataset, gpus, steps):
    """Run combinations in parallel across multiple GPUs."""
    gpu_list = [int(g.strip()) for g in gpus.split(',')]
    combinations = list(product(THRESHOLDS, ALPHA_LOSSES))
    
    processes = []
    results = []
    
    # Queue combinations to GPUs
    for i, (threshold, alpha_loss) in enumerate(combinations):
        gpu = gpu_list[i % len(gpu_list)]
        proc = run_sweep_experiment(base_dir, dataset, threshold, alpha_loss, gpu, steps)
        processes.append({
            'proc': proc,
            'threshold': threshold,
            'alpha_loss': alpha_loss,
            'gpu': gpu
        })
        time.sleep(2)  # Stagger starts
    
    # Wait for all to complete
    for p in processes:
        p['proc'].wait()
        results.append({
            'threshold': p['threshold'],
            'alpha_loss': p['alpha_loss'],
            'gpu': p['gpu'],
            'status': 'completed' if p['proc'].returncode == 0 else 'failed'
        })
    
    return results

def collect_visualizations(base_dir):
    """Collect supervision images from all sweep experiments."""
    sweep_dir = Path(base_dir) / "checkpoints" / "sweep"
    vis_data = []
    
    for exp_dir in sweep_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('sweep_'):
            # Parse parameters from name
            parts = exp_dir.name.replace('sweep_t', '').split('_a')
            if len(parts) == 2:
                threshold = float(parts[0]) / 10 if len(parts[0]) == 1 else float(parts[0][:1] + '.' + parts[0][1:])
                alpha_loss = float(parts[1]) / 10 if len(parts[1]) == 1 else float(parts[1][:1] + '.' + parts[1][1:])
                
                # Find latest supervision image
                vis_files = list(exp_dir.glob('**/supervision_*.jpg')) + list(exp_dir.glob('**/supervision_*.png'))
                if vis_files:
                    latest = max(vis_files, key=lambda x: x.stat().st_mtime)
                    vis_data.append({
                        'threshold': threshold,
                        'alpha_loss': alpha_loss,
                        'image_path': str(latest)
                    })
    
    return vis_data

def create_comparison_grid(base_dir, vis_data):
    """Create a comparison grid image."""
    try:
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: PIL/matplotlib not available for grid creation")
        return None
    
    # Create grid
    fig, axes = plt.subplots(len(ALPHA_LOSSES), len(THRESHOLDS), figsize=(15, 20))
    fig.suptitle('Alpha Sweep: Mask Visualization Comparison', fontsize=16)
    
    for data in vis_data:
        t_idx = THRESHOLDS.index(data['threshold']) if data['threshold'] in THRESHOLDS else -1
        a_idx = ALPHA_LOSSES.index(data['alpha_loss']) if data['alpha_loss'] in ALPHA_LOSSES else -1
        
        if t_idx >= 0 and a_idx >= 0:
            img = Image.open(data['image_path'])
            axes[a_idx, t_idx].imshow(img)
            axes[a_idx, t_idx].set_title(f't={data["threshold"]}, a={data["alpha_loss"]}')
            axes[a_idx, t_idx].axis('off')
    
    # Labels
    for i, a in enumerate(ALPHA_LOSSES):
        axes[i, 0].set_ylabel(f'alpha_loss={a}', fontsize=10)
    for j, t in enumerate(THRESHOLDS):
        axes[0, j].set_xlabel(f'threshold={t}', fontsize=10)
    
    output_path = Path(base_dir) / "experiments" / "sweep_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison grid saved to: {output_path}")
    return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='Alpha Parameter Sweep')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID(s), comma-separated for parallel')
    parser.add_argument('--steps', type=int, default=200, help='Training steps per experiment')
    parser.add_argument('--dataset', type=str, default='D7_1', help='Dataset to use')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel across GPUs')
    parser.add_argument('--collect-only', action='store_true', help='Only collect and visualize results')
    parser.add_argument('--base-dir', type=str, default='/home/joon/dev/FaceLift', help='Base directory')
    
    args = parser.parse_args()
    
    if args.collect_only:
        vis_data = collect_visualizations(args.base_dir)
        print(f"Found {len(vis_data)} visualization files")
        create_comparison_grid(args.base_dir, vis_data)
        return
    
    print(f"Alpha Sweep Configuration:")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Alpha losses: {ALPHA_LOSSES}")
    print(f"  Total combinations: {len(THRESHOLDS) * len(ALPHA_LOSSES)}")
    print(f"  Steps per experiment: {args.steps}")
    print(f"  GPUs: {args.gpu}")
    print(f"  Mode: {'parallel' if args.parallel else 'sequential'}")
    print()
    
    if args.parallel:
        results = run_parallel(args.base_dir, args.dataset, args.gpu, args.steps)
    else:
        results = run_sequential(args.base_dir, args.dataset, args.gpu, args.steps)
    
    # Save results
    results_path = Path(args.base_dir) / "experiments" / "sweep_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Collect visualizations
    vis_data = collect_visualizations(args.base_dir)
    if vis_data:
        create_comparison_grid(args.base_dir, vis_data)

if __name__ == '__main__':
    main()
