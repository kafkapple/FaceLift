#!/usr/bin/env python3
"""
Debug script to visualize mask and image loading for GS-LRM training.

This script helps verify:
1. Images are loaded correctly with alpha channel (if present)
2. Mask is properly extracted from alpha channel
3. Pixel values are in expected range [0, 1]
4. Background color matches between GT and rendering

Usage:
    python scripts/debug_mask_visualization.py --config configs/mouse_gslrm_local_rtx3060.yaml
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import yaml
from easydict import EasyDict as edict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gslrm.data.mouse_dataset import MouseViewDataset


def load_config(config_path: str) -> edict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def visualize_sample(dataset, idx: int, output_dir: str):
    """Visualize a single sample's images and masks."""
    sample = dataset[idx]

    images = sample['image']  # [num_views, C, H, W]
    num_views = images.shape[0]
    num_channels = images.shape[1]

    print(f"\n{'='*60}")
    print(f"Sample {idx}")
    print(f"{'='*60}")
    print(f"  Image shape: {images.shape}")
    print(f"  Num channels: {num_channels} ({'RGBA' if num_channels == 4 else 'RGB'})")
    print(f"  Value range: [{images.min():.4f}, {images.max():.4f}]")
    print(f"  Camera c2w shape: {sample['c2w'].shape}")
    print(f"  Background color: {sample['bg_color']}")

    # Create figure
    if num_channels == 4:
        fig, axes = plt.subplots(3, num_views, figsize=(num_views * 3, 9))
        titles = ['RGB Image', 'Alpha (Mask)', 'Masked RGB']
    else:
        fig, axes = plt.subplots(1, num_views, figsize=(num_views * 3, 3))
        axes = axes.reshape(1, -1)
        titles = ['RGB Image']

    for v in range(num_views):
        img = images[v].numpy()

        # RGB image
        rgb = img[:3].transpose(1, 2, 0)  # [H, W, 3]
        if len(axes.shape) == 1:
            ax = axes[v]
        else:
            ax = axes[0, v]
        ax.imshow(rgb.clip(0, 1))
        ax.set_title(f'View {v}: RGB')
        ax.axis('off')

        if num_channels == 4:
            # Alpha channel (mask)
            alpha = img[3]  # [H, W]
            axes[1, v].imshow(alpha, cmap='gray', vmin=0, vmax=1)
            axes[1, v].set_title(f'View {v}: Alpha')
            axes[1, v].axis('off')

            # Masked RGB (overlay)
            mask_binary = (alpha > 0.5).astype(float)
            masked_rgb = rgb * mask_binary[:, :, None]
            axes[2, v].imshow(masked_rgb.clip(0, 1))
            axes[2, v].set_title(f'View {v}: Masked')
            axes[2, v].axis('off')

            # Print mask statistics
            fg_ratio = mask_binary.mean() * 100
            print(f"  View {v}: Foreground ratio = {fg_ratio:.1f}%")

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'sample_{idx:04d}_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")

    return {
        'num_channels': num_channels,
        'has_mask': num_channels == 4,
        'value_range': (images.min().item(), images.max().item()),
    }


def analyze_dataset(dataset, num_samples: int = 5, output_dir: str = 'experiments/debug'):
    """Analyze multiple samples from the dataset."""
    print("\n" + "="*70)
    print("MOUSE DATASET MASK ANALYSIS")
    print("="*70)

    results = []
    for i in range(min(num_samples, len(dataset))):
        result = visualize_sample(dataset, i, output_dir)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    has_mask_count = sum(1 for r in results if r['has_mask'])
    print(f"  Samples with mask (4 channels): {has_mask_count}/{len(results)}")

    if has_mask_count == 0:
        print("\n  WARNING: No samples have alpha channel!")
        print("  This means masked_l2_loss and masked_ssim_loss will have no effect.")
        print("  Consider:")
        print("    1. Check if source images have alpha channel")
        print("    2. Set remove_alpha: false in config")
        print("    3. Use background subtraction/segmentation preprocessing")

    value_ranges = [r['value_range'] for r in results]
    min_val = min(v[0] for v in value_ranges)
    max_val = max(v[1] for v in value_ranges)
    print(f"  Overall value range: [{min_val:.4f}, {max_val:.4f}]")

    if max_val > 1.0 or min_val < 0.0:
        print("\n  WARNING: Values outside [0, 1] range!")
        print("  This may cause issues with loss computation.")


def main():
    parser = argparse.ArgumentParser(description='Debug mask visualization for GS-LRM')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='experiments/debug',
                        help='Output directory for visualizations')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    # Create dataset
    dataset = MouseViewDataset(
        config=config,
        data_path=config.training.dataset.dataset_path,
        num_views=config.training.dataset.num_views,
        num_input_views=config.training.dataset.num_input_views,
        target_has_input=config.training.dataset.target_has_input,
        remove_alpha=config.training.dataset.get('remove_alpha', False),
        bg_color=config.training.dataset.get('background_color', 'white'),
        normalize_cameras=config.mouse.get('normalize_cameras', False),
        target_camera_distance=config.mouse.get('target_camera_distance', 0.0),
        normalize_to_z_up=config.mouse.get('normalize_to_z_up', False),
        use_augmentation=False,
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Config settings:")
    print(f"  remove_alpha: {config.training.dataset.get('remove_alpha', False)}")
    print(f"  background_color: {config.training.dataset.get('background_color', 'white')}")
    print(f"  masked_l2_loss: {config.training.losses.get('masked_l2_loss', False)}")
    print(f"  masked_ssim_loss: {config.training.losses.get('masked_ssim_loss', False)}")

    # Analyze dataset
    analyze_dataset(dataset, args.num_samples, args.output_dir)

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Check the generated visualization images in:", args.output_dir)
    print("2. Verify masks properly segment the mouse from background")
    print("3. If no masks, consider preprocessing to add alpha channel")
    print("4. Run training with masked losses enabled")


if __name__ == '__main__':
    main()
