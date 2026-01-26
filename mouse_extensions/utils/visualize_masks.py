#!/usr/bin/env python3
"""
Visualize alpha-based masks at different thresholds and compare with RGB-based masks.

Usage:
    python scripts/visualize_alpha_masks.py --checkpoint checkpoints/mouse_gslrm_v25/ckpt_*.pt --output experiments/mask_comparison/

This script:
1. Loads a trained model checkpoint
2. Renders images with alpha channel
3. Creates masks at different alpha thresholds (0.1, 0.3, 0.5, 0.7, 0.9)
4. Compares with RGB-based threshold masks
5. Saves comparison visualizations
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gslrm.model.gslrm import GSLRM
from mouse_extensions.data.mouse_dataset import MouseDataset
from omegaconf import OmegaConf


def create_mask_from_alpha(alpha: torch.Tensor, threshold: float) -> torch.Tensor:
    """Create binary mask from alpha channel."""
    return (alpha > threshold).float()


def create_mask_from_rgb(image: torch.Tensor, bg_color: tuple = (1.0, 1.0, 1.0), threshold: float = 0.1) -> torch.Tensor:
    """Create binary mask from RGB distance to background color."""
    bg = torch.tensor(bg_color, device=image.device).view(3, 1, 1)
    distance = ((image - bg) ** 2).sum(dim=0, keepdim=True).sqrt()
    return (distance > threshold).float()


def visualize_masks(
    image: np.ndarray,
    alpha: np.ndarray,
    gt_mask: np.ndarray,
    output_path: str,
    sample_id: str,
):
    """Create comparison visualization of different masking methods."""
    alpha_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    rgb_thresholds = [0.05, 0.1, 0.2, 0.3]
    
    n_rows = 3
    n_cols = max(len(alpha_thresholds), len(rgb_thresholds)) + 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f'Mask Comparison - {sample_id}', fontsize=16)
    
    # Row 1: Original image and alpha channel
    axes[0, 0].imshow(image.transpose(1, 2, 0))
    axes[0, 0].set_title('Rendered Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(alpha.squeeze(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Alpha Channel')
    axes[0, 1].axis('off')
    
    # Alpha-based masks
    for i, thresh in enumerate(alpha_thresholds):
        mask = (alpha > thresh).astype(float)
        axes[0, i + 2].imshow(mask.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0, i + 2].set_title(f'Alpha > {thresh}')
        axes[0, i + 2].axis('off')
    
    # Row 2: GT mask and RGB-based masks
    if gt_mask is not None:
        axes[1, 0].imshow(gt_mask.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title('GT Mask')
    else:
        axes[1, 0].text(0.5, 0.5, 'No GT', ha='center', va='center')
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, 'RGB Masks\n(below)', ha='center', va='center', fontsize=12)
    axes[1, 1].axis('off')
    
    # RGB-based masks
    bg_color = np.array([1.0, 1.0, 1.0]).reshape(3, 1, 1)
    rgb_distance = np.sqrt(((image - bg_color) ** 2).sum(axis=0, keepdims=True))
    
    for i, thresh in enumerate(rgb_thresholds):
        mask = (rgb_distance > thresh).astype(float)
        axes[1, i + 2].imshow(mask.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[1, i + 2].set_title(f'RGB dist > {thresh}')
        axes[1, i + 2].axis('off')
    
    # Hide unused axes in row 2
    for i in range(len(rgb_thresholds) + 2, n_cols):
        axes[1, i].axis('off')
    
    # Row 3: Overlay comparison (best alpha vs best RGB vs GT)
    best_alpha_thresh = 0.5
    best_rgb_thresh = 0.1
    
    alpha_mask = (alpha > best_alpha_thresh).squeeze()
    rgb_mask = (rgb_distance > best_rgb_thresh).squeeze()
    
    # Create RGB overlay images
    overlay_alpha = image.copy().transpose(1, 2, 0)
    overlay_alpha[alpha_mask == 0] = overlay_alpha[alpha_mask == 0] * 0.3 + np.array([0, 0, 1]) * 0.7
    
    overlay_rgb = image.copy().transpose(1, 2, 0)
    overlay_rgb[rgb_mask == 0] = overlay_rgb[rgb_mask == 0] * 0.3 + np.array([0, 1, 0]) * 0.7
    
    axes[2, 0].imshow(overlay_alpha)
    axes[2, 0].set_title(f'Alpha > {best_alpha_thresh} (blue=bg)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(overlay_rgb)
    axes[2, 1].set_title(f'RGB > {best_rgb_thresh} (green=bg)')
    axes[2, 1].axis('off')
    
    # Difference between alpha and RGB masks
    diff = alpha_mask.astype(float) - rgb_mask.astype(float)
    axes[2, 2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
    axes[2, 2].set_title('Diff (Alpha - RGB)')
    axes[2, 2].axis('off')
    
    # IoU calculation
    if gt_mask is not None:
        gt_binary = gt_mask.squeeze() > 0.5
        alpha_iou = compute_iou(alpha_mask, gt_binary)
        rgb_iou = compute_iou(rgb_mask, gt_binary)
        
        axes[2, 3].text(0.5, 0.5, f'IoU vs GT:\nAlpha: {alpha_iou:.3f}\nRGB: {rgb_iou:.3f}', 
                       ha='center', va='center', fontsize=14)
    else:
        axes[2, 3].text(0.5, 0.5, 'No GT for IoU', ha='center', va='center', fontsize=12)
    axes[2, 3].axis('off')
    
    # Hide remaining axes
    for i in range(4, n_cols):
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / max(union, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/mouse/gslrm_v25_original_ratio.yaml')
    parser.add_argument('--output', type=str, default='experiments/mask_comparison/')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load config
    config = OmegaConf.load(args.config)
    
    print(f'Loading checkpoint: {args.checkpoint}')
    print(f'Output directory: {args.output}')
    
    # This is a placeholder - actual implementation would load model and dataset
    # For now, we'll create dummy data to test the visualization
    print('\nNote: This script requires a trained model and dataset.')
    print('Full implementation would:')
    print('1. Load the GSLRM model from checkpoint')
    print('2. Load validation dataset')
    print('3. Render images with alpha channel')
    print('4. Create mask comparison visualizations')
    print('\nTo use, ensure render_opencv_cam returns "alpha" key.')
    
    # Create example visualization with dummy data
    print('\nCreating example visualization with random data...')
    
    for i in range(min(args.num_samples, 3)):
        # Dummy data for demonstration
        h, w = 512, 512
        image = np.random.rand(3, h, w).astype(np.float32)
        image = image * 0.3 + 0.7  # Mostly white
        
        # Create a circular mouse region
        y, x = np.ogrid[:h, :w]
        center = (h//2 + np.random.randint(-50, 50), w//2 + np.random.randint(-50, 50))
        radius = 100 + np.random.randint(-30, 30)
        mouse_mask = ((y - center[0])**2 + (x - center[1])**2 < radius**2)
        
        # Add mouse color
        image[:, mouse_mask] = np.array([0.4, 0.35, 0.3])[:, None]
        
        # Create alpha (similar to mask but with gradient edges)
        alpha = np.zeros((1, h, w), dtype=np.float32)
        dist = np.sqrt((y - center[0])**2 + (x - center[1])**2)
        alpha[0] = np.clip(1 - (dist - radius + 20) / 40, 0, 1)
        
        gt_mask = mouse_mask.astype(np.float32)[None, :, :]
        
        output_path = os.path.join(args.output, f'mask_comparison_sample_{i:03d}.png')
        visualize_masks(image, alpha, gt_mask, output_path, f'sample_{i:03d}')
    
    print(f'\nDone. Check {args.output} for visualizations.')


if __name__ == '__main__':
    main()
