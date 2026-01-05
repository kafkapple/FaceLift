#!/usr/bin/env python3
"""
Debug script for GS-LRM mask visualization.

This script visualizes:
1. GT images with auto-generated masks
2. Gaussian rendered images with generated masks
3. Comparison between GT mask and rendered mask

Key insight: Gaussian rendering produces RGB, not RGBA.
We generate mask for rendered images using the same threshold method as GT.

Usage:
    python scripts/debug_gslrm_mask_visualization.py \
        --config configs/mouse_gslrm_local_rtx3060.yaml \
        --checkpoint checkpoints/gslrm/ckpt_0000000000021125.pt \
        --num_samples 3 \
        --output_dir experiments/debug_mask
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from easydict import EasyDict as edict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> edict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)


def generate_mask_from_white_bg(image: torch.Tensor, threshold: float = 0.98) -> torch.Tensor:
    """
    Generate binary mask from image with white background.

    Args:
        image: [C, H, W] or [B, C, H, W] tensor in [0, 1] range
        threshold: Pixels with all RGB > threshold are background

    Returns:
        Binary mask [1, H, W] or [B, 1, H, W] where 1 = foreground
    """
    if image.dim() == 3:
        # [C, H, W] -> check if all channels > threshold
        is_background = (image > threshold).all(dim=0, keepdim=True)  # [1, H, W]
        mask = (~is_background).float()
    else:
        # [B, C, H, W]
        is_background = (image > threshold).all(dim=1, keepdim=True)  # [B, 1, H, W]
        mask = (~is_background).float()

    return mask


def visualize_gt_vs_rendered(
    gt_images: torch.Tensor,
    rendered_images: torch.Tensor,
    gt_masks: torch.Tensor,
    rendered_masks: torch.Tensor,
    view_indices: list,
    sample_idx: int,
    output_dir: str
):
    """
    Visualize GT vs Rendered images with their masks.

    Args:
        gt_images: [V, C, H, W] GT images (C=3 or 4)
        rendered_images: [V, 3, H, W] Rendered images
        gt_masks: [V, 1, H, W] GT masks
        rendered_masks: [V, 1, H, W] Rendered masks
        view_indices: List of view indices
        sample_idx: Sample index for filename
        output_dir: Output directory
    """
    num_views = gt_images.shape[0]

    fig, axes = plt.subplots(4, num_views, figsize=(num_views * 3, 12))

    row_titles = ['GT Image', 'GT Mask', 'Rendered Image', 'Rendered Mask']

    for v in range(num_views):
        # GT Image (RGB only)
        gt_rgb = gt_images[v, :3].permute(1, 2, 0).numpy()
        axes[0, v].imshow(gt_rgb.clip(0, 1))
        axes[0, v].set_title(f'View {view_indices[v]}: GT')
        axes[0, v].axis('off')

        # GT Mask
        gt_mask = gt_masks[v, 0].numpy()
        axes[1, v].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        fg_ratio = gt_mask.mean() * 100
        axes[1, v].set_title(f'GT Mask ({fg_ratio:.1f}% FG)')
        axes[1, v].axis('off')

        # Rendered Image
        rendered_rgb = rendered_images[v].permute(1, 2, 0).numpy()
        axes[2, v].imshow(rendered_rgb.clip(0, 1))
        axes[2, v].set_title(f'View {view_indices[v]}: Rendered')
        axes[2, v].axis('off')

        # Rendered Mask
        rendered_mask = rendered_masks[v, 0].numpy()
        axes[3, v].imshow(rendered_mask, cmap='gray', vmin=0, vmax=1)
        fg_ratio = rendered_mask.mean() * 100
        axes[3, v].set_title(f'Rendered Mask ({fg_ratio:.1f}% FG)')
        axes[3, v].axis('off')

    # Add row labels
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=12, rotation=90, labelpad=20, va='center')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'sample_{sample_idx:04d}_gt_vs_rendered.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def visualize_mask_difference(
    gt_masks: torch.Tensor,
    rendered_masks: torch.Tensor,
    view_indices: list,
    sample_idx: int,
    output_dir: str
):
    """
    Visualize difference between GT and rendered masks.

    Green = Both agree (correct)
    Red = GT foreground, Rendered background (missed)
    Blue = GT background, Rendered foreground (extra)
    """
    num_views = gt_masks.shape[0]

    fig, axes = plt.subplots(1, num_views, figsize=(num_views * 3, 3))
    if num_views == 1:
        axes = [axes]

    for v in range(num_views):
        gt_mask = gt_masks[v, 0].numpy() > 0.5
        rendered_mask = rendered_masks[v, 0].numpy() > 0.5

        # Create difference visualization
        diff_img = np.zeros((*gt_mask.shape, 3))

        # Both agree (green)
        both_fg = gt_mask & rendered_mask
        both_bg = (~gt_mask) & (~rendered_mask)
        diff_img[both_fg | both_bg] = [0.2, 0.8, 0.2]  # Green

        # GT foreground, Rendered background (red - missed)
        missed = gt_mask & (~rendered_mask)
        diff_img[missed] = [0.9, 0.2, 0.2]  # Red

        # GT background, Rendered foreground (blue - extra)
        extra = (~gt_mask) & rendered_mask
        diff_img[extra] = [0.2, 0.2, 0.9]  # Blue

        axes[v].imshow(diff_img)

        # Calculate IoU
        intersection = (gt_mask & rendered_mask).sum()
        union = (gt_mask | rendered_mask).sum()
        iou = intersection / max(union, 1)

        axes[v].set_title(f'View {view_indices[v]}: IoU={iou:.3f}')
        axes[v].axis('off')

    plt.suptitle('Mask Difference: Green=Agree, Red=Missed, Blue=Extra', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'sample_{sample_idx:04d}_mask_diff.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {save_path}")


def run_inference_and_visualize(config, model, dataloader, num_samples, output_dir, device):
    """Run inference and visualize masks."""
    model.eval()

    print("\n" + "="*70)
    print("RUNNING GS-LRM INFERENCE WITH MASK VISUALIZATION")
    print("="*70)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            print(f"\nSample {i}:")

            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Get GT images and masks
            gt_images = batch['image']  # [B, V, C, H, W]
            B, V, C, H, W = gt_images.shape

            # Extract or generate GT masks
            if C == 4:
                gt_masks = gt_images[:, :, 3:4, :, :]  # [B, V, 1, H, W]
                gt_rgb = gt_images[:, :, :3, :, :]
            else:
                gt_rgb = gt_images
                # Generate mask from white background
                threshold = config.mouse.get('mask_threshold', 250) / 255.0
                gt_masks = []
                for b in range(B):
                    view_masks = []
                    for v in range(V):
                        mask = generate_mask_from_white_bg(gt_images[b, v], threshold)
                        view_masks.append(mask)
                    gt_masks.append(torch.stack(view_masks))
                gt_masks = torch.stack(gt_masks)  # [B, V, 1, H, W]

            print(f"  GT images: {gt_rgb.shape}, GT masks: {gt_masks.shape}")

            # Run model inference
            try:
                result = model(batch, create_visual=False)
                rendered_images = result.render  # [B, V, 3, H, W]

                if rendered_images is None:
                    print("  WARNING: No rendered images returned")
                    continue

                print(f"  Rendered images: {rendered_images.shape}")

                # Generate masks for rendered images
                rendered_masks = []
                for b in range(B):
                    view_masks = []
                    for v in range(V):
                        mask = generate_mask_from_white_bg(
                            rendered_images[b, v],
                            threshold=0.98
                        )
                        view_masks.append(mask)
                    rendered_masks.append(torch.stack(view_masks))
                rendered_masks = torch.stack(rendered_masks)  # [B, V, 1, H, W]

                print(f"  Rendered masks: {rendered_masks.shape}")

                # Get view indices from batch
                if 'index' in batch:
                    view_indices = batch['index'][0, :, 0].cpu().numpy().tolist()
                else:
                    view_indices = list(range(V))

                # Visualize for each batch item
                for b in range(B):
                    visualize_gt_vs_rendered(
                        gt_rgb[b].cpu(),
                        rendered_images[b].cpu(),
                        gt_masks[b].cpu(),
                        rendered_masks[b].cpu(),
                        view_indices,
                        i * B + b,
                        output_dir
                    )

                    visualize_mask_difference(
                        gt_masks[b].cpu(),
                        rendered_masks[b].cpu(),
                        view_indices,
                        i * B + b,
                        output_dir
                    )

            except Exception as e:
                print(f"  ERROR during inference: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Debug GS-LRM mask visualization')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='experiments/debug_mask',
                        help='Output directory for visualizations')
    parser.add_argument('--data_only', action='store_true',
                        help='Only visualize data loading (no model inference)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.data_only:
        # Just visualize data loading
        from gslrm.data.mouse_dataset import MouseViewDataset
        from torch.utils.data import DataLoader

        # Disable augmentation for visualization
        config.mouse.augmentation.enabled = False
        dataset = MouseViewDataset(config=config, split="train")

        print(f"Dataset loaded: {len(dataset)} samples")

        os.makedirs(args.output_dir, exist_ok=True)

        for i in range(min(args.num_samples, len(dataset))):
            sample = dataset[i]
            images = sample['image']  # [V, C, H, W]

            V, C, H, W = images.shape
            print(f"\nSample {i}: {images.shape}, channels={C}")

            if C == 4:
                gt_masks = images[:, 3:4, :, :]  # [V, 1, H, W]
                # Debug: check mask values
                print(f"  Mask from alpha: min={gt_masks.min():.3f}, max={gt_masks.max():.3f}, mean={gt_masks.mean():.3f}")
            else:
                threshold = config.mouse.get('mask_threshold', 250) / 255.0
                gt_masks = []
                for v in range(V):
                    mask = generate_mask_from_white_bg(images[v, :3], threshold)
                    gt_masks.append(mask)
                gt_masks = torch.stack(gt_masks)
                print(f"  Mask generated: min={gt_masks.min():.3f}, max={gt_masks.max():.3f}, mean={gt_masks.mean():.3f}")

            # Get view indices
            if 'index' in sample:
                view_indices = sample['index'][:, 0].numpy().tolist()
            else:
                view_indices = list(range(V))

            # Visualize GT only
            fig, axes = plt.subplots(2, V, figsize=(V * 3, 6))

            for v in range(V):
                # RGB
                rgb = images[v, :3].permute(1, 2, 0).numpy()
                axes[0, v].imshow(rgb.clip(0, 1))
                axes[0, v].set_title(f'View {view_indices[v]}: RGB')
                axes[0, v].axis('off')

                # Mask
                mask = gt_masks[v, 0].numpy()
                axes[1, v].imshow(mask, cmap='gray', vmin=0, vmax=1)
                fg_ratio = mask.mean() * 100
                axes[1, v].set_title(f'Mask ({fg_ratio:.1f}% FG)')
                axes[1, v].axis('off')

            plt.tight_layout()
            save_path = os.path.join(args.output_dir, f'sample_{i:04d}_data_only.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_path}")

        return

    # Full inference with model
    print("\nLoading model for full inference...")

    # Import model and dataset
    from gslrm.model.gslrm import GSLRM
    from gslrm.data.mouse_dataset import MouseViewDataset
    from torch.utils.data import DataLoader

    # Create dataset
    dataset = MouseViewDataset(config=config, split="train")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Load model
    model = GSLRM(config)

    checkpoint_path = args.checkpoint or config.training.checkpointing.get('resume_ckpt')
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print("WARNING: No checkpoint loaded, using random weights")

    model = model.to(device)

    # Run visualization
    run_inference_and_visualize(
        config, model, dataloader,
        args.num_samples, args.output_dir, device
    )


if __name__ == '__main__':
    main()
