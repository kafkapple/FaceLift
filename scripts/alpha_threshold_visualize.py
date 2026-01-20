#!/usr/bin/env python3
"""
Alpha Threshold Visualization - WandB Style
Compare masks at different thresholds with proper labeling.
"""

import os, sys
sys.path.insert(0, '/home/joon/dev/FaceLift')

import argparse, re, torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf
from easydict import EasyDict as edict

DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7]

# WandB style colors
FG_COLOR = np.array([0.0, 0.8, 0.0])  # Green for foreground
BG_COLOR = np.array([1.0, 0.7, 0.7])  # Pink for background


def load_checkpoint_and_model(ckpt_dir, device='cuda'):
    from gslrm.model.gslrm import GSLRM
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir.parent / 'config.yaml'
    ckpt_files = list(ckpt_dir.parent.glob('ckpt_*.pt'))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir.parent}")
    iter_match = re.search(r'iter_0*(\d+)', str(ckpt_dir))
    if iter_match:
        iter_num = int(iter_match.group(1))
        def get_step(p):
            m = re.search(r'ckpt_0*(\d+)\.pt', p.name)
            return int(m.group(1)) if m else 0
        valid = [c for c in ckpt_files if get_step(c) <= iter_num]
        if valid: ckpt_files = valid
    latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading: {latest_ckpt}")
    config = OmegaConf.load(config_path)
    model = GSLRM(config)
    state = torch.load(latest_ckpt, map_location='cpu')
    if 'model' in state: state = state['model']
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), config


def get_sample_batch(config, device='cuda', sample_idx=0):
    from gslrm.data.mouse_dataset import MouseViewDataset
    dataset = MouseViewDataset(config, split='val')
    sample = dataset[sample_idx]
    batch = edict()
    for k, v in sample.items():
        batch[k] = v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
    return batch


def mask_to_rgb(mask, fg_color=FG_COLOR, bg_color=BG_COLOR):
    """Convert binary mask to RGB with WandB colors."""
    mask_np = mask.squeeze().detach().cpu().numpy()
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3))
    rgb[mask_np > 0.5] = fg_color
    rgb[mask_np <= 0.5] = bg_color
    return rgb


def compute_iou(pred, gt):
    """Compute IoU between pred and gt masks."""
    pred = pred.float()
    gt = gt.float()
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter / (union + 1e-8)).item()


def create_diff_heatmap(pred, gt):
    """Create diff heatmap: TP=green, FP=red, FN=blue."""
    pred_np = pred.squeeze().detach().cpu().numpy()
    gt_np = gt.squeeze().detach().cpu().numpy()
    h, w = pred_np.shape
    diff = np.zeros((h, w, 3))
    # True Positive: green
    diff[..., 1] = (pred_np > 0.5) & (gt_np > 0.5)
    # False Positive: red
    diff[..., 0] = (pred_np > 0.5) & (gt_np <= 0.5)
    # False Negative: blue
    diff[..., 2] = (pred_np <= 0.5) & (gt_np > 0.5)
    return diff


def create_visualization(batch, outputs, thresholds, output_path, config_name, step):
    """Create WandB-style visualization with multiple thresholds."""
    
    # Extract data
    # Input image: (B, V, C, H, W)
    input_rgb = batch.image[0, 0, :3].float().detach().cpu()  # First view RGB
    gt_mask = batch.image[0, 0, 3].float().detach().cpu() if batch.image.shape[2] == 4 else None
    
    # Rendered outputs
    rendered_rgb = outputs['render'][0, 0].float().detach().cpu()  # First view rendered
    rendered_alpha = outputs['rendered_alpha'][0, 0, 0].float().detach().cpu()  # First view alpha
    
    n_thresh = len(thresholds)
    
    # Create figure: 6 rows
    # Row 0: Input RGB, Rendered RGB, GT Mask
    # Row 1: Rendered Alpha (continuous), Alpha histogram
    # Row 2-4: Pred Mask at each threshold
    # Row 5: Diff heatmaps
    
    fig = plt.figure(figsize=(4 * (n_thresh + 2), 4 * 5))
    
    # Title
    fig.suptitle(f'Alpha Threshold Comparison\nConfig: {config_name} | Step: {step}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(5, n_thresh + 2, hspace=0.3, wspace=0.1)
    
    # === Row 0: RGB Images ===
    ax = fig.add_subplot(gs[0, 0])
    rgb_np = input_rgb.permute(1, 2, 0).numpy()
    rgb_np = np.clip(rgb_np, 0, 1)
    ax.imshow(rgb_np)
    ax.set_title('Input RGB (View 0)', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    render_np = rendered_rgb.permute(1, 2, 0).numpy()
    render_np = np.clip(render_np, 0, 1)
    ax.imshow(render_np)
    ax.set_title('Rendered RGB (GS)', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    if gt_mask is not None:
        ax = fig.add_subplot(gs[0, 2])
        gt_rgb = mask_to_rgb(gt_mask)
        ax.imshow(gt_rgb)
        ax.set_title('GT Mask', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # === Row 1: Alpha visualization ===
    ax = fig.add_subplot(gs[1, 0])
    alpha_np = rendered_alpha.numpy()
    im = ax.imshow(alpha_np, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Rendered Alpha\n(Gaussian Splatting)', fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(alpha_np.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='t=0.5')
    for t in thresholds:
        if t != 0.5:
            ax.axvline(x=t, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_title('Alpha Distribution', fontsize=10, fontweight='bold')
    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)
    
    # Info text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    info_text = f"""
    Alpha Statistics:
    - Min: {alpha_np.min():.4f}
    - Max: {alpha_np.max():.4f}
    - Mean: {alpha_np.mean():.4f}
    - Std: {alpha_np.std():.4f}
    
    Foreground (>0.5): {(alpha_np > 0.5).mean()*100:.1f}%
    """
    ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # === Row 2: Pred Masks at different thresholds ===
    ious = []
    for i, thresh in enumerate(thresholds):
        ax = fig.add_subplot(gs[2, i])
        pred_mask = (rendered_alpha > thresh).float()
        pred_rgb = mask_to_rgb(pred_mask)
        ax.imshow(pred_rgb)
        
        iou = compute_iou(pred_mask, gt_mask) if gt_mask is not None else 0
        ious.append(iou)
        
        coverage = pred_mask.mean().item() * 100
        ax.set_title(f'Pred Mask\nthreshold={thresh:.1f}\nIoU={iou:.4f}\nCoverage={coverage:.1f}%', 
                     fontsize=9, fontweight='bold')
        ax.axis('off')
    
    # Legend for masks
    ax = fig.add_subplot(gs[2, n_thresh])
    ax.axis('off')
    fg_patch = mpatches.Patch(color=FG_COLOR, label='Foreground (α > t)')
    bg_patch = mpatches.Patch(color=BG_COLOR, label='Background (α ≤ t)')
    ax.legend(handles=[fg_patch, bg_patch], loc='center', fontsize=10)
    ax.set_title('Mask Legend', fontsize=10, fontweight='bold')
    
    # === Row 3: GT Mask repeated for comparison ===
    if gt_mask is not None:
        for i in range(n_thresh):
            ax = fig.add_subplot(gs[3, i])
            gt_rgb = mask_to_rgb(gt_mask)
            ax.imshow(gt_rgb)
            ax.set_title(f'GT Mask\n(for comparison)', fontsize=9)
            ax.axis('off')
    
    # === Row 4: Diff Heatmaps ===
    if gt_mask is not None:
        for i, thresh in enumerate(thresholds):
            ax = fig.add_subplot(gs[4, i])
            pred_mask = (rendered_alpha > thresh).float()
            diff = create_diff_heatmap(pred_mask, gt_mask)
            ax.imshow(diff)
            
            # Calculate error rates
            pred_np = (rendered_alpha.numpy() > thresh)
            gt_np = gt_mask.numpy() > 0.5
            fp_rate = ((pred_np) & (~gt_np)).mean() * 100
            fn_rate = ((~pred_np) & (gt_np)).mean() * 100
            
            ax.set_title(f'Diff (t={thresh:.1f})\nFP={fp_rate:.1f}% FN={fn_rate:.1f}%', 
                         fontsize=9, fontweight='bold')
            ax.axis('off')
        
        # Legend for diff
        ax = fig.add_subplot(gs[4, n_thresh])
        ax.axis('off')
        tp_patch = mpatches.Patch(color=[0, 1, 0], label='TP (Correct FG)')
        fp_patch = mpatches.Patch(color=[1, 0, 0], label='FP (False FG)')
        fn_patch = mpatches.Patch(color=[0, 0, 1], label='FN (Missed FG)')
        ax.legend(handles=[tp_patch, fp_patch, fn_patch], loc='center', fontsize=9)
        ax.set_title('Diff Legend', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_path}")
    print(f"\n{'='*50}")
    print(f"Alpha Threshold Comparison Summary")
    print(f"{'='*50}")
    print(f"Config: {config_name}")
    print(f"Step: {step}")
    print(f"\nThreshold | IoU      | Coverage")
    print(f"-" * 35)
    for i, thresh in enumerate(thresholds):
        pred_mask = (rendered_alpha > thresh).float()
        coverage = pred_mask.mean().item() * 100
        print(f"  {thresh:.1f}     | {ious[i]:.4f}   | {coverage:.1f}%")
    
    best_idx = np.argmax(ious)
    print(f"\n★ Best: threshold={thresholds[best_idx]:.1f}, IoU={ious[best_idx]:.4f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True)
    parser.add_argument('--thresholds', '-t', type=str, default=None,
                        help='Comma-separated thresholds (default: 0.3,0.5,0.7)')
    parser.add_argument('--output', '-o', default='outputs/threshold_viz')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--sample_idx', type=int, default=0)
    args = parser.parse_args()
    
    thresholds = [float(t) for t in args.thresholds.split(',')] if args.thresholds else DEFAULT_THRESHOLDS
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Thresholds: {thresholds}")
    
    model, config = load_checkpoint_and_model(args.checkpoint, args.device)
    batch = get_sample_batch(config, args.device, args.sample_idx)
    
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = model(batch, create_visual=True)
    
    if outputs.get('rendered_alpha') is None:
        print(f"Error: No rendered_alpha. Keys: {list(outputs.keys())}")
        return
    
    # Extract config name and step
    ckpt_path = Path(args.checkpoint)
    config_name = ckpt_path.parent.name
    step_match = re.search(r'iter_0*(\d+)', str(ckpt_path))
    step = int(step_match.group(1)) if step_match else 0
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) / f"threshold_comparison_{config_name}_step{step}.png"
    
    create_visualization(batch, outputs, thresholds, output_path, config_name, step)


if __name__ == '__main__':
    main()
