#!/usr/bin/env python3
"""
Alpha Inversion Analysis Script
Analyzes actual rendered alpha values from checkpoint to verify inversion hypothesis.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, '/home/joon/dev/FaceLift')

def analyze_rendered_alpha(checkpoint_path: str, data_dir: str, sample_idx: int = 0):
    """Load checkpoint and analyze rendered alpha values."""
    from omegaconf import OmegaConf
    from gslrm.model.gslrm import GSLRM
    from gslrm.datasets.gobjaverse import load_mviews
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load config
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / 'config.yaml'
    config = OmegaConf.load(config_path)
    
    # Load model
    device = torch.device('cuda:0')
    model = GSLRM(config)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Device: {device}")
    
    # Load sample data
    from gslrm.datasets.mouse_dataset import MouseDataset
    dataset = MouseDataset(config, split='train')
    
    sample = dataset[sample_idx]
    
    # Prepare batch
    batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in sample.items()}
    
    print(f"Sample loaded. Keys: {list(batch.keys())}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    # Extract rendered alpha
    # rendered_alpha shape: [B, V, 1, H, W] or [B, V, H, W]
    rendered_alpha = outputs.get('rendered_alpha')
    gt_mask = batch.get('masks')  # [B, V, 1, H, W]
    
    if rendered_alpha is None:
        print("ERROR: rendered_alpha not in outputs")
        print(f"Available keys: {list(outputs.keys())}")
        return
    
    print(f"rendered_alpha shape: {rendered_alpha.shape}")
    print(f"gt_mask shape: {gt_mask.shape if gt_mask is not None else 'None'}")
    
    # Analyze per view
    B, V = rendered_alpha.shape[:2]
    
    print("\n" + "="*70)
    print("ALPHA ANALYSIS PER VIEW")
    print("="*70)
    print(f"{'View':<6} {'Type':<8} {'Alpha FG':<12} {'Alpha BG':<12} {'Status':<12}")
    print("-"*70)
    
    for v in range(V):
        alpha_v = rendered_alpha[0, v].squeeze().cpu().numpy()  # [H, W]
        
        if gt_mask is not None:
            mask_v = gt_mask[0, v].squeeze().cpu().numpy()  # [H, W]
            fg_region = mask_v > 0.5
            bg_region = mask_v <= 0.5
        else:
            # Assume center is foreground
            H, W = alpha_v.shape
            fg_region = np.zeros((H, W), dtype=bool)
            fg_region[H//4:3*H//4, W//4:3*W//4] = True
            bg_region = ~fg_region
        
        alpha_fg = alpha_v[fg_region].mean() if fg_region.sum() > 0 else 0
        alpha_bg = alpha_v[bg_region].mean() if bg_region.sum() > 0 else 0
        
        view_type = "Input" if v < V-1 else "Novel"
        is_inverted = alpha_fg < alpha_bg
        status = "⚠️ INVERTED" if is_inverted else "✅ Normal"
        
        print(f"{v:<6} {view_type:<8} {alpha_fg:.4f}       {alpha_bg:.4f}       {status}")
    
    print("="*70)
    
    return rendered_alpha, gt_mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--sample-idx', '-s', type=int, default=0)
    args = parser.parse_args()
    
    analyze_rendered_alpha(args.checkpoint, None, args.sample_idx)
