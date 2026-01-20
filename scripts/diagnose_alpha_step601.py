#!/usr/bin/env python3
"""
Direct Alpha Analysis for Step 601
Loads checkpoint and analyzes rendered alpha to verify inversion hypothesis.
"""

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, '/home/joon/dev/FaceLift')

def main():
    from omegaconf import OmegaConf
    from gslrm.model.gslrm import GSLRM
    from gslrm.data.mouse_dataset import MouseViewDataset
    
    # Paths
    ckpt_path = '/home/joon/dev/FaceLift/checkpoints/gslrm/D7_t_E4_2_5v_alpha_loss/ckpt_0000000000000600.pt'
    config_path = '/home/joon/dev/FaceLift/checkpoints/gslrm/D7_t_E4_2_5v_alpha_loss/config.yaml'
    output_dir = Path('/home/joon/dev/FaceLift/reports/alpha_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading config: {config_path}")
    config = OmegaConf.load(config_path)
    
    print(f"Loading model...")
    device = torch.device('cuda:0')
    model = GSLRM(config)
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model = model.to(device)
    model.eval()
    print(f"Model loaded.")
    
    # Load dataset
    print(f"Loading dataset...")
    dataset = MouseViewDataset(config, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample 92 (from the supervision image filename)
    sample_idx = 92
    sample = dataset[sample_idx]
    
    # Print sample keys
    print(f"Sample keys: {list(sample.keys())}")
    
    # Prepare batch
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
            print(f"  {k}: {v.shape} -> {batch[k].shape}")
        else:
            batch[k] = v
            print(f"  {k}: {type(v)}")
    
    # Forward pass with hook to capture rendered_alpha
    captured = {}
    
    def capture_hook(module, input, output):
        if isinstance(output, tuple) and len(output) == 2:
            # gaussian_renderer returns (images, alpha)
            captured['rendered_alpha'] = output[1].detach().cpu()
            captured['rendered_images'] = output[0].detach().cpu()
    
    # Register hook on gaussian_renderer
    hook = model.gaussian_renderer.register_forward_hook(capture_hook)
    
    try:
        with torch.no_grad():
            outputs = model(batch)
    finally:
        hook.remove()
    
    if 'rendered_alpha' not in captured:
        print("ERROR: Could not capture rendered_alpha")
        print(f"Output keys: {list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}")
        return
    
    rendered_alpha = captured['rendered_alpha']  # [B, V, 1, H, W] or [B, V, H, W]
    print(f"\nCaptured rendered_alpha shape: {rendered_alpha.shape}")
    
    # Get GT mask
    gt_mask = batch.get('masks')
    if gt_mask is not None:
        gt_mask = gt_mask.cpu()
        print(f"GT mask shape: {gt_mask.shape}")
    
    # Analyze per view
    if len(rendered_alpha.shape) == 5:
        B, V, C, H, W = rendered_alpha.shape
        rendered_alpha = rendered_alpha.squeeze(2)  # [B, V, H, W]
    else:
        B, V, H, W = rendered_alpha.shape
    
    print("\n" + "="*70)
    print("ALPHA INVERSION ANALYSIS - ACTUAL CHECKPOINT DATA")
    print("="*70)
    print(f"{'View':<6} {'Type':<8} {'Alpha FG':<12} {'Alpha BG':<12} {'Corr':<8} {'Status':<12}")
    print("-"*70)
    
    input_inverted = 0
    novel_inverted = 0
    
    for v in range(V):
        alpha_v = rendered_alpha[0, v].numpy()  # [H, W]
        
        if gt_mask is not None and v < gt_mask.shape[1]:
            mask_v = gt_mask[0, v].squeeze().numpy()  # [H, W]
            if mask_v.max() > 1:
                mask_v = mask_v / 255.0
            fg_region = mask_v > 0.5
            bg_region = mask_v <= 0.5
        else:
            # Fallback: assume center region is foreground
            fg_region = np.zeros((H, W), dtype=bool)
            fg_region[H//4:3*H//4, W//4:3*W//4] = True
            bg_region = ~fg_region
        
        alpha_fg = alpha_v[fg_region].mean() if fg_region.sum() > 0 else 0
        alpha_bg = alpha_v[bg_region].mean() if bg_region.sum() > 0 else 0
        
        # Correlation
        if gt_mask is not None and v < gt_mask.shape[1]:
            corr = np.corrcoef(alpha_v.flatten(), mask_v.flatten())[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        view_type = "Input" if v < V-1 else "Novel"
        is_inverted = alpha_fg < alpha_bg
        status = "INVERTED" if is_inverted else "Normal"
        
        if v < V-1 and is_inverted:
            input_inverted += 1
        elif v == V-1 and is_inverted:
            novel_inverted += 1
        
        print(f"{v:<6} {view_type:<8} {alpha_fg:.4f}       {alpha_bg:.4f}       {corr:+.3f}   {status}")
        
        # Save alpha visualization
        alpha_img = (alpha_v * 255).astype(np.uint8)
        Image.fromarray(alpha_img).save(output_dir / f'alpha_view_{v}.png')
    
    print("-"*70)
    print(f"\nInput views inverted: {input_inverted}/{V-1}")
    print(f"Novel view inverted: {novel_inverted}/1")
    
    if input_inverted >= 3 and novel_inverted == 0:
        print("\n⚠️  HYPOTHESIS CONFIRMED: Input views have inverted alpha!")
    elif input_inverted == 0 and novel_inverted == 0:
        print("\n✅  All views have correct alpha orientation.")
    else:
        print(f"\n❓ Mixed results - needs further investigation.")
    
    print(f"\nAlpha images saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
