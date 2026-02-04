#!/usr/bin/env python3
"""Test temporal training integration.

Tests the combined main loss + temporal loss backward.
"""

import sys
import torch
from pathlib import Path
from easydict import EasyDict as edict

def test_combined_loss_backward():
    """Test that main loss + temporal loss can backward together."""
    print('=== Combined Loss Backward Test ===')
    
    from mouse_extensions.training import TemporalLossComputer, TemporalTrainingConfig
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
    
    # Load model
    print('Loading GS-LRM model...')
    gslrm = GSLRMInference(
        config_path='/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/config.yaml',
        checkpoint_path='/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt',
        device='cuda',
        image_size=512,
    )
    gslrm.model.train()
    
    # Setup temporal loss computer
    temporal_config = TemporalTrainingConfig(
        enabled=True,
        temporal_window=2,
        arap_weight=0.1,
        velocity_weight=0.01,
        warmup_steps=0,
        rampup_steps=0,
    )
    temporal_computer = TemporalLossComputer(temporal_config).cuda()
    
    # Load 2 consecutive frames
    data_dir = '/home/joon/data/preprocessed/FaceLift_mouse/M5'
    sample_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])[:2]
    
    print(f'Processing {len(sample_dirs)} frames...')
    
    scaler = torch.cuda.amp.GradScaler()
    xyz_buffer = []
    
    # Process both frames, accumulating xyz
    for i, sample_dir in enumerate(sample_dirs):
        images, c2ws, fxfycxcys, index = load_sample_data(str(sample_dir), 512, 'cuda')
        batch = edict(image=images, c2w=c2ws, fxfycxcy=fxfycxcys, index=index)
        
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            result = gslrm.model.forward(batch, create_visual=False, split_data=True)
        
        xyz = result.gaussian_params_raw.xyz[0]  # [N, 3]
        xyz_buffer.append({'xyz': xyz})
        
        print(f'  Frame {i}: main_loss={result.loss_metrics.loss.item():.6f}')
    
    # Compute temporal loss (now have 2 frames)
    temporal_loss, log_dict = temporal_computer.compute_temporal_loss(xyz_buffer, current_step=0)
    print(f'  Temporal loss: {temporal_loss.item():.8f}')
    print(f'  Log: {log_dict}')
    
    # Get main loss from last result (still has gradients)
    main_loss = result.loss_metrics.loss
    
    # Combine and backward
    print('\nCombining losses and backward...')
    total_loss = main_loss + temporal_loss
    print(f'  Total loss: {total_loss.item():.6f}')
    
    scaler.scale(total_loss).backward()
    
    # Check gradients
    grad_params = []
    for name, param in gslrm.model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_params.append((name, param.grad.abs().max().item()))
    
    if grad_params:
        print(f'\n✅ Test PASSED! {len(grad_params)} params have gradients')
        print('Top 3 by gradient magnitude:')
        for name, grad_max in sorted(grad_params, key=lambda x: -x[1])[:3]:
            print(f'  {name[:45]}: {grad_max:.8f}')
        return True
    else:
        print('\n❌ Test FAILED: No gradients')
        return False


if __name__ == '__main__':
    success = test_combined_loss_backward()
    sys.exit(0 if success else 1)
