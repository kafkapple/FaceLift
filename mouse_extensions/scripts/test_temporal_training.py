#!/usr/bin/env python3
"""Test script for temporal training integration.

Tests that temporal losses (ARAP, velocity) can be computed and gradients
flow back through the GS-LRM model.
"""

import torch
from pathlib import Path
from easydict import EasyDict as edict

def test_temporal_training(
    config_path: str,
    checkpoint_path: str,
    data_dir: str,
    num_frames: int = 3,
    device: str = 'cuda',
):
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
    from mouse_extensions.training import TemporalTrainingConfig, TemporalLossComputer
    
    print('=== Temporal Training Integration Test ===')
    
    # Load model
    print('Loading GS-LRM model...')
    gslrm = GSLRMInference(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        image_size=512,
    )
    gslrm.model.train()
    
    # Setup temporal loss (no rampup for testing)
    temporal_config = TemporalTrainingConfig(
        enabled=True,
        temporal_window=num_frames,
        arap_weight=0.1,
        velocity_weight=0.01,
        warmup_steps=0,
        rampup_steps=0,  # Full weight immediately
    )
    temporal_computer = TemporalLossComputer(temporal_config).to(device)
    
    # Get consecutive samples
    samples_dir = Path(data_dir)
    sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])[:num_frames]
    print(f'Using {len(sample_dirs)} frames: {[d.name for d in sample_dirs]}')
    
    # Load and forward pass for each frame
    xyz_list = []
    for sample_dir in sample_dirs:
        images, c2ws, fxfycxcys, index = load_sample_data(str(sample_dir), 512, device)
        batch = edict(image=images, c2w=c2ws, fxfycxcy=fxfycxcys, index=index)
        
        with torch.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
            result = gslrm.model.forward(batch, create_visual=False, split_data=True)
        
        xyz = result.gaussian_params_raw.xyz[0]  # [N, 3] with gradients
        xyz_list.append(xyz)
        print(f'  Frame {sample_dir.name}: xyz shape={xyz.shape}, requires_grad={xyz.requires_grad}')
    
    # Compute temporal loss
    gaussian_params = [{'xyz': xyz} for xyz in xyz_list]
    loss, log_dict = temporal_computer.compute_temporal_loss(gaussian_params, current_step=0)
    
    print(f'\nTemporal Loss: {loss.item():.8f}')
    print(f'  ARAP: {log_dict.get("temporal/arap", 0):.8f}')
    print(f'  Velocity: {log_dict.get("temporal/velocity", 0):.8f}')
    print(f'  requires_grad: {loss.requires_grad}')
    
    # Check gradient flow
    if loss.requires_grad and loss.item() > 0:
        loss.backward()
        
        grad_params = []
        for name, param in gslrm.model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                grad_params.append((name, param.grad.abs().max().item()))
        
        if grad_params:
            print(f'\n✅ Gradient Flow Test PASSED!')
            print(f'   {len(grad_params)} parameters received gradients')
            print(f'   Top 3 by gradient magnitude:')
            for name, grad_max in sorted(grad_params, key=lambda x: -x[1])[:3]:
                print(f'     {name[:45]}: {grad_max:.8f}')
            return True
        else:
            print('\n❌ Test FAILED: No gradients in model parameters')
            return False
    else:
        print('\n❌ Test FAILED: Cannot compute gradients')
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test temporal training gradient flow')
    parser.add_argument('--config', type=str, 
                       default='/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/config.yaml')
    parser.add_argument('--checkpoint', type=str,
                       default='/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/best_psnr.pt')
    parser.add_argument('--data_dir', type=str,
                       default='/home/joon/data/preprocessed/FaceLift_mouse/M5')
    parser.add_argument('--num_frames', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    success = test_temporal_training(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        num_frames=args.num_frames,
        device=args.device,
    )
    
    exit(0 if success else 1)
