#!/usr/bin/env python3
"""
Unified Pipeline Runner

전처리 → Split 생성 → 실험 설정 생성을 한 번에 실행

Usage:
    # 전체 파이프라인
    python scripts/run_pipeline.py --config configs/preprocessing/data_mouse_v12_centered.yaml
    
    # 개별 단계
    python scripts/run_pipeline.py --config ... --step preprocess
    python scripts/run_pipeline.py --config ... --step split
    python scripts/run_pipeline.py --config ... --step experiment
"""

import argparse
import os
import sys
import subprocess
import random
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_preprocessing(config: dict) -> bool:
    """Run preprocessing script."""
    print("\n" + "="*60)
    print("Step 1: Preprocessing")
    print("="*60)
    
    cmd = [
        "python", "scripts/convert_markerless_unified.py",
        "--version", config['preprocessing']['version'],
        "--input_dir", config['paths']['input_dir'],
        "--output_dir", config['paths']['output_dir'],
        "--frame_interval", str(config['sampling']['frame_interval'])
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def generate_split(config: dict) -> bool:
    """Generate train/val split."""
    print("\n" + "="*60)
    print("Step 2: Generate Train/Val Split")
    print("="*60)
    
    output_dir = Path(config['paths']['output_dir'])
    
    # Find all sample directories
    samples = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')])
    
    if not samples:
        print(f"ERROR: No samples found in {output_dir}")
        return False
    
    print(f"Found {len(samples)} samples")
    
    # Shuffle with seed
    random.seed(config['split']['seed'])
    random.shuffle(samples)
    
    # Split
    val_ratio = config['split']['val_ratio']
    n_val = max(1, int(len(samples) * val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Write split files
    train_file = output_dir / f"{config['name']}_train.txt"
    val_file = output_dir / f"{config['name']}_val.txt"
    
    with open(train_file, 'w') as f:
        for s in train_samples:
            f.write(str(s) + '\n')
    
    with open(val_file, 'w') as f:
        for s in val_samples:
            f.write(str(s) + '\n')
    
    print(f"Written: {train_file}")
    print(f"Written: {val_file}")
    
    return True


def generate_experiment_config(config: dict, exp_name: str = None) -> bool:
    """Generate experiment YAML config."""
    print("\n" + "="*60)
    print("Step 3: Generate Experiment Config")
    print("="*60)
    
    if exp_name is None:
        exp_name = f"gslrm_{config['name']}"
    
    output_dir = Path(config['paths']['output_dir'])
    train_file = output_dir / f"{config['name']}_train.txt"
    val_file = output_dir / f"{config['name']}_val.txt"
    
    defaults = config.get('experiment_defaults', {})
    
    exp_config = {
        'profile': False,
        'debug': False,
        'model': {
            'class_name': 'gslrm.model.gslrm.GSLRM',
            'image_tokenizer': {
                'image_size': config['target']['image_size'],
                'patch_size': 8,
                'in_channels': 9
            },
            'transformer': {'d': 1024, 'd_head': 64, 'n_layer': 24},
            'gaussians': {
                'n_gaussians': 2,
                'sh_degree': 0,
                'upsampler': {'upsample_factor': 1}
            },
            'add_refsrc_marker': False,
            'hard_pixelalign': True,
            'use_custom_plucker': True,
            'clip_xyz': True
        },
        'training': {
            'runtime': {
                'use_tf32': True,
                'use_amp': True,
                'amp_dtype': 'bf16',
                'torch_compile': False,
                'grad_accum_steps': 2,
                'grad_clip_norm': 50.0,
                'grad_checkpoint_every': 1
            },
            'dataset': {
                'dataset_path': str(train_file),
                'maximize_view_overlap': False,
                'num_views': defaults.get('num_views', 6),
                'num_input_views': defaults.get('num_input_views', 5),
                'target_has_input': True,
                'normalize_distance_to': 0.0,
                'remove_alpha': False,
                'background_color': 'white',
                'random_view_selection': True
            },
            'dataloader': {
                'batch_size_per_gpu': 2,
                'num_workers': 8,
                'num_threads': 16,
                'prefetch_factor': 8
            },
            'losses': {
                'l2_loss_weight': 1.0,
                'lpips_loss_weight': 0.0,
                'perceptual_loss_weight': 0.5,
                'ssim_loss_weight': 0.0,
                'pixelalign_loss_weight': 0.0,
                'masked_pixelalign_loss': defaults.get('masked_l2_loss', True),
                'masked_l2_loss': defaults.get('masked_l2_loss', True),
                'masked_ssim_loss': defaults.get('masked_ssim_loss', True),
                'pointsdist_loss_weight': 0.0,
                'warmup_pointsdist': False,
                'distill_loss_weight': 0.0,
                'clamp_rendering': True,
                'use_predicted_mask': False,
                'use_rendered_alpha_mask': defaults.get('use_rendered_alpha_mask', True),
                'alpha_mask_threshold': 0.5
            },
            'optimizer': {
                'lr': defaults.get('lr', 1.0e-6),
                'beta1': 0.9,
                'beta2': 0.95,
                'weight_decay': 0.05,
                'reset_lr': True,
                'reset_weight_decay': False,
                'reset_training_state': True
            },
            'schedule': {
                'num_epochs': 50000,
                'early_stop_after_epochs': 50000,
                'max_fwdbwd_passes': 20000,
                'warmup': 500,
                'l2_warmup_steps': 500
            },
            'checkpointing': {
                'checkpoint_dir': f'checkpoints/mouse_{exp_name}',
                'checkpoint_every': 500,
                'resume_ckpt': 'checkpoints/gslrm/ckpt_0000000000021125.pt'
            },
            'logging': {
                'print_every': 20,
                'vis_every': 250,
                'wandb': {
                    'project': 'mouse_facelift',
                    'group': 'gslrm',
                    'job_type': 'finetune',
                    'exp_name': f'mouse_{exp_name}',
                    'log_every': 50,
                    'offline': False
                }
            }
        },
        'mouse': {
            'use_mouse_dataset': True,
            'normalize_cameras': False,
            'target_camera_distance': 0.0,
            'normalize_to_z_up': True,
            'auto_generate_mask': False,
            'mask_threshold': 250
        },
        'validation': {
            'enabled': True,
            'dataset_path': str(val_file),
            'output_dir': f'experiments/validation/mouse_{exp_name}',
            'val_every': 100
        },
        'inference': {
            'enabled': False,
            'output_dir': f'experiments/inference/mouse_{exp_name}'
        }
    }
    
    # Write config
    exp_config_path = Path(f'configs/mouse/{exp_name}.yaml')
    exp_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(exp_config_path, 'w') as f:
        # Add header comment
        f.write(f"# Auto-generated from: {config['name']}\n")
        f.write(f"# Description: {config.get('description', '')}\n\n")
        yaml.dump(exp_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Written: {exp_config_path}")
    
    # Print run command
    print(f"\nTo run experiment:")
    print(f"  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \\\n    train_gslrm.py -c {exp_config_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Unified Pipeline Runner')
    parser.add_argument('--config', '-c', required=True, help='Path to preprocessing config YAML')
    parser.add_argument('--step', choices=['all', 'preprocess', 'split', 'experiment'], 
                        default='all', help='Which step to run')
    parser.add_argument('--exp_name', help='Custom experiment name')
    parser.add_argument('--skip_preprocess', action='store_true', 
                        help='Skip preprocessing if output already exists')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"\nLoaded config: {config['name']}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Status: {config.get('status', 'active')}")
    
    success = True
    
    if args.step in ['all', 'preprocess']:
        output_exists = Path(config['paths']['output_dir']).exists()
        if args.skip_preprocess and output_exists:
            print(f"\nSkipping preprocessing (output exists): {config['paths']['output_dir']}")
        else:
            success = run_preprocessing(config) and success
    
    if args.step in ['all', 'split']:
        success = generate_split(config) and success
    
    if args.step in ['all', 'experiment']:
        success = generate_experiment_config(config, args.exp_name) and success
    
    print("\n" + "="*60)
    print("Pipeline Complete" if success else "Pipeline Failed")
    print("="*60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
