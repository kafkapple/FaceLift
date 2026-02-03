#!/usr/bin/env python3
# Copyright 2026 FaceLift Mouse Extensions
# Training script for Deformation Network

"""
Train Deformation Network for temporal Gaussian consistency.

Usage:
    # Dry run (test config loading)
    python -m mouse_extensions.scripts.train_deformation --config configs/deformation/default.yaml --dry_run
    
    # Pre-compute Gaussian cache
    python -m mouse_extensions.scripts.train_deformation --config configs/deformation/default.yaml --precompute_cache
    
    # Full training
    python -m mouse_extensions.scripts.train_deformation --config configs/deformation/default.yaml

Training paradigm:
1. Load consecutive frame pairs (t, t+1)
2. Generate Gaussians G_t, G_{t+1} using pre-trained GS-LRM (or load from cache)
3. Train deformation network: G_t -> D(G_t) -> G'_{t+1} ≈ G_{t+1}
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mouse_extensions.model.deformation import (
    DeformationNetwork,
    DeformationConfig,
    DeformationTrainer,
    TrainerConfig,
    GaussianParams,
    GSLRMGaussianGenerator,
    GaussianCache,
)
from mouse_extensions.data.temporal_dataset import (
    TemporalPairDataset,
    TemporalDatasetConfig,
)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_deform_config(cfg: dict) -> DeformationConfig:
    """Build DeformationConfig from yaml dict."""
    model_cfg = cfg.get("model", {})
    return DeformationConfig(
        input_dim=model_cfg.get("input_dim", 3),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 8),
        activation=model_cfg.get("activation", "relu"),
        predict_position=model_cfg.get("predict_position", True),
        predict_opacity=model_cfg.get("predict_opacity", True),
        predict_scale=model_cfg.get("predict_scale", True),
        anisotropic_scale=model_cfg.get("anisotropic_scale", False),
        predict_rotation=model_cfg.get("predict_rotation", False),
        dropout=model_cfg.get("dropout", 0.0),
        zero_init_output=model_cfg.get("zero_init_output", True),
        use_positional_encoding=model_cfg.get("use_positional_encoding", False),
        pe_freq_bands=model_cfg.get("pe_freq_bands", 10),
    )


def build_trainer_config(cfg: dict, deform_config: DeformationConfig) -> TrainerConfig:
    """Build TrainerConfig from yaml dict."""
    train_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})
    
    return TrainerConfig(
        deform_config=deform_config,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        max_steps=train_cfg.get("max_steps", 10000),
        warmup_steps=train_cfg.get("warmup_steps", 500),
        l2_weight=train_cfg.get("l2_weight", 1.0),
        perceptual_weight=train_cfg.get("perceptual_weight", 0.1),
        temporal_weight=train_cfg.get("temporal_weight", 0.01),
        log_every=train_cfg.get("log_every", 100),
        save_every=train_cfg.get("save_every", 1000),
        output_dir=output_cfg.get("dir", "outputs/deformation"),
        device=cfg.get("device", "cuda"),
    )


def build_dataset_config(cfg: dict, split: str = "train") -> TemporalDatasetConfig:
    """Build TemporalDatasetConfig from yaml dict."""
    data_cfg = cfg.get("data", {})
    
    split_file = data_cfg.get(f"{split}_split", data_cfg.get("train_split"))
    
    return TemporalDatasetConfig(
        data_dir=data_cfg.get("data_dir", ""),
        split_file=split_file,
        sequence_length=data_cfg.get("sequence_length", 2),
        frame_stride=data_cfg.get("frame_stride", 1),
        num_views=data_cfg.get("num_views", 6),
    )


def precompute_gaussian_cache(
    cfg: dict,
    dataset: TemporalPairDataset,
    cache_dir: str,
    device: str = "cuda",
):
    """
    Pre-compute all Gaussians using GS-LRM and save to cache.
    
    This is done once before training to avoid repeated GS-LRM inference.
    """
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
    from mouse_extensions.data.mouse_dataset import MouseDataset
    
    gslrm_cfg = cfg.get("gslrm", {})
    data_cfg = cfg.get("data", {})
    
    print("\n=== Pre-computing Gaussian Cache ===")
    print(f"  GS-LRM checkpoint: {gslrm_cfg.get('checkpoint')}")
    print(f"  Cache directory: {cache_dir}")
    
    # Initialize cache
    cache = GaussianCache(cache_dir=cache_dir)
    
    # Get unique frame indices
    all_frames = set()
    for seq in dataset.sequences:
        all_frames.update(seq)
    all_frames = sorted(all_frames)
    
    print(f"  Total unique frames: {len(all_frames)}")
    
    # Check which frames need processing
    frames_to_process = [f for f in all_frames if not cache.has(f)]
    print(f"  Frames to process: {len(frames_to_process)}")
    
    if not frames_to_process:
        print("  All frames already cached!")
        return cache
    
    # Load GS-LRM model
    # Note: This requires proper config setup for the GS-LRM model
    # For now, we provide a placeholder that shows the intended flow
    
    print("\n  [INFO] Full GS-LRM caching requires:")
    print("    1. GS-LRM config file")
    print("    2. GS-LRM checkpoint")
    print("    3. MouseDataset for loading images/cameras")
    print("\n  See configs/deformation/default.yaml for configuration.")
    
    return cache


def train_loop(
    trainer: DeformationTrainer,
    dataset: TemporalPairDataset,
    cache: GaussianCache,
    cfg: dict,
):
    """
    Main training loop.
    
    Args:
        trainer: DeformationTrainer instance
        dataset: TemporalPairDataset with frame pairs
        cache: GaussianCache with pre-computed Gaussians
        cfg: Full config dict
    """
    train_cfg = cfg.get("training", {})
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one pair at a time for now
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for cache access
    )
    
    max_steps = trainer.config.max_steps
    log_every = trainer.config.log_every
    save_every = trainer.config.save_every
    
    print(f"\n=== Starting Training ===")
    print(f"  Max steps: {max_steps}")
    print(f"  Dataset pairs: {len(dataset)}")
    
    # Training loop
    pbar = tqdm(total=max_steps, desc="Training")
    epoch = 0
    
    while trainer.global_step < max_steps:
        epoch += 1
        
        for batch in dataloader:
            if trainer.global_step >= max_steps:
                break
            
            # Get frame indices
            t = batch["t"][0].item()
            t1 = batch["t1"][0].item()
            
            # Get cached Gaussians
            G_t = cache.get(t)
            G_t1 = cache.get(t1)
            
            if G_t is None or G_t1 is None:
                print(f"Warning: Missing cache for frames {t}, {t1}")
                continue
            
            # Train step
            losses = trainer.train_step(G_t, G_t1)
            
            # Logging
            if trainer.global_step % log_every == 0:
                loss_str = ", ".join(
                    f"{k}={v.item():.4f}" for k, v in losses.items()
                )
                tqdm.write(f"Step {trainer.global_step}: {loss_str}")
            
            # Checkpointing
            if trainer.global_step % save_every == 0:
                trainer.save_checkpoint()
            
            pbar.update(1)
    
    pbar.close()
    
    # Final checkpoint
    trainer.save_checkpoint()
    print(f"\nTraining complete! Final step: {trainer.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train Deformation Network")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Test without training")
    parser.add_argument("--precompute_cache", action="store_true", help="Only precompute Gaussian cache")
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    cfg = load_config(args.config)
    
    # Set seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    
    # Build configs
    deform_config = build_deform_config(cfg)
    trainer_config = build_trainer_config(cfg, deform_config)
    dataset_config = build_dataset_config(cfg, split="train")
    
    print(f"\nDeformation Network:")
    print(f"  Layers: {deform_config.num_layers}")
    print(f"  Hidden: {deform_config.hidden_dim}")
    print(f"  Output: {deform_config.output_dim}")
    
    # Create trainer
    trainer = DeformationTrainer(trainer_config)
    print(f"\nTrainer created: {trainer.deform_net.get_num_params():,} parameters")
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create dataset
    print(f"\nLoading dataset from {dataset_config.data_dir}")
    dataset = TemporalPairDataset(dataset_config)
    print(f"  Frames: {len(dataset.frames)}")
    print(f"  Pairs: {len(dataset)}")
    
    if args.dry_run:
        print("\n=== Dry run complete ===")
        return
    
    # Cache directory
    output_cfg = cfg.get("output", {})
    cache_dir = Path(output_cfg.get("dir", "outputs/deformation")) / "gaussian_cache"
    
    # Pre-compute cache if requested
    if args.precompute_cache:
        cache = precompute_gaussian_cache(cfg, dataset, str(cache_dir))
        print(f"\nCache contains {len(cache)} frames")
        return
    
    # Load or create cache
    cache = GaussianCache(cache_dir=str(cache_dir))
    print(f"\nGaussian cache: {len(cache)} frames cached")
    
    if len(cache) == 0:
        print("\n[WARNING] No cached Gaussians found!")
        print("Run with --precompute_cache first to generate Gaussian cache.")
        print("Or provide pre-computed Gaussians in the cache directory.")
        return
    
    # Start training
    train_loop(trainer, dataset, cache, cfg)


if __name__ == "__main__":
    main()
