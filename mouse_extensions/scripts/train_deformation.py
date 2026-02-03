#!/usr/bin/env python3
# Copyright 2026 FaceLift Mouse Extensions
# Training script for Deformation Network

"""
Train Deformation Network for temporal Gaussian consistency.

Usage:
    python -m mouse_extensions.scripts.train_deformation --config configs/deformation/default.yaml

Training paradigm:
1. Load consecutive frame pairs (t, t+1)
2. Generate Gaussians G_t, G_{t+1} using pre-trained GS-LRM
3. Train deformation network: G_t -> D(G_t) -> G'_{t+1} ≈ G_{t+1}
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
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


def build_dataset_config(cfg: dict) -> TemporalDatasetConfig:
    """Build TemporalDatasetConfig from yaml dict."""
    data_cfg = cfg.get("data", {})
    
    return TemporalDatasetConfig(
        data_dir=data_cfg.get("data_dir", ""),
        split_file=data_cfg.get("train_split", None),
        sequence_length=data_cfg.get("sequence_length", 2),
        frame_stride=data_cfg.get("frame_stride", 1),
        num_views=data_cfg.get("num_views", 6),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Deformation Network")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Test without training")
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
    dataset_config = build_dataset_config(cfg)
    
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
    
    # Create dataloader
    train_cfg = cfg.get("training", {})
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    
    print(f"\n=== Starting Training ===")
    print(f"  Max steps: {trainer_config.max_steps}")
    print(f"  Output: {trainer_config.output_dir}")
    
    # Training loop placeholder
    # Note: Full integration with GS-LRM requires additional work
    print("\n[TODO] Full training loop requires GS-LRM integration")
    print("Current implementation provides:")
    print("  - DeformationNetwork (tested)")
    print("  - GaussianParams (tested)")
    print("  - TemporalPipeline (tested)")
    print("  - DeformationTrainer (tested)")
    print("  - TemporalDataset (tested)")
    print("\nNext step: Integrate with GS-LRM inference for pseudo GT generation")


if __name__ == "__main__":
    main()
