"""
Evaluate model on hold-out test set.

Usage:
    python -m mouse_extensions.scripts.evaluate_test \
        --checkpoint checkpoints/gslrm/M3_3t_E1/iter_00005000/model.pt \
        --config configs/mouse/M3_3t_E1.yaml \
        --output_dir outputs/test_eval/M3_3t_E1
"""

import argparse
import os
import sys
import torch
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from easydict import EasyDict as edict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_config(config_path: str) -> edict:
    """Load and merge config files."""
    cfg = OmegaConf.load(config_path)
    
    # Merge with base config if specified
    if hasattr(cfg, 'base') and cfg.base:
        base_dir = Path(config_path).parent
        base_cfg = OmegaConf.load(base_dir / cfg.base)
        cfg = OmegaConf.merge(base_cfg, cfg)
    
    return edict(OmegaConf.to_container(cfg, resolve=True))


def evaluate_test(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    batch_size: int = 1,
    save_visualizations: bool = True,
    device: str = "cuda",
):
    """
    Evaluate model on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        save_visualizations: Whether to save visual outputs
        device: Device to use
    """
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    # Check test dataset path
    test_path = config.get("test", {}).get("dataset_path", "")
    if not test_path or not os.path.exists(test_path):
        print(f"Error: Test dataset path not found: {test_path}")
        print("Make sure config has 'test.dataset_path' set.")
        return None
    
    print(f"Test dataset: {test_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    from gslrm.model.gslrm import GSLRM
    
    model = GSLRM(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # Create test dataset
    print("Creating test dataset...")
    from mouse_extensions.data.mouse_dataset import MouseViewDataset
    
    test_dataset = MouseViewDataset(config, split="test")
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Evaluate
    print(f"\nEvaluating on test set ({len(test_dataset)} samples)...")
    
    all_metrics = {"psnr": [], "lpips": [], "ssim": [], "mask_iou": []}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            # Move batch to device
            batch = edict(batch)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            result = model.forward(batch)
            
            # Save validation results (includes metrics computation)
            save_vis = save_visualizations and batch_idx < 10  # Save first 10
            metrics = model.save_validation_results(
                output_dir, result, batch, test_dataset, save_vis
            )
            
            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(metrics[key])
    
    # Aggregate metrics
    final_metrics = {}
    for key in all_metrics:
        if all_metrics[key]:
            final_metrics[key] = sum(all_metrics[key]) / len(all_metrics[key])
    
    # Print results
    print(f"\n" + "="*50)
    print("TEST SET RESULTS (Hold-out)")
    print("="*50)
    for key, value in final_metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    print("="*50)
    
    # Save results
    results = {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "test_dataset": test_path,
        "num_samples": len(test_dataset),
        "metrics": final_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    results_path = os.path.join(output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate on hold-out test set")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--no_vis", action="store_true", help="Disable visualizations")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    evaluate_test(
        args.checkpoint,
        args.config,
        args.output_dir,
        batch_size=args.batch_size,
        save_visualizations=not args.no_vis,
        device=args.device,
    )


if __name__ == "__main__":
    main()
