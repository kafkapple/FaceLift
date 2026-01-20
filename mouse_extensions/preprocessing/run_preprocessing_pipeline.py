#!/usr/bin/env python3
"""
Unified Preprocessing Pipeline for Mouse Dataset

Runs:
1. Preprocessing (D6-1 or D6-3) - skipped if data exists
2. Verification (PP check)
3. Experiment config generation
4. Training command output

Usage:
    # Full pipeline (auto-detects existing data)
    python -m mouse_extensions.preprocessing.run_preprocessing_pipeline --method D6-3
    
    # Force re-preprocessing
    python -m mouse_extensions.preprocessing.run_preprocessing_pipeline --method D6-3 --force

Author: Joon Park
Date: 2026-01-18
"""

import argparse
import subprocess
import sys
import json
import os
from pathlib import Path
import yaml
import re


def check_existing_data(output_dir: str) -> dict:
    """Check if preprocessed data already exists."""
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    result = {
        "exists": False,
        "train_samples": 0,
        "val_samples": 0,
        "has_cameras": False,
    }
    
    if train_dir.exists():
        train_samples = list(train_dir.iterdir())
        result["train_samples"] = len(train_samples)
        
        # Check if camera files exist
        if train_samples:
            cam_file = train_samples[0] / "opencv_cameras.json"
            result["has_cameras"] = cam_file.exists()
    
    if val_dir.exists():
        result["val_samples"] = len(list(val_dir.iterdir()))
    
    result["exists"] = result["train_samples"] > 0 and result["has_cameras"]
    
    return result


def run_preprocessing(method: str, input_dir: str, output_dir: str) -> bool:
    """Run D6 preprocessing."""
    print("\n" + "=" * 60)
    print(f"Step 1: Running {method} Preprocessing")
    print("=" * 60)
    
    cmd = [
        sys.executable, "-m", "mouse_extensions.preprocessing.preprocessor_d6",
        "--method", method,
        "--input_dir", input_dir,
        "--output_dir", output_dir
    ]
    
    result = subprocess.run(cmd, cwd="/home/joon/dev/FaceLift")
    return result.returncode == 0


def run_verification(dataset_dir: str) -> dict:
    """Run dataset verification and return stats."""
    print("\n" + "=" * 60)
    print("Step 2: Verifying Dataset")
    print("=" * 60)
    
    train_dir = Path(dataset_dir) / "train"
    if not train_dir.exists():
        print(f"Error: {train_dir} does not exist")
        return {"success": False}
    
    sample_dirs = sorted(train_dir.iterdir())[:5]
    
    all_cx, all_cy, all_fx = [], [], []
    for sample_dir in sample_dirs:
        cam_path = sample_dir / "opencv_cameras.json"
        if cam_path.exists():
            with open(cam_path) as f:
                data = json.load(f)
            for frame in data["frames"]:
                all_cx.append(frame["cx"])
                all_cy.append(frame["cy"])
                all_fx.append(frame["fx"])
    
    if not all_cx:
        print("Error: No camera data found")
        return {"success": False}
    
    # Check if all are 256 (bug) or varying (correct)
    all_256 = all(abs(cx - 256) < 0.1 and abs(cy - 256) < 0.1 
                  for cx, cy in zip(all_cx, all_cy))
    
    cx_mean = sum(all_cx) / len(all_cx)
    cy_mean = sum(all_cy) / len(all_cy)
    fx_mean = sum(all_fx) / len(all_fx)
    
    stats = {
        "success": True,
        "cx_mean": cx_mean,
        "cy_mean": cy_mean,
        "fx_mean": fx_mean,
        "cx_std": (sum((x - cx_mean)**2 for x in all_cx) / len(all_cx)) ** 0.5,
        "cy_std": (sum((y - cy_mean)**2 for y in all_cy) / len(all_cy)) ** 0.5,
        "all_256": all_256,
        "num_samples": len(sample_dirs),
    }
    
    print(f"\nPP Statistics (from {len(sample_dirs)} samples, {len(all_cx)} views):")
    print(f"  cx: mean={stats['cx_mean']:.1f}, std={stats['cx_std']:.1f}")
    print(f"  cy: mean={stats['cy_mean']:.1f}, std={stats['cy_std']:.1f}")
    print(f"  fx: mean={stats['fx_mean']:.1f}")
    
    if all_256:
        print("\n[FAIL] All cx, cy are 256 (D4 bug present!)")
    else:
        print("\n[PASS] cx, cy are NOT all 256 (PP correctly computed)")
    
    # Check fx normalization
    if abs(fx_mean - 549) < 50:
        print(f"[PASS] fx ~ 549 (normalized)")
    else:
        print(f"[WARNING] fx = {fx_mean:.1f} (expected ~549)")
    
    return stats


def generate_experiment_configs(method: str, output_dir: str, config_dir: str) -> list:
    """Generate experiment configs from D4 templates."""
    print("\n" + "=" * 60)
    print("Step 3: Generating Experiment Configs")
    print("=" * 60)
    
    config_path = Path(config_dir)
    d4_configs = list(config_path.glob("D4_E*.yaml"))
    
    if not d4_configs:
        print(f"Warning: No D4 configs found in {config_dir}")
        return []
    
    method_short = method.replace("-", "")  # D6-3 -> D63
    generated = []
    skipped = []
    
    for d4_config in sorted(d4_configs):
        # Create new config name
        old_name = d4_config.stem
        new_name = old_name.replace("D4", method_short)
        new_config_path = config_path / f"{new_name}.yaml"
        
        # Check if already exists
        if new_config_path.exists():
            skipped.append(new_config_path)
            continue
        
        # Read D4 config
        with open(d4_config) as f:
            config = yaml.safe_load(f)
        
        # Update paths and names
        if "training" in config and "dataset" in config["training"]:
            old_path = config["training"]["dataset"].get("dataset_path", "")
            config["training"]["dataset"]["dataset_path"] = old_path.replace("/D4/", f"/{method}/")
        
        if "validation" in config:
            old_val_path = config["validation"].get("dataset_path", "")
            config["validation"]["dataset_path"] = old_val_path.replace("/D4/", f"/{method}/")
            old_val_output = config["validation"].get("output_dir", "")
            config["validation"]["output_dir"] = old_val_output.replace("D4", method_short)
        
        if "training" in config and "checkpointing" in config["training"]:
            old_ckpt = config["training"]["checkpointing"].get("checkpoint_dir", "")
            config["training"]["checkpointing"]["checkpoint_dir"] = old_ckpt.replace("D4", method_short)
        
        if "training" in config and "logging" in config["training"]:
            if "wandb" in config["training"]["logging"]:
                config["training"]["logging"]["wandb"]["exp_name"] = new_name
                config["training"]["logging"]["wandb"]["group"] = method
        
        if "inference" in config:
            old_inf = config["inference"].get("output_dir", "")
            config["inference"]["output_dir"] = old_inf.replace("D4", method_short)
        
        # Write new config
        with open(new_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        generated.append(new_config_path)
        print(f"  Created: {new_config_path.name}")
    
    if skipped:
        print(f"\n  Skipped (already exist): {len(skipped)} configs")
        for s in skipped:
            print(f"    - {s.name}")
    
    print(f"\nGenerated {len(generated)} new configs, {len(skipped)} existing")
    
    # Return all configs (generated + skipped)
    all_configs = generated + skipped
    return sorted(all_configs, key=lambda x: x.name)


def print_training_commands(method: str, configs: list):
    """Print training commands for all configs."""
    print("\n" + "=" * 60)
    print("Step 4: Training Commands")
    print("=" * 60)
    
    print("\n# Copy and run these commands:\n")
    print("cd /home/joon/dev/FaceLift\n")
    
    gpu_assignments = {
        "E1": 0, "E2": 1, "E3": 2, "E4": 3, 
        "E5": 4, "E6": 5, "E7": 6
    }
    
    for config in configs:
        config_name = config.stem
        match = re.search(r'E(\d+)', config_name)
        if match:
            exp_num = f"E{match.group(1)}"
            gpu = gpu_assignments.get(exp_num, 0)
        else:
            gpu = 0
        
        log_name = config_name.lower()
        print(f"# {config_name}")
        print(f"CUDA_VISIBLE_DEVICES={gpu} /home/joon/anaconda3/envs/facelift/bin/torchrun \\")
        print(f"    --standalone --nproc_per_node=1 train_gslrm.py \\")
        print(f"    --config configs/mouse/{config.name} > logs/{log_name}.log 2>&1 &")
        print()


def main():
    parser = argparse.ArgumentParser(description="Unified Preprocessing Pipeline")
    parser.add_argument("--method", required=True, choices=["D6-1", "D6-2", "D6-3"],
                        help="Preprocessing method")
    parser.add_argument("--input_dir", default="/home/joon/data/raw/markerless_mouse_1_nerf",
                        help="Input raw data directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: auto)")
    parser.add_argument("--config_dir", default="/home/joon/dev/FaceLift/configs/mouse",
                        help="Config directory")
    parser.add_argument("--force", action="store_true",
                        help="Force re-preprocessing even if data exists")
    parser.add_argument("--skip_configs", action="store_true",
                        help="Skip config generation")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"/home/joon/data/preprocessed/FaceLift_mouse/{args.method}"
    
    print("\n" + "#" * 60)
    print(f"# Mouse Preprocessing Pipeline: {args.method}")
    print("#" * 60)
    
    # Check existing data
    existing = check_existing_data(args.output_dir)
    
    # Step 1: Preprocessing (skip if data exists)
    if existing["exists"] and not args.force:
        print("\n" + "=" * 60)
        print("Step 1: Preprocessing SKIPPED (data exists)")
        print("=" * 60)
        print(f"  Found: {existing['train_samples']} train, {existing['val_samples']} val samples")
        print(f"  Location: {args.output_dir}")
        print("  Use --force to re-run preprocessing")
    else:
        if args.force and existing["exists"]:
            print(f"\n  Force mode: re-running preprocessing...")
        success = run_preprocessing(args.method, args.input_dir, args.output_dir)
        if not success:
            print("\nPreprocessing failed!")
            return 1
    
    # Step 2: Verification
    stats = run_verification(args.output_dir)
    if not stats["success"]:
        print("\nVerification failed!")
        return 1
    
    if stats["all_256"]:
        print("\n[WARNING] PP values are all 256 - this indicates the D4 bug!")
    
    # Step 3: Generate configs
    if not args.skip_configs:
        configs = generate_experiment_configs(args.method, args.output_dir, args.config_dir)
    else:
        method_short = args.method.replace("-", "")
        configs = sorted(Path(args.config_dir).glob(f"{method_short}_E*.yaml"))
        print(f"\nUsing existing configs: {len(configs)} found")
    
    # Step 4: Print training commands
    if configs:
        print_training_commands(args.method, configs)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Dataset: {args.output_dir}")
    print(f"  Configs: {len(configs)} experiments ready")
    print(f"  PP Status: {'CORRECT (varying)' if not stats['all_256'] else 'BUG (all 256)'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
