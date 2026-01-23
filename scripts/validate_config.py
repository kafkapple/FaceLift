#!/usr/bin/env python3
"""Config validation script for FaceLift experiments.

Usage:
    python scripts/validate_config.py configs/mouse/D7_1_E2_gt_alpha.yaml
    python scripts/validate_config.py --all  # Validate all D7_1 configs
"""

import yaml
import sys
import os
from pathlib import Path

def validate_config(cfg_path: str) -> list:
    """Validate config structure and return list of errors."""
    errors = []
    warnings = []
    
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return [f"Failed to load: {e}"], []
    
    # 1. Model section required keys
    model = cfg.get("model", {})
    if not model:
        errors.append("Missing model section")
    else:
        if "num_views" not in model:
            errors.append("model.num_views required")
        if "num_input_views" not in model:
            errors.append("model.num_input_views required")
        if model.get("num_input_views", 0) > model.get("num_views", 0):
            errors.append("num_input_views cannot exceed num_views")
    
    # 2. Training section
    training = cfg.get("training", {})
    if not training:
        errors.append("Missing training section")
    else:
        dataset = training.get("dataset", {})
        if "data_list" not in dataset:
            errors.append("training.dataset.data_list required")
        
        # Wrong location check
        if "num_input_views" in dataset:
            errors.append("WRONG: num_input_views should be in model, not training.dataset")
        if "num_views" in dataset:
            errors.append("WRONG: num_views should be in model, not training.dataset")
        
        losses = training.get("losses", {})
        if not losses:
            warnings.append("No losses section - will use defaults")
    
    # 3. Validation section
    if "validation" not in cfg:
        warnings.append("No validation section")
    
    # 4. wandb section
    wandb = cfg.get("wandb", {})
    if wandb:
        cfg_name = Path(cfg_path).stem
        wandb_name = wandb.get("name", "")
        if wandb_name != cfg_name:
            warnings.append(f"wandb.name {wandb_name} != filename {cfg_name}")
    
    return errors, warnings

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config.yaml> [--all]")
        sys.exit(1)
    
    if sys.argv[1] == "--all":
        configs = list(Path("configs/mouse").glob("D7_1_E*.yaml"))
        configs = [c for c in configs if not c.is_symlink()]
    else:
        configs = [Path(sys.argv[1])]
    
    print("Config Validation Report")
    print("=" * 60)
    
    all_passed = True
    for cfg_path in sorted(configs):
        errors, warnings = validate_config(str(cfg_path))
        
        status = "PASS" if not errors else "FAIL"
        print(f"[{status}] {cfg_path.name}")
        
        for err in errors:
            print(f"    ERROR: {err}")
            all_passed = False
        for warn in warnings:
            print(f"    WARN: {warn}")
    
    print("=" * 60)
    if all_passed:
        print("All configs passed validation!")
    else:
        print("Some configs have errors - please fix before running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
