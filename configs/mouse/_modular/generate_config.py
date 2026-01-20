#!/usr/bin/env python3
"""
Modular Config Generator
========================

Combines base + schema + dataset to generate full experiment config.

Usage:
    python generate_config.py --dataset D7_t --schema 5v_alpha
    python generate_config.py --dataset D7_1 --schema 5v_alpha --exp_num 1
    python generate_config.py --list  # List available options
    
Output:
    configs/mouse/{dataset}_E{N}_{schema}.yaml
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import re


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> Dict:
    """Load YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_next_exp_num(dataset: str, schema: str, output_dir: Path) -> int:
    """Find next available experiment number."""
    pattern = re.compile(rf"{dataset}_E(\d+)_{schema}\.yaml")
    max_num = 0
    for f in output_dir.glob(f"{dataset}_E*_{schema}.yaml"):
        match = pattern.match(f.name)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def generate_config(
    dataset: str,
    schema: str,
    exp_num: int = None,
    output_dir: Path = None
) -> Path:
    """Generate experiment config from modular components."""
    
    modular_dir = Path(__file__).parent
    base_dir = modular_dir / "base"
    schema_dir = modular_dir / "schemas"
    dataset_dir = modular_dir / "datasets"
    
    if output_dir is None:
        output_dir = modular_dir.parent
    
    # Load base configs
    model_cfg = load_yaml(base_dir / "model.yaml")
    runtime_cfg = load_yaml(base_dir / "runtime.yaml")
    
    # Load schema
    schema_path = schema_dir / f"{schema}.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema}")
    schema_cfg = load_yaml(schema_path)
    
    # Load dataset
    dataset_path = dataset_dir / f"{dataset}.yaml"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    dataset_cfg = load_yaml(dataset_path)
    
    # Get experiment number
    if exp_num is None:
        exp_num = get_next_exp_num(dataset, schema, output_dir)
    
    # Merge configs
    config = {}
    config = deep_merge(config, runtime_cfg)
    config = deep_merge(config, model_cfg)
    config = deep_merge(config, schema_cfg)
    config = deep_merge(config, dataset_cfg)
    
    # Remove metadata fields
    config.pop('_schema', None)
    config.pop('_dataset', None)
    
    # Set experiment-specific paths
    exp_name = f"{dataset}_E{exp_num}_{schema}"
    config['training']['checkpointing']['checkpoint_dir'] = f"checkpoints/gslrm/{exp_name}"
    config['training']['logging']['wandb']['group'] = dataset
    config['training']['logging']['wandb']['job_type'] = f"E{exp_num}_{schema}"
    config['training']['logging']['wandb']['exp_name'] = exp_name
    config['validation']['output_dir'] = f"experiments/validation/{exp_name}"
    config['inference']['output_dir'] = f"experiments/inference/{exp_name}"
    
    # Add header comment
    header = f"""# {exp_name}
# Generated from: dataset={dataset}, schema={schema}
# 
# Dataset: {dataset_cfg.get('_dataset', {}).get('description', 'N/A')}
# Schema: {schema_cfg.get('_schema', {}).get('description', 'N/A')}
#
# To regenerate: python _modular/generate_config.py --dataset {dataset} --schema {schema} --exp_num {exp_num}
"""
    
    # Write config
    output_path = output_dir / f"{exp_name}.yaml"
    with open(output_path, 'w') as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return output_path


def list_options(modular_dir: Path):
    """List available datasets and schemas."""
    print("\n=== Available Datasets ===")
    for f in sorted((modular_dir / "datasets").glob("*.yaml")):
        cfg = load_yaml(f)
        meta = cfg.get('_dataset', {})
        status = meta.get('status', 'READY')
        print(f"  {f.stem:12} - {meta.get('description', 'N/A')} [{status}]")
    
    print("\n=== Available Schemas ===")
    for f in sorted((modular_dir / "schemas").glob("*.yaml")):
        cfg = load_yaml(f)
        meta = cfg.get('_schema', {})
        print(f"  {f.stem:12} - {meta.get('description', 'N/A')}")
    
    print("\n=== Example Usage ===")
    print("  python generate_config.py --dataset D7_t --schema 5v_alpha")
    print("  python generate_config.py --dataset D7_1 --schema 5v_alpha --exp_num 1")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate experiment config from modular components")
    parser.add_argument('--dataset', '-d', help='Dataset name (e.g., D7, D7_t, D7_1)')
    parser.add_argument('--schema', '-s', help='Schema name (e.g., 5v_alpha, 4v_random)')
    parser.add_argument('--exp_num', '-n', type=int, help='Experiment number (auto if not specified)')
    parser.add_argument('--list', '-l', action='store_true', help='List available options')
    
    args = parser.parse_args()
    modular_dir = Path(__file__).parent
    
    if args.list or (not args.dataset and not args.schema):
        list_options(modular_dir)
        return
    
    if not args.dataset or not args.schema:
        parser.error("Both --dataset and --schema are required")
    
    output_path = generate_config(args.dataset, args.schema, args.exp_num)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
