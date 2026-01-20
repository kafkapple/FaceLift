#!/usr/bin/env python3
"""
Preprocessing CLI Entry Point

Usage:
    python -m mouse_extensions.scripts.preprocess -c configs/preprocessing/v12.yaml
    python -m mouse_extensions.scripts.preprocess --version v12 --input_dir /path/to/raw
"""

import argparse
import sys
import os

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mouse_extensions.preprocessing.convert_unified import main as run_preprocessing
from mouse_extensions.preprocessing.run_pipeline import main as run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Mouse Data Preprocessing")
    parser.add_argument("-c", "--config", help="YAML config file path")
    parser.add_argument("--version", choices=["v5", "v10", "v11", "v12"], default="v12",
                       help="Preprocessing version (default: v12)")
    parser.add_argument("--input_dir", help="Input data directory")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--frame_interval", type=int, default=5,
                       help="Frame interval for sampling")
    
    args = parser.parse_args()
    
    if args.config:
        # Use YAML-based pipeline
        sys.argv = ["run_pipeline.py", "-c", args.config]
        run_pipeline()
    elif args.input_dir and args.output_dir:
        # Direct preprocessing
        sys.argv = [
            "convert_unified.py",
            "--version", args.version,
            "--input_dir", args.input_dir,
            "--output_dir", args.output_dir,
            "--frame_interval", str(args.frame_interval)
        ]
        run_preprocessing()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
