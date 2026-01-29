#!/usr/bin/env python3
"""Unified inference CLI.

Usage:
    # With config file
    python -m mouse_extensions.inference.run --config configs/inference/default.yaml
    
    # With overrides
    python -m mouse_extensions.inference.run \
        --config configs/inference/default.yaml \
        input.data_dir=/path/to/data \
        input.frame_range.end=100
    
    # Simple mode (legacy compatible)
    python -m mouse_extensions.inference.run \
        --data_dir /path/to/data \
        --checkpoint M5t_E0_1_facelift \
        --output_dir outputs/test
"""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified FaceLift Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/inference/default.yaml",
        help="Path to config YAML file",
    )
    
    # Simple mode arguments (legacy compatible)
    parser.add_argument("--data_dir", type=str, help="Data directory (batch mode)")
    parser.add_argument("--sample_dir", type=str, help="Sample directory (single mode)")
    parser.add_argument("--input_image", type=str, help="Input image path")
    parser.add_argument("--view_idx", type=int, help="Input view index (0-5)")
    parser.add_argument("--start_frame", type=int, help="Start frame index")
    parser.add_argument("--end_frame", type=int, help="End frame index")
    parser.add_argument("--frame_step", type=int, help="Frame step")
    parser.add_argument("--checkpoint", type=str, help="GS-LRM checkpoint")
    parser.add_argument("--mvdiffusion_checkpoint", type=str, help="MVDiffusion checkpoint")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory")
    parser.add_argument("--resolution", type=int, help="Rendering resolution")
    parser.add_argument("--num_views", type=int, help="Number of turntable views")
    parser.add_argument("--fps", type=int, help="Video FPS")
    parser.add_argument("--no_video", action="store_true", help="Disable video output")
    parser.add_argument("--no_gaussian", action="store_true", help="Disable Gaussian export")
    parser.add_argument("--no_rerun", action="store_true", help="Disable Rerun export")
    
    # Catch remaining args as config overrides
    args, overrides = parser.parse_known_args()
    
    return args, overrides


def build_config(args, overrides):
    """Build config from file and command line arguments."""
    # Load base config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / args.config
    
    if config_path.exists():
        config = OmegaConf.load(config_path)
        
        # Handle _base_ inheritance
        if "_base_" in config:
            base_path = config_path.parent / config._base_
            if base_path.exists():
                base_config = OmegaConf.load(base_path)
                del config._base_
                config = OmegaConf.merge(base_config, config)
    else:
        # Start with empty config
        config = OmegaConf.create()
    
    # Apply simple mode arguments
    if args.data_dir:
        OmegaConf.update(config, "input.mode", "batch")
        OmegaConf.update(config, "input.data_dir", args.data_dir)
    if args.sample_dir:
        OmegaConf.update(config, "input.mode", "single_sample")
        OmegaConf.update(config, "input.sample_dir", args.sample_dir)
    if args.input_image:
        OmegaConf.update(config, "input.mode", "single_image")
        OmegaConf.update(config, "input.image_path", args.input_image)
    if args.view_idx is not None:
        OmegaConf.update(config, "input.view_idx", args.view_idx)
    if args.start_frame is not None:
        OmegaConf.update(config, "input.frame_range.start", args.start_frame)
    if args.end_frame is not None:
        OmegaConf.update(config, "input.frame_range.end", args.end_frame)
    if args.frame_step is not None:
        OmegaConf.update(config, "input.frame_range.step", args.frame_step)
    if args.checkpoint:
        OmegaConf.update(config, "pipeline.gslrm.checkpoint", args.checkpoint)
    if args.mvdiffusion_checkpoint:
        OmegaConf.update(config, "pipeline.mvdiffusion.checkpoint", args.mvdiffusion_checkpoint)
        OmegaConf.update(config, "pipeline.mvdiffusion.enabled", True)
    if args.output_dir:
        OmegaConf.update(config, "output.dir", args.output_dir)
    if args.resolution:
        OmegaConf.update(config, "rendering.resolution", args.resolution)
    if args.num_views:
        OmegaConf.update(config, "rendering.num_views", args.num_views)
    if args.fps:
        OmegaConf.update(config, "output.video.fps", args.fps)
    if args.no_video:
        OmegaConf.update(config, "output.video.enabled", False)
    if args.no_gaussian:
        OmegaConf.update(config, "output.gaussian", False)
    if args.no_rerun:
        OmegaConf.update(config, "output.rerun", False)
    
    # Apply dotlist overrides (e.g., input.data_dir=/path)
    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_conf)
    
    return config


def main():
    args, overrides = parse_args()
    config = build_config(args, overrides)
    
    # Print config summary
    print("=" * 60)
    print("FaceLift Unified Inference Pipeline")
    print("=" * 60)
    print(f"Mode: {config.input.mode}")
    print(f"MVDiffusion: {config.pipeline.mvdiffusion.enabled}")
    print(f"GS-LRM: {config.pipeline.gslrm.checkpoint}")
    print(f"Output: {config.output.dir}")
    print("=" * 60)
    
    # Run pipeline
    from mouse_extensions.inference.unified_pipeline import UnifiedPipeline
    
    pipeline = UnifiedPipeline(config)
    
    try:
        result = pipeline.run()
        
        print("\n" + "=" * 60)
        print("Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print("=" * 60)
        
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
