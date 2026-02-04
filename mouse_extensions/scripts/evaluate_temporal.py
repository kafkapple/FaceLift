#!/usr/bin/env python3
"""CLI for temporal deformation evaluation."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal deformation")
    
    parser.add_argument('--original_dir', type=str, required=True)
    parser.add_argument('--smoothed_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # GS-LRM (optional for now, rendering not yet implemented)
    parser.add_argument('--gslrm_config', type=str, default=None)
    parser.add_argument('--gslrm_checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    
    # Options
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    
    # WandB
    parser.add_argument('--wandb_project', type=str, default='FaceLift-Mouse')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    from mouse_extensions.evaluation.temporal_evaluator import (
        TemporalEvaluator, EvaluationConfig
    )
    
    config = EvaluationConfig(
        original_dir=args.original_dir,
        smoothed_dir=args.smoothed_dir,
        output_dir=args.output_dir,
        gslrm_config=args.gslrm_config,
        gslrm_checkpoint=args.gslrm_checkpoint,
        dataset_path=args.dataset_path,
        max_frames=args.max_frames,
        render_resolution=args.resolution,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        device=args.device,
    )
    
    evaluator = TemporalEvaluator(config)
    evaluator.run_full_evaluation()
    evaluator.finish()


if __name__ == '__main__':
    main()
