"""Main evaluator for temporal deformation results with WandB integration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class EvaluationConfig:
    original_dir: str
    smoothed_dir: str
    output_dir: str
    gslrm_config: Optional[str] = None
    gslrm_checkpoint: Optional[str] = None
    dataset_path: Optional[str] = None
    max_frames: Optional[int] = None
    render_resolution: int = 512
    video_fps: int = 30
    turntable_views: int = 60
    wandb_project: str = 'FaceLift-Mouse'
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    use_wandb: bool = True
    device: str = 'cuda'


class TemporalEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_run = None
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
    
    def _init_wandb(self):
        run_name = self.config.wandb_run_name or f"temporal_eval_{Path(self.config.original_dir).parent.name}"
        tags = self.config.wandb_tags or ['temporal', 'deformation', 'evaluation']
        
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            tags=tags,
            config={
                'original_dir': self.config.original_dir,
                'smoothed_dir': self.config.smoothed_dir,
                'max_frames': self.config.max_frames,
            },
        )
        print(f"WandB initialized: {self.wandb_run.url}")
    
    def run_full_evaluation(self) -> Dict:
        print("\n" + "=" * 60)
        print("TEMPORAL DEFORMATION EVALUATION")
        print("=" * 60)
        
        results = {}
        
        # 1. Compute metrics
        print("\n[1/2] Computing quantitative metrics...")
        metrics = self._compute_metrics()
        results['metrics'] = metrics
        
        # 2. Plot metrics
        print("\n[2/2] Plotting metrics...")
        plot_path = self._plot_metrics(metrics)
        results['metrics_plot'] = plot_path
        
        # Log to WandB
        if self.wandb_run:
            self._log_to_wandb(results)
        
        # Print summary
        print("\n" + "=" * 60)
        print(metrics.summary())
        print("=" * 60)
        print(f"\nOutputs saved to: {self.output_dir}")
        
        return results
    
    def _compute_metrics(self):
        from .temporal_metrics import TemporalMetrics
        
        metrics = TemporalMetrics(device=self.config.device)
        result = metrics.compute_from_files(
            self.config.original_dir,
            self.config.smoothed_dir,
            max_frames=self.config.max_frames,
        )
        
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Saved metrics to {metrics_path}")
        
        return result
    
    def _plot_metrics(self, metrics):
        from .temporal_visualizer import TemporalVisualizer
        
        vis = TemporalVisualizer(
            gslrm_config=self.config.gslrm_config or '',
            gslrm_checkpoint=self.config.gslrm_checkpoint or '',
            device=self.config.device,
        )
        
        plot_path = str(self.output_dir / 'metrics_plot.png')
        vis.plot_metrics_over_time(metrics, plot_path)
        return plot_path
    
    def _log_to_wandb(self, results: Dict):
        metrics = results['metrics']
        
        # Log scalar metrics
        wandb.log(metrics.to_dict())
        
        # Log plot
        if results.get('metrics_plot'):
            wandb.log({'temporal/metrics_plot': wandb.Image(results['metrics_plot'])})
        
        # Log curves
        frames = list(range(len(metrics.per_frame_displacement)))
        
        # Create table for displacement
        disp_table = wandb.Table(
            columns=['frame', 'displacement'],
            data=[[i, d] for i, d in enumerate(metrics.per_frame_displacement)]
        )
        wandb.log({'temporal/displacement_table': disp_table})
        
        # Create table for jitter comparison
        jitter_data = [[i, b, a] for i, (b, a) in 
                       enumerate(zip(metrics.per_frame_jitter_before, 
                                    metrics.per_frame_jitter_after))]
        jitter_table = wandb.Table(
            columns=['frame', 'jitter_before', 'jitter_after'],
            data=jitter_data
        )
        wandb.log({'temporal/jitter_table': jitter_table})
        
        print(f"Results logged to WandB: {self.wandb_run.url}")
    
    def finish(self):
        if self.wandb_run:
            wandb.finish()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate temporal deformation")
    parser.add_argument('--original_dir', type=str, required=True)
    parser.add_argument('--smoothed_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gslrm_config', type=str, default=None)
    parser.add_argument('--gslrm_checkpoint', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--wandb_project', type=str, default='FaceLift-Mouse')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
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
    results = evaluator.run_full_evaluation()
    evaluator.finish()
    
    return results


if __name__ == '__main__':
    main()
