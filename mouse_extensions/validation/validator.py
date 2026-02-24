"""
Validation Runner - Extracted from gslrm.py for modularity.

Handles:
- Metric computation (PSNR, LPIPS, SSIM, Mask IoU)
- Visualization saving (GT vs Pred, Alpha comparison)
- Turntable rendering (video, grid, dataset views)
"""

import os
import numpy as np
import torch
from typing import Dict, Any
from PIL import Image
from einops import rearrange

# From mouse_extensions
from mouse_extensions.model.visualization_extensions import (
    VisualizationConfig,
    create_validation_visual,
)
from mouse_extensions.visualization import (
    visualize_alpha_comparison,
    compute_alpha_metrics,
    should_visualize_alpha,
)

# From gslrm






class ValidationRunner:
    """Handles validation logic extracted from GSLRM model."""
    
    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config
    
    def run(
        self,
        output_directory: str,
        model_results: Any,
        batch_data: Any,
        dataset: Any,
        save_visualizations: bool = False
    ) -> Dict[str, float]:
        """Run validation and save results."""
        from mouse_extensions.evaluation import MetricsComputer
        skip_lpips = self.config.get('validation', {}).get('skip_lpips', False)
        # Determine device: use CPU if tensors are already on CPU (low-VRAM mode)
        metrics_device = 'cpu' if (hasattr(model_results, 'render') and
                                    isinstance(model_results.render, torch.Tensor) and
                                    not model_results.render.is_cuda) else 'cuda'
        metrics_computer = MetricsComputer(
            device=metrics_device,
            compute_lpips=not skip_lpips,
        )
        
        os.makedirs(output_directory, exist_ok=True)
        input_data, target_data = model_results.input, model_results.target
        validation_metrics = {"psnr": [], "lpips": [], "ssim": [], "mask_iou": [], "l1": []}
        
        for batch_idx in range(input_data.image.size(0)):
            item_uid = input_data.index[batch_idx, 0, -1].item()
            should_save_visuals = (batch_idx == 0) and save_visualizations
            
            # Compute metrics
            metrics = self._compute_batch_metrics(
                target_data, model_results, batch_idx, metrics_computer
            )
            
            for key in validation_metrics:
                if key in metrics:
                    validation_metrics[key].append(metrics[key])
            
            if batch_idx == 0:
                validation_metrics["per_view_psnr"] = metrics["per_view_psnr"]
                validation_metrics["per_view_lpips"] = metrics["per_view_lpips"]
                validation_metrics["per_view_ssim"] = metrics["per_view_ssim"]
            
            if should_save_visuals:
                self._save_visualizations(
                    output_directory, item_uid, batch_idx,
                    input_data, target_data, model_results, metrics
                )
        
        return self._aggregate_results(validation_metrics)
    
    def _compute_batch_metrics(
        self, target_data, model_results, batch_idx, metrics_computer
    ) -> Dict[str, Any]:
        """
        Compute metrics for a single batch item.

        Uses MetricsComputer to avoid the "3-place modification" bug:
        adding new metrics only requires updating MetricsComputer class.
        """
        full_target = target_data.image[batch_idx]
        target_image = full_target[:, :3, ...]
        rendered = model_results.render[batch_idx]
        gt_mask = full_target[:, 3:4, :, :] if full_target.size(1) == 4 else None

        # Extract rendered alpha for mask_iou (always computed when available)
        batch_alpha = None
        if hasattr(model_results, 'rendered_alpha') and model_results.rendered_alpha is not None:
            batch_alpha = model_results.rendered_alpha[batch_idx]

        return metrics_computer.compute_per_view_metrics(
            target_image, rendered, gt_mask, rendered_alpha=batch_alpha
        )
    
    def _save_visualizations(
        self, output_directory, item_uid, batch_idx,
        input_data, target_data, model_results, metrics
    ):
        """Save all visualization outputs."""
        item_dir = os.path.join(output_directory, f"{item_uid:08d}")
        os.makedirs(item_dir, exist_ok=True)
        
        # Input image (for turntable overlay)
        input_np = self._save_input_image(input_data, batch_idx, item_dir)
        
        # GT vs Pred comparison
        self._save_comparison(target_data, model_results, batch_idx, item_dir)
        
        # Alpha comparison
        self._save_alpha_comparison(target_data, model_results, batch_idx, item_uid, item_dir)
        
        # Metrics files
        self._save_metrics_files(target_data, batch_idx, metrics, item_uid, item_dir)
        
        # Gaussian PLY
        self._save_gaussian_ply(model_results, batch_idx, item_dir)
        
        # Turntable
        self._create_turntable(input_data, target_data, model_results, batch_idx, item_uid, item_dir, input_np)
    
    def _save_input_image(self, input_data, batch_idx, output_dir) -> np.ndarray:
        """Save concatenated input views."""
        input_image = rearrange(
            input_data.image[batch_idx][:, :3, ...],
            "v c h w -> h (v w) c"
        )
        input_np = (input_image.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(input_np).save(os.path.join(output_dir, "input.png"))
        return input_np
    
    def _save_comparison(self, target_data, model_results, batch_idx, output_dir):
        """Save GT vs Prediction comparison."""
        full_target = target_data.image[batch_idx]
        rendered = model_results.render[batch_idx]
        h = full_target.size(2)
        
        batch_alpha = None
        if hasattr(model_results, 'rendered_alpha') and model_results.rendered_alpha is not None:
            batch_alpha = model_results.rendered_alpha[batch_idx]
        
        vis_config = VisualizationConfig.from_training_config(self.config)
        view_ids = target_data.index[batch_idx, :, 0].cpu().numpy().tolist()
        
        comparison, error_stats = create_validation_visual(
            full_target, rendered, vis_config,
            rendered_alpha=batch_alpha, view_indices=view_ids
        )
        
        # Always show mask overlay when alpha data is available
        has_mask_data = full_target.size(1) == 4 or batch_alpha is not None
        num_rows = 5 if has_mask_data else 3
        
        comparison = self.model.loss_calculator._add_error_scale_annotation(
            comparison, error_stats, h, num_rows
        )
        Image.fromarray(comparison).save(os.path.join(output_dir, "gt_vs_pred.png"))
    
    def _save_alpha_comparison(self, target_data, model_results, batch_idx, item_uid, output_dir):
        """Save alpha mask comparison if enabled."""
        if not should_visualize_alpha(self.config):
            return
        
        try:
            full_target = target_data.image[batch_idx]
            batch_alpha = getattr(model_results, 'rendered_alpha', None)
            if batch_alpha is not None:
                batch_alpha = batch_alpha[batch_idx]
            
            if batch_alpha is not None and full_target.size(1) == 4:
                gt_mask = full_target[:, 3:4, :, :]
                alpha_vis = visualize_alpha_comparison(
                    gt_mask, batch_alpha, num_views=gt_mask.size(0), threshold=0.5
                )
                Image.fromarray(alpha_vis).save(os.path.join(output_dir, f"alpha_comparison_{item_uid}.jpg"))
                
                alpha_metrics = compute_alpha_metrics(gt_mask, batch_alpha)
                with open(os.path.join(output_dir, "alpha_metrics.txt"), "w") as f:
                    for k, v in alpha_metrics.items():
                        f.write(f"{k}: {v:.4f}\n")
        except Exception as e:
            print(f"Warning: Could not save alpha comparison: {e}")
    
    def _save_metrics_files(self, target_data, batch_idx, metrics, item_uid, output_dir):
        """Save per-view and averaged metrics."""
        view_ids = target_data.index[batch_idx, :, 0].cpu().numpy()
        
        with open(os.path.join(output_dir, "perview_metrics.txt"), "w") as f:
            for i, vid in enumerate(view_ids):
                f.write(f"view {vid:0>6}, psnr: {metrics['per_view_psnr'][i]:.4f}, "
                        f"lpips: {metrics['per_view_lpips'][i]:.4f}, ssim: {metrics['per_view_ssim'][i]:.4f}\n")
        
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(f"psnr: {metrics['psnr']:.4f}\nlpips: {metrics['lpips']:.4f}\nssim: {metrics['ssim']:.4f}\n")
        
        print(f"Validation UID {item_uid}: PSNR={metrics['psnr']:.4f}, LPIPS={metrics['lpips']:.4f}, SSIM={metrics['ssim']:.4f}")
    
    def _save_gaussian_ply(self, model_results, batch_idx, output_dir):
        """Save filtered Gaussian model."""
        crop_box = None
        if self.config.model.get("clip_xyz", False):
            half_size = self.config.model.get("half_bbx_size", 0.91)
            crop_box = [-half_size, half_size] * 3
        
        model_results.gaussians[batch_idx].apply_all_filters(
            opacity_thres=0.02, crop_bbx=crop_box, cam_origins=None, nearfar_percent=(0.0001, 1.0)
        ).save_ply(os.path.join(output_dir, "gaussians.ply"))
    
    def _create_turntable(self, input_data, target_data, model_results, batch_idx, item_uid, output_dir, input_np):
        # Skip turntable on low-VRAM GPUs to avoid OOM
        if self.config.get('validation', {}).get('skip_turntable', False):
            print(f"  Skipping turntable for UID {item_uid} (skip_turntable=True)")
            return
        """Create turntable video, grid, and orbit views (unified via TurntableRenderer)."""
        from mouse_extensions.visualization.turntable_renderer import TurntableRenderer, TurntableVideoConfig

        turntable_video_cfg = TurntableVideoConfig.from_config(self.config)
        render_res = input_np.shape[0]
        input_res = input_data.image.size(3)

        c2ws = target_data.c2w[batch_idx].cpu().numpy()
        fxfycxcy = target_data.fxfycxcy[batch_idx].cpu().numpy()

        # Get view indices for proper camera->tensor mapping
        view_indices = None
        if hasattr(target_data, "index") and target_data.index is not None:
            view_indices = target_data.index[batch_idx, :, 0].cpu().numpy().tolist()

        # Get actual input camera indices
        if hasattr(input_data, "index") and input_data.index is not None:
            input_indices = input_data.index[batch_idx, :, 0].cpu().numpy().tolist()
        else:
            input_indices = list(range(input_data.image.shape[1]))

        renderer = TurntableRenderer(turntable_video_cfg)
        renderer.render_all(
            gaussians=model_results.gaussians[batch_idx],
            output_dir=output_dir,
            uid=str(item_uid),
            rendering_resolution=render_res,
            dataset_c2ws=c2ws,
            dataset_fxfycxcy=fxfycxcy,
            original_resolution=input_res,
            target_images=target_data.image[batch_idx],
            input_indices=input_indices,
            view_indices=view_indices,
        )

    def _aggregate_results(self, metrics: Dict) -> Dict[str, float]:
        """Aggregate validation metrics."""
        result = {
            "psnr": torch.tensor(metrics["psnr"]).mean().item(),
            "lpips": torch.tensor(metrics["lpips"]).mean().item(),
            "ssim": torch.tensor(metrics["ssim"]).mean().item(),
            "mask_iou": torch.tensor(metrics["mask_iou"]).mean().item(),
            "l1": torch.tensor(metrics.get("l1", [0.0])).mean().item(),
        }
        
        if "per_view_psnr" in metrics and metrics["per_view_psnr"]:
            result["per_view_psnr"] = metrics["per_view_psnr"]
            result["per_view_lpips"] = metrics["per_view_lpips"]
            result["per_view_ssim"] = metrics["per_view_ssim"]
            if "per_view_l1" in metrics:
                result["per_view_l1"] = metrics["per_view_l1"]
        
        return result


def run_validation(
    model: Any,
    output_directory: str,
    model_results: Any,
    batch_data: Any,
    dataset: Any,
    save_visualizations: bool = False
) -> Dict[str, float]:
    """Convenience function for running validation."""
    runner = ValidationRunner(model, model.config)
    return runner.run(output_directory, model_results, batch_data, dataset, save_visualizations)
