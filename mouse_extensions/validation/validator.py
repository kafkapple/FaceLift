"""
Validation Runner - Extracted from gslrm.py for modularity.

Handles:
- Metric computation (PSNR, LPIPS, SSIM, Mask IoU)
- Visualization saving (GT vs Pred, Alpha comparison)
- Turntable rendering (video, grid, dataset views)
"""

import os
import numpy as np
import cv2
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
    MOUSE_CAMERA_ORDER,
    create_dataset_views_video,
)

# From gslrm
from gslrm.model.gaussians_renderer import (
    render_turntable,
    render_dataset_views,
    render_dataset_trajectory,
    imageseq2video,
    add_row_labels_to_grid,
    add_left_row_labels,
)


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
        from gslrm.model.utils_metrics import compute_psnr, compute_lpips, compute_ssim, compute_mask_iou
        
        os.makedirs(output_directory, exist_ok=True)
        input_data, target_data = model_results.input, model_results.target
        validation_metrics = {"psnr": [], "lpips": [], "ssim": [], "mask_iou": []}
        
        for batch_idx in range(input_data.image.size(0)):
            item_uid = input_data.index[batch_idx, 0, -1].item()
            should_save_visuals = (batch_idx == 0) and save_visualizations
            
            # Compute metrics
            metrics = self._compute_batch_metrics(
                target_data, model_results, batch_idx,
                compute_psnr, compute_lpips, compute_ssim, compute_mask_iou
            )
            
            for key in validation_metrics:
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
        self, target_data, model_results, batch_idx,
        compute_psnr, compute_lpips, compute_ssim, compute_mask_iou
    ) -> Dict[str, Any]:
        """Compute metrics for a single batch item."""
        full_target = target_data.image[batch_idx]
        target_image = full_target[:, :3, ...]
        rendered = model_results.render[batch_idx]
        
        gt_mask = full_target[:, 3:4, :, :] if full_target.size(1) == 4 else None
        
        per_view_psnr = compute_psnr(target_image, rendered, mask=gt_mask)
        per_view_lpips = compute_lpips(target_image, rendered, mask=gt_mask)
        per_view_ssim = compute_ssim(target_image, rendered, mask=gt_mask)
        
        mask_iou = 0.0
        if gt_mask is not None:
            per_view_iou = compute_mask_iou(rendered, gt_mask, bg_threshold=0.1)
            mask_iou = per_view_iou.mean().item()
        
        return {
            "psnr": per_view_psnr.mean().item(),
            "lpips": per_view_lpips.mean().item(),
            "ssim": per_view_ssim.mean().item(),
            "mask_iou": mask_iou,
            "per_view_psnr": per_view_psnr.cpu().tolist(),
            "per_view_lpips": per_view_lpips.cpu().tolist(),
            "per_view_ssim": per_view_ssim.cpu().tolist(),
        }
    
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
        
        mask_mode = self.config.training.losses.get("mask_mode", None)
        num_rows = 5 if (full_target.size(1) == 4 and mask_mode != "none") else 3
        
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
        """Create turntable video, grid, and dataset views."""
        cfg = self.config.get("visualization", {}).get("turntable", {})
        
        render_res = input_np.shape[0]
        input_res = input_data.image.size(3)
        
        smooth = cfg.get("smooth_trajectory", True)
        camera_order = cfg.get("camera_order", MOUSE_CAMERA_ORDER)
        loop = cfg.get("loop", True)
        fps = cfg.get("fps", 30)
        num_views = cfg.get("video_views", 144)
        
        c2ws = target_data.c2w[batch_idx].cpu().numpy()
        fxfycxcy = target_data.fxfycxcy[batch_idx].cpu().numpy()
        gaussians = model_results.gaussians[batch_idx]
        
        # Render frames
        if smooth:
            frames, _ = render_dataset_trajectory(
                gaussians, c2ws, fxfycxcy,
                rendering_resolution=render_res, num_views=num_views,
                camera_order=camera_order, loop=loop,
                show_overlay=False, original_resolution=input_res,
            )
        else:
            center = gaussians._xyz.mean(dim=0).detach().cpu().numpy()
            turntable_img = render_turntable(
                gaussians, rendering_resolution=render_res, num_views=num_views, center=center
            )
            frames = rearrange(turntable_img, "h (v w) c -> v h w c", v=num_views)
        
        frames = np.ascontiguousarray(frames)
        
        # Save video
        imageseq2video(frames, os.path.join(output_dir, "turntable.mp4"), fps=fps)
        
        # Save grid
        self._save_grid(frames, cfg, item_uid, output_dir)
        
        # Save with input overlay
        self._save_with_input(frames, input_np, render_res, fps, output_dir)
        
        # Dataset views
        if cfg.get("save_dataset_views", False):
            self._save_dataset_views(gaussians, c2ws, fxfycxcy, render_res, input_res, cfg, item_uid, output_dir)
        
        # Orbit turntable (standard 360-degree rotation)
        if cfg.get("save_orbit_turntable", True):
            self._save_orbit_turntable(gaussians, render_res, fps, cfg, item_uid, output_dir, input_np)
    

    def _save_orbit_turntable(self, gaussians, render_res, fps, cfg, item_uid, output_dir, input_np):
        """Save standard 360-degree orbit turntable video."""
        try:
            orbit_views = cfg.get("orbit_views", 120)
            orbit_elevation = cfg.get("elevation", 20)
            
            # Use opacity-weighted center for better focus
            xyz = gaussians._xyz.detach()
            opacity = gaussians.get_opacity.detach().squeeze()
            weights = opacity / (opacity.sum() + 1e-8)
            center = (xyz * weights.unsqueeze(-1)).sum(dim=0).cpu().numpy()
            
            # Use same radius as normalized data (~2.7)
            orbit_radius = cfg.get("orbit_radius", cfg.get("radius", 2.7))
            
            orbit_img = render_turntable(
                gaussians,
                rendering_resolution=render_res,
                num_views=orbit_views,
                elevation=orbit_elevation,
                radius=orbit_radius,
                center=center,
            )
            orbit_frames = rearrange(orbit_img, "h (v w) c -> v h w c", v=orbit_views)
            orbit_frames = np.ascontiguousarray(orbit_frames)
            
            # Save orbit video
            imageseq2video(orbit_frames, os.path.join(output_dir, f"turntable_orbit_{item_uid}.mp4"), fps=fps)
            
            # Save orbit with input strip
            border = 2
            target_h = int(input_np.shape[0] / input_np.shape[1] * render_res)
            resized = cv2.resize(input_np, (render_res - border * 2, target_h - border * 2), interpolation=cv2.INTER_AREA)
            bordered = np.pad(resized, ((border, border), (border, border), (0, 0)), mode="constant", constant_values=200)
            input_seq = np.tile(bordered[None], (orbit_frames.shape[0], 1, 1, 1))
            combined = np.concatenate((orbit_frames, input_seq), axis=1)
            imageseq2video(combined, os.path.join(output_dir, f"turntable_orbit_with_input_{item_uid}.mp4"), fps=fps)
        except Exception as e:
            print(f"Warning: Could not save orbit turntable: {e}")

    def _save_grid(self, frames, cfg, item_uid, output_dir):
        """Create and save turntable grid."""
        rows = cfg.get("grid_rows", 6)
        cols = cfg.get("grid_cols", 6)
        n_grid = rows * cols
        n_frames = frames.shape[0]
        
        if n_frames > n_grid:
            indices = np.linspace(0, n_frames - 1, n_grid, dtype=int)
            grid_frames = frames[indices]
        else:
            grid_frames = frames[:n_grid]
        
        h = grid_frames.shape[1]
        grid = rearrange(grid_frames, "(r c) h w ch -> (r h) (c w) ch", r=rows, c=cols)
        
        if cfg.get("add_row_labels", True):
            cam_order = cfg.get("camera_order", MOUSE_CAMERA_ORDER)
            if cfg.get("label_position", "left") == "left":
                grid = add_left_row_labels(grid, cam_order, rows, cols, h)
            else:
                grid = add_row_labels_to_grid(grid, cam_order, rows, cols, h)
        
        Image.fromarray(grid).save(os.path.join(output_dir, f"turntable_{item_uid}.jpg"))
    
    def _save_with_input(self, frames, input_np, render_res, fps, output_dir):
        """Save turntable with input overlay."""
        border = 2
        target_h = int(input_np.shape[0] / input_np.shape[1] * render_res)
        
        resized = cv2.resize(input_np, (render_res - border * 2, target_h - border * 2), interpolation=cv2.INTER_AREA)
        bordered = np.pad(resized, ((border, border), (border, border), (0, 0)), mode="constant", constant_values=200)
        
        input_seq = np.tile(bordered[None], (frames.shape[0], 1, 1, 1))
        combined = np.concatenate((frames, input_seq), axis=1)
        
        imageseq2video(combined, os.path.join(output_dir, "turntable_with_input.mp4"), fps=fps)
    
    def _save_dataset_views(self, gaussians, c2ws, fxfycxcy, render_res, input_res, cfg, item_uid, output_dir):
        """Save rendered dataset camera views."""
        try:
            views = render_dataset_views(
                gaussians, c2ws, fxfycxcy,
                rendering_resolution=render_res,
                show_overlay=False, original_resolution=input_res,
            )
            strip = rearrange(views, "v h w c -> h (v w) c")
            Image.fromarray(strip).save(os.path.join(output_dir, f"dataset_views_{item_uid}.jpg"))
            
            if cfg.get("save_dataset_views_video", False):
                create_dataset_views_video(views, os.path.join(output_dir, f"dataset_views_{item_uid}.mp4"), cfg, imageseq2video)
        except Exception as e:
            print(f"Warning: Could not save dataset_views: {e}")
    
    def _aggregate_results(self, metrics: Dict) -> Dict[str, float]:
        """Aggregate validation metrics."""
        result = {
            "psnr": torch.tensor(metrics["psnr"]).mean().item(),
            "lpips": torch.tensor(metrics["lpips"]).mean().item(),
            "ssim": torch.tensor(metrics["ssim"]).mean().item(),
            "mask_iou": torch.tensor(metrics["mask_iou"]).mean().item(),
        }
        
        if "per_view_psnr" in metrics and metrics["per_view_psnr"]:
            result["per_view_psnr"] = metrics["per_view_psnr"]
            result["per_view_lpips"] = metrics["per_view_lpips"]
            result["per_view_ssim"] = metrics["per_view_ssim"]
        
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
