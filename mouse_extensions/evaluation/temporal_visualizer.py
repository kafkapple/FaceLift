"""Visualization tools for temporal deformation evaluation."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


class TemporalVisualizer:
    """Visualize temporal deformation results."""
    
    def __init__(
        self,
        gslrm_config: str,
        gslrm_checkpoint: str,
        device: str = 'cuda',
    ):
        self.device = device
        self.gslrm_config = gslrm_config
        self.gslrm_checkpoint = gslrm_checkpoint
        
        # Lazy load renderer
        self._renderer = None
        print("Visualizer initialized (renderer will be loaded on first use)")
    
    def _get_renderer(self):
        """Lazy load the renderer."""
        if self._renderer is None:
            from gslrm.model.gaussians_renderer import GaussianRenderer
            self._renderer = GaussianRenderer()
        return self._renderer
    
    def render_gaussian(
        self,
        gaussian_dict: Dict,
        c2w: torch.Tensor,
        fxfycxcy: torch.Tensor,
        resolution: int = 512,
    ) -> np.ndarray:
        """Render a single Gaussian to image."""
        from gslrm.model.gaussians_renderer import GaussianModel
        
        renderer = self._get_renderer()
        
        # Reconstruct GaussianModel
        gm = GaussianModel(sh_degree=2)
        gm._xyz = gaussian_dict['xyz'].to(self.device)
        
        # Split features into dc and rest
        features = gaussian_dict['features'].to(self.device)
        n_gaussians = features.shape[0]
        
        # SH features: degree 2 = (2+1)^2 = 9 coefficients per channel
        # For 3 channels: 9*3 = 27 total, but stored as [N, C, 3]
        # DC: [N, 1, 3], Rest: [N, 8, 3]
        if len(features.shape) == 2:
            if features.shape[1] == 27:
                # Stored as [N, 27] - reshape to [N, 9, 3] then split
                features_reshaped = features.reshape(n_gaussians, 9, 3)
                gm._features_dc = features_reshaped[:, :1, :]  # [N, 1, 3]
                gm._features_rest = features_reshaped[:, 1:, :]  # [N, 8, 3]
            else:
                gm._features_dc = features[:, :3].unsqueeze(1)
                gm._features_rest = None
        else:
            gm._features_dc = features[:, :1]
            gm._features_rest = features[:, 1:] if features.shape[1] > 1 else None
        
        gm._scaling = gaussian_dict['scaling'].to(self.device)
        gm._rotation = gaussian_dict['rotation'].to(self.device)
        gm._opacity = gaussian_dict['opacity'].to(self.device)
        
        # Ensure correct shapes
        if len(c2w.shape) == 2:
            c2w = c2w.unsqueeze(0)
        if len(fxfycxcy.shape) == 1:
            fxfycxcy = fxfycxcy.unsqueeze(0)
        
        c2w = c2w.to(self.device)
        fxfycxcy = fxfycxcy.to(self.device)
        
        # Render
        with torch.no_grad():
            result = renderer.render(
                gm,
                c2w,
                fxfycxcy,
                resolution,
                resolution,
            )
        
        # Convert to numpy
        image = result['render'].squeeze().permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)
        
        return image
    
    def create_comparison_frame(
        self,
        original: np.ndarray,
        smoothed: np.ndarray,
        frame_idx: int,
        metrics: Optional[Dict] = None,
    ) -> np.ndarray:
        """Create side-by-side comparison frame."""
        h, w = original.shape[:2]
        
        # Create comparison (side by side)
        comparison = np.zeros((h, w * 2, 3), dtype=np.float32)
        comparison[:, :w] = original
        comparison[:, w:] = smoothed
        
        # Convert to uint8 for text overlay
        comparison_uint8 = (comparison * 255).astype(np.uint8)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison_uint8, f'Original (Frame {frame_idx})', 
                    (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison_uint8, f'Smoothed (Frame {frame_idx})', 
                    (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Add metrics if provided
        if metrics:
            y_offset = 60
            for key, value in metrics.items():
                text = f'{key}: {value:.4f}'
                cv2.putText(comparison_uint8, text, (10, y_offset), 
                           font, 0.5, (255, 255, 0), 1)
                y_offset += 20
        
        return comparison_uint8
    
    def create_comparison_video(
        self,
        original_dir: str,
        smoothed_dir: str,
        output_path: str,
        camera_params: Dict,
        max_frames: Optional[int] = None,
        fps: int = 30,
        resolution: int = 512,
    ):
        """Create before/after comparison video."""
        orig_dir = Path(original_dir)
        smooth_dir = Path(smoothed_dir)
        
        orig_files = sorted(orig_dir.glob('*.pt'))
        smooth_files = sorted(smooth_dir.glob('*.pt'))
        
        if max_frames:
            orig_files = orig_files[:max_frames]
            smooth_files = smooth_files[:max_frames]
        
        n_frames = min(len(orig_files), len(smooth_files))
        print(f"Creating comparison video for {n_frames} frames...")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (resolution * 2, resolution))
        
        c2w = camera_params['c2w']
        fxfycxcy = camera_params['fxfycxcy']
        
        for i in tqdm(range(n_frames), desc="Rendering comparison"):
            orig_data = torch.load(orig_files[i], weights_only=True)
            smooth_data = torch.load(smooth_files[i], weights_only=True)
            
            # Render both
            orig_img = self.render_gaussian(orig_data, c2w, fxfycxcy, resolution)
            smooth_img = self.render_gaussian(smooth_data, c2w, fxfycxcy, resolution)
            
            # Compute frame metrics
            disp = (smooth_data['xyz'] - orig_data['xyz']).norm(dim=-1).mean().item()
            frame_metrics = {'displacement': disp}
            
            # Create comparison frame
            comp = self.create_comparison_frame(orig_img, smooth_img, i, frame_metrics)
            
            # Write frame (BGR for OpenCV)
            out.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Saved comparison video to {output_path}")
    
    def create_turntable_comparison(
        self,
        original_gaussian: Dict,
        smoothed_gaussian: Dict,
        output_path: str,
        n_views: int = 60,
        fps: int = 30,
        resolution: int = 512,
        radius: float = 2.7,
    ):
        """Create turntable video comparing original vs smoothed."""
        print(f"Creating turntable comparison with {n_views} views...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (resolution * 2, resolution))
        
        for i in tqdm(range(n_views), desc="Turntable"):
            angle = 2 * np.pi * i / n_views
            
            # Create camera for this angle
            c2w, fxfycxcy = self._create_orbit_camera(angle, radius, resolution)
            
            # Render both
            orig_img = self.render_gaussian(original_gaussian, c2w, fxfycxcy, resolution)
            smooth_img = self.render_gaussian(smoothed_gaussian, c2w, fxfycxcy, resolution)
            
            # Compute displacement
            disp = (smoothed_gaussian['xyz'] - original_gaussian['xyz']).norm(dim=-1).mean().item()
            
            # Create comparison
            comp = self.create_comparison_frame(orig_img, smooth_img, i, {'displacement': disp})
            out.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"Saved turntable comparison to {output_path}")
    
    def _create_orbit_camera(
        self,
        angle: float,
        radius: float,
        resolution: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create camera parameters for orbit view."""
        # Camera position on orbit
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = 0.5  # Slight elevation
        
        # Look at origin
        eye = np.array([x, y, z])
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        
        # Camera matrix (look-at)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # c2w (camera to world)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, -up, forward], axis=1)
        c2w[:3, 3] = eye
        
        # Intrinsics
        fx = fy = 549.0
        cx = cy = resolution / 2
        
        return (
            torch.tensor(c2w, dtype=torch.float32),
            torch.tensor([fx, fy, cx, cy], dtype=torch.float32),
        )
    
    def plot_metrics_over_time(
        self,
        metrics_result,
        output_path: str,
    ):
        """Plot metrics over time and save figure."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Displacement over time
        ax = axes[0, 0]
        ax.plot(metrics_result.per_frame_displacement)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mean Displacement')
        ax.set_title('Displacement per Frame')
        ax.grid(True)
        
        # Jitter comparison
        ax = axes[0, 1]
        ax.plot(metrics_result.per_frame_jitter_before, label='Before', alpha=0.7)
        ax.plot(metrics_result.per_frame_jitter_after, label='After', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Jitter')
        ax.set_title(f'Temporal Jitter (Reduction: {metrics_result.jitter_reduction:.1f}%)')
        ax.legend()
        ax.grid(True)
        
        # Jitter histogram
        ax = axes[1, 0]
        ax.hist(metrics_result.per_frame_jitter_before, bins=30, alpha=0.5, label='Before')
        ax.hist(metrics_result.per_frame_jitter_after, bins=30, alpha=0.5, label='After')
        ax.set_xlabel('Jitter')
        ax.set_ylabel('Count')
        ax.set_title('Jitter Distribution')
        ax.legend()
        
        # Summary text
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f'''
Summary Statistics

Displacement:
  Mean: {metrics_result.mean_displacement:.6f}
  Std:  {metrics_result.std_displacement:.6f}
  Max:  {metrics_result.max_displacement:.6f}

Temporal Jitter:
  Before: {metrics_result.temporal_jitter_before:.6f}
  After:  {metrics_result.temporal_jitter_after:.6f}
  Reduction: {metrics_result.jitter_reduction:.1f}%
'''
        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved metrics plot to {output_path}")
