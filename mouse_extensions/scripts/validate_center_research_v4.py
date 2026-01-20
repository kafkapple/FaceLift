#!/usr/bin/env python3
"""
Multi-View Center Estimation: Comprehensive Statistical Analysis v4

Improvements:
1. Show corrected centers on actual images
2. Better 3D visualization with camera poses and orientations
3. Camera parameter table (original, corrected, FaceLift settings)
4. Detailed per-view mask overlay comparison
5. Fixed metrics (cross-view std instead of deviation from self)
6. Unified y-axis scales
7. Summary table with all parameters and results
"""

import sys
sys.path.insert(0, '/home/joon/dev/FaceLift')

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

from mouse_extensions.preprocessing.center_estimation import CenterEstimator
from mouse_extensions.preprocessing.data_loader import DataLoader


def get_2d_centroid(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return np.array([256.0, 256.0])
    return np.array([float(xs.mean()), float(ys.mean())])


def get_camera_center(cam: Dict) -> np.ndarray:
    R = np.array(cam['R'])
    T = np.array(cam['T']).flatten()
    return -R.T @ T


def get_camera_axes(cam: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get camera X, Y, Z axes in world coordinates."""
    R = np.array(cam['R'])
    # Camera axes are rows of R transposed (columns of R)
    x_axis = R[0, :]  # Right
    y_axis = R[1, :]  # Down
    z_axis = R[2, :]  # Forward (looking direction)
    return R.T @ np.array([1,0,0]), R.T @ np.array([0,1,0]), R.T @ np.array([0,0,1])


class ComprehensiveValidator:
    METHODS = ['no_correction', 'per_view_2d', 'triangulation', 'visual_hull']
    COLORS = {
        'no_correction': '#f39c12',
        'per_view_2d': '#e74c3c', 
        'triangulation': '#2ecc71',
        'visual_hull': '#3498db'
    }
    MARKERS = {
        'no_correction': 's',
        'per_view_2d': 'x',
        'triangulation': 'o',
        'visual_hull': '^'
    }
    
    def __init__(self, data_dir: str, output_dir: str, num_samples: int = 5):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        
        self.loader = DataLoader(data_dir, source_type='raw', num_views=6)
        self.cameras = self.loader.cameras
        self.n_views = len(self.cameras)
        
        total = self.loader.total_frames
        step = max(1, total // num_samples)
        self.frame_indices = list(range(0, total, step))[:num_samples]
        
        self.frame_results = []
        self.summary = {}
        
    def run(self):
        print(f"\n{'='*70}")
        print("Multi-View Center Estimation: Comprehensive Analysis v4")
        print(f"{'='*70}\n")
        
        for frame_idx in self.frame_indices:
            result = self._analyze_frame(frame_idx)
            if result:
                self.frame_results.append(result)
                print(f"Frame {frame_idx}: processed")
        
        self._compute_summary()
        self._generate_all_figures()
        self._generate_report()
        
        print(f"\nReport: {self.output_dir / 'report.html'}")
        
    def _analyze_frame(self, frame_idx: int) -> Optional[Dict]:
        try:
            images, masks = self.loader.load_frame(frame_idx)
            if images is None:
                return None
            
            # Original camera parameters
            original_params = []
            for cam in self.cameras:
                original_params.append({
                    'fx': cam['fx'], 'fy': cam['fy'],
                    'cx': cam['cx'], 'cy': cam['cy'],
                })
            
            # Method 1: No correction - image center
            centers_no_corr = np.array([[256.0, 256.0]] * self.n_views)
            
            # Method 2: Per-view 2D centroid
            centers_pv2d = np.array([get_2d_centroid(m) for m in masks])
            
            # Method 3: Triangulation
            tri_est = CenterEstimator(self.cameras, method='triangulation')
            tri_result = tri_est.estimate(masks)
            centers_tri = tri_result.centers_2d
            center_3d_tri = tri_result.center_3d
            
            # Method 4: Visual hull
            vh_est = CenterEstimator(self.cameras, method='visual_hull')
            vh_result = vh_est.estimate(masks)
            centers_vh = vh_result.centers_2d
            center_3d_vh = vh_result.center_3d
            
            all_centers = {
                'no_correction': centers_no_corr,
                'per_view_2d': centers_pv2d,
                'triangulation': centers_tri,
                'visual_hull': centers_vh,
            }
            
            # Compute metrics for each method
            metrics = {}
            for method, centers in all_centers.items():
                # Cross-view consistency: std of distances from mean
                center_mean = centers.mean(axis=0)
                dists_from_mean = np.linalg.norm(centers - center_mean, axis=1)
                
                # Per-view spread (how much centers differ across views)
                pairwise_dists = []
                for i in range(self.n_views):
                    for j in range(i+1, self.n_views):
                        pairwise_dists.append(np.linalg.norm(centers[i] - centers[j]))
                
                # Distance from triangulation reference (for non-triangulation methods)
                dist_from_tri = np.linalg.norm(centers - centers_tri, axis=1)
                
                metrics[method] = {
                    'centers_2d': centers.tolist(),
                    'center_mean': center_mean.tolist(),
                    'cross_view_std': float(np.std(dists_from_mean)),
                    'cross_view_range': float(np.max(dists_from_mean) - np.min(dists_from_mean)),
                    'pairwise_dist_mean': float(np.mean(pairwise_dists)) if pairwise_dists else 0,
                    'pairwise_dist_max': float(np.max(pairwise_dists)) if pairwise_dists else 0,
                    'dist_from_tri_per_view': dist_from_tri.tolist(),
                    'dist_from_tri_mean': float(dist_from_tri.mean()),
                }
            
            return {
                'frame_idx': frame_idx,
                'metrics': metrics,
                'centers_3d': {
                    'triangulation': center_3d_tri.tolist(),
                    'visual_hull': center_3d_vh.tolist(),
                },
                'original_params': original_params,
                'images': images,
                'masks': masks,
            }
            
        except Exception as e:
            print(f"Error frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_summary(self):
        for method in self.METHODS:
            cross_view_stds = [r['metrics'][method]['cross_view_std'] for r in self.frame_results]
            pairwise_means = [r['metrics'][method]['pairwise_dist_mean'] for r in self.frame_results]
            dist_from_tri = [r['metrics'][method]['dist_from_tri_mean'] for r in self.frame_results]
            
            self.summary[method] = {
                'cross_view_std': {
                    'mean': float(np.mean(cross_view_stds)),
                    'std': float(np.std(cross_view_stds)),
                },
                'pairwise_dist': {
                    'mean': float(np.mean(pairwise_means)),
                    'std': float(np.std(pairwise_means)),
                },
                'dist_from_tri': {
                    'mean': float(np.mean(dist_from_tri)),
                    'std': float(np.std(dist_from_tri)),
                },
                'n_samples': len(cross_view_stds),
            }
    
    def _generate_all_figures(self):
        if not self.frame_results:
            return
        
        self._fig1_centers_on_images()
        self._fig2_3d_camera_scene()
        self._fig3_method_comparison()
        self._fig4_per_view_detailed()
        self._fig5_mask_overlay()
        
    def _fig1_centers_on_images(self):
        """Figure 1: Centers overlaid on actual images for each method."""
        r = self.frame_results[0]
        images = r['images']
        
        fig, axes = plt.subplots(4, 6, figsize=(24, 16))
        
        for row, method in enumerate(self.METHODS):
            centers = np.array(r['metrics'][method]['centers_2d'])
            color = self.COLORS[method]
            
            for col in range(self.n_views):
                ax = axes[row, col]
                img = images[col]
                if img.max() <= 1:
                    img = (img * 255).astype(np.uint8)
                ax.imshow(img)
                
                # Draw center point
                ax.scatter(centers[col, 0], centers[col, 1], 
                          c=color, s=300, marker='o', edgecolors='white', linewidths=2)
                
                # Draw crosshair
                ax.axhline(y=centers[col, 1], color=color, alpha=0.5, linestyle='--')
                ax.axvline(x=centers[col, 0], color=color, alpha=0.5, linestyle='--')
                
                # Annotate coordinates
                ax.text(10, 30, f'({centers[col,0]:.1f}, {centers[col,1]:.1f})', 
                       fontsize=9, color='white', backgroundcolor=color)
                
                if row == 0:
                    ax.set_title(f'View {col}', fontsize=12)
                if col == 0:
                    ax.set_ylabel(method, fontsize=11, fontweight='bold')
                
                ax.set_xlim(0, 512)
                ax.set_ylim(512, 0)
                ax.axis('off')
        
        plt.suptitle(f'Figure 1: Center Positions by Method (Frame {r["frame_idx"]})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_centers_on_images.png', dpi=150)
        plt.close()
        
    def _fig2_3d_camera_scene(self):
        """Figure 2: 3D camera setup with poses and rays."""
        r = self.frame_results[0]
        
        fig = plt.figure(figsize=(20, 8))
        
        # Camera positions and orientations
        cam_positions = np.array([get_camera_center(c) for c in self.cameras])
        
        for plot_idx, (method, title) in enumerate([
            ('per_view_2d', 'Per-View 2D: Original Centroids'),
            ('triangulation', 'Triangulation: Back-Projected Centers'),
        ]):
            ax = fig.add_subplot(1, 2, plot_idx + 1, projection='3d')
            
            centers_2d = np.array(r['metrics'][method]['centers_2d'])
            center_3d = np.array(r['centers_3d'].get('triangulation', [0,0,0]))
            
            # World coordinate axes
            axis_len = 30
            ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=3)
            ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.1, linewidth=3)
            ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.1, linewidth=3)
            ax.text(axis_len*1.2, 0, 0, 'X', fontsize=12, color='red', fontweight='bold')
            ax.text(0, axis_len*1.2, 0, 'Y', fontsize=12, color='green', fontweight='bold')
            ax.text(0, 0, axis_len*1.2, 'Z', fontsize=12, color='blue', fontweight='bold')
            
            # Draw cameras with orientation
            cam_colors = plt.cm.tab10(np.linspace(0, 1, self.n_views))
            for i, cam in enumerate(self.cameras):
                C = cam_positions[i]
                
                # Camera position
                ax.scatter([C[0]], [C[1]], [C[2]], c=[cam_colors[i]], s=150, marker='^')
                
                # Camera forward direction (looking direction)
                R = np.array(cam['R'])
                forward = R.T @ np.array([0, 0, 1])  # Z-axis in camera frame
                forward = forward / np.linalg.norm(forward) * 50
                ax.quiver(C[0], C[1], C[2], forward[0], forward[1], forward[2],
                         color=cam_colors[i], arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
                
                # Label
                ax.text(C[0], C[1], C[2]+15, f'C{i}', fontsize=10, ha='center')
                
                # Ray to 2D center
                K = np.array(cam['K'])
                K_inv = np.linalg.inv(K)
                pt2d = centers_2d[i]
                ray_cam = K_inv @ np.array([pt2d[0], pt2d[1], 1.0])
                ray_cam = ray_cam / np.linalg.norm(ray_cam)
                ray_world = R.T @ ray_cam
                
                ray_len = 250
                end = C + ray_world * ray_len
                
                ray_color = 'lime' if method == 'triangulation' else 'red'
                ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]],
                       c=ray_color, alpha=0.5, linewidth=1.5)
            
            # 3D center
            ax.scatter([center_3d[0]], [center_3d[1]], [center_3d[2]],
                      c='gold', s=400, marker='*', edgecolors='black', linewidths=2,
                      label='3D Center', zorder=10)
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(title, fontsize=12)
            ax.legend(loc='upper right')
            ax.view_init(elev=25, azim=45)
        
        plt.suptitle('Figure 2: 3D Camera Setup & Ray Visualization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_3d_camera_scene.png', dpi=150)
        plt.close()
        
    def _fig3_method_comparison(self):
        """Figure 3: Method comparison with unified y-axis."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = self.METHODS
        colors = [self.COLORS[m] for m in methods]
        
        # Find global max for y-axis
        max_pairwise = max(self.summary[m]['pairwise_dist']['mean'] + 
                          self.summary[m]['pairwise_dist']['std'] for m in methods)
        max_dist_tri = max(self.summary[m]['dist_from_tri']['mean'] + 
                          self.summary[m]['dist_from_tri']['std'] for m in methods)
        y_max = max(max_pairwise, max_dist_tri) * 1.2
        
        # Plot 1: Pairwise distance (spread across views)
        ax = axes[0]
        means = [self.summary[m]['pairwise_dist']['mean'] for m in methods]
        stds = [self.summary[m]['pairwise_dist']['std'] for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_ylabel('Pairwise Distance (px)')
        ax.set_title('Mean Pairwise Distance\n(How spread out are centers across views)')
        ax.set_ylim(0, y_max)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f}', ha='center', fontsize=9)
        
        # Plot 2: Distance from triangulation
        ax = axes[1]
        means = [self.summary[m]['dist_from_tri']['mean'] for m in methods]
        stds = [self.summary[m]['dist_from_tri']['std'] for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_ylabel('Distance from Triangulation (px)')
        ax.set_title('Distance from Triangulation Reference\n(Triangulation = 0 by definition)')
        ax.set_ylim(0, y_max)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{val:.1f}', ha='center', fontsize=9)
        
        # Plot 3: Cross-view std
        ax = axes[2]
        means = [self.summary[m]['cross_view_std']['mean'] for m in methods]
        stds = [self.summary[m]['cross_view_std']['std'] for m in methods]
        bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right')
        ax.set_ylabel('Cross-View Std (px)')
        ax.set_title('Cross-View Standard Deviation\n(Variance of center positions)')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', fontsize=9)
        
        plt.suptitle('Figure 3: Method Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_method_comparison.png', dpi=150)
        plt.close()
        
    def _fig4_per_view_detailed(self):
        """Figure 4: Per-view analysis with consistent y-axis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # Find global max across all methods and views
        global_max = 0
        for method in self.METHODS:
            for r in self.frame_results:
                dists = r['metrics'][method]['dist_from_tri_per_view']
                global_max = max(global_max, max(dists))
        global_max *= 1.3
        
        for idx, method in enumerate(self.METHODS):
            ax = axes[idx]
            
            # Collect per-view distances from triangulation
            per_view_data = [[] for _ in range(self.n_views)]
            for r in self.frame_results:
                dists = r['metrics'][method]['dist_from_tri_per_view']
                for v in range(self.n_views):
                    per_view_data[v].append(dists[v])
            
            # Box plot
            bp = ax.boxplot(per_view_data, tick_labels=[f'V{i}' for i in range(self.n_views)],
                           patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(self.COLORS[method])
                patch.set_alpha(0.7)
            
            ax.set_xlabel('View')
            ax.set_ylabel('Distance from Triangulation (px)')
            ax.set_title(f'{method}\n(Mean: {self.summary[method]["dist_from_tri"]["mean"]:.1f}px)', 
                        fontsize=11)
            ax.set_ylim(0, global_max)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Note for triangulation
            if method == 'triangulation':
                ax.text(0.5, 0.5, 'Reference\n(0 by definition)', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='gray', alpha=0.7)
        
        plt.suptitle('Figure 4: Per-View Distance from Triangulation Reference', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_per_view_analysis.png', dpi=150)
        plt.close()
        
    def _fig5_mask_overlay(self):
        """Figure 5: Detailed mask overlay with all centers."""
        r = self.frame_results[0]
        masks = r['masks']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for v in range(self.n_views):
            ax = axes[v]
            
            # Show mask
            mask = masks[v]
            if mask.ndim == 3:
                mask = mask.mean(axis=-1)
            ax.imshow(mask, cmap='gray', alpha=0.7)
            
            # Plot all method centers
            for method in self.METHODS:
                center = r['metrics'][method]['centers_2d'][v]
                ax.scatter(center[0], center[1], 
                          c=self.COLORS[method], s=200, 
                          marker=self.MARKERS[method],
                          edgecolors='white' if method != 'per_view_2d' else self.COLORS[method],
                          linewidths=2, label=method if v == 0 else None,
                          zorder=10)
            
            # Draw lines between per_view_2d and triangulation to show the difference
            pv_center = r['metrics']['per_view_2d']['centers_2d'][v]
            tri_center = r['metrics']['triangulation']['centers_2d'][v]
            diff = np.linalg.norm(np.array(pv_center) - np.array(tri_center))
            
            ax.plot([pv_center[0], tri_center[0]], [pv_center[1], tri_center[1]],
                   'b--', linewidth=2, alpha=0.7)
            ax.text((pv_center[0]+tri_center[0])/2, (pv_center[1]+tri_center[1])/2 - 15,
                   f'{diff:.1f}px', fontsize=10, color='blue', ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'View {v}', fontsize=12)
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)
            ax.axis('off')
        
        # Legend
        handles = [plt.scatter([], [], c=self.COLORS[m], marker=self.MARKERS[m], 
                              s=100, label=m) for m in self.METHODS]
        fig.legend(handles, self.METHODS, loc='lower center', ncol=4, fontsize=11)
        
        plt.suptitle(f'Figure 5: Mask Overlay with All Centers (Frame {r["frame_idx"]})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(self.output_dir / 'fig5_mask_overlay.png', dpi=150)
        plt.close()
        
    def _generate_report(self):
        """Generate comprehensive HTML report."""
        r = self.frame_results[0]
        
        # Camera parameters table
        cam_table_rows = []
        for v in range(self.n_views):
            cam = self.cameras[v]
            C = get_camera_center(cam)
            cam_table_rows.append(f'''
            <tr>
                <td>View {v}</td>
                <td>{cam['fx']:.1f}</td>
                <td>{cam['fy']:.1f}</td>
                <td>{cam['cx']:.1f}</td>
                <td>{cam['cy']:.1f}</td>
                <td>({C[0]:.1f}, {C[1]:.1f}, {C[2]:.1f})</td>
            </tr>''')
        
        # Summary metrics table
        metrics_rows = []
        for method in self.METHODS:
            s = self.summary[method]
            # Get sample corrected cx, cy (varies by frame and view, show first frame mean)
            centers = np.array(r['metrics'][method]['centers_2d'])
            cx_mean, cy_mean = centers.mean(axis=0)
            
            metrics_rows.append(f'''
            <tr>
                <td><span style="color:{self.COLORS[method]}">{method}</span></td>
                <td>{cx_mean:.1f}</td>
                <td>{cy_mean:.1f}</td>
                <td>{s['pairwise_dist']['mean']:.2f} ± {s['pairwise_dist']['std']:.2f}</td>
                <td>{s['dist_from_tri']['mean']:.2f} ± {s['dist_from_tri']['std']:.2f}</td>
                <td>{s['cross_view_std']['mean']:.2f} ± {s['cross_view_std']['std']:.2f}</td>
            </tr>''')
        
        # Per-view detail table
        detail_rows = []
        for v in range(self.n_views):
            row = f'<tr><td>View {v}</td>'
            for method in self.METHODS:
                center = r['metrics'][method]['centers_2d'][v]
                dist = r['metrics'][method]['dist_from_tri_per_view'][v]
                row += f'<td>({center[0]:.1f}, {center[1]:.1f})<br><small>Δ={dist:.1f}px</small></td>'
            row += '</tr>'
            detail_rows.append(row)
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Center Estimation Analysis v4</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ border-bottom: 3px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #333; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
        th, td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
        th {{ background: #f5f5f5; }}
        img {{ max-width: 100%; margin: 15px 0; border: 1px solid #ddd; }}
        .highlight {{ background: #fffde7; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .note {{ background: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0; font-size: 12px; }}
        .metadata {{ color: #666; font-size: 12px; }}
    </style>
</head>
<body>

<h1>Multi-View Center Estimation: Comprehensive Analysis</h1>
<p class="metadata">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Samples: {len(self.frame_results)} frames</p>

<h2>1. Camera Parameters (Original)</h2>
<table>
<tr><th>View</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Position (X,Y,Z) mm</th></tr>
{''.join(cam_table_rows)}
</table>
<p class="note"><strong>FaceLift Training Default:</strong> fx=fy=549, cx=cy=256, distance=2.7</p>

<h2>2. Method Comparison Summary</h2>
<table>
<tr>
    <th>Method</th>
    <th>Mean cx (px)</th>
    <th>Mean cy (px)</th>
    <th>Pairwise Dist (px)</th>
    <th>Dist from Tri (px)</th>
    <th>Cross-View Std (px)</th>
</tr>
{''.join(metrics_rows)}
</table>

<div class="highlight">
<strong>Key Finding:</strong> per_view_2d differs from triangulation by <strong>{self.summary['per_view_2d']['dist_from_tri']['mean']:.1f}px</strong> on average.
This inconsistency propagates to 3D reconstruction as ghosting.
</div>

<h2>3. Per-View Center Coordinates (Frame {r['frame_idx']})</h2>
<table>
<tr><th>View</th><th style="color:{self.COLORS['no_correction']}">no_correction</th>
<th style="color:{self.COLORS['per_view_2d']}">per_view_2d</th>
<th style="color:{self.COLORS['triangulation']}">triangulation</th>
<th style="color:{self.COLORS['visual_hull']}">visual_hull</th></tr>
{''.join(detail_rows)}
</table>

<h2>4. Visual Analysis</h2>

<h3>Figure 1: Centers on Images</h3>
<img src="fig1_centers_on_images.png" alt="Centers on Images">
<p class="note">Each row shows center positions for one method across all 6 views. Crosshairs indicate the center location.</p>

<h3>Figure 2: 3D Camera Setup</h3>
<img src="fig2_3d_camera_scene.png" alt="3D Camera Scene">
<p class="note">Camera positions (triangles) with forward direction arrows. Rays show lines from camera to estimated centers. 
Left: per_view_2d uses original 2D centroids. Right: triangulation uses back-projected consistent centers.</p>

<h3>Figure 3: Method Comparison</h3>
<img src="fig3_method_comparison.png" alt="Method Comparison">

<h3>Figure 4: Per-View Analysis</h3>
<img src="fig4_per_view_analysis.png" alt="Per-View Analysis">
<p class="note">Triangulation is the reference (0 by definition). Other methods show their distance from triangulation.</p>

<h3>Figure 5: Mask Overlay</h3>
<img src="fig5_mask_overlay.png" alt="Mask Overlay">
<p class="note">Blue dashed lines show the difference between per_view_2d (red X) and triangulation (green O) for each view.</p>

<h2>5. Interpretation</h2>
<ul>
<li><strong>no_correction</strong>: Fixed at image center (256, 256) - high deviation because mouse is not centered</li>
<li><strong>per_view_2d</strong>: Each view uses its own 2D centroid - causes {self.summary['per_view_2d']['dist_from_tri']['mean']:.1f}px cross-view inconsistency</li>
<li><strong>triangulation</strong>: Computes unified 3D center, back-projects to consistent 2D - reference method</li>
<li><strong>visual_hull</strong>: Similar to triangulation but uses full silhouette - slightly different 3D center possible</li>
</ul>

<hr>
<p class="metadata">Report generated by validate_center_research_v4.py</p>
</body>
</html>'''
        
        with open(self.output_dir / 'report.html', 'w') as f:
            f.write(html)
        
        # Save JSON
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(self.frame_results),
            },
            'camera_params': [{
                'view': i,
                'fx': c['fx'], 'fy': c['fy'], 'cx': c['cx'], 'cy': c['cy'],
                'position': get_camera_center(c).tolist()
            } for i, c in enumerate(self.cameras)],
            'summary': self.summary,
            'per_frame': [{k: v for k, v in r.items() if k not in ['images', 'masks']} 
                         for r in self.frame_results],
        }
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(output, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/joon/data/markerless_mouse_1_nerf')
    parser.add_argument('--output_dir', default='/home/joon/dev/FaceLift/mouse_extensions/reports/center_analysis_v4')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    validator = ComprehensiveValidator(args.data_dir, args.output_dir, args.num_samples)
    validator.run()


if __name__ == '__main__':
    main()
