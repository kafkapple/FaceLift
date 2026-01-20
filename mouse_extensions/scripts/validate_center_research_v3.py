#!/usr/bin/env python3
"""
Multi-View Center Estimation: Statistical Analysis Report

Objective: Compare center estimation methods without bias.
Methods evaluated:
  1. no_correction: Use image center (256, 256)
  2. per_view_2d: Use 2D centroid from each view's mask
  3. triangulation: DLT triangulation -> back-project to 2D
  4. visual_hull: Shape carving -> back-project to 2D

Metrics:
  - Cross-view consistency (px): Std of crop centers across views
  - Center deviation from triangulation (px): Distance to reference
  - Ray convergence error (mm): 3D accuracy
"""

import sys
sys.path.insert(0, '/home/joon/dev/FaceLift')

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
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
    """Get camera center in world coordinates: C = -R^T @ T"""
    R = np.array(cam['R'])
    T = np.array(cam['T']).flatten()
    return -R.T @ T


def get_camera_forward(cam: Dict) -> np.ndarray:
    """Get camera forward direction (Z-axis in camera frame) in world coords."""
    R = np.array(cam['R'])
    # Camera Z-axis is third row of R (looking direction)
    return R[2, :]


def project_3d_to_2d(point_3d: np.ndarray, cam: Dict) -> np.ndarray:
    """Project 3D point to 2D image coordinates."""
    K = np.array(cam['K'])
    R = np.array(cam['R'])
    T = np.array(cam['T']).flatten()
    
    # Transform to camera coordinates
    point_cam = R @ point_3d + T
    
    # Project to image
    point_2d = K @ point_cam
    point_2d = point_2d[:2] / point_2d[2]
    return point_2d


class StatisticalValidator:
    """Unbiased statistical analysis of center estimation methods."""
    
    METHODS = ['no_correction', 'per_view_2d', 'triangulation', 'visual_hull']
    
    def __init__(self, data_dir: str, output_dir: str, num_samples: int = 10):
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
        print(f"\n{'='*60}")
        print("Multi-View Center Estimation: Statistical Analysis")
        print(f"{'='*60}")
        print(f"Methods: {', '.join(self.METHODS)}")
        print(f"Samples: {self.num_samples} frames")
        print(f"{'='*60}\n")
        
        for frame_idx in self.frame_indices:
            result = self._analyze_frame(frame_idx)
            if result:
                self.frame_results.append(result)
                print(f"Frame {frame_idx:>5}: ", end='')
                for m in self.METHODS:
                    cv = result['metrics'][m]['cross_view_std']
                    print(f"{m[:4]}={cv:.1f}px ", end='')
                print()
        
        self._compute_summary()
        self._generate_visualizations()
        self._generate_report()
        
    def _analyze_frame(self, frame_idx: int) -> Optional[Dict]:
        try:
            images, masks = self.loader.load_frame(frame_idx)
            if images is None:
                return None
            
            # Method 1: No correction - use image center
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
            
            centers_3d = {
                'triangulation': center_3d_tri,
                'visual_hull': center_3d_vh,
            }
            
            # Compute metrics for each method
            metrics = {}
            for method, centers in all_centers.items():
                # Cross-view consistency: std of centers
                center_mean = centers.mean(axis=0)
                distances_from_mean = np.linalg.norm(centers - center_mean, axis=1)
                cross_view_std = float(distances_from_mean.std())
                cross_view_max = float(distances_from_mean.max())
                
                # Deviation from triangulation (reference)
                dev_from_tri = np.linalg.norm(centers - centers_tri, axis=1)
                
                metrics[method] = {
                    'centers_2d': centers.tolist(),
                    'center_mean': center_mean.tolist(),
                    'cross_view_std': cross_view_std,
                    'cross_view_max': cross_view_max,
                    'deviation_from_tri_mean': float(dev_from_tri.mean()),
                    'deviation_from_tri_per_view': dev_from_tri.tolist(),
                }
            
            return {
                'frame_idx': frame_idx,
                'metrics': metrics,
                'centers_3d': {k: v.tolist() for k, v in centers_3d.items()},
                'images': images,  # Keep for visualization
                'masks': masks,
            }
            
        except Exception as e:
            print(f"Error frame {frame_idx}: {e}")
            return None
    
    def _compute_summary(self):
        """Compute summary statistics across all frames."""
        for method in self.METHODS:
            cross_view_stds = [r['metrics'][method]['cross_view_std'] for r in self.frame_results]
            dev_from_tri = [r['metrics'][method]['deviation_from_tri_mean'] for r in self.frame_results]
            
            self.summary[method] = {
                'cross_view_consistency': {
                    'mean': float(np.mean(cross_view_stds)),
                    'std': float(np.std(cross_view_stds)),
                    'min': float(np.min(cross_view_stds)),
                    'max': float(np.max(cross_view_stds)),
                },
                'deviation_from_triangulation': {
                    'mean': float(np.mean(dev_from_tri)),
                    'std': float(np.std(dev_from_tri)),
                },
                'n_samples': len(cross_view_stds),
            }
    
    def _generate_visualizations(self):
        if not self.frame_results:
            return
        
        self._plot_2d_centers_on_images()
        self._plot_3d_camera_scene()
        self._plot_method_comparison()
        self._plot_per_view_analysis()
        
    def _plot_2d_centers_on_images(self):
        """Show 2D centers overlaid on actual images for one frame."""
        r = self.frame_results[0]
        images = r['images']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = {
            'no_correction': ('yellow', 's', 'No Correction'),
            'per_view_2d': ('red', 'x', 'Per-View 2D'),
            'triangulation': ('lime', 'o', 'Triangulation'),
            'visual_hull': ('cyan', '^', 'Visual Hull'),
        }
        
        for i in range(self.n_views):
            ax = axes[i]
            
            # Show image
            img = images[i]
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            
            # Plot centers for each method
            for method, (color, marker, label) in colors.items():
                center = r['metrics'][method]['centers_2d'][i]
                ax.scatter(center[0], center[1], c=color, s=150, marker=marker,
                          edgecolors='black', linewidths=1.5, label=label if i==0 else None,
                          zorder=10)
            
            ax.set_title(f'View {i}', fontsize=12)
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)
            ax.axis('off')
        
        # Add legend
        handles = [plt.scatter([], [], c=c, marker=m, s=100, label=l, edgecolors='black')
                   for method, (c, m, l) in colors.items()]
        fig.legend(handles, [c[2] for c in colors.values()], loc='lower center', ncol=4, fontsize=11)
        
        plt.suptitle(f'2D Center Comparison (Frame {r["frame_idx"]})', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(self.output_dir / 'fig1_2d_centers_on_images.png', dpi=150)
        plt.close()
        
    def _plot_3d_camera_scene(self):
        """3D visualization with cameras, coordinate axes, and estimated centers."""
        r = self.frame_results[0]
        
        fig = plt.figure(figsize=(16, 7))
        
        # Get camera positions
        cam_positions = np.array([get_camera_center(c) for c in self.cameras])
        cam_forwards = np.array([get_camera_forward(c) for c in self.cameras])
        
        # 3D centers
        center_tri = np.array(r['centers_3d']['triangulation'])
        center_vh = np.array(r['centers_3d']['visual_hull'])
        
        for plot_idx, (title, show_rays_to) in enumerate([
            ('Per-View 2D: Rays to Original Centroids', 'per_view_2d'),
            ('Triangulation: Rays to Back-Projected Centers', 'triangulation'),
        ]):
            ax = fig.add_subplot(1, 2, plot_idx + 1, projection='3d')
            
            # Draw world coordinate axes at origin
            axis_len = 50
            ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', arrow_length_ratio=0.1, linewidth=2)
            ax.text(axis_len*1.1, 0, 0, 'X', fontsize=10, color='red')
            ax.text(0, axis_len*1.1, 0, 'Y', fontsize=10, color='green')
            ax.text(0, 0, axis_len*1.1, 'Z', fontsize=10, color='blue')
            
            # Draw cameras
            ax.scatter(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
                      c='blue', s=100, marker='^', label='Cameras')
            
            # Label cameras
            for i, pos in enumerate(cam_positions):
                ax.text(pos[0], pos[1], pos[2]+10, f'C{i}', fontsize=9)
            
            # Draw rays from cameras
            centers_2d = np.array(r['metrics'][show_rays_to]['centers_2d'])
            
            ray_color = 'red' if show_rays_to == 'per_view_2d' else 'lime'
            for i, cam in enumerate(self.cameras):
                C = cam_positions[i]
                
                # Compute ray direction from 2D point
                K = np.array(cam['K'])
                R = np.array(cam['R'])
                K_inv = np.linalg.inv(K)
                point_2d = centers_2d[i]
                ray_cam = K_inv @ np.array([point_2d[0], point_2d[1], 1.0])
                ray_cam = ray_cam / np.linalg.norm(ray_cam)
                ray_world = R.T @ ray_cam
                
                # Draw ray
                ray_len = 300
                end = C + ray_world * ray_len
                ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]],
                       c=ray_color, alpha=0.6, linewidth=2)
            
            # Draw 3D centers
            ax.scatter([center_tri[0]], [center_tri[1]], [center_tri[2]],
                      c='lime', s=200, marker='*', edgecolors='black', linewidths=1,
                      label='Triangulation Center')
            ax.scatter([center_vh[0]], [center_vh[1]], [center_vh[2]],
                      c='cyan', s=150, marker='D', edgecolors='black', linewidths=1,
                      label='Visual Hull Center')
            
            # Mark the relevant center
            if show_rays_to == 'triangulation':
                ax.scatter([center_tri[0]], [center_tri[1]], [center_tri[2]],
                          c='yellow', s=400, marker='o', alpha=0.3, label='Convergence Point')
            
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(title, fontsize=11)
            ax.legend(loc='upper left', fontsize=8)
            
            # Set view angle
            ax.view_init(elev=20, azim=45)
        
        plt.suptitle(f'3D Camera Setup & Ray Visualization (Frame {r["frame_idx"]})', 
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_3d_camera_scene.png', dpi=150)
        plt.close()
        
    def _plot_method_comparison(self):
        """Bar chart comparing all methods."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        methods = self.METHODS
        colors = ['#f39c12', '#e74c3c', '#2ecc71', '#3498db']
        
        # Plot 1: Cross-view consistency
        ax = axes[0]
        means = [self.summary[m]['cross_view_consistency']['mean'] for m in methods]
        stds = [self.summary[m]['cross_view_consistency']['std'] for m in methods]
        bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_ylabel('Cross-View Std (pixels)')
        ax.set_title('Cross-View Consistency\n(Lower = More Consistent)', fontsize=12)
        ax.set_xticklabels(methods, rotation=30, ha='right')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Deviation from triangulation
        ax = axes[1]
        means = [self.summary[m]['deviation_from_triangulation']['mean'] for m in methods]
        stds = [self.summary[m]['deviation_from_triangulation']['std'] for m in methods]
        bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_ylabel('Deviation from Triangulation (pixels)')
        ax.set_title('Distance from Triangulation Reference\n(0 = Same as Triangulation)', fontsize=12)
        ax.set_xticklabels(methods, rotation=30, ha='right')
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_method_comparison.png', dpi=150)
        plt.close()
        
    def _plot_per_view_analysis(self):
        """Per-view breakdown for each method."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        colors = ['#f39c12', '#e74c3c', '#2ecc71', '#3498db']
        
        for idx, method in enumerate(self.METHODS):
            ax = axes[idx]
            
            # Collect per-view deviations across frames
            per_view_data = [[] for _ in range(self.n_views)]
            for r in self.frame_results:
                devs = r['metrics'][method]['deviation_from_tri_per_view']
                for v in range(self.n_views):
                    per_view_data[v].append(devs[v])
            
            # Box plot
            bp = ax.boxplot(per_view_data, labels=[f'V{i}' for i in range(self.n_views)],
                           patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[idx])
                patch.set_alpha(0.7)
            
            ax.set_xlabel('View')
            ax.set_ylabel('Deviation from Triangulation (px)')
            ax.set_title(f'{method}', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Per-View Deviation Analysis', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_per_view_analysis.png', dpi=150)
        plt.close()
        
    def _generate_report(self):
        """Generate HTML report in research note style."""
        
        # Build data table
        table_rows = []
        for method in self.METHODS:
            s = self.summary[method]
            table_rows.append(f'''
            <tr>
                <td>{method}</td>
                <td>{s['cross_view_consistency']['mean']:.2f} ± {s['cross_view_consistency']['std']:.2f}</td>
                <td>{s['cross_view_consistency']['max']:.2f}</td>
                <td>{s['deviation_from_triangulation']['mean']:.2f} ± {s['deviation_from_triangulation']['std']:.2f}</td>
                <td>{s['n_samples']}</td>
            </tr>''')
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Center Estimation Analysis Report</title>
    <style>
        body {{ font-family: 'CMU Serif', Georgia, serif; max-width: 900px; margin: 0 auto; 
               padding: 40px; background: #fff; line-height: 1.6; }}
        h1 {{ font-size: 24px; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ font-size: 18px; margin-top: 30px; color: #333; }}
        h3 {{ font-size: 14px; margin-top: 20px; color: #555; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #ccc; text-align: right; }}
        th {{ background: #f5f5f5; font-weight: bold; text-align: center; }}
        td:first-child {{ text-align: left; font-family: monospace; }}
        img {{ max-width: 100%; margin: 15px 0; border: 1px solid #ddd; }}
        .abstract {{ background: #f9f9f9; padding: 15px; margin: 20px 0; font-style: italic; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure-caption {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .metadata {{ font-size: 12px; color: #888; }}
    </style>
</head>
<body>

<h1>Multi-View Center Estimation: Statistical Analysis</h1>

<p class="metadata">
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 
Data: {self.data_dir.name} | 
Samples: {len(self.frame_results)} frames
</p>

<div class="abstract">
<strong>Abstract:</strong> This report presents a statistical comparison of four center estimation 
methods for multi-view object cropping. Metrics include cross-view consistency (standard deviation 
of crop centers across views) and deviation from triangulation reference.
</div>

<h2>1. Methods</h2>
<ul>
<li><strong>no_correction</strong>: Fixed image center (256, 256) for all views</li>
<li><strong>per_view_2d</strong>: 2D centroid computed independently per view from mask</li>
<li><strong>triangulation</strong>: DLT triangulation from all 2D centroids, back-projected to each view</li>
<li><strong>visual_hull</strong>: Shape carving to find 3D volume, back-projected to each view</li>
</ul>

<h2>2. Quantitative Results</h2>

<table>
<tr>
    <th>Method</th>
    <th>Cross-View Std (px)</th>
    <th>Max Deviation (px)</th>
    <th>Dist. from Triangulation (px)</th>
    <th>N</th>
</tr>
{''.join(table_rows)}
</table>

<h2>3. Visual Analysis</h2>

<h3>Figure 1: 2D Centers on Images</h3>
<div class="figure">
<img src="fig1_2d_centers_on_images.png" alt="2D Centers">
<p class="figure-caption">Crop centers for each method overlaid on input images. 
Yellow=no_correction, Red=per_view_2d, Green=triangulation, Cyan=visual_hull.</p>
</div>

<h3>Figure 2: 3D Camera Setup & Ray Visualization</h3>
<div class="figure">
<img src="fig2_3d_camera_scene.png" alt="3D Scene">
<p class="figure-caption">Left: Rays from original 2D centroids (per_view_2d). 
Right: Rays from back-projected centers (triangulation). 
Coordinate axes: X(red), Y(green), Z(blue). Stars indicate estimated 3D centers.</p>
</div>

<h3>Figure 3: Method Comparison</h3>
<div class="figure">
<img src="fig3_method_comparison.png" alt="Method Comparison">
<p class="figure-caption">Left: Cross-view consistency (lower=better). 
Right: Deviation from triangulation reference.</p>
</div>

<h3>Figure 4: Per-View Analysis</h3>
<div class="figure">
<img src="fig4_per_view_analysis.png" alt="Per-View Analysis">
<p class="figure-caption">Box plots showing per-view deviation from triangulation across all sampled frames.</p>
</div>

<h2>4. Observations</h2>
<ul>
<li>Cross-view consistency: triangulation ({self.summary['triangulation']['cross_view_consistency']['mean']:.2f}px) 
vs per_view_2d ({self.summary['per_view_2d']['cross_view_consistency']['mean']:.2f}px)</li>
<li>Per-view 2D shows mean {self.summary['per_view_2d']['deviation_from_triangulation']['mean']:.2f}px 
deviation from triangulation reference</li>
<li>Visual hull produces similar results to triangulation 
(deviation: {self.summary['visual_hull']['deviation_from_triangulation']['mean']:.2f}px)</li>
</ul>

<hr>
<p class="metadata">Report generated by validate_center_research_v3.py</p>

</body>
</html>'''
        
        with open(self.output_dir / 'report.html', 'w') as f:
            f.write(html)
        
        # Save JSON
        output = {
            'metadata': {
                'data_dir': str(self.data_dir),
                'num_samples': len(self.frame_results),
                'timestamp': datetime.now().isoformat(),
                'methods': self.METHODS,
            },
            'summary': self.summary,
            'per_frame': [{k: v for k, v in r.items() if k not in ['images', 'masks']} 
                          for r in self.frame_results],
        }
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(output, f, indent=2)
            
    def print_summary(self):
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print(f"{'='*60}")
        print(f"{'Method':<16} {'Cross-View Std':>15} {'Dev from Tri':>15}")
        print(f"{'-'*60}")
        for m in self.METHODS:
            s = self.summary[m]
            print(f"{m:<16} {s['cross_view_consistency']['mean']:>10.2f} px    {s['deviation_from_triangulation']['mean']:>10.2f} px")
        print(f"{'='*60}")
        print(f"Report: {self.output_dir / 'report.html'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/joon/data/markerless_mouse_1_nerf')
    parser.add_argument('--output_dir', default='/home/joon/dev/FaceLift/mouse_extensions/reports/center_analysis_v3')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    
    validator = StatisticalValidator(args.data_dir, args.output_dir, args.num_samples)
    validator.run()
    validator.print_summary()


if __name__ == '__main__':
    main()
