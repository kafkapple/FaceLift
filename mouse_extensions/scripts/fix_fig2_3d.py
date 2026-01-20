#!/usr/bin/env python3
"""Figure 2: Ray convergence comparison between per_view_2d and triangulation.

Key insight: The 12px 2D difference comes from:
- per_view_2d: independent 2D centroids from each view (don't perfectly converge)
- triangulation: back-projected 2D from single 3D point (converge by construction)
"""

import sys
sys.path.insert(0, '/home/joon/dev/FaceLift')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from mouse_extensions.preprocessing.center_estimation import CenterEstimator
from mouse_extensions.preprocessing.data_loader import DataLoader


def get_camera_center(cam):
    R = np.array(cam['R'])
    T = np.array(cam['T']).flatten()
    return -R.T @ T


def get_2d_centroid(mask):
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return np.array([256.0, 256.0])
    return np.array([float(xs.mean()), float(ys.mean())])


def ray_from_2d(cam, pt2d):
    K = np.array(cam['K'])
    R = np.array(cam['R'])
    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([pt2d[0], pt2d[1], 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    return R.T @ ray_cam  # world coordinates


def main():
    data_dir = '/home/joon/data/markerless_mouse_1_nerf'
    output_dir = Path('/home/joon/dev/FaceLift/mouse_extensions/reports/center_analysis_v4')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = DataLoader(data_dir, source_type='raw', num_views=6)
    cameras = loader.cameras
    n_views = len(cameras)
    
    # Load first frame
    images, masks = loader.load_frame(0)
    
    # Compute methods
    tri_est = CenterEstimator(cameras, method='triangulation')
    tri_result = tri_est.estimate(masks)
    center_3d_tri = tri_result.center_3d
    centers_2d_tri = tri_result.centers_2d  # back-projected, consistent
    
    # Per-view 2D: raw centroids
    centers_2d_pv2d = np.array([get_2d_centroid(m) for m in masks])
    
    cam_positions = np.array([get_camera_center(c) for c in cameras])
    
    # Compute 2D difference
    diff_2d = np.linalg.norm(centers_2d_pv2d - centers_2d_tri, axis=1)
    
    # === Figure: Side-by-side ray comparison ===
    fig = plt.figure(figsize=(24, 10))
    
    methods = [
        ('Per-View 2D (Independent)', centers_2d_pv2d, '#e74c3c', 'red'),
        ('Triangulation (Consistent)', centers_2d_tri, '#2ecc71', 'lime'),
    ]
    
    for idx, (title, centers_2d, point_color, ray_color) in enumerate(methods):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        
        # World axes
        axis_len = 30
        ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', alpha=0.5, linewidth=2)
        ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', alpha=0.5, linewidth=2)
        ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', alpha=0.5, linewidth=2)
        
        # Draw cameras and rays
        cam_colors = plt.cm.Set2(np.linspace(0, 1, n_views))
        for i, cam in enumerate(cameras):
            C = cam_positions[i]
            ax.scatter([C[0]], [C[1]], [C[2]], c=[cam_colors[i]], s=150, marker='^')
            ax.text(C[0], C[1], C[2]+15, f'C{i}', fontsize=9, ha='center')
            
            # Ray from 2D center
            ray_world = ray_from_2d(cam, centers_2d[i])
            ray_len = 280
            end = C + ray_world * ray_len
            
            ax.plot([C[0], end[0]], [C[1], end[1]], [C[2], end[2]],
                   c=ray_color, alpha=0.6, linewidth=2)
        
        # Draw 3D center (same for both)
        ax.scatter([center_3d_tri[0]], [center_3d_tri[1]], [center_3d_tri[2]],
                  c='gold', s=500, marker='*', edgecolors='black', linewidths=2,
                  label='3D Center', zorder=10)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{title}\nRays show where 2D centers project in 3D', fontsize=12)
        ax.legend(loc='upper right')
        ax.view_init(elev=20, azim=50)
    
    # Info text
    info = f'''2D Center Differences (per-view):
View | Per-View 2D | Triangulation | Diff (px)
-----|-------------|---------------|----------'''
    for i in range(n_views):
        pv = centers_2d_pv2d[i]
        tr = centers_2d_tri[i]
        info += f'\n  {i}  | ({pv[0]:.0f}, {pv[1]:.0f}) | ({tr[0]:.0f}, {tr[1]:.0f}) | {diff_2d[i]:.1f}'
    info += f'\n\nMean 2D difference: {diff_2d.mean():.1f} px'
    info += f'\nThis causes cross-view inconsistency!'
    
    fig.text(0.02, 0.02, info, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.suptitle('Figure 2: Ray Convergence - Per-View 2D vs Triangulation', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.15, 1, 0.96])
    plt.savefig(output_dir / 'fig2_3d_camera_scene_improved.png', dpi=150)
    plt.close()
    
    print(f"""
=== 2D Center Comparison ===
View | Per-View 2D     | Triangulation   | Diff
-----|-----------------|-----------------|------""")
    for i in range(n_views):
        pv = centers_2d_pv2d[i]
        tr = centers_2d_tri[i]
        print(f"  {i}  | ({pv[0]:6.1f}, {pv[1]:6.1f}) | ({tr[0]:6.1f}, {tr[1]:6.1f}) | {diff_2d[i]:.1f} px")
    
    print(f"""
Mean 2D difference: {diff_2d.mean():.1f} px
Max 2D difference:  {diff_2d.max():.1f} px

Key insight: 
  Per-View 2D uses independent 2D centroids -> rays don't converge perfectly
  Triangulation back-projects from single 3D -> rays converge by construction
  
Saved: {output_dir / 'fig2_3d_camera_scene_improved.png'}
""")

if __name__ == '__main__':
    main()
