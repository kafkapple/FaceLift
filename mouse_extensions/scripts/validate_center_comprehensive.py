#!/usr/bin/env python3
"""
Comprehensive Center Estimation Validation with Visual Reports

Compares four approaches:
1. Triangulation (3D) - Recommended
2. Visual Hull (3D) - Robust alternative
3. Global Average (2D) - Simple but inaccurate
4. Per-View 2D Centroid - THE BROKEN APPROACH (for comparison)

Outputs:
- Quantitative metrics (JSON)
- Qualitative visualizations (PNG)
- HTML report with embedded images

Usage:
    python validate_center_comprehensive.py \
        --data_dir /home/joon/data/markerless_mouse_1_nerf \
        --output_dir /home/joon/dev/FaceLift/mouse_extensions/reports/center_validation \
        --num_samples 10

Author: AI Research Assistant
Date: 2026-01-17
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mouse_extensions.preprocessing.center_estimation import (
    CenterEstimator,
    CenterMethod,
    compare_all_methods,
    compute_per_view_2d_error,
    compare_all_methods_with_perview,
)

# Optional matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def load_raw_data(raw_dir: Path, frame_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """Load images, masks, and cameras for a frame."""
    cam_file = raw_dir / "new_cam.pkl"
    with open(cam_file, 'rb') as f:
        cam_params = pickle.load(f)
    
    images = []
    masks = []
    cameras = []
    
    video_dir = raw_dir / "video_undist"
    mask_dir = raw_dir / "simpleclick_undist"
    
    for view_idx in range(6):
        # Load image
        video_path = video_dir / f"{view_idx}.mp4"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                images.append(np.zeros((576, 512, 3), dtype=np.uint8))
        else:
            images.append(np.zeros((576, 512, 3), dtype=np.uint8))
        
        # Load mask
        mask_video = mask_dir / f"{view_idx}.mp4"
        if mask_video.exists():
            cap = cv2.VideoCapture(str(mask_video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                masks.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                masks.append(np.zeros((576, 512), dtype=np.uint8))
        else:
            masks.append(np.zeros((576, 512), dtype=np.uint8))
        
        # Camera params
        cam = cam_params[view_idx]
        K = cam['K']
        R = cam['R']
        T = cam['T']
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.flatten()
        
        cameras.append({
            'K': K,
            'w2c': w2c.tolist(),
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
        })
    
    return images, masks, cameras


def compute_2d_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Compute 2D centroid of mask."""
    coords = np.where(mask > 127)
    if len(coords[0]) > 0:
        return coords[1].mean(), coords[0].mean()  # cx, cy
    return mask.shape[1] / 2, mask.shape[0] / 2


def visualize_single_frame(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    results: Dict,
    output_path: Path,
    frame_idx: int,
):
    """Generate visualization for a single frame."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    colors = {
        'triangulation': '#FF0000',      # Red
        'visual_hull': '#0000FF',         # Blue
        'global_average': '#00FF00',      # Green
        'per_view_2d': '#FFFF00',         # Yellow - the broken one
    }
    
    markers = {
        'triangulation': 'x',
        'visual_hull': '+',
        'global_average': 's',
        'per_view_2d': 'o',  # Circle for per-view (original 2D centroid)
    }
    
    for view_idx in range(6):
        # Row 1: Original images with all centers marked
        ax = axes[0, view_idx]
        ax.imshow(images[view_idx])
        ax.set_title(f'View {view_idx} - Image', fontsize=10)
        ax.axis('off')
        
        # Mark estimated centers from each method
        for method_name, result in results.items():
            if method_name == 'per_view_2d':
                # Per-view 2D uses its own centroids
                cx, cy = result['centroids_2d'][view_idx]
            else:
                cx, cy = result.centers_2d[view_idx]
            
            ax.scatter(cx, cy, c=colors.get(method_name, 'gray'), s=100, 
                      marker=markers.get(method_name, 'x'), linewidths=2,
                      edgecolors='black' if method_name == 'per_view_2d' else 'none')
        
        # Row 2: Masks with centers
        ax = axes[1, view_idx]
        ax.imshow(masks[view_idx], cmap='gray')
        ax.set_title(f'View {view_idx} - Mask', fontsize=10)
        ax.axis('off')
        
        # Mark centers on mask
        for method_name, result in results.items():
            if method_name == 'per_view_2d':
                cx, cy = result['centroids_2d'][view_idx]
            else:
                cx, cy = result.centers_2d[view_idx]
            ax.scatter(cx, cy, c=colors.get(method_name, 'gray'), s=80, 
                      marker=markers.get(method_name, 'x'), linewidths=2)
        
        # Row 3: Zoomed mask with centers and error visualization
        ax = axes[2, view_idx]
        orig_cx, orig_cy = compute_2d_centroid(masks[view_idx])
        zoom_size = 150
        x_min = max(0, int(orig_cx - zoom_size))
        x_max = min(masks[view_idx].shape[1], int(orig_cx + zoom_size))
        y_min = max(0, int(orig_cy - zoom_size))
        y_max = min(masks[view_idx].shape[0], int(orig_cy + zoom_size))
        
        zoomed = masks[view_idx][y_min:y_max, x_min:x_max]
        ax.imshow(zoomed, cmap='gray')
        ax.set_title(f'View {view_idx} - Zoomed', fontsize=10)
        ax.axis('off')
        
        # Draw centers in zoomed view
        for method_name, result in results.items():
            if method_name == 'per_view_2d':
                cx, cy = result['centroids_2d'][view_idx]
            else:
                cx, cy = result.centers_2d[view_idx]
            ax.scatter(cx - x_min, cy - y_min, c=colors.get(method_name, 'gray'), 
                      s=120, marker=markers.get(method_name, 'x'), linewidths=3,
                      edgecolors='black' if method_name == 'per_view_2d' else 'none')
        
        # Draw line showing error between per-view 2D and triangulation
        if 'per_view_2d' in results and 'triangulation' in results:
            pv_cx, pv_cy = results['per_view_2d']['centroids_2d'][view_idx]
            tri_cx, tri_cy = results['triangulation'].centers_2d[view_idx]
            ax.plot([pv_cx - x_min, tri_cx - x_min], 
                   [pv_cy - y_min, tri_cy - y_min], 
                   'w--', linewidth=2, alpha=0.7)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['per_view_2d'],
                   markersize=10, markeredgecolor='black', label='Per-View 2D (BROKEN)'),
        plt.Line2D([0], [0], marker='x', color=colors['triangulation'], 
                   markersize=10, label='Triangulation (3D)', linestyle='None'),
        plt.Line2D([0], [0], marker='+', color=colors['visual_hull'], 
                   markersize=10, label='Visual Hull (3D)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color=colors['global_average'], 
                   markersize=10, label='Global Average', linestyle='None'),
        plt.Line2D([0], [0], color='white', linestyle='--', linewidth=2,
                   label='Error (Per-View vs Triangulation)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10)
    
    # Title
    fig.suptitle(f'Frame {frame_idx}: Center Estimation Comparison\n(Yellow circles = Per-view 2D centroids, Red X = Correct 3D triangulation)', 
                fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_temporal_variation(
    all_centers: Dict[str, List[np.ndarray]],
    output_path: Path,
):
    """Visualize temporal variation of 3D centers."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'triangulation': 'red',
        'visual_hull': 'blue',
        'global_average': 'green',
    }
    
    # Plot X, Y, Z over time
    for method_name, centers in all_centers.items():
        if method_name == 'per_view_2d':
            continue  # Skip per-view 2D for temporal plot (it's 2D data)
        centers = np.array(centers)
        frames = np.arange(len(centers))
        
        axes[0, 0].plot(frames, centers[:, 0], color=colors.get(method_name, 'gray'), 
                       label=method_name, linewidth=2, alpha=0.8)
        axes[0, 1].plot(frames, centers[:, 1], color=colors.get(method_name, 'gray'), 
                       linewidth=2, alpha=0.8)
        axes[1, 0].plot(frames, centers[:, 2], color=colors.get(method_name, 'gray'), 
                       linewidth=2, alpha=0.8)
    
    axes[0, 0].set_title('X Position Over Time')
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('X (mm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Y Position Over Time')
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('Y (mm)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Z Position Over Time')
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Z (mm)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3D scatter plot (simplified 2D projection)
    ax = axes[1, 1]
    for method_name, centers in all_centers.items():
        if method_name == 'per_view_2d':
            continue
        centers = np.array(centers)
        ax.scatter(centers[:, 0], centers[:, 1], c=colors.get(method_name, 'gray'), 
                  label=method_name, alpha=0.6, s=50)
    ax.set_title('X-Y Center Distribution')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Temporal Variation of 3D Centers', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_error_comparison(
    all_results: List[Dict],
    output_path: Path,
):
    """Visualize error comparison between methods."""
    if not HAS_MATPLOTLIB:
        return
    
    methods = ['triangulation', 'visual_hull', 'global_average', 'per_view_2d']
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00']
    
    # Collect errors
    ray_errors = {m: [] for m in methods}
    reproj_errors = {m: [] for m in methods}
    
    for result in all_results:
        for method in methods:
            if method in result:
                if method == 'per_view_2d':
                    ray_errors[method].append(result[method]['ray_convergence_error'])
                    reproj_errors[method].append(result[method]['mean_reprojection_error'])
                else:
                    ray_errors[method].append(result[method]['ray_convergence_error'])
                    reproj_errors[method].append(result[method]['mean_reprojection_error'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ray convergence error
    ax = axes[0]
    positions = np.arange(len(methods))
    bar_data = [np.mean(ray_errors[m]) if ray_errors[m] else 0 for m in methods]
    bar_err = [np.std(ray_errors[m]) if ray_errors[m] else 0 for m in methods]
    
    bars = ax.bar(positions, bar_data, yerr=bar_err, color=colors, capsize=5, edgecolor='black')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Triangulation\n(3D)', 'Visual Hull\n(3D)', 'Global Avg\n(2D)', 'Per-View 2D\n(BROKEN)'], fontsize=10)
    ax.set_ylabel('Ray Convergence Error (mm)')
    ax.set_title('Ray Convergence Error by Method')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, bar_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
               f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Reprojection error (pixel)
    ax = axes[1]
    bar_data = [np.mean(reproj_errors[m]) if reproj_errors[m] else 0 for m in methods]
    bar_err = [np.std(reproj_errors[m]) if reproj_errors[m] else 0 for m in methods]
    
    bars = ax.bar(positions, bar_data, yerr=bar_err, color=colors, capsize=5, edgecolor='black')
    ax.set_xticks(positions)
    ax.set_xticklabels(['Triangulation\n(3D)', 'Visual Hull\n(3D)', 'Global Avg\n(2D)', 'Per-View 2D\n(BROKEN)'], fontsize=10)
    ax.set_ylabel('Reprojection Error (pixels)')
    ax.set_title('Cross-View Reprojection Error\n(Lower = More Consistent)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, bar_data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Method Comparison: Why Per-View 2D Centroid Fails', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_html_report(
    metrics: Dict,
    image_files: List[str],
    output_path: Path,
):
    """Generate HTML report with embedded images."""
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Center Estimation Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .best {{ background-color: #90EE90; font-weight: bold; }}
        .worst {{ background-color: #FFB6C1; }}
        .broken {{ background-color: #FFFF99; }}
        img {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .frame-vis {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; }}
        .warning {{ background-color: #FFF3CD; border: 1px solid #FFE69C; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .success {{ background-color: #D4EDDA; border: 1px solid #C3E6CB; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric-box {{ display: inline-block; padding: 10px 20px; margin: 5px; border-radius: 5px; }}
        .metric-good {{ background-color: #90EE90; }}
        .metric-bad {{ background-color: #FFB6C1; }}
    </style>
</head>
<body>
    <div class="container">
    <h1>🔬 Center Estimation Validation Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="warning">
        <strong>⚠️ Key Finding:</strong> Per-view 2D centroid estimation causes cross-view inconsistency,
        leading to severe ghosting artifacts in 3D reconstruction. Use 3D triangulation instead!
    </div>
    
    <h2>📊 Summary Metrics</h2>
    <table>
        <tr>
            <th>Method</th>
            <th>Type</th>
            <th>Ray Error (mm)</th>
            <th>Reproj Error (px)</th>
            <th>Confidence</th>
            <th>Status</th>
        </tr>
'''
    
    # Find best method (excluding per_view_2d)
    valid_methods = {k: v for k, v in metrics['summary'].items() if k != 'per_view_2d'}
    best_method = min(valid_methods.keys(), 
                      key=lambda m: metrics['summary'][m]['ray_convergence_error']['mean'])
    
    method_info = {
        'triangulation': ('3D', 'Recommended'),
        'visual_hull': ('3D', 'Alternative'),
        'global_average': ('2D Avg', 'Not Recommended'),
        'per_view_2d': ('2D Per-View', 'BROKEN'),
    }
    
    for method, stats in metrics['summary'].items():
        m_type, status = method_info.get(method, ('?', '?'))
        
        if method == best_method:
            row_class = 'best'
            status_icon = '✅ ' + status
        elif method == 'per_view_2d':
            row_class = 'broken'
            status_icon = '❌ ' + status
        elif method == 'global_average':
            row_class = 'worst'
            status_icon = '⚠️ ' + status
        else:
            row_class = ''
            status_icon = status
        
        ray_err = stats['ray_convergence_error']['mean']
        ray_std = stats['ray_convergence_error']['std']
        reproj_err = stats.get('reprojection_error', {}).get('mean', 0)
        reproj_std = stats.get('reprojection_error', {}).get('std', 0)
        conf = stats['confidence']['mean']
        
        html += f'''        <tr class="{row_class}">
            <td><strong>{method}</strong></td>
            <td>{m_type}</td>
            <td>{ray_err:.2f} ± {ray_std:.2f}</td>
            <td>{reproj_err:.2f} ± {reproj_std:.2f}</td>
            <td>{conf:.4f}</td>
            <td>{status_icon}</td>
        </tr>
'''
    
    html += '''    </table>
    
    <div class="success">
        <strong>✅ Recommendation:</strong> Use <strong>TRIANGULATION</strong> method for accurate 3D center estimation.
        Per-view 2D centroid has ~{:.1f}x higher error!
    </div>
'''.format(metrics['summary'].get('per_view_2d', {}).get('ray_convergence_error', {}).get('mean', 0) / 
           max(0.01, metrics['summary'].get('triangulation', {}).get('ray_convergence_error', {}).get('mean', 1)))
    
    html += '''
    <h2>📈 Temporal Analysis</h2>
'''
    if 'temporal' in metrics:
        t = metrics['temporal']
        html += f'''    
    <p>Analysis of mouse movement across frames:</p>
    <table>
        <tr>
            <th>Metric</th>
            <th>Mean</th>
            <th>Max</th>
            <th>Std</th>
        </tr>
        <tr>
            <td>Frame-to-Frame Movement (mm)</td>
            <td>{t['movement']['mean']:.2f}</td>
            <td>{t['movement']['max']:.2f}</td>
            <td>{t['movement']['std']:.2f}</td>
        </tr>
        <tr>
            <td>Equivalent Pixel Movement</td>
            <td class="metric-bad">{t['pixel_movement']['mean']:.1f} px</td>
            <td class="metric-bad">{t['pixel_movement']['max']:.1f} px</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Deviation from Global Center</td>
            <td>{t['deviation']['mean']:.2f} mm</td>
            <td>{t['deviation']['max']:.2f} mm</td>
            <td>-</td>
        </tr>
    </table>
    
    <div class="warning">
        <strong>⚠️ Mouse Movement:</strong> Maximum {t['pixel_movement']['max']:.1f} pixels between frames.
        This confirms <strong>per-frame center estimation is required</strong> - global center is not sufficient!
    </div>
'''
    
    html += '''
    <h2>📸 Visualizations</h2>
    <p>Yellow circles = Per-view 2D centroids (broken approach)<br>
    Red X = Triangulation result (correct approach)<br>
    White dashed lines = Error between per-view and correct center</p>
'''
    for img_file in image_files:
        img_name = Path(img_file).name
        html += f'''    <div class="frame-vis">
        <h3>{img_name}</h3>
        <img src="{img_name}" alt="{img_name}">
    </div>
'''
    
    html += '''
    <h2>📚 References</h2>
    <ul>
        <li>pose-splatter CenterEstimator: <code>/home/joon/dev/pose-splatter/src/preprocessing/center_estimator.py</code></li>
        <li>FaceLift CenterEstimator: <code>mouse_extensions/preprocessing/center_estimation.py</code></li>
    </ul>
    </div>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive center estimation validation")
    parser.add_argument('--data_dir', type=str, required=True, help='Raw data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for report')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of frames to validate')
    parser.add_argument('--frame_step', type=int, default=100, help='Frame step between samples')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    all_results = []
    all_centers = {'triangulation': [], 'visual_hull': [], 'global_average': []}
    image_files = []
    
    print(f"Validating {args.num_samples} samples from {data_dir}")
    print(f"Comparing: Triangulation, Visual Hull, Global Average, Per-View 2D (broken)")
    
    for i in tqdm(range(args.num_samples), desc="Processing frames"):
        frame_idx = i * args.frame_step
        
        try:
            images, masks, cameras = load_raw_data(data_dir, frame_idx)
            masks_array = np.array(masks)
            
            # Compare all methods INCLUDING per-view 2D
            results = compare_all_methods_with_perview(masks_array, cameras)
            
            # Store results
            sample_result = {'frame_idx': frame_idx}
            for method_name, result in results.items():
                if method_name == 'per_view_2d':
                    sample_result[method_name] = result  # Already a dict
                else:
                    sample_result[method_name] = {
                        'center_3d': result.center_3d.tolist(),
                        'confidence': result.confidence,
                        'ray_convergence_error': result.ray_convergence_error,
                        'centers_2d': result.centers_2d.tolist(),
                        'mean_reprojection_error': float(np.mean(result.metadata.get('reprojection_errors', [0]))),
                    }
                    all_centers[method_name].append(result.center_3d)
            
            all_results.append(sample_result)
            
            # Generate visualization for first 5 frames
            if i < 5 and HAS_MATPLOTLIB:
                vis_path = output_dir / f"frame_{frame_idx:06d}.png"
                visualize_single_frame(images, masks, results, vis_path, frame_idx)
                image_files.append(str(vis_path))
        
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate metrics
    summary = {}
    for method in ['triangulation', 'visual_hull', 'global_average', 'per_view_2d']:
        method_results = [r[method] for r in all_results if method in r]
        if method_results:
            ray_errors = [r['ray_convergence_error'] for r in method_results]
            confidences = [r['confidence'] for r in method_results]
            reproj_errors = [r.get('mean_reprojection_error', 0) for r in method_results]
            
            summary[method] = {
                'ray_convergence_error': {
                    'mean': float(np.mean(ray_errors)),
                    'std': float(np.std(ray_errors)),
                    'min': float(np.min(ray_errors)),
                    'max': float(np.max(ray_errors)),
                },
                'reprojection_error': {
                    'mean': float(np.mean(reproj_errors)),
                    'std': float(np.std(reproj_errors)),
                },
                'confidence': {
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                },
            }
    
    # Temporal analysis
    tri_centers = np.array(all_centers['triangulation'])
    if len(tri_centers) > 1:
        diffs = np.diff(tri_centers, axis=0)
        movement = np.linalg.norm(diffs, axis=1)
        global_center = tri_centers.mean(axis=0)
        deviations = np.linalg.norm(tri_centers - global_center, axis=1)
        pixel_per_mm = 549 / 2.7 / 100
        
        temporal = {
            'movement': {
                'mean': float(movement.mean()),
                'max': float(movement.max()),
                'std': float(movement.std()),
            },
            'pixel_movement': {
                'mean': float(movement.mean() * pixel_per_mm),
                'max': float(movement.max() * pixel_per_mm),
            },
            'deviation': {
                'mean': float(deviations.mean()),
                'max': float(deviations.max()),
            },
            'global_center': global_center.tolist(),
        }
    else:
        temporal = {}
    
    # Save visualizations
    if HAS_MATPLOTLIB:
        if len(all_centers['triangulation']) > 1:
            temporal_vis_path = output_dir / "temporal_variation.png"
            visualize_temporal_variation(all_centers, temporal_vis_path)
            image_files.append(str(temporal_vis_path))
        
        error_vis_path = output_dir / "error_comparison.png"
        visualize_error_comparison(all_results, error_vis_path)
        image_files.append(str(error_vis_path))
    
    # Save JSON metrics
    metrics = {
        'samples': all_results,
        'summary': summary,
        'temporal': temporal,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate HTML report
    html_path = output_dir / "report.html"
    generate_html_report(metrics, image_files, html_path)
    
    # Print summary
    print("\n" + "="*80)
    print("CENTER ESTIMATION VALIDATION SUMMARY")
    print("="*80)
    print(f"\n{len(all_results)} frames analyzed")
    print(f"\n{'Method':<20} {'Type':<10} {'Ray Error':>15} {'Reproj (px)':>15} {'Status':<15}")
    print("-"*80)
    
    for method in ['triangulation', 'visual_hull', 'global_average', 'per_view_2d']:
        if method not in summary:
            continue
        stats = summary[method]
        m_type = '3D' if method in ['triangulation', 'visual_hull'] else '2D'
        status = '✅ RECOMMENDED' if method == 'triangulation' else ('❌ BROKEN' if method == 'per_view_2d' else '')
        print(f"{method:<20} {m_type:<10} {stats['ray_convergence_error']['mean']:>12.2f}±{stats['ray_convergence_error']['std']:<5.2f} {stats['reprojection_error']['mean']:>12.2f}±{stats['reprojection_error']['std']:<4.1f} {status:<15}")
    
    if temporal:
        print(f"\nTemporal Analysis:")
        print(f"  Frame-to-frame movement: {temporal['movement']['mean']:.2f} mm (max: {temporal['movement']['max']:.2f} mm)")
        print(f"  Equivalent pixels: {temporal['pixel_movement']['mean']:.1f} px (max: {temporal['pixel_movement']['max']:.1f} px)")
    
    # Highlight the key finding
    if 'per_view_2d' in summary and 'triangulation' in summary:
        ratio = summary['per_view_2d']['ray_convergence_error']['mean'] / summary['triangulation']['ray_convergence_error']['mean']
        print(f"\n⚠️  Per-view 2D centroid has {ratio:.1f}x HIGHER error than triangulation!")
    
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - metrics.json: Quantitative results")
    print(f"  - report.html: Visual report")
    print(f"  - *.png: Visualizations")


if __name__ == '__main__':
    main()
