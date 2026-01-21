#!/usr/bin/env python3
"""
Camera Setup Visualization v4 - Enhanced with Bug Fixes
=========================================================

Improvements:
- Fixed forward direction convention (removed erroneous negative sign)
- Large mouse images with camera index labels
- HTML report format
- Precise camera parameter display
"""

import numpy as np
import json
import os
import argparse
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from datetime import datetime


class Arrow3D:
    """Simple 3D arrow using quiver"""
    pass


def load_cameras(sample_dir):
    """Load camera data from opencv_cameras.json"""
    json_path = os.path.join(sample_dir, 'opencv_cameras.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    cameras = []
    for frame in data['frames']:
        w2c = np.array(frame['w2c'])
        c2w = np.linalg.inv(w2c)

        pos = c2w[:3, 3]
        R = w2c[:3, :3]

        # Forward direction - OpenCV convention: camera looks along +Z
        # c2w[:3, 2] = third column of C2W = Z-axis of camera in world
        # NO negative sign here!
        forward = c2w[:3, 2]  # FIXED: was -c2w[:3, 2]
        forward = forward / np.linalg.norm(forward)

        # Up direction (Y-axis, but negated for OpenCV where Y points down)
        up = -c2w[:3, 1]
        up = up / np.linalg.norm(up)

        # Right direction (X-axis)
        right = c2w[:3, 0]
        right = right / np.linalg.norm(right)

        dist = np.linalg.norm(pos)
        elev = np.degrees(np.arcsin(pos[2] / dist)) if dist > 0 else 0
        azim = np.degrees(np.arctan2(pos[1], pos[0]))

        cameras.append({
            'view_id': frame.get('view_id', len(cameras)),
            'position': pos,
            'forward': forward,
            'up': up,
            'right': right,
            'distance': dist,
            'elevation': elev,
            'azimuth': azim,
            'fx': frame['fx'],
            'fy': frame['fy'],
            'cx': frame['cx'],
            'cy': frame['cy'],
            'w2c': w2c,
            'c2w': c2w,
            'file_path': frame.get('file_path', '')
        })

    return cameras


def load_images(sample_dir, cameras):
    """Load mouse images for each camera view"""
    images = []
    for cam in cameras:
        img_path = os.path.join(sample_dir, cam['file_path'])
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(np.array(img))
        else:
            # Fallback: look in images/ folder
            alt_path = os.path.join(sample_dir, 'images', f"cam_{cam['view_id']:03d}.png")
            if os.path.exists(alt_path):
                img = Image.open(alt_path)
                images.append(np.array(img))
            else:
                images.append(np.zeros((512, 512, 3), dtype=np.uint8))
    return images


def create_3d_camera_plot(cameras, images, output_path, title="Camera Setup"):
    """Create 3D plot with camera positions and arrows pointing to center"""
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.Set1(np.linspace(0, 1, len(cameras)))

    # Plot origin (mouse location)
    ax.scatter([0], [0], [0], c='red', s=200, marker='*', label='Mouse (Origin)')

    for i, (cam, color) in enumerate(zip(cameras, colors)):
        pos = cam['position']
        forward = cam['forward']

        # Camera position
        ax.scatter(*pos, c=[color], s=100, marker='o')

        # Forward direction arrow (pointing along camera's Z-axis)
        arrow_len = 0.5
        ax.quiver(pos[0], pos[1], pos[2],
                  forward[0]*arrow_len, forward[1]*arrow_len, forward[2]*arrow_len,
                  color=color, arrow_length_ratio=0.3, linewidth=2)

        # Camera label
        ax.text(pos[0], pos[1], pos[2] + 0.15,
                f'Cam {i}', fontsize=12, fontweight='bold', color=color)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Equal aspect ratio
    max_range = max([np.abs(cam['position']).max() for cam in cameras]) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_image_grid_with_labels(cameras, images, output_path):
    """Create large image grid with camera labels and info"""
    n_cams = len(cameras)
    cols = 3
    rows = (n_cams + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    axes = axes.flatten() if n_cams > 1 else [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, n_cams))

    for i, (cam, img, ax) in enumerate(zip(cameras, images, axes)):
        ax.imshow(img)
        ax.set_title(f"Camera {i}\n"
                     f"Azim: {cam['azimuth']:.1f}° | Elev: {cam['elevation']:.1f}°\n"
                     f"fx={cam['fx']:.4f}, fy={cam['fy']:.4f}",
                     fontsize=12, fontweight='bold', color=colors[i])
        ax.axis('off')

        # Add colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(colors[i])
            spine.set_linewidth(4)

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_html_report(cameras, images, sample_dir, output_dir):
    """Generate comprehensive HTML report"""

    # Save individual camera images
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    for i, img in enumerate(images):
        img_path = os.path.join(img_dir, f'cam_{i}.png')
        Image.fromarray(img).save(img_path)

    # Generate HTML
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Camera Setup Report - D7_1</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 15px; }}
        h2 {{ color: #16213e; margin-top: 40px; }}
        .summary-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; padding: 25px; border-radius: 15px; margin: 20px 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }}
        .metric {{ background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; text-align: center; }}
        .metric-value {{ font-size: 28px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; opacity: 0.9; }}
        .camera-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 25px; margin: 30px 0; }}
        .camera-card {{ background: white; border-radius: 15px; overflow: hidden;
                       box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.3s; }}
        .camera-card:hover {{ transform: translateY(-5px); }}
        .camera-img {{ width: 100%; height: auto; }}
        .camera-info {{ padding: 20px; }}
        .camera-title {{ font-size: 20px; font-weight: bold; margin-bottom: 10px; }}
        .camera-params {{ font-size: 13px; color: #555; line-height: 1.8; }}
        .param-row {{ display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 5px 0; }}
        .param-name {{ color: #888; }}
        .param-value {{ font-weight: 500; font-family: monospace; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; border-radius: 10px; overflow: hidden; }}
        th {{ background: #4a90d9; color: white; padding: 15px; text-align: center; }}
        td {{ padding: 12px; text-align: center; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .viz-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }}
        .viz-img {{ width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .convention-box {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .success {{ color: #28a745; }}
        .code {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 8px;
                font-family: 'Consolas', monospace; overflow-x: auto; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Camera Setup Report - D7_1 Dataset</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Sample:</strong> {sample_dir}</p>

        <div class="summary-box">
            <h2 style="color:white; margin-top:0;">Summary</h2>
            <div class="summary-grid">
                <div class="metric">
                    <div class="metric-value">{len(cameras)}</div>
                    <div class="metric-label">Camera Views</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{cameras[0]['fx']:.4f}</div>
                    <div class="metric-label">Focal Length (fx=fy)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{cameras[0]['cx']:.1f}</div>
                    <div class="metric-label">Principal Point (cx=cy)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{cameras[0]['distance']:.4f}</div>
                    <div class="metric-label">Camera Distance</div>
                </div>
            </div>
        </div>

        <div class="convention-box">
            <h3>📐 Camera Convention (OpenCV)</h3>
            <ul>
                <li><strong>Forward</strong>: +Z axis (camera looks along positive Z)</li>
                <li><strong>Up</strong>: -Y axis (Y points down in image coordinates)</li>
                <li><strong>Right</strong>: +X axis</li>
            </ul>
            <p><strong>Forward direction</strong>: <code>c2w[:3, 2]</code> (NOT <code>-c2w[:3, 2]</code>)</p>
        </div>

        <h2>📷 Camera Views</h2>
        <div class="camera-grid">
'''

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for i, cam in enumerate(cameras):
        color = colors[i % len(colors)]
        html_content += f'''
            <div class="camera-card" style="border-top: 5px solid {color};">
                <img src="images/cam_{i}.png" class="camera-img" alt="Camera {i}">
                <div class="camera-info">
                    <div class="camera-title" style="color: {color};">Camera {i}</div>
                    <div class="camera-params">
                        <div class="param-row">
                            <span class="param-name">Azimuth</span>
                            <span class="param-value">{cam['azimuth']:.2f}°</span>
                        </div>
                        <div class="param-row">
                            <span class="param-name">Elevation</span>
                            <span class="param-value">{cam['elevation']:.2f}°</span>
                        </div>
                        <div class="param-row">
                            <span class="param-name">Distance</span>
                            <span class="param-value">{cam['distance']:.6f}</span>
                        </div>
                        <div class="param-row">
                            <span class="param-name">fx, fy</span>
                            <span class="param-value">{cam['fx']:.6f}</span>
                        </div>
                        <div class="param-row">
                            <span class="param-name">cx, cy</span>
                            <span class="param-value">{cam['cx']:.1f}</span>
                        </div>
                    </div>
                </div>
            </div>
'''

    html_content += '''
        </div>

        <h2>🔢 Detailed Parameters</h2>
        <table>
            <tr>
                <th>Camera</th>
                <th>Azimuth (°)</th>
                <th>Elevation (°)</th>
                <th>Distance</th>
                <th>Position (x, y, z)</th>
                <th>Forward (x, y, z)</th>
            </tr>
'''

    for i, cam in enumerate(cameras):
        pos = cam['position']
        fwd = cam['forward']
        html_content += f'''
            <tr>
                <td><strong>Cam {i}</strong></td>
                <td>{cam['azimuth']:.2f}</td>
                <td>{cam['elevation']:.2f}</td>
                <td>{cam['distance']:.6f}</td>
                <td>({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})</td>
                <td>({fwd[0]:.4f}, {fwd[1]:.4f}, {fwd[2]:.4f})</td>
            </tr>
'''

    html_content += '''
        </table>

        <h2>📊 3D Visualization</h2>
        <div class="viz-container">
            <div>
                <img src="camera_3d_setup.png" class="viz-img" alt="3D Camera Setup">
                <p style="text-align:center; color:#666;">Camera positions with forward direction arrows</p>
            </div>
            <div>
                <img src="camera_top_view.png" class="viz-img" alt="Top View">
                <p style="text-align:center; color:#666;">Top view (X-Y plane)</p>
            </div>
        </div>

        <h2>⚠️ Convention Notes</h2>
        <div class="code">
# Correct forward direction extraction (OpenCV convention)
c2w = np.linalg.inv(w2c)
forward = c2w[:3, 2]  # Third column of C2W matrix

# WRONG (causes 180° direction error):
# forward = -c2w[:3, 2]  # Extra negative sign is incorrect!
        </div>

        <hr>
        <p style="color:#888; font-size:12px; text-align:center;">
            Generated by visualize_camera_setup_v4.py | FaceLift Mouse Extension
        </p>
    </div>
</body>
</html>
'''

    html_path = os.path.join(output_dir, 'camera_report.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Saved: {html_path}")

    return html_path


def create_top_view(cameras, output_path):
    """Create top-down view (X-Y plane)"""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.Set1(np.linspace(0, 1, len(cameras)))

    # Plot origin
    ax.scatter([0], [0], c='red', s=200, marker='*', zorder=10, label='Mouse')

    for i, (cam, color) in enumerate(zip(cameras, colors)):
        pos = cam['position']
        forward = cam['forward']

        # Camera position (X-Y)
        ax.scatter(pos[0], pos[1], c=[color], s=150, marker='o', zorder=5)

        # Forward direction arrow (X-Y projection)
        arrow_len = 0.4
        ax.annotate('', xy=(pos[0] + forward[0]*arrow_len, pos[1] + forward[1]*arrow_len),
                    xytext=(pos[0], pos[1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Label
        ax.annotate(f'Cam {i}', (pos[0], pos[1]), xytext=(5, 5),
                    textcoords='offset points', fontsize=11, fontweight='bold', color=color)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Top View (X-Y Plane)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Camera Setup Visualization v4')
    parser.add_argument('--sample_dir', type=str, required=True,
                       help='Path to sample directory with opencv_cameras.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.sample_dir), 'camera_viz_v4')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading cameras from: {args.sample_dir}")
    cameras = load_cameras(args.sample_dir)
    print(f"Loaded {len(cameras)} cameras")

    print("Loading images...")
    images = load_images(args.sample_dir, cameras)

    print("Creating visualizations...")

    # 3D plot
    create_3d_camera_plot(cameras, images,
                          os.path.join(args.output_dir, 'camera_3d_setup.png'),
                          "D7_1 Camera Setup (OpenCV Convention)")

    # Top view
    create_top_view(cameras, os.path.join(args.output_dir, 'camera_top_view.png'))

    # Image grid
    create_image_grid_with_labels(cameras, images,
                                  os.path.join(args.output_dir, 'camera_images_grid.png'))

    # HTML report
    generate_html_report(cameras, images, args.sample_dir, args.output_dir)

    # Save camera data as JSON
    camera_data = []
    for cam in cameras:
        camera_data.append({
            'view_id': cam['view_id'],
            'position': cam['position'].tolist(),
            'forward': cam['forward'].tolist(),
            'azimuth': cam['azimuth'],
            'elevation': cam['elevation'],
            'distance': cam['distance'],
            'fx': cam['fx'],
            'fy': cam['fy'],
            'cx': cam['cx'],
            'cy': cam['cy']
        })

    with open(os.path.join(args.output_dir, 'camera_data.json'), 'w') as f:
        json.dump(camera_data, f, indent=2)

    print(f"\nDone! Output: {args.output_dir}")


if __name__ == '__main__':
    main()
