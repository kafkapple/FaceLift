#!/usr/bin/env python3
"""
Coordinate System & Preprocessing Report Generator v2
======================================================

Generates comprehensive reports with coordinate system visualizations.
Uses plain text in figures (no LaTeX), LaTeX only in HTML report.

Usage:
    python coordinate_report_generator.py --camera-pkl /path/to/cam.pkl --output reports/
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass
class CameraParams:
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray
    t: np.ndarray
    view_id: int = 0


def load_cameras_from_pkl(pkl_path: str) -> List[CameraParams]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    cameras = []
    for i, cam in enumerate(data):
        K = cam['K']
        R = cam['R']
        t = cam['T'].flatten()
        cameras.append(CameraParams(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            R=R, t=t, view_id=i
        ))
    return cameras


def create_default_camera() -> CameraParams:
    return CameraParams(
        fx=1632.3, fy=1639.3, cx=601.3, cy=491.2,
        R=np.eye(3), t=np.array([0, 0, 5]), view_id=0
    )


def fig_coordinate_systems(output_dir: Path) -> str:
    """Generate 3D coordinate systems visualization."""
    fig = plt.figure(figsize=(16, 6))

    # World Coordinate System
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('(a) World Coordinates', fontsize=12, fontweight='bold')
    ax1.quiver(0, 0, 0, 2, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, 0, 2, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
    ax1.quiver(0, 0, 0, 0, 0, 2, color='b', arrow_length_ratio=0.1, linewidth=2)
    ax1.text(2.2, 0, 0, 'X_w', fontsize=10, color='r')
    ax1.text(0, 2.2, 0, 'Y_w', fontsize=10, color='g')
    ax1.text(0, 0, 2.2, 'Z_w', fontsize=10, color='b')
    ax1.scatter([0.5], [0.3], [0.2], s=200, c='orange', marker='o')
    ax1.text(0.5, 0.3, 0.5, 'Object', fontsize=9)
    cam_positions = [(3, 1, 2), (3, -1, 2), (2, 0, 3)]
    for i, pos in enumerate(cam_positions):
        ax1.scatter(*pos, s=100, c='purple', marker='^')
        ax1.text(pos[0]+0.2, pos[1], pos[2], f'Cam{i}', fontsize=8)
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(-2, 3)
    ax1.set_zlim(-1, 4)

    # Camera Coordinate System
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('(b) Camera Coordinates', fontsize=12, fontweight='bold')
    ax2.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 1.5, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
    ax2.quiver(0, 0, 0, 0, 0, 2, color='b', arrow_length_ratio=0.1, linewidth=2)
    ax2.text(1.7, 0, 0, 'X_c (right)', fontsize=9, color='r')
    ax2.text(0, 1.7, 0, 'Y_c (down)', fontsize=9, color='g')
    ax2.text(0, 0, 2.2, 'Z_c (forward)', fontsize=9, color='b')
    img_plane = [[-0.5, -0.4, 1], [0.5, -0.4, 1], [0.5, 0.4, 1], [-0.5, 0.4, 1]]
    ax2.add_collection3d(Poly3DCollection([img_plane], alpha=0.3, facecolor='cyan', edgecolor='blue'))
    ax2.text(0, 0, 1.1, 'Image Plane', fontsize=8, ha='center')
    ax2.scatter([0.2], [0.15], [2.5], s=150, c='orange', marker='o')
    ax2.plot([0, 0.2], [0, 0.15], [0, 2.5], 'k--', alpha=0.5)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1, 1.5)
    ax2.set_zlim(0, 3)

    # Pixel Coordinate System
    ax3 = fig.add_subplot(133)
    ax3.set_title('(c) Pixel Coordinates', fontsize=12, fontweight='bold')
    ax3.add_patch(Rectangle((0, 0), 1152, 1024, fill=False, edgecolor='black', linewidth=2))
    ax3.annotate('', xy=(200, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.annotate('', xy=(0, 200), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax3.text(220, 30, 'u', fontsize=12, color='red', fontweight='bold')
    ax3.text(30, 220, 'v', fontsize=12, color='green', fontweight='bold')
    cx, cy = 601, 491
    ax3.plot(cx, cy, 'b+', markersize=15, markeredgewidth=2)
    ax3.plot(cx, cy, 'bo', markersize=20, fillstyle='none', markeredgewidth=1)
    ax3.text(cx+30, cy+30, f'PP ({cx}, {cy})', fontsize=10, color='blue')
    ax3.plot(576, 512, 'g+', markersize=12, markeredgewidth=2)
    ax3.text(576+30, 512-30, 'Center (576, 512)', fontsize=9, color='green')
    ax3.scatter([650], [420], s=100, c='orange', zorder=5)
    ax3.set_xlim(-50, 1200)
    ax3.set_ylim(1100, -50)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'fig1_coordinate_systems.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(output_path)


def fig_projection_pipeline(output_dir: Path) -> str:
    """Generate projection pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Camera Projection Pipeline', fontsize=14, fontweight='bold', pad=20)

    boxes = [
        (1, 4, 'World\n(X, Y, Z)'),
        (5, 4, 'Camera\n(Xc, Yc, Zc)'),
        (9, 4, 'Normalized\n(x, y)'),
        (13, 4, 'Pixel\n(u, v)'),
    ]

    for x, y, text in boxes:
        box = FancyBboxPatch((x-1, y-0.8), 2, 1.6, boxstyle="round,pad=0.1",
                             facecolor='lightblue', edgecolor='navy', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')

    arrow_style = dict(arrowstyle='->', color='darkgreen', lw=2)
    ax.annotate('', xy=(4, 4), xytext=(2, 4), arrowprops=arrow_style)
    ax.text(3, 5.2, '[R | t]', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    ax.text(3, 2.8, 'Pc = R*Pw + t', fontsize=9, ha='center', style='italic')

    ax.annotate('', xy=(8, 4), xytext=(6, 4), arrowprops=arrow_style)
    ax.text(7, 5.2, '/ Zc', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    ax.text(7, 2.8, 'x = Xc/Zc\ny = Yc/Zc', fontsize=9, ha='center', style='italic')

    ax.annotate('', xy=(12, 4), xytext=(10, 4), arrowprops=arrow_style)
    ax.text(11, 5.2, 'K', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    ax.text(11, 2.8, 'u = fx*x + cx\nv = fy*y + cy', fontsize=9, ha='center', style='italic')

    # Intrinsic matrix (plain text)
    ax.text(7, 1.2, 'K = [fx  0  cx]\n    [ 0 fy cy]\n    [ 0  0  1]',
            fontsize=10, ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    output_path = output_dir / 'fig2_projection_pipeline.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(output_path)


def fig_pp_centered_shift(cam: CameraParams, output_dir: Path) -> str:
    """Generate PP-centered shift visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scale = 549 / cam.fx

    # Original
    ax1 = axes[0]
    ax1.set_title('(a) Original (1152x1024)', fontsize=11, fontweight='bold')
    ax1.add_patch(Rectangle((0, 0), 1152, 1024, fill=False, edgecolor='black', linewidth=2))
    ax1.plot(cam.cx, cam.cy, 'r+', markersize=15, markeredgewidth=3)
    ax1.plot(cam.cx, cam.cy, 'ro', markersize=25, fillstyle='none', markeredgewidth=2)
    ax1.text(cam.cx+20, cam.cy+20, f'PP\n({cam.cx:.0f}, {cam.cy:.0f})', fontsize=9, color='red')
    ax1.plot(576, 512, 'g+', markersize=12, markeredgewidth=2)
    mouse_x, mouse_y = 650, 450
    ax1.scatter([mouse_x], [mouse_y], s=300, c='orange', marker='o', alpha=0.7)
    ax1.set_xlim(-50, 1200)
    ax1.set_ylim(1100, -50)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # After scaling
    ax2 = axes[1]
    ax2.set_title(f'(b) After Scale ({scale:.4f})', fontsize=11, fontweight='bold')
    new_w, new_h = int(1152 * scale), int(1024 * scale)
    ax2.add_patch(Rectangle((0, 0), new_w, new_h, fill=False, edgecolor='black', linewidth=2))
    scaled_cx, scaled_cy = cam.cx * scale, cam.cy * scale
    ax2.plot(scaled_cx, scaled_cy, 'r+', markersize=15, markeredgewidth=3)
    ax2.plot(scaled_cx, scaled_cy, 'ro', markersize=25, fillstyle='none', markeredgewidth=2)
    ax2.text(scaled_cx+10, scaled_cy+10, f'PP\n({scaled_cx:.0f}, {scaled_cy:.0f})', fontsize=9, color='red')
    ax2.plot(256, 256, 'b+', markersize=12, markeredgewidth=2)
    ax2.axhline(y=256, color='blue', linestyle='--', alpha=0.5)
    ax2.axvline(x=256, color='blue', linestyle='--', alpha=0.5)
    shift_x, shift_y = 256 - scaled_cx, 256 - scaled_cy
    ax2.annotate('', xy=(256, 256), xytext=(scaled_cx, scaled_cy),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax2.text((scaled_cx+256)/2, (scaled_cy+256)/2 - 20,
             f'shift\n({shift_x:+.1f}, {shift_y:+.1f})', fontsize=9, color='purple', ha='center')
    ax2.set_xlim(-50, 450)
    ax2.set_ylim(400, -50)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Final
    ax3 = axes[2]
    ax3.set_title('(c) Final (512x512, PP=256)', fontsize=11, fontweight='bold')
    ax3.add_patch(Rectangle((0, 0), 512, 512, fill=False, edgecolor='black', linewidth=2))
    ax3.plot(256, 256, 'r+', markersize=15, markeredgewidth=3)
    ax3.plot(256, 256, 'ro', markersize=25, fillstyle='none', markeredgewidth=2)
    ax3.text(256+15, 256+15, 'PP = (256, 256)', fontsize=10, color='red', fontweight='bold')
    ax3.axhline(y=256, color='blue', linestyle='--', alpha=0.5)
    ax3.axvline(x=256, color='blue', linestyle='--', alpha=0.5)
    new_mouse_x = mouse_x * scale + shift_x
    new_mouse_y = mouse_y * scale + shift_y
    ax3.scatter([new_mouse_x], [new_mouse_y], s=300, c='orange', marker='o', alpha=0.7)
    ax3.set_xlim(-20, 530)
    ax3.set_ylim(530, -20)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'fig3_pp_centered_shift.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(output_path)


def fig_scale_mode_comparison(cam: CameraParams, output_dir: Path) -> str:
    """Generate scale mode comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    target_f = 549
    scale_x_raw = target_f / cam.fx
    scale_y_raw = target_f / cam.fy
    scale_avg = (scale_x_raw + scale_y_raw) / 2

    modes = [
        ('D7 (fx_only)', scale_x_raw, scale_x_raw, target_f, target_f),
        ('D7.1 (individual)', scale_x_raw, scale_y_raw, target_f, target_f),
        ('D7.2 (average)', scale_avg, scale_avg, cam.fx * scale_avg, cam.fy * scale_avg),
    ]

    for i, (name, sx, sy, fx_out, fy_out) in enumerate(modes):
        # Top row: Scale visualization
        ax = axes[0, i]
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        orig = Rectangle((0.1, 0.1), 0.3, 0.3, fill=False, edgecolor='gray', linewidth=2, linestyle='--')
        ax.add_patch(orig)
        scaled_w = 0.3 * (sx / scale_x_raw)
        scaled_h = 0.3 * (sy / scale_x_raw)
        scaled = Rectangle((0.5, 0.5 - scaled_h/2), scaled_w, scaled_h,
                           fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2, alpha=0.7)
        ax.add_patch(scaled)
        ax.text(0.25, 0.02, 'Original', ha='center', fontsize=9)
        ax.text(0.5 + scaled_w/2, 0.08, f'sx={sx:.4f}\nsy={sy:.4f}', ha='center', fontsize=9, color='blue')
        aniso = abs(sx - sy) / sx * 100
        color = 'yellow' if aniso > 0.1 else 'lightgreen'
        ax.text(0.5, 0.92, f'Anisotropy: {aniso:.2f}%', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Bottom row: Parameters
        ax2 = axes[1, i]
        fx_ok = abs(fx_out - 549) < 1
        fy_ok = abs(fy_out - 549) < 1
        params = [
            ('scale_x', f'{sx:.6f}', 'white'),
            ('scale_y', f'{sy:.6f}', 'white'),
            ("fx'", f'{fx_out:.1f}', 'lightgreen' if fx_ok else 'lightyellow'),
            ("fy'", f'{fy_out:.1f}', 'lightgreen' if fy_ok else 'lightyellow'),
            ("cx'", '256.0', 'lightgreen'),
            ("cy'", '256.0', 'lightgreen'),
        ]
        table_data = [[p[0], p[1]] for p in params]
        colors = [[p[2], p[2]] for p in params]
        table = ax2.table(cellText=table_data, cellColours=colors,
                         colLabels=['Parameter', 'Value'], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        if name == 'D7 (fx_only)':
            actual_fy = cam.fy * sx
            delta_fy = abs(actual_fy - target_f)
            ray_error = np.degrees(np.arctan(256 * delta_fy / (target_f ** 2)))
            ray_text, ray_color = f'Ray Error: ~{ray_error:.2f}deg', 'orange'
        elif name == 'D7.1 (individual)':
            ray_text, ray_color = 'Ray Error: 0deg', 'green'
        else:
            ray_text, ray_color = 'Ray Error: ~0deg', 'green'
        ax2.text(0.5, -0.1, ray_text, ha='center', fontsize=11, fontweight='bold',
                color=ray_color, transform=ax2.transAxes)
        ax2.axis('off')

    plt.suptitle(f'Scale Mode Comparison (Original: fx={cam.fx:.1f}, fy={cam.fy:.1f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'fig4_scale_mode_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(output_path)


def fig_ray_direction_error(cam: CameraParams, output_dir: Path) -> str:
    """Generate ray direction error visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Ray visualization
    ax1 = axes[0]
    ax1.set_title('Ray Direction with PP Error', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='gray', linewidth=2)
    ax1.text(300, 20, 'Image Plane', fontsize=10, ha='center')
    ax1.plot(256, -300, 'ko', markersize=10)
    ax1.text(256, -330, 'Camera Center', ha='center', fontsize=9)
    ax1.plot([256, 400], [-300, 0], 'g-', linewidth=2, label='Correct Ray')
    ax1.annotate('', xy=(400, 0), xytext=(256, -300),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.plot([256, 456], [-300, 0], 'r--', linewidth=2, label='Incorrect Ray')
    ax1.annotate('', xy=(456, 0), xytext=(256, -300),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='dashed'))
    ax1.annotate('', xy=(430, -50), xytext=(400, -50),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    ax1.text(415, -80, 'Error', fontsize=10, color='purple', ha='center')
    ax1.set_xlim(100, 550)
    ax1.set_ylim(-350, 50)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Right: Calculation
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('Ray Error Calculation', fontsize=12, fontweight='bold')

    scale_x = 549 / cam.fx
    expected_fy = cam.fy * scale_x
    delta_fy = abs(expected_fy - 549)
    ray_error = np.degrees(np.arctan(256 * delta_fy / (549 ** 2)))

    formula = f'''Ray Direction Error Formula
--------------------------
theta = arctan(256 * delta_fy / fy^2)

where delta_fy = |fy_expected - fy_recorded|


Example (D7 fx_only)
--------------------
Original: fx = {cam.fx:.1f}, fy = {cam.fy:.1f}
Scale: s = 549 / {cam.fx:.1f} = {scale_x:.4f}

Expected fy' = {cam.fy:.1f} x {scale_x:.4f} = {expected_fy:.1f}
Recorded fy' = 549.0 (forced)

delta_fy = |{expected_fy:.1f} - 549.0| = {delta_fy:.1f}

theta = arctan(256 x {delta_fy:.1f} / 549^2)
      = {ray_error:.2f} degrees


Risk Levels
-----------
LOW:    < 0.5 deg (acceptable)
MEDIUM: 0.5 - 2 deg (caution)
HIGH:   > 2 deg (ghosting likely)
'''
    ax2.text(0.05, 0.95, formula, fontsize=10, va='top', transform=ax2.transAxes,
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.tight_layout()
    output_path = output_dir / 'fig5_ray_direction_error.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(output_path)


def generate_html_report(cam: CameraParams, figures: Dict[str, str], output_dir: Path) -> str:
    """Generate comprehensive HTML report with MathJax LaTeX."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    scale_x = 549 / cam.fx
    scale_y = 549 / cam.fy
    scale_avg = (scale_x + scale_y) / 2
    shift_x = 256 - cam.cx * scale_x
    shift_y = 256 - cam.cy * scale_y

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>D7 Coordinate System & Preprocessing Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: Georgia, serif; max-width: 1200px; margin: 0 auto; padding: 40px; line-height: 1.8; }}
        h1 {{ font-size: 28px; border-bottom: 3px solid #2c3e50; padding-bottom: 15px; }}
        h2 {{ font-size: 22px; color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }}
        .formula-box {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #3498db; border-radius: 5px; }}
        .figure {{ text-align: center; margin: 30px 0; }}
        .figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .figure-caption {{ font-style: italic; color: #666; margin-top: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: center; }}
        th {{ background: #3498db; color: white; }}
        .highlight {{ background: #e8f6f3; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .toc {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>

<h1>D7 Coordinate System & Preprocessing Analysis</h1>
<p><strong>Generated:</strong> {now}</p>
<p><strong>Camera:</strong> fx={cam.fx:.1f}, fy={cam.fy:.1f}, cx={cam.cx:.1f}, cy={cam.cy:.1f}</p>

<div class="toc">
<h3>Contents</h3>
<ol>
<li><a href="#coord">Coordinate Systems</a></li>
<li><a href="#projection">Projection Pipeline</a></li>
<li><a href="#ppshift">PP-Centered Shift</a></li>
<li><a href="#scalemodes">Scale Mode Comparison</a></li>
<li><a href="#rayerror">Ray Direction Error</a></li>
</ol>
</div>

<h2 id="coord">1. Coordinate Systems</h2>

<div class="figure">
<img src="{Path(figures['coordinate_systems']).name}" alt="Coordinate Systems">
<p class="figure-caption">Figure 1: World, Camera, and Pixel Coordinate Systems</p>
</div>

<div class="formula-box">
<h3>World Coordinates (X_w, Y_w, Z_w)</h3>
<p>3D 공간의 절대 좌표계. 모든 카메라와 객체의 위치를 정의.</p>

<h3>Camera Coordinates (X_c, Y_c, Z_c)</h3>
<p>X_c: 오른쪽, Y_c: 아래쪽, Z_c: 카메라 시선 방향 (광축)</p>

<h3>Pixel Coordinates (u, v)</h3>
<p>Principal Point (cx, cy): 광축이 이미지와 만나는 점</p>
</div>

<h2 id="projection">2. Projection Pipeline</h2>

<div class="figure">
<img src="{Path(figures['projection_pipeline']).name}" alt="Projection Pipeline">
<p class="figure-caption">Figure 2: Camera Projection Pipeline</p>
</div>

<div class="formula-box">
<h3>Complete Projection</h3>
\\[
\\begin{{bmatrix}} u \\\\ v \\\\ 1 \\end{{bmatrix}} \\sim
\\mathbf{{K}} \\cdot [\\mathbf{{R}} | \\mathbf{{t}}] \\cdot
\\begin{{bmatrix}} X \\\\ Y \\\\ Z \\\\ 1 \\end{{bmatrix}}
\\]

<p>where:</p>
\\[
\\mathbf{{K}} = \\begin{{bmatrix}} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{{bmatrix}}
\\]
</div>

<h2 id="ppshift">3. PP-Centered Shift</h2>

<div class="figure">
<img src="{Path(figures['pp_centered_shift']).name}" alt="PP-Centered Shift">
<p class="figure-caption">Figure 3: PP-Centered Shift Process</p>
</div>

<div class="formula-box">
<h3>Scale & Shift Calculation</h3>
\\[
\\text{{scale}}_x = \\frac{{549}}{{{cam.fx:.1f}}} = {scale_x:.6f}
\\]
\\[
\\text{{shift}}_x = 256 - ({cam.cx:.1f} \\times {scale_x:.4f}) = {shift_x:+.1f}
\\]
\\[
\\text{{shift}}_y = 256 - ({cam.cy:.1f} \\times {scale_y:.4f}) = {shift_y:+.1f}
\\]
</div>

<h2 id="scalemodes">4. Scale Mode Comparison</h2>

<div class="figure">
<img src="{Path(figures['scale_mode_comparison']).name}" alt="Scale Mode Comparison">
<p class="figure-caption">Figure 4: D7, D7.1, D7.2 Comparison</p>
</div>

<table>
<tr><th>Mode</th><th>scale_x</th><th>scale_y</th><th>fx'</th><th>fy'</th><th>Ray Error</th></tr>
<tr><td>D7 (fx_only)</td><td>{scale_x:.6f}</td><td>{scale_x:.6f}</td><td>549.0</td><td>549.0</td><td>~0.4°</td></tr>
<tr style="background:#e8f6f3"><td><b>D7.1 (individual)</b></td><td>{scale_x:.6f}</td><td>{scale_y:.6f}</td><td>549.0</td><td>549.0</td><td><b>0°</b></td></tr>
<tr><td>D7.2 (average)</td><td>{scale_avg:.6f}</td><td>{scale_avg:.6f}</td><td>{cam.fx*scale_avg:.1f}</td><td>{cam.fy*scale_avg:.1f}</td><td>~0°</td></tr>
</table>

<h2 id="rayerror">5. Ray Direction Error</h2>

<div class="figure">
<img src="{Path(figures['ray_direction_error']).name}" alt="Ray Error">
<p class="figure-caption">Figure 5: Ray Direction Error Analysis</p>
</div>

<div class="formula-box">
<h3>Ray Error Formula</h3>
\\[
\\theta_{{error}} = \\arctan\\left(\\frac{{256 \\cdot \\Delta f_y}}{{f_y^2}}\\right)
\\]
</div>

<div class="highlight">
<h3>Recommendation</h3>
<p><b>D7.1 (individual)</b>: Ray error = 0, geometrically correct</p>
</div>

<hr>
<p style="color:#888; font-size:12px;">Generated by coordinate_report_generator.py | {now}</p>
</body>
</html>
'''
    output_path = output_dir / 'coordinate_analysis_report.html'
    with open(output_path, 'w') as f:
        f.write(html)
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate coordinate report")
    parser.add_argument('--camera-pkl', type=str)
    parser.add_argument('--output', '-o', type=str, default='reports/coordinate_analysis')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.camera_pkl:
        cameras = load_cameras_from_pkl(args.camera_pkl)
        cam = cameras[0]
        print(f"Loaded camera from {args.camera_pkl}")
    else:
        cam = create_default_camera()
        print("Using default camera")

    print(f"Camera: fx={cam.fx:.1f}, fy={cam.fy:.1f}, cx={cam.cx:.1f}, cy={cam.cy:.1f}")
    print(f"Output: {output_dir}\n")

    print("Generating figures...")
    figures = {}
    figures['coordinate_systems'] = fig_coordinate_systems(output_dir)
    print("  [1/5] Coordinate systems")
    figures['projection_pipeline'] = fig_projection_pipeline(output_dir)
    print("  [2/5] Projection pipeline")
    figures['pp_centered_shift'] = fig_pp_centered_shift(cam, output_dir)
    print("  [3/5] PP-centered shift")
    figures['scale_mode_comparison'] = fig_scale_mode_comparison(cam, output_dir)
    print("  [4/5] Scale mode comparison")
    figures['ray_direction_error'] = fig_ray_direction_error(cam, output_dir)
    print("  [5/5] Ray direction error")

    print("\nGenerating HTML report...")
    report_path = generate_html_report(cam, figures, output_dir)

    print(f"\nComplete!")
    print(f"Report: {report_path}")
    print(f"Figures: {len(figures)}")


if __name__ == "__main__":
    main()
