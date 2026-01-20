#!/usr/bin/env python3
"""
Preprocessing Report Generator v2
=================================

Generates comprehensive preprocessing reports with:
- Dataset analysis and verification
- Mathematical formula derivation (LaTeX)
- Visualization figures (matplotlib)
- Comparison tables

Usage:
    python generate_report.py --datasets D7,D7_1,D7_2 --output reports/comparison
    python generate_report.py --datasets D7 --with-figures
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, figures will be skipped")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReportConfig:
    """Report generation configuration."""
    title: str = "Preprocessing Comparison Report"
    base_path: str = "/home/joon/data/preprocessed/FaceLift_mouse"
    raw_data_path: str = "/home/joon/data/markerless_mouse_1_nerf"
    output_dir: str = "/home/joon/dev/FaceLift/mouse_extensions/reports/generated"
    max_samples: int = 50
    generate_figures: bool = True
    figure_dpi: int = 150


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_cameras(dataset_path: str, max_samples: int = 50) -> List[Dict]:
    """Load camera parameters from a dataset."""
    cameras = []
    dataset_path = Path(dataset_path)

    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        samples = sorted(os.listdir(split_path))[:max_samples]
        for sample in samples:
            cam_file = split_path / sample / 'opencv_cameras.json'
            if cam_file.exists():
                with open(cam_file) as f:
                    data = json.load(f)
                    for frame in data.get('frames', []):
                        frame['_sample'] = sample
                        frame['_split'] = split
                        frame['_dataset_path'] = str(dataset_path)
                        cameras.append(frame)
    return cameras


def load_sample_image(dataset_path: str, sample: str = "train/000000", view: int = 0) -> Optional[np.ndarray]:
    """Load a sample image from dataset."""
    if not HAS_PIL:
        return None
    path = Path(dataset_path) / sample / "images" / f"cam_{view:03d}.png"
    if path.exists():
        return np.array(Image.open(path))
    return None


def analyze_dataset(cameras: List[Dict], name: str) -> Dict:
    """Analyze camera parameters for a dataset."""
    if not cameras:
        return {'name': name, 'error': 'No cameras found'}

    # Extract parameters
    fx = [c['fx'] for c in cameras]
    fy = [c['fy'] for c in cameras]
    cx = [c['cx'] for c in cameras]
    cy = [c['cy'] for c in cameras]

    # Original parameters
    orig_fx, orig_fy, orig_cx, orig_cy = [], [], [], []
    for c in cameras:
        if '_original' in c:
            o = c['_original']
            orig_fx.append(o.get('fx', 0))
            orig_fy.append(o.get('fy', 0))
            orig_cx.append(o.get('cx', 0))
            orig_cy.append(o.get('cy', 0))

    # Transform metadata
    scale_mode = 'unknown'
    scale_x, scale_y = [], []
    for c in cameras:
        if '_transform' in c:
            t = c['_transform']
            scale_mode = t.get('scale_mode', t.get('method', 'unknown'))
            if 'scale_x' in t:
                scale_x.append(t['scale_x'])
                scale_y.append(t['scale_y'])
            elif 'scale' in t:
                scale_x.append(t['scale'])
                scale_y.append(t['scale'])
            elif 'scale_avg' in t:
                scale_x.append(t['scale_avg'])
                scale_y.append(t['scale_avg'])

    # Compute ray error
    ray_errors = []
    for c in cameras:
        if '_original' in c and '_transform' in c:
            o = c['_original']
            t = c['_transform']
            # Get scale
            if 'scale_y' in t:
                s = t['scale_y']
            elif 'scale' in t:
                s = t['scale']
            elif 'scale_avg' in t:
                s = t['scale_avg']
            else:
                continue

            expected_fy = o['fy'] * s
            recorded_fy = c['fy']
            delta = abs(expected_fy - recorded_fy)
            if recorded_fy > 0:
                angle = np.degrees(np.arctan(256 * delta / (recorded_fy ** 2)))
                ray_errors.append(angle)

    # Dataset path for image loading
    dataset_path = cameras[0].get('_dataset_path', '') if cameras else ''

    return {
        'name': name,
        'count': len(cameras),
        'scale_mode': scale_mode,
        'dataset_path': dataset_path,
        'fx': {'mean': np.mean(fx), 'std': np.std(fx), 'min': np.min(fx), 'max': np.max(fx), 'values': fx},
        'fy': {'mean': np.mean(fy), 'std': np.std(fy), 'min': np.min(fy), 'max': np.max(fy), 'values': fy},
        'cx': {'mean': np.mean(cx), 'std': np.std(cx), 'values': cx},
        'cy': {'mean': np.mean(cy), 'std': np.std(cy), 'values': cy},
        'orig_fx': {'mean': np.mean(orig_fx), 'values': orig_fx} if orig_fx else None,
        'orig_fy': {'mean': np.mean(orig_fy), 'values': orig_fy} if orig_fy else None,
        'scale_x': {'mean': np.mean(scale_x), 'std': np.std(scale_x), 'values': scale_x} if scale_x else None,
        'scale_y': {'mean': np.mean(scale_y), 'std': np.std(scale_y), 'values': scale_y} if scale_y else None,
        'ray_error': {'mean': np.mean(ray_errors), 'max': np.max(ray_errors), 'values': ray_errors} if ray_errors else None,
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def generate_parameter_distribution_figure(analyses: List[Dict], output_dir: Path) -> Optional[str]:
    """Generate parameter distribution comparison figure."""
    if not HAS_MATPLOTLIB:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Camera Parameter Distribution Comparison", fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(analyses)))

    params = [('fx', 'Focal Length X (fx)'), ('fy', 'Focal Length Y (fy)'),
              ('cx', 'Principal Point X (cx)'), ('cy', 'Principal Point Y (cy)')]

    for ax, (param, title) in zip(axes.flat, params):
        for i, a in enumerate(analyses):
            if 'error' in a:
                continue
            values = a[param].get('values', [])
            if values:
                ax.hist(values, bins=30, alpha=0.5, label=a['name'], color=colors[i])

        ax.set_xlabel(param)
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()
        ax.axvline(x=549 if param in ['fx', 'fy'] else 256, color='red',
                   linestyle='--', alpha=0.7, label='Target')

    plt.tight_layout()
    output_path = output_dir / "fig_parameter_distribution.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(output_path)


def generate_scale_comparison_figure(analyses: List[Dict], output_dir: Path) -> Optional[str]:
    """Generate scale factor comparison figure."""
    if not HAS_MATPLOTLIB:
        return None

    # Filter analyses with scale data
    valid = [a for a in analyses if a.get('scale_x') and a.get('scale_y')]
    if not valid:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scale X vs Scale Y scatter
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))

    for i, a in enumerate(valid):
        sx = a['scale_x']['values']
        sy = a['scale_y']['values']
        ax1.scatter(sx, sy, alpha=0.5, label=a['name'], color=colors[i], s=20)

    # Add diagonal line (isotropic)
    lim = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
           max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
    ax1.plot(lim, lim, 'k--', alpha=0.5, label='Isotropic (sx=sy)')
    ax1.set_xlabel('Scale X')
    ax1.set_ylabel('Scale Y')
    ax1.set_title('Scale Factor Distribution')
    ax1.legend()
    ax1.set_aspect('equal')

    # Right: Bar chart of mean scales
    ax2 = axes[1]
    names = [a['name'] for a in valid]
    x = np.arange(len(names))
    width = 0.35

    sx_means = [a['scale_x']['mean'] for a in valid]
    sy_means = [a['scale_y']['mean'] for a in valid]

    bars1 = ax2.bar(x - width/2, sx_means, width, label='Scale X', alpha=0.8)
    bars2 = ax2.bar(x + width/2, sy_means, width, label='Scale Y', alpha=0.8)

    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Scale Factor')
    ax2.set_title('Mean Scale Factors by Dataset')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()

    # Add difference annotation
    for i, (sx, sy) in enumerate(zip(sx_means, sy_means)):
        diff_pct = abs(sx - sy) / sx * 100
        ax2.annotate(f'{diff_pct:.2f}%', xy=(i, max(sx, sy) + 0.001),
                     ha='center', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "fig_scale_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(output_path)


def generate_sample_images_figure(analyses: List[Dict], output_dir: Path) -> Optional[str]:
    """Generate sample images with PP markers."""
    if not HAS_MATPLOTLIB or not HAS_PIL:
        return None

    # Find analyses with valid paths
    valid = [a for a in analyses if a.get('dataset_path') and 'error' not in a]
    if not valid:
        return None

    n_datasets = len(valid)
    n_views = 6

    fig, axes = plt.subplots(n_datasets, n_views, figsize=(18, 3*n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Sample Images with Principal Point Markers", fontsize=14, fontweight='bold')

    for i, a in enumerate(valid):
        for j in range(n_views):
            ax = axes[i, j]
            img = load_sample_image(a['dataset_path'], "train/000000", j)

            if img is not None:
                ax.imshow(img)
                # Draw PP marker
                cx, cy = a['cx']['mean'], a['cy']['mean']
                ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2, label='PP')
                ax.plot(256, 256, 'gx', markersize=12, markeredgewidth=2, label='Center')
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
            else:
                ax.text(0.5, 0.5, 'No Image', ha='center', va='center', transform=ax.transAxes)

            ax.axis('off')
            if i == 0:
                ax.set_title(f'View {j}')
            if j == 0:
                ax.set_ylabel(a['name'], fontsize=12, rotation=0, ha='right', va='center')

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='+', color='red', linestyle='None', markersize=10, label='Principal Point'),
        Line2D([0], [0], marker='x', color='green', linestyle='None', markersize=10, label='Image Center (256,256)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "fig_sample_images.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(output_path)


def generate_ray_error_figure(analyses: List[Dict], output_dir: Path) -> Optional[str]:
    """Generate ray error analysis figure."""
    if not HAS_MATPLOTLIB:
        return None

    valid = [a for a in analyses if a.get('ray_error')]
    if not valid:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Box plot
    ax1 = axes[0]
    data = [a['ray_error']['values'] for a in valid]
    names = [a['name'] for a in valid]

    bp = ax1.boxplot(data, labels=names, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Ray Direction Error (degrees)')
    ax1.set_title('Ray Error Distribution')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Low Risk Threshold')
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
    ax1.legend()

    # Right: Bar chart with risk levels
    ax2 = axes[1]
    means = [a['ray_error']['mean'] for a in valid]
    maxs = [a['ray_error']['max'] for a in valid]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, means, width, label='Mean Error', alpha=0.8)
    bars2 = ax2.bar(x + width/2, maxs, width, label='Max Error', alpha=0.8)

    # Color bars by risk level
    for bar, val in zip(bars2, maxs):
        if val < 0.5:
            bar.set_color('green')
        elif val < 2.0:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Ray Error (degrees)')
    ax2.set_title('Ray Error by Dataset')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()

    plt.tight_layout()
    output_path = output_dir / "fig_ray_error.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(output_path)


# =============================================================================
# Report Generation
# =============================================================================

def generate_html_report(analyses: List[Dict], output_path: Path, title: str = "Preprocessing Report",
                         figures: Dict[str, str] = None):
    """Generate HTML report with LaTeX formulas and figures."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    figures = figures or {}

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: "CMU Serif", Georgia, serif; max-width: 1200px; margin: 0 auto; padding: 40px; line-height: 1.8; background: #fafafa; }}
        h1 {{ font-size: 28px; border-bottom: 3px solid #333; padding-bottom: 15px; }}
        h2 {{ font-size: 22px; color: #2c3e50; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }}
        h3 {{ font-size: 18px; color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; background: white; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: center; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        .formula-box {{ background: white; padding: 20px; margin: 20px 0; border-left: 4px solid #3498db; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .key-point {{ background: #e8f6f3; padding: 15px; border-left: 4px solid #1abc9c; margin: 15px 0; }}
        .warning {{ background: #fdf2e9; padding: 15px; border-left: 4px solid #e67e22; }}
        .good {{ background: #e8f5e9; }}
        .bad {{ background: #ffebee; }}
        .recommended {{ background: #e3f2fd; border-left: 4px solid #2196f3; padding: 15px; margin: 15px 0; }}
        code {{ background: #f5f5f5; padding: 2px 6px; font-family: monospace; border-radius: 3px; }}
        pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; overflow-x: auto; border-radius: 5px; }}
        .figure {{ text-align: center; margin: 30px 0; }}
        .figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .figure-caption {{ font-style: italic; color: #666; margin-top: 10px; }}
        .toc {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .toc ul {{ list-style-type: none; padding-left: 0; }}
        .toc li {{ margin: 8px 0; }}
        .toc a {{ text-decoration: none; color: #3498db; }}
        .toc a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>

<h1>{title}</h1>
<p><strong>Generated:</strong> {now}</p>

<div class="toc">
<h3>Table of Contents</h3>
<ul>
<li><a href="#summary">1. Executive Summary</a></li>
<li><a href="#theory">2. Mathematical Foundation</a></li>
<li><a href="#analysis">3. Detailed Analysis</a></li>
<li><a href="#visualization">4. Visualization</a></li>
<li><a href="#recommendations">5. Recommendations</a></li>
<li><a href="#commands">6. Preprocessing Commands</a></li>
</ul>
</div>

<h2 id="summary">1. Executive Summary</h2>

<table>
<tr><th>Dataset</th><th>Scale Mode</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Ray Error</th><th>Status</th></tr>
"""

    for a in analyses:
        if 'error' in a:
            html += f'<tr><td>{a["name"]}</td><td colspan="7">{a["error"]}</td></tr>\n'
            continue
        fx_str = f"{a['fx']['mean']:.1f}" if a['fx']['std'] < 0.1 else f"~{a['fx']['mean']:.1f}"
        fy_str = f"{a['fy']['mean']:.1f}" if a['fy']['std'] < 0.1 else f"~{a['fy']['mean']:.1f}"
        cx_str = f"{a['cx']['mean']:.1f}"
        cy_str = f"{a['cy']['mean']:.1f}"
        ray_str = f"~{a['ray_error']['mean']:.2f}°" if a['ray_error'] else "N/A"

        # Determine status
        is_good = a['fx']['std'] < 1 and a['fy']['std'] < 1 and abs(a['cx']['mean'] - 256) < 1
        ray_ok = a['ray_error'] is None or a['ray_error']['max'] < 0.5
        status = "✅ Good" if (is_good and ray_ok) else "⚠️ Check"
        row_class = "good" if (is_good and ray_ok) else ""

        html += f'<tr class="{row_class}"><td><strong>{a["name"]}</strong></td><td>{a["scale_mode"]}</td>'
        html += f'<td>{fx_str}</td><td>{fy_str}</td><td>{cx_str}</td><td>{cy_str}</td>'
        html += f'<td>{ray_str}</td><td>{status}</td></tr>\n'

    html += """</table>

<h2 id="theory">2. Mathematical Foundation</h2>

<div class="formula-box">
<h3>2.1 Pinhole Camera Model</h3>
<p>The projection from 3D world coordinates to 2D pixel coordinates:</p>
\\[
\\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} \\sim
\\mathbf{K} \\cdot [\\mathbf{R} | \\mathbf{T}] \\cdot
\\begin{bmatrix} X \\\\ Y \\\\ Z \\\\ 1 \\end{bmatrix}
\\]
<p>where \\(\\mathbf{K}\\) is the intrinsic matrix:</p>
\\[
\\mathbf{K} = \\begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \\end{bmatrix}
\\]
</div>

<div class="formula-box">
<h3>2.2 Ray Direction Calculation</h3>
<p>For novel view synthesis, the ray direction from pixel (u, v) in camera coordinates:</p>
\\[
\\mathbf{d}_c = \\begin{bmatrix} (u - c_x) / f_x \\\\ (v - c_y) / f_y \\\\ 1 \\end{bmatrix}
\\]
<p><strong>Critical:</strong> Incorrect \\(c_x, c_y\\) leads to wrong ray directions, causing multi-view inconsistency and ghosting artifacts.</p>
</div>

<div class="formula-box">
<h3>2.3 Ray Error Quantification</h3>
<p>When fy is incorrectly recorded, the ray direction error at image edge (256 pixels from center):</p>
\\[
\\theta_{error} = \\arctan\\left(\\frac{256 \\cdot |f_y^{expected} - f_y^{recorded}|}{(f_y)^2}\\right)
\\]
<p>Risk levels: <span style="color:green">LOW (&lt;0.5°)</span>, <span style="color:orange">MEDIUM (0.5-2°)</span>, <span style="color:red">HIGH (&gt;2°)</span></p>
</div>

<div class="formula-box">
<h3>2.4 Scale Factor Calculations</h3>

<p><strong>fx_only (D7):</strong> Same scale for both axes, fy forced to target value</p>
\\[
s = \\frac{f_x^{target}}{f_x^{orig}}, \\quad f_x' = f_y' = 549
\\]

<p><strong>individual (D7.1):</strong> Different scales for exact focal lengths</p>
\\[
s_x = \\frac{f_x^{target}}{f_x^{orig}}, \\quad s_y = \\frac{f_y^{target}}{f_y^{orig}}, \\quad f_x' = f_y' = 549 \\text{ (exact)}
\\]

<p><strong>average (D7.2):</strong> Isotropic scaling with averaged scale factor</p>
\\[
s = \\frac{s_x + s_y}{2}, \\quad f_x' = f_x^{orig} \\cdot s, \\quad f_y' = f_y^{orig} \\cdot s
\\]
</div>

<div class="formula-box">
<h3>2.5 PP-Centered Shift</h3>
<p>After scaling, shift image to center Principal Point at (256, 256):</p>
\\[
\\Delta_x = 256 - c_x^{scaled}, \\quad \\Delta_y = 256 - c_y^{scaled}
\\]
<p>The complete affine transform:</p>
\\[
\\begin{bmatrix} u' \\\\ v' \\end{bmatrix} =
\\begin{bmatrix} s_x & 0 \\\\ 0 & s_y \\end{bmatrix}
\\begin{bmatrix} u \\\\ v \\end{bmatrix} +
\\begin{bmatrix} \\Delta_x \\\\ \\Delta_y \\end{bmatrix}
\\]
</div>

<h2 id="analysis">3. Detailed Analysis</h2>
"""

    for idx, a in enumerate(analyses):
        if 'error' in a:
            continue

        is_recommended = 'D7.1' in a['name'] or 'D7_1' in a['name']
        div_class = "recommended" if is_recommended else ""

        html += f"""<h3>3.{idx+1} {a['name']}</h3>
<div class="{div_class}">
{'<p><strong>★ RECOMMENDED</strong></p>' if is_recommended else ''}
<p><strong>Scale Mode:</strong> <code>{a['scale_mode']}</code></p>
<p><strong>Camera Views Analyzed:</strong> {a['count']}</p>

<table>
<tr><th>Parameter</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Target</th><th>Status</th></tr>
<tr>
    <td>fx</td>
    <td>{a['fx']['mean']:.4f}</td>
    <td>{a['fx']['std']:.4f}</td>
    <td>{a['fx']['min']:.4f}</td>
    <td>{a['fx']['max']:.4f}</td>
    <td>549.0</td>
    <td>{'✅' if abs(a['fx']['mean'] - 549) < 1 else '⚠️'}</td>
</tr>
<tr>
    <td>fy</td>
    <td>{a['fy']['mean']:.4f}</td>
    <td>{a['fy']['std']:.4f}</td>
    <td>{a['fy']['min']:.4f}</td>
    <td>{a['fy']['max']:.4f}</td>
    <td>549.0</td>
    <td>{'✅' if abs(a['fy']['mean'] - 549) < 1 else '⚠️'}</td>
</tr>
<tr>
    <td>cx</td>
    <td>{a['cx']['mean']:.4f}</td>
    <td>{a['cx']['std']:.4f}</td>
    <td>-</td>
    <td>-</td>
    <td>256.0</td>
    <td>{'✅' if abs(a['cx']['mean'] - 256) < 1 else '⚠️'}</td>
</tr>
<tr>
    <td>cy</td>
    <td>{a['cy']['mean']:.4f}</td>
    <td>{a['cy']['std']:.4f}</td>
    <td>-</td>
    <td>-</td>
    <td>256.0</td>
    <td>{'✅' if abs(a['cy']['mean'] - 256) < 1 else '⚠️'}</td>
</tr>
</table>
"""

        if a['scale_x'] and a['scale_y']:
            diff = abs(a['scale_x']['mean'] - a['scale_y']['mean'])
            diff_pct = diff / a['scale_x']['mean'] * 100
            html += f"""
<p><strong>Scale Factors:</strong></p>
<ul>
<li>scale_x: {a['scale_x']['mean']:.6f} (std: {a['scale_x']['std']:.6f})</li>
<li>scale_y: {a['scale_y']['mean']:.6f} (std: {a['scale_y']['std']:.6f})</li>
<li>Anisotropy: {diff:.6f} ({diff_pct:.3f}%)</li>
</ul>
"""

        if a['ray_error']:
            risk = 'LOW' if a['ray_error']['max'] < 0.5 else 'MEDIUM' if a['ray_error']['max'] < 2 else 'HIGH'
            risk_color = 'green' if risk == 'LOW' else 'orange' if risk == 'MEDIUM' else 'red'
            html += f"""
<p><strong>Ray Error Analysis:</strong></p>
<ul>
<li>Mean: {a['ray_error']['mean']:.4f}°</li>
<li>Max: {a['ray_error']['max']:.4f}°</li>
<li>Risk Level: <span style="color:{risk_color}; font-weight:bold">{risk}</span></li>
</ul>
"""

        html += "</div>\n"

    # Visualization section
    html += """<h2 id="visualization">4. Visualization</h2>
"""

    if figures:
        for fig_name, fig_path in figures.items():
            if fig_path:
                rel_path = Path(fig_path).name
                caption = fig_name.replace('_', ' ').title()
                html += f"""
<div class="figure">
<img src="{rel_path}" alt="{caption}">
<p class="figure-caption">Figure: {caption}</p>
</div>
"""
    else:
        html += """<p class="warning">⚠️ Figures not generated. Run with <code>--with-figures</code> to enable visualization.</p>
"""

    html += """
<h2 id="recommendations">5. Recommendations</h2>

<div class="key-point">
<h3>For New Experiments</h3>
<ol>
<li><strong>D7.1 (individual)</strong> - Recommended for geometric accuracy
    <ul>
    <li>Exact fx=fy=549 with different scale factors</li>
    <li>~0.6% anisotropy (visually negligible)</li>
    <li>Zero ray direction error</li>
    </ul>
</li>
<li><strong>D7.2 (average)</strong> - Alternative for isotropic scaling
    <ul>
    <li>Isotropic scaling preserves aspect ratio</li>
    <li>fx, fy may vary slightly from 549</li>
    </ul>
</li>
<li><strong>D7 (fx_only)</strong> - Current production
    <ul>
    <li>Validated performance</li>
    <li>Small ray error (~0.4°, LOW risk)</li>
    </ul>
</li>
</ol>
</div>

<h2 id="commands">6. Preprocessing Commands</h2>

<pre>
# D7.1 (Recommended) - Individual scale factors
python -m mouse_extensions.scripts.preprocess_D7_pp_centered \\
    --data-dir /home/joon/data/markerless_mouse_1_nerf \\
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \\
    --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \\
    --frame-interval 5 \\
    --scale-mode individual

# D7.2 (Alternative) - Average scale factor
python -m mouse_extensions.scripts.preprocess_D7_pp_centered \\
    --data-dir /home/joon/data/markerless_mouse_1_nerf \\
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_2 \\
    --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \\
    --frame-interval 5 \\
    --scale-mode average
</pre>

<hr>
<p style="color:#888; font-size:12px;">
Report generated by <code>generate_report.py</code> | FaceLift Mouse Extension<br>
Template version: v2.0 | {now}
</p>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate preprocessing comparison report")
    parser.add_argument('--datasets', '-d', type=str,
                        help='Comma-separated dataset names (e.g., D7,D7_1,D7_2)')
    parser.add_argument('--base-path', type=str,
                        default='/home/joon/data/preprocessed/FaceLift_mouse',
                        help='Base path for datasets')
    parser.add_argument('--output', '-o', type=str,
                        default='/home/joon/dev/FaceLift/mouse_extensions/reports/generated',
                        help='Output directory')
    parser.add_argument('--title', type=str,
                        default='Preprocessing Comparison Report',
                        help='Report title')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Max samples to analyze per dataset')
    parser.add_argument('--with-figures', action='store_true',
                        help='Generate visualization figures')

    args = parser.parse_args()

    # Parse datasets
    if args.datasets:
        dataset_names = [d.strip() for d in args.datasets.split(',')]
    else:
        dataset_names = ['D7', 'D7_1_test', 'D7_2_test']

    # Analyze datasets
    print(f"Analyzing datasets: {dataset_names}")
    analyses = []
    for name in dataset_names:
        path = f"{args.base_path}/{name}"
        print(f"  Loading {name} from {path}...")
        cameras = load_dataset_cameras(path, args.max_samples)
        if cameras:
            analysis = analyze_dataset(cameras, name)
            analyses.append(analysis)
            print(f"    Found {len(cameras)} camera views, scale_mode={analysis.get('scale_mode', 'unknown')}")
        else:
            analyses.append({'name': name, 'error': f'No data found at {path}'})
            print(f"    No data found")

    # Generate figures
    figures = {}
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.with_figures and HAS_MATPLOTLIB:
        print("\nGenerating visualization figures...")
        figures['parameter_distribution'] = generate_parameter_distribution_figure(analyses, output_dir)
        figures['scale_comparison'] = generate_scale_comparison_figure(analyses, output_dir)
        figures['sample_images'] = generate_sample_images_figure(analyses, output_dir)
        figures['ray_error'] = generate_ray_error_figure(analyses, output_dir)
        print(f"  Generated {len([f for f in figures.values() if f])} figures")

    # Generate report
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    output_path = output_dir / f"{timestamp}_comparison_report.html"

    generate_html_report(analyses, output_path, args.title, figures)
    print(f"\n✅ Report saved: {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for a in analyses:
        if 'error' in a:
            print(f"  {a['name']}: {a['error']}")
        else:
            ray = f"ray_error={a['ray_error']['mean']:.3f}°" if a['ray_error'] else "ray_error=N/A"
            print(f"  {a['name']}: fx={a['fx']['mean']:.1f}, fy={a['fy']['mean']:.1f}, {ray}")


if __name__ == "__main__":
    main()
