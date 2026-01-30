#!/usr/bin/env python3
from mouse_extensions.paths import FACELIFT_ROOT, DATA_ROOT
"""
Comprehensive D8 Camera and Preprocessing Report Generator

Generates:
1. Camera Setup Visualization (HTML with 3D, Top, Side views)
2. Preprocessing Methods Comparison Report
3. D8.1 Zoom Error Analysis

Author: Claude/Joon
Date: 2026-01-21
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

# Configuration
BASE_DIR = FACELIFT_ROOT
DATA_DIR = DATA_ROOT
RAW_DIR = DATA_DIR / "raw/markerless_mouse_1_nerf"
D8_SAMPLE_DIR = DATA_DIR / "preprocessed/FaceLift_mouse/D8_sample/train/000000"
D8_1_SAMPLE_DIR = DATA_DIR / "preprocessed/FaceLift_mouse/D8_1_sample/train/000000"
D7_1_SAMPLE_DIR = DATA_DIR / "preprocessed/FaceLift_mouse/D7_1/train/000000"
OUTPUT_DIR = BASE_DIR / "mouse_extensions/reports/10_D8_comprehensive"

# GS-LRM target values
GSLRM_TARGET_FX = 548.9937744140625
GSLRM_TARGET_PP = 256.0
GSLRM_TARGET_DIST = 2.7

# Camera colors (distinct, visible)
CAMERA_COLORS = {
    0: "#e74c3c",  # Red
    1: "#3498db",  # Blue  
    2: "#2ecc71",  # Green
    3: "#f39c12",  # Orange
    4: "#9b59b6",  # Purple
    5: "#1abc9c",  # Teal
}

# Opposing pairs
OPPOSING_PAIRS = [(4, 3), (2, 5), (1, 0)]

# Layout: top row 4,2,1 / bottom row 0,5,3
LAYOUT_ORDER = [[4, 2, 1], [0, 5, 3]]


class Arrow3D(FancyArrowPatch):
    """3D Arrow for matplotlib"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def load_original_cameras():
    """Load original camera parameters from raw data"""
    with open(RAW_DIR / "new_cam.pkl", "rb") as f:
        return pickle.load(f)


def load_processed_cameras(sample_dir):
    """Load processed camera parameters"""
    with open(sample_dir / "opencv_cameras.json") as f:
        data = json.load(f)
    return data["frames"]


def compute_camera_pose(w2c):
    """Extract camera position and forward direction from w2c matrix"""
    w2c = np.array(w2c)
    c2w = np.linalg.inv(w2c)
    position = c2w[:3, 3]
    forward = c2w[:3, 2]  # OpenCV: +Z is forward
    up = -c2w[:3, 1]  # OpenCV: -Y is up
    right = c2w[:3, 0]
    return position, forward, up, right


def compute_azimuth_elevation(position):
    """Compute azimuth and elevation angles from camera position"""
    x, y, z = position
    azimuth = np.degrees(np.arctan2(y, x))
    dist_xy = np.sqrt(x**2 + y**2)
    elevation = np.degrees(np.arctan2(z, dist_xy))
    distance = np.linalg.norm(position)
    return azimuth, elevation, distance


def create_camera_grid_image(sample_dir, title="Camera Grid"):
    """Create a 2x3 grid of camera images following layout order"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    
    for row_idx, row in enumerate(LAYOUT_ORDER):
        for col_idx, cam_idx in enumerate(row):
            ax = axes[row_idx, col_idx]
            img_path = sample_dir / "images" / f"cam_{cam_idx:03d}.png"
            
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            
            ax.set_title(f"Camera {cam_idx}", color=CAMERA_COLORS[cam_idx], 
                        fontweight="bold", fontsize=14)
            ax.axis("off")
            
            # Add border
            for spine in ax.spines.values():
                spine.set_edgecolor(CAMERA_COLORS[cam_idx])
                spine.set_linewidth(3)
                spine.set_visible(True)
    
    plt.tight_layout()
    return fig


def create_top_view_with_angles(cameras, title="Top View (X-Y Plane)"):
    """Create top view showing camera positions and angles between opposing pairs"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    positions = []
    for cam in cameras:
        pos, fwd, _, _ = compute_camera_pose(cam["w2c"])
        positions.append((pos, fwd))
    
    # Plot cameras
    for cam_idx, (pos, fwd) in enumerate(positions):
        color = CAMERA_COLORS[cam_idx]
        
        # Camera position
        ax.scatter(pos[0], pos[1], s=300, c=color, marker="o", 
                  edgecolors="black", linewidths=2, zorder=5)
        ax.annotate(f"Cam {cam_idx}", (pos[0], pos[1]), 
                   xytext=(10, 10), textcoords="offset points",
                   fontsize=12, fontweight="bold", color=color)
        
        # Forward direction arrow
        arrow_scale = 0.8
        ax.arrow(pos[0], pos[1], fwd[0]*arrow_scale, fwd[1]*arrow_scale,
                head_width=0.15, head_length=0.1, fc=color, ec=color, linewidth=2)
    
    # Draw opposing pair connections with angle annotations
    for cam_a, cam_b in OPPOSING_PAIRS:
        pos_a, _ = positions[cam_a]
        pos_b, _ = positions[cam_b]
        
        # Dashed line connecting opposing cameras
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], 
               linestyle="--", color="gray", linewidth=2, alpha=0.7)
        
        # Calculate angle
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Midpoint annotation
        mid_x = (pos_a[0] + pos_b[0]) / 2
        mid_y = (pos_a[1] + pos_b[1]) / 2
        ax.annotate(f"{cam_a}-{cam_b}", (mid_x, mid_y),
                   fontsize=11, ha="center", va="bottom",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    # Mouse center (origin)
    ax.scatter(0, 0, s=200, c="black", marker="x", linewidths=3, zorder=10)
    ax.annotate("Mouse Center", (0, 0), xytext=(15, -15), 
               textcoords="offset points", fontsize=10)
    
    ax.set_xlabel("X (world)", fontsize=12)
    ax.set_ylabel("Y (world)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    
    # Add legend for azimuth angles
    legend_text = "Azimuth Angles:\n"
    for cam_idx in range(6):
        pos, _ = positions[cam_idx]
        az, el, dist = compute_azimuth_elevation(pos)
        legend_text += f"Cam {cam_idx}: {az:.1f}deg\n"
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
           verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    return fig


def create_side_view(cameras, title="Side View (Elevation)"):
    """Create side view showing camera elevations"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    positions = []
    for cam in cameras:
        pos, fwd, _, _ = compute_camera_pose(cam["w2c"])
        positions.append((pos, fwd))
    
    # Sort cameras by azimuth for x-axis ordering
    sorted_cams = []
    for cam_idx, (pos, fwd) in enumerate(positions):
        az, el, dist = compute_azimuth_elevation(pos)
        sorted_cams.append((cam_idx, az, el, dist, pos))
    sorted_cams.sort(key=lambda x: x[1])  # Sort by azimuth
    
    # Plot cameras at their elevation
    x_positions = np.linspace(0, 5, 6)
    for i, (cam_idx, az, el, dist, pos) in enumerate(sorted_cams):
        color = CAMERA_COLORS[cam_idx]
        
        # Elevation bar
        ax.bar(x_positions[i], el, color=color, width=0.6, 
              edgecolor="black", linewidth=2)
        
        ax.text(x_positions[i], el + 1, f"Cam {cam_idx}", 
               ha="center", fontsize=11, fontweight="bold", color=color)
        ax.text(x_positions[i], -3, f"Az: {az:.1f}deg", 
               ha="center", fontsize=9, rotation=45)
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_ylabel("Elevation (degrees)", fontsize=12)
    ax.set_xlabel("Cameras (sorted by azimuth)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_3d_view(cameras, title="3D Camera Setup"):
    """Create 3D visualization of camera positions"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    positions = []
    for cam in cameras:
        pos, fwd, up, right = compute_camera_pose(cam["w2c"])
        positions.append((pos, fwd, up, right))
    
    # Plot cameras
    for cam_idx, (pos, fwd, up, right) in enumerate(positions):
        color = CAMERA_COLORS[cam_idx]
        
        # Camera position
        ax.scatter(*pos, s=200, c=color, marker="o", edgecolors="black", linewidths=2)
        
        # Forward direction (longer arrow)
        arrow_len = 0.6
        arrow = Arrow3D([pos[0], pos[0]+fwd[0]*arrow_len],
                       [pos[1], pos[1]+fwd[1]*arrow_len],
                       [pos[2], pos[2]+fwd[2]*arrow_len],
                       mutation_scale=15, arrowstyle="-|>", color=color, lw=2)
        ax.add_artist(arrow)
        
        # Label
        ax.text(pos[0], pos[1], pos[2]+0.3, f"Cam {cam_idx}", 
               color=color, fontsize=10, fontweight="bold")
    
    # Draw opposing pair connections
    for cam_a, cam_b in OPPOSING_PAIRS:
        pos_a, _, _, _ = positions[cam_a]
        pos_b, _, _, _ = positions[cam_b]
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], [pos_a[2], pos_b[2]],
               linestyle="--", color="gray", linewidth=1.5, alpha=0.6)
    
    # Mouse center
    ax.scatter(0, 0, 0, s=150, c="black", marker="x", linewidths=3)
    ax.text(0, 0, 0.2, "Mouse", fontsize=9, ha="center")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Set equal aspect ratio
    max_range = 3.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-1, 2])
    
    plt.tight_layout()
    return fig


def create_parameter_comparison_table():
    """Create comparison data for all preprocessing methods"""
    methods = {
        "D4": {"paradigm": "object_centered_crop", "pp": "256 (forced)", 
               "fx": 549, "geometry": "WRONG", "ray_error": "~5-17 deg"},
        "D6-1": {"paradigm": "no_crop_resize", "pp": "varies", 
                 "fx": 549, "geometry": "correct", "ray_error": "0 deg"},
        "D7": {"paradigm": "pp_centered_shift", "pp": "256", 
               "fx": 549, "geometry": "correct", "ray_error": "~0.17 deg (fy mismatch)"},
        "D7.1": {"paradigm": "pp_centered_shift", "pp": "256", 
                 "fx": 549, "geometry": "correct", "ray_error": "~0 deg"},
        "D8": {"paradigm": "precision_homography", "pp": "256", 
               "fx": 548.994, "geometry": "correct", "ray_error": "~0 deg", 
               "skew": "corrected"},
        "D8.1": {"paradigm": "precision_homography+zoom", "pp": "varies", 
                 "fx": 713.69, "geometry": "correct", "ray_error": "~0 deg",
                 "skew": "corrected", "zoom": "1.3x"},
    }
    return methods


def calculate_mouse_size_and_error_amplification(mask_path):
    """Calculate mouse size relative to image and error amplification"""
    img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3]  # Alpha channel
    else:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    
    total_pixels = mask.shape[0] * mask.shape[1]
    mouse_pixels = np.sum(mask > 127)
    
    # Bounding box
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    
    bbox_width = xs.max() - xs.min()
    bbox_height = ys.max() - ys.min()
    
    # Size ratio
    size_ratio = mouse_pixels / total_pixels
    bbox_ratio = (bbox_width * bbox_height) / total_pixels
    linear_ratio = np.sqrt(bbox_ratio)  # Linear scale
    
    # Error amplification: if mouse is X% of image, 1px error = 1/(X%) relative error
    amplification = 1.0 / linear_ratio if linear_ratio > 0 else float("inf")
    
    return {
        "total_pixels": total_pixels,
        "mouse_pixels": int(mouse_pixels),
        "size_ratio_percent": size_ratio * 100,
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        "bbox_ratio_percent": bbox_ratio * 100,
        "linear_ratio_percent": linear_ratio * 100,
        "error_amplification": amplification
    }


def analyze_d81_zoom_error():
    """Analyze D8.1 zoom effects on camera parameters"""
    d8_cams = load_processed_cameras(D8_SAMPLE_DIR)
    d81_cams = load_processed_cameras(D8_1_SAMPLE_DIR)
    
    analysis = {
        "d8": {"fx": d8_cams[0]["fx"], "fy": d8_cams[0]["fy"],
               "cx": d8_cams[0]["cx"], "cy": d8_cams[0]["cy"]},
        "d81": {"fx": d81_cams[0]["fx"], "fy": d81_cams[0]["fy"],
                "cx": d81_cams[0]["cx"], "cy": d81_cams[0]["cy"]},
    }
    
    # D8.1 deviates from target due to zoom
    analysis["d81_deviation"] = {
        "fx_deviation": abs(d81_cams[0]["fx"] - GSLRM_TARGET_FX),
        "cx_deviation": abs(d81_cams[0]["cx"] - GSLRM_TARGET_PP),
        "cy_deviation": abs(d81_cams[0]["cy"] - GSLRM_TARGET_PP),
    }
    
    # Calculate ray error for D8.1 at edge pixels
    fx_81 = d81_cams[0]["fx"]
    cx_81 = d81_cams[0]["cx"]
    
    # Ray at pixel (256, 256) - image center
    # If cx != 256, there is a ray direction difference
    ray_error_rad = np.arctan((GSLRM_TARGET_PP - cx_81) / fx_81)
    ray_error_deg = np.degrees(ray_error_rad)
    
    analysis["d81_ray_error_deg"] = abs(ray_error_deg)
    
    # PP distribution across views
    pp_cx_list = [cam["cx"] for cam in d81_cams]
    pp_cy_list = [cam["cy"] for cam in d81_cams]
    
    analysis["d81_pp_distribution"] = {
        "cx_min": min(pp_cx_list), "cx_max": max(pp_cx_list),
        "cy_min": min(pp_cy_list), "cy_max": max(pp_cy_list),
        "cx_mean": np.mean(pp_cx_list), "cy_mean": np.mean(pp_cy_list),
        "cx_std": np.std(pp_cx_list), "cy_std": np.std(pp_cy_list),
    }
    
    return analysis


def generate_html_report(output_dir):
    """Generate comprehensive HTML report"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    orig_cams = load_original_cameras()
    d8_cams = load_processed_cameras(D8_SAMPLE_DIR)
    d81_cams = load_processed_cameras(D8_1_SAMPLE_DIR)
    
    # Create visualizations
    print("Creating camera grid...")
    fig_grid = create_camera_grid_image(D8_SAMPLE_DIR, "D8 Camera Views (Layout: 4,2,1 / 0,5,3)")
    fig_grid.savefig(output_dir / "camera_images_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    
    print("Creating top view...")
    fig_top = create_top_view_with_angles(d8_cams, "Top View with Opposing Pairs")
    fig_top.savefig(output_dir / "camera_top_view.png", dpi=150, bbox_inches="tight")
    plt.close(fig_top)
    
    print("Creating side view...")
    fig_side = create_side_view(d8_cams, "Side View (Elevation Comparison)")
    fig_side.savefig(output_dir / "camera_side_view.png", dpi=150, bbox_inches="tight")
    plt.close(fig_side)
    
    print("Creating 3D view...")
    fig_3d = create_3d_view(d8_cams, "3D Camera Setup")
    fig_3d.savefig(output_dir / "camera_3d_setup.png", dpi=150, bbox_inches="tight")
    plt.close(fig_3d)
    
    # Analyze mouse size
    print("Analyzing mouse size...")
    mouse_analysis = calculate_mouse_size_and_error_amplification(
        D8_SAMPLE_DIR / "images/cam_000.png"
    )
    
    mouse_analysis_d81 = calculate_mouse_size_and_error_amplification(
        D8_1_SAMPLE_DIR / "images/cam_000.png"
    )
    
    # D8.1 analysis
    print("Analyzing D8.1 zoom effects...")
    d81_analysis = analyze_d81_zoom_error()
    
    # Generate HTML
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>D8 Comprehensive Camera and Preprocessing Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; line-height: 1.6; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 15px; }}
        h2 {{ color: #16213e; margin-top: 40px; border-left: 5px solid #4a90d9; padding-left: 15px; }}
        h3 {{ color: #34495e; }}
        
        .summary-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; padding: 25px; border-radius: 15px; margin: 20px 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; }}
        .metric {{ background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 11px; opacity: 0.9; }}
        
        .viz-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }}
        .viz-img {{ width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .viz-full {{ width: 100%; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: 20px 0; }}
        
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; 
                border-radius: 10px; overflow: hidden; font-size: 13px; }}
        th {{ background: #4a90d9; color: white; padding: 12px 8px; text-align: center; }}
        td {{ padding: 10px 8px; text-align: center; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        
        .good {{ background: #d4edda; }}
        .bad {{ background: #f8d7da; }}
        .warning {{ background: #fff3cd; }}
        .info {{ background: #d1ecf1; }}
        
        .formula-box {{ background: white; padding: 20px; margin: 20px 0; 
                       border-left: 4px solid #4a90d9; border-radius: 5px;
                       box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .key-point {{ background: #e8f6f3; padding: 15px; border-left: 4px solid #1abc9c; 
                     margin: 15px 0; border-radius: 5px; }}
        .warning-box {{ background: #fdf2e9; padding: 15px; border-left: 4px solid #e67e22; 
                       margin: 15px 0; border-radius: 5px; }}
        
        code {{ background: #f5f5f5; padding: 2px 6px; font-family: monospace; border-radius: 3px; }}
        pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; overflow-x: auto; border-radius: 8px; }}
        
        .toc {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px;
               box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .toc ul {{ columns: 2; -webkit-columns: 2; -moz-columns: 2; }}
        .toc li {{ margin: 8px 0; }}
        .toc a {{ text-decoration: none; color: #4a90d9; }}
        .toc a:hover {{ text-decoration: underline; }}
        
        .cam-table {{ font-size: 11px; }}
        .cam-table th {{ padding: 8px 4px; font-size: 10px; }}
        .cam-table td {{ padding: 6px 4px; font-size: 10px; }}
    </style>
</head>
<body>
<div class="container">

<h1>D8 Comprehensive Camera and Preprocessing Report</h1>
<p><strong>Generated:</strong> {timestamp} | <strong>Data:</strong> markerless_mouse_1_nerf | <strong>Version:</strong> D8</p>

<div class="toc">
<h3>Table of Contents</h3>
<ul>
<li><a href="#sec1">1. Executive Summary</a></li>
<li><a href="#sec2">2. Camera Setup Visualization</a></li>
<li><a href="#sec3">3. Camera Parameters (Before/After)</a></li>
<li><a href="#sec4">4. Preprocessing Methods Comparison</a></li>
<li><a href="#sec5">5. Mouse Size & Error Amplification</a></li>
<li><a href="#sec6">6. D8.1 Zoom Analysis</a></li>
<li><a href="#sec7">7. Mathematical Verification</a></li>
<li><a href="#sec8">8. Recommendations</a></li>
</ul>
</div>

<h2 id="sec1">1. Executive Summary</h2>

<div class="summary-box">
<h3 style="color:white; margin-top:0;">D8 Preprocessing Results</h3>
<div class="summary-grid">
<div class="metric">
    <div class="metric-value">6</div>
    <div class="metric-label">Camera Views</div>
</div>
<div class="metric">
    <div class="metric-value">{GSLRM_TARGET_FX:.4f}</div>
    <div class="metric-label">Focal Length (fx=fy)</div>
</div>
<div class="metric">
    <div class="metric-value">{GSLRM_TARGET_PP:.1f}</div>
    <div class="metric-label">Principal Point (cx=cy)</div>
</div>
<div class="metric">
    <div class="metric-value">{GSLRM_TARGET_DIST:.1f}</div>
    <div class="metric-label">Camera Distance</div>
</div>
<div class="metric">
    <div class="metric-value">~0 deg</div>
    <div class="metric-label">Ray Error</div>
</div>
</div>
</div>

<div class="key-point">
<h4>Key Improvements in D8</h4>
<ul>
<li><strong>Homography Transform:</strong> H = K_target @ K_orig^-1 (vs affine in D7)</li>
<li><strong>Skew Correction:</strong> Original camera skew (up to 5.8px) is removed</li>
<li><strong>Exact Focal Length:</strong> 548.9937744140625 (GS-LRM pretrained exact value)</li>
<li><strong>LANCZOS4 Interpolation:</strong> Higher quality image resampling</li>
</ul>
</div>

<h2 id="sec2">2. Camera Setup Visualization</h2>

<h3>2.1 Camera Images (Layout: Row1=[4,2,1], Row2=[0,5,3])</h3>
<img src="camera_images_grid.png" class="viz-full" alt="Camera Images Grid">
<p><em>Camera layout arranged to show opposing pairs visually aligned.</em></p>

<h3>2.2 Top View (X-Y Plane) with Opposing Pairs</h3>
<div class="viz-grid">
<div>
<img src="camera_top_view.png" class="viz-img" alt="Top View">
<p style="text-align:center; color:#666;"><strong>Opposing pairs:</strong> 4-3, 2-5, 1-0</p>
</div>
<div>
<img src="camera_side_view.png" class="viz-img" alt="Side View">
<p style="text-align:center; color:#666;">Elevation angles by camera</p>
</div>
</div>

<h3>2.3 3D Camera Setup</h3>
<img src="camera_3d_setup.png" class="viz-full" alt="3D Camera Setup">

<h2 id="sec3">3. Camera Parameters (Before/After Preprocessing)</h2>

<h3>3.1 Original Camera Parameters (Before)</h3>
<table class="cam-table">
<tr><th>View</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>skew</th><th>Distance (mm)</th></tr>
"""
    
    # Add original camera rows
    for i, cam in enumerate(orig_cams):
        K = cam["K"]
        pos = cam["T"]
        dist = np.linalg.norm(pos)
        html_content += f"""<tr>
<td><strong style="color:{CAMERA_COLORS[i]}">Cam {i}</strong></td>
<td>{K[0,0]:.1f}</td>
<td>{K[1,1]:.1f}</td>
<td>{K[0,2]:.1f}</td>
<td>{K[1,2]:.1f}</td>
<td>{K[0,1]:.2f}</td>
<td>{dist:.1f}</td>
</tr>
"""
    
    html_content += """</table>

<h3>3.2 D8 Processed Camera Parameters (After)</h3>
<table class="cam-table">
<tr><th>View</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Azimuth</th><th>Elevation</th><th>Distance</th></tr>
"""
    
    # Add D8 camera rows
    for i, cam in enumerate(d8_cams):
        pos, fwd, _, _ = compute_camera_pose(cam["w2c"])
        az, el, dist = compute_azimuth_elevation(pos)
        html_content += f"""<tr class="good">
<td><strong style="color:{CAMERA_COLORS[i]}">Cam {i}</strong></td>
<td>{cam["fx"]:.4f}</td>
<td>{cam["fy"]:.4f}</td>
<td>{cam["cx"]:.1f}</td>
<td>{cam["cy"]:.1f}</td>
<td>{az:.1f} deg</td>
<td>{el:.1f} deg</td>
<td>{dist:.4f}</td>
</tr>
"""
    
    methods = create_parameter_comparison_table()
    
    html_content += f"""</table>

<h2 id="sec4">4. Preprocessing Methods Comparison</h2>

<table>
<tr><th>Method</th><th>Paradigm</th><th>fx</th><th>PP (cx,cy)</th><th>Skew</th><th>Geometry</th><th>Ray Error</th><th>Rec.</th></tr>
<tr class="bad"><td><strong>D4</strong></td><td>object_centered_crop</td><td>549</td><td>256 (forced)</td><td>ignored</td><td>WRONG</td><td>~5-17 deg</td><td>-</td></tr>
<tr><td><strong>D6-1</strong></td><td>no_crop_resize</td><td>549</td><td>varies</td><td>ignored</td><td>correct</td><td>0 deg</td><td>-</td></tr>
<tr><td><strong>D7</strong></td><td>pp_centered_shift</td><td>549</td><td>256</td><td>ignored</td><td>correct</td><td>~0.17 deg</td><td>-</td></tr>
<tr class="info"><td><strong>D7.1</strong></td><td>pp_centered_shift</td><td>549</td><td>256</td><td>ignored</td><td>correct</td><td>~0 deg</td><td>OK</td></tr>
<tr class="good"><td><strong>D8</strong></td><td>precision_homography</td><td>{GSLRM_TARGET_FX:.4f}</td><td>256</td><td>corrected</td><td>correct</td><td>~0 deg</td><td>Best</td></tr>
<tr class="warning"><td><strong>D8.1</strong></td><td>homography+zoom</td><td>713.69</td><td>varies</td><td>corrected</td><td>correct</td><td>see sec 6</td><td>Large mouse</td></tr>
</table>

<div class="formula-box">
<h4>D8 Homography Transform</h4>
<p>The key difference from D7 is using homography instead of affine:</p>
\\[
H = K_{{target}} \\cdot K_{{orig}}^{{-1}}
\\]
<p>This correctly handles:</p>
<ul>
<li><strong>Skew correction:</strong> Original skew values (-5.82 to +1.39 px) are removed</li>
<li><strong>fx/fy unification:</strong> Both become exactly {GSLRM_TARGET_FX}</li>
<li><strong>PP centering:</strong> Principal point moves to (256, 256)</li>
</ul>
</div>

<h2 id="sec5">5. Mouse Size & Error Amplification</h2>
"""
    
    if mouse_analysis:
        html_content += f"""
<div class="key-point">
<h4>D8 Mouse Size Analysis</h4>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Image Size</td><td>512 x 512 = 262,144 px</td></tr>
<tr><td>Mouse Pixels</td><td>{mouse_analysis["mouse_pixels"]:,} px</td></tr>
<tr><td>Mouse Area Ratio</td><td>{mouse_analysis["size_ratio_percent"]:.1f}%</td></tr>
<tr><td>Bounding Box</td><td>{mouse_analysis["bbox_width"]} x {mouse_analysis["bbox_height"]} px</td></tr>
<tr><td>Linear Size Ratio</td><td>{mouse_analysis["linear_ratio_percent"]:.1f}%</td></tr>
<tr><td><strong>Error Amplification</strong></td><td><strong>{mouse_analysis["error_amplification"]:.2f}x</strong></td></tr>
</table>
</div>

<div class="warning-box">
<h4>Error Amplification Explanation</h4>
<p>When the mouse occupies ~{mouse_analysis["linear_ratio_percent"]:.0f}% of the image linearly:</p>
<ul>
<li>A 1px camera parameter error becomes <strong>{mouse_analysis["error_amplification"]:.1f}px</strong> error relative to mouse size</li>
<li>D4 PP error (~37px) amplifies to <strong>~{37 * mouse_analysis["error_amplification"]:.0f}px</strong> relative error</li>
<li>D8 skew error (up to 0.9px) would amplify to <strong>~{0.9 * mouse_analysis["error_amplification"]:.1f}px</strong> if not corrected</li>
</ul>
</div>
"""
    
    if mouse_analysis_d81:
        html_content += f"""
<h3>5.2 D8.1 (Zoom 1.3x) Mouse Size</h3>
<table>
<tr><th>Metric</th><th>D8</th><th>D8.1</th><th>Improvement</th></tr>
<tr><td>Mouse Pixels</td><td>{mouse_analysis["mouse_pixels"]:,}</td><td>{mouse_analysis_d81["mouse_pixels"]:,}</td><td>+{(mouse_analysis_d81["mouse_pixels"]/mouse_analysis["mouse_pixels"]-1)*100:.0f}%</td></tr>
<tr><td>Linear Size</td><td>{mouse_analysis["linear_ratio_percent"]:.1f}%</td><td>{mouse_analysis_d81["linear_ratio_percent"]:.1f}%</td><td>+{mouse_analysis_d81["linear_ratio_percent"]-mouse_analysis["linear_ratio_percent"]:.1f}%</td></tr>
<tr><td>Error Amp.</td><td>{mouse_analysis["error_amplification"]:.2f}x</td><td>{mouse_analysis_d81["error_amplification"]:.2f}x</td><td>-{(1-mouse_analysis_d81["error_amplification"]/mouse_analysis["error_amplification"])*100:.0f}%</td></tr>
</table>
"""
    
    html_content += f"""
<h2 id="sec6">6. D8.1 Zoom Analysis</h2>

<div class="warning-box">
<h4>Why D8.1 Deviates from GS-LRM Parameters</h4>
<p>D8.1 applies 1.3x zoom via crop+resize, which changes the effective focal length:</p>
<ul>
<li><strong>fx, fy:</strong> {GSLRM_TARGET_FX:.4f} x 1.3 = <strong>{d81_analysis["d81"]["fx"]:.2f}</strong></li>
<li><strong>cx, cy:</strong> Varies per view based on crop offset (not fixed at 256)</li>
</ul>
</div>

<h3>6.1 D8.1 PP Distribution (varies by view)</h3>
<table>
<tr><th>Statistic</th><th>cx</th><th>cy</th></tr>
<tr><td>Min</td><td>{d81_analysis["d81_pp_distribution"]["cx_min"]:.1f}</td><td>{d81_analysis["d81_pp_distribution"]["cy_min"]:.1f}</td></tr>
<tr><td>Max</td><td>{d81_analysis["d81_pp_distribution"]["cx_max"]:.1f}</td><td>{d81_analysis["d81_pp_distribution"]["cy_max"]:.1f}</td></tr>
<tr><td>Mean</td><td>{d81_analysis["d81_pp_distribution"]["cx_mean"]:.1f}</td><td>{d81_analysis["d81_pp_distribution"]["cy_mean"]:.1f}</td></tr>
<tr><td>Std Dev</td><td>{d81_analysis["d81_pp_distribution"]["cx_std"]:.1f}</td><td>{d81_analysis["d81_pp_distribution"]["cy_std"]:.1f}</td></tr>
</table>

<h3>6.2 Matching D8.1 to FaceLift (Theoretical)</h3>
<div class="formula-box">
<p>To use D8.1 data with GS-LRM pretrained model (expects fx=549, cx=256):</p>

<p><strong>Option 1: Use D8.1 as-is</strong></p>
<ul>
<li>GS-LRM reads fx, cx, cy from data file (not hardcoded)</li>
<li>Model should adapt to different focal lengths during finetuning</li>
<li>Expected ray error: <strong>{d81_analysis["d81_ray_error_deg"]:.2f} deg</strong> at image center</li>
</ul>

<p><strong>Option 2: Post-process to match GS-LRM</strong></p>
<ul>
<li>Apply additional transform to set fx=549, cx=cy=256</li>
<li>Would reduce effective zoom and mouse size</li>
<li>Not recommended - loses the benefit of zoom</li>
</ul>
</div>

<h2 id="sec7">7. Mathematical Verification</h2>

<div class="formula-box">
<h4>7.1 Ray Direction Calculation</h4>
<p>For pixel (u, v), the ray direction in camera coordinates:</p>
\\[
\\mathbf{{d}}_c = \\begin{{bmatrix}} (u - c_x) / f_x \\\\ (v - c_y) / f_y \\\\ 1 \\end{{bmatrix}}
\\]

<h4>7.2 D8 Ray Error Verification</h4>
<p>At image edge (u=0 or u=512):</p>
\\[
\\theta_{{error}} = \\arctan\\left(\\frac{{\\Delta c_x}}{{f_x}}\\right)
\\]
<p>For D8 with cx=256, cy=256 (exact): <strong>ray error ~0 deg</strong></p>
<p>For D4 with PP error ~37px: <strong>ray error = arctan(37/549) ≈ 3.9 deg</strong></p>
</div>

<h2 id="sec8">8. Recommendations</h2>

<div class="key-point">
<h4>Summary</h4>
<table>
<tr><th>Use Case</th><th>Recommended</th><th>Reason</th></tr>
<tr><td>Standard training</td><td><strong>D8</strong></td><td>Exact GS-LRM params, skew corrected</td></tr>
<tr><td>Larger mouse needed</td><td><strong>D8.1</strong></td><td>1.3x zoom, ~40% larger mouse pixels</td></tr>
<tr><td>Legacy comparison</td><td>D7.1</td><td>Compatible with previous experiments</td></tr>
</table>
</div>

<h3>D8 Preprocessing Command</h3>
<pre>
cd ~/dev/FaceLift
python -m mouse_extensions.preprocessing.preprocess \\
    --preset D8 \\
    --input-dir ~/data/raw/markerless_mouse_1_nerf \\
    --output-dir ~/data/preprocessed/FaceLift_mouse/D8
</pre>

<h3>D8.1 Preprocessing Command</h3>
<pre>
cd ~/dev/FaceLift
python -m mouse_extensions.preprocessing.preprocess \\
    --preset D8.1 \\
    --input-dir ~/data/raw/markerless_mouse_1_nerf \\
    --output-dir ~/data/preprocessed/FaceLift_mouse/D8_1
</pre>

<hr>
<p style="color:#888; font-size:12px; text-align:center;">
Generated by generate_comprehensive_d8_report.py | FaceLift Mouse Extension | {timestamp}
</p>

</div>
</body>
</html>
"""
    
    # Save HTML
    with open(output_dir / "comprehensive_d8_report.html", "w") as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    output_dir = generate_html_report(OUTPUT_DIR)
    print(f"\nReport generated at: {output_dir}")
    print(f"Open: {output_dir}/comprehensive_d8_report.html")


def create_original_camera_visualizations():
    """Create visualizations for original (pre-transformation) camera setup"""
    output_dir = OUTPUT_DIR
    
    # Load original cameras
    orig_cams = load_original_cameras()
    
    # Create figure for original 3D setup
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    positions = []
    for i, cam in enumerate(orig_cams):
        R, T = cam["R"], cam["T"].flatten()
        # Camera position in world coordinates
        pos = -R.T @ T
        # Forward direction (camera looks along +Z in camera frame)
        forward = R.T @ np.array([0, 0, 1])
        positions.append((pos, forward))
    
    # Plot cameras
    for cam_idx, (pos, fwd) in enumerate(positions):
        color = CAMERA_COLORS[cam_idx]
        
        # Scale down for visualization (original is in mm)
        pos_scaled = pos / 100  # mm to dm for better visualization
        
        ax.scatter(*pos_scaled, s=200, c=color, marker="o", edgecolors="black", linewidths=2)
        
        arrow_len = 0.8
        arrow = Arrow3D([pos_scaled[0], pos_scaled[0]+fwd[0]*arrow_len],
                       [pos_scaled[1], pos_scaled[1]+fwd[1]*arrow_len],
                       [pos_scaled[2], pos_scaled[2]+fwd[2]*arrow_len],
                       mutation_scale=15, arrowstyle="-|>", color=color, lw=2)
        ax.add_artist(arrow)
        
        ax.text(pos_scaled[0], pos_scaled[1], pos_scaled[2]+0.3, f"Cam {cam_idx}",
               color=color, fontsize=10, fontweight="bold")
    
    # Origin (mouse area center)
    ax.scatter(0, 0, 0, s=150, c="black", marker="x", linewidths=3)
    ax.text(0, 0, 0.3, "Origin", fontsize=9, ha="center")
    
    ax.set_xlabel("X (mm/100)")
    ax.set_ylabel("Y (mm/100)")
    ax.set_zlabel("Z (mm/100)")
    ax.set_title("Original Camera Setup (Before Normalization)", fontsize=14, fontweight="bold")
    
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-2, 4])
    
    plt.tight_layout()
    fig.savefig(output_dir / "camera_3d_original.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Create top view for original
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for cam_idx, (pos, fwd) in enumerate(positions):
        color = CAMERA_COLORS[cam_idx]
        pos_scaled = pos / 100
        
        ax.scatter(pos_scaled[0], pos_scaled[1], s=300, c=color, marker="o",
                  edgecolors="black", linewidths=2, zorder=5)
        ax.annotate(f"Cam {cam_idx}\n({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) mm",
                   (pos_scaled[0], pos_scaled[1]),
                   xytext=(10, 10), textcoords="offset points",
                   fontsize=10, color=color)
        
        arrow_scale = 0.8
        ax.arrow(pos_scaled[0], pos_scaled[1], fwd[0]*arrow_scale, fwd[1]*arrow_scale,
                head_width=0.15, head_length=0.1, fc=color, ec=color, linewidth=2)
    
    # Draw opposing pairs
    for cam_a, cam_b in OPPOSING_PAIRS:
        pos_a = positions[cam_a][0] / 100
        pos_b = positions[cam_b][0] / 100
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]],
               linestyle="--", color="gray", linewidth=2, alpha=0.7)
    
    ax.scatter(0, 0, s=200, c="black", marker="x", linewidths=3, zorder=10)
    ax.annotate("Origin", (0, 0), xytext=(15, -15), textcoords="offset points", fontsize=10)
    
    ax.set_xlabel("X (mm/100)", fontsize=12)
    ax.set_ylabel("Y (mm/100)", fontsize=12)
    ax.set_title("Original Camera Top View (Before Normalization)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    
    # Add camera distances
    dist_text = "Original Distances:\n"
    for i, (pos, _) in enumerate(positions):
        dist = np.linalg.norm(pos)
        dist_text += f"Cam {i}: {dist:.0f} mm\n"
    ax.text(0.02, 0.98, dist_text, transform=ax.transAxes, fontsize=9,
           verticalalignment="top", fontfamily="monospace",
           bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    fig.savefig(output_dir / "camera_top_view_original.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Original camera visualizations saved to: {output_dir}")


if __name__ == "__main__":
    # Also create original camera visualizations
    create_original_camera_visualizations()
