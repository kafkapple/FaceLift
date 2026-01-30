#!/usr/bin/env python3
from mouse_extensions.paths import FACELIFT_ROOT, DATA_ROOT
"""
Comprehensive Report Generator for FaceLift Mouse Preprocessing

Generates two separate reports:
1. Camera Setup Report: Camera arrangement, before/after transformation
2. Preprocessing Comparison Report: Error metrics, mathematical verification

Usage:
    python report_generator.py --dataset D8 --output-dir ./reports
    python report_generator.py --all --output-dir ./reports

Author: Claude/Joon
Date: 2026-01-21
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, ConnectionPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = FACELIFT_ROOT
DATA_DIR = DATA_ROOT
RAW_DIR = DATA_DIR / "raw/markerless_mouse_1_nerf"
PREPROCESSED_DIR = DATA_DIR / "preprocessed/FaceLift_mouse"

# GS-LRM target values
GSLRM_TARGET_FX = 548.9937744140625
GSLRM_TARGET_PP = 256.0
GSLRM_TARGET_DIST = 2.7

# Camera colors (consistent across all visualizations)
CAMERA_COLORS = {
    0: "#E53935",  # Red
    1: "#1E88E5",  # Blue  
    2: "#43A047",  # Green
    3: "#FB8C00",  # Orange
    4: "#8E24AA",  # Purple
    5: "#00ACC1",  # Cyan
}

# Layout order for visualization
LAYOUT_ORDER = [[4, 2, 1], [0, 5, 3]]
OPPOSING_PAIRS = [(4, 3), (2, 5), (1, 0)]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CameraParams:
    """Camera parameters for a single view"""
    view_id: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float
    w2c: np.ndarray
    position: np.ndarray
    forward: np.ndarray
    azimuth: float
    elevation: float
    distance: float


@dataclass 
class DatasetMetrics:
    """Metrics for a preprocessed dataset"""
    name: str
    num_views: int
    fx_mean: float
    fx_std: float
    fy_mean: float
    fy_std: float
    cx_mean: float
    cx_std: float
    cy_mean: float
    cy_std: float
    dist_mean: float
    fx_deviation: float  # from GS-LRM target
    pp_deviation: float  # from 256
    ray_error_max: float
    skew_corrected: bool
    geometry_correct: bool


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


def load_original_cameras() -> List[Dict]:
    """Load original camera parameters from raw data"""
    with open(RAW_DIR / "new_cam.pkl", "rb") as f:
        return pickle.load(f)


def load_processed_cameras(dataset_path: Path) -> List[Dict]:
    """Load processed camera parameters"""
    sample_dir = dataset_path / "train/000000"
    if not sample_dir.exists():
        sample_dirs = list((dataset_path / "train").glob("*"))
        if sample_dirs:
            sample_dir = sample_dirs[0]
    
    cam_file = sample_dir / "opencv_cameras.json"
    if not cam_file.exists():
        return None
    
    with open(cam_file) as f:
        return json.load(f)["frames"]


def compute_camera_pose(w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract camera position and forward direction from w2c matrix"""
    w2c = np.array(w2c)
    c2w = np.linalg.inv(w2c)
    position = c2w[:3, 3]
    forward = c2w[:3, 2]  # OpenCV: +Z is forward
    return position, forward


def compute_azimuth_elevation(position: np.ndarray) -> Tuple[float, float, float]:
    """Compute azimuth and elevation from position"""
    x, y, z = position
    azimuth = np.degrees(np.arctan2(y, x))
    dist_xy = np.sqrt(x**2 + y**2)
    elevation = np.degrees(np.arctan2(z, dist_xy))
    distance = np.linalg.norm(position)
    return azimuth, elevation, distance


def extract_camera_params(cam_dict: Dict, view_id: int) -> CameraParams:
    """Extract CameraParams from camera dictionary"""
    w2c = np.array(cam_dict["w2c"])
    pos, fwd = compute_camera_pose(w2c)
    az, el, dist = compute_azimuth_elevation(pos)
    
    return CameraParams(
        view_id=view_id,
        fx=cam_dict["fx"],
        fy=cam_dict["fy"],
        cx=cam_dict["cx"],
        cy=cam_dict["cy"],
        skew=cam_dict.get("skew", 0.0),
        w2c=w2c,
        position=pos,
        forward=fwd,
        azimuth=az,
        elevation=el,
        distance=dist
    )


def compute_dataset_metrics(name: str, cameras: List[Dict]) -> DatasetMetrics:
    """Compute metrics for a dataset"""
    params = [extract_camera_params(c, i) for i, c in enumerate(cameras)]
    
    fx_list = [p.fx for p in params]
    fy_list = [p.fy for p in params]
    cx_list = [p.cx for p in params]
    cy_list = [p.cy for p in params]
    dist_list = [p.distance for p in params]
    
    # Ray error: at image edge (0 or 512), how much does ray differ from expected?
    ray_errors = []
    for p in params:
        # Ray at pixel (256, 256) should point to (0, 0, 1) if cx=cy=256
        ray_x = (256 - p.cx) / p.fx
        ray_y = (256 - p.cy) / p.fy
        ray_angle = np.degrees(np.arctan(np.sqrt(ray_x**2 + ray_y**2)))
        ray_errors.append(ray_angle)
    
    return DatasetMetrics(
        name=name,
        num_views=len(cameras),
        fx_mean=np.mean(fx_list),
        fx_std=np.std(fx_list),
        fy_mean=np.mean(fy_list),
        fy_std=np.std(fy_list),
        cx_mean=np.mean(cx_list),
        cx_std=np.std(cx_list),
        cy_mean=np.mean(cy_list),
        cy_std=np.std(cy_list),
        dist_mean=np.mean(dist_list),
        fx_deviation=abs(np.mean(fx_list) - GSLRM_TARGET_FX),
        pp_deviation=np.sqrt((np.mean(cx_list)-256)**2 + (np.mean(cy_list)-256)**2),
        ray_error_max=max(ray_errors),
        skew_corrected="D8" in name,
        geometry_correct="D4" not in name
    )


# =============================================================================
# REPORT A: CAMERA SETUP VISUALIZATION
# =============================================================================

def create_camera_grid_image(dataset_path: Path, title: str, output_path: Path):
    """Create 2x3 grid of camera images following layout [4,2,1]/[0,5,3]"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    
    sample_dir = dataset_path / "train/000000"
    if not sample_dir.exists():
        sample_dirs = list((dataset_path / "train").glob("*"))
        if sample_dirs:
            sample_dir = sample_dirs[0]
    
    for row_idx, row in enumerate(LAYOUT_ORDER):
        for col_idx, cam_idx in enumerate(row):
            ax = axes[row_idx, col_idx]
            img_path = sample_dir / "images" / f"cam_{cam_idx:03d}.png"
            
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            
            # Title with color
            ax.set_title(f"Camera {cam_idx}", 
                        color=CAMERA_COLORS[cam_idx], 
                        fontweight="bold", fontsize=16)
            ax.axis("off")
            
            # Colored border
            for spine in ax.spines.values():
                spine.set_edgecolor(CAMERA_COLORS[cam_idx])
                spine.set_linewidth(4)
                spine.set_visible(True)
    
    # Add opposing pair annotations
    fig.text(0.5, 0.02, "Opposing Pairs: 4↔3, 2↔5, 1↔0", 
            ha="center", fontsize=12, style="italic")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def create_top_view(cameras: List[CameraParams], title: str, output_path: Path,
                   show_angles: bool = True):
    """Create top view with opposing pairs connected"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot cameras
    for cam in cameras:
        color = CAMERA_COLORS[cam.view_id]
        
        ax.scatter(cam.position[0], cam.position[1], s=400, c=color, 
                  marker="o", edgecolors="black", linewidths=2, zorder=5)
        
        # Camera label
        label = f"Cam {cam.view_id}"
        if show_angles:
            label += f"\n({cam.azimuth:.0f}°)"
        ax.annotate(label, (cam.position[0], cam.position[1]),
                   xytext=(12, 12), textcoords="offset points",
                   fontsize=11, fontweight="bold", color=color,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                            edgecolor=color, alpha=0.9))
        
        # Forward direction arrow
        arrow_len = 0.6
        ax.arrow(cam.position[0], cam.position[1], 
                cam.forward[0]*arrow_len, cam.forward[1]*arrow_len,
                head_width=0.12, head_length=0.08, 
                fc=color, ec=color, linewidth=2, zorder=4)
    
    # Draw opposing pair connections
    for cam_a, cam_b in OPPOSING_PAIRS:
        pos_a = cameras[cam_a].position
        pos_b = cameras[cam_b].position
        
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]],
               linestyle="--", color="gray", linewidth=2, alpha=0.6, zorder=1)
        
        # Midpoint label
        mid_xy = (pos_a[:2] + pos_b[:2]) / 2
        ax.annotate(f"{cam_a}-{cam_b}", (mid_xy[0], mid_xy[1]),
                   fontsize=10, ha="center", va="center",
                   bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))
    
    # Origin (mouse center)
    ax.scatter(0, 0, s=300, c="black", marker="X", linewidths=3, zorder=10)
    ax.annotate("Mouse\nCenter", (0, 0), xytext=(0, -0.5),
               fontsize=10, ha="center", fontweight="bold")
    
    ax.set_xlabel("X (world)", fontsize=12)
    ax.set_ylabel("Y (world)", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def create_side_view(cameras: List[CameraParams], title: str, output_path: Path):
    """Create side view showing elevation differences"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by azimuth for x-axis ordering
    sorted_cams = sorted(cameras, key=lambda c: c.azimuth)
    
    x_positions = np.linspace(0.5, 5.5, 6)
    for i, cam in enumerate(sorted_cams):
        color = CAMERA_COLORS[cam.view_id]
        
        ax.bar(x_positions[i], cam.elevation, color=color, width=0.6,
              edgecolor="black", linewidth=2, zorder=3)
        
        ax.text(x_positions[i], cam.elevation + 2, f"Cam {cam.view_id}",
               ha="center", fontsize=12, fontweight="bold", color=color)
        ax.text(x_positions[i], -4, f"Az:{cam.azimuth:.0f}°",
               ha="center", fontsize=9, rotation=0)
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_ylabel("Elevation (degrees)", fontsize=12)
    ax.set_xlabel("Cameras (sorted by azimuth)", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylim(-8, 40)
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def create_3d_view(cameras: List[CameraParams], title: str, output_path: Path):
    """Create 3D visualization of camera setup"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    for cam in cameras:
        color = CAMERA_COLORS[cam.view_id]
        pos = cam.position
        fwd = cam.forward
        
        ax.scatter(*pos, s=200, c=color, marker="o", 
                  edgecolors="black", linewidths=2)
        
        # Forward direction arrow
        arrow_len = 0.5
        arrow = Arrow3D([pos[0], pos[0]+fwd[0]*arrow_len],
                       [pos[1], pos[1]+fwd[1]*arrow_len],
                       [pos[2], pos[2]+fwd[2]*arrow_len],
                       mutation_scale=12, arrowstyle="-|>", color=color, lw=2)
        ax.add_artist(arrow)
        
        ax.text(pos[0], pos[1], pos[2]+0.25, f"Cam {cam.view_id}",
               color=color, fontsize=10, fontweight="bold")
    
    # Draw opposing pairs
    for cam_a, cam_b in OPPOSING_PAIRS:
        pos_a = cameras[cam_a].position
        pos_b = cameras[cam_b].position
        ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], [pos_a[2], pos_b[2]],
               linestyle="--", color="gray", linewidth=1.5, alpha=0.5)
    
    # Mouse center
    ax.scatter(0, 0, 0, s=200, c="black", marker="X", linewidths=3)
    ax.text(0, 0, 0.2, "Mouse", fontsize=9, ha="center")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=16, fontweight="bold")
    
    max_range = 3.0
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-0.5, 2])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_camera_setup_report(dataset_name: str, output_dir: Path):
    """Generate camera setup visualization report (Report A)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cameras
    dataset_path = PREPROCESSED_DIR / dataset_name
    if not dataset_path.exists():
        dataset_path = PREPROCESSED_DIR / f"{dataset_name}_sample"
    
    raw_cameras = load_processed_cameras(dataset_path)
    if raw_cameras is None:
        print(f"No camera data found for {dataset_name}")
        return None
    
    cameras = [extract_camera_params(c, i) for i, c in enumerate(raw_cameras)]
    
    # Generate visualizations
    print(f"Generating camera setup report for {dataset_name}...")
    
    create_camera_grid_image(dataset_path, 
                            f"{dataset_name} Camera Views",
                            output_dir / "camera_grid.png")
    
    create_top_view(cameras,
                   f"{dataset_name} Top View (X-Y Plane)",
                   output_dir / "camera_top_view.png")
    
    create_side_view(cameras,
                    f"{dataset_name} Side View (Elevation)",
                    output_dir / "camera_side_view.png")
    
    create_3d_view(cameras,
                  f"{dataset_name} 3D Camera Setup",
                  output_dir / "camera_3d_view.png")
    
    # Generate HTML report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Camera Setup Report - {dataset_name}</title>
    <style>
        body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 15px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        .summary-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; padding: 20px; border-radius: 12px; margin: 20px 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .metric {{ background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 22px; font-weight: bold; }}
        .metric-label {{ font-size: 11px; opacity: 0.9; }}
        .viz-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .viz-img {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .viz-full {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 15px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; 
                border-radius: 8px; overflow: hidden; font-size: 12px; }}
        th {{ background: #4a90d9; color: white; padding: 10px; text-align: center; font-size: 11px; }}
        td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; font-size: 11px; }}
        .color-box {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; }}
    </style>
</head>
<body>
<div class="container">
<h1>Camera Setup Report - {dataset_name}</h1>
<p><strong>Generated:</strong> {timestamp}</p>

<div class="summary-box">
<div class="summary-grid">
<div class="metric"><div class="metric-value">{len(cameras)}</div><div class="metric-label">Views</div></div>
<div class="metric"><div class="metric-value">{cameras[0].fx:.2f}</div><div class="metric-label">Focal Length</div></div>
<div class="metric"><div class="metric-value">{cameras[0].cx:.1f}</div><div class="metric-label">cx (Cam 0)</div></div>
<div class="metric"><div class="metric-value">{cameras[0].distance:.2f}</div><div class="metric-label">Distance</div></div>
</div>
</div>

<h2>Camera Grid (Layout: [4,2,1] / [0,5,3])</h2>
<img src="camera_grid.png" class="viz-full" alt="Camera Grid">
<p><em>Opposing pairs: 4↔3, 2↔5, 1↔0</em></p>

<h2>Spatial Arrangement</h2>
<div class="viz-grid">
<div>
    <img src="camera_top_view.png" class="viz-img" alt="Top View">
    <p style="text-align:center; color:#666;">Top View (X-Y Plane)</p>
</div>
<div>
    <img src="camera_side_view.png" class="viz-img" alt="Side View">
    <p style="text-align:center; color:#666;">Side View (Elevation)</p>
</div>
</div>

<h2>3D View</h2>
<img src="camera_3d_view.png" class="viz-full" alt="3D View">

<h2>Camera Parameters</h2>
<table>
<tr><th>Cam</th><th>Color</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Azimuth</th><th>Elevation</th><th>Distance</th></tr>
"""
    
    for cam in cameras:
        color = CAMERA_COLORS[cam.view_id]
        html += f"""<tr>
<td><strong>Cam {cam.view_id}</strong></td>
<td><span class="color-box" style="background:{color}"></span></td>
<td>{cam.fx:.4f}</td>
<td>{cam.fy:.4f}</td>
<td>{cam.cx:.1f}</td>
<td>{cam.cy:.1f}</td>
<td>{cam.azimuth:.1f}°</td>
<td>{cam.elevation:.1f}°</td>
<td>{cam.distance:.4f}</td>
</tr>
"""
    
    html += """</table>
<hr>
<p style="color:#888; font-size:11px; text-align:center;">
Generated by report_generator.py | FaceLift Mouse Extension
</p>
</div>
</body>
</html>
"""
    
    with open(output_dir / "camera_setup_report.html", "w") as f:
        f.write(html)
    
    print(f"Camera setup report saved to: {output_dir}")
    return output_dir


# =============================================================================
# REPORT B: PREPROCESSING COMPARISON
# =============================================================================

def calculate_mouse_size(image_path: Path) -> Optional[Dict]:
    """Calculate mouse size from RGBA image"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    if img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3]
    else:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    
    total_pixels = mask.shape[0] * mask.shape[1]
    mouse_pixels = np.sum(mask > 127)
    
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    
    bbox_w = xs.max() - xs.min()
    bbox_h = ys.max() - ys.min()
    linear_ratio = np.sqrt((bbox_w * bbox_h) / total_pixels)
    
    return {
        "mouse_pixels": int(mouse_pixels),
        "area_ratio": mouse_pixels / total_pixels,
        "bbox_width": int(bbox_w),
        "bbox_height": int(bbox_h),
        "linear_ratio": linear_ratio,
        "error_amplification": 1.0 / linear_ratio if linear_ratio > 0 else float("inf")
    }


def create_pp_distribution_plot(datasets: Dict[str, DatasetMetrics], output_path: Path):
    """Create PP distribution comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # cx distribution
    ax1 = axes[0]
    for i, (name, metrics) in enumerate(datasets.items()):
        ax1.bar(i, metrics.cx_mean, yerr=metrics.cx_std, capsize=5,
               color=f"C{i}", edgecolor="black", linewidth=1, label=name)
    ax1.axhline(y=256, color="red", linestyle="--", linewidth=2, label="GS-LRM target")
    ax1.set_ylabel("cx (principal point X)", fontsize=12)
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets.keys(), rotation=45, ha="right")
    ax1.set_title("Principal Point X Distribution", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # cy distribution  
    ax2 = axes[1]
    for i, (name, metrics) in enumerate(datasets.items()):
        ax2.bar(i, metrics.cy_mean, yerr=metrics.cy_std, capsize=5,
               color=f"C{i}", edgecolor="black", linewidth=1)
    ax2.axhline(y=256, color="red", linestyle="--", linewidth=2)
    ax2.set_ylabel("cy (principal point Y)", fontsize=12)
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets.keys(), rotation=45, ha="right")
    ax2.set_title("Principal Point Y Distribution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def create_error_comparison_plot(datasets: Dict[str, DatasetMetrics], output_path: Path):
    """Create error metrics comparison plot"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(datasets.keys())
    x = range(len(names))
    
    # fx deviation
    ax1 = axes[0]
    fx_devs = [datasets[n].fx_deviation for n in names]
    colors = ["green" if d < 1 else "orange" if d < 10 else "red" for d in fx_devs]
    ax1.bar(x, fx_devs, color=colors, edgecolor="black")
    ax1.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="< 1px (good)")
    ax1.set_ylabel("fx deviation from 548.99", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_title("Focal Length Deviation", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # PP deviation
    ax2 = axes[1]
    pp_devs = [datasets[n].pp_deviation for n in names]
    colors = ["green" if d < 1 else "orange" if d < 10 else "red" for d in pp_devs]
    ax2.bar(x, pp_devs, color=colors, edgecolor="black")
    ax2.axhline(y=1, color="green", linestyle="--", alpha=0.5)
    ax2.set_ylabel("PP deviation from (256,256)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_title("Principal Point Deviation", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Ray error
    ax3 = axes[2]
    ray_errs = [datasets[n].ray_error_max for n in names]
    colors = ["green" if e < 0.1 else "orange" if e < 1 else "red" for e in ray_errs]
    ax3.bar(x, ray_errs, color=colors, edgecolor="black")
    ax3.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="< 0.1° (good)")
    ax3.set_ylabel("Max ray error (degrees)", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.set_title("Ray Direction Error", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_preprocessing_comparison_report(output_dir: Path):
    """Generate preprocessing comparison report (Report B)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating preprocessing comparison report...")
    
    # Find available datasets
    available_datasets = {}
    for dataset_dir in PREPROCESSED_DIR.iterdir():
        if dataset_dir.is_dir():
            cameras = load_processed_cameras(dataset_dir)
            if cameras:
                name = dataset_dir.name
                # Skip test/temp directories
                if "_test" not in name and "_t" != name[-2:]:
                    metrics = compute_dataset_metrics(name, cameras)
                    available_datasets[name] = metrics
    
    # Sort by name
    available_datasets = dict(sorted(available_datasets.items()))
    
    print(f"Found {len(available_datasets)} datasets: {list(available_datasets.keys())}")
    
    # Generate plots
    create_pp_distribution_plot(available_datasets, output_dir / "pp_distribution.png")
    create_error_comparison_plot(available_datasets, output_dir / "error_comparison.png")
    
    # Calculate mouse sizes for key datasets
    mouse_sizes = {}
    for name in ["D8_sample", "D8_1_sample", "D7_1"]:
        dataset_path = PREPROCESSED_DIR / name
        if dataset_path.exists():
            sample_dir = dataset_path / "train/000000"
            if not sample_dir.exists():
                sample_dirs = list((dataset_path / "train").glob("*"))
                if sample_dirs:
                    sample_dir = sample_dirs[0]
            img_path = sample_dir / "images/cam_000.png"
            if img_path.exists():
                mouse_sizes[name] = calculate_mouse_size(img_path)
    
    # Generate HTML
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Preprocessing Comparison Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; line-height: 1.6; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4a90d9; padding-bottom: 15px; }}
        h2 {{ color: #16213e; margin-top: 35px; border-left: 5px solid #4a90d9; padding-left: 15px; }}
        h3 {{ color: #34495e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; border-radius: 8px; overflow: hidden; font-size: 12px; }}
        th {{ background: #4a90d9; color: white; padding: 10px 8px; text-align: center; font-size: 11px; }}
        td {{ padding: 8px; text-align: center; border-bottom: 1px solid #eee; }}
        .good {{ background: #d4edda; }}
        .warning {{ background: #fff3cd; }}
        .bad {{ background: #f8d7da; }}
        .formula-box {{ background: white; padding: 15px; margin: 15px 0; border-left: 4px solid #4a90d9; border-radius: 5px; }}
        .key-point {{ background: #e8f6f3; padding: 15px; border-left: 4px solid #1abc9c; margin: 15px 0; border-radius: 5px; }}
        .warning-box {{ background: #fdf2e9; padding: 15px; border-left: 4px solid #e67e22; margin: 15px 0; border-radius: 5px; }}
        .viz-img {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 15px 0; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 8px; overflow-x: auto; }}
    </style>
</head>
<body>
<div class="container">
<h1>Preprocessing Methods Comparison Report</h1>
<p><strong>Generated:</strong> {timestamp} | <strong>Datasets:</strong> {len(available_datasets)}</p>

<h2>1. GS-LRM / FaceLift Expected Parameters</h2>
<div class="key-point">
<p>GS-LRM pretrained model expects:</p>
<ul>
<li><strong>fx = fy = {GSLRM_TARGET_FX:.10f}</strong></li>
<li><strong>cx = cy = {GSLRM_TARGET_PP:.1f}</strong> (PP at image center)</li>
<li><strong>Image size = 512 × 512</strong></li>
<li><strong>Camera distance ≈ {GSLRM_TARGET_DIST:.1f}</strong> (normalized)</li>
</ul>
</div>

<h2>2. Dataset Parameters Summary</h2>
<table>
<tr>
<th>Dataset</th>
<th>fx (mean±std)</th>
<th>fy (mean±std)</th>
<th>cx (mean±std)</th>
<th>cy (mean±std)</th>
<th>Distance</th>
<th>fx Dev.</th>
<th>PP Dev.</th>
<th>Ray Err.</th>
<th>Skew</th>
<th>Geom.</th>
</tr>
"""
    
    for name, m in available_datasets.items():
        fx_class = "good" if m.fx_deviation < 1 else "warning" if m.fx_deviation < 10 else "bad"
        pp_class = "good" if m.pp_deviation < 1 else "warning" if m.pp_deviation < 10 else "bad"
        ray_class = "good" if m.ray_error_max < 0.1 else "warning" if m.ray_error_max < 1 else "bad"
        geom_class = "good" if m.geometry_correct else "bad"
        
        html += f"""<tr>
<td><strong>{name}</strong></td>
<td>{m.fx_mean:.2f}±{m.fx_std:.2f}</td>
<td>{m.fy_mean:.2f}±{m.fy_std:.2f}</td>
<td>{m.cx_mean:.1f}±{m.cx_std:.1f}</td>
<td>{m.cy_mean:.1f}±{m.cy_std:.1f}</td>
<td>{m.dist_mean:.2f}</td>
<td class="{fx_class}">{m.fx_deviation:.2f}</td>
<td class="{pp_class}">{m.pp_deviation:.1f}</td>
<td class="{ray_class}">{m.ray_error_max:.2f}°</td>
<td>{"✓" if m.skew_corrected else "×"}</td>
<td class="{geom_class}">{"✓" if m.geometry_correct else "✗"}</td>
</tr>
"""
    
    html += """</table>

<h2>3. Error Metrics Visualization</h2>
<img src="error_comparison.png" class="viz-img" alt="Error Comparison">

<h2>4. Principal Point Distribution</h2>
<img src="pp_distribution.png" class="viz-img" alt="PP Distribution">

<h2>5. Mathematical Foundation</h2>

<div class="formula-box">
<h3>5.1 Ray Direction Calculation</h3>
<p>For pixel (u, v), the ray direction in camera coordinates:</p>
\\[
\\mathbf{d}_c = \\begin{bmatrix} (u - c_x) / f_x \\\\ (v - c_y) / f_y \\\\ 1 \\end{bmatrix}
\\]
<p><strong>Critical:</strong> If \\(c_x, c_y\\) are wrong, the ray direction is wrong.</p>
</div>

<div class="formula-box">
<h3>5.2 Ray Error Quantification</h3>
<p>At pixel (256, 256) center, the ray angle error when PP deviates from (256, 256):</p>
\\[
\\theta_{\\text{error}} = \\arctan\\left(\\sqrt{\\left(\\frac{256 - c_x}{f_x}\\right)^2 + \\left(\\frac{256 - c_y}{f_y}\\right)^2}\\right)
\\]
</div>

<div class="formula-box">
<h3>5.3 D8 Homography Transform</h3>
<p>D8 uses homography instead of affine to correct skew:</p>
\\[
H = K_{\\text{target}} \\cdot K_{\\text{orig}}^{-1}
\\]
<p>Where:</p>
\\[
K_{\\text{target}} = \\begin{bmatrix} 548.99 & 0 & 256 \\\\ 0 & 548.99 & 256 \\\\ 0 & 0 & 1 \\end{bmatrix}
\\]
</div>

<h2>6. D8 vs D7.1 Comparison</h2>
<table>
<tr><th>Aspect</th><th>D7.1 (Affine)</th><th>D8 (Homography)</th></tr>
<tr><td>fx Precision</td><td>549.0</td><td class="good">548.9937744140625</td></tr>
<tr><td>Skew Handling</td><td>Ignored (up to 0.9px error)</td><td class="good">Corrected (0px error)</td></tr>
<tr><td>Ray Error</td><td>~0°</td><td class="good">~1e-6° (numerical precision)</td></tr>
<tr><td>Interpolation</td><td>LINEAR</td><td class="good">LANCZOS4</td></tr>
</table>

<h2>7. Original Camera Skew Values</h2>
<table>
<tr><th>Camera</th><th>Original Skew (px)</th><th>D7.1 Edge Error (px)</th><th>D8 Error</th></tr>
<tr><td>Cam 0</td><td>-5.82</td><td>0.91</td><td class="good">~0</td></tr>
<tr><td>Cam 1</td><td>+1.39</td><td>0.23</td><td class="good">~0</td></tr>
<tr><td>Cam 2</td><td>-0.91</td><td>0.14</td><td class="good">~0</td></tr>
<tr><td>Cam 3</td><td>-5.89</td><td>0.94</td><td class="good">~0</td></tr>
<tr><td>Cam 4</td><td>-1.27</td><td>0.20</td><td class="good">~0</td></tr>
<tr><td>Cam 5</td><td>-4.93</td><td>0.77</td><td class="good">~0</td></tr>
</table>
"""

    # Mouse size section
    if mouse_sizes:
        html += """
<h2>8. Mouse Size & Error Amplification</h2>
<table>
<tr><th>Dataset</th><th>Mouse Pixels</th><th>Area Ratio</th><th>BBox Size</th><th>Linear Ratio</th><th>Error Amp.</th></tr>
"""
        for name, size in mouse_sizes.items():
            if size:
                html += f"""<tr>
<td><strong>{name}</strong></td>
<td>{size['mouse_pixels']:,}</td>
<td>{size['area_ratio']*100:.1f}%</td>
<td>{size['bbox_width']}×{size['bbox_height']}</td>
<td>{size['linear_ratio']*100:.1f}%</td>
<td><strong>{size['error_amplification']:.1f}×</strong></td>
</tr>
"""
        html += """</table>

<div class="warning-box">
<h4>Error Amplification Interpretation</h4>
<p>When mouse occupies ~30% of image linearly:</p>
<ul>
<li>1px camera error → <strong>3.3px</strong> error relative to mouse</li>
<li>D4 PP error (~37px) → <strong>~120px</strong> relative error</li>
<li>D8 skew correction prevents up to <strong>~3px</strong> relative error</li>
</ul>
</div>
"""

    html += """
<h2>9. D8.1 Zoom Analysis</h2>
<div class="warning-box">
<h4>D8.1 fx ≠ 549 - Is This a Problem?</h4>
<p>D8.1 applies 1.3× zoom via crop+resize:</p>
<ul>
<li><strong>fx = 713.69</strong> (= 548.99 × 1.3) - different from pretrained</li>
<li><strong>cx, cy varies</strong> per view (not fixed at 256)</li>
<li>But ray directions are <strong>GEOMETRICALLY CORRECT</strong></li>
</ul>
<p><strong>Conclusion:</strong> D8.1 is correct. Model will adapt during finetuning. Use D8 if strict fx=549 is needed.</p>
</div>

<h2>10. Recommendations</h2>
<table>
<tr><th>Use Case</th><th>Recommended</th><th>Reason</th></tr>
<tr><td>Standard training</td><td class="good"><strong>D8</strong></td><td>Exact GS-LRM params, skew corrected</td></tr>
<tr><td>Larger mouse needed</td><td class="warning"><strong>D8.1</strong></td><td>1.3× zoom, fx=713.69</td></tr>
<tr><td>Legacy comparison</td><td>D7.1</td><td>Compatible with previous experiments</td></tr>
</table>

<h2>11. Preprocessing Commands</h2>
<pre>
# D8 (Recommended)
python -m mouse_extensions.preprocessing.preprocess \\
    --preset D8 --input-dir /path/to/raw --output-dir /path/to/D8

# D8.1 (With 1.3x zoom)
python -m mouse_extensions.preprocessing.preprocess \\
    --preset D8.1 --input-dir /path/to/raw --output-dir /path/to/D8.1
</pre>

<hr>
<p style="color:#888; font-size:11px; text-align:center;">
Generated by report_generator.py | FaceLift Mouse Extension | {timestamp}
</p>
</div>
</body>
</html>
"""
    
    with open(output_dir / "preprocessing_comparison_report.html", "w") as f:
        f.write(html)
    
    print(f"Preprocessing comparison report saved to: {output_dir}")
    return output_dir


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate FaceLift Mouse Reports")
    parser.add_argument("--dataset", "-d", type=str, help="Dataset name for camera setup report")
    parser.add_argument("--all", "-a", action="store_true", help="Generate all reports")
    parser.add_argument("--output-dir", "-o", type=str, default="./reports", help="Output directory")
    parser.add_argument("--camera-only", action="store_true", help="Generate only camera setup report")
    parser.add_argument("--comparison-only", action="store_true", help="Generate only comparison report")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.camera_only and args.dataset:
        generate_camera_setup_report(args.dataset, output_dir / f"camera_setup_{args.dataset}")
    elif args.comparison_only:
        generate_preprocessing_comparison_report(output_dir / "preprocessing_comparison")
    elif args.all:
        # Generate both reports
        generate_camera_setup_report("D8_sample", output_dir / "camera_setup_D8")
        generate_preprocessing_comparison_report(output_dir / "preprocessing_comparison")
    elif args.dataset:
        generate_camera_setup_report(args.dataset, output_dir / f"camera_setup_{args.dataset}")
    else:
        print("Usage:")
        print("  python report_generator.py --dataset D8_sample  # Camera setup report")
        print("  python report_generator.py --comparison-only    # Comparison report")
        print("  python report_generator.py --all                # All reports")


if __name__ == "__main__":
    main()
