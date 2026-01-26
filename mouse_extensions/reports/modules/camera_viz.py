"""
Camera Visualization Module - 3D/2D camera arrangement visualizations.

Generates camera grid, top view, side view, and 3D perspective plots.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Camera colors (consistent across all visualizations)
CAMERA_COLORS = {
    0: '#E53935',  # Red
    1: '#1E88E5',  # Blue  
    2: '#43A047',  # Green
    3: '#FB8C00',  # Orange
    4: '#8E24AA',  # Purple
    5: '#00ACC1',  # Cyan
}

# Layout order for visualization
LAYOUT_ORDER = [[4, 2, 1], [0, 5, 3]]
OPPOSING_PAIRS = [(4, 3), (2, 5), (1, 0)]


@dataclass
class CameraParams:
    """Camera parameters for visualization."""
    view_id: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float
    position: np.ndarray
    forward: np.ndarray
    azimuth: float
    elevation: float
    distance: float


class CameraVisualizationModule:
    """Generates camera arrangement visualizations."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path('.')
        self.cameras: List[CameraParams] = []
    
    def load_cameras_from_pkl(self, pkl_path: Path) -> List[CameraParams]:
        """Load original cameras from pickle file."""
        with open(pkl_path, 'rb') as f:
            raw_cams = pickle.load(f)
        
        cameras = []
        for i, cam in enumerate(raw_cams[:6]):
            K = np.array(cam['K'])
            R = np.array(cam['R'])
            T = np.array(cam['T']).flatten()
            
            # Build w2c from R, T
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            
            c2w = np.linalg.inv(w2c)
            
            position = c2w[:3, 3]
            forward = c2w[:3, 2]  # OpenCV convention
            
            # Calculate azimuth/elevation
            azimuth = np.degrees(np.arctan2(position[0], position[2]))
            dist = np.linalg.norm(position)
            elevation = np.degrees(np.arcsin(position[1] / dist)) if dist > 0 else 0
            
            cameras.append(CameraParams(
                view_id=i,
                fx=K[0, 0], fy=K[1, 1],
                cx=K[0, 2], cy=K[1, 2],
                skew=K[0, 1] if K.shape[1] > 2 else 0,
                position=position,
                forward=forward,
                azimuth=azimuth,
                elevation=elevation,
                distance=dist
            ))
        
        self.cameras = cameras
        return cameras
    
    def load_cameras_from_json(self, json_path: Path) -> List[CameraParams]:
        """Load processed cameras from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        cameras = []
        for i, frame in enumerate(data['frames'][:6]):
            K = np.array(frame['K'])
            w2c = np.array(frame['w2c'])
            c2w = np.linalg.inv(w2c)
            
            position = c2w[:3, 3]
            forward = c2w[:3, 2]
            
            dist = np.linalg.norm(position)
            azimuth = np.degrees(np.arctan2(position[0], position[2]))
            elevation = np.degrees(np.arcsin(position[1] / dist)) if dist > 0 else 0
            
            cameras.append(CameraParams(
                view_id=i,
                fx=K[0, 0], fy=K[1, 1],
                cx=K[0, 2], cy=K[1, 2],
                skew=K[0, 1] if len(K) > 2 else 0,
                position=position,
                forward=forward,
                azimuth=azimuth,
                elevation=elevation,
                distance=dist
            ))
        
        self.cameras = cameras
        return cameras
    
    def generate_camera_table_html(self) -> str:
        """Generate HTML table of camera parameters."""
        rows = []
        for cam in self.cameras:
            color = CAMERA_COLORS.get(cam.view_id, '#888888')
            rows.append(f'''
<tr>
    <td><strong>Cam {cam.view_id}</strong></td>
    <td><span class="color-box" style="background:{color}"></span></td>
    <td>{cam.fx:.2f}</td>
    <td>{cam.fy:.2f}</td>
    <td>{cam.cx:.1f}</td>
    <td>{cam.cy:.1f}</td>
    <td>{cam.azimuth:.1f}</td>
    <td>{cam.elevation:.1f}</td>
    <td>{cam.distance:.4f}</td>
</tr>''')
        
        return f'''
<h3>Original Camera Parameters (6 Views)</h3>
<table>
<tr>
    <th>Cam</th><th>Color</th><th>fx</th><th>fy</th>
    <th>cx</th><th>cy</th><th>Azimuth</th><th>Elevation</th><th>Distance (mm)</th>
</tr>
{''.join(rows)}
</table>
<p><em>Opposing pairs: 4-3, 2-5, 1-0</em></p>
'''
    
    def plot_top_view(self, save_path: Path = None) -> Path:
        """Generate top-down view of camera arrangement."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot mouse position (origin)
        ax.scatter(0, 0, s=200, c='brown', marker='o', zorder=10, label='Mouse')
        ax.add_patch(plt.Circle((0, 0), 20, color='brown', alpha=0.3))
        
        # Plot cameras
        for cam in self.cameras:
            color = CAMERA_COLORS.get(cam.view_id, '#888888')
            x, z = cam.position[0], cam.position[2]
            
            # Camera position
            ax.scatter(x, z, s=150, c=color, marker='s', zorder=5)
            ax.annotate(f'Cam {cam.view_id}', (x, z), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
            
            # Direction arrow (towards origin)
            dx, dz = -x / cam.distance * 30, -z / cam.distance * 30
            ax.arrow(x, z, dx, dz, head_width=10, 
                    head_length=5, fc=color, ec=color, alpha=0.7)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title('Top View (X-Z Plane)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        if save_path is None:
            save_path = self.output_dir / 'camera_top_view.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_side_view(self, save_path: Path = None) -> Path:
        """Generate side view showing elevation."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot cameras by azimuth (x) and elevation (y)
        for cam in self.cameras:
            color = CAMERA_COLORS.get(cam.view_id, '#888888')
            ax.scatter(cam.azimuth, cam.elevation, s=150, c=color, marker='s')
            ax.annotate(f'Cam {cam.view_id}\n(d={cam.distance:.0f})', 
                       (cam.azimuth, cam.elevation),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Azimuth (degrees)')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_title('Side View (Azimuth vs Elevation)')
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / 'camera_side_view.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_3d_view(self, save_path: Path = None) -> Path:
        """Generate 3D view of camera arrangement."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mouse at origin
        ax.scatter(0, 0, 0, s=200, c='brown', marker='o', label='Mouse')
        
        # Plot cameras
        for cam in self.cameras:
            color = CAMERA_COLORS.get(cam.view_id, '#888888')
            x, y, z = cam.position
            
            ax.scatter(x, y, z, s=100, c=color, marker='s')
            ax.text(x, y, z, f'  Cam {cam.view_id}', fontsize=9)
            
            # Ray towards origin
            ax.plot([x, 0], [y, 0], [z, 0], c=color, alpha=0.5, linestyle='--')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)') 
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Camera Arrangement')
        
        if save_path is None:
            save_path = self.output_dir / 'camera_3d_view.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def generate_all_visualizations(self) -> Dict[str, Path]:
        """Generate all camera visualizations."""
        return {
            'top_view': self.plot_top_view(),
            'side_view': self.plot_side_view(),
            '3d_view': self.plot_3d_view(),
        }
    
    def generate_camera_section_html(self, image_prefix: str = '') -> str:
        """Generate complete camera section HTML."""
        return f'''
<h2 id="camera">1. Camera Setup</h2>

{self.generate_camera_table_html()}

<h3>Spatial Arrangement</h3>
<div class="viz-grid">
    <div>
        <img src="{image_prefix}camera_top_view.png" class="viz-img" alt="Top View">
        <p style="text-align:center; color:#666;">Top View (X-Z Plane)</p>
    </div>
    <div>
        <img src="{image_prefix}camera_side_view.png" class="viz-img" alt="Side View">
        <p style="text-align:center; color:#666;">Side View (Azimuth vs Elevation)</p>
    </div>
</div>

<h3>3D View</h3>
<img src="{image_prefix}camera_3d_view.png" class="viz-full" alt="3D View">
'''
