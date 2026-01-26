"""
Principal Point Analysis Module - PP distribution and error analysis.

Analyzes PP (cx, cy) across datasets, generates distribution plots,
and calculates ray direction errors.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class PPStats:
    """Principal Point statistics for a dataset."""
    dataset: str
    cx_mean: float
    cx_std: float
    cy_mean: float
    cy_std: float
    cx_min: float
    cx_max: float
    cy_min: float
    cy_max: float
    ray_error_mean: float
    ray_error_max: float
    samples: int
    fx_mean: float = 549.0


class PPAnalysisModule:
    """Analyzes Principal Point distributions across datasets."""
    
    TARGET_PP = 256.0
    TARGET_FX = 548.9937744140625
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path('.')
        self.stats: Dict[str, PPStats] = {}
    
    def _find_sample_dirs(self, dataset_path: Path) -> List[Path]:
        """Find sample directories (handles both train/ and samples/ structures)."""
        # Try train/ first (D7_1, D8 structure)
        train_dir = dataset_path / 'train'
        if train_dir.exists():
            return [d for d in train_dir.iterdir() if d.is_dir()]
        
        # Try samples/ (M3 series structure)
        samples_dir = dataset_path / 'samples'
        if samples_dir.exists():
            return [d for d in samples_dir.iterdir() if d.is_dir()]
        
        raise FileNotFoundError(f"No train/ or samples/ directory found in {dataset_path}")
    
    def analyze_dataset(self, dataset_path: Path, dataset_name: str, max_samples: int = 100) -> PPStats:
        """Analyze PP distribution for a dataset."""
        cx_values = []
        cy_values = []
        fx_values = []
        
        sample_dirs = self._find_sample_dirs(dataset_path)
        
        # Limit samples for speed
        sample_dirs = sample_dirs[:max_samples]
        
        for sample_dir in sample_dirs:
            cam_file = sample_dir / 'opencv_cameras.json'
            if not cam_file.exists():
                continue
            
            with open(cam_file, 'r') as f:
                data = json.load(f)
            
            for frame in data.get('frames', []):
                # Handle direct fx, fy, cx, cy format (GS-LRM style)
                if 'cx' in frame and 'cy' in frame:
                    cx_values.append(frame['cx'])
                    cy_values.append(frame['cy'])
                    fx_values.append(frame.get('fx', 549.0))
                # Handle K matrix format (legacy)
                elif 'K' in frame:
                    K = np.array(frame['K'])
                    cx_values.append(K[0, 2])
                    cy_values.append(K[1, 2])
                    fx_values.append(K[0, 0])
        
        if not cx_values:
            raise ValueError(f"No camera data found in {dataset_path}")
        
        cx_values = np.array(cx_values)
        cy_values = np.array(cy_values)
        fx_values = np.array(fx_values)
        
        # Calculate ray errors
        ray_errors = self._compute_ray_errors(cx_values, cy_values, np.mean(fx_values))
        
        stats = PPStats(
            dataset=dataset_name,
            cx_mean=np.mean(cx_values),
            cx_std=np.std(cx_values),
            cy_mean=np.mean(cy_values),
            cy_std=np.std(cy_values),
            cx_min=np.min(cx_values),
            cx_max=np.max(cx_values),
            cy_min=np.min(cy_values),
            cy_max=np.max(cy_values),
            ray_error_mean=np.mean(ray_errors),
            ray_error_max=np.max(ray_errors),
            samples=len(cx_values),
            fx_mean=np.mean(fx_values)
        )
        
        self.stats[dataset_name] = stats
        return stats
    
    def _compute_ray_errors(self, cx_values: np.ndarray, cy_values: np.ndarray, fx: float) -> np.ndarray:
        """Compute ray direction errors in degrees."""
        offset_x = self.TARGET_PP - cx_values
        offset_y = self.TARGET_PP - cy_values
        
        errors_rad = np.arctan(np.sqrt(
            (offset_x / fx)**2 + 
            (offset_y / fx)**2
        ))
        return np.degrees(errors_rad)
    
    def plot_pp_distribution(self, save_path: Path = None) -> Path:
        """Plot PP distribution scatter for all datasets."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.stats)))
        
        for (name, stats), color in zip(self.stats.items(), colors):
            # Plot mean with error bars
            ax.errorbar(
                stats.cx_mean, stats.cy_mean,
                xerr=stats.cx_std, yerr=stats.cy_std,
                fmt='o', markersize=10, color=color,
                capsize=5, label=f'{name} (n={stats.samples})'
            )
        
        # Target PP
        ax.axhline(y=256, color='red', linestyle='--', alpha=0.5, label='Target (256)')
        ax.axvline(x=256, color='red', linestyle='--', alpha=0.5)
        ax.scatter(256, 256, s=200, c='red', marker='*', zorder=10)
        
        ax.set_xlabel('cx (Principal Point X)', fontsize=12)
        ax.set_ylabel('cy (Principal Point Y)', fontsize=12)
        ax.set_title('Principal Point Distribution by Dataset', fontsize=14)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Auto-scale with margin
        all_cx = [s.cx_mean for s in self.stats.values()]
        all_cy = [s.cy_mean for s in self.stats.values()]
        all_cx_std = [s.cx_std for s in self.stats.values()]
        all_cy_std = [s.cy_std for s in self.stats.values()]
        
        min_x = min(min(all_cx) - max(all_cx_std) - 10, 240)
        max_x = max(max(all_cx) + max(all_cx_std) + 10, 270)
        min_y = min(min(all_cy) - max(all_cy_std) - 10, 240)
        max_y = max(max(all_cy) + max(all_cy_std) + 10, 270)
        
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        
        if save_path is None:
            save_path = self.output_dir / 'pp_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_ray_error_comparison(self, save_path: Path = None) -> Path:
        """Plot ray error comparison bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = list(self.stats.keys())
        means = [s.ray_error_mean for s in self.stats.values()]
        maxs = [s.ray_error_max for s in self.stats.values()]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, means, width, label='Mean Ray Error', color='steelblue')
        bars2 = ax.bar(x + width/2, maxs, width, label='Max Ray Error', color='coral')
        
        # Risk threshold lines
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Low risk (<0.5°)')
        ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Medium risk (<2°)')
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Ray Direction Error (degrees)', fontsize=12)
        ax.set_title('Ray Error Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = self.output_dir / 'ray_error_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    def generate_pp_stats_table_html(self) -> str:
        """Generate HTML table of PP statistics."""
        rows = []
        for name, stats in self.stats.items():
            # Determine status based on PP deviation and ray error
            pp_dev = np.sqrt((stats.cx_mean - 256)**2 + (stats.cy_mean - 256)**2)
            
            if pp_dev < 1.0 and stats.ray_error_max < 0.5:
                status_class = 'good'
                status = '✅ Good'
            elif pp_dev < 10.0 and stats.ray_error_max < 2.0:
                status_class = 'warning'
                status = '⚠️ Check'
            else:
                status_class = 'bad'
                status = '❌ Error'
            
            rows.append(f'''
<tr class="{status_class}">
    <td><strong>{name}</strong></td>
    <td>{stats.cx_mean:.1f} ± {stats.cx_std:.1f}</td>
    <td>{stats.cy_mean:.1f} ± {stats.cy_std:.1f}</td>
    <td>{pp_dev:.1f}</td>
    <td>{stats.fx_mean:.1f}</td>
    <td>{stats.ray_error_mean:.2f}°</td>
    <td>{stats.ray_error_max:.2f}°</td>
    <td>{stats.samples}</td>
    <td>{status}</td>
</tr>''')
        
        return f'''
<h3>PP Statistics Summary</h3>
<table>
<tr>
    <th>Dataset</th>
    <th>cx (mean±std)</th>
    <th>cy (mean±std)</th>
    <th>PP Deviation</th>
    <th>fx (mean)</th>
    <th>Mean Ray Error</th>
    <th>Max Ray Error</th>
    <th>Samples</th>
    <th>Status</th>
</tr>
{''.join(rows)}
</table>
<p><em>PP Deviation = √((cx-256)² + (cy-256)²). Target: PP=256, fx=549, Ray Error=0°</em></p>
'''
    
    def generate_pp_section_html(self, image_prefix: str = '') -> str:
        """Generate complete PP analysis section."""
        return f'''
<h2 id="pp-analysis">5. Principal Point Analysis</h2>

{self.generate_pp_stats_table_html()}

<h3>PP Distribution</h3>
<img src="{image_prefix}pp_distribution.png" class="viz-full" alt="PP Distribution">
<p><em>Each point shows dataset mean PP with ±1 std error bars. Red star = Target (256, 256).</em></p>

<h3>Ray Error Comparison</h3>
<img src="{image_prefix}ray_error_comparison.png" class="viz-full" alt="Ray Error">
<p><em>Green line: Low risk (&lt;0.5°). Orange line: Medium risk threshold (2°).</em></p>
'''
