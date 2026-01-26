#!/usr/bin/env python3
"""
Unified Preprocessing Report Generator

Generates comprehensive HTML report combining:
- Camera setup and visualization
- MVG theory and formulas
- Dataset comparison (M1, M2, M3 series)
- PP analysis and error metrics
- Hypothesis testing section

Usage:
    python -m mouse_extensions.reports.unified_report --output-dir ./reports
    python -m mouse_extensions.reports.unified_report --datasets M3_1,M3_2 --output-dir ./reports

Author: FaceLift Team
Date: 2026-01-27
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from .modules.theory import TheoryModule
from .modules.camera_viz import CameraVisualizationModule
from .modules.pp_analysis import PPAnalysisModule
from .modules.dataset_comparison import DatasetComparisonModule


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path('/home/joon/dev/FaceLift')
DATA_DIR = Path('/home/joon/data')
RAW_DIR = DATA_DIR / 'raw/markerless_mouse_1_nerf'
PREPROCESSED_DIR = DATA_DIR / 'preprocessed/FaceLift_mouse'

HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>FaceLift Unified Preprocessing Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            font-family: "CMU Serif", Georgia, serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px;
            background: #fff;
            line-height: 1.8;
        }}
        h1 {{ font-size: 28px; border-bottom: 3px solid #333; padding-bottom: 15px; margin-bottom: 30px; }}
        h2 {{ font-size: 22px; color: #2c3e50; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }}
        h3 {{ font-size: 18px; color: #34495e; margin-top: 25px; }}
        h4 {{ font-size: 16px; color: #555; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: center; }}
        th {{ background: #f8f9fa; font-weight: bold; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .formula-box {{
            background: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
            overflow-x: auto;
        }}
        .key-point {{
            background: #e8f6f3;
            padding: 15px;
            border-left: 4px solid #1abc9c;
            margin: 15px 0;
        }}
        .warning {{
            background: #fdf2e9;
            padding: 15px;
            border-left: 4px solid #e67e22;
            margin: 15px 0;
        }}
        .good {{ background: #d4edda; }}
        .warning {{ background: #fff3cd; }}
        .bad {{ background: #f8d7da; }}
        .toc {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }}
        .toc ul {{ list-style-type: none; padding-left: 20px; }}
        .toc a {{ text-decoration: none; color: #3498db; }}
        code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        pre {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; overflow-x: auto; border-radius: 5px; font-size: 13px; }}
        .viz-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .viz-img {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .viz-full {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 15px 0; }}
        .color-box {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; }}
        .metadata {{ font-size: 12px; color: #888; margin-top: 50px; }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 12px; margin: 20px 0;
        }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .metric {{ background: rgba(255,255,255,0.2); padding: 12px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 22px; font-weight: bold; }}
        .metric-label {{ font-size: 11px; opacity: 0.9; }}
    </style>
</head>
<body>

<h1>FaceLift Mouse Dataset Preprocessing Report</h1>
<p><strong>Generated:</strong> {timestamp} |
<strong>Version:</strong> Unified v1.0 |
<strong>Data:</strong> markerless_mouse_1_nerf (DANNCE)</p>

<div class="summary-box">
<h3 style="margin-top:0;">Quick Summary</h3>
<div class="summary-grid">
    <div class="metric"><div class="metric-value">M3_2</div><div class="metric-label">Recommended</div></div>
    <div class="metric"><div class="metric-value">256</div><div class="metric-label">Target PP</div></div>
    <div class="metric"><div class="metric-value">549</div><div class="metric-label">Target fx</div></div>
    <div class="metric"><div class="metric-value">2.7</div><div class="metric-label">Target Distance</div></div>
</div>
</div>

<div class="toc">
<h3>Table of Contents</h3>
<ul>
    <li><a href="#camera">1. Camera Setup</a></li>
    <li><a href="#theory">2. MVG Theory Foundation</a></li>
    <li><a href="#comparison">3. Dataset Comparison</a></li>
    <li><a href="#hypothesis">4. Hypothesis Testing</a></li>
    <li><a href="#commands">5. Preprocessing Commands</a></li>
    <li><a href="#pp-analysis">6. Principal Point Analysis</a></li>
    <li><a href="#appendix">Appendix: Full Dataset Parameters</a></li>
</ul>
</div>

{camera_section}

{theory_section}

{comparison_section}

{pp_section}

<h2 id="appendix">Appendix: GS-LRM Expected Parameters</h2>

<div class="key-point">
<h3>Pretrained Model Expectations</h3>
<p>GS-LRM/FaceLift was trained on Objaverse synthetic data with:</p>
<ul>
    <li><strong>fx = fy = 548.9937744140625</strong> (exact)</li>
    <li><strong>cx = cy = 256.0</strong> (PP at image center)</li>
    <li><strong>Image size = 512 × 512</strong></li>
    <li><strong>Camera distance ≈ 2.7</strong> (normalized)</li>
</ul>
</div>

<table>
<tr><th>Property</th><th>Mouse (Original)</th><th>Mouse (Preprocessed)</th><th>FaceLift (Objaverse)</th></tr>
<tr><td>Image Size</td><td>1152 × 1024</td><td>512 × 512</td><td>512 × 512</td></tr>
<tr><td>Focal Length</td><td>1557-1637</td><td>549</td><td>549</td></tr>
<tr><td>PP (cx, cy)</td><td>(583-642, 418-552)</td><td>(256, 256) for M3_2</td><td>(256, 256)</td></tr>
<tr><td>Camera Distance</td><td>246-415 mm</td><td>2.7 (normalized)</td><td>~2.7</td></tr>
<tr><td>Elevation</td><td>~10-30° (varies)</td><td>~10-30°</td><td>-10° to 40° (varied)</td></tr>
<tr><td>Number of Views</td><td>6 fixed</td><td>4 input / 6 total</td><td>1-8 input</td></tr>
</table>

<hr>
<p class="metadata">
<strong>Report generated by:</strong> unified_report.py<br>
<strong>Data source:</strong> markerless_mouse_1_nerf (DANNCE, Bolaños et al. 2021)<br>
<strong>Reference:</strong> <a href="https://www.nature.com/articles/s41592-021-01103-9">Nature Methods 18, 378-381 (2021)</a><br>
<strong>FaceLift Project:</strong> <a href="https://github.com/joon/FaceLift">/home/joon/dev/FaceLift</a>
</p>

</body>
</html>
'''


class UnifiedReportGenerator:
    """Generates unified preprocessing report."""
    
    def __init__(self, output_dir: Path, datasets: List[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets = datasets or ['D7_1', 'D8', 'M3_1', 'M3_2']
        
        # Initialize modules
        self.theory = TheoryModule()
        self.camera_viz = CameraVisualizationModule(output_dir=self.output_dir)
        self.pp_analysis = PPAnalysisModule(output_dir=self.output_dir)
        self.comparison = DatasetComparisonModule(datasets=self.datasets)
    
    def load_cameras(self):
        """Load camera data."""
        # Try to load from raw data first
        pkl_path = RAW_DIR / 'new_cam.pkl'
        if pkl_path.exists():
            self.camera_viz.load_cameras_from_pkl(pkl_path)
            return True
        
        # Fallback to processed data
        for dataset in self.datasets:
            dataset_path = PREPROCESSED_DIR / dataset
            if dataset_path.exists():
                sample_dirs = list((dataset_path / 'train').glob('*'))
                if sample_dirs:
                    cam_file = sample_dirs[0] / 'opencv_cameras.json'
                    if cam_file.exists():
                        self.camera_viz.load_cameras_from_json(cam_file)
                        return True
        
        return False
    
    def analyze_datasets(self):
        """Analyze PP distribution for all datasets."""
        for dataset in self.datasets:
            dataset_path = PREPROCESSED_DIR / dataset
            if dataset_path.exists():
                try:
                    self.pp_analysis.analyze_dataset(dataset_path, dataset)
                    print(f"  Analyzed: {dataset}")
                except Exception as e:
                    print(f"  Skip {dataset}: {e}")
    
    def generate_visualizations(self):
        """Generate all visualization images."""
        print("Generating visualizations...")
        
        if self.camera_viz.cameras:
            self.camera_viz.plot_top_view()
            self.camera_viz.plot_side_view()
            self.camera_viz.plot_3d_view()
            print("  Camera visualizations: Done")
        
        if self.pp_analysis.stats:
            self.pp_analysis.plot_pp_distribution()
            self.pp_analysis.plot_ray_error_comparison()
            print("  PP analysis plots: Done")
    
    def generate_report(self) -> Path:
        """Generate the complete unified report."""
        print("=" * 60)
        print("FaceLift Unified Preprocessing Report Generator")
        print("=" * 60)
        
        # Load data
        print("\nLoading cameras...")
        self.load_cameras()
        
        print("\nAnalyzing datasets...")
        self.analyze_datasets()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate HTML sections
        print("\nGenerating HTML...")
        
        camera_section = ""
        if self.camera_viz.cameras:
            camera_section = self.camera_viz.generate_camera_section_html()
        
        theory_section = self.theory.generate_full_theory_section()
        comparison_section = self.comparison.generate_full_comparison_section()
        
        pp_section = ""
        if self.pp_analysis.stats:
            pp_section = self.pp_analysis.generate_pp_section_html()
        
        # Combine into final HTML
        html_content = HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
            camera_section=camera_section,
            theory_section=theory_section,
            comparison_section=comparison_section,
            pp_section=pp_section
        )
        
        # Save report
        report_path = self.output_dir / 'unified_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"\n✅ Report generated: {report_path}")
        print(f"   Images: {self.output_dir}")
        
        return report_path


def main():
    parser = argparse.ArgumentParser(description='Generate unified preprocessing report')
    parser.add_argument('--output-dir', '-o', type=str, default='./reports/unified',
                       help='Output directory for report and images')
    parser.add_argument('--datasets', '-d', type=str, default='D7_1,D8,M3_1,M3_2',
                       help='Comma-separated list of datasets to analyze')
    
    args = parser.parse_args()
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    generator = UnifiedReportGenerator(
        output_dir=Path(args.output_dir),
        datasets=datasets
    )
    
    generator.generate_report()


if __name__ == '__main__':
    main()
