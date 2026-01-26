#!/usr/bin/env python3
"""
Clipping Analyzer for Preprocessed Datasets.

Supports two directory structures:
1. Single folder: dataset/samples/{sample_id}/images/
2. Train/val split: dataset/{train,val}/{sample_id}/images/

Usage:
    python clipping_analyzer.py /path/to/dataset --output /path/to/output
    python clipping_analyzer.py /path/to/dataset --sample-rate 100
"""

import argparse
import json
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


class ClippingAnalyzer:
    """Analyze foreground clipping in preprocessed datasets."""
    
    def __init__(
        self,
        dataset_path: str,
        output_path: Optional[str] = None,
        alpha_threshold: int = 127,
    ):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path) if output_path else self.dataset_path / "clipping_check"
        self.alpha_threshold = alpha_threshold
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Detect directory structure
        self.structure = self._detect_structure()
        
    def _detect_structure(self) -> str:
        """Detect dataset directory structure."""
        if (self.dataset_path / "samples").exists():
            return "samples"
        elif (self.dataset_path / "train").exists():
            return "train_val"
        else:
            raise ValueError(f"Unknown dataset structure in {self.dataset_path}")
    
    def _get_sample_dirs(self) -> List[Tuple[str, Path]]:
        """Get list of (sample_id, sample_path) tuples."""
        if self.structure == "samples":
            samples_dir = self.dataset_path / "samples"
            return [(d.name, d) for d in sorted(samples_dir.iterdir()) if d.is_dir()]
        else:
            results = []
            for split in ["train", "val"]:
                split_dir = self.dataset_path / split
                if split_dir.exists():
                    for d in sorted(split_dir.iterdir()):
                        if d.is_dir():
                            results.append((f"{split}/{d.name}", d))
            return results
        
    def analyze_sample(self, sample_id: str, sample_path: Path) -> dict:
        """Analyze a single sample for clipping."""
        images_dir = sample_path / "images"
        if not images_dir.exists():
            images_dir = sample_path  # Some datasets have images directly in sample folder
            
        if not images_dir.exists():
            return None
            
        sample_result = {"id": sample_id, "cams": [], "path": str(sample_path)}
        
        for cam_file in sorted(images_dir.iterdir()):
            if not cam_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                continue
                
            try:
                img = Image.open(cam_file)
            except Exception:
                continue
            
            # Handle different image modes
            if img.mode == "RGBA":
                alpha = np.array(img)[:, :, 3]
            elif img.mode == "LA":
                alpha = np.array(img)[:, :, 1]
            else:
                # No alpha channel - check if mask file exists
                mask_dir = sample_path / "masks"
                mask_path = mask_dir / cam_file.name if mask_dir.exists() else None
                if mask_path and mask_path.exists():
                    alpha = np.array(Image.open(mask_path).convert("L"))
                else:
                    continue
            
            fg = alpha > self.alpha_threshold
            
            # Check edges
            touching = []
            if fg[0, :].any(): touching.append("top")
            if fg[-1, :].any(): touching.append("bottom")
            if fg[:, 0].any(): touching.append("left")
            if fg[:, -1].any(): touching.append("right")
            
            if touching:
                sample_result["cams"].append({
                    "cam": cam_file.name,
                    "edges": touching,
                    "fg_ratio": float(fg.sum() / fg.size),
                })
        
        return sample_result if sample_result["cams"] else None
    
    def analyze_dataset(self, sample_rate: int = 1) -> list:
        """Analyze entire dataset.
        
        Args:
            sample_rate: Check every Nth sample (1 = all, 100 = every 100th)
        """
        all_samples = self._get_sample_dirs()
        samples_to_check = all_samples[::sample_rate]
        
        results = []
        for sample_id, sample_path in tqdm(samples_to_check, desc="Analyzing clipping"):
            result = self.analyze_sample(sample_id, sample_path)
            if result:
                results.append(result)
        
        return results
    
    def save_report(self, results: list) -> Path:
        """Save analysis results to JSON."""
        report_path = self.output_path / "clipping_report.json"
        
        # Calculate statistics
        total_samples_checked = len(self._get_sample_dirs())
        total_clipped_cams = sum(len(r["cams"]) for r in results)
        
        # Edge statistics
        edge_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        for r in results:
            for c in r["cams"]:
                for e in c["edges"]:
                    edge_counts[e] += 1
        
        report = {
            "summary": {
                "dataset": str(self.dataset_path),
                "structure": self.structure,
                "total_samples_in_dataset": total_samples_checked,
                "clipped_samples": len(results),
                "clipped_ratio": f"{len(results)/total_samples_checked*100:.1f}%" if total_samples_checked > 0 else "N/A",
                "total_clipped_cameras": total_clipped_cams,
                "edge_breakdown": edge_counts,
            },
            "samples": results,
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def copy_samples(self, results: list, max_samples: int = 10) -> None:
        """Copy clipped samples for visual inspection."""
        for result in results[:max_samples]:
            sample_path = Path(result["path"])
            sample_id = result["id"].replace("/", "_")  # train/000001 -> train_000001
            
            images_dir = sample_path / "images"
            if not images_dir.exists():
                images_dir = sample_path
                
            dst_dir = self.output_path / sample_id
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            for cam_file in images_dir.iterdir():
                if cam_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    img = Image.open(cam_file)
                    img.save(dst_dir / cam_file.name)
    
    def generate_html_viewer(self, results: list, max_samples: int = 20) -> Path:
        """Generate HTML viewer for visual inspection."""
        
        def img_to_base64(img_path: Path) -> str:
            img = Image.open(img_path)
            # Convert RGBA to RGB with white background for display
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (40, 40, 40))
                background.paste(img, mask=img.split()[3])
                img = background
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        # Build lookup
        clipped_lookup = {}
        for r in results:
            sample_id = r["id"].replace("/", "_")
            clipped_lookup[sample_id] = {c["cam"]: c["edges"] for c in r["cams"]}
        
        html = f'''<!DOCTYPE html>
<html>
<head>
<title>Clipping Analysis - {self.dataset_path.name}</title>
<style>
body {{ font-family: Arial; margin: 20px; background: #1a1a1a; color: #fff; }}
h1 {{ color: #4CAF50; }}
.summary {{ background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
.summary table {{ width: 100%; border-collapse: collapse; }}
.summary td {{ padding: 5px 10px; }}
.summary td:first-child {{ font-weight: bold; width: 200px; }}
.sample {{ margin: 20px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; }}
.sample-header {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
.images {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.cam {{ text-align: center; }}
.cam img {{ width: 180px; height: 180px; border: 2px solid #333; object-fit: contain; background: #282828; }}
.cam.clipped img {{ border-color: #f44336; }}
.cam-label {{ font-size: 11px; margin-top: 5px; }}
.clipped .cam-label {{ color: #f44336; font-weight: bold; }}
</style>
</head>
<body>
<h1>Clipping Analysis: {self.dataset_path.name}</h1>
<div class="summary">
<table>
<tr><td>Dataset</td><td>{self.dataset_path}</td></tr>
<tr><td>Structure</td><td>{self.structure}</td></tr>
<tr><td>Clipped samples</td><td>{len(results)}</td></tr>
<tr><td>Alpha threshold</td><td>{self.alpha_threshold}</td></tr>
</table>
</div>
'''
        
        # Get copied sample dirs
        sample_dirs = sorted([d for d in self.output_path.iterdir() 
                            if d.is_dir()])[:max_samples]
        
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            clipped_cams = clipped_lookup.get(sample_id, {})
            
            html += f'<div class="sample"><div class="sample-header">{sample_id}</div><div class="images">'
            
            for img_file in sorted(sample_dir.iterdir()):
                if not img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    continue
                    
                img_b64 = img_to_base64(img_file)
                is_clipped = img_file.name in clipped_cams
                edges = ", ".join(clipped_cams.get(img_file.name, []))
                
                css_class = "cam clipped" if is_clipped else "cam"
                label = f"{img_file.name}<br/>CLIPPED ({edges})" if is_clipped else img_file.name
                
                html += f'<div class="{css_class}"><img src="data:image/png;base64,{img_b64}"/><div class="cam-label">{label}</div></div>'
            
            html += '</div></div>'
        
        html += '</body></html>'
        
        viewer_path = self.output_path / "viewer.html"
        with open(viewer_path, "w") as f:
            f.write(html)
        
        return viewer_path
    
    def print_summary(self, results: list) -> None:
        """Print summary to console."""
        total = len(self._get_sample_dirs())
        clipped = len(results)
        ratio = clipped / total * 100 if total > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"Dataset: {self.dataset_path.name}")
        print(f"Structure: {self.structure}")
        print(f"Total samples: {total}")
        print(f"Clipped samples: {clipped} ({ratio:.1f}%)")
        
        if results:
            edge_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
            for r in results:
                for c in r["cams"]:
                    for e in c["edges"]:
                        edge_counts[e] += 1
            print(f"Edge breakdown: {edge_counts}")
        print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Analyze foreground clipping in datasets")
    parser.add_argument("dataset_path", help="Path to preprocessed dataset")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--sample-rate", "-r", type=int, default=100, 
                       help="Check every Nth sample (default: 100)")
    parser.add_argument("--alpha-threshold", "-t", type=int, default=127,
                       help="Alpha threshold for foreground (default: 127)")
    parser.add_argument("--max-samples", "-m", type=int, default=10,
                       help="Max samples to copy for viewer (default: 10)")
    
    args = parser.parse_args()
    
    analyzer = ClippingAnalyzer(
        args.dataset_path,
        args.output,
        args.alpha_threshold,
    )
    
    print(f"Analyzing: {args.dataset_path} (structure: {analyzer.structure})")
    results = analyzer.analyze_dataset(args.sample_rate)
    
    report_path = analyzer.save_report(results)
    print(f"Report saved: {report_path}")
    
    analyzer.copy_samples(results, args.max_samples)
    
    viewer_path = analyzer.generate_html_viewer(results, args.max_samples)
    print(f"HTML viewer: {viewer_path}")
    
    analyzer.print_summary(results)


if __name__ == "__main__":
    main()
