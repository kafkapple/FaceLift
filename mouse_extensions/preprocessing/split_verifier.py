#!/usr/bin/env python3
"""
Split Verifier: Analyze train/val split characteristics.

Detects potential issues:
1. Temporal clustering (val samples from specific time periods)
2. Distribution mismatch (camera params, image stats)
3. Data leakage indicators

Usage:
    python -m mouse_extensions.preprocessing.split_verifier \
        --data-dir /path/to/dataset \
        --output split_report.json
    
    # Compare multiple datasets
    python -m mouse_extensions.preprocessing.split_verifier \
        --data-dirs D3_normalized D7_1 D8 M3 \
        --output comparison_report.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import cv2


@dataclass
class SplitStats:
    """Statistics for a single split."""
    name: str
    count: int
    # Camera stats
    cam_trans_mean: float
    cam_trans_std: float
    cam_fx_mean: float
    cam_fx_std: float
    cam_fy_mean: float
    cam_fy_std: float
    # Image stats
    img_brightness_mean: float
    img_brightness_std: float
    img_contrast_mean: float
    img_contrast_std: float
    # Sample indices (if available)
    sample_indices: Optional[List[int]] = None


@dataclass 
class DatasetReport:
    """Full report for a dataset."""
    dataset_name: str
    dataset_path: str
    total_samples: int
    train_stats: SplitStats
    val_stats: SplitStats
    # Comparison metrics
    trans_diff_pct: float  # % difference in translation
    brightness_diff_pct: float
    contrast_diff_pct: float
    fx_fy_ratio_train: float
    fx_fy_ratio_val: float
    # Warnings
    warnings: List[str]


def load_camera_params(sample_dir: Path) -> Optional[Dict]:
    """Load camera parameters from sample directory."""
    cam_file = sample_dir / "opencv_cameras.json"
    if not cam_file.exists():
        cam_file = sample_dir / "cameras.json"
    if not cam_file.exists():
        return None
    
    try:
        with open(cam_file) as f:
            data = json.load(f)
        return data
    except:
        return None


def extract_camera_stats(sample_dir: Path) -> Optional[Dict]:
    """Extract camera statistics from a sample."""
    data = load_camera_params(sample_dir)
    if data is None:
        return None
    
    frames = data.get("frames", [])
    if not frames:
        return None
    
    translations = []
    fxs, fys = [], []
    
    for frame in frames:
        # Extract translation from w2c matrix
        w2c = frame.get("w2c", [])
        if len(w2c) >= 3:
            t = [w2c[i][3] for i in range(3)]
            translations.append(np.linalg.norm(t))
        
        fxs.append(frame.get("fx", 0))
        fys.append(frame.get("fy", 0))
    
    return {
        "trans": np.mean(translations) if translations else 0,
        "fx": np.mean(fxs),
        "fy": np.mean(fys),
    }


def extract_image_stats(sample_dir: Path, max_images: int = 2) -> Optional[Dict]:
    """Extract image statistics from a sample."""
    img_dir = sample_dir / "images"
    if not img_dir.exists():
        return None
    
    images = list(img_dir.glob("*.png"))[:max_images]
    if not images:
        images = list(img_dir.glob("*.jpg"))[:max_images]
    
    if not images:
        return None
    
    brightnesses, contrasts = [], []
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            brightnesses.append(float(img.mean()))
            contrasts.append(float(img.std()))
    
    if not brightnesses:
        return None
    
    return {
        "brightness": np.mean(brightnesses),
        "contrast": np.mean(contrasts),
    }


def analyze_split(split_dir: Path, split_name: str, max_samples: int = 500) -> SplitStats:
    """Analyze a single split (train or val)."""
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")
    
    samples = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    
    # Limit samples for speed
    if len(samples) > max_samples:
        step = len(samples) // max_samples
        samples = samples[::step][:max_samples]
    
    trans_list, fx_list, fy_list = [], [], []
    brightness_list, contrast_list = [], []
    sample_indices = []
    
    for sample_dir in tqdm(samples, desc=f"Analyzing {split_name}", leave=False):
        # Try to extract sample index from name
        try:
            idx = int(sample_dir.name)
            sample_indices.append(idx)
        except:
            pass
        
        # Camera stats
        cam_stats = extract_camera_stats(sample_dir)
        if cam_stats:
            trans_list.append(cam_stats["trans"])
            fx_list.append(cam_stats["fx"])
            fy_list.append(cam_stats["fy"])
        
        # Image stats
        img_stats = extract_image_stats(sample_dir)
        if img_stats:
            brightness_list.append(img_stats["brightness"])
            contrast_list.append(img_stats["contrast"])
    
    return SplitStats(
        name=split_name,
        count=len(list(split_dir.iterdir())),
        cam_trans_mean=float(np.mean(trans_list)) if trans_list else 0,
        cam_trans_std=float(np.std(trans_list)) if trans_list else 0,
        cam_fx_mean=float(np.mean(fx_list)) if fx_list else 0,
        cam_fx_std=float(np.std(fx_list)) if fx_list else 0,
        cam_fy_mean=float(np.mean(fy_list)) if fy_list else 0,
        cam_fy_std=float(np.std(fy_list)) if fy_list else 0,
        img_brightness_mean=float(np.mean(brightness_list)) if brightness_list else 0,
        img_brightness_std=float(np.std(brightness_list)) if brightness_list else 0,
        img_contrast_mean=float(np.mean(contrast_list)) if contrast_list else 0,
        img_contrast_std=float(np.std(contrast_list)) if contrast_list else 0,
        sample_indices=sample_indices if sample_indices else None,
    )


def pct_diff(a: float, b: float) -> float:
    """Calculate percentage difference."""
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b)) * 100


def analyze_dataset(data_dir: Path, max_samples: int = 500) -> DatasetReport:
    """Analyze a complete dataset."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    # Also check samples/ structure
    if not train_dir.exists():
        samples_dir = data_dir / "samples"
        if samples_dir.exists():
            # Need to parse split files
            raise NotImplementedError("samples/ structure not yet supported")
    
    print(f"\nAnalyzing: {data_dir.name}")
    print("=" * 50)
    
    train_stats = analyze_split(train_dir, "train", max_samples)
    val_stats = analyze_split(val_dir, "val", max_samples)
    
    # Calculate comparison metrics
    trans_diff = pct_diff(train_stats.cam_trans_mean, val_stats.cam_trans_mean)
    brightness_diff = pct_diff(train_stats.img_brightness_mean, val_stats.img_brightness_mean)
    contrast_diff = pct_diff(train_stats.img_contrast_mean, val_stats.img_contrast_mean)
    
    fx_fy_train = train_stats.cam_fx_mean / train_stats.cam_fy_mean if train_stats.cam_fy_mean else 1
    fx_fy_val = val_stats.cam_fx_mean / val_stats.cam_fy_mean if val_stats.cam_fy_mean else 1
    
    # Generate warnings
    warnings = []
    
    if trans_diff > 5:
        warnings.append(f"Camera translation differs by {trans_diff:.1f}% between train/val")
    
    if brightness_diff > 10:
        warnings.append(f"Image brightness differs by {brightness_diff:.1f}% between train/val")
    
    if contrast_diff > 10:
        warnings.append(f"Image contrast differs by {contrast_diff:.1f}% between train/val")
    
    if abs(fx_fy_train - 1.0) > 0.01:
        warnings.append(f"fx/fy ratio != 1.0 in train: {fx_fy_train:.4f}")
    
    if abs(fx_fy_val - 1.0) > 0.01:
        warnings.append(f"fx/fy ratio != 1.0 in val: {fx_fy_val:.4f}")
    
    # Check for small differences (potential "easy val")
    if trans_diff < 1 and brightness_diff < 2 and contrast_diff < 2:
        if train_stats.count > 100:  # Only warn for substantial datasets
            warnings.append("⚠️ Train/Val distributions very similar - possible easy val or data leakage")
    
    return DatasetReport(
        dataset_name=data_dir.name,
        dataset_path=str(data_dir),
        total_samples=train_stats.count + val_stats.count,
        train_stats=train_stats,
        val_stats=val_stats,
        trans_diff_pct=trans_diff,
        brightness_diff_pct=brightness_diff,
        contrast_diff_pct=contrast_diff,
        fx_fy_ratio_train=fx_fy_train,
        fx_fy_ratio_val=fx_fy_val,
        warnings=warnings,
    )


def print_report(report: DatasetReport):
    """Print formatted report."""
    print(f"\n{'='*60}")
    print(f"Dataset: {report.dataset_name}")
    print(f"{'='*60}")
    print(f"Total samples: {report.total_samples} (train: {report.train_stats.count}, val: {report.val_stats.count})")
    print(f"\nCamera Parameters:")
    print(f"  {'Metric':<20} {'Train':>12} {'Val':>12} {'Diff%':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Translation mean':<20} {report.train_stats.cam_trans_mean:>12.3f} {report.val_stats.cam_trans_mean:>12.3f} {report.trans_diff_pct:>9.1f}%")
    print(f"  {'fx mean':<20} {report.train_stats.cam_fx_mean:>12.1f} {report.val_stats.cam_fx_mean:>12.1f}")
    print(f"  {'fy mean':<20} {report.train_stats.cam_fy_mean:>12.1f} {report.val_stats.cam_fy_mean:>12.1f}")
    print(f"  {'fx/fy ratio':<20} {report.fx_fy_ratio_train:>12.4f} {report.fx_fy_ratio_val:>12.4f}")
    
    print(f"\nImage Statistics:")
    print(f"  {'Metric':<20} {'Train':>12} {'Val':>12} {'Diff%':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Brightness mean':<20} {report.train_stats.img_brightness_mean:>12.1f} {report.val_stats.img_brightness_mean:>12.1f} {report.brightness_diff_pct:>9.1f}%")
    print(f"  {'Contrast mean':<20} {report.train_stats.img_contrast_mean:>12.1f} {report.val_stats.img_contrast_mean:>12.1f} {report.contrast_diff_pct:>9.1f}%")
    
    if report.warnings:
        print(f"\n⚠️  Warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    else:
        print(f"\n✅ No issues detected")


def print_comparison_table(reports: List[DatasetReport]):
    """Print comparison table for multiple datasets."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Total':>8} {'Trans%':>8} {'Bright%':>8} {'Contr%':>8} {'fx/fy':>8} {'Issues':>8}")
    print(f"{'-'*80}")
    
    for r in reports:
        issues = len(r.warnings)
        issue_str = f"{issues} ⚠️" if issues > 0 else "✅"
        print(f"{r.dataset_name:<20} {r.total_samples:>8} {r.trans_diff_pct:>7.1f}% {r.brightness_diff_pct:>7.1f}% {r.contrast_diff_pct:>7.1f}% {r.fx_fy_ratio_train:>8.4f} {issue_str:>8}")


def main():
    parser = argparse.ArgumentParser(description="Split Verifier")
    parser.add_argument("--data-dir", type=Path, help="Single dataset directory")
    parser.add_argument("--data-dirs", nargs="+", help="Multiple dataset names (relative to base)")
    parser.add_argument("--base-dir", type=Path, 
                       default=Path("/home/joon/data/preprocessed/FaceLift_mouse"),
                       help="Base directory for datasets")
    parser.add_argument("--output", type=Path, help="Output JSON report")
    parser.add_argument("--max-samples", type=int, default=300, help="Max samples to analyze per split")
    
    args = parser.parse_args()
    
    reports = []
    
    if args.data_dir:
        report = analyze_dataset(args.data_dir, args.max_samples)
        reports.append(report)
        print_report(report)
    
    elif args.data_dirs:
        for name in args.data_dirs:
            data_dir = args.base_dir / name
            if not data_dir.exists():
                print(f"⚠️ Dataset not found: {data_dir}")
                continue
            try:
                report = analyze_dataset(data_dir, args.max_samples)
                reports.append(report)
                print_report(report)
            except Exception as e:
                print(f"❌ Error analyzing {name}: {e}")
    
    if len(reports) > 1:
        print_comparison_table(reports)
    
    if args.output and reports:
        output_data = {
            "reports": [asdict(r) for r in reports],
        }
        # Remove sample_indices from output (too large)
        for r in output_data["reports"]:
            r["train_stats"].pop("sample_indices", None)
            r["val_stats"].pop("sample_indices", None)
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
