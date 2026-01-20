#!/usr/bin/env python3
"""
Unified Report Generator for FaceLift Mouse
============================================

Generates comprehensive preprocessing and experiment reports.

Usage:
    # Full preprocessing comparison
    python unified_report_generator.py --type preprocessing \
        --datasets v13,D1,D7_1,D7_1_t \
        --output reports/09_unified/

    # Quick dataset summary
    python unified_report_generator.py --type summary --datasets D7_1_t

Author: Claude Code
Version: v1.0 (2026-01-21)
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np


@dataclass
class ReportConfig:
    """Report generation configuration."""
    title: str = "FaceLift Mouse Report"
    base_path: str = "/home/joon/data/preprocessed/FaceLift_mouse"
    output_dir: str = "/home/joon/dev/FaceLift/mouse_extensions/reports/09_unified"
    max_samples: int = 100


DATASET_REGISTRY = {
    "v13": {
        "name": "v13",
        "path": "data_mouse_v13_original_ratio",
        "description": "Legacy PP=256 forced (whitening baseline)",
        "status": "DEPRECATED",
        "ray_error": "5-13 deg",
        "issue": "PP forced causes ray direction error"
    },
    "D1": {
        "name": "D1",
        "path": "D1_pp_centered",
        "description": "PP-centered crop (ghosting baseline)",
        "status": "DEPRECATED",
        "ray_error": "~13 deg",
        "issue": "View-wise crop causes cross-view inconsistency"
    },
    "D4": {
        "name": "D4",
        "path": "D4",
        "description": "Triangulation center + PP=256 forced",
        "status": "DEPRECATED",
        "ray_error": "5-13 deg",
        "issue": "PP forced despite triangulation"
    },
    "D7": {
        "name": "D7",
        "path": "D7",
        "description": "PP-shift + fy=549 forced",
        "status": "CURRENT",
        "ray_error": "~0.4 deg",
        "issue": "Minor fy approximation"
    },
    "D7_1": {
        "name": "D7_1",
        "path": "D7_1",
        "description": "Individual scale (geometrically correct)",
        "status": "RECOMMENDED",
        "ray_error": "~0 deg",
        "issue": "None"
    },
    "D7_1_t": {
        "name": "D7_1_t",
        "path": "D7_1_t",
        "description": "D7.1 + temporal split",
        "status": "RECOMMENDED",
        "ray_error": "~0 deg",
        "issue": "None"
    },
    "D7_2": {
        "name": "D7_2",
        "path": "D7_2",
        "description": "Average scale (isotropic)",
        "status": "ALTERNATIVE",
        "ray_error": "~0.2 deg",
        "issue": "Minor fx/fy deviation"
    },
}


def load_dataset_info(dataset_name: str, base_path: str) -> Dict:
    """Load dataset information."""
    if dataset_name not in DATASET_REGISTRY:
        return {"error": f"Unknown dataset: {dataset_name}"}

    info = DATASET_REGISTRY[dataset_name].copy()
    dataset_path = Path(base_path) / info["path"]

    if not dataset_path.exists():
        info["exists"] = False
        return info

    info["exists"] = True
    info["full_path"] = str(dataset_path)

    # Load preprocessing info
    info_file = dataset_path / "preprocessing_info.json"
    if info_file.exists():
        with open(info_file) as f:
            info["preprocessing"] = json.load(f)

    # Count samples
    for split in ["train", "val", "test"]:
        txt_file = dataset_path / f"data_mouse_{split}.txt"
        if txt_file.exists():
            with open(txt_file) as f:
                info[f"{split}_samples"] = len(f.readlines())

    return info


def load_camera_stats(dataset_name: str, base_path: str, max_samples: int = 50) -> Dict:
    """Load camera parameter statistics."""
    if dataset_name not in DATASET_REGISTRY:
        return {}

    dataset_path = Path(base_path) / DATASET_REGISTRY[dataset_name]["path"]

    fx_values, fy_values, cx_values, cy_values = [], [], [], []

    for split in ["train", "val"]:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue

        for i, sample_dir in enumerate(sorted(split_dir.iterdir())):
            if i >= max_samples:
                break

            cam_file = sample_dir / "opencv_cameras.json"
            if not cam_file.exists():
                continue

            with open(cam_file) as f:
                data = json.load(f)

            frames = data.get("frames", data)
            for frame in frames:
                # Support both K matrix format and direct fx/fy/cx/cy format
                if "K" in frame:
                    K = np.array(frame["K"])
                    fx_values.append(K[0, 0])
                    fy_values.append(K[1, 1])
                    cx_values.append(K[0, 2])
                    cy_values.append(K[1, 2])
                elif "fx" in frame:
                    fx_values.append(frame["fx"])
                    fy_values.append(frame["fy"])
                    cx_values.append(frame["cx"])
                    cy_values.append(frame["cy"])

    if not fx_values:
        return {}

    return {
        "fx": {"mean": np.mean(fx_values), "std": np.std(fx_values)},
        "fy": {"mean": np.mean(fy_values), "std": np.std(fy_values)},
        "cx": {"mean": np.mean(cx_values), "std": np.std(cx_values)},
        "cy": {"mean": np.mean(cy_values), "std": np.std(cy_values)},
        "n_samples": len(fx_values)
    }


def generate_preprocessing_report(datasets: List[str], config: ReportConfig) -> str:
    """Generate preprocessing comparison report in Markdown."""
    date_str = datetime.now().strftime("%y%m%d")

    lines = [
        "# Preprocessing Comparison Report",
        "",
        f"> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> **Datasets**: {', '.join(datasets)}",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Dataset | Status | Ray Error | Train | Val | Test |",
        "|---------|--------|-----------|-------|-----|------|",
    ]

    all_info = {}
    for ds in datasets:
        info = load_dataset_info(ds, config.base_path)
        all_info[ds] = info

        status = info.get("status", "UNKNOWN")
        ray_error = info.get("ray_error", "?")
        train = info.get("train_samples", "-")
        val = info.get("val_samples", "-")
        test = info.get("test_samples", "-")

        lines.append(f"| {ds} | {status} | {ray_error} | {train} | {val} | {test} |")

    lines.extend([
        "",
        "---",
        "",
        "## Camera Parameter Statistics",
        "",
        "| Dataset | fx | fy | cx | cy |",
        "|---------|-----|-----|-----|-----|",
    ])

    for ds in datasets:
        stats = load_camera_stats(ds, config.base_path)
        if not stats:
            lines.append(f"| {ds} | N/A | N/A | N/A | N/A |")
            continue

        fx = f"{stats['fx']['mean']:.1f}+/-{stats['fx']['std']:.1f}"
        fy = f"{stats['fy']['mean']:.1f}+/-{stats['fy']['std']:.1f}"
        cx = f"{stats['cx']['mean']:.1f}+/-{stats['cx']['std']:.1f}"
        cy = f"{stats['cy']['mean']:.1f}+/-{stats['cy']['std']:.1f}"

        lines.append(f"| {ds} | {fx} | {fy} | {cx} | {cy} |")

    lines.extend([
        "",
        "---",
        "",
        "## Dataset Details",
        "",
    ])

    for ds in datasets:
        info = all_info[ds]
        full_path = info.get('full_path', 'N/A')
        lines.extend([
            f"### {ds}",
            "",
            f"- **Description**: {info.get('description', 'N/A')}",
            f"- **Status**: {info.get('status', 'UNKNOWN')}",
            f"- **Ray Error**: {info.get('ray_error', '?')}",
            f"- **Issue**: {info.get('issue', 'N/A')}",
            f"- **Path**: `{full_path}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Generated by unified_report_generator.py*",
    ])

    return "\n".join(lines)


def generate_summary_report(datasets: List[str], config: ReportConfig) -> str:
    """Generate quick summary."""
    lines = ["# Quick Dataset Summary", ""]

    for ds in datasets:
        info = load_dataset_info(ds, config.base_path)
        lines.extend([
            f"## {ds}",
            f"- Status: {info.get('status', 'UNKNOWN')}",
            f"- Samples: train={info.get('train_samples', '?')}, val={info.get('val_samples', '?')}",
            f"- Ray Error: {info.get('ray_error', '?')}",
            "",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate FaceLift Mouse reports")
    parser.add_argument("--type", choices=["preprocessing", "experiment", "summary"],
                        default="preprocessing", help="Report type")
    parser.add_argument("--datasets", type=str, default="D7_1,D7_1_t",
                        help="Comma-separated dataset names")
    parser.add_argument("--output", type=str,
                        default="/home/joon/dev/FaceLift/mouse_extensions/reports/09_unified",
                        help="Output directory")
    parser.add_argument("--base-path", type=str,
                        default="/home/joon/data/preprocessed/FaceLift_mouse",
                        help="Base path for datasets")

    args = parser.parse_args()

    config = ReportConfig(
        output_dir=args.output,
        base_path=args.base_path
    )

    datasets = [d.strip() for d in args.datasets.split(",")]

    # Generate report
    if args.type == "preprocessing":
        report = generate_preprocessing_report(datasets, config)
        filename = f"{datetime.now().strftime('%y%m%d')}_preprocessing_comparison.md"
    elif args.type == "summary":
        report = generate_summary_report(datasets, config)
        filename = f"{datetime.now().strftime('%y%m%d')}_summary.md"
    else:
        print(f"Report type '{args.type}' not yet implemented")
        return

    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = Path(config.output_dir) / filename
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
