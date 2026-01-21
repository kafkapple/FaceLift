#!/usr/bin/env python3
"""
Analyze Camera Parameter Consistency Across Frames
===================================================

Checks if opencv_cameras.json values are identical across all samples
in a mouse dataset (since cameras are physically fixed).

Usage:
    python -m mouse_extensions.scripts.analyze_camera_consistency \
        --dataset_dir /path/to/D7_1 \
        --num_samples 100
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_camera_json(sample_path):
    """Load opencv_cameras.json from a sample directory."""
    json_path = os.path.join(sample_path, "opencv_cameras.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def extract_camera_params(data):
    """Extract key camera parameters for comparison."""
    params = []
    for frame in data["frames"]:
        params.append({
            "fx": frame["fx"],
            "fy": frame["fy"],
            "cx": frame["cx"],
            "cy": frame["cy"],
            "w2c": np.array(frame["w2c"]),
            "view_id": frame.get("view_id", len(params))
        })
    return params


def compare_params(params1, params2):
    """Compare two sets of camera parameters."""
    diffs = []
    for p1, p2 in zip(params1, params2):
        diff = {
            "view_id": p1["view_id"],
            "fx_diff": abs(p1["fx"] - p2["fx"]),
            "fy_diff": abs(p1["fy"] - p2["fy"]),
            "cx_diff": abs(p1["cx"] - p2["cx"]),
            "cy_diff": abs(p1["cy"] - p2["cy"]),
            "w2c_diff": np.max(np.abs(p1["w2c"] - p2["w2c"]))
        }
        diffs.append(diff)
    return diffs


def analyze_dataset(dataset_dir, num_samples=None, split="train"):
    """Analyze camera parameter consistency across samples."""
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.exists(split_dir):
        print("Split directory not found: {}".format(split_dir))
        return None
    
    samples = sorted([d for d in os.listdir(split_dir) 
                      if os.path.isdir(os.path.join(split_dir, d))])
    
    if num_samples:
        samples = samples[:num_samples]
    
    print("Analyzing {} samples from {}".format(len(samples), split_dir))
    
    # Load reference (first sample)
    ref_path = os.path.join(split_dir, samples[0])
    ref_data = load_camera_json(ref_path)
    if ref_data is None:
        print("Could not load reference sample")
        return None
    
    ref_params = extract_camera_params(ref_data)
    print("Reference sample: {}".format(samples[0]))
    print("Number of cameras: {}".format(len(ref_params)))
    
    # Compare all samples to reference
    results = {
        "reference": samples[0],
        "num_samples": len(samples),
        "num_cameras": len(ref_params),
        "differences": defaultdict(list),
        "identical_count": 0,
        "different_samples": []
    }
    
    # Track per-camera statistics
    stats = {cam["view_id"]: {
        "fx": [], "fy": [], "cx": [], "cy": [], "w2c_max": []
    } for cam in ref_params}
    
    for sample in samples[1:]:
        sample_path = os.path.join(split_dir, sample)
        data = load_camera_json(sample_path)
        if data is None:
            continue
        
        params = extract_camera_params(data)
        diffs = compare_params(ref_params, params)
        
        is_identical = True
        for diff in diffs:
            vid = diff["view_id"]
            stats[vid]["fx"].append(diff["fx_diff"])
            stats[vid]["fy"].append(diff["fy_diff"])
            stats[vid]["cx"].append(diff["cx_diff"])
            stats[vid]["cy"].append(diff["cy_diff"])
            stats[vid]["w2c_max"].append(diff["w2c_diff"])
            
            # Check if any difference exceeds threshold
            if any([diff["fx_diff"] > 1e-6, diff["fy_diff"] > 1e-6,
                    diff["cx_diff"] > 1e-6, diff["cy_diff"] > 1e-6,
                    diff["w2c_diff"] > 1e-6]):
                is_identical = False
        
        if is_identical:
            results["identical_count"] += 1
        else:
            results["different_samples"].append(sample)
    
    # Compute statistics
    results["per_camera_stats"] = {}
    for vid, s in stats.items():
        if len(s["fx"]) > 0:
            results["per_camera_stats"][vid] = {
                "fx": {"mean": np.mean(s["fx"]), "max": np.max(s["fx"]), "std": np.std(s["fx"])},
                "fy": {"mean": np.mean(s["fy"]), "max": np.max(s["fy"]), "std": np.std(s["fy"])},
                "cx": {"mean": np.mean(s["cx"]), "max": np.max(s["cx"]), "std": np.std(s["cx"])},
                "cy": {"mean": np.mean(s["cy"]), "max": np.max(s["cy"]), "std": np.std(s["cy"])},
                "w2c_max": {"mean": np.mean(s["w2c_max"]), "max": np.max(s["w2c_max"]), "std": np.std(s["w2c_max"])}
            }
    
    return results


def print_report(results):
    """Print analysis report."""
    print("\n" + "="*60)
    print("CAMERA CONSISTENCY ANALYSIS REPORT")
    print("="*60)
    
    print("\nDataset Statistics:")
    print("  Reference sample: {}".format(results["reference"]))
    print("  Total samples analyzed: {}".format(results["num_samples"]))
    print("  Number of cameras: {}".format(results["num_cameras"]))
    
    identical_pct = (results["identical_count"] / (results["num_samples"]-1)) * 100 if results["num_samples"] > 1 else 100
    print("\nConsistency Check:")
    print("  Identical to reference: {}/{} ({:.1f}%)".format(
        results["identical_count"], results["num_samples"]-1, identical_pct))
    
    if results["different_samples"]:
        print("  Different samples: {}".format(results["different_samples"][:10]))
        if len(results["different_samples"]) > 10:
            print("    ... and {} more".format(len(results["different_samples"]) - 10))
    
    print("\nPer-Camera Parameter Differences (vs reference):")
    print("-"*60)
    print("{:>6} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Cam", "fx_max", "fy_max", "cx_max", "cy_max", "w2c_max"))
    print("-"*60)
    
    for vid, stats in sorted(results["per_camera_stats"].items()):
        print("{:>6} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
            vid,
            stats["fx"]["max"],
            stats["fy"]["max"],
            stats["cx"]["max"],
            stats["cy"]["max"],
            stats["w2c_max"]["max"]
        ))
    
    print("-"*60)
    
    # Summary
    all_fx_max = max(s["fx"]["max"] for s in results["per_camera_stats"].values())
    all_w2c_max = max(s["w2c_max"]["max"] for s in results["per_camera_stats"].values())
    
    print("\nSummary:")
    if all_fx_max < 1e-6 and all_w2c_max < 1e-6:
        print("  ✓ All camera parameters are IDENTICAL across all samples")
        print("  → Cameras are truly fixed (no per-frame variation)")
    else:
        print("  ⚠ Camera parameters VARY across samples!")
        print("  → Max intrinsics diff: {:.6f}".format(all_fx_max))
        print("  → Max extrinsics diff: {:.6f}".format(all_w2c_max))
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Analyze camera parameter consistency")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset (e.g., D7_1)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to analyze")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (train/val)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    results = analyze_dataset(args.dataset_dir, args.num_samples, args.split)
    
    if results:
        print_report(results)
        
        if args.output:
            # Convert numpy types for JSON serialization
            output_data = {
                "reference": results["reference"],
                "num_samples": results["num_samples"],
                "num_cameras": results["num_cameras"],
                "identical_count": results["identical_count"],
                "different_samples": results["different_samples"][:20],
                "per_camera_stats": {
                    str(k): {pk: {sk: float(sv) for sk, sv in pv.items()} 
                             for pk, pv in v.items()}
                    for k, v in results["per_camera_stats"].items()
                }
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print("\nResults saved to: {}".format(args.output))


if __name__ == "__main__":
    main()
