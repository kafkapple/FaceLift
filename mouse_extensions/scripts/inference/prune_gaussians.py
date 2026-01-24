#!/usr/bin/env python3
"""
Simple Gaussian Pruning - 낮은 opacity Gaussian 제거

Usage:
    python -m mouse_extensions.scripts.inference.prune_gaussians \
        --input gaussians.ply \
        --output gaussians_pruned.ply \
        --opacity_threshold 0.1
"""

import argparse
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement


def prune_gaussians(
    input_path: str,
    output_path: str,
    opacity_threshold: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Remove Gaussians with opacity below threshold.
    
    Args:
        input_path: Input .ply file
        output_path: Output .ply file
        opacity_threshold: Remove if opacity < threshold
        verbose: Print statistics
        
    Returns:
        dict with pruning statistics
    """
    # Load PLY
    plydata = PlyData.read(input_path)
    vertex = plydata["vertex"]
    
    # Get opacity (stored as DC component or direct)
    if "opacity" in vertex.data.dtype.names:
        opacity = vertex["opacity"]
    elif "f_dc_0" in vertex.data.dtype.names:
        # GS-LRM format: opacity might be in different field
        # Try to find opacity-like field
        opacity = None
        for name in vertex.data.dtype.names:
            if "opac" in name.lower():
                opacity = vertex[name]
                break
        if opacity is None:
            # Assume all are valid
            opacity = np.ones(len(vertex.data))
    else:
        opacity = np.ones(len(vertex.data))
    
    # Sigmoid if raw logits
    if opacity.min() < 0 or opacity.max() > 1:
        opacity = 1 / (1 + np.exp(-opacity))
    
    # Create mask
    keep_mask = opacity >= opacity_threshold
    
    # Statistics
    n_original = len(vertex.data)
    n_kept = keep_mask.sum()
    n_pruned = n_original - n_kept
    
    stats = {
        "original": n_original,
        "kept": int(n_kept),
        "pruned": int(n_pruned),
        "prune_ratio": n_pruned / n_original if n_original > 0 else 0,
        "opacity_threshold": opacity_threshold,
    }
    
    if verbose:
        print(f"Original: {n_original:,} Gaussians")
        print(f"Kept: {n_kept:,} ({n_kept/n_original*100:.1f}%)")
        print(f"Pruned: {n_pruned:,} ({n_pruned/n_original*100:.1f}%)")
        print(f"Threshold: {opacity_threshold}")
    
    # Filter vertex data
    filtered_data = vertex.data[keep_mask]
    
    # Create new PLY
    new_vertex = PlyElement.describe(filtered_data, "vertex")
    new_plydata = PlyData([new_vertex])
    new_plydata.write(output_path)
    
    if verbose:
        print(f"Saved: {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prune low-opacity Gaussians")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .ply")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .ply")
    parser.add_argument("--opacity_threshold", "-t", type=float, default=0.1)
    args = parser.parse_args()
    
    if args.output is None:
        p = Path(args.input)
        args.output = str(p.parent / f"{p.stem}_pruned{p.suffix}")
    
    prune_gaussians(args.input, args.output, args.opacity_threshold)


if __name__ == "__main__":
    main()
