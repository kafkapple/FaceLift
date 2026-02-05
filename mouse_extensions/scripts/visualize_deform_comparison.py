#!/usr/bin/env python3
"""
Visualize V1 vs V2 Deformation Comparison.

Creates:
1. Drift over time plot (V1 cumulative vs V2 bounded)
2. Simulated position trajectory visualization
3. Frame-by-frame comparison if real data available

Usage:
    python -m mouse_extensions.scripts.visualize_deform_comparison \
        --output_dir outputs/deform_comparison \
        --num_frames 50
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[3]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_v1_v2_comparison(num_frames: int = 50, num_points: int = 100, seed: int = 42):
    """
    Simulate V1 (autoregressive) vs V2 (per-frame reference) behavior.
    
    V1: Each frame uses previous OUTPUT → error accumulates
    V2: Each frame uses original REFERENCE → error bounded
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Ground truth trajectory: smooth motion
    t = torch.linspace(0, 2 * np.pi, num_frames)
    
    # Base positions (centroid motion)
    base_x = torch.sin(t) * 0.5
    base_y = torch.cos(t) * 0.3
    base_z = t / (2 * np.pi) * 0.2
    
    # Per-frame Gaussians (with small jitter simulating GS-LRM output)
    original_positions = []
    for i in range(num_frames):
        # Random point cloud around base position
        points = torch.randn(num_points, 3) * 0.1
        points[:, 0] += base_x[i]
        points[:, 1] += base_y[i]
        points[:, 2] += base_z[i]
        # Add per-frame jitter (simulating GS-LRM reconstruction noise)
        points += torch.randn_like(points) * 0.02
        original_positions.append(points)
    
    # V1: Autoregressive (uses previous output)
    v1_positions = [original_positions[0].clone()]
    v1_drift_per_step = torch.randn(num_points, 3) * 0.01  # Small consistent drift
    
    for i in range(1, num_frames):
        # V1 uses PREVIOUS OUTPUT, adds drift
        prev = v1_positions[-1]
        new_pos = prev + v1_drift_per_step  # Cumulative!
        v1_positions.append(new_pos)
    
    # V2: Per-frame reference (uses original)
    v2_positions = []
    
    for i in range(num_frames):
        if i < num_frames - 1:
            # V2 uses ORIGINAL next frame as reference
            current = original_positions[i]
            reference = original_positions[i + 1]
            
            # Deformation toward reference (bounded)
            direction = reference - current
            delta = direction * 0.3  # Bounded step (30% toward reference)
            new_pos = current + delta
        else:
            new_pos = original_positions[i].clone()
        
        v2_positions.append(new_pos)
    
    # Compute drift from original
    v1_drifts = []
    v2_drifts = []
    
    for i in range(num_frames):
        v1_drift = (original_positions[i] - v1_positions[i]).norm(dim=-1).mean().item()
        v2_drift = (original_positions[i] - v2_positions[i]).norm(dim=-1).mean().item()
        v1_drifts.append(v1_drift)
        v2_drifts.append(v2_drift)
    
    return {
        'original': original_positions,
        'v1': v1_positions,
        'v2': v2_positions,
        'v1_drifts': v1_drifts,
        'v2_drifts': v2_drifts,
        'base_trajectory': (base_x, base_y, base_z),
    }


def plot_drift_comparison(results: dict, output_path: str):
    """Plot drift over time for V1 vs V2."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    frames = range(len(results['v1_drifts']))
    
    # Plot 1: Drift over time
    ax1 = axes[0]
    ax1.plot(frames, results['v1_drifts'], 'r-', linewidth=2, label='V1 (Autoregressive)', marker='o', markersize=3)
    ax1.plot(frames, results['v2_drifts'], 'g-', linewidth=2, label='V2 (Per-frame Ref)', marker='s', markersize=3)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Drift from Original', fontsize=12)
    ax1.set_title('Drift Accumulation: V1 vs V2', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(frames) - 1)
    
    # Annotate final values
    ax1.annotate(f'V1: {results["v1_drifts"][-1]:.3f}', 
                xy=(len(frames)-1, results['v1_drifts'][-1]),
                xytext=(len(frames)*0.7, results['v1_drifts'][-1]*0.8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax1.annotate(f'V2: {results["v2_drifts"][-1]:.3f}', 
                xy=(len(frames)-1, results['v2_drifts'][-1]),
                xytext=(len(frames)*0.7, max(results['v2_drifts'])*2),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    # Plot 2: Centroid trajectory
    ax2 = axes[1]
    
    # Get centroids
    orig_centroids = torch.stack([p.mean(dim=0) for p in results['original']])
    v1_centroids = torch.stack([p.mean(dim=0) for p in results['v1']])
    v2_centroids = torch.stack([p.mean(dim=0) for p in results['v2']])
    
    ax2.plot(orig_centroids[:, 0], orig_centroids[:, 1], 'b-', linewidth=2, label='Original (GS-LRM)', alpha=0.7)
    ax2.plot(v1_centroids[:, 0], v1_centroids[:, 1], 'r--', linewidth=2, label='V1 (Drift)', alpha=0.7)
    ax2.plot(v2_centroids[:, 0], v2_centroids[:, 1], 'g-.', linewidth=2, label='V2 (Bounded)', alpha=0.7)
    
    # Mark start and end
    ax2.scatter([orig_centroids[0, 0]], [orig_centroids[0, 1]], c='blue', s=100, marker='o', zorder=5, label='Start')
    ax2.scatter([orig_centroids[-1, 0]], [orig_centroids[-1, 1]], c='blue', s=100, marker='X', zorder=5, label='End')
    ax2.scatter([v1_centroids[-1, 0]], [v1_centroids[-1, 1]], c='red', s=100, marker='X', zorder=5)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Centroid Trajectory (Top View)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved drift comparison plot to {output_path}")


def plot_3d_trajectories(results: dict, output_path: str):
    """Plot 3D trajectories."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    # Get centroids
    orig_centroids = torch.stack([p.mean(dim=0) for p in results['original']]).numpy()
    v1_centroids = torch.stack([p.mean(dim=0) for p in results['v1']]).numpy()
    v2_centroids = torch.stack([p.mean(dim=0) for p in results['v2']]).numpy()
    
    # Original vs V1
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(orig_centroids[:, 0], orig_centroids[:, 1], orig_centroids[:, 2], 
             'b-', linewidth=2, label='Original')
    ax1.plot(v1_centroids[:, 0], v1_centroids[:, 1], v1_centroids[:, 2], 
             'r--', linewidth=2, label='V1 (Drift)')
    ax1.set_title('V1: Autoregressive Drift', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Original vs V2
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(orig_centroids[:, 0], orig_centroids[:, 1], orig_centroids[:, 2], 
             'b-', linewidth=2, label='Original')
    ax2.plot(v2_centroids[:, 0], v2_centroids[:, 1], v2_centroids[:, 2], 
             'g--', linewidth=2, label='V2 (Bounded)')
    ax2.set_title('V2: Per-frame Reference', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved 3D trajectory plot to {output_path}")


def create_animation(results: dict, output_path: str, fps: int = 10):
    """Create animated comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    num_frames = len(results['original'])
    
    def update(frame):
        for ax in axes:
            ax.clear()
        
        # Original
        orig = results['original'][frame].numpy()
        axes[0].scatter(orig[:, 0], orig[:, 1], c='blue', s=1, alpha=0.5)
        axes[0].set_title(f'Original (GS-LRM)\nFrame {frame}', fontsize=12)
        axes[0].set_xlim(-1, 1)
        axes[0].set_ylim(-1, 1)
        
        # V1
        v1 = results['v1'][frame].numpy()
        axes[1].scatter(v1[:, 0], v1[:, 1], c='red', s=1, alpha=0.5)
        axes[1].set_title(f'V1 (Autoregressive)\nDrift: {results["v1_drifts"][frame]:.3f}', fontsize=12)
        axes[1].set_xlim(-1, 1)
        axes[1].set_ylim(-1, 1)
        
        # V2
        v2 = results['v2'][frame].numpy()
        axes[2].scatter(v2[:, 0], v2[:, 1], c='green', s=1, alpha=0.5)
        axes[2].set_title(f'V2 (Per-frame Ref)\nDrift: {results["v2_drifts"][frame]:.3f}', fontsize=12)
        axes[2].set_xlim(-1, 1)
        axes[2].set_ylim(-1, 1)
        
        return axes
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000//fps, blit=False)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close()
    
    logger.info(f"Saved animation to {output_path}")


def print_summary(results: dict):
    """Print numerical summary."""
    print("\n" + "=" * 60)
    print("DEFORMATION COMPARISON SUMMARY")
    print("=" * 60)
    
    v1_final = results['v1_drifts'][-1]
    v2_final = results['v2_drifts'][-1]
    v1_max = max(results['v1_drifts'])
    v2_max = max(results['v2_drifts'])
    
    print(f"\nFrames: {len(results['v1_drifts'])}")
    print(f"Points per frame: {results['original'][0].shape[0]}")
    
    print(f"\n{'Metric':<25} {'V1 (Autoregressive)':<20} {'V2 (Per-frame Ref)':<20}")
    print("-" * 65)
    print(f"{'Final Drift':<25} {v1_final:<20.4f} {v2_final:<20.4f}")
    print(f"{'Max Drift':<25} {v1_max:<20.4f} {v2_max:<20.4f}")
    print(f"{'Drift Ratio (V1/V2)':<25} {v1_final/max(v2_final, 1e-8):<20.1f}x")
    
    print(f"\n{'Pattern':<25} {'CUMULATIVE (↑)':<20} {'BOUNDED (→)':<20}")
    print("=" * 60)
    
    print("\nConclusion:")
    print("  V1: Error accumulates over time → 'melting' effect")
    print("  V2: Error bounded by original reference → stable output")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize V1 vs V2 Deformation")
    parser.add_argument("--output_dir", type=str, default="outputs/deform_comparison")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--create_animation", action="store_true", help="Create animated GIF")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate comparison
    logger.info(f"Simulating V1 vs V2 with {args.num_frames} frames, {args.num_points} points")
    results = simulate_v1_v2_comparison(args.num_frames, args.num_points)
    
    # Print summary
    print_summary(results)
    
    # Create visualizations
    plot_drift_comparison(results, str(output_dir / "drift_comparison.png"))
    plot_3d_trajectories(results, str(output_dir / "trajectory_3d.png"))
    
    if args.create_animation:
        create_animation(results, str(output_dir / "animation.gif"))
    
    logger.info(f"All visualizations saved to {output_dir}")
    print(f"\nVisualization files:")
    print(f"  - {output_dir}/drift_comparison.png")
    print(f"  - {output_dir}/trajectory_3d.png")
    if args.create_animation:
        print(f"  - {output_dir}/animation.gif")


if __name__ == "__main__":
    main()
