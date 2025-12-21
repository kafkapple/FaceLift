#!/usr/bin/env python3
"""
Camera Normalization Visualization Script

Visualizes the effect of camera normalization (Y-up alignment) on mouse data.
Generates comparison grids showing:
1. Camera positions before/after normalization
2. Camera orientations (arrows)
3. Sample images with camera info

Usage:
    python scripts/visualize_camera_normalization.py \
        --sample_dir data_mouse/sample_000000 \
        --output_dir outputs/camera_normalization_viz/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gslrm.data.mouse_dataset import normalize_cameras_to_y_up, normalize_cameras_to_z_up, normalize_camera_distance


def load_cameras(json_path: str):
    """Load camera data from opencv_cameras.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    cameras = []
    for frame in data["frames"]:
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        cameras.append({
            "c2w": c2w,
            "w2c": w2c,
            "fx": frame["fx"],
            "fy": frame["fy"],
            "cx": frame["cx"],
            "cy": frame["cy"],
            "file_path": frame["file_path"]
        })
    return cameras


def extract_camera_info(c2w_matrices):
    """Extract positions and directions from c2w matrices"""
    positions = []
    forward_dirs = []  # -Z in camera space
    up_dirs = []       # -Y in camera space (camera up)

    for c2w in c2w_matrices:
        pos = c2w[:3, 3]
        forward = -c2w[:3, 2]  # -Z axis (look direction)
        up = -c2w[:3, 1]       # -Y axis (up direction)

        positions.append(pos)
        forward_dirs.append(forward)
        up_dirs.append(up)

    return np.array(positions), np.array(forward_dirs), np.array(up_dirs)


def estimate_up_direction(c2w_matrices):
    """Estimate up direction using PCA (same as in mouse_dataset.py)"""
    positions = np.array([c2w[:3, 3] for c2w in c2w_matrices])
    center = np.mean(positions, axis=0)
    centered = positions - center

    # PCA to find orbit plane normal
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    min_idx = np.argmin(eigenvalues.real)
    up_direction = eigenvectors[:, min_idx].real

    # Ensure consistent direction with camera ups
    avg_cam_up = np.mean([-c2w[:3, 1] for c2w in c2w_matrices], axis=0)
    if np.dot(up_direction, avg_cam_up) < 0:
        up_direction = -up_direction

    return up_direction / np.linalg.norm(up_direction)


def plot_cameras_3d(ax, positions, forward_dirs, up_dirs, title, color='blue'):
    """Plot camera positions and orientations in 3D"""
    # Plot camera positions with labels
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=color, s=150, marker='o', edgecolors='black', linewidths=1)

    # Add camera number labels
    for i, pos in enumerate(positions):
        ax.text(pos[0], pos[1], pos[2] + 0.15, f'C{i}', fontsize=12, fontweight='bold',
                ha='center', va='bottom')

    # Plot forward directions (red arrows) - where camera looks
    scale = 0.3
    for i, (pos, fwd) in enumerate(zip(positions, forward_dirs)):
        ax.quiver(pos[0], pos[1], pos[2],
                  fwd[0]*scale, fwd[1]*scale, fwd[2]*scale,
                  color='red', alpha=0.8, arrow_length_ratio=0.2, linewidth=2)

    # Plot up directions (green arrows) - camera's up direction
    for i, (pos, up) in enumerate(zip(positions, up_dirs)):
        ax.quiver(pos[0], pos[1], pos[2],
                  up[0]*scale, up[1]*scale, up[2]*scale,
                  color='green', alpha=0.8, arrow_length_ratio=0.2, linewidth=2)

    # Plot origin (subject location)
    ax.scatter([0], [0], [0], c='yellow', s=300, marker='*', edgecolors='black',
               linewidths=2, label='Subject', zorder=10)

    # Draw world coordinate axes with labels
    axis_len = 0.5
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='red', linewidth=3)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', linewidth=3)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='blue', linewidth=3)
    ax.text(axis_len+0.1, 0, 0, 'X', fontsize=14, fontweight='bold', color='red')
    ax.text(0, axis_len+0.1, 0, 'Y', fontsize=14, fontweight='bold', color='green')
    ax.text(0, 0, axis_len+0.1, 'Z', fontsize=14, fontweight='bold', color='blue')

    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Equal aspect ratio
    max_range = np.max(np.abs(positions)) * 1.3
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # Add legend for arrows
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Look direction'),
        Line2D([0], [0], color='green', linewidth=2, label='Camera Up'),
        Line2D([0], [0], marker='*', color='yellow', markersize=15,
               markeredgecolor='black', linestyle='None', label='Subject')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)


def create_comparison_figure(sample_dir: str, output_dir: str):
    """Create comparison visualization of camera normalization"""
    os.makedirs(output_dir, exist_ok=True)

    # Load cameras
    json_path = os.path.join(sample_dir, "opencv_cameras.json")
    cameras = load_cameras(json_path)

    # Get c2w matrices
    c2w_original = np.array([cam["c2w"] for cam in cameras])

    # Estimate and normalize (both Y-up and Z-up)
    up_direction = estimate_up_direction(c2w_original)
    c2w_yup = normalize_cameras_to_y_up(c2w_original.copy(), up_direction)
    c2w_zup = normalize_cameras_to_z_up(c2w_original.copy(), up_direction)

    # Also apply distance normalization
    c2w_yup = normalize_camera_distance(c2w_yup, target_distance=2.7)
    c2w_zup = normalize_camera_distance(c2w_zup, target_distance=2.7)
    c2w_normalized = c2w_zup  # Use Z-up as default (matches human data)

    # Extract camera info
    pos_orig, fwd_orig, up_orig = extract_camera_info(c2w_original)
    pos_norm, fwd_norm, up_norm = extract_camera_info(c2w_normalized)

    # ============ Figure 1: 3D Camera Comparison ============
    fig = plt.figure(figsize=(16, 8))

    # Original cameras
    ax1 = fig.add_subplot(121, projection='3d')
    plot_cameras_3d(ax1, pos_orig, fwd_orig, up_orig,
                    f'Original Cameras\nUp: [{up_direction[0]:.2f}, {up_direction[1]:.2f}, {up_direction[2]:.2f}]',
                    color='blue')

    # Normalized cameras
    ax2 = fig.add_subplot(122, projection='3d')
    plot_cameras_3d(ax2, pos_norm, fwd_norm, up_norm,
                    'Normalized Cameras (Y-up)',
                    color='orange')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'camera_positions_3d.png'), dpi=150)
    plt.close()

    # ============ Figure 2: Camera Statistics ============
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Position comparison
    for i, (label, data_orig, data_norm) in enumerate([
        ('X', pos_orig[:, 0], pos_norm[:, 0]),
        ('Y', pos_orig[:, 1], pos_norm[:, 1]),
        ('Z', pos_orig[:, 2], pos_norm[:, 2])
    ]):
        axes[0, i].bar(['Original', 'Normalized'],
                       [np.mean(data_orig), np.mean(data_norm)],
                       yerr=[np.std(data_orig), np.std(data_norm)],
                       capsize=5)
        axes[0, i].set_title(f'Camera {label} Position (mean ± std)')
        axes[0, i].set_ylabel(label)

    # Up direction comparison
    avg_up_orig = np.mean(up_orig, axis=0)
    avg_up_norm = np.mean(up_norm, axis=0)

    for i, (label, idx) in enumerate([('X', 0), ('Y', 1), ('Z', 2)]):
        axes[1, i].bar(['Original', 'Normalized'],
                       [avg_up_orig[idx], avg_up_norm[idx]])
        axes[1, i].set_title(f'Avg Camera Up {label}')
        axes[1, i].axhline(y=1.0 if idx == 1 else 0.0, color='r', linestyle='--',
                          label='Target' if idx == 1 else None)
        if idx == 1:
            axes[1, i].legend()

    plt.suptitle('Camera Normalization Statistics', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'camera_statistics.png'), dpi=150)
    plt.close()

    # ============ Figure 3: Image Grid with Camera Info ============
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))

    for i, cam in enumerate(cameras):
        img_path = os.path.join(sample_dir, cam["file_path"])
        if os.path.exists(img_path):
            img = Image.open(img_path)

            # Original row
            axes[0, i].imshow(img)
            axes[0, i].set_title(
                f'Cam {i} (Original)\n'
                f'Up: [{up_orig[i, 0]:.2f}, {up_orig[i, 1]:.2f}, {up_orig[i, 2]:.2f}]\n'
                f'Pos: [{pos_orig[i, 0]:.1f}, {pos_orig[i, 1]:.1f}, {pos_orig[i, 2]:.1f}]',
                fontsize=9, fontweight='bold')
            axes[0, i].axis('off')
            # Add red border for original
            for spine in axes[0, i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)

            # Normalized row (same image, different camera info)
            axes[1, i].imshow(img)
            axes[1, i].set_title(
                f'Cam {i} (Normalized)\n'
                f'Up: [{up_norm[i, 0]:.2f}, {up_norm[i, 1]:.2f}, {up_norm[i, 2]:.2f}]\n'
                f'Pos: [{pos_norm[i, 0]:.1f}, {pos_norm[i, 1]:.1f}, {pos_norm[i, 2]:.1f}]',
                fontsize=9, fontweight='bold')
            axes[1, i].axis('off')
            # Add green border for normalized
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)

    # Add row labels
    fig.text(0.02, 0.75, 'ORIGINAL\n(Z-up)', fontsize=14, fontweight='bold',
             color='red', rotation=90, va='center')
    fig.text(0.02, 0.25, 'NORMALIZED\n(Y-up)', fontsize=14, fontweight='bold',
             color='green', rotation=90, va='center')

    plt.suptitle(
        f'Camera Normalization Effect on Sample: {os.path.basename(sample_dir)}\n'
        f'Estimated Up Direction: [{up_direction[0]:.3f}, {up_direction[1]:.3f}, {up_direction[2]:.3f}]\n'
        f'⚠️ Images are IDENTICAL - only camera parameters (c2w matrices) are transformed',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.92])
    plt.savefig(os.path.join(output_dir, 'image_grid_with_cameras.png'), dpi=150)
    plt.close()

    # ============ Save numerical data ============
    np.savez(os.path.join(output_dir, 'camera_data.npz'),
             c2w_original=c2w_original,
             c2w_normalized=c2w_normalized,
             up_direction=up_direction,
             positions_original=pos_orig,
             positions_normalized=pos_norm,
             up_vectors_original=up_orig,
             up_vectors_normalized=up_norm)

    # Print summary
    print(f"\n{'='*60}")
    print("Camera Normalization Summary")
    print(f"{'='*60}")
    print(f"Sample: {sample_dir}")
    print(f"Estimated Up Direction: [{up_direction[0]:.4f}, {up_direction[1]:.4f}, {up_direction[2]:.4f}]")
    print(f"\nOriginal Camera Up (avg): [{avg_up_orig[0]:.4f}, {avg_up_orig[1]:.4f}, {avg_up_orig[2]:.4f}]")
    print(f"Normalized Camera Up (avg): [{avg_up_norm[0]:.4f}, {avg_up_norm[1]:.4f}, {avg_up_norm[2]:.4f}]")
    print(f"Target Y-up: [0.0, 1.0, 0.0]")
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - camera_positions_3d.png")
    print(f"  - camera_statistics.png")
    print(f"  - image_grid_with_cameras.png")
    print(f"  - camera_data.npz")

    return up_direction, c2w_original, c2w_normalized


def main():
    parser = argparse.ArgumentParser(description="Visualize camera normalization effect")
    parser.add_argument("--sample_dir", type=str, default="data_mouse/sample_000000",
                        help="Path to sample directory with opencv_cameras.json")
    parser.add_argument("--output_dir", type=str, default="outputs/camera_normalization_viz",
                        help="Output directory for visualizations")
    args = parser.parse_args()

    create_comparison_figure(args.sample_dir, args.output_dir)


if __name__ == "__main__":
    main()
