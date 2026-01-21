#!/usr/bin/env python3
"""
Visualize Camera Setup for Mouse Dataset
=========================================

Visualizes 6 camera positions, orientations, and RGB images in 3D space.

Usage:
    python -m mouse_extensions.scripts.visualize_camera_setup \
        --sample_dir /path/to/sample \
        --output_dir /path/to/output
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


def load_camera_data(sample_dir):
    """Load camera parameters and images from sample directory."""
    json_path = os.path.join(sample_dir, "opencv_cameras.json")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    cameras = []
    images = []
    
    for idx, frame in enumerate(data["frames"]):
        fx, fy = frame["fx"], frame["fy"]
        cx, cy = frame["cx"], frame["cy"]
        w, h = frame["w"], frame["h"]
        
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        
        position = c2w[:3, 3]
        rotation = c2w[:3, :3]
        
        forward = rotation @ np.array([0, 0, 1])
        up = rotation @ np.array([0, -1, 0])
        right = rotation @ np.array([1, 0, 0])
        
        cameras.append({
            "position": position,
            "forward": forward,
            "up": up,
            "right": right,
            "rotation": rotation,
            "fx": fx, "fy": fy,
            "cx": cx, "cy": cy,
            "w": w, "h": h,
            "view_id": frame.get("view_id", idx)
        })
        
        img_path = os.path.join(sample_dir, frame["file_path"])
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert("RGB"))
            images.append(img)
        else:
            images.append(None)
    
    return cameras, images


def draw_camera_frustum(ax, cam, color, scale=0.3, alpha=0.3):
    """Draw camera frustum as a pyramid."""
    pos = cam["position"]
    forward = cam["forward"]
    up = cam["up"]
    right = cam["right"]
    
    fov_scale = scale
    corners = [
        pos + forward * scale + (up + right) * fov_scale,
        pos + forward * scale + (up - right) * fov_scale,
        pos + forward * scale + (-up - right) * fov_scale,
        pos + forward * scale + (-up + right) * fov_scale,
    ]
    
    for corner in corners:
        ax.plot3D([pos[0], corner[0]], [pos[1], corner[1]], [pos[2], corner[2]], 
                  color=color, alpha=0.5, linewidth=1)
    
    verts = [corners]
    face = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(face)


def draw_image_billboard(ax, cam, img, scale=0.2, distance=0.15):
    """Draw RGB image as a billboard in front of camera."""
    if img is None:
        return
    
    pos = cam["position"]
    forward = cam["forward"]
    up = cam["up"]
    right = cam["right"]
    
    center = pos + forward * distance
    half_w = scale
    half_h = scale * img.shape[0] / img.shape[1]
    
    img_small = np.array(Image.fromarray(img).resize((32, 32))) / 255.0
    
    for i in range(img_small.shape[0]):
        for j in range(img_small.shape[1]):
            u = (j / img_small.shape[1] - 0.5) * 2 * half_w
            v = (0.5 - i / img_small.shape[0]) * 2 * half_h
            point = center + right * u + up * v
            color = img_small[i, j]
            ax.scatter(point[0], point[1], point[2], c=[color], s=2, alpha=0.8)


def visualize_cameras(cameras, images, output_path=None, show_images=True, title="Camera Setup"):
    """Create 3D visualization of camera setup."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(cameras)))
    
    ax.scatter([0], [0], [0], c="black", s=100, marker="o", label="Origin")
    
    axis_length = 0.5
    ax.quiver(0, 0, 0, axis_length, 0, 0, color="red", arrow_length_ratio=0.1, label="X")
    ax.quiver(0, 0, 0, 0, axis_length, 0, color="green", arrow_length_ratio=0.1, label="Y")
    ax.quiver(0, 0, 0, 0, 0, axis_length, color="blue", arrow_length_ratio=0.1, label="Z")
    
    for i, (cam, img) in enumerate(zip(cameras, images)):
        pos = cam["position"]
        forward = cam["forward"]
        color = colors[i]
        vid = cam["view_id"]
        
        ax.scatter(pos[0], pos[1], pos[2], c=[color], s=150, marker="^", 
                   label="Cam {}".format(vid))
        
        arrow_len = 0.4
        ax.quiver(pos[0], pos[1], pos[2], 
                  forward[0]*arrow_len, forward[1]*arrow_len, forward[2]*arrow_len,
                  color=color, arrow_length_ratio=0.2, linewidth=2)
        
        draw_camera_frustum(ax, cam, color, scale=0.25, alpha=0.2)
        
        ax.text(pos[0], pos[1], pos[2] + 0.15, "Cam {}".format(vid), fontsize=10, 
                ha="center", color=color)
        
        if show_images and img is not None:
            draw_image_billboard(ax, cam, img, scale=0.15, distance=0.2)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14)
    
    all_pos = np.array([cam["position"] for cam in cameras])
    max_range = np.max(np.abs(all_pos)) * 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print("Saved: {}".format(output_path))
    
    return fig, ax


def create_camera_info_table(cameras, output_path=None):
    """Create summary table of camera parameters."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    
    headers = ["Cam", "Position (x,y,z)", "fx", "fy", "cx", "cy", "Distance"]
    rows = []
    
    for cam in cameras:
        pos = cam["position"]
        dist = np.linalg.norm(pos)
        vid = cam["view_id"]
        rows.append([
            str(vid),
            "({:.2f}, {:.2f}, {:.2f})".format(pos[0], pos[1], pos[2]),
            "{:.1f}".format(cam["fx"]),
            "{:.1f}".format(cam["fy"]),
            "{:.1f}".format(cam["cx"]),
            "{:.1f}".format(cam["cy"]),
            "{:.2f}".format(dist)
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title("Camera Parameters Summary", fontsize=14, pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print("Saved: {}".format(output_path))
    
    return fig


def create_image_grid(cameras, images, output_path=None):
    """Create grid of all camera images with labels."""
    n_cams = len(cameras)
    cols = 3
    rows = (n_cams + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten() if n_cams > 1 else [axes]
    
    for i, (cam, img) in enumerate(zip(cameras, images)):
        if img is not None:
            axes[i].imshow(img)
        pos = cam["position"]
        vid = cam["view_id"]
        axes[i].set_title("Cam {}\npos=({:.1f}, {:.1f}, {:.1f})".format(vid, pos[0], pos[1], pos[2]))
        axes[i].axis("off")
    
    for i in range(len(cameras), len(axes)):
        axes[i].axis("off")
    
    plt.suptitle("Camera Views", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print("Saved: {}".format(output_path))
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize camera setup for mouse dataset")
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Path to sample directory with opencv_cameras.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for visualizations")
    parser.add_argument("--no_images", action="store_true",
                        help="Skip image billboards in 3D view")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.sample_dir, "camera_visualization")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading camera data from: {}".format(args.sample_dir))
    cameras, images = load_camera_data(args.sample_dir)
    print("Loaded {} cameras".format(len(cameras)))
    
    sample_name = os.path.basename(args.sample_dir.rstrip("/"))
    
    print("\n=== Creating 3D Camera Visualization ===")
    fig_3d, _ = visualize_cameras(
        cameras, images,
        output_path=os.path.join(args.output_dir, "camera_3d_view.png"),
        show_images=not args.no_images,
        title="Camera Setup - {}".format(sample_name)
    )
    
    for angle, name in [(0, "top"), (90, "front"), (45, "iso")]:
        fig_3d.axes[0].view_init(elev=30 if name != "top" else 90, azim=angle)
        fig_3d.savefig(
            os.path.join(args.output_dir, "camera_3d_{}.png".format(name)),
            dpi=150, bbox_inches="tight"
        )
        print("Saved: camera_3d_{}.png".format(name))
    
    print("\n=== Creating Camera Info Table ===")
    create_camera_info_table(
        cameras,
        output_path=os.path.join(args.output_dir, "camera_info_table.png")
    )
    
    print("\n=== Creating Image Grid ===")
    create_image_grid(
        cameras, images,
        output_path=os.path.join(args.output_dir, "camera_images_grid.png")
    )
    
    positions = {}
    for cam in cameras:
        vid = cam["view_id"]
        positions["cam_{}".format(vid)] = {
            "position": cam["position"].tolist(),
            "forward": cam["forward"].tolist(),
            "distance": float(np.linalg.norm(cam["position"])),
            "fx": cam["fx"], "fy": cam["fy"],
            "cx": cam["cx"], "cy": cam["cy"]
        }
    
    json_path = os.path.join(args.output_dir, "camera_positions.json")
    with open(json_path, "w") as f:
        json.dump(positions, f, indent=2)
    print("Saved: {}".format(json_path))
    
    print("\n=== All outputs saved to: {} ===".format(args.output_dir))
    
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
