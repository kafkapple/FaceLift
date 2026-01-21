#\!/usr/bin/env python3
"""
Camera Setup Visualization v2
=============================

Enhanced visualization with:
- Precise camera center, PP, and ray direction
- Longer pointed arrows for viewing direction
- Camera angle calculations (azimuth, elevation)
- RGB images connected to cameras with lines
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


class Arrow3D(FancyArrowPatch):
    """3D arrow with pointed tip."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def load_camera_data(sample_dir):
    """Load camera parameters and images."""
    json_path = os.path.join(sample_dir, "opencv_cameras.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    cameras = []
    images = []
    
    for idx, frame in enumerate(data["frames"]):
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        
        position = c2w[:3, 3]
        rotation = c2w[:3, :3]
        
        # Camera coordinate axes in world frame
        forward = rotation @ np.array([0, 0, 1])  # Z-axis (viewing direction)
        up = rotation @ np.array([0, -1, 0])       # -Y (up in OpenCV)
        right = rotation @ np.array([1, 0, 0])     # X-axis
        
        # Calculate angles
        # Azimuth: angle in XY plane from X-axis
        azimuth = np.degrees(np.arctan2(position[1], position[0]))
        # Elevation: angle from XY plane
        dist_xy = np.sqrt(position[0]**2 + position[1]**2)
        elevation = np.degrees(np.arctan2(position[2], dist_xy))
        # Distance from origin
        distance = np.linalg.norm(position)
        
        cameras.append({
            "position": position,
            "forward": forward,
            "up": up,
            "right": right,
            "rotation": rotation,
            "fx": frame["fx"], "fy": frame["fy"],
            "cx": frame["cx"], "cy": frame["cy"],
            "w": frame["w"], "h": frame["h"],
            "view_id": frame.get("view_id", idx),
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance
        })
        
        img_path = os.path.join(sample_dir, frame["file_path"])
        if os.path.exists(img_path):
            images.append(np.array(Image.open(img_path).convert("RGB")))
        else:
            images.append(None)
    
    return cameras, images


def draw_camera_with_image(ax, cam, img, color, arrow_length=0.8, img_distance=0.4, img_scale=0.3):
    """Draw camera with pointed arrow and connected RGB image."""
    pos = cam["position"]
    forward = cam["forward"]
    up = cam["up"]
    right = cam["right"]
    vid = cam["view_id"]
    
    # 1. Camera position marker
    ax.scatter(pos[0], pos[1], pos[2], c=[color], s=200, marker="o", 
               edgecolors="black", linewidths=1.5, zorder=5)
    
    # 2. Pointed arrow for viewing direction (longer)
    arrow_end = pos + forward * arrow_length
    arrow = Arrow3D([pos[0], arrow_end[0]], 
                    [pos[1], arrow_end[1]], 
                    [pos[2], arrow_end[2]],
                    mutation_scale=15, lw=2, arrowstyle="-|>", color=color)
    ax.add_artist(arrow)
    
    # 3. Camera frustum (smaller, more precise)
    frustum_scale = 0.15
    frustum_depth = 0.25
    corners = [
        pos + forward * frustum_depth + (up + right) * frustum_scale,
        pos + forward * frustum_depth + (up - right) * frustum_scale,
        pos + forward * frustum_depth + (-up - right) * frustum_scale,
        pos + forward * frustum_depth + (-up + right) * frustum_scale,
    ]
    for corner in corners:
        ax.plot3D([pos[0], corner[0]], [pos[1], corner[1]], [pos[2], corner[2]], 
                  color=color, alpha=0.6, linewidth=1)
    # Near plane
    for i in range(4):
        ax.plot3D([corners[i][0], corners[(i+1)%4][0]], 
                  [corners[i][1], corners[(i+1)%4][1]], 
                  [corners[i][2], corners[(i+1)%4][2]], 
                  color=color, alpha=0.6, linewidth=1)
    
    # 4. Image billboard connected with dashed line
    if img is not None:
        img_center = pos + forward * img_distance
        
        # Dashed line connecting camera to image
        ax.plot3D([pos[0], img_center[0]], [pos[1], img_center[1]], [pos[2], img_center[2]],
                  color=color, linestyle="--", alpha=0.5, linewidth=1)
        
        # Draw image as colored scatter (simplified)
        img_small = np.array(Image.fromarray(img).resize((16, 16))) / 255.0
        half_w = img_scale * 0.5
        half_h = half_w * img.shape[0] / img.shape[1]
        
        for i in range(img_small.shape[0]):
            for j in range(img_small.shape[1]):
                u = (j / img_small.shape[1] - 0.5) * 2 * half_w
                v = (0.5 - i / img_small.shape[0]) * 2 * half_h
                point = img_center + right * u + up * v
                ax.scatter(point[0], point[1], point[2], c=[img_small[i, j]], s=3, alpha=0.9)
    
    # 5. Label with coordinates and angles
    label = "Cam {}\n({:.1f}, {:.1f}, {:.1f})\nAz:{:.0f}° El:{:.0f}°".format(
        vid, pos[0], pos[1], pos[2], cam["azimuth"], cam["elevation"])
    ax.text(pos[0], pos[1], pos[2] + 0.25, label, fontsize=8, 
            ha="center", color=color, fontweight="bold")
    
    return arrow_end


def visualize_cameras_v2(cameras, images, output_path=None, title="Camera Setup"):
    """Enhanced 3D visualization."""
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection="3d")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(cameras)))
    
    # Origin and coordinate axes
    ax.scatter([0], [0], [0], c="black", s=150, marker="x", linewidths=3, label="Origin (Mouse)")
    
    axis_len = 0.6
    for axis, color, label in [([1,0,0], "red", "X"), ([0,1,0], "green", "Y"), ([0,0,1], "blue", "Z")]:
        ax.quiver(0, 0, 0, axis[0]*axis_len, axis[1]*axis_len, axis[2]*axis_len, 
                  color=color, arrow_length_ratio=0.1, linewidth=2)
        ax.text(axis[0]*axis_len*1.2, axis[1]*axis_len*1.2, axis[2]*axis_len*1.2, 
                label, color=color, fontsize=12, fontweight="bold")
    
    # Draw each camera
    for i, (cam, img) in enumerate(zip(cameras, images)):
        draw_camera_with_image(ax, cam, img, colors[i], 
                               arrow_length=0.6, img_distance=0.35, img_scale=0.25)
    
    # Legend
    for i, cam in enumerate(cameras):
        ax.scatter([], [], [], c=[colors[i]], s=100, label="Cam {} (d={:.2f})".format(
            cam["view_id"], cam["distance"]))
    
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # Equal aspect
    all_pos = np.array([cam["position"] for cam in cameras])
    max_range = np.max(np.abs(all_pos)) * 1.3
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range/2, max_range])
    
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print("Saved: {}".format(output_path))
    
    return fig, ax


def create_camera_angle_table(cameras, output_path=None):
    """Create table with camera angles and parameters."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    
    headers = ["Cam", "Position (x,y,z)", "Distance", "Azimuth", "Elevation", 
               "fx", "fy", "cx", "cy"]
    rows = []
    
    for cam in cameras:
        pos = cam["position"]
        rows.append([
            str(cam["view_id"]),
            "({:.2f}, {:.2f}, {:.2f})".format(pos[0], pos[1], pos[2]),
            "{:.2f}".format(cam["distance"]),
            "{:.1f}°".format(cam["azimuth"]),
            "{:.1f}°".format(cam["elevation"]),
            "{:.1f}".format(cam["fx"]),
            "{:.1f}".format(cam["fy"]),
            "{:.1f}".format(cam["cx"]),
            "{:.1f}".format(cam["cy"])
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    
    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    
    ax.set_title("Camera Parameters with Angles", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print("Saved: {}".format(output_path))
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Enhanced camera visualization v2")
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_images", action="store_true")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.sample_dir, "camera_viz_v2")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading: {}".format(args.sample_dir))
    cameras, images = load_camera_data(args.sample_dir)
    if args.no_images:
        images = [None] * len(images)
    print("Loaded {} cameras".format(len(cameras)))
    
    sample_name = os.path.basename(args.sample_dir.rstrip("/"))
    
    # 3D visualization
    fig, ax = visualize_cameras_v2(
        cameras, images,
        output_path=os.path.join(args.output_dir, "camera_3d_enhanced.png"),
        title="Camera Setup - {} (with RGB)".format(sample_name)
    )
    
    # Multiple angles
    for elev, azim, name in [(30, 45, "iso"), (90, 0, "top"), (0, 0, "front"), (0, 90, "side")]:
        ax.view_init(elev=elev, azim=azim)
        fig.savefig(os.path.join(args.output_dir, "camera_3d_{}.png".format(name)),
                    dpi=150, bbox_inches="tight")
        print("Saved: camera_3d_{}.png".format(name))
    
    # Angle table
    create_camera_angle_table(
        cameras,
        output_path=os.path.join(args.output_dir, "camera_angles_table.png")
    )
    
    # Save JSON with angles
    data = {}
    for cam in cameras:
        data["cam_{}".format(cam["view_id"])] = {
            "position": cam["position"].tolist(),
            "forward": cam["forward"].tolist(),
            "distance": cam["distance"],
            "azimuth_deg": cam["azimuth"],
            "elevation_deg": cam["elevation"],
            "fx": cam["fx"], "fy": cam["fy"],
            "cx": cam["cx"], "cy": cam["cy"]
        }
    
    with open(os.path.join(args.output_dir, "camera_data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("Saved: camera_data.json")
    
    print("\nDone: {}".format(args.output_dir))


if __name__ == "__main__":
    main()
