#!/usr/bin/env python3
"""
Blender Multi-View Rendering Script for GS-LRM Synthetic Dataset

Usage:
    blender --background model.blend -P render_gslrm_dataset.py -- [options]

Options:
    --output_dir PATH   Output directory (default: /tmp/synthetic)
    --num_views N       Number of views (default: 6)
    --elevation DEG     Camera elevation angle (default: 20)
    --distance D        Camera distance from origin (default: 2.7)
    --resolution N      Image resolution (default: 512)
    --fov DEG           Field of view in degrees (default: 50)

Example:
    blender --background bunny.blend -P render_gslrm_dataset.py -- \\
        --output_dir /home/joon/data/synthetic/bunny_001 \\
        --num_views 6 --elevation 20 --distance 2.7

Requirements:
    - Blender 4.0+ with Cycles renderer
    - Scene should have object at origin
    - Object should be normalized to fit within unit sphere

Author: FaceLift Team
Date: 2026-01-28
"""

import bpy
import numpy as np
import json
import math
import os
import sys
import argparse
from mathutils import Vector, Matrix, Euler


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "output_dir": "/tmp/synthetic_gslrm",
    "num_views": 6,
    "elevation_deg": 20,
    "distance": 2.7,
    "resolution": 512,
    "fov_deg": 50.0,
    "sensor_width_mm": 36.0,
    "render_samples": 128,
    "use_denoising": True,
}


# =============================================================================
# Camera Utilities
# =============================================================================

def compute_focal_length_mm(fov_deg: float, sensor_width_mm: float) -> float:
    """Compute focal length in mm from FOV and sensor width."""
    fov_rad = math.radians(fov_deg)
    return (sensor_width_mm / 2.0) / math.tan(fov_rad / 2.0)


def compute_focal_length_px(fov_deg: float, image_size: int) -> float:
    """Compute focal length in pixels from FOV and image size."""
    fov_rad = math.radians(fov_deg)
    return image_size / (2.0 * math.tan(fov_rad / 2.0))


def spherical_to_cartesian(
    azimuth_deg: float, 
    elevation_deg: float, 
    distance: float
) -> tuple:
    """Convert spherical coordinates to Cartesian (Z-up)."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    
    x = distance * math.cos(el) * math.sin(az)
    y = distance * math.cos(el) * math.cos(az)
    z = distance * math.sin(el)
    
    return (x, y, z)


def look_at_rotation(camera_pos: tuple, target: tuple = (0, 0, 0)) -> Euler:
    """Compute rotation to look at target from camera position."""
    direction = Vector(target) - Vector(camera_pos)
    # Track -Z axis, Y up
    rot_quat = direction.to_track_quat("-Z", "Y")
    return rot_quat.to_euler()


# =============================================================================
# Scene Setup
# =============================================================================

def setup_render_settings(config: dict):
    """Configure Blender render settings."""
    scene = bpy.context.scene
    
    # Engine
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = config["render_samples"]
    scene.cycles.use_denoising = config["use_denoising"]
    
    # Resolution
    scene.render.resolution_x = config["resolution"]
    scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    
    # Output format
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 15
    
    # Transparent background for alpha
    scene.render.film_transparent = True
    
    print(f"[Setup] Render: {config[resolution]}x{config[resolution]}, "
          f"{config[render_samples]} samples, GPU Cycles")


def setup_lighting():
    """Setup basic three-point lighting."""
    # Remove existing lights
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Key light (Sun)
    key_light = bpy.data.lights.new(name="KeyLight", type="SUN")
    key_light.energy = 3.0
    key_obj = bpy.data.objects.new("KeyLight", key_light)
    bpy.context.collection.objects.link(key_obj)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Fill light
    fill_light = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_light.energy = 1.0
    fill_obj = bpy.data.objects.new("FillLight", fill_light)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(45), 0, math.radians(-135))
    
    # Rim light
    rim_light = bpy.data.lights.new(name="RimLight", type="SUN")
    rim_light.energy = 1.5
    rim_obj = bpy.data.objects.new("RimLight", rim_light)
    bpy.context.collection.objects.link(rim_obj)
    rim_obj.rotation_euler = (math.radians(-30), 0, math.radians(180))
    
    print("[Setup] Three-point lighting configured")


def setup_camera(config: dict):
    """Create and configure camera."""
    # Remove existing cameras
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Create camera
    cam_data = bpy.data.cameras.new(name="RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # Set focal length
    focal_mm = compute_focal_length_mm(config["fov_deg"], config["sensor_width_mm"])
    cam_data.lens = focal_mm
    cam_data.sensor_width = config["sensor_width_mm"]
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    
    print(f"[Setup] Camera: FOV={config[fov_deg]}°, "
          f"focal={focal_mm:.2f}mm, sensor={config[sensor_width_mm]}mm")
    
    return cam_obj, cam_data


# =============================================================================
# Camera Parameter Extraction
# =============================================================================

def extract_camera_params(
    cam_obj, 
    cam_data, 
    view_id: int,
    azimuth_deg: float,
    config: dict
) -> dict:
    """Extract camera parameters in GS-LRM format."""
    
    # Intrinsics (in pixels)
    fx = fy = compute_focal_length_px(config["fov_deg"], config["resolution"])
    cx = cy = config["resolution"] / 2.0
    
    # Extrinsics
    # Blender matrix_world is camera-to-world (c2w)
    c2w = np.array(cam_obj.matrix_world)
    
    # GS-LRM expects world-to-camera (w2c)
    w2c = np.linalg.inv(c2w)
    
    return {
        "w": config["resolution"],
        "h": config["resolution"],
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w2c": w2c.tolist(),
        "c2w": c2w.tolist(),
        "file_path": f"images/cam_{view_id:03d}.png",
        "view_id": view_id,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": config["elevation_deg"],
        "camera_distance": config["distance"],
    }


# =============================================================================
# Rendering
# =============================================================================

def render_view(output_path: str, filename: str):
    """Render current view to file."""
    filepath = os.path.join(output_path, filename)
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered: {filename}")


def render_multiview_dataset(config: dict):
    """Main function to render complete multi-view dataset."""
    
    output_dir = config["output_dir"]
    num_views = config["num_views"]
    
    print(f"\n{=*60}")
    print(f"GS-LRM Synthetic Dataset Rendering")
    print(f"{=*60}")
    print(f"Output: {output_dir}")
    print(f"Views: {num_views}")
    print(f"Resolution: {config[resolution]}x{config[resolution]}")
    print(f"Camera: distance={config[distance]}, elevation={config[elevation_deg]}°")
    print(f"{=*60}\n")
    
    # Setup
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    setup_render_settings(config)
    setup_lighting()
    cam_obj, cam_data = setup_camera(config)
    
    frames = []
    
    # Render each view
    for view_id in range(num_views):
        azimuth_deg = view_id * (360.0 / num_views)
        
        print(f"\n[View {view_id}/{num_views}] Azimuth: {azimuth_deg:.1f}°")
        
        # Position camera
        cam_pos = spherical_to_cartesian(
            azimuth_deg, 
            config["elevation_deg"], 
            config["distance"]
        )
        cam_obj.location = cam_pos
        cam_obj.rotation_euler = look_at_rotation(cam_pos)
        
        # Update scene
        bpy.context.view_layer.update()
        
        # Render
        filename = f"cam_{view_id:03d}.png"
        render_view(os.path.join(output_dir, "images"), filename)
        
        # Extract parameters
        params = extract_camera_params(cam_obj, cam_data, view_id, azimuth_deg, config)
        frames.append(params)
    
    # Save camera parameters
    camera_data = {
        "frames": frames,
        "_metadata": {
            "generator": "render_gslrm_dataset.py",
            "blender_version": bpy.app.version_string,
            "num_views": num_views,
            "resolution": config["resolution"],
            "fov_deg": config["fov_deg"],
            "elevation_deg": config["elevation_deg"],
            "camera_distance": config["distance"],
            "fx": frames[0]["fx"],
            "fy": frames[0]["fy"],
            "cx": frames[0]["cx"],
            "cy": frames[0]["cy"],
        }
    }
    
    json_path = os.path.join(output_dir, "opencv_cameras.json")
    with open(json_path, "w") as f:
        json.dump(camera_data, f, indent=2)
    
    print(f"\n{=*60}")
    print(f"Dataset saved to: {output_dir}")
    print(f"Camera params: {json_path}")
    print(f"Images: {output_dir}/images/cam_*.png")
    print(f"{=*60}\n")
    
    return camera_data


# =============================================================================
# Validation
# =============================================================================

def validate_output(camera_data: dict) -> bool:
    """Validate generated camera parameters."""
    frames = camera_data["frames"]
    meta = camera_data["_metadata"]
    
    print("\n[Validation]")
    
    # Check intrinsics consistency
    fx_values = [f["fx"] for f in frames]
    fx_std = np.std(fx_values)
    if fx_std > 0.01:
        print(f"  ⚠️ fx inconsistent: std={fx_std:.4f}")
        return False
    print(f"  ✓ Intrinsics: fx={fx_values[0]:.2f}, cx={frames[0][cx]:.1f}")
    
    # Check w2c matrices
    for i, f in enumerate(frames):
        w2c = np.array(f["w2c"])
        if w2c.shape != (4, 4):
            print(f"  ⚠️ Frame {i}: w2c shape {w2c.shape} != (4,4)")
            return False
        
        # Check rotation matrix orthogonality
        R = w2c[:3, :3]
        RRT = R @ R.T
        if not np.allclose(RRT, np.eye(3), atol=0.01):
            print(f"  ⚠️ Frame {i}: rotation not orthogonal")
            return False
    print(f"  ✓ All {len(frames)} w2c matrices valid")
    
    # Check camera distances
    distances = []
    for f in frames:
        c2w = np.array(f["c2w"])
        cam_pos = c2w[:3, 3]
        dist = np.linalg.norm(cam_pos)
        distances.append(dist)
    
    avg_dist = np.mean(distances)
    dist_std = np.std(distances)
    if dist_std > 0.01:
        print(f"  ⚠️ Camera distances vary: {avg_dist:.3f} ± {dist_std:.4f}")
        return False
    print(f"  ✓ Camera distance: {avg_dist:.3f}")
    
    # Check azimuth coverage
    azimuths = [f["azimuth_deg"] for f in frames]
    print(f"  ✓ Azimuths: {[f{a:.0f}° for a in azimuths]}")
    
    print("\n  ✓ Validation PASSED")
    return True


# =============================================================================
# CLI Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    # Find -- separator
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]
    except ValueError:
        args = []
    
    parser = argparse.ArgumentParser(description="Render GS-LRM synthetic dataset")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--num_views", type=int, default=DEFAULT_CONFIG["num_views"])
    parser.add_argument("--elevation", type=float, default=DEFAULT_CONFIG["elevation_deg"])
    parser.add_argument("--distance", type=float, default=DEFAULT_CONFIG["distance"])
    parser.add_argument("--resolution", type=int, default=DEFAULT_CONFIG["resolution"])
    parser.add_argument("--fov", type=float, default=DEFAULT_CONFIG["fov_deg"])
    parser.add_argument("--samples", type=int, default=DEFAULT_CONFIG["render_samples"])
    
    return parser.parse_args(args)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    config = {
        "output_dir": args.output_dir,
        "num_views": args.num_views,
        "elevation_deg": args.elevation,
        "distance": args.distance,
        "resolution": args.resolution,
        "fov_deg": args.fov,
        "sensor_width_mm": DEFAULT_CONFIG["sensor_width_mm"],
        "render_samples": args.samples,
        "use_denoising": DEFAULT_CONFIG["use_denoising"],
    }
    
    camera_data = render_multiview_dataset(config)
    validate_output(camera_data)
