#!/usr/bin/env python3
"""
MAMMAL Mesh를 32-View Orbit 카메라로 렌더링
FaceLift 원본 데이터 형식과 동일하게 출력

Usage:
    blender --background --python render_mammal_32view.py -- \
        --experiment MAMMAL_CENTER \
        --output_dir /path/to/output \
        --num_samples 100
"""

import bpy
import sys
import os
import json
import argparse
import numpy as np
import math
from pathlib import Path
from mathutils import Matrix, Vector

# Blender args parsing
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="MAMMAL_CENTER",
                    choices=["MAMMAL_CENTER", "MAMMAL_OFFSET", "MAMMAL_OFFSET_PP"])
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--mammal_results", type=str, 
                    default="/home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249")
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--frame_step", type=int, default=5, help="Sample every N frames from MAMMAL results")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args(argv)

# ==============================================================================
# Constants (FaceLift compatible)
# ==============================================================================

NUM_VIEWS = 32
IMAGE_SIZE = 512
FX = FY = 548.9937744140625  # FaceLift standard
CX = CY = 256.0
CAMERA_DISTANCE = 2.7
ELEVATION = 20  # degrees
TARGET_OBJECT_SIZE = 0.25  # 25% of unit sphere

# Experiment configurations
EXPERIMENTS = {
    "MAMMAL_CENTER": {
        "offset": (0.0, 0.0, 0.0),
        "pp_correct": False,
        "desc": "Mouse at center (baseline)"
    },
    "MAMMAL_OFFSET": {
        "offset": (0.3, 0.2, 0.0),  # Right and up offset
        "pp_correct": False,
        "desc": "Mouse offset, PP unchanged (expect ghosting)"
    },
    "MAMMAL_OFFSET_PP": {
        "offset": (0.3, 0.2, 0.0),
        "pp_correct": True,
        "desc": "Mouse offset with PP correction"
    }
}

# ==============================================================================
# Utility Functions
# ==============================================================================

def clear_scene():
    """Clear all objects from scene"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def load_mammal_obj(obj_path):
    """Load MAMMAL OBJ mesh"""
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    return obj

def get_mesh_bounds(obj):
    """Get mesh bounding box in world coordinates"""
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_coord = Vector((min(v[i] for v in bbox) for i in range(3)))
    max_coord = Vector((max(v[i] for v in bbox) for i in range(3)))
    return min_coord, max_coord

def normalize_mesh(obj, target_size=TARGET_OBJECT_SIZE):
    """
    Normalize mesh: center at origin and scale to target size
    MAMMAL meshes are in mm, need to convert and normalize
    """
    # Get current bounds
    min_coord, max_coord = get_mesh_bounds(obj)
    center = (min_coord + max_coord) / 2
    size = max(max_coord - min_coord)
    
    # Move to origin
    obj.location = -center
    bpy.context.view_layer.update()
    
    # Scale to target size (mm to normalized units)
    # MAMMAL mesh is ~70mm, we want 0.25 units
    scale_factor = target_size / (size / 1000)  # mm to m, then to target
    obj.scale = (scale_factor, scale_factor, scale_factor)
    
    # Apply transforms
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Verify
    min_coord, max_coord = get_mesh_bounds(obj)
    new_size = max(max_coord - min_coord)
    print(f"Mesh normalized: size {size:.1f}mm -> {new_size:.3f} units")
    
    return obj

def apply_offset(obj, offset):
    """Apply position offset to mesh"""
    obj.location = Vector(offset)
    bpy.context.view_layer.update()

def create_orbit_camera(cam_idx, azimuth, elevation, distance):
    """Create a camera at orbit position"""
    # Create camera
    cam_data = bpy.data.cameras.new(f"Camera_{cam_idx:03d}")
    cam_obj = bpy.data.objects.new(f"Camera_{cam_idx:03d}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    # Set camera properties
    cam_data.type = "PERSP"
    cam_data.lens_unit = "FOV"
    # FOV from fx: fov = 2 * atan(image_size / (2 * fx))
    fov = 2 * math.atan(IMAGE_SIZE / (2 * FX))
    cam_data.angle = fov
    
    # Calculate position from spherical coordinates
    azim_rad = math.radians(azimuth)
    elev_rad = math.radians(elevation)
    
    x = distance * math.cos(elev_rad) * math.sin(azim_rad)
    y = distance * math.cos(elev_rad) * math.cos(azim_rad)
    z = distance * math.sin(elev_rad)
    
    cam_obj.location = (x, y, z)
    
    # Point camera at origin
    direction = Vector((0, 0, 0)) - cam_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    return cam_obj, cam_data

def get_camera_matrices(cam_obj):
    """Get camera matrices in OpenCV convention"""
    # World to camera transform
    w2c = cam_obj.matrix_world.inverted()
    
    # Convert Blender (Y-up, -Z forward) to OpenCV (Z-forward, Y-down)
    # Blender: X-right, Y-up, Z-back
    # OpenCV: X-right, Y-down, Z-forward
    flip = Matrix([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    w2c_opencv = flip @ w2c
    
    # Camera to world
    c2w_opencv = w2c_opencv.inverted()
    
    return np.array(w2c_opencv), np.array(c2w_opencv)

def setup_render_engine():
    """Setup Cycles render engine"""
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.render.resolution_x = IMAGE_SIZE
    bpy.context.scene.render.resolution_y = IMAGE_SIZE
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    
    # Setup GPU
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True

def setup_lighting():
    """Setup basic lighting"""
    # Environment light
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)  # White background
    bg.inputs[1].default_value = 0.5  # Strength
    
    # Key light
    light_data = bpy.data.lights.new("KeyLight", "AREA")
    light_data.energy = 100
    light_obj = bpy.data.objects.new("KeyLight", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (2, 2, 3)
    light_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

def setup_material(obj):
    """Setup simple gray material for mesh"""
    mat = bpy.data.materials.new("MouseMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.6, 0.55, 0.5, 1)  # Mouse gray-brown
    bsdf.inputs["Roughness"].default_value = 0.7
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def render_all_views(cameras, output_dir):
    """Render all camera views"""
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for cam_idx, (cam_obj, _) in enumerate(cameras):
        bpy.context.scene.camera = cam_obj
        output_path = str(images_dir / f"cam_{cam_idx:03d}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

def save_cameras_json(cameras, output_dir, offset=(0, 0, 0), pp_correct=False):
    """Save camera parameters in FaceLift format"""
    frames = []
    
    for cam_idx, (cam_obj, cam_data) in enumerate(cameras):
        w2c, c2w = get_camera_matrices(cam_obj)
        
        # PP correction for offset
        cx, cy = CX, CY
        if pp_correct and (offset[0] != 0 or offset[1] != 0):
            # dcx = offset_x * fx / distance
            cx = CX + offset[0] * FX / CAMERA_DISTANCE
            cy = CY + offset[1] * FY / CAMERA_DISTANCE
        
        frame = {
            "w": IMAGE_SIZE,
            "h": IMAGE_SIZE,
            "fx": FX,
            "fy": FY,
            "cx": cx,
            "cy": cy,
            "w2c": w2c.tolist(),
            "c2w": c2w.tolist(),
            "file_path": f"images/cam_{cam_idx:03d}.png"
        }
        frames.append(frame)
    
    cameras_json = {"frames": frames}
    
    output_path = Path(output_dir) / "opencv_cameras.json"
    with open(output_path, "w") as f:
        json.dump(cameras_json, f, indent=2)

def get_mammal_obj_files(mammal_dir, num_samples, frame_step):
    """Get list of MAMMAL OBJ files to process"""
    obj_dir = Path(mammal_dir) / "obj"
    all_files = sorted(obj_dir.glob("step_2_frame_*.obj"))
    
    # Sample evenly
    step = max(1, len(all_files) // num_samples)
    selected = all_files[::step][:num_samples]
    
    return selected

# ==============================================================================
# Main
# ==============================================================================

def render_sample(obj_path, sample_idx, output_dir, experiment_config):
    """Render a single MAMMAL mesh sample"""
    clear_scene()
    
    # Load and normalize mesh
    obj = load_mammal_obj(str(obj_path))
    obj = normalize_mesh(obj, TARGET_OBJECT_SIZE)
    
    # Apply offset
    offset = experiment_config["offset"]
    if offset != (0, 0, 0):
        apply_offset(obj, offset)
    
    # Setup material
    setup_material(obj)
    
    # Create 32 orbit cameras
    cameras = []
    for cam_idx in range(NUM_VIEWS):
        azimuth = cam_idx * (360.0 / NUM_VIEWS)
        cam_obj, cam_data = create_orbit_camera(cam_idx, azimuth, ELEVATION, CAMERA_DISTANCE)
        cameras.append((cam_obj, cam_data))
    
    # Setup rendering
    setup_render_engine()
    setup_lighting()
    
    # Render
    sample_dir = Path(output_dir) / f"sample_{sample_idx:05d}"
    render_all_views(cameras, sample_dir)
    
    # Save cameras
    save_cameras_json(cameras, sample_dir, offset, experiment_config["pp_correct"])
    
    # Save mesh info
    mesh_info = {
        "source_obj": str(obj_path),
        "offset": offset,
        "pp_corrected": experiment_config["pp_correct"]
    }
    with open(sample_dir / "mesh_info.json", "w") as f:
        json.dump(mesh_info, f, indent=2)

def main():
    """Main rendering pipeline"""
    np.random.seed(args.seed)
    
    experiment = args.experiment
    config = EXPERIMENTS[experiment]
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*60)
    print(f"MAMMAL 32-View Rendering")
    print(f"Experiment: {experiment}")
    print(f"Description: {config['desc']}")
    print(f"Offset: {config['offset']}")
    print(f"PP Correction: {config['pp_correct']}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # Get OBJ files
    obj_files = get_mammal_obj_files(args.mammal_results, args.num_samples, args.frame_step)
    print(f"Found {len(obj_files)} OBJ files to process")
    
    # Render each sample
    for sample_idx, obj_path in enumerate(obj_files):
        print(f"[{sample_idx+1}/{len(obj_files)}] Rendering {obj_path.name}...")
        render_sample(obj_path, sample_idx, output_dir, config)
    
    # Save experiment info
    exp_info = {
        "experiment": experiment,
        "config": config,
        "num_samples": len(obj_files),
        "camera_settings": {
            "num_views": NUM_VIEWS,
            "fx": FX, "fy": FY,
            "cx": CX, "cy": CY,
            "distance": CAMERA_DISTANCE,
            "elevation": ELEVATION,
            "image_size": IMAGE_SIZE
        }
    }
    with open(output_dir / "experiment_info.json", "w") as f:
        json.dump(exp_info, f, indent=2)
    
    # Create data list file
    sample_dirs = sorted(output_dir.glob("sample_*"))
    with open(output_dir / "data_train.txt", "w") as f:
        for d in sample_dirs:
            f.write(str(d) + "\n")
    
    print("\n" + "="*60)
    print(f"Complete: {output_dir}")
    print(f"Data list: {output_dir}/data_train.txt")
    print("="*60)

if __name__ == "__main__":
    main()
