#!/usr/bin/env python3
"""
MAMMAL Mesh를 32-View Orbit 카메라로 렌더링 (v2 - 좌표계 수정)
"""

import bpy
import sys
import os
import json
import argparse
import numpy as np
import math
from pathlib import Path
from mathutils import Matrix, Vector, Euler

# Args parsing
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
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args(argv)

# ==============================================================================
# Constants
# ==============================================================================

NUM_VIEWS = 32
IMAGE_SIZE = 512
FX = FY = 548.9937744140625
CX = CY = 256.0
CAMERA_DISTANCE = 2.7
ELEVATION = 20
TARGET_OBJECT_SIZE = 1.5

EXPERIMENTS = {
    "MAMMAL_CENTER": {"offset": (0.0, 0.0, 0.0), "pp_correct": False, "desc": "Center baseline"},
    "MAMMAL_OFFSET": {"offset": (0.3, 0.2, 0.0), "pp_correct": False, "desc": "Offset, no PP fix"},
    "MAMMAL_OFFSET_PP": {"offset": (0.3, 0.2, 0.0), "pp_correct": True, "desc": "Offset + PP fix"},
}

# ==============================================================================
# Core Functions
# ==============================================================================

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

def load_mammal_obj(obj_path):
    """Load MAMMAL OBJ and normalize to origin with correct scale"""
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    
    # Get vertices in local space
    verts = [v.co for v in obj.data.vertices]
    verts_np = np.array([[v.x, v.y, v.z] for v in verts])
    
    # MAMMAL is in mm, center at ~(99, 24, 35)
    center = verts_np.mean(axis=0)
    size = verts_np.max(axis=0) - verts_np.min(axis=0)
    max_dim = size.max()
    
    print(f"Original: center={center}, size={size}, max_dim={max_dim:.1f}mm")
    
    # Transform: center to origin, mm to meters, scale to target size
    # target_size / (max_dim_in_meters) = target_size / (max_dim_mm * 0.001)
    scale = TARGET_OBJECT_SIZE / (max_dim * 0.001)
    
    # Apply to each vertex directly
    for v in obj.data.vertices:
        # Center
        v.co.x -= center[0]
        v.co.y -= center[1]
        v.co.z -= center[2]
        # mm to m and scale
        v.co.x *= 0.001 * scale
        v.co.y *= 0.001 * scale
        v.co.z *= 0.001 * scale
    
    # MAMMAL coordinate system fix: -Y up -> Z up (Blender)
    # Rotate +90 degrees around X axis: (x, y, z) -> (x, z, -y)
    import math as _math
    for v in obj.data.vertices:
        old_y = v.co.y
        old_z = v.co.z
        v.co.y = old_z
        v.co.z = -old_y
    
    obj.data.update()
    
    # Verify
    verts_new = np.array([[v.co.x, v.co.y, v.co.z] for v in obj.data.vertices])
    new_size = verts_new.max(axis=0) - verts_new.min(axis=0)
    print(f"Normalized: size={new_size}, max_dim={new_size.max():.3f}")
    
    return obj

def create_camera(cam_idx, azimuth_deg, elevation_deg, distance):
    """Create camera pointing at origin"""
    cam_data = bpy.data.cameras.new(f"Cam_{cam_idx:03d}")
    cam_obj = bpy.data.objects.new(f"Cam_{cam_idx:03d}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    # Spherical to Cartesian (Blender: Z-up, Y-forward)
    azim = math.radians(azimuth_deg)
    elev = math.radians(elevation_deg)
    
    # Standard spherical coordinates with Z-up
    x = distance * math.cos(elev) * math.sin(azim)
    y = -distance * math.cos(elev) * math.cos(azim)
    z = distance * math.sin(elev)
    
    cam_obj.location = (x, y, z)
    
    # Point at origin using track constraint
    direction = -cam_obj.location.normalized()
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    # Set focal length from fx
    # fx = f_mm * width_px / sensor_width_mm
    # f_mm = fx * sensor_width / width_px
    sensor_width = 36  # default full frame
    cam_data.lens = FX * sensor_width / IMAGE_SIZE
    cam_data.sensor_width = sensor_width
    
    return cam_obj, cam_data

def get_opencv_matrices(cam_obj):
    """Get w2c and c2w in OpenCV convention"""
    # Blender camera: -Z forward, Y up
    # OpenCV camera: Z forward, -Y up
    
    cam_matrix = cam_obj.matrix_world.copy()
    
    # Blender to OpenCV conversion
    flip = Matrix([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    c2w_opencv = cam_matrix @ flip
    w2c_opencv = c2w_opencv.inverted()
    
    return np.array(w2c_opencv), np.array(c2w_opencv)

def setup_render():
    """Setup Cycles renderer"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 64
    scene.render.resolution_x = IMAGE_SIZE
    scene.render.resolution_y = IMAGE_SIZE
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    
    # GPU setup
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True

def setup_lighting():
    """Basic lighting"""
    # World background
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
    
    # Area light
    light = bpy.data.lights.new("AreaLight", "AREA")
    light.energy = 50
    light_obj = bpy.data.objects.new("AreaLight", light)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (2, -2, 3)

def setup_material(obj):
    """Simple diffuse material"""
    mat = bpy.data.materials.new("MouseMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.6, 0.55, 0.5, 1)
    bsdf.inputs["Roughness"].default_value = 0.8
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def render_sample(obj_path, sample_idx, output_dir, config):
    """Render one sample with 32 views"""
    clear_scene()
    
    # Load mesh
    obj = load_mammal_obj(str(obj_path))
    
    # Apply offset if any
    offset = config["offset"]
    if offset != (0, 0, 0):
        obj.location = Vector(offset)
    
    setup_material(obj)
    setup_render()
    setup_lighting()
    
    # Create 32 cameras
    cameras = []
    for i in range(NUM_VIEWS):
        azim = i * (360.0 / NUM_VIEWS)
        cam_obj, cam_data = create_camera(i, azim, ELEVATION, CAMERA_DISTANCE)
        cameras.append((cam_obj, cam_data))
    
    # Render each view
    sample_dir = Path(output_dir) / f"sample_{sample_idx:05d}"
    images_dir = sample_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (cam_obj, _) in enumerate(cameras):
        bpy.context.scene.camera = cam_obj
        bpy.context.scene.render.filepath = str(images_dir / f"cam_{i:03d}.png")
        bpy.ops.render.render(write_still=True)
    
    # Save cameras JSON
    frames = []
    for i, (cam_obj, _) in enumerate(cameras):
        w2c, c2w = get_opencv_matrices(cam_obj)
        
        cx, cy = CX, CY
        if config["pp_correct"] and offset != (0, 0, 0):
            cx = CX + offset[0] * FX / CAMERA_DISTANCE
            cy = CY + offset[1] * FY / CAMERA_DISTANCE
        
        frames.append({
            "w": IMAGE_SIZE, "h": IMAGE_SIZE,
            "fx": FX, "fy": FY, "cx": cx, "cy": cy,
            "w2c": w2c.tolist(), "c2w": c2w.tolist(),
            "file_path": f"images/cam_{i:03d}.png"
        })
    
    with open(sample_dir / "opencv_cameras.json", "w") as f:
        json.dump({"frames": frames}, f, indent=2)

def main():
    config = EXPERIMENTS[args.experiment]
    output_dir = Path(args.output_dir)
    
    print("\n" + "="*60)
    print(f"MAMMAL 32-View Rendering v2")
    print(f"Experiment: {args.experiment} - {config['desc']}")
    print(f"Samples: {args.num_samples}")
    print("="*60 + "\n")
    
    # Get OBJ files
    obj_dir = Path(args.mammal_results) / "obj"
    all_objs = sorted(obj_dir.glob("step_2_frame_*.obj"))
    step = max(1, len(all_objs) // args.num_samples)
    selected = all_objs[::step][:args.num_samples]
    
    print(f"Selected {len(selected)} frames from {len(all_objs)} total")
    
    for idx, obj_path in enumerate(selected):
        print(f"[{idx+1}/{len(selected)}] {obj_path.name}")
        render_sample(obj_path, idx, output_dir, config)
    
    # Create data list
    with open(output_dir / "data_train.txt", "w") as f:
        for i in range(len(selected)):
            f.write(str(output_dir / f"sample_{i:05d}") + "\n")
    
    print(f"\nComplete: {output_dir}")

if __name__ == "__main__":
    main()
