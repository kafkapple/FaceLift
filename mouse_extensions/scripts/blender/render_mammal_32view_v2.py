#!/usr/bin/env python3
"""
MAMMAL Mesh를 N-View Orbit 카메라로 렌더링 (v2.1 - procedural material + 6-view mode)

Usage:
    # 32-view dense orbit (default)
    blender --background --python render_mammal_32view_v2.py -- \
        --experiment MAMMAL_CENTER --output_dir /path/to/output --num_samples 100

    # 6-view arena-like (fixed cameras)
    blender --background --python render_mammal_32view_v2.py -- \
        --experiment MAMMAL_CENTER --output_dir /path/to/output --num_views 6

    # With procedural fur material
    blender --background --python render_mammal_32view_v2.py -- \
        --experiment MAMMAL_CENTER --output_dir /path/to/output --material fur
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
                    default=str(Path.home()) + "/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249")
parser.add_argument("--num_samples", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--texture", type=str, default=None, help="Path to texture PNG file")
parser.add_argument("--material", type=str, default="fur",
                    choices=["simple", "fur"], help="Material type")
parser.add_argument("--num_views", type=int, default=32, help="Number of views (6 for arena-like, 32 for dense)")
parser.add_argument("--mesh", type=str, default=None, help="Path to specific OBJ mesh file")
args = parser.parse_args(argv)

# ==============================================================================
# Constants
# ==============================================================================

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
    
    verts = [v.co for v in obj.data.vertices]
    verts_np = np.array([[v.x, v.y, v.z] for v in verts])
    
    center = verts_np.mean(axis=0)
    size = verts_np.max(axis=0) - verts_np.min(axis=0)
    max_dim = size.max()
    
    print(f"Original: center={center}, size={size}, max_dim={max_dim:.1f}mm")
    
    scale = TARGET_OBJECT_SIZE / (max_dim * 0.001)
    
    for v in obj.data.vertices:
        v.co.x = (v.co.x - center[0]) * 0.001 * scale
        v.co.y = (v.co.y - center[1]) * 0.001 * scale
        v.co.z = (v.co.z - center[2]) * 0.001 * scale
    
    # MAMMAL -Y up -> Blender Z up: (x, y, z) -> (x, z, -y)
    for v in obj.data.vertices:
        old_y = v.co.y
        old_z = v.co.z
        v.co.y = old_z
        v.co.z = -old_y
    
    obj.data.update()
    
    verts_new = np.array([[v.co.x, v.co.y, v.co.z] for v in obj.data.vertices])
    new_size = verts_new.max(axis=0) - verts_new.min(axis=0)
    print(f"Normalized: size={new_size}, max_dim={new_size.max():.3f}")
    
    return obj

def create_camera(cam_idx, azimuth_deg, elevation_deg, distance):
    """Create camera pointing at origin"""
    cam_data = bpy.data.cameras.new(f"Cam_{cam_idx:03d}")
    cam_obj = bpy.data.objects.new(f"Cam_{cam_idx:03d}", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    azim = math.radians(azimuth_deg)
    elev = math.radians(elevation_deg)
    
    x = distance * math.cos(elev) * math.sin(azim)
    y = -distance * math.cos(elev) * math.cos(azim)
    z = distance * math.sin(elev)
    
    cam_obj.location = (x, y, z)
    
    direction = -cam_obj.location.normalized()
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    sensor_width = 36
    cam_data.lens = FX * sensor_width / IMAGE_SIZE
    cam_data.sensor_width = sensor_width
    
    return cam_obj, cam_data

def get_opencv_matrices(cam_obj):
    """Get w2c and c2w in OpenCV convention"""
    cam_matrix = cam_obj.matrix_world.copy()
    
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
    
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True

def setup_lighting():
    """Multi-light setup for realistic mouse rendering"""
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.9, 0.9, 0.95, 1)
    bg.inputs[1].default_value = 0.3
    
    # Key light
    light = bpy.data.lights.new("KeyLight", "AREA")
    light.energy = 50
    light.size = 2.0
    light_obj = bpy.data.objects.new("KeyLight", light)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (2, -2, 3)
    
    # Fill light
    fill = bpy.data.lights.new("FillLight", "AREA")
    fill.energy = 20
    fill.size = 3.0
    fill_obj = bpy.data.objects.new("FillLight", fill)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (-2, -1, 2)

def setup_material_simple(obj, texture_path=None):
    """Simple diffuse material."""
    mat = bpy.data.materials.new("MouseMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    if texture_path and os.path.exists(texture_path):
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(texture_path)
        mat.node_tree.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.6
        print(f"Loaded texture: {texture_path}")
    else:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.55, 0.5, 1)
        bsdf.inputs["Roughness"].default_value = 0.8
        if texture_path:
            print(f"WARNING: Texture not found: {texture_path}")
    
    obj.data.materials.clear()
    obj.data.materials.append(mat)

def setup_material_fur(obj):
    """Procedural mouse fur material with noise-based color variation."""
    mat = bpy.data.materials.new("MouseFur")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)
    
    # Output
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (600, 0)
    
    # Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Specular IOR Level"].default_value = 0.1
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    
    # Color mix: base fur color + noise variation
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    mix.location = (0, 0)
    mix.inputs[0].default_value = 0.3  # Factor
    # Light brown base
    mix.inputs[6].default_value = (0.45, 0.35, 0.28, 1)
    # Darker brown variation
    mix.inputs[7].default_value = (0.3, 0.22, 0.15, 1)
    links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    
    # Noise texture for fur pattern
    noise = nodes.new("ShaderNodeTexNoise")
    noise.location = (-200, 0)
    noise.inputs["Scale"].default_value = 15.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.7
    links.new(noise.outputs["Fac"], mix.inputs[0])
    
    # Texture coordinate (object space)
    coord = nodes.new("ShaderNodeTexCoord")
    coord.location = (-400, 0)
    links.new(coord.outputs["Object"], noise.inputs["Vector"])
    
    # Bump for surface detail
    bump = nodes.new("ShaderNodeBump")
    bump.location = (100, -200)
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.01
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    print("Applied procedural fur material")

def setup_material(obj, texture_path=None, material_type="fur"):
    """Material dispatcher."""
    if texture_path:
        setup_material_simple(obj, texture_path)
    elif material_type == "fur":
        setup_material_fur(obj)
    else:
        setup_material_simple(obj)

def render_sample(obj_path, sample_idx, output_dir, config):
    """Render one sample with N views"""
    clear_scene()
    
    obj = load_mammal_obj(str(obj_path))
    
    offset = config["offset"]
    if offset != (0, 0, 0):
        obj.location = Vector(offset)
    
    setup_material(obj, texture_path=args.texture, material_type=args.material)
    setup_render()
    setup_lighting()
    
    # Create cameras
    cameras = []
    num_views = args.num_views
    for i in range(num_views):
        azim = i * (360.0 / num_views)
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
    print(f"MAMMAL {args.num_views}-View Rendering v2.1")
    print(f"Experiment: {args.experiment} - {config['desc']}")
    print(f"Material: {args.material}, Views: {args.num_views}")
    print(f"Samples: {args.num_samples}, Texture: {args.texture}, Mesh: {args.mesh}")
    print("="*60 + "\n")
    
    # Get OBJ files
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.exists():
            print(f"ERROR: Mesh not found: {mesh_path}")
            return
        selected = [mesh_path] * args.num_samples
        print(f"Single mesh mode: {mesh_path.name} x {args.num_samples} samples")
    else:
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
