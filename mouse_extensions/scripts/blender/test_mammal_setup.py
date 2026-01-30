#!/usr/bin/env python3
"""
Blender에서 MAMMAL 메시 + 카메라 시각화 테스트
GUI 모드에서 실행: blender --python test_mammal_setup.py
"""

import bpy
import math
from mathutils import Vector

# ==============================================================================
# Configuration
# ==============================================================================

MAMMAL_OBJ = str(Path.home()) + "/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/step_2_frame_000000.obj"

# FaceLift camera settings
NUM_CAMERAS = 6
CAMERA_DISTANCE = 2.7
ELEVATION = 20  # degrees
FX = 549
IMAGE_SIZE = 512

# ==============================================================================
# Functions
# ==============================================================================

def clear_scene():
    """Clear all objects"""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def load_and_center_mesh(obj_path):
    """Load OBJ and center at origin"""
    # Import OBJ
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.name = "MouseMesh"
    
    # Get bounding box center
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8
    
    # Move to origin
    obj.location = -center
    
    # Convert mm to meters and scale for viewing
    # MAMMAL is in mm, ~100mm body → 0.1m
    # For camera distance 2.7, object should be ~0.25 units
    # Current size ~100mm = 0.1m, target ~0.25
    scale_factor = 0.25 / 0.1  # = 2.5
    # But we need to convert mm to Blender units (meters) first
    # 100mm = 0.1m, so scale = 0.001 (mm to m) * 2.5 (to target size)
    total_scale = 0.001 * 2.5
    obj.scale = (total_scale, total_scale, total_scale)
    
    # Apply transforms
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    return obj

def create_orbit_cameras(num_cams, distance, elevation):
    """Create orbit cameras around origin"""
    cameras = []
    
    for i in range(num_cams):
        azimuth = i * (360.0 / num_cams)
        azim_rad = math.radians(azimuth)
        elev_rad = math.radians(elevation)
        
        # Position
        x = distance * math.cos(elev_rad) * math.sin(azim_rad)
        y = -distance * math.cos(elev_rad) * math.cos(azim_rad)  # Blender Y forward
        z = distance * math.sin(elev_rad)
        
        # Create camera
        cam_data = bpy.data.cameras.new(f"Camera_{i:03d}")
        cam_obj = bpy.data.objects.new(f"Camera_{i:03d}", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
        
        cam_obj.location = (x, y, z)
        
        # Point at origin
        direction = Vector((0, 0, 0)) - cam_obj.location
        rot_quat = direction.to_track_quat("-Z", "Y")
        cam_obj.rotation_euler = rot_quat.to_euler()
        
        # Set focal length
        cam_data.lens = FX * 36 / IMAGE_SIZE  # Convert to mm focal length
        
        cameras.append(cam_obj)
        print(f"Camera {i}: azimuth={azimuth:.0f}°, pos=({x:.2f}, {y:.2f}, {z:.2f})")
    
    return cameras

def add_coordinate_axes():
    """Add XYZ axes for reference"""
    # X axis (red)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1, location=(0.5, 0, 0))
    x_axis = bpy.context.active_object
    x_axis.name = "X_Axis"
    x_axis.rotation_euler = (0, math.radians(90), 0)
    mat_x = bpy.data.materials.new("Red")
    mat_x.diffuse_color = (1, 0, 0, 1)
    x_axis.data.materials.append(mat_x)
    
    # Y axis (green)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1, location=(0, 0.5, 0))
    y_axis = bpy.context.active_object
    y_axis.name = "Y_Axis"
    y_axis.rotation_euler = (math.radians(90), 0, 0)
    mat_y = bpy.data.materials.new("Green")
    mat_y.diffuse_color = (0, 1, 0, 1)
    y_axis.data.materials.append(mat_y)
    
    # Z axis (blue)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.02, depth=1, location=(0, 0, 0.5))
    z_axis = bpy.context.active_object
    z_axis.name = "Z_Axis"
    mat_z = bpy.data.materials.new("Blue")
    mat_z.diffuse_color = (0, 0, 1, 1)
    z_axis.data.materials.append(mat_z)

def setup_simple_lighting():
    """Add basic lighting"""
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = 2
    light_obj = bpy.data.objects.new("Sun", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (5, 5, 10)

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("MAMMAL Mesh + Camera Visualization Test")
    print("="*60 + "\n")
    
    # Clear and setup
    clear_scene()
    
    # Load mesh
    print("Loading MAMMAL mesh...")
    mesh = load_and_center_mesh(MAMMAL_OBJ)
    print(f"Mesh loaded: {mesh.name}")
    
    # Get final bounds
    bbox = [mesh.matrix_world @ Vector(corner) for corner in mesh.bound_box]
    min_c = Vector((min(v[i] for v in bbox) for i in range(3)))
    max_c = Vector((max(v[i] for v in bbox) for i in range(3)))
    print(f"Final bounds: min={min_c}, max={max_c}")
    print(f"Final size: {max_c - min_c}")
    
    # Create cameras
    print("\nCreating cameras...")
    cameras = create_orbit_cameras(NUM_CAMERAS, CAMERA_DISTANCE, ELEVATION)
    
    # Add reference
    add_coordinate_axes()
    setup_simple_lighting()
    
    # Set first camera as active
    bpy.context.scene.camera = cameras[0]
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("- Use viewport to check mesh visibility")
    print("- Press Numpad 0 to view through camera")
    print("- Check if mouse mesh is visible in camera views")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
