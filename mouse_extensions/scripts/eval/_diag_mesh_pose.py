"""Diagnostic: Compare flat vs textured mesh geometry for frame 0."""
import numpy as np
import trimesh

# Load flat mesh
flat = trimesh.load(
    '/home/joon/dev/MAMMAL_mouse/results/fitting/'
    'markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/step_2_frame_000000.obj',
    process=False,
)
print(f'Flat mesh: {flat.vertices.shape} vertices, {flat.faces.shape} faces')
print(f'  bounds: {flat.vertices.min(0)} ~ {flat.vertices.max(0)}')
print(f'  centroid: {flat.vertices.mean(0)}')

# Parse template OBJ
def parse_obj(path):
    verts, uvs, faces_v, faces_vt = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v ') and not line.startswith('vt'):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('vt '):
                p = line.split()
                uvs.append([float(p[1]), float(p[2])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                fv, ft = [], []
                for p in parts:
                    idx = p.split('/')
                    fv.append(int(idx[0]) - 1)
                    ft.append(int(idx[1]) - 1)
                faces_v.append(fv)
                faces_vt.append(ft)
    return np.array(verts), np.array(uvs), np.array(faces_v), np.array(faces_vt)

t_verts, t_uvs, t_faces_v, t_faces_vt = parse_obj(
    '/home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj'
)
print(f'\nTemplate: {len(t_verts)} verts, {len(t_uvs)} UVs, {len(t_faces_v)} faces')

# Build uv_to_v mapping
uv_to_v = np.zeros(len(t_uvs), dtype=np.int32)
for fv, ft in zip(t_faces_v, t_faces_vt):
    for vi, ti in zip(fv, ft):
        uv_to_v[ti] = vi

# Check coverage
unique_uv_assigned = len(np.unique(np.concatenate([ft for ft in t_faces_vt])))
print(f'UV indices covered by faces: {unique_uv_assigned} / {len(t_uvs)}')
print(f'uv_to_v range: [{uv_to_v.min()}, {uv_to_v.max()}]')

# Expand frame vertices
frame_verts = flat.vertices
expanded_verts = frame_verts[uv_to_v]
print(f'\nExpanded: {expanded_verts.shape}')

# Compare face triangles
print('\nFace geometry comparison (flat vs textured):')
mismatches = 0
for i in range(min(len(flat.faces), len(t_faces_vt))):
    flat_tri = frame_verts[flat.faces[i]]
    tex_tri = expanded_verts[t_faces_vt[i]]
    if not np.allclose(flat_tri, tex_tri, atol=1e-4):
        mismatches += 1
        if mismatches <= 5:
            print(f'  Face {i}: MISMATCH')
            print(f'    Flat indices: {flat.faces[i]} -> verts: {flat_tri}')
            print(f'    UV indices: {t_faces_vt[i]} -> uv_to_v: {uv_to_v[t_faces_vt[i]]} -> verts: {tex_tri}')
print(f'\nTotal mismatches: {mismatches} / {len(flat.faces)}')

# Also check: is the flat material rendering actually showing correct pose?
# Compare MAMMAL keypoints with mesh centroid
kp_path = '/home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/keypoints_22_3d.npz'
data = np.load(kp_path)
kp_all = data[list(data.keys())[0]]
kp0 = kp_all[0]  # Frame 0 keypoints

# Coordinate transforms — SSOT: mouse_extensions/coordinate_utils.py
from mouse_extensions.coordinate_utils import mammal_to_gslrm

kp0_fl = mammal_to_gslrm(kp0)
mesh_centroid_fl = mammal_to_gslrm(flat.vertices.mean(0))

print(f'\n=== FaceLift Space Alignment ===')
print(f'Keypoint centroid: {kp0_fl.mean(0)}')
print(f'Mesh centroid:     {mesh_centroid_fl}')
print(f'Distance:          {np.linalg.norm(kp0_fl.mean(0) - mesh_centroid_fl):.4f}')

# Check: compare with GS-LRM camera config to see where the cameras look
import json
config_path = '/home/joon/dev/FaceLift/outputs/poc_mesh_gs_pairs/frame_00000/camera_config_frame_00000.json'
with open(config_path) as f:
    config = json.load(f)

print(f'\n=== Camera Info ===')
for name, cam in list(config['gt_cameras'].items())[:2]:
    c2w = np.array(cam['c2w'])
    cam_pos = c2w[:3, 3]
    cam_forward = c2w[:3, 2]  # Z-forward (OpenCV)
    print(f'{name}: pos={cam_pos}, forward={cam_forward}')

print(f'\nMesh centroid (FL): {mesh_centroid_fl}')
print(f'Camera 0 looks at: pos + forward = {np.array(config["gt_cameras"]["cam_000"]["c2w"])[:3,3] + np.array(config["gt_cameras"]["cam_000"]["c2w"])[:3,2]}')
