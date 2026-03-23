"""Pre-compute coordinate maps for neural texture training.

Supports 3 modes:
  uv8:  Standard 8-bit UV (baseline, 255² = 65K distinct positions)
  uv16: Two-pass 16-bit UV (65535² = 4B distinct positions)
  xyz:  3D world coordinates (8-bit × 3ch = 255³ = 16.5M distinct positions)

Usage:
    PYOPENGL_PLATFORM=egl python -m \
        mouse_extensions.scripts.neural_texture.precompute_maps \
        --mode xyz --output-dir outputs/neural_texture/xyz_maps
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
from pathlib import Path

import numpy as np
import pyrender
import trimesh

M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785

def mammal_to_gslrm(xyz):
    return (xyz - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def _make_cam_pose(w2c):
    c2w = np.linalg.inv(np.array(w2c))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ flip


def _render_with_colors(mesh_verts, mesh_faces, colors_uint8, cam_params, size=512):
    """Render mesh with given uint8 vertex colors, return (H,W,4) RGBA."""
    m = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces,
                        vertex_colors=colors_uint8, process=False)
    mp = pyrender.Mesh.from_trimesh(m, smooth=False)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    scene.add(mp)
    cam = pyrender.IntrinsicsCamera(
        fx=cam_params["fx"], fy=cam_params["fy"],
        cx=cam_params["cx"], cy=cam_params["cy"],
        znear=0.01, zfar=100)
    scene.add(cam, pose=_make_cam_pose(cam_params["w2c"]))
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)
    r.delete()
    return color


def precompute_uv8(mesh, cam_params, size=512):
    """Standard 8-bit UV map."""
    uv = mesh.visual.uv
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 0] = np.clip(uv[:, 0] * 255, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(uv[:, 1] * 255, 0, 255).astype(np.uint8)
    colors[:, 2] = 255; colors[:, 3] = 255

    rgba = _render_with_colors(mesh.vertices, mesh.faces, colors, cam_params, size)
    mask = rgba[:, :, 3] > 0
    coord_map = np.zeros((size, size, 2), dtype=np.float32)
    coord_map[:, :, 0] = rgba[:, :, 0].astype(np.float32) / 255.0
    coord_map[:, :, 1] = rgba[:, :, 1].astype(np.float32) / 255.0
    return coord_map, mask, 2  # input_dim=2


def precompute_uv16(mesh, cam_params, size=512):
    """Two-pass 16-bit UV map. Renders high-byte and low-byte separately."""
    uv = mesh.visual.uv
    uv_16 = np.clip(uv * 65535, 0, 65535).astype(np.uint16)

    # Pass 1: high bytes
    hi_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    hi_colors[:, 0] = (uv_16[:, 0] >> 8).astype(np.uint8)
    hi_colors[:, 1] = (uv_16[:, 1] >> 8).astype(np.uint8)
    hi_colors[:, 2] = 255; hi_colors[:, 3] = 255
    hi_rgba = _render_with_colors(mesh.vertices, mesh.faces, hi_colors, cam_params, size)

    # Pass 2: low bytes
    lo_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    lo_colors[:, 0] = (uv_16[:, 0] & 0xFF).astype(np.uint8)
    lo_colors[:, 1] = (uv_16[:, 1] & 0xFF).astype(np.uint8)
    lo_colors[:, 2] = 255; lo_colors[:, 3] = 255
    lo_rgba = _render_with_colors(mesh.vertices, mesh.faces, lo_colors, cam_params, size)

    mask = hi_rgba[:, :, 3] > 0
    # Reconstruct 16-bit UV
    u_16 = hi_rgba[:, :, 0].astype(np.float32) * 256 + lo_rgba[:, :, 0].astype(np.float32)
    v_16 = hi_rgba[:, :, 1].astype(np.float32) * 256 + lo_rgba[:, :, 1].astype(np.float32)

    coord_map = np.zeros((size, size, 2), dtype=np.float32)
    coord_map[:, :, 0] = u_16 / 65535.0
    coord_map[:, :, 1] = v_16 / 65535.0
    return coord_map, mask, 2  # input_dim=2


def precompute_xyz(mesh, cam_params, size=512):
    """3D world coordinates as vertex colors (8-bit × 3 channels)."""
    verts = mesh.vertices  # already in GS-LRM space [-1, 1]
    # Normalize to [0, 1] for uint8 encoding
    v_min = verts.min(0)
    v_max = verts.max(0)
    v_range = v_max - v_min
    v_range = np.maximum(v_range, 1e-6)
    verts_norm = (verts - v_min) / v_range  # [0, 1]

    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 0] = np.clip(verts_norm[:, 0] * 255, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(verts_norm[:, 1] * 255, 0, 255).astype(np.uint8)
    colors[:, 2] = np.clip(verts_norm[:, 2] * 255, 0, 255).astype(np.uint8)
    colors[:, 3] = 255

    rgba = _render_with_colors(mesh.vertices, mesh.faces, colors, cam_params, size)
    mask = rgba[:, :, 3] > 0
    coord_map = np.zeros((size, size, 3), dtype=np.float32)
    coord_map[:, :, 0] = rgba[:, :, 0].astype(np.float32) / 255.0
    coord_map[:, :, 1] = rgba[:, :, 1].astype(np.float32) / 255.0
    coord_map[:, :, 2] = rgba[:, :, 2].astype(np.float32) / 255.0
    return coord_map, mask, 3  # input_dim=3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["uv8", "uv16", "xyz"], default="xyz")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--obj-dir", default="/home/joon/data/synthetic/textured_obj")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5_4")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/neural_texture/{args.mode}_maps"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    precompute_fn = {"uv8": precompute_uv8, "uv16": precompute_uv16, "xyz": precompute_xyz}[args.mode]
    obj_files = sorted(Path(args.obj_dir).glob("step_2_frame_*.obj"))[:args.num_frames]
    print(f"Mode: {args.mode}, {len(obj_files)} OBJs → {output_dir}")

    total = 0
    input_dim = None
    xyz_bounds = None  # for xyz mode: save normalization bounds

    for oi, obj_path in enumerate(obj_files):
        mammal_frame = int(obj_path.stem.split("_")[-1])
        m5_frame = mammal_frame // 5
        cam_path = Path(args.m5_dir) / f"{m5_frame:06d}" / "opencv_cameras.json"
        if not cam_path.exists():
            continue

        mesh = trimesh.load(str(obj_path), process=False)
        mesh.vertices = mammal_to_gslrm(mesh.vertices)

        # Save xyz normalization bounds (needed for inference)
        if args.mode == "xyz" and xyz_bounds is None:
            xyz_bounds = {
                "min": mesh.vertices.min(0).tolist(),
                "max": mesh.vertices.max(0).tolist(),
            }

        cam_data = json.load(open(cam_path))
        for ci, frame in enumerate(cam_data["frames"]):
            coord_map, mask, input_dim = precompute_fn(mesh, frame)
            np.savez_compressed(
                output_dir / f"map_{m5_frame:06d}_cam{ci}.npz",
                coord_map=coord_map, mask=mask,
                m5_frame=m5_frame, cam_idx=ci, mammal_frame=mammal_frame,
            )
            total += 1

        if (oi + 1) % 20 == 0:
            print(f"  {oi+1}/{len(obj_files)} ({total} maps)")

    meta = {
        "mode": args.mode, "input_dim": input_dim, "total_maps": total,
        "num_frames": len(obj_files), "resolution": 512,
    }
    if xyz_bounds:
        meta["xyz_bounds"] = xyz_bounds
    json.dump(meta, open(output_dir / "metadata.json", "w"), indent=2)
    print(f"Done: {total} maps ({args.mode})")


if __name__ == "__main__":
    main()
