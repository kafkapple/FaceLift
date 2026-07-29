"""Pre-compute UV maps for all mesh-camera pairs using pyrender.

Rasterizes MAMMAL mesh from each GT camera, encoding UV coordinates as
vertex colors (R=U, G=V). Output: per-pixel UV maps stored as .npz files.

Usage:
    PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=4 python -m \
        mouse_extensions.scripts.neural_texture.precompute_uv_maps \
        --output-dir outputs/analysis/mouse/neural_texture/uv_maps \
        --num-frames 100
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image
from mouse_extensions.paths import M5_DATA

# MAMMAL → GS-LRM coordinate transform (from coordinate_utils.py)
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785


def mammal_to_gslrm(xyz_mm: np.ndarray) -> np.ndarray:
    return (xyz_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def load_mesh_with_uv_colors(obj_path: str) -> pyrender.Mesh:
    """Load MAMMAL OBJ, transform to GS-LRM space, encode UV as vertex colors."""
    mesh = trimesh.load(obj_path, process=False)

    # Transform vertices to GS-LRM normalized space
    mesh.vertices = mammal_to_gslrm(mesh.vertices)

    # Encode UV as vertex colors (R=U*255, G=V*255, B=255 mask, A=255)
    uv = mesh.visual.uv
    colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 0] = np.clip(uv[:, 0] * 255, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(uv[:, 1] * 255, 0, 255).astype(np.uint8)
    colors[:, 2] = 255
    colors[:, 3] = 255

    uv_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces,
        vertex_colors=colors, process=False,
    )
    return pyrender.Mesh.from_trimesh(uv_mesh, smooth=False)


def render_uv_map(
    mesh_py: pyrender.Mesh, cam_params: dict, img_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Render UV map from a single camera viewpoint.

    Returns:
        uv_map: (H, W, 2) float32 UV coordinates in [0, 1]
        mask: (H, W) bool — True where mesh is visible
    """
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    scene.add(mesh_py)

    cam = pyrender.IntrinsicsCamera(
        fx=cam_params["fx"], fy=cam_params["fy"],
        cx=cam_params["cx"], cy=cam_params["cy"],
        znear=0.01, zfar=100,
    )

    # OpenCV w2c → OpenGL camera pose
    w2c = np.array(cam_params["w2c"])
    c2w = np.linalg.inv(w2c)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    cam_pose = c2w @ flip

    scene.add(cam, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    color, _ = renderer.render(
        scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA
    )
    renderer.delete()

    mask = color[:, :, 3] > 0
    uv_map = np.zeros((img_size, img_size, 2), dtype=np.float32)
    uv_map[:, :, 0] = color[:, :, 0].astype(np.float32) / 255.0  # U
    uv_map[:, :, 1] = color[:, :, 1].astype(np.float32) / 255.0  # V

    return uv_map, mask


def main():
    parser = argparse.ArgumentParser(description="Pre-compute UV maps")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis/mouse/neural_texture/uv_maps")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--obj-dir", type=str,
                        default="/home/joon/data/synthetic/textured_obj")
    parser.add_argument("--m5-dir", type=str,
                        default=str(M5_DATA))
    parser.add_argument("--save-preview", action="store_true",
                        help="Save UV map as PNG for visual inspection")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover OBJ files
    obj_dir = Path(args.obj_dir)
    obj_files = sorted(obj_dir.glob("step_2_frame_*.obj"))[:args.num_frames]
    print(f"Found {len(obj_files)} OBJ files")

    # Frame mapping: OBJ filename → MAMMAL frame → M5 frame
    total_pairs = 0
    for obj_idx, obj_path in enumerate(obj_files):
        mammal_frame = int(obj_path.stem.split("_")[-1])
        m5_frame = mammal_frame // 5

        cam_json_path = Path(args.m5_dir) / f"{m5_frame:06d}" / "opencv_cameras.json"
        if not cam_json_path.exists():
            print(f"  SKIP: M5 frame {m5_frame} — no camera data")
            continue

        cam_data = json.load(open(cam_json_path))

        # Load mesh once per frame
        mesh_py = load_mesh_with_uv_colors(str(obj_path))

        # Render UV map from each camera
        for cam_idx, frame in enumerate(cam_data["frames"]):
            uv_map, mask = render_uv_map(mesh_py, frame)

            # Save as compressed npz
            out_name = f"uv_{m5_frame:06d}_cam{cam_idx:d}.npz"
            np.savez_compressed(
                output_dir / out_name,
                uv_map=uv_map,
                mask=mask,
                m5_frame=m5_frame,
                cam_idx=cam_idx,
                mammal_frame=mammal_frame,
            )

            if args.save_preview and cam_idx == 0 and obj_idx % 20 == 0:
                preview = np.zeros((512, 512, 3), dtype=np.uint8)
                preview[:, :, 0] = (uv_map[:, :, 0] * 255).astype(np.uint8)
                preview[:, :, 1] = (uv_map[:, :, 1] * 255).astype(np.uint8)
                preview[:, :, 2] = mask.astype(np.uint8) * 255
                Image.fromarray(preview).save(
                    output_dir / f"preview_{m5_frame:06d}_cam0.png"
                )

            total_pairs += 1

        if (obj_idx + 1) % 10 == 0:
            print(f"  Processed {obj_idx + 1}/{len(obj_files)} frames ({total_pairs} pairs)")

    print(f"\nDone: {total_pairs} UV maps saved to {output_dir}")

    # Save metadata
    meta = {
        "total_pairs": total_pairs,
        "num_frames": len(obj_files),
        "num_cameras": 6,
        "resolution": 512,
        "coordinate_transform": "mammal_to_gslrm",
        "scene_center": M5_SCENE_CENTER.tolist(),
        "distance_scale": float(M5_DISTANCE_SCALE),
    }
    json.dump(meta, open(output_dir / "metadata.json", "w"), indent=2)


if __name__ == "__main__":
    main()
