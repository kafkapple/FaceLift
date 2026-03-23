"""Generate novel view pseudo-GT renders using trained neural texture.

For each mesh frame, renders the neural-textured mesh at novel viewpoints
(bottom, top, front_low, side_low) + GT camera views. These renders serve as
pseudo-GT for DiFix Type 2 training pairs.

Usage:
    PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=7 python -m \
        mouse_extensions.scripts.neural_texture.generate_pseudo_gt \
        --checkpoint outputs/analysis/mouse/neural_texture/runs/uv16_lpips_500ep/best.pt \
        --output-dir outputs/analysis/mouse/neural_texture/pseudo_gt
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import json
from pathlib import Path

import numpy as np
import pyrender
import trimesh
import torch
from PIL import Image

M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785
TURNTABLE_RADIUS = 2.7
TURNTABLE_FX = 549.0

NOVEL_VIEWS = {
    "bottom":    {"elevation": -70, "azimuth": 0},
    "top":       {"elevation": 70,  "azimuth": 0},
    "front_low": {"elevation": -30, "azimuth": 0},
    "side_low":  {"elevation": -30, "azimuth": 90},
}


def mammal_to_gslrm(xyz):
    return (xyz - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def spherical_to_c2w(elev_deg, azim_deg, radius=TURNTABLE_RADIUS):
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    x = radius * np.cos(elev) * np.cos(azim)
    y = radius * np.cos(elev) * np.sin(azim)
    z = radius * np.sin(elev)
    cam_pos = np.array([x, y, z])
    target = np.zeros(3)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0, 0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1.0, 0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = -up
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos
    return c2w


def _make_pyrender_pose(c2w):
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ flip


def render_neural_texture_image(mesh, model, device, cam_params=None, c2w=None, size=512):
    """Render mesh with neural texture: rasterize UV → query MLP → image."""
    # UV-as-color mesh
    uv = mesh.visual.uv
    uv_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    uv_colors[:, 0] = np.clip(uv[:, 0] * 255, 0, 255).astype(np.uint8)
    uv_colors[:, 1] = np.clip(uv[:, 1] * 255, 0, 255).astype(np.uint8)
    uv_colors[:, 2] = 255
    uv_colors[:, 3] = 255

    uv_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                               vertex_colors=uv_colors, process=False)
    mesh_py = pyrender.Mesh.from_trimesh(uv_mesh, smooth=False)

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1, 1, 1])
    scene.add(mesh_py)

    if cam_params is not None:
        cam = pyrender.IntrinsicsCamera(fx=cam_params["fx"], fy=cam_params["fy"],
                                         cx=cam_params["cx"], cy=cam_params["cy"],
                                         znear=0.01, zfar=100)
        w2c = np.array(cam_params["w2c"])
        pose = _make_pyrender_pose(np.linalg.inv(w2c))
    else:
        cam = pyrender.IntrinsicsCamera(fx=TURNTABLE_FX, fy=TURNTABLE_FX,
                                         cx=256, cy=256, znear=0.01, zfar=100)
        pose = _make_pyrender_pose(c2w)

    scene.add(cam, pose=pose)
    r = pyrender.OffscreenRenderer(size, size)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)
    r.delete()

    # Extract UV and query MLP
    mask = np.any(color[:, :, :3] < 250, axis=-1)
    if mask.sum() == 0:
        return np.ones((size, size, 3), dtype=np.uint8) * 255, mask

    vy, vx = np.where(mask)
    uv_coords = np.zeros((len(vy), 2), dtype=np.float32)
    uv_coords[:, 0] = color[vy, vx, 0].astype(np.float32) / 255.0
    uv_coords[:, 1] = color[vy, vx, 1].astype(np.float32) / 255.0

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(uv_coords).float().to(device)).cpu().numpy()

    result = np.ones((size, size, 3), dtype=np.uint8) * 255
    result[vy, vx] = (pred * 255).clip(0, 255).astype(np.uint8)
    return result, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/neural_texture/pseudo_gt")
    parser.add_argument("--obj-dir", default="/home/joon/data/synthetic/textured_obj")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5_4")
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--render-gt-views", action="store_true",
                        help="Also render at GT camera positions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    input_dim = ckpt.get("input_dim", 2)

    from mouse_extensions.scripts.neural_texture.train_exp import build_model
    model = build_model(
        input_dim=input_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 6),
        num_freqs=config.get("num_freqs", 10),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: epoch={ckpt.get('epoch')}, val={ckpt.get('val_loss', '?'):.4f}")

    # Process frames
    obj_files = sorted(Path(args.obj_dir).glob("step_2_frame_*.obj"))[:args.num_frames]
    print(f"Generating pseudo-GT for {len(obj_files)} frames...")

    total = 0
    for oi, obj_path in enumerate(obj_files):
        mammal_frame = int(obj_path.stem.split("_")[-1])
        m5_frame = mammal_frame // 5

        mesh = trimesh.load(str(obj_path), process=False)
        mesh.vertices = mammal_to_gslrm(mesh.vertices)

        # Novel views
        for view_name, params in NOVEL_VIEWS.items():
            c2w = spherical_to_c2w(params["elevation"], params["azimuth"])
            img, mask = render_neural_texture_image(mesh, model, device, c2w=c2w)

            view_dir = output_dir / "novel" / view_name
            view_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(view_dir / f"f{m5_frame:06d}.png")
            total += 1

        # GT camera views (optional)
        if args.render_gt_views:
            cam_path = Path(args.m5_dir) / f"{m5_frame:06d}" / "opencv_cameras.json"
            if cam_path.exists():
                cam_data = json.load(open(cam_path))
                for ci, frame in enumerate(cam_data["frames"]):
                    img, mask = render_neural_texture_image(
                        mesh, model, device, cam_params=frame)
                    gt_dir = output_dir / "gt_view" / f"cam{ci}"
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(img).save(gt_dir / f"f{m5_frame:06d}.png")
                    total += 1

        if (oi + 1) % 20 == 0:
            print(f"  {oi+1}/{len(obj_files)} ({total} renders)")

    # Save metadata
    meta = {
        "checkpoint": str(args.checkpoint),
        "total_renders": total,
        "num_frames": len(obj_files),
        "novel_views": list(NOVEL_VIEWS.keys()),
        "gt_views_rendered": args.render_gt_views,
    }
    json.dump(meta, open(output_dir / "metadata.json", "w"), indent=2)
    print(f"\nDone: {total} renders saved to {output_dir}")


if __name__ == "__main__":
    main()
