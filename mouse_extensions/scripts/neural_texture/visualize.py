"""Visualize Neural Texture results: grid comparison + novel view renders.

Creates side-by-side comparison of baseline (uniform color mesh) vs neural texture
renders at GT camera views and novel viewpoints (bottom, top, front_low, side_low).

Usage:
    PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=4 python -m \
        mouse_extensions.scripts.neural_texture.visualize \
        --checkpoint outputs/analysis/mouse/neural_texture/checkpoints/best.pt \
        --frame 0
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
from PIL import Image, ImageDraw, ImageFont

from mouse_extensions.model.neural_texture import build_neural_texture

# Coordinate transform
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785

def mammal_to_gslrm(xyz):
    return (xyz - M5_SCENE_CENTER) * M5_DISTANCE_SCALE

# Novel view camera definitions (spherical → c2w)
NOVEL_VIEWS = {
    "bottom":    {"elevation": -70, "azimuth": 0},
    "top":       {"elevation": 70,  "azimuth": 0},
    "front_low": {"elevation": -30, "azimuth": 0},
    "side_low":  {"elevation": -30, "azimuth": 90},
}

TURNTABLE_RADIUS = 2.7
TURNTABLE_FX = 549.0


def spherical_to_c2w(elev_deg, azim_deg, radius=TURNTABLE_RADIUS):
    """Create look-at c2w matrix from spherical coordinates."""
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)

    # Camera position
    x = radius * np.cos(elev) * np.cos(azim)
    y = radius * np.cos(elev) * np.sin(azim)
    z = radius * np.sin(elev)
    cam_pos = np.array([x, y, z])

    # Look-at origin
    target = np.array([0, 0, 0])
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    # Up vector (world Z-up)
    world_up = np.array([0, 0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0, 1.0, 0])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # c2w (OpenCV convention: Z forward, Y down, X right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = -up
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos

    return c2w


def load_mesh_transformed(obj_path):
    """Load MAMMAL mesh and transform to GS-LRM space."""
    mesh = trimesh.load(obj_path, process=False)
    mesh.vertices = mammal_to_gslrm(mesh.vertices)
    return mesh


def render_with_vertex_colors(mesh, vertex_colors, cam_params=None,
                              c2w=None, img_size=512):
    """Render mesh with given vertex colors using pyrender.

    Args:
        mesh: trimesh.Trimesh (already in GS-LRM space)
        vertex_colors: (N_verts, 3) float RGB in [0, 1] or (N_verts, 4) RGBA uint8
        cam_params: dict with fx, fy, cx, cy, w2c (for GT cameras)
        c2w: 4×4 camera-to-world matrix (for novel cameras, OpenCV convention)
    """
    # Prepare colors
    if vertex_colors.dtype == np.float32 or vertex_colors.dtype == np.float64:
        colors_uint8 = (np.clip(vertex_colors, 0, 1) * 255).astype(np.uint8)
    else:
        colors_uint8 = vertex_colors

    if colors_uint8.shape[1] == 3:
        alpha = np.full((len(colors_uint8), 1), 255, dtype=np.uint8)
        colors_uint8 = np.concatenate([colors_uint8, alpha], axis=1)

    colored_mesh = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces,
        vertex_colors=colors_uint8, process=False,
    )
    mesh_py = pyrender.Mesh.from_trimesh(colored_mesh, smooth=False)

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.8, 0.8, 0.8])
    scene.add(mesh_py)

    # Light
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0)

    if cam_params is not None:
        cam = pyrender.IntrinsicsCamera(
            fx=cam_params["fx"], fy=cam_params["fy"],
            cx=cam_params["cx"], cy=cam_params["cy"],
            znear=0.01, zfar=100,
        )
        w2c = np.array(cam_params["w2c"])
        c2w_mat = np.linalg.inv(w2c)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        cam_pose = c2w_mat @ flip
    else:
        cam = pyrender.IntrinsicsCamera(
            fx=TURNTABLE_FX, fy=TURNTABLE_FX, cx=256, cy=256,
            znear=0.01, zfar=100,
        )
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        cam_pose = c2w @ flip

    scene.add(cam, pose=cam_pose)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(img_size, img_size)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA)
    renderer.delete()

    return color[:, :, :3]  # RGB uint8


def render_neural_texture(mesh, model, device, cam_params=None, c2w=None, img_size=512):
    """Render mesh with neural texture: rasterize UV → query MLP → reconstruct image."""
    # Step 1: Get UV map by rendering mesh with UV-as-color
    uv = mesh.visual.uv
    uv_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    uv_colors[:, 0] = np.clip(uv[:, 0] * 255, 0, 255).astype(np.uint8)
    uv_colors[:, 1] = np.clip(uv[:, 1] * 255, 0, 255).astype(np.uint8)
    uv_colors[:, 2] = 255
    uv_colors[:, 3] = 255

    uv_render = render_with_vertex_colors(
        mesh, uv_colors, cam_params=cam_params, c2w=c2w, img_size=img_size,
    )

    # Step 2: Extract valid pixels and their UV coordinates
    # Mask: blue channel = 255 where mesh is visible (from our UV encoding)
    # Actually: check if any non-white pixel exists
    mask = np.any(uv_render < 250, axis=-1)

    if mask.sum() == 0:
        return np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    valid_y, valid_x = np.where(mask)
    uv_coords = np.zeros((len(valid_y), 2), dtype=np.float32)
    uv_coords[:, 0] = uv_render[valid_y, valid_x, 0].astype(np.float32) / 255.0
    uv_coords[:, 1] = uv_render[valid_y, valid_x, 1].astype(np.float32) / 255.0

    # Step 3: Query MLP
    model.eval()
    with torch.no_grad():
        uv_tensor = torch.from_numpy(uv_coords).float().to(device)
        pred_rgb = model(uv_tensor).cpu().numpy()

    # Step 4: Reconstruct image
    result = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    result[valid_y, valid_x] = (pred_rgb * 255).clip(0, 255).astype(np.uint8)

    return result


def create_comparison_grid(images, labels, row_labels=None, cell_size=256):
    """Create a grid of images with labels.

    Args:
        images: list of lists — images[row][col] = numpy array HxWx3
        labels: list of column labels
        row_labels: list of row labels (optional)
    """
    n_rows = len(images)
    n_cols = len(labels)

    # Resize all images
    cells = []
    for row in images:
        row_cells = []
        for img in row:
            if img is not None:
                pil = Image.fromarray(img).resize((cell_size, cell_size), Image.LANCZOS)
            else:
                pil = Image.new("RGB", (cell_size, cell_size), (200, 200, 200))
                draw = ImageDraw.Draw(pil)
                draw.text((cell_size // 3, cell_size // 2), "N/A",
                          fill=(128, 128, 128))
            row_cells.append(pil)
        cells.append(row_cells)

    # Compose grid
    label_h = 30
    row_label_w = 80 if row_labels else 0
    grid_w = row_label_w + n_cols * cell_size
    grid_h = label_h + n_rows * cell_size

    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # Column labels
    for j, label in enumerate(labels):
        x = row_label_w + j * cell_size + cell_size // 4
        draw.text((x, 5), label, fill=(200, 200, 200))

    # Row labels and cells
    for i, row in enumerate(cells):
        y = label_h + i * cell_size
        if row_labels:
            draw.text((5, y + cell_size // 2 - 10), row_labels[i],
                       fill=(200, 200, 200))
        for j, cell in enumerate(row):
            x = row_label_w + j * cell_size
            grid.paste(cell, (x, y))

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize Neural Texture")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/analysis/mouse/neural_texture/checkpoints/best.pt")
    parser.add_argument("--obj-dir", type=str,
                        default="/home/joon/data/synthetic/textured_obj")
    parser.add_argument("--m5-dir", type=str,
                        default="/home/joon/data/preprocessed/FaceLift_mouse/M5_4")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/analysis/mouse/neural_texture/visualizations")
    parser.add_argument("--frame", type=int, default=0,
                        help="M5 frame index to visualize")
    parser.add_argument("--cell-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (supports both NeuralTextureMLP and train_exp Model)
    print("Loading model from", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    input_dim = ckpt.get("input_dim", config.get("input_dim", 2))

    # Detect model type from state_dict keys
    state_keys = list(ckpt["model_state_dict"].keys())
    if any(k.startswith("uv_encoder") for k in state_keys):
        # NeuralTextureMLP from model/neural_texture.py
        model = build_neural_texture(
            use_pose=config.get("use_pose", False),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 6),
            num_freqs=config.get("num_freqs", 8),
        ).to(device)
    else:
        # Model from train_exp.py
        from mouse_extensions.scripts.neural_texture.train_exp import build_model
        model = build_model(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 6),
            num_freqs=config.get("num_freqs", 10),
        ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}, input_dim={input_dim}")

    # Load mesh for requested frame
    mammal_frame = args.frame * 5
    obj_path = Path(args.obj_dir) / f"step_2_frame_{mammal_frame:06d}.obj"
    if not obj_path.exists():
        obj_path = Path(args.obj_dir) / "step_2_frame_000000.obj"
        print(f"  Frame {args.frame} mesh not found, using frame 0")
    mesh = load_mesh_transformed(str(obj_path))
    print(f"  Mesh: {len(mesh.vertices)} vertices")

    # Load GT cameras
    cam_path = Path(args.m5_dir) / f"{args.frame:06d}" / "opencv_cameras.json"
    cam_data = json.load(open(cam_path))

    # === Render GT views ===
    print("Rendering GT views...")
    gt_renders = []
    baseline_renders = []
    neural_renders = []

    # Baseline: uniform gray color
    baseline_colors = np.full((len(mesh.vertices), 3), 180, dtype=np.uint8)

    for i, frame in enumerate(cam_data["frames"]):
        # GT image
        gt_path = Path(args.m5_dir) / f"{args.frame:06d}" / "images" / f"cam_{i:03d}.png"
        gt_img = np.array(Image.open(gt_path))[:, :, :3]
        gt_renders.append(gt_img)

        # Baseline render (uniform gray)
        baseline = render_with_vertex_colors(mesh, baseline_colors, cam_params=frame)
        baseline_renders.append(baseline)

        # Neural texture render
        neural = render_neural_texture(mesh, model, device, cam_params=frame)
        neural_renders.append(neural)

    # === Render novel views ===
    print("Rendering novel views...")
    novel_gt = []
    novel_baseline = []
    novel_neural = []
    novel_names = []

    for name, params in NOVEL_VIEWS.items():
        c2w = spherical_to_c2w(params["elevation"], params["azimuth"])
        novel_names.append(name)

        # No GT for novel views
        novel_gt.append(None)

        # Baseline
        baseline = render_with_vertex_colors(mesh, baseline_colors, c2w=c2w)
        novel_baseline.append(baseline)

        # Neural texture
        neural = render_neural_texture(mesh, model, device, c2w=c2w)
        novel_neural.append(neural)

    # === Create comparison grid ===
    print("Creating grid...")
    col_labels = ["GT", "Baseline", "Neural Texture"]
    row_labels = [f"Cam {i}" for i in range(6)] + novel_names

    images = []
    for i in range(6):
        images.append([gt_renders[i], baseline_renders[i], neural_renders[i]])
    for i in range(len(novel_names)):
        images.append([novel_gt[i], novel_baseline[i], novel_neural[i]])

    grid = create_comparison_grid(images, col_labels, row_labels, args.cell_size)
    grid_path = output_dir / f"grid_f{args.frame:04d}.png"
    grid.save(grid_path)
    print(f"  Saved: {grid_path}")

    # === Save individual renders ===
    for i in range(6):
        Image.fromarray(neural_renders[i]).save(
            output_dir / f"neural_cam{i}_f{args.frame:04d}.png")
    for i, name in enumerate(novel_names):
        Image.fromarray(novel_neural[i]).save(
            output_dir / f"neural_{name}_f{args.frame:04d}.png")

    print("Done!")


if __name__ == "__main__":
    main()
