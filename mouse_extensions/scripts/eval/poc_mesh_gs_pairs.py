#!/usr/bin/env python3
"""
PoC: MAMMAL Mesh vs GS-LRM Image Pair Collection

Two-phase script for collecting image pairs at identical camera poses.
Phase 1 (facelift env): GS-LRM inference + rendering + save cameras
Phase 2 (mammal_stable env): MAMMAL mesh rendering at saved cameras
Phase 3 (any env): Comparison grid generation

Usage:
    # Phase 1: GS-LRM rendering (facelift conda env)
    conda activate facelift
    python poc_mesh_gs_pairs.py --phase gslrm --frames 0 100 200

    # Phase 2: MAMMAL mesh rendering (mammal_stable conda env)
    conda activate mammal_stable
    python poc_mesh_gs_pairs.py --phase mammal --frames 0 100 200

    # Phase 3: Comparison grids (any env with cv2/numpy)
    python poc_mesh_gs_pairs.py --phase compare --frames 0 100 200
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# === Shared Constants ===
# MAMMAL world coords (mm) -> FaceLift normalized space
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785  # ~0.008781

MAMMAL_OBJ_DIR = (
    "/home/joon/dev/MAMMAL_mouse/results/fitting/"
    "markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj"
)
MAMMAL_TEXTURE_PNG = "/home/joon/dev/MAMMAL_mouse/exports/texture_final.png"
MAMMAL_TEXTURED_OBJ = "/home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj"
M5_DATA_DIR = "/home/joon/data/preprocessed/FaceLift_mouse/M5"
GSLRM_CONFIG = "/home/joon/dev/FaceLift/configs/base/gslrm_mouse.yaml"
GSLRM_CHECKPOINT = "/home/joon/dev/FaceLift/checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt"
OUTPUT_BASE = "/home/joon/dev/FaceLift/outputs/poc_mesh_gs_pairs"

# Novel view definitions (spherical coords in FaceLift normalized space)
# elevation: positive = looking down from above, negative = looking up from below
NOVEL_VIEWS = {
    "bottom": {"elevation": -70.0, "azimuth": 0.0},
    "top": {"elevation": 70.0, "azimuth": 0.0},
    "front_low": {"elevation": -30.0, "azimuth": 0.0},
    "side_low": {"elevation": -30.0, "azimuth": 90.0},
}
RENDER_RESOLUTION = 384
TURNTABLE_RADIUS = 2.7  # GS-LRM default


def mammal_to_facelift(points_mm: np.ndarray) -> np.ndarray:
    """Transform MAMMAL world coordinates (mm) to FaceLift normalized space."""
    return (points_mm - M5_SCENE_CENTER) * M5_DISTANCE_SCALE


def facelift_to_mammal(points_fl: np.ndarray) -> np.ndarray:
    """Transform FaceLift normalized coords back to MAMMAL world (mm)."""
    return points_fl / M5_DISTANCE_SCALE + M5_SCENE_CENTER


def get_frame_obj_path(frame_idx: int) -> str:
    """Get MAMMAL OBJ file path for a given M5 frame index.

    MAMMAL fitting outputs OBJ at step=5 (100fps video → 20fps fitting).
    M5 sample N corresponds to video frame N*5 → step_2_frame_{N*5:06d}.obj
    """
    mammal_frame = frame_idx * 5
    obj_name = f"step_2_frame_{mammal_frame:06d}.obj"
    return os.path.join(MAMMAL_OBJ_DIR, obj_name)


def get_sample_dir(frame_idx: int) -> str:
    """Get M5 data sample directory for a given frame index."""
    # M5 data uses 6-digit zero-padded directories: 000000, 000100, etc.
    return os.path.join(M5_DATA_DIR, f"{frame_idx:06d}")


def make_look_at_c2w(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])
) -> np.ndarray:
    """Create a camera-to-world matrix looking from eye toward target.

    Uses OpenCV convention: X-right, Y-down, Z-forward (into screen).
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([1, 0, 0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # OpenCV: Y-down
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def generate_novel_cameras(
    center: np.ndarray = np.zeros(3),
    radius: float = TURNTABLE_RADIUS,
    resolution: int = RENDER_RESOLUTION,
) -> dict:
    """Generate novel view cameras around a center point.

    Returns dict of {view_name: {"c2w": [4,4], "fxfycxcy": [4]}}.
    """
    # Use GT camera intrinsics (scaled to render resolution)
    # Original: fx=fy=548.99 at 512x512 → scales proportionally with resolution
    gt_fx_at_512 = 548.99
    fx = fy = gt_fx_at_512 * (resolution / 512.0)
    cx = cy = resolution / 2.0
    fxfycxcy = np.array([fx, fy, cx, cy])

    cameras = {}
    for name, params in NOVEL_VIEWS.items():
        elev_rad = np.radians(params["elevation"])
        azim_rad = np.radians(params["azimuth"])

        # Spherical to Cartesian
        x = radius * np.cos(elev_rad) * np.cos(azim_rad)
        y = radius * np.cos(elev_rad) * np.sin(azim_rad)
        z = radius * np.sin(elev_rad)

        eye = center + np.array([x, y, z])
        c2w = make_look_at_c2w(eye, center)

        cameras[name] = {
            "c2w": c2w.tolist(),
            "fxfycxcy": fxfycxcy.tolist(),
            "resolution": resolution,
        }
    return cameras


def save_camera_config(
    output_dir: str,
    gt_cameras: dict,
    novel_cameras: dict,
    frame_idx: int,
):
    """Save camera configuration JSON for cross-env sharing."""
    config = {
        "frame_idx": frame_idx,
        "coordinate_system": "facelift_normalized",
        "transform": {
            "M5_SCENE_CENTER": M5_SCENE_CENTER.tolist(),
            "M5_DISTANCE_SCALE": M5_DISTANCE_SCALE,
            "formula": "point_fl = (point_mammal_mm - center) * scale",
        },
        "gt_cameras": gt_cameras,
        "novel_cameras": novel_cameras,
    }
    path = os.path.join(output_dir, f"camera_config_frame_{frame_idx:05d}.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved camera config: {path}")
    return path


# ============================================================
# Phase 1: GS-LRM Rendering (facelift env)
# ============================================================
def phase_gslrm(frame_indices: list, output_dir: str, checkpoint: str, config: str = GSLRM_CONFIG):
    """Run GS-LRM inference and render at GT + novel camera poses."""
    import torch

    # Import FaceLift modules
    sys.path.insert(0, "/home/joon/dev/FaceLift")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
    from gslrm.model.gaussians_renderer import render_opencv_cam

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load GS-LRM pipeline
    print("Loading GS-LRM pipeline...")
    pipeline = GSLRMInference(
        config_path=config,
        checkpoint_path=checkpoint,
        device=device,
    )

    for frame_idx in frame_indices:
        print(f"\n=== Frame {frame_idx} ===")
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx:05d}")
        os.makedirs(os.path.join(frame_dir, "gslrm_gt"), exist_ok=True)
        os.makedirs(os.path.join(frame_dir, "gslrm_novel"), exist_ok=True)
        os.makedirs(os.path.join(frame_dir, "gt_rgb"), exist_ok=True)

        # Load sample data
        sample_dir = get_sample_dir(frame_idx)
        if not os.path.exists(sample_dir):
            print(f"  WARNING: Sample dir not found: {sample_dir}")
            continue

        images, c2ws, fxfycxcys, index = load_sample_data(
            sample_dir, image_size=RENDER_RESOLUTION, device=device
        )

        # Run GS-LRM inference (6-view input)
        print("  Running GS-LRM inference...")
        result = pipeline.predict(images, c2ws, fxfycxcys, index)
        gaussians = result["gaussians"][0]

        # Extract GT camera info
        gt_cameras = {}
        c2ws_np = c2ws[0].cpu().numpy()  # [V, 4, 4]
        fxfycxcys_np = fxfycxcys[0].cpu().numpy()  # [V, 4]

        for v in range(c2ws_np.shape[0]):
            cam_name = f"cam_{v:03d}"
            gt_cameras[cam_name] = {
                "c2w": c2ws_np[v].tolist(),
                "fxfycxcy": fxfycxcys_np[v].tolist(),
                "resolution": RENDER_RESOLUTION,
            }

            # Render GS-LRM at GT camera
            c2w_t = torch.from_numpy(c2ws_np[v].astype(np.float32)).to(device)
            fxfy_t = torch.tensor(fxfycxcys_np[v], dtype=torch.float32, device=device)

            rendered = render_opencv_cam(
                gaussians,
                height=RENDER_RESOLUTION,
                width=RENDER_RESOLUTION,
                C2W=c2w_t,
                fxfycxcy=fxfy_t,
                bg_color=(1.0, 1.0, 1.0),
            )
            # Save rendered image
            img = rendered["render"].detach().cpu().permute(1, 2, 0).numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            save_path = os.path.join(frame_dir, "gslrm_gt", f"{cam_name}.png")
            _save_image(img, save_path)

            # Copy GT RGB image (M5 uses cam_NNN.png in images/ subdir)
            gt_img_src = os.path.join(sample_dir, "images", f"cam_{v:03d}.png")
            if not os.path.exists(gt_img_src):
                gt_img_src = os.path.join(sample_dir, f"cam_{v:03d}.png")
            if os.path.exists(gt_img_src):
                import shutil
                gt_img_dst = os.path.join(frame_dir, "gt_rgb", f"{cam_name}.png")
                shutil.copy2(gt_img_src, gt_img_dst)

        # Generate and render novel views
        novel_cameras = generate_novel_cameras()
        for name, cam_params in novel_cameras.items():
            c2w_np = np.array(cam_params["c2w"], dtype=np.float32)
            fxfy_np = np.array(cam_params["fxfycxcy"], dtype=np.float32)

            c2w_t = torch.from_numpy(c2w_np).to(device)
            fxfy_t = torch.from_numpy(fxfy_np).to(device)

            rendered = render_opencv_cam(
                gaussians,
                height=cam_params["resolution"],
                width=cam_params["resolution"],
                C2W=c2w_t,
                fxfycxcy=fxfy_t,
                bg_color=(1.0, 1.0, 1.0),
            )
            img = rendered["render"].detach().cpu().permute(1, 2, 0).numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            save_path = os.path.join(frame_dir, "gslrm_novel", f"{name}.png")
            _save_image(img, save_path)

        # Save camera config for MAMMAL rendering phase
        save_camera_config(frame_dir, gt_cameras, novel_cameras, frame_idx)

    print(f"\nPhase 1 complete. Outputs in: {output_dir}")


# ============================================================
# Phase 2: MAMMAL Mesh Rendering (mammal_stable env)
# ============================================================
def _parse_textured_obj(obj_path: str):
    """Parse OBJ with f v/vt format. Returns vertices, uvs, faces_v, faces_vt."""
    vertices = []
    uvs = []
    faces_v = []
    faces_vt = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):
                parts = line.split()
                uvs.append([float(parts[1]), float(parts[2])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                fv = []
                ft = []
                for p in parts:
                    indices = p.split('/')
                    fv.append(int(indices[0]) - 1)  # 1-indexed → 0-indexed
                    ft.append(int(indices[1]) - 1)
                faces_v.append(fv)
                faces_vt.append(ft)
    return (
        np.array(vertices),
        np.array(uvs),
        np.array(faces_v),
        np.array(faces_vt),
    )


# Cache parsed template data (same for all frames)
_TEMPLATE_CACHE = {}


def _load_textured_mesh(frame_obj_path: str):
    """Load per-frame geometry with UV texture from template.

    Strategy (v2 — fixed vertex mapping):
    1. Load template OBJ via trimesh (handles v/vt UV expansion natively: 14,522→15,399)
    2. Build expanded→original vertex mapping by face correspondence
    3. For per-frame OBJ, swap vertex positions using the correct mapping

    Previous approach manually built faces from faces_vt indices, but trimesh
    internally reindexes vertices during OBJ loading. The manual faces_vt didn't
    match trimesh's internal face indices, causing vertex positions to be scrambled
    — producing the "skeleton-like" distortion artifact.
    """
    import trimesh
    from PIL import Image

    global _TEMPLATE_CACHE

    # Load template mesh via trimesh once and cache
    if not _TEMPLATE_CACHE:
        print(f"  Loading UV template via trimesh: {MAMMAL_TEXTURED_OBJ}")
        template_mesh = trimesh.load(MAMMAL_TEXTURED_OBJ, process=False)
        print(f"  Template: {len(template_mesh.vertices)} expanded verts, "
              f"{len(template_mesh.faces)} faces, UV: {template_mesh.visual.uv.shape}")

        # Parse original vertex indices from OBJ faces (v index in f v/vt entries)
        faces_v_orig = []
        with open(MAMMAL_TEXTURED_OBJ) as f:
            for line in f:
                if line.startswith('f '):
                    parts = line.split()[1:]
                    fv = [int(p.split('/')[0]) - 1 for p in parts]
                    faces_v_orig.append(fv)
        faces_v_orig = np.array(faces_v_orig)

        # Build mapping: trimesh expanded vertex index → original 14,522 vertex index
        # For each face, trimesh.faces[i][j] (expanded) ↔ faces_v_orig[i][j] (original)
        expanded_to_orig = np.full(len(template_mesh.vertices), -1, dtype=np.int32)
        for i in range(len(template_mesh.faces)):
            for j in range(3):
                exp_idx = template_mesh.faces[i][j]
                orig_idx = faces_v_orig[i][j]
                expanded_to_orig[exp_idx] = orig_idx

        n_mapped = (expanded_to_orig >= 0).sum()
        n_unique = len(np.unique(expanded_to_orig[expanded_to_orig >= 0]))
        print(f"  Mapping: {n_mapped}/{len(expanded_to_orig)} expanded verts mapped "
              f"→ {n_unique} unique original verts")

        _TEMPLATE_CACHE = {
            "template_mesh": template_mesh,
            "expanded_to_orig": expanded_to_orig,  # [15399] → original vertex index
            "n_orig_verts": n_unique,
        }

    cache = _TEMPLATE_CACHE

    # Load per-frame geometry
    frame_mesh = trimesh.load(frame_obj_path, process=False)
    frame_verts = np.array(frame_mesh.vertices)

    if len(frame_verts) != cache["n_orig_verts"]:
        print(f"  WARNING: vertex count mismatch: expected {cache['n_orig_verts']}, got {len(frame_verts)}")
        return None

    # Create a copy of template mesh (preserves faces, UV, material)
    mesh = cache["template_mesh"].copy()

    # Swap vertex positions: expanded_verts[i] = frame_verts[expanded_to_orig[i]]
    mesh.vertices = frame_verts[cache["expanded_to_orig"]]

    return mesh


def phase_mammal(frame_indices: list, output_dir: str, diag_texture: bool = False):
    """Render MAMMAL mesh at saved camera poses using pyrender.

    Args:
        diag_texture: If True, render both textured and flat versions for comparison.
    """
    import trimesh
    import pyrender
    from PIL import Image

    # Check if texture assets exist
    has_texture = os.path.exists(MAMMAL_TEXTURED_OBJ) and os.path.exists(MAMMAL_TEXTURE_PNG)
    if has_texture:
        print(f"UV texture available: {MAMMAL_TEXTURE_PNG}")
    else:
        print("WARNING: UV texture not found, using flat material")

    for frame_idx in frame_indices:
        print(f"\n=== Frame {frame_idx} ===")
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx:05d}")
        os.makedirs(os.path.join(frame_dir, "mammal_gt"), exist_ok=True)
        os.makedirs(os.path.join(frame_dir, "mammal_novel"), exist_ok=True)
        if diag_texture:
            os.makedirs(os.path.join(frame_dir, "mammal_flat_gt"), exist_ok=True)

        # Load camera config saved by Phase 1
        config_path = os.path.join(
            frame_dir, f"camera_config_frame_{frame_idx:05d}.json"
        )
        if not os.path.exists(config_path):
            print(f"  ERROR: Camera config not found: {config_path}")
            print("  Run Phase 1 (--phase gslrm) first.")
            continue

        with open(config_path) as f:
            config = json.load(f)

        # Load MAMMAL OBJ mesh
        obj_path = get_frame_obj_path(frame_idx)
        if not os.path.exists(obj_path):
            print(f"  ERROR: OBJ not found: {obj_path}")
            continue

        # Load flat mesh (always needed for diag or fallback)
        flat_mesh_trimesh = trimesh.load(obj_path, process=False)
        flat_verts = np.array(flat_mesh_trimesh.vertices)
        flat_verts_fl = mammal_to_facelift(flat_verts)
        flat_mesh_trimesh.vertices = flat_verts_fl

        # Try loading with UV texture
        textured_mesh = None
        if has_texture:
            textured_mesh = _load_textured_mesh(obj_path)

        if textured_mesh is not None:
            mesh_trimesh = textured_mesh
            verts = np.array(mesh_trimesh.vertices)
            verts_fl = mammal_to_facelift(verts)
            mesh_trimesh.vertices = verts_fl
            print(f"  Using UV textured mesh (bounds: {np.array2string(verts_fl.min(axis=0), precision=4)} ~ {np.array2string(verts_fl.max(axis=0), precision=4)})")
        else:
            mesh_trimesh = flat_mesh_trimesh
            print(f"  Using flat mesh (bounds: {np.array2string(flat_verts_fl.min(axis=0), precision=4)} ~ {np.array2string(flat_verts_fl.max(axis=0), precision=4)})")

        # Create pyrender meshes
        if textured_mesh is not None:
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
        else:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.75, 0.7, 1.0],
                metallicFactor=0.1,
                roughnessFactor=0.8,
            )
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, material=material)

        # Also create flat pyrender mesh for diagnostic comparison
        flat_pyrender = None
        if diag_texture and has_texture:
            flat_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.75, 0.7, 1.0],
                metallicFactor=0.1,
                roughnessFactor=0.8,
            )
            flat_pyrender = pyrender.Mesh.from_trimesh(flat_mesh_trimesh, material=flat_material)

        # Setup offscreen renderer
        renderer = pyrender.OffscreenRenderer(
            viewport_width=RENDER_RESOLUTION,
            viewport_height=RENDER_RESOLUTION,
        )

        # Render at GT cameras
        print("  Rendering at GT cameras...")
        for cam_name, cam_params in config["gt_cameras"].items():
            img = _render_pyrender(
                mesh_pyrender, cam_params, renderer,
                use_texture=(textured_mesh is not None),
            )
            save_path = os.path.join(frame_dir, "mammal_gt", f"{cam_name}.png")
            _save_image(img, save_path)

            # Diagnostic: also render flat version
            if flat_pyrender is not None:
                flat_img = _render_pyrender(
                    flat_pyrender, cam_params, renderer,
                    use_texture=False,
                )
                save_path = os.path.join(frame_dir, "mammal_flat_gt", f"{cam_name}.png")
                _save_image(flat_img, save_path)

        # Render at novel cameras
        print("  Rendering at novel cameras...")
        for view_name, cam_params in config["novel_cameras"].items():
            img = _render_pyrender(
                mesh_pyrender, cam_params, renderer,
                use_texture=(textured_mesh is not None),
            )
            save_path = os.path.join(frame_dir, "mammal_novel", f"{view_name}.png")
            _save_image(img, save_path)

        renderer.delete()

    print(f"\nPhase 2 complete. Outputs in: {output_dir}")


def _render_pyrender(mesh, cam_params, renderer, use_texture: bool = False):
    """Render a mesh at a given camera pose using pyrender.

    Camera params are in FaceLift/OpenCV convention (X-right, Y-down, Z-forward).
    pyrender uses OpenGL convention (X-right, Y-up, Z-backward).
    Conversion: flip Y and Z axes.

    When use_texture=True, reduces lighting intensity to avoid double-lighting
    with baked texture colors.
    """
    import pyrender

    c2w_cv = np.array(cam_params["c2w"])
    fxfycxcy = np.array(cam_params["fxfycxcy"])

    # OpenCV C2W -> OpenGL C2W (flip Y and Z)
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w_gl = c2w_cv @ cv_to_gl

    # Create camera with intrinsics
    fx, fy, cx, cy = fxfycxcy
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

    # Build scene
    # High ambient for textured mesh (texture has baked color, we want to see it clearly)
    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[1.0, 1.0, 1.0] if use_texture else [0.3, 0.3, 0.3],
    )
    scene.add(mesh)
    scene.add(camera, pose=c2w_gl)

    if use_texture:
        # Strong multi-directional lighting for dark C57BL/6 mouse texture
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene.add(key_light, pose=c2w_gl)
        # Fill lights from multiple directions
        for offset in [[0, 0, 3], [0, 3, 0], [3, 0, 0], [-3, 0, 0]]:
            fill_pose = np.eye(4)
            fill_pose[:3, 3] = offset
            fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(fill_light, pose=fill_pose)
    else:
        # Lighting for flat material
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=c2w_gl)
        ambient = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        ambient_pose = np.eye(4)
        ambient_pose[:3, 3] = [0, 0, 5]
        scene.add(ambient, pose=ambient_pose)

    # Render
    color, depth = renderer.render(scene)
    return color  # uint8 [H, W, 3]


# ============================================================
# Phase 3: Comparison Grid Generation
# ============================================================
def phase_compare(frame_indices: list, output_dir: str):
    """Create comparison grids from rendered image pairs."""
    import cv2

    for frame_idx in frame_indices:
        print(f"\n=== Frame {frame_idx} ===")
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx:05d}")
        grid_dir = os.path.join(frame_dir, "comparison")
        os.makedirs(grid_dir, exist_ok=True)

        # GT View comparison: GT RGB | MAMMAL render | GS-LRM render
        print("  Creating GT view comparisons...")
        gt_rgb_dir = os.path.join(frame_dir, "gt_rgb")
        mammal_gt_dir = os.path.join(frame_dir, "mammal_gt")
        gslrm_gt_dir = os.path.join(frame_dir, "gslrm_gt")

        gt_grid_images = []
        for cam_name in sorted(os.listdir(gslrm_gt_dir)):
            if not cam_name.endswith(".png"):
                continue
            stem = cam_name.replace(".png", "")

            imgs = []
            labels = []
            for subdir, label in [
                (gt_rgb_dir, "GT RGB"),
                (mammal_gt_dir, "MAMMAL Mesh"),
                (gslrm_gt_dir, "GS-LRM"),
            ]:
                path = os.path.join(subdir, cam_name)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    img = cv2.resize(img, (RENDER_RESOLUTION, RENDER_RESOLUTION))
                    imgs.append(img)
                    labels.append(label)
                else:
                    # Placeholder
                    placeholder = np.ones(
                        (RENDER_RESOLUTION, RENDER_RESOLUTION, 3), dtype=np.uint8
                    ) * 200
                    imgs.append(placeholder)
                    labels.append(f"{label} (N/A)")

            # Add labels
            for i, (img, label) in enumerate(zip(imgs, labels)):
                cv2.putText(
                    img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
                cv2.putText(
                    img, stem, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1,
                )

            row = np.concatenate(imgs, axis=1)
            gt_grid_images.append(row)

            # Save individual comparison
            save_path = os.path.join(grid_dir, f"gt_view_{stem}.png")
            cv2.imwrite(save_path, row)

        # Save full GT grid (all cameras stacked)
        if gt_grid_images:
            full_grid = np.concatenate(gt_grid_images, axis=0)
            save_path = os.path.join(grid_dir, "gt_views_all.png")
            cv2.imwrite(save_path, full_grid)
            print(f"  Saved: {save_path}")

        # Novel View comparison: MAMMAL render | GS-LRM render
        print("  Creating novel view comparisons...")
        mammal_novel_dir = os.path.join(frame_dir, "mammal_novel")
        gslrm_novel_dir = os.path.join(frame_dir, "gslrm_novel")

        novel_grid_images = []
        for view_file in sorted(os.listdir(gslrm_novel_dir)):
            if not view_file.endswith(".png"):
                continue
            stem = view_file.replace(".png", "")

            imgs = []
            labels = []
            for subdir, label in [
                (mammal_novel_dir, "MAMMAL Mesh"),
                (gslrm_novel_dir, "GS-LRM"),
            ]:
                path = os.path.join(subdir, view_file)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    img = cv2.resize(img, (RENDER_RESOLUTION, RENDER_RESOLUTION))
                    imgs.append(img)
                    labels.append(label)
                else:
                    placeholder = np.ones(
                        (RENDER_RESOLUTION, RENDER_RESOLUTION, 3), dtype=np.uint8
                    ) * 200
                    imgs.append(placeholder)
                    labels.append(f"{label} (N/A)")

            for i, (img, label) in enumerate(zip(imgs, labels)):
                cv2.putText(
                    img, f"{label} [{stem}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            row = np.concatenate(imgs, axis=1)
            novel_grid_images.append(row)

            save_path = os.path.join(grid_dir, f"novel_view_{stem}.png")
            cv2.imwrite(save_path, row)

        if novel_grid_images:
            full_grid = np.concatenate(novel_grid_images, axis=0)
            save_path = os.path.join(grid_dir, "novel_views_all.png")
            cv2.imwrite(save_path, full_grid)
            print(f"  Saved: {save_path}")

        # Compute basic metrics (PSNR for GT views)
        _compute_metrics(frame_dir, grid_dir)

    print(f"\nPhase 3 complete. Comparison grids in: {output_dir}")


def _compute_metrics(frame_dir: str, grid_dir: str):
    """Compute PSNR/SSIM between image pairs."""
    try:
        import cv2
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError:
        print("  Skipping metrics (skimage not available)")
        return

    metrics = {}
    gt_rgb_dir = os.path.join(frame_dir, "gt_rgb")
    mammal_gt_dir = os.path.join(frame_dir, "mammal_gt")
    gslrm_gt_dir = os.path.join(frame_dir, "gslrm_gt")

    for cam_file in sorted(os.listdir(gslrm_gt_dir)):
        if not cam_file.endswith(".png"):
            continue
        stem = cam_file.replace(".png", "")
        cam_metrics = {}

        gt_path = os.path.join(gt_rgb_dir, cam_file)
        mammal_path = os.path.join(mammal_gt_dir, cam_file)
        gslrm_path = os.path.join(gslrm_gt_dir, cam_file)

        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path)
            gt_img = cv2.resize(gt_img, (RENDER_RESOLUTION, RENDER_RESOLUTION))

            if os.path.exists(mammal_path):
                mammal_img = cv2.imread(mammal_path)
                mammal_img = cv2.resize(
                    mammal_img, (RENDER_RESOLUTION, RENDER_RESOLUTION)
                )
                psnr = peak_signal_noise_ratio(gt_img, mammal_img)
                ssim = structural_similarity(
                    gt_img, mammal_img, channel_axis=2, data_range=255
                )
                cam_metrics["gt_vs_mammal"] = {
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 4),
                }

            if os.path.exists(gslrm_path):
                gslrm_img = cv2.imread(gslrm_path)
                gslrm_img = cv2.resize(
                    gslrm_img, (RENDER_RESOLUTION, RENDER_RESOLUTION)
                )
                psnr = peak_signal_noise_ratio(gt_img, gslrm_img)
                ssim = structural_similarity(
                    gt_img, gslrm_img, channel_axis=2, data_range=255
                )
                cam_metrics["gt_vs_gslrm"] = {
                    "psnr": round(psnr, 2),
                    "ssim": round(ssim, 4),
                }

        if cam_metrics:
            metrics[stem] = cam_metrics

    if metrics:
        metrics_path = os.path.join(grid_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved metrics: {metrics_path}")

        # Print summary
        print("  Metrics summary:")
        for cam, m in metrics.items():
            for pair, vals in m.items():
                print(f"    {cam} {pair}: PSNR={vals['psnr']:.1f} SSIM={vals['ssim']:.3f}")


def _save_image(img: np.ndarray, path: str):
    """Save image using PIL (available in most envs)."""
    from PIL import Image

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ============================================================
# M5t2 Split Definitions
# ============================================================
M5T2_SPLITS = {
    "train": list(range(0, 2880)),
    "val": list(range(2880, 3240)),
    "test": list(range(3240, 3600)),
}


def get_split_name(frame_idx: int) -> str:
    """Return split name (train/val/test) for a given M5 frame index."""
    if frame_idx < 2880:
        return "train"
    elif frame_idx < 3240:
        return "val"
    else:
        return "test"


# ============================================================
# Phase 4: FG-Only Comparison + Video Generation
# ============================================================
def _extract_fg_white_bg(img: np.ndarray, gt_rgba_path: str) -> np.ndarray:
    """Extract foreground from image using GT alpha mask, white background.

    Args:
        img: [H,W,3] uint8 BGR image
        gt_rgba_path: Path to GT RGBA image (alpha channel = mask)

    Returns:
        [H,W,3] uint8 BGR image with foreground on white background
    """
    import cv2
    from PIL import Image as PILImage

    # Load GT alpha mask
    gt_rgba = PILImage.open(gt_rgba_path).convert("RGBA")
    alpha = np.array(gt_rgba.split()[-1])  # [H_orig, W_orig]

    # Resize alpha to match image
    h, w = img.shape[:2]
    if alpha.shape[0] != h or alpha.shape[1] != w:
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    # Binary mask (threshold at 128)
    mask = (alpha > 128).astype(np.float32)

    # Composite on white background
    fg = img.astype(np.float32)
    white = np.ones_like(fg) * 255.0
    result = fg * mask[:, :, None] + white * (1.0 - mask[:, :, None])
    return result.astype(np.uint8)


def phase_compare_fg(frame_indices: list, output_dir: str):
    """Create FG-only comparison grids using GT alpha mask."""
    import cv2

    m5_data_dir = M5_DATA_DIR

    for frame_idx in frame_indices:
        print(f"\n=== Frame {frame_idx} (FG comparison) ===")
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx:05d}")
        fg_dir = os.path.join(frame_dir, "comparison_fg")
        os.makedirs(fg_dir, exist_ok=True)

        # Paths
        sample_dir = get_sample_dir(frame_idx)
        gt_rgb_dir = os.path.join(frame_dir, "gt_rgb")
        mammal_gt_dir = os.path.join(frame_dir, "mammal_gt")
        gslrm_gt_dir = os.path.join(frame_dir, "gslrm_gt")

        fg_grid_images = []
        fg_metrics = {}

        for v in range(6):
            cam_name = f"cam_{v:03d}"
            gt_rgba_path = os.path.join(sample_dir, "images", f"cam_{v:03d}.png")
            if not os.path.exists(gt_rgba_path):
                continue

            imgs_fg = []
            labels = ["GT FG", "MAMMAL FG", "GS-LRM FG"]
            for subdir in [gt_rgb_dir, mammal_gt_dir, gslrm_gt_dir]:
                path = os.path.join(subdir, f"{cam_name}.png")
                if os.path.exists(path):
                    img = cv2.imread(path)
                    img = cv2.resize(img, (RENDER_RESOLUTION, RENDER_RESOLUTION))
                    fg_img = _extract_fg_white_bg(img, gt_rgba_path)
                    imgs_fg.append(fg_img)
                else:
                    placeholder = np.ones(
                        (RENDER_RESOLUTION, RENDER_RESOLUTION, 3), dtype=np.uint8
                    ) * 255
                    imgs_fg.append(placeholder)

            # Add labels
            for i, (img, label) in enumerate(zip(imgs_fg, labels)):
                cv2.putText(
                    img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            row = np.concatenate(imgs_fg, axis=1)
            fg_grid_images.append(row)
            cv2.imwrite(os.path.join(fg_dir, f"fg_view_{cam_name}.png"), row)

            # FG-only PSNR
            try:
                from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                gt_fg = imgs_fg[0]
                for j, (comp_fg, pair_name) in enumerate([
                    (imgs_fg[1], "gt_vs_mammal_fg"),
                    (imgs_fg[2], "gt_vs_gslrm_fg"),
                ]):
                    if j + 1 < len(imgs_fg):
                        psnr = peak_signal_noise_ratio(gt_fg, comp_fg)
                        ssim = structural_similarity(
                            gt_fg, comp_fg, channel_axis=2, data_range=255
                        )
                        fg_metrics.setdefault(cam_name, {})[pair_name] = {
                            "psnr": round(psnr, 2), "ssim": round(ssim, 4),
                        }
            except ImportError:
                pass

        # Save stacked grid
        if fg_grid_images:
            full_grid = np.concatenate(fg_grid_images, axis=0)
            cv2.imwrite(os.path.join(fg_dir, "fg_views_all.png"), full_grid)
            print(f"  Saved FG comparison grid")

        if fg_metrics:
            import json as _json
            with open(os.path.join(fg_dir, "metrics_fg.json"), "w") as f:
                _json.dump(fg_metrics, f, indent=2)
            print(f"  FG metrics saved")
            for cam, m in fg_metrics.items():
                for pair, vals in m.items():
                    print(f"    {cam} {pair}: PSNR={vals['psnr']:.1f}")


def phase_video(frame_indices: list, output_dir: str, cam_idx: int = 0,
                split_mode: bool = False, fps: int = 10):
    """Generate comparison videos from consecutive rendered frames.

    Stitches per-frame renderings into side-by-side MP4 videos.
    Uses imageio for encoding (available in mammal_stable env).

    Args:
        frame_indices: List of M5 frame indices (should be consecutive for video)
        output_dir: Base output directory
        cam_idx: Camera index to use for video (default: 0)
        split_mode: If True, organize by train/val/test subfolders
        fps: Video framerate
    """
    import cv2

    print(f"\n=== Video Generation ===")
    print(f"  Frames: {len(frame_indices)} ({frame_indices[0]}~{frame_indices[-1]})")
    print(f"  Camera: cam_{cam_idx:03d}, FPS: {fps}")

    # Collect frames
    gt_frames = []
    mammal_frames = []
    gslrm_frames = []
    valid_indices = []

    m5_data_dir = M5_DATA_DIR

    for frame_idx in frame_indices:
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx:05d}")
        cam_name = f"cam_{cam_idx:03d}"

        gt_path = os.path.join(frame_dir, "gt_rgb", f"{cam_name}.png")
        mammal_path = os.path.join(frame_dir, "mammal_gt", f"{cam_name}.png")
        gslrm_path = os.path.join(frame_dir, "gslrm_gt", f"{cam_name}.png")

        if not all(os.path.exists(p) for p in [gt_path, mammal_path, gslrm_path]):
            continue

        gt_img = cv2.imread(gt_path)
        mammal_img = cv2.imread(mammal_path)
        gslrm_img = cv2.imread(gslrm_path)

        # Resize to uniform size
        sz = RENDER_RESOLUTION
        gt_img = cv2.resize(gt_img, (sz, sz))
        mammal_img = cv2.resize(mammal_img, (sz, sz))
        gslrm_img = cv2.resize(gslrm_img, (sz, sz))

        gt_frames.append(gt_img)
        mammal_frames.append(mammal_img)
        gslrm_frames.append(gslrm_img)
        valid_indices.append(frame_idx)

    if not valid_indices:
        print("  ERROR: No rendered frames found. Run gslrm + mammal phases first.")
        return

    print(f"  Found {len(valid_indices)} valid frames")

    # Determine output directory
    if split_mode:
        # Group by split
        split_frames = {"train": [], "val": [], "test": []}
        for i, fidx in enumerate(valid_indices):
            split_name = get_split_name(fidx)
            split_frames[split_name].append(i)

        for split_name, indices in split_frames.items():
            if not indices:
                continue
            split_dir = os.path.join(output_dir, "videos", split_name)
            os.makedirs(split_dir, exist_ok=True)

            _write_comparison_video(
                [gt_frames[i] for i in indices],
                [mammal_frames[i] for i in indices],
                [gslrm_frames[i] for i in indices],
                split_dir, cam_idx, fps,
                label_prefix=f"[{split_name}] ",
            )
            print(f"  {split_name}: {len(indices)} frames → {split_dir}")
    else:
        video_dir = os.path.join(output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        _write_comparison_video(
            gt_frames, mammal_frames, gslrm_frames,
            video_dir, cam_idx, fps,
        )

    # Also generate FG-only video if possible
    fg_gt_frames = []
    fg_mammal_frames = []
    fg_gslrm_frames = []

    for i, frame_idx in enumerate(valid_indices):
        sample_dir = get_sample_dir(frame_idx)
        gt_rgba_path = os.path.join(sample_dir, "images", f"cam_{cam_idx:03d}.png")
        if os.path.exists(gt_rgba_path):
            fg_gt = _extract_fg_white_bg(gt_frames[i], gt_rgba_path)
            fg_mammal = _extract_fg_white_bg(mammal_frames[i], gt_rgba_path)
            fg_gslrm = _extract_fg_white_bg(gslrm_frames[i], gt_rgba_path)
            fg_gt_frames.append(fg_gt)
            fg_mammal_frames.append(fg_mammal)
            fg_gslrm_frames.append(fg_gslrm)

    if fg_gt_frames:
        if split_mode:
            # Group FG frames by split
            fg_split_frames = {"train": [], "val": [], "test": []}
            for i, fidx in enumerate(valid_indices):
                split_name = get_split_name(fidx)
                fg_split_frames[split_name].append(i)

            for split_name, indices in fg_split_frames.items():
                if not indices:
                    continue
                fg_dir = os.path.join(output_dir, "videos", split_name, "fg_only")
                os.makedirs(fg_dir, exist_ok=True)
                _write_comparison_video(
                    [fg_gt_frames[i] for i in indices],
                    [fg_mammal_frames[i] for i in indices],
                    [fg_gslrm_frames[i] for i in indices],
                    fg_dir, cam_idx, fps,
                    label_prefix=f"[{split_name} FG] ",
                )
                print(f"  {split_name} FG video: {len(indices)} frames → {fg_dir}")
        else:
            fg_dir = os.path.join(output_dir, "videos", "fg_only")
            os.makedirs(fg_dir, exist_ok=True)
            _write_comparison_video(
                fg_gt_frames, fg_mammal_frames, fg_gslrm_frames,
                fg_dir, cam_idx, fps, label_prefix="[FG] ",
            )
            print(f"  FG-only video: {len(fg_gt_frames)} frames → {fg_dir}")


def _write_comparison_video(
    gt_frames: list, mammal_frames: list, gslrm_frames: list,
    output_dir: str, cam_idx: int, fps: int, label_prefix: str = "",
):
    """Write a 3-column comparison video (GT | MAMMAL | GS-LRM)."""
    import cv2
    try:
        import imageio.v2 as imageio
        use_imageio = True
    except ImportError:
        use_imageio = False

    # Build comparison frames [GT | MAMMAL | GS-LRM]
    video_frames = []
    for gt, mammal, gslrm in zip(gt_frames, mammal_frames, gslrm_frames):
        # Add column headers
        gt_labeled = gt.copy()
        mammal_labeled = mammal.copy()
        gslrm_labeled = gslrm.copy()

        cv2.putText(gt_labeled, f"{label_prefix}GT RGB", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(mammal_labeled, f"{label_prefix}MAMMAL", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(gslrm_labeled, f"{label_prefix}GS-LRM", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        row = np.concatenate([gt_labeled, mammal_labeled, gslrm_labeled], axis=1)
        video_frames.append(row)

    # Encode video
    filename = f"comparison_cam{cam_idx:03d}.mp4"
    output_path = os.path.join(output_dir, filename)

    # Try imageio-ffmpeg first, then cv2 fallback
    written = False
    if use_imageio:
        try:
            import imageio_ffmpeg
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video_frames]
            writer = imageio.get_writer(
                output_path, fps=fps, codec="libx264",
                quality=8, pixelformat="yuv420p",
                format="FFMPEG",
            )
            for frame in rgb_frames:
                writer.append_data(frame)
            writer.close()
            written = True
        except (ImportError, Exception) as e:
            print(f"  imageio-ffmpeg failed ({e}), falling back to cv2")

    if not written:
        # cv2 fallback
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in video_frames:
            writer.write(frame)
        writer.release()

    print(f"  Video saved: {output_path} ({len(video_frames)} frames, {fps}fps)")


# ============================================================
# Phase: Texture Diagnostic (flat vs textured comparison)
# ============================================================
def phase_diag_texture(frame_indices: list, output_dir: str, cam_idx: int = 0):
    """Generate 4-column diagnostic grid: GT | Flat Mesh | Textured Mesh | GS-LRM.

    Helps isolate whether visual artifacts come from UV texture mapping
    or from the mesh geometry / pyrender rendering itself.
    """
    import cv2

    print(f"\n=== Texture Diagnostic ===")
    print(f"  Comparing flat vs textured mesh rendering")

    all_rows = []

    for frame_idx in frame_indices:
        frame_dir = os.path.join(output_dir, f"frame_{frame_idx:05d}")
        cam_name = f"cam_{cam_idx:03d}"

        gt_path = os.path.join(frame_dir, "gt_rgb", f"{cam_name}.png")
        flat_path = os.path.join(frame_dir, "mammal_flat_gt", f"{cam_name}.png")
        textured_path = os.path.join(frame_dir, "mammal_gt", f"{cam_name}.png")
        gslrm_path = os.path.join(frame_dir, "gslrm_gt", f"{cam_name}.png")

        if not os.path.exists(textured_path):
            print(f"  Frame {frame_idx}: mammal_gt not found, skipping")
            continue
        if not os.path.exists(flat_path):
            print(f"  Frame {frame_idx}: mammal_flat_gt not found (run --phase mammal --diag_texture first)")
            continue

        sz = RENDER_RESOLUTION
        imgs = []
        labels = ["GT RGB", "Flat Mesh", "Textured Mesh", "GS-LRM"]
        paths = [gt_path, flat_path, textured_path, gslrm_path]

        for path, label in zip(paths, labels):
            if os.path.exists(path):
                img = cv2.imread(path)
                img = cv2.resize(img, (sz, sz))
            else:
                img = np.ones((sz, sz, 3), dtype=np.uint8) * 200

            cv2.putText(img, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(img, f"F{frame_idx} {cam_name}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
            imgs.append(img)

        row = np.concatenate(imgs, axis=1)
        all_rows.append(row)

    if not all_rows:
        print("  No frames with both flat and textured renders found.")
        return

    # Save grid
    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    full_grid = np.concatenate(all_rows, axis=0)
    grid_path = os.path.join(diag_dir, f"texture_diag_cam{cam_idx:03d}.png")
    cv2.imwrite(grid_path, full_grid)
    print(f"  Saved diagnostic grid: {grid_path} ({len(all_rows)} frames)")

    # Also save per-frame grids
    for i, (row, frame_idx) in enumerate(zip(all_rows, frame_indices)):
        per_path = os.path.join(diag_dir, f"texture_diag_frame{frame_idx:05d}_cam{cam_idx:03d}.png")
        cv2.imwrite(per_path, row)

    print(f"  Diagnostics saved to: {diag_dir}")


def cleanup_frame_dirs(output_dir: str, keep_frames: list = None):
    """Remove per-frame directories, keeping only videos and diagnostics.

    Args:
        keep_frames: List of frame indices to keep as representative samples.
                     If None, keeps first, middle, and last from each split.
    """
    import shutil

    frame_dirs = sorted([
        d for d in os.listdir(output_dir)
        if d.startswith("frame_") and os.path.isdir(os.path.join(output_dir, d))
    ])

    if not frame_dirs:
        print("  No frame directories to clean up.")
        return

    # Parse frame indices from directory names
    all_indices = []
    for d in frame_dirs:
        try:
            idx = int(d.replace("frame_", ""))
            all_indices.append(idx)
        except ValueError:
            pass

    if keep_frames is None:
        # Keep representative frames: first, last of each split that exists
        keep_set = set()
        splits = {"train": [], "val": [], "test": []}
        for idx in all_indices:
            splits[get_split_name(idx)].append(idx)
        for split_name, indices in splits.items():
            if indices:
                keep_set.add(indices[0])
                keep_set.add(indices[-1])
                if len(indices) > 2:
                    keep_set.add(indices[len(indices) // 2])
        keep_frames = sorted(keep_set)

    keep_set = set(keep_frames)
    removed = 0
    for idx in all_indices:
        if idx not in keep_set:
            dir_path = os.path.join(output_dir, f"frame_{idx:05d}")
            shutil.rmtree(dir_path)
            removed += 1

    kept = len(all_indices) - removed
    print(f"  Cleanup: removed {removed} frame dirs, kept {kept} (frames: {sorted(keep_set)})")


# ============================================================
# Coordinate Verification (Phase 0)
# ============================================================
def phase_verify(frame_indices: list, output_dir: str):
    """Verify coordinate transform by comparing MAMMAL keypoints with GS-LRM scene.

    This phase loads MAMMAL 3D keypoints, transforms them to FaceLift space,
    and checks if they land within the GS-LRM scene bounds.
    """
    kp_path = os.path.join(
        os.path.dirname(MAMMAL_OBJ_DIR), "keypoints_22_3d.npz"
    )
    if not os.path.exists(kp_path):
        print(f"Keypoints file not found: {kp_path}")
        return

    data = np.load(kp_path)
    kp_all = data["keypoints"] if "keypoints" in data else data[list(data.keys())[0]]
    print(f"Loaded keypoints: shape={kp_all.shape}")

    for frame_idx in frame_indices:
        if frame_idx >= len(kp_all):
            print(f"  Frame {frame_idx} out of range (max {len(kp_all)-1})")
            continue

        kp_mammal = kp_all[frame_idx]  # [22, 3] in mm
        kp_fl = mammal_to_facelift(kp_mammal)

        print(f"\n  Frame {frame_idx}:")
        print(f"    MAMMAL keypoints range: {kp_mammal.min(axis=0)} ~ {kp_mammal.max(axis=0)} (mm)")
        print(f"    FaceLift keypoints range: {np.array2string(kp_fl.min(axis=0), precision=4)} ~ {np.array2string(kp_fl.max(axis=0), precision=4)}")
        print(f"    FaceLift centroid: {np.array2string(kp_fl.mean(axis=0), precision=4)}")
        print(f"    Expected: near origin (0,0,0), extents ~0.1-0.5")

        # Check if in reasonable range for GS-LRM (radius 2.7)
        max_dist = np.linalg.norm(kp_fl, axis=1).max()
        print(f"    Max distance from origin: {max_dist:.4f}")
        if max_dist < 2.0:
            print("    ✓ Within GS-LRM scene bounds")
        else:
            print("    ⚠ Outside expected GS-LRM scene bounds!")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="PoC: MAMMAL Mesh vs GS-LRM Image Pair Collection"
    )
    parser.add_argument(
        "--phase",
        choices=["verify", "gslrm", "mammal", "compare", "compare_fg", "video",
                 "diag_texture", "cleanup"],
        required=True,
        help="Which phase to run",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[0, 100, 200],
        help="Frame indices to process",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_BASE,
        help="Output base directory",
    )
    parser.add_argument(
        "--checkpoint",
        default=GSLRM_CHECKPOINT,
        help="GS-LRM checkpoint path (Phase 1 only)",
    )
    parser.add_argument(
        "--config",
        default=GSLRM_CONFIG,
        help="GS-LRM config YAML path (Phase 1 only)",
    )
    parser.add_argument(
        "--cam_idx",
        type=int,
        default=0,
        help="Camera index for video generation (Phase video)",
    )
    parser.add_argument(
        "--split_mode",
        action="store_true",
        help="Organize video output by train/val/test split",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video framerate (Phase video)",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        default=None,
        help="Frame range [start, end) for video/batch processing",
    )
    parser.add_argument(
        "--diag_texture",
        action="store_true",
        help="Render both flat and textured mesh (mammal phase diagnostic)",
    )
    parser.add_argument(
        "--keep_frames",
        nargs="*",
        type=int,
        default=None,
        help="Frame indices to keep during cleanup (default: auto-select representatives)",
    )
    args = parser.parse_args()

    # Resolve frame indices
    if args.frame_range:
        frame_indices = list(range(args.frame_range[0], args.frame_range[1]))
    else:
        frame_indices = args.frames

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase == "verify":
        phase_verify(frame_indices, args.output_dir)
    elif args.phase == "gslrm":
        phase_gslrm(frame_indices, args.output_dir, args.checkpoint, args.config)
    elif args.phase == "mammal":
        phase_mammal(frame_indices, args.output_dir, diag_texture=args.diag_texture)
    elif args.phase == "compare":
        phase_compare(frame_indices, args.output_dir)
    elif args.phase == "compare_fg":
        phase_compare_fg(frame_indices, args.output_dir)
    elif args.phase == "video":
        phase_video(frame_indices, args.output_dir,
                    cam_idx=args.cam_idx, split_mode=args.split_mode, fps=args.fps)
    elif args.phase == "diag_texture":
        phase_diag_texture(frame_indices, args.output_dir, cam_idx=args.cam_idx)
    elif args.phase == "cleanup":
        cleanup_frame_dirs(args.output_dir, keep_frames=args.keep_frames)


if __name__ == "__main__":
    main()
