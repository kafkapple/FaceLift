# no-split: single dataset pipeline — 5 modes share camera defs, paths, frame indexing. Splitting fragments shared state.
#!/usr/bin/env python3
"""Novel View Dataset Collection Pipeline v2.0

Systematic pipeline for collecting novel-view renders from GS-LRM and MAMMAL mesh,
organized in a tier-based directory structure with per-frame metadata.

Directory structure:
    outputs/novel_view_dataset/
    ├── manifest.json                # Dataset-level metadata
    ├── cameras/
    │   └── novel_views.json         # Global novel view camera params
    ├── mouse_m5t2/                  # Species + dataset
    │   ├── tier0_raw/               # GS-LRM novel view renders
    │   │   ├── bottom/              # One dir per view type
    │   │   │   ├── 00000.png
    │   │   │   └── ...
    │   │   ├── top/
    │   │   ├── front_low/
    │   │   └── side_low/
    │   ├── pseudo_gt/               # MAMMAL mesh renders at novel cameras
    │   │   └── (same view structure)
    │   ├── gt_views/                # GS-LRM renders at GT camera poses
    │   │   ├── cam_000/
    │   │   └── ...
    │   ├── gt_rgb/                  # Original GT camera images
    │   │   └── (same cam structure)
    │   └── metadata/                # Per-frame metadata JSON
    │       ├── 00000.json
    │       └── ...
    ├── splits/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    └── visualizations/              # Comparison grids, videos

Usage:
    # Migrate existing PoC data
    python -m mouse_extensions.scripts.novel_view.collect_dataset \\
        --mode migrate \\
        --poc_dir outputs/poc_mesh_gs_pairs_v0_archive

    # Generate fresh data for a frame range
    python -m mouse_extensions.scripts.novel_view.collect_dataset \\
        --mode generate --phase gslrm \\
        --frame_range 0 150

    # Generate MAMMAL pseudo-GT (requires mammal_stable env)
    python -m mouse_extensions.scripts.novel_view.collect_dataset \\
        --mode generate --phase mammal \\
        --frame_range 0 150

    # Build manifest and splits
    python -m mouse_extensions.scripts.novel_view.collect_dataset \\
        --mode manifest
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Coordinate transforms — SSOT: mouse_extensions/coordinate_utils.py
from mouse_extensions.coordinate_utils import (
    M5_SCENE_CENTER, M5_DISTANCE_SCALE,
    mammal_to_gslrm, gslrm_to_mammal,
)

NOVEL_VIEWS = {
    "bottom": {"elevation": -70.0, "azimuth": 0.0},
    "top": {"elevation": 70.0, "azimuth": 0.0},
    "front_low": {"elevation": -30.0, "azimuth": 0.0},
    "side_low": {"elevation": -30.0, "azimuth": 90.0},
}

RENDER_RESOLUTION = 512  # Must match GS-LRM training resolution for best quality
TURNTABLE_RADIUS = 2.7

# Paths
M5_DATA_DIR = "/home/joon/data/preprocessed/FaceLift_mouse/M5"
MAMMAL_OBJ_DIR = (
    "/home/joon/dev/MAMMAL_mouse/results/fitting/"
    "production_3600_slerp/obj_textured"
)
MAMMAL_TEXTURED_OBJ = "/home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj"
MAMMAL_TEXTURE_PNG = "/home/joon/dev/MAMMAL_mouse/exports/texture_final.png"
GSLRM_CHECKPOINT = (
    "/node_data/joon/checkpoints/FaceLift/gslrm/"
    "M5t2_6view_alpha03_v3/best_psnr.pt"
)
GSLRM_CONFIG = (
    "/node_data/joon/checkpoints/FaceLift/gslrm/"
    "M5t2_6view_alpha03_v3/config.yaml"
)

OUTPUT_BASE = "/home/joon/dev/FaceLift/outputs/datasets/novel_view"
SPECIES_DATASET = "mouse_m5t2"

# M5t2 split boundaries
M5T2_SPLITS = {
    "train": (0, 2880),
    "val": (2880, 3240),
    "test": (3240, 3600),
}

VIEW_NAMES = list(NOVEL_VIEWS.keys())
NUM_GT_CAMERAS = 6

# Ablation uses the 6-view checkpoint with first-N input slicing
# (same approach as H4 view ablation experiment)


# Aliases for backward compatibility within this file
mammal_to_facelift = mammal_to_gslrm
facelift_to_mammal = gslrm_to_mammal


# ============================================================
# Camera generation
# ============================================================
def make_look_at_c2w(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])
) -> np.ndarray:
    """Create camera-to-world matrix (OpenCV: X-right, Y-down, Z-forward)."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([1, 0, 0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

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
    """Generate novel view cameras. Same for all frames."""
    gt_fx_at_512 = 548.99
    fx = fy = gt_fx_at_512 * (resolution / 512.0)
    cx = cy = resolution / 2.0
    fxfycxcy = [fx, fy, cx, cy]

    cameras = {}
    for name, params in NOVEL_VIEWS.items():
        elev_rad = np.radians(params["elevation"])
        azim_rad = np.radians(params["azimuth"])

        x = radius * np.cos(elev_rad) * np.cos(azim_rad)
        y = radius * np.cos(elev_rad) * np.sin(azim_rad)
        z = radius * np.sin(elev_rad)

        eye = center + np.array([x, y, z])
        c2w = make_look_at_c2w(eye, center)

        cameras[name] = {
            "c2w": c2w.tolist(),
            "fxfycxcy": fxfycxcy,
            "resolution": resolution,
            "elevation": params["elevation"],
            "azimuth": params["azimuth"],
        }
    return cameras


# ============================================================
# Path helpers
# ============================================================
def get_dataset_dir(output_base: str = OUTPUT_BASE) -> str:
    return os.path.join(output_base, SPECIES_DATASET)


def get_tier_dir(tier: str, output_base: str = OUTPUT_BASE) -> str:
    return os.path.join(get_dataset_dir(output_base), tier)


def get_view_path(
    tier: str, view_name: str, frame_idx: int, output_base: str = OUTPUT_BASE
) -> str:
    """Get image path: {dataset}/{tier}/{view_name}/{frame_idx:05d}.png"""
    return os.path.join(
        get_tier_dir(tier, output_base), view_name, f"{frame_idx:05d}.png"
    )


def get_gt_view_path(
    subdir: str, cam_idx: int, frame_idx: int, output_base: str = OUTPUT_BASE
) -> str:
    """Get GT camera path: {dataset}/{subdir}/cam_{cam_idx:03d}/{frame_idx:05d}.png"""
    return os.path.join(
        get_dataset_dir(output_base),
        subdir,
        f"cam_{cam_idx:03d}",
        f"{frame_idx:05d}.png",
    )


def get_metadata_path(frame_idx: int, output_base: str = OUTPUT_BASE) -> str:
    return os.path.join(
        get_dataset_dir(output_base), "metadata", f"{frame_idx:05d}.json"
    )


def get_sample_dir(frame_idx: int) -> str:
    """M5 data sample directory for a given frame index."""
    return os.path.join(M5_DATA_DIR, f"{frame_idx:06d}")


def get_frame_obj_path(frame_idx: int) -> str:
    """MAMMAL OBJ path. M5 idx N → MAMMAL video frame N*5 (100fps→20fps)."""
    mammal_frame = frame_idx * 5
    # Try step_2_frame (current MAMMAL output format) then frame_ (legacy)
    path = os.path.join(MAMMAL_OBJ_DIR, f"step_2_frame_{mammal_frame:06d}.obj")
    if not os.path.exists(path):
        path = os.path.join(MAMMAL_OBJ_DIR, f"frame_{mammal_frame:05d}.obj")
    return path


def get_ablation_view_path(
    n_views: int, cam_idx: int, frame_idx: int, output_base: str = OUTPUT_BASE
) -> str:
    """Get ablation render path: {dataset}/ablation_{N}view/cam_{cam_idx:03d}/{frame:05d}.png"""
    return os.path.join(
        get_dataset_dir(output_base),
        f"ablation_{n_views}view",
        f"cam_{cam_idx:03d}",
        f"{frame_idx:05d}.png",
    )


def get_split_name(frame_idx: int) -> str:
    """Return train/val/test for M5t2 frame index."""
    for split, (start, end) in M5T2_SPLITS.items():
        if start <= frame_idx < end:
            return split
    return "unknown"


# ============================================================
# Image I/O
# ============================================================
def _save_image(img: np.ndarray, path: str):
    """Save uint8 image via PIL."""
    from PIL import Image

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ============================================================
# Per-frame metadata
# ============================================================
def create_frame_metadata(
    frame_idx: int,
    gt_cameras: dict | None = None,
    has_tier0: bool = False,
    has_pseudo_gt: bool = False,
    has_gt_views: bool = False,
    has_gt_rgb: bool = False,
    output_base: str = OUTPUT_BASE,
) -> dict:
    """Create per-frame metadata JSON."""
    mammal_frame = frame_idx * 5
    split = get_split_name(frame_idx)

    meta = {
        "frame_idx": frame_idx,
        "mammal_video_frame": mammal_frame,
        "split": split,
        "species": "mouse",
        "dataset": "M5t2",
        "resolution": RENDER_RESOLUTION,
        "coordinate_system": "facelift_normalized",
        "transform": {
            "M5_SCENE_CENTER": M5_SCENE_CENTER.tolist(),
            "M5_DISTANCE_SCALE": float(M5_DISTANCE_SCALE),
            "formula": "point_fl = (point_mammal_mm - center) * scale",
        },
        "data_available": {
            "tier0_raw": has_tier0,
            "pseudo_gt": has_pseudo_gt,
            "gt_views": has_gt_views,
            "gt_rgb": has_gt_rgb,
            "tier1_cleaned": False,
            "tier2_enhanced": False,
            "artifact_masks": False,
        },
        "mammal_obj_path": get_frame_obj_path(frame_idx),
        "m5_sample_dir": get_sample_dir(frame_idx),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if gt_cameras:
        meta["gt_cameras"] = gt_cameras

    return meta


def save_frame_metadata(frame_idx: int, meta: dict, output_base: str = OUTPUT_BASE):
    """Save per-frame metadata JSON."""
    path = get_metadata_path(frame_idx, output_base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Mode: Migrate from PoC
# ============================================================
def migrate_poc(poc_dir: str, output_base: str = OUTPUT_BASE, dry_run: bool = False):
    """Migrate PoC flat structure to tier-based dataset structure.

    PoC structure:  frame_NNNNN/{gslrm_novel,mammal_novel,gslrm_gt,gt_rgb}/
    New structure:  mouse_m5t2/{tier0_raw,pseudo_gt,gt_views,gt_rgb}/{view_or_cam}/
    """
    print(f"=== Migrating PoC data ===")
    print(f"  Source: {poc_dir}")
    print(f"  Target: {output_base}")

    if not os.path.exists(poc_dir):
        print(f"  ERROR: PoC directory not found: {poc_dir}")
        return

    # Discover frame directories
    frame_dirs = sorted([
        d for d in os.listdir(poc_dir)
        if d.startswith("frame_") and os.path.isdir(os.path.join(poc_dir, d))
    ])

    if not frame_dirs:
        print("  ERROR: No frame directories found")
        return

    frame_indices = []
    for d in frame_dirs:
        try:
            idx = int(d.replace("frame_", ""))
            frame_indices.append(idx)
        except ValueError:
            continue

    print(f"  Found {len(frame_indices)} frames: {frame_indices[0]}..{frame_indices[-1]}")

    # Mapping: PoC subdir → (new_tier, naming_scheme)
    # gslrm_novel/{view}.png → tier0_raw/{view}/{frame}.png
    # mammal_novel/{view}.png → pseudo_gt/{view}/{frame}.png
    # gslrm_gt/cam_NNN.png → gt_views/cam_NNN/{frame}.png
    # gt_rgb/cam_NNN.png → gt_rgb/cam_NNN/{frame}.png

    total_copied = 0
    total_skipped = 0
    frames_with_data = []

    for frame_idx in frame_indices:
        poc_frame_dir = os.path.join(poc_dir, f"frame_{frame_idx:05d}")
        has_tier0 = False
        has_pseudo_gt = False
        has_gt_views = False
        has_gt_rgb = False

        # Migrate novel view renders
        for poc_subdir, tier in [("gslrm_novel", "tier0_raw"), ("mammal_novel", "pseudo_gt")]:
            src_dir = os.path.join(poc_frame_dir, poc_subdir)
            if not os.path.isdir(src_dir):
                continue

            for img_file in os.listdir(src_dir):
                if not img_file.endswith(".png"):
                    continue
                view_name = img_file.replace(".png", "")
                src = os.path.join(src_dir, img_file)
                dst = get_view_path(tier, view_name, frame_idx, output_base)

                if not dry_run:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                total_copied += 1

                if tier == "tier0_raw":
                    has_tier0 = True
                else:
                    has_pseudo_gt = True

        # Migrate GT camera renders
        for poc_subdir, new_subdir in [("gslrm_gt", "gt_views"), ("gt_rgb", "gt_rgb")]:
            src_dir = os.path.join(poc_frame_dir, poc_subdir)
            if not os.path.isdir(src_dir):
                continue

            for img_file in os.listdir(src_dir):
                if not img_file.endswith(".png"):
                    continue
                cam_name = img_file.replace(".png", "")
                cam_idx = int(cam_name.replace("cam_", ""))
                src = os.path.join(src_dir, img_file)
                dst = get_gt_view_path(new_subdir, cam_idx, frame_idx, output_base)

                if not dry_run:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                total_copied += 1

                if new_subdir == "gt_views":
                    has_gt_views = True
                else:
                    has_gt_rgb = True

        # Load GT cameras from PoC camera config
        gt_cameras = None
        config_path = os.path.join(
            poc_frame_dir, f"camera_config_frame_{frame_idx:05d}.json"
        )
        if os.path.exists(config_path):
            with open(config_path) as f:
                poc_config = json.load(f)
            gt_cameras = poc_config.get("gt_cameras")

        # Create per-frame metadata
        if not dry_run:
            meta = create_frame_metadata(
                frame_idx,
                gt_cameras=gt_cameras,
                has_tier0=has_tier0,
                has_pseudo_gt=has_pseudo_gt,
                has_gt_views=has_gt_views,
                has_gt_rgb=has_gt_rgb,
                output_base=output_base,
            )
            save_frame_metadata(frame_idx, meta, output_base)

        if has_tier0 or has_pseudo_gt:
            frames_with_data.append(frame_idx)

    print(f"  Copied: {total_copied} files")
    print(f"  Frames with data: {len(frames_with_data)}")

    # Save global camera config
    if not dry_run:
        _save_global_cameras(output_base)
        _build_splits(output_base)
        _build_manifest(output_base)

    print(f"  Migration complete.")


# ============================================================
# Mode: Generate (GS-LRM phase)
# ============================================================
def generate_gslrm(
    frame_indices: list[int],
    output_base: str = OUTPUT_BASE,
    checkpoint: str = GSLRM_CHECKPOINT,
    config: str = GSLRM_CONFIG,
    force: bool = False,
):
    """Run GS-LRM inference and render at GT + novel camera poses."""
    import torch

    sys.path.insert(0, "/home/joon/dev/FaceLift")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
    from mouse_extensions.visualization import render_opencv_cam

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading GS-LRM pipeline...")
    pipeline = GSLRMInference(
        config_path=config,
        checkpoint_path=checkpoint,
        device=device,
    )

    novel_cameras = generate_novel_cameras()

    for frame_idx in frame_indices:
        # Resume support: skip if all outputs already exist
        if not force:
            meta_path = get_metadata_path(frame_idx, output_base)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    existing = json.load(f)
                avail = existing.get("data_available", {})
                if avail.get("tier0_raw") and avail.get("gt_views") and avail.get("gt_rgb"):
                    print(f"  Frame {frame_idx}: already complete, skipping (use --force to override)")
                    continue

        print(f"\n=== Frame {frame_idx} ===")

        sample_dir = get_sample_dir(frame_idx)
        if not os.path.exists(sample_dir):
            print(f"  WARNING: Sample dir not found: {sample_dir}, skipping")
            continue

        # Load 6-view input
        images, c2ws, fxfycxcys, index = load_sample_data(
            sample_dir, image_size=RENDER_RESOLUTION, device=device
        )

        # GS-LRM inference
        print("  Running GS-LRM inference...")
        result = pipeline.predict(images, c2ws, fxfycxcys, index)
        gaussians = result["gaussians"][0]

        c2ws_np = c2ws[0].cpu().numpy()
        fxfycxcys_np = fxfycxcys[0].cpu().numpy()
        gt_cameras = {}

        # Render at GT cameras
        for v in range(c2ws_np.shape[0]):
            cam_name = f"cam_{v:03d}"
            gt_cameras[cam_name] = {
                "c2w": c2ws_np[v].tolist(),
                "fxfycxcy": fxfycxcys_np[v].tolist(),
                "resolution": RENDER_RESOLUTION,
            }

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
            img = rendered["render"].detach().cpu().permute(1, 2, 0).numpy()
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

            # Save GS-LRM render at GT camera
            dst = get_gt_view_path("gt_views", v, frame_idx, output_base)
            _save_image(img, dst)

            # Copy GT RGB
            gt_img_src = os.path.join(sample_dir, "images", f"cam_{v:03d}.png")
            if not os.path.exists(gt_img_src):
                gt_img_src = os.path.join(sample_dir, f"cam_{v:03d}.png")
            if os.path.exists(gt_img_src):
                gt_dst = get_gt_view_path("gt_rgb", v, frame_idx, output_base)
                os.makedirs(os.path.dirname(gt_dst), exist_ok=True)
                shutil.copy2(gt_img_src, gt_dst)

        # Render at novel cameras
        print("  Rendering novel views...")
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

            dst = get_view_path("tier0_raw", name, frame_idx, output_base)
            _save_image(img, dst)

        # Save metadata
        meta = create_frame_metadata(
            frame_idx,
            gt_cameras=gt_cameras,
            has_tier0=True,
            has_pseudo_gt=False,
            has_gt_views=True,
            has_gt_rgb=True,
            output_base=output_base,
        )
        save_frame_metadata(frame_idx, meta, output_base)
        print(f"  Frame {frame_idx} complete.")

    print(f"\nGS-LRM generation complete.")


# ============================================================
# Mode: Generate (MAMMAL phase)
# ============================================================
# Template mesh cache for UV texture handling
_TEMPLATE_CACHE = None


def _load_textured_mesh(frame_obj_path: str):
    """Load MAMMAL mesh with UV texture, handling vertex expansion."""
    import trimesh

    global _TEMPLATE_CACHE

    if _TEMPLATE_CACHE is None:
        if not os.path.exists(MAMMAL_TEXTURED_OBJ) or not os.path.exists(MAMMAL_TEXTURE_PNG):
            return None

        template = trimesh.load(MAMMAL_TEXTURED_OBJ, process=False)
        n_expanded = len(template.vertices)

        # Parse original OBJ for non-expanded vertex count
        n_unique = 0
        with open(MAMMAL_TEXTURED_OBJ) as f:
            for line in f:
                if line.startswith("v "):
                    n_unique += 1

        # Build expanded→original vertex mapping
        # Template has face-level vertex expansion for UV
        expanded_to_orig = np.zeros(n_expanded, dtype=np.int64)
        orig_verts = []
        with open(MAMMAL_TEXTURED_OBJ) as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    orig_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        orig_verts = np.array(orig_verts)

        # Match expanded vertices to original by position
        from scipy.spatial import cKDTree

        tree = cKDTree(orig_verts)
        dists, indices = tree.query(template.vertices)
        expanded_to_orig = indices

        _TEMPLATE_CACHE = {
            "template_mesh": template,
            "expanded_to_orig": expanded_to_orig,
            "n_orig_verts": n_unique,
        }

    cache = _TEMPLATE_CACHE
    frame_mesh = trimesh.load(frame_obj_path, process=False)
    frame_verts = np.array(frame_mesh.vertices)

    if len(frame_verts) != cache["n_orig_verts"]:
        print(f"  WARNING: vertex count mismatch: expected {cache['n_orig_verts']}, got {len(frame_verts)}")
        return None

    mesh = cache["template_mesh"].copy()
    mesh.vertices = frame_verts[cache["expanded_to_orig"]]
    return mesh


def _render_pyrender(mesh, cam_params, renderer, use_texture: bool = False):
    """Render mesh at camera pose. OpenCV→OpenGL convention conversion."""
    import pyrender

    c2w_cv = np.array(cam_params["c2w"])
    fxfycxcy = np.array(cam_params["fxfycxcy"])

    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float64)
    c2w_gl = c2w_cv @ cv_to_gl

    fx, fy, cx, cy = fxfycxcy
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],
        ambient_light=[1.0, 1.0, 1.0] if use_texture else [0.3, 0.3, 0.3],
    )
    scene.add(mesh)
    scene.add(camera, pose=c2w_gl)

    if use_texture:
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        scene.add(key_light, pose=c2w_gl)
        for offset in [[0, 0, 3], [0, 3, 0], [3, 0, 0], [-3, 0, 0]]:
            fill_pose = np.eye(4)
            fill_pose[:3, 3] = offset
            fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            scene.add(fill_light, pose=fill_pose)
    else:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=c2w_gl)
        ambient = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        ambient_pose = np.eye(4)
        ambient_pose[:3, 3] = [0, 0, 5]
        scene.add(ambient, pose=ambient_pose)

    color, depth = renderer.render(scene)
    return color


def generate_mammal(
    frame_indices: list[int],
    output_base: str = OUTPUT_BASE,
    force: bool = False,
    use_texture: bool = False,
):
    """Render MAMMAL mesh at novel view cameras (pseudo-GT).

    Args:
        use_texture: If True, render with UV texture and save to pseudo_gt_textured/.
                     If False, render untextured and save to pseudo_gt/.
    """
    import trimesh
    import pyrender

    output_tier = "pseudo_gt_textured" if use_texture else "pseudo_gt"

    if use_texture:
        has_texture = os.path.exists(MAMMAL_TEXTURED_OBJ) and os.path.exists(MAMMAL_TEXTURE_PNG)
        if not has_texture:
            print(f"ERROR: Texture files not found:")
            print(f"  OBJ: {MAMMAL_TEXTURED_OBJ} (exists={os.path.exists(MAMMAL_TEXTURED_OBJ)})")
            print(f"  PNG: {MAMMAL_TEXTURE_PNG} (exists={os.path.exists(MAMMAL_TEXTURE_PNG)})")
            return

    novel_cameras = generate_novel_cameras()

    for frame_idx in frame_indices:
        # Resume support: skip if output_tier already complete
        if not force:
            meta_path = get_metadata_path(frame_idx, output_base)
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    existing = json.load(f)
                if existing.get("data_available", {}).get(output_tier):
                    print(f"  Frame {frame_idx}: {output_tier} exists, skipping (use --force)")
                    continue

        print(f"\n=== Frame {frame_idx} (MAMMAL {'textured' if use_texture else 'untextured'}) ===")

        # Load metadata to get GT cameras
        meta_path = get_metadata_path(frame_idx, output_base)
        if not os.path.exists(meta_path):
            print(f"  WARNING: No metadata found. Run gslrm phase first or migrate PoC data.")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        gt_cameras = meta.get("gt_cameras", {})

        # Load MAMMAL mesh
        obj_path = get_frame_obj_path(frame_idx)
        if not os.path.exists(obj_path):
            print(f"  ERROR: OBJ not found: {obj_path}")
            continue

        # Load mesh with UV texture if requested
        if use_texture:
            mesh_trimesh = _load_textured_mesh(obj_path)
            if mesh_trimesh is None:
                print(f"  ERROR: Failed to load textured mesh for frame {frame_idx}")
                continue
            verts = np.array(mesh_trimesh.vertices)
            mesh_trimesh.vertices = mammal_to_facelift(verts)
            render_textured = True
        else:
            mesh_trimesh = trimesh.load(obj_path, process=False)
            verts = np.array(mesh_trimesh.vertices)
            mesh_trimesh.vertices = mammal_to_facelift(verts)
            render_textured = False

        # Create pyrender mesh
        if render_textured:
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
        else:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.8, 0.75, 0.7, 1.0],
                metallicFactor=0.1,
                roughnessFactor=0.8,
            )
            mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_trimesh, material=material)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=RENDER_RESOLUTION,
            viewport_height=RENDER_RESOLUTION,
        )

        # Render at novel cameras → output_tier
        print(f"  Rendering {output_tier} at novel cameras...")
        for view_name, cam_params in novel_cameras.items():
            img = _render_pyrender(mesh_pyrender, cam_params, renderer, render_textured)
            dst = get_view_path(output_tier, view_name, frame_idx, output_base)
            _save_image(img, dst)

        renderer.delete()

        # Update metadata
        meta["data_available"][output_tier] = True
        save_frame_metadata(frame_idx, meta, output_base)
        print(f"  Frame {frame_idx} {output_tier} complete.")

    print(f"\nMAMMAL {output_tier} generation complete.")


# ============================================================
# Mode: Generate (Ablation phase)
# ============================================================
def generate_ablation(
    frame_indices: list[int],
    n_views_list: list[int],
    output_base: str = OUTPUT_BASE,
    config: str = GSLRM_CONFIG,
    checkpoint: str = GSLRM_CHECKPOINT,
    force: bool = False,
):
    """Generate N-view ablation renders at GT camera poses.

    Uses the 6-view checkpoint with first-N input slicing (same as H4 ablation):
      1. Load 6-view checkpoint (once)
      2. For each frame and N: load 6 GT views, slice first N
      3. GS-LRM inference -> Gaussians
      4. Render at all 6 GT cameras
      5. Save to ablation_{N}view/cam_NNN/NNNNN.png
    """
    import torch

    sys.path.insert(0, "/home/joon/dev/FaceLift")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
    from mouse_extensions.visualization import render_opencv_cam

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading GS-LRM pipeline (6-view checkpoint)...")
    print(f"  Checkpoint: {checkpoint}")
    pipeline = GSLRMInference(
        config_path=config,
        checkpoint_path=checkpoint,
        device=device,
    )

    for n_views in n_views_list:
        print(f"\n{'='*60}")
        print(f"=== Ablation: {n_views}-view (slice first {n_views} of 6) ===")
        print(f"{'='*60}")

        for frame_idx in frame_indices:
            # Resume: check if all 6 GT camera renders exist
            if not force:
                all_exist = all(
                    os.path.exists(get_ablation_view_path(n_views, v, frame_idx, output_base))
                    for v in range(NUM_GT_CAMERAS)
                )
                if all_exist:
                    print(f"  Frame {frame_idx}: {n_views}v ablation exists, skipping")
                    continue

            sample_dir = get_sample_dir(frame_idx)
            if not os.path.exists(sample_dir):
                print(f"  WARNING: Sample dir not found: {sample_dir}, skipping")
                continue

            # Load all 6 views, then slice first N
            images, c2ws, fxfycxcys, index = load_sample_data(
                sample_dir, image_size=RENDER_RESOLUTION, device=device
            )

            # Slice to first N views for inference
            images_n = images[:, :n_views]
            c2ws_n = c2ws[:, :n_views]
            fxfycxcys_n = fxfycxcys[:, :n_views]

            # GS-LRM inference with N views
            result = pipeline.predict(images_n, c2ws_n, fxfycxcys_n, index)
            gaussians = result["gaussians"][0]

            # Render at all 6 GT cameras
            c2ws_np = c2ws[0].cpu().numpy()
            fxfycxcys_np = fxfycxcys[0].cpu().numpy()

            for v in range(c2ws_np.shape[0]):
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
                img = rendered["render"].detach().cpu().permute(1, 2, 0).numpy()
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

                dst = get_ablation_view_path(n_views, v, frame_idx, output_base)
                _save_image(img, dst)

            print(f"  Frame {frame_idx}: {n_views}v ablation complete (6 cams)")

    print(f"\nAblation generation complete.")


# ============================================================
# Mode: Video (3-column comparison)
# ============================================================
def generate_video(
    output_base: str = OUTPUT_BASE,
    fps: int = 20,
    n_views_list: list[int] | None = None,
):
    """Generate comparison videos: GT | GS-LRM | MAMMAL (+ optional ablation columns).

    Creates per-view MP4 videos showing side-by-side comparisons across all frames.
    """
    import cv2

    viz_dir = os.path.join(output_base, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Discover available frames from metadata
    meta_dir = os.path.join(get_dataset_dir(output_base), "metadata")
    if not os.path.exists(meta_dir):
        print("ERROR: No metadata found. Run generate first.")
        return

    frame_indices = sorted(
        int(f.replace(".json", ""))
        for f in os.listdir(meta_dir)
        if f.endswith(".json")
    )
    print(f"Found {len(frame_indices)} frames for video generation")

    # Generate video for each novel view
    for view_name in VIEW_NAMES:
        print(f"\n=== Video: {view_name} ===")

        # Determine columns
        columns = []
        col_labels = []

        # Column 1: GS-LRM (tier0_raw)
        columns.append(("tier0_raw", view_name))
        col_labels.append(f"GS-LRM 6v [{view_name}]")

        # Column 2: MAMMAL pseudo-GT
        columns.append(("pseudo_gt", view_name))
        col_labels.append(f"MAMMAL [{view_name}]")

        # Optional ablation columns
        if n_views_list:
            for nv in sorted(n_views_list):
                # For novel views, ablation is at GT cameras only — skip
                pass

        n_cols = len(columns)
        frame_w = RENDER_RESOLUTION
        frame_h = RENDER_RESOLUTION
        canvas_w = frame_w * n_cols
        canvas_h = frame_h

        video_path = os.path.join(viz_dir, f"comparison_{view_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (canvas_w, canvas_h))

        written = 0
        for frame_idx in frame_indices:
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 200

            for col_i, (tier, vname) in enumerate(columns):
                img_path = get_view_path(tier, vname, frame_idx, output_base)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (frame_w, frame_h))
                    else:
                        img = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 200
                else:
                    img = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 200

                # Add label
                cv2.putText(img, col_labels[col_i], (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(img, f"F{frame_idx:05d}", (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

                x_off = col_i * frame_w
                canvas[:, x_off:x_off + frame_w] = img

            writer.write(canvas)
            written += 1

        writer.release()
        print(f"  Saved: {video_path} ({written} frames, {fps} fps)")

    # Also generate GT camera comparison videos (GT RGB | GS-LRM 6v | ablation variants)
    for cam_idx in range(NUM_GT_CAMERAS):
        cam_name = f"cam_{cam_idx:03d}"
        print(f"\n=== Video: {cam_name} ===")

        columns = []
        col_labels = []

        # GT RGB
        columns.append(("gt_rgb", cam_idx, "gt"))
        col_labels.append(f"GT [{cam_name}]")

        # GS-LRM 6v
        columns.append(("gt_views", cam_idx, "gt"))
        col_labels.append(f"GS-LRM 6v [{cam_name}]")

        # Ablation columns
        if n_views_list:
            for nv in sorted(n_views_list):
                columns.append(("ablation", cam_idx, nv))
                col_labels.append(f"{nv}v [{cam_name}]")

        n_cols = len(columns)
        canvas_w = RENDER_RESOLUTION * n_cols
        canvas_h = RENDER_RESOLUTION

        video_path = os.path.join(viz_dir, f"comparison_{cam_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (canvas_w, canvas_h))

        written = 0
        for frame_idx in frame_indices:
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 200

            for col_i, col_spec in enumerate(columns):
                if col_spec[2] == "gt":
                    img_path = get_gt_view_path(col_spec[0], col_spec[1], frame_idx, output_base)
                else:
                    nv = col_spec[2]
                    img_path = get_ablation_view_path(nv, col_spec[1], frame_idx, output_base)

                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (RENDER_RESOLUTION, RENDER_RESOLUTION))
                    else:
                        img = np.ones((RENDER_RESOLUTION, RENDER_RESOLUTION, 3), dtype=np.uint8) * 200
                else:
                    img = np.ones((RENDER_RESOLUTION, RENDER_RESOLUTION, 3), dtype=np.uint8) * 200

                cv2.putText(img, col_labels[col_i], (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(img, f"F{frame_idx:05d}", (5, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)

                x_off = col_i * RENDER_RESOLUTION
                canvas[:, x_off:x_off + RENDER_RESOLUTION] = img

            writer.write(canvas)
            written += 1

        writer.release()
        print(f"  Saved: {video_path} ({written} frames, {fps} fps)")

    print(f"\nVideo generation complete.")


# ============================================================
# Mode: Build manifest and splits
# ============================================================
def _save_global_cameras(output_base: str = OUTPUT_BASE):
    """Save global novel view camera parameters."""
    cameras_dir = os.path.join(output_base, "cameras")
    os.makedirs(cameras_dir, exist_ok=True)

    novel_cameras = generate_novel_cameras()
    config = {
        "coordinate_system": "facelift_normalized_opencv",
        "convention": "X-right, Y-down, Z-forward (OpenCV)",
        "resolution": RENDER_RESOLUTION,
        "intrinsics": {
            "fx": novel_cameras["bottom"]["fxfycxcy"][0],
            "fy": novel_cameras["bottom"]["fxfycxcy"][1],
            "cx": novel_cameras["bottom"]["fxfycxcy"][2],
            "cy": novel_cameras["bottom"]["fxfycxcy"][3],
            "source": "GT camera fx=548.99@512 scaled to 384",
        },
        "views": {},
    }

    for name, cam in novel_cameras.items():
        config["views"][name] = {
            "c2w": cam["c2w"],
            "elevation_deg": cam["elevation"],
            "azimuth_deg": cam["azimuth"],
            "radius": TURNTABLE_RADIUS,
        }

    path = os.path.join(cameras_dir, "novel_views.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved global camera config: {path}")


def _build_splits(output_base: str = OUTPUT_BASE):
    """Build train/val/test split JSONs from available metadata."""
    meta_dir = os.path.join(get_dataset_dir(output_base), "metadata")
    if not os.path.exists(meta_dir):
        print("  No metadata directory found, skipping splits.")
        return

    splits = {"train": [], "val": [], "test": []}

    for meta_file in sorted(os.listdir(meta_dir)):
        if not meta_file.endswith(".json"):
            continue
        frame_idx = int(meta_file.replace(".json", ""))
        split = get_split_name(frame_idx)
        splits[split].append(frame_idx)

    splits_dir = os.path.join(output_base, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    for split_name, indices in splits.items():
        path = os.path.join(splits_dir, f"{split_name}.json")
        with open(path, "w") as f:
            json.dump({
                "split": split_name,
                "n_frames": len(indices),
                "frame_indices": sorted(indices),
            }, f, indent=2)
        print(f"  {split_name}: {len(indices)} frames")


def _build_manifest(output_base: str = OUTPUT_BASE):
    """Build dataset-level manifest.json."""
    meta_dir = os.path.join(get_dataset_dir(output_base), "metadata")

    # Count frames and data availability
    total_frames = 0
    availability = {
        "tier0_raw": 0,
        "pseudo_gt": 0,
        "pseudo_gt_textured": 0,
        "gt_views": 0,
        "gt_rgb": 0,
        "tier1_cleaned": 0,
        "tier2_enhanced": 0,
        "artifact_masks": 0,
    }

    if os.path.exists(meta_dir):
        for meta_file in sorted(os.listdir(meta_dir)):
            if not meta_file.endswith(".json"):
                continue
            total_frames += 1
            with open(os.path.join(meta_dir, meta_file)) as f:
                meta = json.load(f)
            for key, available in meta.get("data_available", {}).items():
                if available and key in availability:
                    availability[key] += 1

    manifest = {
        "dataset_name": "FaceLift Novel View Mouse Dataset",
        "version": "1.0.0",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "species": "mouse",
        "source_dataset": "M5t2 (markerless_mouse_1_nerf, 6-camera)",
        "resolution": RENDER_RESOLUTION,
        "novel_views": list(NOVEL_VIEWS.keys()),
        "novel_view_params": {
            name: {"elevation": p["elevation"], "azimuth": p["azimuth"]}
            for name, p in NOVEL_VIEWS.items()
        },
        "total_frames": total_frames,
        "data_availability": availability,
        "splits": {
            name: {"start": s, "end": e}
            for name, (s, e) in M5T2_SPLITS.items()
        },
        "pipeline": {
            "stage1_model": "GS-LRM (6-view GT input)",
            "stage1_checkpoint": os.path.basename(GSLRM_CHECKPOINT),
            "pseudo_gt_source": "MAMMAL mesh fitting (v012345_kp22)",
            "tier0": "Raw GS-LRM novel view renders",
            "tier1": "OpenCV artifact removal (morphological + TELEA inpainting)",
            "tier2": "AI enhancement (Nano Banana / Gemini, qualitative only)",
        },
        "coordinate_transform": {
            "M5_SCENE_CENTER_mm": M5_SCENE_CENTER.tolist(),
            "M5_DISTANCE_SCALE": float(M5_DISTANCE_SCALE),
            "formula": "point_fl = (point_mm - center) * scale",
            "turntable_radius": TURNTABLE_RADIUS,
        },
        "target": "NeurIPS 2026 Evaluations & Datasets Track",
    }

    path = os.path.join(output_base, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {path} ({total_frames} frames)")


# ============================================================
# Mode: Visualize (comparison grids)
# ============================================================
def visualize(
    frame_indices: list[int],
    output_base: str = OUTPUT_BASE,
):
    """Generate comparison grids: tier0_raw vs pseudo_gt for novel views."""
    import cv2

    viz_dir = os.path.join(output_base, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    for frame_idx in frame_indices:
        print(f"  Frame {frame_idx}: ", end="")
        rows = []

        for view_name in VIEW_NAMES:
            tier0_path = get_view_path("tier0_raw", view_name, frame_idx, output_base)
            pgt_path = get_view_path("pseudo_gt", view_name, frame_idx, output_base)

            imgs = []
            labels = [f"GS-LRM [{view_name}]", f"MAMMAL [{view_name}]"]

            for path in [tier0_path, pgt_path]:
                if os.path.exists(path):
                    img = cv2.imread(path)
                    img = cv2.resize(img, (RENDER_RESOLUTION, RENDER_RESOLUTION))
                else:
                    img = np.ones((RENDER_RESOLUTION, RENDER_RESOLUTION, 3), dtype=np.uint8) * 200

                imgs.append(img)

            for img, label in zip(imgs, labels):
                cv2.putText(img, label, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img, f"F{frame_idx:05d}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

            row = np.concatenate(imgs, axis=1)
            rows.append(row)

        if rows:
            grid = np.concatenate(rows, axis=0)
            grid_path = os.path.join(viz_dir, f"novel_views_{frame_idx:05d}.png")
            cv2.imwrite(grid_path, grid)
            print(f"saved {grid_path}")
        else:
            print("no data")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Novel View Dataset Collection Pipeline v2.0"
    )
    parser.add_argument(
        "--mode",
        choices=["migrate", "generate", "manifest", "visualize", "video"],
        required=True,
    )
    parser.add_argument(
        "--phase",
        choices=["gslrm", "mammal", "ablation"],
        help="Generation phase (for --mode generate)",
    )
    parser.add_argument(
        "--poc_dir",
        default="/home/joon/dev/FaceLift/outputs/_archive/poc_v0",
        help="PoC directory to migrate from",
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_BASE,
        help="Output base directory",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=None,
        help="Specific frame indices",
    )
    parser.add_argument(
        "--frame_range",
        nargs=2,
        type=int,
        default=None,
        help="Frame range [start, end)",
    )
    parser.add_argument(
        "--n_views",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="View counts for ablation phase (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--checkpoint",
        default=GSLRM_CHECKPOINT,
    )
    parser.add_argument(
        "--config",
        default=GSLRM_CONFIG,
    )
    parser.add_argument(
        "--use-texture",
        action="store_true",
        help="Use UV texture for MAMMAL rendering (outputs to pseudo_gt_textured/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if outputs already exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview migration without copying files",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video FPS (default: 20)",
    )
    args = parser.parse_args()

    # Resolve frame indices
    if args.frame_range:
        frame_indices = list(range(args.frame_range[0], args.frame_range[1]))
    elif args.frames:
        frame_indices = args.frames
    else:
        frame_indices = None

    if args.mode == "migrate":
        migrate_poc(args.poc_dir, args.output_dir, dry_run=args.dry_run)

    elif args.mode == "generate":
        if not args.phase:
            print("ERROR: --phase required for generate mode")
            return
        if not frame_indices:
            print("ERROR: --frames or --frame_range required for generate mode")
            return

        if args.phase == "gslrm":
            generate_gslrm(
                frame_indices, args.output_dir, args.checkpoint, args.config,
                force=args.force,
            )
        elif args.phase == "mammal":
            generate_mammal(
                frame_indices, args.output_dir,
                force=args.force, use_texture=args.use_texture,
            )
        elif args.phase == "ablation":
            generate_ablation(
                frame_indices, args.n_views, args.output_dir, args.config,
                checkpoint=args.checkpoint, force=args.force,
            )

        # Rebuild manifest after generation
        _save_global_cameras(args.output_dir)
        _build_splits(args.output_dir)
        _build_manifest(args.output_dir)

    elif args.mode == "manifest":
        _save_global_cameras(args.output_dir)
        _build_splits(args.output_dir)
        _build_manifest(args.output_dir)

    elif args.mode == "video":
        generate_video(
            output_base=args.output_dir,
            fps=args.fps,
            n_views_list=args.n_views,
        )

    elif args.mode == "visualize":
        if not frame_indices:
            # Auto-select: first 5 frames from metadata
            meta_dir = os.path.join(get_dataset_dir(args.output_dir), "metadata")
            if os.path.exists(meta_dir):
                files = sorted(os.listdir(meta_dir))[:5]
                frame_indices = [int(f.replace(".json", "")) for f in files if f.endswith(".json")]
            else:
                print("ERROR: No metadata found. Provide --frames or --frame_range.")
                return
        visualize(frame_indices, args.output_dir)


if __name__ == "__main__":
    main()
