"""Deformation visualization: render original vs deformed Gaussians.

Generates:
  1. Temporal video at fixed novel camera (original + deformed side-by-side)
  2. Turntable orbit at key frames (original vs deformed)
  3. Temporal metrics (tOF, TLPIPS, FF-SSIM)

Reuses existing visualization infrastructure: render_cam, get_turntable_cameras,
save_video, compute_temporal_metrics.

Usage:
    # After Phase 0 cache generation:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.viz_deformation \
        --cache_dir /node_data/joon/outputs/FaceLift/gaussians/M5t2_temporal \
        --output_dir outputs/viz/deformation/baseline \
        --frame_range 3300:3320 --resolution 512

    # After Phase 1 training (with deformation checkpoint):
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.viz_deformation \
        --cache_dir /node_data/joon/outputs/FaceLift/gaussians/M5t2_temporal \
        --deform_ckpt /node_data/joon/checkpoints/FaceLift/deform_v2/M5t2/latest.pt \
        --output_dir outputs/viz/deformation/deform_v2 \
        --frame_range 3300:3320
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from mouse_extensions.visualization import save_video, get_turntable_cameras


# ---------------------------------------------------------------------------
# Gaussian loading (PT cache → GaussianModel)
# ---------------------------------------------------------------------------

def load_gaussian_from_pt(pt_path: str, device: str = "cuda"):
    """Load GaussianParams .pt file and set into a GaussianModel for rendering."""
    from gslrm.model.gaussians_renderer import GaussianModel

    data = torch.load(pt_path, map_location="cpu")

    # Handle both dict and GaussianParams formats
    if isinstance(data, dict):
        xyz = data["xyz"]
        features = data.get("features", data.get("features_dc"))
        scaling = data["scaling"]
        rotation = data["rotation"]
        opacity = data["opacity"]
    else:
        # GaussianParams object
        xyz = data.xyz
        features = data.features
        scaling = data.scaling
        rotation = data.rotation
        opacity = data.opacity

    # Ensure correct shapes
    if features.dim() == 2:
        features = features.unsqueeze(1)  # [N, F] → [N, 1, F] for SH dim

    gm = GaussianModel(sh_degree=0)
    gm.set_data(
        xyz.to(device),
        features.to(device),
        scaling.to(device),
        rotation.to(device),
        opacity.to(device),
    )
    return gm


def load_frame_sequence(
    cache_dir: str, frame_range: Tuple[int, int], device: str = "cuda"
) -> List[Tuple[int, object]]:
    """Load sequence of Gaussian frames from cache directory."""
    cache_path = Path(cache_dir)
    frames = []
    for fi in range(frame_range[0], frame_range[1]):
        pt_file = cache_path / f"frame_{fi:06d}.pt"
        if not pt_file.exists():
            print(f"  Skip missing: {pt_file.name}")
            continue
        gm = load_gaussian_from_pt(str(pt_file), device)
        frames.append((fi, gm))
    print(f"  Loaded {len(frames)} frames from {cache_dir}")
    return frames


# ---------------------------------------------------------------------------
# Rendering utilities (reuse existing infrastructure)
# ---------------------------------------------------------------------------

def render_gaussian_at_camera(gm, c2w, fxfy, w, h, bg=(1.0, 1.0, 1.0), device="cuda"):
    """Render a GaussianModel at given camera pose. Reuses render_opencv_cam."""
    from mouse_extensions.visualization import render_opencv_cam

    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    fxfy_t = torch.tensor(fxfy, dtype=torch.float32, device=device)
    with torch.no_grad():
        r = render_opencv_cam(gm, h, w, c2w_t, fxfy_t, bg_color=bg)
    img = r["render"].permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)


def make_novel_cameras(resolution: int = 512, radius: float = 2.7, hfov: float = 50):
    """Standard novel view cameras for deformation evaluation."""
    cameras = {}
    view_defs = [
        ("front", 20, 0),
        ("side", 20, 90),
        ("bottom", -70, 0),
        ("top", 70, 0),
    ]
    fx = resolution / (2 * np.tan(np.deg2rad(hfov) / 2.0))
    fxfy = np.array([fx, fx, resolution / 2.0, resolution / 2.0])

    for name, elev, azim in view_defs:
        elev_r, azim_r = np.deg2rad(elev), np.deg2rad(azim)
        z = radius * np.sin(elev_r)
        base = radius * np.cos(elev_r)
        pos = np.array([base * np.cos(azim_r), base * np.sin(azim_r), z])
        fwd = -pos / np.linalg.norm(pos)
        up_v = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_v)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack((right, -up, fwd), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, pos[:, None]), axis=1)
        cameras[name] = (c2w, fxfy)
    return cameras


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def render_temporal_video(
    frames: List[Tuple[int, object]],
    cam_name: str,
    c2w: np.ndarray,
    fxfy: np.ndarray,
    resolution: int,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Render temporal sequence at a fixed camera."""
    imgs = []
    for fi, gm in frames:
        img = render_gaussian_at_camera(gm, c2w, fxfy, resolution, resolution, device=device)
        # Add frame index label
        u8 = (img * 255).astype(np.uint8).copy()
        cv2.putText(u8, f"f{fi} | {cam_name}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(u8, f"f{fi} | {cam_name}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        imgs.append(u8)
    return imgs


def render_turntable_at_frame(
    gm, resolution: int = 512, n_views: int = 60,
    radius: float = 2.7, hfov: float = 50, elev: float = 20,
    device: str = "cuda",
) -> List[np.ndarray]:
    """Render 360° turntable orbit for a single Gaussian frame."""
    _, _, _, fxfy_arr, c2ws = get_turntable_cameras(
        hfov=hfov, num_views=n_views, w=resolution, h=resolution,
        radius=radius, elevation=elev)
    imgs = []
    for i in range(n_views):
        img = render_gaussian_at_camera(
            gm, c2ws[i], fxfy_arr[i], resolution, resolution, device=device)
        imgs.append((img * 255).astype(np.uint8))
    return imgs


def make_comparison_grid(img_a: np.ndarray, img_b: np.ndarray,
                         label_a: str = "Original", label_b: str = "Deformed") -> np.ndarray:
    """Side-by-side comparison with labels."""
    h, w = img_a.shape[:2]
    canvas = np.ones((h, w * 2 + 4, 3), dtype=np.uint8) * 200
    canvas[:, :w] = img_a
    canvas[:, w + 4:] = img_b
    for lbl, x0 in [(label_a, 8), (label_b, w + 12)]:
        cv2.putText(canvas, lbl, (x0, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (x0, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deformation Visualization")
    parser.add_argument("--cache_dir", required=True, help="Gaussian cache directory (frame_*.pt)")
    parser.add_argument("--deform_ckpt", default=None, help="Deformation checkpoint (optional)")
    parser.add_argument("--output_dir", default="outputs/viz/deformation/baseline")
    parser.add_argument("--frame_range", default="3300:3320", help="start:end")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = args.resolution

    # Parse frame range
    parts = args.frame_range.split(":")
    fr = (int(parts[0]), int(parts[1]))

    print(f"Loading Gaussians from {args.cache_dir} [{fr[0]}:{fr[1]}]...")
    frames = load_frame_sequence(args.cache_dir, fr, args.device)
    if not frames:
        print("ERROR: No frames loaded")
        return

    # Novel cameras (consistent with temporal eval standard)
    cameras = make_novel_cameras(res)

    # --- 1. Temporal videos at each novel camera ---
    print("\n=== Temporal Videos ===")
    for cam_name, (c2w, fxfy) in cameras.items():
        print(f"  Rendering {cam_name} ({len(frames)} frames)...")
        imgs = render_temporal_video(frames, cam_name, c2w, fxfy, res, args.device)
        path = str(out / f"temporal_{cam_name}.mp4")
        save_video(np.array(imgs), path, args.fps)
        print(f"  Saved: {path}")

    # --- 2. Turntable at key frames (first, middle, last) ---
    print("\n=== Turntable Orbits ===")
    key_indices = [0, len(frames) // 2, len(frames) - 1]
    for ki in key_indices:
        fi, gm = frames[ki]
        print(f"  Turntable at frame {fi}...")
        orbit_imgs = render_turntable_at_frame(gm, res, n_views=60, device=args.device)
        path = str(out / f"turntable_f{fi:06d}.mp4")
        save_video(np.array(orbit_imgs), path, 30)
        print(f"  Saved: {path}")

    # --- 3. 4-view temporal grid (2x2) ---
    print("\n=== 4-View Temporal Grid ===")
    grid_frames = []
    cam_names = list(cameras.keys())
    for frame_idx in range(len(frames)):
        fi, gm = frames[frame_idx]
        cell_imgs = []
        for cn in cam_names:
            c2w, fxfy = cameras[cn]
            img = render_gaussian_at_camera(gm, c2w, fxfy, res, res, device=args.device)
            u8 = (img * 255).astype(np.uint8)
            # Resize to half for 2x2 grid
            cell = cv2.resize(u8, (res // 2, res // 2), interpolation=cv2.INTER_AREA)
            # Add view label
            cv2.putText(cell, cn, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.putText(cell, cn, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (200, 200, 200), 1, cv2.LINE_AA)
            cell_imgs.append(cell)
        # Compose 2x2 grid
        half = res // 2
        grid = np.ones((res, res, 3), dtype=np.uint8) * 255
        grid[:half, :half] = cell_imgs[0]
        grid[:half, half:] = cell_imgs[1]
        grid[half:, :half] = cell_imgs[2]
        grid[half:, half:] = cell_imgs[3]
        grid_frames.append(grid)

    path = str(out / "temporal_4view_grid.mp4")
    save_video(np.array(grid_frames), path, args.fps)
    print(f"  Saved: {path}")

    # --- 4. Temporal metrics ---
    print("\n=== Temporal Metrics ===")
    try:
        from mouse_extensions.evaluation.temporal_smoothing import compute_temporal_metrics
        for cam_name in cam_names:
            c2w, fxfy = cameras[cam_name]
            rendered = []
            for fi, gm in frames:
                img = render_gaussian_at_camera(gm, c2w, fxfy, res, res, device=args.device)
                rendered.append((img * 255).astype(np.uint8))
            metrics = compute_temporal_metrics(rendered)
            print(f"  {cam_name}: tOF={metrics.tof_mean:.4f} "
                  f"TLPIPS={metrics.tlpips_mean:.4f} "
                  f"FF-SSIM-var={metrics.ff_ssim_var:.6f}")
    except Exception as e:
        print(f"  Metrics skipped: {e}")

    # --- 5. Summary ---
    n_files = len(list(out.glob("*.mp4")))
    print(f"\n✅ DEFORM_VIZ_DONE: {n_files} videos → {out}")


if __name__ == "__main__":
    main()
