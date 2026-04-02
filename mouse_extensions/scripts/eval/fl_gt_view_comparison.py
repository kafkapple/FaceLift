"""FL vs GT comparison grid video for representative viewpoints.

Renders FaceLift (GS-LRM) reconstruction side-by-side with GT input images
for 4 representative camera angles: bottom, top, side-left, side-right.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.fl_gt_view_comparison \
        --config mouse_extensions/behavior/cinematic_default.yaml \
        --output-dir outputs/viz/comparison/mouse/fl_gt_4views \
        --frame-range 195:315

Output:
    fl_gt_comparison.mp4  — side-by-side 2-row grid (GT top, FL bottom), 4 cols
    fl_gt_comparison_*.mp4  — per-view individual videos
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from mouse_extensions.behavior.camera_system import gt_camera_c2w_original as _gt_c2w
from mouse_extensions.behavior.camera_system import gt_camera_fxfycxcy as _gt_fxfy
from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts
from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.visualization.camera_utils import get_turntable_cameras
from mouse_extensions.visualization.video_io import save_video


# ---------------------------------------------------------------------------
# Representative view definitions (elevation, azimuth in degrees)
# ---------------------------------------------------------------------------
REPRESENTATIVE_VIEWS = {
    "bottom":     {"elevation": -80, "azimuth": 0,   "label": "Bottom (-80°)"},
    "top":        {"elevation":  60, "azimuth": 0,   "label": "Top (+60°)"},
    "side_left":  {"elevation":   5, "azimuth": 90,  "label": "Side Left (90°)"},
    "side_right": {"elevation":   5, "azimuth": 270, "label": "Side Right (270°)"},
}


def get_novel_camera(elevation: float, azimuth: float, radius: float = 2.7,
                     hfov: float = 50, resolution: int = 512) -> Tuple:
    """Get (c2w [4,4], fxfycxcy [4]) for a single turntable view.

    Computes c2w directly from elevation and azimuth (degrees) without
    relying on get_turntable_cameras azimuth_start (which does not exist).
    """
    import math

    elev_rad = math.radians(elevation)
    azim_rad = math.radians(azimuth)
    up_vector = np.array([0.0, 0.0, 1.0])
    center = np.zeros(3)

    z = radius * math.sin(elev_rad)
    base = radius * math.cos(elev_rad)
    cam_pos = np.array([base * math.cos(azim_rad), base * math.sin(azim_rad), z])

    forward = center - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_vector)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.stack((right, -up, forward), axis=1)
    c2w = np.eye(4)
    c2w[:3, :4] = np.concatenate((R, cam_pos[:, None]), axis=1)

    fx = (resolution / 2) / math.tan(math.radians(hfov / 2))
    fxfycxcy = np.array([fx, fx, resolution / 2, resolution / 2], dtype=np.float32)

    return c2w, fxfycxcy


def render_gaussian_at_view(gaussians, camera, resolution: int = 512,
                            bg_color=(1.0, 1.0, 1.0), device="cuda",
                            vis_mask: np.ndarray = None):
    """Render 3D Gaussians at a specific camera viewpoint.

    Args:
        vis_mask: Optional boolean numpy array (N,). If provided, Gaussians
                  where mask is False are temporarily set to invisible.
    """
    from mouse_extensions.visualization import render_opencv_cam

    c2w, fxfycxcy = camera
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    fxfy_t = torch.tensor(fxfycxcy, dtype=torch.float32, device=device)

    # Apply visibility mask by zeroing out opacity for excluded Gaussians
    old_opacity = None
    if vis_mask is not None:
        mask_t = torch.from_numpy(vis_mask).to(device)
        old_opacity = gaussians._opacity.data.clone()
        gaussians._opacity.data[~mask_t] = -100.0

    try:
        with torch.no_grad():
            result = render_opencv_cam(
                gaussians,
                height=resolution,
                width=resolution,
                C2W=c2w_t,
                fxfycxcy=fxfy_t,
                bg_color=bg_color,
            )
        img = result["render"]  # (C, H, W)
        img_np = img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    finally:
        if old_opacity is not None:
            gaussians._opacity.data = old_opacity

    return (img_np * 255).astype(np.uint8)


def load_gt_image(frame_dir: str, view_idx: int, resolution: int = 512) -> np.ndarray:
    """Load GT camera image for a given view index."""
    img_path = Path(frame_dir) / "images" / f"cam_{view_idx:03d}.png"
    if not img_path.exists():
        img_path = Path(frame_dir) / f"cam_{view_idx:03d}.png"

    import cv2
    from PIL import Image
    img = Image.open(str(img_path))
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    img = img.resize((resolution, resolution), Image.LANCZOS)
    return np.array(img)


def add_label(img: np.ndarray, text: str, row_label: str = "",
              color=(255, 255, 0), outline: bool = False) -> np.ndarray:
    """Add overlay labels to image.

    Args:
        text: Primary label (top-left).
        row_label: Secondary label (bottom-left, smaller).
        color: Text color (B,G,R). Default yellow for legacy compat.
        outline: If True, draw dark outline behind text for contrast.
    """
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Primary label (top-left)
    if outline:
        cv2.putText(out, text, (8, 28), font, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (8, 28), font, 0.65, color, 2, cv2.LINE_AA)
    # Secondary label (bottom-left, smaller)
    if row_label:
        if outline:
            cv2.putText(out, row_label, (8, out.shape[0] - 10),
                        font, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, row_label, (8, out.shape[0] - 10),
                    font, 0.45, color, 1, cv2.LINE_AA)
    return out


def generate_comparison_video(
    config_path: str,
    output_dir: str,
    frame_range: str = "195:315",
    resolution: int = 512,
    fps: int = 15,
    n_filter: int = 2,
    device: str = "cuda",
):
    """Generate FL vs GT comparison grid video for 4 representative views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m5_dir = Path(cfg["model"]["m5_dir"])
    resolution = cfg["global"].get("resolution", resolution)
    fps = cfg["global"].get("fps", fps)
    n_filter = cfg["global"].get("n_filter", n_filter)

    # Parse frame range
    start, end = map(int, frame_range.split(":"))
    frame_indices = list(range(start, end))

    # Load model
    print("Loading GS-LRM model...")
    model = GSLRMInference(
        config_path=cfg["model"]["config"],
        checkpoint_path=cfg["model"]["checkpoint"],
        device=device,
    )
    # Apply num_input_views override
    if "num_input_views" in cfg.get("model", {}):
        model.config.model.num_input_views = cfg["model"]["num_input_views"]
        print(f"  Patched num_input_views = {cfg['model']['num_input_views']}")

    # Pre-compute novel cameras for representative views
    view_names = list(REPRESENTATIVE_VIEWS.keys())
    cameras = {}
    for vname, vdef in REPRESENTATIVE_VIEWS.items():
        cameras[vname] = get_novel_camera(
            elevation=vdef["elevation"],
            azimuth=vdef["azimuth"],
            resolution=resolution,
        )

    # GT camera index closest to each novel view (use first GT cam for GT display)
    # We'll display GT cam 0 image for all novel-view columns (honest: novel views don't have GT)
    # Instead: show nearest GT input camera image for each novel view direction

    n_views = len(REPRESENTATIVE_VIEWS)  # 4
    H, W = resolution, resolution

    # Video writers — one combined, one per view
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    combined_path = str(output_dir / "fl_gt_comparison.mp4")
    combined_writer = cv2.VideoWriter(combined_path, fourcc, fps, (W * n_views, H * 2))

    per_view_writers = {}
    for vname in view_names:
        vpath = str(output_dir / f"fl_gt_{vname}.mp4")
        per_view_writers[vname] = cv2.VideoWriter(vpath, fourcc, fps, (W * 2, H))

    print(f"Generating {len(frame_indices)} frames × {n_views} views...")

    for fi in frame_indices:
        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            continue

        # Load 6-view data and infer Gaussians
        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=resolution, device=device)
        result = model.predict(imgs, c2ws, fxfys, idx)
        gaussians = result.gaussians[0]

        # Apply visibility filter (n_filter >= 2)
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        vc = compute_visibility_counts(xyz, str(fd), n_views=6)
        vis_mask = vc >= n_filter

        # Load GT image (view 0 as reference, shown as "GT Input")
        try:
            gt_img = load_gt_image(str(fd), view_idx=0, resolution=resolution)
        except Exception:
            gt_img = np.full((H, W, 3), 128, dtype=np.uint8)

        # Render FL at each representative view + build grid
        gt_row_cells = []
        fl_row_cells = []

        for vname, vdef in REPRESENTATIVE_VIEWS.items():
            cam = cameras[vname]
            label = vdef["label"]

            # FL render at novel view (vis_mask applied inside render function)
            try:
                fl_img = render_gaussian_at_view(
                    gaussians, cam, resolution=resolution, device=device,
                    vis_mask=vis_mask,
                )
            except Exception as e:
                print(f"  [warn] render failed for {vname} frame {fi}: {e}")
                fl_img = np.full((H, W, 3), 80, dtype=np.uint8)

            # GT display: use GT input image (view 0) for all columns
            # Label it clearly as "GT Input View 0"
            gt_cell = add_label(gt_img.copy(), f"GT | cam_000", "input view")
            fl_cell = add_label(fl_img, f"FL | {label}", "reconstruction")

            gt_row_cells.append(gt_cell)
            fl_row_cells.append(fl_cell)

            # Per-view video: [GT | FL] side by side
            per_frame = np.hstack([gt_cell, fl_cell])
            per_view_writers[vname].write(cv2.cvtColor(per_frame, cv2.COLOR_RGB2BGR))

        # Combined grid: 2 rows × 4 cols
        gt_row = np.hstack(gt_row_cells)
        fl_row = np.hstack(fl_row_cells)

        # Frame number overlay
        combined = np.vstack([gt_row, fl_row])
        cv2.putText(
            combined, f"Frame {fi:04d}",
            (8, H * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        combined_writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if fi % 20 == 0:
            print(f"  frame {fi}/{end}")

    # Release
    combined_writer.release()
    for w in per_view_writers.values():
        w.release()

    print(f"\nSaved:")
    print(f"  Combined: {combined_path}")
    for vname, vdef in REPRESENTATIVE_VIEWS.items():
        print(f"  {vname}: {output_dir}/fl_gt_{vname}.mp4")


def main():
    parser = argparse.ArgumentParser(description="FL vs GT comparison grid video")
    parser.add_argument("--config", default="mouse_extensions/behavior/cinematic_default.yaml")
    parser.add_argument("--output-dir", default="outputs/viz/comparison/mouse/fl_gt_4views")
    parser.add_argument("--frame-range", default="195:315")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    generate_comparison_video(
        config_path=args.config,
        output_dir=args.output_dir,
        frame_range=args.frame_range,
        resolution=args.resolution,
        fps=args.fps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
