"""Render body-part Gaussians via GS-LRM inference with proper colors.

Runs GS-LRM inference → gets full GaussianModel (with SH colors) →
applies body-part mask → renders each part separately with correct appearance.

Also renders unfiltered (all Gaussians) as control.

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.behavior.render_bodypart_gaussians \
        --frame-idx 0 500 1000 \
        --views 0 2 4 \
        --output-dir outputs/viz/bodypart/mouse/renders
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from mouse_extensions.constants import SKELETON_BONES  # SSOT (2026-03-23)
from mouse_extensions.behavior.view_projected_filtering import (
    KP_NAMES, BODY_PARTS, BODY_PART_COLORS, load_camera, load_keypoints_gslrm,
    project_points_to_2d,
)

# Body part to bone segment mapping for 3D assignment
# Each keypoint's body part
KP_TO_PART = {}
for part_name, kp_indices in BODY_PARTS.items():
    for ki in kp_indices:
        KP_TO_PART[ki] = part_name

BONE_SEGMENTS = [
    (2, 0), (2, 1), (2, 3),
    (3, 4), (4, 5), (5, 6), (6, 7),
    (3, 11), (11, 10), (10, 8), (8, 9),
    (3, 15), (15, 14), (14, 12), (12, 13),
    (5, 18), (18, 17), (17, 16),   # tail_root→L_hip→L_knee→L_foot (matches YAML skeleton)
    (5, 21), (21, 20), (20, 19),   # tail_root→R_hip→R_knee→R_foot
]


def point_to_segment_distance(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Compute distance from points to line segment [a, b]."""
    ab = seg_b - seg_a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return np.linalg.norm(points - seg_a, axis=1)
    ap = points - seg_a
    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    proj = seg_a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def assign_gaussians_to_bodyparts_3d(
    xyz: np.ndarray, keypoints_3d: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Assign Gaussians to body parts via nearest bone segment."""
    N = len(xyz)
    best_dist = np.full(N, np.inf)
    best_part = np.empty(N, dtype=object)

    for ki_a, ki_b in BONE_SEGMENTS:
        seg_a = keypoints_3d[ki_a]
        seg_b = keypoints_3d[ki_b]
        dists = point_to_segment_distance(xyz, seg_a, seg_b)

        part_a = KP_TO_PART.get(ki_a)
        part_b = KP_TO_PART.get(ki_b)
        if part_a == part_b:
            part = part_a
        elif part_a == "torso":
            part = part_b
        elif part_b == "torso":
            part = part_a
        else:
            part = part_a

        if part is None:
            continue

        closer = dists < best_dist
        best_dist[closer] = dists[closer]
        best_part[closer] = part

    part_masks = {}
    for part_name in BODY_PARTS:
        part_masks[part_name] = (best_part == part_name)
    return part_masks


def render_gaussians_masked(
    gaussians, mask_tensor: torch.Tensor, cam: Dict, device: str = "cuda",
) -> np.ndarray:
    """Render only masked Gaussians using the real GS-LRM renderer.

    Creates a shallow copy of the GaussianModel with zeroed-out opacity
    for non-selected Gaussians, preserving SH colors.
    """
    from mouse_extensions.visualization import render_opencv_cam

    # Create a copy with masked opacity
    masked_gaussians = copy.copy(gaussians)

    # Zero out opacity for non-selected Gaussians
    original_opacity = gaussians.get_opacity.clone()
    masked_opacity = original_opacity.clone()
    masked_opacity[~mask_tensor] = 0.0

    # Temporarily replace opacity
    # Store in _opacity field (pre-activation) — but we need to handle this
    # depending on GaussianModel implementation
    old_opacity_raw = gaussians._opacity.data.clone()
    # Set non-masked to very negative logit so sigmoid → 0
    gaussians._opacity.data[~mask_tensor] = -100.0

    w2c = torch.tensor(cam["w2c"], dtype=torch.float32, device=device)
    c2w = torch.inverse(w2c)
    fxfycxcy = torch.tensor(
        [cam["fx"], cam["fy"], cam["cx"], cam["cy"]],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        result = render_opencv_cam(
            gaussians, cam["h"], cam["w"], c2w, fxfycxcy,
            bg_color=(1.0, 1.0, 1.0),  # White background
        )

    rendered = result["render"].permute(1, 2, 0).cpu().numpy()

    # Restore original opacity
    gaussians._opacity.data = old_opacity_raw

    return np.clip(rendered, 0, 1)


def render_all_gaussians(
    gaussians, cam: Dict, device: str = "cuda",
) -> np.ndarray:
    """Render all Gaussians (control/unfiltered)."""
    from mouse_extensions.visualization import render_opencv_cam

    w2c = torch.tensor(cam["w2c"], dtype=torch.float32, device=device)
    c2w = torch.inverse(w2c)
    fxfycxcy = torch.tensor(
        [cam["fx"], cam["fy"], cam["cx"], cam["cy"]],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        result = render_opencv_cam(
            gaussians, cam["h"], cam["w"], c2w, fxfycxcy,
            bg_color=(1.0, 1.0, 1.0),
        )

    rendered = result["render"].permute(1, 2, 0).cpu().numpy()
    return np.clip(rendered, 0, 1)


# MAMMAL-standard keypoint colors (RGB, consistent with keypoint_overlay.py)
# Converted from BGR (OpenCV) → RGB (matplotlib)
MAMMAL_KP_COLORS = {
    # head: yellow
    0: "#FFFF00", 1: "#FFFF00", 2: "#FFFF00",
    # body: magenta
    3: "#FF00FF", 4: "#FF00FF",
    # tail: orange
    5: "#FFA500", 6: "#FFA500", 7: "#FFA500",
    # left_front: blue
    8: "#0000FF", 9: "#0000FF", 10: "#0000FF", 11: "#0000FF",
    # right_front: green
    12: "#00FF00", 13: "#00FF00", 14: "#00FF00", 15: "#00FF00",
    # left_hind: cyan
    16: "#00FFFF", 17: "#00FFFF", 18: "#00FFFF",
    # right_hind: red
    19: "#FF0000", 20: "#FF0000", 21: "#FF0000",
}


def draw_keypoints_on_image(
    image: np.ndarray,
    kp_2d: np.ndarray,
    kp_valid: np.ndarray,
    highlight_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Draw keypoints + skeleton on image with consistent body-part colors."""
    H, W = image.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(image)

    # Skeleton bones — thin white with slight transparency
    for ki_a, ki_b in SKELETON_BONES:
        if kp_valid[ki_a] and kp_valid[ki_b]:
            ax.plot(
                [kp_2d[ki_a, 0], kp_2d[ki_b, 0]],
                [kp_2d[ki_a, 1], kp_2d[ki_b, 1]],
                color="white", linewidth=1.0, alpha=0.5, zorder=5,
            )

    # Keypoints — MAMMAL-standard colors, highlighted ones get white ring
    for i in range(len(kp_2d)):
        if not kp_valid[i]:
            continue
        base_color = MAMMAL_KP_COLORS.get(i, "#FFFFFF")
        is_hl = highlight_indices is not None and i in highlight_indices
        ms = 8 if is_hl else 5
        ew = 2.0 if is_hl else 0.8
        ec = "white" if is_hl else "black"
        ax.plot(kp_2d[i, 0], kp_2d[i, 1], "o", markersize=ms, color=base_color,
                markeredgecolor=ec, markeredgewidth=ew, zorder=10)

    ax.axis("off")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if buf.shape[:2] != (H, W):
        from PIL import Image as PILImage
        buf = np.array(PILImage.fromarray(buf).resize((W, H), PILImage.LANCZOS))
    return buf


def process_frame(
    frame_idx: int,
    model,  # GSLRMInference
    kp_path: str,
    m5_dir: str,
    views: List[int],
    output_dir: Path,
    device: str = "cuda",
    selected_parts: Optional[List[str]] = None,
    save_images: bool = True,
):
    """Run GS-LRM inference, then render per body part across views."""
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data

    frame_dir = Path(m5_dir) / f"{frame_idx:06d}"
    if not frame_dir.exists():
        print(f"  Frame {frame_idx}: data dir not found")
        return

    # Run GS-LRM inference (full pipeline with SH colors)
    images, c2ws, fxfycxcys, index = load_sample_data(
        str(frame_dir), image_size=512, device=device,
    )
    result = model.predict(images, c2ws, fxfycxcys, index)
    gaussians = result.gaussians[0]  # GaussianModel with full SH

    xyz_np = gaussians.get_xyz.detach().cpu().numpy()
    N = len(xyz_np)
    print(f"  GS-LRM: {N:,} Gaussians (with SH colors)")

    # Load keypoints
    kp_gslrm = load_keypoints_gslrm(kp_path, frame_idx)

    # 3D body-part assignment
    part_masks_np = assign_gaussians_to_bodyparts_3d(xyz_np, kp_gslrm)
    for pn, m in part_masks_np.items():
        print(f"    {pn}: {m.sum():,}")

    all_part_names = list(BODY_PARTS.keys())
    part_names = [p for p in all_part_names if selected_parts is None or p in selected_parts]
    part_kp_desc = {
        "face": "nose, L/R ear, neck",
        "left_paw": "L_paw, L_paw_end, L_elbow",
        "right_paw": "R_paw, R_paw_end, R_elbow",
        "tail": "tail_root/mid/end",
        "torso": "body_mid, shoulders, hips",
    }
    n_views = len(views)
    row_labels = ["GT Image", "All Gaussians (control)"] + [
        f"{pn} ({part_kp_desc[pn]})" for pn in part_names
    ]
    n_rows = len(row_labels)

    # Pre-render all views' data
    view_data = []
    for view_idx in views:
        cam_path = frame_dir / "opencv_cameras.json"
        cam = load_camera(str(cam_path), view_idx)
        img_path = frame_dir / "images" / f"cam_{view_idx:03d}.png"
        if img_path.exists():
            from PIL import Image
            gt_image = np.array(Image.open(img_path))[:, :, :3] / 255.0
        else:
            gt_image = np.ones((512, 512, 3)) * 0.5

        kp_2d, kp_valid = project_points_to_2d(
            kp_gslrm, np.array(cam["w2c"]),
            cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        )
        kp_in_img = (
            kp_valid
            & (kp_2d[:, 0] >= 0) & (kp_2d[:, 0] < cam["w"])
            & (kp_2d[:, 1] >= 0) & (kp_2d[:, 1] < cam["h"])
        )

        all_render = render_all_gaussians(gaussians, cam, device)
        part_renders = {}
        for part_name in part_names:  # Only selected parts
            mask_np = part_masks_np[part_name]
            mask_t = torch.from_numpy(mask_np).to(device)
            part_renders[part_name] = render_gaussians_masked(gaussians, mask_t, cam, device)

        view_data.append({
            "view_idx": view_idx, "cam": cam, "gt": gt_image,
            "kp_2d": kp_2d, "kp_in_img": kp_in_img,
            "all_render": all_render, "part_renders": part_renders,
        })

    if not save_images:
        # Skip image generation, just return render data for video
        frame_renders = {
            "frame_idx": frame_idx,
            "view_data": view_data,
            "part_masks": part_masks_np,
        }
        del result, gaussians, images, c2ws, fxfycxcys
        torch.cuda.empty_cache()
        return frame_renders

    # Generate TWO versions: with and without keypoints
    for show_kp, suffix in [(True, "kp"), (False, "clean")]:
        fig, axes = plt.subplots(n_rows, n_views, figsize=(5 * n_views, 4.5 * n_rows))
        if n_views == 1:
            axes = axes[:, None]

        for vi, vd in enumerate(view_data):
            kp_2d = vd["kp_2d"]
            kp_in_img = vd["kp_in_img"]

            def _maybe_kp(img, hl=None):
                if show_kp:
                    return draw_keypoints_on_image(img, kp_2d, kp_in_img, hl)
                return (img * 255).astype(np.uint8) if img.max() <= 1.0 else img

            # Row 0: GT
            ax = axes[0, vi]
            ax.imshow(_maybe_kp(vd["gt"]))
            ax.set_title(f"View {vd['view_idx']}", fontsize=14, fontweight="bold")
            ax.axis("off")

            # Row 1: All Gaussians
            ax = axes[1, vi]
            ax.imshow(_maybe_kp(vd["all_render"]))
            ax.axis("off")

            # Rows 2+: Body parts
            for pi, part_name in enumerate(part_names):
                ax = axes[pi + 2, vi]
                render = vd["part_renders"][part_name]
                hl = BODY_PARTS[part_name] if show_kp else None
                ax.imshow(_maybe_kp(render, hl))
                n_gauss = int(part_masks_np[part_name].sum())
                ax.set_title(f"{n_gauss:,} Gaussians", fontsize=12)
                ax.axis("off")

        # Row labels on left — use text annotation for reliable placement
        for ri, label in enumerate(row_labels):
            if ri >= 2:
                pn = part_names[ri - 2]
                color = BODY_PART_COLORS[pn]
            else:
                color = "black"
            axes[ri, 0].annotate(
                label, xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-10, 0), textcoords="offset points",
                fontsize=13, fontweight="bold", color=color,
                ha="right", va="center", rotation=90,
            )

        kp_tag = "with keypoints" if show_kp else "no keypoints"
        plt.suptitle(
            f"Body-Part Gaussian Rendering — Frame {frame_idx} ({kp_tag})\n"
            f"GS-LRM 6-view inference | White BG = no Gaussians",
            fontsize=16, fontweight="bold",
        )
        plt.tight_layout(rect=[0.12, 0, 1, 0.96])
        out_path = output_dir / f"frame_{frame_idx:06d}_{suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    # Return raw renders for video generation
    frame_renders = {
        "frame_idx": frame_idx,
        "view_data": view_data,
        "part_masks": part_masks_np,
    }

    # Cleanup
    del result, gaussians, images, c2ws, fxfycxcys
    torch.cuda.empty_cache()

    return frame_renders


def _generate_videos(
    all_frame_renders: List[Dict],
    views: List[int],
    output_dir: Path,
    fps: int = 10,
):
    """Generate per-body-part 6-view GRID MP4 videos from consecutive frame renders."""
    try:
        import cv2
    except ImportError:
        print("  cv2 not available, skipping video generation")
        return

    first_parts = list(all_frame_renders[0]["view_data"][0]["part_renders"].keys())
    n_views = len(views)

    # For each body part + "all", create a GRID video (all views side by side)
    for part_name in ["all"] + first_parts:
        grid_frames = []
        for fr in all_frame_renders:
            row_imgs = []
            for vd in fr["view_data"]:
                if part_name == "all":
                    img = vd["all_render"]
                else:
                    img = vd["part_renders"].get(part_name)
                    if img is None:
                        continue
                img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                row_imgs.append(img_u8)

            if not row_imgs:
                continue

            # Arrange views in a grid: 2 rows × 3 cols for 6 views, or 1 row for ≤3
            if n_views <= 3:
                grid = np.concatenate(row_imgs, axis=1)
            else:
                # 2 rows
                n_top = (n_views + 1) // 2
                top = np.concatenate(row_imgs[:n_top], axis=1)
                bot_imgs = row_imgs[n_top:]
                # Pad if uneven
                if len(bot_imgs) < n_top:
                    h, w = row_imgs[0].shape[:2]
                    bot_imgs.append(np.ones((h, w, 3), dtype=np.uint8) * 255)
                bot = np.concatenate(bot_imgs[:n_top], axis=1)
                grid = np.concatenate([top, bot], axis=0)

            grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            grid_frames.append(grid_bgr)

        if not grid_frames:
            continue

        h, w = grid_frames[0].shape[:2]
        video_path = output_dir / f"video_{part_name}_grid.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )
        for f in grid_frames:
            writer.write(f)
        writer.release()
        print(f"  Grid video: {video_path} ({len(grid_frames)} frames, {n_views}-view grid)")


def _generate_novel_view_videos(
    all_frame_renders: List[Dict],
    model,
    m5_dir: str,
    output_dir: Path,
    device: str = "cuda",
    fps: int = 10,
):
    """Generate novel view videos: extrapolated (bottom/top) + interpolated.

    Uses get_turntable_cameras for novel camera generation.
    """
    try:
        import cv2
        from mouse_extensions.visualization import render_opencv_cam, get_turntable_cameras
    except ImportError as e:
        print(f"  Novel views skipped: {e}")
        return

    frame_indices = [fr["frame_idx"] for fr in all_frame_renders]
    n_frames = len(frame_indices)

    # Define novel view sets
    novel_configs = {
        "extrapolated": [
            ("bottom_-30", -30),
            ("level_0", 0),
            ("top_20", 20),
            ("top_40", 40),
            ("top_60", 60),
            ("top_80", 80),
        ],
    }

    for config_name, elev_list in novel_configs.items():
        n_novel = len(elev_list)
        grid_frames = []

        for fr in all_frame_renders:
            fi = fr["frame_idx"]
            # Need to re-run inference for this frame's gaussians
            # (already done in process_frame, but gaussians were freed)
            # Use cached view_data's cam for intrinsics reference
            vd0 = fr["view_data"][0]
            cam = vd0["cam"]

            # Re-infer for novel views
            from mouse_extensions.inference.gslrm_pipeline import load_sample_data
            frame_dir = Path(m5_dir) / f"{fi:06d}"
            images, c2ws, fxfycxcys, index = load_sample_data(
                str(frame_dir), image_size=512, device=device,
            )
            result = model.predict(images, c2ws, fxfycxcys, index)
            gaussians = result.gaussians[0]

            # Render at each novel elevation (fixed azimuth = 0)
            row_imgs = []
            for label, elev in elev_list:
                cams = get_turntable_cameras(
                    hfov=50, num_views=1, w=512, h=512,
                    radius=2.7, elevation=elev,
                )
                c2w_novel = torch.tensor(cams[0], dtype=torch.float32, device=device)
                fxfycxcy_novel = torch.tensor(
                    [cam["fx"], cam["fy"], cam["cx"], cam["cy"]],
                    dtype=torch.float32, device=device,
                )
                with torch.no_grad():
                    out = render_opencv_cam(
                        gaussians, 512, 512, c2w_novel, fxfycxcy_novel,
                        bg_color=(1.0, 1.0, 1.0),
                    )
                img = out["render"].permute(1, 2, 0).cpu().numpy()
                img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                row_imgs.append(img_u8)

            # Make 2×3 or 1×N grid
            if n_novel <= 3:
                grid = np.concatenate(row_imgs, axis=1)
            else:
                n_top = (n_novel + 1) // 2
                top = np.concatenate(row_imgs[:n_top], axis=1)
                bot = row_imgs[n_top:]
                if len(bot) < n_top:
                    bot.append(np.ones_like(row_imgs[0]) * 255)
                bot = np.concatenate(bot[:n_top], axis=1)
                grid = np.concatenate([top, bot], axis=0)

            grid_frames.append(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

            del result, gaussians, images, c2ws, fxfycxcys
            torch.cuda.empty_cache()

        if grid_frames:
            h, w = grid_frames[0].shape[:2]
            path = output_dir / f"novel_{config_name}_grid.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for f in grid_frames:
                writer.write(f)
            writer.release()
            print(f"  Novel view ({config_name}): {path} ({n_frames}f, {n_novel} views)")


def _generate_temporal_turntable(
    model, frame_list: List[int], m5_dir: str, output_dir: Path,
    device: str = "cuda", fps: int = 10,
):
    """Generate temporal turntable: orbit rotates while frames progress.

    Uses existing TemporalVideoRenderer for time_fixed + time_rotating videos.
    """
    try:
        from mouse_extensions.inference.gslrm_pipeline import load_sample_data
        from mouse_extensions.visualization import render_turntable
        from mouse_extensions.visualization.turntable_renderer import (
            TemporalVideoRenderer, TurntableVideoConfig,
        )
    except ImportError as e:
        print(f"  Temporal turntable skipped: {e}")
        return

    all_turntables = []
    for fi in frame_list:
        frame_dir = Path(m5_dir) / f"{fi:06d}"
        if not frame_dir.exists():
            continue

        images, c2ws, fxfycxcys, index = load_sample_data(
            str(frame_dir), image_size=512, device=device,
        )
        result = model.predict(images, c2ws, fxfycxcys, index)
        gaussians = result.gaussians[0]

        # Render turntable for this frame (36 views for temporal)
        strip = render_turntable(
            gaussians, rendering_resolution=512, num_views=36,
            elevation=20, radius=2.7, trajectory_mode="turntable",
        )
        if strip.dtype != np.uint8:
            strip = (strip * 255).clip(0, 255).astype(np.uint8)

        h = strip.shape[0]
        w = strip.shape[1] // 36
        frames_arr = strip.reshape(h, 36, w, 3).transpose(1, 0, 2, 3)  # [V, H, W, 3]
        all_turntables.append(frames_arr)

        del result, gaussians, images, c2ws, fxfycxcys
        torch.cuda.empty_cache()

    if len(all_turntables) < 2:
        print("  Need ≥2 frames for temporal turntable")
        return

    # Use existing TemporalVideoRenderer
    cfg = TurntableVideoConfig()
    cfg.temporal_fps = fps
    renderer = TemporalVideoRenderer(config=cfg)

    turntable_dir = str(output_dir / "temporal_turntable")
    results = renderer.render_temporal(
        all_turntables,
        turntable_dir,
        fps=fps,
        fixed_angles=[0, 9, 18, 27],  # 4 views: front, right, back, left
        rotation_speed=1.0,
    )
    for name, path in results.items():
        print(f"  Temporal turntable ({name}): {path}")


def main():
    from mouse_extensions.behavior.paths import GPU03_KEYPOINTS, GPU03_M5_DATA

    parser = argparse.ArgumentParser(description="Body-Part Gaussian Rendering (GS-LRM)")
    parser.add_argument("--frame-idx", nargs="+", type=int, default=[0, 500, 1000],
                        help="Frame indices. For video, use consecutive frames e.g. 0 1 2 3 4")
    parser.add_argument("--frame-range", type=str, default=None,
                        help="Consecutive frame range 'start:end' e.g. '0:30' (overrides --frame-idx)")
    parser.add_argument("--views", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--checkpoint", default="checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt")
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--kp-path", default=GPU03_KEYPOINTS)
    parser.add_argument("--m5-dir", default=GPU03_M5_DATA)
    parser.add_argument("--output-dir", default="outputs/viz/bodypart/mouse/renders")
    parser.add_argument("--parts", nargs="+", type=str,
                        default=["face", "tail", "torso"],
                        help="Body parts to render (default: face tail torso)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", action="store_true",
                        help="Generate MP4 video from consecutive frames")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GS-LRM model once
    print("Loading GS-LRM model...")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
    model = GSLRMInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("Model loaded")

    # Determine frame list
    if args.frame_range:
        start, end = map(int, args.frame_range.split(":"))
        frame_list = list(range(start, end))
    else:
        frame_list = args.frame_idx

    all_frame_renders = []
    for i, fi in enumerate(frame_list):
        print(f"\n{'='*60}")
        print(f"Frame {fi} ({i+1}/{len(frame_list)})")
        print(f"{'='*60}")
        save_img = (i == 0)  # Only save images for first frame
        fr = process_frame(
            fi, model, args.kp_path, args.m5_dir, args.views, output_dir,
            args.device, selected_parts=args.parts, save_images=save_img,
        )
        if fr is not None:
            all_frame_renders.append(fr)

    # Generate video from consecutive frames if requested
    if args.video and len(all_frame_renders) > 1:
        print("\nGenerating body-part grid videos...")
        _generate_videos(all_frame_renders, args.views, output_dir, args.fps)

        print("\nGenerating novel view videos...")
        _generate_novel_view_videos(
            all_frame_renders, model, args.m5_dir, output_dir, args.device, args.fps,
        )

        print("\nGenerating temporal turntable...")
        _generate_temporal_turntable(
            model, frame_list, args.m5_dir, output_dir, args.device, args.fps,
        )

    print(f"\nAll renders saved to {output_dir}")


if __name__ == "__main__":
    main()
