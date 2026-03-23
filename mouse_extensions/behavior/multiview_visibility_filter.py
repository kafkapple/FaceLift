"""Multi-View Visibility Filtering for Gaussian body-part features.

Projects all Gaussians to each of 6 camera views, checks foreground mask
membership, and filters by N-threshold (visible in at least N views).

Produces side-by-side comparison grid: N=0..6 × white/black BG × 6 views.

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.multiview_visibility_filter \
        --frame-idx 0 222 1000 \
        --n-thresholds 0 1 2 3 4 5 6 \
        --output-dir outputs/analysis/mouse/filtering/multiview_filter
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from mouse_extensions.behavior.view_projected_filtering import (
    KP_NAMES, BODY_PARTS, BODY_PART_COLORS, load_camera, load_keypoints_gslrm,
    project_points_to_2d,
)
from mouse_extensions.behavior.render_bodypart_gaussians import (
    assign_gaussians_to_bodyparts_3d, render_gaussians_masked, render_all_gaussians,
)


# ---------------------------------------------------------------------------
# Core: Multi-view visibility counting
# ---------------------------------------------------------------------------

def load_foreground_mask(frame_dir: str, view_idx: int, alpha_threshold: int = 128) -> np.ndarray:
    """Load GT foreground mask from RGBA alpha channel.

    Args:
        frame_dir: Path to frame directory (contains images/)
        view_idx: Camera view index
        alpha_threshold: Pixels with alpha > threshold are foreground

    Returns:
        mask: (H, W) boolean array
    """
    from PIL import Image
    img_path = Path(frame_dir) / "images" / f"cam_{view_idx:03d}.png"
    img = np.array(Image.open(img_path))
    if img.shape[-1] == 4:
        return img[:, :, 3] > alpha_threshold
    else:
        # Fallback: non-black pixels are foreground
        gray = img.mean(axis=-1)
        return gray > 10


def compute_visibility_counts(
    xyz: np.ndarray,
    frame_dir: str,
    n_views: int = 6,
    alpha_threshold: int = 128,
) -> np.ndarray:
    """Count how many views each Gaussian center projects into the foreground mask.

    Args:
        xyz: (N, 3) Gaussian centers in world coordinates
        frame_dir: Path to frame directory
        n_views: Number of camera views
        alpha_threshold: Alpha threshold for foreground mask

    Returns:
        counts: (N,) int array, number of views where Gaussian is in foreground
    """
    N = len(xyz)
    counts = np.zeros(N, dtype=np.int32)

    for view_idx in range(n_views):
        # Load camera
        cam_path = Path(frame_dir) / "opencv_cameras.json"
        cam = load_camera(str(cam_path), view_idx)

        # Load foreground mask
        fg_mask = load_foreground_mask(frame_dir, view_idx, alpha_threshold)
        H, W = fg_mask.shape

        # Project Gaussian centers to 2D
        uv, valid = project_points_to_2d(
            xyz, np.array(cam["w2c"]),
            cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        )

        # Check which are inside image bounds AND in foreground
        u_int = np.round(uv[:, 0]).astype(np.int32)
        v_int = np.round(uv[:, 1]).astype(np.int32)

        in_bounds = (
            valid
            & (u_int >= 0) & (u_int < W)
            & (v_int >= 0) & (v_int < H)
        )

        # Check foreground membership
        in_fg = np.zeros(N, dtype=bool)
        idx = np.where(in_bounds)[0]
        in_fg[idx] = fg_mask[v_int[idx], u_int[idx]]

        counts += in_fg.astype(np.int32)

    return counts


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_filtered_gaussians(
    gaussians, mask: np.ndarray, cam: Dict, bg_color: Tuple[float, ...],
    device: str = "cuda",
) -> np.ndarray:
    """Render Gaussians with a boolean mask, on specified background color."""
    from gslrm.model.gaussians_renderer import render_opencv_cam

    mask_t = torch.from_numpy(mask).to(device)

    # Zero out opacity for non-selected Gaussians
    old_opacity_raw = gaussians._opacity.data.clone()
    gaussians._opacity.data[~mask_t] = -100.0

    w2c = torch.tensor(cam["w2c"], dtype=torch.float32, device=device)
    c2w = torch.inverse(w2c)
    fxfycxcy = torch.tensor(
        [cam["fx"], cam["fy"], cam["cx"], cam["cy"]],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        result = render_opencv_cam(
            gaussians, cam["h"], cam["w"], c2w, fxfycxcy,
            bg_color=bg_color,
        )

    rendered = result["render"].permute(1, 2, 0).cpu().numpy()

    # Restore
    gaussians._opacity.data = old_opacity_raw
    return np.clip(rendered, 0, 1)


# ---------------------------------------------------------------------------
# Main experiment: N-threshold sweep with visualization
# ---------------------------------------------------------------------------

def process_frame_sweep(
    frame_idx: int,
    model,
    kp_path: str,
    m5_dir: str,
    n_thresholds: List[int],
    output_dir: Path,
    device: str = "cuda",
    render_views: Optional[List[int]] = None,
):
    """Run GS-LRM inference, compute visibility, generate N-sweep comparison.

    For each N threshold:
      - Filter Gaussians with visibility >= N
      - Render on white and black backgrounds
      - Compute body-part distribution
      - Save comparison grid
    """
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data

    frame_dir = Path(m5_dir) / f"{frame_idx:06d}"
    if not frame_dir.exists():
        print(f"  Frame {frame_idx}: not found")
        return None

    if render_views is None:
        render_views = [0, 2, 4]  # 3 representative views

    print(f"\n{'='*60}")
    print(f"Frame {frame_idx}: Multi-View Visibility Filter Sweep")
    print(f"{'='*60}")

    # Step 1: GS-LRM inference
    print("  [1/4] GS-LRM inference...")
    images, c2ws, fxfycxcys, index = load_sample_data(
        str(frame_dir), image_size=512, device=device,
    )
    result = model.predict(images, c2ws, fxfycxcys, index)
    gaussians = result.gaussians[0]

    xyz_np = gaussians.get_xyz.detach().cpu().numpy()
    N_total = len(xyz_np)
    print(f"  Total Gaussians: {N_total:,}")

    # Step 2: Compute visibility counts
    print("  [2/4] Computing 6-view visibility...")
    vis_counts = compute_visibility_counts(xyz_np, str(frame_dir), n_views=6)

    # Report distribution
    print(f"  Visibility distribution:")
    for n in range(7):
        cnt = (vis_counts == n).sum()
        print(f"    N={n}: {cnt:,} ({100*cnt/N_total:.1f}%)")

    # Cumulative (>= N)
    print(f"  Cumulative (>= N):")
    for n in range(7):
        cnt = (vis_counts >= n).sum()
        print(f"    N>={n}: {cnt:,} ({100*cnt/N_total:.1f}%)")

    # Step 3: Load keypoints for body-part assignment
    kp_gslrm = load_keypoints_gslrm(kp_path, frame_idx)

    # Step 4: Render N-sweep comparison
    print("  [3/4] Rendering N-sweep comparison...")

    # Prepare camera data for render views
    cam_list = []
    gt_images = []
    fg_masks = []
    for vi in render_views:
        cam_path = frame_dir / "opencv_cameras.json"
        cam = load_camera(str(cam_path), vi)
        cam_list.append(cam)

        from PIL import Image
        img_path = frame_dir / "images" / f"cam_{vi:03d}.png"
        img = np.array(Image.open(img_path))
        gt_images.append(img[:, :, :3] / 255.0)
        fg_masks.append(img[:, :, 3] > 128)

    # Collect results per N threshold
    sweep_results = {}
    for n_thresh in n_thresholds:
        if n_thresh == 0:
            mask = np.ones(N_total, dtype=bool)
        else:
            mask = vis_counts >= n_thresh

        n_kept = mask.sum()

        # Body-part distribution
        if n_kept > 0:
            part_masks = assign_gaussians_to_bodyparts_3d(xyz_np[mask], kp_gslrm)
            bp_counts = {pn: int(m.sum()) for pn, m in part_masks.items()}
        else:
            bp_counts = {pn: 0 for pn in BODY_PARTS}

        # Render on white and black BG
        renders_white = []
        renders_black = []
        for cam in cam_list:
            rw = render_filtered_gaussians(gaussians, mask, cam, (1.0, 1.0, 1.0), device)
            rb = render_filtered_gaussians(gaussians, mask, cam, (0.0, 0.0, 0.0), device)
            renders_white.append(rw)
            renders_black.append(rb)

        sweep_results[n_thresh] = {
            "n_kept": int(n_kept),
            "pct": 100.0 * n_kept / N_total,
            "bp_counts": bp_counts,
            "renders_white": renders_white,
            "renders_black": renders_black,
            "mask": mask,
        }

        bp_str = ", ".join(f"{k}={v:,}" for k, v in bp_counts.items())
        print(f"    N>={n_thresh}: {n_kept:,} Gaussians ({100*n_kept/N_total:.1f}%) | {bp_str}")

    # Step 5: Generate comparison grids
    print("  [4/4] Generating comparison grids...")

    # Save quantitative results as JSON
    quant = {
        "frame_idx": frame_idx,
        "n_total": N_total,
        "visibility_distribution": {str(n): int((vis_counts == n).sum()) for n in range(7)},
        "sweep": {},
    }
    for n_thresh, res in sweep_results.items():
        quant["sweep"][str(n_thresh)] = {
            "n_kept": res["n_kept"],
            "pct": res["pct"],
            "bp_counts": res["bp_counts"],
        }

    json_path = output_dir / f"frame_{frame_idx:06d}_stats.json"
    with open(json_path, "w") as f:
        json.dump(quant, f, indent=2)
    print(f"  Stats: {json_path}")

    # Generate grid images for both backgrounds
    for bg_name, bg_key in [("white", "renders_white"), ("black", "renders_black")]:
        n_rows = len(n_thresholds) + 1  # +1 for GT row
        n_cols = len(render_views)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_cols == 1:
            axes = axes[:, None]

        # Row 0: GT images with foreground mask overlay
        for ci, (gt, fg) in enumerate(zip(gt_images, fg_masks)):
            ax = axes[0, ci]
            ax.imshow(gt)
            # Overlay mask boundary
            from scipy.ndimage import binary_dilation
            boundary = binary_dilation(fg, iterations=1) & ~fg
            overlay = gt.copy()
            overlay[boundary] = [1, 0, 0]  # Red boundary
            ax.imshow(overlay)
            ax.set_title(f"View {render_views[ci]} (GT + mask)", fontsize=11, fontweight="bold")
            ax.axis("off")

        # Label row 0
        axes[0, 0].annotate(
            "GT + FG Mask", xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-10, 0), textcoords="offset points",
            fontsize=12, fontweight="bold", ha="right", va="center", rotation=90,
        )

        # Rows 1+: N-threshold renders
        for ri, n_thresh in enumerate(n_thresholds):
            res = sweep_results[n_thresh]
            renders = res[bg_key]
            for ci, render_img in enumerate(renders):
                ax = axes[ri + 1, ci]
                ax.imshow(render_img)
                ax.set_title(f"{res['n_kept']:,} Gaussians", fontsize=10)
                ax.axis("off")

            label = f"N>={n_thresh}" if n_thresh > 0 else "All (N>=0)"
            axes[ri + 1, 0].annotate(
                f"{label}\n{res['pct']:.1f}%",
                xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-10, 0), textcoords="offset points",
                fontsize=11, fontweight="bold", ha="right", va="center", rotation=90,
            )

        bg_label = "White BG" if bg_name == "white" else "Black BG"
        fig.suptitle(
            f"Multi-View Visibility Filter — Frame {frame_idx} ({bg_label})\n"
            f"N-threshold sweep: keep Gaussians visible in >= N of 6 views",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout(rect=[0.08, 0, 1, 0.95])
        out_path = output_dir / f"frame_{frame_idx:06d}_sweep_{bg_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Grid: {out_path}")

    # Generate body-part balance chart
    _plot_bodypart_balance(sweep_results, n_thresholds, frame_idx, output_dir)

    # Cleanup GPU
    del result, gaussians, images, c2ws, fxfycxcys
    torch.cuda.empty_cache()

    return sweep_results


def _plot_bodypart_balance(
    sweep_results: Dict, n_thresholds: List[int], frame_idx: int, output_dir: Path
):
    """Plot body-part distribution balance across N thresholds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    part_names = list(BODY_PARTS.keys())
    colors = [BODY_PART_COLORS[pn] for pn in part_names]

    # Left: stacked bar chart of absolute counts
    x = np.arange(len(n_thresholds))
    bottoms = np.zeros(len(n_thresholds))
    for pi, pn in enumerate(part_names):
        vals = [sweep_results[n]["bp_counts"].get(pn, 0) for n in n_thresholds]
        ax1.bar(x, vals, bottom=bottoms, label=pn, color=colors[pi], alpha=0.8)
        bottoms += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"N>={n}" if n > 0 else "All" for n in n_thresholds])
    ax1.set_ylabel("Gaussian Count")
    ax1.set_title("Body-Part Distribution (Absolute)")
    ax1.legend(loc="upper right")

    # Right: max/min ratio (balance metric)
    ratios = []
    for n in n_thresholds:
        counts = [sweep_results[n]["bp_counts"].get(pn, 0) for pn in part_names]
        if min(counts) > 0:
            ratios.append(max(counts) / min(counts))
        else:
            ratios.append(float("inf"))

    ax2.bar(x, [r if r != float("inf") else 0 for r in ratios], color="steelblue", alpha=0.8)
    for i, r in enumerate(ratios):
        label = f"{r:.1f}" if r != float("inf") else "inf"
        ax2.text(i, min(r, 100) + 1, label, ha="center", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"N>={n}" if n > 0 else "All" for n in n_thresholds])
    ax2.set_ylabel("Max/Min Ratio")
    ax2.set_title("Body-Part Balance (lower = better)")
    ax2.set_ylim(0, min(max(r for r in ratios if r != float("inf")), 100) * 1.2)

    fig.suptitle(f"Frame {frame_idx} — Body-Part Balance vs. Visibility Threshold", fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / f"frame_{frame_idx:06d}_balance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Balance: {out_path}")


# ---------------------------------------------------------------------------
# Video generation: temporal body-part grid with N>=K filtering
# ---------------------------------------------------------------------------

def process_frame_for_video(
    frame_idx: int,
    model,
    kp_path: str,
    m5_dir: str,
    n_thresh: int,
    views: List[int],
    parts: List[str],
    device: str = "cuda",
) -> Optional[Dict]:
    """Run inference + N-threshold filter + body-part renders for one frame.

    Returns dict with per-view renders for each body part + "all_filtered".
    """
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data

    frame_dir = Path(m5_dir) / f"{frame_idx:06d}"
    if not frame_dir.exists():
        return None

    # GS-LRM inference
    images, c2ws, fxfycxcys, index = load_sample_data(
        str(frame_dir), image_size=512, device=device,
    )
    result = model.predict(images, c2ws, fxfycxcys, index)
    gaussians = result.gaussians[0]

    xyz_np = gaussians.get_xyz.detach().cpu().numpy()
    N_total = len(xyz_np)

    # Visibility filter
    vis_counts = compute_visibility_counts(xyz_np, str(frame_dir), n_views=6)
    vis_mask = vis_counts >= n_thresh if n_thresh > 0 else np.ones(N_total, dtype=bool)
    n_kept = vis_mask.sum()

    # Body-part assignment on filtered set
    kp_gslrm = load_keypoints_gslrm(kp_path, frame_idx)
    # Assign on full set, then intersect with vis_mask
    full_part_masks = assign_gaussians_to_bodyparts_3d(xyz_np, kp_gslrm)

    print(f"  f{frame_idx}: {n_kept:,}/{N_total:,} Gaussians (N>={n_thresh})")

    # Render per-view: all_filtered + each body part
    view_data = []
    for vi in views:
        cam_path = frame_dir / "opencv_cameras.json"
        cam = load_camera(str(cam_path), vi)

        # GT image
        from PIL import Image
        img_path = frame_dir / "images" / f"cam_{vi:03d}.png"
        gt = np.array(Image.open(img_path))[:, :, :3] / 255.0

        renders = {}
        # All filtered (white BG)
        renders["all_white"] = render_filtered_gaussians(
            gaussians, vis_mask, cam, (1.0, 1.0, 1.0), device)
        # All filtered (black BG)
        renders["all_black"] = render_filtered_gaussians(
            gaussians, vis_mask, cam, (0.0, 0.0, 0.0), device)

        # Per body-part (intersect vis_mask + part_mask)
        for pn in parts:
            bp_mask = full_part_masks[pn] & vis_mask
            renders[f"{pn}_white"] = render_filtered_gaussians(
                gaussians, bp_mask, cam, (1.0, 1.0, 1.0), device)
            renders[f"{pn}_black"] = render_filtered_gaussians(
                gaussians, bp_mask, cam, (0.0, 0.0, 0.0), device)

        view_data.append({"view_idx": vi, "gt": gt, "renders": renders})

    # Cleanup
    del result, gaussians, images, c2ws, fxfycxcys
    torch.cuda.empty_cache()

    return {"frame_idx": frame_idx, "view_data": view_data, "n_kept": n_kept}


def _make_grid(images: List[np.ndarray], n_views: int) -> np.ndarray:
    """Arrange view images into a grid (2×3 for 6 views, 1-row for <=3)."""
    if n_views <= 3:
        return np.concatenate(images, axis=1)
    n_top = (n_views + 1) // 2
    top = np.concatenate(images[:n_top], axis=1)
    bot = images[n_top:]
    if len(bot) < n_top:
        h, w = images[0].shape[:2]
        bot.append(np.ones((h, w, 3), dtype=np.uint8) * 255)
    bot = np.concatenate(bot[:n_top], axis=1)
    return np.concatenate([top, bot], axis=0)


def generate_videos(
    all_frames: List[Dict],
    views: List[int],
    parts: List[str],
    output_dir: Path,
    fps: int = 10,
):
    """Generate grid MP4 videos from temporal frame renders."""
    try:
        import cv2
    except ImportError:
        print("  cv2 not available, skipping video generation")
        return

    n_views = len(views)

    # Video configs: (render_key, filename)
    video_configs = [
        ("all_white", "video_all_filtered_white_grid"),
        ("all_black", "video_all_filtered_black_grid"),
    ]
    for pn in parts:
        video_configs.append((f"{pn}_white", f"video_{pn}_white_grid"))
        video_configs.append((f"{pn}_black", f"video_{pn}_black_grid"))

    # Also GT video
    video_configs.append(("gt", "video_gt_grid"))

    for render_key, filename in video_configs:
        grid_frames = []
        for fr in all_frames:
            row_imgs = []
            for vd in fr["view_data"]:
                if render_key == "gt":
                    img = vd["gt"]
                else:
                    img = vd["renders"].get(render_key)
                    if img is None:
                        continue
                img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                row_imgs.append(img_u8)

            if len(row_imgs) < n_views:
                continue
            grid = _make_grid(row_imgs, n_views)
            grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            grid_frames.append(grid_bgr)

        if not grid_frames:
            continue

        h, w = grid_frames[0].shape[:2]
        video_path = output_dir / f"{filename}.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
        )
        for f in grid_frames:
            writer.write(f)
        writer.release()
        print(f"  Video: {video_path} ({len(grid_frames)} frames)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-View Visibility Filter Sweep")
    parser.add_argument("--mode", choices=["sweep", "video"], default="sweep",
                        help="sweep: N-threshold comparison images; video: temporal MP4s")
    parser.add_argument("--frame-idx", type=int, nargs="+", default=[0, 222, 1000])
    parser.add_argument("--frame-range", type=str, default=None,
                        help="Consecutive frame range 'start:end' (overrides --frame-idx)")
    parser.add_argument("--n-thresholds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--n-filter", type=int, default=2,
                        help="N threshold for video mode (default: 2)")
    parser.add_argument("--render-views", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                        help="Camera views for video mode (default: all 6)")
    parser.add_argument("--parts", type=str, nargs="+", default=["face", "tail", "torso"],
                        help="Body parts for video mode")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5t")
    from mouse_extensions.paths import KP_22
    parser.add_argument("--kp-path", default=str(KP_22))
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt")
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/filtering/multiview_filter")
    parser.add_argument("--alpha-threshold", type=int, default=128)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GS-LRM model
    print("Loading GS-LRM model...")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GSLRMInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    print(f"  Model loaded on {device}")

    if args.mode == "sweep":
        # Original sweep mode
        all_results = {}
        for frame_idx in args.frame_idx:
            results = process_frame_sweep(
                frame_idx=frame_idx,
                model=model,
                kp_path=args.kp_path,
                m5_dir=args.m5_dir,
                n_thresholds=args.n_thresholds,
                output_dir=output_dir,
                device=device,
                render_views=args.render_views,
            )
            if results:
                all_results[frame_idx] = results

        if all_results:
            _print_cross_frame_summary(all_results, args.n_thresholds)

    elif args.mode == "video":
        # Video mode: temporal body-part grid with N>=K filter
        if args.frame_range:
            start, end = map(int, args.frame_range.split(":"))
            frame_list = list(range(start, end))
        else:
            frame_list = args.frame_idx

        print(f"\nVideo mode: {len(frame_list)} frames, N>={args.n_filter}, "
              f"{len(args.views)} views, parts={args.parts}")

        all_frames = []
        for i, fi in enumerate(frame_list):
            fr = process_frame_for_video(
                frame_idx=fi,
                model=model,
                kp_path=args.kp_path,
                m5_dir=args.m5_dir,
                n_thresh=args.n_filter,
                views=args.views,
                parts=args.parts,
                device=device,
            )
            if fr:
                all_frames.append(fr)

        if len(all_frames) > 1:
            print(f"\nGenerating {len(all_frames)}-frame videos...")
            generate_videos(all_frames, args.views, args.parts, output_dir, args.fps)

    print(f"\nAll outputs saved to: {output_dir}")


def _print_cross_frame_summary(all_results: Dict, n_thresholds: List[int]):
    """Print summary table across all frames."""
    print(f"\n{'='*70}")
    print("Cross-Frame Summary")
    print(f"{'='*70}")
    print(f"{'N':>5} | ", end="")
    for fid in sorted(all_results.keys()):
        print(f"f{fid:<8}", end="")
    print("| Mean %")
    print("-" * 70)

    for n in n_thresholds:
        label = f"N>={n}" if n > 0 else "All"
        print(f"{label:>5} | ", end="")
        pcts = []
        for fid in sorted(all_results.keys()):
            res = all_results[fid][n]
            print(f"{res['n_kept']:>7,}", end=" ")
            pcts.append(res["pct"])
        print(f"| {np.mean(pcts):.1f}%")


if __name__ == "__main__":
    main()
