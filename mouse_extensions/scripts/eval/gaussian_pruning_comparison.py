"""Gaussian pruning strategy comparison for GS-LRM mouse reconstruction.

Applies 5 pruning strategies to a sample frame and renders from 4 representative
viewpoints for visual comparison. Outputs a comparison grid image + per-strategy stats.

Strategies compared:
  1. no_filter   - apply_all_filters only (no visibility/mask filter)
  2. vis_2       - visibility n>=2 (current default)
  3. opacity_15  - vis_2 + opacity threshold >= 0.15
  4. anisotropy  - vis_2 + remove large flat Gaussians (anisotropy >= 30, max_scale >= 0.05)
  5. combined    - vis_2 + opacity_15 + anisotropy

Background: GS-LRM produces ~100K Gaussians in white-background scenes. Background
Gaussians appear white and are hard to distinguish by color from mouse fur/teeth.
We use geometric + visibility cues instead of color proximity.

Audit findings applied:
  - Color-based pruning AVOIDED (mouse fur is white)
  - Multi-view visibility used instead of single-camera mask
  - SH coefficients NOT used for pruning (degree ambiguity)
  - AND-logic mask: keep if visible in ANY view (n_filter=1 is more conservative)

Views: bottom(-80 deg elev), top(+60 deg elev), side_left(90 deg azim), side_right(270 deg azim)

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.gaussian_pruning_comparison \\
        --config mouse_extensions/behavior/cinematic_default.yaml \\
        --frame-idx 222 \\
        --output-dir outputs/viz/comparison/mouse/pruning_strategies

    # Multiple frames (random sample):
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.gaussian_pruning_comparison \\
        --config mouse_extensions/behavior/cinematic_default.yaml \\
        --frame-range 195:315 --n-sample 3 \\
        --output-dir outputs/viz/comparison/mouse/pruning_strategies
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Representative views for comparison
# ---------------------------------------------------------------------------

VIEWS = [
    {"name": "bottom",      "elevation": -80, "azimuth":   0},
    {"name": "top",         "elevation":  60, "azimuth":   0},
    {"name": "side_left",   "elevation":   5, "azimuth":  90},
    {"name": "side_right",  "elevation":   5, "azimuth": 270},
]

STRATEGY_LABELS = {
    "no_filter":   "No Mask Filter\n(apply_all only)",
    "vis_2":       "Vis n≥2\n(current default)",
    "opacity_15":  "Vis n≥2 +\nOpacity ≥0.15",
    "anisotropy":  "Vis n≥2 +\nAnisotropy",
    "combined":    "Combined\n(vis+opacity+aniso)",
}

STRATEGY_ORDER = ["no_filter", "vis_2", "opacity_15", "anisotropy", "combined"]


# ---------------------------------------------------------------------------
# Pruning strategy implementations
# ---------------------------------------------------------------------------

def compute_pruning_masks(
    gaussians,
    frame_dir: str,
    n_views: int = 6,
    opacity_threshold: float = 0.15,
    anisotropy_threshold: float = 30.0,
    max_scale_threshold: float = 0.05,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Compute boolean keep-masks for each pruning strategy.

    Returns:
        dict: strategy_name -> (N,) bool array (True = keep)
    """
    from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts

    N = gaussians.get_xyz.shape[0]
    xyz = gaussians.get_xyz.detach().cpu().numpy()

    # --- Visibility counts (multi-view mask projection) ---
    vc = compute_visibility_counts(xyz, frame_dir, n_views=n_views)

    # --- Opacity ---
    opacity = torch.sigmoid(gaussians._opacity).squeeze(-1).detach().cpu().numpy()  # (N,)

    # --- Anisotropy (geometric flatness) ---
    # Background Gaussians are often large and flat (pancake-shaped)
    scales = torch.exp(gaussians.get_scaling).detach().cpu().numpy()  # (N, 3)
    max_scale = scales.max(axis=1)
    min_scale = scales.min(axis=1).clip(min=1e-6)
    anisotropy = max_scale / min_scale  # (N,)

    # --- Strategy masks ---
    # Note: no_filter just uses apply_all_filters result (all True here, applied earlier)
    mask_no_filter = np.ones(N, dtype=bool)

    mask_vis2 = vc >= 2

    mask_opacity = mask_vis2 & (opacity >= opacity_threshold)

    # Anisotropy: prune Gaussians that are large AND flat, seen in <2 views
    # Logic: remove if (anisotropy > thr AND max_scale > thr AND vis < 2)
    # Equivalently: keep if vis>=2 AND NOT(large AND flat)
    is_large_flat = (anisotropy >= anisotropy_threshold) & (max_scale >= max_scale_threshold)
    mask_anisotropy = mask_vis2 & ~is_large_flat

    mask_combined = mask_vis2 & (opacity >= opacity_threshold) & ~is_large_flat

    return {
        "no_filter":  mask_no_filter,
        "vis_2":      mask_vis2,
        "opacity_15": mask_opacity,
        "anisotropy": mask_anisotropy,
        "combined":   mask_combined,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def get_camera_for_view(view: dict, radius: float = 2.7, hfov: float = 50,
                         resolution: int = 512) -> Tuple:
    """Get (c2w [4,4], fxfycxcy [4]) for a single turntable view.

    Computes c2w directly from elevation and azimuth (in degrees) without
    relying on get_turntable_cameras, which does not support azimuth_start.
    Uses the same math as camera_utils.get_turntable_cameras internally.
    """
    import math

    elev_rad = math.radians(view["elevation"])
    azim_rad = math.radians(view["azimuth"])
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


def render_gaussians_masked(
    gaussians,
    bool_mask: np.ndarray,
    c2w: np.ndarray,
    fxfycxcy: np.ndarray,
    resolution: int = 512,
    bg_color: Tuple = (1.0, 1.0, 1.0),
    device: str = "cuda",
) -> np.ndarray:
    """Render Gaussians with a boolean keep-mask, returning (H, W, 3) uint8.

    Uses render_opencv_cam (the public API) rather than render_view.
    """
    from mouse_extensions.visualization import render_opencv_cam

    mask_t = torch.from_numpy(bool_mask).to(device)
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    fxfy_t = torch.tensor(fxfycxcy, dtype=torch.float32, device=device)

    # Temporarily zero out opacity for masked-out Gaussians
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
        gaussians._opacity.data = old_opacity

    return (img_np * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grid composition
# ---------------------------------------------------------------------------

def add_text_overlay(
    img: np.ndarray,
    text: str,
    pos: str = "top",
    font_scale: float = 0.45,
    thickness: int = 1,
    color: Tuple[int, int, int] = (50, 50, 200),
) -> np.ndarray:
    """Add multi-line text overlay to image."""
    out = img.copy()
    H, W = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = text.split("\n")
    line_h = int(20 * font_scale / 0.45)

    if pos == "top":
        y0 = 18
    else:
        y0 = H - line_h * len(lines) - 5

    for i, line in enumerate(lines):
        y = y0 + i * line_h
        # Shadow
        cv2.putText(out, line, (9, y + 1), font, font_scale, (255, 255, 255), thickness + 1, cv2.LINE_AA)
        # Text
        cv2.putText(out, line, (8, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return out


def compose_comparison_grid(
    renders: Dict[str, List[np.ndarray]],
    strategy_order: List[str],
    view_names: List[str],
    strategy_labels: Dict[str, str],
    counts: Dict[str, int],
    cell_size: int = 256,
    border: int = 2,
) -> np.ndarray:
    """Compose N_strategies x N_views comparison grid.

    Args:
        renders: {strategy -> [img_view0, img_view1, ...]}
        strategy_order: list of strategy names (rows)
        view_names: list of view names (cols)
        strategy_labels: human-readable labels
        counts: {strategy -> n_gaussians_retained}
        cell_size: pixel size of each cell
        border: pixel border between cells

    Returns:
        (H, W, 3) uint8 grid image
    """
    n_rows = len(strategy_order)
    n_cols = len(view_names)

    # Header sizes
    row_header_w = 130  # for strategy label
    col_header_h = 28   # for view name

    total_w = row_header_w + n_cols * (cell_size + border) + border
    total_h = col_header_h + n_rows * (cell_size + border) + border

    grid = np.full((total_h, total_w, 3), 240, dtype=np.uint8)  # light gray bg

    # Column headers (view names)
    for c, vname in enumerate(view_names):
        x = row_header_w + c * (cell_size + border) + border + cell_size // 2 - 30
        cv2.putText(grid, vname, (x, col_header_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)

    for r, strat in enumerate(strategy_order):
        label = strategy_labels.get(strat, strat)
        count = counts.get(strat, -1)
        y_top = col_header_h + r * (cell_size + border) + border

        # Row header
        lines = label.split("\n") + [f"N={count:,}"]
        for li, ln in enumerate(lines):
            yy = y_top + 18 + li * 16
            cv2.putText(grid, ln, (3, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)

        for c, vname in enumerate(view_names):
            x_left = row_header_w + c * (cell_size + border) + border

            img = renders.get(strat, [None] * len(view_names))[c]
            if img is not None:
                # Resize to cell_size
                img_r = cv2.resize(img, (cell_size, cell_size))
                grid[y_top:y_top + cell_size, x_left:x_left + cell_size] = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)

    # Title border line
    cv2.line(grid, (0, col_header_h), (total_w, col_header_h), (180, 180, 180), 1)
    cv2.line(grid, (row_header_w, 0), (row_header_w, total_h), (180, 180, 180), 1)

    return grid


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_frame(
    frame_idx: int,
    m5_dir: Path,
    model,
    output_dir: Path,
    resolution: int = 512,
    n_filter_base: int = 2,
    opacity_threshold: float = 0.15,
    anisotropy_threshold: float = 30.0,
    max_scale_threshold: float = 0.05,
    radius: float = 2.7,
    hfov: float = 50.0,
    device: str = "cuda",
) -> Optional[str]:
    """Process a single frame: infer, prune, render, compose grid.

    Returns:
        Path to output grid image, or None on failure.
    """
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data

    fd = m5_dir / f"{frame_idx:06d}"
    if not fd.exists():
        print(f"  [skip] {frame_idx:06d} — directory not found")
        return None

    print(f"  Processing frame {frame_idx:06d}...")

    # --- Inference ---
    try:
        imgs, c2ws_in, fxfys_in, idx = load_sample_data(
            str(fd), image_size=resolution, device=device
        )
        result = model.predict(imgs, c2ws_in, fxfys_in, idx)
        gaussians_raw = result.gaussians[0]
    except Exception as e:
        print(f"  [ERROR] inference failed: {e}")
        return None

    # Apply baseline filters (opacity/scale/floater) — all strategies share this
    gaussians = gaussians_raw.apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
    )
    total_after_filter = gaussians.get_xyz.shape[0]
    print(f"    Gaussians after apply_all_filters: {total_after_filter:,}")

    # --- Compute pruning masks ---
    try:
        masks = compute_pruning_masks(
            gaussians,
            str(fd),
            n_views=6,
            opacity_threshold=opacity_threshold,
            anisotropy_threshold=anisotropy_threshold,
            max_scale_threshold=max_scale_threshold,
            device=device,
        )
    except Exception as e:
        print(f"  [ERROR] pruning mask computation failed: {e}")
        return None

    counts = {s: int(m.sum()) for s, m in masks.items()}
    for s, c in counts.items():
        pct = 100 * c / max(total_after_filter, 1)
        print(f"    {s:12s}: {c:6,} ({pct:.1f}%)")

    # --- Pre-compute cameras ---
    cameras = {}
    for view in VIEWS:
        cam = get_camera_for_view(view, radius=radius, hfov=hfov, resolution=resolution)
        cameras[view["name"]] = cam

    # --- Render each strategy × view ---
    renders: Dict[str, List[np.ndarray]] = {}
    for strat in STRATEGY_ORDER:
        bool_mask = masks[strat]
        renders[strat] = []
        for view in VIEWS:
            vname = view["name"]
            cam = cameras[vname]
            try:
                img = render_gaussians_masked(
                    gaussians, bool_mask,
                    c2w=cam[0], fxfycxcy=cam[1],
                    resolution=resolution,
                    device=device,
                )
            except Exception as e:
                print(f"    [warn] render {strat}/{vname}: {e}")
                img = np.full((resolution, resolution, 3), 200, dtype=np.uint8)
            renders[strat].append(img)
        print(f"    Rendered {strat}")

    # --- Compose grid ---
    view_names = [v["name"] for v in VIEWS]
    grid = compose_comparison_grid(
        renders=renders,
        strategy_order=STRATEGY_ORDER,
        view_names=view_names,
        strategy_labels=STRATEGY_LABELS,
        counts=counts,
        cell_size=min(resolution // 2, 256),
    )

    # Save
    out_path = output_dir / f"pruning_comparison_{frame_idx:06d}.png"
    cv2.imwrite(str(out_path), grid)
    print(f"    Saved: {out_path}")

    # Save stats JSON
    stats = {
        "frame_idx": frame_idx,
        "total_after_apply_all_filters": total_after_filter,
        "counts": counts,
        "retention_pct": {s: round(100 * c / max(total_after_filter, 1), 1) for s, c in counts.items()},
        "thresholds": {
            "opacity": opacity_threshold,
            "anisotropy": anisotropy_threshold,
            "max_scale": max_scale_threshold,
        },
    }
    stats_path = output_dir / f"pruning_stats_{frame_idx:06d}.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return str(out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pruning_comparison(
    config_path: str,
    output_dir: str,
    frame_indices: Optional[List[int]] = None,
    frame_range: Optional[str] = None,
    n_sample: Optional[int] = None,
    opacity_threshold: float = 0.15,
    anisotropy_threshold: float = 30.0,
    max_scale_threshold: float = 0.05,
    device: str = "cuda",
) -> List[str]:
    """Run Gaussian pruning comparison for one or more frames.

    Args:
        config_path: cinematic_default.yaml
        output_dir: where to save comparison grids
        frame_indices: explicit list of frame indices
        frame_range: "start:end" (alternative to frame_indices)
        n_sample: randomly sample N frames from range (default: all)
        opacity_threshold: threshold for opacity_15 strategy
        anisotropy_threshold: anisotropy ratio threshold
        max_scale_threshold: minimum max_scale for large-flat classification
        device: CUDA device string

    Returns:
        list of output image paths
    """
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m5_dir = Path(cfg["model"]["m5_dir"])
    resolution = cfg["global"].get("resolution", 512)
    radius = cfg.get("camera", {}).get("radius", 2.7)
    hfov = cfg.get("camera", {}).get("hfov", 50.0)

    # Determine frame indices
    if frame_indices is None:
        rng = frame_range or cfg.get("frame_range", "195:315")
        start, end = map(int, rng.split(":"))
        frame_indices = list(range(start, end))

    if n_sample is not None and n_sample < len(frame_indices):
        rng_state = np.random.RandomState(42)
        frame_indices = sorted(rng_state.choice(frame_indices, n_sample, replace=False).tolist())

    print(f"Frames: {frame_indices}")
    print(f"Output: {output_dir}")
    print(f"Strategies: {STRATEGY_ORDER}")

    # Load model
    print("\nLoading GS-LRM model...")
    model = GSLRMInference(
        config_path=cfg["model"]["config"],
        checkpoint_path=cfg["model"]["checkpoint"],
        device=device,
    )
    if "num_input_views" in cfg.get("model", {}):
        model.config.model.num_input_views = cfg["model"]["num_input_views"]
        print(f"  num_input_views patched -> {cfg['model']['num_input_views']}")

    outputs = []
    for fi in frame_indices:
        result = process_frame(
            frame_idx=fi,
            m5_dir=m5_dir,
            model=model,
            output_dir=output_dir,
            resolution=resolution,
            opacity_threshold=opacity_threshold,
            anisotropy_threshold=anisotropy_threshold,
            max_scale_threshold=max_scale_threshold,
            radius=radius,
            hfov=hfov,
            device=device,
        )
        if result:
            outputs.append(result)

    print(f"\nDone: {len(outputs)}/{len(frame_indices)} frames")
    print(f"Output dir: {output_dir}")
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian pruning strategy comparison for GS-LRM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single frame
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.gaussian_pruning_comparison \\
        --config mouse_extensions/behavior/cinematic_default.yaml \\
        --frame-idx 222 \\
        --output-dir outputs/viz/comparison/mouse/pruning_strategies

    # Random 3 frames from range
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.gaussian_pruning_comparison \\
        --config mouse_extensions/behavior/cinematic_default.yaml \\
        --frame-range 195:315 --n-sample 3 \\
        --output-dir outputs/viz/comparison/mouse/pruning_strategies

    # Tune thresholds
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.eval.gaussian_pruning_comparison \\
        --opacity-threshold 0.20 --anisotropy-threshold 20.0
""",
    )
    parser.add_argument(
        "--config", default="mouse_extensions/behavior/cinematic_default.yaml"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/viz/comparison/mouse/pruning_strategies",
    )
    parser.add_argument("--frame-idx", type=int, nargs="+",
                        help="Explicit frame index(es)")
    parser.add_argument("--frame-range", default=None,
                        help="Frame range 'start:end'")
    parser.add_argument("--n-sample", type=int, default=None,
                        help="Randomly sample N frames from range")
    parser.add_argument("--opacity-threshold", type=float, default=0.15,
                        help="Opacity threshold for opacity_15 strategy (default: 0.15)")
    parser.add_argument("--anisotropy-threshold", type=float, default=30.0,
                        help="Anisotropy ratio threshold (default: 30.0)")
    parser.add_argument("--max-scale-threshold", type=float, default=0.05,
                        help="Min max_scale to classify as large-flat (default: 0.05)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_pruning_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        frame_indices=args.frame_idx,
        frame_range=args.frame_range,
        n_sample=args.n_sample,
        opacity_threshold=args.opacity_threshold,
        anisotropy_threshold=args.anisotropy_threshold,
        max_scale_threshold=args.max_scale_threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
