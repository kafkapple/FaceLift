"""View-Projected Gaussian Filtering: Body-part selective dense features.

Two modes:
  bbox   — Bounding box filtering (original)
  radial — Circle-based filtering with radius sweep grid visualization

Projects all Gaussians to a GT camera view, filters by 2D membership.

Body parts defined:
  - face: nose + L_ear + R_ear + neck
  - left_paw: L_paw + L_paw_end + L_elbow
  - right_paw: R_paw + R_paw_end + R_elbow
  - tail: tail_root + tail_middle + tail_end
  - torso: body_middle + L_shoulder + R_shoulder + L_hip + R_hip

Usage:
    cd /home/joon/dev/FaceLift

    # Original bbox mode
    python -m mouse_extensions.behavior.view_projected_filtering \
        --frame-idx 0 100 500 1000 2000

    # Radial mode with radius sweep
    python -m mouse_extensions.behavior.view_projected_filtering \
        --mode radial --radii 10 20 30 50 70 100 \
        --frame-idx 0 500 1000 \
        --output-dir outputs/analysis/mouse/filtering/radial_filtering
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# MAMMAL 22 keypoint names (same order as keypoints_22_3d.npz)
KP_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder", "L_foot", "L_knee",
    "L_hip", "R_foot", "R_knee", "R_hip",
]

# Body part definitions: name -> list of keypoint indices
BODY_PARTS = {
    "face": [0, 1, 2, 3],          # L_ear, R_ear, nose, neck
    "left_paw": [8, 9, 10],        # L_paw, L_paw_end, L_elbow
    "right_paw": [12, 13, 14],     # R_paw, R_paw_end, R_elbow
    "tail": [5, 6, 7],             # tail_root, tail_middle, tail_end
    "torso": [4, 11, 15, 18, 21],  # body_middle, L/R_shoulder, L/R_hip
}

BODY_PART_COLORS = {
    "face": "#FFD700",       # gold
    "left_paw": "#4169E1",   # royal blue
    "right_paw": "#32CD32",  # lime green
    "tail": "#FF6347",       # tomato
    "torso": "#DA70D6",      # orchid
}


def load_camera(cam_json_path: str, view_idx: int = 0) -> Dict:
    """Load camera parameters from opencv_cameras.json."""
    with open(cam_json_path) as f:
        cams = json.load(f)
    cam = cams["frames"][view_idx]
    return {
        "w2c": np.array(cam["w2c"], dtype=np.float64),
        "fx": float(cam["fx"]),
        "fy": float(cam["fy"]),
        "cx": float(cam["cx"]),
        "cy": float(cam["cy"]),
        "w": int(cam["w"]),
        "h": int(cam["h"]),
        "view_id": int(cam["view_id"]),
    }


def load_gaussians(npz_path: str) -> Dict[str, np.ndarray]:
    """Load Gaussian parameters from NPZ."""
    d = np.load(npz_path)
    xyz = d["xyz"].astype(np.float32)
    opacity_logit = d["opacity"].astype(np.float32).flatten()
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    scale = d["scale"].astype(np.float32)
    rotation = d["rotation"].astype(np.float32)
    return {
        "xyz": xyz,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
        "n_gaussians": len(xyz),
    }


def load_keypoints_gslrm(kp_path: str, frame_idx: int) -> np.ndarray:
    """Load keypoints and convert MAMMAL mm -> GS-LRM normalized space."""
    from mouse_extensions.coordinate_utils import mammal_to_gslrm
    kp_data = np.load(kp_path)
    kp_mm = kp_data["keypoints"][frame_idx]  # (22, 3) in MAMMAL mm
    kp_gslrm = mammal_to_gslrm(kp_mm)  # (22, 3) in GS-LRM normalized space
    return kp_gslrm


def project_points_to_2d(
    points_3d: np.ndarray, w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) in world coordinates
        w2c: (4, 4) world-to-camera matrix

    Returns:
        uv: (N, 2) pixel coordinates
        valid: (N,) boolean mask for points in front of camera
    """
    N = len(points_3d)
    pts_h = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
    pts_cam = (w2c @ pts_h.T).T[:, :3]  # (N, 3) in camera space

    z = pts_cam[:, 2]
    valid = z > 1e-6

    u = np.zeros(N)
    v = np.zeros(N)
    u[valid] = fx * pts_cam[valid, 0] / z[valid] + cx
    v[valid] = fy * pts_cam[valid, 1] / z[valid] + cy

    return np.stack([u, v], axis=1), valid


def compute_body_part_bboxes(
    kp_2d: np.ndarray, kp_valid: np.ndarray, padding_px: int = 20, img_size: int = 512
) -> Dict[str, Dict]:
    """Compute bounding boxes for each body part from projected keypoints.

    Returns dict of {part_name: {"bbox": [x_min, y_min, x_max, y_max], "kp_indices": [...]}}.
    """
    bboxes = {}
    for part_name, kp_indices in BODY_PARTS.items():
        part_kp = kp_2d[kp_indices]
        part_valid = kp_valid[kp_indices]

        if not part_valid.any():
            continue

        valid_kp = part_kp[part_valid]
        x_min = max(0, valid_kp[:, 0].min() - padding_px)
        y_min = max(0, valid_kp[:, 1].min() - padding_px)
        x_max = min(img_size, valid_kp[:, 0].max() + padding_px)
        y_max = min(img_size, valid_kp[:, 1].max() + padding_px)

        bboxes[part_name] = {
            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
            "kp_indices": kp_indices,
            "n_visible_kp": int(part_valid.sum()),
        }
    return bboxes


def filter_gaussians_by_bbox(
    gauss_2d: np.ndarray,
    gauss_valid: np.ndarray,
    bbox: List[float],
) -> np.ndarray:
    """Return boolean mask of Gaussians within a 2D bounding box."""
    x_min, y_min, x_max, y_max = bbox
    in_bbox = (
        gauss_valid
        & (gauss_2d[:, 0] >= x_min) & (gauss_2d[:, 0] <= x_max)
        & (gauss_2d[:, 1] >= y_min) & (gauss_2d[:, 1] <= y_max)
    )
    return in_bbox


def filter_gaussians_by_radius(
    gauss_2d: np.ndarray,
    gauss_valid: np.ndarray,
    kp_2d: np.ndarray,
    kp_valid: np.ndarray,
    radius_px: float,
    body_parts: Dict[str, List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Filter Gaussians within radius of projected keypoints per body part.

    Uses nearest-keypoint assignment for overlapping regions.

    Args:
        gauss_2d: (N, 2) projected Gaussian positions
        gauss_valid: (N,) valid mask
        kp_2d: (22, 2) projected keypoint positions
        kp_valid: (22,) valid keypoint mask
        radius_px: radius in pixels
        body_parts: dict of {part_name: [kp_indices]}

    Returns:
        dict of {part_name: boolean mask (N,)}
    """
    if body_parts is None:
        body_parts = BODY_PARTS

    N = len(gauss_2d)

    # Compute distance from each valid Gaussian to each valid keypoint
    # For soft assignment: assign to nearest keypoint's body part
    all_kp_indices = []
    all_kp_positions = []
    kp_to_part_map = {}
    for part_name, kp_indices in body_parts.items():
        for ki in kp_indices:
            if kp_valid[ki]:
                all_kp_indices.append(ki)
                all_kp_positions.append(kp_2d[ki])
                kp_to_part_map[ki] = part_name

    if not all_kp_positions:
        return {pn: np.zeros(N, dtype=bool) for pn in body_parts}

    kp_pos = np.array(all_kp_positions)  # (K_valid, 2)
    kp_idx = all_kp_indices

    # Distances: (N, K_valid)
    valid_gauss = gauss_valid.copy()
    dists = np.full((N, len(kp_pos)), np.inf)
    if valid_gauss.any():
        diff = gauss_2d[valid_gauss, None, :] - kp_pos[None, :, :]  # (N_valid, K, 2)
        dists[valid_gauss] = np.sqrt((diff ** 2).sum(axis=2))

    # Nearest keypoint assignment
    nearest_kp_idx_local = dists.argmin(axis=1)  # index into kp_idx
    nearest_dist = dists[np.arange(N), nearest_kp_idx_local]

    # Build masks: within radius AND nearest keypoint belongs to this part
    part_masks = {}
    for part_name in body_parts:
        mask = np.zeros(N, dtype=bool)
        for local_i, ki in enumerate(kp_idx):
            if kp_to_part_map[ki] == part_name:
                is_nearest = nearest_kp_idx_local == local_i
                within_radius = nearest_dist <= radius_px
                mask |= (is_nearest & within_radius & gauss_valid)
        part_masks[part_name] = mask

    return part_masks


def visualize_radius_sweep(
    frame_idx: int,
    gt_image: np.ndarray,
    kp_2d: np.ndarray,
    kp_valid: np.ndarray,
    gauss_2d: np.ndarray,
    gauss_valid: np.ndarray,
    gauss_opacity: np.ndarray,
    radii: List[float],
    output_dir: Path,
):
    """Generate grid visualization sweeping radius parameter.

    Creates a grid: rows = radii, cols = body parts + overview.
    """
    n_radii = len(radii)
    n_parts = len(BODY_PARTS)
    fig, axes = plt.subplots(n_radii, n_parts + 1, figsize=(4 * (n_parts + 1), 4 * n_radii))
    if n_radii == 1:
        axes = axes[None, :]

    coverage_stats = []

    for ri, radius in enumerate(radii):
        part_masks = filter_gaussians_by_radius(
            gauss_2d, gauss_valid, kp_2d, kp_valid, radius
        )

        total_assigned = sum(m.sum() for m in part_masks.values())
        total_visible = gauss_valid.sum()
        coverage = total_assigned / max(total_visible, 1)
        coverage_stats.append({
            "radius": float(radius),
            "total_assigned": int(total_assigned),
            "total_visible": int(total_visible),
            "coverage": float(coverage),
            "per_part": {pn: int(m.sum()) for pn, m in part_masks.items()},
        })

        # Column 0: Overview (all parts color-coded)
        ax = axes[ri, 0]
        ax.imshow(gt_image, alpha=0.3)
        for part_name, mask in part_masks.items():
            if mask.sum() > 0:
                ax.scatter(
                    gauss_2d[mask, 0], gauss_2d[mask, 1],
                    c=BODY_PART_COLORS[part_name], s=0.3, alpha=0.4,
                )
        # Draw radius circles around keypoints
        for part_name, kp_indices in BODY_PARTS.items():
            for ki in kp_indices:
                if kp_valid[ki]:
                    circle = plt.Circle(
                        (kp_2d[ki, 0], kp_2d[ki, 1]), radius,
                        fill=False, edgecolor=BODY_PART_COLORS[part_name],
                        linewidth=0.8, linestyle="--", alpha=0.6,
                    )
                    ax.add_patch(circle)
        ax.set_xlim(0, 512)
        ax.set_ylim(512, 0)
        ax.set_ylabel(f"r={radius:.0f}px", fontsize=11, fontweight="bold")
        ax.set_title(f"All parts ({coverage:.0%} coverage)" if ri == 0 else f"{coverage:.0%} coverage",
                     fontsize=9)
        ax.axis("off")

        # Columns 1-5: Individual body parts
        for pi, part_name in enumerate(BODY_PARTS.keys()):
            ax = axes[ri, pi + 1]
            mask = part_masks[part_name]
            ax.imshow(gt_image, alpha=0.3)
            if mask.sum() > 0:
                ax.scatter(
                    gauss_2d[mask, 0], gauss_2d[mask, 1],
                    c=BODY_PART_COLORS[part_name], s=0.5, alpha=0.5,
                )
            # Draw circles for this part's keypoints
            for ki in BODY_PARTS[part_name]:
                if kp_valid[ki]:
                    circle = plt.Circle(
                        (kp_2d[ki, 0], kp_2d[ki, 1]), radius,
                        fill=False, edgecolor=BODY_PART_COLORS[part_name],
                        linewidth=1.2, linestyle="-",
                    )
                    ax.add_patch(circle)
                    ax.plot(kp_2d[ki, 0], kp_2d[ki, 1], "*", markersize=6,
                            color="white", markeredgecolor="black", markeredgewidth=0.5)
            ax.set_xlim(0, 512)
            ax.set_ylim(512, 0)
            if ri == 0:
                ax.set_title(part_name, fontsize=10, fontweight="bold",
                             color=BODY_PART_COLORS[part_name])
            n_gauss = mask.sum()
            ax.text(5, 20, f"N={n_gauss:,}", fontsize=8, color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
            ax.axis("off")

    plt.suptitle(
        f"Radius Sweep — Frame {frame_idx}\n"
        f"Radii: {[f'{r:.0f}' for r in radii]} px | "
        f"Total visible: {int(gauss_valid.sum()):,} Gaussians",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / f"frame_{frame_idx:06d}_radius_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # Coverage curve
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total coverage vs radius
    rs = [s["radius"] for s in coverage_stats]
    covs = [s["coverage"] for s in coverage_stats]
    ax2[0].plot(rs, covs, "o-", color="steelblue", linewidth=2, markersize=6)
    ax2[0].set_xlabel("Radius (px)")
    ax2[0].set_ylabel("Coverage (fraction of visible Gaussians)")
    ax2[0].set_title("Total Coverage vs Radius")
    ax2[0].axhline(0.9, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    ax2[0].axhline(0.95, color="orange", linestyle="--", alpha=0.5, label="95% threshold")
    ax2[0].legend()
    ax2[0].grid(True, alpha=0.3)

    # Right: per-part count vs radius
    for part_name in BODY_PARTS:
        counts = [s["per_part"][part_name] for s in coverage_stats]
        ax2[1].plot(rs, counts, "o-", color=BODY_PART_COLORS[part_name],
                    linewidth=2, markersize=5, label=part_name)
    ax2[1].set_xlabel("Radius (px)")
    ax2[1].set_ylabel("Gaussians assigned")
    ax2[1].set_title("Per-Part Gaussians vs Radius")
    ax2[1].legend()
    ax2[1].grid(True, alpha=0.3)

    fig2.suptitle(f"Radius Sweep Statistics — Frame {frame_idx}", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    curve_path = output_dir / f"frame_{frame_idx:06d}_coverage_curve.png"
    fig2.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {curve_path}")

    return coverage_stats


def process_frame_radial(
    frame_idx: int,
    gauss_dir: str,
    kp_path: str,
    m5_dir: str,
    view_idx: int = 0,
    radii: List[float] = None,
    output_dir: Path = Path("outputs"),
) -> Dict:
    """Process a single frame with radial filtering + radius sweep."""
    if radii is None:
        radii = [10, 20, 30, 40, 50, 70, 100]

    # Load Gaussians
    gauss_path = Path(gauss_dir) / f"{frame_idx:06d}.npz"
    if not gauss_path.exists():
        print(f"  Frame {frame_idx}: NPZ not found at {gauss_path}")
        return {}

    gauss = load_gaussians(str(gauss_path))
    print(f"  Gaussians: {gauss['n_gaussians']:,}")

    # Load camera
    cam_path = Path(m5_dir) / f"{frame_idx:06d}" / "opencv_cameras.json"
    cam = load_camera(str(cam_path), view_idx)

    # Load GT image
    img_path = Path(m5_dir) / f"{frame_idx:06d}" / "images" / f"cam_{view_idx:03d}.png"
    if img_path.exists():
        from PIL import Image
        gt_image = np.array(Image.open(img_path))[:, :, :3]
    else:
        gt_image = np.ones((512, 512, 3), dtype=np.uint8) * 128

    # Load keypoints
    kp_gslrm = load_keypoints_gslrm(kp_path, frame_idx)

    # 3D spatial pre-filter
    kp_min = kp_gslrm.min(axis=0) - 0.3
    kp_max = kp_gslrm.max(axis=0) + 0.3
    xyz = gauss["xyz"]
    spatial_mask = (
        (xyz[:, 0] >= kp_min[0]) & (xyz[:, 0] <= kp_max[0]) &
        (xyz[:, 1] >= kp_min[1]) & (xyz[:, 1] <= kp_max[1]) &
        (xyz[:, 2] >= kp_min[2]) & (xyz[:, 2] <= kp_max[2])
    )

    # Project keypoints to 2D
    kp_2d, kp_valid = project_points_to_2d(
        kp_gslrm, cam["w2c"], cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    )
    kp_in_img = (
        kp_valid
        & (kp_2d[:, 0] >= 0) & (kp_2d[:, 0] < cam["w"])
        & (kp_2d[:, 1] >= 0) & (kp_2d[:, 1] < cam["h"])
    )

    # Project spatially-filtered Gaussians to 2D
    gauss_2d = np.zeros((gauss["n_gaussians"], 2), dtype=np.float32)
    gauss_valid = np.zeros(gauss["n_gaussians"], dtype=bool)
    if spatial_mask.sum() > 0:
        g2d, gv = project_points_to_2d(
            gauss["xyz"][spatial_mask], cam["w2c"], cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        )
        gauss_2d[spatial_mask] = g2d
        gauss_valid[spatial_mask] = gv

    gauss_in_img = (
        gauss_valid
        & (gauss_2d[:, 0] >= 0) & (gauss_2d[:, 0] < cam["w"])
        & (gauss_2d[:, 1] >= 0) & (gauss_2d[:, 1] < cam["h"])
    )
    print(f"  Visible Gaussians: {gauss_in_img.sum():,}, Keypoints: {kp_in_img.sum()}/22")

    # Radius sweep visualization
    coverage_stats = visualize_radius_sweep(
        frame_idx, gt_image, kp_2d, kp_in_img,
        gauss_2d, gauss_in_img, gauss["opacity"],
        radii, output_dir,
    )

    return {
        "frame_idx": frame_idx,
        "view_idx": view_idx,
        "n_gaussians_total": gauss["n_gaussians"],
        "n_gaussians_visible": int(gauss_in_img.sum()),
        "radii_sweep": coverage_stats,
    }


def filter_by_opacity_topk(opacity: np.ndarray, k: int) -> np.ndarray:
    """Return boolean mask for top-K Gaussians by opacity."""
    if k >= len(opacity):
        return np.ones(len(opacity), dtype=bool)
    threshold_idx = np.argpartition(opacity, -k)[-k:]
    mask = np.zeros(len(opacity), dtype=bool)
    mask[threshold_idx] = True
    return mask


def visualize_frame(
    frame_idx: int,
    gt_image: np.ndarray,
    kp_2d: np.ndarray,
    kp_valid: np.ndarray,
    gauss_2d: np.ndarray,
    gauss_valid: np.ndarray,
    gauss_opacity: np.ndarray,
    bboxes: Dict,
    part_masks: Dict[str, np.ndarray],
    output_dir: Path,
    top_k: int = 50000,
):
    """Create comprehensive visualization for one frame."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # === Panel 1: GT image + keypoints + bboxes ===
    ax = axes[0, 0]
    ax.imshow(gt_image)
    for part_name, bbox_info in bboxes.items():
        bb = bbox_info["bbox"]
        rect = mpatches.Rectangle(
            (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
            linewidth=2, edgecolor=BODY_PART_COLORS[part_name],
            facecolor="none", linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(bb[0], bb[1] - 5, part_name, fontsize=8,
                color=BODY_PART_COLORS[part_name], fontweight="bold")

    # Draw keypoints
    for i in range(len(kp_2d)):
        if kp_valid[i]:
            ax.plot(kp_2d[i, 0], kp_2d[i, 1], "o", markersize=4, color="white",
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.text(kp_2d[i, 0] + 3, kp_2d[i, 1], KP_NAMES[i], fontsize=5, color="white")
    ax.set_title(f"Frame {frame_idx}: GT + Keypoints + BBoxes")
    ax.axis("off")

    # === Panel 2: All Gaussians (top-K opacity, color by opacity) ===
    ax = axes[0, 1]
    ax.imshow(gt_image, alpha=0.3)
    top_mask = filter_by_opacity_topk(gauss_opacity, top_k) & gauss_valid
    sc = ax.scatter(
        gauss_2d[top_mask, 0], gauss_2d[top_mask, 1],
        c=gauss_opacity[top_mask], cmap="hot", s=0.1, alpha=0.3, vmin=0.5, vmax=0.73
    )
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_title(f"All Gaussians (top-{top_k//1000}K by opacity)")
    ax.axis("off")

    # === Panel 3: Gaussians colored by body part ===
    ax = axes[0, 2]
    ax.imshow(gt_image, alpha=0.3)
    for part_name, mask in part_masks.items():
        color = BODY_PART_COLORS[part_name]
        part_top = mask & filter_by_opacity_topk(gauss_opacity, top_k)
        if part_top.sum() > 0:
            ax.scatter(
                gauss_2d[part_top, 0], gauss_2d[part_top, 1],
                c=color, s=0.5, alpha=0.5, label=f"{part_name} ({part_top.sum():,})"
            )
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.legend(fontsize=7, loc="upper right", markerscale=10)
    ax.set_title("Gaussians by Body Part (2D filtered)")
    ax.axis("off")

    # === Panels 4-6: Individual body parts (face, left_paw, tail) ===
    highlight_parts = ["face", "left_paw", "tail"]
    for ax, part_name in zip(axes[1], highlight_parts):
        if part_name not in part_masks or part_name not in bboxes:
            ax.text(0.5, 0.5, f"{part_name}: not visible", transform=ax.transAxes,
                    ha="center", fontsize=12)
            ax.axis("off")
            continue

        mask = part_masks[part_name]
        bb = bboxes[part_name]["bbox"]
        color = BODY_PART_COLORS[part_name]

        # Zoom to bbox region with margin
        margin = 30
        x0 = max(0, bb[0] - margin)
        y0 = max(0, bb[1] - margin)
        x1 = min(512, bb[2] + margin)
        y1 = min(512, bb[3] + margin)

        ax.imshow(gt_image)
        ax.scatter(
            gauss_2d[mask, 0], gauss_2d[mask, 1],
            c=color, s=2, alpha=0.6
        )
        # Draw keypoints in this part
        for kp_i in BODY_PARTS[part_name]:
            if kp_valid[kp_i]:
                ax.plot(kp_2d[kp_i, 0], kp_2d[kp_i, 1], "*", markersize=12,
                        color="white", markeredgecolor="black", markeredgewidth=1)

        rect = mpatches.Rectangle(
            (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"{part_name}: {mask.sum():,} Gaussians")
        ax.axis("off")

    plt.suptitle(
        f"View-Projected Gaussian Filtering — Frame {frame_idx}\n"
        f"Total Gaussians: {len(gauss_opacity):,}, Visible: {gauss_valid.sum():,}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(output_dir / f"frame_{frame_idx:06d}_body_parts.png", dpi=150, bbox_inches="tight")
    plt.close()


def process_frame(
    frame_idx: int,
    gauss_dir: str,
    kp_path: str,
    m5_dir: str,
    view_idx: int = 0,
    padding_px: int = 20,
    output_dir: Path = Path("outputs"),
) -> Dict:
    """Process a single frame: load, project, filter, visualize."""
    # Load Gaussians
    gauss_path = Path(gauss_dir) / f"{frame_idx:06d}.npz"
    if not gauss_path.exists():
        print(f"  Frame {frame_idx}: Gaussian NPZ not found at {gauss_path}")
        return {}

    gauss = load_gaussians(str(gauss_path))
    print(f"  Gaussians: {gauss['n_gaussians']:,}")

    # Load camera
    cam_path = Path(m5_dir) / f"{frame_idx:06d}" / "opencv_cameras.json"
    cam = load_camera(str(cam_path), view_idx)

    # Load GT image
    img_path = Path(m5_dir) / f"{frame_idx:06d}" / "images" / f"cam_{view_idx:03d}.png"
    if img_path.exists():
        from PIL import Image
        gt_image = np.array(Image.open(img_path))[:, :, :3]
    else:
        gt_image = np.ones((512, 512, 3), dtype=np.uint8) * 128

    # Load keypoints (convert MAMMAL mm → GS-LRM normalized space)
    kp_gslrm = load_keypoints_gslrm(kp_path, frame_idx)  # (22, 3)

    # Pre-filter Gaussians: keep only those within expanded keypoint bounding box in 3D
    kp_min = kp_gslrm.min(axis=0) - 0.3  # margin in GS-LRM units
    kp_max = kp_gslrm.max(axis=0) + 0.3
    xyz = gauss["xyz"]
    spatial_mask = (
        (xyz[:, 0] >= kp_min[0]) & (xyz[:, 0] <= kp_max[0]) &
        (xyz[:, 1] >= kp_min[1]) & (xyz[:, 1] <= kp_max[1]) &
        (xyz[:, 2] >= kp_min[2]) & (xyz[:, 2] <= kp_max[2])
    )
    print(f"  3D spatial filter: {spatial_mask.sum():,}/{len(xyz):,} Gaussians near object")

    # Project keypoints to 2D
    kp_2d, kp_valid = project_points_to_2d(
        kp_gslrm, cam["w2c"], cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    )

    # Check if keypoints project within image
    kp_in_img = (
        kp_valid
        & (kp_2d[:, 0] >= 0) & (kp_2d[:, 0] < cam["w"])
        & (kp_2d[:, 1] >= 0) & (kp_2d[:, 1] < cam["h"])
    )
    print(f"  Keypoints visible: {kp_in_img.sum()}/22")

    # Project only spatially-filtered Gaussians to 2D (saves compute)
    gauss_2d_full = np.zeros((gauss["n_gaussians"], 2), dtype=np.float32)
    gauss_valid_full = np.zeros(gauss["n_gaussians"], dtype=bool)
    if spatial_mask.sum() > 0:
        g2d, gv = project_points_to_2d(
            gauss["xyz"][spatial_mask], cam["w2c"], cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        )
        gauss_2d_full[spatial_mask] = g2d
        gauss_valid_full[spatial_mask] = gv
    gauss_2d = gauss_2d_full
    gauss_valid = gauss_valid_full

    gauss_in_img = (
        gauss_valid
        & (gauss_2d[:, 0] >= 0) & (gauss_2d[:, 0] < cam["w"])
        & (gauss_2d[:, 1] >= 0) & (gauss_2d[:, 1] < cam["h"])
    )
    print(f"  Gaussians in image: {gauss_in_img.sum():,}/{gauss['n_gaussians']:,} "
          f"(after 3D filter: {spatial_mask.sum():,})")

    # Compute body part bounding boxes
    bboxes = compute_body_part_bboxes(kp_2d, kp_in_img, padding_px, cam["w"])
    print(f"  Body parts visible: {list(bboxes.keys())}")

    # Filter Gaussians by body part
    part_masks = {}
    part_stats = {}
    for part_name, bbox_info in bboxes.items():
        mask = filter_gaussians_by_bbox(gauss_2d, gauss_in_img, bbox_info["bbox"])
        part_masks[part_name] = mask
        part_stats[part_name] = {
            "n_gaussians": int(mask.sum()),
            "bbox": bbox_info["bbox"],
            "mean_opacity": float(gauss["opacity"][mask].mean()) if mask.sum() > 0 else 0,
        }
        print(f"    {part_name}: {mask.sum():,} Gaussians")

    # Visualize
    visualize_frame(
        frame_idx, gt_image, kp_2d, kp_in_img,
        gauss_2d, gauss_in_img, gauss["opacity"],
        bboxes, part_masks, output_dir
    )

    return {
        "frame_idx": frame_idx,
        "view_idx": view_idx,
        "n_gaussians_total": gauss["n_gaussians"],
        "n_gaussians_visible": int(gauss_in_img.sum()),
        "n_keypoints_visible": int(kp_in_img.sum()),
        "body_parts": part_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="View-Projected Gaussian Filtering")
    parser.add_argument("--mode", choices=["bbox", "radial"], default="bbox",
                        help="Filtering mode: bbox (original) or radial (circle + sweep)")
    parser.add_argument("--frame-idx", nargs="+", type=int, default=[0, 100, 500, 1000, 2000])
    parser.add_argument("--view-idx", type=int, default=0, help="Camera view index (0-5)")
    parser.add_argument("--padding-px", type=int, default=20, help="BBox padding in pixels")
    parser.add_argument("--radii", nargs="+", type=float, default=[10, 20, 30, 40, 50, 70, 100],
                        help="Radii to sweep in radial mode (pixels)")
    from mouse_extensions.paths import KP_22, GAUSSIANS_RAW_DIR
    parser.add_argument("--gauss-dir", default=str(GAUSSIANS_RAW_DIR))
    parser.add_argument("--kp-path",
                        default=str(KP_22))
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/filtering/view_projected_filtering")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for fi in args.frame_idx:
        print(f"\nProcessing frame {fi} (mode={args.mode})...")
        if args.mode == "radial":
            result = process_frame_radial(
                fi, args.gauss_dir, args.kp_path, args.m5_dir,
                args.view_idx, args.radii, output_dir
            )
        else:
            result = process_frame(
                fi, args.gauss_dir, args.kp_path, args.m5_dir,
                args.view_idx, args.padding_px, output_dir
            )
        if result:
            all_results.append(result)

    # Save summary
    summary_path = output_dir / "filtering_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"SUMMARY: View-Projected Gaussian Filtering (mode={args.mode})")
    print("=" * 70)
    for r in all_results:
        if args.mode == "radial":
            print(f"\nFrame {r['frame_idx']} (view {r['view_idx']}):")
            print(f"  Total: {r['n_gaussians_total']:,}, Visible: {r['n_gaussians_visible']:,}")
            for sweep in r.get("radii_sweep", []):
                print(f"  r={sweep['radius']:5.0f}px: "
                      f"coverage={sweep['coverage']:.1%}, "
                      f"assigned={sweep['total_assigned']:,}")
        else:
            print(f"\nFrame {r['frame_idx']} (view {r['view_idx']}):")
            print(f"  Total: {r['n_gaussians_total']:,}, Visible: {r['n_gaussians_visible']:,}")
            for part, stats in r["body_parts"].items():
                print(f"  {part:12s}: {stats['n_gaussians']:6,} Gaussians, "
                      f"opacity={stats['mean_opacity']:.4f}")


if __name__ == "__main__":
    main()
