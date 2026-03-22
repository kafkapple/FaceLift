#!/usr/bin/env python3
"""
Multi-View Triangulation Analysis for 3D Keypoints.

Analyzes triangulation accuracy using DANNCE 2D detections and MAMMAL 3D GT.
Supports both real camera views and GS-LRM novel views.

Core workflow:
    1. Load DANNCE 2D keypoint detections (6 views)
    2. Load raw camera parameters (intrinsics + extrinsics)
    3. DLT triangulate from N views
    4. Compare with MAMMAL 3D ground truth (MPJPE)
    5. Analyze view count vs accuracy tradeoff

Usage:
    from mouse_extensions.analysis.triangulation_analysis import (
        load_dannce_2d, load_raw_cameras, run_gt_view_experiment,
    )

    dannce_2d = load_dannce_2d(data_dir)
    cameras = load_raw_cameras(cam_pkl_path)
    results = run_gt_view_experiment(dannce_2d, mammal_3d, cameras)

Date: 2026-03-04
"""

import pickle
import itertools
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_dannce_2d(
    data_dir: Union[str, Path],
    views: List[int] = None,
) -> np.ndarray:
    """Load DANNCE 2D keypoint detections from per-view .pkl files.

    Args:
        data_dir: Directory containing keypoints2d_undist/result_view_{i}.pkl
        views: Which view indices to load (default: [0,1,2,3,4,5])

    Returns:
        (N_views, N_frames, 22, 3) — (x_pixel, y_pixel, confidence)
    """
    if views is None:
        views = list(range(6))

    data_dir = Path(data_dir)
    kp_dir = data_dir / "keypoints2d_undist"

    all_views = []
    for v in views:
        pkl_path = kp_dir / f"result_view_{v}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing DANNCE 2D file: {pkl_path}")

        with open(pkl_path, "rb") as f:
            kp_data = pickle.load(f)

        # Expected shape: (18000, 22, 3)
        if isinstance(kp_data, dict):
            # Some formats store as dict with 'keypoints' key
            kp_data = kp_data.get("keypoints", kp_data.get("predictions"))
        kp_data = np.array(kp_data)

        logger.info(f"View {v}: shape={kp_data.shape}")
        all_views.append(kp_data)

    result = np.stack(all_views, axis=0)  # (N_views, N_frames, 22, 3)
    logger.info(f"Loaded DANNCE 2D: {result.shape}")
    return result


def load_mammal_3d(
    npz_path: Union[str, Path],
) -> np.ndarray:
    """Load MAMMAL 3D keypoint ground truth.

    Args:
        npz_path: Path to keypoints_22_3d.npz

    Returns:
        (N_frames, 22, 3) in world coordinates (mm)
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    # Try common key names
    for key in ["keypoints_3d", "keypoints", "arr_0"]:
        if key in data:
            kp3d = data[key]
            logger.info(f"Loaded MAMMAL 3D: {kp3d.shape} from key='{key}'")
            return kp3d

    # If only one array, use it
    keys = list(data.keys())
    if len(keys) == 1:
        kp3d = data[keys[0]]
        logger.info(f"Loaded MAMMAL 3D: {kp3d.shape} from key='{keys[0]}'")
        return kp3d

    raise KeyError(f"Cannot find keypoints in {npz_path}. Keys: {keys}")


def load_raw_cameras(
    cam_pkl_path: Union[str, Path],
) -> List[Dict[str, np.ndarray]]:
    """Load raw camera parameters from new_cam.pkl.

    Args:
        cam_pkl_path: Path to new_cam.pkl

    Returns:
        List of dicts with keys: 'K' (3,3), 'R' (3,3), 't' (3,1),
        'dist' (optional distortion coeffs)
    """
    cam_pkl_path = Path(cam_pkl_path)

    with open(cam_pkl_path, "rb") as f:
        cam_data = pickle.load(f)

    cameras = []

    if isinstance(cam_data, dict):
        # Format: dict with camera indices or names as keys
        n_cams = len(cam_data)
        for i in range(n_cams):
            key = i if i in cam_data else str(i)
            if key not in cam_data:
                # Try other key patterns
                for k in cam_data:
                    if str(i) in str(k):
                        key = k
                        break

            cam = cam_data[key]
            cameras.append(_parse_camera_entry(cam, i))
    elif isinstance(cam_data, list):
        for i, cam in enumerate(cam_data):
            cameras.append(_parse_camera_entry(cam, i))
    else:
        raise ValueError(f"Unexpected camera data format: {type(cam_data)}")

    logger.info(f"Loaded {len(cameras)} cameras from {cam_pkl_path}")
    return cameras


def _parse_camera_entry(cam: dict, idx: int) -> Dict[str, np.ndarray]:
    """Parse a single camera entry from various formats."""
    result = {}

    # Intrinsics
    if "K" in cam:
        result["K"] = np.array(cam["K"], dtype=np.float64)
    elif "intrinsic" in cam:
        result["K"] = np.array(cam["intrinsic"], dtype=np.float64)
    elif "fx" in cam:
        result["K"] = np.array([
            [cam["fx"], 0, cam["cx"]],
            [0, cam["fy"], cam["cy"]],
            [0, 0, 1],
        ], dtype=np.float64)
    else:
        raise ValueError(f"Camera {idx}: no intrinsics found. Keys: {list(cam.keys())}")

    # Extrinsics: R and t
    if "R" in cam:
        result["R"] = np.array(cam["R"], dtype=np.float64)
    elif "rotation" in cam:
        result["R"] = np.array(cam["rotation"], dtype=np.float64)
    else:
        raise ValueError(f"Camera {idx}: no rotation found")

    if "t" in cam:
        result["t"] = np.array(cam["t"], dtype=np.float64).reshape(3, 1)
    elif "T" in cam:
        result["t"] = np.array(cam["T"], dtype=np.float64).reshape(3, 1)
    elif "translation" in cam:
        result["t"] = np.array(cam["translation"], dtype=np.float64).reshape(3, 1)
    else:
        raise ValueError(f"Camera {idx}: no translation found")

    # Optional distortion
    if "dist" in cam:
        result["dist"] = np.array(cam["dist"], dtype=np.float64)
    elif "distortion" in cam:
        result["dist"] = np.array(cam["distortion"], dtype=np.float64)

    return result


# =============================================================================
# Projection & Triangulation
# =============================================================================

def build_projection_matrix(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Build 3x4 projection matrix P = K @ [R | t].

    Args:
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation matrix (world-to-camera)
        t: (3, 1) translation vector

    Returns:
        (3, 4) projection matrix
    """
    Rt = np.hstack([R, t.reshape(3, 1)])  # (3, 4)
    return K @ Rt


def triangulate_dlt(
    points_2d: np.ndarray,
    proj_matrices: np.ndarray,
    confidences: Optional[np.ndarray] = None,
) -> np.ndarray:
    """DLT triangulation from N views for a single 3D point.

    Solves the linear system A @ X = 0 using SVD.

    Args:
        points_2d: (N_views, 2) pixel coordinates
        proj_matrices: (N_views, 3, 4) projection matrices
        confidences: (N_views,) optional weights per view

    Returns:
        (3,) 3D point in world coordinates
    """
    A = []
    for i in range(len(points_2d)):
        x, y = points_2d[i]
        P = proj_matrices[i]

        row1 = x * P[2] - P[0]
        row2 = y * P[2] - P[1]

        if confidences is not None:
            w = confidences[i]
            row1 = row1 * w
            row2 = row2 * w

        A.append(row1)
        A.append(row2)

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / (X[3] + 1e-10)


def triangulate_batch(
    points_2d: np.ndarray,
    proj_matrices: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    conf_threshold: float = 0.1,
) -> np.ndarray:
    """Triangulate multiple keypoints across N views.

    Args:
        points_2d: (N_views, N_joints, 3) — (x, y, conf) per view per joint
        proj_matrices: (N_views, 3, 4) projection matrices
        confidences: External confidence override (N_views, N_joints).
                     If None, uses the confidence channel from points_2d.
        conf_threshold: Minimum confidence to include a view

    Returns:
        (N_joints, 3) triangulated 3D points
    """
    n_views, n_joints = points_2d.shape[:2]
    result = np.zeros((n_joints, 3))

    for j in range(n_joints):
        # Gather 2D coordinates and confidences for this joint
        pts = points_2d[:, j, :2]  # (N_views, 2)
        if confidences is not None:
            confs = confidences[:, j]
        else:
            confs = points_2d[:, j, 2]  # confidence channel

        # Filter by confidence threshold
        valid = confs > conf_threshold
        if valid.sum() < 2:
            # Need at least 2 views; fall back to top-2
            top2 = np.argsort(confs)[-2:]
            valid = np.zeros(n_views, dtype=bool)
            valid[top2] = True

        pts_valid = pts[valid]
        P_valid = proj_matrices[valid]
        confs_valid = confs[valid]

        result[j] = triangulate_dlt(pts_valid, P_valid, confs_valid)

    return result


def project_3d_to_2d(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to 2D pixel coordinates.

    Args:
        points_3d: (N, 3) points in world coordinates
        K: (3, 3) intrinsic matrix
        R: (3, 3) rotation (world-to-camera)
        t: (3, 1) translation

    Returns:
        (N, 2) pixel coordinates
    """
    t = t.reshape(3, 1)
    # World to camera
    pts_cam = R @ points_3d.T + t  # (3, N)
    # Camera to pixel
    pts_px = K @ pts_cam  # (3, N)
    pts_px = pts_px[:2] / (pts_px[2:] + 1e-10)  # (2, N)
    return pts_px.T  # (N, 2)


# =============================================================================
# Metrics
# =============================================================================

def compute_mpjpe(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
) -> Dict[str, float]:
    """Compute MPJPE (Mean Per-Joint Position Error).

    Args:
        pred_3d: (N_joints, 3) or (N_frames, N_joints, 3)
        gt_3d: Same shape as pred_3d

    Returns:
        Dict with 'mpjpe', 'per_joint' errors
    """
    diff = pred_3d - gt_3d
    per_point_error = np.linalg.norm(diff, axis=-1)  # (..., N_joints)

    return {
        "mpjpe": float(np.mean(per_point_error)),
        "mpjpe_std": float(np.std(per_point_error)),
        "per_joint": per_point_error.mean(axis=0).tolist()
            if per_point_error.ndim > 1 else per_point_error.tolist(),
        "median": float(np.median(per_point_error)),
        "max": float(np.max(per_point_error)),
    }


def compute_pa_mpjpe(
    pred_3d: np.ndarray,
    gt_3d: np.ndarray,
) -> Dict[str, float]:
    """Compute PA-MPJPE (Procrustes-Aligned MPJPE).

    Finds optimal rotation, translation, and scale to align pred to gt,
    then computes MPJPE on the aligned prediction.

    Args:
        pred_3d: (N_joints, 3)
        gt_3d: (N_joints, 3)

    Returns:
        Dict with 'pa_mpjpe' and alignment info
    """
    # Center both
    pred_centered = pred_3d - pred_3d.mean(axis=0)
    gt_centered = gt_3d - gt_3d.mean(axis=0)

    # Optimal rotation via SVD (Kabsch algorithm)
    H = pred_centered.T @ gt_centered  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R_opt = Vt.T @ sign_matrix @ U.T

    # Optimal scale
    scale = np.trace(R_opt @ H) / (np.sum(pred_centered ** 2) + 1e-10)

    # Align
    pred_aligned = scale * (pred_centered @ R_opt.T) + gt_3d.mean(axis=0)

    # Compute error on aligned
    diff = pred_aligned - gt_3d
    per_point_error = np.linalg.norm(diff, axis=-1)

    return {
        "pa_mpjpe": float(np.mean(per_point_error)),
        "pa_mpjpe_std": float(np.std(per_point_error)),
        "scale": float(scale),
    }


# =============================================================================
# View Selection Strategies
# =============================================================================

def select_views_uniform(n_total: int, n_select: int) -> List[int]:
    """Select N views with uniform angular spacing from available views.

    Assumes cameras are roughly equally spaced around the subject.
    """
    if n_select >= n_total:
        return list(range(n_total))
    indices = np.linspace(0, n_total, n_select, endpoint=False, dtype=int)
    return indices.tolist()


def select_views_by_confidence(
    confidences: np.ndarray,
    n_select: int,
) -> List[int]:
    """Select N views with highest average confidence.

    Args:
        confidences: (N_views, N_joints) confidence scores
        n_select: Number of views to select

    Returns:
        List of selected view indices
    """
    mean_conf = confidences.mean(axis=1)  # (N_views,)
    return np.argsort(mean_conf)[-n_select:].tolist()


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_gt_view_experiment(
    dannce_2d: np.ndarray,
    mammal_3d: np.ndarray,
    cameras: List[Dict[str, np.ndarray]],
    view_counts: List[int] = None,
    frame_range: Tuple[int, int] = (3240, 3600),
    frame_step: int = 5,
    selection_strategy: str = "uniform",
) -> Dict[str, dict]:
    """Run triangulation accuracy experiment with varying view counts.

    For each view count, selects views, triangulates DANNCE 2D detections,
    and compares with MAMMAL 3D ground truth.

    Args:
        dannce_2d: (6, N_dannce_frames, 22, 3) — DANNCE detections at 100fps
        mammal_3d: (3600, 22, 3) — MAMMAL GT at 20fps (M5 frames)
        cameras: List of camera dicts from load_raw_cameras()
        view_counts: List of N values to test (default: [2,3,4,5,6])
        frame_range: (start, end) M5 frame indices for test set
        frame_step: Step between evaluated frames (default: 5 for speed)
        selection_strategy: "uniform" or "confidence"

    Returns:
        {n_views: {mpjpe, pa_mpjpe, per_joint, n_frames, selected_views, ...}}
    """
    if view_counts is None:
        view_counts = [2, 3, 4, 5, 6]

    # Build projection matrices for all cameras
    n_cams = len(cameras)
    proj_matrices_all = np.zeros((n_cams, 3, 4))
    for i, cam in enumerate(cameras):
        proj_matrices_all[i] = build_projection_matrix(cam["K"], cam["R"], cam["t"])

    start_frame, end_frame = frame_range
    # DANNCE frame index = M5_frame * 5
    m5_frames = list(range(start_frame, end_frame, frame_step))

    results = {}

    for n_views in view_counts:
        logger.info(f"Testing {n_views}-view triangulation...")

        # Select views
        if selection_strategy == "uniform":
            selected = select_views_uniform(n_cams, n_views)
        elif selection_strategy == "confidence":
            # Use average confidence across test frames
            mean_conf = dannce_2d[:, :, :, 2].mean(axis=(1, 2))
            selected = select_views_by_confidence(
                dannce_2d[:, :, :, 2].mean(axis=1), n_views
            )
        else:
            raise ValueError(f"Unknown strategy: {selection_strategy}")

        proj_matrices = proj_matrices_all[selected]

        all_pred_3d = []
        all_gt_3d = []

        for m5_frame in m5_frames:
            # Map M5 frame to DANNCE frame index
            dannce_frame = m5_frame * 5

            if dannce_frame >= dannce_2d.shape[1]:
                continue

            # Get 2D detections for selected views
            kp2d = dannce_2d[selected, dannce_frame]  # (n_views, 22, 3)

            # Triangulate
            pred_3d = triangulate_batch(kp2d, proj_matrices)  # (22, 3)
            gt_3d = mammal_3d[m5_frame]  # (22, 3)

            all_pred_3d.append(pred_3d)
            all_gt_3d.append(gt_3d)

        all_pred_3d = np.array(all_pred_3d)  # (N_frames, 22, 3)
        all_gt_3d = np.array(all_gt_3d)      # (N_frames, 22, 3)

        # Compute metrics
        mpjpe_results = compute_mpjpe(all_pred_3d, all_gt_3d)

        # PA-MPJPE per frame, then average
        pa_errors = []
        for i in range(len(all_pred_3d)):
            pa = compute_pa_mpjpe(all_pred_3d[i], all_gt_3d[i])
            pa_errors.append(pa["pa_mpjpe"])

        results[n_views] = {
            **mpjpe_results,
            "pa_mpjpe": float(np.mean(pa_errors)),
            "pa_mpjpe_std": float(np.std(pa_errors)),
            "n_frames": len(all_pred_3d),
            "selected_views": selected,
        }

        logger.info(
            f"  {n_views} views: MPJPE={mpjpe_results['mpjpe']:.2f}mm, "
            f"PA-MPJPE={results[n_views]['pa_mpjpe']:.2f}mm"
        )

    return results


def run_noise_experiment(
    points_3d_gt: np.ndarray,
    cameras: List[Dict[str, np.ndarray]],
    proj_matrices: np.ndarray,
    noise_levels: List[float] = None,
    view_counts: List[int] = None,
) -> Dict[str, Dict[str, dict]]:
    """Run triangulation experiment with synthetic noise on projected 2D points.

    Projects GT 3D keypoints to 2D, adds Gaussian noise, then triangulates back.
    This isolates the effect of detection noise from actual detector errors.

    Args:
        points_3d_gt: (N_frames, 22, 3) GT 3D keypoints
        cameras: Camera parameters
        proj_matrices: (N_views, 3, 4) projection matrices
        noise_levels: Noise sigma in pixels (default: [0, 1, 2, 5])
        view_counts: Number of views to test (default: [6, 9, 12, 24])

    Returns:
        {n_views: {noise_sigma: {mpjpe, ...}}}
    """
    if noise_levels is None:
        noise_levels = [0.0, 1.0, 2.0, 5.0]
    if view_counts is None:
        view_counts = [6, 9, 12, 24]

    results = {}
    n_total = len(proj_matrices)

    for n_views in view_counts:
        # Select views uniformly
        selected = select_views_uniform(n_total, n_views)
        P_sel = proj_matrices[selected]

        results[n_views] = {}

        for sigma in noise_levels:
            all_pred = []

            for frame_idx in range(len(points_3d_gt)):
                gt_3d = points_3d_gt[frame_idx]  # (22, 3)
                n_joints = gt_3d.shape[0]

                # Project to 2D for each selected view
                pts_2d = np.zeros((len(selected), n_joints, 3))
                for vi, cam_idx in enumerate(selected):
                    cam = cameras[cam_idx]
                    projected = project_3d_to_2d(gt_3d, cam["K"], cam["R"], cam["t"])
                    # Add Gaussian noise
                    if sigma > 0:
                        noise = np.random.randn(*projected.shape) * sigma
                        projected = projected + noise
                    pts_2d[vi, :, :2] = projected
                    pts_2d[vi, :, 2] = 1.0  # perfect confidence

                pred_3d = triangulate_batch(pts_2d, P_sel)
                all_pred.append(pred_3d)

            all_pred = np.array(all_pred)
            metrics = compute_mpjpe(all_pred, points_3d_gt)

            results[n_views][sigma] = metrics
            logger.info(
                f"  {n_views} views, σ={sigma:.1f}px: "
                f"MPJPE={metrics['mpjpe']:.4f}mm"
            )

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_accuracy_vs_views(
    results: Dict[int, dict],
    output_path: Union[str, Path],
    title: str = "Triangulation Accuracy vs Number of Views",
):
    """Plot MPJPE vs view count.

    Args:
        results: Output from run_gt_view_experiment()
        output_path: Where to save the plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    view_counts = sorted(results.keys())
    mpjpe_values = [results[n]["mpjpe"] for n in view_counts]
    pa_mpjpe_values = [results[n]["pa_mpjpe"] for n in view_counts]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(view_counts, mpjpe_values, "o-", label="MPJPE", color="tab:blue", linewidth=2)
    ax.plot(view_counts, pa_mpjpe_values, "s--", label="PA-MPJPE", color="tab:orange", linewidth=2)

    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("Error (mm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(view_counts)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved plot: {output_path}")


def plot_noise_experiment(
    results: Dict[int, Dict[float, dict]],
    output_path: Union[str, Path],
    title: str = "Triangulation: Views vs Noise",
):
    """Plot MPJPE heatmap/lines for view count × noise level.

    Args:
        results: Output from run_noise_experiment()
        output_path: Where to save the plot
        title: Plot title
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    view_counts = sorted(results.keys())
    noise_levels = sorted(results[view_counts[0]].keys())

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(noise_levels)))

    for ni, sigma in enumerate(noise_levels):
        mpjpes = [results[n][sigma]["mpjpe"] for n in view_counts]
        label = f"σ={sigma:.0f}px" if sigma > 0 else "σ=0 (perfect)"
        ax.plot(view_counts, mpjpes, "o-", label=label, color=colors[ni], linewidth=2)

    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(view_counts)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved plot: {output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run GT view experiment from command line."""
    import argparse
    from mouse_extensions.paths import KP_22

    parser = argparse.ArgumentParser(
        description="Triangulation accuracy analysis with DANNCE 2D & MAMMAL 3D"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/data/raw/markerless_mouse_1_nerf",
        help="Directory with DANNCE keypoints2d_undist/",
    )
    parser.add_argument(
        "--cam-pkl",
        type=str,
        default="~/data/raw/markerless_mouse_1_nerf/new_cam.pkl",
        help="Path to camera parameters pkl",
    )
    parser.add_argument(
        "--mammal-3d",
        type=str,
        default=str(KP_22),
        help="Path to MAMMAL 3D keypoints npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/triangulation_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        default=[3240, 3600],
        help="M5 frame range (test set)",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=5,
        help="Step between evaluated frames",
    )
    parser.add_argument(
        "--view-counts",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="Number of views to test",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Expand paths
    data_dir = Path(args.data_dir).expanduser()
    cam_pkl = Path(args.cam_pkl).expanduser()
    mammal_path = Path(args.mammal_3d).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading DANNCE 2D keypoints...")
    dannce_2d = load_dannce_2d(data_dir)

    logger.info("Loading MAMMAL 3D ground truth...")
    mammal_3d = load_mammal_3d(mammal_path)

    logger.info("Loading camera parameters...")
    cameras = load_raw_cameras(cam_pkl)

    # Run experiment
    logger.info("Running GT view experiment...")
    results = run_gt_view_experiment(
        dannce_2d=dannce_2d,
        mammal_3d=mammal_3d,
        cameras=cameras,
        view_counts=args.view_counts,
        frame_range=tuple(args.frame_range),
        frame_step=args.frame_step,
    )

    # Save results
    results_path = output_dir / "gt_view_results.json"

    # Convert numpy types for JSON
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {}
    for k, v in results.items():
        serializable[str(k)] = {
            kk: _convert(vv) for kk, vv in v.items()
        }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved results: {results_path}")

    # Plot
    plot_path = output_dir / "accuracy_vs_views.png"
    plot_accuracy_vs_views(results, plot_path)

    # Print summary
    print("\n" + "=" * 60)
    print("TRIANGULATION ANALYSIS RESULTS")
    print("=" * 60)
    for n_views in sorted(results.keys()):
        r = results[n_views]
        print(
            f"  {n_views} views: MPJPE={r['mpjpe']:.2f}mm "
            f"(±{r['mpjpe_std']:.2f}), "
            f"PA-MPJPE={r['pa_mpjpe']:.2f}mm, "
            f"views={r['selected_views']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
