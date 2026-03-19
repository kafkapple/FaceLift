"""Strategy C: Temporal Derivative Feature Extraction for BehaviorBench.

Extracts temporal dynamics features from 3D keypoint sequences without
requiring Gaussian correspondence across frames. This is possible because
keypoints (from MAMMAL) have consistent identity across frames, unlike
per-frame independent GS-LRM Gaussians.

Feature groups:
    1. Centroid velocity/acceleration (7 dims)
    2. Body-part delta features (88 dims)
    3. Rigid-nonrigid decomposition via Kabsch (25 dims)
    4. Sliding window statistics (9 dims)
    Total: 129 dims per frame

Usage:
    python -m mouse_extensions.behavior.extract_temporal_features \
        --keypoints /path/to/keypoints_22_3d.npz \
        --output-dir /path/to/output/ \
        --fps 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


# Frames with tracking jumps (session boundaries in MAMMAL data)
FRAME_JUMPS = {
    1178, 1179, 1180, 1181, 1182,
    2358, 2359, 2360, 2361, 2362,
    3538, 3539, 3540, 3541, 3542,
}


def compute_centroid_features(keypoints: np.ndarray, fps: float) -> np.ndarray:
    """Compute centroid velocity and acceleration from keypoint sequences.

    Args:
        keypoints: (N, 22, 3) keypoint positions across frames.
        fps: frames per second for scaling derivatives.

    Returns:
        features: (N, 7) array with [vx, vy, vz, speed, ax, ay, az].
            First frame is zero-padded (no preceding frame for diff).
    """
    N = len(keypoints)
    # Centroid: mean of all 22 keypoints per frame
    centroid = keypoints.mean(axis=1)  # (N, 3)

    # Velocity: finite difference scaled by fps
    # Use forward diff, pad last frame with zeros
    velocity = np.zeros((N, 3), dtype=np.float64)
    velocity[:-1] = np.diff(centroid, axis=0) * fps  # (N-1, 3)

    # Speed: L2 norm of velocity
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)  # (N, 1)

    # Acceleration: finite difference of velocity
    acceleration = np.zeros((N, 3), dtype=np.float64)
    acceleration[:-1] = np.diff(velocity, axis=0) * fps  # (N-1, 3)

    # Stack: [vx, vy, vz, speed, ax, ay, az] = 7 dims
    features = np.concatenate([velocity, speed, acceleration], axis=1)

    return features.astype(np.float32)


def compute_bodypart_delta_features(keypoints: np.ndarray, fps: float) -> np.ndarray:
    """Compute per-keypoint velocity (delta) features.

    Args:
        keypoints: (N, 22, 3) keypoint positions.
        fps: frames per second.

    Returns:
        features: (N, 88) array with per-keypoint [dx, dy, dz] (66) + speed (22).
    """
    N, J = keypoints.shape[:2]  # J = 22

    # Per-keypoint delta: kp[t+1] - kp[t], scaled by fps
    deltas = np.zeros((N, J, 3), dtype=np.float64)
    deltas[:-1] = np.diff(keypoints, axis=0) * fps  # (N-1, 22, 3)

    # Per-keypoint speed
    speeds = np.linalg.norm(deltas, axis=2)  # (N, 22)

    # Flatten: 22*3 = 66 + 22 = 88 dims
    deltas_flat = deltas.reshape(N, J * 3)  # (N, 66)
    features = np.concatenate([deltas_flat, speeds], axis=1)  # (N, 88)

    return features.astype(np.float32)


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal rotation and translation (Kabsch algorithm).

    Finds R, t that minimizes ||Q - (R @ P.T).T - t||^2.

    Args:
        P: (K, 3) source points.
        Q: (K, 3) target points.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
    """
    # Center both point sets
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_c = P - centroid_P
    Q_c = Q - centroid_Q

    # Cross-covariance matrix
    H = P_c.T @ Q_c  # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1, not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T
    t = centroid_Q - R @ centroid_P

    return R, t


def rotation_angle(R: np.ndarray) -> float:
    """Extract rotation angle (radians) from a 3x3 rotation matrix.

    Uses the formula: angle = arccos((trace(R) - 1) / 2).
    """
    trace_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace_val))


def compute_rigid_nonrigid_features(keypoints: np.ndarray) -> np.ndarray:
    """Decompose frame-to-frame motion into rigid and nonrigid components.

    Uses the Kabsch algorithm to find the optimal rigid transform between
    consecutive frames, then measures the residual (nonrigid deformation).

    Args:
        keypoints: (N, 22, 3) keypoint positions.

    Returns:
        features: (N, 25) array with:
            - per-keypoint nonrigid magnitude (22 dims)
            - total nonrigid energy (1 dim)
            - rotation angle in radians (1 dim)
            - translation magnitude (1 dim)
    """
    N, J = keypoints.shape[:2]
    features = np.zeros((N, J + 3), dtype=np.float64)  # 22 + 3 = 25

    for t in range(N - 1):
        P = keypoints[t]      # (22, 3) source
        Q = keypoints[t + 1]  # (22, 3) target

        # Kabsch: find best rigid alignment P -> Q
        R, tvec = kabsch_rotation(P, Q)

        # Rigid prediction
        Q_rigid = (R @ P.T).T + tvec  # (22, 3)

        # Nonrigid residual
        residual = Q - Q_rigid  # (22, 3)
        residual_mag = np.linalg.norm(residual, axis=1)  # (22,)

        # Per-keypoint nonrigid magnitude
        features[t, :J] = residual_mag

        # Total nonrigid energy (RMS of residuals)
        features[t, J] = float(np.sqrt((residual_mag ** 2).mean()))

        # Rotation angle
        features[t, J + 1] = rotation_angle(R)

        # Translation magnitude
        features[t, J + 2] = float(np.linalg.norm(tvec))

    return features.astype(np.float32)


def compute_sliding_window_features(
    speed: np.ndarray,
    nonrigid_energy: np.ndarray,
    windows: list[int] = None,
) -> np.ndarray:
    """Compute sliding window statistics over speed and nonrigid energy.

    Args:
        speed: (N,) per-frame speed (centroid speed).
        nonrigid_energy: (N,) per-frame total nonrigid energy.
        windows: list of window sizes (default: [5, 10, 20]).

    Returns:
        features: (N, len(windows) * 3) with [mean_speed, std_speed, mean_nonrigid]
            per window. Uses causal (backward-looking) windows, zero-padded at start.
    """
    if windows is None:
        windows = [5, 10, 20]

    N = len(speed)
    n_stats = 3  # mean_speed, std_speed, mean_nonrigid
    features = np.zeros((N, len(windows) * n_stats), dtype=np.float64)

    for wi, w in enumerate(windows):
        offset = wi * n_stats
        for t in range(N):
            # Causal window: [max(0, t-w+1) : t+1]
            start = max(0, t - w + 1)
            win_speed = speed[start:t + 1]
            win_nonrigid = nonrigid_energy[start:t + 1]

            features[t, offset] = win_speed.mean()
            features[t, offset + 1] = win_speed.std() if len(win_speed) > 1 else 0.0
            features[t, offset + 2] = win_nonrigid.mean()

    return features.astype(np.float32)


def mask_frame_jumps(features: np.ndarray, valid_frames: np.ndarray) -> np.ndarray:
    """Zero out features at session boundary frames where diffs are invalid.

    Frames immediately after a gap in frame indices get zeroed because
    the temporal derivative across a session boundary is meaningless.

    Args:
        features: (N, D) feature matrix.
        valid_frames: (N,) sorted frame indices.

    Returns:
        features: (N, D) with boundary frames zeroed.
    """
    # Find where frame index jumps by more than 1
    diffs = np.diff(valid_frames)
    jump_indices = np.where(diffs > 1)[0]

    # Zero the frame right before the jump (its forward diff is invalid)
    for idx in jump_indices:
        features[idx] = 0.0

    return features


def extract_temporal_features(
    keypoints: np.ndarray,
    valid_frames: np.ndarray,
    fps: float = 20.0,
    windows: list[int] = None,
) -> dict[str, np.ndarray]:
    """Extract all temporal feature groups from keypoint sequences.

    Args:
        keypoints: (N, 22, 3) keypoint positions for valid frames.
        valid_frames: (N,) frame indices (sorted, no FRAME_JUMPS).
        fps: frame rate.
        windows: sliding window sizes.

    Returns:
        Dictionary with feature arrays and metadata.
    """
    N, J, _ = keypoints.shape
    print(f"Extracting temporal features: {N} frames, {J} keypoints, fps={fps}")

    t0 = time.time()

    # 1. Centroid velocity/acceleration (7 dims)
    print("  [1/4] Centroid velocity & acceleration...")
    centroid_feats = compute_centroid_features(keypoints, fps)
    centroid_feats = mask_frame_jumps(centroid_feats, valid_frames)
    print(f"         Shape: {centroid_feats.shape} ({time.time()-t0:.1f}s)")

    # 2. Body-part deltas (88 dims)
    print("  [2/4] Body-part delta features...")
    bodypart_feats = compute_bodypart_delta_features(keypoints, fps)
    bodypart_feats = mask_frame_jumps(bodypart_feats, valid_frames)
    print(f"         Shape: {bodypart_feats.shape} ({time.time()-t0:.1f}s)")

    # 3. Rigid-nonrigid decomposition (25 dims)
    print("  [3/4] Rigid-nonrigid decomposition (Kabsch)...")
    rigid_nonrigid_feats = compute_rigid_nonrigid_features(keypoints)
    rigid_nonrigid_feats = mask_frame_jumps(rigid_nonrigid_feats, valid_frames)
    print(f"         Shape: {rigid_nonrigid_feats.shape} ({time.time()-t0:.1f}s)")

    # 4. Sliding window statistics (9 dims)
    print("  [4/4] Sliding window statistics...")
    # Use centroid speed and nonrigid energy as inputs
    centroid_speed = centroid_feats[:, 3]  # speed column
    nonrigid_energy = rigid_nonrigid_feats[:, J]  # total nonrigid energy column
    window_feats = compute_sliding_window_features(
        centroid_speed, nonrigid_energy, windows=windows,
    )
    print(f"         Shape: {window_feats.shape} ({time.time()-t0:.1f}s)")

    # Concatenate all features
    all_features = np.concatenate([
        centroid_feats,        # 7
        bodypart_feats,        # 88
        rigid_nonrigid_feats,  # 25
        window_feats,          # 9
    ], axis=1)

    total_time = time.time() - t0
    print(f"\n  Total: {all_features.shape[1]} dims, {total_time:.1f}s")

    return {
        # Individual feature groups
        "centroid_features": centroid_feats,            # (N, 7)
        "bodypart_delta_features": bodypart_feats,      # (N, 88)
        "rigid_nonrigid_features": rigid_nonrigid_feats,  # (N, 25)
        "sliding_window_features": window_feats,        # (N, 9)
        # Concatenated
        "all_features": all_features,                   # (N, 129)
        # Metadata
        "valid_frames": valid_frames,                   # (N,)
        "feature_dims": np.array([7, 88, 25, 9]),
        "feature_names": np.array([
            "centroid_velocity_acceleration",
            "bodypart_deltas",
            "rigid_nonrigid_kabsch",
            "sliding_window_stats",
        ]),
    }


def print_summary(result: dict[str, np.ndarray]) -> None:
    """Print summary statistics for extracted features."""
    print(f"\n{'='*70}")
    print("TEMPORAL FEATURE EXTRACTION SUMMARY")
    print(f"{'='*70}")

    dims = result["feature_dims"]
    names = result["feature_names"]
    all_feat = result["all_features"]
    N = len(all_feat)

    print(f"Frames: {N}")
    print(f"Total dimensions: {all_feat.shape[1]}")
    print()

    print(f"{'Feature Group':<40} {'Dims':>5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 85)

    col_offset = 0
    for name, d in zip(names, dims):
        group = all_feat[:, col_offset:col_offset + d]
        # Skip zero-only rows for stats (boundary frames)
        nonzero_mask = np.any(group != 0, axis=1)
        if nonzero_mask.sum() > 0:
            g = group[nonzero_mask]
            print(f"  {str(name):<38} {d:>5} {g.mean():>10.4f} {g.std():>10.4f} "
                  f"{g.min():>10.4f} {g.max():>10.4f}")
        else:
            print(f"  {str(name):<38} {d:>5} {'N/A':>10}")
        col_offset += d

    # NaN/Inf check
    n_nan = np.isnan(all_feat).sum()
    n_inf = np.isinf(all_feat).sum()
    print(f"\nNaN count: {n_nan}, Inf count: {n_inf}")

    # Zero-row count (boundary frames)
    zero_rows = np.all(all_feat == 0, axis=1).sum()
    print(f"Zero-padded frames (boundaries + last): {zero_rows}")


def main():
    parser = argparse.ArgumentParser(
        description="Strategy C: Extract temporal derivative features from 3D keypoints",
    )
    parser.add_argument(
        "--keypoints", type=str, required=True,
        help="Path to keypoints_22_3d.npz (MAMMAL output)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for temporal_features.npz",
    )
    parser.add_argument(
        "--fps", type=float, default=20.0,
        help="Frame rate for scaling temporal derivatives (default: 20)",
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", default=[5, 10, 20],
        help="Sliding window sizes in frames (default: 5 10 20)",
    )
    parser.add_argument(
        "--total-frames", type=int, default=3600,
        help="Total number of frames in dataset (default: 3600)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load keypoints
    print(f"Loading keypoints: {args.keypoints}")
    kp_data = np.load(args.keypoints, allow_pickle=True)
    kp_all = kp_data["keypoints"]  # (T, 22, 3) in MAMMAL mm space
    print(f"  Raw shape: {kp_all.shape}")

    # Filter valid frames (exclude session boundary jumps)
    all_frames = np.arange(min(args.total_frames, len(kp_all)))
    valid_mask = ~np.isin(all_frames, list(FRAME_JUMPS))
    valid_frames = all_frames[valid_mask]
    keypoints = kp_all[valid_frames]  # (N, 22, 3)
    print(f"  Valid frames: {len(valid_frames)} / {len(all_frames)} "
          f"(excluded {len(all_frames) - len(valid_frames)} jump frames)")

    # Extract features (in MAMMAL mm space — temporal derivatives are
    # coordinate-system agnostic for relative measures like velocity/speed)
    result = extract_temporal_features(
        keypoints, valid_frames, fps=args.fps, windows=args.windows,
    )

    # Print summary
    print_summary(result)

    # Save
    out_path = output_dir / "temporal_features.npz"
    np.savez_compressed(
        out_path,
        # Feature groups
        centroid_features=result["centroid_features"],
        bodypart_delta_features=result["bodypart_delta_features"],
        rigid_nonrigid_features=result["rigid_nonrigid_features"],
        sliding_window_features=result["sliding_window_features"],
        # Concatenated
        all_features=result["all_features"],
        # Metadata
        valid_frames=result["valid_frames"],
        feature_dims=result["feature_dims"],
        feature_names=result["feature_names"],
        fps=np.array(args.fps),
        window_sizes=np.array(args.windows),
    )
    print(f"\nSaved: {out_path}")
    print(f"  Feature matrix: {result['all_features'].shape}")

    # Also save a metadata JSON for quick inspection
    meta = {
        "experiment": "Strategy_C_Temporal_Derivatives",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "keypoints_source": str(args.keypoints),
        "fps": args.fps,
        "window_sizes": args.windows,
        "n_frames": int(len(valid_frames)),
        "total_dims": int(result["all_features"].shape[1]),
        "feature_groups": {
            str(name): int(d)
            for name, d in zip(result["feature_names"], result["feature_dims"])
        },
        "stats": {
            "mean": float(result["all_features"].mean()),
            "std": float(result["all_features"].std()),
            "nan_count": int(np.isnan(result["all_features"]).sum()),
            "inf_count": int(np.isinf(result["all_features"]).sum()),
        },
    }
    meta_path = output_dir / "temporal_features_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
