"""Extract per-body-part covariance eigenvalue features from raw Gaussian NPZ.

Computes individual Gaussian covariance matrices (from scale + rotation),
extracts eigenvalues, and aggregates per skeleton joint. Also computes
temporal features (Jacobian, sliding window).

Unlike extract_bodypart_features.py which uses aggregate statistics of
raw scale values (constant due to renderer convergence), this script
extracts the GEOMETRIC SHAPE of individual Gaussians via their full
covariance matrix Sigma = R @ diag(exp(s))^2 @ R^T.

Usage on gpu03:
    python -m mouse_extensions.behavior.extract_covariance_features
    python -m mouse_extensions.behavior.extract_covariance_features --n_filter 2  # N>=2 visibility

Output:
    - covariance_static.npy:   (N_frames, 22*7) per-joint shape features
    - covariance_temporal.npy: (N_frames, 22*9) temporal change features
    - covariance_meta.json:    metadata + quality report
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


FRAME_JUMPS = {1178, 1179, 1180, 1181, 1182, 2358, 2359, 2360, 2361, 2362,
               3538, 3539, 3540, 3541, 3542}


def quat_to_rotation_matrix(q):
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix.

    Handles both single quaternion (4,) and batch (N, 4).
    """
    if q.ndim == 1:
        q = q[np.newaxis, :]

    # Normalize
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-10)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.zeros((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def compute_gaussian_covariance(scale_raw, rotation_raw):
    """Compute covariance matrices from raw (pre-activation) Gaussian params.

    Args:
        scale_raw: (N, 3) log-space scale (needs exp)
        rotation_raw: (N, 4) unnormalized quaternion (needs normalize)

    Returns:
        eigenvalues: (N, 3) sorted descending per Gaussian
    """
    # Activate: exp for scale, normalize for rotation
    scale = np.exp(scale_raw.astype(np.float32))
    R = quat_to_rotation_matrix(rotation_raw.astype(np.float32))  # (N, 3, 3)

    # Covariance: Sigma = R @ diag(s^2) @ R^T
    # Eigenvalues of Sigma = s^2 (rotated), but actual eigenvalues
    # of R @ diag(s^2) @ R^T are just s^2 (rotation doesn't change eigenvalues)
    # However, we still compute via full matrix for correctness with
    # potential numerical issues.
    #
    # Actually: eigenvalues of R @ D @ R^T = eigenvalues of D (similarity transform)
    # So eigenvalues = scale^2 directly.
    # BUT: the interesting part is the SPATIAL distribution of these shapes
    # when aggregated per joint — how the orientations (R) relate to each other.
    #
    # For per-Gaussian features: lambda_i = scale_i^2
    # For per-joint features: we need to also capture orientation diversity.

    s2 = scale ** 2  # (N, 3) — eigenvalues of individual Gaussian covariance

    # Sort descending per Gaussian
    eigenvalues = np.sort(s2, axis=1)[:, ::-1]  # (N, 3) lambda1 >= lambda2 >= lambda3

    return eigenvalues, scale, R


def compute_per_joint_features(xyz, eigenvalues, scale, R, opacity,
                               keypoints, n_joints=22, pruning_level=12500,
                               vis_mask=None):
    """Compute per-joint covariance-based features.

    Per joint (7d):
        - mean_log_volume (1d): mean(log(lambda1 * lambda2 * lambda3))
        - mean_eigenvalues (3d): mean(lambda1), mean(lambda2), mean(lambda3)
        - mean_anisotropy (1d): mean(lambda1 / lambda3)
        - orientation_diversity (1d): entropy of principal axis directions
        - count (1d): number of Gaussians assigned

    Total: 22 * 7 = 154d

    Returns:
        features: (22 * 7,) flat feature vector
        assignments: (N,) per-Gaussian joint assignment
    """
    N = len(xyz)

    if vis_mask is not None:
        # N>=2 multi-view visibility filter (preferred over opacity pruning)
        idx = np.where(vis_mask)[0]
        xyz = xyz[idx]
        eigenvalues = eigenvalues[idx]
        scale = scale[idx]
        R = R[idx]
    else:
        # Legacy: opacity-based pruning (foreground focus)
        op = opacity.flatten()
        if N > pruning_level:
            idx = np.argsort(op)[-pruning_level:]
            xyz = xyz[idx]
            eigenvalues = eigenvalues[idx]
            scale = scale[idx]
            R = R[idx]

    # NN hard assignment
    dists = cdist(xyz, keypoints)  # (N, 22)
    assignments = dists.argmin(axis=1)

    features = []
    for j in range(n_joints):
        mask = assignments == j
        count = mask.sum()

        if count < 2:
            features.extend([0.0] * 7)
            continue

        ev = eigenvalues[mask]  # (count, 3)
        local_R = R[mask]       # (count, 3, 3)

        # Log volume: log(product of eigenvalues) = sum of log(eigenvalues)
        log_vol = np.log(ev + 1e-10).sum(axis=1)  # (count,)
        mean_log_vol = float(log_vol.mean())

        # Mean eigenvalues
        mean_ev = ev.mean(axis=0)  # (3,)

        # Anisotropy: lambda1 / lambda3
        anisotropy = (ev[:, 0] / (ev[:, 2] + 1e-10))
        mean_anisotropy = float(anisotropy.mean())

        # Orientation diversity: dispersion of principal axes
        # Principal axis = first column of R (direction of lambda1)
        principal_axes = local_R[:, :, 0]  # (count, 3)
        # Use angular variance: 1 - |mean(normalized_axes)|
        mean_axis = principal_axes.mean(axis=0)
        orientation_div = float(1.0 - np.linalg.norm(mean_axis) / (count + 1e-10) * count)
        orientation_div = max(0.0, min(1.0, orientation_div))

        part_feat = [
            mean_log_vol,
            float(mean_ev[0]), float(mean_ev[1]), float(mean_ev[2]),
            mean_anisotropy,
            orientation_div,
            float(count),
        ]
        features.extend(part_feat)

    return np.array(features, dtype=np.float32), assignments


def compute_temporal_features(static_features_seq, fps=20.0, window_sizes=(5, 10, 20)):
    """Compute temporal features from a sequence of static features.

    Per joint (9d):
        - delta_eigenvalues (3d): frame-to-frame eigenvalue change (Jacobian)
        - sliding_window_std (3d): std of eigenvalues over 10-frame window
        - sliding_window_delta_std (3d): std of eigenvalue deltas over 10-frame window

    Total: 22 * 9 = 198d

    Args:
        static_features_seq: (T, 22*7) static features over time
        fps: frame rate
        window_sizes: tuple of window sizes to use

    Returns:
        temporal_features: (T, 22*9) temporal features
    """
    T = len(static_features_seq)
    n_joints = 22
    dims_per_joint = 7
    temporal_dims_per_joint = 9

    # Reshape to (T, 22, 7)
    static = static_features_seq.reshape(T, n_joints, dims_per_joint)

    # Extract eigenvalue components: indices 1,2,3 per joint
    eigenvals = static[:, :, 1:4]  # (T, 22, 3)

    temporal = np.zeros((T, n_joints, temporal_dims_per_joint), dtype=np.float32)

    # 1. Frame-to-frame delta (Jacobian) — 3d per joint
    delta = np.zeros_like(eigenvals)
    delta[1:] = eigenvals[1:] - eigenvals[:-1]
    delta[0] = delta[1]  # pad first frame
    temporal[:, :, 0:3] = delta * fps  # scale by fps for rate

    # 2. Sliding window std of eigenvalues — 3d per joint
    w = window_sizes[1] if len(window_sizes) > 1 else 10  # default 10 frames
    half_w = w // 2
    for t in range(T):
        start = max(0, t - half_w)
        end = min(T, t + half_w + 1)
        window = eigenvals[start:end]  # (window_len, 22, 3)
        temporal[t, :, 3:6] = window.std(axis=0)

    # 3. Sliding window std of deltas — 3d per joint (measures "jerkiness")
    for t in range(T):
        start = max(0, t - half_w)
        end = min(T, t + half_w + 1)
        window_delta = delta[start:end]
        temporal[t, :, 6:9] = window_delta.std(axis=0)

    return temporal.reshape(T, n_joints * temporal_dims_per_joint)


def main():
    from mouse_extensions.behavior.paths import FEATURES_DIR, GPU03_KEYPOINTS, ensure_dirs
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Extract covariance eigenvalue features")
    parser.add_argument("--npz_dir", default=str(FEATURES_DIR / "gaussians_raw"))
    parser.add_argument("--pruning_level", type=int, default=12500)
    parser.add_argument("--n_filter", type=int, default=0,
                        help="Multi-view visibility threshold (0=disabled, 2=recommended)")
    parser.add_argument("--m5_dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5t",
                        help="M5t frame directory (for visibility filter)")
    parser.add_argument("--output_dir", default=str(FEATURES_DIR / "covariance"))
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    m5_dir = Path(args.m5_dir)
    output_dir = Path(args.output_dir)
    if args.n_filter > 0:
        output_dir = Path(str(output_dir) + f"_n{args.n_filter}")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_vis_filter = args.n_filter > 0
    if use_vis_filter:
        from mouse_extensions.behavior.multiview_visibility_filter import compute_visibility_counts
        print(f"Visibility filter: N>={args.n_filter} (multi-view)")
    else:
        print(f"Visibility filter: disabled (opacity pruning, level={args.pruning_level})")

    # Load keypoints → GS-LRM space
    from mouse_extensions.coordinate_utils import mammal_to_gslrm, assert_mammal_space, assert_gslrm_space
    kp_all_mm = np.load(GPU03_KEYPOINTS, allow_pickle=True)["keypoints"]
    assert_mammal_space(kp_all_mm, "raw keypoints")
    kp_all = np.stack([mammal_to_gslrm(kp_all_mm[i]) for i in range(len(kp_all_mm))])
    assert_gslrm_space(kp_all, "transformed keypoints")
    print(f"Keypoints: MAMMAL mm -> GS-LRM normalized")

    all_valid = sorted(set(range(3600)) - FRAME_JUMPS)

    # Find NPZ files
    frame_npz = {}
    for p in sorted(npz_dir.glob("*.npz")):
        fi = int(p.stem)
        if fi not in FRAME_JUMPS:
            frame_npz[fi] = p
    frames = sorted(f for f in frame_npz if f in set(all_valid))
    N = len(frames)
    print(f"Frames: {N}, Pruning: {args.pruning_level}")

    # Extract static features
    t0 = time.time()
    static_features = []
    quality_report = {"constant_dims": 0, "nan_count": 0, "inf_count": 0}

    for i, fi in enumerate(frames):
        try:
            data = np.load(frame_npz[fi])
            xyz = data["xyz"].astype(np.float32)
            opacity = data["opacity"].astype(np.float32).flatten()
            scale_raw = data["scale"].astype(np.float32)
            rotation_raw = data["rotation"].astype(np.float32)
        except Exception as e:
            static_features.append(np.zeros(22 * 7, dtype=np.float32))
            continue

        # Compute covariance eigenvalues
        eigenvalues, scale, R = compute_gaussian_covariance(scale_raw, rotation_raw)

        # Apply N>=2 multi-view visibility filter
        vis_mask = None
        if use_vis_filter:
            frame_dir = str(m5_dir / f"{fi:06d}")
            if Path(frame_dir).exists():
                counts = compute_visibility_counts(xyz, frame_dir)
                vis_mask = counts >= args.n_filter
            else:
                vis_mask = opacity.flatten() > 0.5  # fallback

        # Compute per-joint features
        kp = kp_all[fi]
        feat, _ = compute_per_joint_features(
            xyz, eigenvalues, scale, R, opacity, kp,
            pruning_level=args.pruning_level,
            vis_mask=vis_mask,
        )
        static_features.append(feat)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{N} ({elapsed:.0f}s)")

    static_features = np.stack(static_features)  # (N, 154)
    print(f"Static features: {static_features.shape} ({time.time()-t0:.0f}s)")

    # Quality check
    stds = static_features.std(0)
    const_dims = int((stds < 0.01).sum())
    nan_count = int(np.isnan(static_features).sum())
    inf_count = int(np.isinf(static_features).sum())
    print(f"Quality: {const_dims}/{static_features.shape[1]} constant, {nan_count} NaN, {inf_count} Inf")

    # Replace NaN/Inf
    static_features = np.nan_to_num(static_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute temporal features
    print("Computing temporal features...")
    temporal_features = compute_temporal_features(static_features)
    print(f"Temporal features: {temporal_features.shape}")
    temporal_stds = temporal_features.std(0)
    temporal_const = int((temporal_stds < 0.01).sum())
    print(f"Temporal quality: {temporal_const}/{temporal_features.shape[1]} constant")

    # Save
    np.save(output_dir / "covariance_static.npy", static_features)
    np.save(output_dir / "covariance_temporal.npy", temporal_features)

    # Per-dim-type analysis
    dim_labels = ["log_vol", "ev1", "ev2", "ev3", "aniso", "orient_div", "count"]
    dim_report = {}
    for d, name in enumerate(dim_labels):
        indices = list(range(d, 22 * 7, 7))
        dim_stds = stds[indices]
        dim_report[name] = {
            "const_joints": int((dim_stds < 0.01).sum()),
            "min_std": float(dim_stds.min()),
            "max_std": float(dim_stds.max()),
            "mean_std": float(dim_stds.mean()),
        }
        status = "✅" if (dim_stds < 0.01).sum() == 0 else f"⚠️ {(dim_stds < 0.01).sum()}/22"
        print(f"  {name:>12}: {status} (std range [{dim_stds.min():.6f}, {dim_stds.max():.4f}])")

    meta = {
        "shape_static": list(static_features.shape),
        "shape_temporal": list(temporal_features.shape),
        "constant_dims_static": const_dims,
        "constant_dims_temporal": temporal_const,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "dim_labels_static": dim_labels,
        "dims_per_joint_static": 7,
        "dims_per_joint_temporal": 9,
        "temporal_labels": ["delta_ev1", "delta_ev2", "delta_ev3",
                           "window_std_ev1", "window_std_ev2", "window_std_ev3",
                           "window_delta_std_ev1", "window_delta_std_ev2", "window_delta_std_ev3"],
        "dim_report": dim_report,
        "pruning_level": args.pruning_level,
        "n_filter": args.n_filter,
        "visibility_filter": use_vis_filter,
        "m5_dir": str(m5_dir) if use_vis_filter else None,
        "n_frames": N,
        "extraction_time_s": round(time.time() - t0, 1),
    }
    with open(output_dir / "covariance_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {output_dir}")
    print(f"  covariance_static.npy:   {static_features.shape}")
    print(f"  covariance_temporal.npy: {temporal_features.shape}")
    print(f"  covariance_meta.json")


if __name__ == "__main__":
    main()
