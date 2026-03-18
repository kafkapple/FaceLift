"""Body-Part Gaussian Grouping: structured dense features from 3DGS.

Assigns pruned Gaussians to nearest MAMMAL skeleton joints, then computes
per-body-part statistics (shape, appearance, density, motion).

This bridges sparse keypoints (body part identity) with dense 3DGS
(local shape/appearance), capturing information that raw keypoints miss:
body part shape, surface deformation, local density patterns.

Literature basis:
- Skeleton-guided point cloud features (PointNet++ with joint-centric sampling)
- Graph Attention Networks on body parts (Veličković et al., 2018)
- Body-part-aware 3D representations for action recognition

Usage:
    python -m mouse_extensions.behavior.extract_bodypart_features
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


FRAME_JUMPS = {1178,1179,1180,1181,1182,2358,2359,2360,2361,2362,3538,3539,3540,3541,3542}

# MAMMAL bone lengths (approximate, for adaptive radius)
# Connected joints for radius estimation
BONE_PAIRS = [
    (2, 0), (2, 1), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
    (3, 11), (11, 10), (10, 8), (8, 9),
    (3, 15), (15, 14), (14, 12), (12, 13),
    (4, 18), (18, 17), (17, 16),
    (4, 21), (21, 20), (20, 19),
]


def compute_bodypart_features(gaussians_xyz, gaussians_opacity, gaussians_scale,
                              keypoints_3d, n_joints=22, top_k=None):
    """Compute per-body-part Gaussian statistics.

    Args:
        gaussians_xyz: (N, 3) Gaussian positions
        gaussians_opacity: (N,) opacity values
        gaussians_scale: (N, 3) scale values
        keypoints_3d: (22, 3) joint positions
        n_joints: number of joints
        top_k: if set, use only top-k nearest Gaussians per joint

    Returns:
        feature_vector: (n_joints * n_features_per_part,) flat feature vector
    """
    N = len(gaussians_xyz)

    # Soft assignment: compute distances from each Gaussian to each joint
    dists = cdist(gaussians_xyz, keypoints_3d)  # (N, 22)

    # Hard assignment: each Gaussian → nearest joint
    assignments = dists.argmin(axis=1)  # (N,)

    features = []
    for j in range(n_joints):
        mask = assignments == j

        if top_k is not None and mask.sum() > top_k:
            # Keep only top_k nearest Gaussians
            joint_dists = dists[mask, j]
            top_indices = np.argsort(joint_dists)[:top_k]
            local_xyz = gaussians_xyz[mask][top_indices]
            local_opacity = gaussians_opacity[mask][top_indices]
            local_scale = gaussians_scale[mask][top_indices]
            count = top_k
        elif mask.sum() > 0:
            local_xyz = gaussians_xyz[mask]
            local_opacity = gaussians_opacity[mask]
            local_scale = gaussians_scale[mask]
            count = mask.sum()
        else:
            # No Gaussians assigned to this joint
            features.extend([0.0] * 14)
            continue

        # Relative positions (body-part local coordinates)
        rel_xyz = local_xyz - keypoints_3d[j]

        # Shape descriptors: eigenvalues of position covariance
        if count >= 4:
            cov = np.cov(rel_xyz.T)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending
        else:
            eigvals = np.array([0.0, 0.0, 0.0])

        # Per-body-part features (14 per joint)
        part_feat = [
            float(np.linalg.norm(rel_xyz, axis=1).mean()),  # mean distance from joint
            float(eigvals[0]),     # largest eigenvalue (elongation)
            float(eigvals[1]),     # middle eigenvalue
            float(eigvals[2]),     # smallest eigenvalue (flatness)
            float(eigvals[0] / (eigvals[2] + 1e-10)),  # anisotropy ratio
            float(local_opacity.mean()),    # mean opacity
            float(local_opacity.std()),     # opacity variation
            float(local_scale.mean()),      # mean scale
            float(local_scale.std()),       # scale variation
            float(local_scale[:, 0].mean() / (local_scale[:, 2].mean() + 1e-10)),  # scale anisotropy
            float(count),                    # Gaussian count (density)
            float(rel_xyz[:, 0].std()),     # spread in x
            float(rel_xyz[:, 1].std()),     # spread in y
            float(rel_xyz[:, 2].std()),     # spread in z
        ]
        features.extend(part_feat)

    return np.array(features, dtype=np.float32)


def compute_scene_flow_features(gaussians_xyz_t, gaussians_xyz_t1,
                                keypoints_t, keypoints_t1, n_joints=22):
    """Compute per-body-part scene flow (displacement) features.

    Uses nearest-neighbor correspondence between frames.

    Args:
        gaussians_xyz_t: (N, 3) positions at frame t
        gaussians_xyz_t1: (M, 3) positions at frame t+1
        keypoints_t: (22, 3) joints at frame t
        keypoints_t1: (22, 3) joints at frame t+1

    Returns:
        flow_features: (n_joints * 6,) per-part velocity stats
    """
    # Assign Gaussians at frame t to nearest joint
    dists = cdist(gaussians_xyz_t, keypoints_t)
    assignments = dists.argmin(axis=1)

    # Find nearest neighbor in frame t+1 using KDTree (memory efficient)
    from scipy.spatial import cKDTree
    tree = cKDTree(gaussians_xyz_t1)
    nn_dists, nn_idx = tree.query(gaussians_xyz_t, k=1)
    displacements = gaussians_xyz_t1[nn_idx] - gaussians_xyz_t  # (N, 3)

    # Per-joint displacement (relative to joint motion)
    joint_motion = keypoints_t1 - keypoints_t  # (22, 3)

    features = []
    for j in range(n_joints):
        mask = assignments == j
        if mask.sum() < 2:
            features.extend([0.0] * 6)
            continue

        # Relative displacement (local deformation, removing rigid joint motion)
        local_disp = displacements[mask] - joint_motion[j]

        features.extend([
            float(np.linalg.norm(local_disp, axis=1).mean()),  # mean local speed
            float(np.linalg.norm(local_disp, axis=1).std()),   # speed variation
            float(local_disp[:, 0].mean()),  # mean deformation x
            float(local_disp[:, 1].mean()),  # mean deformation y
            float(local_disp[:, 2].mean()),  # mean deformation z
            float(np.linalg.norm(local_disp.mean(0))),  # coherent motion magnitude
        ])

    return np.array(features, dtype=np.float32)


def temporal_metrics(labels, fps=20.0):
    """Compute temporal metrics."""
    valid = labels >= 0
    lab = labels[valid]
    changes = np.where(np.diff(lab) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(lab)]])) / fps
    tc = 1.0 - len(changes) / (len(lab) - 1) if len(lab) > 1 else 1.0
    short_pct = float((bouts < 0.3).mean() * 100) if len(bouts) > 0 else 0
    return {
        "bout_mean": round(float(bouts.mean()), 3),
        "tc": round(float(tc), 4),
        "short_pct": round(short_pct, 1),
    }


def main():
    from mouse_extensions.behavior.paths import (
        RESULTS_DIR, FEATURES_DIR, GPU03_KEYPOINTS, ensure_dirs,
    )
    ensure_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", default=str(FEATURES_DIR / "gaussians_raw"))
    parser.add_argument("--pruning_level", type=int, default=12500)
    parser.add_argument("--output_dir", default=str(RESULTS_DIR / "bodypart_features"))
    parser.add_argument("--top_k_per_joint", type=int, default=200,
                        help="Max Gaussians per joint (0=all)")
    parser.add_argument("--compute_flow", action="store_true",
                        help="Compute scene flow features (slower)")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load keypoints and transform to GS-LRM normalized space
    from mouse_extensions.behavior.keypoint_projection import mammal_to_gslrm
    kp_all_mm = np.load(GPU03_KEYPOINTS, allow_pickle=True)["keypoints"]
    # CRITICAL: Gaussians are in GS-LRM normalized space, keypoints in MAMMAL mm
    # Must transform keypoints to match Gaussian coordinate system
    kp_all = np.stack([mammal_to_gslrm(kp_all_mm[i]) for i in range(len(kp_all_mm))])
    print(f"Keypoints transformed: MAMMAL mm → GS-LRM normalized")
    all_valid = sorted(set(range(3600)) - FRAME_JUMPS)

    # Find NPZ files
    frame_npz = {}
    for p in sorted(npz_dir.glob("*.npz")):
        fi = int(p.stem)
        if fi not in FRAME_JUMPS:
            frame_npz[fi] = p
    frames = sorted(f for f in frame_npz if f in set(all_valid))
    N = len(frames)
    print(f"Frames: {N}, Pruning: {args.pruning_level}, TopK/joint: {args.top_k_per_joint}")

    # Extract body-part features
    t0 = time.time()
    static_features = []
    flow_features = []
    prev_xyz = None

    for i, fi in enumerate(frames):
        try:
            data = np.load(frame_npz[fi])
            xyz = data["xyz"].astype(np.float32)
            opacity = data["opacity"].astype(np.float32).flatten()
            scale = data["scale"].astype(np.float32)
        except Exception:
            # Bad NPZ, use zeros
            static_features.append(np.zeros(22 * 14, dtype=np.float32))
            if args.compute_flow:
                flow_features.append(np.zeros(22 * 6, dtype=np.float32))
            prev_xyz = None
            continue

        # Opacity-based pruning
        sorted_idx = np.argsort(opacity)
        top_idx = sorted_idx[-min(args.pruning_level, len(opacity)):]
        xyz_pruned = xyz[top_idx]
        opacity_pruned = opacity[top_idx]
        scale_pruned = scale[top_idx] if scale.ndim == 2 else scale[top_idx].reshape(-1, 3)

        # Body-part features
        kp = kp_all[fi]
        top_k = args.top_k_per_joint if args.top_k_per_joint > 0 else None
        feat = compute_bodypart_features(xyz_pruned, opacity_pruned, scale_pruned,
                                         kp, top_k=top_k)
        static_features.append(feat)

        # Scene flow features (frame-to-frame displacement)
        if args.compute_flow and prev_xyz is not None and i > 0:
            kp_prev = kp_all[frames[i - 1]]
            flow = compute_scene_flow_features(prev_xyz, xyz_pruned, kp_prev, kp)
            flow_features.append(flow)
        elif args.compute_flow:
            flow_features.append(np.zeros(22 * 6, dtype=np.float32))

        prev_xyz = xyz_pruned

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N} ({time.time()-t0:.0f}s)")

    print(f"Feature extraction done ({time.time()-t0:.0f}s)")

    static_features = np.stack(static_features)  # (N, 22*14=308)
    print(f"Static features shape: {static_features.shape}")

    # Combine with flow if available
    if args.compute_flow and flow_features:
        flow_features = np.stack(flow_features)
        print(f"Flow features shape: {flow_features.shape}")
        # Align lengths (flow has N-1 or N entries)
        min_len = min(len(static_features), len(flow_features))
        all_features = np.concatenate([
            static_features[:min_len],
            flow_features[:min_len],
        ], axis=1)
        print(f"Combined features shape: {all_features.shape}")
    else:
        all_features = static_features

    # Save raw features
    np.save(output_dir / "bodypart_features.npy", all_features)

    # Also load SP_RawPCA for comparison (use original mm keypoints for PCA)
    kp_frames = kp_all_mm[np.array(frames)]
    kp_c = kp_frames - kp_frames.mean(1, keepdims=True)
    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)

    # Clustering comparison
    feat_s = StandardScaler().fit_transform(all_features)
    # Remove NaN/Inf
    feat_s = np.nan_to_num(feat_s, nan=0.0, posinf=0.0, neginf=0.0)
    feat_pca = PCA(n_components=min(30, feat_s.shape[1])).fit_transform(feat_s)

    print(f"\n{'='*90}")
    print("BODY-PART GAUSSIAN FEATURES — Clustering Comparison")
    print(f"{'='*90}")
    print(f"{'Pipeline':<35} {'K':>3} | {'Sil':>7} {'CH':>7} {'DB':>6} | {'Bout':>6} {'TC':>6}")
    print("-" * 80)

    results = {}
    for name, feat in [("SP_RawPCA", kp_pca), ("DN_BodyPart", feat_pca)]:
        for k in [4, 8]:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat)
            sil = silhouette_score(feat, labels, sample_size=min(5000, len(feat)))
            ch = calinski_harabasz_score(feat, labels)
            db = davies_bouldin_score(feat, labels)
            tm = temporal_metrics(labels)
            key = f"{name}_KMeans_K{k}"
            results[key] = {
                "silhouette": round(float(sil), 4),
                "calinski_harabasz": round(float(ch), 1),
                "davies_bouldin": round(float(db), 3),
                **tm,
            }
            print(f"  {key:<35} {k:>3} | {sil:>7.3f} {ch:>7.0f} {db:>6.2f} | "
                  f"{tm['bout_mean']:>6.2f} {tm['tc']:>6.3f}")

    # Hybrid: SP_RawPCA + DN_BodyPart
    hybrid = np.concatenate([kp_pca, feat_pca], axis=1)
    for k in [4, 8]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(hybrid)
        sil = silhouette_score(hybrid, labels, sample_size=min(5000, len(hybrid)))
        ch = calinski_harabasz_score(hybrid, labels)
        db = davies_bouldin_score(hybrid, labels)
        tm = temporal_metrics(labels)
        key = f"HY_BodyPart_KMeans_K{k}"
        results[key] = {
            "silhouette": round(float(sil), 4),
            "calinski_harabasz": round(float(ch), 1),
            "davies_bouldin": round(float(db), 3),
            **tm,
        }
        print(f"  {key:<35} {k:>3} | {sil:>7.3f} {ch:>7.0f} {db:>6.2f} | "
              f"{tm['bout_mean']:>6.2f} {tm['tc']:>6.3f}")

    # Save results
    with open(output_dir / "bodypart_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_dir}")


if __name__ == "__main__":
    main()
