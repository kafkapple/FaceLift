"""Phase 3A: Sparse Keypoint Ablation Study.

Runs SUBTLE + generic (UMAP+HDBSCAN) clustering on MAMMAL 22-joint 3D keypoints
with progressive joint ablation (22 → 15 → 7 → 4).

Usage:
    python -m mouse_extensions.behavior.run_sparse_ablation \
        --keypoints outputs/clustering/sparse/keypoints_22_3d.npz \
        --output-dir outputs/clustering/sparse/results
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Frame jump indices to exclude (±2 frames around each)
FRAME_JUMP_INDICES = [1180, 2360, 3540]
FRAME_JUMP_MARGIN = 2

# Joint ablation groups (indices into MAMMAL 22-joint skeleton)
JOINT_GROUPS = {
    "full_22": list(range(22)),
    "core_15": [
        0, 1, 2, 3,     # L_ear, R_ear, nose, neck
        4, 5, 7,         # body_middle, tail_root, tail_end
        8, 12,           # L_paw, R_paw
        16, 19,          # L_foot, R_foot
        18, 21,          # L_hip, R_hip
        11, 15,          # L_shoulder, R_shoulder
    ],
    "minimal_7": [
        2,               # nose
        0, 1,            # L_ear, R_ear
        3,               # neck
        18, 21,          # L_hip, R_hip
        5,               # tail_root
    ],
    "ultra_4": [
        2,               # nose
        3,               # neck
        4,               # body_middle
        5,               # tail_root
    ],
}


def mask_frame_jumps(keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove frames near known frame jumps.

    Returns:
        masked_keypoints: (T', K, 3) with jump frames removed
        valid_mask: (T,) boolean mask of kept frames
    """
    T = keypoints.shape[0]
    valid_mask = np.ones(T, dtype=bool)
    for idx in FRAME_JUMP_INDICES:
        start = max(0, idx - FRAME_JUMP_MARGIN)
        end = min(T, idx + FRAME_JUMP_MARGIN + 1)
        valid_mask[start:end] = False
    return keypoints[valid_mask], valid_mask


def compute_metrics(labels: np.ndarray, features: np.ndarray) -> dict:
    """Compute clustering quality metrics."""
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    valid = labels >= 0
    n_valid = valid.sum()
    n_clusters = len(set(labels[valid]))

    if n_clusters < 2 or n_valid < 10:
        return {
            "n_clusters": int(n_clusters),
            "n_valid_frames": int(n_valid),
            "n_noise": int((~valid).sum()),
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }

    feat_valid = features[valid]
    lab_valid = labels[valid]

    sample_size = min(5000, n_valid)
    sil = silhouette_score(feat_valid, lab_valid, sample_size=sample_size)
    ch = calinski_harabasz_score(feat_valid, lab_valid)
    db = davies_bouldin_score(feat_valid, lab_valid)

    return {
        "n_clusters": int(n_clusters),
        "n_valid_frames": int(n_valid),
        "n_noise": int((~valid).sum()),
        "silhouette": round(float(sil), 4),
        "calinski_harabasz": round(float(ch), 2),
        "davies_bouldin": round(float(db), 4),
    }


def compute_temporal_metrics(labels: np.ndarray, fps: float = 20.0) -> dict:
    """Compute temporal consistency metrics."""
    valid = labels >= 0
    lab = labels[valid]

    if len(lab) < 2:
        return {"bout_mean_sec": 0, "bout_median_sec": 0, "transition_rate": 0}

    # Bout duration
    changes = np.where(np.diff(lab) != 0)[0]
    bout_lengths = np.diff(np.concatenate([[0], changes + 1, [len(lab)]]))
    bout_durations = bout_lengths / fps

    # Transition rate
    n_transitions = len(changes)
    transition_rate = n_transitions / (len(lab) / fps)  # transitions per second

    return {
        "bout_mean_sec": round(float(bout_durations.mean()), 3),
        "bout_median_sec": round(float(np.median(bout_durations)), 3),
        "bout_std_sec": round(float(bout_durations.std()), 3),
        "n_bouts": int(len(bout_durations)),
        "transition_rate_per_sec": round(float(transition_rate), 3),
    }


def run_bsoid_clustering(keypoints: np.ndarray, fps: float = 20.0) -> dict:
    """Run B-SOiD clustering on 3D keypoints (extended from 2D-only)."""
    try:
        from behavior_lab.models.discovery.bsoid import BSOiD

        model = BSOiD(fps=int(fps), min_cluster_size=30)
        result = model.fit_predict(keypoints)

        labels = result.labels if hasattr(result, "labels") else result["labels"]
        embeddings = (
            result.embeddings
            if hasattr(result, "embeddings")
            else result.get("embedding_2d")
        )
        features_out = (
            result.features
            if hasattr(result, "features")
            else result.get("features")
        )

        feat_for_metrics = keypoints[: len(labels)].reshape(len(labels), -1)
        metrics = compute_metrics(labels, feat_for_metrics)
        temporal = compute_temporal_metrics(labels, fps)

        return {
            "method": "B-SOiD",
            "labels": labels,
            "embeddings": embeddings,
            "metrics": {**metrics, **temporal},
        }
    except Exception as e:
        print(f"  B-SOiD failed: {e}")
        return {"method": "B-SOiD", "labels": None, "error": str(e)}


def run_subtle_clustering(keypoints: np.ndarray, fps: float = 20.0) -> dict:
    """Run SUBTLE clustering on 3D keypoints."""
    try:
        from behavior_lab.models.discovery.subtle_wrapper import SUBTLE

        model = SUBTLE(fps=fps, n_train_frames=min(120000, len(keypoints)))
        result = model.fit([keypoints])

        labels = result.labels if hasattr(result, "labels") else result["labels"]
        embeddings = (
            result.embeddings
            if hasattr(result, "embeddings")
            else result.get("embedding_2d")
        )

        # Flatten keypoints for metric computation
        features = keypoints[: len(labels)].reshape(len(labels), -1)

        metrics = compute_metrics(labels, features)
        temporal = compute_temporal_metrics(labels, fps)

        return {
            "method": "SUBTLE",
            "labels": labels,
            "embeddings": embeddings,
            "metrics": {**metrics, **temporal},
        }
    except Exception as e:
        print(f"  SUBTLE failed: {e}")
        return {"method": "SUBTLE", "labels": None, "error": str(e)}


def run_generic_clustering(
    keypoints: np.ndarray, n_clusters: int = 8
) -> dict:
    """Run generic PCA+UMAP+HDBSCAN clustering."""
    try:
        from behavior_lab.models.discovery.clustering import cluster_features

        features = keypoints.reshape(len(keypoints), -1)
        result = cluster_features(features, n_clusters=n_clusters)

        labels = result["labels"] if isinstance(result, dict) else result.labels
        embeddings = (
            result.get("embedding_2d") if isinstance(result, dict) else result.embeddings
        )

        metrics = compute_metrics(labels, features)
        temporal = compute_temporal_metrics(labels)

        return {
            "method": "Generic_UMAP_HDBSCAN",
            "labels": labels,
            "embeddings": embeddings,
            "metrics": {**metrics, **temporal},
        }
    except Exception as e:
        print(f"  Generic clustering failed: {e}")
        return {"method": "Generic_UMAP_HDBSCAN", "labels": None, "error": str(e)}


def save_visualizations(
    results: dict,
    keypoints: np.ndarray,
    joint_names: list[str],
    output_dir: Path,
    group_name: str,
) -> None:
    """Save UMAP plots and ethograms."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for method_result in results:
            method = method_result["method"]
            labels = method_result.get("labels")
            embeddings = method_result.get("embeddings")

            if labels is None:
                continue

            # UMAP scatter
            if embeddings is not None and len(embeddings.shape) == 2:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                valid = labels >= 0
                scatter = ax.scatter(
                    embeddings[valid, 0],
                    embeddings[valid, 1],
                    c=labels[valid],
                    cmap="tab20",
                    s=1,
                    alpha=0.5,
                )
                ax.set_title(f"{group_name} | {method} | UMAP")
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                plt.colorbar(scatter, ax=ax, label="Cluster")
                fig.savefig(
                    output_dir / f"{group_name}_{method}_umap.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

            # Ethogram (temporal raster)
            fig, ax = plt.subplots(1, 1, figsize=(16, 2))
            ax.imshow(
                labels[np.newaxis, :],
                aspect="auto",
                cmap="tab20",
                interpolation="nearest",
            )
            ax.set_xlabel("Frame")
            ax.set_yticks([])
            ax.set_title(f"{group_name} | {method} | Ethogram")
            fig.savefig(
                output_dir / f"{group_name}_{method}_ethogram.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    except ImportError:
        print("  matplotlib not available, skipping visualizations")


def main():
    parser = argparse.ArgumentParser(description="Sparse keypoint ablation study")
    parser.add_argument(
        "--keypoints",
        type=str,
        default="outputs/clustering/sparse/keypoints_22_3d.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/clustering/sparse/results",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(JOINT_GROUPS.keys()),
        help="Joint groups to evaluate",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load keypoints
    print(f"Loading keypoints from {args.keypoints}")
    data = np.load(args.keypoints, allow_pickle=True)
    keypoints_raw = data["keypoints"]  # (3600, 22, 3)
    keypoint_names = list(data["keypoint_names"])
    print(f"  Raw shape: {keypoints_raw.shape}")

    # Apply frame jump masking
    keypoints, valid_mask = mask_frame_jumps(keypoints_raw)
    n_excluded = (~valid_mask).sum()
    print(f"  Frame jump masking: excluded {n_excluded} frames → {keypoints.shape[0]} remaining")

    all_results = {}

    for group_name in args.groups:
        joint_indices = JOINT_GROUPS[group_name]
        n_joints = len(joint_indices)
        kp_subset = keypoints[:, joint_indices, :]
        group_names = [keypoint_names[i] for i in joint_indices]

        print(f"\n{'='*60}")
        print(f"Joint Group: {group_name} ({n_joints} joints)")
        print(f"  Joints: {group_names}")
        print(f"  Data shape: {kp_subset.shape}")

        group_results = []

        # B-SOiD (3D-extended)
        print(f"  Running B-SOiD (3D)...")
        t0 = time.time()
        bsoid_result = run_bsoid_clustering(kp_subset, fps=args.fps)
        bsoid_result["time_sec"] = round(time.time() - t0, 1)
        group_results.append(bsoid_result)
        if bsoid_result.get("metrics"):
            print(f"    Clusters: {bsoid_result['metrics'].get('n_clusters', 'N/A')}")
            print(f"    Silhouette: {bsoid_result['metrics'].get('silhouette', 'N/A')}")
            print(f"    Bout mean: {bsoid_result['metrics'].get('bout_mean_sec', 'N/A')}s")

        # SUBTLE (3D clustering)
        print(f"  Running SUBTLE...")
        t0 = time.time()
        subtle_result = run_subtle_clustering(kp_subset, fps=args.fps)
        subtle_result["time_sec"] = round(time.time() - t0, 1)
        group_results.append(subtle_result)
        if subtle_result.get("metrics"):
            print(f"    Clusters: {subtle_result['metrics'].get('n_clusters', 'N/A')}")
            print(f"    Silhouette: {subtle_result['metrics'].get('silhouette', 'N/A')}")
            print(f"    Bout mean: {subtle_result['metrics'].get('bout_mean_sec', 'N/A')}s")

        # Generic UMAP+HDBSCAN
        print(f"  Running Generic UMAP+HDBSCAN...")
        t0 = time.time()
        generic_result = run_generic_clustering(kp_subset)
        generic_result["time_sec"] = round(time.time() - t0, 1)
        group_results.append(generic_result)
        if generic_result.get("metrics"):
            print(f"    Clusters: {generic_result['metrics'].get('n_clusters', 'N/A')}")
            print(f"    Silhouette: {generic_result['metrics'].get('silhouette', 'N/A')}")

        # Save labels
        for r in group_results:
            if r.get("labels") is not None:
                np.save(
                    output_dir / f"{group_name}_{r['method']}_labels.npy",
                    r["labels"],
                )
                if r.get("embeddings") is not None:
                    np.save(
                        output_dir / f"{group_name}_{r['method']}_embeddings.npy",
                        r["embeddings"],
                    )

        # Visualizations
        if not args.skip_viz:
            save_visualizations(group_results, kp_subset, group_names, output_dir, group_name)

        # Store results (without numpy arrays for JSON serialization)
        all_results[group_name] = [
            {
                "method": r["method"],
                "metrics": r.get("metrics"),
                "time_sec": r.get("time_sec"),
                "error": r.get("error"),
            }
            for r in group_results
        ]

    # Save summary
    summary = {
        "experiment": "Phase_3A_Sparse_Keypoint_Ablation",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": "M5t2",
        "total_frames": int(keypoints_raw.shape[0]),
        "frames_after_masking": int(keypoints.shape[0]),
        "excluded_frames": int(n_excluded),
        "fps": args.fps,
        "joint_groups": {k: len(v) for k, v in JOINT_GROUPS.items()},
        "results": all_results,
    }

    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Summary saved to {summary_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Group':<12} {'Method':<22} {'Clusters':>8} {'Silhouette':>10} {'CH':>10} {'Bout(s)':>8}")
    print("-" * 72)
    for group_name, results in all_results.items():
        for r in results:
            m = r.get("metrics", {})
            if m:
                print(
                    f"{group_name:<12} {r['method']:<22} "
                    f"{m.get('n_clusters', 'N/A'):>8} "
                    f"{m.get('silhouette', 'N/A'):>10} "
                    f"{m.get('calinski_harabasz', 'N/A'):>10} "
                    f"{m.get('bout_mean_sec', 'N/A'):>8}"
                )
            else:
                print(f"{group_name:<12} {r['method']:<22} {'ERROR':>8}")

    # Generate HTML report
    try:
        from mouse_extensions.behavior.report_generator import generate_report

        report_path = generate_report(
            results_dir=str(output_dir),
            keypoints_path=args.keypoints,
        )
        print(f"\nHTML Report: {report_path}")
    except Exception as e:
        print(f"\nReport generation failed: {e}")


if __name__ == "__main__":
    main()
