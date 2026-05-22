"""Unified Behavior Clustering Pipeline.

Manages: preprocessing presets × joint groups × clustering methods
in a single consistent framework. All methods receive the same preprocessed
data and z-scored features.

Usage:
    cd /home/joon/dev/FaceLift
    python -m mouse_extensions.behavior.unified_pipeline \
        --presets centered bsoid_standard full \
        --groups minimal_7 kinematic_18 full_22 \
        --methods pca_kmeans umap_hdbscan bsoid
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from mouse_extensions.behavior.preprocessing import (
    preprocess, PreprocessConfig, PRESETS,
)
from mouse_extensions.constants import MOUSE_KP_NAMES as KEYPOINT_NAMES

# Expanded joint groups (fine-grained ablation)
JOINT_GROUPS = {
    "ultra_4": [2, 3, 4, 5],
    "minimal_7": [2, 3, 4, 5, 8, 12, 16],
    "loco_10": [2, 3, 4, 5, 8, 12, 16, 19, 11, 15],
    "posture_12": [2, 3, 4, 5, 8, 12, 16, 19, 11, 15, 18, 21],
    "forelimb_15": [2, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 18, 19, 21],
    "kinematic_18": [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21],
    "full_22": list(range(22)),
}


def extract_features_and_standardize(
    kp_subset: np.ndarray,
) -> np.ndarray:
    """Flatten keypoints and z-score standardize.

    This is the critical step that fixes B-SOiD/UMAP numerical issues:
    After body-size normalization, values are ~0-1.5 range.
    Z-scoring brings them to mean=0, std=1 → UMAP/HDBSCAN-friendly.

    Returns:
        (T, K*D) z-scored feature matrix
    """
    T = kp_subset.shape[0]
    features = kp_subset.reshape(T, -1)
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def run_method(
    method_name: str,
    kp_subset: np.ndarray,
    features_zscored: np.ndarray,
    fps: float = 20.0,
) -> dict:
    """Run a single clustering method.

    B-SOiD/SUBTLE receive raw (non-z-scored) keypoints because they
    extract their own features internally. But we also pass z-scored
    features for methods that need them (PCA+KMeans, UMAP+HDBSCAN).
    """
    try:
        if method_name == "pca_kmeans":
            return _run_pca_kmeans(features_zscored, fps)
        elif method_name == "umap_hdbscan":
            return _run_umap_hdbscan(features_zscored, fps)
        elif method_name == "bsoid":
            return _run_bsoid(kp_subset, fps)
        elif method_name == "subtle":
            return _run_subtle(kp_subset, fps)
        else:
            return {"method": method_name, "error": f"Unknown method: {method_name}"}
    except Exception as e:
        return {"method": method_name, "error": str(e)}


def _run_pca_kmeans(features: np.ndarray, fps: float) -> dict:
    pca = PCA(n_components=min(20, features.shape[1]))
    feat_pca = pca.fit_transform(features)

    best_sil, best_k = -1, 4
    for k in [4, 6, 8, 10, 12]:
        if k >= len(feat_pca):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(feat_pca)
        sil = silhouette_score(feat_pca, labels, sample_size=min(3000, len(feat_pca)))
        if sil > best_sil:
            best_sil, best_k = sil, k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(feat_pca)

    return {
        "method": "PCA_KMeans",
        "labels": labels,
        "pca_var": round(float(pca.explained_variance_ratio_.sum()), 4),
        **_compute_all_metrics(labels, feat_pca, fps),
    }


def _run_umap_hdbscan(features: np.ndarray, fps: float) -> dict:
    import umap
    import hdbscan

    pca = PCA(n_components=min(30, features.shape[1]))
    feat_pca = pca.fit_transform(features)

    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.0, random_state=42)
    embedding = reducer.fit_transform(feat_pca)

    # Use larger min_cluster_size to avoid over-segmentation
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20)
    labels = clusterer.fit_predict(embedding)

    return {
        "method": "UMAP_HDBSCAN",
        "labels": labels,
        "embedding_2d": embedding,
        **_compute_all_metrics(labels, feat_pca, fps),
    }


def _run_bsoid(kp_subset: np.ndarray, fps: float) -> dict:
    """B-SOiD receives preprocessed (centered, optionally normalized) keypoints.

    B-SOiD extracts its own features internally (displacement, pairwise dist, angles)
    and applies its own UMAP + HDBSCAN. The features are standardized internally
    via StandardScaler in the B-SOiD pipeline.
    """
    from behavior_lab.models.discovery.bsoid import BSOiD

    model = BSOiD(fps=int(fps), min_cluster_size=30)
    result = model.fit_predict(kp_subset)
    labels = result.labels
    feat = kp_subset[:len(labels)].reshape(len(labels), -1)

    return {
        "method": "B-SOiD",
        "labels": labels,
        **_compute_all_metrics(labels, feat, fps),
    }


def _run_subtle(kp_subset: np.ndarray, fps: float) -> dict:
    from behavior_lab.models.discovery.subtle_wrapper import SUBTLE

    model = SUBTLE(fps=int(fps), n_train_frames=min(120000, len(kp_subset)))
    result = model.fit([kp_subset])
    labels = result.labels if hasattr(result, "labels") else result["labels"]
    feat = kp_subset[:len(labels)].reshape(len(labels), -1)

    return {
        "method": "SUBTLE",
        "labels": labels,
        **_compute_all_metrics(labels, feat, fps),
    }


def _compute_all_metrics(labels: np.ndarray, features: np.ndarray, fps: float) -> dict:
    """Compute clustering quality + temporal metrics."""
    valid = labels >= 0
    n_valid = valid.sum()
    n_clusters = len(set(labels[valid])) if n_valid > 0 else 0

    metrics = {
        "n_clusters": int(n_clusters),
        "n_valid": int(n_valid),
        "n_noise": int((~valid).sum()),
        "noise_ratio": round(float((~valid).sum() / len(labels)), 4) if len(labels) > 0 else 0,
    }

    if n_clusters >= 2 and n_valid >= 10:
        f, l = features[valid], labels[valid]
        sample = min(5000, n_valid)
        metrics["silhouette"] = round(float(silhouette_score(f, l, sample_size=sample)), 4)
        metrics["calinski_harabasz"] = round(float(calinski_harabasz_score(f, l)), 2)
        metrics["davies_bouldin"] = round(float(davies_bouldin_score(f, l)), 4)

    # Temporal metrics
    lab = labels[valid]
    if len(lab) >= 2:
        changes = np.where(np.diff(lab) != 0)[0]
        bouts = np.diff(np.concatenate([[0], changes + 1, [len(lab)]])) / fps
        metrics["bout_mean_sec"] = round(float(bouts.mean()), 3)
        metrics["bout_median_sec"] = round(float(np.median(bouts)), 3)
        metrics["n_bouts"] = int(len(bouts))
        metrics["transition_rate"] = round(float(len(changes) / (len(lab) / fps)), 3)

    return metrics


def run_experiment(
    kp_raw: np.ndarray,
    presets: list[str],
    groups: list[str],
    methods: list[str],
    output_dir: Path,
    fps: float = 20.0,
) -> dict:
    """Run full experiment grid: presets × groups × methods."""

    all_results = {}

    for preset_name in presets:
        print(f"\n{'#'*70}")
        print(f"PRESET: {preset_name}")
        print(f"{'#'*70}")

        kp_proc, preproc_meta = preprocess(kp_raw, preset=preset_name)
        print(f"  Preprocessed: {kp_proc.shape}, range [{kp_proc.min():.3f}, {kp_proc.max():.3f}]")

        preset_results = {"preprocessing": preproc_meta, "groups": {}}

        for group_name in groups:
            indices = JOINT_GROUPS[group_name]
            kp_g = kp_proc[:, indices, :]
            features_z = extract_features_and_standardize(kp_g)

            print(f"\n  Group: {group_name} ({len(indices)} joints)")

            group_results = {}

            for method_name in methods:
                print(f"    {method_name}...", end=" ", flush=True)
                t0 = time.time()

                result = run_method(method_name, kp_g, features_z, fps)
                elapsed = round(time.time() - t0, 1)

                # Save labels
                if "labels" in result and result["labels"] is not None:
                    label_path = output_dir / f"{preset_name}_{group_name}_{method_name}_labels.npy"
                    np.save(label_path, result["labels"])
                    if "embedding_2d" in result and result["embedding_2d"] is not None:
                        np.save(
                            output_dir / f"{preset_name}_{group_name}_{method_name}_embed.npy",
                            result["embedding_2d"],
                        )

                # Store metrics (no numpy arrays)
                metrics = {
                    k: v for k, v in result.items()
                    if k not in ("labels", "embedding_2d", "features_pca")
                }
                metrics["time_sec"] = elapsed

                sil = metrics.get("silhouette", "err")
                k = metrics.get("n_clusters", "err")
                print(f"K={k} Sil={sil} ({elapsed}s)")

                group_results[method_name] = metrics

            preset_results["groups"][group_name] = {
                "n_joints": len(indices),
                "joint_names": [KEYPOINT_NAMES[i] for i in indices],
                "methods": group_results,
            }

        all_results[preset_name] = preset_results

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Unified Behavior Clustering Pipeline")
    from mouse_extensions.paths import KP_22
    parser.add_argument(
        "--keypoints",
        default=str(KP_22),
    )
    parser.add_argument("--presets", nargs="+", default=["raw", "centered", "bsoid_standard"])
    parser.add_argument("--groups", nargs="+", default=["minimal_7", "kinematic_18", "full_22"])
    parser.add_argument("--methods", nargs="+", default=["pca_kmeans", "umap_hdbscan"])
    from mouse_extensions.behavior.paths import RESULTS_DIR
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "unified"))
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading {args.keypoints}")
    data = np.load(args.keypoints, allow_pickle=True)
    kp_raw = data["keypoints"]
    print(f"Shape: {kp_raw.shape}")

    # Run experiment grid
    results = run_experiment(
        kp_raw, args.presets, args.groups, args.methods, output_dir, args.fps,
    )

    # Save summary
    summary = {
        "experiment": "Unified_Behavior_Clustering",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "presets": args.presets,
        "groups": args.groups,
        "methods": args.methods,
        "results": results,
    }
    summary_path = output_dir / "unified_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nSummary: {summary_path}")

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"{'Preset':<18} {'Group':<14} {'Method':<16} {'K':>4} {'Sil':>8} {'CH':>10} {'DB':>8} {'Bout':>6}")
    print("-" * 90)
    for preset_name, pdata in results.items():
        for group_name, gdata in pdata.get("groups", {}).items():
            for method_name, mdata in gdata.get("methods", {}).items():
                sil = mdata.get("silhouette", "-")
                ch = mdata.get("calinski_harabasz", "-")
                db = mdata.get("davies_bouldin", "-")
                bout = mdata.get("bout_mean_sec", "-")
                k = mdata.get("n_clusters", "-")
                err = mdata.get("error", "")
                if err:
                    print(f"{preset_name:<18} {group_name:<14} {method_name:<16} {'ERR':>4} {err[:40]}")
                else:
                    print(f"{preset_name:<18} {group_name:<14} {method_name:<16} {k:>4} {sil:>8} {ch:>10} {db:>8} {bout:>6}")


if __name__ == "__main__":
    main()
