"""Clustering PoC for BehaviorSplatter: frame length ablation × feature type.

Runs K-Means and HDBSCAN clustering on sparse features, varying:
  - Frame lengths: [1500, 5000, 15000, 45000, 90000]
  - Feature sets: S1 (skeleton), S3 (engineered), S1+S3 (combined)
  - K values (for K-Means): [4, 6, 8, 12, 16]
  - Seeds: 5 per configuration

Outputs: clustering metrics (Silhouette, CH, DB, TPI) as CSV + summary plots.

Usage:
    python run_clustering_poc.py \
        --features outputs/features/sparse_features.npz \
        --output outputs/clustering_poc/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def compute_tpi(labels: np.ndarray, window: int = 20) -> float:
    """Temporal Persistence Index (SUBTLE, IJCV 2024).

    Fraction of consecutive frame pairs where the label stays the same.
    Higher = more temporally coherent clusters.
    """
    if len(labels) < 2:
        return 0.0
    same = (labels[1:] == labels[:-1]).mean()
    return float(same)


def compute_entropy_rate(labels: np.ndarray) -> float:
    """Entropy rate of label transitions."""
    unique = np.unique(labels)
    n_states = len(unique)
    if n_states <= 1:
        return 0.0

    # Transition matrix
    T_mat = np.zeros((n_states, n_states))
    label_map = {l: i for i, l in enumerate(unique)}
    for i in range(len(labels) - 1):
        T_mat[label_map[labels[i]], label_map[labels[i + 1]]] += 1

    # Normalize rows
    row_sums = T_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_norm = T_mat / row_sums

    # Stationary distribution (left eigenvector)
    pi = T_mat.sum(axis=1)
    pi = pi / pi.sum()

    # Entropy rate
    h = 0.0
    for i in range(n_states):
        for j in range(n_states):
            if T_norm[i, j] > 0:
                h -= pi[i] * T_norm[i, j] * np.log2(T_norm[i, j])
    return float(h)


def compute_bout_stats(labels: np.ndarray, fps: float = 50.0) -> dict:
    """Compute bout duration statistics."""
    bouts = []
    current_label = labels[0]
    current_len = 1
    for i in range(1, len(labels)):
        if labels[i] == current_label:
            current_len += 1
        else:
            bouts.append(current_len / fps)
            current_label = labels[i]
            current_len = 1
    bouts.append(current_len / fps)
    bouts = np.array(bouts)
    return {
        "bout_median_s": float(np.median(bouts)),
        "bout_mean_s": float(np.mean(bouts)),
        "bout_count": len(bouts),
    }


def run_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run K-Means and return labels."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    return km.fit_predict(X)


def run_single_experiment(
    X: np.ndarray,
    k: int,
    seed: int,
    fps: float = 50.0,
) -> dict:
    """Run one clustering experiment and compute all metrics."""
    labels = run_kmeans(X, k, seed)

    result = {
        "k": k,
        "seed": seed,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "silhouette": float(silhouette_score(X, labels, sample_size=min(10000, len(X)), random_state=seed)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "tpi": compute_tpi(labels),
        "entropy_rate": compute_entropy_rate(labels),
    }
    result.update(compute_bout_stats(labels, fps))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Path to sparse_features.npz")
    parser.add_argument("--output", default="outputs/clustering_poc/", help="Output dir")
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    print(f"Loading features from {args.features}...")
    data = np.load(args.features, allow_pickle=True)

    feature_sets = {
        "S1_single_rat1": data["s1_single_rat1"],
        "S1_dyadic": data["s1_dyadic"],
        "S3_engineered": data["s3_engineered"],
        "S1_S3_combined": np.concatenate([data["s1_dyadic"], data["s3_engineered"]], axis=1),
    }

    frame_lengths = [1500, 5000, 15000, 45000, 90000]
    k_values = [4, 6, 8, 12, 16]
    n_seeds = 5

    total_experiments = (
        len(feature_sets) * len(frame_lengths) * len(k_values) * n_seeds
    )
    print(f"Total experiments: {total_experiments}")
    print(f"Feature sets: {list(feature_sets.keys())}")
    print(f"Frame lengths: {frame_lengths}")
    print(f"K values: {k_values}")

    all_results = []
    exp_count = 0
    t0 = time.time()

    for feat_name, feat_data in feature_sets.items():
        for n_frames in frame_lengths:
            if n_frames > feat_data.shape[0]:
                n_frames = feat_data.shape[0]

            X = feat_data[:n_frames]

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            for k in k_values:
                if k >= n_frames:
                    continue

                for seed in range(n_seeds):
                    result = run_single_experiment(X_scaled, k, seed, args.fps)
                    result["feature_set"] = feat_name
                    result["frame_length"] = n_frames
                    all_results.append(result)

                    exp_count += 1
                    if exp_count % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"  [{exp_count}/{total_experiments}] "
                              f"{feat_name} n={n_frames} k={k} "
                              f"sil={result['silhouette']:.3f} "
                              f"tpi={result['tpi']:.3f} "
                              f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nCompleted {exp_count} experiments in {elapsed:.1f}s")

    # Save raw results
    results_path = output_dir / "clustering_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {results_path}")

    # Summary: best config per feature set
    print(f"\n{'='*80}")
    print("SUMMARY: Best Silhouette per Feature Set × Frame Length")
    print(f"{'='*80}")
    print(f"{'Feature':<25s} {'Frames':>8s} {'K':>4s} {'Silhouette':>11s} {'TPI':>6s} {'CH':>10s} {'Bout(s)':>8s}")
    print("-" * 80)

    for feat_name in feature_sets:
        for n_frames in frame_lengths:
            subset = [r for r in all_results
                      if r["feature_set"] == feat_name and r["frame_length"] == n_frames]
            if not subset:
                continue
            best = max(subset, key=lambda x: x["silhouette"])
            print(f"{feat_name:<25s} {best['frame_length']:>8d} {best['k']:>4d} "
                  f"{best['silhouette']:>11.4f} {best['tpi']:>6.3f} "
                  f"{best['calinski_harabasz']:>10.1f} {best['bout_median_s']:>8.3f}")

    # Generate aggregated summary CSV
    import csv
    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
