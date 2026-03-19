"""Analyze BAMS embeddings: clustering + visualization.

Runs K-Means/HDBSCAN on BAMS short/long/combined embeddings,
compares with raw sparse features (S1), generates UMAP + ethogram plots.

Usage on gpu03 (bams env):
    source /home/joon/anaconda3/etc/profile.d/conda.sh && conda activate bams
    cd /home/joon/dev/FaceLift
    python -m mouse_extensions.behavior.analyze_bams_embeddings \
        --bams_emb outputs/sdannce_poc/bams/bams_embeddings.npz \
        --sparse_feat outputs/sdannce_poc/features/sparse_features.npz \
        --output outputs/sdannce_poc/bams_analysis/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def compute_tpi(labels):
    if len(labels) < 2:
        return 0.0
    return float((labels[1:] == labels[:-1]).mean())


def compute_entropy_rate(labels):
    unique = np.unique(labels)
    n = len(unique)
    if n <= 1:
        return 0.0
    T_mat = np.zeros((n, n))
    lmap = {l: i for i, l in enumerate(unique)}
    for i in range(len(labels) - 1):
        T_mat[lmap[labels[i]], lmap[labels[i + 1]]] += 1
    row_sums = T_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_norm = T_mat / row_sums
    pi = T_mat.sum(axis=1)
    pi = pi / pi.sum()
    h = 0.0
    for i in range(n):
        for j in range(n):
            if T_norm[i, j] > 0:
                h -= pi[i] * T_norm[i, j] * np.log2(T_norm[i, j])
    return float(h)


def compute_bout_stats(labels, fps=50.0):
    bouts = []
    cur, cur_len = labels[0], 1
    for i in range(1, len(labels)):
        if labels[i] == cur:
            cur_len += 1
        else:
            bouts.append(cur_len / fps)
            cur, cur_len = labels[i], 1
    bouts.append(cur_len / fps)
    bouts = np.array(bouts)
    return {
        "bout_median_s": float(np.median(bouts)),
        "bout_mean_s": float(np.mean(bouts)),
        "bout_count": len(bouts),
    }


def run_clustering(X, k, seed=0, fps=50.0):
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
    n_sample = min(10000, len(X))
    return {
        "k": k,
        "seed": seed,
        "silhouette": float(silhouette_score(X, labels, sample_size=n_sample, random_state=seed)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "tpi": compute_tpi(labels),
        "entropy_rate": compute_entropy_rate(labels),
        **compute_bout_stats(labels, fps),
    }, labels


def try_umap_plot(X, labels, title, out_path, n_samples=10000):
    """Generate UMAP scatter plot if umap-learn is available."""
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        idx = np.random.RandomState(42).choice(len(X), min(n_samples, len(X)), replace=False)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
        emb_2d = reducer.fit_transform(X[idx])

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels[idx],
                            cmap="tab10", s=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        print(f"  UMAP saved: {out_path}")
        return True
    except ImportError:
        print("  UMAP not available (pip install umap-learn)")
        return False


def plot_ethogram(labels, title, out_path, fps=50.0, max_frames=10000):
    """Generate ethogram (temporal raster) plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(max_frames, len(labels))
        t = np.arange(n) / fps

        fig, ax = plt.subplots(1, 1, figsize=(16, 2))
        ax.scatter(t, np.zeros_like(t), c=labels[:n], cmap="tab10", s=0.5, marker="|")
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xlim(0, t[-1])
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        print(f"  Ethogram saved: {out_path}")
    except ImportError:
        print("  matplotlib not available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bams_emb", required=True)
    parser.add_argument("--sparse_feat", required=True)
    parser.add_argument("--output", default="outputs/sdannce_poc/bams_analysis/")
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load BAMS embeddings
    print("Loading BAMS embeddings...")
    bams = np.load(args.bams_emb, allow_pickle=True)
    emb_short = bams["short_term"]  # (179K, 64)
    emb_long = bams["long_term"]    # (179K, 64)
    emb_combined = bams["combined"] # (179K, 128)
    print(f"  short: {emb_short.shape}, long: {emb_long.shape}, combined: {emb_combined.shape}")

    # Load sparse features for comparison
    print("Loading sparse features...")
    sparse = np.load(args.sparse_feat, allow_pickle=True)
    s1 = sparse["s1_single_rat1"]  # (90K, 69)
    s3 = sparse["s3_engineered"]   # (90K, 12)
    print(f"  S1: {s1.shape}, S3: {s3.shape}")

    # Align lengths (BAMS has 179K from overlapping windows, S1 has 90K)
    # Use first 90K of BAMS (covers full session with overlap)
    # Actually BAMS windows overlap so embeddings are per-window-frame, not per-original-frame
    # For fair comparison, use the same number of frames
    n_compare = min(len(s1), len(emb_combined))
    print(f"  Comparison length: {n_compare} frames")

    # Feature sets for clustering comparison
    feature_sets = {
        "S1_skeleton_69d": StandardScaler().fit_transform(s1[:n_compare]),
        "S3_engineered_12d": StandardScaler().fit_transform(s3[:n_compare]),
        "BAMS_short_64d": StandardScaler().fit_transform(emb_short[:n_compare]),
        "BAMS_long_64d": StandardScaler().fit_transform(emb_long[:n_compare]),
        "BAMS_combined_128d": StandardScaler().fit_transform(emb_combined[:n_compare]),
    }

    # Clustering comparison
    k_values = [4, 6, 8, 12]
    n_seeds = 3
    all_results = []

    print(f"\nRunning clustering: {len(feature_sets)} features × {len(k_values)} K × {n_seeds} seeds")
    t0 = time.time()

    best_labels = {}  # Store best labels for visualization

    for feat_name, X in feature_sets.items():
        best_sil = -1
        for k in k_values:
            for seed in range(n_seeds):
                result, labels = run_clustering(X, k, seed, args.fps)
                result["feature_set"] = feat_name
                all_results.append(result)

                if result["silhouette"] > best_sil:
                    best_sil = result["silhouette"]
                    best_labels[feat_name] = (labels, k, result)

    elapsed = time.time() - t0
    print(f"Completed {len(all_results)} experiments in {elapsed:.1f}s")

    # Save results
    json_path = out_dir / "bams_clustering_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*90}")
    print("SUMMARY: Best Silhouette per Feature Set")
    print(f"{'='*90}")
    fmt = "{:<25s} {:>4s} {:>8s} {:>6s} {:>10s} {:>8s} {:>8s}"
    print(fmt.format("Feature", "K", "Sil", "TPI", "CH", "Bout(s)", "ER"))
    print("-" * 90)

    for feat_name in feature_sets:
        _, k, r = best_labels[feat_name]
        print(fmt.format(
            feat_name, str(k),
            f"{r['silhouette']:.4f}", f"{r['tpi']:.3f}",
            f"{r['calinski_harabasz']:.1f}", f"{r['bout_median_s']:.3f}",
            f"{r['entropy_rate']:.3f}"
        ))

    # Visualizations
    print("\nGenerating visualizations...")
    for feat_name, (labels, k, _) in best_labels.items():
        X = feature_sets[feat_name]
        short_name = feat_name.replace("_", " ")

        # UMAP
        try_umap_plot(
            X, labels,
            f"{short_name} (K={k})",
            out_dir / f"umap_{feat_name}.png",
        )

        # Ethogram (first 200s = 10K frames)
        plot_ethogram(
            labels,
            f"{short_name} (K={k})",
            out_dir / f"ethogram_{feat_name}.png",
            fps=args.fps, max_frames=10000,
        )

    # Save best labels for further analysis
    labels_path = out_dir / "best_cluster_labels.npz"
    np.savez_compressed(
        str(labels_path),
        **{f"{name}_labels": best_labels[name][0] for name in best_labels},
        **{f"{name}_k": best_labels[name][1] for name in best_labels},
        n_frames=n_compare,
    )
    print(f"\nLabels saved: {labels_path}")
    print(f"All results saved: {json_path}")
    print(f"Visualizations in: {out_dir}")


if __name__ == "__main__":
    main()
