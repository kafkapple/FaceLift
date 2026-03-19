"""Plot BAMS vs Sparse comparison: ethogram + UMAP + cluster distribution."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    out_dir = Path("outputs/sdannce_poc/bams_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    bams = np.load("outputs/sdannce_poc/bams/bams_embeddings.npz", allow_pickle=True)
    sparse = np.load("outputs/sdannce_poc/features/sparse_features.npz", allow_pickle=True)

    s1 = sparse["s1_single_rat1"][:90000]
    bams_long = bams["long_term"][:90000]
    bams_comb = bams["combined"][:90000]

    # Cluster
    labels_s1 = KMeans(4, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(s1))
    labels_bl = KMeans(4, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(bams_long))
    labels_bc = KMeans(4, random_state=0, n_init=10).fit_predict(StandardScaler().fit_transform(bams_comb))

    # 1. Ethogram
    fps = 50
    n_show = 15000
    t = np.arange(n_show) / fps

    fig, axes = plt.subplots(3, 1, figsize=(18, 5), sharex=True)
    configs = [
        (labels_s1, "S1 Skeleton (Sil=0.362, TPI=0.990, Bout=1.06s)"),
        (labels_bl, "BAMS Long (Sil=0.359, TPI=0.900, Bout=0.06s)"),
        (labels_bc, "BAMS Combined (Sil=0.294, TPI=0.916, Bout=0.06s)"),
    ]
    for ax, (labels, title) in zip(axes, configs):
        for k in range(4):
            mask = labels[:n_show] == k
            ax.fill_between(t, 0, 1, where=mask, alpha=0.7, label="C%d" % k)
        ax.set_ylabel(title, fontsize=8)
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Ethogram: S1 vs BAMS (K=4, s-DANNCE rat1)", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out_dir / "ethogram_comparison.png"), dpi=150)
    plt.close()
    print("Saved ethogram_comparison.png")

    # 2. Cluster distribution
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (labels, title) in zip(axes, [
        (labels_s1, "S1"), (labels_bl, "BAMS Long"), (labels_bc, "BAMS Combined")
    ]):
        counts = np.bincount(labels, minlength=4)
        ax.bar(range(4), counts / len(labels) * 100)
        ax.set_title(title)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("% Frames")
        ax.set_ylim(0, 60)
    plt.suptitle("Cluster Distribution (K=4)")
    plt.tight_layout()
    plt.savefig(str(out_dir / "cluster_distribution.png"), dpi=150)
    plt.close()
    print("Saved cluster_distribution.png")

    # 3. UMAP
    try:
        import umap
        idx = np.random.RandomState(42).choice(90000, 10000, replace=False)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        datasets = [
            (StandardScaler().fit_transform(s1), labels_s1, "S1 Skeleton (69d)"),
            (StandardScaler().fit_transform(bams_long), labels_bl, "BAMS Long (64d)"),
            (StandardScaler().fit_transform(bams_comb), labels_bc, "BAMS Combined (128d)"),
        ]
        for ax, (X, labels, title) in zip(axes, datasets):
            reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
            emb_2d = reducer.fit_transform(X[idx])
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels[idx], cmap="tab10", s=1, alpha=0.4)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("UMAP: S1 vs BAMS (K=4, 10K samples)", fontsize=12)
        plt.tight_layout()
        plt.savefig(str(out_dir / "umap_comparison.png"), dpi=150)
        plt.close()
        print("Saved umap_comparison.png")
    except Exception as e:
        print("UMAP failed: %s" % e)

    print("Done!")


if __name__ == "__main__":
    main()
