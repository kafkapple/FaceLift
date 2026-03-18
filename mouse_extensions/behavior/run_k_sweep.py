"""H0: K selection sweep — determine optimal K for behavior clustering.

Runs KMeans with K=2..20 on SP_MAE and SP_RawPCA, computing full metrics.
Outputs JSON results + prints summary table.

Usage:
    python -m mouse_extensions.behavior.run_k_sweep
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


FRAME_JUMPS = {1178,1179,1180,1181,1182,2358,2359,2360,2361,2362,3538,3539,3540,3541,3542}
K_RANGE = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]


def compute_metrics(features, labels, fps=20.0):
    """Compute key metrics for a clustering result."""
    valid = labels >= 0
    lab_v = labels[valid]
    feat_v = features[valid]

    if len(set(lab_v)) < 2:
        return None

    sil = float(silhouette_score(feat_v, lab_v, sample_size=min(5000, len(feat_v))))
    ch = float(calinski_harabasz_score(feat_v, lab_v))
    db = float(davies_bouldin_score(feat_v, lab_v))

    # Temporal metrics
    changes = np.where(np.diff(lab_v) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(lab_v)]])) / fps
    tc = 1.0 - len(changes) / (len(lab_v) - 1) if len(lab_v) > 1 else 1.0
    short_pct = float((bouts < 0.3).mean() * 100) if len(bouts) > 0 else 0

    # TPI
    unique = sorted(set(lab_v))
    K = len(unique)
    lm = {l: i for i, l in enumerate(unique)}
    T = np.zeros((K, K))
    for t in range(len(lab_v) - 1):
        T[lm[lab_v[t]], lm[lab_v[t + 1]]] += 1
    rs = T.sum(1, keepdims=True); rs[rs == 0] = 1; P = T / rs
    tpi = np.clip(1.0 / (1.0 - np.diag(P) + 1e-6), 0, 10000)

    # Entropy rate
    counts = np.zeros(K)
    for l in lab_v: counts[lm[l]] += 1
    pi = counts / counts.sum()
    ent = 0.0
    for i in range(K):
        for j in range(K):
            if P[i, j] > 0:
                ent -= pi[i] * P[i, j] * np.log2(P[i, j])

    # Inertia (for elbow method)
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(feat_v)
    inertia = float(km.inertia_)

    return {
        "k": K,
        "silhouette": round(sil, 4),
        "calinski_harabasz": round(ch, 1),
        "davies_bouldin": round(db, 3),
        "bout_mean_sec": round(float(bouts.mean()), 3),
        "bout_median_sec": round(float(np.median(bouts)), 3),
        "tc": round(tc, 4),
        "tpi_mean": round(float(tpi.mean()), 1),
        "entropy_rate": round(ent, 4),
        "short_pct": round(short_pct, 1),
        "n_bouts": int(len(bouts)),
        "inertia": round(inertia, 1),
    }


def main():
    from mouse_extensions.behavior.paths import RESULTS_DIR, FEATURES_DIR, GPU03_KEYPOINTS, ensure_dirs
    ensure_dirs()

    output_dir = RESULTS_DIR / "k_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load keypoints
    kp_all = np.load(GPU03_KEYPOINTS, allow_pickle=True)["keypoints"]
    all_valid = sorted(set(range(3600)) - FRAME_JUMPS)
    kp = kp_all[all_valid]
    kp_c = kp - kp.mean(1, keepdims=True)
    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)
    N = len(kp_pca)
    print(f"Loaded {N} frames")

    # Load hBehaveMAE
    mae_path = FEATURES_DIR / "behavemae_features.npy"
    mae_pca = None
    if mae_path.exists():
        mae_all = np.load(mae_path)
        if len(mae_all) == N:
            mae_s = StandardScaler().fit_transform(mae_all)
            mae_pca = PCA(n_components=30).fit_transform(mae_s)
            print(f"Loaded hBehaveMAE features ({mae_pca.shape})")

    results = {}

    # SP_RawPCA sweep
    print(f"\n{'='*90}")
    print(f"SP_RawPCA K Sweep")
    print(f"{'K':>4} {'Sil':>7} {'CH':>8} {'DB':>7} | {'Bout':>6} {'TC':>6} {'TPI':>7} {'Ent':>6} {'Short%':>7} {'Inertia':>10}")
    print(f"{'-'*90}")

    for k in K_RANGE:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kp_pca)
        m = compute_metrics(kp_pca, labels)
        if m:
            key = f"SP_RawPCA_K{k}"
            results[key] = m
            print(f"{k:>4} {m['silhouette']:>7.3f} {m['calinski_harabasz']:>8.0f} {m['davies_bouldin']:>7.2f} | "
                  f"{m['bout_mean_sec']:>6.2f} {m['tc']:>6.3f} {m['tpi_mean']:>7.1f} {m['entropy_rate']:>6.3f} "
                  f"{m['short_pct']:>7.1f} {m['inertia']:>10.0f}")

    # SP_MAE sweep
    if mae_pca is not None:
        print(f"\n{'='*90}")
        print(f"SP_MAE K Sweep")
        print(f"{'K':>4} {'Sil':>7} {'CH':>8} {'DB':>7} | {'Bout':>6} {'TC':>6} {'TPI':>7} {'Ent':>6} {'Short%':>7} {'Inertia':>10}")
        print(f"{'-'*90}")

        for k in K_RANGE:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(mae_pca)
            m = compute_metrics(mae_pca, labels)
            if m:
                key = f"SP_MAE_K{k}"
                results[key] = m
                print(f"{k:>4} {m['silhouette']:>7.3f} {m['calinski_harabasz']:>8.0f} {m['davies_bouldin']:>7.2f} | "
                      f"{m['bout_mean_sec']:>6.2f} {m['tc']:>6.3f} {m['tpi_mean']:>7.1f} {m['entropy_rate']:>6.3f} "
                      f"{m['short_pct']:>7.1f} {m['inertia']:>10.0f}")

    # Save
    output_path = output_dir / "k_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # Best K analysis
    print(f"\n{'='*60}")
    print("OPTIMAL K ANALYSIS")
    print(f"{'='*60}")
    for prefix in ["SP_RawPCA", "SP_MAE"]:
        keys = [k for k in results if k.startswith(prefix)]
        if not keys:
            continue
        best_sil_k = max(keys, key=lambda k: results[k]["silhouette"])
        best_tc_k = max(keys, key=lambda k: results[k]["tc"])
        print(f"\n{prefix}:")
        print(f"  Best Sil: K={results[best_sil_k]['k']} (Sil={results[best_sil_k]['silhouette']:.3f})")
        print(f"  Best TC:  K={results[best_tc_k]['k']} (TC={results[best_tc_k]['tc']:.3f})")

        # Elbow: find largest drop in inertia
        ks = sorted(keys, key=lambda k: results[k]["k"])
        inertias = [results[k]["inertia"] for k in ks]
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow_idx = np.argmax(diffs2) + 2  # +2 for double diff offset
            if elbow_idx < len(ks):
                print(f"  Elbow:    K={results[ks[elbow_idx]]['k']} (inertia 2nd derivative max)")


if __name__ == "__main__":
    main()
