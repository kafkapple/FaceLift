"""Gaussian count ablation: extract features at multiple pruning levels + cluster.

Reads raw Gaussian NPZ files, applies top-K opacity pruning at various levels,
extracts behavior features, and runs clustering comparison with multiple methods.

Methods: KMeans, GMM, HMM (+ B-SOiD, SUBTLE on sparse only)
Metrics: Full suite via compute_all_metrics() — Sil, CH, DB, TPI, Entropy Rate,
         TC, Bout duration, Bootstrap CI, Temporal shuffle p-value.

Usage:
    python -m mouse_extensions.behavior.run_gaussian_ablation \
        --npz_dir outputs/report/clustering/features/gaussians_raw \
        --levels 1000 5000 12500 50000 200000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


FRAME_JUMPS = {1178,1179,1180,1181,1182,2358,2359,2360,2361,2362,3538,3539,3540,3541,3542}


def compute_summary(xyz, opacity, scale):
    """Compute summary statistics feature vector from pruned Gaussians."""
    f = []
    f.extend(xyz.mean(0)); f.extend(xyz.std(0))
    f.extend(np.percentile(xyz, 25, axis=0)); f.extend(np.percentile(xyz, 75, axis=0))
    f.extend([opacity.mean(), opacity.std(), float(np.median(opacity))])
    sc = scale.reshape(-1, 3) if scale.ndim == 2 else scale.reshape(-1, 3)
    f.extend(sc.mean(0)); f.extend(sc.std(0))
    f.append(sc[:, 0].mean() - sc[:, 2].mean())
    f.extend(np.percentile(sc, [25, 75], axis=0).mean(1))
    xyz_c = xyz - xyz.mean(0)
    cov = np.cov(xyz_c.T) if xyz_c.shape[0] > 3 else np.eye(3)
    eigvals = np.linalg.eigvalsh(cov)
    f.extend(sorted(eigvals, reverse=True))
    f.extend(xyz.max(0) - xyz.min(0))
    f.append(float(len(xyz)))
    return np.array(f, dtype=np.float32)


def cluster_kmeans(features, k, random_state=42):
    """KMeans clustering."""
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(features)


def cluster_gmm(features, k, random_state=42):
    """Gaussian Mixture Model clustering."""
    try:
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3,
                              covariance_type="full", max_iter=200, reg_covar=1e-4)
        return gmm.fit_predict(features)
    except Exception as e:
        print(f"  [WARN] GMM failed (K={k}): {e}")
        return None


def cluster_hmm(features, k, random_state=42):
    """Hidden Markov Model clustering (Gaussian emissions)."""
    try:
        from hmmlearn.hmm import GaussianHMM
        hmm = GaussianHMM(n_components=k, covariance_type="diag",
                          n_iter=100, random_state=random_state, verbose=False)
        hmm.fit(features)
        return hmm.predict(features)
    except ImportError:
        print("  [WARN] hmmlearn not installed, skipping HMM")
        return None
    except Exception as e:
        print(f"  [WARN] HMM failed: {e}")
        return None


def cluster_bsoid(kp_centered, fps=20.0):
    """B-SOiD clustering on keypoints (requires behavior_lab)."""
    try:
        from behavior_lab.models.discovery.bsoid import BSOiD
        model = BSOiD(fps=fps)
        labels = model.fit_predict(kp_centered)
        return labels
    except ImportError:
        print("  [WARN] behavior_lab not installed, skipping B-SOiD")
        return None
    except Exception as e:
        print(f"  [WARN] B-SOiD failed: {e}")
        return None


def cluster_subtle(kp_centered, fps=20.0):
    """SUBTLE clustering on keypoints (requires behavior_lab)."""
    try:
        from behavior_lab.models.discovery.subtle_wrapper import SUBTLE
        model = SUBTLE(fps=fps, n_train_frames=len(kp_centered))
        labels = model.fit_predict(kp_centered)
        # SUBTLE may return more labels than frames (wavelet upsampling)
        if len(labels) > len(kp_centered):
            stride = len(labels) // len(kp_centered)
            labels = labels[::stride][:len(kp_centered)]
        return labels
    except ImportError:
        print("  [WARN] behavior_lab not installed, skipping SUBTLE")
        return None
    except Exception as e:
        print(f"  [WARN] SUBTLE failed: {e}")
        return None


CLUSTERING_METHODS = {
    "KMeans": cluster_kmeans,
    "GMM": cluster_gmm,
    "HMM": cluster_hmm,
}


def main():
    from mouse_extensions.behavior.paths import RESULTS_DIR, FEATURES_DIR, GPU03_KEYPOINTS, ensure_dirs
    from mouse_extensions.behavior.metrics import compute_all_metrics
    ensure_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", default=str(FEATURES_DIR / "gaussians_raw"))
    parser.add_argument("--levels", nargs="+", type=int, default=[1000, 5000, 12500, 50000, 200000])
    parser.add_argument("--output_dir", default=str(RESULTS_DIR / "gaussian_ablation"))
    parser.add_argument("--k_values", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--methods", nargs="+", default=["KMeans", "GMM", "HMM"],
                        help="Clustering methods: KMeans, GMM, HMM")
    parser.add_argument("--skip_bsoid", action="store_true", help="Skip B-SOiD (sparse only)")
    parser.add_argument("--skip_subtle", action="store_true", help="Skip SUBTLE (sparse only)")
    parser.add_argument("--skip_significance", action="store_true",
                        help="Skip bootstrap CI / temporal shuffle (faster)")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all NPZ files
    npz_files = sorted(npz_dir.glob("*.npz"))
    frame_npz = {}
    for p in npz_files:
        fi = int(p.stem)
        if fi not in FRAME_JUMPS:
            frame_npz[fi] = p
    frames = sorted(frame_npz.keys())
    print(f"NPZ files: {len(frames)} frames")

    if len(frames) < 100:
        print("Too few frames. Waiting for extraction to complete.")
        return

    # Extract features at each pruning level
    print(f"\nExtracting features at {len(args.levels)} pruning levels...")
    level_features = {k: [] for k in args.levels}
    t0 = time.time()

    bad_frames = set()
    for i, fi in enumerate(frames):
        try:
            data = np.load(frame_npz[fi])
            xyz = data["xyz"].astype(np.float32)
            opacity = data["opacity"].astype(np.float32).flatten()
            scale = data["scale"].astype(np.float32)
        except Exception as e:
            if len(bad_frames) < 5:
                print(f"  [WARN] Bad NPZ frame {fi}: {e}")
            bad_frames.add(fi)
            continue

        sorted_idx = np.argsort(opacity)

        for k in args.levels:
            top_idx = sorted_idx[-min(k, len(opacity)):]
            feat = compute_summary(xyz[top_idx], opacity[top_idx], scale[top_idx])
            level_features[k].append(feat)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(frames)} ({time.time()-t0:.0f}s)")

    if bad_frames:
        print(f"  Skipped {len(bad_frames)} bad NPZ files")
        # Remove bad frames from the frame list
        frames = [f for f in frames if f not in bad_frames]

    print(f"Feature extraction done ({time.time()-t0:.0f}s)")

    # Load sparse keypoints for comparison
    kp_all = np.load(GPU03_KEYPOINTS, allow_pickle=True)["keypoints"]
    kp_frames = kp_all[np.array(frames)]
    kp_c = kp_frames - kp_frames.mean(1, keepdims=True)
    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)

    # Load hBehaveMAE features if available
    mae_path = FEATURES_DIR / "behavemae_features.npy"
    mae_pca = None
    if mae_path.exists():
        mae_all = np.load(mae_path)
        if len(mae_all) == 3585:
            all_valid = sorted(set(range(3600)) - FRAME_JUMPS)
            frame_to_idx = {f: i for i, f in enumerate(all_valid)}
            mae_indices = [frame_to_idx[f] for f in frames if f in frame_to_idx]
            if len(mae_indices) == len(frames):
                mae_feat = mae_all[mae_indices]
                mae_s = StandardScaler().fit_transform(mae_feat)
                mae_pca = PCA(n_components=30).fit_transform(mae_s)

    # ==================== Clustering ====================
    bootstrap_n = 0 if args.skip_significance else 100
    compute_sig = not args.skip_significance
    results = {}

    def run_and_record(name, features, labels):
        """Compute full metrics and record result."""
        m = compute_all_metrics(labels, features, fps=20.0,
                                bootstrap_n=bootstrap_n,
                                compute_significance=compute_sig)
        d = m.to_dict()
        results[name] = d
        # Print summary row
        sil = d.get("silhouette") or 0
        ch = d.get("calinski_harabasz") or 0
        db = d.get("davies_bouldin") or 0
        bout = d.get("bout_mean_sec") or 0
        tc = d.get("temporal_consistency") or 0
        tpi = d.get("tpi_mean") or 0
        ent = d.get("entropy_rate") or 0
        short = 0
        if d.get("n_bouts") and d.get("bout_mean_sec"):
            # Compute short bout pct from bout stats
            changes = np.where(np.diff(labels[labels >= 0]) != 0)[0]
            bouts = np.diff(np.concatenate([[0], changes + 1, [int((labels >= 0).sum())]])) / 20.0
            short = float((bouts < 0.3).mean() * 100) if len(bouts) > 0 else 0
        p_val = d.get("p_value_vs_shuffle")
        p_str = f"{p_val:.3f}" if p_val is not None else "N/A"
        print(f"  {name:<35} Sil={sil:.3f} CH={ch:.0f} DB={db:.2f} | "
              f"Bout={bout:.2f}s TC={tc:.3f} TPI={tpi:.1f} Ent={ent:.3f} Short={short:.0f}% | p={p_str}")
        return d

    # Header
    print(f"\n{'='*120}")
    print(f"  {'Method':<35} {'--- Intrinsic ---':>25} | {'--- Temporal ---':>45} | {'Stat':>6}")
    print(f"{'='*120}")

    # ============================================================
    # Naming convention: {Input}_{Representation}_{Clustering}_K{k}
    #   Input:   SP (Sparse keypoint), DN (Dense 3DGS), HY (Hybrid)
    #   Repr:    RawPCA, MAE (hBehaveMAE), Stats (summary), Concat
    #   Cluster: KMeans, GMM, HMM, BSOID, SUBTLE
    # ============================================================

    # === SP_RawPCA: Sparse Input + Raw PCA ===
    print("\n[SP_RawPCA] Sparse Keypoint → Raw PCA(20)")
    for method_name in args.methods:
        cluster_fn = CLUSTERING_METHODS.get(method_name)
        if cluster_fn is None:
            continue
        for k in args.k_values:
            labels = cluster_fn(kp_pca, k)
            if labels is None:
                continue
            run_and_record(f"SP_RawPCA_{method_name}_K{k}", kp_pca, labels)

    # B-SOiD on sparse (density-based, auto K)
    if not args.skip_bsoid:
        print("\n[SP_RawPCA] Sparse Keypoint → B-SOiD (UMAP+HDBSCAN)")
        labels = cluster_bsoid(kp_c.reshape(len(kp_c), -1))
        if labels is not None and len(labels) == len(kp_pca):
            run_and_record("SP_RawPCA_BSOID", kp_pca, labels)

    # SUBTLE on sparse
    if not args.skip_subtle:
        print("\n[SP_RawPCA] Sparse Keypoint → SUBTLE (Wavelet+Phenograph)")
        labels = cluster_subtle(kp_c.reshape(len(kp_c), -1))
        if labels is not None and len(labels) == len(kp_pca):
            run_and_record("SP_RawPCA_SUBTLE", kp_pca, labels)

    # === SP_MAE: Sparse Input + hBehaveMAE (learned representation) ===
    if mae_pca is not None:
        print("\n[SP_MAE] Sparse Keypoint → hBehaveMAE → PCA(30)")
        for method_name in args.methods:
            cluster_fn = CLUSTERING_METHODS.get(method_name)
            if cluster_fn is None:
                continue
            for k in args.k_values:
                labels = cluster_fn(mae_pca, k)
                if labels is None:
                    continue
                run_and_record(f"SP_MAE_{method_name}_K{k}", mae_pca, labels)

    # === DN_Stats: Dense Input (3DGS) + Summary Statistics ===
    for n_gauss in args.levels:
        features = np.stack(level_features[n_gauss])
        feat_s = StandardScaler().fit_transform(features)
        feat_pca = PCA(n_components=min(20, feat_s.shape[1])).fit_transform(feat_s)

        print(f"\n[DN_Stats_{n_gauss:,}] Dense 3DGS (top-{n_gauss:,} opacity) → Summary Stats → PCA(20)")
        for method_name in args.methods:
            cluster_fn = CLUSTERING_METHODS.get(method_name)
            if cluster_fn is None:
                continue
            for k in args.k_values:
                labels = cluster_fn(feat_pca, k)
                if labels is None:
                    continue
                run_and_record(f"DN_Stats{n_gauss}_{method_name}_K{k}", feat_pca, labels)

        # === HY_Concat: Hybrid (Sparse + Dense) ===
        hybrid = np.concatenate([kp_pca, feat_pca], axis=1)
        print(f"  [HY_Concat_{n_gauss:,}] SP_RawPCA + DN_Stats → Concat")
        for method_name in args.methods:
            cluster_fn = CLUSTERING_METHODS.get(method_name)
            if cluster_fn is None:
                continue
            labels = cluster_fn(hybrid, 8)
            if labels is None:
                continue
            run_and_record(f"HY_Concat{n_gauss}_{method_name}_K8", hybrid, labels)

    # ==================== Save ====================
    output_path = output_dir / "gaussian_ablation_full_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*120}")
    print(f"Results saved: {output_path}")
    print(f"Total experiments: {len(results)}")

    # Summary table: best per feature type
    print(f"\n{'='*100}")
    print("BEST PER FEATURE PIPELINE (KMeans K=8)")
    print(f"  Input → Representation → Clustering")
    print(f"{'='*100}")
    summary_keys = [
        ("SP_RawPCA_KMeans_K8",       "Sparse → Raw PCA(20)"),
        ("SP_MAE_KMeans_K8",          "Sparse → hBehaveMAE → PCA(30)"),
        ("DN_Stats1000_KMeans_K8",    "Dense 3DGS (1K) → Summary Stats"),
        ("DN_Stats5000_KMeans_K8",    "Dense 3DGS (5K) → Summary Stats"),
        ("DN_Stats12500_KMeans_K8",   "Dense 3DGS (12.5K) → Summary Stats"),
        ("DN_Stats50000_KMeans_K8",   "Dense 3DGS (50K) → Summary Stats"),
        ("DN_Stats200000_KMeans_K8",  "Dense 3DGS (200K) → Summary Stats"),
        ("HY_Concat12500_KMeans_K8",  "Sparse+Dense (12.5K) → Concat"),
    ]
    for key, desc in summary_keys:
        if key in results:
            d = results[key]
            sil = d.get("silhouette") or 0
            bout = d.get("bout_mean_sec") or 0
            tc = d.get("temporal_consistency") or 0
            tpi = d.get("tpi_mean") or 0
            ent = d.get("entropy_rate") or 0
            print(f"  {desc:<42} Sil={sil:.3f}  Bout={bout:.2f}s  TC={tc:.3f}  TPI={tpi:.1f}  Ent={ent:.3f}")


if __name__ == "__main__":
    main()
