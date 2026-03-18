"""Multi-method clustering comparison: B-SOiD, SUBTLE, GMM, KMeans, HMM.

All methods receive the SAME data (centered mm-scale, NOT pre-normalized).
Each method handles its own internal normalization.

Goal: Find 2+ methods showing consistent dense/hybrid temporal advantage.
"""
import numpy as np, json, sys, time, traceback
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
sys.path.insert(0, ".")

def load_data():
    from plyfile import PlyData
    kp_all = np.load("/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
                     allow_pickle=True)["keypoints"]
    plys = sorted(Path("outputs/experiments/phase3_e2e/H3_resume_pose").rglob("gaussians.ply"))
    frame_ply = {}
    for p in plys:
        for part in p.parts:
            if part.isdigit() and len(part) == 6:
                fi = int(part)
                if fi not in frame_ply:
                    frame_ply[fi] = str(p)
                break
    frames = np.array(sorted(frame_ply.keys()))
    kp = kp_all[frames]
    # COM centering only (mm scale preserved for B-SOiD/SUBTLE)
    kp_centered_mm = kp - kp.mean(1, keepdims=True)

    def load_gauss(p):
        ply = PlyData.read(p)
        v = ply["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
        op = np.array(v["opacity"])
        f = []
        f.extend(xyz.mean(0)); f.extend(xyz.std(0))
        f.extend(np.percentile(xyz, [25,75], axis=0).flatten())
        f.extend([op.mean(), op.std(), np.median(op)])
        f.extend(sc.mean(0)); f.extend(sc.std(0))
        f.append(sc[:,0].mean() - sc[:,2].mean())
        xyz_c = xyz - xyz.mean(0)
        eigvals = np.linalg.eigvalsh(np.cov(xyz_c.T))
        f.extend(sorted(eigvals, reverse=True))
        f.extend(xyz.max(0) - xyz.min(0))
        return np.array(f, dtype=np.float32)

    gauss_feats = np.stack([load_gauss(frame_ply[f]) for f in frames])
    return kp_centered_mm, gauss_feats, frames


def temporal_metrics(labels, fps=20.0):
    valid = labels >= 0
    lab = valid_labels = labels[valid]
    if len(lab) < 2:
        return {"bout_mean": 0, "tc": 0, "short_pct": 0, "n_bouts": 0}
    changes = np.where(np.diff(lab) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(lab)]])) / fps
    tc = 1.0 - len(changes) / (len(lab) - 1) if len(lab) > 1 else 1.0
    return {
        "bout_mean": round(float(bouts.mean()), 3),
        "tc": round(float(tc), 4),
        "short_pct": round(float((bouts < 0.3).mean() * 100), 1),
        "n_bouts": int(len(bouts)),
        "n_noise": int((~valid).sum()),
    }


def run_kmeans(features_2d, k=8):
    """PCA+KMeans on pre-flattened features."""
    scaler = StandardScaler()
    feat = scaler.fit_transform(features_2d)
    pca = PCA(n_components=min(20, feat.shape[1]))
    feat_pca = pca.fit_transform(feat)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat_pca)
    sil = silhouette_score(feat_pca, labels)
    return labels, feat_pca, sil


def run_gmm(features_2d, k=8):
    """PCA+GMM (handles non-spherical clusters)."""
    scaler = StandardScaler()
    feat = scaler.fit_transform(features_2d)
    pca = PCA(n_components=min(20, feat.shape[1]))
    feat_pca = pca.fit_transform(feat)
    gmm = GaussianMixture(n_components=k, random_state=42, covariance_type="full", max_iter=200)
    labels = gmm.fit_predict(feat_pca)
    sil = silhouette_score(feat_pca, labels) if len(set(labels)) >= 2 else 0
    return labels, feat_pca, sil


def run_bsoid(kp_3d, fps=20):
    """B-SOiD on raw centered mm-scale 3D keypoints."""
    from behavior_lab.models.discovery.bsoid import BSOiD
    model = BSOiD(fps=fps, min_cluster_size=30)
    result = model.fit_predict(kp_3d)
    labels = result.labels
    feat = kp_3d[:len(labels)].reshape(len(labels), -1)
    sil = silhouette_score(feat, labels, sample_size=min(3000, len(feat))) if len(set(labels[labels>=0])) >= 2 else 0
    return labels, feat, sil


def run_subtle(kp_3d, fps=20):
    """SUBTLE on raw centered mm-scale 3D keypoints (list input!)."""
    from behavior_lab.models.discovery.subtle_wrapper import SUBTLE
    model = SUBTLE(fps=fps, n_train_frames=min(120000, len(kp_3d)))
    result = model.fit([kp_3d])  # Must be list!
    # Result structure varies - try different access patterns
    if hasattr(result, "labels"):
        labels = result.labels
    elif isinstance(result, dict):
        if "labels" in result:
            labels = result["labels"]
        elif "subclusters" in result:
            labels = result["subclusters"]
        else:
            print("    SUBTLE result keys:", list(result.keys()) if isinstance(result, dict) else dir(result))
            return None, None, 0
    else:
        # Try to access as attribute
        for attr in ["labels", "subclusters", "cluster_labels"]:
            if hasattr(result, attr):
                labels = getattr(result, attr)
                break
        else:
            print("    SUBTLE result type:", type(result))
            return None, None, 0

    feat = kp_3d[:len(labels)].reshape(len(labels), -1)
    valid = labels >= 0
    sil = silhouette_score(feat[valid], labels[valid]) if valid.sum() > 10 and len(set(labels[valid])) >= 2 else 0
    return labels, feat, sil


def run_hmm_fallback(features_2d, k=8):
    """PCA+GaussianHMM (temporal model, captures transitions)."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("    hmmlearn not installed, skipping HMM")
        return None, None, 0

    scaler = StandardScaler()
    feat = scaler.fit_transform(features_2d)
    pca = PCA(n_components=min(10, feat.shape[1]))
    feat_pca = pca.fit_transform(feat)

    model = GaussianHMM(n_components=k, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(feat_pca)
    labels = model.predict(feat_pca)
    sil = silhouette_score(feat_pca, labels) if len(set(labels)) >= 2 else 0
    return labels, feat_pca, sil


def main():
    kp_mm, gauss_feats, frames = load_data()
    print("Data: %d frames, kp %s, gauss %s" % (len(frames), kp_mm.shape, gauss_feats.shape))

    # Prepare features for flat-input methods
    kp_flat = kp_mm.reshape(len(kp_mm), -1)  # (360, 66)
    gauss_flat = gauss_feats  # already 2D (360, 28)
    hybrid_flat = np.concatenate([
        StandardScaler().fit_transform(kp_flat),
        StandardScaler().fit_transform(gauss_flat)
    ], axis=1)

    K = 8  # Fixed K for comparison
    methods = [
        ("KMeans", run_kmeans),
        ("GMM", run_gmm),
    ]

    feature_sets = {
        "Sparse": kp_flat,
        "Dense": gauss_flat,
        "Hybrid": hybrid_flat,
    }

    results = {}

    # Standard methods (KMeans, GMM) on all feature sets
    print("\n" + "=" * 80)
    print("MULTI-METHOD COMPARISON (K=%d)" % K)
    print("=" * 80)
    print("%-12s %-10s %8s %8s %8s %6s %6s" % ("Method", "Feature", "Sil", "Bout_m", "TC", "Short%", "NBout"))
    print("-" * 65)

    for method_name, method_fn in methods:
        for feat_name, feat in feature_sets.items():
            try:
                labels, feat_pca, sil = method_fn(feat, k=K)
                tm = temporal_metrics(labels)
                key = "%s_%s" % (method_name, feat_name)
                results[key] = {"sil": round(sil, 4), **tm}
                print("%-12s %-10s %8.4f %8.3f %8.4f %6.1f %6d" % (
                    method_name, feat_name, sil, tm["bout_mean"], tm["tc"], tm["short_pct"], tm["n_bouts"]))
            except Exception as e:
                print("%-12s %-10s ERROR: %s" % (method_name, feat_name, str(e)[:50]))

    # B-SOiD (3D keypoints only - it extracts its own features)
    print("\n--- B-SOiD (3D, mm-scale) ---")
    try:
        labels_bsoid, feat_bsoid, sil_bsoid = run_bsoid(kp_mm, fps=20)
        tm = temporal_metrics(labels_bsoid)
        results["BSOID_Sparse"] = {"sil": round(sil_bsoid, 4), **tm}
        print("%-12s %-10s %8.4f %8.3f %8.4f %6.1f %6d" % (
            "B-SOiD", "Sparse(mm)", sil_bsoid, tm["bout_mean"], tm["tc"], tm["short_pct"], tm["n_bouts"]))
    except Exception as e:
        print("B-SOiD error:", str(e)[:80])
        traceback.print_exc()

    # SUBTLE (3D keypoints, list input)
    print("\n--- SUBTLE (3D, mm-scale) ---")
    try:
        labels_subtle, feat_subtle, sil_subtle = run_subtle(kp_mm, fps=20)
        if labels_subtle is not None:
            tm = temporal_metrics(labels_subtle)
            results["SUBTLE_Sparse"] = {"sil": round(sil_subtle, 4), **tm}
            print("%-12s %-10s %8.4f %8.3f %8.4f %6.1f %6d" % (
                "SUBTLE", "Sparse(mm)", sil_subtle, tm["bout_mean"], tm["tc"], tm["short_pct"], tm["n_bouts"]))
    except Exception as e:
        print("SUBTLE error:", str(e)[:80])
        traceback.print_exc()

    # HMM (temporal model)
    print("\n--- HMM (PCA+GaussianHMM) ---")
    for feat_name, feat in feature_sets.items():
        try:
            labels_hmm, feat_hmm, sil_hmm = run_hmm_fallback(feat, k=K)
            if labels_hmm is not None:
                tm = temporal_metrics(labels_hmm)
                results["HMM_%s" % feat_name] = {"sil": round(sil_hmm, 4), **tm}
                print("%-12s %-10s %8.4f %8.3f %8.4f %6.1f %6d" % (
                    "HMM", feat_name, sil_hmm, tm["bout_mean"], tm["tc"], tm["short_pct"], tm["n_bouts"]))
        except Exception as e:
            print("HMM %s error: %s" % (feat_name, str(e)[:50]))

    # Summary: check if multiple methods show consistent pattern
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK: Dense/Hybrid temporal advantage across methods")
    print("=" * 80)

    for method_base in ["KMeans", "GMM", "HMM"]:
        sp_key = "%s_Sparse" % method_base
        hy_key = "%s_Hybrid" % method_base
        if sp_key in results and hy_key in results:
            sp = results[sp_key]
            hy = results[hy_key]
            sil_delta = hy["sil"] - sp["sil"]
            bout_delta = hy["bout_mean"] - sp["bout_mean"]
            tc_delta = hy["tc"] - sp["tc"]
            short_delta = hy["short_pct"] - sp["short_pct"]
            print("%s:" % method_base)
            print("  Sil: Sparse=%.4f Hybrid=%.4f (delta=%+.4f)" % (sp["sil"], hy["sil"], sil_delta))
            print("  Bout: Sparse=%.3f Hybrid=%.3f (delta=%+.3f %s)" % (
                sp["bout_mean"], hy["bout_mean"], bout_delta,
                "HYBRID LONGER" if bout_delta > 0.05 else "SIMILAR"))
            print("  TC: Sparse=%.4f Hybrid=%.4f (delta=%+.4f)" % (sp["tc"], hy["tc"], tc_delta))
            print("  Short%%: Sparse=%.1f Hybrid=%.1f (delta=%+.1f %s)" % (
                sp["short_pct"], hy["short_pct"], short_delta,
                "HYBRID FEWER" if short_delta < -3 else "SIMILAR"))

    # Save
    out = Path("outputs/clustering/multi_method")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "multi_method_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved:", out / "multi_method_results.json")


if __name__ == "__main__":
    main()
