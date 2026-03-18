"""Full dataset multi-method experiment.

Phase 1: Sparse keypoints on ALL 3585 frames (B-SOiD, SUBTLE, KMeans, GMM, HMM)
Phase 2: Dense/Hybrid on 360 test frames (same methods for comparison)
Phase 3: Sparse on SAME 360 frames (fair comparison with Phase 2)
"""
import numpy as np, json, sys, time, traceback
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
sys.path.insert(0, ".")

def temporal_metrics(labels, fps=20.0):
    valid = labels >= 0
    lab = labels[valid]
    if len(lab) < 2:
        return {"bout_mean": 0, "tc": 0, "short_pct": 0, "n_bouts": 0, "n_noise": int((~valid).sum())}
    changes = np.where(np.diff(lab) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(lab)]])) / fps
    tc = 1.0 - len(changes) / (len(lab) - 1)
    return {
        "bout_mean": round(float(bouts.mean()), 3),
        "bout_median": round(float(np.median(bouts)), 3),
        "tc": round(float(tc), 4),
        "short_pct": round(float((bouts < 0.3).mean() * 100), 1),
        "n_bouts": int(len(bouts)),
        "n_noise": int((~valid).sum()),
        "n_clusters": int(len(set(lab))),
    }

def run_method(name, feat_or_kp, k=8, is_3d_kp=False):
    """Run a single clustering method. Returns (labels, features_2d, sil)."""
    try:
        if name == "KMeans":
            s = StandardScaler().fit_transform(feat_or_kp.reshape(len(feat_or_kp), -1))
            p = PCA(n_components=min(20, s.shape[1])).fit_transform(s)
            l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(p)
            return l, p, silhouette_score(p, l)

        elif name == "GMM":
            s = StandardScaler().fit_transform(feat_or_kp.reshape(len(feat_or_kp), -1))
            p = PCA(n_components=min(20, s.shape[1])).fit_transform(s)
            l = GaussianMixture(n_components=k, random_state=42, covariance_type="full").fit_predict(p)
            return l, p, silhouette_score(p, l) if len(set(l)) >= 2 else 0

        elif name == "HMM":
            from hmmlearn.hmm import GaussianHMM
            s = StandardScaler().fit_transform(feat_or_kp.reshape(len(feat_or_kp), -1))
            p = PCA(n_components=min(10, s.shape[1])).fit_transform(s)
            m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=100, random_state=42)
            m.fit(p)
            l = m.predict(p)
            return l, p, silhouette_score(p, l) if len(set(l)) >= 2 else 0

        elif name == "B-SOiD":
            assert is_3d_kp, "B-SOiD needs 3D keypoints"
            from behavior_lab.models.discovery.bsoid import BSOiD
            model = BSOiD(fps=20, min_cluster_size=30)
            result = model.fit_predict(feat_or_kp)
            l = result.labels
            f = feat_or_kp[:len(l)].reshape(len(l), -1)
            valid = l >= 0
            sil = silhouette_score(f[valid], l[valid]) if valid.sum() > 10 and len(set(l[valid])) >= 2 else 0
            return l, f, sil

        elif name == "SUBTLE":
            assert is_3d_kp, "SUBTLE needs 3D keypoints"
            from behavior_lab.models.discovery.subtle_wrapper import SUBTLE
            model = SUBTLE(fps=20, n_train_frames=min(120000, len(feat_or_kp)))
            result = model.fit([feat_or_kp])
            # Extract labels from result
            if hasattr(result, "labels"):
                l = result.labels
            elif isinstance(result, dict):
                l = result.get("labels", result.get("subclusters", None))
                if l is None:
                    # Try ClusteringResult
                    for attr in ["labels", "subclusters"]:
                        if attr in result:
                            l = result[attr]
                            break
            else:
                l = getattr(result, "labels", getattr(result, "subclusters", None))

            if l is None:
                print("    SUBTLE: could not extract labels from result type", type(result))
                return None, None, 0

            f = feat_or_kp[:len(l)].reshape(len(l), -1)
            valid = l >= 0
            sil = silhouette_score(f[valid], l[valid]) if valid.sum() > 10 and len(set(l[valid])) >= 2 else 0
            return l, f, sil

    except Exception as e:
        print("    %s ERROR: %s" % (name, str(e)[:80]))
        return None, None, 0


def main():
    # Load all data
    print("Loading data...")
    kp_all = np.load("/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
                     allow_pickle=True)["keypoints"]

    # Frame jump masking
    valid_mask = np.ones(3600, dtype=bool)
    for idx in [1180, 2360, 3540]:
        valid_mask[max(0,idx-2):min(3600,idx+3)] = False
    kp_full = kp_all[valid_mask]  # (3585, 22, 3)
    kp_full_centered = kp_full - kp_full.mean(1, keepdims=True)

    # Gaussian features (test set only)
    from plyfile import PlyData
    plys = sorted(Path("outputs/experiments/phase3_e2e/H3_resume_pose").rglob("gaussians.ply"))
    frame_ply = {}
    for p in plys:
        for part in p.parts:
            if part.isdigit() and len(part) == 6:
                fi = int(part)
                if fi not in frame_ply:
                    frame_ply[fi] = str(p)
                break
    test_frames = np.array(sorted(frame_ply.keys()))

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

    gauss_feats = np.stack([load_gauss(frame_ply[f]) for f in test_frames])

    # Test set keypoints (same 360 frames)
    kp_test = kp_all[test_frames]
    kp_test_centered = kp_test - kp_test.mean(1, keepdims=True)

    print("Full dataset: %d frames" % len(kp_full))
    print("Test set: %d frames (for dense/hybrid comparison)" % len(test_frames))

    methods_flat = ["KMeans", "GMM", "HMM"]
    methods_3d = ["B-SOiD", "SUBTLE"]
    K = 8
    results = {}

    # ==========================================
    # PHASE 1: Sparse on FULL dataset (3585 frames)
    # ==========================================
    print("\n" + "=" * 80)
    print("PHASE 1: Sparse on FULL dataset (%d frames)" % len(kp_full))
    print("=" * 80)
    print("%-12s %8s %8s %8s %6s %6s %6s" % ("Method", "Sil", "Bout_m", "TC", "Short%", "K_eff", "Noise"))
    print("-" * 60)

    for mname in methods_flat + methods_3d:
        is_3d = mname in methods_3d
        data = kp_full_centered if is_3d else kp_full_centered.reshape(len(kp_full_centered), -1)
        t0 = time.time()
        labels, feat, sil = run_method(mname, data, k=K, is_3d_kp=is_3d)
        elapsed = time.time() - t0
        if labels is not None:
            tm = temporal_metrics(labels)
            key = "full_%s_Sparse" % mname
            results[key] = {"sil": round(sil, 4), **tm, "time_sec": round(elapsed, 1)}
            print("%-12s %8.4f %8.3f %8.4f %6.1f %6d %6d (%.1fs)" % (
                mname, sil, tm["bout_mean"], tm["tc"], tm["short_pct"], tm["n_clusters"], tm["n_noise"], elapsed))

    # ==========================================
    # PHASE 2+3: Dense/Hybrid/Sparse on TEST set (360 frames, fair comparison)
    # ==========================================
    print("\n" + "=" * 80)
    print("PHASE 2+3: All features on TEST set (%d frames, fair comparison)" % len(test_frames))
    print("=" * 80)

    kp_flat = kp_test_centered.reshape(len(kp_test_centered), -1)
    gauss_flat = gauss_feats
    hybrid_flat = np.concatenate([
        StandardScaler().fit_transform(kp_flat),
        StandardScaler().fit_transform(gauss_flat)
    ], axis=1)

    feature_sets = {
        "Sparse": (kp_flat, kp_test_centered),
        "Dense": (gauss_flat, None),
        "Hybrid": (hybrid_flat, None),
    }

    print("%-12s %-8s %8s %8s %8s %6s %6s %6s" % ("Method", "Feature", "Sil", "Bout_m", "TC", "Short%", "K_eff", "Noise"))
    print("-" * 70)

    for mname in methods_flat:
        for fname, (feat_flat, kp_3d) in feature_sets.items():
            t0 = time.time()
            labels, feat, sil = run_method(mname, feat_flat, k=K)
            elapsed = time.time() - t0
            if labels is not None:
                tm = temporal_metrics(labels)
                key = "test_%s_%s" % (mname, fname)
                results[key] = {"sil": round(sil, 4), **tm, "time_sec": round(elapsed, 1)}
                print("%-12s %-8s %8.4f %8.3f %8.4f %6.1f %6d %6d" % (
                    mname, fname, sil, tm["bout_mean"], tm["tc"], tm["short_pct"], tm["n_clusters"], tm["n_noise"]))

    # B-SOiD on test set sparse (3D keypoints)
    for mname in methods_3d:
        t0 = time.time()
        labels, feat, sil = run_method(mname, kp_test_centered, k=K, is_3d_kp=True)
        elapsed = time.time() - t0
        if labels is not None:
            tm = temporal_metrics(labels)
            key = "test_%s_Sparse" % mname
            results[key] = {"sil": round(sil, 4), **tm, "time_sec": round(elapsed, 1)}
            print("%-12s %-8s %8.4f %8.3f %8.4f %6.1f %6d %6d" % (
                mname, "Sparse", sil, tm["bout_mean"], tm["tc"], tm["short_pct"], tm["n_clusters"], tm["n_noise"]))

    # ==========================================
    # CONSISTENCY CHECK
    # ==========================================
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK: Hybrid temporal advantage (test set)")
    print("=" * 80)
    for mname in methods_flat:
        sp_key = "test_%s_Sparse" % mname
        hy_key = "test_%s_Hybrid" % mname
        if sp_key in results and hy_key in results:
            sp, hy = results[sp_key], results[hy_key]
            print("%s: Bout %+.3fs | TC %+.4f | Short%% %+.1fpp | Sil %+.4f" % (
                mname,
                hy["bout_mean"] - sp["bout_mean"],
                hy["tc"] - sp["tc"],
                hy["short_pct"] - sp["short_pct"],
                hy["sil"] - sp["sil"],
            ))

    # Save
    out = Path("outputs/clustering/full_dataset")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "full_dataset_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved:", out / "full_dataset_results.json")


if __name__ == "__main__":
    main()
