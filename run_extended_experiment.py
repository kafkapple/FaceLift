"""Extended experiment: K sweep + BayesianGMM auto-K + cluster interpretation.

Runs on test set (360 frames) with Sparse, Dense, Hybrid features.
Adds: wider K range, DPGMM auto-K, per-cluster kinematic stats.
"""
import numpy as np, json, sys, time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
sys.path.insert(0, ".")

KEYPOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder", "L_foot", "L_knee",
    "L_hip", "R_foot", "R_knee", "R_hip",
]

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
    kp_c = kp - kp.mean(1, keepdims=True)
    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)

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
    gauss_s = StandardScaler().fit_transform(gauss_feats)
    gauss_pca = PCA(n_components=min(15, gauss_s.shape[1])).fit_transform(gauss_s)
    hybrid = np.concatenate([kp_pca, gauss_pca], axis=1)
    return kp_c, kp_pca, gauss_pca, hybrid, frames

def temporal_metrics(labels, fps=20.0):
    changes = np.where(np.diff(labels) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / fps
    tc = 1.0 - len(changes) / (len(labels) - 1)
    short_pct = (bouts < 0.3).mean() * 100 if len(bouts) > 0 else 0
    return {
        "bout_mean": round(float(bouts.mean()), 3) if len(bouts) > 0 else 0,
        "bout_median": round(float(np.median(bouts)), 3) if len(bouts) > 0 else 0,
        "n_bouts": int(len(bouts)),
        "tc": round(float(tc), 4),
        "short_bout_pct": round(float(short_pct), 1),
    }

def run_k_sweep():
    kp_c, kp_pca, gauss_pca, hybrid, frames = load_data()
    print("Data loaded: %d frames" % len(frames))

    K_VALUES = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25]
    features_dict = {"Sparse": kp_pca, "Dense": gauss_pca, "Hybrid": hybrid}

    # K sweep
    print("\n" + "=" * 80)
    print("K SWEEP (KMeans)")
    print("=" * 80)
    header = "%-8s %-8s %4s %8s %8s %8s %8s %8s %6s" % (
        "Feature", "K", "Sil", "CH", "DB", "Bout_m", "TC", "Short%", "NBout")
    print(header)
    print("-" * 80)

    results = {}
    for fname, feat in features_dict.items():
        for k in K_VALUES:
            if k >= len(feat):
                continue
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat)
            sil = silhouette_score(feat, labels)
            ch = calinski_harabasz_score(feat, labels)
            db = davies_bouldin_score(feat, labels)
            tm = temporal_metrics(labels)
            key = "%s_K%d" % (fname, k)
            results[key] = {"sil": round(sil, 4), "ch": round(ch, 1), "db": round(db, 4), **tm}
            print("%-8s %-8d %8.4f %8.1f %8.4f %8.3f %8.4f %6.1f %6d" % (
                fname, k, sil, ch, db, tm["bout_mean"], tm["tc"], tm["short_bout_pct"], tm["n_bouts"]))

    # BayesianGMM (auto-K)
    print("\n" + "=" * 80)
    print("BAYESIAN GMM (Auto-K via Dirichlet Process)")
    print("=" * 80)
    for fname, feat in features_dict.items():
        bgm = BayesianGaussianMixture(
            n_components=25, weight_concentration_prior=0.1,
            covariance_type="full", random_state=42, max_iter=500,
        )
        labels = bgm.fit_predict(feat)
        eff_k = len(set(labels))
        weights = bgm.weights_
        active = (weights > 0.01).sum()
        sil = silhouette_score(feat, labels) if eff_k >= 2 else 0
        tm = temporal_metrics(labels)
        print("%-8s: effective K=%d (active=%d), Sil=%.4f, bout=%.3fs, TC=%.4f, short%%=%.1f" % (
            fname, eff_k, active, sil, tm["bout_mean"], tm["tc"], tm["short_bout_pct"]))
        results["%s_DPGMM" % fname] = {
            "method": "BayesianGMM", "effective_k": eff_k, "active_k": int(active),
            "sil": round(sil, 4), **tm,
        }

    # Per-cluster kinematic stats (Sparse K=4, for cluster interpretation)
    print("\n" + "=" * 80)
    print("CLUSTER INTERPRETATION (Sparse K=4)")
    print("=" * 80)
    labels_k4 = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(kp_pca)
    for c in range(4):
        mask = labels_k4 == c
        kp_cluster = kp_c[mask]  # (n, 22, 3)
        n = mask.sum()
        # Kinematic features
        nose_tail = np.linalg.norm(kp_cluster[:, 2] - kp_cluster[:, 5], axis=1)
        com_height = kp_cluster.mean(axis=1)[:, 2]  # Z coordinate of COM
        speed = np.linalg.norm(np.diff(kp_cluster.mean(axis=1), axis=0), axis=1) if n > 1 else [0]
        paw_height = kp_cluster[:, [8, 12, 16, 19], 2].mean(axis=1)  # avg paw Z

        print("  Cluster %d (%d frames, %.1f%%):" % (c, n, n/len(labels_k4)*100))
        print("    nose-tail dist: %.1f +/- %.1f mm (body extension)" % (nose_tail.mean(), nose_tail.std()))
        print("    COM height (Z): %.1f +/- %.1f mm (rearing indicator)" % (com_height.mean(), com_height.std()))
        print("    speed (COM):    %.2f +/- %.2f mm/frame" % (np.mean(speed), np.std(speed)))
        print("    paw height (Z): %.1f +/- %.1f mm" % (paw_height.mean(), paw_height.std()))

        cluster_info = {
            "n_frames": int(n), "pct": round(n/len(labels_k4)*100, 1),
            "nose_tail_mean": round(float(nose_tail.mean()), 1),
            "com_height_mean": round(float(com_height.mean()), 1),
            "speed_mean": round(float(np.mean(speed)), 2),
            "paw_height_mean": round(float(paw_height.mean()), 1),
        }
        results["cluster_%d_kinematics" % c] = cluster_info

    # Save
    out = Path("outputs/clustering/extended")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "extended_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved:", out / "extended_results.json")

if __name__ == "__main__":
    run_k_sweep()
