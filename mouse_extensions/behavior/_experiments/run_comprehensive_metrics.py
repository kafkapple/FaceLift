"""Run comprehensive metrics on all existing results + significance tests."""
import numpy as np, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")

from mouse_extensions.behavior.metrics import (
    compute_all_metrics, permutation_test_two_methods
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
kp_all = np.load("/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
                 allow_pickle=True)["keypoints"]

# Test set frames (where PLY exists)
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
frames = np.array(sorted(frame_ply.keys()))
print("Frames:", len(frames))

# Prepare sparse features
kp = kp_all[frames]
kp_c = kp - kp.mean(1, keepdims=True)
kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
kp_pca = PCA(n_components=20).fit_transform(kp_flat)

# Prepare Gaussian features
def load_gauss(ply_path):
    ply = PlyData.read(ply_path)
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

# Hybrid
hybrid = np.concatenate([kp_pca, gauss_pca], axis=1)

# Cluster all methods
results = {}
for name, feat in [("Sparse", kp_pca), ("Dense_Gauss", gauss_pca), ("Hybrid", hybrid)]:
    for k in [4, 8]:
        key = "%s_K%d" % (name, k)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat)
        print("\n--- %s ---" % key)
        m = compute_all_metrics(labels, feat, fps=20.0, bootstrap_n=50, compute_significance=True)
        results[key] = m.to_dict()
        print("  Sil=%.4f [%.4f, %.4f]" % (m.silhouette, m.silhouette_ci_lower, m.silhouette_ci_upper))
        print("  CH=%.1f DB=%.4f" % (m.calinski_harabasz, m.davies_bouldin))
        print("  Bout=%.3fs TPI=%.1f Entropy=%.3f TC=%.4f" % (
            m.bout_mean_sec, m.tpi_mean, m.entropy_rate, m.temporal_consistency))
        print("  p_value_vs_shuffle=%.4f" % m.p_value_vs_shuffle)

# Significance tests: Sparse vs Hybrid at K=8
print("\n" + "=" * 60)
print("SIGNIFICANCE TESTS: Sparse_K8 vs Hybrid_K8")
print("=" * 60)

lab_sp = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(kp_pca)
lab_hy = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(hybrid)

for metric in ["silhouette", "temporal_consistency", "tpi"]:
    r = permutation_test_two_methods(lab_sp, lab_hy, kp_pca, hybrid, metric=metric, n_permutations=500)
    sig = "***" if r["significant_001"] else "**" if r["significant_005"] else "ns"
    print("  %s: Sparse=%.4f Hybrid=%.4f diff=%+.4f p=%.4f %s" % (
        metric, r["method_a"], r["method_b"], r["observed_diff"], r["p_value"], sig))

# Save
out = Path("outputs/clustering/comprehensive_metrics")
out.mkdir(parents=True, exist_ok=True)
with open(out / "all_metrics.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved:", out / "all_metrics.json")
