"""Deduplicated Gaussian feature comparison."""
from plyfile import PlyData
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_rich(p):
    ply = PlyData.read(str(p))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
    sh = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    op = np.array(v["opacity"])
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    f = []
    f.extend(xyz.mean(0)); f.extend(xyz.std(0))
    f.extend(np.percentile(xyz, 25, axis=0)); f.extend(np.percentile(xyz, 75, axis=0))
    f.extend(rgb.mean(0)); f.extend(rgb.std(0)); f.extend(sh.mean(0))
    f.extend([op.mean(), op.std(), np.median(op)])
    f.extend(sc.mean(0)); f.extend(sc.std(0))
    f.append(sc[:,0].mean() - sc[:,2].mean())
    f.extend(np.percentile(sc, [25, 75], axis=0).mean(1))
    f.extend(rot.mean(0)); f.extend(rot.std(0))
    xyz_c = xyz - xyz.mean(0)
    eigvals = np.linalg.eigvalsh(np.cov(xyz_c.T))
    f.extend(sorted(eigvals, reverse=True))
    f.extend(xyz.max(0) - xyz.min(0))
    f.append(float(len(xyz)))
    return np.array(f, dtype=np.float32)

# Deduplicate: use only one experiment path
plys = sorted(Path("outputs/experiments/phase3_e2e/H3_resume_pose").rglob("gaussians.ply"))
print("PLY files (single experiment):", len(plys))

frame_ply = {}
for p in plys:
    for part in p.parts:
        if part.isdigit() and len(part) == 6:
            fi = int(part)
            if fi not in frame_ply:
                frame_ply[fi] = p
            break

frames_sorted = sorted(frame_ply.keys())
print("Unique frames:", len(frames_sorted), "range:", frames_sorted[0], "-", frames_sorted[-1])

features = np.stack([load_rich(frame_ply[f]) for f in frames_sorted])
frames_arr = np.array(frames_sorted)
print("Features:", features.shape)

scaler = StandardScaler()
feat_s = scaler.fit_transform(features)
feat_pca = PCA(n_components=min(30, feat_s.shape[1])).fit_transform(feat_s)

# Sparse on same frames
kp = np.load("/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
             allow_pickle=True)["keypoints"]
kp_sub = kp[frames_arr]
kp_c = kp_sub - kp_sub.mean(1, keepdims=True)
kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
kp_pca = PCA(n_components=20).fit_transform(kp_flat)

print()
header = "%-35s %3s %8s %10s" % ("Method", "K", "Sil", "delta")
print(header)
print("-" * 60)

sparse_sils = {}
for k in [4, 6, 8, 10, 12]:
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kp_pca)
    s = silhouette_score(kp_pca, l)
    sparse_sils[k] = s
    print("%-35s %3d %8.4f %10s" % ("Sparse K=%d" % k, k, s, "BASE"))

base = sparse_sils[4]
for k in [4, 6, 8, 10, 12]:
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat_pca)
    s = silhouette_score(feat_pca, l)
    d = s - base
    v = "BETTER" if d > 0.01 else "SAME" if abs(d) <= 0.01 else "WORSE"
    print("%-35s %3d %8.4f %+10.4f %s" % ("3D Gaussian rich K=%d" % k, k, s, d, v))

hybrid = np.concatenate([kp_flat, feat_s], axis=1)
hybrid_pca = PCA(n_components=30).fit_transform(hybrid)
for k in [4, 6, 8, 10, 12]:
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(hybrid_pca)
    s = silhouette_score(hybrid_pca, l)
    d = s - base
    v = "BETTER" if d > 0.01 else "SAME" if abs(d) <= 0.01 else "WORSE"
    print("%-35s %3d %8.4f %+10.4f %s" % ("Hybrid K=%d" % k, k, s, d, v))
