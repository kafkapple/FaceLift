from plyfile import PlyData
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_rich_features(ply_path):
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    N = len(v.data)
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
    sh_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    opacity = np.array(v["opacity"])
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
    feats = []
    feats.extend(xyz.mean(0)); feats.extend(xyz.std(0))
    feats.extend(np.percentile(xyz, 25, axis=0)); feats.extend(np.percentile(xyz, 75, axis=0))
    feats.extend(rgb.mean(0)); feats.extend(rgb.std(0)); feats.extend(sh_dc.mean(0))
    feats.extend([opacity.mean(), opacity.std(), np.median(opacity)])
    feats.extend(scale.mean(0)); feats.extend(scale.std(0))
    feats.append(scale[:,0].mean() - scale[:,2].mean())
    feats.extend(np.percentile(scale, [25, 75], axis=0).mean(1))
    feats.extend(rot.mean(0)); feats.extend(rot.std(0))
    xyz_c = xyz - xyz.mean(0)
    eigvals = np.linalg.eigvalsh(np.cov(xyz_c.T))
    feats.extend(sorted(eigvals, reverse=True))
    feats.extend(xyz.max(0) - xyz.min(0))
    feats.append(float(N))
    return np.array(feats, dtype=np.float32)

plys = sorted(Path("outputs").rglob("gaussians.ply"))
print("PLY files:", len(plys))
frames, features = [], []
for ply in plys:
    for part in ply.parts:
        if part.isdigit() and len(part) == 6:
            features.append(load_rich_features(ply))
            frames.append(int(part))
            break
features = np.stack(features)
frames = np.array(frames)
print("Rich features:", features.shape, "dim =", features.shape[1])

scaler = StandardScaler()
feat_s = scaler.fit_transform(features)
feat_pca = PCA(n_components=min(30, feat_s.shape[1])).fit_transform(feat_s)

kp = np.load("/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
             allow_pickle=True)["keypoints"]
kp_test = kp[frames]
kp_c = kp_test - kp_test.mean(1, keepdims=True)
kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
kp_pca = PCA(n_components=20).fit_transform(kp_flat)

print("\n%-35s %3s %8s %10s" % ("Method", "K", "Sil", "delta"))
print("-" * 60)

sparse_sils = {}
for k in [4, 6, 8]:
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kp_pca)
    s = silhouette_score(kp_pca, l, sample_size=min(500, len(kp_pca)))
    sparse_sils[k] = s
    print("%-35s %3d %8.4f %10s" % ("Sparse K=%d" % k, k, s, "BASE"))

sparse_k4 = sparse_sils[4]

for k in [4, 6, 8, 10, 12]:
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat_pca)
    s = silhouette_score(feat_pca, l, sample_size=min(500, len(feat_pca)))
    d = s - sparse_k4
    v = "BETTER" if d > 0.01 else "SAME" if abs(d) <= 0.01 else "WORSE"
    print("%-35s %3d %8.4f %+10.4f %s" % ("3D Gaussian rich K=%d" % k, k, s, d, v))

hybrid = np.concatenate([kp_flat, feat_s], axis=1)
hybrid_pca = PCA(n_components=30).fit_transform(hybrid)
for k in [4, 6, 8]:
    l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(hybrid_pca)
    s = silhouette_score(hybrid_pca, l, sample_size=min(500, len(hybrid_pca)))
    d = s - sparse_k4
    v = "BETTER" if d > 0.01 else "SAME" if abs(d) <= 0.01 else "WORSE"
    print("%-35s %3d %8.4f %+10.4f %s" % ("Hybrid sparse+gauss K=%d" % k, k, s, d, v))
