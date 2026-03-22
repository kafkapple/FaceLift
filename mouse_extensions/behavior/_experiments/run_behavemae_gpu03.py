"""hBehaveMAE feature extraction + clustering on gpu03."""
import numpy as np, sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "/home/joon/dev/behavior-lab/src")
sys.path.insert(0, "/home/joon/dev/behavior-lab/external/BehaveMAE")

from behavior_lab.models.discovery.behavemae import BehaveMAE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load model
print("Loading hBehaveMAE...")
model = BehaveMAE.from_pretrained(
    "/home/joon/dev/behavior-lab/checkpoints/hBehaveMAE_MABe22.pth",
    dataset="mabe22", device="cuda"
)
print("Model loaded on GPU")

# Load keypoints
from mouse_extensions.paths import KP_22
kp = np.load(str(KP_22),
             allow_pickle=True)["keypoints"]
valid_mask = np.ones(3600, dtype=bool)
for idx in [1180, 2360, 3540]:
    valid_mask[max(0,idx-2):min(3600,idx+3)] = False
kp = kp[valid_mask]
kp_c = kp - kp.mean(1, keepdims=True)
N = len(kp_c)

# Format for MABe22: (T, 3, 24) — pack first 12 joints XY
kp_2d = kp_c[:, :, :2]  # (3585, 22, 2)
kp_flat = kp_2d.reshape(N, -1)  # (3585, 44)
kp_mabe = np.zeros((N, 3, 24), dtype=np.float32)
kp_mabe[:, 0, :24] = kp_flat[:, :24]

# Extract per-frame features using sliding window
print("Extracting per-frame features (%d frames)..." % N)
window = 900
frame_features = np.zeros((N, 256), dtype=np.float32)
counts = np.zeros(N, dtype=np.float32)
t0 = time.time()

for start in range(0, N - window + 1, window // 4):  # 75% overlap
    end = start + window
    chunk = kp_mabe[start:end]
    feat = model.encode(chunk)  # (60, 256)
    token_size = window // feat.shape[0]  # 15
    for ti in range(feat.shape[0]):
        fs = start + ti * token_size
        fe = min(fs + token_size, N)
        frame_features[fs:fe] += feat[ti]
        counts[fs:fe] += 1
    if start % (window * 2) == 0:
        print("  start=%d/%d (%.1fs)" % (start, N, time.time() - t0))

valid_c = counts > 0
frame_features[valid_c] /= counts[valid_c, np.newaxis]
frame_features[~valid_c] = frame_features[valid_c].mean(0)
print("Features: %s (%.1fs)" % (str(frame_features.shape), time.time() - t0))

# Save features
np.save("outputs/clustering/behavemae_features.npy", frame_features)

# Cluster and compare
scaler = StandardScaler()
feat_s = scaler.fit_transform(frame_features)
feat_pca = PCA(n_components=min(30, feat_s.shape[1])).fit_transform(feat_s)

kp_flat_all = StandardScaler().fit_transform(kp_c.reshape(N, -1))
kp_pca = PCA(n_components=20).fit_transform(kp_flat_all)

hybrid = np.concatenate([kp_pca, feat_pca], axis=1)

print()
print("%-30s %4s %8s %8s %8s %6s" % ("Method", "K", "Sil", "Bout", "TC", "Short%"))
print("-" * 65)

for feat_name, feat in [("Sparse_keypoint", kp_pca), ("hBehaveMAE", feat_pca),
                         ("Hybrid_kp+MAE", hybrid)]:
    for k in [4, 8]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat)
        sil = silhouette_score(feat, labels)
        changes = np.where(np.diff(labels) != 0)[0]
        bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / 20.0
        tc = 1.0 - len(changes) / (len(labels) - 1)
        short = (bouts < 0.3).mean() * 100
        print("%-30s %4d %8.4f %8.3f %8.4f %6.1f" % (
            feat_name + "_K%d" % k, k, sil, bouts.mean(), tc, short))

# GMM too
for feat_name, feat in [("Sparse_GMM", kp_pca), ("hBehaveMAE_GMM", feat_pca),
                         ("Hybrid_GMM", hybrid)]:
    labels = GaussianMixture(n_components=8, random_state=42).fit_predict(feat)
    sil = silhouette_score(feat, labels) if len(set(labels)) >= 2 else 0
    changes = np.where(np.diff(labels) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / 20.0
    tc = 1.0 - len(changes) / (len(labels) - 1)
    short = (bouts < 0.3).mean() * 100
    print("%-30s %4d %8.4f %8.3f %8.4f %6.1f" % (
        feat_name + "_K8", 8, sil, bouts.mean(), tc, short))
