"""Fix SUBTLE label length mismatch + test supercluster on full dataset."""
import numpy as np, sys
sys.path.insert(0, ".")

from mouse_extensions.paths import KP_22
kp = np.load(str(KP_22),
             allow_pickle=True)["keypoints"]
valid_mask = np.ones(3600, dtype=bool)
for idx in [1180, 2360, 3540]:
    valid_mask[max(0,idx-2):min(3600,idx+3)] = False
kp = kp[valid_mask]
kp_c = kp - kp.mean(1, keepdims=True)
N = len(kp_c)

from behavior_lab.models.discovery.subtle_wrapper import SUBTLE
model = SUBTLE(fps=20, n_train_frames=min(120000, N))
result = model.fit_predict([kp_c], use_superclusters=True)

raw_labels = result.labels
print(f"Raw labels: {raw_labels.shape}, kp frames: {N}")

# Downsample labels to match keypoint frames
if len(raw_labels) != N:
    ratio = len(raw_labels) / N
    print(f"Ratio: {ratio:.2f}x — downsampling via strided indexing")
    # Use strided indexing (every ratio-th label)
    indices = np.linspace(0, len(raw_labels)-1, N, dtype=int)
    labels = raw_labels[indices]
else:
    labels = raw_labels

print(f"Downsampled labels: {labels.shape}")
print(f"Unique: {sorted(set(labels))}, K={len(set(labels))}")

# Temporal metrics
from sklearn.metrics import silhouette_score
changes = np.where(np.diff(labels) != 0)[0]
bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / 20.0
tc = 1.0 - len(changes) / (len(labels) - 1)
short_pct = (bouts < 0.3).mean() * 100

feat = kp_c.reshape(N, -1)
sil = silhouette_score(feat, labels, sample_size=min(3000, N)) if len(set(labels)) >= 2 else 0

print(f"\nSUBTLE Superclusters (full {N}f, downsampled):")
print(f"  K={len(set(labels))}, Sil={sil:.4f}, Bout={bouts.mean():.3f}s, TC={tc:.4f}, Short%={short_pct:.1f}")

# Also get subclusters for comparison
sub_result = model.fit_predict([kp_c], use_superclusters=False)
sub_raw = sub_result.labels
if len(sub_raw) != N:
    sub_labels = sub_raw[np.linspace(0, len(sub_raw)-1, N, dtype=int)]
else:
    sub_labels = sub_raw
sub_changes = np.where(np.diff(sub_labels) != 0)[0]
sub_bouts = np.diff(np.concatenate([[0], sub_changes + 1, [len(sub_labels)]])) / 20.0
sub_tc = 1.0 - len(sub_changes) / (len(sub_labels) - 1)
sub_sil = silhouette_score(feat, sub_labels, sample_size=min(3000, N)) if len(set(sub_labels)) >= 2 else 0

print(f"\nSUBTLE Subclusters (full {N}f, downsampled):")
print(f"  K={len(set(sub_labels))}, Sil={sub_sil:.4f}, Bout={sub_bouts.mean():.3f}s, TC={sub_tc:.4f}, Short%={(sub_bouts<0.3).mean()*100:.1f}")

# Save labels
np.save("outputs/clustering/full_dataset/subtle_super_labels.npy", labels)
np.save("outputs/clustering/full_dataset/subtle_sub_labels.npy", sub_labels)
print("\nLabels saved.")
