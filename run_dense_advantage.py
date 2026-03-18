"""Find specific scenarios where dense features outperform sparse.

Strategy 1: Sub-clustering within sparse clusters (dense discovers subtypes)
Strategy 2: Temporal consistency comparison (dense more stable bouts?)
Strategy 3: Shape encoding (body tension/posture subtypes via Gaussian shape)
"""
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

def load_data():
    # Sparse keypoints (test set frames)
    kp = np.load("/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
                 allow_pickle=True)["keypoints"]

    # Gaussian features
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

    frames = sorted(frame_ply.keys())
    # Load Gaussian features
    gauss_feats = []
    for fi in frames:
        ply = PlyData.read(frame_ply[fi])
        v = ply["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
        op = np.array(v["opacity"])
        f = []
        f.extend(xyz.mean(0)); f.extend(xyz.std(0))
        f.extend(np.percentile(xyz, [25, 75], axis=0).flatten())
        f.extend([op.mean(), op.std(), np.median(op)])
        f.extend(sc.mean(0)); f.extend(sc.std(0))
        f.append(sc[:,0].mean() - sc[:,2].mean())
        xyz_c = xyz - xyz.mean(0)
        eigvals = np.linalg.eigvalsh(np.cov(xyz_c.T))
        f.extend(sorted(eigvals, reverse=True))
        f.extend(xyz.max(0) - xyz.min(0))
        gauss_feats.append(np.array(f, dtype=np.float32))

    frames = np.array(frames)
    gauss_feats = np.stack(gauss_feats)

    kp_sub = kp[frames]
    kp_c = kp_sub - kp_sub.mean(1, keepdims=True)

    return kp_c, gauss_feats, frames


def strategy1_subclustering(kp_c, gauss_feats):
    """Sub-cluster within sparse clusters using dense features."""
    print("\n" + "=" * 60)
    print("Strategy 1: Sub-clustering (dense discovers subtypes within sparse clusters)")
    print("=" * 60)

    # First: sparse clustering K=4
    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)
    sparse_labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(kp_pca)

    # For each sparse cluster, sub-cluster with sparse vs dense
    gauss_s = StandardScaler().fit_transform(gauss_feats)
    gauss_pca = PCA(n_components=min(15, gauss_feats.shape[1])).fit_transform(gauss_s)

    print("\nPer sparse cluster sub-clustering (K_sub=3):")
    print("%-10s %6s %12s %12s %12s" % ("Cluster", "Size", "Sparse Sub", "Dense Sub", "Hybrid Sub"))
    print("-" * 55)

    results = []
    for c in range(4):
        mask = sparse_labels == c
        n = mask.sum()
        if n < 20:
            continue

        # Sub-cluster with sparse
        sub_kp = kp_pca[mask]
        k_sub = min(3, n // 5)
        if k_sub < 2:
            continue
        l_sp = KMeans(n_clusters=k_sub, random_state=42, n_init=5).fit_predict(sub_kp)
        sil_sp = silhouette_score(sub_kp, l_sp) if len(set(l_sp)) >= 2 else 0

        # Sub-cluster with dense
        sub_gs = gauss_pca[mask]
        l_gs = KMeans(n_clusters=k_sub, random_state=42, n_init=5).fit_predict(sub_gs)
        sil_gs = silhouette_score(sub_gs, l_gs) if len(set(l_gs)) >= 2 else 0

        # Sub-cluster with hybrid
        sub_hybrid = np.concatenate([sub_kp, sub_gs], axis=1)
        l_hy = KMeans(n_clusters=k_sub, random_state=42, n_init=5).fit_predict(sub_hybrid)
        sil_hy = silhouette_score(sub_hybrid, l_hy) if len(set(l_hy)) >= 2 else 0

        winner = "DENSE" if sil_gs > sil_sp + 0.01 else "HYBRID" if sil_hy > sil_sp + 0.01 else "SPARSE"

        print("%-10s %6d %12.4f %12.4f %12.4f  %s" % (
            "C%d" % c, n, sil_sp, sil_gs, sil_hy, winner))
        results.append({"cluster": c, "size": int(n), "sparse_sub": round(sil_sp, 4),
                        "dense_sub": round(sil_gs, 4), "hybrid_sub": round(sil_hy, 4), "winner": winner})

    return results


def strategy2_temporal(kp_c, gauss_feats):
    """Compare temporal consistency of sparse vs dense clusters."""
    print("\n" + "=" * 60)
    print("Strategy 2: Temporal consistency (which features give more stable bouts?)")
    print("=" * 60)

    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)
    gauss_s = StandardScaler().fit_transform(gauss_feats)
    gauss_pca = PCA(n_components=min(15, gauss_feats.shape[1])).fit_transform(gauss_s)
    hybrid = np.concatenate([kp_pca, gauss_pca], axis=1)

    print("\n%-25s %4s %8s %8s %8s %8s" % ("Feature", "K", "Sil", "Bout_m", "Bout_md", "TransR"))
    print("-" * 65)

    results = []
    for name, feat in [("Sparse", kp_pca), ("Dense_Gauss", gauss_pca), ("Hybrid", hybrid)]:
        for k in [4, 8]:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat)
            sil = silhouette_score(feat, labels)
            changes = np.where(np.diff(labels) != 0)[0]
            bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / 20.0
            trans_rate = len(changes) / (len(labels) / 20.0)

            print("%-25s %4d %8.4f %8.3f %8.3f %8.3f" % (
                name, k, sil, bouts.mean(), np.median(bouts), trans_rate))
            results.append({"feature": name, "k": k, "sil": round(sil, 4),
                            "bout_mean": round(bouts.mean(), 3),
                            "bout_median": round(float(np.median(bouts)), 3),
                            "transition_rate": round(trans_rate, 3)})

    return results


def strategy3_shape(kp_c, gauss_feats):
    """Test if Gaussian shape features distinguish body posture subtypes."""
    print("\n" + "=" * 60)
    print("Strategy 3: Shape encoding (Gaussian eigenvalues as body shape descriptor)")
    print("=" * 60)

    # Extract shape-specific features from Gaussian
    # Eigenvalues (indices 18-20 in our feature vector) + bbox (21-23)
    shape_feats = gauss_feats[:, 18:24]  # eigenvalues + bbox
    shape_s = StandardScaler().fit_transform(shape_feats)

    # Also: nose-tail distance from keypoints as shape proxy
    nose_tail = np.linalg.norm(kp_c[:, 2] - kp_c[:, 5], axis=1)  # nose - tail_root

    print("\nCorrelation: Gaussian shape features vs keypoint nose-tail distance")
    for i, name in enumerate(["eigval_1", "eigval_2", "eigval_3", "bbox_x", "bbox_y", "bbox_z"]):
        corr = np.corrcoef(nose_tail, shape_feats[:, i])[0, 1]
        print("  %s: r=%.4f" % (name, corr))

    # Cluster with shape features only vs full sparse
    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)

    for k in [4, 6]:
        l_sp = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kp_pca)
        sil_sp = silhouette_score(kp_pca, l_sp)

        l_sh = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(shape_s)
        sil_sh = silhouette_score(shape_s, l_sh)

        # Shape added to sparse
        combined = np.concatenate([kp_pca, shape_s], axis=1)
        l_cb = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(combined)
        sil_cb = silhouette_score(combined, l_cb)

        print("\nK=%d: Sparse=%.4f, Shape-only=%.4f, Sparse+Shape=%.4f" % (k, sil_sp, sil_sh, sil_cb))
        better = sil_cb > sil_sp + 0.005
        print("  Sparse+Shape %s Sparse: %s" % (">" if better else "<=", "YES - shape helps!" if better else "no improvement"))


def main():
    kp_c, gauss_feats, frames = load_data()
    print("Data: %d frames, kp %s, gauss %s" % (len(frames), kp_c.shape, gauss_feats.shape))

    r1 = strategy1_subclustering(kp_c, gauss_feats)
    r2 = strategy2_temporal(kp_c, gauss_feats)
    strategy3_shape(kp_c, gauss_feats)

    # Save
    out = Path("outputs/clustering/dense_advantage")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dense_advantage_results.json", "w") as f:
        json.dump({"subclustering": r1, "temporal": r2}, f, indent=2)
    print("\nResults saved:", out / "dense_advantage_results.json")


if __name__ == "__main__":
    main()
