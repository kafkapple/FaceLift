"""Phase 3D: Extract behavior features from 3D Gaussian parameters.

Loads PLY files from GS-LRM inference outputs and computes summary statistics
as behavior features.

Usage on gpu03:
    python -m mouse_extensions.behavior.extract_gaussian_features
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

SPARSE_BASELINE_SIL = 0.276


def load_ply_features(ply_path: str) -> np.ndarray:
    """Extract summary statistics from a Gaussian PLY file.

    Returns a fixed-size feature vector (22-dim baseline, expandable).
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("pip install plyfile")

    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)  # (N, 3)

    # Opacity
    op = np.array(v["opacity"]) if "opacity" in v.data.dtype.names else np.ones(len(xyz))

    features = []

    # === Position statistics (12 dims) ===
    features.extend(xyz.mean(axis=0))  # 3: center of mass
    features.extend(xyz.std(axis=0))  # 3: spatial spread
    features.extend(np.percentile(xyz, 25, axis=0))  # 3: Q1
    features.extend(np.percentile(xyz, 75, axis=0))  # 3: Q3

    # === Opacity statistics (3 dims) ===
    features.extend([float(op.mean()), float(op.std()), float(np.median(op))])

    # === Bounding box (3 dims) ===
    bbox = xyz.max(axis=0) - xyz.min(axis=0)
    features.extend(bbox)

    # === Shape descriptor: PCA eigenvalues (3 dims) ===
    xyz_c = xyz - xyz.mean(axis=0)
    cov = np.cov(xyz_c.T)
    eigvals = np.linalg.eigvalsh(cov)
    features.extend(sorted(eigvals, reverse=True))

    # === Count (1 dim) ===
    features.append(float(len(xyz)))

    # === Scale statistics if available (optional, +6 dims) ===
    scale_names = [f"scale_{i}" for i in range(3)]
    has_scale = all(s in v.data.dtype.names for s in scale_names)
    if has_scale:
        scales = np.stack([v[s] for s in scale_names], axis=1)  # (N, 3)
        features.extend(scales.mean(axis=0))  # 3: mean scale per axis
        features.extend(scales.std(axis=0))  # 3: scale variation

    # === Rotation statistics if available (optional, +4 dims) ===
    rot_names = [f"rot_{i}" for i in range(4)]
    has_rot = all(r in v.data.dtype.names for r in rot_names)
    if has_rot:
        rots = np.stack([v[r] for r in rot_names], axis=1)  # (N, 4)
        features.extend(rots.mean(axis=0))  # 4: mean quaternion

    return np.array(features, dtype=np.float32)


def find_ply_files(base_dir: str) -> list[tuple[int, str]]:
    """Find all PLY files and extract frame indices."""
    base = Path(base_dir)
    results = []
    for ply in sorted(base.rglob("gaussians.ply")):
        # Extract frame index from path
        for part in ply.parts:
            if part.isdigit() and len(part) == 6:
                results.append((int(part), str(ply)))
                break
    return results


def extract_all_features(base_dir: str, output_dir: str) -> str:
    """Extract Gaussian features for all available PLY files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_files = find_ply_files(base_dir)
    print(f"Found {len(ply_files)} PLY files")

    if not ply_files:
        print("No PLY files found!")
        return None

    frame_indices = []
    features_list = []
    feat_dim = None

    t0 = time.time()
    for i, (frame_idx, ply_path) in enumerate(ply_files):
        try:
            feat = load_ply_features(ply_path)
            if feat_dim is None:
                feat_dim = len(feat)
                print(f"Feature dimension: {feat_dim}")
            elif len(feat) != feat_dim:
                # Pad or truncate to match
                if len(feat) < feat_dim:
                    feat = np.pad(feat, (0, feat_dim - len(feat)))
                else:
                    feat = feat[:feat_dim]

            features_list.append(feat)
            frame_indices.append(frame_idx)
        except Exception as e:
            print(f"  Warning: frame {frame_idx}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ply_files)} processed ({time.time()-t0:.1f}s)")

    features = np.stack(features_list)
    frame_indices = np.array(frame_indices)

    out_path = output_dir / "gaussian_features.npz"
    np.savez_compressed(out_path, features=features, frame_indices=frame_indices,
                        feat_dim=feat_dim, n_frames=len(features))
    print(f"\nSaved: {out_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Frames: {frame_indices.min()}-{frame_indices.max()}")
    print(f"  Time: {time.time()-t0:.1f}s")

    return str(out_path)


def cluster_and_compare(features_path: str, output_dir: str):
    """Cluster Gaussian features and compare to sparse baseline."""
    output_dir = Path(output_dir)
    data = np.load(features_path)
    features = data["features"]
    frame_indices = data["frame_indices"]

    print(f"\nClustering {features.shape[0]} frames, {features.shape[1]} features")

    # Z-score standardize
    scaler = StandardScaler()
    feat_s = scaler.fit_transform(features)

    # PCA
    n_comp = min(20, feat_s.shape[1])
    pca = PCA(n_components=n_comp)
    feat_pca = pca.fit_transform(feat_s)
    var_exp = pca.explained_variance_ratio_.sum()
    print(f"PCA: {features.shape[1]} → {n_comp} dims ({var_exp:.3f} variance)")

    results = {}

    for method_name, cluster_fn in [
        ("PCA_KMeans_K4", lambda f: KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(f)),
        ("PCA_KMeans_K6", lambda f: KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(f)),
        ("PCA_KMeans_K8", lambda f: KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(f)),
        ("PCA_KMeans_auto", lambda f: _auto_kmeans(f)),
    ]:
        labels = cluster_fn(feat_pca)
        n_clusters = len(set(labels[labels >= 0]))

        if n_clusters >= 2:
            sil = silhouette_score(feat_pca, labels, sample_size=min(3000, len(feat_pca)))
            ch = calinski_harabasz_score(feat_pca, labels)
            db = davies_bouldin_score(feat_pca, labels)
        else:
            sil, ch, db = 0, 0, 0

        # Temporal metrics
        changes = np.where(np.diff(labels) != 0)[0]
        bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / 20.0

        delta = sil - SPARSE_BASELINE_SIL
        verdict = "BETTER" if delta > 0.01 else "SIMILAR" if abs(delta) <= 0.01 else "WORSE"

        results[method_name] = {
            "n_clusters": int(n_clusters),
            "silhouette": round(float(sil), 4),
            "calinski_harabasz": round(float(ch), 2),
            "davies_bouldin": round(float(db), 4),
            "bout_mean_sec": round(float(bouts.mean()), 3),
            "bout_median_sec": round(float(np.median(bouts)), 3),
            "delta_vs_sparse": round(float(delta), 4),
            "verdict": verdict,
        }

        np.save(output_dir / f"gaussian_{method_name}_labels.npy", labels)
        print(f"  {method_name}: K={n_clusters} Sil={sil:.4f} CH={ch:.1f} bout={bouts.mean():.2f}s → {verdict}")

    # Also compare with sparse baseline on SAME frames (test set only)
    print("\n--- Sparse baseline on same frames (test set) ---")
    from mouse_extensions.paths import KP_22
    kp_path = str(KP_22)
    kp_data = np.load(kp_path, allow_pickle=True)
    kp_all = kp_data["keypoints"]

    # COM centering
    kp_test = kp_all[frame_indices]
    com = kp_test.mean(axis=1, keepdims=True)
    kp_centered = kp_test - com

    kp_flat = kp_centered.reshape(len(kp_centered), -1)
    kp_scaler = StandardScaler()
    kp_s = kp_scaler.fit_transform(kp_flat)
    kp_pca = PCA(n_components=20).fit_transform(kp_s)

    km_sparse = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_sparse = km_sparse.fit_predict(kp_pca)
    sil_sparse = silhouette_score(kp_pca, labels_sparse, sample_size=min(3000, len(kp_pca)))
    print(f"  Sparse (same frames): Sil={sil_sparse:.4f}")
    results["sparse_same_frames"] = {"silhouette": round(float(sil_sparse), 4), "n_clusters": 4}

    # Save summary
    summary = {
        "experiment": "Phase_3D_Gaussian_Features",
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "n_frames": int(len(features)),
        "frame_range": f"{int(frame_indices.min())}-{int(frame_indices.max())}",
        "feat_dim": int(features.shape[1]),
        "pca_variance": round(float(var_exp), 4),
        "sparse_baseline_global": SPARSE_BASELINE_SIL,
        "results": results,
    }
    with open(output_dir / "gaussian_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {output_dir / 'gaussian_results.json'}")


def _auto_kmeans(features: np.ndarray) -> np.ndarray:
    """KMeans with automatic K selection (best silhouette)."""
    best_sil, best_k, best_labels = -1, 4, None
    for k in [4, 6, 8, 10, 12]:
        if k >= len(features):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(features)
        sil = silhouette_score(features, labels, sample_size=min(3000, len(features)))
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels
    return best_labels


def main():
    parser = argparse.ArgumentParser()
    from mouse_extensions.behavior.paths import FEATURES_DIR, RESULTS_DIR
    parser.add_argument("--base-dir", default="outputs", help="Base dir to search for PLY files")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "gaussian_3d"))
    args = parser.parse_args()

    features_path = extract_all_features(args.base_dir, args.output_dir)
    if features_path:
        cluster_and_compare(features_path, args.output_dir)


if __name__ == "__main__":
    main()
