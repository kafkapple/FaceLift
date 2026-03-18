"""Phase 3D-next: PointNet on raw Gaussian parameters.

Treats each frame's 17K Gaussians as a point cloud and uses a lightweight
PointNet (not PointNet++) to extract a global feature vector per frame.
No training required — uses the raw PointNet architecture as a deterministic
feature extractor, then clusters the output.

PointNet processes each point independently through MLPs, then max-pools
to get a global feature — preserving permutation invariance while capturing
spatial structure.

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.behavior.pointnet_gaussian
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SPARSE_BASELINE_SIL = 0.376  # test set, K=4


class SimplePointNet(nn.Module):
    """Lightweight PointNet for Gaussian feature extraction.

    Input: (B, N, C) point cloud with C channels per point
    Output: (B, D) global feature vector
    """

    def __init__(self, in_channels: int = 7, feature_dim: int = 256):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) point features
        Returns:
            (B, feature_dim) global features via max pooling
        """
        B, N, C = x.shape
        # Per-point MLP (shared weights)
        x = x.reshape(B * N, C)
        x = self.mlp1(x)  # (B*N, 64)
        x = self.mlp2(x)  # (B*N, 128)
        x = self.mlp3(x)  # (B*N, feature_dim)
        x = x.reshape(B, N, -1)
        # Max pooling over points
        global_feat = x.max(dim=1)[0]  # (B, feature_dim)
        return global_feat


def load_gaussian_pointcloud(ply_path: str, max_points: int = 4096) -> np.ndarray:
    """Load Gaussian params as point cloud.

    Returns: (N, C) where C = xyz(3) + scale(3) + opacity(1) = 7
    Optionally subsamples to max_points for memory efficiency.
    """
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    v = ply["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    opacity = np.array(v["opacity"]).reshape(-1, 1)

    points = np.concatenate([xyz, scale, opacity], axis=1)  # (N, 7)

    # Center xyz
    points[:, :3] -= points[:, :3].mean(axis=0)

    # Subsample if needed
    if len(points) > max_points:
        idx = np.random.RandomState(42).choice(len(points), max_points, replace=False)
        points = points[idx]

    return points.astype(np.float32)


def extract_pointnet_features(
    ply_dir: str,
    feature_dim: int = 256,
    max_points: int = 4096,
    batch_size: int = 16,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract PointNet features for all PLY files."""

    # Find deduplicated PLYs
    plys = sorted(Path(ply_dir).rglob("gaussians.ply"))
    frame_ply = {}
    for p in plys:
        for part in p.parts:
            if part.isdigit() and len(part) == 6:
                fi = int(part)
                if fi not in frame_ply:
                    frame_ply[fi] = str(p)
                break

    frames_sorted = sorted(frame_ply.keys())
    n_frames = len(frames_sorted)
    print(f"Frames: {n_frames} (range {frames_sorted[0]}-{frames_sorted[-1]})")

    # Initialize PointNet (random weights — used as deterministic feature extractor)
    torch.manual_seed(42)
    model = SimplePointNet(in_channels=7, feature_dim=feature_dim).to(device)
    model.eval()

    features = np.zeros((n_frames, feature_dim), dtype=np.float32)

    t0 = time.time()
    for batch_start in range(0, n_frames, batch_size):
        batch_frames = frames_sorted[batch_start:batch_start + batch_size]
        batch_points = []

        for fi in batch_frames:
            pc = load_gaussian_pointcloud(frame_ply[fi], max_points)
            batch_points.append(pc)

        # Pad to same size if needed
        max_n = max(len(p) for p in batch_points)
        padded = np.zeros((len(batch_points), max_n, 7), dtype=np.float32)
        for i, p in enumerate(batch_points):
            padded[i, :len(p)] = p

        tensor = torch.from_numpy(padded).to(device)

        with torch.no_grad():
            feats = model(tensor).cpu().numpy()

        for i, fi in enumerate(batch_frames):
            idx = frames_sorted.index(fi)
            features[idx] = feats[i]

        done = min(batch_start + batch_size, n_frames)
        if done % 100 == 0 or done == n_frames:
            elapsed = time.time() - t0
            print(f"  {done}/{n_frames} ({elapsed:.1f}s)")

    return features, np.array(frames_sorted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply-dir", default="outputs/experiments/phase3_e2e/H3_resume_pose")
    parser.add_argument("--output-dir", default="outputs/clustering/pointnet_3d")
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import json

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract PointNet features
    features, frames = extract_pointnet_features(
        args.ply_dir, args.feature_dim, args.max_points, device=args.device,
    )
    print(f"PointNet features: {features.shape}")

    # Save features
    np.savez_compressed(output_dir / "pointnet_features.npz",
                        features=features, frames=frames)

    # PCA reduce
    scaler = StandardScaler()
    feat_s = scaler.fit_transform(features)
    feat_pca = PCA(n_components=min(30, feat_s.shape[1])).fit_transform(feat_s)

    # Sparse baseline on same frames
    kp = np.load("/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz",
                 allow_pickle=True)["keypoints"]
    kp_sub = kp[frames]
    kp_c = kp_sub - kp_sub.mean(1, keepdims=True)
    kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
    kp_pca = PCA(n_components=20).fit_transform(kp_flat)

    print("\n%-35s %3s %8s %10s" % ("Method", "K", "Sil", "delta"))
    print("-" * 60)

    results = {}
    for k in [4, 6, 8, 10, 12]:
        l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kp_pca)
        s = silhouette_score(kp_pca, l)
        results["sparse_K%d" % k] = {"sil": round(s, 4), "k": k}
        print("%-35s %3d %8.4f %10s" % ("Sparse K=%d" % k, k, s, "BASE"))

    base = results["sparse_K4"]["sil"]
    for k in [4, 6, 8, 10, 12]:
        l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat_pca)
        s = silhouette_score(feat_pca, l)
        d = s - base
        v = "BETTER" if d > 0.01 else "SAME" if abs(d) <= 0.01 else "WORSE"
        results["pointnet_K%d" % k] = {"sil": round(s, 4), "k": k, "delta": round(d, 4), "verdict": v}
        print("%-35s %3d %8.4f %+10.4f %s" % ("PointNet K=%d" % k, k, s, d, v))
        np.save(output_dir / ("pointnet_K%d_labels.npy" % k), l)

    # Hybrid: sparse + pointnet
    hybrid = np.concatenate([kp_flat, feat_s], axis=1)
    hybrid_pca = PCA(n_components=30).fit_transform(hybrid)
    for k in [4, 6, 8, 10, 12]:
        l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(hybrid_pca)
        s = silhouette_score(hybrid_pca, l)
        d = s - base
        v = "BETTER" if d > 0.01 else "SAME" if abs(d) <= 0.01 else "WORSE"
        results["hybrid_pointnet_K%d" % k] = {"sil": round(s, 4), "k": k, "delta": round(d, 4), "verdict": v}
        print("%-35s %3d %8.4f %+10.4f %s" % ("Hybrid sp+pointnet K=%d" % k, k, s, d, v))

    with open(output_dir / "pointnet_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {output_dir / 'pointnet_results.json'}")


if __name__ == "__main__":
    main()
