#!/usr/bin/env python3
"""
Camera Pose Encoder Visualization
- Visualize how different encoding methods represent camera poses
- Test before diffusion training to verify encoder quality
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mvdiffusion.models.pose_conditioning import (
    CameraPoseConditioner,
    SphericalPoseEncoder,
    PluckerRayEncoder,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_mouse_cameras(data_dir: str, num_frames: int = 10):
    """Load camera poses from mouse dataset.

    Supports two formats:
    1. opencv_cameras.json format (actual mouse data):
       - sample_XXXXXX/opencv_cameras.json with w2c matrices
    2. Legacy format:
       - frame_*/cameras.json with c2w matrices
    """
    data_path = Path(data_dir)

    all_c2w = []
    all_intrinsics = []

    # Try opencv_cameras.json format first (actual mouse data)
    sample_dirs = sorted(data_path.glob("sample_*"))[:num_frames]

    if sample_dirs:
        print(f"Found {len(sample_dirs)} sample directories (opencv_cameras.json format)")
        for sample_dir in sample_dirs:
            cam_file = sample_dir / "opencv_cameras.json"
            if not cam_file.exists():
                continue

            with open(cam_file) as f:
                cams = json.load(f)

            if "frames" not in cams:
                continue

            frames = cams["frames"]
            if len(frames) != 6:
                continue

            # Sort by view_id to ensure consistent ordering
            frames = sorted(frames, key=lambda x: x.get("view_id", 0))

            frame_c2w = []
            frame_intrinsics = []

            for frame in frames:
                # w2c (world-to-camera) -> c2w (camera-to-world)
                w2c = np.array(frame["w2c"])
                c2w = np.linalg.inv(w2c)

                # Intrinsics
                fx = frame.get("fx", 512)
                fy = frame.get("fy", 512)
                cx = frame.get("cx", 256)
                cy = frame.get("cy", 256)

                frame_c2w.append(c2w)
                frame_intrinsics.append([fx, fy, cx, cy])

            if len(frame_c2w) == 6:
                all_c2w.append(np.stack(frame_c2w))
                all_intrinsics.append(np.stack(frame_intrinsics))

    # Fallback to legacy format
    if not all_c2w:
        frame_dirs = sorted(data_path.glob("frame_*"))[:num_frames]

        for frame_dir in frame_dirs:
            cam_file = frame_dir / "cameras.json"
            if not cam_file.exists():
                continue

            with open(cam_file) as f:
                cams = json.load(f)

            frame_c2w = []
            frame_intrinsics = []

            for view_idx in range(6):
                view_key = f"view_{view_idx}"
                if view_key not in cams:
                    continue

                cam = cams[view_key]
                c2w = np.array(cam["c2w"])

                # Intrinsics
                fx = cam.get("fx", 512)
                fy = cam.get("fy", 512)
                cx = cam.get("cx", 256)
                cy = cam.get("cy", 256)

                frame_c2w.append(c2w)
                frame_intrinsics.append([fx, fy, cx, cy])

            if len(frame_c2w) == 6:
                all_c2w.append(np.stack(frame_c2w))
                all_intrinsics.append(np.stack(frame_intrinsics))

    if not all_c2w:
        print("No camera data found, using synthetic cameras")
        return create_synthetic_cameras()

    c2w = torch.tensor(np.stack(all_c2w), dtype=torch.float32)
    intrinsics = torch.tensor(np.stack(all_intrinsics), dtype=torch.float32)

    print(f"Loaded {len(all_c2w)} frames with real camera data")

    return c2w, intrinsics


def create_synthetic_cameras(num_frames: int = 10, num_views: int = 6):
    """Create synthetic camera poses matching mouse camera arrangement."""
    # Mouse camera azimuths (approximate)
    azimuths = np.array([0, 13.5, 36, 72, 99, 151.4]) * np.pi / 180
    elevation = 15 * np.pi / 180  # Slight elevation
    distance = 2.7

    all_c2w = []

    for frame_idx in range(num_frames):
        frame_c2w = []
        for az in azimuths:
            # Camera position
            x = distance * np.sin(az) * np.cos(elevation)
            y = distance * np.sin(elevation)
            z = distance * np.cos(az) * np.cos(elevation)

            # Look at origin
            forward = -np.array([x, y, z])
            forward = forward / np.linalg.norm(forward)

            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            # c2w matrix
            c2w = np.eye(4)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward
            c2w[:3, 3] = [x, y, z]

            frame_c2w.append(c2w)

        all_c2w.append(np.stack(frame_c2w))

    c2w = torch.tensor(np.stack(all_c2w), dtype=torch.float32)
    intrinsics = torch.tensor([[512, 512, 256, 256]] * num_views, dtype=torch.float32)
    intrinsics = intrinsics.unsqueeze(0).expand(num_frames, -1, -1)

    return c2w, intrinsics


def visualize_spherical_encoding(c2w: torch.Tensor, save_path: str):
    """Visualize spherical coordinate encoding."""
    B, N = c2w.shape[:2]

    # Extract spherical coordinates
    az, el, dist = SphericalPoseEncoder.from_camera_matrix(c2w)

    # Create encoder and get embeddings
    encoder = CameraPoseConditioner(method='spherical', embed_dim=256)
    with torch.no_grad():
        embeddings = encoder(c2w)  # [B, N, 256]

    embeddings_flat = embeddings.reshape(-1, 256).numpy()

    # Dimensionality reduction
    if embeddings_flat.shape[0] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_flat)
    else:
        embeddings_2d = embeddings_flat[:, :2]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Spherical coordinates
    ax = axes[0]
    az_flat = az.reshape(-1).numpy() * 180 / np.pi
    el_flat = el.reshape(-1).numpy() * 180 / np.pi
    colors = np.tile(np.arange(N), B)
    scatter = ax.scatter(az_flat, el_flat, c=colors, cmap='tab10', s=50)
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_title('Camera Positions (Spherical)')
    ax.legend(*scatter.legend_elements(), title="View")

    # Plot 2: PCA of embeddings
    ax = axes[1]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='tab10', s=50)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Spherical Embeddings (PCA)')

    # Plot 3: Embedding similarity matrix
    ax = axes[2]
    embeddings_norm = embeddings[0] / embeddings[0].norm(dim=-1, keepdim=True)
    similarity = (embeddings_norm @ embeddings_norm.T).numpy()
    im = ax.imshow(similarity, cmap='viridis', vmin=-1, vmax=1)
    ax.set_xlabel('View Index')
    ax.set_ylabel('View Index')
    ax.set_title('Embedding Similarity (First Frame)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return embeddings


def visualize_extrinsic_encoding(c2w: torch.Tensor, save_path: str):
    """Visualize extrinsic parameter encoding."""
    B, N = c2w.shape[:2]

    # Create encoder and get embeddings
    encoder = CameraPoseConditioner(method='extrinsic', embed_dim=256)
    with torch.no_grad():
        embeddings = encoder(c2w)  # [B, N, 256]

    embeddings_flat = embeddings.reshape(-1, 256).numpy()

    # Dimensionality reduction
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_flat)

    # t-SNE for comparison
    if embeddings_flat.shape[0] >= 5:
        tsne = TSNE(n_components=2, perplexity=min(5, embeddings_flat.shape[0]-1))
        embeddings_tsne = tsne.fit_transform(embeddings_flat)
    else:
        embeddings_tsne = embeddings_2d

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = np.tile(np.arange(N), B)

    # Plot 1: PCA
    ax = axes[0]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='tab10', s=50)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Extrinsic Embeddings (PCA)')
    ax.legend(*scatter.legend_elements(), title="View")

    # Plot 2: t-SNE
    ax = axes[1]
    scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=colors, cmap='tab10', s=50)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Extrinsic Embeddings (t-SNE)')

    # Plot 3: Similarity matrix
    ax = axes[2]
    embeddings_norm = embeddings[0] / embeddings[0].norm(dim=-1, keepdim=True)
    similarity = (embeddings_norm @ embeddings_norm.T).numpy()
    im = ax.imshow(similarity, cmap='viridis', vmin=-1, vmax=1)
    ax.set_xlabel('View Index')
    ax.set_ylabel('View Index')
    ax.set_title('Embedding Similarity (First Frame)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return embeddings


def visualize_plucker_encoding(c2w: torch.Tensor, intrinsics: torch.Tensor, save_path: str):
    """Visualize Plücker ray encoding."""
    B, N = c2w.shape[:2]
    H, W = 64, 64  # Small resolution for visualization

    # Compute Plücker coordinates for first frame
    c2w_first = c2w[0]  # [N, 4, 4]
    intrinsics_first = intrinsics[0]  # [N, 4]

    fig, axes = plt.subplots(2, N, figsize=(3*N, 6))

    for view_idx in range(N):
        c2w_view = c2w_first[view_idx:view_idx+1]  # [1, 4, 4]
        intr_view = intrinsics_first[view_idx:view_idx+1]  # [1, 4]

        # Compute Plücker coordinates
        plucker = PluckerRayEncoder.compute_plucker_coordinates(
            c2w_view, intr_view, H, W
        )  # [1, 6, H, W]

        plucker = plucker[0].numpy()  # [6, H, W]

        # Direction (first 3 channels) - normalize for visualization
        direction = plucker[:3]
        direction_norm = np.linalg.norm(direction, axis=0)
        direction_rgb = (direction.transpose(1, 2, 0) + 1) / 2  # Map [-1,1] to [0,1]

        # Moment (last 3 channels)
        moment = plucker[3:]
        moment_mag = np.linalg.norm(moment, axis=0)

        # Plot direction as RGB
        axes[0, view_idx].imshow(direction_rgb)
        axes[0, view_idx].set_title(f'View {view_idx}\nDirection (RGB)')
        axes[0, view_idx].axis('off')

        # Plot moment magnitude
        im = axes[1, view_idx].imshow(moment_mag, cmap='viridis')
        axes[1, view_idx].set_title('Moment Magnitude')
        axes[1, view_idx].axis('off')

    plt.suptitle('Plücker Ray Encoding: Direction (top) and Moment (bottom)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def visualize_encoding_comparison(c2w: torch.Tensor, save_path: str):
    """Compare all three encoding methods."""
    B, N = c2w.shape[:2]

    # Get embeddings from all methods
    spherical_enc = CameraPoseConditioner(method='spherical', embed_dim=256)
    extrinsic_enc = CameraPoseConditioner(method='extrinsic', embed_dim=256)

    with torch.no_grad():
        spherical_emb = spherical_enc(c2w)  # [B, N, 256]
        extrinsic_emb = extrinsic_enc(c2w)  # [B, N, 256]

    # Compute view-to-view angular distances (ground truth)
    positions = c2w[0, :, :3, 3].numpy()  # [N, 3]
    angular_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cos_angle = np.dot(positions[i], positions[j]) / (
                np.linalg.norm(positions[i]) * np.linalg.norm(positions[j])
            )
            angular_dist[i, j] = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    # Compute embedding similarities
    spherical_sim = compute_similarity(spherical_emb[0])
    extrinsic_sim = compute_similarity(extrinsic_emb[0])

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth angular distance
    im0 = axes[0].imshow(angular_dist, cmap='viridis')
    axes[0].set_title('Angular Distance (degrees)\n(Ground Truth)')
    axes[0].set_xlabel('View Index')
    axes[0].set_ylabel('View Index')
    plt.colorbar(im0, ax=axes[0])

    # Add text annotations
    for i in range(N):
        for j in range(N):
            axes[0].text(j, i, f'{angular_dist[i,j]:.0f}°', ha='center', va='center', fontsize=8)

    # Spherical embedding similarity
    im1 = axes[1].imshow(spherical_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('Spherical Embedding\nCosine Similarity')
    axes[1].set_xlabel('View Index')
    axes[1].set_ylabel('View Index')
    plt.colorbar(im1, ax=axes[1])

    # Extrinsic embedding similarity
    im2 = axes[2].imshow(extrinsic_sim, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title('Extrinsic Embedding\nCosine Similarity')
    axes[2].set_xlabel('View Index')
    axes[2].set_ylabel('View Index')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle('Encoding Quality: Does similarity reflect angular distance?', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # Print correlation analysis
    angular_flat = angular_dist[np.triu_indices(N, k=1)]
    spherical_flat = (1 - spherical_sim)[np.triu_indices(N, k=1)]  # Convert sim to dist
    extrinsic_flat = (1 - extrinsic_sim)[np.triu_indices(N, k=1)]

    from scipy.stats import spearmanr
    spherical_corr = spearmanr(angular_flat, spherical_flat)[0]
    extrinsic_corr = spearmanr(angular_flat, extrinsic_flat)[0]

    print(f"\n=== Encoding Quality (Spearman Correlation with Angular Distance) ===")
    print(f"Spherical: {spherical_corr:.3f}")
    print(f"Extrinsic: {extrinsic_corr:.3f}")
    print(f"(Higher = embedding distance better reflects actual angular distance)")


def compute_similarity(embeddings: torch.Tensor) -> np.ndarray:
    """Compute cosine similarity matrix."""
    embeddings_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return (embeddings_norm @ embeddings_norm.T).numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data_mouse_centered',
                        help='Path to mouse data directory')
    parser.add_argument('--output_dir', type=str, default='experiments/pose_encoding_viz',
                        help='Output directory for visualizations')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to load')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading camera data...")
    c2w, intrinsics = load_mouse_cameras(args.data_dir, args.num_frames)
    print(f"Loaded {c2w.shape[0]} frames with {c2w.shape[1]} views each")

    print("\n1. Visualizing Spherical encoding...")
    visualize_spherical_encoding(c2w, str(output_dir / "spherical_encoding.png"))

    print("\n2. Visualizing Extrinsic encoding...")
    visualize_extrinsic_encoding(c2w, str(output_dir / "extrinsic_encoding.png"))

    print("\n3. Visualizing Plücker encoding...")
    visualize_plucker_encoding(c2w, intrinsics, str(output_dir / "plucker_encoding.png"))

    print("\n4. Comparing all encodings...")
    visualize_encoding_comparison(c2w, str(output_dir / "encoding_comparison.png"))

    print(f"\n=== All visualizations saved to {output_dir} ===")


if __name__ == '__main__':
    main()
