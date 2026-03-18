"""Phase 3B: Extract DINOv2 features from 6 GT camera views.

Extracts CLS token features from DINOv2-ViT-B/14 for each camera view,
then aggregates per frame via mean pooling.

Usage on gpu03:
    cd /home/joon/dev/FaceLift
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.behavior.extract_dinov2_features

Output: outputs/features/dinov2/dinov2_features.npz
    - per_view: (T, 6, 768) — per-view CLS tokens
    - mean_pool: (T, 768) — mean-pooled across views
    - frame_indices: (T,) — valid frame indices
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Frame jump masking
FRAME_JUMP_INDICES = [1180, 2360, 3540]
FRAME_JUMP_MARGIN = 2


def get_valid_frame_indices(total_frames: int) -> np.ndarray:
    """Return frame indices excluding jump regions."""
    valid = np.ones(total_frames, dtype=bool)
    for idx in FRAME_JUMP_INDICES:
        s, e = max(0, idx - FRAME_JUMP_MARGIN), min(total_frames, idx + FRAME_JUMP_MARGIN + 1)
        valid[s:e] = False
    return np.where(valid)[0]


def load_dinov2(model_name: str = "dinov2_vitb14", device: str = "cuda"):
    """Load DINOv2 model from torch hub."""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess


def extract_features(
    data_dir: str,
    output_dir: str,
    model_name: str = "dinov2_vitb14",
    batch_size: int = 64,
    device: str = "cuda",
    n_cameras: int = 6,
    total_frames: int = 3600,
):
    """Extract DINOv2 features for all views and frames."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Valid frames after jump masking
    valid_indices = get_valid_frame_indices(total_frames)
    n_valid = len(valid_indices)
    print(f"Frames: {total_frames} total, {n_valid} valid (excluding {total_frames - n_valid} jump frames)")

    # Load model
    print(f"Loading {model_name}...")
    model, preprocess = load_dinov2(model_name, device)

    # Determine feature dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        feat_dim = model(dummy).shape[-1]
    print(f"Feature dimension: {feat_dim}")

    # Storage
    per_view_features = np.zeros((n_valid, n_cameras, feat_dim), dtype=np.float32)

    # Process in batches (batch across views and frames)
    all_tasks = []  # (frame_local_idx, cam_idx, image_path)
    for local_idx, frame_idx in enumerate(valid_indices):
        frame_dir = data_dir / f"{frame_idx:06d}" / "images"
        for cam_idx in range(n_cameras):
            img_path = frame_dir / f"cam_{cam_idx:03d}.png"
            all_tasks.append((local_idx, cam_idx, img_path))

    total_tasks = len(all_tasks)
    print(f"Total images to process: {total_tasks} ({n_valid} frames × {n_cameras} views)")

    t0 = time.time()
    for batch_start in range(0, total_tasks, batch_size):
        batch_tasks = all_tasks[batch_start:batch_start + batch_size]

        # Load and preprocess batch
        images = []
        indices = []
        for local_idx, cam_idx, img_path in batch_tasks:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img)
                images.append(img_tensor)
                indices.append((local_idx, cam_idx))
            except Exception as e:
                print(f"  Warning: Failed to load {img_path}: {e}")
                continue

        if not images:
            continue

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            features = model(batch_tensor).cpu().numpy()  # (B, feat_dim)

        for feat, (local_idx, cam_idx) in zip(features, indices):
            per_view_features[local_idx, cam_idx] = feat

        # Progress
        done = min(batch_start + batch_size, total_tasks)
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (total_tasks - done) / rate if rate > 0 else 0
        if done % (batch_size * 10) == 0 or done == total_tasks:
            print(f"  {done}/{total_tasks} ({done/total_tasks*100:.1f}%) — {rate:.1f} img/s — ETA {eta:.0f}s")

    # Mean pooling across views
    mean_pool = per_view_features.mean(axis=1)  # (T, feat_dim)

    # Save
    output_path = output_dir / "dinov2_features.npz"
    np.savez_compressed(
        output_path,
        per_view=per_view_features,
        mean_pool=mean_pool,
        frame_indices=valid_indices,
        model_name=model_name,
        feat_dim=feat_dim,
    )
    print(f"\nSaved: {output_path}")
    print(f"  per_view: {per_view_features.shape}")
    print(f"  mean_pool: {mean_pool.shape}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return output_path


def cluster_and_compare(
    features_path: str,
    sparse_baseline_sil: float = 0.276,
    output_dir: str = "outputs/clustering/dense_2d",
):
    """Phase 3B Step 2: Cluster DINOv2 features and compare to sparse baseline."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(features_path)
    mean_pool = data["mean_pool"]  # (T, 768)
    per_view = data["per_view"]  # (T, 6, 768)

    print(f"Features loaded: mean_pool {mean_pool.shape}, per_view {per_view.shape}")

    results = {}

    for agg_name, features in [
        ("mean_pool", mean_pool),
        ("concat_pca", per_view.reshape(len(per_view), -1)),  # (T, 6*768=4608)
    ]:
        print(f"\n--- Aggregation: {agg_name} ---")

        # Z-score standardize
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(features)

        # PCA
        n_comp = min(128, feat_scaled.shape[1])
        pca = PCA(n_components=n_comp)
        feat_pca = pca.fit_transform(feat_scaled)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {features.shape[1]} → {n_comp} dims ({var_explained:.3f} variance)")

        # KMeans with optimal K
        best_sil, best_k = -1, 4
        for k in [4, 6, 8, 10, 12]:
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(feat_pca)
            sil = silhouette_score(feat_pca, labels, sample_size=min(3000, len(feat_pca)))
            if sil > best_sil:
                best_sil, best_k = sil, k

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(feat_pca)
        sil = silhouette_score(feat_pca, labels, sample_size=min(3000, len(feat_pca)))
        ch = calinski_harabasz_score(feat_pca, labels)
        db = davies_bouldin_score(feat_pca, labels)

        # Bout duration
        changes = np.where(np.diff(labels) != 0)[0]
        bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / 20.0

        delta = sil - sparse_baseline_sil
        verdict = "BETTER" if delta > 0.01 else "SIMILAR" if abs(delta) <= 0.01 else "WORSE"

        results[agg_name] = {
            "n_clusters": best_k,
            "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2),
            "davies_bouldin": round(db, 4),
            "bout_mean_sec": round(bouts.mean(), 3),
            "pca_variance": round(var_explained, 4),
            "vs_sparse_delta": round(delta, 4),
            "verdict": verdict,
        }

        np.save(output_dir / f"dinov2_{agg_name}_labels.npy", labels)

        print(f"  K={best_k}, Sil={sil:.4f}, CH={ch:.1f}, DB={db:.4f}")
        print(f"  Bout mean={bouts.mean():.2f}s")
        print(f"  vs Sparse baseline (0.276): {'+' if delta > 0 else ''}{delta:.4f} → {verdict}")

    # Save results
    import json
    with open(output_dir / "dense_2d_results.json", "w") as f:
        json.dump({
            "experiment": "Phase_3B_DINOv2_2D_Dense",
            "sparse_baseline_sil": sparse_baseline_sil,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved: {output_dir / 'dense_2d_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser()
    from mouse_extensions.behavior.paths import GPU03_M5_DATA, FEATURES_DIR
    parser.add_argument("--data-dir", default=GPU03_M5_DATA)
    parser.add_argument("--output-dir", default=str(FEATURES_DIR))
    parser.add_argument("--model", default="dinov2_vitb14")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cluster", action="store_true", help="Also run clustering after extraction")
    args = parser.parse_args()

    features_path = extract_features(
        args.data_dir, args.output_dir, args.model, args.batch_size, args.device,
    )

    if args.cluster:
        cluster_and_compare(str(features_path))


if __name__ == "__main__":
    main()
