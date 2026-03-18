"""Phase 3B-ext: DINOv2 with foreground masking.

M5 images are RGBA — alpha channel provides foreground mask.
Three strategies:
1. Masked: set background to black using alpha mask
2. Cropped: crop to bounding box of foreground + DINOv2
3. Patch-level: use DINOv2 spatial tokens, select foreground patches only

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.behavior.extract_dinov2_masked
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

FRAME_JUMP_INDICES = [1180, 2360, 3540]
FRAME_JUMP_MARGIN = 2


def get_valid_indices(total: int = 3600) -> np.ndarray:
    valid = np.ones(total, dtype=bool)
    for idx in FRAME_JUMP_INDICES:
        valid[max(0, idx - FRAME_JUMP_MARGIN):min(total, idx + FRAME_JUMP_MARGIN + 1)] = False
    return np.where(valid)[0]


def load_masked_image(img_path: str, strategy: str = "masked") -> Image.Image:
    """Load RGBA image and apply foreground strategy.

    Args:
        strategy: "masked" (bg=black), "cropped" (bbox crop), "full" (ignore alpha)
    """
    img = Image.open(img_path)  # RGBA

    if strategy == "full":
        return img.convert("RGB")

    r, g, b, a = img.split()
    alpha = np.array(a)

    if strategy == "masked":
        # Set background pixels to black
        rgb = np.array(img.convert("RGB"))
        mask = alpha > 10  # threshold
        rgb[~mask] = 0  # black background
        return Image.fromarray(rgb)

    elif strategy == "cropped":
        # Crop to bounding box of foreground
        mask = alpha > 10
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return img.convert("RGB")
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Add margin (10%)
        h, w = rmax - rmin, cmax - cmin
        margin = max(int(max(h, w) * 0.1), 5)
        rmin, rmax = max(0, rmin - margin), min(alpha.shape[0], rmax + margin)
        cmin, cmax = max(0, cmin - margin), min(alpha.shape[1], cmax + margin)
        cropped = img.crop((cmin, rmin, cmax, rmax)).convert("RGB")
        return cropped

    return img.convert("RGB")


def extract_masked_features(
    data_dir: str,
    output_dir: str,
    strategy: str = "masked",
    model_name: str = "dinov2_vitb14",
    batch_size: int = 64,
    device: str = "cuda",
    n_cameras: int = 6,
    total_frames: int = 3600,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_indices = get_valid_indices(total_frames)
    n_valid = len(valid_indices)

    print(f"Strategy: {strategy}")
    print(f"Frames: {n_valid} valid")

    # Load model
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        feat_dim = model(torch.randn(1, 3, 224, 224).to(device)).shape[-1]
    print(f"Feature dim: {feat_dim}")

    per_view = np.zeros((n_valid, n_cameras, feat_dim), dtype=np.float32)

    # Build task list
    tasks = []
    for li, fi in enumerate(valid_indices):
        for ci in range(n_cameras):
            tasks.append((li, ci, data_dir / f"{fi:06d}" / "images" / f"cam_{ci:03d}.png"))

    t0 = time.time()
    for bs in range(0, len(tasks), batch_size):
        batch = tasks[bs:bs + batch_size]
        images, indices = [], []
        for li, ci, path in batch:
            try:
                img = load_masked_image(str(path), strategy)
                images.append(preprocess(img))
                indices.append((li, ci))
            except Exception:
                continue

        if not images:
            continue

        tensor = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model(tensor).cpu().numpy()

        for feat, (li, ci) in zip(feats, indices):
            per_view[li, ci] = feat

        done = min(bs + batch_size, len(tasks))
        if done % (batch_size * 10) == 0 or done >= len(tasks):
            elapsed = time.time() - t0
            rate = done / elapsed
            print(f"  {done}/{len(tasks)} ({done/len(tasks)*100:.1f}%) — {rate:.1f} img/s")

    mean_pool = per_view.mean(axis=1)

    out_path = output_dir / f"dinov2_{strategy}_features.npz"
    np.savez_compressed(out_path, per_view=per_view, mean_pool=mean_pool,
                        frame_indices=valid_indices, strategy=strategy)
    print(f"Saved: {out_path} ({mean_pool.shape})")
    print(f"Time: {time.time() - t0:.1f}s")
    return str(out_path)


def cluster_all_strategies(output_dir: str, sparse_sil: float = 0.276):
    """Cluster and compare all strategy results."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    import json

    output_dir = Path(output_dir)
    results = {}

    for npz_path in sorted(output_dir.glob("dinov2_*_features.npz")):
        strategy = npz_path.stem.split("_")[1]  # e.g., "masked"
        data = np.load(npz_path)
        features = data["mean_pool"]

        scaler = StandardScaler()
        feat_s = scaler.fit_transform(features)
        pca = PCA(n_components=min(128, feat_s.shape[1]))
        feat_pca = pca.fit_transform(feat_s)

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

        changes = np.where(np.diff(labels) != 0)[0]
        bouts = np.diff(np.concatenate([[0], changes + 1, [len(labels)]])) / 20.0

        delta = sil - sparse_sil
        verdict = "BETTER" if delta > 0.01 else "SIMILAR" if abs(delta) <= 0.01 else "WORSE"

        results[strategy] = {
            "n_clusters": best_k, "silhouette": round(sil, 4),
            "calinski_harabasz": round(ch, 2), "davies_bouldin": round(db, 4),
            "bout_mean_sec": round(bouts.mean(), 3),
            "pca_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
            "delta_vs_sparse": round(delta, 4), "verdict": verdict,
        }

        np.save(output_dir / f"labels_{strategy}.npy", labels)
        print(f"  {strategy}: K={best_k} Sil={sil:.4f} delta={delta:+.4f} → {verdict}")

    with open(output_dir / "masked_comparison.json", "w") as f:
        json.dump({"sparse_baseline": sparse_sil, "results": results}, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--output-dir", default="outputs/features/dinov2_masked")
    parser.add_argument("--strategies", nargs="+", default=["masked", "cropped"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    for strategy in args.strategies:
        extract_masked_features(
            args.data_dir, args.output_dir, strategy=strategy,
            device=args.device, batch_size=args.batch_size,
        )

    print("\n=== Clustering Comparison ===")
    # Also copy the full (unmasked) features for comparison
    full_path = Path("outputs/features/dinov2/dinov2_features.npz")
    if full_path.exists():
        import shutil
        dest = Path(args.output_dir) / "dinov2_full_features.npz"
        if not dest.exists():
            shutil.copy(full_path, dest)

    cluster_all_strategies(args.output_dir)


if __name__ == "__main__":
    main()
