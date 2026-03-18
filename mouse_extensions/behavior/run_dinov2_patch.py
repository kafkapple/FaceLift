"""H2: DINOv2 patch-level feature extraction + temporal aggregation.

Previous experiment used DINOv2 CLS token only → Sil=0.132 (failed).
This script tests patch-level pooling ("mean", "concat") and temporal
window aggregation to see if DINOv2 can work for behavior clustering.

Uses behavior-lab's DINOv2Backend and temporal aggregation.

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.run_dinov2_patch \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5
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


FRAME_JUMPS = {1178,1179,1180,1181,1182,2358,2359,2360,2361,2362,3538,3539,3540,3541,3542}


def compute_metrics(features, labels, fps=20.0):
    """Compute key metrics."""
    valid = labels >= 0
    lab_v, feat_v = labels[valid], features[valid]
    if len(set(lab_v)) < 2:
        return None

    sil = float(silhouette_score(feat_v, lab_v, sample_size=min(5000, len(feat_v))))
    ch = float(calinski_harabasz_score(feat_v, lab_v))
    db = float(davies_bouldin_score(feat_v, lab_v))

    changes = np.where(np.diff(lab_v) != 0)[0]
    bouts = np.diff(np.concatenate([[0], changes + 1, [len(lab_v)]])) / fps
    tc = 1.0 - len(changes) / (len(lab_v) - 1) if len(lab_v) > 1 else 1.0
    short_pct = float((bouts < 0.3).mean() * 100) if len(bouts) > 0 else 0

    return {
        "silhouette": round(sil, 4),
        "calinski_harabasz": round(ch, 1),
        "davies_bouldin": round(db, 3),
        "bout_mean_sec": round(float(bouts.mean()), 3),
        "tc": round(tc, 4),
        "short_pct": round(short_pct, 1),
    }


def main():
    import sys
    sys.path.insert(0, "/home/joon/dev/behavior-lab/src")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5")
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument("--sample_frames", type=int, default=0,
                        help="0 = all frames, >0 = subsample for speed")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from mouse_extensions.behavior.paths import RESULTS_DIR, ensure_dirs
    ensure_dirs()
    output_dir = RESULTS_DIR / "dinov2_patch"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    # Frame list
    all_valid = sorted(set(range(3600)) - FRAME_JUMPS)
    if args.sample_frames > 0:
        step = max(1, len(all_valid) // args.sample_frames)
        all_valid = all_valid[::step][:args.sample_frames]
    N = len(all_valid)
    print(f"Processing {N} frames, {args.n_views} views")

    # Try behavior-lab DINOv2Backend
    try:
        from behavior_lab.data.features.visual_backend import DINOv2Backend
        print("Using behavior-lab DINOv2Backend")
        use_bl = True
    except ImportError:
        print("behavior-lab not available, using torch.hub directly")
        use_bl = False

    results = {}

    for pool_mode in ["cls", "mean", "concat"]:
        print(f"\n{'='*60}")
        print(f"DINOv2 pool={pool_mode}")
        print(f"{'='*60}")

        if use_bl:
            backend = DINOv2Backend(
                model_name="dinov2_vits14",
                device=args.device,
                pool=pool_mode,
                batch_size=32,
            )
            dim = backend.dim
        else:
            import torch
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            model = model.to(args.device).eval()
            dim = 384 if pool_mode != "concat" else 768

        print(f"  Feature dim: {dim}")

        # Extract per-view, per-frame features
        all_features = []  # (N, n_views * dim) or aggregated
        t0 = time.time()

        for fi_idx, fi in enumerate(all_valid):
            frame_dir = data_dir / f"{fi:06d}" / "images"
            view_feats = []

            for cam in range(args.n_views):
                img_path = frame_dir / f"cam_{cam:03d}.png"
                if not img_path.exists():
                    continue

                from PIL import Image
                img = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3) uint8

                if use_bl:
                    feat = backend.extract(img[np.newaxis])  # (1, dim)
                else:
                    import torch
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.ToPILImage(), T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    x = transform(img).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        out = model.forward_features(x)
                        if pool_mode == "cls":
                            feat = out["x_norm_clstoken"].cpu().numpy()
                        elif pool_mode == "mean":
                            feat = out["x_norm_patchtokens"].mean(1).cpu().numpy()
                        else:
                            cls = out["x_norm_clstoken"]
                            patch = out["x_norm_patchtokens"].mean(1)
                            feat = torch.cat([cls, patch], dim=-1).cpu().numpy()

                view_feats.append(feat.flatten())

            if len(view_feats) == args.n_views:
                # Aggregation strategies
                stacked = np.stack(view_feats)  # (n_views, dim)
                mean_feat = stacked.mean(0)  # (dim,) — mean across views
                concat_feat = stacked.flatten()  # (n_views * dim,)
                all_features.append({"mean": mean_feat, "concat": concat_feat})

            if (fi_idx + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  {fi_idx+1}/{N} ({elapsed:.0f}s, {(fi_idx+1)/elapsed:.1f} fps)")

        print(f"  Extracted {len(all_features)} frames ({time.time()-t0:.0f}s)")

        if len(all_features) < 100:
            print("  Too few frames, skipping")
            continue

        # Test both view aggregation methods
        for view_agg in ["mean_views", "concat_views"]:
            key_suffix = "mean" if view_agg == "mean_views" else "concat"
            feat_array = np.stack([f[key_suffix.replace("_views", "")] for f in all_features])

            # Standardize + PCA
            feat_s = StandardScaler().fit_transform(feat_array)
            n_pca = min(30, feat_s.shape[1])
            feat_pca = PCA(n_components=n_pca).fit_transform(feat_s)

            pipeline_name = f"DN_DINOv2_{pool_mode}_{view_agg}"
            print(f"\n  [{pipeline_name}] dim={feat_array.shape[1]} → PCA({n_pca})")

            for k in [4, 8]:
                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feat_pca)
                m = compute_metrics(feat_pca, labels)
                if m:
                    result_key = f"{pipeline_name}_K{k}"
                    results[result_key] = m
                    print(f"    K={k}: Sil={m['silhouette']:.3f} CH={m['calinski_harabasz']:.0f} "
                          f"Bout={m['bout_mean_sec']:.2f}s TC={m['tc']:.3f} Short={m['short_pct']:.0f}%")

    # Temporal window aggregation (on best pool mode)
    print(f"\n{'='*60}")
    print(f"Temporal Window Aggregation")
    print(f"{'='*60}")

    # Use mean_views features from the last pool mode run
    if all_features:
        feat_array = np.stack([f["mean"] for f in all_features])
        feat_s = StandardScaler().fit_transform(feat_array)
        feat_pca = PCA(n_components=min(30, feat_s.shape[1])).fit_transform(feat_s)

        for window in [5, 10, 15, 30]:
            # Sliding window mean
            T_len = len(feat_pca)
            n_seg = T_len - window + 1
            if n_seg < 50:
                continue
            windowed = np.stack([feat_pca[i:i+window].mean(0) for i in range(n_seg)])

            for k in [4, 8]:
                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(windowed)
                m = compute_metrics(windowed, labels, fps=20.0)
                if m:
                    result_key = f"DN_DINOv2_temporal_w{window}_K{k}"
                    results[result_key] = m
                    print(f"  window={window} K={k}: Sil={m['silhouette']:.3f} "
                          f"Bout={m['bout_mean_sec']:.2f}s TC={m['tc']:.3f}")

    # Save
    output_path = output_dir / "dinov2_patch_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
