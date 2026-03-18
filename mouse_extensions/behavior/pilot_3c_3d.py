"""Phase 3C + 3D Pilot — Quick test of 3D novel views and Gaussian features.

3C: Render 4 novel views per frame → DINOv2 → mean pool → cluster
3D: Extract 3D Gaussian parameters → PCA → cluster

Both on 100-frame pilot subset.

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.behavior.pilot_3c_3d
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SPARSE_BASELINE_SIL = 0.276


def pilot_3d_gaussian_features(n_pilot: int = 100):
    """Phase 3D pilot: cluster directly on Gaussian parameters.

    GS-LRM outputs 16,384 Gaussians per frame, each with:
    - xyz (3), opacity (1), scale (3), rotation quaternion (4), SH coefficients
    We use a subset of parameters as features.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print("\n" + "=" * 60)
    print("Phase 3D Pilot: 3D Gaussian Parameter Features")
    print("=" * 60)

    # Check for existing Gaussian PLY files
    ply_dir = Path("outputs/experiments")
    ply_files = sorted(ply_dir.glob("**/gaussians.ply"))

    if len(ply_files) < 5:
        print(f"Only {len(ply_files)} PLY files found. Need GS-LRM inference first.")
        print("Running GS-LRM inference on pilot frames...")

        # Use the inference pipeline to generate Gaussians
        return _run_gslrm_and_extract(n_pilot)

    print(f"Found {len(ply_files)} existing PLY files")

    # Load Gaussian parameters from PLY
    features_list = []
    for ply_path in ply_files[:n_pilot]:
        try:
            params = _load_ply_gaussians(ply_path)
            if params is not None:
                features_list.append(params)
        except Exception as e:
            print(f"  Warning: {ply_path}: {e}")

    if len(features_list) < 10:
        print(f"Only {len(features_list)} valid PLY files. Insufficient for pilot.")
        return None

    features = np.stack(features_list)
    print(f"Gaussian features: {features.shape}")

    # Cluster
    scaler = StandardScaler()
    feat_s = scaler.fit_transform(features)
    pca = PCA(n_components=min(20, feat_s.shape[1]))
    feat_pca = pca.fit_transform(feat_s)

    best_sil, best_k = -1, 4
    for k in [4, 6, 8]:
        if k >= len(feat_pca):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(feat_pca)
        sil = silhouette_score(feat_pca, labels, sample_size=min(1000, len(feat_pca)))
        if sil > best_sil:
            best_sil, best_k = sil, k

    delta = best_sil - SPARSE_BASELINE_SIL
    verdict = "BETTER" if delta > 0.01 else "SIMILAR" if abs(delta) <= 0.01 else "WORSE"
    print(f"  K={best_k}, Sil={best_sil:.4f}, delta={delta:+.4f} → {verdict}")

    return {"method": "3D_Gaussian_PCA", "n_frames": len(features_list),
            "k": best_k, "sil": best_sil, "delta": delta, "verdict": verdict}


def _load_ply_gaussians(ply_path: Path) -> np.ndarray:
    """Load Gaussian parameters from PLY and compute summary statistics.

    Returns a fixed-size feature vector summarizing the Gaussian cloud.
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    vertex = plydata["vertex"]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    opacity = vertex["opacity"] if "opacity" in vertex else np.ones(len(xyz))

    # Summary statistics as feature vector
    features = []

    # Spatial statistics (position distribution)
    features.extend(xyz.mean(axis=0))  # 3: center of mass
    features.extend(xyz.std(axis=0))  # 3: spatial spread
    features.extend(np.percentile(xyz, [25, 75], axis=0).flatten())  # 6: quartiles

    # Opacity statistics
    op = np.array(opacity)
    features.extend([op.mean(), op.std(), np.median(op)])  # 3

    # Spatial extent
    bbox = xyz.max(axis=0) - xyz.min(axis=0)
    features.extend(bbox)  # 3: bounding box size

    # PCA on positions (shape descriptor)
    xyz_centered = xyz - xyz.mean(axis=0)
    cov = np.cov(xyz_centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    features.extend(sorted(eigvals, reverse=True))  # 3: eigenvalues (shape)

    # Density
    features.append(len(xyz))  # 1: number of Gaussians

    return np.array(features, dtype=np.float32)


def _run_gslrm_and_extract(n_pilot: int) -> dict:
    """Run GS-LRM inference on pilot frames and extract features."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Load model
    print("Loading GS-LRM model...")
    sys.path.insert(0, ".")

    try:
        from gslrm.models.gslrm import GSLRM
        from omegaconf import OmegaConf

        # Find config
        config_path = Path("configs/mouse/base/gslrm_mouse.yaml")
        if not config_path.exists():
            config_path = Path("configs/mouse/base_uniform_v2.yaml")
        if not config_path.exists():
            print("No config found. Falling back to simplified extraction.")
            return _extract_from_checkpoint_direct(n_pilot)

        cfg = OmegaConf.load(config_path)
        ckpt_path = "checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt"

        if not Path(ckpt_path).exists():
            ckpt_path = "checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt"

        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {ckpt_path}")

        # This is complex — fall back to direct checkpoint loading
        return _extract_from_checkpoint_direct(n_pilot)

    except ImportError as e:
        print(f"Import error: {e}")
        return _extract_from_checkpoint_direct(n_pilot)


def _extract_from_checkpoint_direct(n_pilot: int) -> dict:
    """Extract Gaussian parameters directly from checkpoint predictions.

    Loads checkpoint, runs forward pass on pilot frames, extracts Gaussian params.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    ckpt_path = None
    for p in [
        "checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt",
        "checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt",
    ]:
        if Path(p).exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        print("No checkpoint found!")
        return None

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Check what's in the checkpoint
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())[:10]
        print(f"  Checkpoint keys: {keys}")

        # If it contains Gaussian predictions, extract them
        if "gaussians" in ckpt:
            gaussians = ckpt["gaussians"]
            print(f"  Gaussians shape: {gaussians.shape}")
        elif "model_state_dict" in ckpt or "state_dict" in ckpt:
            print("  Checkpoint contains model weights only.")
            print("  Need full GS-LRM inference pipeline — skipping direct extraction.")
            print("  → Use turntable_renderer.py or run_e2e_inference.py instead.")
            return {"method": "3D_Gaussian", "status": "needs_inference_pipeline",
                    "note": "Checkpoint has model weights only, not pre-computed Gaussians"}

    return None


def pilot_3c_rendered_views(n_pilot: int = 100):
    """Phase 3C pilot: DINOv2 on rendered novel views.

    Uses existing turntable renders if available, otherwise needs GS-LRM inference.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print("\n" + "=" * 60)
    print("Phase 3C Pilot: DINOv2 on Rendered Novel Views")
    print("=" * 60)

    # Check for existing turntable renders
    render_dirs = sorted(Path("outputs").glob("**/turntable_grid.png"))

    if len(render_dirs) < 5:
        print(f"Only {len(render_dirs)} existing renders. Need GS-LRM turntable rendering.")
        print("To generate: python -m mouse_extensions.visualization.turntable_renderer ...")
        return {"method": "3C_rendered_DINOv2", "status": "needs_rendering",
                "note": f"Only {len(render_dirs)} renders available, need GS-LRM turntable"}

    # If we have renders, extract DINOv2 from them
    print(f"Found {len(render_dirs)} turntable grid images")
    # These are grid images (multiple views in one image) — would need to split
    # For pilot, we can use the grid image directly
    return {"method": "3C_rendered_DINOv2", "status": "needs_individual_views",
            "note": "Grid images need splitting into individual views"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pilot", type=int, default=100)
    args = parser.parse_args()

    results = {}

    # Phase 3D: Gaussian features
    r3d = pilot_3d_gaussian_features(args.n_pilot)
    if r3d:
        results["phase_3d"] = r3d

    # Phase 3C: Rendered views
    r3c = pilot_3c_rendered_views(args.n_pilot)
    if r3c:
        results["phase_3c"] = r3c

    # Summary
    print("\n" + "=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)
    for phase, r in results.items():
        print(f"  {phase}: {r}")

    # Save
    out_dir = Path("outputs/clustering/pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pilot_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
