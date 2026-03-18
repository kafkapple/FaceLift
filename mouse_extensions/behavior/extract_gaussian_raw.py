"""Extract raw Gaussian parameters as compact NPZ (no PLY).

Runs GS-LRM inference → saves xyz/opacity/scale/rotation as float16 NPZ.
Supports post-hoc pruning at any threshold without re-inference.

Usage on gpu03:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.behavior.extract_gaussian_raw \
        --start_frame 0 --end_frame 3600

Output: outputs/report/clustering/features/gaussians_raw/{frame:06d}.npz (~8MB each)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict


def main():
    from mouse_extensions.behavior.paths import FEATURES_DIR, GPU03_M5_DATA, ensure_dirs
    ensure_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=GPU03_M5_DATA)
    parser.add_argument("--checkpoint", default="checkpoints/gslrm/base_uniform_v2_6view_v2/best_psnr.pt")
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--output_dir", default=str(FEATURES_DIR / "gaussians_raw"))
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=3600)
    parser.add_argument("--device", default="cuda")
    # Feature extraction options
    parser.add_argument("--save_raw", action="store_true", default=True,
                        help="Save raw params NPZ (float16, ~8MB/frame)")
    parser.add_argument("--save_features", action="store_true", default=True,
                        help="Also save compact feature vector per frame")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = FEATURES_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Frame jump masking
    jump_frames = set()
    for idx in [1180, 2360, 3540]:
        for d in range(-2, 3):
            jump_frames.add(idx + d)

    # Find frames to process
    data_dir = Path(args.data_dir)
    frames_to_process = []
    for fi in range(args.start_frame, args.end_frame):
        if fi in jump_frames:
            continue
        frame_dir = data_dir / f"{fi:06d}"
        if not frame_dir.exists():
            continue
        npz_path = output_dir / f"{fi:06d}.npz"
        if npz_path.exists():
            continue  # skip existing
        frames_to_process.append(fi)

    print(f"Frames to process: {len(frames_to_process)}")
    print(f"Output: {output_dir}")

    if not frames_to_process:
        print("All frames already processed!")
        return

    # Load model
    print(f"Loading GS-LRM from {args.checkpoint}...")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference

    model = GSLRMInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("Model loaded")

    # Process frames
    all_features = []
    all_frame_indices = []
    t0 = time.time()

    for i, fi in enumerate(frames_to_process):
        frame_dir = data_dir / f"{fi:06d}"

        try:
            from mouse_extensions.inference.gslrm_pipeline import load_sample_data

            # Load sample using existing pipeline
            images, c2ws, fxfycxcys, index = load_sample_data(
                str(frame_dir), image_size=512, device=args.device,
            )

            # Predict using existing GSLRMInference
            result = model.predict(images, c2ws, fxfycxcys, index)

            # Extract raw Gaussian parameters
            gaussians = result.gaussians[0]  # first (only) batch element
            xyz = gaussians.get_xyz.detach().cpu().half().numpy()       # (N, 3)
            opacity = gaussians.get_opacity.detach().cpu().half().numpy()  # (N, 1)
            scale = gaussians.get_scaling.detach().cpu().half().numpy()    # (N, 3)
            rotation = gaussians.get_rotation.detach().cpu().half().numpy()  # (N, 4)

            # Save raw params NPZ
            if args.save_raw:
                npz_path = output_dir / f"{fi:06d}.npz"
                np.savez_compressed(npz_path, xyz=xyz, opacity=opacity,
                                    scale=scale, rotation=rotation)

            # Extract compact feature vector
            if args.save_features:
                feat = _extract_feature(xyz.astype(np.float32),
                                       opacity.astype(np.float32),
                                       scale.astype(np.float32))
                all_features.append(feat)
                all_frame_indices.append(fi)

            # Clear GPU cache
            del result, gaussians, images, c2ws, fxfycxcys, index
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Frame {fi}: ERROR {str(e)[:60]}")
            torch.cuda.empty_cache()
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(frames_to_process) - i - 1) / rate
            size = (output_dir / f"{fi:06d}.npz").stat().st_size / 1e6 if (output_dir / f"{fi:06d}.npz").exists() else 0
            print(f"  {i+1}/{len(frames_to_process)} ({rate:.1f} f/s, ETA {eta/60:.0f}min, NPZ {size:.1f}MB)")

    # Save all features
    if all_features and args.save_features:
        features = np.stack(all_features)
        indices = np.array(all_frame_indices)
        feat_path = feature_dir / "gaussian_raw_features.npz"
        np.savez_compressed(feat_path, features=features, frame_indices=indices)
        print(f"Features saved: {feat_path} ({features.shape})")

    print(f"\nDone. {len(frames_to_process)} frames in {(time.time()-t0)/60:.1f} min")


def _extract_feature(xyz, opacity, scale, top_k=20000):
    """Extract compact behavior feature from raw Gaussian params.

    Applies opacity-based top-K pruning, then computes summary statistics.
    """
    N = xyz.shape[0]
    op = opacity.flatten()

    # Top-K by opacity (foreground focus)
    if N > top_k:
        idx = np.argsort(op)[-top_k:]
        xyz, op, scale = xyz[idx], op[idx], scale[idx].reshape(-1, 3)
    else:
        scale = scale.reshape(-1, 3)

    f = []
    # Position stats (12)
    f.extend(xyz.mean(0)); f.extend(xyz.std(0))
    f.extend(np.percentile(xyz, 25, axis=0)); f.extend(np.percentile(xyz, 75, axis=0))
    # Opacity stats (3)
    f.extend([op.mean(), op.std(), float(np.median(op))])
    # Scale stats (9)
    f.extend(scale.mean(0)); f.extend(scale.std(0))
    f.append(scale[:, 0].mean() - scale[:, 2].mean())  # anisotropy
    f.extend(np.percentile(scale, [25, 75], axis=0).mean(1))
    # Shape PCA (3)
    xyz_c = xyz - xyz.mean(0)
    eigvals = np.linalg.eigvalsh(np.cov(xyz_c.T))
    f.extend(sorted(eigvals, reverse=True))
    # BBox + count (4)
    f.extend(xyz.max(0) - xyz.min(0))
    f.append(float(len(xyz)))

    return np.array(f, dtype=np.float32)


if __name__ == "__main__":
    main()
