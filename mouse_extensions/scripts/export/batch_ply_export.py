"""Batch PLY export: best_psnr.pt → per-frame Gaussian .ply files.

Runs GS-LRM inference on each frame and saves the resulting 3D Gaussians as .ply.
Applies visibility filter (n_filter >= 2 cameras) before saving.

Expected output: ~56-100 KB/frame PLY files (visibility-filtered).
120 frames ≈ 7-12 MB total.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.export.batch_ply_export \
        --config mouse_extensions/behavior/cinematic_default.yaml \
        --output-dir outputs/datasets/novel_view/ply/mouse_6view_best \
        --frame-range 195:315

    # Dry-run (3 frames only):
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.export.batch_ply_export \
        --config mouse_extensions/behavior/cinematic_default.yaml \
        --dry-run

    # Skip already-exported frames (resume-safe):
    CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.export.batch_ply_export \
        --skip-existing
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from mouse_extensions.behavior.multiview_visibility_filter import (
    compute_visibility_counts,
)
from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.visualization.gaussian_export import GaussianExporter


def export_ply_sequence(
    config_path: str,
    output_dir: str,
    frame_range: str = "195:315",
    n_filter: int = 2,
    use_fp16: bool = False,
    dry_run: bool = False,
    skip_existing: bool = True,
    device: str = "cuda",
) -> dict:
    """
    Export per-frame PLY files from GS-LRM best checkpoint.

    Args:
        config_path:  cinematic_default.yaml (provides checkpoint + data paths)
        output_dir:   Directory to write .ply files into
        frame_range:  "start:end" frame index range (inclusive start, exclusive end)
        n_filter:     Min camera visibility count for Gaussian retention
        use_fp16:     Write fp16 PLY (smaller, ~half size, less precision)
        dry_run:      Only process first 3 frames for quick validation
        skip_existing: Skip frames whose .ply already exists (resume support)
        device:       CUDA device string

    Returns:
        Summary dict with saved count, total size MB, failed frames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    m5_dir = Path(cfg["model"]["m5_dir"])
    resolution = cfg["global"].get("resolution", 512)

    # Parse frame range
    start, end = map(int, frame_range.split(":"))
    frame_indices = list(range(start, end))
    if dry_run:
        frame_indices = frame_indices[:3]
        print("[DRY RUN] Processing first 3 frames only.")

    print(f"Frames: {len(frame_indices)} ({start}~{end-1})")
    print(f"Output: {output_dir}")
    print(f"Checkpoint: {cfg['model']['checkpoint']}")

    # Load model
    print("\nLoading GS-LRM model...")
    model = GSLRMInference(
        config_path=cfg["model"]["config"],
        checkpoint_path=cfg["model"]["checkpoint"],
        device=device,
    )
    # Apply num_input_views override from cinematic config
    if "num_input_views" in cfg.get("model", {}):
        model.config.model.num_input_views = cfg["model"]["num_input_views"]
        print(f"  num_input_views patched → {cfg['model']['num_input_views']}")

    exporter = GaussianExporter()
    failed = []
    saved_paths = []
    t0 = time.time()

    for i, fi in enumerate(frame_indices):
        ply_path = output_dir / f"frame_{fi:06d}.ply"

        # Resume support
        if skip_existing and ply_path.exists() and ply_path.stat().st_size > 0:
            saved_paths.append(ply_path)
            continue

        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            print(f"  [skip] {fi:06d} — directory not found")
            continue

        try:
            # Infer Gaussians from 6-view input
            imgs, c2ws, fxfys, idx = load_sample_data(
                str(fd), image_size=resolution, device=device
            )
            result = model.predict(imgs, c2ws, fxfys, idx)
            gaussians = result.gaussians[0]

            # Apply visibility filter
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            vc = compute_visibility_counts(xyz, str(fd), n_views=6)
            vis_mask = vc >= n_filter

            # Export PLY
            GaussianExporter.to_ply(
                gaussians,
                str(ply_path),
                use_fp16=use_fp16,
                enable_gs_viewer=True,
                filter_mask=vis_mask,
            )

            size_kb = ply_path.stat().st_size / 1024
            saved_paths.append(ply_path)

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(frame_indices) - i - 1) / max(rate, 1e-6)
            print(
                f"  [{i+1:3d}/{len(frame_indices)}] frame_{fi:06d}.ply "
                f"({size_kb:.0f} KB) | ETA {eta:.0f}s"
            )

        except Exception as e:
            print(f"  [ERROR] frame {fi:06d}: {e}")
            failed.append((fi, str(e)))

    # Summary
    total_mb = sum(p.stat().st_size for p in saved_paths) / 1024**2
    avg_kb = (total_mb * 1024) / max(len(saved_paths), 1)

    print(f"\n{'='*55}")
    print(f"  Saved:   {len(saved_paths)}/{len(frame_indices)} frames")
    print(f"  Total:   {total_mb:.1f} MB")
    print(f"  Avg:     {avg_kb:.0f} KB/frame")
    print(f"  Output:  {output_dir}")
    if failed:
        print(f"  Failed:  {len(failed)} → {[f[0] for f in failed[:5]]}")

    return {
        "saved": len(saved_paths),
        "total_mb": total_mb,
        "avg_kb": avg_kb,
        "failed": failed,
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch PLY export from GS-LRM checkpoint")
    parser.add_argument(
        "--config", default="mouse_extensions/behavior/cinematic_default.yaml",
        help="Cinematic config YAML (provides checkpoint + data paths)"
    )
    parser.add_argument(
        "--output-dir",
        # 260814: 구 기본값 "/home/joon/data/derived/..." 은 개인 계정 심링크 경유라
        # 계정 정리와 함께 끊긴다. 호스트 중립 상대경로로 바꾸고, 필요하면 --output-dir 로 지정.
        default="outputs/novel_view/ply/mouse_6view_best",
        help="Output directory for .ply files"
    )
    parser.add_argument(
        "--frame-range", default="195:315",
        help="Frame range 'start:end' (default: 195:315)"
    )
    parser.add_argument("--n-filter", type=int, default=2,
                        help="Min camera visibility count (default: 2)")
    parser.add_argument("--fp16", action="store_true",
                        help="Write fp16 PLY (~half file size)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process only 3 frames for validation")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-export even if .ply already exists")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    export_ply_sequence(
        config_path=args.config,
        output_dir=args.output_dir,
        frame_range=args.frame_range,
        n_filter=args.n_filter,
        use_fp16=args.fp16,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip,
        device=args.device,
    )


if __name__ == "__main__":
    main()
