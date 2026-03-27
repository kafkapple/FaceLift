"""Test orientation-aware filter on a few frames with before/after comparison.

Renders novel bottom view with and without the Z-aligned flat Gaussian filter,
saving side-by-side comparison images + statistics.

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.test_orientation_filter \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --n-frames 5 \
        --output-dir outputs/analysis/mouse/filtering/orientation_filter_test
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
from mouse_extensions.model.orientation_filter import suppress_z_aligned_flat
from mouse_extensions.scripts.eval.fl_gt_view_comparison import (
    get_novel_camera,
    render_gaussian_at_view,
)

# Bottom view: elevation -80° (looking up from below)
BOTTOM_CAM_PARAMS = {"elevation": -80, "azimuth": 0}

# Default filter settings matching gslrm_pipeline.py
FILTER_PARAMS = dict(
    opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
    crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
)

TEST_RANGE = (3240, 3599)  # M5t2 test split (80:10:10 temporal)


def render_bottom_view(gaussians, resolution: int, device: str) -> np.ndarray:
    """Render from bottom camera, return (H, W, 3) uint8."""
    cam = get_novel_camera(
        BOTTOM_CAM_PARAMS["elevation"], BOTTOM_CAM_PARAMS["azimuth"],
        resolution=resolution,
    )
    try:
        return render_gaussian_at_view(gaussians, cam, resolution=resolution, device=device)
    except Exception as e:
        print(f"  [render error] {e}")
        return np.full((resolution, resolution, 3), 80, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Test orientation-aware filter")
    parser.add_argument("--m5-dir", required=True)
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml")
    parser.add_argument("--checkpoint",
                        default="/node_data/joon/checkpoints/FaceLift/gslrm/"
                                "base_uniform_v2_6view_v2/best_psnr.pt")
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/filtering/orientation_filter_test")
    parser.add_argument("--seed", type=int, default=42)
    # Filter params
    parser.add_argument("--ratio-thresh", type=float, default=30.0)
    parser.add_argument("--z-align-thresh", type=float, default=0.15)
    parser.add_argument("--opacity-ceil", type=float, default=0.4)
    parser.add_argument("--attenuation", type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    m5_dir = Path(args.m5_dir)

    # Sample test frames
    rng = np.random.RandomState(args.seed)
    all_frames = list(range(TEST_RANGE[0], TEST_RANGE[1] + 1))
    n = min(args.n_frames, len(all_frames))
    frame_indices = sorted(rng.choice(all_frames, n, replace=False).tolist())
    print(f"Testing on {n} frames: {frame_indices}")

    print(f"Loading model from {args.checkpoint}...")
    model = GSLRMInference(config_path=args.config, checkpoint_path=args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_stats = []

    for fi in frame_indices:
        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            print(f"  [skip] {fi:06d} not found")
            continue

        print(f"\n--- Frame {fi:06d} ---")
        imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)
        result = model.predict(imgs, c2ws, fxfys, idx)

        # ---- BEFORE: standard filters only ----
        gs_before = result.gaussians[0]
        gs_before.apply_all_filters(**FILTER_PARAMS)
        n_before = gs_before.get_xyz.shape[0]
        img_before = render_bottom_view(gs_before, args.resolution, device)

        # ---- AFTER: standard filters + orientation filter ----
        # Re-run inference (apply_all_filters is in-place)
        result2 = model.predict(imgs, c2ws, fxfys, idx)
        gs_after = result2.gaussians[0]
        gs_after.apply_all_filters(**FILTER_PARAMS)

        stats = suppress_z_aligned_flat(
            gs_after,
            ratio_thresh=args.ratio_thresh,
            z_align_thresh=args.z_align_thresh,
            opacity_ceil=args.opacity_ceil,
            attenuation=args.attenuation,
        )
        n_after_effective = gs_after.get_xyz.shape[0]
        img_after = render_bottom_view(gs_after, args.resolution, device)

        # Print statistics
        print(f"  Total: {stats['n_total']:,}")
        print(f"  Flat (ratio>{args.ratio_thresh}): {stats['n_flat']:,} "
              f"({100*stats['n_flat']/max(stats['n_total'],1):.1f}%)")
        print(f"  Z-aligned (>{args.z_align_thresh}): {stats['n_z_aligned']:,} "
              f"({100*stats['n_z_aligned']/max(stats['n_total'],1):.1f}%)")
        print(f"  Suppressed: {stats['n_suppressed']:,} "
              f"({100*stats['n_suppressed']/max(stats['n_total'],1):.1f}%)")

        all_stats.append({"frame": fi, **stats})

        # Save side-by-side comparison
        label_h = 30
        h, w = img_before.shape[:2]
        canvas = np.full((h + label_h, w * 2 + 10, 3), 255, dtype=np.uint8)

        # Before
        canvas[label_h:label_h + h, :w] = img_before
        cv2.putText(canvas, "Before (standard filters)", (5, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Separator
        canvas[:, w:w + 10] = 128

        # After
        canvas[label_h:label_h + h, w + 10:w + 10 + w] = img_after
        cv2.putText(canvas, f"After (suppressed {stats['n_suppressed']:,})", (w + 15, 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        fname = out / f"comparison_{fi:06d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {fname}")

    # Summary
    if all_stats:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        avg_suppressed = np.mean([s["n_suppressed"] for s in all_stats])
        avg_total = np.mean([s["n_total"] for s in all_stats])
        avg_pct = 100 * avg_suppressed / max(avg_total, 1)
        print(f"  Avg suppressed: {avg_suppressed:.0f} / {avg_total:.0f} ({avg_pct:.1f}%)")
        print(f"  Avg flat: {np.mean([s['n_flat'] for s in all_stats]):.0f}")
        print(f"  Avg Z-aligned: {np.mean([s['n_z_aligned'] for s in all_stats]):.0f}")
        print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
