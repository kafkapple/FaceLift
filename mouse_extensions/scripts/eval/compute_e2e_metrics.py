#!/usr/bin/env python3
"""
Compute metrics for E2E inference outputs vs GT.

Supports multiple evaluation protocols for fair literature comparison:
  - full_white:   White-BG composite + full-image (LGM/PoseSplatter standard)
  - fg_only:      Binary mask (alpha > 0.5) foreground-only (diagnostic)
  - alpha_weight:  Continuous alpha-weighted (GS-LRM training consistent)

Also computes PoseSplatter-compatible metrics:
  - Masked L1:    sum|pred-gt| / (3 * sum(mask))
  - Silhouette IoU: rendered alpha vs GT alpha

Usage:
    # All modes at once
    python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
        --output_dir outputs/h5_e2e/cfgr_ckpt10000 \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5

    # Compare two experiments
    python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
        --output_dir outputs/h5_e2e/cfgr_ckpt10000 outputs/h5_e2e/baseline_ckpt5000 \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5

    # Skip input view from evaluation
    python -m mouse_extensions.scripts.eval.compute_e2e_metrics \
        --output_dir outputs/h5_e2e/cfgr_ckpt10000 \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --skip_input_view 0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mouse_extensions.evaluation.metrics import MetricsComputer, MetricResult


def load_image_rgba(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load image and separate into RGB [0,1] and alpha [0,1].

    Returns:
        rgb: (H, W, 3) float64 in [0,1]
        alpha: (H, W) float64 in [0,1] or None if no alpha channel
    """
    img = np.array(Image.open(path)).astype(np.float64) / 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    elif img.ndim == 3 and img.shape[2] == 3:
        return img, None
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")


def composite_white_bg(rgb: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    """Composite RGB onto white background using alpha.

    This matches the GS-LRM dataset loading and PoseSplatter evaluation protocol.
    """
    if alpha is None:
        return rgb
    alpha_3d = alpha[:, :, np.newaxis]
    return rgb * alpha_3d + (1.0 - alpha_3d)  # white = 1.0


def compute_psnr_full(pred: np.ndarray, gt: np.ndarray) -> float:
    """Full-image PSNR (no masking). Standard for LGM, PoseSplatter."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_psnr_masked(pred: np.ndarray, gt: np.ndarray,
                        mask: np.ndarray) -> float:
    """Foreground-only PSNR with binary mask."""
    mask_3d = mask[:, :, np.newaxis]
    n_fg = mask.sum()
    if n_fg < 1:
        return 0.0
    mse = np.sum((pred - gt) ** 2 * mask_3d) / (n_fg * 3)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_psnr_alpha_weighted(pred: np.ndarray, gt: np.ndarray,
                                alpha: np.ndarray) -> float:
    """Alpha-weighted PSNR (continuous weights). GS-LRM training consistent."""
    alpha_3d = alpha[:, :, np.newaxis]
    alpha_sum = alpha.sum()
    if alpha_sum < 1e-8:
        return 0.0
    mse = np.sum((pred - gt) ** 2 * alpha_3d) / (alpha_sum * 3)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_ssim_full(pred: np.ndarray, gt: np.ndarray) -> float:
    """Full-image SSIM."""
    try:
        from skimage.metrics import structural_similarity as ssim
        pred_u8 = (pred * 255).astype(np.uint8)
        gt_u8 = (gt * 255).astype(np.uint8)
        return float(ssim(pred_u8, gt_u8, channel_axis=2, data_range=255))
    except ImportError:
        return 0.0


def compute_masked_l1(pred: np.ndarray, gt: np.ndarray,
                      mask: np.ndarray) -> float:
    """PoseSplatter masked L1: sum|pred-gt| / (3 * sum(mask))."""
    mask_3d = mask[:, :, np.newaxis]
    n_fg = mask.sum()
    if n_fg < 1:
        return 0.0
    return float(np.sum(np.abs(pred - gt) * mask_3d) / (3 * n_fg))


def extract_silhouette_from_white_bg(rgb: np.ndarray,
                                     threshold: float = 0.98) -> np.ndarray:
    """Extract binary silhouette mask from white-bg rendered image.

    For GS-LRM renders without alpha channel: pixels significantly
    different from white are considered foreground.

    Args:
        rgb: (H, W, 3) in [0,1]
        threshold: pixels with ALL channels > threshold are background
    Returns:
        (H, W) binary mask (1=foreground, 0=background)
    """
    is_bg = np.all(rgb > threshold, axis=2)
    return (~is_bg).astype(np.float64)


def compute_silhouette_iou(pred_alpha: Optional[np.ndarray],
                           gt_alpha: np.ndarray,
                           pred_rgb: Optional[np.ndarray] = None,
                           threshold: float = 0.5) -> float:
    """Silhouette IoU between predicted and GT masks.

    If pred_alpha is None but pred_rgb is provided, extracts silhouette
    from white-bg render (common for GS-LRM outputs without alpha).
    """
    if pred_alpha is not None:
        pred_mask = (pred_alpha > threshold).astype(np.float64)
    elif pred_rgb is not None:
        pred_mask = extract_silhouette_from_white_bg(pred_rgb)
    else:
        return 0.0
    gt_mask = (gt_alpha > threshold).astype(np.float64)
    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).astype(np.float64).sum()
    if union < 1:
        return 0.0
    return float(intersection / union)


def compute_all_metrics(
    render_path: Path,
    gt_path: Path,
    sample_id: str = "",
) -> Dict[str, float]:
    """Compute all metric variants for a single view pair.

    Returns dict with keys:
        psnr_full_white, psnr_fg_only, psnr_alpha_weighted,
        ssim_full_white, masked_l1, silhouette_iou, fg_fraction
    """
    # Load images
    render_rgb, render_alpha = load_image_rgba(render_path)
    gt_rgb, gt_alpha = load_image_rgba(gt_path)

    # Ensure same shape
    if render_rgb.shape[:2] != gt_rgb.shape[:2]:
        h, w = gt_rgb.shape[:2]
        render_pil = Image.fromarray((render_rgb * 255).astype(np.uint8))
        render_pil = render_pil.resize((w, h), Image.BILINEAR)
        render_rgb = np.array(render_pil).astype(np.float64) / 255.0
        if render_alpha is not None:
            alpha_pil = Image.fromarray((render_alpha * 255).astype(np.uint8))
            alpha_pil = alpha_pil.resize((w, h), Image.BILINEAR)
            render_alpha = np.array(alpha_pil).astype(np.float64) / 255.0

    # Binary mask from GT alpha
    if gt_alpha is not None:
        binary_mask = (gt_alpha > 0.5).astype(np.float64)
    else:
        binary_mask = np.ones(gt_rgb.shape[:2], dtype=np.float64)

    # FG fraction
    total_pixels = binary_mask.size
    fg_pixels = binary_mask.sum()
    fg_fraction = fg_pixels / total_pixels

    # --- Mode 1: White-BG full-image (LGM/PoseSplatter standard) ---
    gt_white = composite_white_bg(gt_rgb, gt_alpha)
    render_white = composite_white_bg(render_rgb, render_alpha)
    psnr_full_white = compute_psnr_full(render_white, gt_white)
    ssim_full_white = compute_ssim_full(render_white, gt_white)

    # --- Mode 2: FG-only binary masked ---
    psnr_fg_only = compute_psnr_masked(render_white, gt_white, binary_mask)

    # --- Mode 3: Alpha-weighted (GS-LRM training) ---
    if gt_alpha is not None:
        psnr_alpha_weighted = compute_psnr_alpha_weighted(
            render_white, gt_white, gt_alpha
        )
    else:
        psnr_alpha_weighted = psnr_full_white

    # --- PoseSplatter metrics ---
    masked_l1 = compute_masked_l1(render_white, gt_white, binary_mask)
    iou = compute_silhouette_iou(
        render_alpha,
        gt_alpha if gt_alpha is not None else binary_mask,
        pred_rgb=render_white,  # Fallback: extract silhouette from white-bg
    )

    return {
        "psnr_full_white": psnr_full_white,
        "psnr_fg_only": psnr_fg_only,
        "psnr_alpha_weighted": psnr_alpha_weighted,
        "ssim_full_white": ssim_full_white,
        "masked_l1": masked_l1,
        "silhouette_iou": iou,
        "fg_fraction": fg_fraction,
    }


def compute_e2e_metrics(
    output_dir: str,
    data_dir: str,
    views: list = None,
    skip_input_view: int = None,
    device: str = "cuda",
    no_lpips: bool = False,
) -> dict:
    """Compute all metric variants for E2E inference outputs."""
    output_path = Path(output_dir)
    data_path = Path(data_dir)
    samples_dir = output_path / "samples"

    if not samples_dir.exists():
        print(f"ERROR: No samples/ directory in {output_dir}")
        return {}

    # Determine views to evaluate
    if views is None:
        views = list(range(6))
    if skip_input_view is not None and skip_input_view in views:
        views = [v for v in views if v != skip_input_view]

    # LPIPS computer (separate, operates on full images)
    lpips_computer = MetricsComputer(device=device, compute_lpips=not no_lpips)

    sample_dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
    )

    # Metric keys to aggregate
    metric_keys = [
        "psnr_full_white", "psnr_fg_only", "psnr_alpha_weighted",
        "ssim_full_white", "masked_l1", "silhouette_iou", "fg_fraction",
    ]

    per_view_results = {v: {k: [] for k in metric_keys} for v in views}
    all_results = {k: [] for k in metric_keys}
    lpips_per_view = {v: [] for v in views}
    lpips_all = []
    errors = []

    print(f"Evaluating {len(sample_dirs)} samples, views: {views}")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        gt_images_dir = data_path / sample_id / "images"

        if not gt_images_dir.exists():
            errors.append(f"GT not found: {sample_id}")
            continue

        cam_subdirs = [d for d in sample_dir.iterdir()
                       if d.is_dir() and d.name.startswith("cam_")]
        render_dir = cam_subdirs[0] if cam_subdirs else sample_dir

        for view_idx in views:
            render_path = render_dir / f"render_view_{view_idx:02d}.png"
            gt_path = gt_images_dir / f"cam_{view_idx:03d}.png"

            if not render_path.exists() or not gt_path.exists():
                continue

            # Compute all metric modes
            metrics = compute_all_metrics(render_path, gt_path, f"{sample_id}_v{view_idx}")

            for k in metric_keys:
                per_view_results[view_idx][k].append(metrics[k])
                all_results[k].append(metrics[k])

            # LPIPS (on white-bg composited images)
            if not no_lpips:
                gt_rgb, gt_alpha = load_image_rgba(gt_path)
                render_rgb, render_alpha = load_image_rgba(render_path)
                gt_white = composite_white_bg(gt_rgb, gt_alpha)
                render_white = composite_white_bg(render_rgb, render_alpha)
                lpips_val = lpips_computer._compute_lpips(
                    render_white.astype(np.float32),
                    gt_white.astype(np.float32),
                )
                lpips_per_view[view_idx].append(lpips_val)
                lpips_all.append(lpips_val)

    # Aggregate
    def agg(values):
        if not values:
            return {"mean": 0, "std": 0}
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    overall = {k: agg(all_results[k]) for k in metric_keys}
    overall["n_samples"] = len(set(
        s.name for s in sample_dirs
        if (data_path / s.name / "images").exists()
    ))
    if lpips_all:
        overall["lpips_full_white"] = agg(lpips_all)

    per_view_agg = {}
    for v in views:
        if any(per_view_results[v][k] for k in metric_keys):
            view_metrics = {k: agg(per_view_results[v][k]) for k in metric_keys}
            if lpips_per_view[v]:
                view_metrics["lpips_full_white"] = agg(lpips_per_view[v])
            per_view_agg[f"view_{v}"] = view_metrics

    summary = {
        "experiment": output_path.name,
        "timestamp": datetime.now().isoformat(),
        "protocol_version": "v2.0",
        "n_samples": len(sample_dirs),
        "n_evaluated": overall["n_samples"],
        "views_evaluated": views,
        "overall": overall,
        "per_view": per_view_agg,
    }

    if errors:
        summary["errors"] = errors[:10]

    return summary


def print_summary(summary: dict, label: str = ""):
    """Pretty print multi-mode metrics summary."""
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    o = summary["overall"]
    n = summary.get("n_evaluated", 0)
    views = summary.get("views_evaluated", [])
    fg_pct = o.get("fg_fraction", {}).get("mean", 0) * 100

    print(f"\n  Samples: {n} | Views: {views} | FG fraction: {fg_pct:.1f}%")

    print(f"\n  {'Mode':<25} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'L1':>8} {'IoU':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Full-image white-bg (literature standard)
    psnr_fw = o.get("psnr_full_white", {}).get("mean", 0)
    ssim_fw = o.get("ssim_full_white", {}).get("mean", 0)
    lpips_fw = o.get("lpips_full_white", {}).get("mean", 0) if "lpips_full_white" in o else 0
    l1 = o.get("masked_l1", {}).get("mean", 0)
    iou = o.get("silhouette_iou", {}).get("mean", 0)
    print(f"  {'full_white (literature)':25} {psnr_fw:8.2f} {ssim_fw:8.4f} {lpips_fw:8.4f} {l1:8.4f} {iou:8.3f}")

    # FG-only
    psnr_fg = o.get("psnr_fg_only", {}).get("mean", 0)
    print(f"  {'fg_only (diagnostic)':25} {psnr_fg:8.2f} {'':>8} {'':>8} {'':>8} {'':>8}")

    # Alpha-weighted
    psnr_aw = o.get("psnr_alpha_weighted", {}).get("mean", 0)
    print(f"  {'alpha_weighted (GS-LRM)':25} {psnr_aw:8.2f} {'':>8} {'':>8} {'':>8} {'':>8}")

    # Per-view breakdown (full_white only for compactness)
    print(f"\n  Per-view (full_white):")
    for vk in sorted(summary.get("per_view", {}).keys()):
        vv = summary["per_view"][vk]
        vp = vv.get("psnr_full_white", {}).get("mean", 0)
        vs = vv.get("ssim_full_white", {}).get("mean", 0)
        vl = vv.get("lpips_full_white", {}).get("mean", 0) if "lpips_full_white" in vv else 0
        vl1 = vv.get("masked_l1", {}).get("mean", 0)
        print(f"    {vk}: PSNR={vp:.2f}, SSIM={vs:.4f}, LPIPS={vl:.4f}, L1={vl1:.4f}")

    # Per-view FG-only
    print(f"\n  Per-view (fg_only):")
    for vk in sorted(summary.get("per_view", {}).keys()):
        vv = summary["per_view"][vk]
        vp = vv.get("psnr_fg_only", {}).get("mean", 0)
        print(f"    {vk}: PSNR={vp:.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Compute E2E inference metrics (multi-mode)")
    parser.add_argument(
        "--output_dir", type=str, nargs="+", required=True,
        help="E2E output directories (1 or more for comparison)",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="GT dataset root (e.g. M5)",
    )
    parser.add_argument(
        "--views", type=int, nargs="+", default=None,
        help="View indices to evaluate (default: all 6)",
    )
    parser.add_argument(
        "--skip_input_view", type=int, default=None,
        help="Skip input view from evaluation (e.g. 0 for MVDiffusion input)",
    )
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results = []
    for out_dir in args.output_dir:
        print(f"\nProcessing: {out_dir}")
        summary = compute_e2e_metrics(
            output_dir=out_dir,
            data_dir=args.data_dir,
            views=args.views,
            skip_input_view=args.skip_input_view,
            device=args.device,
            no_lpips=args.no_lpips,
        )
        results.append(summary)

        # Save per-experiment results
        metrics_path = Path(out_dir) / "metrics_v2.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {metrics_path}")

        print_summary(summary, label=Path(out_dir).name)

    # Comparison table
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"  COMPARISON (all modes)")
        print(f"{'=' * 70}")
        print(f"\n  {'Experiment':<25} {'PSNR_wh':>8} {'PSNR_fg':>8} {'PSNR_aw':>8} {'SSIM':>8} {'L1':>8} {'IoU':>6} {'N':>4}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*4}")
        for r in results:
            o = r["overall"]
            p_wh = o.get("psnr_full_white", {}).get("mean", 0)
            p_fg = o.get("psnr_fg_only", {}).get("mean", 0)
            p_aw = o.get("psnr_alpha_weighted", {}).get("mean", 0)
            ssim = o.get("ssim_full_white", {}).get("mean", 0)
            l1 = o.get("masked_l1", {}).get("mean", 0)
            iou = o.get("silhouette_iou", {}).get("mean", 0)
            n = r.get("n_evaluated", 0)
            print(f"  {r['experiment']:<25} {p_wh:>8.2f} {p_fg:>8.2f} {p_aw:>8.2f} {ssim:>8.4f} {l1:>8.4f} {iou:>6.3f} {n:>4}")
        print()


if __name__ == "__main__":
    main()
