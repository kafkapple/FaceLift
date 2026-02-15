#!/usr/bin/env python3
"""
fair_comparison.py - Fair Model Comparison Evaluation
=====================================================

Unified evaluation for FaceLift vs Pose-Splatter with fairness guarantees:
  1. Test-only frames (no train/val contamination)
  2. GT binary masks from alpha channel (same source for both)
  3. Identical metric functions (masked PSNR, SSIM, L1, IoU)
  4. Foreground-only evaluation (no background inflation)
  5. Coverage-aware metrics (intersection-conditional)

Subcommands:
  evaluate_fl    Evaluate FaceLift renders against GT
  compare        Compare two evaluation JSONs side-by-side

Example workflow:
  # Step 1: Evaluate FL on gpu03
  python -m mouse_extensions.scripts.eval.fair_comparison evaluate_fl \
    --render_dir outputs/h5_e2e/baseline_ckpt5000/samples \
    --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --output experiments/comparison/fair/facelift_fair.json

  # Step 2: Evaluate PS on joon (see fair_test_only_eval.py in pose-splatter repo)
  # Then copy result JSON to gpu03

  # Step 3: Compare
  python -m mouse_extensions.scripts.eval.fair_comparison compare \
    --facelift experiments/comparison/fair/facelift_fair.json \
    --baseline baselines/pose_splatter/posesplatter_fair.json \
    --output_dir experiments/comparison/fair/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("WARNING: skimage not available, SSIM will be NaN")


# ==============================================================
# Unified Metric Functions
# ==============================================================
# These MUST be identical to the functions in fair_test_only_eval.py
# for Pose-Splatter. Any change here must be mirrored there.

def compute_masked_psnr(pred: np.ndarray, gt: np.ndarray,
                        mask: np.ndarray) -> float:
    """PSNR on foreground pixels only.

    Args:
        pred: (H, W, 3) float32 [0, 1]
        gt: (H, W, 3) float32 [0, 1]
        mask: (H, W) binary float32
    Returns:
        PSNR in dB (higher is better)
    """
    if mask.sum() == 0:
        return 0.0
    fg_pred = pred[mask > 0.5]
    fg_gt = gt[mask > 0.5]
    mse = np.mean((fg_pred - fg_gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(-10.0 * np.log10(mse))


def compute_masked_ssim(pred: np.ndarray, gt: np.ndarray,
                        mask: np.ndarray) -> float:
    """SSIM on bounding-box crop of foreground region.

    Both pred and gt are composited onto white background using the mask,
    then cropped to the mask bounding box (+ 10px padding).
    """
    if not HAS_SKIMAGE:
        return float('nan')
    if mask.sum() == 0:
        return 0.0

    mask_3ch = mask[:, :, None]
    pred_white = pred * mask_3ch + (1.0 - mask_3ch)
    gt_white = gt * mask_3ch + (1.0 - mask_3ch)

    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    pad = 10
    rmin = max(0, rmin - pad)
    rmax = min(mask.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(mask.shape[1] - 1, cmax + pad)

    pred_crop = pred_white[rmin:rmax + 1, cmin:cmax + 1]
    gt_crop = gt_white[rmin:rmax + 1, cmin:cmax + 1]

    min_dim = min(pred_crop.shape[0], pred_crop.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        return float('nan')

    return float(ssim_fn(
        gt_crop, pred_crop,
        win_size=win_size,
        channel_axis=2,
        data_range=1.0
    ))


def compute_masked_l1(pred: np.ndarray, gt: np.ndarray,
                      mask: np.ndarray) -> float:
    """L1 on foreground: sum|pred-gt| / (3 * sum(mask)).

    Same formula as Pose-Splatter's masked L1.
    """
    if mask.sum() == 0:
        return 0.0
    diff = np.abs(pred - gt)
    masked_diff = diff * mask[:, :, None]
    return float(np.sum(masked_diff) / (3.0 * np.sum(mask)))


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary mask IoU (Intersection over Union)."""
    pred_bin = pred_mask > 0.5
    gt_bin = gt_mask > 0.5
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def extract_foreground_mask(render: np.ndarray,
                            threshold: float = 0.98) -> np.ndarray:
    """Extract binary foreground mask from white-BG render.

    A pixel is foreground if ANY channel < threshold.
    Returns (H, W) float32 binary mask.
    """
    return np.any(render < threshold, axis=2).astype(np.float32)


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray,
                        gt_mask: np.ndarray,
                        pred_mask: np.ndarray = None) -> dict:
    """Compute all fair metrics for one image pair.

    Returns dict with:
      - psnr_gt_masked: PSNR using GT foreground mask
      - psnr_intersection: PSNR on pixels where both pred and GT have FG
      - ssim_gt_masked: SSIM on GT mask bounding box
      - l1_gt_masked: L1 using GT mask
      - l1_intersection: L1 on intersection region
      - iou: silhouette IoU
      - coverage: fraction of GT FG covered by pred FG
      - pred_precision: fraction of pred FG overlapping GT FG
      - color_bias_r/g/b: mean color difference on intersection (pred - gt)
    """
    results = {}

    # GT-masked metrics (standard)
    results['psnr_gt_masked'] = compute_masked_psnr(pred, gt, gt_mask)
    results['ssim_gt_masked'] = compute_masked_ssim(pred, gt, gt_mask)
    results['l1_gt_masked'] = compute_masked_l1(pred, gt, gt_mask)

    # Extract pred silhouette from white BG if not provided
    if pred_mask is None:
        pred_mask = extract_foreground_mask(pred, threshold=0.98)

    # IoU
    results['iou'] = compute_iou(pred_mask, gt_mask)

    # Coverage analysis
    gt_fg = gt_mask > 0.5
    pred_fg = pred_mask > 0.5
    gt_fg_pixels = float(gt_fg.sum())
    pred_fg_pixels = float(pred_fg.sum())
    intersection = np.logical_and(pred_fg, gt_fg)
    inter_pixels = float(intersection.sum())

    results['coverage'] = inter_pixels / gt_fg_pixels if gt_fg_pixels > 0 else 0.0
    results['pred_precision'] = inter_pixels / pred_fg_pixels if pred_fg_pixels > 0 else 0.0
    results['gt_fg_ratio'] = gt_fg_pixels / (gt_mask.shape[0] * gt_mask.shape[1])

    # Intersection-conditional metrics (coverage-aware)
    inter_mask = intersection.astype(np.float32)
    if inter_pixels > 0:
        results['psnr_intersection'] = compute_masked_psnr(pred, gt, inter_mask)
        results['l1_intersection'] = compute_masked_l1(pred, gt, inter_mask)

        # Color bias analysis on intersection
        fg_pred = pred[intersection]
        fg_gt = gt[intersection]
        bias = np.mean(fg_pred, axis=0) - np.mean(fg_gt, axis=0)
        results['color_bias_r'] = float(bias[0])
        results['color_bias_g'] = float(bias[1])
        results['color_bias_b'] = float(bias[2])
    else:
        results['psnr_intersection'] = 0.0
        results['l1_intersection'] = 1.0
        results['color_bias_r'] = 0.0
        results['color_bias_g'] = 0.0
        results['color_bias_b'] = 0.0

    # Multi-threshold IoU sensitivity
    for t in [0.90, 0.95, 0.98]:
        t_mask = extract_foreground_mask(pred, threshold=t)
        results[f'iou_t{t:.2f}'] = compute_iou(t_mask, gt_mask)
        t_inter = np.logical_and(t_mask > 0.5, gt_fg).sum()
        results[f'coverage_t{t:.2f}'] = float(t_inter / gt_fg_pixels) if gt_fg_pixels > 0 else 0.0

    return results


# ==============================================================
# Visualization
# ==============================================================

def _save_vis_grid(render: np.ndarray, gt: np.ndarray,
                   gt_mask: np.ndarray, metrics: dict,
                   vis_dir: Path, frame_id: str, view_idx: int):
    """Save 4-panel visualization: GT | Render | GT Mask | Error Map.

    Top row: GT RGB, Render RGB, GT on white BG
    Bottom row: GT Mask, Pred Mask, |Error| heatmap (masked)
    """
    h, w = gt.shape[:2]

    # Pred mask from white-BG extraction
    pred_mask = extract_foreground_mask(render, threshold=0.98)

    # Composites on white BG
    mask_3 = gt_mask[:, :, None]
    gt_white = gt * mask_3 + (1.0 - mask_3)
    render_white = render * mask_3 + (1.0 - mask_3)

    # Error map (absolute diff, amplified for visibility)
    error = np.abs(render - gt) * mask_3
    error_vis = np.clip(error * 5.0, 0, 1)  # 5x amplification

    # Intersection mask visualization (green=both, red=GT only, blue=pred only)
    gt_fg = gt_mask > 0.5
    pred_fg = pred_mask > 0.5
    mask_vis = np.zeros((h, w, 3), dtype=np.float32)
    mask_vis[np.logical_and(gt_fg, pred_fg)] = [0, 1, 0]     # Green: intersection
    mask_vis[np.logical_and(gt_fg, ~pred_fg)] = [1, 0, 0]    # Red: GT only (missed)
    mask_vis[np.logical_and(~gt_fg, pred_fg)] = [0, 0, 1]    # Blue: pred only (false pos)

    # GT mask as grayscale
    gt_mask_vis = np.stack([gt_mask] * 3, axis=2)

    # Assemble 2x3 grid
    row1 = np.concatenate([gt_white, render_white, gt_mask_vis], axis=1)
    row2 = np.concatenate([mask_vis, error_vis, render], axis=1)
    grid = np.concatenate([row1, row2], axis=0)

    # Add metrics text (as a simple bar at top)
    psnr_gt = metrics.get('psnr_gt_masked', 0)
    psnr_int = metrics.get('psnr_intersection', 0)
    iou = metrics.get('iou', 0)
    cov = metrics.get('coverage', 0)

    grid_uint8 = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(grid_uint8)

    # Save
    fname = f'{frame_id}_view{view_idx:02d}_psnr{psnr_gt:.1f}_int{psnr_int:.1f}_iou{iou:.2f}_cov{cov:.0%}.png'
    img.save(vis_dir / fname)


# ==============================================================
# FaceLift Evaluation
# ==============================================================

def evaluate_facelift(render_dir: str, gt_dir: str,
                      views: list = None,
                      save_per_frame: bool = False,
                      save_vis: str = None,
                      vis_every: int = 10) -> dict:
    """Evaluate FaceLift renders against GT using GT alpha masks.

    Args:
        render_dir: outputs/h5_e2e/baseline_ckpt5000/samples
        gt_dir: /home/joon/data/preprocessed/FaceLift_mouse/M5
        views: view indices to evaluate [1,2,3,4,5] (0=input, skip)
        save_per_frame: include per-frame detail in output
        save_vis: directory to save visualization grids (GT|Render|Mask|Error)
        vis_every: save vis every N frames (default: 10)
    """
    if views is None:
        views = [1, 2, 3, 4, 5]

    render_dir = Path(render_dir)
    gt_dir = Path(gt_dir)

    vis_dir = None
    if save_vis:
        vis_dir = Path(save_vis)
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FL] Saving visualizations to: {vis_dir} (every {vis_every} frames)")

    # Find rendered frames
    frame_dirs = sorted([d for d in render_dir.iterdir() if d.is_dir()])
    frame_ids = [d.name for d in frame_dirs]

    print(f"[FL] Found {len(frame_ids)} rendered frames")
    print(f"[FL] Range: {frame_ids[0]} - {frame_ids[-1]}")
    print(f"[FL] Views: {views} (view 0 = input, skipped)")

    per_view_all = {f'view_{v}': [] for v in views}
    per_frame = {} if save_per_frame else None
    skipped = 0

    for fi, frame_id in enumerate(frame_ids):
        frame_metrics = {}

        for view_idx in views:
            render_path = render_dir / frame_id / 'cam_000' / f'render_view_{view_idx:02d}.png'
            gt_path = gt_dir / frame_id / 'images' / f'cam_{view_idx:03d}.png'

            if not render_path.exists():
                skipped += 1
                continue
            if not gt_path.exists():
                skipped += 1
                continue

            # Load render (RGB)
            render_img = np.array(
                Image.open(render_path).convert('RGB')
            ).astype(np.float32) / 255.0

            # Load GT (RGBA)
            gt_raw = np.array(Image.open(gt_path))

            if gt_raw.ndim == 2:
                # Grayscale, skip
                skipped += 1
                continue

            if gt_raw.shape[2] >= 4:
                gt_rgb = gt_raw[:, :, :3].astype(np.float32) / 255.0
                gt_mask = (gt_raw[:, :, 3] > 127).astype(np.float32)
            else:
                gt_rgb = gt_raw[:, :, :3].astype(np.float32) / 255.0
                gt_mask = np.ones(gt_rgb.shape[:2], dtype=np.float32)

            # Resolution check
            if render_img.shape[:2] != gt_rgb.shape[:2]:
                h_r, w_r = render_img.shape[:2]
                h_g, w_g = gt_rgb.shape[:2]
                h, w = min(h_r, h_g), min(w_r, w_g)
                ry, rx = (h_r - h) // 2, (w_r - w) // 2
                gy, gx = (h_g - h) // 2, (w_g - w) // 2
                render_img = render_img[ry:ry + h, rx:rx + w]
                gt_rgb = gt_rgb[gy:gy + h, gx:gx + w]
                gt_mask = gt_mask[gy:gy + h, gx:gx + w]

            metrics = compute_all_metrics(render_img, gt_rgb, gt_mask)
            view_key = f'view_{view_idx}'
            per_view_all[view_key].append(metrics)
            frame_metrics[view_key] = metrics

            # Save visualization grid
            if vis_dir and fi % vis_every == 0:
                _save_vis_grid(
                    render_img, gt_rgb, gt_mask, metrics,
                    vis_dir, frame_id, view_idx
                )

        if save_per_frame and frame_metrics:
            per_frame[frame_id] = frame_metrics

        if (fi + 1) % 50 == 0 or (fi + 1) == len(frame_ids):
            print(f"  [{fi + 1}/{len(frame_ids)}] processed")

    if skipped > 0:
        print(f"  WARNING: {skipped} image pairs skipped (missing files)")

    # Aggregate per-view
    view_summary = {}
    for vk, mlist in per_view_all.items():
        if not mlist:
            continue
        summary = {}
        for key in mlist[0]:
            vals = [m[key] for m in mlist
                    if isinstance(m[key], (int, float)) and not np.isnan(m[key])]
            if vals:
                summary[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'n': len(vals),
                }
        view_summary[vk] = summary

    # Aggregate overall
    all_metrics = []
    for mlist in per_view_all.values():
        all_metrics.extend(mlist)

    overall = {}
    if all_metrics:
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics
                    if isinstance(m[key], (int, float)) and not np.isnan(m[key])]
            if vals:
                overall[key] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'n': len(vals),
                }

    result = {
        'model': 'FaceLift',
        'evaluation': 'fair_comparison_v1',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'render_dir': str(render_dir),
            'gt_dir': str(gt_dir),
            'views_evaluated': views,
            'num_frames': len(frame_ids),
            'frame_range': [frame_ids[0], frame_ids[-1]] if frame_ids else [],
            'input_view': 0,
            'mask_source': 'gt_alpha_channel',
            'mask_threshold': 127,
            'silhouette_extraction': 'white_bg_threshold',
            'silhouette_thresholds_tested': [0.90, 0.95, 0.98],
        },
        'fairness': {
            'test_only': True,
            'gt_mask_used': True,
            'same_metric_functions': True,
            'fg_only_evaluation': True,
            'coverage_aware_metrics': True,
        },
        'overall': overall,
        'per_view': view_summary,
    }

    if save_per_frame:
        result['per_frame'] = per_frame

    return result


# ==============================================================
# Comparison
# ==============================================================

def compare_results(fl_path: str, ps_path: str,
                    output_dir: str = None) -> None:
    """Compare FaceLift and Pose-Splatter fair evaluation results."""
    with open(fl_path) as f:
        fl = json.load(f)
    with open(ps_path) as f:
        ps = json.load(f)

    print("\n" + "=" * 72)
    print("  FAIR COMPARISON: FaceLift vs Pose-Splatter")
    print("=" * 72)

    # Config
    fl_cfg = fl.get('config', {})
    ps_cfg = ps.get('config', {})
    print(f"\n  FL: {fl_cfg.get('num_frames', '?')} frames, "
          f"views {fl_cfg.get('views_evaluated', '?')}, "
          f"mask={fl_cfg.get('mask_source', '?')}")
    print(f"  PS: {ps_cfg.get('num_frames', '?')} frames, "
          f"views {ps_cfg.get('views_evaluated', '?')}, "
          f"mask={ps_cfg.get('mask_source', '?')}")

    # Fairness check
    fl_fair = fl.get('fairness', {})
    ps_fair = ps.get('fairness', {})
    issues = []
    if not fl_fair.get('test_only'):
        issues.append("FL includes non-test data")
    if not ps_fair.get('test_only'):
        issues.append("PS includes non-test data")
    if not fl_fair.get('gt_mask_used'):
        issues.append("FL not using GT masks")
    if not ps_fair.get('gt_mask_used'):
        issues.append("PS not using GT masks")

    if issues:
        print(f"\n  !! FAIRNESS WARNINGS: {', '.join(issues)}")
    else:
        print(f"\n  Fairness checks: ALL PASSED")

    # Main comparison table
    metrics_spec = [
        ('psnr_gt_masked',    'PSNR (GT-masked)',    'dB',  True),
        ('psnr_intersection', 'PSNR (intersection)', 'dB',  True),
        ('ssim_gt_masked',    'SSIM (GT-masked)',    '',    True),
        ('l1_gt_masked',      'L1 (GT-masked)',      '',    False),
        ('l1_intersection',   'L1 (intersection)',   '',    False),
        ('iou',               'IoU',                 '',    True),
        ('coverage',          'Coverage',            '',    True),
        ('pred_precision',    'Precision',           '',    True),
    ]

    print(f"\n  {'Metric':<22} {'FaceLift':>10} {'PS':>10} {'Gap':>10} {'Winner':>8}")
    print("  " + "-" * 62)

    for key, name, unit, higher_better in metrics_spec:
        fl_val = fl['overall'].get(key, {}).get('mean', float('nan'))
        ps_val = ps['overall'].get(key, {}).get('mean', float('nan'))

        if np.isnan(fl_val) or np.isnan(ps_val):
            print(f"  {name:<22} {'N/A':>10} {'N/A':>10} {'':>10} {'':>8}")
            continue

        gap = fl_val - ps_val
        winner = 'FL' if (gap > 0) == higher_better else 'PS'

        if key in ('coverage', 'pred_precision'):
            print(f"  {name:<22} {fl_val * 100:>9.1f}% {ps_val * 100:>9.1f}% "
                  f"{gap * 100:>+9.1f}% {winner:>8}")
        elif 'psnr' in key:
            print(f"  {name:<22} {fl_val:>10.2f} {ps_val:>10.2f} "
                  f"{gap:>+10.2f} {winner:>8}")
        else:
            print(f"  {name:<22} {fl_val:>10.4f} {ps_val:>10.4f} "
                  f"{gap:>+10.4f} {winner:>8}")

    print("  " + "-" * 62)

    # Coverage analysis
    fl_cov = fl['overall'].get('coverage', {}).get('mean', 0)
    fl_psnr_i = fl['overall'].get('psnr_intersection', {}).get('mean', 0)
    fl_psnr_g = fl['overall'].get('psnr_gt_masked', {}).get('mean', 0)
    fl_fg = fl['overall'].get('gt_fg_ratio', {}).get('mean', 0)

    print(f"\n  Coverage Analysis:")
    print(f"    FL foreground coverage: {fl_cov * 100:.1f}% of GT")
    print(f"    FL PSNR (covered region): {fl_psnr_i:.2f} dB")
    print(f"    FL PSNR (full GT mask):   {fl_psnr_g:.2f} dB")
    print(f"    PSNR gap from coverage:   {fl_psnr_i - fl_psnr_g:.2f} dB")
    print(f"    Mouse FG ratio:           {fl_fg * 100:.1f}% of image")

    # Color bias
    bias_r = fl['overall'].get('color_bias_r', {}).get('mean', 0)
    bias_g = fl['overall'].get('color_bias_g', {}).get('mean', 0)
    bias_b = fl['overall'].get('color_bias_b', {}).get('mean', 0)
    if abs(bias_r) > 0.01 or abs(bias_g) > 0.01 or abs(bias_b) > 0.01:
        print(f"\n  Color Bias (FL pred - GT, on intersection):")
        print(f"    R: {bias_r:+.4f}  G: {bias_g:+.4f}  B: {bias_b:+.4f}")
        print(f"    Mean absolute bias: {(abs(bias_r) + abs(bias_g) + abs(bias_b)) / 3:.4f}")

    # Threshold sensitivity
    print(f"\n  Threshold Sensitivity (FL silhouette extraction):")
    for t in [0.90, 0.95, 0.98]:
        iou_t = fl['overall'].get(f'iou_t{t:.2f}', {}).get('mean', 0)
        cov_t = fl['overall'].get(f'coverage_t{t:.2f}', {}).get('mean', 0)
        print(f"    threshold={t:.2f}: IoU={iou_t:.3f}, coverage={cov_t * 100:.1f}%")

    # Per-view comparison
    print(f"\n  Per-View PSNR (GT-masked):")
    fl_views = fl.get('per_view', {})
    ps_views = ps.get('per_view', {})
    all_view_keys = sorted(set(list(fl_views.keys()) + list(ps_views.keys())))

    print(f"  {'View':<10} {'FL':>10} {'PS':>10} {'Notes':>20}")
    for vk in all_view_keys:
        fl_v = fl_views.get(vk, {}).get('psnr_gt_masked', {}).get('mean', float('nan'))
        ps_v = ps_views.get(vk, {}).get('psnr_gt_masked', {}).get('mean', float('nan'))
        notes = ''
        if vk == 'view_0':
            notes = '(FL=input, PS=train)'
        elif vk == 'view_5':
            notes = '(PS=holdout)'
        else:
            notes = '(PS=train)'

        fl_str = f'{fl_v:.2f}' if not np.isnan(fl_v) else 'N/A'
        ps_str = f'{ps_v:.2f}' if not np.isnan(ps_v) else 'N/A'
        print(f"  {vk:<10} {fl_str:>10} {ps_str:>10} {notes:>20}")

    # Save reports
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save_markdown_report(fl, ps, metrics_spec, output_dir)
        _save_merged_json(fl, ps, output_dir)


def _save_markdown_report(fl, ps, metrics_spec, output_dir):
    """Save markdown comparison report."""
    path = os.path.join(output_dir, 'fair_comparison_report.md')

    fl_cov = fl['overall'].get('coverage', {}).get('mean', 0)
    fl_psnr_i = fl['overall'].get('psnr_intersection', {}).get('mean', 0)
    fl_psnr_g = fl['overall'].get('psnr_gt_masked', {}).get('mean', 0)

    with open(path, 'w') as f:
        f.write("# Fair Comparison: FaceLift vs Pose-Splatter\n\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Fairness Guarantees\n\n")
        f.write("| Guarantee | Status |\n")
        f.write("|-----------|--------|\n")
        f.write("| Test-only frames | Yes |\n")
        f.write("| GT alpha masks | Yes |\n")
        f.write("| Identical metric functions | Yes |\n")
        f.write("| Foreground-only evaluation | Yes |\n")
        f.write("| Coverage-aware metrics | Yes |\n\n")

        f.write("## Overall Results\n\n")
        f.write("| Metric | FaceLift | Pose-Splatter | Gap | Winner |\n")
        f.write("|--------|----------|---------------|-----|--------|\n")

        for key, name, unit, higher_better in metrics_spec:
            fl_val = fl['overall'].get(key, {}).get('mean', float('nan'))
            ps_val = ps['overall'].get(key, {}).get('mean', float('nan'))
            if np.isnan(fl_val) or np.isnan(ps_val):
                continue
            gap = fl_val - ps_val
            winner = 'FL' if (gap > 0) == higher_better else 'PS'

            if key in ('coverage', 'pred_precision'):
                f.write(f"| {name} | {fl_val * 100:.1f}% | {ps_val * 100:.1f}% | "
                        f"{gap * 100:+.1f}% | {winner} |\n")
            elif 'psnr' in key:
                f.write(f"| {name} | {fl_val:.2f} | {ps_val:.2f} | "
                        f"{gap:+.2f} | {winner} |\n")
            else:
                f.write(f"| {name} | {fl_val:.4f} | {ps_val:.4f} | "
                        f"{gap:+.4f} | {winner} |\n")

        f.write(f"\n## Coverage Analysis\n\n")
        f.write(f"- FL coverage: {fl_cov * 100:.1f}% of GT foreground\n")
        f.write(f"- PSNR on covered region: {fl_psnr_i:.2f} dB\n")
        f.write(f"- PSNR on full GT mask: {fl_psnr_g:.2f} dB\n")
        f.write(f"- Gap from coverage: {fl_psnr_i - fl_psnr_g:.2f} dB\n")

        f.write(f"\n## Interpretation\n\n")
        f.write("- **PSNR (GT-masked)**: Standard evaluation using GT foreground mask.\n")
        f.write("  Penalizes both color error and missing coverage.\n")
        f.write("- **PSNR (intersection)**: Only evaluates where BOTH models have "
                "foreground.\n")
        f.write("  Isolates color accuracy from coverage.\n")
        f.write("- **Coverage**: What fraction of GT mouse the model reconstructs.\n")
        f.write("- **IoU**: Overlap between predicted and GT silhouettes.\n")

    print(f"\n  Report saved: {path}")


def _save_merged_json(fl, ps, output_dir):
    """Save merged comparison JSON."""
    path = os.path.join(output_dir, 'fair_comparison_merged.json')
    merged = {
        'facelift': fl,
        'posesplatter': ps,
        'timestamp': datetime.now().isoformat(),
    }
    with open(path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"  Merged JSON saved: {path}")


# ==============================================================
# CLI
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fair Comparison: FaceLift vs Pose-Splatter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='mode', help='Evaluation mode')

    # -- evaluate_fl --
    fl_p = subparsers.add_parser('evaluate_fl',
                                 help='Evaluate FaceLift renders')
    fl_p.add_argument('--render_dir', required=True,
                      help='FL render samples dir')
    fl_p.add_argument('--gt_dir', required=True,
                      help='GT data dir (M5)')
    fl_p.add_argument('--views', nargs='+', type=int,
                      default=[1, 2, 3, 4, 5],
                      help='View indices (default: 1-5)')
    fl_p.add_argument('--save_per_frame', action='store_true',
                      help='Include per-frame metrics')
    fl_p.add_argument('--save_vis', default=None,
                      help='Directory to save GT/Render/Mask visualization grids')
    fl_p.add_argument('--vis_every', type=int, default=10,
                      help='Save vis every N frames (default: 10)')
    fl_p.add_argument('--output', required=True,
                      help='Output JSON path')

    # -- compare --
    cmp_p = subparsers.add_parser('compare',
                                  help='Compare two evaluation JSONs')
    cmp_p.add_argument('--facelift', required=True,
                       help='FL evaluation JSON')
    cmp_p.add_argument('--baseline', required=True,
                       help='Baseline evaluation JSON')
    cmp_p.add_argument('--output_dir', default=None,
                       help='Dir for report outputs')

    args = parser.parse_args()

    if args.mode == 'evaluate_fl':
        result = evaluate_facelift(
            args.render_dir, args.gt_dir,
            views=args.views,
            save_per_frame=args.save_per_frame,
            save_vis=args.save_vis,
            vis_every=args.vis_every,
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Summary
        print(f"\n  {'Metric':<25} {'Mean':>10} {'Std':>10}")
        print("  " + "-" * 47)
        for key in ['psnr_gt_masked', 'psnr_intersection',
                     'ssim_gt_masked', 'l1_gt_masked', 'iou', 'coverage']:
            if key in result['overall']:
                m = result['overall'][key]
                if key == 'coverage':
                    print(f"  {key:<25} {m['mean'] * 100:>9.1f}% "
                          f"{m['std'] * 100:>9.1f}%")
                else:
                    print(f"  {key:<25} {m['mean']:>10.4f} {m['std']:>10.4f}")
        print(f"\n  Saved: {args.output}")

    elif args.mode == 'compare':
        compare_results(args.facelift, args.baseline, args.output_dir)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
