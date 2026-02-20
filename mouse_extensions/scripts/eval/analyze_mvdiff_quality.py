"""A1: MVDiff Output Quality Analysis

Compare MVDiff-generated views vs GT views to diagnose E2E bottleneck.

Answers:
- H1a: View consistency (geometric error)?
- H1b: Silhouette accuracy (shape error)?
- H1c: Texture quality (color error)?

Usage:
    python analyze_mvdiff_quality.py \
        --e2e_dir outputs/phase3_e2e/E1_cosine_20k \
        --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --output_dir experiments/analysis/mvdiff_quality
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

def extract_fg_mask_threshold(img_rgb, bg_color='auto', threshold=20):
    """Extract foreground mask from RGB image by detecting background."""
    h, w = img_rgb.shape[:2]
    # Check corners to detect background color
    corners = [img_rgb[0,0], img_rgb[0,-1], img_rgb[-1,0], img_rgb[-1,-1]]
    corner_mean = np.mean(corners, axis=0)

    if corner_mean.mean() > 200:  # white-ish background
        is_bg = np.all(img_rgb > (255 - threshold), axis=-1)
    elif corner_mean.mean() < 50:  # black-ish background
        is_bg = np.all(img_rgb < threshold, axis=-1)
    else:
        # Use generic: pixels very close to corner color
        diff = np.abs(img_rgb.astype(float) - corner_mean.astype(float))
        is_bg = np.all(diff < threshold, axis=-1)

    return ~is_bg


def compute_metrics(gen_rgb, gt_rgba, view_idx):
    """Compute comprehensive metrics between generated and GT view."""
    gt_rgb = gt_rgba[:, :, :3].astype(np.float64)
    gt_alpha = gt_rgba[:, :, 3]
    gen = gen_rgb.astype(np.float64)

    # GT mask
    gt_mask = gt_alpha > 0
    gt_fg_ratio = gt_mask.mean()

    # Generated mask (from RGB)
    gen_mask = extract_fg_mask_threshold(gen_rgb)
    gen_fg_ratio = gen_mask.mean()

    # Silhouette metrics
    intersection = gt_mask & gen_mask
    union = gt_mask | gen_mask
    iou = intersection.sum() / max(union.sum(), 1)
    coverage = intersection.sum() / max(gt_mask.sum(), 1)
    precision = intersection.sum() / max(gen_mask.sum(), 1)

    # Pixels only in one mask
    gt_only = (gt_mask & ~gen_mask).sum()
    gen_only = (~gt_mask & gen_mask).sum()

    result = {
        'view': view_idx,
        'gt_fg_ratio': float(gt_fg_ratio),
        'gen_fg_ratio': float(gen_fg_ratio),
        'silhouette_iou': float(iou),
        'silhouette_coverage': float(coverage),
        'silhouette_precision': float(precision),
        'gt_only_px': int(gt_only),
        'gen_only_px': int(gen_only),
    }

    # Color metrics on GT foreground region
    if gt_mask.sum() > 0:
        gt_fg = gt_rgb[gt_mask]
        gen_fg = gen[gt_mask]
        mse_gt = np.mean((gt_fg - gen_fg) ** 2)
        psnr_gt = 10 * np.log10(255**2 / max(mse_gt, 1e-10))
        l1_gt = np.mean(np.abs(gt_fg - gen_fg)) / 255.0
        result['psnr_gt_fg'] = float(psnr_gt)
        result['l1_gt_fg'] = float(l1_gt)

    # Color metrics on intersection region
    if intersection.sum() > 0:
        gt_int = gt_rgb[intersection]
        gen_int = gen[intersection]
        mse_int = np.mean((gt_int - gen_int) ** 2)
        psnr_int = 10 * np.log10(255**2 / max(mse_int, 1e-10))
        l1_int = np.mean(np.abs(gt_int - gen_int)) / 255.0
        result['psnr_intersection'] = float(psnr_int)
        result['l1_intersection'] = float(l1_int)

    # Full image PSNR (white BG for both)
    gen_white = np.ones_like(gen) * 255
    gt_white = np.ones_like(gt_rgb) * 255
    gen_white[gen_mask] = gen[gen_mask]
    gt_white[gt_mask] = gt_rgb[gt_mask]
    mse_full = np.mean((gen_white - gt_white) ** 2)
    psnr_full = 10 * np.log10(255**2 / max(mse_full, 1e-10))
    result['psnr_white_bg'] = float(psnr_full)

    return result


def main():
    parser = argparse.ArgumentParser(description='A1: MVDiff Output Quality Analysis')
    parser.add_argument('--e2e_dir', type=str, required=True,
                        help='E2E output directory (e.g. outputs/phase3_e2e/E1_cosine_20k)')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='GT dataset root (e.g. M5)')
    parser.add_argument('--output_dir', type=str, default='experiments/analysis/mvdiff_quality')
    parser.add_argument('--max_frames', type=int, default=0, help='Limit frames (0=all)')
    args = parser.parse_args()

    e2e_dir = Path(args.e2e_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_dir = e2e_dir / 'samples'
    frame_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
    if args.max_frames > 0:
        frame_dirs = frame_dirs[:args.max_frames]

    print('A1: MVDiff Output Quality Analysis')
    print('=' * 60)
    print('E2E dir: %s' % e2e_dir)
    print('GT dir:  %s' % gt_dir)
    print('Frames:  %d' % len(frame_dirs))
    print()

    # View mapping: generated view_XX -> GT cam_XXX
    view_to_cam = {0: 'cam_000', 1: 'cam_001', 2: 'cam_002',
                   3: 'cam_003', 4: 'cam_004', 5: 'cam_005'}

    all_results = []
    per_view = defaultdict(list)

    for fi, frame_dir in enumerate(frame_dirs):
        frame_id = frame_dir.name
        gen_views_dir = frame_dir / 'cam_000' / 'generated_views'

        if not gen_views_dir.exists():
            print('  SKIP %s: no generated_views' % frame_id)
            continue

        gt_frame_dir = gt_dir / frame_id / 'images'
        if not gt_frame_dir.exists():
            print('  SKIP %s: no GT' % frame_id)
            continue

        for vidx in range(6):
            gen_path = gen_views_dir / ('view_%02d.png' % vidx)
            gt_path = gt_frame_dir / ('%s.png' % view_to_cam[vidx])

            if not gen_path.exists() or not gt_path.exists():
                continue

            gen_img = np.array(Image.open(gen_path))
            gt_img = np.array(Image.open(gt_path))

            # Ensure same size
            if gen_img.shape[:2] != gt_img.shape[:2]:
                gen_pil = Image.open(gen_path).resize(
                    (gt_img.shape[1], gt_img.shape[0]), Image.LANCZOS)
                gen_img = np.array(gen_pil)

            if gt_img.shape[-1] != 4:
                continue

            metrics = compute_metrics(gen_img[:, :, :3], gt_img, vidx)
            metrics['frame'] = frame_id
            all_results.append(metrics)
            per_view[vidx].append(metrics)

        if (fi + 1) % 50 == 0:
            print('  [%d/%d] processed' % (fi + 1, len(frame_dirs)))

    print('  [%d/%d] processed' % (len(frame_dirs), len(frame_dirs)))
    print()

    # Aggregate
    metric_keys = ['silhouette_iou', 'silhouette_coverage', 'silhouette_precision',
                   'psnr_gt_fg', 'psnr_intersection', 'psnr_white_bg',
                   'l1_gt_fg', 'l1_intersection', 'gt_fg_ratio', 'gen_fg_ratio']

    print('=' * 60)
    print('OVERALL (all views, all frames)')
    print('=' * 60)
    overall = {}
    for key in metric_keys:
        vals = [r[key] for r in all_results if key in r]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            overall[key] = {'mean': float(m), 'std': float(s), 'n': len(vals)}
            if 'ratio' in key:
                print('  %-25s  %.2f%% +/- %.2f%%' % (key, m * 100, s * 100))
            elif 'psnr' in key:
                print('  %-25s  %.2f dB +/- %.2f' % (key, m, s))
            elif 'iou' in key or 'coverage' in key or 'precision' in key:
                print('  %-25s  %.3f +/- %.3f' % (key, m, s))
            else:
                print('  %-25s  %.4f +/- %.4f' % (key, m, s))

    print()
    print('=' * 60)
    print('PER-VIEW BREAKDOWN')
    print('=' * 60)
    header = '  %-8s %8s %8s %8s %8s %8s %8s'
    print(header % ('View', 'Sil_IoU', 'Sil_Cov', 'PSNR_gt', 'PSNR_int', 'L1_gt', 'fg_ratio'))
    print('  ' + '-' * 60)

    per_view_summary = {}
    for vidx in range(6):
        if vidx not in per_view:
            continue
        vr = per_view[vidx]
        row = {}
        for key in metric_keys:
            vals = [r[key] for r in vr if key in r]
            if vals:
                row[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        per_view_summary[vidx] = row

        is_input = ' (input)' if vidx == 0 else ''
        print('  view_%d%s %8.3f %8.3f %8.2f %8.2f %8.4f %7.2f%%' % (
            vidx, is_input,
            row.get('silhouette_iou', {}).get('mean', 0),
            row.get('silhouette_coverage', {}).get('mean', 0),
            row.get('psnr_gt_fg', {}).get('mean', 0),
            row.get('psnr_intersection', {}).get('mean', 0),
            row.get('l1_gt_fg', {}).get('mean', 0),
            row.get('gen_fg_ratio', {}).get('mean', 0) * 100,
        ))

    # Diagnosis
    print()
    print('=' * 60)
    print('DIAGNOSIS')
    print('=' * 60)

    avg_iou = overall.get('silhouette_iou', {}).get('mean', 0)
    avg_psnr_int = overall.get('psnr_intersection', {}).get('mean', 0)
    avg_cov = overall.get('silhouette_coverage', {}).get('mean', 0)
    avg_prec = overall.get('silhouette_precision', {}).get('mean', 0)
    avg_gt_fg = overall.get('gt_fg_ratio', {}).get('mean', 0)
    avg_gen_fg = overall.get('gen_fg_ratio', {}).get('mean', 0)

    print()
    print('  Shape Analysis (Silhouette):')
    print('    IoU: %.3f (perfect=1.0)' % avg_iou)
    print('    Coverage: %.1f%% (GT covered by gen)' % (avg_cov * 100))
    print('    Precision: %.1f%% (gen that is real fg)' % (avg_prec * 100))
    print('    GT fg: %.2f%%, Gen fg: %.2f%% (ratio: %.1fx)' % (
        avg_gt_fg * 100, avg_gen_fg * 100,
        avg_gen_fg / max(avg_gt_fg, 1e-6)))

    if avg_iou < 0.5:
        print('    >> SEVERE shape error: MVDiff silhouettes poorly match GT')
    elif avg_iou < 0.8:
        print('    >> MODERATE shape error: significant silhouette mismatch')
    else:
        print('    >> Shape is reasonable, not the primary bottleneck')

    print()
    print('  Color Analysis (on intersection region):')
    print('    PSNR_intersection: %.2f dB' % avg_psnr_int)

    if avg_psnr_int < 15:
        print('    >> SEVERE color error: poor texture quality')
    elif avg_psnr_int < 20:
        print('    >> MODERATE color error: noticeable texture degradation')
    else:
        print('    >> Color quality is decent')

    print()
    print('  View Consistency:')
    if per_view_summary:
        ious = [per_view_summary[v]['silhouette_iou']['mean']
                for v in per_view_summary if v != 0]
        psnrs = [per_view_summary[v].get('psnr_intersection', {}).get('mean', 0)
                 for v in per_view_summary if v != 0]
        if ious:
            print('    IoU range: %.3f - %.3f (spread: %.3f)' % (
                min(ious), max(ious), max(ious) - min(ious)))
        if psnrs:
            print('    PSNR_int range: %.2f - %.2f (spread: %.2f dB)' % (
                min(psnrs), max(psnrs), max(psnrs) - min(psnrs)))

    # Compare with E2E results
    print()
    print('  Pipeline Impact:')
    print('    MVDiff silhouette IoU: %.3f' % avg_iou)
    print('    E2E final IoU:        0.528 (from fair_comparison)')
    print('    MVDiff PSNR_int:      %.2f dB' % avg_psnr_int)
    print('    E2E final PSNR_int:   16.15 dB (from fair_comparison)')
    if avg_iou > 0:
        print('    IoU preservation:     %.1f%% (E2E/MVDiff)' % (0.528 / avg_iou * 100))

    # Save results
    output = {
        'config': {
            'e2e_dir': str(e2e_dir),
            'gt_dir': str(gt_dir),
            'num_frames': len(frame_dirs),
            'num_samples': len(all_results),
        },
        'overall': overall,
        'per_view': {str(k): v for k, v in per_view_summary.items()},
    }
    out_file = out_dir / 'mvdiff_quality_analysis.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print()
    print('  Saved: %s' % out_file)


if __name__ == '__main__':
    main()
