"""A2: Oracle MVDiff - Prepare Hybrid Datasets

Creates hybrid datasets mixing GT and MVDiff-generated views.
Measures GS-LRM sensitivity to input view quality.

Levels:
  0: 6 GT views (already done: PSNR_gt=21.02)
  1: v0=GT, v1-4=GT, v5=MVDiff
  2: v0=GT, v1-3=GT, v4-5=MVDiff
  3: v0=GT, v1-2=GT, v3-5=MVDiff
  4: v0=GT, v1=GT, v2-5=MVDiff
  5: v0=GT, v1-5=MVDiff (approx E2E: PSNR_gt=7.90)

View 0 is always GT (input view in E2E pipeline).
Views replaced in order 5,4,3,2,1 (farthest-to-closest from input).

Usage:
    # Step 1: Prepare datasets
    python prepare_oracle_mvdiff.py prepare \
        --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --mvdiff_dir outputs/phase3_e2e/E1_cosine_20k/samples \
        --output_base outputs/analysis/oracle_mvdiff \
        --split data_mouse_t2_test.txt

    # Step 2: Run inference (see run_oracle_mvdiff.sh)

    # Step 3: Compute metrics
    python prepare_oracle_mvdiff.py metrics \
        --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --output_base outputs/analysis/oracle_mvdiff

    # Quick sanity check (5 frames only)
    python prepare_oracle_mvdiff.py prepare \
        --gt_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --mvdiff_dir outputs/phase3_e2e/E1_cosine_20k/samples \
        --output_base outputs/analysis/oracle_mvdiff_test \
        --split data_mouse_t2_test.txt \
        --max_frames 5
"""

import argparse
import json
import os
import shutil
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict


# Hybrid levels: which views use MVDiff output
# View 0 = input (always GT). Replace from farthest (v5) to closest (v1).
HYBRID_LEVELS = {
    1: [5],           # 5 GT + 1 MVDiff
    2: [4, 5],        # 4 GT + 2 MVDiff
    3: [3, 4, 5],     # 3 GT + 3 MVDiff
    4: [2, 3, 4, 5],  # 2 GT + 4 MVDiff
}

VIEW_TO_CAM = {0: 'cam_000', 1: 'cam_001', 2: 'cam_002',
               3: 'cam_003', 4: 'cam_004', 5: 'cam_005'}


def extract_fg_mask(img_rgb, threshold=20):
    """Extract foreground mask from RGB image via corner-based BG detection."""
    corners = [img_rgb[0, 0], img_rgb[0, -1], img_rgb[-1, 0], img_rgb[-1, -1]]
    corner_mean = np.mean(corners, axis=0)

    if corner_mean.mean() > 200:
        is_bg = np.all(img_rgb > (255 - threshold), axis=-1)
    elif corner_mean.mean() < 50:
        is_bg = np.all(img_rgb < threshold, axis=-1)
    else:
        diff = np.abs(img_rgb.astype(float) - corner_mean.astype(float))
        is_bg = np.all(diff < threshold, axis=-1)

    return ~is_bg


def convert_rgb_to_rgba(rgb_img):
    """Convert MVDiff RGB image to RGBA by extracting foreground mask."""
    mask = extract_fg_mask(rgb_img)
    h, w = rgb_img.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_img
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    return rgba


def load_frame_list(gt_dir, split_file):
    """Load frame IDs from split file."""
    split_path = gt_dir / split_file
    if not split_path.exists():
        # Try in parent
        split_path = gt_dir.parent / split_file
    if not split_path.exists():
        raise FileNotFoundError('Split file not found: %s' % split_file)

    with open(split_path) as f:
        frames = []
        for line in f:
            line = line.strip().rstrip('/')
            if not line:
                continue
            # Extract frame ID from full path or bare name
            frame_id = Path(line).name
            frames.append(frame_id)
    return frames


def cmd_prepare(args):
    """Prepare hybrid datasets."""
    from PIL import Image

    gt_dir = Path(args.gt_dir)
    mvdiff_dir = Path(args.mvdiff_dir)
    output_base = Path(args.output_base)

    frames = load_frame_list(gt_dir, args.split)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    print('A2: Oracle MVDiff - Prepare Hybrid Datasets')
    print('=' * 60)
    print('GT dir:     %s' % gt_dir)
    print('MVDiff dir: %s' % mvdiff_dir)
    print('Output:     %s' % output_base)
    print('Frames:     %d' % len(frames))
    print()

    # Verify first frame has expected structure
    test_frame = frames[0]
    test_gt = gt_dir / test_frame / 'images' / 'cam_000.png'
    test_mv = mvdiff_dir / test_frame / 'cam_000' / 'generated_views' / 'view_00.png'
    if not test_gt.exists():
        print('ERROR: GT not found: %s' % test_gt)
        sys.exit(1)
    if not test_mv.exists():
        # Try absolute path
        test_mv_abs = Path(args.mvdiff_dir).resolve() / test_frame / 'cam_000' / 'generated_views' / 'view_00.png'
        if test_mv_abs.exists():
            mvdiff_dir = Path(args.mvdiff_dir).resolve()
            test_mv = test_mv_abs
        else:
            print('ERROR: MVDiff not found: %s' % test_mv)
            print('  Also tried: %s' % test_mv_abs)
            sys.exit(1)
    print('Verified: GT and MVDiff data found for %s' % test_frame)
    print('  GT:     %s' % test_gt)
    print('  MVDiff: %s' % test_mv)

    stats = defaultdict(int)

    for level, mvdiff_views in HYBRID_LEVELS.items():
        level_dir = output_base / ('hybrid_level_%d' % level)
        gt_views = [v for v in range(6) if v not in mvdiff_views]

        print()
        print('Level %d: %d GT + %d MVDiff (replace views %s)' % (
            level, len(gt_views), len(mvdiff_views), mvdiff_views))

        # Symlink root-level files (metadata, split files)
        level_dir.mkdir(parents=True, exist_ok=True)
        for root_file in ['metadata.json', 'split.json']:
            src = gt_dir / root_file
            dst = level_dir / root_file
            if src.exists() and not dst.exists():
                os.symlink(str(src.resolve()), str(dst))
        # Symlink all split text files
        for txt_file in gt_dir.glob('data_mouse_*.txt'):
            dst = level_dir / txt_file.name
            if not dst.exists():
                os.symlink(str(txt_file.resolve()), str(dst))

        for fi, frame_id in enumerate(frames):
            frame_out = level_dir / frame_id
            images_out = frame_out / 'images'
            images_out.mkdir(parents=True, exist_ok=True)

            # Symlink camera JSON
            cam_json_src = gt_dir / frame_id / 'opencv_cameras.json'
            cam_json_dst = frame_out / 'opencv_cameras.json'
            if not cam_json_dst.exists():
                os.symlink(str(cam_json_src.resolve()), str(cam_json_dst))

            # GT views: symlink
            for vidx in gt_views:
                cam_name = VIEW_TO_CAM[vidx]
                src = gt_dir / frame_id / 'images' / ('%s.png' % cam_name)
                dst = images_out / ('%s.png' % cam_name)
                if not dst.exists():
                    os.symlink(str(src.resolve()), str(dst))
                stats['gt_symlinks'] += 1

            # MVDiff views: convert RGB→RGBA and save
            for vidx in mvdiff_views:
                cam_name = VIEW_TO_CAM[vidx]
                mv_path = mvdiff_dir / frame_id / 'cam_000' / 'generated_views' / ('view_%02d.png' % vidx)
                dst = images_out / ('%s.png' % cam_name)

                if not dst.exists():
                    if not mv_path.exists():
                        print('  WARN: missing MVDiff view %s/%d' % (frame_id, vidx))
                        stats['missing'] += 1
                        # Fallback: use GT
                        src = gt_dir / frame_id / 'images' / ('%s.png' % cam_name)
                        os.symlink(str(src.resolve()), str(dst))
                        continue

                    mv_rgb = np.array(Image.open(mv_path))
                    if mv_rgb.shape[-1] == 4:
                        # Already RGBA (shouldn't happen but handle)
                        rgba = mv_rgb
                    else:
                        rgba = convert_rgb_to_rgba(mv_rgb)

                    Image.fromarray(rgba).save(str(dst))
                    stats['mvdiff_converted'] += 1

            if (fi + 1) % 100 == 0:
                print('  [%d/%d]' % (fi + 1, len(frames)))

        # Copy split file to level dir
        split_src = gt_dir / args.split
        if not split_src.exists():
            split_src = gt_dir.parent / args.split
        if split_src.exists():
            split_dst = level_dir / args.split
            if not split_dst.exists():
                shutil.copy2(str(split_src), str(split_dst))

        print('  Done: %s' % level_dir)

    print()
    print('Stats: %d GT symlinks, %d MVDiff converted, %d missing' % (
        stats['gt_symlinks'], stats['mvdiff_converted'], stats['missing']))
    print()
    print('Next: Run GS-LRM inference for each level (see run_oracle_mvdiff.sh)')


def cmd_metrics(args):
    """Compute fair metrics for all hybrid levels."""
    from PIL import Image

    gt_dir = Path(args.gt_dir)
    output_base = Path(args.output_base)

    print('A2: Oracle MVDiff - Compute Metrics')
    print('=' * 60)

    # Collect results from all levels
    all_levels = {}

    # Add known endpoints
    all_levels[0] = {
        'label': '6 GT views (upper bound)',
        'source': 'gslrm_6view_fair.json',
        'psnr_gt': 21.02, 'psnr_int': 22.36, 'iou': 0.943, 'coverage': 0.989
    }
    all_levels[5] = {
        'label': '1 GT + 5 MVDiff (E2E)',
        'source': 'e1_cosine_20k_fair.json',
        'psnr_gt': 7.90, 'psnr_int': 16.15, 'iou': 0.528, 'coverage': 0.705
    }

    # Check for fair comparison results for hybrid levels
    for level in [1, 2, 3, 4]:
        json_path = output_base / ('hybrid_level_%d_fair.json' % level)
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            overall = data.get('overall', data.get('summary', {}))
            mvdiff_views = HYBRID_LEVELS[level]

            def _get_mean(d, key):
                v = d.get(key, {})
                if isinstance(v, dict):
                    return v.get('mean', 0)
                return v

            all_levels[level] = {
                'label': '%d GT + %d MVDiff (replace v%s)' % (
                    6 - len(mvdiff_views), len(mvdiff_views),
                    ','.join(str(v) for v in mvdiff_views)),
                'source': json_path.name,
                'psnr_gt': _get_mean(overall, 'psnr_gt_masked'),
                'psnr_int': _get_mean(overall, 'psnr_intersection'),
                'iou': _get_mean(overall, 'iou'),
                'coverage': _get_mean(overall, 'coverage'),
            }
        else:
            print('  MISSING: %s (run inference + fair_comparison first)' % json_path)

    # Print results table
    print()
    print('=' * 80)
    print('ORACLE MVDiff SENSITIVITY ANALYSIS')
    print('=' * 80)
    header = '  %-5s %-35s %8s %8s %8s %8s'
    print(header % ('Level', 'Config', 'PSNR_gt', 'PSNR_int', 'IoU', 'Cov%'))
    print('  ' + '-' * 75)

    for level in sorted(all_levels.keys()):
        r = all_levels[level]
        print('  %-5d %-35s %8.2f %8.2f %8.3f %7.1f%%' % (
            level, r['label'][:35],
            r.get('psnr_gt', 0), r.get('psnr_int', 0),
            r.get('iou', 0), r.get('coverage', 0) * 100))

    # Analyze degradation pattern
    print()
    print('DEGRADATION ANALYSIS')
    print('=' * 80)

    levels_with_data = sorted([l for l in all_levels if 'psnr_gt' in all_levels[l]])
    if len(levels_with_data) >= 3:
        prev = None
        for level in levels_with_data:
            r = all_levels[level]
            if prev is not None:
                dp = r['psnr_gt'] - prev['psnr_gt']
                di = r['iou'] - prev['iou']
                print('  Level %d→%d: dPSNR_gt=%+.2f dB, dIoU=%+.3f' % (
                    prev_level, level, dp, di))
            prev = r
            prev_level = level

        # Check linearity
        l0 = all_levels.get(0, {})
        l5 = all_levels.get(5, {})
        if l0 and l5:
            total_drop_psnr = l5['psnr_gt'] - l0['psnr_gt']
            total_drop_iou = l5['iou'] - l0['iou']
            print()
            print('  Total drop (0→5): PSNR_gt=%+.2f dB, IoU=%+.3f' % (
                total_drop_psnr, total_drop_iou))

            # Check if degradation is linear or threshold-like
            for level in [1, 2, 3, 4]:
                if level in all_levels:
                    r = all_levels[level]
                    expected_psnr = l0['psnr_gt'] + (total_drop_psnr * level / 5)
                    actual_psnr = r['psnr_gt']
                    deviation = actual_psnr - expected_psnr
                    print('  Level %d: expected=%.2f, actual=%.2f, dev=%+.2f dB (%s)' % (
                        level, expected_psnr, actual_psnr, deviation,
                        'sub-linear' if deviation > 0.5 else
                        'super-linear' if deviation < -0.5 else 'linear'))

    print()
    print('INTERPRETATION')
    print('=' * 80)

    mid_levels = [l for l in [2, 3] if l in all_levels]
    if mid_levels:
        mid = all_levels[mid_levels[0]]
        l0_psnr = all_levels[0]['psnr_gt']
        l5_psnr = all_levels[5]['psnr_gt']
        mid_ratio = (mid['psnr_gt'] - l5_psnr) / (l0_psnr - l5_psnr) * 100

        if mid_ratio > 70:
            print('  >> Sub-linear degradation: first few bad views have minimal impact')
            print('     GS-LRM is robust to partial view corruption')
            print('     Implication: improving worst MVDiff views would help most')
        elif mid_ratio < 30:
            print('  >> Super-linear degradation: quality drops sharply with any bad views')
            print('     GS-LRM requires all views to be high quality')
            print('     Implication: all MVDiff views must improve together')
        else:
            print('  >> Approximately linear degradation')
            print('     Each additional MVDiff view contributes proportionally')
            print('     Implication: any improvement in MVDiff helps proportionally')

    # Save results
    results_file = output_base / 'oracle_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(all_levels, f, indent=2, default=str)
    print()
    print('  Saved: %s' % results_file)


def main():
    parser = argparse.ArgumentParser(description='A2: Oracle MVDiff Analysis')
    subparsers = parser.add_subparsers(dest='command')

    # Prepare command
    p_prep = subparsers.add_parser('prepare', help='Prepare hybrid datasets')
    p_prep.add_argument('--gt_dir', required=True)
    p_prep.add_argument('--mvdiff_dir', required=True,
                        help='E2E samples dir with generated_views')
    p_prep.add_argument('--output_base', required=True)
    p_prep.add_argument('--split', default='data_mouse_t2_test.txt')
    p_prep.add_argument('--max_frames', type=int, default=0)

    # Metrics command
    p_met = subparsers.add_parser('metrics', help='Compute metrics for all levels')
    p_met.add_argument('--gt_dir', required=True)
    p_met.add_argument('--output_base', required=True)

    args = parser.parse_args()
    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'metrics':
        cmd_metrics(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
