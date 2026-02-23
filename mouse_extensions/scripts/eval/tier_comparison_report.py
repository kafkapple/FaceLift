#!/usr/bin/env python3
"""
tier_comparison_report.py - Tiered Comparison Report Generator
===============================================================

Generates a structured comparison report across tiers:
  Tier A: Fair quantitative (GS-LRM 6-view GT vs Pose-Splatter 6-view GT)
  Tier B: Qualitative (E2E 1-view vs PS 6-view, with disclaimer)
  Tier C: Bottleneck analysis (6-view GT -> 1-view GT -> E2E pipeline)

Usage:
  python -m mouse_extensions.scripts.eval.tier_comparison_report \
    --gslrm_6view experiments/comparison/tier/gslrm_6view_fair.json \
    --gslrm_1view experiments/comparison/tier/gslrm_1view_fair.json \
    --e2e experiments/comparison/fair/facelift_fair.json \
    --baseline baselines/pose_splatter/posesplatter_fair.json \
    --output_dir experiments/comparison/tier/
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


# Key metrics to report
MAIN_METRICS = [
    ('psnr_gt_masked',    'PSNR (GT-masked)',    'dB',  True,  '.2f'),
    ('psnr_intersection', 'PSNR (intersection)', 'dB',  True,  '.2f'),
    ('ssim_gt_masked',    'SSIM (GT-masked)',    '',     True,  '.4f'),
    ('l1_gt_masked',      'L1 (GT-masked)',      '',     False, '.4f'),
    ('iou',               'IoU',                 '',     True,  '.3f'),
    ('coverage',          'Coverage',            '%',    True,  '.1%'),
    ('pred_precision',    'Precision',           '%',    True,  '.1%'),
]


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_metric(data: dict, key: str) -> float:
    """Extract mean metric value from evaluation JSON."""
    overall = data.get('overall', {})
    entry = overall.get(key, {})
    if isinstance(entry, dict):
        return entry.get('mean', float('nan'))
    return float(entry) if entry else float('nan')


def get_metric_std(data: dict, key: str) -> float:
    """Extract std of metric from evaluation JSON."""
    overall = data.get('overall', {})
    entry = overall.get(key, {})
    if isinstance(entry, dict):
        return entry.get('std', float('nan'))
    return float('nan')


def fmt_val(val: float, fmt: str, is_pct: bool = False) -> str:
    """Format a metric value for display."""
    if np.isnan(val):
        return 'N/A'
    if is_pct:
        return f'{val * 100:.1f}%'
    return f'{val:{fmt}}'


def fmt_delta(val: float, fmt: str, is_pct: bool = False) -> str:
    """Format a delta value with sign."""
    if np.isnan(val):
        return 'N/A'
    if is_pct:
        return f'{val * 100:+.1f}%'
    return f'{val:+{fmt}}'


def build_tier_a_table(gslrm_6v: dict, ps: dict) -> list[str]:
    """Tier A: GS-LRM 6-view GT vs Pose-Splatter (same input condition)."""
    lines = []
    lines.append("## Tier A: Fair Quantitative Comparison (Same Input = 6 GT Views)")
    lines.append("")
    lines.append("> Both models receive 6 ground-truth camera views as input.")
    lines.append("> Evaluated on test frames only (3240-3599).")
    lines.append("")
    lines.append("| Metric | GS-LRM 6-view | Pose-Splatter | Delta | Winner |")
    lines.append("|--------|:-------------:|:-------------:|:-----:|:------:|")

    for key, name, unit, higher_better, fmt in MAIN_METRICS:
        v_gs = get_metric(gslrm_6v, key)
        v_ps = get_metric(ps, key)
        is_pct = unit == '%'

        delta = v_gs - v_ps
        if np.isnan(delta):
            winner = '-'
        else:
            winner = '**GS-LRM**' if (delta > 0) == higher_better else '**PS**'

        lines.append(
            f"| {name} | {fmt_val(v_gs, fmt, is_pct)} | {fmt_val(v_ps, fmt, is_pct)} "
            f"| {fmt_delta(delta, fmt, is_pct)} | {winner} |"
        )

    lines.append("")
    return lines


def build_tier_b_table(e2e: dict, ps: dict) -> list[str]:
    """Tier B: E2E (1-view) vs PS (6-view) — qualitative with disclaimer."""
    lines = []
    lines.append("## Tier B: Cross-Condition Comparison (Different Input)")
    lines.append("")
    lines.append("> **Disclaimer**: FaceLift E2E uses 1 input image (MVDiffusion generates views);")
    lines.append("> Pose-Splatter uses 6 GT views including target-view mask guidance.")
    lines.append("> This comparison is **not apples-to-apples** and is included for reference only.")
    lines.append("")
    lines.append("| Metric | E2E (1 image) | PS (6 GT views) | Delta |")
    lines.append("|--------|:------------:|:---------------:|:-----:|")

    for key, name, unit, higher_better, fmt in MAIN_METRICS:
        v_e2e = get_metric(e2e, key)
        v_ps = get_metric(ps, key)
        is_pct = unit == '%'
        delta = v_e2e - v_ps

        lines.append(
            f"| {name} | {fmt_val(v_e2e, fmt, is_pct)} | {fmt_val(v_ps, fmt, is_pct)} "
            f"| {fmt_delta(delta, fmt, is_pct)} |"
        )

    lines.append("")
    return lines


def build_tier_c_table(gslrm_6v: dict, gslrm_1v: dict, e2e: dict, ps: dict) -> list[str]:
    """Tier C: Bottleneck analysis — degradation from 6-view GT to E2E."""
    lines = []
    lines.append("## Tier C: Bottleneck Analysis (Quality Degradation Path)")
    lines.append("")
    lines.append("> Tracks how reconstruction quality degrades from ideal (6-view GT) to E2E pipeline.")
    lines.append("")
    lines.append("| Stage | Input | PSNR (GT) | PSNR (int) | IoU | Coverage | Drop from prev |")
    lines.append("|-------|:-----:|:---------:|:----------:|:---:|:--------:|:--------------:|")

    stages = [
        ('GS-LRM GT 6-view', '6 GT views', gslrm_6v),
        ('GS-LRM GT 1-view', '1 GT view', gslrm_1v),
        ('E2E (MVDiff+GS-LRM)', '1 image', e2e),
    ]

    prev_psnr = None
    for name, input_desc, data in stages:
        psnr_gt = get_metric(data, 'psnr_gt_masked')
        psnr_int = get_metric(data, 'psnr_intersection')
        iou = get_metric(data, 'iou')
        cov = get_metric(data, 'coverage')

        if prev_psnr is not None and not np.isnan(psnr_gt) and not np.isnan(prev_psnr):
            drop = f'{psnr_gt - prev_psnr:+.2f} dB'
        else:
            drop = 'baseline'

        lines.append(
            f"| {name} | {input_desc} | {fmt_val(psnr_gt, '.2f')} "
            f"| {fmt_val(psnr_int, '.2f')} | {fmt_val(iou, '.3f')} "
            f"| {fmt_val(cov, '.1%', True)} | {drop} |"
        )
        prev_psnr = psnr_gt

    # Add PS as reference row
    ps_psnr_gt = get_metric(ps, 'psnr_gt_masked')
    ps_psnr_int = get_metric(ps, 'psnr_intersection')
    ps_iou = get_metric(ps, 'iou')
    ps_cov = get_metric(ps, 'coverage')
    lines.append(
        f"| Pose-Splatter (ref) | 6 GT views | {fmt_val(ps_psnr_gt, '.2f')} "
        f"| {fmt_val(ps_psnr_int, '.2f')} | {fmt_val(ps_iou, '.3f')} "
        f"| {fmt_val(ps_cov, '.1%', True)} | (reference) |"
    )

    lines.append("")
    return lines


def build_per_view_section(gslrm_6v: dict, gslrm_1v: dict, e2e: dict, ps: dict) -> list[str]:
    """Per-view PSNR breakdown."""
    lines = []
    lines.append("## Per-View PSNR Breakdown (GT-masked)")
    lines.append("")
    lines.append("| View | GS-LRM 6v | GS-LRM 1v | E2E | PS |")
    lines.append("|:----:|:---------:|:---------:|:---:|:--:|")

    models = [gslrm_6v, gslrm_1v, e2e, ps]

    for v in range(6):
        vk = f'view_{v}'
        vals = []
        for m in models:
            pv = m.get('per_view', {}).get(vk, {}).get('psnr_gt_masked', {})
            val = pv.get('mean', float('nan')) if isinstance(pv, dict) else float('nan')
            vals.append(fmt_val(val, '.2f'))
        lines.append(f"| {v} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")

    lines.append("")
    return lines


def build_executive_summary(gslrm_6v: dict, gslrm_1v: dict, e2e: dict, ps: dict) -> list[str]:
    """One-paragraph summary of key findings."""
    lines = []
    lines.append("## Executive Summary")
    lines.append("")

    gs6_psnr = get_metric(gslrm_6v, 'psnr_gt_masked')
    gs1_psnr = get_metric(gslrm_1v, 'psnr_gt_masked')
    e2e_psnr = get_metric(e2e, 'psnr_gt_masked')
    ps_psnr = get_metric(ps, 'psnr_gt_masked')

    gs6_iou = get_metric(gslrm_6v, 'iou')
    ps_iou = get_metric(ps, 'iou')

    # Tier A verdict
    tier_a_delta = gs6_psnr - ps_psnr
    if not np.isnan(tier_a_delta):
        if tier_a_delta > 0:
            tier_a_verdict = f"GS-LRM 6-view outperforms PS by {tier_a_delta:+.2f} dB"
        else:
            tier_a_verdict = f"PS outperforms GS-LRM 6-view by {-tier_a_delta:.2f} dB"
    else:
        tier_a_verdict = "comparison unavailable"

    # Bottleneck
    drop_6to1 = gs1_psnr - gs6_psnr if not (np.isnan(gs1_psnr) or np.isnan(gs6_psnr)) else float('nan')
    drop_1toe2e = e2e_psnr - gs1_psnr if not (np.isnan(e2e_psnr) or np.isnan(gs1_psnr)) else float('nan')

    summary_parts = []
    summary_parts.append(
        f"**Tier A** (fair, same 6-view input): {tier_a_verdict} "
        f"(PSNR: {fmt_val(gs6_psnr, '.2f')} vs {fmt_val(ps_psnr, '.2f')}, "
        f"IoU: {fmt_val(gs6_iou, '.3f')} vs {fmt_val(ps_iou, '.3f')})."
    )

    if not np.isnan(drop_6to1) and not np.isnan(drop_1toe2e):
        summary_parts.append(
            f"**Tier C** bottleneck: reducing from 6 to 1 GT view costs {fmt_delta(drop_6to1, '.2f')} dB; "
            f"the multi-view diffusion generation step costs an additional {fmt_delta(drop_1toe2e, '.2f')} dB "
            f"(total E2E: {fmt_val(e2e_psnr, '.2f')} dB)."
        )

    lines.append(" ".join(summary_parts))
    lines.append("")
    return lines


def build_conclusions(gslrm_6v: dict, gslrm_1v: dict, e2e: dict, ps: dict) -> list[str]:
    """Key conclusions section."""
    lines = []
    lines.append("## Conclusions")
    lines.append("")

    gs6_psnr = get_metric(gslrm_6v, 'psnr_gt_masked')
    gs1_psnr = get_metric(gslrm_1v, 'psnr_gt_masked')
    e2e_psnr = get_metric(e2e, 'psnr_gt_masked')

    drop_6to1 = gs1_psnr - gs6_psnr if not (np.isnan(gs1_psnr) or np.isnan(gs6_psnr)) else float('nan')
    drop_1toe2e = e2e_psnr - gs1_psnr if not (np.isnan(e2e_psnr) or np.isnan(gs1_psnr)) else float('nan')
    total_drop = e2e_psnr - gs6_psnr if not (np.isnan(e2e_psnr) or np.isnan(gs6_psnr)) else float('nan')

    lines.append("1. **View reduction (6→1)** is the first bottleneck:")
    if not np.isnan(drop_6to1):
        lines.append(f"   - {fmt_delta(drop_6to1, '.2f')} dB from losing 5 input views")
    lines.append("")

    lines.append("2. **multi-view diffusion generation** is the second bottleneck:")
    if not np.isnan(drop_1toe2e):
        lines.append(f"   - {fmt_delta(drop_1toe2e, '.2f')} dB from generated (vs GT) input views")
    lines.append("")

    if not np.isnan(total_drop):
        lines.append(f"3. **Total E2E degradation**: {fmt_delta(total_drop, '.2f')} dB from ideal (6-view GT)")
    lines.append("")

    lines.append("4. **Improving Stage 1 quality** is the most impactful path to closing the gap with PS.")
    lines.append("")
    return lines


def build_caveats_section(gslrm_6v: dict, gslrm_1v: dict, e2e: dict, ps: dict) -> list[str]:
    """Methodological caveats and known asymmetries."""
    lines = []
    lines.append("## Caveats & Methodological Notes")
    lines.append("")

    # Frame count asymmetry
    cfg_6v = gslrm_6v.get('config', {})
    cfg_1v = gslrm_1v.get('config', {})
    cfg_e2e = e2e.get('config', {})
    cfg_ps = ps.get('config', {})

    n_6v = cfg_6v.get('num_frames', '?')
    n_1v = cfg_1v.get('num_frames', '?')
    n_e2e = cfg_e2e.get('num_frames', '?')
    n_ps = cfg_ps.get('num_frames', '?')

    lines.append("### 1. Frame Count Asymmetry")
    lines.append("")
    lines.append("| Model | Frames | Range | Note |")
    lines.append("|-------|:------:|-------|------|")

    fr_6v = cfg_6v.get('frame_range', ['?', '?'])
    fr_e2e = cfg_e2e.get('frame_range', ['?', '?'])
    step_ps = cfg_ps.get('frame_step', '?')

    lines.append(f"| GS-LRM 6-view | {n_6v} | {fr_6v[0]}-{fr_6v[1]} | Full test set |")
    lines.append(f"| GS-LRM 1-view | {n_1v} | {fr_6v[0]}-{fr_6v[1]} | Full test set |")
    lines.append(f"| E2E | {n_e2e} | {fr_e2e[0]}-{fr_e2e[1]} | Partial (earlier inference run) |")
    lines.append(f"| Pose-Splatter | {n_ps} | 3240-3599 | frame_step={step_ps} |")
    lines.append("")
    lines.append("> E2E evaluated only 200/360 test frames. "
                 "Per-frame metrics should be stable, but statistical power is lower.")
    lines.append("")

    # View evaluation asymmetry
    views_6v = cfg_6v.get('views_evaluated', [])
    views_ps = cfg_ps.get('views_evaluated', [])
    lines.append("### 2. Evaluated Views Asymmetry")
    lines.append("")
    lines.append(f"- GS-LRM / E2E: views {views_6v} (view 0 = input, excluded)")
    lines.append(f"- Pose-Splatter: views {views_ps} (view 0 included)")
    lines.append(f"- PS view 0 inclusion effect: +0.13 dB PSNR, +0.003 IoU (minor)")
    lines.append(f"- PS views 1-5 only: PSNR 16.67, IoU 0.824")
    lines.append("")

    # GT FG ratio difference
    fg_6v = get_metric(gslrm_6v, 'gt_fg_ratio')
    fg_ps = get_metric(ps, 'gt_fg_ratio')
    lines.append("### 3. GT Foreground Ratio Difference")
    lines.append("")
    lines.append(f"- GS-LRM / E2E: gt_fg_ratio = {fg_6v:.1%} (512x512, GT RGBA alpha)")
    lines.append(f"- Pose-Splatter: gt_fg_ratio = {fg_ps:.1%} (crop_to=512, white-BG extraction)")
    lines.append("- Ratio is 2.2x higher for PS, likely due to tighter crop and different mask method")
    lines.append("- This affects IoU comparability: higher FG ratio = easier IoU")
    lines.append("")

    # Mask source
    mask_fl = cfg_6v.get('mask_source', '?')
    mask_ps = cfg_ps.get('mask_source', '?')
    lines.append("### 4. Mask Source")
    lines.append("")
    lines.append(f"- GS-LRM / E2E: `{mask_fl}` (GT RGBA alpha > 127)")
    lines.append(f"- Pose-Splatter: `{mask_ps}` (threshold on white-BG render)")
    lines.append("")

    # 1-view IoU explanation
    lines.append("### 5. GS-LRM 1-view IoU = 0.028")
    lines.append("")
    lines.append("- **Not a bug**. With 1 input view, the model produces artifacts covering ~97% of pixels.")
    lines.append("- Coverage = 99.95% (GT FG almost fully inside predicted FG)")
    lines.append("- Precision = 2.76% (only 2.76% of predicted FG is actual FG)")
    lines.append("- IoU ≈ Precision when Coverage → 1.0 (degenerate case)")
    lines.append("")

    return lines


def build_experiment_details() -> list[str]:
    """Detailed experiment configuration section."""
    lines = []
    lines.append("### Experiment Details")
    lines.append("")

    lines.append("**Checkpoints:**")
    lines.append("")
    lines.append("| Model | Checkpoint | Training | Val PSNR |")
    lines.append("|-------|-----------|----------|:--------:|")
    lines.append("| GS-LRM 6-view | `gslrm/base_uniform_v2_6view_v2/best_psnr.pt` | M5t2 train, 6 input views | 24.49 |")
    lines.append("| GS-LRM 1-view | `gslrm/base_uniform_v2_1view_v2/best_psnr.pt` | M5t2 train, 1 input view | 11.08 |")
    lines.append("| E2E MVDiff | `mvdiffusion/mouse_M5t2/checkpoint-5000` | M5t2 train, sparse attn | N/A |")
    lines.append("| E2E GS-LRM | `gslrm/M5t2_E0_1_facelift/best_psnr.pt` | M5t2 train, 4 input views | 22.34 |")
    lines.append("| Pose-Splatter | `facelift_compare_5cam/latest` | M5t2 train, per-scene opt | N/A |")
    lines.append("")

    lines.append("**Data:**")
    lines.append("")
    lines.append("- Dataset: M5 (Mouse 5, single subject)")
    lines.append("- Split: M5t2 — Train 0-2879 (80%), Val 2880-3239 (10%), Test 3240-3599 (10%)")
    lines.append("- Resolution: 512x512 RGBA")
    lines.append("- Cameras: 6 views (opencv_cameras.json, fixed positions)")
    lines.append("")

    lines.append("**Metric Protocol:**")
    lines.append("")
    lines.append("- PSNR_gt_masked: PSNR computed on GT foreground pixels only (alpha > 127)")
    lines.append("- PSNR_intersection: PSNR on intersection of GT and predicted foreground")
    lines.append("- IoU: Jaccard index between GT silhouette and predicted silhouette (threshold=0.98)")
    lines.append("- Coverage: fraction of GT FG pixels covered by predicted FG")
    lines.append("- Precision: fraction of predicted FG pixels that are actual GT FG")
    lines.append("- Silhouette extraction: white-background threshold (default 0.98)")
    lines.append("")

    lines.append("**Hardware:**")
    lines.append("")
    lines.append("- Server: gpu03 (NVIDIA RTX A6000, Ampere)")
    lines.append("- GPU 4-7 only (GPU 0-3 = Blackwell RTX PRO 6000, PyTorch CUDA kernels unavailable)")
    lines.append("")

    return lines


def generate_report(gslrm_6v: dict, gslrm_1v: dict, e2e: dict, ps: dict,
                    output_dir: str) -> tuple[str, str]:
    """Generate full tiered comparison report.

    Returns (json_path, md_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build markdown
    md_lines = []
    md_lines.append("# Tiered Comparison Report: FaceLift vs Pose-Splatter")
    md_lines.append("")
    md_lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append(f"> Test frames: 3240-3599 (M5t2 split)")
    md_lines.append("")

    md_lines.extend(build_executive_summary(gslrm_6v, gslrm_1v, e2e, ps))
    md_lines.extend(build_tier_a_table(gslrm_6v, ps))
    md_lines.extend(build_tier_b_table(e2e, ps))
    md_lines.extend(build_tier_c_table(gslrm_6v, gslrm_1v, e2e, ps))
    md_lines.extend(build_per_view_section(gslrm_6v, gslrm_1v, e2e, ps))
    md_lines.extend(build_conclusions(gslrm_6v, gslrm_1v, e2e, ps))

    # Caveats and methodology
    md_lines.append("---")
    md_lines.append("")
    md_lines.extend(build_caveats_section(gslrm_6v, gslrm_1v, e2e, ps))
    md_lines.extend(build_experiment_details())

    # Save markdown
    md_path = output_dir / "tier_comparison_report.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Report saved: {md_path}")

    # Save unified JSON
    unified = {
        'timestamp': datetime.now().isoformat(),
        'test_split': 'M5t2 (frames 3240-3599)',
        'models': {
            'gslrm_6view': {
                'overall': gslrm_6v.get('overall', {}),
                'config': gslrm_6v.get('config', {}),
            },
            'gslrm_1view': {
                'overall': gslrm_1v.get('overall', {}),
                'config': gslrm_1v.get('config', {}),
            },
            'e2e': {
                'overall': e2e.get('overall', {}),
                'config': e2e.get('config', {}),
            },
            'posesplatter': {
                'overall': ps.get('overall', {}),
                'config': ps.get('config', {}),
            },
        },
        'tier_a': {
            'description': 'Fair comparison: GS-LRM 6-view GT vs PS (same input)',
            'gslrm_psnr': get_metric(gslrm_6v, 'psnr_gt_masked'),
            'ps_psnr': get_metric(ps, 'psnr_gt_masked'),
            'delta_psnr': get_metric(gslrm_6v, 'psnr_gt_masked') - get_metric(ps, 'psnr_gt_masked'),
        },
        'tier_c': {
            'description': 'Bottleneck analysis: 6-view GT -> 1-view GT -> E2E',
            'psnr_6view': get_metric(gslrm_6v, 'psnr_gt_masked'),
            'psnr_1view': get_metric(gslrm_1v, 'psnr_gt_masked'),
            'psnr_e2e': get_metric(e2e, 'psnr_gt_masked'),
            'drop_6to1': get_metric(gslrm_1v, 'psnr_gt_masked') - get_metric(gslrm_6v, 'psnr_gt_masked'),
            'drop_1to_e2e': get_metric(e2e, 'psnr_gt_masked') - get_metric(gslrm_1v, 'psnr_gt_masked'),
            'total_drop': get_metric(e2e, 'psnr_gt_masked') - get_metric(gslrm_6v, 'psnr_gt_masked'),
        },
    }

    json_path = output_dir / "tier_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(unified, f, indent=2)
    print(f"JSON saved: {json_path}")

    return str(json_path), str(md_path)


def main():
    parser = argparse.ArgumentParser(
        description='Tiered Comparison Report: FaceLift vs Pose-Splatter',
    )
    parser.add_argument('--gslrm_6view', required=True,
                        help='GS-LRM 6-view fair evaluation JSON')
    parser.add_argument('--gslrm_1view', required=True,
                        help='GS-LRM 1-view fair evaluation JSON')
    parser.add_argument('--e2e', required=True,
                        help='E2E (MVDiff+GS-LRM) fair evaluation JSON')
    parser.add_argument('--baseline', required=True,
                        help='Pose-Splatter fair evaluation JSON')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for report files')

    args = parser.parse_args()

    gslrm_6v = load_json(args.gslrm_6view)
    gslrm_1v = load_json(args.gslrm_1view)
    e2e = load_json(args.e2e)
    ps = load_json(args.baseline)

    print("Loaded evaluation results:")
    print(f"  GS-LRM 6-view: {args.gslrm_6view}")
    print(f"  GS-LRM 1-view: {args.gslrm_1view}")
    print(f"  E2E:           {args.e2e}")
    print(f"  PS baseline:   {args.baseline}")

    json_path, md_path = generate_report(gslrm_6v, gslrm_1v, e2e, ps, args.output_dir)

    # Print summary to console
    print("\n" + "=" * 60)
    print("  TIERED COMPARISON SUMMARY")
    print("=" * 60)

    for key, name, unit, higher_better, fmt in MAIN_METRICS[:5]:
        gs6 = get_metric(gslrm_6v, key)
        gs1 = get_metric(gslrm_1v, key)
        e2e_v = get_metric(e2e, key)
        ps_v = get_metric(ps, key)
        is_pct = unit == '%'
        print(f"  {name:<22} 6v={fmt_val(gs6, fmt, is_pct):>8} "
              f"1v={fmt_val(gs1, fmt, is_pct):>8} "
              f"E2E={fmt_val(e2e_v, fmt, is_pct):>8} "
              f"PS={fmt_val(ps_v, fmt, is_pct):>8}")

    print("=" * 60)


if __name__ == '__main__':
    main()
