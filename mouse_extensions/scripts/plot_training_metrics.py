#!/usr/bin/env python3
"""
Reusable training metrics plotter for FaceLift GS-LRM experiments.

Parses training log to extract Val PSNR/SSIM/LPIPS + param_update_step trend.
Generates 4-panel diagnostic plot.

Usage:
    python plot_training_metrics.py <log_path> [--output <png_path>] [--title <title>]

Examples:
    python plot_training_metrics.py /node_data/joon/logs_FaceLift/rat_v13_fullft.log
    python plot_training_metrics.py logs/train.log --output results/metrics.png --title "v13 Rat 7M"
"""
import argparse
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Parse FaceLift training log for val metrics + param_update."""
    val_records = []
    train_records = []

    with open(log_path) as f:
        for line in f:
            # Val summary: [Val] step=4401, samples=179, psnr=13.6633, ssim=0.5764, lpips=0.6171,
            m = re.search(r'\[Val\] step=(\d+),.*psnr=([\d.]+),.*ssim=([\d.]+),.*lpips=([\d.]+)', line)
            if m:
                val_records.append({
                    'step': int(m.group(1)),
                    'psnr': float(m.group(2)),
                    'ssim': float(m.group(3)),
                    'lpips': float(m.group(4)),
                })
            # Train step: [Train] epoch: 2, step: 4400/1435, time: 7.50, param_update_step: 382, lr:
            m = re.search(r'\[Train\].*step:\s*(\d+)/.*param_update_step:\s*(\d+)', line)
            if m:
                train_records.append({
                    'step': int(m.group(1)),
                    'param_update': int(m.group(2)),
                })

    return val_records, train_records


def plot_metrics(val_records, train_records, title="Training Progress", output_path=None,
                 pass_threshold=18.0, marginal_threshold=15.0, best_psnr=None):
    """Generate 4-panel diagnostic plot."""
    if not val_records:
        print("No val records found"); return

    steps = [r['step'] for r in val_records]
    psnr = [r['psnr'] for r in val_records]
    ssim = [r['ssim'] for r in val_records]
    lpips = [r['lpips'] for r in val_records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title}\n'
                 f'Val PSNR range: {min(psnr):.2f} - {max(psnr):.2f} dB, '
                 f'peak @ step {steps[psnr.index(max(psnr))]}',
                 fontsize=13, fontweight='bold')

    # Panel 1: Val PSNR
    ax = axes[0, 0]
    ax.plot(steps, psnr, 'b-o', markersize=4, linewidth=2, label='Val PSNR')
    if best_psnr:
        ax.axhline(y=best_psnr, color='green', linestyle='--', alpha=0.7, label=f'best {best_psnr:.2f}')
    ax.axhline(y=pass_threshold, color='red', linestyle='--', alpha=0.5, label=f'PASS {pass_threshold}')
    ax.axhline(y=marginal_threshold, color='orange', linestyle='--', alpha=0.5, label=f'MARGINAL {marginal_threshold}')
    ax.set_xlabel('Step'); ax.set_ylabel('PSNR (dB)')
    ax.set_title('Val PSNR vs Step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: SSIM + LPIPS
    ax = axes[0, 1]
    ax.plot(steps, ssim, 'g-s', markersize=3, label='SSIM (higher=better)')
    ax2 = ax.twinx()
    ax2.plot(steps, lpips, 'r-^', markersize=3, label='LPIPS (lower=better)')
    ax.set_xlabel('Step'); ax.set_ylabel('SSIM', color='green')
    ax2.set_ylabel('LPIPS', color='red')
    ax.set_title('SSIM / LPIPS'); ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: Param update rate
    ax = axes[1, 0]
    t_pu, t_steps, rate = [], [], 0.0  # defaults (scope safety for Panel 4)
    if train_records:
        subsample = train_records[::max(1, len(train_records)//100)]
        t_steps = [r['step'] for r in subsample]
        t_pu = [r['param_update'] for r in subsample]
        ax.plot(t_steps, t_pu, 'k-', linewidth=1.5, label='param_update_step')
        start = t_steps[0]
        ax.plot([start, t_steps[-1]], [0, t_steps[-1]-start], 'g--', alpha=0.5, label='ideal 1:1')
        rate = t_pu[-1] / max(1, t_steps[-1] - start) * 100
        ax.set_title(f'Gradient Updates ({rate:.1f}% update rate)')
    else:
        ax.set_title('Gradient Updates (no data)')
    ax.set_xlabel('Step'); ax.set_ylabel('param_update_step')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 4: Summary stats
    ax = axes[1, 1]
    ax.axis('off')
    peak_step = steps[psnr.index(max(psnr))]
    last_step = steps[-1]
    if len(psnr) < 2:
        trend = "INSUFFICIENT_DATA"
    else:
        trend = "RISING" if psnr[-1] > psnr[0] else "DECLINING" if psnr[-1] < psnr[-2] else "PLATEAU"

    summary = f"""
  Steps: {steps[0]} - {last_step}
  Val PSNR: {min(psnr):.2f} - {max(psnr):.2f} dB
  Peak: {max(psnr):.2f} @ step {peak_step}
  Latest: {psnr[-1]:.2f} @ step {last_step}
  Trend: {trend}
  SSIM range: {min(ssim):.3f} - {max(ssim):.3f}
  LPIPS range: {min(lpips):.3f} - {max(lpips):.3f}
"""
    if t_pu:  # Panel 3 populated variables
        summary += f"  Param updates: {t_pu[-1]} / {t_steps[-1]-t_steps[0]} ({rate:.1f}%)\n"

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    if output_path is None:
        output_path = log_path.replace('.log', '_metrics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Plot FaceLift training metrics from log')
    ap.add_argument('log_path', help='Path to training log file')
    ap.add_argument('--output', '-o', help='Output PNG path')
    ap.add_argument('--title', '-t', default='Training Progress')
    ap.add_argument('--pass_threshold', type=float, default=18.0)
    ap.add_argument('--marginal_threshold', type=float, default=15.0)
    ap.add_argument('--best_psnr', type=float, default=None)
    args = ap.parse_args()

    val_records, train_records = parse_log(args.log_path)
    print(f'Parsed: {len(val_records)} val records, {len(train_records)} train records')

    if not val_records:
        print("ERROR: No [Val] step=... lines found in log"); sys.exit(1)

    plot_metrics(val_records, train_records, title=args.title, output_path=args.output,
                 pass_threshold=args.pass_threshold, marginal_threshold=args.marginal_threshold,
                 best_psnr=args.best_psnr)


if __name__ == '__main__':
    main()
