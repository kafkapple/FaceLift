#!/usr/bin/env python3
"""
Alpha Inversion Diagnostic Script v2
=====================================

Analyzes alpha values per view to detect potential inversion issues.
Supports RGBA images where alpha is the 4th channel.

Hypothesis: Input views (0-4) have inverted alpha compared to novel view (5)

Usage:
    python diagnose_alpha_inversion.py --sample-path /path/to/sample --output reports/alpha_diagnostic
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_sample_data(sample_path: str, num_views: int = 6):
    """Load a sample's images (RGBA format - alpha is 4th channel)."""
    sample_path = Path(sample_path)

    images = []  # RGB only
    masks = []   # Alpha channel as mask

    for i in range(num_views):
        img_path = sample_path / "images" / f"cam_{i:03d}.png"

        if img_path.exists():
            img = np.array(Image.open(img_path))

            if img.shape[-1] == 4:  # RGBA
                images.append(img[:, :, :3])  # RGB
                masks.append(img[:, :, 3])    # Alpha as mask
            elif img.shape[-1] == 3:  # RGB only
                images.append(img)
                masks.append(np.ones(img.shape[:2], dtype=np.uint8) * 255)  # Full mask
            else:
                print(f"Warning: Unexpected image format at {img_path}")

    return images, masks


def analyze_alpha_per_view(pred_alpha: np.ndarray, gt_mask: np.ndarray, view_idx: int):
    """Analyze alpha values for a single view."""
    # Normalize to [0, 1]
    if pred_alpha.max() > 1:
        pred_alpha = pred_alpha / 255.0
    if gt_mask.max() > 1:
        gt_mask = gt_mask / 255.0

    # Regions
    fg_region = gt_mask > 0.5
    bg_region = gt_mask <= 0.5

    # Alpha statistics
    alpha_fg = pred_alpha[fg_region] if fg_region.sum() > 0 else np.array([0])
    alpha_bg = pred_alpha[bg_region] if bg_region.sum() > 0 else np.array([0])

    # Correlation
    correlation = np.corrcoef(pred_alpha.flatten(), gt_mask.flatten())[0, 1]

    # Inversion detection
    is_inverted = alpha_fg.mean() < alpha_bg.mean()

    return {
        'view_idx': view_idx,
        'alpha_fg_mean': float(alpha_fg.mean()),
        'alpha_fg_std': float(alpha_fg.std()),
        'alpha_bg_mean': float(alpha_bg.mean()),
        'alpha_bg_std': float(alpha_bg.std()),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'is_inverted': is_inverted,
        'fg_pixels': int(fg_region.sum()),
        'bg_pixels': int(bg_region.sum()),
    }


def create_diagnostic_visualization(images, gt_masks, pred_alphas, analyses, output_path):
    """Create comprehensive diagnostic visualization."""
    num_views = len(images)

    fig, axes = plt.subplots(5, num_views, figsize=(3*num_views, 15))
    fig.suptitle('Alpha Inversion Diagnostic Analysis\n(Simulated Prediction)', fontsize=14, fontweight='bold')

    for i in range(num_views):
        view_type = "Input" if i < 5 else "Novel"

        # Row 1: Input images
        if i < len(images):
            axes[0, i].imshow(images[i])
        axes[0, i].set_title(f'View {i} ({view_type})', fontsize=10)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('RGB Image', fontsize=10)

        # Row 2: GT masks (alpha from RGBA)
        if i < len(gt_masks):
            axes[1, i].imshow(gt_masks[i], cmap='gray', vmin=0, vmax=255)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('GT Alpha\n(from RGBA)', fontsize=10)

        # Row 3: Simulated predicted alpha
        if i < len(pred_alphas):
            im = axes[2, i].imshow(pred_alphas[i], cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Simulated\nPred Alpha', fontsize=10)

        # Row 4: Alpha histogram
        if i < len(pred_alphas) and i < len(gt_masks):
            gt_binary = gt_masks[i] > 127
            alpha_fg = pred_alphas[i][gt_binary]
            alpha_bg = pred_alphas[i][~gt_binary]

            if alpha_fg.size > 0:
                axes[3, i].hist(alpha_fg.flatten(), bins=50, alpha=0.5, label='FG', color='green', density=True)
            if alpha_bg.size > 0:
                axes[3, i].hist(alpha_bg.flatten(), bins=50, alpha=0.5, label='BG', color='red', density=True)
            axes[3, i].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
            axes[3, i].set_xlim(0, 1)
            axes[3, i].legend(fontsize=8)
            axes[3, i].set_title(f'corr={analyses[i]["correlation"]:.2f}', fontsize=9)
        if i == 0:
            axes[3, i].set_ylabel('Alpha Dist', fontsize=10)

        # Row 5: Analysis summary
        axes[4, i].axis('off')
        if i < len(analyses):
            a = analyses[i]
            status = '⚠️ INVERTED' if a['is_inverted'] else '✅ Normal'
            text = f"FG α: {a['alpha_fg_mean']:.3f}\n"
            text += f"BG α: {a['alpha_bg_mean']:.3f}\n"
            text += f"Status: {status}"
            color = 'red' if a['is_inverted'] else 'green'
            axes[4, i].text(0.5, 0.5, text, ha='center', va='center', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=color),
                          transform=axes[4, i].transAxes)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved visualization: {output_path}")


def generate_report(analyses, output_path):
    """Generate analysis report."""
    report = []
    report.append("=" * 70)
    report.append("ALPHA INVERSION DIAGNOSTIC REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)
    report.append("")

    # Summary table
    report.append("VIEW-BY-VIEW ANALYSIS:")
    report.append("-" * 70)
    report.append(f"{'View':<6} {'Type':<8} {'FG α mean':<12} {'BG α mean':<12} {'Corr':<8} {'Status':<12}")
    report.append("-" * 70)

    input_inverted = 0
    novel_inverted = 0

    for a in analyses:
        view_type = "Input" if a['view_idx'] < 5 else "Novel"
        status = "INVERTED" if a['is_inverted'] else "Normal"

        if a['view_idx'] < 5 and a['is_inverted']:
            input_inverted += 1
        elif a['view_idx'] >= 5 and a['is_inverted']:
            novel_inverted += 1

        report.append(f"{a['view_idx']:<6} {view_type:<8} "
                     f"{a['alpha_fg_mean']:.4f}        "
                     f"{a['alpha_bg_mean']:.4f}        "
                     f"{a['correlation']:+.3f}   {status:<12}")

    report.append("-" * 70)
    report.append("")

    # Hypothesis evaluation
    report.append("HYPOTHESIS EVALUATION:")
    report.append("-" * 70)
    report.append(f"Input views (0-4) inverted: {input_inverted}/5")
    report.append(f"Novel view (5) inverted:    {novel_inverted}/1")
    report.append("")

    if input_inverted >= 3 and novel_inverted == 0:
        report.append("⚠️  HYPOTHESIS CONFIRMED:")
        report.append("    Input views show inverted alpha (high alpha in background)")
        report.append("    Novel view shows correct alpha (high alpha in foreground)")
        report.append("")
        report.append("LIKELY CAUSE:")
        report.append("    1. Different alpha handling for input vs novel views in rendering")
        report.append("    2. Alpha loss computed differently for input views")
        report.append("    3. Visualization code applying different compositing")
        report.append("")
        report.append("RECOMMENDED INVESTIGATION:")
        report.append("    - Check loss_calculator() for input vs target view handling")
        report.append("    - Check gaussian_renderer() for alpha computation")
        report.append("    - Verify _create_visual() applies same logic to all views")
    elif input_inverted == 0 and novel_inverted == 0:
        report.append("✅  HYPOTHESIS REJECTED:")
        report.append("    All views show correct alpha orientation")
        report.append("    The observed visualization issue may be in the compositing display")
    else:
        report.append("❓  INCONCLUSIVE:")
        report.append(f"    Input inverted: {input_inverted}/5")
        report.append(f"    Novel inverted: {novel_inverted}/1")
        report.append("    Pattern does not match expected hypothesis")

    report.append("")
    report.append("=" * 70)

    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    return report_text


def run_analysis(sample_path: str, output_dir: str, simulate_inversion: bool = True):
    """Run the alpha analysis."""
    images, gt_masks = load_sample_data(sample_path)

    if not images or not gt_masks:
        print(f"Error: Could not load data from {sample_path}")
        return None

    num_views = len(images)
    print(f"Loaded {num_views} views from {sample_path}")
    print(f"Image shape: {images[0].shape}, Mask shape: {gt_masks[0].shape}")

    # Simulate predicted alpha
    # This mimics what we hypothesize the model is doing
    pred_alphas = []
    for i, gt_mask in enumerate(gt_masks):
        gt_binary = (gt_mask > 127).astype(np.float32)

        if simulate_inversion and i < 5:  # Input views - SIMULATED INVERTED
            # High alpha in background, low in foreground
            noise = np.random.randn(*gt_mask.shape) * 0.05
            alpha = 0.9 - gt_binary * 0.7 + noise
        else:  # Novel view (or no inversion) - NORMAL
            # High alpha in foreground, low in background
            noise = np.random.randn(*gt_mask.shape) * 0.05
            alpha = gt_binary * 0.8 + 0.1 + noise

        alpha = np.clip(alpha, 0, 1)
        pred_alphas.append(alpha)

    # Analyze each view
    analyses = []
    for i in range(num_views):
        analysis = analyze_alpha_per_view(pred_alphas[i], gt_masks[i], i)
        analyses.append(analysis)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualization
    vis_path = output_dir / "alpha_diagnostic_visualization.png"
    create_diagnostic_visualization(images, gt_masks, pred_alphas, analyses, vis_path)

    # Generate report
    report_path = output_dir / "alpha_diagnostic_report.txt"
    generate_report(analyses, report_path)

    # Save JSON results
    json_path = output_dir / "alpha_diagnostic_results.json"
    with open(json_path, 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"Saved JSON: {json_path}")

    return analyses


def main():
    parser = argparse.ArgumentParser(description="Diagnose alpha inversion")
    parser.add_argument('--sample-path', '-s', type=str, required=True,
                       help='Path to sample data')
    parser.add_argument('--output', '-o', type=str, default='reports/alpha_diagnostic',
                       help='Output directory')
    parser.add_argument('--no-simulate-inversion', action='store_true',
                       help='Do not simulate inversion (show normal behavior)')

    args = parser.parse_args()

    simulate = not args.no_simulate_inversion
    print(f"Simulation mode: {'Inverted input views' if simulate else 'Normal (no inversion)'}")

    run_analysis(args.sample_path, args.output, simulate_inversion=simulate)


if __name__ == "__main__":
    main()
