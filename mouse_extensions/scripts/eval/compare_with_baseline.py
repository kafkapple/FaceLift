#!/usr/bin/env python3
"""
Unified comparison of FaceLift vs Pose-Splatter (baseline).

Loads pre-computed metrics from both models and generates a unified
comparison report with consistent metric definitions.

Supports two modes:
  1. Load pre-computed: Read existing metrics JSON from both models
  2. Recompute FaceLift: Re-evaluate FaceLift renders against GT

Metrics (unified):
  - PSNR_full_white: White-BG composite + full-image MSE (literature standard)
  - SSIM_full_white: White-BG composite + skimage SSIM
  - LPIPS_full_white: AlexNet backbone (FaceLift only, PS doesn't compute)
  - Mask_IoU: Silhouette IoU (alpha > 0.5)
  - L1_masked: sum|pred-gt| / (3 * sum(mask))

Usage:
    # Mode 1: Load pre-computed metrics from both models
    python -m mouse_extensions.scripts.eval.compare_with_baseline \
        --facelift_metrics outputs/h5_e2e/cfgr_ckpt10000/metrics_v2.json \
        --baseline_metrics /path/to/pose-splatter/paper_standard_evaluation.json \
        --output_dir experiments/comparison/FL_vs_PS/

    # Mode 2: Recompute FaceLift metrics + load PS metrics
    python -m mouse_extensions.scripts.eval.compare_with_baseline \
        --facelift_dir outputs/h5_e2e/cfgr_ckpt10000 \
        --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --baseline_metrics /path/to/pose-splatter/paper_standard_evaluation.json \
        --output_dir experiments/comparison/FL_vs_PS/

    # Mode 3: Compare with manually specified baseline values
    python -m mouse_extensions.scripts.eval.compare_with_baseline \
        --facelift_metrics outputs/h5_e2e/cfgr_ckpt10000/metrics_v2.json \
        --baseline_values '{"psnr": 24.68, "ssim": 0.963, "iou": 0.829, "l1": 0.097}' \
        --output_dir experiments/comparison/FL_vs_PS/

Created: 2026-02-15
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# Metric Normalization
# ============================================================

def normalize_facelift_metrics(metrics_v2: Dict) -> Dict[str, Any]:
    """Normalize FaceLift metrics_v2.json to unified format.

    FaceLift metrics_v2.json structure:
        overall: {psnr_full_white, psnr_fg_only, psnr_alpha_weighted,
                  ssim_full_white, masked_l1, silhouette_iou, fg_fraction,
                  lpips_full_white}
        per_view: {view_0: {...}, ...}
    """
    overall = metrics_v2.get("overall", {})

    def get_mean(d: Dict, key: str) -> Optional[float]:
        v = d.get(key, {})
        if isinstance(v, dict):
            return v.get("mean")
        return v if isinstance(v, (int, float)) else None

    def get_std(d: Dict, key: str) -> Optional[float]:
        v = d.get(key, {})
        if isinstance(v, dict):
            return v.get("std")
        return None

    result = {
        "model": "FaceLift",
        "source": "metrics_v2.json",
        "n_samples": metrics_v2.get("n_evaluated", metrics_v2.get("n_samples", 0)),
        "views_evaluated": metrics_v2.get("views_evaluated", []),
        "overall": {
            "psnr": {
                "mean": get_mean(overall, "psnr_full_white"),
                "std": get_std(overall, "psnr_full_white"),
            },
            "ssim": {
                "mean": get_mean(overall, "ssim_full_white"),
                "std": get_std(overall, "ssim_full_white"),
            },
            "lpips": {
                "mean": get_mean(overall, "lpips_full_white"),
                "std": get_std(overall, "lpips_full_white"),
            },
            "mask_iou": {
                "mean": get_mean(overall, "silhouette_iou"),
                "std": get_std(overall, "silhouette_iou"),
            },
            "l1_masked": {
                "mean": get_mean(overall, "masked_l1"),
                "std": get_std(overall, "masked_l1"),
            },
        },
        "extra": {
            "psnr_fg_only": {
                "mean": get_mean(overall, "psnr_fg_only"),
                "std": get_std(overall, "psnr_fg_only"),
            },
            "psnr_alpha_weighted": {
                "mean": get_mean(overall, "psnr_alpha_weighted"),
                "std": get_std(overall, "psnr_alpha_weighted"),
            },
            "fg_fraction": {
                "mean": get_mean(overall, "fg_fraction"),
                "std": get_std(overall, "fg_fraction"),
            },
        },
    }

    # Per-view
    per_view = {}
    for vk, vv in metrics_v2.get("per_view", {}).items():
        per_view[vk] = {
            "psnr": {"mean": get_mean(vv, "psnr_full_white"),
                     "std": get_std(vv, "psnr_full_white")},
            "ssim": {"mean": get_mean(vv, "ssim_full_white"),
                     "std": get_std(vv, "ssim_full_white")},
            "mask_iou": {"mean": get_mean(vv, "silhouette_iou"),
                         "std": get_std(vv, "silhouette_iou")},
            "l1_masked": {"mean": get_mean(vv, "masked_l1"),
                          "std": get_std(vv, "masked_l1")},
        }
    result["per_view"] = per_view

    return result


def normalize_posesplatter_metrics(ps_metrics: Dict) -> Dict[str, Any]:
    """Normalize Pose-Splatter paper_standard_evaluation.json to unified format.

    PS paper_standard_evaluation.json structure:
        overall: {psnr, ssim, iou, l1} (each with mean/std/min/max)
        holdout_only: {psnr, ssim, iou, l1}
        train_only: {psnr, ssim, iou, l1}
        per_view: {view_0: {is_holdout, psnr, ssim, iou, l1}, ...}

    IMPORTANT metric difference:
        PS uses MASKED metrics (foreground-only) by default.
        FaceLift uses WHITE-BG composite + full-image.
        These are NOT directly comparable without re-evaluation.
        We flag this in the output.
    """
    overall = ps_metrics.get("overall", {})
    holdout = ps_metrics.get("holdout_only", {})

    def get_stat(d: Dict, key: str) -> Dict:
        v = d.get(key, {})
        if isinstance(v, dict):
            return {"mean": v.get("mean"), "std": v.get("std")}
        return {"mean": v if isinstance(v, (int, float)) else None, "std": None}

    result = {
        "model": "Pose-Splatter",
        "source": "paper_standard_evaluation.json",
        "n_frames": ps_metrics.get("num_frames", 0),
        "holdout_views": ps_metrics.get("holdout_views", []),
        "metric_type": "masked",  # PS uses foreground-only metrics
        "overall": {
            "psnr": get_stat(overall, "psnr"),
            "ssim": get_stat(overall, "ssim"),
            "lpips": {"mean": None, "std": None},  # PS doesn't compute LPIPS
            "mask_iou": get_stat(overall, "iou"),
            "l1_masked": get_stat(overall, "l1"),
        },
        "holdout_only": {
            "psnr": get_stat(holdout, "psnr"),
            "ssim": get_stat(holdout, "ssim"),
            "mask_iou": get_stat(holdout, "iou"),
            "l1_masked": get_stat(holdout, "l1"),
        },
    }

    # Per-view
    per_view = {}
    for vk, vv in ps_metrics.get("per_view", {}).items():
        per_view[vk] = {
            "is_holdout": vv.get("is_holdout", False),
            "psnr": get_stat(vv, "psnr"),
            "ssim": get_stat(vv, "ssim"),
            "mask_iou": get_stat(vv, "iou"),
            "l1_masked": get_stat(vv, "l1"),
        }
    result["per_view"] = per_view

    return result


def normalize_manual_values(values: Dict) -> Dict[str, Any]:
    """Normalize manually specified baseline values."""
    result = {
        "model": "Pose-Splatter",
        "source": "manual",
        "metric_type": "unknown",
        "overall": {
            "psnr": {"mean": values.get("psnr"), "std": None},
            "ssim": {"mean": values.get("ssim"), "std": None},
            "lpips": {"mean": values.get("lpips"), "std": None},
            "mask_iou": {"mean": values.get("iou", values.get("mask_iou")),
                         "std": None},
            "l1_masked": {"mean": values.get("l1", values.get("l1_masked")),
                          "std": None},
        },
        "per_view": {},
    }
    return result


# ============================================================
# Comparison Logic
# ============================================================

def compute_comparison(
    fl_norm: Dict[str, Any],
    ps_norm: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute side-by-side comparison between normalized metrics."""
    metrics_keys = ["psnr", "ssim", "lpips", "mask_iou", "l1_masked"]

    # Overall comparison
    overall_comparison = {}
    for key in metrics_keys:
        fl_val = fl_norm["overall"].get(key, {}).get("mean")
        ps_val = ps_norm["overall"].get(key, {}).get("mean")

        entry = {
            "facelift": fl_val,
            "posesplatter": ps_val,
            "diff": None,
            "better": None,
        }

        if fl_val is not None and ps_val is not None:
            entry["diff"] = fl_val - ps_val
            # Higher is better for PSNR, SSIM, IoU; Lower is better for LPIPS, L1
            if key in ("lpips", "l1_masked"):
                entry["better"] = "FaceLift" if fl_val < ps_val else "Pose-Splatter"
            else:
                entry["better"] = "FaceLift" if fl_val > ps_val else "Pose-Splatter"

        overall_comparison[key] = entry

    # Per-view comparison (match by view index)
    per_view_comparison = {}
    fl_views = fl_norm.get("per_view", {})
    ps_views = ps_norm.get("per_view", {})

    all_view_keys = sorted(set(list(fl_views.keys()) + list(ps_views.keys())))
    for vk in all_view_keys:
        fl_vv = fl_views.get(vk, {})
        ps_vv = ps_views.get(vk, {})

        view_comp = {}
        for key in metrics_keys:
            fl_val = fl_vv.get(key, {}).get("mean") if fl_vv else None
            ps_val = ps_vv.get(key, {}).get("mean") if ps_vv else None

            view_comp[key] = {
                "facelift": fl_val,
                "posesplatter": ps_val,
            }

        # Check holdout status
        is_holdout = ps_vv.get("is_holdout", False) if ps_vv else False
        view_comp["is_holdout"] = is_holdout

        per_view_comparison[vk] = view_comp

    return {
        "overall": overall_comparison,
        "per_view": per_view_comparison,
    }


# ============================================================
# Output Formatting
# ============================================================

def fmt(val, precision=4):
    """Format a numeric value, handling None."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{precision}f}"
    return str(val)


def print_comparison_table(
    comparison: Dict[str, Any],
    fl_norm: Dict[str, Any],
    ps_norm: Dict[str, Any],
):
    """Print formatted comparison table to stdout."""
    print(f"\n{'=' * 78}")
    print(f"  FaceLift vs Pose-Splatter  —  Unified Comparison")
    print(f"{'=' * 78}")

    # Metadata
    fl_src = fl_norm.get("source", "unknown")
    ps_src = ps_norm.get("source", "unknown")
    ps_type = ps_norm.get("metric_type", "unknown")
    print(f"\n  FaceLift source:       {fl_src}")
    print(f"  Pose-Splatter source:  {ps_src}")
    if ps_type == "masked":
        print(f"  ⚠  PS metrics are MASKED (foreground-only); FL uses white-BG composite.")
        print(f"     PSNR/SSIM not directly comparable without re-evaluation on same protocol.")

    # Overall table
    print(f"\n  {'Metric':<14} {'FaceLift':>12} {'PoseSplatter':>14} {'Diff':>10} {'Better':>16}")
    print(f"  {'-'*14} {'-'*12} {'-'*14} {'-'*10} {'-'*16}")

    oc = comparison["overall"]

    # Direction indicators
    arrows = {"psnr": "↑", "ssim": "↑", "lpips": "↓", "mask_iou": "↑", "l1_masked": "↓"}
    labels = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
        "mask_iou": "Mask IoU",
        "l1_masked": "L1 (masked)",
    }

    for key in ["psnr", "ssim", "lpips", "mask_iou", "l1_masked"]:
        entry = oc[key]
        label = f"{labels[key]} {arrows[key]}"
        fl_v = fmt(entry["facelift"])
        ps_v = fmt(entry["posesplatter"])
        diff = fmt(entry["diff"])
        if entry["diff"] is not None:
            diff = f"{entry['diff']:+.4f}"
        better = entry["better"] or "—"
        print(f"  {label:<14} {fl_v:>12} {ps_v:>14} {diff:>10} {better:>16}")

    # Per-view breakdown
    pv = comparison.get("per_view", {})
    if pv:
        print(f"\n  Per-view PSNR comparison:")
        print(f"  {'View':<10} {'FaceLift':>10} {'PoseSplatter':>14} {'Holdout':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*14} {'-'*10}")
        for vk in sorted(pv.keys()):
            vv = pv[vk]
            fl_p = fmt(vv.get("psnr", {}).get("facelift"), 2)
            ps_p = fmt(vv.get("psnr", {}).get("posesplatter"), 2)
            holdout = "✓" if vv.get("is_holdout") else ""
            print(f"  {vk:<10} {fl_p:>10} {ps_p:>14} {holdout:>10}")

    print()


def generate_markdown_report(
    comparison: Dict[str, Any],
    fl_norm: Dict[str, Any],
    ps_norm: Dict[str, Any],
    output_path: Path,
):
    """Generate Markdown comparison report."""
    lines = []
    lines.append("# FaceLift vs Pose-Splatter: Unified Comparison Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**FaceLift source**: {fl_norm.get('source', 'unknown')}")
    lines.append(f"**Pose-Splatter source**: {ps_norm.get('source', 'unknown')}")
    lines.append("")

    # Metric protocol warning
    ps_type = ps_norm.get("metric_type", "unknown")
    if ps_type == "masked":
        lines.append("> **Warning**: Pose-Splatter uses **masked** (foreground-only) metrics,")
        lines.append("> while FaceLift uses **white-BG composite** + full-image metrics.")
        lines.append("> PSNR and SSIM are not directly comparable without re-evaluation")
        lines.append("> on the same protocol. IoU and L1 are more comparable.")
        lines.append("")

    # Overall table
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | FaceLift | Pose-Splatter | Diff | Better |")
    lines.append("|--------|----------|---------------|------|--------|")

    oc = comparison["overall"]
    labels = {
        "psnr": "PSNR ↑",
        "ssim": "SSIM ↑",
        "lpips": "LPIPS ↓",
        "mask_iou": "Mask IoU ↑",
        "l1_masked": "L1 (masked) ↓",
    }

    for key in ["psnr", "ssim", "lpips", "mask_iou", "l1_masked"]:
        entry = oc[key]
        label = labels[key]
        fl_v = fmt(entry["facelift"])
        ps_v = fmt(entry["posesplatter"])
        diff = fmt(entry["diff"])
        if entry["diff"] is not None:
            diff = f"{entry['diff']:+.4f}"
        better = entry["better"] or "—"

        # Bold the winner
        if better == "FaceLift":
            fl_v = f"**{fl_v}**"
        elif better == "Pose-Splatter":
            ps_v = f"**{ps_v}**"

        lines.append(f"| {label} | {fl_v} | {ps_v} | {diff} | {better} |")

    lines.append("")

    # Per-view table
    pv = comparison.get("per_view", {})
    if pv:
        lines.append("## Per-View Breakdown (PSNR)")
        lines.append("")
        lines.append("| View | FaceLift | Pose-Splatter | Holdout |")
        lines.append("|------|----------|---------------|---------|")
        for vk in sorted(pv.keys()):
            vv = pv[vk]
            fl_p = fmt(vv.get("psnr", {}).get("facelift"), 2)
            ps_p = fmt(vv.get("psnr", {}).get("posesplatter"), 2)
            holdout = "Yes" if vv.get("is_holdout") else ""
            lines.append(f"| {vk} | {fl_p} | {ps_p} | {holdout} |")
        lines.append("")

    # Model info
    lines.append("## Model Information")
    lines.append("")
    lines.append("| | FaceLift | Pose-Splatter |")
    lines.append("|--|----------|---------------|")
    lines.append("| **Paper** | Lyu et al., ICCV 2025 | Goffinet et al., NeurIPS 2025 |")
    lines.append("| **Architecture** | SD2.1-UnCLIP MVDiffusion + GSLRM | Shape Carving + Stacked U-Net + gsplat |")
    lines.append("| **Inference** | Feed-forward (two-stage) | Feed-forward (~30ms/frame) |")
    lines.append("| **Metric protocol** | White-BG composite + full-image | Masked (foreground-only) |")
    lines.append("")

    # Methodology notes
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("### Metric Protocol Differences")
    lines.append("")
    lines.append("1. **PSNR**: FaceLift composites onto white background then computes full-image MSE.")
    lines.append("   PS extracts foreground pixels via binary mask and computes MSE on those pixels only.")
    lines.append("   White-BG PSNR is inflated by easy-to-predict background pixels.")
    lines.append("")
    lines.append("2. **SSIM**: FaceLift uses skimage on white-BG composite.")
    lines.append("   PS uses torchmetrics SSIM on masked regions with fallback to correlation-based.")
    lines.append("")
    lines.append("3. **L1 (masked)**: Both use the same formula: `sum|pred-gt| / (3 * sum(mask))`.")
    lines.append("   This is the most directly comparable metric.")
    lines.append("")
    lines.append("4. **IoU**: Both use alpha > 0.5 binary mask intersection/union.")
    lines.append("   Directly comparable.")
    lines.append("")
    lines.append("### Recommendations for Fair Comparison")
    lines.append("")
    lines.append("- Re-evaluate both models using **white-BG composite + full-image** protocol")
    lines.append("- Or re-evaluate both using **masked foreground-only** protocol")
    lines.append("- Compare L1 and IoU directly (protocol-agnostic)")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by compare_with_baseline.py | {datetime.now().strftime('%Y-%m-%d')}*")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"  Markdown report saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FaceLift vs Pose-Splatter unified comparison"
    )

    # FaceLift input (one of these)
    fl_group = parser.add_mutually_exclusive_group(required=True)
    fl_group.add_argument(
        "--facelift_metrics", type=str,
        help="Path to FaceLift metrics_v2.json (pre-computed)",
    )
    fl_group.add_argument(
        "--facelift_dir", type=str,
        help="Path to FaceLift E2E output directory (recompute metrics)",
    )

    # Baseline input (one of these)
    bl_group = parser.add_mutually_exclusive_group(required=True)
    bl_group.add_argument(
        "--baseline_metrics", type=str,
        help="Path to PS paper_standard_evaluation.json",
    )
    bl_group.add_argument(
        "--baseline_values", type=str,
        help='JSON string of baseline values, e.g. \'{"psnr": 24.68, "ssim": 0.963}\'',
    )

    # Recompute options (only used with --facelift_dir)
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="GT dataset root (required if --facelift_dir is used)",
    )
    parser.add_argument("--views", type=int, nargs="+", default=None)
    parser.add_argument("--skip_input_view", type=int, default=None)
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="experiments/comparison/FL_vs_PS",
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    # ---- Load FaceLift metrics ----
    if args.facelift_metrics:
        fl_path = Path(args.facelift_metrics)
        if not fl_path.exists():
            print(f"ERROR: FaceLift metrics not found: {fl_path}")
            sys.exit(1)
        with open(fl_path) as f:
            fl_raw = json.load(f)
        fl_norm = normalize_facelift_metrics(fl_raw)
        print(f"Loaded FaceLift metrics from {fl_path}")

    elif args.facelift_dir:
        if not args.data_dir:
            print("ERROR: --data_dir required when using --facelift_dir")
            sys.exit(1)
        # Import and recompute
        try:
            from mouse_extensions.scripts.eval.compute_e2e_metrics import (
                compute_e2e_metrics,
            )
        except ImportError:
            print("ERROR: Cannot import compute_e2e_metrics. Run from FaceLift repo root.")
            sys.exit(1)

        print(f"Recomputing FaceLift metrics from {args.facelift_dir}...")
        fl_raw = compute_e2e_metrics(
            output_dir=args.facelift_dir,
            data_dir=args.data_dir,
            views=args.views,
            skip_input_view=args.skip_input_view,
            device=args.device,
            no_lpips=args.no_lpips,
        )
        fl_norm = normalize_facelift_metrics(fl_raw)

    # ---- Load baseline (Pose-Splatter) metrics ----
    if args.baseline_metrics:
        bl_path = Path(args.baseline_metrics)
        if not bl_path.exists():
            print(f"ERROR: Baseline metrics not found: {bl_path}")
            sys.exit(1)
        with open(bl_path) as f:
            ps_raw = json.load(f)
        ps_norm = normalize_posesplatter_metrics(ps_raw)
        print(f"Loaded Pose-Splatter metrics from {bl_path}")

    elif args.baseline_values:
        try:
            values = json.loads(args.baseline_values)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON for --baseline_values: {e}")
            sys.exit(1)
        ps_norm = normalize_manual_values(values)
        print(f"Using manual baseline values: {values}")

    # ---- Compute comparison ----
    comparison = compute_comparison(fl_norm, ps_norm)

    # ---- Print to terminal ----
    print_comparison_table(comparison, fl_norm, ps_norm)

    # ---- Save outputs ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    output_json = {
        "timestamp": datetime.now().isoformat(),
        "facelift": fl_norm,
        "posesplatter": ps_norm,
        "comparison": comparison,
    }
    json_path = output_dir / "metrics_comparison.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  JSON saved: {json_path}")

    # Markdown report
    md_path = output_dir / "comparison_report.md"
    generate_markdown_report(comparison, fl_norm, ps_norm, md_path)

    print(f"\nDone. Results in {output_dir}/")


if __name__ == "__main__":
    main()
