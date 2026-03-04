#!/usr/bin/env python3
"""
Multi-View Triangulation Visualization Module.

Generates quantitative and qualitative visualizations for
triangulation accuracy analysis across view counts and noise levels.

Outputs:
    1. Heatmap: MPJPE across (view_count × noise_level)
    2. Line chart: MPJPE vs views for each noise level
    3. Per-joint error bar chart: joint-level breakdown
    4. Triangulated 3D scatter: GT vs predicted keypoints
    5. Multi-view camera layout: bird's-eye view of camera arrangement
    6. Representative renders: N-view grids with projected keypoints

Usage:
    python -m mouse_extensions.analysis.triangulation_visualizer \
        --results-json outputs/triangulation_eval/multiview_results.json \
        --gt-results-json outputs/triangulation_analysis/gt_view_results.json \
        --mammal-3d /node_data/joon/.../keypoints_22_3d.npz \
        --output-dir outputs/triangulation_viz

Date: 2026-03-04
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

# Joint names for mouse 22-keypoint model
JOINT_NAMES = [
    "Nose", "Head", "Neck", "UpperBack", "MidBack", "LowerBack", "TailRoot",
    "RF_Shoulder", "RF_Elbow", "RF_Paw",
    "LF_Shoulder", "LF_Elbow", "LF_Paw",
    "RH_Hip", "RH_Knee", "RH_Paw",
    "LH_Hip", "LH_Knee", "LH_Paw",
    "J19", "J20", "J21",
]

JOINT_GROUPS = {
    "Spine": [0, 1, 2, 3, 4, 5, 6],
    "Right Forelimb": [7, 8, 9],
    "Left Forelimb": [10, 11, 12],
    "Right Hindlimb": [13, 14, 15],
    "Left Hindlimb": [16, 17, 18],
}

GROUP_COLORS = {
    "Spine": "#e74c3c",
    "Right Forelimb": "#3498db",
    "Left Forelimb": "#2ecc71",
    "Right Hindlimb": "#f39c12",
    "Left Hindlimb": "#9b59b6",
}


def load_results(json_path: str) -> dict:
    """Load results JSON with numeric key conversion."""
    with open(json_path, "r") as f:
        raw = json.load(f)

    results = {}
    for nv_str, noise_dict in raw.items():
        nv = int(nv_str)
        results[nv] = {}
        for sigma_str, metrics in noise_dict.items():
            sigma = float(sigma_str)
            results[nv][sigma] = metrics
    return results


def load_gt_results(json_path: str) -> dict:
    """Load GT view experiment results."""
    with open(json_path, "r") as f:
        raw = json.load(f)

    results = {}
    for nv_str, metrics in raw.items():
        results[int(nv_str)] = metrics
    return results


# =============================================================================
# Plot 1: Heatmap — MPJPE across views × noise
# =============================================================================

def plot_heatmap(
    results: dict,
    output_path: str,
    gt_results: Optional[dict] = None,
):
    """Heatmap of MPJPE: rows=noise levels, cols=view counts."""
    view_counts = sorted(results.keys())
    noise_levels = sorted(results[view_counts[0]].keys())

    # Build MPJPE matrix
    data = np.zeros((len(noise_levels), len(view_counts)))
    for ni, sigma in enumerate(noise_levels):
        for vi, nv in enumerate(view_counts):
            data[ni, vi] = results[nv][sigma]["mpjpe"]

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label="MPJPE (mm)")

    # Annotate cells
    for ni in range(len(noise_levels)):
        for vi in range(len(view_counts)):
            val = data[ni, vi]
            text = f"{val:.2f}" if val > 0.01 else "~0"
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(vi, ni, text, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(len(view_counts)))
    ax.set_xticklabels([str(n) for n in view_counts])
    ax.set_yticks(range(len(noise_levels)))
    ax.set_yticklabels([f"σ={s:.0f}px" if s > 0 else "σ=0\n(perfect)" for s in noise_levels])
    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("Detection Noise", fontsize=12)
    ax.set_title("Triangulation MPJPE: Views × Noise Level", fontsize=14)

    # Add GT DANNCE reference line as text
    if gt_results:
        gt6 = gt_results.get(6, {}).get("mpjpe", None)
        if gt6:
            ax.text(
                0.98, 0.02, f"DANNCE 6-view real: {gt6:.2f}mm",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, style="italic", color="darkblue",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved heatmap: {output_path}")


# =============================================================================
# Plot 2: Line chart — MPJPE vs views, with GT overlay
# =============================================================================

def plot_lines_with_gt(
    results: dict,
    output_path: str,
    gt_results: Optional[dict] = None,
):
    """Line chart: synthetic noise results + DANNCE real baseline."""
    fig, ax = plt.subplots(figsize=(9, 6))

    view_counts = sorted(results.keys())
    noise_levels = sorted(results[view_counts[0]].keys())

    cmap = plt.cm.Reds
    colors = [cmap(0.2 + 0.7 * i / (len(noise_levels) - 1)) for i in range(len(noise_levels))]

    for ni, sigma in enumerate(noise_levels):
        mpjpes = [results[nv][sigma]["mpjpe"] for nv in view_counts]
        label = f"Synthetic σ={sigma:.0f}px" if sigma > 0 else "Synthetic σ=0 (perfect)"
        ax.plot(view_counts, mpjpes, "o-", label=label, color=colors[ni],
                linewidth=2, markersize=7)

    # Overlay DANNCE real results
    if gt_results:
        gt_views = sorted(gt_results.keys())
        gt_mpjpes = [gt_results[nv]["mpjpe"] for nv in gt_views]
        ax.plot(gt_views, gt_mpjpes, "s--", label="DANNCE Real 2D",
                color="darkblue", linewidth=2.5, markersize=9, zorder=10)

        # Shade the region between σ=5 and DANNCE to show "unexplained error"
        if 6 in gt_results and 6 in results and 5.0 in results[6]:
            ax.annotate(
                f"Gap = {gt_results[6]['mpjpe'] - results[6][5.0]['mpjpe']:.1f}mm\n"
                f"(calibration + body model error)",
                xy=(6, gt_results[6]["mpjpe"]),
                xytext=(8, gt_results[6]["mpjpe"] + 1),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="darkblue"),
                color="darkblue",
            )

    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title("Triangulation Accuracy: Synthetic Noise vs DANNCE Real", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(view_counts)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved line chart: {output_path}")


# =============================================================================
# Plot 3: Per-joint error bar chart
# =============================================================================

def plot_per_joint_errors(
    results: dict,
    output_path: str,
    view_count: int = 6,
    noise_level: float = 5.0,
    gt_results: Optional[dict] = None,
):
    """Per-joint MPJPE bar chart comparing synthetic vs real."""
    fig, ax = plt.subplots(figsize=(14, 5))

    n_joints = 22
    x = np.arange(n_joints)
    width = 0.35

    # Synthetic per-joint errors
    syn_per_joint = results[view_count][noise_level].get("per_joint", [0] * n_joints)
    bars1 = ax.bar(x - width / 2, syn_per_joint[:n_joints], width,
                   label=f"Synthetic σ={noise_level:.0f}px, {view_count}v",
                   color="#e74c3c", alpha=0.7)

    # GT DANNCE per-joint errors
    if gt_results and view_count in gt_results:
        gt_per_joint = gt_results[view_count].get("per_joint", [0] * n_joints)
        bars2 = ax.bar(x + width / 2, gt_per_joint[:n_joints], width,
                       label=f"DANNCE Real, {view_count}v",
                       color="#3498db", alpha=0.7)

    # Color bars by joint group
    for group_name, joint_indices in JOINT_GROUPS.items():
        color = GROUP_COLORS[group_name]
        for ji in joint_indices:
            if ji < n_joints:
                ax.axvspan(ji - 0.5, ji + 0.5, alpha=0.05, color=color)

    # Labels
    labels = JOINT_NAMES[:n_joints]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (mm)", fontsize=11)
    ax.set_title(f"Per-Joint Error: {view_count} Views", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Group labels at bottom
    group_positions = {
        "Spine": (0, 6), "RF": (7, 9), "LF": (10, 12),
        "RH": (13, 15), "LH": (16, 18),
    }
    for gname, (start, end) in group_positions.items():
        mid = (start + end) / 2
        ax.annotate(gname, xy=(mid, -0.05), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved per-joint chart: {output_path}")


# =============================================================================
# Plot 4: Noise reduction factor chart
# =============================================================================

def plot_noise_reduction(
    results: dict,
    output_path: str,
):
    """Show how much each additional view reduces error (diminishing returns)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    view_counts = sorted(results.keys())
    noise_levels = [s for s in sorted(results[view_counts[0]].keys()) if s > 0]

    # Left: Relative error (normalized to 6-view)
    for sigma in noise_levels:
        baseline = results[view_counts[0]][sigma]["mpjpe"]
        ratios = [results[nv][sigma]["mpjpe"] / baseline * 100 for nv in view_counts]
        ax1.plot(view_counts, ratios, "o-", label=f"σ={sigma:.0f}px", linewidth=2)

    ax1.set_xlabel("Number of Views", fontsize=11)
    ax1.set_ylabel("Relative MPJPE (%, baseline=6v)", fontsize=11)
    ax1.set_title("Error Reduction vs View Count", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(view_counts)
    ax1.axhline(y=100, color="gray", linestyle=":", alpha=0.5)

    # Right: Theoretical √N improvement
    theoretical_n = np.array(view_counts, dtype=float)
    theoretical_reduction = np.sqrt(view_counts[0] / theoretical_n) * 100

    for sigma in noise_levels:
        baseline = results[view_counts[0]][sigma]["mpjpe"]
        ratios = [results[nv][sigma]["mpjpe"] / baseline * 100 for nv in view_counts]
        ax2.plot(view_counts, ratios, "o-",
                 label=f"Actual σ={sigma:.0f}px", linewidth=2, alpha=0.8)

    ax2.plot(view_counts, theoretical_reduction, "k--",
             label="Theoretical √(6/N)", linewidth=2, alpha=0.6)

    ax2.set_xlabel("Number of Views", fontsize=11)
    ax2.set_ylabel("Relative MPJPE (%)", fontsize=11)
    ax2.set_title("Actual vs Theoretical √N Improvement", fontsize=13)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(view_counts)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved noise reduction chart: {output_path}")


# =============================================================================
# Plot 5: Camera layout bird's-eye view
# =============================================================================

def plot_camera_layouts(
    output_path: str,
    view_counts: List[int] = None,
):
    """Show camera arrangements for different view counts (top-down)."""
    if view_counts is None:
        view_counts = [6, 9, 12, 24]

    fig, axes = plt.subplots(1, len(view_counts), figsize=(4 * len(view_counts), 4))
    if len(view_counts) == 1:
        axes = [axes]

    for ax, nv in zip(axes, view_counts):
        # Turntable camera positions
        angles = np.linspace(0, 2 * np.pi, nv, endpoint=False)
        radius = 2.7
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        # Plot camera positions
        ax.scatter(x, y, s=60, c="steelblue", zorder=5, edgecolors="navy", linewidth=0.5)

        # Draw viewing directions (toward center)
        for xi, yi in zip(x, y):
            dx = -xi * 0.15
            dy = -yi * 0.15
            ax.arrow(xi, yi, dx, dy, head_width=0.08, head_length=0.05,
                     fc="steelblue", ec="navy", alpha=0.6, linewidth=0.5)

        # Subject at center
        ax.scatter([0], [0], s=200, c="coral", marker="*", zorder=10,
                   edgecolors="darkred", linewidth=0.5, label="Subject")

        # Circle showing orbit
        circle = plt.Circle((0, 0), radius, fill=False, linestyle="--",
                            color="gray", alpha=0.3)
        ax.add_patch(circle)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(f"N={nv} views\n(Δθ={360/nv:.0f}°)", fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X (m)")
        if ax == axes[0]:
            ax.set_ylabel("Y (m)")

    plt.suptitle("Camera Arrangements (Top-Down View)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved camera layout: {output_path}")


# =============================================================================
# Plot 6: 3D keypoint scatter (GT vs predicted)
# =============================================================================

def plot_3d_keypoints(
    gt_3d: np.ndarray,
    pred_3d_per_config: Dict[str, np.ndarray],
    output_path: str,
    frame_idx: int = 0,
):
    """3D scatter plot of GT vs predicted keypoints for different configs.

    Args:
        gt_3d: (22, 3) single frame GT
        pred_3d_per_config: {"6v σ=5": (22,3), "24v σ=5": (22,3), ...}
        output_path: Save path
        frame_idx: Frame number for title
    """
    fig = plt.figure(figsize=(12, 5))
    n_configs = len(pred_3d_per_config) + 1  # +1 for GT

    for i, (config_name, pred_3d) in enumerate(pred_3d_per_config.items()):
        ax = fig.add_subplot(1, n_configs, i + 1, projection="3d")

        # GT points
        ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2],
                   c="blue", s=30, alpha=0.5, label="GT")

        # Predicted points
        ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2],
                   c="red", s=30, alpha=0.5, label="Pred")

        # Error lines
        for j in range(len(gt_3d)):
            ax.plot([gt_3d[j, 0], pred_3d[j, 0]],
                    [gt_3d[j, 1], pred_3d[j, 1]],
                    [gt_3d[j, 2], pred_3d[j, 2]],
                    "g-", alpha=0.3, linewidth=0.5)

        err = np.linalg.norm(gt_3d - pred_3d, axis=1).mean()
        ax.set_title(f"{config_name}\nMPJPE={err:.2f}mm", fontsize=9)
        ax.legend(fontsize=7)

    plt.suptitle(f"3D Keypoints: GT vs Triangulated (Frame {frame_idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved 3D scatter: {output_path}")


# =============================================================================
# Plot 7: Summary dashboard
# =============================================================================

def plot_summary_dashboard(
    results: dict,
    output_path: str,
    gt_results: Optional[dict] = None,
):
    """Combined dashboard with heatmap + line chart + noise reduction."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    view_counts = sorted(results.keys())
    noise_levels = sorted(results[view_counts[0]].keys())
    noise_nonzero = [s for s in noise_levels if s > 0]

    # --- Panel 1: Heatmap ---
    ax1 = fig.add_subplot(gs[0, 0])
    data = np.zeros((len(noise_levels), len(view_counts)))
    for ni, sigma in enumerate(noise_levels):
        for vi, nv in enumerate(view_counts):
            data[ni, vi] = results[nv][sigma]["mpjpe"]

    im = ax1.imshow(data, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax1, label="MPJPE (mm)", shrink=0.8)

    for ni in range(len(noise_levels)):
        for vi in range(len(view_counts)):
            val = data[ni, vi]
            text = f"{val:.2f}" if val > 0.01 else "~0"
            color = "white" if val > data.max() * 0.6 else "black"
            ax1.text(vi, ni, text, ha="center", va="center", fontsize=9, color=color)

    ax1.set_xticks(range(len(view_counts)))
    ax1.set_xticklabels([str(n) for n in view_counts])
    ax1.set_yticks(range(len(noise_levels)))
    ax1.set_yticklabels([f"σ={s:.0f}" if s > 0 else "σ=0" for s in noise_levels])
    ax1.set_xlabel("Views")
    ax1.set_ylabel("Noise (px)")
    ax1.set_title("A. MPJPE Heatmap (mm)", fontweight="bold")

    # --- Panel 2: Line chart with GT ---
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = plt.cm.Reds
    for ni, sigma in enumerate(noise_nonzero):
        mpjpes = [results[nv][sigma]["mpjpe"] for nv in view_counts]
        c = cmap(0.3 + 0.6 * ni / (len(noise_nonzero) - 1))
        ax2.plot(view_counts, mpjpes, "o-", label=f"σ={sigma:.0f}px",
                 color=c, linewidth=2)

    if gt_results:
        gt_views = sorted(gt_results.keys())
        gt_mpjpes = [gt_results[nv]["mpjpe"] for nv in gt_views]
        ax2.plot(gt_views, gt_mpjpes, "s--", label="DANNCE Real",
                 color="darkblue", linewidth=2.5, markersize=8, zorder=10)

    ax2.set_xlabel("Views")
    ax2.set_ylabel("MPJPE (mm)")
    ax2.set_title("B. Synthetic vs Real Detection", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(view_counts)

    # --- Panel 3: Noise reduction ---
    ax3 = fig.add_subplot(gs[1, 0])
    theoretical_n = np.array(view_counts, dtype=float)
    theoretical_reduction = np.sqrt(view_counts[0] / theoretical_n) * 100

    for sigma in noise_nonzero:
        baseline = results[view_counts[0]][sigma]["mpjpe"]
        ratios = [results[nv][sigma]["mpjpe"] / baseline * 100 for nv in view_counts]
        ax3.plot(view_counts, ratios, "o-", label=f"σ={sigma:.0f}px", linewidth=2)

    ax3.plot(view_counts, theoretical_reduction, "k--",
             label="√(6/N) theory", linewidth=2, alpha=0.5)
    ax3.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
    ax3.set_xlabel("Views")
    ax3.set_ylabel("Relative MPJPE (%, 6v=100%)")
    ax3.set_title("C. Noise Reduction vs View Count", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(view_counts)

    # --- Panel 4: Key findings text ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    findings = []
    # Best reduction
    for sigma in noise_nonzero:
        base = results[view_counts[0]][sigma]["mpjpe"]
        best = results[view_counts[-1]][sigma]["mpjpe"]
        reduction = (1 - best / base) * 100
        findings.append(f"σ={sigma:.0f}px: {view_counts[-1]}v reduces error by {reduction:.0f}%")

    if gt_results and 6 in gt_results:
        findings.append(f"\nDANNCE 6-view real: {gt_results[6]['mpjpe']:.2f}mm")
        if 5.0 in results.get(6, {}):
            gap = gt_results[6]["mpjpe"] - results[6][5.0]["mpjpe"]
            findings.append(f"Gap vs σ=5 synthetic: {gap:.2f}mm")
            findings.append(f"→ Non-noise error (calib/model): ~{gap:.1f}mm")

    findings_text = "Key Findings:\n\n" + "\n".join(f"• {f}" for f in findings)
    ax4.text(0.05, 0.95, findings_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax4.set_title("D. Summary", fontweight="bold")

    plt.suptitle("Multi-View Triangulation Analysis Dashboard", fontsize=15, fontweight="bold")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved dashboard: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Triangulation visualization")
    parser.add_argument("--results-json", type=str, required=True,
                        help="Phase 2 multiview_results.json")
    parser.add_argument("--gt-results-json", type=str, default=None,
                        help="Phase 1 gt_view_results.json (DANNCE real)")
    parser.add_argument("--output-dir", type=str, default="outputs/triangulation_viz")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.results_json)
    gt_results = load_gt_results(args.gt_results_json) if args.gt_results_json else None

    logger.info(f"Loaded synthetic results: {sorted(results.keys())} views")
    if gt_results:
        logger.info(f"Loaded GT results: {sorted(gt_results.keys())} views")

    # Generate all plots
    plot_heatmap(results, output_dir / "01_heatmap.png", gt_results)
    plot_lines_with_gt(results, output_dir / "02_lines_with_gt.png", gt_results)
    plot_per_joint_errors(results, output_dir / "03_per_joint_6v.png",
                          view_count=6, noise_level=5.0, gt_results=gt_results)
    plot_noise_reduction(results, output_dir / "04_noise_reduction.png")
    plot_camera_layouts(output_dir / "05_camera_layouts.png")
    plot_summary_dashboard(results, output_dir / "06_dashboard.png", gt_results)

    logger.info(f"All plots saved to {output_dir}")
    print(f"\nGenerated 6 visualization files in {output_dir}/")


if __name__ == "__main__":
    main()
