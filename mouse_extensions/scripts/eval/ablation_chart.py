"""Ablation summary charts — view count and alpha weight.

Uses hardcoded fair-eval results (M5t2, test set). Runs locally (no GPU).
Source: CLAUDE.md §11 + memory/project_alpha_loss_study.md (2026-03-23 S26)

Usage:
    python -m mouse_extensions.scripts.eval.ablation_chart
    python -m mouse_extensions.scripts.eval.ablation_chart --output-dir outputs/analysis/mouse/ablation_charts
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Hardcoded results (fair eval, PSNR_gt, M5t2 test set)
# ---------------------------------------------------------------------------

# View ablation: M5t2, 6-view model trained per N views, 360 test frames × 5 views
VIEW_DATA = {
    "views":    [1,     2,     3,     4,     5,     6],
    "psnr_gt":  [10.47, 15.95, 18.56, 20.66, 22.16, 23.84],
    "psnr_int": [10.47, 17.91, 19.54, 21.29, 22.56, 24.02],
    "iou":      [0.028, 0.858, 0.899, 0.926, 0.942, 0.954],
}

# Alpha ablation: 4-view models, M5t2
ALPHA_4V = {
    "alpha":   [0.0,   0.3,   0.5,   1.0],
    "psnr_gt": [20.66, 19.88, 19.72, 19.49],
    "iou":     [0.926, 0.914, 0.912, 0.908],
}

# Alpha ablation: 6-view models, M5t2
ALPHA_6V = {
    "alpha":   [0.0,   0.3,   0.5,   1.0],
    "psnr_gt": [23.84, 23.29, 23.00, 22.55],
    "iou":     [0.954, 0.956, 0.953, 0.949],
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _annotate_points(ax, xs, ys, color, offset=(0, 8)):
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y),
                    textcoords="offset points", xytext=offset,
                    ha="center", fontsize=8.5, color=color)


def plot_view_ablation(output_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "View Ablation  —  Fair Eval (M5t2 test set, 360 frames × 5 views)",
        fontsize=13, fontweight="bold",
    )

    v = VIEW_DATA["views"]
    c_psnr = "#1565C0"
    c_int  = "#90CAF9"
    c_iou  = "#C62828"

    # --- PSNR ---
    ax1.plot(v, VIEW_DATA["psnr_gt"],  "o-",  color=c_psnr, linewidth=2.2, markersize=8,
             label="PSNR_gt (masked fg)")
    ax1.plot(v, VIEW_DATA["psnr_int"], "s--", color=c_int,  linewidth=1.5, markersize=6, alpha=0.8,
             label="PSNR_int (white-bg)")
    ax1.axvline(x=6, color="green", linestyle=":", alpha=0.4)
    ax1.set_xlabel("Number of Input Views", fontsize=11)
    ax1.set_ylabel("PSNR (dB)", fontsize=11)
    ax1.set_title("PSNR vs Input Views")
    ax1.set_xticks(v)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)
    _annotate_points(ax1, v, VIEW_DATA["psnr_gt"], c_psnr, offset=(0, 9))

    # --- IoU ---
    ax2.plot(v, VIEW_DATA["iou"], "o-", color=c_iou, linewidth=2.2, markersize=8)
    ax2.axvline(x=6, color="green", linestyle=":", alpha=0.4, label="best (6-view)")
    ax2.set_xlabel("Number of Input Views", fontsize=11)
    ax2.set_ylabel("IoU", fontsize=11)
    ax2.set_title("IoU vs Input Views")
    ax2.set_xticks(v)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    _annotate_points(ax2, v, VIEW_DATA["iou"], c_iou, offset=(0, 6))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_alpha_ablation(output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Alpha Supervision Ablation  —  Fair Eval (M5t2 test set, PSNR_gt)\n"
        "Note: PSNR_gt ↓ with higher α, but IoU improves at 6-view α=0.3",
        fontsize=12, fontweight="bold",
    )

    c4 = "#E64A19"   # 4-view color
    c6 = "#6A1B9A"   # 6-view color

    def _ax_psnr(ax, data4, data6):
        ax.plot(data4["alpha"], data4["psnr_gt"], "o-",  color=c4, linewidth=2.2,
                markersize=8, label="4-view")
        ax.plot(data6["alpha"], data6["psnr_gt"], "s-",  color=c6, linewidth=2.2,
                markersize=8, label="6-view")
        ax.set_xlabel("α (alpha loss weight)", fontsize=10)
        ax.set_ylabel("PSNR_gt (dB)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        _annotate_points(ax, data4["alpha"], data4["psnr_gt"], c4, offset=(-12, 7))
        _annotate_points(ax, data6["alpha"], data6["psnr_gt"], c6, offset=(12, -14))

    def _ax_iou(ax, data4, data6):
        ax.plot(data4["alpha"], data4["iou"], "o-",  color=c4, linewidth=2.2,
                markersize=8, label="4-view")
        ax.plot(data6["alpha"], data6["iou"], "s-",  color=c6, linewidth=2.2,
                markersize=8, label="6-view")
        ax.set_xlabel("α (alpha loss weight)", fontsize=10)
        ax.set_ylabel("IoU", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        _annotate_points(ax, data4["alpha"], data4["iou"], c4, offset=(-12, 7))
        _annotate_points(ax, data6["alpha"], data6["iou"], c6, offset=(12, -10))

    axes[0, 0].set_title("PSNR_gt vs α")
    _ax_psnr(axes[0, 0], ALPHA_4V, ALPHA_6V)

    axes[0, 1].set_title("IoU vs α")
    _ax_iou(axes[0, 1], ALPHA_4V, ALPHA_6V)

    # PSNR delta from baseline
    axes[1, 0].set_title("ΔPSNR_gt vs α  (relative to α=0)")
    base4, base6 = ALPHA_4V["psnr_gt"][0], ALPHA_6V["psnr_gt"][0]
    d4 = [p - base4 for p in ALPHA_4V["psnr_gt"]]
    d6 = [p - base6 for p in ALPHA_6V["psnr_gt"]]
    axes[1, 0].bar(np.array(ALPHA_4V["alpha"]) - 0.02, d4, width=0.04, color=c4, alpha=0.8, label="4-view")
    axes[1, 0].bar(np.array(ALPHA_6V["alpha"]) + 0.02, d6, width=0.04, color=c6, alpha=0.8, label="6-view")
    axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_xlabel("α", fontsize=10)
    axes[1, 0].set_ylabel("ΔPSNR_gt (dB)", fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.25)

    # IoU delta from baseline
    axes[1, 1].set_title("ΔIoU vs α  (relative to α=0)")
    di4 = [i - ALPHA_4V["iou"][0] for i in ALPHA_4V["iou"]]
    di6 = [i - ALPHA_6V["iou"][0] for i in ALPHA_6V["iou"]]
    axes[1, 1].bar(np.array(ALPHA_4V["alpha"]) - 0.02, di4, width=0.04, color=c4, alpha=0.8, label="4-view")
    axes[1, 1].bar(np.array(ALPHA_6V["alpha"]) + 0.02, di6, width=0.04, color=c6, alpha=0.8, label="6-view")
    axes[1, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].set_xlabel("α", fontsize=10)
    axes[1, 1].set_ylabel("ΔIoU", fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation summary charts (no GPU)")
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/ablation_charts")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_view_ablation(out / "view_ablation_chart.png")
    plot_alpha_ablation(out / "alpha_ablation_chart.png")

    print(f"\nAll charts saved to: {out}/")
    print("  view_ablation_chart.png  — PSNR/IoU vs number of input views (1-6)")
    print("  alpha_ablation_chart.png — PSNR/IoU vs alpha weight (4v + 6v)")


if __name__ == "__main__":
    main()
