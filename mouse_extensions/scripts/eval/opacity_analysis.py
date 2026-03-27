# no-split: single analysis script with tightly coupled collect→plot pipeline
"""Gaussian opacity and scaling distribution analysis for GS-LRM.

Samples test frames, runs inference, and generates distribution plots:
  1. opacity_histogram.png       — overall opacity distribution + modality test
  2. opacity_by_filter.png       — before/after each apply_all_filters stage
  3. scaling_anisotropy.png      — max/min scaling ratio per Gaussian
  4. multi_checkpoint_opacity.png — compare opacity across checkpoints

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.opacity_analysis \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --n-frames 20 \
        --output-dir outputs/analysis/mouse/opacity

    # Multi-checkpoint comparison:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.opacity_analysis \
        --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
        --checkpoints alpha0.0=/path/to/ckpt0.pt alpha0.3=/path/to/ckpt3.pt \
        --n-frames 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Test frame range for M5t2 dataset
TEST_RANGE = (3240, 3599)

# Default filter params (matches save_outputs in gslrm_pipeline.py)
FILTER_PARAMS = dict(opacity_thres=0.04, scaling_thres=0.1, floater_thres=0.6,
                     crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0])


def collect_gaussian_stats(
    model, m5_dir: Path, frame_indices: List[int], device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Run inference on frames and collect opacity/scaling arrays.

    Returns dict with keys: opacity_raw, opacity_filtered, scaling_filtered,
    opacity_post_opacity, opacity_post_scaling, opacity_post_floater.
    """
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data

    all_raw, all_filt, all_scale = [], [], []
    all_post_op, all_post_sc, all_post_fl = [], [], []

    for fi in frame_indices:
        fd = m5_dir / f"{fi:06d}"
        if not fd.exists():
            print(f"  [skip] {fi:06d} not found")
            continue
        try:
            imgs, c2ws, fxfys, idx = load_sample_data(str(fd), image_size=512, device=device)
            result = model.predict(imgs, c2ws, fxfys, idx)
            gs = result.gaussians[0]
        except Exception as e:
            print(f"  [ERROR] frame {fi:06d}: {e}")
            continue

        # Raw opacity (before any filter)
        raw_op = gs.get_opacity.squeeze(-1).detach().cpu().numpy()
        all_raw.append(raw_op)

        # Stage-by-stage filter tracking (all methods are IN-PLACE)
        # Stage 1: opacity prune
        gs.prune(FILTER_PARAMS["opacity_thres"])
        all_post_op.append(gs.get_opacity.squeeze(-1).detach().cpu().numpy())

        # Stage 2: scaling prune
        gs.prune_by_scaling(FILTER_PARAMS["scaling_thres"])
        all_post_sc.append(gs.get_opacity.squeeze(-1).detach().cpu().numpy())

        # Stage 3: floater crop
        gs.crop_by_xyz(FILTER_PARAMS["floater_thres"])
        all_post_fl.append(gs.get_opacity.squeeze(-1).detach().cpu().numpy())

        # Stage 4: bbox crop (final stage of apply_all_filters)
        gs.crop(FILTER_PARAMS["crop_bbx"])
        op_f = gs.get_opacity.squeeze(-1).detach().cpu().numpy()
        sc_f = gs.get_scaling.detach().cpu().numpy()  # (N, 3)
        all_filt.append(op_f)
        all_scale.append(sc_f)

        n_raw, n_filt = len(raw_op), len(op_f)
        print(f"  frame {fi:06d}: {n_raw:,} raw -> {n_filt:,} filtered")

    return {
        "opacity_raw": np.concatenate(all_raw) if all_raw else np.array([]),
        "opacity_filtered": np.concatenate(all_filt) if all_filt else np.array([]),
        "scaling_filtered": np.vstack(all_scale) if all_scale else np.zeros((0, 3)),
        "opacity_post_opacity": np.concatenate(all_post_op) if all_post_op else np.array([]),
        "opacity_post_scaling": np.concatenate(all_post_sc) if all_post_sc else np.array([]),
        "opacity_post_floater": np.concatenate(all_post_fl) if all_post_fl else np.array([]),
    }


def _stat_text(arr: np.ndarray) -> str:
    """Return summary statistics string."""
    if len(arr) == 0:
        return "N=0"
    from scipy import stats as sp_stats
    mode_val = float(sp_stats.mode(np.round(arr, 2), keepdims=False).mode)
    pct_low = 100 * np.mean(arr < 0.1)
    pct_mid = 100 * np.mean((arr >= 0.1) & (arr < 0.5))
    pct_high = 100 * np.mean(arr >= 0.5)
    return (f"N={len(arr):,}  mean={arr.mean():.3f}  med={np.median(arr):.3f}  "
            f"mode={mode_val:.2f}\n<0.1: {pct_low:.1f}%  0.1-0.5: {pct_mid:.1f}%  "
            f">=0.5: {pct_high:.1f}%")


def plot_opacity_histogram(data: Dict, out_dir: Path) -> None:
    """Fig 1: Overall opacity distribution with modality analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, title in [
        (axes[0], "opacity_raw", "Raw (pre-filter)"),
        (axes[1], "opacity_filtered", "After apply_all_filters"),
    ]:
        arr = data[key]
        if len(arr) == 0:
            ax.set_title(f"{title} — no data"); continue
        ax.hist(arr, bins=100, range=(0, 1), color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_xlabel("Opacity"); ax.set_ylabel("Count"); ax.set_title(title)
        ax.text(0.02, 0.95, _stat_text(arr), transform=ax.transAxes,
                fontsize=8, va="top", family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.suptitle("Gaussian Opacity Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "opacity_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved opacity_histogram.png")


def plot_opacity_histogram_enhanced(data: Dict, out_dir: Path) -> None:
    """Fig 1b: Enhanced opacity views — log-scale, zoomed, and cumulative."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Gaussian Opacity Distribution (Enhanced Views)", fontsize=14, fontweight="bold")

    for col, (key, label) in enumerate([
        ("opacity_raw", "Raw"),
        ("opacity_filtered", "Filtered"),
    ]):
        arr = data[key]
        if len(arr) == 0:
            continue

        # Row 0, left: Log-scale Y axis
        ax = axes[0, col]
        ax.hist(arr, bins=200, range=(0, 1), color="steelblue", edgecolor="none", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Opacity"); ax.set_ylabel("Count (log)")
        ax.set_title(f"{label} — Log Scale")
        ax.grid(True, alpha=0.3)

        # Row 1, left: Zoomed into 0.0-0.2 range (where most mass is)
        ax2 = axes[1, col]
        ax2.hist(arr[arr < 0.2], bins=100, range=(0, 0.2), color="coral", edgecolor="none", alpha=0.8)
        ax2.set_xlabel("Opacity (zoomed 0-0.2)"); ax2.set_ylabel("Count")
        ax2.set_title(f"{label} — Zoomed Low Range")
        ax2.grid(True, alpha=0.3)

    # Right column: Cumulative distribution (both overlaid)
    ax_cdf = axes[0, 2]
    for key, label, color in [
        ("opacity_raw", "Raw", "steelblue"),
        ("opacity_filtered", "Filtered", "coral"),
    ]:
        arr = data[key]
        if len(arr) == 0:
            continue
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        # Subsample for performance
        step = max(1, len(sorted_arr) // 5000)
        ax_cdf.plot(sorted_arr[::step], cdf[::step], label=f"{label} (N={len(arr):,})", color=color)
    ax_cdf.set_xlabel("Opacity"); ax_cdf.set_ylabel("Cumulative Fraction")
    ax_cdf.set_title("CDF Comparison"); ax_cdf.legend(fontsize=9)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax_cdf.axhline(0.9, color="gray", linestyle="--", alpha=0.5)

    # Bottom right: Filtered log-odds (logit) space
    ax_logit = axes[1, 2]
    arr_f = data["opacity_filtered"]
    if len(arr_f) > 0:
        # Logit transform: log(p / (1-p))
        clamped = np.clip(arr_f, 1e-6, 1 - 1e-6)
        logits = np.log(clamped / (1 - clamped))
        ax_logit.hist(logits, bins=150, color="mediumpurple", edgecolor="none", alpha=0.8)
        ax_logit.set_xlabel("Logit (log-odds)"); ax_logit.set_ylabel("Count")
        ax_logit.set_title("Filtered — Logit Space")
        ax_logit.grid(True, alpha=0.3)
        ax_logit.text(0.02, 0.95, f"mean={logits.mean():.2f}  med={np.median(logits):.2f}",
                      transform=ax_logit.transAxes, fontsize=9, va="top",
                      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    fig.tight_layout()
    fig.savefig(out_dir / "opacity_histogram_enhanced.png", dpi=150)
    plt.close(fig)
    print(f"  Saved opacity_histogram_enhanced.png")


def plot_opacity_by_filter(data: Dict, out_dir: Path) -> None:
    """Fig 2: Opacity distribution at each filter stage."""
    stages = [
        ("opacity_raw", "Raw (all Gaussians)"),
        ("opacity_post_opacity", f"After opacity prune (>{FILTER_PARAMS['opacity_thres']})"),
        ("opacity_post_scaling", f"After scaling prune (<{FILTER_PARAMS['scaling_thres']})"),
        ("opacity_post_floater", f"After floater crop ({FILTER_PARAMS['floater_thres']})"),
        ("opacity_filtered", "After full apply_all_filters"),
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 4), sharey=True)
    for ax, (key, title) in zip(axes, stages):
        arr = data.get(key, np.array([]))
        if len(arr) == 0:
            ax.set_title(title, fontsize=9); continue
        ax.hist(arr, bins=80, range=(0, 1), color="teal", edgecolor="none", alpha=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Opacity")
        ax.text(0.02, 0.95, f"N={len(arr):,}", transform=ax.transAxes, fontsize=8, va="top")
    axes[0].set_ylabel("Count")

    fig.suptitle("Opacity by Filter Stage", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "opacity_by_filter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved opacity_by_filter.png")


def plot_scaling_anisotropy(data: Dict, out_dir: Path) -> None:
    """Fig 3: Scaling anisotropy (max/min ratio) distribution."""
    sc = data["scaling_filtered"]
    if sc.shape[0] == 0:
        print("  [skip] scaling_anisotropy — no data"); return

    max_s = sc.max(axis=1)
    min_s = np.clip(sc.min(axis=1), 1e-7, None)
    ratio = max_s / min_s

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: anisotropy ratio histogram (clipped at 100 for visibility)
    clipped = np.clip(ratio, 0, 100)
    axes[0].hist(clipped, bins=100, color="coral", edgecolor="none", alpha=0.8)
    axes[0].set_xlabel("Anisotropy ratio (max_scale / min_scale)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Scaling Anisotropy Distribution")
    pct_iso = 100 * np.mean(ratio < 3)
    pct_aniso = 100 * np.mean(ratio >= 30)
    axes[0].text(0.55, 0.90, f"N={len(ratio):,}\nmed={np.median(ratio):.1f}\n"
                 f"<3 (isotropic): {pct_iso:.1f}%\n>=30 (flat): {pct_aniso:.1f}%",
                 transform=axes[0].transAxes, fontsize=9, va="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    # Right: per-axis scale distribution
    labels = ["Scale X", "Scale Y", "Scale Z"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, (lbl, c) in enumerate(zip(labels, colors)):
        axes[1].hist(np.clip(sc[:, i], 0, 0.15), bins=80, alpha=0.5, label=lbl, color=c)
    axes[1].set_xlabel("Scale value"); axes[1].set_ylabel("Count")
    axes[1].set_title("Per-axis Scale Distribution"); axes[1].legend()

    fig.suptitle("Gaussian Scaling Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "scaling_anisotropy.png", dpi=150)
    plt.close(fig)
    print(f"  Saved scaling_anisotropy.png")


def plot_multi_checkpoint(
    all_data: Dict[str, Dict], out_dir: Path,
) -> None:
    """Fig 4: Compare opacity distributions across checkpoints."""
    if len(all_data) < 2:
        print("  [skip] multi_checkpoint — need >=2 checkpoints"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_data)))

    for (name, data), color in zip(all_data.items(), colors):
        for ax, key, title in [
            (axes[0], "opacity_raw", "Raw Opacity"),
            (axes[1], "opacity_filtered", "Filtered Opacity"),
        ]:
            arr = data[key]
            if len(arr) == 0:
                continue
            ax.hist(arr, bins=80, range=(0, 1), alpha=0.5, label=f"{name} (N={len(arr):,})",
                    color=color, edgecolor="none")
            ax.set_title(title); ax.set_xlabel("Opacity"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")

    fig.suptitle("Multi-Checkpoint Opacity Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "multi_checkpoint_opacity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved multi_checkpoint_opacity.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Gaussian opacity/scaling distributions from GS-LRM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--m5-dir", required=True, help="M5 preprocessed data directory")
    parser.add_argument("--config", default="configs/base/gslrm_mouse.yaml",
                        help="GS-LRM config YAML")
    parser.add_argument("--checkpoint",
                        default="/node_data/joon/checkpoints/FaceLift/gslrm/"
                                "base_uniform_v2_6view_v2/best_psnr.pt",
                        help="Default checkpoint path")
    parser.add_argument("--checkpoints", nargs="*", metavar="NAME=PATH",
                        help="Multiple checkpoints as name=path pairs for comparison")
    parser.add_argument("--n-frames", type=int, default=20, help="Number of test frames to sample")
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/opacity")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m5_dir = Path(args.m5_dir)

    # Sample test frame indices
    rng = np.random.RandomState(args.seed)
    all_frames = list(range(TEST_RANGE[0], TEST_RANGE[1] + 1))
    n = min(args.n_frames, len(all_frames))
    frame_indices = sorted(rng.choice(all_frames, n, replace=False).tolist())
    print(f"Sampled {n} test frames: {frame_indices[:5]}...{frame_indices[-1]}")

    # Build checkpoint dict
    ckpt_dict: Dict[str, str] = {}
    if args.checkpoints:
        for item in args.checkpoints:
            name, path = item.split("=", 1)
            ckpt_dict[name] = path
    else:
        ckpt_dict["default"] = args.checkpoint

    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference

    all_data: Dict[str, Dict] = {}
    for ckpt_name, ckpt_path in ckpt_dict.items():
        print(f"\n=== Checkpoint: {ckpt_name} ===")
        print(f"  Loading model from {ckpt_path}...")
        model = GSLRMInference(config_path=args.config, checkpoint_path=ckpt_path)
        data = collect_gaussian_stats(model, m5_dir, frame_indices)
        all_data[ckpt_name] = data

        # Save raw stats as JSON
        stats = {k: {"N": len(v), "mean": float(v.mean()) if len(v) > 0 else None,
                      "median": float(np.median(v)) if len(v) > 0 else None}
                 for k, v in data.items() if v.ndim == 1}
        with open(out_dir / f"stats_{ckpt_name}.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Free GPU memory between checkpoints
        del model
        torch.cuda.empty_cache()

    # Use first checkpoint for single-checkpoint plots
    first_data = next(iter(all_data.values()))
    print("\nGenerating plots...")
    plot_opacity_histogram(first_data, out_dir)
    plot_opacity_histogram_enhanced(first_data, out_dir)
    plot_opacity_by_filter(first_data, out_dir)
    plot_scaling_anisotropy(first_data, out_dir)
    plot_multi_checkpoint(all_data, out_dir)

    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
