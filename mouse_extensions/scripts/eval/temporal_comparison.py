"""Temporal Consistency Comparison Pipeline.

Compares multiple temporal smoothing methods across experiment settings.
Generates:
  1. Per-method temporal metrics (tOF, TLPIPS, FF-SSIM-var)
  2. Comparison grid images and videos
  3. Summary JSON and markdown table

Usage:
    python -m mouse_extensions.scripts.eval.temporal_comparison \
        --data_root outputs/datasets/temporal_eval \
        --output_dir outputs/report/temporal_comparison

See: docs/experiments/TEMPORAL_EVAL_STANDARD.md
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from mouse_extensions.evaluation.temporal_smoothing import (
    TemporalStabilityResult,
    bilateral_temporal_sequence,
    compute_temporal_metrics,
    ema_smooth_sequence,
    load_frame_sequence,
    optflow_smooth_sequence,
    save_frame_sequence,
    savgol_temporal_sequence,
    temporal_median_sequence,
)

logger = logging.getLogger(__name__)

# ============================================================
# Standard Configuration (from TEMPORAL_EVAL_STANDARD.md)
# ============================================================

SPARSE_FRAMES = [3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3599]
DENSE_FRAMES = list(range(3300, 3321))  # 3300-3320 inclusive
VIEWS = ["bottom", "top", "front_low", "side_low"]
EXPERIMENTS = ["baseline_6v", "alpha03", "alpha05", "alpha10"]
EXPERIMENT_LABELS = {
    "baseline_6v": "Baseline (α=0)",
    "alpha03": "α=0.3",
    "alpha05": "α=0.5",
    "alpha10": "α=1.0",
}

# Smoothing methods to compare
SMOOTHING_METHODS = {
    "original": {"fn": None, "params": {}},
    "ema_01": {"fn": "ema", "params": {"alpha": 0.1}},
    "ema_03": {"fn": "ema", "params": {"alpha": 0.3}},
    "median_3": {"fn": "median", "params": {"window": 3}},
    "median_5": {"fn": "median", "params": {"window": 5}},
    "bilateral_5": {"fn": "bilateral", "params": {"window": 5, "sigma_intensity": 25.0}},
    "bilateral_3": {"fn": "bilateral", "params": {"window": 3, "sigma_intensity": 15.0}},
    "savgol_5": {"fn": "savgol", "params": {"window": 5, "poly_order": 2}},
    "optflow_03": {"fn": "optflow", "params": {"alpha": 0.3}},
}


def get_frame_dir(data_root: str, experiment: str, view: str) -> Path:
    """Get directory containing rendered frames for an experiment+view."""
    return Path(data_root) / experiment / "mouse_m5t2" / "tier0_raw" / view


def apply_smoothing(
    frames: List[np.ndarray], method: str, params: dict
) -> List[np.ndarray]:
    """Apply a smoothing method to a frame sequence."""
    if method is None or method == "original":
        return list(frames)
    elif method == "ema":
        return ema_smooth_sequence(frames, **params)
    elif method == "median":
        return temporal_median_sequence(frames, **params)
    elif method == "bilateral":
        return bilateral_temporal_sequence(frames, **params)
    elif method == "savgol":
        return savgol_temporal_sequence(frames, **params)
    elif method == "optflow":
        return optflow_smooth_sequence(frames, **params)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


# ============================================================
# Grid Generation
# ============================================================

def create_comparison_grid(
    frames_dict: Dict[str, np.ndarray],
    labels: List[str],
    label_height: int = 36,
    padding: int = 4,
    font_scale: float = 0.7,
) -> np.ndarray:
    """Create a single-row comparison grid with labels.

    Args:
        frames_dict: {label: frame_image} ordered dict
        labels: Column labels
        label_height: Height of label area
        padding: Padding between cells
        font_scale: OpenCV font scale

    Returns:
        [H+label_height, W*N, 3] uint8 image
    """
    frame_list = [frames_dict[l] for l in labels]
    h, w = frame_list[0].shape[:2]
    n = len(frame_list)

    canvas_w = n * w + (n - 1) * padding
    canvas_h = h + label_height
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, (label, frame) in enumerate(zip(labels, frame_list)):
        x = i * (w + padding)
        # Label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        tx = x + (w - text_size[0]) // 2
        cv2.putText(
            canvas, label, (tx, label_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2,
        )
        # Image (convert RGB to BGR for cv2, then back)
        canvas[label_height:, x : x + w] = frame

    return canvas


def create_temporal_strip(
    frames: List[np.ndarray],
    frame_ids: List[int],
    n_show: int = 8,
    cell_size: int = 192,
    label: str = "",
) -> np.ndarray:
    """Create a horizontal strip of consecutive frames for temporal visualization.

    Args:
        frames: Dense frame sequence
        frame_ids: Frame IDs corresponding to frames
        n_show: Number of frames to show
        cell_size: Size to resize each frame
        label: Row label

    Returns:
        [cell_size + label_h, cell_size * n_show, 3] uint8 image
    """
    step = max(1, len(frames) // n_show)
    indices = list(range(0, len(frames), step))[:n_show]

    label_h = 24
    canvas_w = cell_size * n_show
    canvas_h = cell_size + label_h
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, idx in enumerate(indices):
        frame = cv2.resize(frames[idx], (cell_size, cell_size))
        x = i * cell_size
        canvas[label_h:, x : x + cell_size] = frame
        # Frame ID label
        fid = frame_ids[idx] if idx < len(frame_ids) else idx
        cv2.putText(
            canvas, str(fid), (x + 4, label_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1,
        )

    # Row label
    if label:
        cv2.putText(
            canvas, label, (4, cell_size + label_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1,
        )

    return canvas


# ============================================================
# Main Pipeline
# ============================================================

def run_comparison(
    data_root: str,
    output_dir: str,
    experiments: Optional[List[str]] = None,
    views: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    use_lpips: bool = True,
    device: str = "cuda",
) -> Dict:
    """Run full temporal comparison pipeline.

    Args:
        data_root: Root directory with experiment renders
        output_dir: Output directory for results
        experiments: Experiment IDs to compare (default: all 4)
        views: Novel views to evaluate (default: all 4)
        methods: Smoothing methods to compare (default: all)
        use_lpips: Whether to compute TLPIPS
        device: Device for LPIPS

    Returns:
        Dict with all metrics
    """
    experiments = experiments or EXPERIMENTS
    views = views or VIEWS
    methods = methods or list(SMOOTHING_METHODS.keys())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grids").mkdir(exist_ok=True)
    (out / "strips").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)

    all_results = {}

    for exp in experiments:
        exp_label = EXPERIMENT_LABELS.get(exp, exp)
        logger.info(f"Processing experiment: {exp} ({exp_label})")
        all_results[exp] = {}

        for view in views:
            logger.info(f"  View: {view}")
            frame_dir = get_frame_dir(data_root, exp, view)

            if not frame_dir.exists():
                logger.warning(f"  Frame dir not found: {frame_dir}")
                continue

            # Load dense sequence for temporal analysis
            try:
                dense_frames = load_frame_sequence(frame_dir, DENSE_FRAMES)
            except FileNotFoundError as e:
                logger.warning(f"  Missing dense frames: {e}")
                continue

            all_results[exp][view] = {}

            # Compute metrics for each smoothing method
            for method_name in methods:
                method_cfg = SMOOTHING_METHODS[method_name]
                fn_name = method_cfg["fn"]
                params = method_cfg["params"]

                logger.info(f"    Method: {method_name}")

                # Apply smoothing
                smoothed = apply_smoothing(
                    dense_frames,
                    fn_name if fn_name else "original",
                    params,
                )

                # Compute temporal metrics
                metrics = compute_temporal_metrics(
                    smoothed,
                    gt_frames=None,  # No GT for novel views
                    use_lpips=use_lpips,
                    device=device,
                )

                all_results[exp][view][method_name] = metrics.to_dict()
                logger.info(f"      tOF={metrics.tof_mean:.4f} "
                            f"TLPIPS={metrics.tlpips_mean:.4f} "
                            f"FF-SSIM-var={metrics.ff_ssim_var:.6f}")

            # Generate temporal strip comparison
            strip_rows = []
            for method_name in methods:
                method_cfg = SMOOTHING_METHODS[method_name]
                smoothed = apply_smoothing(
                    dense_frames,
                    method_cfg["fn"] if method_cfg["fn"] else "original",
                    method_cfg["params"],
                )
                strip = create_temporal_strip(
                    smoothed, DENSE_FRAMES, n_show=10,
                    cell_size=128, label=method_name,
                )
                strip_rows.append(strip)

            combined_strip = np.vstack(strip_rows)
            strip_path = out / "strips" / f"temporal_strip_{exp}_{view}.png"
            cv2.imwrite(str(strip_path), cv2.cvtColor(combined_strip, cv2.COLOR_RGB2BGR))
            logger.info(f"  Saved strip: {strip_path}")

        # Generate per-experiment comparison grid (mid-frame, all methods × all views)
        mid_frame_id = DENSE_FRAMES[len(DENSE_FRAMES) // 2]  # frame 3310
        grid_frames = {}
        for method_name in methods:
            method_cfg = SMOOTHING_METHODS[method_name]
            row_frames = []
            for view in views:
                frame_dir = get_frame_dir(data_root, exp, view)
                if not frame_dir.exists():
                    continue
                try:
                    frames = load_frame_sequence(frame_dir, [mid_frame_id])
                    smoothed = apply_smoothing(
                        frames,
                        method_cfg["fn"] if method_cfg["fn"] else "original",
                        method_cfg["params"],
                    )
                    row_frames.append(smoothed[0])
                except Exception:
                    row_frames.append(np.zeros((384, 384, 3), dtype=np.uint8))

            if row_frames:
                grid_frames[method_name] = np.hstack(row_frames)

        if grid_frames:
            grid = create_comparison_grid(
                grid_frames, list(grid_frames.keys()), label_height=40,
            )
            grid_path = out / "grids" / f"methods_grid_{exp}.png"
            cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # Save metrics JSON
    metrics_path = out / "metrics" / "temporal_comparison.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    # Generate summary table
    _generate_summary_table(all_results, out / "metrics" / "comparison_table.md", methods)

    return all_results


def _generate_summary_table(
    results: Dict, output_path: Path, methods: List[str],
) -> None:
    """Generate markdown summary table."""
    lines = [
        "# Temporal Consistency Comparison\n",
        "| Experiment | View | Method | tOF↓ | TLPIPS↓ | FF-SSIM-var↓ | Flicker↓ |",
        "|------------|------|--------|------|---------|-------------|---------|",
    ]

    for exp in results:
        exp_label = EXPERIMENT_LABELS.get(exp, exp)
        for view in results[exp]:
            for method in methods:
                if method not in results[exp][view]:
                    continue
                m = results[exp][view][method]
                lines.append(
                    f"| {exp_label} | {view} | {method} | "
                    f"{m.get('tof/mean', 0):.4f} | "
                    f"{m.get('tlpips/mean', 0):.4f} | "
                    f"{m.get('ff_ssim/var', 0):.6f} | "
                    f"{m.get('flicker/rate', 0):.3f} |"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Temporal Consistency Comparison")
    parser.add_argument(
        "--data_root", default="outputs/datasets/temporal_eval",
        help="Root directory with experiment renders",
    )
    parser.add_argument(
        "--output_dir", default="outputs/report/temporal_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiments to compare (default: all 4)",
    )
    parser.add_argument(
        "--views", nargs="+", default=None,
        help="Views to evaluate (default: all 4)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Smoothing methods (default: all)",
    )
    parser.add_argument(
        "--no-lpips", action="store_true",
        help="Skip TLPIPS computation (faster)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for LPIPS (default: cuda)",
    )
    args = parser.parse_args()

    run_comparison(
        data_root=args.data_root,
        output_dir=args.output_dir,
        experiments=args.experiments,
        views=args.views,
        methods=args.methods,
        use_lpips=not args.no_lpips,
        device=args.device,
    )


if __name__ == "__main__":
    main()
