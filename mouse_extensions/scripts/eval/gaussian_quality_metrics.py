"""
Literature-Based Gaussian Quality Metrics

3D Gaussian parameter analysis + rendered alpha metrics for artifact evaluation.
No ground truth novel view images required.

Metrics:
    [3D Gaussian Space — from predicted Gaussians]
    - Anisotropy Ratio: max(s) / min(s) per Gaussian (Compact-3DGS)
    - Isotropy Score: Var(log(s)) per Gaussian (LightGaussian, 2024)
    - Opacity Ambiguity: fraction with 0.1 < opacity < 0.9
    - Outlier Count: Gaussians outside scene bounding box
    - Scale Magnitude: mean ||log(s)||_1 (Compact-3DGS)

    [Rendered Alpha — from rendered images]
    - Alpha Sparsity: mean(min(alpha, 1-alpha)) — fuzziness
    - Alpha Entropy: Shannon entropy of alpha histogram

Literature:
    - LightGaussian (Gao et al., 2024): Isotropy regularization
    - Compact-3DGS (He et al., 2024): Scale L1 regularization
    - Mip-NeRF 360 (Barron et al., 2022): Distortion loss concept
    - InfoNeRF (Kim et al., 2021): Entropy regularization

Usage:
    # On server (gpu03)
    CUDA_VISIBLE_DEVICES=6 python -m mouse_extensions.scripts.eval.gaussian_quality_metrics \\
        --checkpoints base_uniform_v2_6view_v2 base_uniform_v2_4view_v2 \\
                      base_uniform_v2_4view_alpha05_v3 base_uniform_v2_4view_alpha10_v3 \\
        --frames 0 500 1000 1500 2000 \\
        --output_dir outputs/reports/gaussian_quality_metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# 3D Gaussian Metrics (from predicted Gaussian parameters)
# ============================================================

def compute_anisotropy_ratio(scales: np.ndarray) -> dict:
    """
    Anisotropy Ratio: max(s) / min(s) per Gaussian.

    Pancake Gaussians have high ratio (>>1), spherical ≈ 1.
    Used in Compact-3DGS analysis.

    Args:
        scales: [N, 3] — scale parameters (already exp-activated)

    Returns:
        dict with mean, median, p95, p99, max, histogram bins
    """
    s_max = np.max(scales, axis=1)
    s_min = np.clip(np.min(scales, axis=1), 1e-8, None)
    ratios = s_max / s_min

    return {
        "mean": float(np.mean(ratios)),
        "median": float(np.median(ratios)),
        "p95": float(np.percentile(ratios, 95)),
        "p99": float(np.percentile(ratios, 99)),
        "max": float(np.max(ratios)),
        "pct_gt_10": float(np.mean(ratios > 10) * 100),  # % with ratio > 10
        "pct_gt_50": float(np.mean(ratios > 50) * 100),  # % with ratio > 50
        "pct_gt_100": float(np.mean(ratios > 100) * 100),
    }


def compute_isotropy_score(scales: np.ndarray) -> dict:
    """
    Isotropy Score: Var(log(s_x), log(s_y), log(s_z)) per Gaussian.

    From LightGaussian (Gao et al., 2024).
    Lower = more isotropic (spherical). 0 = perfect sphere.

    Args:
        scales: [N, 3]

    Returns:
        dict with mean, median, p95
    """
    log_scales = np.log(np.clip(scales, 1e-8, None))
    # Variance across 3 axes for each Gaussian
    var_per_gaussian = np.var(log_scales, axis=1)

    return {
        "mean": float(np.mean(var_per_gaussian)),
        "median": float(np.median(var_per_gaussian)),
        "p95": float(np.percentile(var_per_gaussian, 95)),
        "p99": float(np.percentile(var_per_gaussian, 99)),
    }


def compute_opacity_distribution(opacities: np.ndarray) -> dict:
    """
    Opacity distribution analysis.

    Ideal: bimodal (peaks at 0 and 1).
    Floaters: significant mass in [0.1, 0.9].

    Args:
        opacities: [N] — sigmoid-activated opacity values [0, 1]

    Returns:
        dict with ambiguity score, distribution stats
    """
    amb_mask = (opacities > 0.1) & (opacities < 0.9)
    low_mask = opacities <= 0.1
    high_mask = opacities >= 0.9

    return {
        "ambiguity_pct": float(np.mean(amb_mask) * 100),
        "low_opacity_pct": float(np.mean(low_mask) * 100),
        "high_opacity_pct": float(np.mean(high_mask) * 100),
        "mean_opacity": float(np.mean(opacities)),
        "median_opacity": float(np.median(opacities)),
    }


def compute_scale_magnitude(scales: np.ndarray) -> dict:
    """
    Scale Magnitude: mean ||log(s)||_1 per Gaussian.

    From Compact-3DGS (He et al., 2024).
    Higher = larger Gaussians (potential artifacts).

    Args:
        scales: [N, 3]
    """
    log_scales = np.log(np.clip(scales, 1e-8, None))
    l1_per_gaussian = np.sum(np.abs(log_scales), axis=1)

    return {
        "mean": float(np.mean(l1_per_gaussian)),
        "median": float(np.median(l1_per_gaussian)),
        "p95": float(np.percentile(l1_per_gaussian, 95)),
    }


def compute_spatial_outliers(
    positions: np.ndarray, margin: float = 0.5
) -> dict:
    """
    Count Gaussians outside scene bounding box.

    For FaceLift mouse: normalized scene ≈ [-0.5, 0.5].

    Args:
        positions: [N, 3] — Gaussian center positions
        margin: distance beyond [-margin, margin] considered outlier
    """
    outside = np.any(np.abs(positions) > margin, axis=1)
    return {
        "outlier_count": int(np.sum(outside)),
        "outlier_pct": float(np.mean(outside) * 100),
        "total_gaussians": int(len(positions)),
        "position_range": {
            "x": [float(positions[:, 0].min()), float(positions[:, 0].max())],
            "y": [float(positions[:, 1].min()), float(positions[:, 1].max())],
            "z": [float(positions[:, 2].min()), float(positions[:, 2].max())],
        },
    }


# ============================================================
# Rendered Alpha Metrics (from rendered alpha maps)
# ============================================================

def compute_alpha_sparsity(alpha_map: np.ndarray) -> float:
    """
    Alpha Sparsity: mean(min(alpha, 1-alpha)).

    From common regularization principle.
    Lower = sharper (cleaner edges). 0 = perfect binary.
    Max = 0.5 (all pixels at alpha=0.5).

    Args:
        alpha_map: [H, W] in [0, 1]
    """
    return float(np.mean(np.minimum(alpha_map, 1.0 - alpha_map)))


def compute_alpha_entropy(alpha_map: np.ndarray, n_bins: int = 256) -> float:
    """
    Alpha Entropy: Shannon entropy of alpha histogram.

    Lower = simpler alpha distribution (mostly 0 and 1).
    Higher = more complex/noisy.

    Args:
        alpha_map: [H, W] in [0, 1]
        n_bins: histogram bins
    """
    hist, _ = np.histogram(alpha_map.ravel(), bins=n_bins, range=(0, 1))
    hist = hist / hist.sum()  # normalize to probabilities
    hist = hist[hist > 0]  # remove zeros
    return float(-np.sum(hist * np.log2(hist)))


# ============================================================
# Pipeline: Extract Gaussians and Compute All Metrics
# ============================================================

def extract_gaussians_from_checkpoint(
    checkpoint_name: str,
    frame_indices: list,
    data_dir: str = "/home/joon/data/preprocessed/FaceLift_mouse/M5",
    device: str = "cuda",
) -> list:
    """
    Load checkpoint, run inference on frames, extract Gaussian parameters.

    Returns:
        List of dicts with 'scales', 'opacities', 'positions', 'alpha_maps'
    """
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference, load_sample_data
    from mouse_extensions.visualization import render_opencv_cam

    ckpt_base = Path("checkpoints/gslrm") / checkpoint_name
    config_path = ckpt_base / "config.yaml"

    pipeline = GSLRMInference(
        config_path=str(config_path),
        checkpoint_path=str(ckpt_base),
        device=device,
    )

    # Novel view cameras for alpha rendering
    from mouse_extensions.scripts.novel_view.collect_dataset import generate_novel_cameras
    novel_cameras = generate_novel_cameras()

    results = []
    for frame_idx in frame_indices:
        sample_dir = os.path.join(data_dir, f"{frame_idx:06d}")
        if not os.path.exists(sample_dir):
            print(f"  Skip frame {frame_idx}: not found")
            continue

        images, c2ws, fxfycxcys, index = load_sample_data(
            sample_dir, image_size=384, device=device
        )
        result = pipeline.predict(images, c2ws, fxfycxcys, index)
        gaussians = result["gaussians"][0]  # GaussianModel object

        # Extract parameters via GaussianModel properties
        xyz = gaussians.get_xyz.detach().cpu().numpy()          # [N, 3]
        scales = gaussians.get_scaling.detach().cpu().numpy()   # [N, 3] (activated)
        opacities = gaussians.get_opacity.detach().cpu().squeeze(-1).numpy()  # [N]
        n_gaussians = xyz.shape[0]

        # Render alpha maps for novel views
        alpha_maps = {}
        for view_name, cam in novel_cameras.items():
            c2w_t = torch.tensor(cam["c2w"], dtype=torch.float32, device=device)
            fxfy_t = torch.tensor(cam["fxfycxcy"], dtype=torch.float32, device=device)
            rendered = render_opencv_cam(
                gaussians, height=512, width=512,
                C2W=c2w_t, fxfycxcy=fxfy_t, bg_color=(1.0, 1.0, 1.0),
            )
            if "alpha" in rendered:
                alpha = rendered["alpha"].detach().cpu().squeeze(0).numpy()
            else:
                # Derive from RGB (fallback)
                rgb = rendered["render"].detach().cpu().permute(1, 2, 0).numpy()
                alpha = 1.0 - np.min(rgb, axis=2)  # approximate
            alpha_maps[view_name] = alpha

        results.append({
            "frame_idx": frame_idx,
            "positions": xyz,
            "scales": scales,
            "opacities": opacities,
            "alpha_maps": alpha_maps,
            "n_gaussians": n_gaussians,
        })
        print(f"  Frame {frame_idx}: {n_gaussians} Gaussians extracted")

    return results


def compute_all_metrics(gaussian_data_list: list) -> dict:
    """Compute all metrics averaged over frames."""
    all_aniso = []
    all_iso = []
    all_opacity = []
    all_scale_mag = []
    all_spatial = []
    all_alpha_sparsity = {v: [] for v in ["bottom", "top", "front_low", "side_low"]}
    all_alpha_entropy = {v: [] for v in ["bottom", "top", "front_low", "side_low"]}

    for gd in gaussian_data_list:
        all_aniso.append(compute_anisotropy_ratio(gd["scales"]))
        all_iso.append(compute_isotropy_score(gd["scales"]))
        all_opacity.append(compute_opacity_distribution(gd["opacities"]))
        all_scale_mag.append(compute_scale_magnitude(gd["scales"]))
        all_spatial.append(compute_spatial_outliers(gd["positions"]))

        for view_name, alpha_map in gd["alpha_maps"].items():
            if view_name in all_alpha_sparsity:
                all_alpha_sparsity[view_name].append(compute_alpha_sparsity(alpha_map))
                all_alpha_entropy[view_name].append(compute_alpha_entropy(alpha_map))

    # Average across frames
    def avg_dict(dicts, keys):
        return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

    return {
        "anisotropy_ratio": avg_dict(all_aniso, ["mean", "median", "p95", "p99", "pct_gt_10", "pct_gt_50", "pct_gt_100"]),
        "isotropy_score": avg_dict(all_iso, ["mean", "median", "p95"]),
        "opacity_distribution": avg_dict(all_opacity, ["ambiguity_pct", "low_opacity_pct", "high_opacity_pct", "mean_opacity"]),
        "scale_magnitude": avg_dict(all_scale_mag, ["mean", "median", "p95"]),
        "spatial_outliers": avg_dict(all_spatial, ["outlier_pct"]),
        "n_gaussians": int(np.mean([gd["n_gaussians"] for gd in gaussian_data_list])),
        "alpha_sparsity": {v: float(np.mean(vals)) if vals else 0 for v, vals in all_alpha_sparsity.items()},
        "alpha_entropy": {v: float(np.mean(vals)) if vals else 0 for v, vals in all_alpha_entropy.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Gaussian quality metrics (literature-based)")
    parser.add_argument(
        "--checkpoints", nargs="+",
        default=[
            "base_uniform_v2_6view_v2",
            "base_uniform_v2_4view_v2",
            "base_uniform_v2_4view_alpha05_v3",
            "base_uniform_v2_4view_alpha10_v3",
        ],
    )
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 500, 1000])
    parser.add_argument("--output_dir", default="outputs/reports/gaussian_quality_metrics")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for ckpt_name in args.checkpoints:
        label = ckpt_name.replace("base_uniform_v2_", "")
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"{'='*60}")

        gaussian_data = extract_gaussians_from_checkpoint(
            ckpt_name, args.frames, device=args.device,
        )
        if not gaussian_data:
            print(f"  No data for {label}")
            continue

        metrics = compute_all_metrics(gaussian_data)
        all_results[label] = metrics

    # Print comparison table
    print("\n" + "=" * 100)
    print("GAUSSIAN QUALITY METRICS COMPARISON (Literature-Based)")
    print("=" * 100)

    print(f"\n{'Metric':<30}", end="")
    for label in all_results:
        print(f" {label:>18}", end="")
    print()
    print("-" * (30 + 19 * len(all_results)))

    rows = [
        ("Aniso Ratio (mean)", lambda m: m["anisotropy_ratio"]["mean"]),
        ("Aniso Ratio (p95)", lambda m: m["anisotropy_ratio"]["p95"]),
        ("Aniso Ratio (p99)", lambda m: m["anisotropy_ratio"]["p99"]),
        ("% Aniso > 10", lambda m: m["anisotropy_ratio"]["pct_gt_10"]),
        ("% Aniso > 50", lambda m: m["anisotropy_ratio"]["pct_gt_50"]),
        ("Isotropy Score (mean)", lambda m: m["isotropy_score"]["mean"]),
        ("Opacity Ambiguity %", lambda m: m["opacity_distribution"]["ambiguity_pct"]),
        ("Mean Opacity", lambda m: m["opacity_distribution"]["mean_opacity"]),
        ("Scale Magnitude (mean)", lambda m: m["scale_magnitude"]["mean"]),
        ("Spatial Outlier %", lambda m: m["spatial_outliers"]["outlier_pct"]),
        ("N Gaussians", lambda m: m["n_gaussians"]),
        ("Alpha Sparsity (bottom)", lambda m: m["alpha_sparsity"].get("bottom", 0)),
        ("Alpha Sparsity (top)", lambda m: m["alpha_sparsity"].get("top", 0)),
        ("Alpha Entropy (bottom)", lambda m: m["alpha_entropy"].get("bottom", 0)),
        ("Alpha Entropy (top)", lambda m: m["alpha_entropy"].get("top", 0)),
    ]

    for row_name, getter in rows:
        print(f"{row_name:<30}", end="")
        for label, metrics in all_results.items():
            val = getter(metrics)
            if isinstance(val, int):
                print(f" {val:>18d}", end="")
            else:
                print(f" {val:>18.4f}", end="")
        print()

    # Save JSON
    report_path = os.path.join(args.output_dir, "gaussian_quality_comparison.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
