"""Gaussian parameter distribution analysis: histograms + modality tests.

Loads raw Gaussian NPZ files, computes per-body-part distributions of
opacity, scale, and rotation, runs modality tests (Hartigan's Dip Test,
GMM BIC), and saves comprehensive visualizations.

Body-part assignment: 3D nearest-bone-segment distance (view-independent).

Usage on gpu03:
    python -m mouse_extensions.behavior.analyze_gaussian_distributions \
        --frame-idx 0 500 1000 2000 \
        --output-dir outputs/analysis/mouse/filtering/gaussian_distributions
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial.distance import cdist

from mouse_extensions.behavior.view_projected_filtering import (
    KP_NAMES, BODY_PARTS, BODY_PART_COLORS, load_gaussians, load_keypoints_gslrm,
)

# Bone segments for distance-based assignment (pairs of keypoint indices)
BONE_SEGMENTS = [
    (2, 0), (2, 1), (2, 3),        # nose-ears, nose-neck
    (3, 4), (4, 5), (5, 6), (6, 7),  # neck-body-tail chain
    (3, 11), (11, 10), (10, 8), (8, 9),   # left arm
    (3, 15), (15, 14), (14, 12), (12, 13),  # right arm
    (4, 18), (18, 17), (17, 16),   # left leg
    (4, 21), (21, 20), (20, 19),   # right leg
]

# Map each keypoint to its body part for bone→part assignment
KP_TO_PART = {}
for part_name, kp_indices in BODY_PARTS.items():
    for ki in kp_indices:
        KP_TO_PART[ki] = part_name


def point_to_segment_distance(points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> np.ndarray:
    """Compute distance from points to a line segment [a, b].

    Args:
        points: (N, 3)
        seg_a: (3,) segment start
        seg_b: (3,) segment end

    Returns:
        distances: (N,) distance to segment
    """
    ab = seg_b - seg_a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return np.linalg.norm(points - seg_a, axis=1)

    ap = points - seg_a
    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    proj = seg_a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def assign_gaussians_to_bodyparts_3d(
    xyz: np.ndarray,
    keypoints_3d: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Assign Gaussians to body parts via nearest bone segment in 3D.

    Returns:
        dict of {part_name: boolean mask of shape (N,)}
    """
    N = len(xyz)

    # Compute distance to each bone segment
    # Track (distance, part_name) per Gaussian
    best_dist = np.full(N, np.inf)
    best_part = np.empty(N, dtype=object)

    for ki_a, ki_b in BONE_SEGMENTS:
        seg_a = keypoints_3d[ki_a]
        seg_b = keypoints_3d[ki_b]
        dists = point_to_segment_distance(xyz, seg_a, seg_b)

        # Determine which body part this bone belongs to
        # Use the part of the first keypoint (or shared part if both same)
        part_a = KP_TO_PART.get(ki_a)
        part_b = KP_TO_PART.get(ki_b)
        # Prefer the part that is NOT torso for limb bones
        if part_a == part_b:
            part = part_a
        elif part_a == "torso":
            part = part_b
        elif part_b == "torso":
            part = part_a
        else:
            part = part_a  # default to first

        if part is None:
            continue

        closer = dists < best_dist
        best_dist[closer] = dists[closer]
        best_part[closer] = part

    # Build masks
    part_masks = {}
    for part_name in BODY_PARTS:
        part_masks[part_name] = (best_part == part_name)

    return part_masks


def hartigan_dip_test(data: np.ndarray, n_boot: int = 1000) -> Tuple[float, float]:
    """Hartigan's dip test for unimodality.

    Returns:
        (dip_statistic, p_value)
    """
    try:
        import diptest
        dip, pval = diptest.diptest(data)
        return float(dip), float(pval)
    except ImportError:
        # Fallback: simple bootstrap approximation
        data_sorted = np.sort(data)
        n = len(data_sorted)
        if n < 10:
            return 0.0, 1.0

        # Compute empirical CDF
        ecdf = np.arange(1, n + 1) / n
        # Greatest convex minorant / least concave majorant approximation
        # Simplified: use uniform CDF as reference
        uniform_cdf = (data_sorted - data_sorted[0]) / (data_sorted[-1] - data_sorted[0] + 1e-10)
        dip = np.max(np.abs(ecdf - uniform_cdf)) / 2

        # Bootstrap p-value
        dips_boot = []
        for _ in range(n_boot):
            boot = np.sort(np.random.uniform(0, 1, n))
            boot_ecdf = np.arange(1, n + 1) / n
            dips_boot.append(np.max(np.abs(boot_ecdf - boot)) / 2)
        pval = np.mean(np.array(dips_boot) >= dip)
        return float(dip), float(pval)


def gmm_bic_modality(data: np.ndarray, max_components: int = 5) -> Dict:
    """Fit GMM with varying components, select best by BIC.

    Returns:
        dict with best_n_components, bic_values, means, weights
    """
    from sklearn.mixture import GaussianMixture

    data_2d = data.reshape(-1, 1)
    bics = []
    models = []

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200)
        gmm.fit(data_2d)
        bics.append(gmm.bic(data_2d))
        models.append(gmm)

    best_k = int(np.argmin(bics) + 1)
    best_model = models[best_k - 1]

    return {
        "best_n_components": best_k,
        "bic_values": [float(b) for b in bics],
        "means": best_model.means_.flatten().tolist(),
        "weights": best_model.weights_.tolist(),
        "covariances": best_model.covariances_.flatten().tolist(),
        "is_multimodal": best_k > 1,
    }


def analyze_parameter(
    data: np.ndarray,
    param_name: str,
    subsample: int = 50000,
) -> Dict:
    """Analyze a single parameter distribution: stats + modality tests.

    Args:
        data: 1D array of parameter values
        param_name: name for reporting
        subsample: max samples for modality tests (performance)

    Returns:
        dict with statistics and test results
    """
    if len(data) == 0:
        return {"param": param_name, "n": 0, "error": "no data"}

    stats = {
        "param": param_name,
        "n": int(len(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "skewness": float(_skewness(data)),
        "kurtosis": float(_kurtosis(data)),
    }

    # Subsample for expensive tests
    if len(data) > subsample:
        rng = np.random.default_rng(42)
        data_sub = rng.choice(data, subsample, replace=False)
    else:
        data_sub = data

    # Modality tests
    dip, dip_p = hartigan_dip_test(data_sub)
    stats["dip_statistic"] = dip
    stats["dip_pvalue"] = dip_p
    stats["dip_unimodal"] = dip_p > 0.05

    gmm_result = gmm_bic_modality(data_sub)
    stats["gmm"] = gmm_result
    stats["is_multimodal"] = gmm_result["is_multimodal"] or (dip_p < 0.05)
    stats["modality_verdict"] = (
        f"{'Multimodal' if stats['is_multimodal'] else 'Unimodal'} "
        f"(Dip p={dip_p:.4f}, GMM best_k={gmm_result['best_n_components']})"
    )

    return stats


def _skewness(data: np.ndarray) -> float:
    m = np.mean(data)
    s = np.std(data)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((data - m) / s) ** 3))


def _kurtosis(data: np.ndarray) -> float:
    m = np.mean(data)
    s = np.std(data)
    if s < 1e-10:
        return 0.0
    return float(np.mean(((data - m) / s) ** 4) - 3.0)


def plot_distributions(
    gauss: Dict[str, np.ndarray],
    part_masks: Dict[str, np.ndarray],
    all_stats: Dict,
    frame_idx: int,
    output_dir: Path,
):
    """Create comprehensive distribution visualization.

    Layout: 4 rows × 3 cols
    Row 1: Global opacity, scale_magnitude, scale_anisotropy
    Row 2: Per-body-part opacity violin, scale violin, opacity KDE overlay
    Row 3: Per-body-part histograms (face, torso, tail)
    Row 4: BIC curves + modality summary table
    """
    opacity = gauss["opacity"]
    scale = gauss["scale"]
    scale_mag = np.sqrt((scale ** 2).sum(axis=1))  # Frobenius norm
    scale_aniso = scale.max(axis=1) / (scale.min(axis=1) + 1e-10)

    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.3)

    # === Row 1: Global histograms with KDE ===
    params_global = [
        ("Opacity", opacity, (0, 1)),
        ("Scale Magnitude", scale_mag, None),
        ("Scale Anisotropy", np.clip(scale_aniso, 0, 50), None),
    ]

    for col, (name, data, xlim) in enumerate(params_global):
        ax = fig.add_subplot(gs[0, col])
        ax.hist(data, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        # KDE overlay
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(data[::max(1, len(data) // 10000)])
            x_range = np.linspace(data.min(), np.percentile(data, 99), 200)
            ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
        except Exception:
            pass
        if xlim:
            ax.set_xlim(xlim)
        ax.set_title(f"Global {name} (N={len(data):,})", fontsize=11)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # === Row 2: Violin plots per body part ===
    part_names = list(BODY_PARTS.keys())
    part_colors = [BODY_PART_COLORS[p] for p in part_names]

    for col, (name, data_all) in enumerate([
        ("Opacity", opacity),
        ("Scale Magnitude", scale_mag),
        ("Scale Anisotropy", np.clip(scale_aniso, 0, 20)),
    ]):
        ax = fig.add_subplot(gs[1, col])
        violin_data = []
        for pn in part_names:
            mask = part_masks[pn]
            d = data_all[mask]
            if len(d) > 5000:
                d = np.random.default_rng(42).choice(d, 5000, replace=False)
            violin_data.append(d)

        vp = ax.violinplot(violin_data, showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(part_colors[i])
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(part_names) + 1))
        ax.set_xticklabels(part_names, rotation=30, fontsize=9)
        ax.set_title(f"{name} by Body Part", fontsize=11)

    # === Row 3: Per-body-part KDE overlays for key params ===
    from scipy.stats import gaussian_kde

    highlight_parts = ["face", "torso", "tail"]
    for col, part_name in enumerate(highlight_parts):
        ax = fig.add_subplot(gs[2, col])
        mask = part_masks[part_name]
        part_opacity = opacity[mask]
        part_scale = scale_mag[mask]

        if len(part_opacity) > 20:
            # Opacity histogram
            ax.hist(part_opacity, bins=60, density=True, alpha=0.5, color=BODY_PART_COLORS[part_name],
                    edgecolor="none", label="Opacity")
            try:
                sub = part_opacity[::max(1, len(part_opacity) // 5000)]
                kde_op = gaussian_kde(sub)
                x = np.linspace(0, 1, 200)
                ax.plot(x, kde_op(x), "-", color=BODY_PART_COLORS[part_name], linewidth=2)
            except Exception:
                pass

            # Mark GMM means if multimodal
            key = f"{part_name}_opacity"
            if key in all_stats and all_stats[key].get("is_multimodal"):
                gmm = all_stats[key]["gmm"]
                for mu, w in zip(gmm["means"], gmm["weights"]):
                    ax.axvline(mu, color="red", linestyle="--", linewidth=1.5,
                               label=f"GMM μ={mu:.3f} (w={w:.2f})")

        ax.set_xlim(0, 1)
        ax.set_title(f"{part_name}: Opacity (N={mask.sum():,})", fontsize=11)
        ax.legend(fontsize=7)

    # === Row 4: BIC curves + summary table ===
    # BIC curve for global opacity
    ax_bic = fig.add_subplot(gs[3, 0])
    if "global_opacity" in all_stats and "gmm" in all_stats["global_opacity"]:
        bics = all_stats["global_opacity"]["gmm"]["bic_values"]
        ax_bic.plot(range(1, len(bics) + 1), bics, "o-", color="steelblue", linewidth=2)
        best_k = all_stats["global_opacity"]["gmm"]["best_n_components"]
        ax_bic.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
        ax_bic.set_xlabel("Number of Components")
        ax_bic.set_ylabel("BIC")
        ax_bic.set_title("GMM BIC: Global Opacity")
        ax_bic.legend()

    # Modality summary table
    ax_table = fig.add_subplot(gs[3, 1:])
    ax_table.axis("off")
    table_data = []
    table_cols = ["Parameter", "N", "Mean", "Std", "Dip p", "GMM K", "Verdict"]
    for key, stats in all_stats.items():
        if "n" not in stats or stats["n"] == 0:
            continue
        table_data.append([
            key,
            f"{stats['n']:,}",
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            f"{stats.get('dip_pvalue', 'N/A'):.4f}" if isinstance(stats.get('dip_pvalue'), float) else "N/A",
            str(stats.get("gmm", {}).get("best_n_components", "N/A")),
            "Multi" if stats.get("is_multimodal") else "Uni",
        ])

    if table_data:
        table = ax_table.table(
            cellText=table_data, colLabels=table_cols,
            loc="center", cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)
        # Color multimodal rows
        for i, row in enumerate(table_data):
            if row[-1] == "Multi":
                for j in range(len(table_cols)):
                    table[i + 1, j].set_facecolor("#FFE0E0")
    ax_table.set_title("Modality Summary", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Gaussian Parameter Distributions — Frame {frame_idx}\n"
        f"Body-part assignment: 3D nearest bone segment",
        fontsize=14, fontweight="bold",
    )

    out_path = output_dir / f"frame_{frame_idx:06d}_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return str(out_path)


def process_frame(
    frame_idx: int,
    gauss_dir: str,
    kp_path: str,
    output_dir: Path,
) -> Dict:
    """Analyze Gaussian distributions for a single frame."""
    gauss_path = Path(gauss_dir) / f"{frame_idx:06d}.npz"
    if not gauss_path.exists():
        print(f"  Frame {frame_idx}: NPZ not found at {gauss_path}")
        return {}

    gauss = load_gaussians(str(gauss_path))
    print(f"  Frame {frame_idx}: {gauss['n_gaussians']:,} Gaussians loaded")

    # Load keypoints
    kp_gslrm = load_keypoints_gslrm(kp_path, frame_idx)

    # 3D bone-based body part assignment
    part_masks = assign_gaussians_to_bodyparts_3d(gauss["xyz"], kp_gslrm)
    for pn, mask in part_masks.items():
        print(f"    {pn}: {mask.sum():,} Gaussians")

    # Derived parameters
    opacity = gauss["opacity"]
    scale = gauss["scale"]
    scale_mag = np.sqrt((scale ** 2).sum(axis=1))
    scale_aniso = scale.max(axis=1) / (scale.min(axis=1) + 1e-10)

    # Analyze distributions
    all_stats = {}

    # Global
    print("  Analyzing global distributions...")
    all_stats["global_opacity"] = analyze_parameter(opacity, "global_opacity")
    all_stats["global_scale_mag"] = analyze_parameter(scale_mag, "global_scale_mag")
    all_stats["global_scale_aniso"] = analyze_parameter(np.clip(scale_aniso, 0, 50), "global_scale_aniso")

    # Per body part
    for part_name in BODY_PARTS:
        mask = part_masks[part_name]
        if mask.sum() < 10:
            continue
        print(f"  Analyzing {part_name}...")
        all_stats[f"{part_name}_opacity"] = analyze_parameter(opacity[mask], f"{part_name}_opacity")
        all_stats[f"{part_name}_scale_mag"] = analyze_parameter(scale_mag[mask], f"{part_name}_scale_mag")

    # Visualize
    plot_path = plot_distributions(gauss, part_masks, all_stats, frame_idx, output_dir)

    # Add assignment counts to stats
    assignment_counts = {pn: int(mask.sum()) for pn, mask in part_masks.items()}
    all_stats["_assignment_counts"] = assignment_counts
    all_stats["_total_gaussians"] = gauss["n_gaussians"]

    return all_stats


def main():
    from mouse_extensions.behavior.paths import GAUSSIANS_RAW_DIR, GPU03_KEYPOINTS

    parser = argparse.ArgumentParser(description="Gaussian Parameter Distribution Analysis")
    parser.add_argument("--frame-idx", nargs="+", type=int, default=[0, 500, 1000, 2000])
    parser.add_argument("--gauss-dir", default=str(GAUSSIANS_RAW_DIR))
    parser.add_argument("--kp-path", default=GPU03_KEYPOINTS)
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/filtering/gaussian_distributions")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for fi in args.frame_idx:
        print(f"\n{'='*60}")
        print(f"Frame {fi}")
        print(f"{'='*60}")
        stats = process_frame(fi, args.gauss_dir, args.kp_path, output_dir)
        if stats:
            all_results[str(fi)] = stats

    # Save JSON summary
    summary_path = output_dir / "distribution_analysis.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON summary saved to {summary_path}")

    # Print modality summary
    print(f"\n{'='*70}")
    print("MODALITY SUMMARY")
    print(f"{'='*70}")
    for frame_key, stats in all_results.items():
        print(f"\nFrame {frame_key}:")
        for key, val in stats.items():
            if key.startswith("_"):
                continue
            verdict = val.get("modality_verdict", "N/A")
            print(f"  {key:30s} → {verdict}")


if __name__ == "__main__":
    main()
