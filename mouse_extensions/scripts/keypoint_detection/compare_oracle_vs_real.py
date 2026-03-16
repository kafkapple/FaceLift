"""Compare oracle triangulation vs neural detector triangulation.

Loads results from the oracle saturation analysis (Phase 2) and neural
detector pipeline (Phase D), estimates effective noise sigma, and
generates comparison plots.

Usage:
    python compare_oracle_vs_real.py \
        --oracle_path ~/outputs/triangulation/saturation_analysis.json \
        --neural_path ~/outputs/neural_triangulation/results/neural_results.json \
        --output_dir ~/outputs/neural_triangulation/comparison
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINT_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle",
    "tail_root", "tail_middle", "tail_end",
    "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip",
    "R_foot", "R_knee", "R_hip",
]

JOINT_GROUPS = {
    "Head": [0, 1, 2],
    "Spine": [3, 4, 5],
    "Tail": [6, 7],
    "Front Legs": [8, 9, 10, 11, 12, 13, 14, 15],
    "Hind Legs": [16, 17, 18, 19, 20, 21],
}


def load_oracle_results(oracle_path: str) -> dict:
    """Load oracle saturation analysis results.

    Expected format: {
        "noise_levels": {"0": {...}, "1": {...}, "2": {...}, "5": {...}},
        "view_counts": [6, 12, 24],
        ...
    }
    Each noise level contains results per view count.
    """
    with open(oracle_path) as f:
        data = json.load(f)
    return data


def load_neural_results(neural_dir: str) -> dict:
    """Load neural detection results for multiple view counts.

    Searches for neural_results.json in subdirectories like
    6views/, 12views/, 24views/.
    """
    results = {}

    # Check if neural_dir contains direct results
    direct_path = os.path.join(neural_dir, "neural_results.json")
    if os.path.exists(direct_path):
        with open(direct_path) as f:
            data = json.load(f)
        nv = data.get("num_views", 0)
        results[nv] = data
        return results

    # Search subdirectories
    parent = os.path.dirname(neural_dir) if os.path.isfile(neural_dir) else neural_dir
    for entry in sorted(os.listdir(parent)):
        if "views" in entry:
            path = os.path.join(parent, entry, "results", "neural_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                nv = int(entry.replace("views", ""))
                results[nv] = data

    return results


def estimate_effective_sigma(oracle_data: dict, neural_mpjpe: float, view_count: int) -> float:
    """Estimate the detector's effective noise sigma by interpolating
    on the oracle MPJPE vs sigma curve.

    Uses the oracle results at different noise levels (sigma=0,1,2,5 px)
    and interpolates to find which sigma gives the same MPJPE as the
    neural detector.
    """
    # Extract oracle MPJPE at each sigma for the given view count
    sigmas = []
    mpjpes = []

    view_key = str(view_count)

    for noise_str, noise_data in oracle_data.get("noise_levels", {}).items():
        sigma = float(noise_str)
        if view_key in noise_data:
            mpjpe = noise_data[view_key].get("mpjpe", None)
            if mpjpe is not None:
                sigmas.append(sigma)
                mpjpes.append(mpjpe)

    if len(sigmas) < 2:
        return float("nan")

    # Sort by sigma
    order = np.argsort(sigmas)
    sigmas = np.array(sigmas)[order]
    mpjpes = np.array(mpjpes)[order]

    # Interpolate: find sigma where mpjpe == neural_mpjpe
    if neural_mpjpe <= mpjpes.min():
        return sigmas[np.argmin(mpjpes)]
    if neural_mpjpe >= mpjpes.max():
        # Extrapolate linearly from last two points
        slope = (mpjpes[-1] - mpjpes[-2]) / (sigmas[-1] - sigmas[-2])
        if slope > 0:
            return sigmas[-1] + (neural_mpjpe - mpjpes[-1]) / slope
        return float("nan")

    # Linear interpolation
    return float(np.interp(neural_mpjpe, mpjpes, sigmas))


def plot_oracle_vs_real(oracle_data: dict, neural_data: dict, output_dir: str):
    """Generate the main comparison chart: MPJPE vs view count."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    view_counts = sorted(set(
        list(neural_data.keys()) +
        [int(k) for noise in oracle_data.get("noise_levels", {}).values()
         for k in noise.keys() if k.isdigit()]
    ))

    # Plot oracle curves for each noise level
    noise_levels = sorted(oracle_data.get("noise_levels", {}).keys(), key=float)
    oracle_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(noise_levels)))

    for i, noise_str in enumerate(noise_levels):
        noise_data = oracle_data["noise_levels"][noise_str]
        vcs = []
        mpjpes = []
        for vc in view_counts:
            vc_str = str(vc)
            if vc_str in noise_data and "mpjpe" in noise_data[vc_str]:
                vcs.append(vc)
                mpjpes.append(noise_data[vc_str]["mpjpe"])

        if vcs:
            sigma = float(noise_str)
            label = f"Oracle sigma={sigma:.0f}px" if sigma == int(sigma) else f"Oracle sigma={sigma}px"
            ax.plot(vcs, mpjpes, "o--", color=oracle_colors[i],
                    label=label, markersize=6, alpha=0.8)

    # Plot neural detector results
    neural_vcs = sorted(neural_data.keys())
    neural_mpjpes = [neural_data[vc]["mpjpe_mean"] for vc in neural_vcs]
    neural_stds = [neural_data[vc].get("mpjpe_std", 0) for vc in neural_vcs]

    ax.errorbar(neural_vcs, neural_mpjpes, yerr=neural_stds,
                fmt="s-", color="red", linewidth=2, markersize=8,
                capsize=5, label="Neural Detector (HRNet-w48)",
                zorder=10)

    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("MPJPE (mm)", fontsize=12)
    ax.set_title("Oracle vs Neural Detector: 3D Keypoint Triangulation", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(view_counts)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "oracle_vs_real.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def plot_per_joint_error(neural_data: dict, output_dir: str):
    """Bar chart of per-joint MPJPE."""
    # Use the results with most views
    max_vc = max(neural_data.keys())
    data = neural_data[max_vc]

    per_joint = data.get("per_joint", {})
    if not per_joint:
        print("  Warning: No per_joint data available")
        return

    names = list(per_joint.keys())
    mpjpes = [per_joint[n]["mpjpe_mean"] for n in names]
    det_rates = [per_joint[n]["detection_rate"] for n in names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # MPJPE bar chart
    x = np.arange(len(names))
    colors = ["#e74c3c" if m > np.mean(mpjpes) * 1.5 else
              "#3498db" if m < np.mean(mpjpes) * 0.5 else
              "#2ecc71" for m in mpjpes]

    ax1.bar(x, mpjpes, color=colors, alpha=0.8)
    ax1.axhline(y=np.mean(mpjpes), color="gray", linestyle="--",
                label=f"Mean: {np.mean(mpjpes):.1f} mm")
    ax1.set_ylabel("MPJPE (mm)", fontsize=11)
    ax1.set_title(f"Per-Joint Error ({max_vc} views)", fontsize=13)
    ax1.legend()

    # Detection rate bar chart
    ax2.bar(x, det_rates, color="#9b59b6", alpha=0.7)
    ax2.axhline(y=np.mean(det_rates), color="gray", linestyle="--",
                label=f"Mean: {np.mean(det_rates):.1%}")
    ax2.set_ylabel("Detection Rate", fontsize=11)
    ax2.set_xlabel("Joint", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "per_joint_error.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def plot_domain_gap(neural_data: dict, output_dir: str):
    """Visualize domain gap: detection rate per joint group across view counts."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    group_colors = plt.cm.Set2(np.linspace(0, 1, len(JOINT_GROUPS)))

    for gi, (group_name, joint_ids) in enumerate(JOINT_GROUPS.items()):
        view_counts = sorted(neural_data.keys())
        group_det_rates = []

        for vc in view_counts:
            per_joint = neural_data[vc].get("per_joint", {})
            if per_joint:
                rates = [
                    per_joint[JOINT_NAMES[j]]["detection_rate"]
                    for j in joint_ids
                    if JOINT_NAMES[j] in per_joint
                ]
                group_det_rates.append(np.mean(rates) if rates else 0)
            else:
                group_det_rates.append(0)

        ax.plot(view_counts, group_det_rates, "o-", color=group_colors[gi],
                label=group_name, markersize=6, linewidth=2)

    ax.set_xlabel("Number of Views", fontsize=12)
    ax.set_ylabel("Detection Rate", fontsize=12)
    ax.set_title("Domain Gap Analysis: Detection Rate by Joint Group", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(neural_data.keys()))

    plt.tight_layout()
    save_path = os.path.join(output_dir, "domain_gap_analysis.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved {save_path}")


def generate_report(oracle_data: dict, neural_data: dict, output_dir: str):
    """Generate a markdown summary report."""
    lines = ["# Neural Detection vs Oracle Triangulation Report\n"]
    lines.append(f"Generated from comparison analysis.\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Views | Oracle sigma=0 | Oracle sigma=2 | Oracle sigma=5 | Neural Detector |")
    lines.append("|-------|---------------|---------------|---------------|-----------------|")

    view_counts = sorted(neural_data.keys())
    for vc in view_counts:
        oracle_vals = []
        for sigma in ["0", "2", "5"]:
            noise_data = oracle_data.get("noise_levels", {}).get(sigma, {})
            val = noise_data.get(str(vc), {}).get("mpjpe", "N/A")
            oracle_vals.append(f"{val:.2f}" if isinstance(val, (int, float)) else str(val))

        neural_mpjpe = neural_data[vc]["mpjpe_mean"]
        neural_std = neural_data[vc].get("mpjpe_std", 0)
        lines.append(
            f"| {vc} | {oracle_vals[0]} | {oracle_vals[1]} | {oracle_vals[2]} | "
            f"{neural_mpjpe:.2f} +/- {neural_std:.2f} |"
        )

    # Effective sigma estimates
    lines.append("\n## Effective Noise Sigma Estimates\n")
    lines.append("| Views | Effective sigma (px) |")
    lines.append("|-------|---------------------|")
    for vc in view_counts:
        sigma = estimate_effective_sigma(
            oracle_data, neural_data[vc]["mpjpe_mean"], vc,
        )
        lines.append(f"| {vc} | {sigma:.1f} |")

    # Per-joint analysis
    lines.append("\n## Per-Joint Analysis\n")
    max_vc = max(view_counts)
    per_joint = neural_data[max_vc].get("per_joint", {})
    if per_joint:
        lines.append(f"### {max_vc}-view results\n")
        lines.append("| Joint | MPJPE (mm) | Detection Rate |")
        lines.append("|-------|-----------|----------------|")
        for name in JOINT_NAMES:
            if name in per_joint:
                pj = per_joint[name]
                lines.append(
                    f"| {name} | {pj['mpjpe_mean']:.2f} | {pj['detection_rate']:.1%} |"
                )

    # Key findings
    lines.append("\n## Key Findings\n")
    mean_det = neural_data[max_vc].get("mean_detection_rate", 0)
    lines.append(f"- Mean detection rate: **{mean_det:.1%}**")
    lines.append(f"- Neural MPJPE ({max_vc}v): **{neural_data[max_vc]['mpjpe_mean']:.2f} mm**")

    # Worst joints
    if per_joint:
        sorted_joints = sorted(per_joint.items(), key=lambda x: x[1]["mpjpe_mean"], reverse=True)
        worst = sorted_joints[:3]
        worst_strs = [f"{n} ({v['mpjpe_mean']:.1f}mm)" for n, v in worst]
        lines.append(f"- Worst joints: {', '.join(worst_strs)}")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare oracle vs neural detector triangulation results"
    )
    parser.add_argument("--oracle_path", type=str, required=True,
                        help="Path to saturation_analysis.json")
    parser.add_argument("--neural_path", type=str, required=True,
                        help="Path to neural_results.json or directory")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser(
                            "~/outputs/neural_triangulation/comparison"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading oracle results...")
    oracle_data = load_oracle_results(args.oracle_path)

    print("Loading neural results...")
    if os.path.isfile(args.neural_path):
        neural_data = load_neural_results(os.path.dirname(args.neural_path))
    else:
        neural_data = load_neural_results(args.neural_path)

    if not neural_data:
        print("ERROR: No neural results found!")
        return

    print(f"Neural results for view counts: {sorted(neural_data.keys())}")

    # Effective sigma estimation
    print("\nEffective sigma estimates:")
    for vc in sorted(neural_data.keys()):
        sigma = estimate_effective_sigma(
            oracle_data, neural_data[vc]["mpjpe_mean"], vc,
        )
        print(f"  {vc} views: sigma = {sigma:.1f} px")

    # Generate plots
    print("\nGenerating plots...")
    plot_oracle_vs_real(oracle_data, neural_data, plots_dir)
    plot_per_joint_error(neural_data, plots_dir)
    plot_domain_gap(neural_data, plots_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(oracle_data, neural_data, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
