"""
Artifact Metrics for Novel View Quality Evaluation

Quantifies "white elongated thin Gaussian artifacts" (floaters) in
extrapolated novel views without ground truth images.

Key insight: Use rendered alpha map to separate foreground from artifacts
via connected components analysis, then compute morphology-aware metrics.

Metrics:
    FAS: Floater Area Score — % of artifact pixels
    EFS: Elongation-weighted Floater Score — penalizes thin streaks
    OAS: Opacity-weighted Area Score — penalizes high-opacity artifacts
    CAS: Composite Artifact Score — weighted combination

Literature basis:
    - Mip-NeRF 360 (CVPR 2022): dist-loss for floater reduction
    - Compact-3DGS (2024): anisotropy regularization
    - StableGS (2025): entropy-based opacity regularization

Usage:
    python -m mouse_extensions.scripts.eval.artifact_metrics \\
        --render_dir outputs/datasets/novel_view_alpha_comparison \\
        --configs baseline_4v alpha05 alpha10 baseline_6v \\
        --output_dir outputs/reports/artifact_metrics

Author: FaceLift Mouse Extensions
Date: 2026-03-16
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class ArtifactMetrics:
    """Per-image artifact metrics."""

    fas: float = 0.0  # Floater Area Score
    efs: float = 0.0  # Elongation-weighted Floater Score
    oas: float = 0.0  # Opacity-weighted Area Score
    cas: float = 0.0  # Composite Artifact Score
    n_components: int = 0  # Number of artifact components
    max_eccentricity: float = 0.0  # Max eccentricity of any component
    artifact_pixel_count: int = 0
    foreground_pixel_count: int = 0
    total_pixels: int = 0

    def to_dict(self) -> dict:
        return {k: round(v, 6) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


def compute_artifact_metrics(
    rgb_image: np.ndarray,
    alpha_map: Optional[np.ndarray] = None,
    bg_threshold: float = 0.05,
    alpha_fg_threshold: float = 0.95,
    alpha_any_threshold: float = 0.01,
    min_component_area: int = 5,
    cas_weights: tuple = (0.2, 0.5, 0.3),
) -> ArtifactMetrics:
    """
    Compute artifact metrics for a single rendered image.

    Uses alpha map (if available) or RGB deviation from white background
    to separate foreground from artifact pixels.

    Args:
        rgb_image: RGB image [H, W, 3], range [0, 1]
        alpha_map: Rendered alpha [H, W], range [0, 1]. If None, derived from RGB.
        bg_threshold: Deviation from white to consider non-background
        alpha_fg_threshold: Alpha threshold for core foreground
        alpha_any_threshold: Alpha threshold for any non-background
        min_component_area: Minimum pixels for a component to count
        cas_weights: (w_fas, w_efs, w_oas) for composite score

    Returns:
        ArtifactMetrics dataclass
    """
    H, W = rgb_image.shape[:2]
    total_pixels = H * W

    # Step 1: Create initial non-background mask
    if alpha_map is not None:
        initial_mask = (alpha_map > alpha_any_threshold).astype(np.uint8)
    else:
        # Derive from RGB: deviation from white
        deviation = np.max(1.0 - rgb_image, axis=2)
        initial_mask = (deviation > bg_threshold).astype(np.uint8)

    # Step 2: Connected components analysis
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        initial_mask, connectivity=8
    )

    if n_labels <= 1:
        return ArtifactMetrics(total_pixels=total_pixels)

    # Step 3: Identify foreground = largest connected component
    # (label 0 = background, skip it)
    areas = stats[1:, cv2.CC_STAT_AREA]
    fg_label = np.argmax(areas) + 1
    foreground_mask = (labels == fg_label).astype(np.uint8)

    # Step 4: Morphological opening to remove thin bridges
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )

    # Step 5: Artifact mask = non-background minus foreground
    artifact_mask = initial_mask & (~foreground_mask.astype(bool)).astype(np.uint8)

    # Step 6: Analyze artifact components
    art_n_labels, art_labels, art_stats, _ = cv2.connectedComponentsWithStats(
        artifact_mask, connectivity=8
    )

    artifact_pixel_count = int(np.sum(artifact_mask))
    foreground_pixel_count = int(np.sum(foreground_mask))

    # FAS: Floater Area Score
    fas = artifact_pixel_count / total_pixels

    # EFS: Elongation-weighted Floater Score
    efs = 0.0
    max_ecc = 0.0
    n_valid_components = 0

    if art_n_labels > 1:
        for i in range(1, art_n_labels):
            area = art_stats[i, cv2.CC_STAT_AREA]
            if area < min_component_area:
                continue
            n_valid_components += 1

            # Compute eccentricity from component moments
            component_mask = (art_labels == i).astype(np.uint8)
            moments = cv2.moments(component_mask)

            if moments["m00"] == 0:
                continue

            # Second-order central moments for eccentricity
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            mu11 = moments["mu11"] / moments["m00"]

            # Eigenvalues of inertia tensor
            delta = np.sqrt(4 * mu11**2 + (mu20 - mu02) ** 2)
            lambda1 = (mu20 + mu02 + delta) / 2
            lambda2 = max((mu20 + mu02 - delta) / 2, 1e-10)

            # Eccentricity: 0 = circle, → 1 = line
            eccentricity = np.sqrt(1 - lambda2 / max(lambda1, 1e-10))
            eccentricity = min(eccentricity, 1.0)
            max_ecc = max(max_ecc, eccentricity)

            efs += area * (eccentricity**2)

    efs /= total_pixels

    # OAS: Opacity-weighted Area Score
    if alpha_map is not None:
        oas = float(np.sum(alpha_map[artifact_mask > 0])) / total_pixels
    else:
        # Use RGB deviation as proxy for opacity
        deviation = np.max(1.0 - rgb_image, axis=2)
        oas = float(np.sum(deviation[artifact_mask > 0])) / total_pixels

    # CAS: Composite Artifact Score
    w1, w2, w3 = cas_weights
    cas = (w1 * fas + w2 * efs + w3 * oas) * 100  # Scale to percentage

    return ArtifactMetrics(
        fas=fas,
        efs=efs,
        oas=oas,
        cas=cas,
        n_components=n_valid_components,
        max_eccentricity=max_ecc,
        artifact_pixel_count=artifact_pixel_count,
        foreground_pixel_count=foreground_pixel_count,
        total_pixels=total_pixels,
    )


def compute_batch_metrics(
    image_dir: str,
    alpha_dir: Optional[str] = None,
    views: Optional[list] = None,
    frame_indices: Optional[list] = None,
) -> dict:
    """
    Compute artifact metrics for a batch of images.

    Args:
        image_dir: Directory with view subdirectories containing PNGs
        alpha_dir: Optional directory with alpha maps
        views: List of view names (default: auto-detect)
        frame_indices: List of frame indices (default: all)

    Returns:
        Dict with per-view and overall metrics
    """
    if views is None:
        views = [
            d for d in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, d))
        ]

    results = {"per_view": {}, "overall": {}}

    all_metrics = []
    for view in sorted(views):
        view_dir = os.path.join(image_dir, view)
        if not os.path.isdir(view_dir):
            continue

        view_metrics = []
        for fname in sorted(os.listdir(view_dir)):
            if not fname.endswith(".png"):
                continue
            frame_idx = int(fname.replace(".png", ""))
            if frame_indices and frame_idx not in frame_indices:
                continue

            img_path = os.path.join(view_dir, fname)
            img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0

            # Load alpha if available
            alpha = None
            if alpha_dir:
                alpha_path = os.path.join(alpha_dir, view, fname)
                if os.path.exists(alpha_path):
                    alpha = np.array(Image.open(alpha_path).convert("L")).astype(np.float32) / 255.0

            metrics = compute_artifact_metrics(img, alpha)
            view_metrics.append(metrics)
            all_metrics.append(metrics)

        if view_metrics:
            results["per_view"][view] = {
                "fas": np.mean([m.fas for m in view_metrics]),
                "efs": np.mean([m.efs for m in view_metrics]),
                "oas": np.mean([m.oas for m in view_metrics]),
                "cas": np.mean([m.cas for m in view_metrics]),
                "n_frames": len(view_metrics),
                "mean_components": np.mean([m.n_components for m in view_metrics]),
                "max_eccentricity": max(m.max_eccentricity for m in view_metrics),
            }

    if all_metrics:
        results["overall"] = {
            "fas": np.mean([m.fas for m in all_metrics]),
            "efs": np.mean([m.efs for m in all_metrics]),
            "oas": np.mean([m.oas for m in all_metrics]),
            "cas": np.mean([m.cas for m in all_metrics]),
            "n_frames": len(all_metrics),
        }

    return results


def create_comparison_report(
    render_base: str,
    config_names: list,
    output_dir: str,
    views: Optional[list] = None,
    frame_indices: Optional[list] = None,
):
    """
    Create a comparison report across multiple configurations.

    Args:
        render_base: Base directory containing config subdirectories
        config_names: List of config directory names
        output_dir: Where to save the report
        views: View names to evaluate
        frame_indices: Frame indices to evaluate
    """
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    for cfg in config_names:
        tier_dir = os.path.join(render_base, cfg, "mouse_m5t2", "tier0_raw")
        if not os.path.exists(tier_dir):
            print(f"  SKIP {cfg}: {tier_dir} not found")
            continue
        print(f"  Processing {cfg}...")
        all_results[cfg] = compute_batch_metrics(
            tier_dir, views=views, frame_indices=frame_indices,
        )

    # Print comparison table
    if views is None:
        views = ["bottom", "top", "front_low", "side_low"]

    print("\n" + "=" * 90)
    print("ARTIFACT METRICS COMPARISON (CAS = Composite Artifact Score, lower = better)")
    print("=" * 90)

    header = f"{'Config':<15}"
    for v in views:
        header += f" {v:>12}"
    header += f" {'OVERALL':>12}"
    print(header)
    print("-" * 90)

    for cfg in config_names:
        if cfg not in all_results:
            continue
        r = all_results[cfg]
        row = f"{cfg:<15}"
        for v in views:
            if v in r["per_view"]:
                cas = r["per_view"][v]["cas"]
                row += f" {cas:>11.4f}%"
            else:
                row += f" {'N/A':>12}"
        if "overall" in r and r["overall"]:
            row += f" {r['overall']['cas']:>11.4f}%"
        print(row)

    # Detailed per-metric table
    print("\n\nDETAILED METRICS (Overall averages)")
    print("-" * 80)
    print(f"{'Config':<15} {'FAS':>10} {'EFS':>10} {'OAS':>10} {'CAS':>10} {'Components':>12}")
    print("-" * 80)
    for cfg in config_names:
        if cfg not in all_results:
            continue
        o = all_results[cfg].get("overall", {})
        if not o:
            continue
        print(
            f"{cfg:<15} {o['fas']*100:>9.4f}% {o['efs']*100:>9.4f}% "
            f"{o['oas']*100:>9.4f}% {o['cas']:>9.4f}% {o['n_frames']:>12}"
        )

    # Save JSON
    report_path = os.path.join(output_dir, "artifact_metrics_comparison.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nReport saved: {report_path}")

    return all_results


def create_artifact_visualization(
    rgb_image: np.ndarray,
    alpha_map: Optional[np.ndarray] = None,
    metrics: Optional[ArtifactMetrics] = None,
) -> np.ndarray:
    """
    Create a visualization showing foreground, artifacts, and metrics overlay.

    Returns:
        Visualization image [H, W*3, 3] with (original, artifact_mask, overlay)
    """
    H, W = rgb_image.shape[:2]

    if metrics is None:
        metrics = compute_artifact_metrics(rgb_image, alpha_map)

    # Recompute masks for visualization
    if alpha_map is not None:
        initial_mask = (alpha_map > 0.01).astype(np.uint8)
    else:
        deviation = np.max(1.0 - rgb_image, axis=2)
        initial_mask = (deviation > 0.05).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        initial_mask, connectivity=8
    )
    if n_labels <= 1:
        blank = np.ones((H, W * 3, 3), dtype=np.uint8) * 255
        blank[:, :W] = (rgb_image * 255).astype(np.uint8)
        return blank

    areas = stats[1:, cv2.CC_STAT_AREA]
    fg_label = np.argmax(areas) + 1
    foreground_mask = (labels == fg_label).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    artifact_mask = initial_mask & (~foreground_mask.astype(bool)).astype(np.uint8)

    # Panel 1: Original
    panel1 = (rgb_image * 255).astype(np.uint8)

    # Panel 2: Mask visualization (green=foreground, red=artifact)
    panel2 = np.ones((H, W, 3), dtype=np.uint8) * 255
    panel2[foreground_mask > 0] = [0, 200, 0]  # green
    panel2[artifact_mask > 0] = [255, 0, 0]  # red

    # Panel 3: Overlay (original with red artifact highlight)
    panel3 = (rgb_image * 255).astype(np.uint8).copy()
    artifact_overlay = artifact_mask > 0
    panel3[artifact_overlay, 0] = 255  # Red channel
    panel3[artifact_overlay, 1] = 0
    panel3[artifact_overlay, 2] = 0

    # Add text overlay with metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel3, f"CAS:{metrics.cas:.3f}%", (5, 20), font, 0.5, (255, 0, 0), 1)
    cv2.putText(panel3, f"N:{metrics.n_components}", (5, 40), font, 0.5, (255, 0, 0), 1)

    return np.concatenate([panel1, panel2, panel3], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Artifact metrics for novel view evaluation")
    parser.add_argument(
        "--render_dir",
        default="outputs/datasets/novel_view_alpha_comparison",
        help="Base directory with config subdirs",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline_4v", "alpha05", "alpha10", "baseline_6v"],
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/reports/artifact_metrics",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["bottom", "top", "front_low", "side_low"],
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate artifact visualization images",
    )
    args = parser.parse_args()

    results = create_comparison_report(
        render_base=args.render_dir,
        config_names=args.configs,
        output_dir=args.output_dir,
        views=args.views,
        frame_indices=args.frames,
    )

    # Generate visualizations if requested
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        for cfg in args.configs:
            tier_dir = os.path.join(
                args.render_dir, cfg, "mouse_m5t2", "tier0_raw"
            )
            for view in args.views:
                view_dir = os.path.join(tier_dir, view)
                if not os.path.isdir(view_dir):
                    continue
                for fname in sorted(os.listdir(view_dir))[:3]:  # First 3 frames
                    if not fname.endswith(".png"):
                        continue
                    img = np.array(Image.open(
                        os.path.join(view_dir, fname)
                    ).convert("RGB")).astype(np.float32) / 255.0
                    viz = create_artifact_visualization(img)
                    out_path = os.path.join(
                        viz_dir, f"{cfg}_{view}_{fname}"
                    )
                    Image.fromarray(viz).save(out_path)
        print(f"Visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
