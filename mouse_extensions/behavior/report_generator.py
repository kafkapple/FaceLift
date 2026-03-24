"""Clustering Report Generator — HTML report with quantitative metrics and qualitative visualizations.

Generates:
- Per-cluster GIF montages (keypoint skeleton animation, RGB multi-view grid if available)
- UMAP scatter plots with cluster coloring
- Ethograms (temporal raster)
- Transition matrices
- Metric comparison tables
- Self-contained HTML report

Usage:
    from mouse_extensions.behavior.report_generator import generate_report
    generate_report(results_dir="outputs/clustering/sparse/results", output_html="report.html")
"""

import base64
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np

from mouse_extensions.constants import SKELETON_BONES as MAMMAL_BONES

# MAMMAL color scheme for behavior visualization (distinct from keypoint overlay)
BODY_PART_COLORS = {
    "head": "#FF6B6B",      # red
    "spine": "#4ECDC4",     # teal
    "tail": "#95E1D3",      # light teal
    "front_L": "#FFD93D",   # yellow
    "front_R": "#6BCB77",   # green
    "hind_L": "#4D96FF",    # blue
    "hind_R": "#9B59B6",    # purple
}

JOINT_BODY_PART = {
    0: "head", 1: "head", 2: "head", 3: "head",
    4: "spine", 5: "tail", 6: "tail", 7: "tail",
    8: "front_L", 9: "front_L", 10: "front_L", 11: "front_L",
    12: "front_R", 13: "front_R", 14: "front_R", 15: "front_R",
    16: "hind_L", 17: "hind_L", 18: "hind_L",
    19: "hind_R", 20: "hind_R", 21: "hind_R",
}


def _make_skeleton_frame(
    keypoints_2d: np.ndarray,
    joint_indices: list[int],
    figsize: tuple = (4, 4),
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> np.ndarray:
    """Render a single skeleton frame as RGB array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("equal")

    # Draw bones
    for i, j in MAMMAL_BONES:
        if i in joint_indices and j in joint_indices:
            ii = joint_indices.index(i)
            jj = joint_indices.index(j)
            part = JOINT_BODY_PART.get(i, "spine")
            color = BODY_PART_COLORS.get(part, "#888888")
            ax.plot(
                [keypoints_2d[ii, 0], keypoints_2d[jj, 0]],
                [keypoints_2d[ii, 1], keypoints_2d[jj, 1]],
                color=color, linewidth=2, alpha=0.8,
            )

    # Draw joints
    for idx, ji in enumerate(joint_indices):
        part = JOINT_BODY_PART.get(ji, "spine")
        color = BODY_PART_COLORS.get(part, "#888888")
        ax.plot(keypoints_2d[idx, 0], keypoints_2d[idx, 1],
                "o", color=color, markersize=5, markeredgecolor="black", markeredgewidth=0.5)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return img


def make_cluster_gif(
    keypoints: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    joint_indices: list[int],
    max_frames: int = 30,
    fps: int = 10,
) -> Optional[bytes]:
    """Create a GIF of skeleton frames for a specific cluster.

    Selects consecutive frames where possible, otherwise nearest frames.
    Returns GIF as bytes, or None if PIL not available.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    # Find frames belonging to this cluster
    cluster_frames = np.where(labels == cluster_id)[0]
    if len(cluster_frames) == 0:
        return None

    # Prefer longest consecutive run
    selected = _select_representative_frames(cluster_frames, max_frames)

    # Project 3D → 2D (top-down: XY plane)
    kp_subset = keypoints[selected][:, joint_indices, :]
    kp_2d = kp_subset[:, :, :2]  # XY projection

    # Compute global limits
    margin = 10
    xlim = (kp_2d[:, :, 0].min() - margin, kp_2d[:, :, 0].max() + margin)
    ylim = (kp_2d[:, :, 1].min() - margin, kp_2d[:, :, 1].max() + margin)

    # Render frames
    frames = []
    for t in range(len(selected)):
        img = _make_skeleton_frame(kp_2d[t], joint_indices, xlim=xlim, ylim=ylim)
        frames.append(Image.fromarray(img))

    if not frames:
        return None

    # Save as GIF
    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=1000 // fps, loop=0,
    )
    return buf.getvalue()


def _select_representative_frames(frame_indices: np.ndarray, max_frames: int) -> np.ndarray:
    """Select representative frames: prefer longest consecutive run, then evenly spaced."""
    if len(frame_indices) <= max_frames:
        return frame_indices

    # Find longest consecutive run
    diffs = np.diff(frame_indices)
    run_starts = np.concatenate([[0], np.where(diffs > 1)[0] + 1])
    run_ends = np.concatenate([np.where(diffs > 1)[0] + 1, [len(frame_indices)]])
    run_lengths = run_ends - run_starts

    best_run_idx = np.argmax(run_lengths)
    best_start = run_starts[best_run_idx]
    best_end = run_ends[best_run_idx]

    if best_end - best_start >= max_frames:
        return frame_indices[best_start:best_start + max_frames]

    # If longest run is shorter than max_frames, use evenly spaced
    indices = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
    return frame_indices[indices]


def generate_report(
    results_dir: str,
    keypoints_path: str = "outputs/clustering/sparse/keypoints_22_3d.npz",
    output_html: Optional[str] = None,
) -> str:
    """Generate comprehensive HTML report from clustering results.

    Args:
        results_dir: Directory containing ablation_summary.json, *_labels.npy, *_embeddings.npy
        keypoints_path: Path to original keypoints NPZ
        output_html: Output path (default: results_dir/report.html)

    Returns:
        Path to generated HTML file
    """
    results_dir = Path(results_dir)
    if output_html is None:
        output_html = str(results_dir / "report.html")

    # Load summary
    summary_path = results_dir / "ablation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary found at {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    # Load keypoints
    data = np.load(keypoints_path, allow_pickle=True)
    keypoints = data["keypoints"]
    keypoint_names = list(data["keypoint_names"])

    # Build HTML
    html_parts = [_html_header(summary)]

    # Metrics comparison table
    html_parts.append(_html_metrics_table(summary))

    # Per-group visualizations
    from mouse_extensions.behavior.run_sparse_ablation import JOINT_GROUPS

    for group_name, group_results in summary.get("results", {}).items():
        joint_indices = JOINT_GROUPS.get(group_name, list(range(22)))
        html_parts.append(f'<h2 id="{group_name}">{group_name} ({len(joint_indices)} joints)</h2>')

        for method_result in group_results:
            method = method_result["method"]
            metrics = method_result.get("metrics", {})

            html_parts.append(f"<h3>{method}</h3>")

            # Metrics card
            if metrics:
                html_parts.append(_html_metric_card(metrics))

            # Load labels and embeddings
            labels_path = results_dir / f"{group_name}_{method}_labels.npy"
            embed_path = results_dir / f"{group_name}_{method}_embeddings.npy"

            if labels_path.exists():
                labels = np.load(labels_path)

                # UMAP plot
                umap_img = results_dir / f"{group_name}_{method}_umap.png"
                if umap_img.exists():
                    html_parts.append(_embed_image(umap_img, f"{group_name} {method} UMAP"))

                # Ethogram
                ethogram_img = results_dir / f"{group_name}_{method}_ethogram.png"
                if ethogram_img.exists():
                    html_parts.append(_embed_image(ethogram_img, f"{group_name} {method} Ethogram"))

                # Cluster GIFs
                n_clusters = metrics.get("n_clusters", 0)
                if n_clusters > 0:
                    html_parts.append("<h4>Cluster Examples (skeleton GIF)</h4>")
                    html_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 10px;">')
                    for cid in range(min(n_clusters, 12)):
                        gif_bytes = make_cluster_gif(
                            keypoints[:len(labels)], labels, cid, joint_indices,
                            max_frames=20, fps=8,
                        )
                        if gif_bytes:
                            b64 = base64.b64encode(gif_bytes).decode()
                            html_parts.append(
                                f'<div style="text-align: center;">'
                                f'<img src="data:image/gif;base64,{b64}" style="width:200px; border:1px solid #ddd;">'
                                f'<br><small>Cluster {cid} ({(labels == cid).sum()} frames)</small></div>'
                            )
                    html_parts.append("</div>")

    html_parts.append(_html_footer())

    html_content = "\n".join(html_parts)
    with open(output_html, "w") as f:
        f.write(html_content)

    print(f"Report saved to {output_html}")
    return output_html


def _html_header(summary: dict) -> str:
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Behavior Clustering Report — {summary.get('experiment', 'Unknown')}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2980b9; margin-top: 40px; }}
h3 {{ color: #7f8c8d; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
th {{ background: #3498db; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.metric-card {{ display: inline-block; background: white; border-radius: 8px; padding: 15px; margin: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 120px; }}
.metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
.metric-label {{ font-size: 12px; color: #7f8c8d; }}
img {{ max-width: 100%; border-radius: 4px; }}
</style>
</head><body>
<h1>{summary.get('experiment', 'Behavior Clustering Report')}</h1>
<p><strong>Date:</strong> {summary.get('date', 'N/A')} |
<strong>Dataset:</strong> {summary.get('dataset', 'N/A')} |
<strong>Frames:</strong> {summary.get('frames_after_masking', 'N/A')} (excluded: {summary.get('excluded_frames', 0)}) |
<strong>FPS:</strong> {summary.get('fps', 'N/A')}</p>
"""


def _html_metrics_table(summary: dict) -> str:
    rows = []
    for group_name, results in summary.get("results", {}).items():
        for r in results:
            m = r.get("metrics", {})
            if m:
                rows.append(
                    f"<tr><td>{group_name}</td><td>{r['method']}</td>"
                    f"<td>{m.get('n_clusters', '-')}</td>"
                    f"<td>{m.get('silhouette', '-')}</td>"
                    f"<td>{m.get('calinski_harabasz', '-')}</td>"
                    f"<td>{m.get('davies_bouldin', '-')}</td>"
                    f"<td>{m.get('bout_mean_sec', '-')}</td>"
                    f"<td>{m.get('transition_rate_per_sec', '-')}</td></tr>"
                )

    return f"""<h2>Comparison Table</h2>
<table>
<tr><th>Joint Group</th><th>Method</th><th>Clusters</th><th>Silhouette ↑</th>
<th>CH ↑</th><th>DB ↓</th><th>Bout Mean (s)</th><th>Trans/sec</th></tr>
{''.join(rows)}
</table>"""


def _html_metric_card(metrics: dict) -> str:
    cards = []
    for key, label in [
        ("n_clusters", "Clusters"),
        ("silhouette", "Silhouette"),
        ("calinski_harabasz", "CH Score"),
        ("bout_mean_sec", "Bout (s)"),
    ]:
        val = metrics.get(key, "-")
        cards.append(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>'
        )
    return '<div style="margin: 10px 0;">' + "".join(cards) + "</div>"


def _embed_image(img_path: Path, alt: str) -> str:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = img_path.suffix.lstrip(".")
    return f'<img src="data:image/{ext};base64,{b64}" alt="{alt}" style="margin: 10px 0;">'


def _html_footer() -> str:
    return """<hr>
<p style="color: #7f8c8d; font-size: 12px;">
Generated by BehaviorSplatter Phase 3A — Sparse Keypoint Ablation Report<br>
<a href="https://github.com/kafkapple/FaceLift">FaceLift</a> ×
<a href="https://github.com/kafkapple/behavior-lab">behavior-lab</a>
</p>
</body></html>"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/clustering/sparse/results")
    parser.add_argument("--keypoints", default="outputs/clustering/sparse/keypoints_22_3d.npz")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    generate_report(args.results_dir, args.keypoints, args.output)
