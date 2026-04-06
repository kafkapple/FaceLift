# no-split: unified cluster visualization pipeline (GIF + comparison + HTML report), all functions share rendering state
"""Cluster visualization module — GIF montages, preprocessing comparison, HTML report.

Generates:
- Per-cluster skeleton GIF (top-down XY projection with MAMMAL color scheme)
- Per-cluster RGB+skeleton overlay GIF
- Preprocessing before/after comparison GIFs
- UMAP scatter, ethogram, transition matrix
- Self-contained HTML report

All outputs saved to `outputs/reports/clustering/` (see paths.py).

Usage on gpu03:
    cd /home/joon/dev/FaceLift
    python -m mouse_extensions.behavior.visualize_clusters
"""

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np

from mouse_extensions.constants import (
    SKELETON_BONES as MAMMAL_BONES,
    MOUSE_KP_NAMES as KEYPOINT_NAMES,
)

# Per-joint color scheme for behavior visualization (distinct from keypoint overlay)
BODY_COLORS = {
    0: "#FF6B6B", 1: "#FF6B6B", 2: "#FF6B6B", 3: "#FF6B6B",  # head
    4: "#4ECDC4", 5: "#95E1D3", 6: "#95E1D3", 7: "#95E1D3",  # spine/tail
    8: "#FFD93D", 9: "#FFD93D", 10: "#FFD93D", 11: "#FFD93D",  # front_L
    12: "#6BCB77", 13: "#6BCB77", 14: "#6BCB77", 15: "#6BCB77",  # front_R
    16: "#4D96FF", 17: "#4D96FF", 18: "#4D96FF",  # hind_L
    19: "#9B59B6", 20: "#9B59B6", 21: "#9B59B6",  # hind_R
}


def render_skeleton_frame(kp_2d: np.ndarray, xlim: tuple, ylim: tuple,
                          title: str = "", figsize=(3, 3)) -> np.ndarray:
    """Render single skeleton frame as RGB array (top-down XY)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")

    K = kp_2d.shape[0]
    for i, j in MAMMAL_BONES:
        if i < K and j < K:
            c = BODY_COLORS.get(i, "#888")
            ax.plot([kp_2d[i, 0], kp_2d[j, 0]], [kp_2d[i, 1], kp_2d[j, 1]],
                    color=c, linewidth=2, alpha=0.8)
    for idx in range(K):
        c = BODY_COLORS.get(idx, "#888")
        ax.plot(kp_2d[idx, 0], kp_2d[idx, 1], "o", color=c,
                markersize=5, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, color="white", fontsize=9, pad=2)

    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def make_cluster_gif(kp: np.ndarray, labels: np.ndarray, cluster_id: int,
                     max_frames: int = 24, fps: int = 8) -> Optional[bytes]:
    """Create GIF of skeleton animation for a specific cluster."""
    try:
        from PIL import Image
    except ImportError:
        return None

    frames_idx = np.where(labels == cluster_id)[0]
    if len(frames_idx) == 0:
        return None

    selected = _select_consecutive(frames_idx, max_frames)
    kp_sel = kp[selected, :, :2]  # XY projection

    margin = 5
    xlim = (kp_sel[:, :, 0].min() - margin, kp_sel[:, :, 0].max() + margin)
    ylim = (kp_sel[:, :, 1].min() - margin, kp_sel[:, :, 1].max() + margin)

    images = []
    for t in range(len(selected)):
        img = render_skeleton_frame(kp_sel[t], xlim, ylim,
                                    title=f"C{cluster_id} f{selected[t]}")
        images.append(Image.fromarray(img))

    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=1000 // fps, loop=0)
    return buf.getvalue()


def make_preprocessing_comparison_gif(kp_raw: np.ndarray, kp_processed: np.ndarray,
                                      frame_range: tuple = (0, 60),
                                      fps: int = 8) -> Optional[bytes]:
    """Side-by-side GIF: raw vs preprocessed keypoints."""
    try:
        from PIL import Image
    except ImportError:
        return None

    start, end = frame_range
    kp_r = kp_raw[start:end, :, :2]
    kp_p = kp_processed[start:end, :, :2]

    margin = 5
    xlim_r = (kp_r[:, :, 0].min() - margin, kp_r[:, :, 0].max() + margin)
    ylim_r = (kp_r[:, :, 1].min() - margin, kp_r[:, :, 1].max() + margin)
    xlim_p = (kp_p[:, :, 0].min() - margin, kp_p[:, :, 0].max() + margin)
    ylim_p = (kp_p[:, :, 1].min() - margin, kp_p[:, :, 1].max() + margin)

    images = []
    for t in range(len(kp_r)):
        img_r = render_skeleton_frame(kp_r[t], xlim_r, ylim_r, title="Raw")
        img_p = render_skeleton_frame(kp_p[t], xlim_p, ylim_p, title="Centered")
        combined = np.concatenate([img_r, img_p], axis=1)
        images.append(Image.fromarray(combined))

    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=1000 // fps, loop=0)
    return buf.getvalue()


def make_umap_plot(embeddings: np.ndarray, labels: np.ndarray, title: str = "") -> bytes:
    """UMAP scatter plot as PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    valid = labels >= 0
    scatter = ax.scatter(embeddings[valid, 0], embeddings[valid, 1],
                         c=labels[valid], cmap="tab20", s=2, alpha=0.5)
    noise = ~valid
    if noise.any():
        ax.scatter(embeddings[noise, 0], embeddings[noise, 1],
                   c="gray", s=1, alpha=0.2, label=f"noise ({noise.sum()})")
        ax.legend(fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def make_ethogram(labels: np.ndarray, fps: float = 20.0, title: str = "") -> bytes:
    """Temporal raster plot as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 1.5), dpi=100)
    time_sec = np.arange(len(labels)) / fps
    ax.imshow(labels[np.newaxis, :], aspect="auto", cmap="tab20",
              interpolation="nearest", extent=[0, time_sec[-1], 0, 1])
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def make_transition_matrix(labels: np.ndarray, title: str = "") -> bytes:
    """Transition probability heatmap as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = labels >= 0
    lab = labels[valid]
    K = len(set(lab))
    T = np.zeros((K, K))
    unique_labels = sorted(set(lab))
    label_map = {l: i for i, l in enumerate(unique_labels)}

    for t in range(len(lab) - 1):
        T[label_map[lab[t]], label_map[lab[t + 1]]] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    im = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("To cluster")
    ax.set_ylabel("From cluster")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(unique_labels, fontsize=8)
    ax.set_yticklabels(unique_labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="P(transition)")

    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if P[i, j] < 0.5 else "white")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


def compute_tpi(labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Temporal Persistence Index (SUBTLE metric)."""
    valid = labels >= 0
    lab = labels[valid]
    unique = sorted(set(lab))
    K = len(unique)
    label_map = {l: i for i, l in enumerate(unique)}

    T = np.zeros((K, K))
    for t in range(len(lab) - 1):
        T[label_map[lab[t]], label_map[lab[t + 1]]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    self_trans = np.diag(P)
    tpi = 1.0 / (1.0 - self_trans + 1e-10)
    return tpi, float(tpi.mean())


def compute_entropy_rate(labels: np.ndarray) -> float:
    """Behavioral entropy rate (Shannon-based)."""
    valid = labels >= 0
    lab = labels[valid]
    unique = sorted(set(lab))
    K = len(unique)
    label_map = {l: i for i, l in enumerate(unique)}

    T = np.zeros((K, K))
    for t in range(len(lab) - 1):
        T[label_map[lab[t]], label_map[lab[t + 1]]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = T / row_sums

    # Stationary distribution (left eigenvector)
    counts = np.zeros(K)
    for l in lab:
        counts[label_map[l]] += 1
    pi = counts / counts.sum()

    H = 0.0
    for i in range(K):
        for j in range(K):
            if P[i, j] > 0:
                H -= pi[i] * P[i, j] * np.log2(P[i, j])
    return float(H)


def _select_consecutive(indices: np.ndarray, max_n: int) -> np.ndarray:
    """Select longest consecutive run, or evenly spaced if short."""
    if len(indices) <= max_n:
        return indices
    diffs = np.diff(indices)
    starts = np.concatenate([[0], np.where(diffs > 1)[0] + 1])
    ends = np.concatenate([np.where(diffs > 1)[0] + 1, [len(indices)]])
    lengths = ends - starts
    best = np.argmax(lengths)
    s, e = starts[best], ends[best]
    if e - s >= max_n:
        return indices[s:s + max_n]
    idx = np.linspace(0, len(indices) - 1, max_n, dtype=int)
    return indices[idx]


def _img_to_b64(data: bytes, fmt: str = "png") -> str:
    return f"data:image/{fmt};base64,{base64.b64encode(data).decode()}"


def generate_html_report(
    kp_raw: np.ndarray,
    kp_processed: np.ndarray,
    results_dir: Path,
    summary: dict,
    output_path: str,
    best_preset: str = "centered",
    best_group: str = "full_22",
    best_method: str = "pca_kmeans",
):
    """Generate comprehensive HTML report."""

    html = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 3A Sparse Keypoint Ablation Report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 20px; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 10px; }}
h2 {{ color: #79c0ff; margin-top: 30px; }}
h3 {{ color: #d2a8ff; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: center; }}
th {{ background: #161b22; color: #58a6ff; }}
tr:nth-child(even) {{ background: #161b22; }}
.best {{ background: #1f6feb33 !important; font-weight: bold; }}
.card {{ display: inline-block; background: #161b22; border-radius: 8px; padding: 12px 20px; margin: 5px; text-align: center; border: 1px solid #30363d; }}
.card .val {{ font-size: 22px; font-weight: bold; color: #58a6ff; }}
.card .lbl {{ font-size: 11px; color: #8b949e; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }}
.grid img {{ border: 1px solid #30363d; border-radius: 6px; }}
img {{ max-width: 100%; }}
a {{ color: #58a6ff; }}
</style></head><body>
<h1>Phase 3A — Sparse Keypoint Ablation Report</h1>
<p>Date: {summary.get('date', 'N/A')} | Dataset: M5t2 | Frames: {kp_processed.shape[0]} | FPS: 20</p>
"""]

    # Preprocessing comparison GIF
    html.append("<h2>1. Preprocessing: Raw vs COM-Centered</h2>")
    gif_data = make_preprocessing_comparison_gif(kp_raw[:60], kp_processed[:60])
    if gif_data:
        html.append(f'<img src="{_img_to_b64(gif_data, "gif")}" style="max-width: 500px;">')
        html.append("<p>Left: Raw (cage position included) | Right: COM-centered (position-invariant)</p>")

    # Main comparison table
    html.append("<h2>2. Preprocessing × Joint Group Ablation (PCA+KMeans)</h2>")
    html.append(_build_ablation_table(summary))

    # Best configuration details
    label_file = results_dir / f"{best_preset}_{best_group}_{best_method}_labels.npy"
    if label_file.exists():
        labels = np.load(label_file)
        n_clusters = len(set(labels[labels >= 0]))

        # TPI and Entropy
        tpi_per, tpi_avg = compute_tpi(labels)
        entropy = compute_entropy_rate(labels)

        html.append(f"<h2>3. Best Config: {best_preset} / {best_group} / {best_method}</h2>")
        html.append('<div>')
        for val, lbl in [(n_clusters, "Clusters"), (f"{tpi_avg:.1f}", "TPI (avg frames)"),
                         (f"{entropy:.2f}", "Entropy Rate (bits)")]:
            html.append(f'<div class="card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>')
        html.append('</div>')

        # Ethogram
        html.append("<h3>Ethogram</h3>")
        eth = make_ethogram(labels, title=f"{best_preset}/{best_group}/{best_method}")
        html.append(f'<img src="{_img_to_b64(eth)}">')

        # Transition matrix
        html.append("<h3>Transition Matrix</h3>")
        tm = make_transition_matrix(labels, title="Cluster Transitions")
        html.append(f'<img src="{_img_to_b64(tm)}" style="max-width: 500px;">')

        # UMAP if available
        embed_file = results_dir / f"{best_preset}_{best_group}_umap_hdbscan_embed.npy"
        label_umap_file = results_dir / f"{best_preset}_{best_group}_umap_hdbscan_labels.npy"
        if embed_file.exists() and label_umap_file.exists():
            embed = np.load(embed_file)
            labels_u = np.load(label_umap_file)
            html.append("<h3>UMAP Embedding (HDBSCAN)</h3>")
            umap_img = make_umap_plot(embed, labels_u, title=f"{best_preset}/{best_group}")
            html.append(f'<img src="{_img_to_b64(umap_img)}" style="max-width: 600px;">')

        # Per-cluster GIFs
        html.append("<h3>Per-Cluster Skeleton GIFs</h3>")
        html.append('<div class="grid">')
        for cid in range(min(n_clusters, 12)):
            count = (labels == cid).sum()
            gif = make_cluster_gif(kp_processed, labels, cid, max_frames=20, fps=8)
            if gif:
                html.append(
                    f'<div style="text-align:center;">'
                    f'<img src="{_img_to_b64(gif, "gif")}" style="width:180px;">'
                    f'<br><small>Cluster {cid} ({count} frames, {count/20:.1f}s)</small></div>'
                )
        html.append('</div>')

        # TPI per cluster
        html.append("<h3>TPI per Cluster (frames)</h3>")
        html.append("<table><tr><th>Cluster</th>" +
                    "".join(f"<th>{i}</th>" for i in range(len(tpi_per))) + "</tr>")
        html.append("<tr><td>TPI</td>" +
                    "".join(f"<td>{t:.1f}</td>" for t in tpi_per) + "</tr></table>")

    html.append("""<hr><p style="color:#8b949e; font-size:11px;">
    Phase 3A Sparse Keypoint Ablation | BehaviorSplatter
    | <a href="https://github.com/kafkapple/FaceLift">FaceLift</a></p></body></html>""")

    with open(output_path, "w") as f:
        f.write("\n".join(html))
    print(f"Report saved: {output_path}")


def _build_ablation_table(summary: dict) -> str:
    """Build HTML table from unified summary."""
    rows = []
    best_sil = 0
    for preset, pdata in summary.get("results", {}).items():
        for group, gdata in pdata.get("groups", {}).items():
            for method, mdata in gdata.get("methods", {}).items():
                if method != "pca_kmeans":
                    continue
                sil = mdata.get("silhouette", "-")
                if isinstance(sil, (int, float)) and sil > best_sil:
                    best_sil = sil
                rows.append((preset, group, gdata.get("n_joints", "?"), mdata))

    html = """<table><tr><th>Preset</th><th>Group</th><th>Joints</th>
    <th>K</th><th>Silhouette</th><th>CH</th><th>DB</th><th>Bout (s)</th></tr>"""
    for preset, group, nj, m in rows:
        sil = m.get("silhouette", "-")
        cls = ' class="best"' if isinstance(sil, (int, float)) and abs(sil - best_sil) < 0.001 else ""
        html += (f'<tr{cls}><td>{preset}</td><td>{group}</td><td>{nj}</td>'
                 f'<td>{m.get("n_clusters", "-")}</td><td>{sil}</td>'
                 f'<td>{m.get("calinski_harabasz", "-")}</td>'
                 f'<td>{m.get("davies_bouldin", "-")}</td>'
                 f'<td>{m.get("bout_mean_sec", "-")}</td></tr>')
    html += "</table>"
    return html


def save_visualization(data: bytes, name: str, subdir: str = "", fmt: str = "png") -> Path:
    """Save a visualization file to the standard output directory.

    Args:
        data: bytes content (PNG, GIF, etc.)
        name: filename without extension
        subdir: subdirectory under visualizations/ (e.g., "umap", "rgb_gifs")
        fmt: file format extension

    Returns:
        Path to saved file
    """
    from mouse_extensions.behavior.paths import VIZ_DIR, ensure_dirs
    ensure_dirs()
    out_dir = VIZ_DIR / subdir if subdir else VIZ_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.{fmt}"
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def save_report(html_content: str, name: str) -> Path:
    """Save an HTML report to the standard reports directory.

    Returns:
        Path to saved file
    """
    from mouse_extensions.behavior.paths import REPORTS_DIR, ensure_dirs
    ensure_dirs()
    out_path = REPORTS_DIR / f"{name}.html"
    with open(out_path, "w") as f:
        f.write(html_content)
    print(f"Report saved: {out_path}")
    return out_path


def main():
    from mouse_extensions.behavior.paths import (
        RESULTS_DIR, REPORTS_DIR, GPU03_KEYPOINTS, ensure_dirs,
    )
    ensure_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR / "sparse_ablation"))
    parser.add_argument("--keypoints", default=GPU03_KEYPOINTS)
    parser.add_argument("--output", default=str(REPORTS_DIR / "phase3a_report.html"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = np.load(args.keypoints, allow_pickle=True)
    kp_raw = data["keypoints"]

    from mouse_extensions.behavior.preprocessing import preprocess
    kp_ctr, _ = preprocess(kp_raw, preset="centered")

    # Try to load summary from various locations
    summary = {}
    for candidate in [
        results_dir / "unified_summary.json",
        results_dir / "full_ablation_summary.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                summary = json.load(f)
            break

    generate_html_report(
        kp_raw=kp_raw[:len(kp_ctr)],
        kp_processed=kp_ctr,
        results_dir=results_dir,
        summary=summary,
        output_path=args.output,
        best_preset="bsoid_standard",
        best_group="full_22",
        best_method="pca_kmeans",
    )


if __name__ == "__main__":
    main()
