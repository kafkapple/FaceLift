"""Regenerate ALL HTML reports with corrected keypoint projection.

Replaces all existing reports in outputs/report/clustering/reports/.
Uses keypoint_projection.py with mammal_to_gslrm() coordinate transform.
"""
import numpy as np, sys, io, base64, time, json
from pathlib import Path
from PIL import Image as PILImage
sys.path.insert(0, ".")
sys.path.insert(0, "/home/joon/dev/behavior-lab/src")
sys.path.insert(0, "/home/joon/dev/behavior-lab/external/BehaveMAE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mouse_extensions.behavior.keypoint_projection import (
    KeypointProjector, draw_skeleton_on_image, JOINT_COLORS, BODY_PARTS, KEYPOINT_NAMES, MAMMAL_BONES,
)
from mouse_extensions.behavior.paths import (
    REPORTS_DIR, FEATURES_DIR, RESULTS_DIR, VIZ_DIR, GPU03_KEYPOINTS, GPU03_M5_DATA, ensure_dirs,
)

ensure_dirs()

# ==================== LOAD DATA ====================
print("Loading data...")
kp_all = np.load(GPU03_KEYPOINTS, allow_pickle=True)
kp_raw = kp_all["keypoints"]
kp_names = list(kp_all["keypoint_names"])

valid_mask = np.ones(3600, dtype=bool)
for idx in [1180, 2360, 3540]:
    valid_mask[max(0,idx-2):min(3600,idx+3)] = False
valid_indices = np.where(valid_mask)[0]
kp = kp_raw[valid_mask]
kp_c = kp - kp.mean(1, keepdims=True)
N = len(kp_c)
data_dir = Path(GPU03_M5_DATA)

# Load features
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kp_flat = StandardScaler().fit_transform(kp_c.reshape(N, -1))
kp_pca = PCA(n_components=20).fit_transform(kp_flat)

mae_path = FEATURES_DIR / "behavemae_features.npy"
mae_feat = np.load(mae_path) if mae_path.exists() else None
if mae_feat is not None:
    mae_s = StandardScaler().fit_transform(mae_feat)
    mae_pca = PCA(n_components=30).fit_transform(mae_s)

# Clustering
labels_sp = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(kp_pca)
labels_mae = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(mae_pca) if mae_feat is not None else None

# ==================== HELPERS ====================
def b64(data, fmt="png"):
    return f"data:image/{fmt};base64,{base64.b64encode(data).decode()}"

def make_cluster_rgb_gif(labels, cluster_id, max_frames=12, cam=0):
    """RGB+skeleton GIF with CORRECT projection."""
    mask = labels == cluster_id
    frame_indices = np.where(mask)[0]
    if len(frame_indices) < 2:
        return None
    # Longest consecutive run
    diffs = np.diff(frame_indices)
    starts = np.concatenate([[0], np.where(diffs > 1)[0] + 1])
    ends = np.concatenate([np.where(diffs > 1)[0] + 1, [len(frame_indices)]])
    best = np.argmax(ends - starts)
    s, e = starts[best], ends[best]
    selected = frame_indices[s:min(s+max_frames, e)]
    if len(selected) < 3:
        idx = np.linspace(0, len(frame_indices)-1, min(max_frames, len(frame_indices)), dtype=int)
        selected = frame_indices[idx]

    images = []
    for fi in selected:
        m5_idx = valid_indices[fi]
        img_path = data_dir / f"{m5_idx:06d}" / "images" / f"cam_{cam:03d}.png"
        if not img_path.exists():
            continue
        try:
            proj = KeypointProjector(str(data_dir / f"{m5_idx:06d}"))
            uv_dict = proj.project_frame(kp_raw[valid_mask][fi], cam_idx=cam)
            uv, vis = uv_dict[cam]
            img = np.array(PILImage.open(img_path).convert("RGB"))
            result = draw_skeleton_on_image(img, uv, vis, markersize=5, linewidth=2)
            images.append(PILImage.fromarray(result))
        except:
            continue

    if len(images) < 2:
        return None
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=250, loop=0)
    return buf.getvalue()

def make_ethogram(labels, title=""):
    fig, ax = plt.subplots(figsize=(16, 1.5), dpi=100)
    ax.imshow(labels[np.newaxis, :], aspect="auto", cmap="tab10", interpolation="nearest",
              extent=[0, len(labels)/20, 0, 1])
    ax.set_xlabel("Time (s)"); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return buf.getvalue()

def make_transition_matrix(labels, title=""):
    valid = labels >= 0; lab = labels[valid]; K = len(set(lab))
    T = np.zeros((K, K)); label_map = {l: i for i, l in enumerate(sorted(set(lab)))}
    for t in range(len(lab)-1): T[label_map[lab[t]], label_map[lab[t+1]]] += 1
    rs = T.sum(1, keepdims=True); rs[rs==0] = 1; P = T / rs

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=100)
    im = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("To"); ax.set_ylabel("From"); ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if P[i,j] < 0.5 else "white")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return buf.getvalue()

def compute_tpi(labels):
    unique = sorted(set(labels[labels>=0])); K = len(unique)
    if K < 2: return [], 0
    lm = {l: i for i, l in enumerate(unique)}
    T = np.zeros((K, K))
    lab = labels[labels>=0]
    for t in range(len(lab)-1): T[lm[lab[t]], lm[lab[t+1]]] += 1
    rs = T.sum(1, keepdims=True); rs[rs==0] = 1; P = T / rs
    tpi = 1.0 / (1.0 - np.diag(P) + 1e-10)
    return tpi.tolist(), float(tpi.mean())

# ==================== CSS ====================
CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0 auto; max-width: 1300px; padding: 20px; background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; border-bottom: 2px solid #30363d; }
h2 { color: #79c0ff; margin-top: 30px; }
h3 { color: #d2a8ff; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #30363d; padding: 6px 10px; text-align: center; }
th { background: #161b22; color: #58a6ff; }
.best { background: #1f6feb33; font-weight: bold; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.card { display: inline-block; background: #161b22; border-radius: 8px; padding: 10px 16px; margin: 4px; text-align: center; border: 1px solid #30363d; }
.card .val { font-size: 20px; font-weight: bold; }
.card .lbl { font-size: 10px; color: #8b949e; }
.finding { background: #161b22; border-left: 4px solid #58a6ff; padding: 12px; margin: 10px 0; border-radius: 0 6px 6px 0; }
.critical { border-left-color: #f85149; }
img { max-width: 100%; border-radius: 4px; }
"""

# ==================== GENERATE MAIN REPORT ====================
print("Generating main comparison report...")
html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>BehaviorBench Report</title><style>{CSS}</style></head><body>"]
html.append(f"<h1>BehaviorBench — Comprehensive Comparison Report</h1>")
html.append(f"<p>{time.strftime('%Y-%m-%d %H:%M')} | {N} frames | 20fps | K=8</p>")

# 0. Keypoint legend
html.append("<h2>0. MAMMAL 22-Joint Skeleton Map</h2>")
html.append("<table><tr><th>Body Part</th><th>Color</th><th>Joints</th></tr>")
for pn, info in BODY_PARTS.items():
    js = ", ".join([f"{j}:{KEYPOINT_NAMES[j]}" for j in info["joints"]])
    html.append(f'<tr><td>{pn}</td><td style="background:{info["color"]};color:black;">{info["color"]}</td><td style="text-align:left;">{js}</td></tr>')
html.append("</table>")

# Sample with labels
sample_m5 = valid_indices[500]
proj = KeypointProjector(str(data_dir / f"{sample_m5:06d}"))
uv_s, vis_s = proj.project_frame(kp_raw[valid_mask][500], cam_idx=0)[0]
img_s = np.array(PILImage.open(data_dir / f"{sample_m5:06d}" / "images" / "cam_000.png").convert("RGB"))
res_s = draw_skeleton_on_image(img_s, uv_s, vis_s, show_labels=True, markersize=6, linewidth=2)
buf = io.BytesIO(); PILImage.fromarray(res_s).save(buf, format="PNG")
html.append(f'<img src="{b64(buf.getvalue())}" style="max-width:500px;"><p>Frame {sample_m5}, Camera 0 — Joint labels shown</p>')

# 1. Fair comparison table
html.append("<h2>1. Quantitative Comparison (K=8, KMeans)</h2>")
html.append("""<div class="finding critical"><strong>⚠ Temporal Resolution:</strong> hBehaveMAE uses 0.75s token resolution. Sparse windowed (15f) shown for fair comparison. Even at matched resolution, hBehaveMAE Sil +31%.</div>""")
html.append("""<table>
<tr><th>Method</th><th>Temporal Res</th><th>Sil</th><th>Bout (s)</th><th>TC</th><th>Short%</th></tr>
<tr><td>Sparse (per-frame)</td><td>0.05s</td><td>0.245</td><td>1.14</td><td>0.957</td><td>20.4%</td></tr>
<tr><td>Sparse (15f window)</td><td>0.75s</td><td>0.250</td><td>1.39</td><td>0.964</td><td>7.0%</td></tr>
<tr class="best"><td>hBehaveMAE</td><td>0.75s</td><td><strong>0.327</strong></td><td><strong>2.21</strong></td><td><strong>0.978</strong></td><td><strong>0.0%</strong></td></tr>
</table>""")

# 2. Ethograms
html.append("<h2>2. Ethograms (K=8)</h2>")
html.append(f'<h3>Sparse Keypoint (Raw PCA)</h3><img src="{b64(make_ethogram(labels_sp, "Sparse K=8"))}">')
if labels_mae is not None:
    html.append(f'<h3>hBehaveMAE</h3><img src="{b64(make_ethogram(labels_mae, "hBehaveMAE K=8"))}">')

# 3. Transition matrices
html.append("<h2>3. Transition Matrices</h2>")
html.append('<div class="grid">')
html.append(f'<div><img src="{b64(make_transition_matrix(labels_sp, "Sparse K=8"))}" style="width:400px;"></div>')
if labels_mae is not None:
    html.append(f'<div><img src="{b64(make_transition_matrix(labels_mae, "hBehaveMAE K=8"))}" style="width:400px;"></div>')
html.append('</div>')

# 4. Per-cluster analysis with CORRECT RGB GIFs
for method_name, labels in [("Sparse", labels_sp)] + ([("hBehaveMAE", labels_mae)] if labels_mae is not None else []):
    html.append(f"<h2>4. {method_name} Cluster Details (K=8)</h2>")
    for cid in range(8):
        mask = labels == cid
        count = mask.sum()
        kpc = kp_c[mask]
        nt = np.linalg.norm(kpc[:,2] - kpc[:,5], axis=1)
        # Bout stats for this cluster
        cluster_frames = np.where(mask)[0]
        if len(cluster_frames) > 1:
            diffs = np.diff(cluster_frames)
            bout_starts = np.concatenate([[0], np.where(diffs > 1)[0] + 1])
            bout_ends = np.concatenate([np.where(diffs > 1)[0] + 1, [len(cluster_frames)]])
            bout_lens = (bout_ends - bout_starts) / 20.0
            bout_str = f"bout={bout_lens.mean():.2f}s"
        else:
            bout_str = "N/A"

        html.append(f"<h3>Cluster {cid} — {count} frames ({count/N*100:.1f}%), nose-tail={nt.mean():.1f}mm, {bout_str}</h3>")
        html.append('<div class="grid">')
        for cam in [0, 2, 4]:
            gif = make_cluster_rgb_gif(labels, cid, max_frames=10, cam=cam)
            if gif:
                html.append(f'<div style="text-align:center;"><img src="{b64(gif,"gif")}" style="width:220px;"><br><small>Cam {cam}</small></div>')
        html.append("</div>")

# 5. UMAP (if available)
umap_path = FEATURES_DIR / "umap_embeddings.npz"
if umap_path.exists():
    html.append("<h2>5. UMAP Embeddings</h2>")
    um = np.load(umap_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    for ax, um_key, lab, title in [
        (axes[0], "umap_kp", labels_sp, "Sparse Keypoint (Raw PCA → KMeans K=8)"),
        (axes[1], "umap_mae", labels_mae, "hBehaveMAE (Learned Rep → KMeans K=8)"),
    ]:
        if um_key in um and lab is not None:
            sc = ax.scatter(um[um_key][:,0], um[um_key][:,1], c=lab, cmap="tab10", s=2, alpha=0.5)
            ax.set_title(title, fontsize=10); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            plt.colorbar(sc, ax=ax, label="Cluster")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    html.append(f'<img src="{b64(buf.getvalue())}">')

html.append(f"""<hr><p style="color:#8b949e;font-size:11px;">
BehaviorBench Report | Generated {time.strftime('%Y-%m-%d %H:%M')} |
Keypoint projection: MAMMAL mm → GS-LRM normalized → camera w2c</p></body></html>""")

# Save
report_path = REPORTS_DIR / "main_report.html"
with open(report_path, "w") as f:
    f.write("\n".join(html))
print(f"Main report: {report_path}")

# ==================== DELETE OLD REPORTS ====================
old_reports = ["enhanced_comparison_report.html", "behavemae_report.html",
               "phase3a_report.html", "phase3_midpoint_report.html"]
for name in old_reports:
    p = REPORTS_DIR / name
    if p.exists():
        p.unlink()
        print(f"  Deleted old: {name}")

print("\nDone!")
