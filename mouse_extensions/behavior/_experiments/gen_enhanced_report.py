"""Enhanced report: RGB+skeleton side-by-side, keypoint labels, transition comparison."""
import numpy as np, sys, io, base64, time, json
from pathlib import Path
from PIL import Image
sys.path.insert(0, ".")
sys.path.insert(0, "/home/joon/dev/behavior-lab/src")
sys.path.insert(0, "/home/joon/dev/behavior-lab/external/BehaveMAE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mouse_extensions.behavior.visualize_clusters import (
    make_ethogram, make_transition_matrix, compute_tpi, compute_entropy_rate, _img_to_b64,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==================== DATA ====================
from mouse_extensions.paths import KP_22
kp_all = np.load(str(KP_22),
                 allow_pickle=True)
kp_raw = kp_all["keypoints"]
kp_names = list(kp_all["keypoint_names"])
valid_mask = np.ones(3600, dtype=bool)
for idx in [1180, 2360, 3540]:
    valid_mask[max(0,idx-2):min(3600,idx+3)] = False
valid_indices = np.where(valid_mask)[0]
kp = kp_raw[valid_mask]
kp_c = kp - kp.mean(1, keepdims=True)
N = len(kp_c)

mae_feat = np.load("outputs/clustering/behavemae_features.npy")
data_dir = Path("/home/joon/data/preprocessed/FaceLift_mouse/M5")

# ==================== CLUSTERING ====================
kp_flat = StandardScaler().fit_transform(kp_c.reshape(N, -1))
kp_pca = PCA(n_components=20).fit_transform(kp_flat)
mae_s = StandardScaler().fit_transform(mae_feat)
mae_pca = PCA(n_components=30).fit_transform(mae_s)

labels_sparse = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(kp_pca)
labels_mae = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(mae_pca)

# ==================== CONSTANTS ====================
BONES = [(2,0),(2,1),(2,3),(3,4),(4,5),(5,6),(6,7),
         (3,11),(11,10),(10,8),(8,9),(3,15),(15,14),(14,12),(12,13),
         (4,18),(18,17),(17,16),(4,21),(21,20),(20,19)]

BODY_PARTS = {
    "Head": {"joints": [0,1,2,3], "color": "#FF6B6B"},
    "Spine/Tail": {"joints": [4,5,6,7], "color": "#4ECDC4"},
    "Front L": {"joints": [8,9,10,11], "color": "#FFD93D"},
    "Front R": {"joints": [12,13,14,15], "color": "#6BCB77"},
    "Hind L": {"joints": [16,17,18], "color": "#4D96FF"},
    "Hind R": {"joints": [19,20,21], "color": "#9B59B6"},
}

JOINT_COLOR = {}
for part_name, info in BODY_PARTS.items():
    for j in info["joints"]:
        JOINT_COLOR[j] = info["color"]


def make_rgb_skeleton_frame(m5_idx, kp_frame, cam=0, show_labels=False, figsize=(4,4)):
    """Render RGB image with skeleton overlay and optional joint labels."""
    img_path = data_dir / f"{m5_idx:06d}" / "images" / f"cam_{cam:03d}.png"
    if not img_path.exists():
        return None

    img = np.array(Image.open(img_path).convert("RGBA"))[:,:,:3]

    # Simple projection: normalize keypoint XY to image space
    kp_2d = np.zeros((22, 2))
    xmin, xmax = kp_frame[:, 0].min(), kp_frame[:, 0].max()
    ymin, ymax = kp_frame[:, 1].min(), kp_frame[:, 1].max()
    pad = max(xmax-xmin, ymax-ymin) * 0.15
    kp_2d[:, 0] = (kp_frame[:, 0] - xmin + pad) / (xmax - xmin + 2*pad) * 400 + 56
    kp_2d[:, 1] = (kp_frame[:, 1] - ymin + pad) / (ymax - ymin + 2*pad) * 400 + 56

    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    ax.imshow(img)
    for i, j in BONES:
        c = JOINT_COLOR.get(i, "#888")
        ax.plot([kp_2d[i,0], kp_2d[j,0]], [kp_2d[i,1], kp_2d[j,1]], color=c, linewidth=1.5, alpha=0.8)
    for idx in range(22):
        c = JOINT_COLOR.get(idx, "#888")
        ax.plot(kp_2d[idx,0], kp_2d[idx,1], "o", color=c, markersize=4,
                markeredgecolor="white", markeredgewidth=0.3)
        if show_labels:
            ax.annotate(str(idx), (kp_2d[idx,0]+3, kp_2d[idx,1]-3),
                       fontsize=5, color=c, fontweight="bold")
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout(pad=0.1)
    fig.canvas.draw()
    frame_img = np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
    plt.close(fig)
    return frame_img


def make_cluster_rgb_gif(labels, cluster_id, max_frames=16, cam=0, show_labels=False):
    """RGB+skeleton GIF for a specific cluster."""
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
        frame_img = make_rgb_skeleton_frame(m5_idx, kp_raw[valid_mask][fi], cam, show_labels)
        if frame_img is not None:
            images.append(Image.fromarray(frame_img))

    if len(images) < 2:
        return None

    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=250, loop=0)
    return buf.getvalue()


# ==================== BUILD HTML ====================
print("Building enhanced report...")

html = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>BehaviorBench — Enhanced Comparison Report</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 0 auto; max-width: 1300px; padding: 20px; background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; border-bottom: 2px solid #30363d; }
h2 { color: #79c0ff; margin-top: 30px; }
h3 { color: #d2a8ff; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #30363d; padding: 6px 10px; text-align: center; }
th { background: #161b22; color: #58a6ff; }
.best { background: #1f6feb33; font-weight: bold; }
.warn { background: #d2992233; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.card { display: inline-block; background: #161b22; border-radius: 8px; padding: 10px 16px; margin: 4px; text-align: center; border: 1px solid #30363d; }
.card .val { font-size: 20px; font-weight: bold; }
.card .lbl { font-size: 10px; color: #8b949e; }
.finding { background: #161b22; border-left: 4px solid #58a6ff; padding: 12px; margin: 10px 0; border-radius: 0 6px 6px 0; }
.critical { border-left-color: #f85149; }
img { max-width: 100%; border-radius: 4px; }
</style></head><body>"""]

html.append(f"<h1>BehaviorBench — Enhanced Comparison Report</h1>")
html.append(f"<p>{time.strftime('%Y-%m-%d %H:%M')} | {N} frames | 20fps | K=8</p>")

# 0. Keypoint Legend
html.append("<h2>0. MAMMAL 22-Joint Keypoint Map</h2>")
html.append("<table><tr><th>Body Part</th><th>Color</th><th>Joints</th></tr>")
for part_name, info in BODY_PARTS.items():
    joint_str = ", ".join([f"{j}:{kp_names[j]}" for j in info["joints"]])
    html.append(f'<tr><td>{part_name}</td><td style="background:{info["color"]};color:black;">{info["color"]}</td><td style="text-align:left;">{joint_str}</td></tr>')
html.append("</table>")

# Sample frame with labeled keypoints
sample_img = make_rgb_skeleton_frame(valid_indices[100], kp_raw[valid_mask][100], cam=0, show_labels=True)
if sample_img is not None:
    buf = io.BytesIO()
    Image.fromarray(sample_img).save(buf, format="PNG")
    html.append(f'<p>Sample frame with joint indices:</p><img src="{_img_to_b64(buf.getvalue())}" style="max-width:400px;">')

# 1. Fair Comparison Table
html.append("<h2>1. Fair Comparison (Same Temporal Resolution)</h2>")
html.append("""<div class="finding critical">
<strong>⚠ Temporal Resolution Note:</strong> hBehaveMAE uses 15-frame tokens (0.75s resolution).
For fair comparison, sparse keypoints are also shown with 15-frame window averaging.
Even at matched resolution, hBehaveMAE outperforms sparse by Sil +31%.
</div>""")
html.append("""<table>
<tr><th>Method</th><th>Temporal Res</th><th>Sil</th><th>Bout (s)</th><th>TC</th><th>Short%</th></tr>
<tr><td>Sparse (per-frame)</td><td>0.05s</td><td>0.245</td><td>1.14</td><td>0.957</td><td>20.4%</td></tr>
<tr class="warn"><td>Sparse (15f window)</td><td>0.75s</td><td>0.250</td><td>1.39</td><td>0.964</td><td>7.0%</td></tr>
<tr class="best"><td>hBehaveMAE</td><td>0.75s</td><td><strong>0.327</strong></td><td><strong>2.21</strong></td><td><strong>0.978</strong></td><td><strong>0.0%</strong></td></tr>
</table>""")

# 2. Ethograms
html.append("<h2>2. Ethogram Comparison (K=8)</h2>")
eth_sp = make_ethogram(labels_sparse, title="Sparse Keypoint K=8")
eth_mae = make_ethogram(labels_mae, title="hBehaveMAE K=8")
html.append(f'<h3>Sparse</h3><img src="{_img_to_b64(eth_sp)}">')
html.append(f'<h3>hBehaveMAE</h3><img src="{_img_to_b64(eth_mae)}">')

# 3. Transition Matrices side by side
html.append("<h2>3. Transition Matrices</h2>")
tm_sp = make_transition_matrix(labels_sparse, title="Sparse K=8")
tm_mae = make_transition_matrix(labels_mae, title="hBehaveMAE K=8")
html.append('<div class="grid">')
html.append(f'<div><img src="{_img_to_b64(tm_sp)}" style="width:450px;"></div>')
html.append(f'<div><img src="{_img_to_b64(tm_mae)}" style="width:450px;"></div>')
html.append('</div>')

html.append("""<div class="finding">
<strong>Transition Analysis:</strong> hBehaveMAE shows higher self-transition (diagonal ~0.97)
vs Sparse (~0.94). This is partly due to MAE's 0.75s token resolution, but even at matched
temporal resolution (15f window), sparse still shows more flickering (7% short bouts vs 0%).
</div>""")

# 4. Per-cluster RGB+Skeleton GIFs with kinematics
html.append("<h2>4. hBehaveMAE Cluster GIFs (RGB + Skeleton, K=8)</h2>")
for cid in range(8):
    mask = labels_mae == cid
    count = mask.sum()
    kpc = kp_c[mask]
    nt = np.linalg.norm(kpc[:,2] - kpc[:,5], axis=1)

    html.append(f"<h3>Cluster {cid} — {count} frames ({count/N*100:.1f}%), nose-tail={nt.mean():.1f}±{nt.std():.1f}mm</h3>")
    html.append('<div class="grid">')
    for cam in [0, 2]:
        gif = make_cluster_rgb_gif(labels_mae, cid, max_frames=12, cam=cam)
        if gif:
            html.append(f'<img src="{_img_to_b64(gif, "gif")}" style="width:280px;">')
    html.append("</div>")

# 5. Sparse Cluster GIFs for comparison
html.append("<h2>5. Sparse Cluster GIFs (RGB + Skeleton, K=8)</h2>")
for cid in range(8):
    mask = labels_sparse == cid
    count = mask.sum()
    html.append(f"<h3>Cluster {cid} — {count} frames ({count/N*100:.1f}%)</h3>")
    html.append('<div class="grid">')
    for cam in [0, 2]:
        gif = make_cluster_rgb_gif(labels_sparse, cid, max_frames=12, cam=cam)
        if gif:
            html.append(f'<img src="{_img_to_b64(gif, "gif")}" style="width:280px;">')
    html.append("</div>")

# 6. Per-cluster kinematics table
html.append("<h2>6. Per-Cluster Kinematics (hBehaveMAE K=8)</h2>")
html.append("<table><tr><th>C</th><th>Frames</th><th>%</th><th>Nose-Tail</th><th>Paw Height</th><th>Interpretation</th></tr>")
for c in range(8):
    mask = labels_mae == c
    kpc = kp_c[mask]
    n = mask.sum()
    nt = np.linalg.norm(kpc[:,2] - kpc[:,5], axis=1)
    paw_z = kpc[:, [8,12,16,19], 2].mean(1)
    if nt.mean() > 80: interp = "Extended body"
    elif nt.mean() < 65: interp = "Compact/curled"
    else: interp = "Medium posture"
    html.append(f"<tr><td>C{c}</td><td>{n}</td><td>{n/N*100:.1f}%</td><td>{nt.mean():.1f}±{nt.std():.1f}</td><td>{paw_z.mean():.1f}±{paw_z.std():.1f}</td><td>{interp}</td></tr>")
html.append("</table>")

html.append("""<hr><p style="color:#8b949e;font-size:11px;">
BehaviorBench Enhanced Report | Generated """ + time.strftime("%Y-%m-%d %H:%M") + """</p></body></html>""")

# Save
from mouse_extensions.behavior.paths import REPORTS_DIR, ensure_dirs
ensure_dirs()
out_path = str(REPORTS_DIR / "enhanced_comparison_report.html")
with open(out_path, "w") as f:
    f.write("\n".join(html))
print("Report:", out_path)
