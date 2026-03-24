"""Figure 2: Real mouse RGB image overlay — BEFORE vs AFTER skeleton.

For 3 sampled frames from fj5 dataset (6 cameras each):
  - Projects 3D keypoints → 2D using camera intrinsics/extrinsics
  - Draws BEFORE skeleton (body_middle→hip, wrong) in red
  - Draws AFTER skeleton (tail_root→hip, correct) in green
  - Overlays body-part colored joints with full keypoint name labels

Layout: 3 frames × 2 cams × 2 versions (BEFORE/AFTER) = 3 rows × 4 cols
"""
import json
import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from mouse_extensions.constants import (
    MOUSE_KP_NAMES, MAMMAL_KP_COLORS, BODY_PARTS, BODY_PART_COLORS,
)
from mouse_extensions.behavior.render_bodypart_gaussians import BONE_SEGMENTS
from mouse_extensions.paths import get_analysis_dir

DATA_DIR = Path("/home/joon/data/preprocessed/markerless_mouse_1_nerf/fj5")

# ── Load data ──────────────────────────────────────────────────────────────
def load_data():
    with open(DATA_DIR / "metadata.json") as f:
        meta = json.load(f)
    img_downsample = meta["preprocessing_config"]["image_downsample"]  # e.g. 4

    with h5py.File(DATA_DIR / "images" / "images.h5", "r") as f:
        images = f["images"][:]              # (N, 6, H, W, 3) uint8

    kp3d = np.load(DATA_DIR / "keypoints_3d.npy")   # (N, 22, 3)
    conf = np.load(DATA_DIR / "keypoints_3d_confidence.npy")  # (N, 22)

    with h5py.File(DATA_DIR / "camera_params.h5", "r") as f:
        K_orig = f["camera_parameters/intrinsic"][:]   # (6, 3, 3) full-res
        R = f["camera_parameters/rotation"][:]
        t = f["camera_parameters/translation"][:]

    # Scale K from full-res to stored (downsampled) resolution
    K = K_orig.copy()
    K[:, :2, :] /= img_downsample   # scale fx, skew, cx (row0) and fy, cy (row1)

    return images, kp3d, conf, K, R, t


def project_kp(kp3d_frame, K_cam, R_cam, t_cam):
    """Project (22, 3) world coords → (22, 2) pixel coords.
    K_cam must already be scaled to the stored image resolution.
    """
    # World → Camera: X_cam = R @ X_world + t
    X_cam = (R_cam @ kp3d_frame.T).T + t_cam       # (22, 3)
    # Camera → Image
    z = X_cam[:, 2:3].clip(min=1e-6)
    xy = X_cam[:, :2] / z                           # (22, 2) normalized
    fx, fy = K_cam[0, 0], K_cam[1, 1]
    cx, cy = K_cam[0, 2], K_cam[1, 2]
    u = xy[:, 0] * fx + cx
    v = xy[:, 1] * fy + cy
    return np.stack([u, v], axis=1)                 # (22, 2)


KP_TO_PART = {i: p for p, idxs in BODY_PARTS.items() for i in idxs}

BONES_BEFORE = [(4, 18), (18, 17), (17, 16),   # OLD: body_middle → hip
                (4, 21), (21, 20), (20, 19)]
BONES_AFTER  = [(5, 18), (18, 17), (17, 16),   # NEW: tail_root → hip
                (5, 21), (21, 20), (20, 19)]

HIND_KPS = {5, 16, 17, 18, 19, 20, 21}
HIND_CONTEXT = {4, 5} | HIND_KPS   # show body_middle for context


def _mouse_crop_bounds(img_rgb, pad=20):
    """Return (r0, r1, c0, c1) tight crop around non-white pixels."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray < 245
    if not mask.any():
        H, W = img_rgb.shape[:2]
        return 0, H, 0, W
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    H, W = img_rgb.shape[:2]
    return (max(0, rows[0] - pad), min(H, rows[-1] + pad),
            max(0, cols[0] - pad), min(W, cols[-1] + pad))


def draw_overlay(ax, img_rgb, kp2d, conf_frame, bones_hind, all_bones,
                 title, hind_color, conf_thresh=0.3):
    """Draw skeleton overlay on a single camera view, zoomed to mouse."""
    H, W = img_rgb.shape[:2]
    r0, r1, c0, c1 = _mouse_crop_bounds(img_rgb)
    ax.imshow(img_rgb)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, color=hind_color, pad=6, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#00000099", ec=hind_color, lw=1.5))

    valid = conf_frame > conf_thresh

    def pt(i):
        return (int(kp2d[i, 0]), int(kp2d[i, 1]))

    def in_frame(p):
        return 0 <= p[0] < W and 0 <= p[1] < H

    # ── Draw non-hind bones (gray, subtle) ──
    for a, b in all_bones:
        if a in HIND_KPS or b in HIND_KPS:
            continue
        if not (valid[a] and valid[b]):
            continue
        pa, pb = pt(a), pt(b)
        if not (in_frame(pa) or in_frame(pb)):
            continue
        col = BODY_PART_COLORS.get(KP_TO_PART.get(b, "torso"), "#778899")
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color=col, lw=2.0, alpha=0.55, solid_capstyle="round", zorder=3)

    # ── Draw hind bones (highlighted) ──
    for a, b in bones_hind:
        if not (valid[a] and valid[b]):
            continue
        pa, pb = pt(a), pt(b)
        if not (in_frame(pa) or in_frame(pb)):
            continue
        is_hip_bone = b in {18, 21}
        col = hind_color if is_hip_bone else "#aaaaff"
        lw = 4.0 if is_hip_bone else 2.5
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color=col, lw=lw, alpha=0.95, solid_capstyle="round", zorder=4)
        if is_hip_bone:
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            ax.annotate("", xy=pb, xytext=pa,
                        arrowprops=dict(arrowstyle="-|>", color=hind_color,
                                        lw=2.5, mutation_scale=16),
                        zorder=5)

    # ── Draw all joints ──
    for i in range(22):
        if not valid[i]:
            continue
        p = pt(i)
        if not in_frame(p):
            continue
        rgb = MAMMAL_KP_COLORS.get(i, (180, 180, 180))
        fc = "#{:02x}{:02x}{:02x}".format(*rgb)
        part = KP_TO_PART.get(i, "torso")
        ring = BODY_PART_COLORS.get(part, "#ffffff")

        is_hind = i in HIND_KPS
        is_parent = i in {4, 5} and any(b in {18, 21} for a, b in bones_hind if a == i)
        size = 160 if is_parent else (110 if is_hind else 55)
        ring_lw = 3.0 if (is_parent or is_hind) else 1.0
        ax.scatter(p[0], p[1], s=size, c=fc, zorder=6,
                   edgecolors=hind_color if is_parent else ring,
                   linewidths=ring_lw)

        # Label: always for hind + parent, skip others if crowded
        if is_hind or is_parent:
            name = MOUSE_KP_NAMES[i]
            dx_off = -8 if p[0] < W / 2 else 8
            ha = "right" if p[0] < W / 2 else "left"
            col_txt = hind_color if is_parent else "white"
            ax.text(p[0] + dx_off, p[1],
                    f"[{i}] {name}", fontsize=9, color=col_txt,
                    ha=ha, va="center", zorder=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#00000099", ec="none"))


# ── Main ───────────────────────────────────────────────────────────────────
print("Loading data...")
images, kp3d, conf, K, R, t = load_data()

# Pick 3 frames with good confidence, 2 camera views (top & side)
frame_idxs = [100, 500, 1200]
cam_idxs = [0, 3]   # cam0 (top/side 1) and cam3 (opposite side)

n_frames = len(frame_idxs)
n_cams = len(cam_idxs)
n_cols = n_cams * 2   # BEFORE + AFTER per cam

fig, axes = plt.subplots(n_frames, n_cols,
                         figsize=(n_cols * 9, n_frames * 8),
                         facecolor="#0d0d1a")
fig.suptitle(
    "FaceLift Mouse · Real RGB Overlay — BEFORE vs AFTER Skeleton Fix\n"
    "RED = BEFORE (body_middle→hip, WRONG)   ·   GREEN = AFTER (tail_root→hip, CORRECT)\n"
    "Dataset: markerless_mouse_1_nerf/fj5   |   2026-03-24",
    fontsize=16, color="white", y=1.01, fontweight="bold"
)

for ri, frame_idx in enumerate(frame_idxs):
    kp3d_f = kp3d[frame_idx]   # (22, 3)
    conf_f = conf[frame_idx]   # (22,)
    col = 0
    for cam_idx in cam_idxs:
        img = images[frame_idx, cam_idx]        # (H, W, 3) uint8 RGB
        kp2d = project_kp(kp3d_f, K[cam_idx], R[cam_idx], t[cam_idx])  # (22, 2)

        # BEFORE
        ax_b = axes[ri, col]
        draw_overlay(
            ax_b, img, kp2d, conf_f,
            bones_hind=BONES_BEFORE,
            all_bones=[(a, b) for a, b in BONE_SEGMENTS],
            title=f"BEFORE  frame={frame_idx} cam={cam_idx}",
            hind_color="#ff6b6b",
        )

        # AFTER
        ax_a = axes[ri, col + 1]
        draw_overlay(
            ax_a, img, kp2d, conf_f,
            bones_hind=BONES_AFTER,
            all_bones=[(a, b) for a, b in BONE_SEGMENTS],
            title=f"AFTER  frame={frame_idx} cam={cam_idx}",
            hind_color="#51cf66",
        )
        col += 2

# Column headers
col_titles = []
for cam_idx in cam_idxs:
    col_titles += [f"Camera {cam_idx} — BEFORE", f"Camera {cam_idx} — AFTER"]
for ci, ct in enumerate(col_titles):
    axes[0, ci].set_title(
        axes[0, ci].get_title(),
        fontsize=13,
        color="#ff6b6b" if "BEFORE" in ct else "#51cf66",
        fontweight="bold", pad=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="#00000099",
                  ec="#ff6b6b" if "BEFORE" in ct else "#51cf66", lw=1.5)
    )

# Body-part legend (bottom)
legend_handles = [
    mpatches.Patch(facecolor=BODY_PART_COLORS[p], edgecolor="white",
                   linewidth=0.8, label=f"  {p}")
    for p in BODY_PARTS
]
legend_handles += [
    mpatches.Patch(facecolor="#ff6b6b", edgecolor="white", label="  WRONG hip connection"),
    mpatches.Patch(facecolor="#51cf66", edgecolor="white", label="  CORRECT hip connection"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=9,
           fontsize=11, facecolor="#1a1a33", edgecolor="#555577",
           labelcolor="white", framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.04, 1, 1.0])

out_dir = get_analysis_dir("mouse", "filtering")
out_dir.mkdir(parents=True, exist_ok=True)
out2 = out_dir / "fig2_real_overlay_260324.png"
fig.savefig(out2, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out2}")
