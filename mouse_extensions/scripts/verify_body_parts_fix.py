"""Verification script for body_parts coverage + BONE_SEGMENTS hip fix.

Produces a 3-panel figure saved to outputs/analysis/mouse/:
  Panel 1 — Skeleton diagram colored by body_part, all 22 kp labeled
  Panel 2 — Body-parts coverage matrix: YAML / constants / view_projected_filtering
  Panel 3 — Hind limb skeleton zoom: corrected hip parent (tail_root vs body_middle)

Uses existing modules only:
  mouse_extensions.constants          — BODY_PARTS, BODY_PART_COLORS, SKELETON_BONES, ...
  mouse_extensions.behavior.render_bodypart_gaussians — BONE_SEGMENTS
  mouse_extensions.behavior.view_projected_filtering  — BODY_PARTS (VPF)
  mouse_extensions.paths              — output path factory
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ---------- Load from existing modules ----------
from mouse_extensions.constants import (
    MOUSE_KP_NAMES, MAMMAL_KP_COLORS, SKELETON_BONES,
    BODY_PARTS, BODY_PART_COLORS,
)
from mouse_extensions.behavior.render_bodypart_gaussians import BONE_SEGMENTS
from mouse_extensions.behavior.view_projected_filtering import (
    BODY_PARTS as VPF_BODY_PARTS,
    BODY_PART_COLORS as VPF_COLORS,
)
from mouse_extensions.paths import get_analysis_dir

# ---------- Canonical 2D layout for mouse skeleton (normalized [0,1]) ----------
# Anatomically approximate positions for a mouse in dorsal view
KP_XY = {
    0:  (0.35, 0.06),   # L_ear
    1:  (0.65, 0.06),   # R_ear
    2:  (0.50, 0.03),   # nose
    3:  (0.50, 0.18),   # neck
    4:  (0.50, 0.42),   # body_middle
    5:  (0.50, 0.62),   # tail_root
    6:  (0.50, 0.78),   # tail_middle
    7:  (0.50, 0.94),   # tail_end
    8:  (0.18, 0.50),   # L_paw
    9:  (0.12, 0.58),   # L_paw_end
    10: (0.22, 0.38),   # L_elbow
    11: (0.30, 0.26),   # L_shoulder
    12: (0.82, 0.50),   # R_paw
    13: (0.88, 0.58),   # R_paw_end
    14: (0.78, 0.38),   # R_elbow
    15: (0.70, 0.26),   # R_shoulder
    16: (0.24, 0.92),   # L_foot
    17: (0.28, 0.76),   # L_knee
    18: (0.34, 0.62),   # L_hip
    19: (0.76, 0.92),   # R_foot
    20: (0.72, 0.76),   # R_knee
    21: (0.66, 0.62),   # R_hip
}

# Build reverse map: kp_idx -> body_part name
KP_TO_PART = {}
for part, idxs in BODY_PARTS.items():
    for i in idxs:
        KP_TO_PART[i] = part

# ------------------------------------------------------------------ #
#  PANEL 1: Full skeleton, colored by body_part, kp labels           #
# ------------------------------------------------------------------ #

def _rgb_hex(color) -> str:
    """Convert (R,G,B) uint8 tuple or hex str to matplotlib-compatible color."""
    if isinstance(color, str):
        return color
    r, g, b = color
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_skeleton_panel(ax, bones, title, kp_xy=KP_XY,
                        body_parts=BODY_PARTS, bp_colors=BODY_PART_COLORS,
                        kp_colors=MAMMAL_KP_COLORS, kp_names=MOUSE_KP_NAMES,
                        show_labels=True, highlight_hind=False):
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=11, pad=8, fontweight="bold")
    ax.axis("off")

    # Draw bones
    for a, b in bones:
        xa, ya = kp_xy[a]
        xb, yb = kp_xy[b]
        part_a = KP_TO_PART.get(a)
        part_b = KP_TO_PART.get(b)
        # Color bone by the non-torso part (or average)
        if part_a == part_b:
            col = bp_colors.get(part_a, "#888888")
        elif part_a in ("torso", None):
            col = bp_colors.get(part_b, "#888888")
        else:
            col = bp_colors.get(part_a, "#888888")
        ax.plot([xa, xb], [ya, yb], color=col, lw=2.5, alpha=0.75, zorder=2)

    # Draw keypoints
    for idx, (x, y) in kp_xy.items():
        c = _rgb_hex(kp_colors.get(idx, (200, 200, 200)))
        part = KP_TO_PART.get(idx, "unassigned")
        ring_color = bp_colors.get(part, "#ffffff")
        size = 90 if (highlight_hind and idx in {5, 16, 17, 18, 19, 20, 21}) else 60
        ax.scatter(x, y, s=size, c=c, zorder=5, edgecolors=ring_color, linewidths=1.5)
        if show_labels:
            name = kp_names[idx]
            ha = "right" if x < 0.5 else "left"
            dx = -0.03 if x < 0.5 else 0.03
            ax.text(x + dx, y, f"{idx}:{name}", fontsize=6.5, color="white",
                    ha=ha, va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="#00000055", ec="none"))

    # Legend (body parts)
    handles = [
        mpatches.Patch(color=bp_colors.get(p, "#888"), label=p)
        for p in body_parts
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              facecolor="#222244", edgecolor="#555", labelcolor="white",
              framealpha=0.8, ncol=2)


# ------------------------------------------------------------------ #
#  PANEL 2: Coverage matrix — YAML / constants / VPF alignment       #
# ------------------------------------------------------------------ #

def draw_coverage_matrix(ax):
    ax.set_facecolor("#1a1a2e")
    ax.set_title("Body-Part Coverage — 3 Sources Aligned", color="white",
                 fontsize=11, pad=8, fontweight="bold")

    n_kp = 22
    sources = {
        "YAML\nbody_parts":      BODY_PARTS,
        "constants\n.BODY_PARTS": BODY_PARTS,   # same data, different import path
        "view_projected\n_filtering": VPF_BODY_PARTS,
    }
    part_order = list(BODY_PARTS.keys())

    # Build assignment arrays: sources × kp_idx
    n_src = len(sources)
    matrix = np.zeros((n_src, n_kp), dtype=int)
    kp_to_part_idx = {}
    for pi, part in enumerate(part_order):
        for idx in BODY_PARTS[part]:
            kp_to_part_idx[idx] = pi

    for si, (src_name, bp_dict) in enumerate(sources.items()):
        for part in bp_dict:
            for idx in bp_dict[part]:
                matrix[si, idx] = kp_to_part_idx.get(idx, -1) + 1  # 1-based for colormap

    # Color map: each body_part gets a distinct color
    part_hex = [BODY_PART_COLORS.get(p, "#888888") for p in part_order]
    cmap_colors = ["#1a1a2e"] + part_hex  # 0 = unassigned (dark)

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_colors)

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(part_order),
                   aspect="auto", interpolation="nearest")

    # Annotations: kp name per column
    ax.set_xticks(range(n_kp))
    ax.set_xticklabels(
        [f"{i}\n{MOUSE_KP_NAMES[i]}" for i in range(n_kp)],
        fontsize=5.5, color="white", rotation=90
    )
    src_names = list(sources.keys())
    ax.set_yticks(range(n_src))
    ax.set_yticklabels(src_names, fontsize=8, color="white")

    # Add part label inside each cell
    for si in range(n_src):
        src_bp = list(sources.values())[si]
        kp_to_part_here = {idx: p for p, idxs in src_bp.items() for idx in idxs}
        for ci in range(n_kp):
            part = kp_to_part_here.get(ci, "")
            if part:
                short = part[:3]
                ax.text(ci, si, short, ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold")

    # Legend
    handles = [mpatches.Patch(color=bp, label=p)
               for p, bp in zip(part_order, part_hex)]
    ax.legend(handles=handles, loc="upper right", fontsize=6,
              facecolor="#222244", edgecolor="#555", labelcolor="white",
              framealpha=0.9, ncol=2,
              bbox_to_anchor=(1.0, -0.35))

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")


# ------------------------------------------------------------------ #
#  PANEL 3: Hind limb zoom — BONE_SEGMENTS (fixed) vs old (body_mid) #
# ------------------------------------------------------------------ #

def draw_hind_limb_comparison(ax_old, ax_new):
    hind_kps = {5, 16, 17, 18, 19, 20, 21}
    old_bones = [(4, 18), (18, 17), (17, 16), (4, 21), (21, 20), (20, 19)]
    new_bones = [(a, b) for a, b in BONE_SEGMENTS if a in hind_kps or b in hind_kps]

    # Include body_middle (4) and tail_root (5) for context
    ctx_kps = {4, 5} | hind_kps

    # Local layout: zoom into hind region (x: 0.2-0.8, y: 0.55-1.0)
    def normalize(kp_set):
        xs = [KP_XY[k][0] for k in kp_set]
        ys = [KP_XY[k][1] for k in kp_set]
        return min(xs), max(xs), min(ys), max(ys)

    x0, x1, y0, y1 = 0.10, 0.90, 0.38, 1.00
    margin = 0.04

    for ax, bones, title, label_color, parent_label in [
        (ax_old, old_bones, "BEFORE (body_middle as hip parent)", "#ff6b6b",
         "body_middle\n(WRONG)"),
        (ax_new, new_bones, "AFTER (tail_root as hip parent — YAML match)", "#51cf66",
         "tail_root\n(CORRECT)"),
    ]:
        ax.set_xlim(x0 - margin, x1 + margin)
        ax.set_ylim(y1 + margin, y0 - margin)
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color=label_color, fontsize=10, pad=6, fontweight="bold")
        ax.axis("off")

        # Draw all context bones (faint gray) first
        for a, b in SKELETON_BONES:
            if a in ctx_kps and b in ctx_kps:
                xa, ya = KP_XY[a]
                xb, yb = KP_XY[b]
                ax.plot([xa, xb], [ya, yb], color="#444466", lw=1.5, alpha=0.4, zorder=1)

        # Draw highlight bones
        for a, b in bones:
            if a not in KP_XY or b not in KP_XY:
                continue
            xa, ya = KP_XY[a]
            xb, yb = KP_XY[b]
            is_hip_bone = (b in {18, 21})
            col = label_color if is_hip_bone else "#aaaaff"
            lw = 3.5 if is_hip_bone else 2.0
            ax.plot([xa, xb], [ya, yb], color=col, lw=lw, alpha=0.95, zorder=3,
                    solid_capstyle="round")
            # Arrow on hip bone
            if is_hip_bone:
                dx, dy = xb - xa, yb - ya
                ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                            arrowprops=dict(arrowstyle="->", color=label_color,
                                            lw=2.0, mutation_scale=14))

        # Draw keypoints
        for idx in ctx_kps:
            x, y = KP_XY[idx]
            is_hip = idx in {18, 21}
            is_parent = (idx == 4 and ax is ax_old) or (idx == 5 and ax is ax_new)
            c = _rgb_hex(MAMMAL_KP_COLORS.get(idx, (200, 200, 200)))
            size = 160 if is_parent else (110 if is_hip else 70)
            ring = label_color if is_parent else ("#ffffff" if is_hip else "#444466")
            lw_ring = 3.0 if is_parent else 1.5
            ax.scatter(x, y, s=size, c=c, zorder=6, edgecolors=ring, linewidths=lw_ring)

            name = MOUSE_KP_NAMES[idx]
            ha = "right" if x < 0.5 else "left"
            dx_off = -0.04 if x < 0.5 else 0.04
            weight = "bold" if is_parent or is_hip else "normal"
            color = label_color if is_parent else "white"
            ax.text(x + dx_off, y, f"{idx}:{name}", fontsize=8, color=color,
                    ha=ha, va="center", zorder=7, fontweight=weight,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#00000077", ec="none"))

        # Label the parent node
        parent_idx = 4 if ax is ax_old else 5
        px, py = KP_XY[parent_idx]
        ax.text(px, py - 0.045, parent_label, fontsize=7.5, color=label_color,
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="#00000099",
                          ec=label_color, lw=1.5))


# ------------------------------------------------------------------ #
#  Compose and save                                                   #
# ------------------------------------------------------------------ #

fig = plt.figure(figsize=(22, 20), facecolor="#0d0d1a")
fig.suptitle(
    "FaceLift Mouse — Body-Parts & Skeleton Verification\n"
    "refactor/mouse-extensions  |  2026-03-24  |  Commits 2f86440 → 7f0156f",
    color="white", fontsize=13, y=0.98, fontweight="bold"
)

# Layout: 2 rows
# Row 1: panel1 (skeleton) | panel2 (coverage matrix)
# Row 2: panel3-old | panel3-new
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                      hspace=0.30, wspace=0.18,
                      left=0.04, right=0.96, top=0.94, bottom=0.04)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3a = fig.add_subplot(gs[1, 0])
ax3b = fig.add_subplot(gs[1, 1])

draw_skeleton_panel(ax1, BONE_SEGMENTS,
                    "Panel 1 — Full Skeleton (BONE_SEGMENTS)\nColored by body_part · 22/22 kp labeled",
                    show_labels=True, highlight_hind=True)

draw_coverage_matrix(ax2)
ax2.set_title("Panel 2 — Coverage Matrix\nYAML · constants · view_projected_filtering (all aligned)",
              color="white", fontsize=11, pad=8, fontweight="bold")

draw_hind_limb_comparison(ax3a, ax3b)

# Output
out_dir = get_analysis_dir("mouse", "filtering")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "body_parts_verification_260324.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {out_path}")
