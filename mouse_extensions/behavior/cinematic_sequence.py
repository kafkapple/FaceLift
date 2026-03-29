# no-split: single cinematic pipeline with 8 tightly-coupled segment handlers sharing state
"""Cinematic Sequence Visualization v4: config-driven, temporal-flow architecture.

Segment types:
  flow_gt              — GT RGB, temporal flow
  flow_mask            — FG mask overlay, temporal flow
  flow_render          — GS-LRM render (GT camera), temporal flow
  freeze_orbit         — Freeze time, 360° turntable
  freeze_elev          — Freeze time, elevation arc
  flow_orbit           — Temporal flow + orbiting camera
  flow_bodypart        — Temporal flow + body-part cycling
  freeze_bodypart      — Freeze time + body-part cycling + orbit
  flow_novel           — Temporal flow at fixed novel camera (elevation param)
  freeze_head_orbit    — Freeze time, head-only Gaussians + orbit + zoom
  flow_head_kp         — Temporal flow, head-only + keypoint overlay + orbit
  flow_render_fullbody — Temporal flow, full body + orbiting camera (alias for flow_orbit)

Usage:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.cinematic_sequence \
        --config mouse_extensions/behavior/cinematic_default.yaml \
        --output-dir outputs/viz/cinematic/mouse/latest

    # Or with CLI overrides:
    CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.cinematic_sequence \
        --frame-range 195:255 --fps 15
"""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from mouse_extensions.constants import MAMMAL_KP_COLORS, MOUSE_KP_NAMES, SKELETON_BONES

# Abbreviated keypoint names for overlay labels (22 keypoints)
KP_ABBREVS = ['LE', 'RE', 'Ns', 'Nk', 'BM', 'TR', 'TM', 'TE',
               'LP', 'LPe', 'LEb', 'LSh', 'RP', 'RPe', 'REb', 'RSh',
               'LF', 'LKn', 'LHp', 'RF', 'RKn', 'RHp']
from mouse_extensions.behavior.camera_system import (
    gt_camera_c2w_original as _cs_gt_camera_c2w,
    gt_camera_fxfycxcy as _cs_gt_camera_fxfycxcy,
    interpolate_cameras,
)
from mouse_extensions.behavior.view_projected_filtering import (
    BODY_PARTS, BODY_PART_COLORS, load_camera, load_keypoints_gslrm,
)
from mouse_extensions.behavior.render_bodypart_gaussians import (
    assign_gaussians_to_bodyparts_3d,
)
from mouse_extensions.behavior.multiview_visibility_filter import (
    compute_visibility_counts, render_filtered_gaussians,
)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# Camera functions: use camera_system.py as SSOT (2026-03-23)
def gt_camera_c2w(frame_dir: str, view_idx: int) -> np.ndarray:
    """Get GT camera c2w (4x4) matrix. Delegates to camera_system."""
    return _cs_gt_camera_c2w(frame_dir, view_idx)


def gt_camera_fxfycxcy(frame_dir: str, view_idx: int) -> np.ndarray:
    """Get GT camera intrinsics as [fx, fy, cx, cy]. Delegates to camera_system."""
    return _cs_gt_camera_fxfycxcy(frame_dir, view_idx)


def gt_camera_spherical(frame_dir: str, view_idx: int) -> Tuple[float, float, float]:
    """Extract (radius, elevation_deg, azimuth_deg) of GT camera from c2w.

    Azimuth: standard math convention, CCW from +X axis (same as _cam_spherical and
    get_turntable_cameras which both use pos=[cos(azim), sin(azim), z]).
    Elevation: angle from XY plane (positive = above).
    """
    cam = load_camera(str(Path(frame_dir) / "opencv_cameras.json"), view_idx)
    w2c = np.array(cam["w2c"])
    c2w = np.linalg.inv(w2c)
    x, y, z = c2w[0, 3], c2w[1, 3], c2w[2, 3]
    radius = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.degrees(np.arcsin(z / max(radius, 1e-8)))
    azimuth = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return radius, elevation, azimuth


def get_orbit_cameras(n_frames: int, elevation: float, radius: float = 2.7,
                      hfov: float = 50, resolution: int = 512,
                      mode: str = "turntable", elevation_end: float = None,
                      match_gt: bool = False, gt_frame_dir: str = None,
                      gt_view: int = 0):
    """Generate camera poses via get_turntable_cameras.

    match_gt is deprecated (always False). All configs use fixed spherical params.
    """
    from mouse_extensions.visualization.camera_utils import get_turntable_cameras

    kw = dict(hfov=hfov, num_views=n_frames, w=resolution, h=resolution,
              radius=radius, elevation=elevation, trajectory_mode=mode)
    if elevation_end is not None:
        kw["elevation_end"] = elevation_end
    w, h, _, fxfy, c2ws = get_turntable_cameras(**kw)

    return c2ws, fxfy, w, h


def render_cam(gaussians, c2w, fxfy, w, h, mask=None,
               bg=(0., 0., 0.), device="cuda"):
    """Render with pre-computed camera matrices."""
    from mouse_extensions.visualization import render_opencv_cam
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    fxfy_t = torch.tensor(fxfy, dtype=torch.float32, device=device)
    if mask is not None:
        old = gaussians._opacity.data.clone()
        gaussians._opacity.data[~torch.from_numpy(mask).to(device)] = -100.0
    with torch.no_grad():
        r = render_opencv_cam(gaussians, h, w, c2w_t, fxfy_t, bg_color=bg)
    img = r["render"].permute(1, 2, 0).cpu().numpy()
    if mask is not None:
        gaussians._opacity.data = old
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Frame inference
# ---------------------------------------------------------------------------

def infer_frame(model, fi, m5_dir, kp_path, n_thresh=2, res=512, device="cuda"):
    from mouse_extensions.inference.gslrm_pipeline import load_sample_data
    fd = Path(m5_dir) / f"{fi:06d}"
    if not fd.exists():
        return None
    imgs, c2ws, fxfy, idx = load_sample_data(str(fd), image_size=res, device=device)
    result = model.predict(imgs, c2ws, fxfy, idx)
    g = result.gaussians[0]
    xyz = g.get_xyz.detach().cpu().numpy()
    vc = compute_visibility_counts(xyz, str(fd), n_views=6)
    vm = vc >= n_thresh
    kp = load_keypoints_gslrm(kp_path, fi)
    pm = assign_gaussians_to_bodyparts_3d(xyz, kp)
    del result, imgs, c2ws, fxfy
    torch.cuda.empty_cache()
    return {"gaussians": g, "vis_mask": vm, "part_masks": pm,
            "frame_dir": str(fd), "frame_idx": fi}


def free_fd(fd):
    if fd and "gaussians" in fd:
        del fd["gaussians"]
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_gt(fd_path, view, bg_color=(1.0, 1.0, 1.0), raw=False, res=None):
    """Load GT RGB. raw=True keeps original background, False composites onto bg_color.
    If res is provided, output is resized to (res, res) so it matches VideoWriter dimensions."""
    from PIL import Image
    p = Path(fd_path) / "images" / f"cam_{view:03d}.png"
    img = np.array(Image.open(p))
    rgb = img[:, :, :3] / 255.0
    if not raw and img.shape[-1] == 4:
        alpha = img[:, :, 3:4] / 255.0
        bg = np.array(bg_color).reshape(1, 1, 3)
        rgb = rgb * alpha + bg * (1 - alpha)
    if res is not None and (rgb.shape[0] != res or rgb.shape[1] != res):
        rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_LINEAR)
    return rgb


def load_mask_overlay(fd_path, view, border_color=None, bg_color=(1.0, 1.0, 1.0), res=None):
    """GT image with foreground highlight on bg_color background.
    If res is provided, output is resized to (res, res) so it matches VideoWriter dimensions."""
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_erosion
    img = np.array(Image.open(Path(fd_path) / "images" / f"cam_{view:03d}.png"))
    rgb = img[:, :, :3] / 255.0
    fg = img[:, :, 3] > 128 if img.shape[-1] == 4 else rgb.mean(-1) > 0.04
    # Composite onto bg_color
    if img.shape[-1] == 4:
        alpha = img[:, :, 3:4] / 255.0
        bg = np.array(bg_color).reshape(1, 1, 3)
        rgb = rgb * alpha + bg * (1 - alpha)
    out = rgb * 0.15 + np.array(bg_color).reshape(1, 1, 3) * 0.85  # dim on bg
    out[fg] = rgb[fg]
    if border_color is not None:
        bd = binary_dilation(fg, iterations=2) & ~binary_erosion(fg, iterations=1)
        out[bd] = border_color
    if res is not None and (out.shape[0] != res or out.shape[1] != res):
        out = cv2.resize(out, (res, res), interpolation=cv2.INTER_LINEAR)
    return out


# Constants imported from mouse_extensions.constants (SSOT, 2026-03-23)


def overlay_keypoints(img, kp_2d, kp_valid, bone_width=2, kp_radius=5,
                       kp_depth=None, show_labels=False, show_legend=False):
    """Keypoint + skeleton overlay with optional depth-based opacity and labels.

    Args:
        kp_depth: (K,) z-depth per keypoint in camera space (larger = farther).
                  Used for depth-based opacity: closer = brighter/larger.
        show_labels: Draw abbreviated keypoint names next to each point.
        show_legend: Draw legend panel on right side of image.
    """
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    H, W = u8.shape[:2]

    # Per-keypoint opacity multiplier from depth (0.35 = far, 1.0 = close)
    depth_alpha = np.ones(len(kp_2d))
    if kp_depth is not None:
        valid_d = kp_depth[kp_valid]
        if len(valid_d) > 1:
            d_min, d_max = valid_d.min(), valid_d.max()
            d_range = max(d_max - d_min, 1e-6)
            depth_alpha = 1.0 - 0.65 * np.clip((kp_depth - d_min) / d_range, 0, 1)

    # Skeleton bones
    for a, b in SKELETON_BONES:
        if kp_valid[a] and kp_valid[b]:
            p1 = (int(kp_2d[a, 0]), int(kp_2d[a, 1]))
            p2 = (int(kp_2d[b, 0]), int(kp_2d[b, 1]))
            ca = np.array(MAMMAL_KP_COLORS.get(a, (200, 200, 200)), dtype=np.float32)
            cb = np.array(MAMMAL_KP_COLORS.get(b, (200, 200, 200)), dtype=np.float32)
            ba = (depth_alpha[a] + depth_alpha[b]) / 2
            bc = tuple(int(v * ba) for v in ((ca + cb) / 2).astype(np.uint8))
            cv2.line(u8, p1, p2, (0, 0, 0), bone_width + 2, cv2.LINE_AA)
            cv2.line(u8, p1, p2, bc, bone_width, cv2.LINE_AA)

    # Keypoints
    for i in range(len(kp_2d)):
        if not kp_valid[i]:
            continue
        x, y = int(kp_2d[i, 0]), int(kp_2d[i, 1])
        if not (0 <= x < W and 0 <= y < H):
            continue
        c_full = np.array(MAMMAL_KP_COLORS.get(i, (200, 200, 200)), dtype=np.float32)
        da = depth_alpha[i]
        c = tuple(int(v * da) for v in c_full.astype(np.uint8))
        r = max(2, int(kp_radius * (0.55 + 0.45 * da)))  # closer = larger dot
        cv2.circle(u8, (x, y), r + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(u8, (x, y), r, c, -1, cv2.LINE_AA)
        if show_labels:
            abbrev = KP_ABBREVS[i] if i < len(KP_ABBREVS) else str(i)
            tx, ty = x + r + 3, y + 4
            if 0 <= tx < W - 20 and 0 <= ty < H - 2:
                cv2.putText(u8, abbrev, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.30, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(u8, abbrev, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.30, (220, 220, 220), 1, cv2.LINE_AA)

    if show_legend:
        u8 = _add_kp_legend(u8)
    return u8 / 255.0


def _add_kp_legend(img_u8):
    """Draw a semi-transparent keypoint legend panel on the right side."""
    H, W = img_u8.shape[:2]
    legend_w = 148
    n_kp = len(KP_ABBREVS)
    row_h = max(14, min(22, (H - 16) // n_kp))
    font_s = max(0.26, row_h / 65.0)

    panel = np.zeros((H, legend_w, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)
    # Header
    cv2.putText(panel, "Keypoints", (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 18), (legend_w, 18), (60, 60, 60), 1)

    for i in range(n_kp):
        y = 24 + i * row_h + row_h // 2
        if y >= H - 4:
            break
        c = MAMMAL_KP_COLORS.get(i, (180, 180, 180))
        cv2.circle(panel, (9, y), 5, tuple(int(v) for v in c), -1)
        name = MOUSE_KP_NAMES[i] if i < len(MOUSE_KP_NAMES) else f"KP{i}"
        abbrev = KP_ABBREVS[i] if i < len(KP_ABBREVS) else str(i)
        text = f"{abbrev} {name}"
        cv2.putText(panel, text, (17, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    font_s, (160, 160, 160), 1, cv2.LINE_AA)

    result = img_u8.copy()
    x0 = max(0, W - legend_w)
    roi = result[:, x0:].astype(np.float32)
    blended = roi * 0.25 + panel[:, :W - x0].astype(np.float32) * 0.75
    result[:, x0:] = blended.clip(0, 255).astype(np.uint8)
    return result


def project_kp_to_view(kp_3d, frame_dir, view_idx):
    """Project 3D keypoints to 2D for a given view."""
    from mouse_extensions.behavior.view_projected_filtering import project_points_to_2d
    cam = load_camera(str(Path(frame_dir) / "opencv_cameras.json"), view_idx)
    uv, valid = project_points_to_2d(
        kp_3d, np.array(cam["w2c"]), cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"])
    return uv, in_img


def project_kp_to_novel_cam(kp_3d, c2w, fxfy, w, h):
    """Project 3D keypoints to 2D using a novel camera (c2w + intrinsics).

    Args:
        kp_3d: (K, 3) 3D keypoints in world coords
        c2w: (4, 4) camera-to-world numpy array
        fxfy: (4,) [fx, fy, cx, cy] or (2,) [fx, fy] intrinsics
        w, h: image resolution
    Returns:
        uv: (K, 2) pixel coordinates
        valid: (K,) boolean mask
    """
    from mouse_extensions.behavior.view_projected_filtering import project_points_to_2d
    w2c = np.linalg.inv(c2w)
    fx, fy = float(fxfy[0]), float(fxfy[1])
    cx = float(fxfy[2]) if len(fxfy) > 2 else w / 2.0
    cy = float(fxfy[3]) if len(fxfy) > 3 else h / 2.0
    uv, valid = project_points_to_2d(kp_3d, w2c, fx, fy, cx, cy)
    in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, in_img


def add_label(img, text, fontscale=0.65, thickness=1):
    """Add text label with semi-transparent background panel (ASCII-safe)."""
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    # Replace non-ASCII chars (e.g. em-dash, degree) with ASCII fallbacks
    safe = text.encode('ascii', 'replace').decode('ascii')
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), base = cv2.getTextSize(safe, font, fontscale, thickness)
    pad = 6
    x0, y0 = 10, 8
    x1, y1 = x0 + tw + pad * 2, y0 + th + base + pad * 2
    # Semi-transparent dark background
    overlay = img_u8.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img_u8, 0.45, 0, img_u8)
    # Text with shadow
    tx, ty = x0 + pad, y0 + th + pad
    cv2.putText(img_u8, safe, (tx, ty), font, fontscale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img_u8, safe, (tx, ty), font, fontscale, (240, 240, 240), thickness, cv2.LINE_AA)
    return img_u8 / 255.0


def zoom_to_content(img, zoom: float = 2.0, bg_thresh: float = 0.05) -> np.ndarray:
    """Crop and resize to zoom into non-background content.

    Finds the bounding box of non-background pixels, adds padding,
    crops, and resizes to original resolution.

    Args:
        img: (H, W, 3) float image 0-1
        zoom: zoom factor (2.0 = 2x magnification)
        bg_thresh: threshold to distinguish content from background
    """
    H, W = img.shape[:2]
    # Detect content: pixels that differ from corners (background)
    bg_sample = img[0, 0]  # top-left pixel as background reference
    diff = np.abs(img - bg_sample).max(axis=-1)
    content = diff > bg_thresh

    if content.sum() < 10:
        return img  # no content found, return as-is

    ys, xs = np.where(content)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2

    # Crop size = original / zoom
    crop_h = int(H / zoom)
    crop_w = int(W / zoom)

    # Center crop around content center, clamped to image bounds
    y1 = max(0, min(cy - crop_h // 2, H - crop_h))
    x1 = max(0, min(cx - crop_w // 2, W - crop_w))
    y2 = y1 + crop_h
    x2 = x1 + crop_w

    cropped = img[y1:y2, x1:x2]
    # Resize back to original resolution
    from PIL import Image as PILImage
    resized = np.array(PILImage.fromarray(
        (np.clip(cropped, 0, 1) * 255).astype(np.uint8)
    ).resize((W, H), PILImage.LANCZOS)) / 255.0
    return resized


def crossfade(a, b, t):
    # Resize if shapes differ (e.g., GT 512px → render 768px)
    if a.shape != b.shape:
        from PIL import Image as PILImage
        h, w = b.shape[:2]
        a = np.array(PILImage.fromarray(
            (np.clip(a, 0, 1) * 255).astype(np.uint8)
        ).resize((w, h), PILImage.LANCZOS)) / 255.0
    return np.clip((1 - t) * a + t * b, 0, 1)


# Icon bar: maps segment type → icon name
SEGMENT_ICON_MAP = {
    "flow_gt": "PLAY", "flow_gt_opener": "PLAY", "flow_mask": "PLAY",
    "flow_render": "PLAY", "flow_render_fullbody": "PLAY",
    "freeze_orbit": "ORBIT", "freeze_elev": "ORBIT",
    "freeze_bodypart": "ORBIT", "freeze_head_orbit": "KP",
    "flow_orbit": "PLAY", "flow_novel": "PLAY", "flow_novel_extra": "PLAY",
    "flow_bodypart": "PLAY", "flow_head_kp": "KP",
    "grid_multiview": "GRID", "grid_panorama": "GRID",
}
ICON_DEFS = [("PLAY", "> PLAY"), ("ORBIT", "() ORBIT"), ("KP", "* KP"), ("GRID", "## GRID")]


def draw_icon_bar(img: np.ndarray, active_icon: str, flash_t: float = 0.0) -> np.ndarray:
    """Draw a semi-transparent control icon bar at the top of the frame.

    Args:
        img: (H, W, 3) float32 image 0-1
        active_icon: icon name from ICON_DEFS keys — highlighted white
        flash_t: 0-1, pulse brightness for new-segment flash (fades over 0.5s)
    """
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    H, W = u8.shape[:2]
    bar_h = max(28, int(H * 0.06))  # ~46px at 768

    # Draw semi-transparent dark bar
    overlay = u8.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, u8, 0.35, 0, u8)

    # Draw each icon
    n_icons = len(ICON_DEFS)
    icon_w = W // (n_icons + 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_s = max(0.30, bar_h / 90.0)
    th = max(1, int(bar_h / 25))

    for idx, (name, label) in enumerate(ICON_DEFS):
        cx = icon_w * (idx + 1)
        is_active = (name == active_icon)
        brightness = 1.0
        if is_active and flash_t > 0:
            brightness = 1.0 + 0.35 * flash_t  # pulse up to +35%
        if is_active:
            base_color = (240, 240, 240)
        else:
            base_color = (80, 80, 80)
        color = tuple(int(min(255, v * brightness)) for v in base_color)

        (tw, _), _ = cv2.getTextSize(label, font, font_s, th)
        tx = cx - tw // 2
        ty = bar_h // 2 + int(bar_h * 0.18)
        cv2.putText(u8, label, (tx, ty), font, font_s, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(u8, label, (tx, ty), font, font_s, color, th, cv2.LINE_AA)

        if is_active:
            # Underline bar for active icon
            ux0 = cx - tw // 2 - 2
            ux1 = cx + tw // 2 + 2
            uy = bar_h - 4
            cv2.line(u8, (ux0, uy), (ux1, uy), color, max(1, th), cv2.LINE_AA)

    return u8 / 255.0


def zoom_out_canvas(img: np.ndarray, zoom: float,
                    bg: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Zoom out: shrink image and center on bg canvas (zoom < 1.0).

    Args:
        img: (H, W, 3) float32 image 0-1
        zoom: < 1.0 → object appears smaller (e.g. 0.75 = 75% of frame)
        bg: background fill color (float 0-1)
    Returns same (H, W, 3) resolution.
    """
    H, W = img.shape[:2]
    nw, nh = int(W * zoom), int(H * zoom)
    small = cv2.resize(img.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), bg, dtype=np.float32)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


# ---------------------------------------------------------------------------
# Cinematic pipeline
# ---------------------------------------------------------------------------

class CinematicPipeline:

    def __init__(self, model, cfg, device="cuda"):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.g = cfg["global"]
        self.fps = self.g["fps"]
        self.res = self.g["resolution"]
        self.bg = tuple(self.g.get("bg_color", [0, 0, 0]))
        self.n_thresh = self.g.get("n_filter", 2)
        self.gt_view = self.g.get("gt_view", 0)
        self.cf_frames = int(self.g.get("crossfade", 0.3) * self.fps)
        self.cam_cfg = cfg.get("camera", {})
        self.radius = self.cam_cfg.get("radius", 2.7)
        self.hfov = self.cam_cfg.get("hfov", 50)
        self.m5 = os.path.expanduser(cfg["model"]["m5_dir"])
        self.kp = os.path.expanduser(cfg["model"]["kp_path"])
        self.kp_overlay = self.g.get("keypoint_overlay", False)
        self.border_color = self.g.get("mask_border_color", None)
        self._show_controls = self.g.get("show_controls", False)

        # State shared across segments
        self._last_orbit_az: float = 270.0  # updated by freeze_orbit handler

        # Pre-load keypoints array if overlay enabled
        self._kp_data = None
        if self.kp_overlay:
            self._kp_data = np.load(self.kp)

        # Parse frame range
        parts = str(cfg["frame_range"]).split(":")
        start, end = int(parts[0]), int(parts[1])
        step = int(parts[2]) if len(parts) > 2 else 1
        self.all_frames = list(range(start, end, step))
        self.fi = 0  # global frame index cursor

    def _next_frames(self, n, step=1):
        """Consume frames from the global cursor, cycling if needed.

        step > 1: each unique frame is repeated 'step' rendered frames
        (slows down temporal playback by factor of step).
        """
        if step <= 1:
            out = []
            for _ in range(n):
                out.append(self.all_frames[self.fi % len(self.all_frames)])
                self.fi += 1
            return out
        # Slow-motion: advance cursor once every 'step' rendered frames
        unique_n = max(1, (n + step - 1) // step)
        unique = []
        for _ in range(unique_n):
            unique.append(self.all_frames[self.fi % len(self.all_frames)])
            self.fi += 1
        # Repeat each frame 'step' times, trim to exactly n
        out = [f for f in unique for _ in range(step)][:n]
        # Pad if needed (edge case)
        while len(out) < n:
            out.append(out[-1])
        return out

    def _peek_frame(self):
        return self.all_frames[(self.fi - 1) % len(self.all_frames)]

    def _n(self, duration):
        """Convert duration (seconds) to frame count."""
        return max(1, int(duration * self.fps))

    def _fd(self, fi):
        return str(Path(self.m5) / f"{fi:06d}")

    def _apply_kp(self, img, fi, view=None):
        """Overlay keypoints on image if enabled."""
        if not self.kp_overlay:
            return img
        if view is None:
            view = self.gt_view
        kp_3d = load_keypoints_gslrm(self.kp, fi)
        uv, valid = project_kp_to_view(kp_3d, self._fd(fi), view)
        return overlay_keypoints(img, uv, valid)

    def generate(self, output_path: str, side_by_side: bool = False,
                 use_cache: bool = False, clear_cache: bool = False,
                 dual_output_dir: str = None):
        """Render all segments and write final MP4.

        Args:
            use_cache: If True, save each segment's frames to .seg_cache/ and
                       reload on subsequent runs (skip re-inference for unchanged segments).
            clear_cache: If True with use_cache, wipe .seg_cache/ before running.
            dual_output_dir: If set, also write a version without control overlays
                             (zero re-inference cost — same frames, no icon bar).
        """
        import shutil
        segments = self.cfg["segments"]
        all_imgs = []        # frames WITHOUT icon bar (base frames)
        icon_events = []     # [(start_frame_idx, icon_name), ...]
        last_img = None
        last_fd = None

        # --- Segment cache setup ---
        cache_dir = Path(output_path).parent / ".seg_cache"
        if use_cache:
            if clear_cache and cache_dir.exists():
                shutil.rmtree(cache_dir)
                print("  [CACHE] Cleared")
            cache_dir.mkdir(parents=True, exist_ok=True)

        # Segments that don't produce inference output → prev_fd passes through unchanged
        _NO_INFER_SEGS = {"flow_gt", "flow_mask", "flow_gt_opener"}

        for si, seg in enumerate(segments):
            stype = seg["type"]
            label = seg.get("label", "")
            dur = seg.get("duration", 2.0)
            n = self._n(dur)
            no_crossfade = seg.get("no_crossfade", False)

            print(f"\n  [{si+1}/{len(segments)}] {stype}: {dur}s ({n}f) — {label}")

            handler = getattr(self, f"_seg_{stype}", None)
            if handler is None:
                print(f"    WARNING: unknown type '{stype}', skipping")
                continue

            cache_file = cache_dir / f"seg_{si:02d}_{stype}.npz" if use_cache else None

            if cache_file and cache_file.exists():
                # --- Cache HIT: load frames, restore fi cursor and prev_fd ---
                d = np.load(cache_file)
                frames = list(d["frames"].astype(np.float32) / 255.0)
                self.fi = int(d["fi_after"][0])

                # Restore prev_fd for downstream segments that need Gaussians
                if stype not in _NO_INFER_SEGS and "last_fi" in d:
                    fd_restored = infer_frame(
                        self.model, int(d["last_fi"][0]),
                        self.m5, self.kp, self.n_thresh, self.res, self.device)
                    if fd_restored is not None:
                        last_fd = fd_restored

                print(f"    [CACHE HIT] {len(frames)}f ← {cache_file.name}")
            else:
                # --- Cache MISS: run handler, then optionally save ---
                fi_before = self.fi
                frames, last_fd = handler(seg, n, last_fd)
                fi_after = self.fi

                if cache_file:
                    step = seg.get("frame_step", 1)
                    unique_n = max(1, (n + step - 1) // step) if step > 1 else n
                    last_fi_val = self.all_frames[
                        (fi_before + unique_n - 1) % len(self.all_frames)]
                    frames_u8 = np.array(
                        [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames],
                        dtype=np.uint8)
                    np.savez_compressed(
                        cache_file,
                        frames=frames_u8,
                        fi_after=np.array([fi_after], dtype=np.int32),
                        last_fi=np.array([last_fi_val], dtype=np.int32))
                    print(f"    [CACHE SAVED] {cache_file.name} "
                          f"({frames_u8.nbytes // 1024 // 1024}MB raw)")

            # Apply labels
            if label:
                frames = [add_label(f, label) for f in frames]

            # Crossfade from previous segment (skip if no_crossfade=true)
            if last_img is not None and self.cf_frames > 0 and frames and not no_crossfade:
                cf = min(self.cf_frames, len(frames))
                for i in range(cf):
                    t = (i + 1) / cf
                    frames[i] = crossfade(last_img, frames[i], t)

            # Track icon events for control overlay rendering
            icon_events.append((len(all_imgs), SEGMENT_ICON_MAP.get(stype, "PLAY")))

            all_imgs.extend(frames)
            if frames:
                last_img = frames[-1]

        # --- Write video helper ---
        def write_video(path: str, with_controls: bool):
            flash_dur = int(self.fps * 0.5)  # 0.5s flash on segment start
            vdur = len(all_imgs) / self.fps
            print(f"\n  Writing: {len(all_imgs)} frames @ {self.fps}fps = {vdur:.1f}s"
                  f"  [controls={'ON' if with_controls else 'OFF'}]")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(path, fourcc, self.fps, (self.res, self.res))

            # Build per-frame icon lookup from icon_events
            frame_icons = {}
            if with_controls and icon_events:
                for ev_idx, (ev_frame, ev_icon) in enumerate(icon_events):
                    next_frame = (icon_events[ev_idx + 1][0]
                                  if ev_idx + 1 < len(icon_events) else len(all_imgs))
                    for f in range(ev_frame, next_frame):
                        frame_icons[f] = (ev_icon, f - ev_frame)

            for fi_out, img in enumerate(all_imgs):
                if with_controls and fi_out in frame_icons:
                    icon_name, frames_since = frame_icons[fi_out]
                    ft = max(0.0, (flash_dur - frames_since) / flash_dur)
                    frame = draw_icon_bar(img, icon_name, flash_t=ft)
                else:
                    frame = img
                u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                vw.write(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"  Saved: {path}")
            print(f"✅ CINEMATIC_DONE: {vdur:.1f}s ({len(all_imgs)}f) → {path}")

        # Write main output (with controls if show_controls=true)
        write_video(output_path, with_controls=self._show_controls)

        # Write dual output (no controls) if requested
        if dual_output_dir and self._show_controls:
            Path(dual_output_dir).mkdir(parents=True, exist_ok=True)
            dual_path = str(Path(dual_output_dir) / "cinematic_demo.mp4")
            write_video(dual_path, with_controls=False)

    # --- Segment handlers ---

    def _seg_flow_gt(self, seg, n, prev_fd):
        step = seg.get("frame_step", 1)
        fis = self._next_frames(n, step=step)
        raw = seg.get("raw", False)
        imgs = [self._apply_kp(
            load_gt(self._fd(fi), self.gt_view, self.bg, raw=raw, res=self.res), fi)
                for fi in fis]
        return imgs, prev_fd

    def _seg_flow_mask(self, seg, n, prev_fd):
        fis = self._next_frames(n, step=seg.get("frame_step", 1))
        imgs = [self._apply_kp(
            load_mask_overlay(self._fd(fi), self.gt_view,
                              border_color=self.border_color, bg_color=self.bg,
                              res=self.res), fi)
                for fi in fis]
        return imgs, prev_fd

    def _seg_flow_render(self, seg, n, prev_fd):
        fis = self._next_frames(n, step=seg.get("frame_step", 1))
        imgs = []
        fd = None
        for fi in fis:
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue
            cam = load_camera(str(Path(fd["frame_dir"]) / "opencv_cameras.json"),
                              self.gt_view)
            img = render_filtered_gaussians(
                fd["gaussians"], fd["vis_mask"], cam, self.bg, self.device)
            if img.shape[0] != self.res or img.shape[1] != self.res:
                img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_LINEAR)
            imgs.append(img)
        return imgs, fd  # keep last fd for freeze segments

    def _seg_freeze_orbit(self, seg, n, prev_fd):
        fd = prev_fd or self._ensure_fd()
        elev = seg.get("elevation", 20)
        transition_frames = seg.get("transition_frames", 15)  # smooth GT->orbit
        match_gt = seg.get("match_gt", False)

        # Generate orbit cameras
        orbit_n = max(1, n - transition_frames)
        c2ws, fxfy, w, h = get_orbit_cameras(
            orbit_n, elev, self.radius, self.hfov, self.res,
            match_gt=match_gt, gt_frame_dir=fd["frame_dir"], gt_view=self.gt_view)

        # Get GT camera c2w for smooth transition
        gt_c2w_mat = gt_camera_c2w(fd["frame_dir"], self.gt_view)
        gt_fxfy = gt_camera_fxfycxcy(fd["frame_dir"], self.gt_view)

        # Interpolate GT camera → first orbit camera
        trans_c2ws = interpolate_cameras(gt_c2w_mat, c2ws[0], transition_frames)

        imgs = []
        # Transition: GT → orbit (smooth SLERP)
        for i in range(transition_frames):
            img = render_cam(fd["gaussians"], trans_c2ws[i], gt_fxfy, w, h,
                             mask=fd["vis_mask"], bg=self.bg, device=self.device)
            imgs.append(img)

        # Orbit
        for i in range(orbit_n):
            img = render_cam(fd["gaussians"], c2ws[i], fxfy[i], w, h,
                             mask=fd["vis_mask"], bg=self.bg, device=self.device)
            imgs.append(img)

        # Store ending azimuth for downstream segments (e.g. flow_novel with use_prev_azimuth)
        last_c2w = c2ws[-1]
        x, y = last_c2w[0, 3], last_c2w[1, 3]
        self._last_orbit_az = (np.degrees(-np.arctan2(-x, y)) + 360) % 360

        return imgs, fd

    def _seg_freeze_elev(self, seg, n, prev_fd):
        fd = prev_fd or self._ensure_fd()
        se = seg.get("start_elevation", -30)
        ee = seg.get("end_elevation", 80)
        if abs(se - ee) < 1.0:
            # Fixed elevation = turntable at that elevation (slow orbit)
            c2ws, fxfy, w, h = get_orbit_cameras(
                n, se, self.radius, self.hfov, self.res, mode="turntable")
        else:
            # Elevation sweep = arc mode
            c2ws, fxfy, w, h = get_orbit_cameras(
                n, se, self.radius, self.hfov, self.res,
                mode="arc", elevation_end=ee)
        imgs = []
        for i in range(n):
            img = render_cam(fd["gaussians"], c2ws[i], fxfy[i], w, h,
                             mask=fd["vis_mask"], bg=self.bg, device=self.device)
            imgs.append(img)
        return imgs, fd

    def _seg_flow_orbit(self, seg, n, prev_fd):
        fis = self._next_frames(n)
        elev = seg.get("elevation", 25)
        c2ws, fxfy, w, h = get_orbit_cameras(
            n, elev, self.radius, self.hfov, self.res)
        imgs = []
        fd = prev_fd
        for i, fi in enumerate(fis):
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue
            img = render_cam(fd["gaussians"], c2ws[i], fxfy[i], w, h,
                             mask=fd["vis_mask"], bg=self.bg, device=self.device)
            imgs.append(img)
        return imgs, fd

    def _seg_flow_bodypart(self, seg, n, prev_fd):
        fis = self._next_frames(n)
        parts = seg.get("parts", ["face", "torso", "tail"])
        elev = seg.get("elevation", 30)
        zoom = seg.get("zoom", 1.0)
        c2ws, fxfy, w, h = get_orbit_cameras(
            n, elev, self.radius, self.hfov, self.res)
        imgs = []
        fd = prev_fd
        for i, fi in enumerate(fis):
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue
            pi = (i * len(parts)) // n
            pn = parts[min(pi, len(parts) - 1)]
            if pn == "all":
                mask = fd["vis_mask"]
            else:
                mask = fd["vis_mask"] & fd["part_masks"].get(pn, fd["vis_mask"])
            img = render_cam(fd["gaussians"], c2ws[i], fxfy[i], w, h,
                             mask=mask, bg=self.bg, device=self.device)
            if zoom > 1.0 and pn != "all":
                img = zoom_to_content(img, zoom)
            imgs.append(img)
        return imgs, fd

    def _seg_freeze_bodypart(self, seg, n, prev_fd):
        fd = prev_fd or self._ensure_fd()
        parts = seg.get("parts", ["face", "torso", "tail", "all"])
        elev = seg.get("elevation", 25)
        max_zoom = seg.get("zoom", 1.0)

        # Use GT camera with smooth transition to orbit
        gt_c2w_mat = gt_camera_c2w(fd["frame_dir"], self.gt_view)
        gt_fxfy = gt_camera_fxfycxcy(fd["frame_dir"], self.gt_view)
        c2ws, fxfy, w, h = get_orbit_cameras(
            n, elev, self.radius, self.hfov, self.res)

        frames_per_part = n // len(parts)
        imgs = []
        for i in range(n):
            pi = min((i * len(parts)) // n, len(parts) - 1)
            pn = parts[pi]
            if pn == "all":
                mask = fd["vis_mask"]
            else:
                mask = fd["vis_mask"] & fd["part_masks"].get(pn, fd["vis_mask"])

            # Use orbit camera (slow orbit during body part display)
            img = render_cam(fd["gaussians"], c2ws[i], fxfy[i], w, h,
                             mask=mask, bg=self.bg, device=self.device)

            # Gradual zoom: ramp up then down within each part
            if max_zoom > 1.0 and pn != "all":
                local_i = i - pi * frames_per_part
                local_n = frames_per_part
                # Ease in (first 30%) → hold → ease out (last 20%)
                if local_n > 0:
                    t = local_i / max(local_n - 1, 1)
                    if t < 0.3:
                        z = 1.0 + (max_zoom - 1.0) * (t / 0.3)
                    elif t > 0.8:
                        z = 1.0 + (max_zoom - 1.0) * ((1.0 - t) / 0.2)
                    else:
                        z = max_zoom
                    img = zoom_to_content(img, z)

            imgs.append(img)
        return imgs, fd

    def _overlay_kp_novel(self, img, fi, c2w, fxfy, w, h,
                           show_labels=False, show_legend=False):
        """Overlay keypoints projected through a novel camera with depth-based opacity."""
        kp_3d = load_keypoints_gslrm(self.kp, fi)
        uv, valid = project_kp_to_novel_cam(kp_3d, c2w, fxfy, w, h)
        # Depth in camera space (OpenCV: z positive forward)
        w2c = np.linalg.inv(c2w)
        kp_h = np.concatenate([kp_3d, np.ones((len(kp_3d), 1))], axis=1)
        depth = (w2c @ kp_h.T).T[:, 2]
        return overlay_keypoints(img, uv, valid, kp_depth=depth,
                                  show_labels=show_labels, show_legend=show_legend)

    def _overlay_kp_novel_adjusted(self, img, fi, c2w, fxfy, w, h,
                                    uv_scale: float = 1.0, show_legend: bool = True):
        """Overlay KPs adjusted for zoom_out_canvas, then draw legend at fixed corner.

        When zoom_out_canvas(z) has been applied, the rendered content is centered
        at scale z. UV coordinates must be adjusted accordingly.
        Legend is drawn at fixed bottom-right corner (not affected by zoom).
        """
        kp_3d = load_keypoints_gslrm(self.kp, fi)
        uv_raw, valid = project_kp_to_novel_cam(kp_3d, c2w, fxfy, w, h)
        w2c = np.linalg.inv(c2w)
        kp_h = np.concatenate([kp_3d, np.ones((len(kp_3d), 1))], axis=1)
        depth = (w2c @ kp_h.T).T[:, 2]

        # Adjust UV for zoom_out_canvas offset: uv_adj = uv * scale + [(1-z)*W/2, (1-z)*H/2]
        H_img, W_img = img.shape[:2]
        offset = np.array([(1.0 - uv_scale) * W_img / 2.0,
                            (1.0 - uv_scale) * H_img / 2.0])
        uv_adj = uv_raw * uv_scale + offset
        # Re-check in-bounds after adjustment
        in_bounds = (uv_adj[:, 0] >= 0) & (uv_adj[:, 0] < W_img) & \
                    (uv_adj[:, 1] >= 0) & (uv_adj[:, 1] < H_img)
        valid_adj = valid & in_bounds

        # Draw KPs (no legend inside overlay_keypoints)
        img_out = overlay_keypoints(img, uv_adj, valid_adj, kp_depth=depth,
                                     show_labels=True, show_legend=False)
        # Legend at fixed corner
        if show_legend:
            img_u8 = (np.clip(img_out, 0, 1) * 255).astype(np.uint8)
            img_u8 = _add_kp_legend(img_u8)
            img_out = img_u8 / 255.0
        return img_out

    def _seg_flow_novel(self, seg, n, prev_fd):
        """Temporal flow at fixed novel camera with smooth elevation-sweep entry.

        transition_frames: number of frames to sweep elevation from prev_elevation
        to the target elevation (camera stays at GT azimuth throughout).
        """
        step = seg.get("frame_step", 1)
        fis = self._next_frames(n, step=step)
        elev = seg.get("elevation", -80)
        prev_elev = seg.get("prev_elevation", 20.0)
        trans_n = min(seg.get("transition_frames", 0), n)
        kp_on = seg.get("keypoint_overlay", False)

        use_prev_az = seg.get("use_prev_azimuth", False)

        # Determine azimuth: use last orbit az if requested, else GT camera az
        gt_az = 270.0
        if use_prev_az:
            gt_az = self._last_orbit_az
        elif prev_fd and "frame_dir" in prev_fd:
            _, _, gt_az = gt_camera_spherical(prev_fd["frame_dir"], self.gt_view)

        # Fixed target camera
        c2w_fixed, fxfy_fixed = self._cam_spherical(elev, gt_az)

        # Transition cameras: smooth elevation sweep (eased)
        trans_c2ws = []
        if trans_n > 0:
            ts = np.linspace(0, 1, trans_n + 1)[1:]  # exclude start
            for t in ts:
                t_ease = t * t * (3 - 2 * t)  # smoothstep
                e = prev_elev + (elev - prev_elev) * t_ease
                trans_c2ws.append(self._cam_spherical(e, gt_az)[0])

        imgs = []
        fd = prev_fd
        for i, fi in enumerate(fis):
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue
            c2w = trans_c2ws[i] if i < trans_n else c2w_fixed
            img = render_cam(fd["gaussians"], c2w, fxfy_fixed, self.res, self.res,
                             mask=fd["vis_mask"], bg=self.bg, device=self.device)
            if kp_on:
                img = self._overlay_kp_novel(img, fi, c2w, fxfy_fixed,
                                              self.res, self.res)
            imgs.append(img)
        return imgs, fd

    def _seg_freeze_head_orbit(self, seg, n, prev_fd):
        """Freeze time, head-only Gaussians, slow orbit with zoom."""
        fd = prev_fd or self._ensure_fd()
        elev = seg.get("elevation", 25)
        zoom = seg.get("zoom", 2.5)
        c2ws, fxfy, w, h = get_orbit_cameras(
            n, elev, self.radius, self.hfov, self.res)
        face_mask = fd["vis_mask"] & fd["part_masks"].get("face", fd["vis_mask"])
        imgs = []
        for i in range(n):
            img = render_cam(fd["gaussians"], c2ws[i], fxfy[i], w, h,
                             mask=face_mask, bg=self.bg, device=self.device)
            if zoom > 1.0:
                img = zoom_to_content(img, zoom)
            imgs.append(img)
        return imgs, fd

    def _seg_flow_head_kp(self, seg, n, prev_fd):
        """Temporal flow, head-only Gaussians, orbit + enhanced keypoint overlay.

        Starts with smooth elevation sweep from prev_elevation to target.
        Keypoints include depth-based opacity, abbreviation labels, and a legend.
        Zoom trajectory: 1.0 → 0.75 → 1.0 (zoom-out then return, triangle wave).
        Legend drawn at fixed corner regardless of zoom.
        """
        step = seg.get("frame_step", 1)
        fis = self._next_frames(n, step=step)
        elev = seg.get("elevation", 25)
        prev_elev = seg.get("prev_elevation", -80.0)
        trans_n = min(seg.get("transition_frames", 0), n)

        # GT azimuth for consistent orientation
        gt_az = 270.0
        if prev_fd and "frame_dir" in prev_fd:
            _, _, gt_az = gt_camera_spherical(prev_fd["frame_dir"], self.gt_view)

        # Orbit cameras for main section
        orbit_n = max(1, n - trans_n)
        c2ws, fxfy_orb, w, h = get_orbit_cameras(
            orbit_n, elev, self.radius, self.hfov, self.res,
            match_gt=False, gt_frame_dir=prev_fd["frame_dir"] if prev_fd else None,
            gt_view=self.gt_view)
        fxfy_orb0 = fxfy_orb[0]  # consistent intrinsics

        # Transition cameras: elevation sweep from prev_elev to target
        trans_c2ws = []
        if trans_n > 0:
            ts = np.linspace(0, 1, trans_n + 1)[1:]
            for t in ts:
                t_ease = t * t * (3 - 2 * t)
                e = prev_elev + (elev - prev_elev) * t_ease
                trans_c2ws.append(self._cam_spherical(e, gt_az)[0])

        ease_start = int(orbit_n * 0.8)
        imgs = []
        fd = prev_fd
        for i, fi in enumerate(fis):
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue

            # Camera: transition first, then orbit
            if i < trans_n:
                c2w_i, fxfy_i = trans_c2ws[i], fxfy_orb0
            else:
                oi = i - trans_n
                c2w_i, fxfy_i = c2ws[oi % orbit_n], fxfy_orb[oi % orbit_n]

            oi = max(0, i - trans_n)
            face_mask = (fd["vis_mask"] & fd["part_masks"].get("face", fd["vis_mask"])
                         if oi < ease_start else fd["vis_mask"])

            img = render_cam(fd["gaussians"], c2w_i, fxfy_i, w, h,
                             mask=face_mask, bg=self.bg, device=self.device)

            # Zoom-out trajectory: 1.0 → 0.75 → 1.0 (triangle wave with smoothstep)
            t_zoom = i / max(n - 1, 1)
            t_ease = t_zoom * t_zoom * (3 - 2 * t_zoom)
            z = 1.0 - 0.25 * (1.0 - abs(2 * t_ease - 1.0))

            if z < 0.99:
                img = zoom_out_canvas(img, z, self.bg)
                img = self._overlay_kp_novel_adjusted(
                    img, fi, c2w_i, fxfy_i, w, h, uv_scale=z, show_legend=True)
            else:
                img = self._overlay_kp_novel(img, fi, c2w_i, fxfy_i, w, h,
                                              show_labels=True, show_legend=True)
            imgs.append(img)
        return imgs, fd

    def _seg_flow_gt_opener(self, seg, n, prev_fd):
        """Opening segment: GT 6-cam mosaic → animated zoom-in to gt_view cell.

        Phase 1 (n//2 frames): static 6-cam mosaic grid (no inference).
        Phase 2 (n - n//2 frames): crop-zoom from grid cell bounds → full frame,
            with crossfade to full-res GT at the end for quality.
        """
        n1 = n // 2
        n2 = n - n1

        cell = self.res // 3       # 256 at 768
        pad_top = (self.res - cell * 2) // 2  # 128 at 768

        # Pick a representative frame from dataset midpoint for consistent GT mosaic
        fi_rep = self.all_frames[len(self.all_frames) // 2]
        fd_rep = self._fd(fi_rep)

        def make_mosaic(fi_use):
            fd_path = self._fd(fi_use)
            canvas = np.ones((self.res, self.res, 3), dtype=np.float32) * np.array(self.bg)
            for v in range(6):
                gt_img = load_gt(fd_path, v, self.bg, raw=True, res=self.res)
                row, col = v // 3, v % 3
                sm = cv2.resize(gt_img, (cell, cell), interpolation=cv2.INTER_AREA)
                y0 = pad_top + row * cell
                canvas[y0:y0 + cell, col * cell:(col + 1) * cell] = sm
            return canvas

        # GT view cell bounds in mosaic
        gt_v = self.gt_view
        cell_row, cell_col = gt_v // 3, gt_v % 3
        cell_x0 = cell_col * cell
        cell_y0 = pad_top + cell_row * cell

        fis = self._next_frames(n2, step=seg.get("frame_step", 2))
        full_gt_last = load_gt(self._fd(fis[-1]), gt_v, self.bg, raw=True, res=self.res)

        imgs = []
        # Phase 1: static mosaic
        mosaic = make_mosaic(fi_rep)
        mosaic_labeled = add_label(mosaic, "GT Views (6 Cameras)", fontscale=0.55)
        imgs.extend([mosaic_labeled] * n1)

        # Phase 2: crop-zoom animation (cell rect → full frame)
        for j, fi in enumerate(fis):
            t = j / max(n2 - 1, 1)
            t_ease = t * t * (3 - 2 * t)  # smoothstep

            # Interpolate crop rect from cell bounds → full frame
            cx0 = int(cell_x0 * (1 - t_ease))
            cy0 = int(cell_y0 * (1 - t_ease))
            cs = int(cell + (self.res - cell) * t_ease)
            cs = max(cs, 1)

            mosaic_f = make_mosaic(fi)
            # Crop + resize to full resolution
            crop = mosaic_f[cy0:cy0 + cs, cx0:cx0 + cs]
            if crop.size == 0:
                crop = mosaic_f
            frame = cv2.resize(crop.astype(np.float32), (self.res, self.res),
                               interpolation=cv2.INTER_LINEAR)

            # Blend with full GT in last 20% for quality
            blend_t = max(0.0, (t - 0.8) / 0.2) if t > 0.8 else 0.0
            if blend_t > 0:
                frame = crossfade(frame, full_gt_last, blend_t)

            imgs.append(frame)

        return imgs, prev_fd  # no inference performed

    def _seg_flow_novel_extra(self, seg, n, prev_fd):
        """Temporal flow cycling through multiple extrapolated novel view cameras.

        Config:
            cam_defs: list of {elevation, azimuth, label}
            include_extreme: bool (default True) — include ±80° elevation cams
            frames_per_cam: auto = n // len(cams), or override per cam
            transition_frames: SLERP frames between consecutive cameras
        """
        step = seg.get("frame_step", 2)
        include_extreme = seg.get("include_extreme", True)
        trans_n = seg.get("transition_frames", 20)

        # Default camera definitions
        default_cams = [
            {"elevation": -80, "azimuth": 0,   "label": "Bottom View"},
            {"elevation":  80, "azimuth": 0,   "label": "Top View"},
            {"elevation":   0, "azimuth": 90,  "label": "Side View (Right)"},
            {"elevation": -40, "azimuth": 180, "label": "Rear-Bottom View"},
        ]
        cam_defs = seg.get("cam_defs", default_cams)

        # Filter extreme views if not wanted
        if not include_extreme:
            cam_defs = [c for c in cam_defs if abs(c.get("elevation", 0)) < 70]
        if not cam_defs:
            cam_defs = default_cams[:2]  # fallback

        n_cams = len(cam_defs)
        base_fpk = n // n_cams
        remainder = n - base_fpk * n_cams
        frames_per_cam = [base_fpk + (1 if i < remainder else 0) for i in range(n_cams)]

        fis = self._next_frames(n, step=step)

        imgs = []
        fd = prev_fd
        fi_cursor = 0

        for cam_idx, cam_def in enumerate(cam_defs):
            cam_n = frames_per_cam[cam_idx]
            cam_elev = cam_def.get("elevation", 0)
            cam_azim = cam_def.get("azimuth", 0)
            cam_label = cam_def.get("label", f"View {cam_idx+1}")
            c2w_target, fxfy_target = self._cam_spherical(cam_elev, cam_azim)

            # Smooth transition from previous camera via SLERP
            if cam_idx > 0 and trans_n > 0 and imgs:
                prev_elev = cam_defs[cam_idx - 1].get("elevation", 0)
                prev_azim = cam_defs[cam_idx - 1].get("azimuth", 0)
                c2w_prev, _ = self._cam_spherical(prev_elev, prev_azim)
                trans_c2ws = interpolate_cameras(c2w_prev, c2w_target,
                                                  min(trans_n, cam_n))
            else:
                trans_c2ws = []

            for j in range(cam_n):
                if fi_cursor >= len(fis):
                    break
                fi = fis[fi_cursor]
                fi_cursor += 1
                if fd:
                    free_fd(fd)
                fd = infer_frame(self.model, fi, self.m5, self.kp,
                                 self.n_thresh, self.res, self.device)
                if fd is None:
                    imgs.append(np.zeros((self.res, self.res, 3)))
                    continue

                c2w_use = trans_c2ws[j] if j < len(trans_c2ws) else c2w_target
                img = render_cam(fd["gaussians"], c2w_use, fxfy_target,
                                 self.res, self.res, mask=fd["vis_mask"],
                                 bg=self.bg, device=self.device)
                img = add_label(img, cam_label)
                imgs.append(img)

        return imgs, fd

    def _seg_grid_panorama(self, seg, n, prev_fd):
        """Panoramic grid: evenly distributed views rendered at cell size for efficiency.

        Config:
            num_views: 24 (4x6) or 36 (6x6), default 24
            include_extreme: bool, include ±60°+ elevation rows (default True)
        Phase 1 (n//2): GS-LRM recon at all angles (temporal flow)
        Phase 2 (n - n//2): Continued temporal flow with different label
        """
        step = seg.get("frame_step", 2)
        num_views = seg.get("num_views", 24)
        include_extreme = seg.get("include_extreme", True)
        fis = self._next_frames(n, step=step)

        # Layout: n_rows x 6 columns
        if num_views <= 24:
            n_cols = 6
            elevs_all = [-60, -20, 20, 60]
        else:
            n_cols = 6
            elevs_all = [-60, -30, 0, 30, 60, 85]

        if not include_extreme:
            elevs_all = [e for e in elevs_all if abs(e) < 60]
        if not elevs_all:
            elevs_all = [-30, 0, 30]

        n_rows = len(elevs_all)
        azims_all = [i * (360 // n_cols) for i in range(n_cols)]
        cam_grid = [(e, a) for e in elevs_all for a in azims_all]
        n_cams = len(cam_grid)

        cell = self.res // n_cols  # 128 at 768 with 6 cols
        cell_h = self.res // n_rows if n_rows > 0 else cell
        # Scale intrinsics for cell-size render (faster)
        cell_scale = cell / self.res
        fx_cell = self.res / (2 * np.tan(np.deg2rad(self.hfov) / 2.0)) * cell_scale
        fxfy_cell_base = np.array([fx_cell, fx_cell, cell / 2.0, cell_h / 2.0])

        # Pre-compute all camera poses
        cams_c2w = [self._cam_spherical(e, a)[0] for e, a in cam_grid]

        def make_panorama_grid(fd_use, phase_label):
            canvas = np.ones((self.res, self.res, 3), dtype=np.float32) * np.array(self.bg)
            for idx, (e, a) in enumerate(cam_grid):
                row_i = idx // n_cols
                col_i = idx % n_cols
                c2w_cam = cams_c2w[idx]
                cell_img = render_cam(fd_use["gaussians"], c2w_cam, fxfy_cell_base,
                                       cell, cell_h,
                                       mask=fd_use["vis_mask"], bg=self.bg,
                                       device=self.device)
                if cell_img.shape[:2] != (cell_h, cell):
                    cell_img = cv2.resize(cell_img, (cell, cell_h),
                                          interpolation=cv2.INTER_AREA)
                y0 = row_i * cell_h
                x0 = col_i * cell
                canvas[y0:y0 + cell_h, x0:x0 + cell] = cell_img
            # Draw per-cell elevation labels (must work on uint8)
            canvas_u8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
            for idx, (e, a) in enumerate(cam_grid):
                row_i = idx // n_cols
                col_i = idx % n_cols
                y0 = row_i * cell_h
                x0 = col_i * cell
                label_txt = f"{e:+d}"
                cv2.putText(canvas_u8, label_txt, (x0 + 2, y0 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, (100, 100, 100), 1)
            canvas = canvas_u8 / 255.0
            return add_label(canvas, phase_label, fontscale=0.50)

        n1 = n // 2
        n2 = n - n1
        phase_labels = [
            f"Panoramic Views ({n_cams} angles) — GS-LRM Reconstruction",
            f"Panoramic Views ({n_cams} angles) — Temporal Sequence",
        ]

        imgs = []
        fd = prev_fd
        for i, fi in enumerate(fis):
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue
            phase = 0 if i < n1 else 1
            grid = make_panorama_grid(fd, phase_labels[phase])
            imgs.append(grid)
        return imgs, fd

    def _seg_flow_render_fullbody(self, seg, n, prev_fd):
        """Temporal flow, full body, orbiting camera — final segment."""
        step = seg.get("frame_step", 1)
        fis = self._next_frames(n, step=step)
        elev = seg.get("elevation", 25)
        c2ws, fxfy, w, h = get_orbit_cameras(
            n, elev, self.radius, self.hfov, self.res)
        imgs = []
        fd = prev_fd
        for i, fi in enumerate(fis):
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue
            img = render_cam(fd["gaussians"], c2ws[i], fxfy[i], w, h,
                             mask=fd["vis_mask"], bg=self.bg, device=self.device)
            imgs.append(img)
        return imgs, fd

    def _seg_grid_multiview(self, seg, n, prev_fd):
        """6-view grid: Phase 1 = GT views, Phase 2 = GS-LRM recon, Phase 3 = novel views.

        Layout: 3 columns x 2 rows (256x256 cells at res=768), centered in canvas.
        """
        step = seg.get("frame_step", 2)
        fis = self._next_frames(n, step=step)
        n1 = n // 3       # GT grid
        n2 = n // 3       # Recon grid
        n3 = n - n1 - n2  # Novel grid

        cell = self.res // 3  # 256 for 768
        pad_top = (self.res - cell * 2) // 2  # 128 for 768

        # 6 novel view cameras evenly spaced (2 elevations x 3 azimuths)
        novel_elevs = [20, 20, 20, -40, -40, -40]
        novel_azims = [0, 120, 240, 60, 180, 300]
        novel_cams = [self._cam_spherical(e, a) for e, a in zip(novel_elevs, novel_azims)]

        def make_grid(cells):
            canvas = np.ones((self.res, self.res, 3), dtype=np.float32) * np.array(self.bg)
            for idx, cell_img in enumerate(cells):
                row, col = idx // 3, idx % 3
                sm = cv2.resize(cell_img, (cell, cell), interpolation=cv2.INTER_AREA)
                y0 = pad_top + row * cell
                canvas[y0:y0 + cell, col * cell:(col + 1) * cell] = sm
            return canvas

        def add_phase_label(canvas, text):
            return add_label(canvas, text, fontscale=0.55)

        imgs = []
        fd = prev_fd
        for i, fi in enumerate(fis):
            if fd:
                free_fd(fd)
            fd = infer_frame(self.model, fi, self.m5, self.kp,
                             self.n_thresh, self.res, self.device)
            if fd is None:
                imgs.append(np.zeros((self.res, self.res, 3)))
                continue

            phase = 0 if i < n1 else (1 if i < n1 + n2 else 2)
            cells = []

            if phase == 0:  # GT views (no inference needed for image loading)
                for v in range(6):
                    gt_img = load_gt(fd["frame_dir"], v, self.bg, raw=True, res=self.res)
                    cells.append(gt_img)
            elif phase == 1:  # GS-LRM recon from GT camera positions
                for v in range(6):
                    c2w_v, fxfy_v = self._cam_from_gt_view(fd["frame_dir"], v)
                    r = render_cam(fd["gaussians"], c2w_v, fxfy_v,
                                   self.res, self.res, mask=fd["vis_mask"],
                                   bg=self.bg, device=self.device)
                    cells.append(r)
            else:  # Novel views
                for nc2w, nfxfy in novel_cams:
                    r = render_cam(fd["gaussians"], nc2w, nfxfy,
                                   self.res, self.res, mask=fd["vis_mask"],
                                   bg=self.bg, device=self.device)
                    cells.append(r)

            grid = make_grid(cells)
            phase_labels = ["GT Views (6 Cameras)", "GS-LRM Reconstruction",
                            "Novel Views (6 Angles)"]
            grid = add_phase_label(grid, phase_labels[phase])
            imgs.append(grid)
        return imgs, fd

    def _cam_spherical(self, elev_deg: float, azim_deg: float):
        """Return (c2w 4x4, fxfycxcy 4) for camera at given spherical coords."""
        elev = np.deg2rad(elev_deg)
        azim = np.deg2rad(azim_deg)
        r = self.radius
        z = r * np.sin(elev)
        base = r * np.cos(elev)
        pos = np.array([base * np.cos(azim), base * np.sin(azim), z])
        fwd = -pos / np.linalg.norm(pos)
        up_v = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_v)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack((right, -up, fwd), axis=1)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((R, pos[:, None]), axis=1)
        fx = self.res / (2 * np.tan(np.deg2rad(self.hfov) / 2.0))
        fxfy = np.array([fx, fx, self.res / 2.0, self.res / 2.0])
        return c2w, fxfy

    def _cam_from_gt_view(self, frame_dir: str, view_idx: int):
        """Return (c2w 4x4, fxfycxcy 4) for a GT camera scaled to self.res."""
        c2w = gt_camera_c2w(frame_dir, view_idx)
        raw_fxfy = gt_camera_fxfycxcy(frame_dir, view_idx)  # for 512-px images
        scale = self.res / 512.0
        return c2w, raw_fxfy * scale

    def _ensure_fd(self):
        fi = self._peek_frame()
        return infer_frame(self.model, fi, self.m5, self.kp,
                           self.n_thresh, self.res, self.device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cinematic Sequence v4")
    parser.add_argument("--config", default="mouse_extensions/behavior/cinematic_default.yaml")
    parser.add_argument("--output-dir", default="outputs/viz/cinematic/mouse/latest")
    parser.add_argument("--frame-range", default=None, help="Override: 'start:end[:step]'")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true",
                        help="Cache segment frames to .seg_cache/ for faster re-runs")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Wipe .seg_cache/ before running (implies --use-cache)")
    parser.add_argument("--dual-output-dir", default=None,
                        help="Also write a version without control overlays to this dir")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Config: {cfg_path}")
    else:
        print(f"Config not found: {cfg_path}, using defaults")
        cfg = yaml.safe_load(Path(__file__).parent / "cinematic_default.yaml")

    # CLI overrides
    if args.frame_range:
        cfg["frame_range"] = args.frame_range
    if args.fps:
        cfg["global"]["fps"] = args.fps

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading GS-LRM model...")
    from mouse_extensions.inference.gslrm_pipeline import GSLRMInference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GSLRMInference(
        config_path=cfg["model"]["config"],
        checkpoint_path=cfg["model"]["checkpoint"],
        device=device,
    )
    # Patch num_input_views from cinematic config if explicitly specified
    if "num_input_views" in cfg.get("model", {}):
        model.config.model.num_input_views = cfg["model"]["num_input_views"]
        print(f"  Patched num_input_views: {cfg['model']['num_input_views']}")
    print("  Model loaded")

    pipeline = CinematicPipeline(model, cfg, device)
    pipeline.generate(
        str(out / "cinematic_demo.mp4"),
        use_cache=args.use_cache or args.clear_cache,
        clear_cache=args.clear_cache,
        dual_output_dir=args.dual_output_dir,
    )
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
