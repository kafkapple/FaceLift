# no-split: single cinematic pipeline with 8 tightly-coupled segment handlers sharing state
"""Cinematic Sequence Visualization v4: config-driven, temporal-flow architecture.

Segment types:
  flow_gt        — GT RGB, temporal flow
  flow_mask      — FG mask overlay, temporal flow
  flow_render    — GS-LRM render (GT camera), temporal flow
  freeze_orbit   — Freeze time, 360° turntable
  freeze_elev    — Freeze time, elevation arc
  flow_orbit     — Temporal flow + orbiting camera
  flow_bodypart  — Temporal flow + body-part cycling
  freeze_bodypart — Freeze time + body-part cycling + orbit

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

from mouse_extensions.constants import MAMMAL_KP_COLORS, SKELETON_BONES
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

    Azimuth: CW from +Y (matching get_turntable_cameras convention).
    Elevation: angle from XY plane (positive = above).
    """
    cam = load_camera(str(Path(frame_dir) / "opencv_cameras.json"), view_idx)
    w2c = np.array(cam["w2c"])
    c2w = np.linalg.inv(w2c)
    x, y, z = c2w[0, 3], c2w[1, 3], c2w[2, 3]
    radius = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.degrees(np.arcsin(z / max(radius, 1e-8)))
    azimuth = (np.degrees(-np.arctan2(-x, y)) + 360) % 360
    return radius, elevation, azimuth


def get_orbit_cameras(n_frames: int, elevation: float, radius: float = 2.7,
                      hfov: float = 50, resolution: int = 512,
                      mode: str = "turntable", elevation_end: float = None,
                      match_gt: bool = False, gt_frame_dir: str = None,
                      gt_view: int = 0):
    """Generate camera poses via get_turntable_cameras.

    If match_gt=True, uses GT camera's actual radius/elevation/azimuth
    so orbit starts exactly from the GT camera position.
    """
    from mouse_extensions.visualization.camera_utils import get_turntable_cameras

    start_azimuth = None
    if match_gt and gt_frame_dir:
        gt_r, gt_elev, gt_az = gt_camera_spherical(gt_frame_dir, gt_view)
        radius = gt_r
        if mode == "turntable":
            elevation = gt_elev
        start_azimuth = gt_az

    kw = dict(hfov=hfov, num_views=n_frames, w=resolution, h=resolution,
              radius=radius, elevation=elevation, trajectory_mode=mode)
    if elevation_end is not None:
        kw["elevation_end"] = elevation_end
    w, h, _, fxfy, c2ws = get_turntable_cameras(**kw)

    if start_azimuth is not None and mode == "turntable":
        # get_turntable_cameras starts at 270° CW. Find offset.
        default_start = 270.0
        offset = start_azimuth - default_start
        # Find closest frame index to offset
        step = 360.0 / n_frames
        shift = int(round(offset / step)) % n_frames
        if shift != 0:
            c2ws = np.roll(c2ws, -shift, axis=0)
            fxfy = np.roll(fxfy, -shift, axis=0)

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

def load_gt(fd_path, view, bg_color=(1.0, 1.0, 1.0), raw=False):
    """Load GT RGB. raw=True keeps original background, False composites onto bg_color."""
    from PIL import Image
    p = Path(fd_path) / "images" / f"cam_{view:03d}.png"
    img = np.array(Image.open(p))
    rgb = img[:, :, :3] / 255.0
    if not raw and img.shape[-1] == 4:
        alpha = img[:, :, 3:4] / 255.0
        bg = np.array(bg_color).reshape(1, 1, 3)
        rgb = rgb * alpha + bg * (1 - alpha)
    return rgb


def load_mask_overlay(fd_path, view, border_color=None, bg_color=(1.0, 1.0, 1.0)):
    """GT image with foreground highlight on bg_color background."""
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
    return out


# Constants imported from mouse_extensions.constants (SSOT, 2026-03-23)


def overlay_keypoints(img, kp_2d, kp_valid):
    """Fast cv2-based keypoint + skeleton overlay. img: float 0-1, returns float 0-1."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    H, W = u8.shape[:2]
    # Skeleton bones
    for a, b in SKELETON_BONES:
        if kp_valid[a] and kp_valid[b]:
            p1 = (int(kp_2d[a, 0]), int(kp_2d[a, 1]))
            p2 = (int(kp_2d[b, 0]), int(kp_2d[b, 1]))
            cv2.line(u8, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
    # Keypoints
    for i in range(len(kp_2d)):
        if not kp_valid[i]:
            continue
        x, y = int(kp_2d[i, 0]), int(kp_2d[i, 1])
        if 0 <= x < W and 0 <= y < H:
            c = MAMMAL_KP_COLORS.get(i, (255, 255, 255))
            cv2.circle(u8, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)  # outline
            cv2.circle(u8, (x, y), 3, c, -1, cv2.LINE_AA)
    return u8 / 255.0


def project_kp_to_view(kp_3d, frame_dir, view_idx):
    """Project 3D keypoints to 2D for a given view."""
    from mouse_extensions.behavior.view_projected_filtering import project_points_to_2d
    cam = load_camera(str(Path(frame_dir) / "opencv_cameras.json"), view_idx)
    uv, valid = project_points_to_2d(
        kp_3d, np.array(cam["w2c"]), cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    in_img = valid & (uv[:, 0] >= 0) & (uv[:, 0] < cam["w"]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["h"])
    return uv, in_img


def add_label(img, text, fontscale=0.6, thickness=1):
    """Add text label to top-left of image."""
    h, w = img.shape[:2]
    img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    # Shadow
    cv2.putText(img_u8, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                fontscale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # White text
    cv2.putText(img_u8, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                fontscale, (255, 255, 255), thickness, cv2.LINE_AA)
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

    def _next_frames(self, n):
        """Consume n frames from the global cursor, cycling if needed."""
        out = []
        for _ in range(n):
            out.append(self.all_frames[self.fi % len(self.all_frames)])
            self.fi += 1
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

    def generate(self, output_path: str, side_by_side: bool = False):
        segments = self.cfg["segments"]
        all_imgs = []
        last_img = None
        last_fd = None

        for si, seg in enumerate(segments):
            stype = seg["type"]
            label = seg.get("label", "")
            dur = seg.get("duration", 2.0)
            n = self._n(dur)

            print(f"\n  [{si+1}/{len(segments)}] {stype}: {dur}s ({n}f) — {label}")

            handler = getattr(self, f"_seg_{stype}", None)
            if handler is None:
                print(f"    WARNING: unknown type '{stype}', skipping")
                continue

            frames, last_fd = handler(seg, n, last_fd)

            # Apply labels
            if label:
                frames = [add_label(f, label) for f in frames]

            # Crossfade from previous segment
            if last_img is not None and self.cf_frames > 0 and frames:
                cf = min(self.cf_frames, len(frames))
                for i in range(cf):
                    t = (i + 1) / cf
                    frames[i] = crossfade(last_img, frames[i], t)

            all_imgs.extend(frames)
            if frames:
                last_img = frames[-1]

        # Write video
        dur = len(all_imgs) / self.fps
        print(f"\n  Writing: {len(all_imgs)} frames @ {self.fps}fps = {dur:.1f}s")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(output_path, fourcc, self.fps, (self.res, self.res))
        for img in all_imgs:
            u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            w.write(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
        w.release()
        print(f"  Saved: {output_path}")

    # --- Segment handlers ---

    def _seg_flow_gt(self, seg, n, prev_fd):
        fis = self._next_frames(n)
        raw = seg.get("raw", False)
        imgs = [self._apply_kp(load_gt(self._fd(fi), self.gt_view, self.bg, raw=raw), fi)
                for fi in fis]
        return imgs, prev_fd

    def _seg_flow_mask(self, seg, n, prev_fd):
        fis = self._next_frames(n)
        imgs = [self._apply_kp(
            load_mask_overlay(self._fd(fi), self.gt_view,
                              border_color=self.border_color, bg_color=self.bg), fi)
                for fi in fis]
        return imgs, prev_fd

    def _seg_flow_render(self, seg, n, prev_fd):
        fis = self._next_frames(n)
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
            imgs.append(img)
        return imgs, fd  # keep last fd for freeze segments

    def _seg_freeze_orbit(self, seg, n, prev_fd):
        fd = prev_fd or self._ensure_fd()
        elev = seg.get("elevation", 20)
        transition_frames = seg.get("transition_frames", 15)  # smooth GT->orbit

        # Generate orbit cameras
        orbit_n = max(1, n - transition_frames)
        c2ws, fxfy, w, h = get_orbit_cameras(
            orbit_n, elev, self.radius, self.hfov, self.res)

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
    pipeline.generate(str(out / "cinematic_demo.mp4"))
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
