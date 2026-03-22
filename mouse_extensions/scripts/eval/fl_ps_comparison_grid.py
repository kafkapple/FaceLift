"""FaceLift vs Pose-Splatter Comparison Grid Generator.

Generates 2x2 comparison grids and side-by-side videos for FL vs PS.
Designed to run on gpu03 with pre-transferred PS renders.

Grid layout (2x2):
  [FL render]         [PS render]
  [FL + KP overlay]   [PS + KP overlay]

Usage:
    cd ~/dev/FaceLift
    python mouse_extensions/scripts/eval/fl_ps_comparison_grid.py \
        --fl_dir outputs/tier_comparison/gslrm_6view_test/samples \
        --ps_dir outputs/comparison/fl_vs_ps_comparison/ps_renders \
        --gt_dir ~/data/preprocessed/FaceLift_mouse/M5 \
        --output_dir outputs/comparison/fl_vs_ps_comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# MAMMAL 22 keypoint names
KP_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip", "R_foot", "R_knee", "R_hip"
]

# Joint group colors (BGR for cv2)
JOINT_COLORS = {
    "head": (0, 200, 255),
    "spine": (0, 255, 0),
    "tail": (255, 100, 0),
    "left_arm": (255, 0, 0),
    "right_arm": (0, 0, 255),
    "left_leg": (255, 128, 0),
    "right_leg": (0, 128, 255),
}

JOINT_GROUPS = {
    "head": [0, 1, 2, 3],
    "spine": [4],
    "tail": [5, 6, 7],
    "left_arm": [8, 9, 10, 11],
    "right_arm": [12, 13, 14, 15],
    "left_leg": [16, 17, 18],
    "right_leg": [19, 20, 21],
}

SKELETON = [
    (2, 0), (2, 1), (2, 3),
    (3, 4), (4, 5), (5, 6), (6, 7),
    (3, 11), (11, 10), (10, 8), (8, 9),
    (3, 15), (15, 14), (14, 12), (12, 13),
    (4, 18), (18, 17), (17, 16),
    (4, 21), (21, 20), (20, 19),
]

# Coordinate transforms — SSOT: mouse_extensions/coordinate_utils.py
from mouse_extensions.coordinate_utils import M5_SCENE_CENTER, M5_DISTANCE_SCALE, mammal_to_gslrm


def get_joint_color(joint_idx):
    for group, indices in JOINT_GROUPS.items():
        if joint_idx in indices:
            return JOINT_COLORS[group]
    return (200, 200, 200)


def load_gt_keypoints_3d(gt_path, frame_idx):
    data = np.load(gt_path)
    kp3d = data["keypoints"]  # (3600, 22, 3) in MAMMAL world (mm)
    return kp3d[frame_idx]


def load_camera(gt_dir, frame_str, view_idx):
    """Load camera parameters from FaceLift preprocessed format.

    Returns K (3x3) and w2c (4x4).
    """
    cam_json = os.path.join(gt_dir, frame_str, "opencv_cameras.json")
    with open(cam_json) as f:
        data = json.load(f)

    cam = data["frames"][view_idx]

    K = np.array([
        [cam["fx"], 0, cam["cx"]],
        [0, cam["fy"], cam["cy"]],
        [0, 0, 1],
    ])

    w2c = np.array(cam["w2c"])  # (4, 4)

    return K, w2c


def project_3d_to_2d(kp3d_mammal, gt_dir, frame_str, view_idx, img_size=512):
    """Project 3D MAMMAL keypoints to 2D image coordinates."""
    K, w2c = load_camera(gt_dir, frame_str, view_idx)

    # Transform MAMMAL mm -> GS-LRM normalized
    kp3d_fl = mammal_to_gslrm(kp3d_mammal)  # (22, 3)

    # Homogeneous
    kp3d_h = np.hstack([kp3d_fl, np.ones((22, 1))])  # (22, 4)

    # World to camera
    cam_coords = (w2c[:3] @ kp3d_h.T).T  # (22, 3)

    # Project to image
    proj = (K @ cam_coords.T).T  # (22, 3)
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    # Visibility
    visible = (cam_coords[:, 2] > 0) & (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)

    kp2d = np.stack([u, v], axis=-1)
    return kp2d, visible


def draw_keypoints(img_np, kp2d, visible, radius=4, thickness=2):
    """Draw keypoints and skeleton on BGR image."""
    img = img_np.copy()

    for i, j in SKELETON:
        if visible[i] and visible[j]:
            color = get_joint_color(i)
            pt1 = (int(kp2d[i, 0]), int(kp2d[i, 1]))
            pt2 = (int(kp2d[j, 0]), int(kp2d[j, 1]))
            cv2.line(img, pt1, pt2, color, thickness=thickness)

    for idx in range(22):
        if visible[idx]:
            color = get_joint_color(idx)
            pt = (int(kp2d[idx, 0]), int(kp2d[idx, 1]))
            cv2.circle(img, pt, radius, color, -1)
            cv2.circle(img, pt, radius, (255, 255, 255), 1)

    return img


def add_label(img_np, text, font_scale=0.6, color=(255, 255, 255)):
    img = img_np.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    org = (8, th + 8)
    cv2.rectangle(img, (org[0] - 4, org[1] - th - 4), (org[0] + tw + 4, org[1] + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return img


def create_2x2_grid(fl_img, ps_img, fl_overlay, ps_overlay, frame_str, view_idx):
    """Create 2x2 grid: [FL|PS] / [FL+KP|PS+KP]."""
    h, w = fl_img.shape[:2]
    pad = 4

    fl_l = add_label(fl_img, f"FaceLift (v{view_idx})")
    ps_l = add_label(ps_img, f"Pose-Splatter (v{view_idx})")
    fl_o = add_label(fl_overlay, "FL + Keypoints")
    ps_o = add_label(ps_overlay, "PS + Keypoints")

    gap_h = np.ones((h, pad, 3), dtype=np.uint8) * 40
    gap_v = np.ones((pad, w * 2 + pad, 3), dtype=np.uint8) * 40

    top = np.hstack([fl_l, gap_h, ps_l])
    bottom = np.hstack([fl_o, gap_h, ps_o])
    grid = np.vstack([top, gap_v, bottom])
    return grid


def process_frame(fl_dir, ps_dir, gt_kp_path, gt_dir, frame_str, views, output_dir):
    """Generate comparison for one frame."""
    frame_idx = int(frame_str)
    kp3d = load_gt_keypoints_3d(gt_kp_path, frame_idx)

    cam_json = os.path.join(gt_dir, frame_str, "opencv_cameras.json")
    if not os.path.exists(cam_json):
        print(f"  Camera not found: {cam_json}, skipping")
        return []

    grids = []

    for view_idx in views:
        view_str = f"{view_idx:02d}"

        fl_path = os.path.join(fl_dir, frame_str, f"render_view_{view_str}.png")
        ps_path = os.path.join(ps_dir, frame_str, f"render_view_{view_str}.png")

        if not os.path.exists(fl_path) or not os.path.exists(ps_path):
            continue

        fl_img = cv2.imread(fl_path)
        ps_img = cv2.imread(ps_path)

        h, w = fl_img.shape[:2]
        if ps_img.shape[:2] != (h, w):
            ps_img = cv2.resize(ps_img, (w, h))

        kp2d, visible = project_3d_to_2d(kp3d, gt_dir, frame_str, view_idx, w)

        fl_overlay = draw_keypoints(fl_img, kp2d, visible)
        ps_overlay = draw_keypoints(ps_img, kp2d, visible)

        grid = create_2x2_grid(fl_img, ps_img, fl_overlay, ps_overlay, frame_str, view_idx)
        grids.append((view_idx, grid))

        # Save individual grid
        frame_out = os.path.join(output_dir, "grids", frame_str)
        os.makedirs(frame_out, exist_ok=True)
        cv2.imwrite(os.path.join(frame_out, f"grid_2x2_view{view_str}.png"), grid)

    # Combined 6-view (2 rows x 3 cols)
    if len(grids) >= 6:
        frame_out = os.path.join(output_dir, "grids", frame_str)
        row1 = np.hstack([g for _, g in grids[:3]])
        row2 = np.hstack([g for _, g in grids[3:6]])
        combined = np.vstack([row1, row2])
        scale = min(1.0, 4096.0 / max(combined.shape[:2]))
        if scale < 1.0:
            combined = cv2.resize(combined, None, fx=scale, fy=scale)
        cv2.imwrite(os.path.join(frame_out, "combined_6view_2x2.png"), combined)

    print(f"  Frame {frame_str}: {len(grids)} views")
    return grids


def create_videos(fl_dir, ps_dir, gt_kp_path, gt_dir, frames, output_dir, view_idx=0, fps=2):
    """Create comparison videos: clean and with keypoint overlay."""
    first_fl = cv2.imread(os.path.join(fl_dir, frames[0], f"render_view_{view_idx:02d}.png"))
    if first_fl is None:
        print("Cannot create video: FL images not found")
        return

    h, w = first_fl.shape[:2]
    vid_w = w * 2 + 4

    for suffix, with_kp in [("clean", False), ("overlay", True)]:
        out_path = os.path.join(output_dir, f"fl_vs_ps_view{view_idx:02d}_{suffix}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (vid_w, h))

        for frame_str in frames:
            fl_path = os.path.join(fl_dir, frame_str, f"render_view_{view_idx:02d}.png")
            ps_path = os.path.join(ps_dir, frame_str, f"render_view_{view_idx:02d}.png")

            fl_img = cv2.imread(fl_path)
            ps_img = cv2.imread(ps_path)
            if fl_img is None or ps_img is None:
                continue

            if ps_img.shape[:2] != (h, w):
                ps_img = cv2.resize(ps_img, (w, h))

            if with_kp:
                frame_idx = int(frame_str)
                kp3d = load_gt_keypoints_3d(gt_kp_path, frame_idx)
                kp2d, visible = project_3d_to_2d(kp3d, gt_dir, frame_str, view_idx, w)
                fl_img = draw_keypoints(fl_img, kp2d, visible)
                ps_img = draw_keypoints(ps_img, kp2d, visible)

            fl_l = add_label(fl_img, f"FaceLift f{frame_str}")
            ps_l = add_label(ps_img, f"PS f{frame_str}")
            gap = np.ones((h, 4, 3), dtype=np.uint8) * 40
            frame_out = np.hstack([fl_l, gap, ps_l])
            writer.write(frame_out)

        writer.release()
        print(f"  Video: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="FL vs PS Comparison Grid Generator")
    parser.add_argument("--fl_dir", required=True)
    parser.add_argument("--ps_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    from mouse_extensions.paths import KP_22
    parser.add_argument("--gt_kp_path",
                        default=str(KP_22))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames", nargs="+",
                        default=["003240", "003300", "003360", "003420", "003480", "003540", "003599"])
    parser.add_argument("--views", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--video_view", type=int, default=0)
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"FL dir: {args.fl_dir}")
    print(f"PS dir: {args.ps_dir}")
    print(f"GT dir: {args.gt_dir}")
    print(f"Frames: {args.frames}")
    print(f"Views: {args.views}")
    print()

    # Step 1: Per-frame grids
    print("=== Step 1: Per-frame 2x2 comparison grids ===")
    for frame_str in args.frames:
        process_frame(args.fl_dir, args.ps_dir, args.gt_kp_path,
                      args.gt_dir, frame_str, args.views, args.output_dir)

    # Step 2: Representative grid (middle frame, view 0)
    print("\n=== Step 2: Representative grid ===")
    mid = args.frames[len(args.frames) // 2]
    frame_idx = int(mid)
    kp3d = load_gt_keypoints_3d(args.gt_kp_path, frame_idx)

    fl_img = cv2.imread(os.path.join(args.fl_dir, mid, "render_view_00.png"))
    ps_img = cv2.imread(os.path.join(args.ps_dir, mid, "render_view_00.png"))

    if fl_img is not None and ps_img is not None:
        h, w = fl_img.shape[:2]
        if ps_img.shape[:2] != (h, w):
            ps_img = cv2.resize(ps_img, (w, h))
        kp2d, visible = project_3d_to_2d(kp3d, args.gt_dir, mid, 0, w)
        fl_ov = draw_keypoints(fl_img, kp2d, visible)
        ps_ov = draw_keypoints(ps_img, kp2d, visible)
        grid = create_2x2_grid(fl_img, ps_img, fl_ov, ps_ov, mid, 0)
        out_path = os.path.join(args.output_dir, "representative_2x2_grid.png")
        cv2.imwrite(out_path, grid)
        print(f"  Saved: {out_path}")

    # Step 3: Videos
    print("\n=== Step 3: Comparison videos ===")
    create_videos(args.fl_dir, args.ps_dir, args.gt_kp_path,
                  args.gt_dir, args.frames, args.output_dir,
                  view_idx=args.video_view, fps=args.fps)

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
