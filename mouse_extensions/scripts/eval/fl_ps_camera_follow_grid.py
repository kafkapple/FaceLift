"""FL vs PS Camera Follow Comparison Grid Video.

Creates 2x2 comparison video:
  [FL clean]         [PS clean]
  [FL + KP overlay]  [PS + KP overlay]

Usage:
    python /tmp/fl_ps_camera_follow_grid.py
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# Keypoint constants (same as fl_ps_comparison_grid.py)
KP_NAMES = [
    "L_ear", "R_ear", "nose", "neck", "body_middle", "tail_root",
    "tail_middle", "tail_end", "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
    "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
    "L_foot", "L_knee", "L_hip", "R_foot", "R_knee", "R_hip"
]

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

M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])
M5_DISTANCE_SCALE = 2.7 / 307.785


def get_joint_color(joint_idx):
    for group, indices in JOINT_GROUPS.items():
        if joint_idx in indices:
            return JOINT_COLORS[group]
    return (200, 200, 200)


def draw_keypoints(img_np, kp2d, visible, radius=4, thickness=2):
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


def project_kp3d_to_camera(kp3d_mammal, c2w, fx, fy, cx, cy, img_size=512):
    """Project 3D keypoints using camera follow c2w and intrinsics."""
    # MAMMAL -> FaceLift normalized
    kp3d_fl = (kp3d_mammal - M5_SCENE_CENTER) * M5_DISTANCE_SCALE  # (22, 3)

    # c2w -> w2c
    w2c = np.linalg.inv(c2w)

    # Homogeneous
    kp3d_h = np.hstack([kp3d_fl, np.ones((22, 1))])  # (22, 4)

    # World to camera
    cam_coords = (w2c[:3] @ kp3d_h.T).T  # (22, 3)

    # Project
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    proj = (K @ cam_coords.T).T
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    visible = (cam_coords[:, 2] > 0) & (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
    kp2d = np.stack([u, v], axis=-1)
    return kp2d, visible


def add_label(img_np, text, font_scale=0.6, color=(255, 255, 255)):
    img = img_np.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    org = (8, th + 8)
    cv2.rectangle(img, (org[0] - 4, org[1] - th - 4), (org[0] + tw + 4, org[1] + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fl_video",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/camera_follow_face_clean.mp4")
    parser.add_argument("--ps_dir",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/ps_renders")
    parser.add_argument("--traj_path",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/trajectory.npz")
    parser.add_argument("--gt_kp_path",
                        default="/node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz")
    parser.add_argument("--output_dir",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--target_size", type=int, default=512)
    args = parser.parse_args()

    # Load trajectory
    traj = np.load(args.traj_path, allow_pickle=True)
    c2ws = traj["c2ws"]  # (360, 4, 4)
    frame_indices = traj["frame_indices"]  # (360,)
    config_raw = traj["config"].item()
    # config may be a CameraFollowConfig object or string repr
    if hasattr(config_raw, "fx"):
        fx, fy, cx, cy = config_raw.fx, config_raw.fy, config_raw.cx, config_raw.cy
    else:
        # Parse from string repr: CameraFollowConfig(..., fx=549.0, ...)
        import re
        s = str(config_raw)
        fx = float(re.search(r"fx=([\d.]+)", s).group(1))
        fy = float(re.search(r"fy=([\d.]+)", s).group(1))
        cx = float(re.search(r"cx=([\d.]+)", s).group(1))
        cy = float(re.search(r"cy=([\d.]+)", s).group(1))
    print(f"Trajectory: {len(frame_indices)} frames, fx={fx}, fy={fy}")

    # Load GT keypoints
    kp_data = np.load(args.gt_kp_path)
    kp3d_all = kp_data["keypoints"]  # (3600, 22, 3)
    print(f"GT keypoints: {kp3d_all.shape}")

    # Open FL video
    fl_cap = cv2.VideoCapture(args.fl_video)
    fl_total = int(fl_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FL video: {fl_total} frames")

    # Check PS renders
    ps_frames = sorted([f for f in os.listdir(args.ps_dir) if f.endswith(".png")])
    print(f"PS renders: {len(ps_frames)} frames")

    S = args.target_size
    pad = 4
    vid_w = S * 2 + pad
    vid_h = S * 2 + pad

    # Create videos: clean and overlay
    for suffix, with_kp in [("clean", False), ("overlay", True)]:
        out_path = os.path.join(args.output_dir, f"fl_vs_ps_camera_follow_{suffix}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (vid_w, vid_h))

        fl_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        written = 0

        for i in range(len(frame_indices)):
            frame_idx = int(frame_indices[i])
            c2w = c2ws[i]

            # Read FL frame
            ret, fl_frame = fl_cap.read()
            if not ret:
                break

            # Read PS frame
            ps_path = os.path.join(args.ps_dir, f"frame_{frame_idx:06d}.png")
            if not os.path.exists(ps_path):
                continue
            ps_frame = cv2.imread(ps_path)

            # Resize to target
            fl_img = cv2.resize(fl_frame, (S, S))
            ps_img = cv2.resize(ps_frame, (S, S))

            if with_kp:
                kp3d = kp3d_all[frame_idx]
                kp2d, visible = project_kp3d_to_camera(kp3d, c2w, fx, fy, cx, cy, S)
                fl_overlay = draw_keypoints(fl_img, kp2d, visible)
                ps_overlay = draw_keypoints(ps_img, kp2d, visible)
            else:
                fl_overlay = fl_img
                ps_overlay = ps_img

            # Labels
            fl_l = add_label(fl_img, f"FaceLift f{frame_idx}")
            ps_l = add_label(ps_img, f"Pose-Splatter f{frame_idx}")

            if with_kp:
                fl_ol = add_label(fl_overlay, "FL + Keypoints")
                ps_ol = add_label(ps_overlay, "PS + Keypoints")
            else:
                fl_ol = fl_l
                ps_ol = ps_l

            # 2x2 grid
            gap_h = np.ones((S, pad, 3), dtype=np.uint8) * 40
            gap_v = np.ones((pad, S * 2 + pad, 3), dtype=np.uint8) * 40

            if with_kp:
                top = np.hstack([fl_l, gap_h, ps_l])
                bottom = np.hstack([fl_ol, gap_h, ps_ol])
                grid = np.vstack([top, gap_v, bottom])
            else:
                # For clean, just side-by-side (1x2)
                grid_1x2 = np.hstack([fl_l, gap_h, ps_l])
                # Pad to same height as overlay version
                grid = np.vstack([grid_1x2, gap_v, grid_1x2.copy()])

            writer.write(grid)
            written += 1

        writer.release()
        print(f"  Video: {out_path} ({written} frames)")

    fl_cap.release()

    # Also save representative frame
    rep_idx = len(frame_indices) // 2
    frame_idx = int(frame_indices[rep_idx])
    c2w = c2ws[rep_idx]

    fl_cap2 = cv2.VideoCapture(args.fl_video)
    fl_cap2.set(cv2.CAP_PROP_POS_FRAMES, rep_idx)
    ret, fl_frame = fl_cap2.read()
    fl_cap2.release()

    ps_path = os.path.join(args.ps_dir, f"frame_{frame_idx:06d}.png")
    if ret and os.path.exists(ps_path):
        ps_frame = cv2.imread(ps_path)
        fl_img = cv2.resize(fl_frame, (S, S))
        ps_img = cv2.resize(ps_frame, (S, S))

        kp3d = kp3d_all[frame_idx]
        kp2d, visible = project_kp3d_to_camera(kp3d, c2w, fx, fy, cx, cy, S)
        fl_ov = draw_keypoints(fl_img, kp2d, visible)
        ps_ov = draw_keypoints(ps_img, kp2d, visible)

        fl_l = add_label(fl_img, f"FaceLift f{frame_idx}")
        ps_l = add_label(ps_img, f"Pose-Splatter f{frame_idx}")
        fl_ol = add_label(fl_ov, "FL + Keypoints")
        ps_ol = add_label(ps_ov, "PS + Keypoints")

        gap_h = np.ones((S, pad, 3), dtype=np.uint8) * 40
        gap_v = np.ones((pad, S * 2 + pad, 3), dtype=np.uint8) * 40
        top = np.hstack([fl_l, gap_h, ps_l])
        bottom = np.hstack([fl_ol, gap_h, ps_ol])
        grid = np.vstack([top, gap_v, bottom])

        out_path = os.path.join(args.output_dir, "fl_vs_ps_camera_follow_representative.png")
        cv2.imwrite(out_path, grid)
        print(f"  Representative: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
