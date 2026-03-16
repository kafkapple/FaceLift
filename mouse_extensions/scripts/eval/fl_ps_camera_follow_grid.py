"""FL vs PS Camera Follow Comparison Grid Video.

Creates 2x2 comparison video:
  [FL clean]         [PS clean]
  [FL + KP overlay]  [PS + KP overlay]

Uses MAMMAL canonical keypoint definitions via keypoint_viz module.
See: mouse_extensions/docs/KEYPOINT_VIZ.md

Usage:
    cd ~/dev/FaceLift
    python mouse_extensions/scripts/eval/fl_ps_camera_follow_grid.py
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for keypoint_viz import
sys.path.insert(0, str(Path(__file__).parent))
from keypoint_viz import (
    draw_keypoints, project_kp3d_to_camera, add_label, draw_legend,
)


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

    if hasattr(config_raw, "fx"):
        fx, fy, cx, cy = config_raw.fx, config_raw.fy, config_raw.cx, config_raw.cy
    else:
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

            ret, fl_frame = fl_cap.read()
            if not ret:
                break

            ps_path = os.path.join(args.ps_dir, f"frame_{frame_idx:06d}.png")
            if not os.path.exists(ps_path):
                continue
            ps_frame = cv2.imread(ps_path)

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

            fl_l = add_label(fl_img, f"FaceLift f{frame_idx}")
            ps_l = add_label(ps_img, f"Pose-Splatter f{frame_idx}")

            if with_kp:
                fl_ol = add_label(fl_overlay, "FL + Keypoints")
                ps_ol = add_label(ps_overlay, "PS + Keypoints")
            else:
                fl_ol = fl_l
                ps_ol = ps_l

            gap_h = np.ones((S, pad, 3), dtype=np.uint8) * 40
            gap_v = np.ones((pad, S * 2 + pad, 3), dtype=np.uint8) * 40

            if with_kp:
                top = np.hstack([fl_l, gap_h, ps_l])
                bottom = np.hstack([fl_ol, gap_h, ps_ol])
                grid = np.vstack([top, gap_v, bottom])
            else:
                grid_1x2 = np.hstack([fl_l, gap_h, ps_l])
                grid = np.vstack([grid_1x2, gap_v, grid_1x2.copy()])

            writer.write(grid)
            written += 1

        writer.release()
        print(f"  Video: {out_path} ({written} frames)")

    fl_cap.release()

    # Save representative frame with legend
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
        grid = draw_legend(grid)

        out_path = os.path.join(args.output_dir, "fl_vs_ps_camera_follow_representative.png")
        cv2.imwrite(out_path, grid)
        print(f"  Representative: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
