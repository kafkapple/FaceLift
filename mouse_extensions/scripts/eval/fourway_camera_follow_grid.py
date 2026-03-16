"""4-Way Camera Follow Comparison Grid Video.

Creates 2x2 comparison video:
  [FL 6v]  [FL 5v]
  [PS 5v]  [PS 6v]

Usage:
    cd ~/dev/FaceLift
    python /tmp/fourway_camera_follow_grid.py
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/joon/dev/FaceLift/mouse_extensions/scripts/eval")
from keypoint_viz import (
    draw_keypoints, project_kp3d_to_camera, add_label, draw_legend,
)


def read_video_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def read_png_frame(directory, frame_idx):
    path = os.path.join(directory, f"frame_{frame_idx:06d}.png")
    if os.path.exists(path):
        return cv2.imread(path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fl6v_video",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/camera_follow_face_clean.mp4")
    parser.add_argument("--fl5v_video",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/fl_5v_renders/camera_follow_face_clean.mp4")
    parser.add_argument("--ps5v_dir",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/ps_renders")
    parser.add_argument("--ps6v_dir",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/ps_6v_renders")
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
    c2ws = traj["c2ws"]
    frame_indices = traj["frame_indices"]

    import re
    s = str(traj["config"].item())
    fx = float(re.search(r"fx=([\d.]+)", s).group(1))
    fy = float(re.search(r"fy=([\d.]+)", s).group(1))
    cx = float(re.search(r"cx=([\d.]+)", s).group(1))
    cy = float(re.search(r"cy=([\d.]+)", s).group(1))
    print(f"Trajectory: {len(frame_indices)} frames, fx={fx:.1f}")

    # Load GT keypoints
    kp3d_all = np.load(args.gt_kp_path)["keypoints"]
    print(f"GT keypoints: {kp3d_all.shape}")

    # Open video sources
    fl6v_cap = cv2.VideoCapture(args.fl6v_video)
    fl5v_cap = cv2.VideoCapture(args.fl5v_video)
    print(f"FL 6v video: {int(fl6v_cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
    print(f"FL 5v video: {int(fl5v_cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
    print(f"PS 5v dir: {len(os.listdir(args.ps5v_dir))} files")
    print(f"PS 6v dir: {len(os.listdir(args.ps6v_dir))} files")

    S = args.target_size
    pad = 4
    vid_w = S * 2 + pad
    vid_h = S * 2 + pad

    labels = ["FaceLift 6v (GT upper bound)", "FaceLift 5v (PSNR 23.02)",
              "Pose-Splatter 5v", "Pose-Splatter 6v (PSNR 29.22)"]

    # Generate clean and overlay versions
    for suffix, with_kp in [("clean", False), ("overlay", True)]:
        out_path = os.path.join(args.output_dir, f"4way_camera_follow_{suffix}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (vid_w, vid_h))

        fl6v_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fl5v_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        written = 0

        for i in range(len(frame_indices)):
            frame_idx = int(frame_indices[i])

            # Read all 4 sources
            ret6, fl6v_frame = fl6v_cap.read()
            ret5, fl5v_frame = fl5v_cap.read()
            ps5v_frame = read_png_frame(args.ps5v_dir, frame_idx)
            ps6v_frame = read_png_frame(args.ps6v_dir, frame_idx)

            if not ret6 or not ret5 or ps5v_frame is None or ps6v_frame is None:
                # Skip if any source missing
                if not ret6 or not ret5:
                    break
                continue

            imgs = [
                cv2.resize(fl6v_frame, (S, S)),
                cv2.resize(fl5v_frame, (S, S)),
                cv2.resize(ps5v_frame, (S, S)),
                cv2.resize(ps6v_frame, (S, S)),
            ]

            if with_kp:
                kp3d = kp3d_all[frame_idx]
                kp2d, visible = project_kp3d_to_camera(kp3d, c2ws[i], fx, fy, cx, cy, S)
                imgs = [draw_keypoints(img, kp2d, visible) for img in imgs]

            # Add labels
            lab = [f"{labels[j]} f{frame_idx}" for j in range(4)]
            imgs = [add_label(imgs[j], lab[j]) for j in range(4)]

            # Assemble 2x2 grid
            gap_h = np.ones((S, pad, 3), dtype=np.uint8) * 40
            gap_v = np.ones((pad, vid_w, 3), dtype=np.uint8) * 40
            top = np.hstack([imgs[0], gap_h, imgs[1]])
            bottom = np.hstack([imgs[2], gap_h, imgs[3]])
            grid = np.vstack([top, gap_v, bottom])

            writer.write(grid)
            written += 1

        writer.release()
        print(f"  Video: {out_path} ({written} frames)")

    fl6v_cap.release()
    fl5v_cap.release()

    # Save representative frame with legend
    rep_idx = len(frame_indices) // 2
    frame_idx = int(frame_indices[rep_idx])

    fl6v_cap2 = cv2.VideoCapture(args.fl6v_video)
    fl5v_cap2 = cv2.VideoCapture(args.fl5v_video)

    fl6v_img = read_video_frame(fl6v_cap2, rep_idx)
    fl5v_img = read_video_frame(fl5v_cap2, rep_idx)
    ps5v_img = read_png_frame(args.ps5v_dir, frame_idx)
    ps6v_img = read_png_frame(args.ps6v_dir, frame_idx)

    fl6v_cap2.release()
    fl5v_cap2.release()

    if all(x is not None for x in [fl6v_img, fl5v_img, ps5v_img, ps6v_img]):
        imgs = [cv2.resize(x, (S, S)) for x in [fl6v_img, fl5v_img, ps5v_img, ps6v_img]]

        kp3d = kp3d_all[frame_idx]
        kp2d, visible = project_kp3d_to_camera(kp3d, c2ws[rep_idx], fx, fy, cx, cy, S)
        imgs_kp = [draw_keypoints(img.copy(), kp2d, visible) for img in imgs]

        # Clean version
        imgs_l = [add_label(imgs[j], labels[j]) for j in range(4)]
        gap_h = np.ones((S, pad, 3), dtype=np.uint8) * 40
        gap_v = np.ones((pad, vid_w, 3), dtype=np.uint8) * 40
        top = np.hstack([imgs_l[0], gap_h, imgs_l[1]])
        bottom = np.hstack([imgs_l[2], gap_h, imgs_l[3]])
        grid_clean = np.vstack([top, gap_v, bottom])

        # Overlay version
        imgs_kp_l = [add_label(imgs_kp[j], f"{labels[j]} + KP") for j in range(4)]
        top_kp = np.hstack([imgs_kp_l[0], gap_h, imgs_kp_l[1]])
        bottom_kp = np.hstack([imgs_kp_l[2], gap_h, imgs_kp_l[3]])
        grid_overlay = np.vstack([top_kp, gap_v, bottom_kp])

        # Stack clean and overlay vertically
        gap_big = np.ones((8, vid_w, 3), dtype=np.uint8) * 40
        combined = np.vstack([grid_clean, gap_big, grid_overlay])
        combined = draw_legend(combined)

        out_path = os.path.join(args.output_dir, "4way_representative.png")
        cv2.imwrite(out_path, combined)
        print(f"  Representative: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
