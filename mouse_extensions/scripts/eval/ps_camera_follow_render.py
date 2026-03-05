"""Render Pose-Splatter from FL camera follow trajectory.

Loads FL trajectory (c2ws in FL coords), transforms to PS coords,
runs PS forward pass per frame, then renders from the trajectory camera.

Usage:
    cd ~/dev/pose-splatter
    CUDA_VISIBLE_DEVICES=4 python /tmp/ps_camera_follow_render.py \
        --output_dir /home/joon/dev/FaceLift/outputs/camera_follow/ps_renders
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/joon/dev/pose-splatter")
from src.utils import get_cam_params
from src.data import FrameDataset
from src.model import PoseSplatter


def load_fl_trajectory(traj_path):
    """Load FL camera follow trajectory."""
    d = np.load(traj_path, allow_pickle=True)
    c2ws = d["c2ws"]  # (N, 4, 4)
    frame_indices = d["frame_indices"]  # (N,)
    return c2ws, frame_indices


def load_fl_to_ps_transform(transform_path):
    """Load FL->PS coordinate transform."""
    d = np.load(transform_path)
    return d["scale"], d["rotation"], d["translation"]


def transform_c2w_fl_to_ps(c2w_fl, scale, R_fl2ps, t_fl2ps):
    """Transform a c2w matrix from FL coords to PS coords.

    FL c2w columns: [R_cam | t_cam] where t_cam is camera position in FL world.
    We need to transform both position and orientation.
    """
    c2w_ps = np.eye(4)

    # Transform camera position
    pos_fl = c2w_fl[:3, 3]
    pos_ps = scale * (R_fl2ps @ pos_fl) + t_fl2ps

    # Transform camera orientation (rotation only, no scale)
    rot_fl = c2w_fl[:3, :3]
    rot_ps = R_fl2ps @ rot_fl

    c2w_ps[:3, :3] = rot_ps
    c2w_ps[:3, 3] = pos_ps
    return c2w_ps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/trajectory.npz")
    parser.add_argument("--transform_path",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/fl_to_ps_transform.npz")
    parser.add_argument("--ps_checkpoint",
                        default="/home/joon/dev/pose-splatter/output/m5_baseline_gs/latest/checkpoint.pt")
    parser.add_argument("--ps_config",
                        default="/home/joon/dev/pose-splatter/output/m5_baseline_gs/latest/config.json")
    parser.add_argument("--preprocess_dir",
                        default="/home/joon/data/preprocessed/markerless_mouse_1_nerf/m5_for_ps_fj1")
    parser.add_argument("--output_dir",
                        default="/home/joon/dev/FaceLift/outputs/camera_follow/ps_renders")
    parser.add_argument("--resolution", type=int, default=512,
                        help="PS render resolution (512 for ds=2)")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.ps_config) as f:
        config_dict = json.load(f)

    # Create simple config object
    class Config:
        pass
    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)

    # Camera params
    ds = getattr(config, "image_downsample", 4)
    K_ps, E_ps, _ = get_cam_params(
        os.path.join(args.preprocess_dir, "camera_params.h5"),
        ds=ds,
        up_fn=os.path.join(args.preprocess_dir, "vertical_lines.npz"),
    )

    holdout_views = getattr(config, "holdout_views", [5])
    grid_size = getattr(config, "grid_size", 112)
    volume_idx = getattr(config, "volume_idx", [[0, grid_size]] * 3)

    W = getattr(config, "image_width", 1152) // ds
    H = getattr(config, "image_height", 1024) // ds

    print(f"PS model: H={H}, W={W}, grid_size={grid_size}, ds={ds}")
    print(f"Holdout views: {holdout_views}")

    # Init model
    model = PoseSplatter(
        intrinsics=K_ps,
        extrinsics=E_ps,
        W=W, H=H,
        ell=getattr(config, "ell", 0.22),
        grid_size=grid_size,
        volume_idx=volume_idx,
        holdout_views=holdout_views,
        gaussian_mode=getattr(config, "gaussian_mode", "3d"),
    ).to(device)

    ckpt = torch.load(args.ps_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print("Model loaded.")

    # Load dataset for forward pass
    angle_fn = os.path.join(args.preprocess_dir, "center_rotation.npz")
    image_dir = os.path.join(args.preprocess_dir, "images", "images.zarr")

    dataset = FrameDataset(
        img_fn=image_dir,
        volume_fn=None,  # Not used in __getitem__
        angle_fn=angle_fn,
        C=6,
        holdout_views=holdout_views,
        split="all_volumes",
    )
    print(f"Dataset: {len(dataset)} frames")

    # Load FL trajectory and transform
    c2ws_fl, frame_indices = load_fl_trajectory(args.traj_path)
    scale, R_fl2ps, t_fl2ps = load_fl_to_ps_transform(args.transform_path)
    print(f"Trajectory: {len(frame_indices)} frames, scale={scale:.4f}")

    # PS intrinsics for novel view rendering (use cam0 after ds)
    ps_intrinsics = torch.tensor([
        K_ps[0][0, 0],  # fx
        K_ps[0][1, 1],  # fy
        K_ps[0][0, 2],  # cx
        K_ps[0][1, 2],  # cy
    ], dtype=torch.float32, device=device)

    # Render
    rendered_count = 0
    total = len(frame_indices)
    if args.max_frames:
        total = min(total, args.max_frames)

    with torch.no_grad():
        for i in range(0, total, args.frame_step):
            frame_idx = int(frame_indices[i])
            c2w_fl = c2ws_fl[i]

            # Transform FL c2w to PS coords
            c2w_ps = transform_c2w_fl_to_ps(c2w_fl, scale, R_fl2ps, t_fl2ps)
            c2w_tensor = torch.tensor(c2w_ps, dtype=torch.float32, device=device)

            # Get dataset item for this frame (need mask, img, p_3d, angle for forward)
            # frame_idx in dataset = absolute index
            # Dataset split="all_volumes" so i1=0, i2=total
            ds_idx = frame_idx  # Direct index since split="all_volumes"
            if ds_idx >= len(dataset):
                print(f"  Frame {frame_idx} out of dataset range, skipping")
                continue

            mask, img, p_3d, angle, view_idx, _ = dataset[ds_idx]
            mask = mask.unsqueeze(0).to(device)
            img = img.unsqueeze(0).to(device)
            p_3d = p_3d.unsqueeze(0).to(device)
            angle_val = float(angle)

            # Forward pass to generate Gaussians (use view 0)
            model(mask, img, p_3d, angle_val, view_num=0)

            # Render from FL trajectory camera
            rgb = model.render_novel_view(
                c2w=c2w_tensor,
                intrinsics=ps_intrinsics,
                h=args.resolution,
                w=args.resolution,
            )

            # Save as image
            rgb_np = rgb.cpu().numpy()
            rgb_np = np.clip(rgb_np * 255, 0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

            out_path = os.path.join(args.output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(out_path, rgb_bgr)
            rendered_count += 1

            if rendered_count % 20 == 0:
                print(f"  Rendered {rendered_count}/{total} (frame {frame_idx})")

    print(f"\nDone. {rendered_count} frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
