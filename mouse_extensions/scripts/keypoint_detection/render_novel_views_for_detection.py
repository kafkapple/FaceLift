"""Render GS-LRM novel views for keypoint detection.

Runs in the **facelift** conda env. Generates turntable-camera rendered
images from GS-LRM predictions, saving them to disk with camera parameters
for downstream MMPose detection and triangulation.

Usage:
    conda activate facelift
    CUDA_VISIBLE_DEVICES=4 python render_novel_views_for_detection.py \
        --config ~/dev/FaceLift/configs/mouse/uniform/base_uniform_v2.yaml \
        --checkpoint ~/dev/FaceLift/checkpoints/gslrm/M5t2_E0_1_facelift/best_psnr.pt \
        --data_root ~/data/preprocessed/FaceLift_mouse/M5 \
        --output_dir ~/outputs/neural_triangulation/renders \
        --num_views 12 \
        --render_size 384
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add FaceLift root to path
FACELIFT_ROOT = os.path.expanduser("~/dev/FaceLift")
sys.path.insert(0, FACELIFT_ROOT)

MOUSE_EXT_ROOT = os.path.join(FACELIFT_ROOT, "mouse_extensions")
sys.path.insert(0, MOUSE_EXT_ROOT)


def load_test_frame_ids(data_root: str) -> list:
    """Load M5t2 test frame IDs from split file.

    Split file may contain full paths (e.g., /path/to/M5/003240/)
    or bare frame IDs (e.g., 003240). Returns bare frame IDs.
    """
    split_file = os.path.join(data_root, "data_mouse_t2_test.txt")
    if os.path.exists(split_file):
        ids = []
        with open(split_file) as f:
            for line in f:
                entry = line.strip().rstrip("/")
                if not entry:
                    continue
                # Extract bare frame ID from full path
                frame_id = os.path.basename(entry)
                ids.append(frame_id)
        return ids

    # Fallback: frames 3240-3599
    return [f"{i:06d}" for i in range(3240, 3600)]


def get_turntable_cameras_safe(num_views, render_size, hfov=50, radius=2.7, elevation=20.0):
    """Generate turntable cameras. Wraps gslrm renderer API."""
    from mouse_extensions.visualization import get_turntable_cameras

    w, h, nv, fxfycxcy, c2ws = get_turntable_cameras(
        hfov=hfov,
        num_views=num_views,
        w=render_size,
        h=render_size,
        radius=radius,
        elevation=elevation,
        trajectory_mode="turntable",
        clockwise=True,
        center=None,
    )
    return w, h, nv, fxfycxcy, c2ws


def render_single_view(gaussians, c2w, fxfycxcy, render_size, device, bg_color=(1.0, 1.0, 1.0)):
    """Render a single view from gaussians."""
    from mouse_extensions.visualization import render_opencv_cam

    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device)
    fxfycxcy_t = torch.tensor(fxfycxcy, dtype=torch.float32, device=device)

    result = render_opencv_cam(
        pc=gaussians,
        height=render_size,
        width=render_size,
        C2W=c2w_t,
        fxfycxcy=fxfycxcy_t,
        bg_color=bg_color,
    )
    return result


def cameras_to_serializable(c2ws, fxfycxcys, render_size):
    """Convert camera parameters to JSON-serializable dict with projection matrices."""
    cameras = []
    for i in range(len(c2ws)):
        c2w = c2ws[i]
        fx, fy, cx, cy = fxfycxcys[i]

        # Camera intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ])

        # World-to-camera: w2c = inv(c2w)
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3:]

        # Projection matrix: P = K @ [R | t]
        Rt = np.hstack([R, t])
        P = K @ Rt

        cameras.append({
            "c2w": c2w.tolist(),
            "w2c": w2c.tolist(),
            "K": K.tolist(),
            "fxfycxcy": [float(fx), float(fy), float(cx), float(cy)],
            "P": P.tolist(),
        })

    return {
        "num_views": len(cameras),
        "width": render_size,
        "height": render_size,
        "cameras": cameras,
    }


def render_frame(
    gslrm_model,
    frame_id: str,
    data_root: str,
    num_views: int,
    render_size: int,
    output_dir: str,
    device: str,
    bg_color: tuple = (1.0, 1.0, 1.0),
):
    """Predict gaussians for one frame and render multiple novel views."""
    from inference.gslrm_pipeline import load_sample_data

    sample_dir = os.path.join(data_root, frame_id)

    # Load input data (4 views from M5)
    images, c2ws, fxfycxcys, index = load_sample_data(
        sample_dir, image_size=512, device=device,
    )

    # Predict gaussians
    with torch.no_grad():
        result = gslrm_model.predict(images, c2ws, fxfycxcys, index)

    gaussians = result.gaussians[0]

    # Filter Gaussians (same as multiview_triangulation_eval.py)
    gaussians = gaussians.apply_all_filters(
        opacity_thres=0.04,
        scaling_thres=0.1,
        floater_thres=0.6,
        crop_bbx=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
    )

    # Generate turntable cameras
    w, h, nv, cam_fxfycxcys, cam_c2ws = get_turntable_cameras_safe(
        num_views=num_views,
        render_size=render_size,
    )

    # Convert to numpy for serialization
    if isinstance(cam_c2ws, torch.Tensor):
        cam_c2ws_np = cam_c2ws.cpu().numpy()
    else:
        cam_c2ws_np = np.array(cam_c2ws)

    if isinstance(cam_fxfycxcys, torch.Tensor):
        cam_fxfycxcys_np = cam_fxfycxcys.cpu().numpy()
    else:
        cam_fxfycxcys_np = np.array(cam_fxfycxcys)

    # Create output directory for this frame
    frame_out = os.path.join(output_dir, frame_id)
    os.makedirs(frame_out, exist_ok=True)

    # Render each view
    rendered_paths = []
    for view_idx in range(num_views):
        render_result = render_single_view(
            gaussians,
            cam_c2ws_np[view_idx],
            cam_fxfycxcys_np[view_idx],
            render_size,
            device,
            bg_color=bg_color,
        )

        # Extract rendered image (handle different output formats)
        render_img = render_result["render"]
        if isinstance(render_img, torch.Tensor):
            render_img = render_img.detach().cpu()
            if render_img.dim() == 3 and render_img.shape[0] in (3, 4):
                # (C, H, W) -> (H, W, C)
                render_img = render_img.permute(1, 2, 0)
            render_img = (render_img.numpy() * 255).clip(0, 255).astype(np.uint8)

        # Save as PNG (preserves alpha if present)
        import cv2
        img_path = os.path.join(frame_out, f"cam_{view_idx:03d}.png")
        if render_img.shape[-1] == 4:
            cv2.imwrite(img_path, cv2.cvtColor(render_img, cv2.COLOR_RGBA2BGRA))
        else:
            cv2.imwrite(img_path, cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR))
        rendered_paths.append(img_path)

    # Save camera parameters
    cam_data = cameras_to_serializable(cam_c2ws_np, cam_fxfycxcys_np, render_size)
    cam_data["frame_id"] = frame_id
    cam_data["bg_color"] = list(bg_color)

    cam_path = os.path.join(frame_out, "cameras.json")
    with open(cam_path, "w") as f:
        json.dump(cam_data, f, indent=2)

    return rendered_paths


def main():
    parser = argparse.ArgumentParser(
        description="Render GS-LRM novel views for keypoint detection"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="GS-LRM config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="GS-LRM checkpoint path")
    parser.add_argument("--data_root", type=str,
                        default=os.path.expanduser(
                            "~/data/preprocessed/FaceLift_mouse/M5"),
                        help="M5 preprocessed data root")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser(
                            "~/outputs/neural_triangulation/renders"),
                        help="Output directory for rendered images")
    parser.add_argument("--num_views", type=int, nargs="+", default=[6, 12, 24],
                        help="Number of turntable views to render")
    parser.add_argument("--render_size", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bg_color", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0],
                        help="Background color (R G B, 0-1)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max test frames to process (for debugging)")
    args = parser.parse_args()

    # Load GS-LRM model
    from inference.gslrm_pipeline import GSLRMInference

    print(f"Loading GS-LRM model from {args.checkpoint}")
    gslrm = GSLRMInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # Load test frame IDs
    frame_ids = load_test_frame_ids(args.data_root)
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]
    print(f"Processing {len(frame_ids)} test frames")

    bg_color = tuple(args.bg_color)

    # Render for each view count
    for num_views in args.num_views:
        view_output_dir = os.path.join(args.output_dir, f"{num_views}views")
        print(f"\n=== Rendering {num_views} views -> {view_output_dir} ===")

        for frame_id in tqdm(frame_ids, desc=f"{num_views}v"):
            try:
                render_frame(
                    gslrm_model=gslrm,
                    frame_id=frame_id,
                    data_root=args.data_root,
                    num_views=num_views,
                    render_size=args.render_size,
                    output_dir=view_output_dir,
                    device=args.device,
                    bg_color=bg_color,
                )
            except Exception as e:
                print(f"  Error rendering frame {frame_id}: {e}")
                continue

    print("\nRendering complete.")
    print(f"Output: {args.output_dir}")
    print("Next: run detect_and_triangulate.py in mmpose env")


if __name__ == "__main__":
    main()
