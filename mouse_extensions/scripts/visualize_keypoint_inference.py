#!/usr/bin/env python3
"""Visualize GS-LRM inference results with MAMMAL 22-keypoint overlay.

Runs GS-LRM inference on selected test frames, renders from GT camera views,
and overlays projected MAMMAL 3D keypoints on both GT and predicted images.

Usage:
    CUDA_VISIBLE_DEVICES=4 /home/joon/anaconda3/envs/facelift/bin/python3 \
        mouse_extensions/scripts/visualize_keypoint_inference.py \
        --checkpoint base_uniform_v2_6view_v2 \
        --keypoints_npz /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
        --data_txt ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
        --frames 3240 3280 3320 3360 3400 \
        --output_dir /node_data/joon/outputs/FaceLift/keypoint_viz
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add FaceLift project root to path for mouse_extensions imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # mouse_extensions/scripts/ → FaceLift/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image


# Coordinate transforms — SSOT: mouse_extensions/coordinate_utils.py
# Derivation: M5 preprocessing _normalize_cameras_batch() in preprocess.py
# Scene center = centroid of 6 camera positions, scale = 2.7 / mean distance
from mouse_extensions.coordinate_utils import mammal_to_gslrm as transform_mammal_to_facelift


def load_keypoints_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load pre-extracted MAMMAL keypoints.

    Returns:
        keypoints: (N, 22, 3) world-space coordinates
        frame_indices: (N,) original video frame indices
        keypoint_names: list of 22 names
    """
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data["keypoints"]
    frame_indices = data["frame_indices"]
    names = list(data["keypoint_names"])
    print(f"Loaded keypoints: {keypoints.shape}, frames: {frame_indices.min()}-{frame_indices.max()}")
    return keypoints, frame_indices, names


def load_camera_data(sample_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict], dict]:
    """Load GT images, w2c matrices, and intrinsics from a FaceLift sample dir.

    Returns:
        images: list of (H, W, 3) BGR images
        w2cs: list of (4, 4) w2c matrices
        intrinsics_list: list of dicts with fx, fy, cx, cy
        preprocessing: dict with version and frame_idx
    """
    cam_json_path = os.path.join(sample_dir, "opencv_cameras.json")
    with open(cam_json_path, "r") as f:
        cam_data = json.load(f)

    preprocessing = cam_data.get("_preprocessing", {})
    frames = cam_data["frames"]

    images = []
    w2cs = []
    intrinsics_list = []

    for frame in frames:
        # Load image (RGBA → BGR)
        img_path = os.path.join(sample_dir, frame["file_path"])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            # RGBA → RGB with white background, then BGR
            alpha = img[:, :, 3:4].astype(float) / 255.0
            rgb = img[:, :, :3].astype(float)
            white_bg = np.ones_like(rgb) * 255.0
            composited = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            img = composited
        elif img is not None and img.shape[2] == 3:
            pass  # Already BGR
        images.append(img)

        # w2c matrix
        w2c = np.array(frame["w2c"], dtype=np.float64)
        w2cs.append(w2c)

        # Intrinsics
        intrinsics_list.append({
            "fx": frame["fx"],
            "fy": frame["fy"],
            "cx": frame["cx"],
            "cy": frame["cy"],
        })

    return images, w2cs, intrinsics_list, preprocessing


def get_m5_frame_index(m5_dir_name: str) -> int:
    """Convert M5 directory name to frame index. '003240' → 3240."""
    return int(m5_dir_name)


def find_keypoint_index(
    frame_indices: np.ndarray,
    m5_frame_idx: int,
) -> Optional[int]:
    """Find the keypoint array index for a given M5 frame index.

    M5 frame i → original video frame i*5 → match in frame_indices.
    """
    original_frame = m5_frame_idx * 5
    matches = np.where(frame_indices == original_frame)[0]
    if len(matches) > 0:
        return int(matches[0])
    return None


def run_gslrm_inference(
    sample_dir: str,
    checkpoint_name: str,
    device: str = "cuda",
) -> Tuple[Optional[object], Optional[list]]:
    """Run GS-LRM inference and return Gaussians + rendered images.

    Returns:
        gaussians: GaussianModel or None
        pred_images: list of (H, W, 3) BGR numpy arrays or None
    """
    try:
        from mouse_extensions.inference.gslrm_pipeline import (
            GSLRMInference,
            load_sample_data,
        )
        from gslrm.model.gaussians_renderer import render_opencv_cam
    except ImportError as e:
        print(f"Warning: Cannot import GS-LRM modules: {e}")
        print("Skipping inference, will only overlay keypoints on GT images.")
        return None, None

    # Resolve checkpoint
    ckpt_base = Path("checkpoints/gslrm")
    config_path = ckpt_base / checkpoint_name / "config.yaml"
    ckpt_dir = ckpt_base / checkpoint_name

    if not config_path.exists():
        print(f"Warning: Checkpoint config not found: {config_path}")
        return None, None

    pipeline = GSLRMInference(
        config_path=str(config_path),
        checkpoint_path=str(ckpt_dir),
        device=device,
    )

    images, c2ws, fxfycxcys, index = load_sample_data(sample_dir, device=device)
    result = pipeline.predict(images, c2ws, fxfycxcys, index)

    # Extract Gaussians for custom rendering (predict returns a list for batch)
    gaussians_raw = result.get("gaussians", None)
    gaussians = None
    if gaussians_raw is not None:
        gaussians = gaussians_raw[0] if isinstance(gaussians_raw, list) else gaussians_raw

    # Render from GT camera poses using Gaussians
    pred_images = []
    if gaussians is not None:
        cam_json_path = os.path.join(sample_dir, "opencv_cameras.json")
        with open(cam_json_path, "r") as f:
            cam_data = json.load(f)

        for frame in cam_data["frames"]:
            w2c = np.array(frame["w2c"], dtype=np.float32)
            c2w = np.linalg.inv(w2c)
            c2w_tensor = torch.from_numpy(c2w).float().to(device)
            fxfycxcy = torch.tensor(
                [frame["fx"], frame["fy"], frame["cx"], frame["cy"]],
                dtype=torch.float32, device=device,
            )

            rendered = render_opencv_cam(
                gaussians,
                height=frame["h"],
                width=frame["w"],
                C2W=c2w_tensor,
                fxfycxcy=fxfycxcy,
            )
            # rendered is dict with 'render' key: (C, H, W) tensor
            render_img = rendered["render"]  # (C, H, W)
            render_np = (render_img.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            # RGB → BGR for OpenCV
            render_np = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)
            pred_images.append(render_np)

    return gaussians, pred_images if pred_images else None


def add_text_label(image: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """Add a text label bar to an image."""
    h, w = image.shape[:2]
    label_h = 25
    label_bar = np.zeros((label_h, w, 3), dtype=np.uint8)
    cv2.putText(
        label_bar, text, (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    if position == "top":
        return np.concatenate([label_bar, image], axis=0)
    return np.concatenate([image, label_bar], axis=0)


def create_frame_visualization(
    gt_images: List[np.ndarray],
    pred_images: Optional[List[np.ndarray]],
    keypoints_3d: np.ndarray,
    w2cs: List[np.ndarray],
    intrinsics_list: List[Dict],
    frame_label: str,
    output_dir: str,
    draw_labels: bool = False,
) -> List[str]:
    """Create all visualization outputs for a single frame.

    Returns:
        List of saved file paths.
    """
    from mouse_extensions.visualization.keypoint_overlay import (
        KeypointVisualizer,
        create_legend,
        project_3d_to_2d,
    )

    viz = KeypointVisualizer(
        joint_radius=4,
        bone_thickness=2,
        draw_labels=draw_labels,
        draw_skeleton=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    num_views = len(gt_images)

    # --- 1. 6-cam GT overlay grid ---
    gt_overlays = viz.overlay_multiview(
        gt_images, keypoints_3d, w2cs, intrinsics_list,
    )
    # 2x3 grid for 6 cameras
    rows = []
    for row_idx in range(2):
        row_imgs = []
        for col_idx in range(3):
            v = row_idx * 3 + col_idx
            if v < num_views:
                labeled = add_text_label(gt_overlays[v], f"cam_{v:03d}")
                row_imgs.append(labeled)
        if row_imgs:
            rows.append(np.concatenate(row_imgs, axis=1))
    gt_grid = np.concatenate(rows, axis=0) if rows else gt_overlays[0]

    path = os.path.join(output_dir, f"{frame_label}_6cam_overlay.png")
    cv2.imwrite(path, gt_grid)
    saved_paths.append(path)

    # --- 2. GT vs Pred comparison (if predictions available) ---
    if pred_images is not None:
        pred_overlays = viz.overlay_multiview(
            pred_images, keypoints_3d, w2cs, intrinsics_list,
        )
        gt_row_imgs = [add_text_label(img, f"GT cam_{i}") for i, img in enumerate(gt_overlays)]
        pred_row_imgs = [add_text_label(img, f"Pred cam_{i}") for i, img in enumerate(pred_overlays)]

        gt_row = np.concatenate(gt_row_imgs[:3], axis=1)
        pred_row = np.concatenate(pred_row_imgs[:3], axis=1)
        comparison_top = np.concatenate([gt_row, pred_row], axis=0)

        if num_views > 3:
            gt_row2 = np.concatenate(gt_row_imgs[3:6], axis=1)
            pred_row2 = np.concatenate(pred_row_imgs[3:6], axis=1)
            comparison_bot = np.concatenate([gt_row2, pred_row2], axis=0)
            comparison = np.concatenate([comparison_top, comparison_bot], axis=0)
        else:
            comparison = comparison_top

        path = os.path.join(output_dir, f"{frame_label}_gt_vs_pred.png")
        cv2.imwrite(path, comparison)
        saved_paths.append(path)

    # --- 3. Labeled keypoint detail (cam_000 with labels) ---
    viz_labeled = KeypointVisualizer(
        joint_radius=5, bone_thickness=2, draw_labels=True, draw_skeleton=True,
    )
    if gt_images:
        detail = viz_labeled.overlay_on_image(
            gt_images[0], keypoints_3d, w2cs[0], intrinsics_list[0],
        )
        legend = create_legend(height=detail.shape[0], width=300)
        if legend.shape[0] != detail.shape[0]:
            legend = cv2.resize(legend, (300, detail.shape[0]))
        detail_with_legend = np.concatenate([detail, legend], axis=1)

        path = os.path.join(output_dir, f"{frame_label}_labeled_detail.png")
        cv2.imwrite(path, detail_with_legend)
        saved_paths.append(path)

    return saved_paths


def print_coordinate_diagnostics(
    keypoints_3d: np.ndarray,
    w2cs: List[np.ndarray],
    intrinsics_list: List[Dict],
    frame_label: str,
):
    """Print diagnostic info for coordinate alignment debugging."""
    from mouse_extensions.visualization.keypoint_overlay import project_3d_to_2d

    print(f"\n{'='*60}")
    print(f"Coordinate Diagnostics: {frame_label}")
    print(f"{'='*60}")

    # 3D keypoint stats
    print(f"\n3D Keypoints (world space):")
    print(f"  Range: [{keypoints_3d.min():.4f}, {keypoints_3d.max():.4f}]")
    print(f"  Center: {keypoints_3d.mean(axis=0)}")
    print(f"  Nose (idx=2): {keypoints_3d[2]}")

    # Camera center positions (from c2w = inv(w2c))
    print(f"\nCamera centers (world space):")
    for i, w2c in enumerate(w2cs):
        c2w = np.linalg.inv(w2c)
        cam_center = c2w[:3, 3]
        print(f"  cam_{i}: {cam_center}")

    # Projection check for cam_000
    kp_2d, valid = project_3d_to_2d(keypoints_3d, w2cs[0], intrinsics_list[0])
    print(f"\n2D Projection (cam_000):")
    print(f"  Valid: {valid.sum()}/{len(valid)}")
    print(f"  u range: [{kp_2d[valid, 0].min():.1f}, {kp_2d[valid, 0].max():.1f}]")
    print(f"  v range: [{kp_2d[valid, 1].min():.1f}, {kp_2d[valid, 1].max():.1f}]")
    img_w = intrinsics_list[0].get("cx", 256) * 2
    img_h = intrinsics_list[0].get("cy", 256) * 2
    in_frame = (
        (kp_2d[valid, 0] >= 0) & (kp_2d[valid, 0] < img_w) &
        (kp_2d[valid, 1] >= 0) & (kp_2d[valid, 1] < img_h)
    )
    print(f"  In frame: {in_frame.sum()}/{valid.sum()} (image: {img_w:.0f}x{img_h:.0f})")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GS-LRM inference with MAMMAL keypoint overlay",
    )
    parser.add_argument(
        "--checkpoint", default="base_uniform_v2_6view_v2",
        help="GS-LRM checkpoint name under checkpoints/gslrm/",
    )
    parser.add_argument(
        "--keypoints_npz", required=True,
        help="Path to extracted MAMMAL keypoints NPZ",
    )
    parser.add_argument(
        "--data_txt", required=True,
        help="Path to data_mouse_t2_test.txt listing sample directories",
    )
    parser.add_argument(
        "--frames", type=int, nargs="+",
        default=[3240, 3280, 3320, 3360, 3400],
        help="M5 frame indices to visualize",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--no_inference", action="store_true",
        help="Skip GS-LRM inference, only overlay keypoints on GT images",
    )
    parser.add_argument(
        "--draw_labels", action="store_true",
        help="Draw keypoint name labels on overlay",
    )
    parser.add_argument(
        "--diagnostics", action="store_true", default=True,
        help="Print coordinate diagnostics for debugging",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # --- Load keypoints ---
    keypoints_all, frame_indices, kp_names = load_keypoints_npz(args.keypoints_npz)

    # --- Load data file list ---
    data_txt = os.path.expanduser(args.data_txt)
    with open(data_txt, "r") as f:
        all_sample_dirs = [line.strip().rstrip("/") for line in f if line.strip()]

    print(f"Total samples in data file: {len(all_sample_dirs)}")
    print(f"Frames to visualize: {args.frames}")

    # --- Process each frame ---
    os.makedirs(args.output_dir, exist_ok=True)
    all_saved = []

    for m5_idx in args.frames:
        frame_label = f"frame_{m5_idx:06d}"
        print(f"\n{'='*60}")
        print(f"Processing {frame_label}")
        print(f"{'='*60}")

        # Find sample directory
        target_dir = None
        for sd in all_sample_dirs:
            dir_name = os.path.basename(sd)
            if get_m5_frame_index(dir_name) == m5_idx:
                target_dir = sd
                break

        if target_dir is None:
            print(f"  Warning: M5 frame {m5_idx} not found in data file, skipping")
            continue

        # Find keypoint index
        kp_idx = find_keypoint_index(frame_indices, m5_idx)
        if kp_idx is None:
            print(f"  Warning: No keypoints for M5 frame {m5_idx} "
                  f"(original frame {m5_idx * 5}), skipping")
            continue

        kp_3d_raw = keypoints_all[kp_idx]  # (22, 3) in MAMMAL world (mm)
        kp_3d = transform_mammal_to_facelift(kp_3d_raw)  # (22, 3) in FaceLift world
        print(f"  Sample dir: {target_dir}")
        print(f"  Keypoint index: {kp_idx} (original frame: {frame_indices[kp_idx]})")
        print(f"  KP raw center: {kp_3d_raw.mean(axis=0)} → transformed: {kp_3d.mean(axis=0)}")

        # Load camera data
        gt_images, w2cs, intrinsics_list, preproc = load_camera_data(target_dir)
        print(f"  Preprocessing: {preproc}")
        print(f"  Loaded {len(gt_images)} GT views")

        # Coordinate diagnostics
        if args.diagnostics:
            print_coordinate_diagnostics(kp_3d, w2cs, intrinsics_list, frame_label)

        # GS-LRM inference
        pred_images = None
        if not args.no_inference:
            _, pred_images = run_gslrm_inference(
                target_dir, args.checkpoint, args.device,
            )
            if pred_images:
                print(f"  Rendered {len(pred_images)} predicted views")
            else:
                print(f"  Warning: Inference failed, using GT-only overlay")

        # Create visualizations
        saved = create_frame_visualization(
            gt_images=gt_images,
            pred_images=pred_images,
            keypoints_3d=kp_3d,
            w2cs=w2cs,
            intrinsics_list=intrinsics_list,
            frame_label=frame_label,
            output_dir=args.output_dir,
            draw_labels=args.draw_labels,
        )
        all_saved.extend(saved)
        print(f"  Saved {len(saved)} images")

    # --- Summary grid ---
    if all_saved:
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Total images saved: {len(all_saved)}")
        for p in all_saved:
            print(f"  {p}")

    print(f"\nDone. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
