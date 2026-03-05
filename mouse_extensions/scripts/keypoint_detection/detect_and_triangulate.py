"""Run MMPose 2D detection on rendered images, then triangulate to 3D.

Runs in the **mmpose** conda env. Reads rendered images + camera params
from render_novel_views_for_detection.py output, detects 2D keypoints
with the fine-tuned HRNet, triangulates across views, and evaluates
against MAMMAL GT.

Usage:
    conda activate mmpose
    CUDA_VISIBLE_DEVICES=4 python detect_and_triangulate.py \
        --render_dir ~/outputs/neural_triangulation/renders/12views \
        --mmpose_config configs/mmpose/hrnet_w48_mouse_22kp.py \
        --mmpose_checkpoint work_dirs/hrnet_w48_mouse_22kp/best_coco_AP.pth \
        --gt_3d_path /node_data/joon/data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz \
        --output_dir ~/outputs/neural_triangulation/results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add mouse_extensions to path for triangulation functions
FACELIFT_ROOT = os.path.expanduser("~/dev/FaceLift")
MOUSE_EXT_ROOT = os.path.join(FACELIFT_ROOT, "mouse_extensions")
sys.path.insert(0, MOUSE_EXT_ROOT)

# Coordinate transform constants (MAMMAL world mm <-> FaceLift normalized)
M5_SCENE_CENTER = np.array([59.672, 51.517, 107.099])  # mm
M5_DISTANCE_SCALE = 2.7 / 307.785  # = 0.008772


def facelift_to_mammal(points_3d: np.ndarray) -> np.ndarray:
    """Convert FaceLift normalized coords back to MAMMAL world (mm)."""
    return points_3d / M5_DISTANCE_SCALE + M5_SCENE_CENTER


def get_bbox_from_alpha(image: np.ndarray, threshold: float = 0.5, padding: float = 0.1):
    """Compute bbox from alpha channel (RGBA image).

    Args:
        image: (H, W, 4) RGBA image
        threshold: Alpha threshold for foreground
        padding: Fractional padding
    Returns:
        [x1, y1, x2, y2] bbox or None if no foreground
    """
    if image.shape[-1] != 4:
        # No alpha channel, use full image
        h, w = image.shape[:2]
        return [0, 0, w, h]

    alpha = image[:, :, 3].astype(float) / 255.0
    mask = alpha > threshold
    if mask.sum() < 10:
        h, w = image.shape[:2]
        return [0, 0, w, h]

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    w = x2 - x1
    h = y2 - y1
    x1 -= w * padding
    y1 -= h * padding
    x2 += w * padding
    y2 += h * padding

    H, W = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    return [float(x1), float(y1), float(x2), float(y2)]


def init_mmpose_model(config_path: str, checkpoint_path: str, device: str = "cuda:0"):
    """Initialize MMPose model for inference."""
    from mmpose.apis import init_model
    model = init_model(config_path, checkpoint_path, device=device)
    return model


def detect_keypoints_single(model, image_path: str, bbox: list = None):
    """Run MMPose inference on a single image.

    Args:
        model: MMPose model
        image_path: Path to image file
        bbox: [x1, y1, x2, y2] bounding box, or None for full image
    Returns:
        keypoints: (22, 3) with [x, y, confidence]
    """
    from mmpose.apis import inference_topdown
    from mmpose.structures import PoseDataSample

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if bbox is None:
        h, w = img.shape[:2]
        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    else:
        bboxes = np.array([bbox], dtype=np.float32)

    results = inference_topdown(model, image_path, bboxes, bbox_format='xyxy')

    if len(results) == 0:
        return np.zeros((22, 3), dtype=np.float64)

    # Extract keypoints from first result
    result = results[0]
    kps = result.pred_instances.keypoints[0]  # (K, 2)
    scores = result.pred_instances.keypoint_scores[0]  # (K,)

    # Combine into (K, 3) format
    keypoints = np.zeros((kps.shape[0], 3), dtype=np.float64)
    keypoints[:, :2] = kps
    keypoints[:, 2] = scores

    return keypoints


def detect_all_views(model, frame_dir: str, num_views: int):
    """Run detection on all rendered views of a frame.

    Returns:
        keypoints_2d: (num_views, 22, 3) with [x, y, confidence]
    """
    all_kps = []

    for view_idx in range(num_views):
        img_path = os.path.join(frame_dir, f"cam_{view_idx:03d}.png")
        if not os.path.exists(img_path):
            all_kps.append(np.zeros((22, 3), dtype=np.float64))
            continue

        # Get bbox from alpha channel
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        bbox = get_bbox_from_alpha(img)

        kps = detect_keypoints_single(model, img_path, bbox)
        all_kps.append(kps)

    return np.stack(all_kps)  # (num_views, 22, 3)


def load_cameras_json(cam_path: str):
    """Load camera parameters from JSON.

    Returns:
        proj_matrices: (num_views, 3, 4) projection matrices
    """
    with open(cam_path) as f:
        cam_data = json.load(f)

    proj_matrices = []
    for cam in cam_data["cameras"]:
        P = np.array(cam["P"])
        proj_matrices.append(P)

    return np.stack(proj_matrices), cam_data


def triangulate_and_eval(
    keypoints_2d: np.ndarray,
    proj_matrices: np.ndarray,
    gt_3d_mm: np.ndarray,
    conf_threshold: float = 0.3,
):
    """Triangulate detected 2D keypoints and evaluate against GT.

    Args:
        keypoints_2d: (num_views, 22, 3) detected [x, y, conf]
        proj_matrices: (num_views, 3, 4) projection matrices
        gt_3d_mm: (22, 3) MAMMAL GT in mm
        conf_threshold: Min confidence for triangulation
    Returns:
        result dict with pred_3d, mpjpe, per_joint_error, etc.
    """
    from analysis.triangulation_analysis import (
        triangulate_batch,
        compute_mpjpe,
        compute_pa_mpjpe,
    )

    # Triangulate in FaceLift normalized space
    pred_3d_fl = triangulate_batch(
        keypoints_2d, proj_matrices, conf_threshold=conf_threshold,
    )

    # Convert to MAMMAL world (mm)
    pred_3d_mm = facelift_to_mammal(pred_3d_fl)

    # Compute metrics
    mpjpe_result = compute_mpjpe(pred_3d_mm, gt_3d_mm)
    pa_mpjpe_result = compute_pa_mpjpe(pred_3d_mm, gt_3d_mm)

    # Per-view detection stats
    n_views = keypoints_2d.shape[0]
    n_joints = keypoints_2d.shape[1]
    detection_rates = []
    for j in range(n_joints):
        detected = (keypoints_2d[:, j, 2] > conf_threshold).sum()
        detection_rates.append(float(detected) / n_views)

    return {
        "pred_3d_mm": pred_3d_mm.tolist(),
        "mpjpe": float(mpjpe_result["mpjpe"]),
        "mpjpe_std": float(mpjpe_result.get("mpjpe_std", 0)),
        "per_joint_error": [float(v) for v in mpjpe_result["per_joint"]],
        "median_error": float(mpjpe_result.get("median", 0)),
        "max_error": float(mpjpe_result.get("max", 0)),
        "pa_mpjpe": float(pa_mpjpe_result["pa_mpjpe"]),
        "detection_rates": detection_rates,
        "mean_detection_rate": float(np.mean(detection_rates)),
        "n_views_used": n_views,
    }


def run_pipeline(args):
    """Main detection + triangulation pipeline."""
    # Load MAMMAL 3D GT
    print(f"Loading GT 3D from {args.gt_3d_path}")
    gt_data = np.load(args.gt_3d_path)
    gt_3d_all = gt_data["keypoints"]  # (3600, 22, 3) in mm
    gt_frame_indices = gt_data["frame_indices"]  # (3600,)

    # Test split: M5 frames 3240-3599
    test_start = 3240
    test_end = 3600
    gt_3d_test = gt_3d_all[test_start:test_end]  # (360, 22, 3)

    # Initialize MMPose model
    print(f"Loading MMPose model from {args.mmpose_checkpoint}")
    model = init_mmpose_model(args.mmpose_config, args.mmpose_checkpoint, args.device)

    # Discover rendered frames
    render_dir = args.render_dir
    frame_dirs = sorted([
        d for d in os.listdir(render_dir)
        if os.path.isdir(os.path.join(render_dir, d))
    ])

    if args.max_frames:
        frame_dirs = frame_dirs[:args.max_frames]

    print(f"Processing {len(frame_dirs)} frames from {render_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Process each frame
    all_results = []
    per_joint_errors = []

    for frame_id in tqdm(frame_dirs, desc="Detecting + triangulating"):
        frame_dir = os.path.join(render_dir, frame_id)
        cam_path = os.path.join(frame_dir, "cameras.json")

        if not os.path.exists(cam_path):
            print(f"  Warning: No cameras.json for frame {frame_id}")
            continue

        # Load camera parameters
        proj_matrices, cam_data = load_cameras_json(cam_path)
        num_views = cam_data["num_views"]

        # Run detection on all views
        keypoints_2d = detect_all_views(model, frame_dir, num_views)

        # Get GT for this frame
        m5_idx = int(frame_id)
        if m5_idx < test_start or m5_idx >= test_end:
            continue
        gt_idx = m5_idx - test_start
        gt_3d = gt_3d_test[gt_idx]  # (22, 3) mm

        # Triangulate and evaluate
        result = triangulate_and_eval(
            keypoints_2d, proj_matrices, gt_3d,
            conf_threshold=args.conf_threshold,
        )
        result["frame_id"] = frame_id
        all_results.append(result)
        per_joint_errors.append(result["per_joint_error"])

        # Save per-frame detection for visualization
        det_path = os.path.join(args.output_dir, f"detections_{frame_id}.json")
        with open(det_path, "w") as f:
            json.dump({
                "frame_id": frame_id,
                "keypoints_2d": keypoints_2d.tolist(),
                "num_views": num_views,
            }, f)

    if not all_results:
        print("No results! Check render_dir and frame IDs.")
        return

    # Aggregate statistics
    mpjpe_values = [r["mpjpe"] for r in all_results]
    pa_mpjpe_values = [r["pa_mpjpe"] for r in all_results]
    per_joint_errors = np.array(per_joint_errors)  # (N_frames, 22)
    detection_rates = np.array([r["detection_rates"] for r in all_results])

    # Canonical keypoint names
    joint_names = [
        "L_ear", "R_ear", "nose", "neck", "body_middle",
        "tail_root", "tail_middle", "tail_end",
        "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
        "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
        "L_foot", "L_knee", "L_hip",
        "R_foot", "R_knee", "R_hip",
    ]

    summary = {
        "num_frames": len(all_results),
        "num_views": all_results[0]["n_views_used"],
        "conf_threshold": args.conf_threshold,
        "mpjpe_mean": float(np.mean(mpjpe_values)),
        "mpjpe_std": float(np.std(mpjpe_values)),
        "mpjpe_median": float(np.median(mpjpe_values)),
        "pa_mpjpe_mean": float(np.mean(pa_mpjpe_values)),
        "pa_mpjpe_std": float(np.std(pa_mpjpe_values)),
        "per_joint": {
            joint_names[j]: {
                "mpjpe_mean": float(per_joint_errors[:, j].mean()),
                "mpjpe_std": float(per_joint_errors[:, j].std()),
                "detection_rate": float(detection_rates[:, j].mean()),
            }
            for j in range(22)
        },
        "mean_detection_rate": float(detection_rates.mean()),
    }

    # Save results
    results_path = os.path.join(args.output_dir, "neural_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    all_results_path = os.path.join(args.output_dir, "neural_results_all_frames.json")
    with open(all_results_path, "w") as f:
        json.dump(all_results, f)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Neural Detection + Triangulation Results ({summary['num_views']} views)")
    print(f"{'=' * 60}")
    print(f"Frames: {summary['num_frames']}")
    print(f"MPJPE:    {summary['mpjpe_mean']:.2f} +/- {summary['mpjpe_std']:.2f} mm")
    print(f"PA-MPJPE: {summary['pa_mpjpe_mean']:.2f} +/- {summary['pa_mpjpe_std']:.2f} mm")
    print(f"Mean detection rate: {summary['mean_detection_rate']:.1%}")
    print(f"\nPer-joint MPJPE (mm):")
    for j in range(22):
        name = joint_names[j]
        err = per_joint_errors[:, j].mean()
        det = detection_rates[:, j].mean()
        print(f"  {name:15s}: {err:6.2f} mm  (det: {det:.1%})")
    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect 2D keypoints on rendered images and triangulate to 3D"
    )
    parser.add_argument("--render_dir", type=str, required=True,
                        help="Directory with rendered images (from render_novel_views)")
    parser.add_argument("--mmpose_config", type=str, required=True,
                        help="MMPose config file path")
    parser.add_argument("--mmpose_checkpoint", type=str, required=True,
                        help="MMPose checkpoint file path")
    parser.add_argument("--gt_3d_path", type=str,
                        default=os.path.expanduser(
                            "/node_data/joon/data/results/MAMMAL_mouse/"
                            "v012345_kp22_20260126/keypoints_22_3d.npz"),
                        help="Path to MAMMAL 3D GT NPZ")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser(
                            "~/outputs/neural_triangulation/results"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Min confidence for triangulation")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
