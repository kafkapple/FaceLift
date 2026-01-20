#!/usr/bin/env python3
"""
D7_5b Preprocessing: PP=256 + Object-Aware Maximum Scaling
===========================================================

Goal: PP=(256,256) exactly while maximizing object size (not clipping any part)

Key improvement over D7.5:
- D7.5: Fits entire IMAGE in 512×512 (conservative)
- D7_5b: Fits entire OBJECT (from mask) in 512×512 (optimal)

This ensures:
1. Mouse tail, nose, limbs are never clipped
2. Maximum zoom while preserving the entire animal
3. PP exactly at (256, 256)

Algorithm:
1. For each view, analyze mask across ALL frames to find max bounding box
2. Compute scale that fits this bbox in 512×512 after PP-centering
3. Add safety margin to account for movement

Created: 2026-01-19
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# Frames with camera jumps to skip
JUMP_FRAMES = {5900, 11800, 17700}


@dataclass
class D75bConfig:
    """D7_5b preprocessing configuration"""
    target_pp: Tuple[float, float] = (256.0, 256.0)
    target_distance: float = 2.7
    output_size: int = 512
    safety_margin: float = 1.05  # 5% safety margin for object movement
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)


class D75bPreprocessor:
    """
    D7_5b: Object-aware optimal scaling with PP=256

    Analyzes masks to find the maximum object extent across all frames,
    then computes optimal scale to fit the object without clipping.
    """

    def __init__(self, config: D75bConfig = None):
        self.config = config or D75bConfig()
        self.view_bboxes = {}  # Store max bbox per view

    def load_original_cameras(self, pkl_path: str) -> List[Dict]:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def analyze_masks_for_bbox(
        self,
        mask_dir: Path,
        num_views: int,
        frame_indices: List[int],
        sample_every: int = 10
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Analyze masks across frames to find maximum bounding box per view.

        Returns:
            Dict mapping view_idx to (min_x, min_y, max_x, max_y)
        """
        print("Analyzing masks to find maximum object extent...")

        view_bboxes = {i: [float('inf'), float('inf'), 0, 0] for i in range(num_views)}

        # Sample frames for analysis (every Nth frame)
        sample_frames = frame_indices[::sample_every]
        print(f"  Sampling {len(sample_frames)} frames for bbox analysis")

        mask_caps = []
        for cam_idx in range(num_views):
            mask_caps.append(cv2.VideoCapture(str(mask_dir / f"{cam_idx}.mp4")))

        for frame_idx in tqdm(sample_frames, desc="  Analyzing"):
            for cam_idx in range(num_views):
                mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, mask = mask_caps[cam_idx].read()

                if not ret:
                    continue

                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                # Find bounding box of mask
                ys, xs = np.where(mask > 127)
                if len(xs) == 0:
                    continue

                min_x, max_x = xs.min(), xs.max()
                min_y, max_y = ys.min(), ys.max()

                # Update max bbox
                bbox = view_bboxes[cam_idx]
                bbox[0] = min(bbox[0], min_x)
                bbox[1] = min(bbox[1], min_y)
                bbox[2] = max(bbox[2], max_x)
                bbox[3] = max(bbox[3], max_y)

        for cap in mask_caps:
            cap.release()

        # Convert to tuples and apply safety margin
        margin = self.config.safety_margin
        result = {}
        for view_idx, bbox in view_bboxes.items():
            if bbox[0] == float('inf'):
                continue
            # Expand by safety margin (center-based)
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = (bbox[2] - bbox[0]) * margin
            h = (bbox[3] - bbox[1]) * margin
            result[view_idx] = (
                int(cx - w/2),
                int(cy - h/2),
                int(cx + w/2),
                int(cy + h/2)
            )
            print(f"  View {view_idx}: Object bbox = ({result[view_idx][0]}, {result[view_idx][1]}) - "
                  f"({result[view_idx][2]}, {result[view_idx][3]}), "
                  f"size = {result[view_idx][2]-result[view_idx][0]}×{result[view_idx][3]-result[view_idx][1]}")

        return result

    def compute_optimal_scale_for_object(
        self,
        cx: float,
        cy: float,
        obj_bbox: Tuple[int, int, int, int],
        orig_w: int = 1152,
        orig_h: int = 1024
    ) -> float:
        """
        Compute maximum scale that fits object bbox in 512×512
        while centering PP at (256, 256).

        After transform:
        - Object left edge: obj_bbox[0] * s + shift_x >= 0
        - Object right edge: obj_bbox[2] * s + shift_x <= 512
        - Object top edge: obj_bbox[1] * s + shift_y >= 0
        - Object bottom edge: obj_bbox[3] * s + shift_y <= 512

        Where shift = (256 - cx*s, 256 - cy*s)

        Substituting:
        - obj_bbox[0] * s + 256 - cx*s >= 0  → s * (obj_bbox[0] - cx) >= -256
        - obj_bbox[2] * s + 256 - cx*s <= 512  → s * (obj_bbox[2] - cx) <= 256
        - obj_bbox[1] * s + 256 - cy*s >= 0  → s * (obj_bbox[1] - cy) >= -256
        - obj_bbox[3] * s + 256 - cy*s <= 512  → s * (obj_bbox[3] - cy) <= 256
        """
        target = self.config.target_pp[0]  # 256

        obj_min_x, obj_min_y, obj_max_x, obj_max_y = obj_bbox

        constraints = []

        # Left edge constraint: s * (obj_min_x - cx) >= -256
        if obj_min_x < cx:
            constraints.append(target / (cx - obj_min_x))

        # Right edge constraint: s * (obj_max_x - cx) <= 256
        if obj_max_x > cx:
            constraints.append(target / (obj_max_x - cx))

        # Top edge constraint: s * (obj_min_y - cy) >= -256
        if obj_min_y < cy:
            constraints.append(target / (cy - obj_min_y))

        # Bottom edge constraint: s * (obj_max_y - cy) <= 256
        if obj_max_y > cy:
            constraints.append(target / (obj_max_y - cy))

        if not constraints:
            # Object is entirely within PP quadrant, use image bounds
            return self.compute_optimal_scale_for_image(cx, cy, orig_w, orig_h)

        return min(constraints)

    def compute_optimal_scale_for_image(
        self,
        cx: float,
        cy: float,
        orig_w: int = 1152,
        orig_h: int = 1024
    ) -> float:
        """Fallback: fit entire image (same as D7.5)"""
        target = self.config.target_pp[0]
        constraints = [
            target / cx,
            target / cy,
            target / (orig_w - cx),
            target / (orig_h - cy),
        ]
        return min(constraints)

    def compute_transform(
        self,
        K: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        obj_bbox: Optional[Tuple[int, int, int, int]] = None,
        orig_size: Tuple[int, int] = (1152, 1024)
    ) -> Tuple[Dict, np.ndarray, dict]:
        """Compute camera transform with object-aware scaling."""
        orig_w, orig_h = orig_size
        cfg = self.config

        orig_fx = K[0, 0]
        orig_fy = K[1, 1]
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]

        # Compute optimal scale
        if obj_bbox is not None:
            scale = self.compute_optimal_scale_for_object(orig_cx, orig_cy, obj_bbox, orig_w, orig_h)
        else:
            scale = self.compute_optimal_scale_for_image(orig_cx, orig_cy, orig_w, orig_h)

        # New focal length
        new_fx = orig_fx * scale
        new_fy = orig_fy * scale

        # Shift for PP=256
        scaled_cx = orig_cx * scale
        scaled_cy = orig_cy * scale
        shift_x = cfg.target_pp[0] - scaled_cx
        shift_y = cfg.target_pp[1] - scaled_cy

        # Affine matrix
        affine_matrix = np.array([
            [scale, 0, shift_x],
            [0, scale, shift_y]
        ], dtype=np.float32)

        # Extrinsics
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.flatten()
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        current_distance = np.linalg.norm(cam_pos)

        distance_scale = cfg.target_distance / current_distance
        new_cam_pos = cam_pos * distance_scale
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = new_cam_pos
        new_w2c = np.linalg.inv(new_c2w)

        # Coverage (approximate)
        scaled_w = orig_w * scale
        scaled_h = orig_h * scale
        coverage = min(1.0, (scaled_w * scaled_h) / (cfg.output_size ** 2))

        camera_dict = {
            "w": cfg.output_size,
            "h": cfg.output_size,
            "fx": new_fx,
            "fy": new_fy,
            "cx": cfg.target_pp[0],
            "cy": cfg.target_pp[1],
            "w2c": new_w2c.tolist(),
            "_original": {
                "fx": float(orig_fx),
                "fy": float(orig_fy),
                "cx": float(orig_cx),
                "cy": float(orig_cy),
                "distance": float(current_distance),
            },
            "_transform": {
                "method": "D7_5b_object_aware",
                "scale": float(scale),
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
                "coverage": float(coverage),
                "object_bbox": obj_bbox,
            },
        }

        stats = {
            "scale": scale,
            "new_fx": new_fx,
            "coverage": coverage,
        }

        return camera_dict, affine_matrix, stats

    def transform_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        affine_matrix: np.ndarray
    ) -> np.ndarray:
        cfg = self.config
        output_size = cfg.output_size

        warped_img = cv2.warpAffine(
            image, affine_matrix, (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        warped_mask = cv2.warpAffine(
            mask, affine_matrix, (output_size, output_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        if len(warped_img.shape) == 2:
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2RGB)
        elif warped_img.shape[2] == 4:
            warped_img = warped_img[:, :, :3]

        output = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        output[:, :, :3] = warped_img
        output[:, :, 3] = warped_mask

        return output


def process_dataset(
    data_dir: str,
    output_dir: str,
    camera_pkl: str,
    config: D75bConfig = None,
    frame_interval: int = 5,
    max_samples: Optional[int] = None,
    val_ratio: float = 0.1
):
    if config is None:
        config = D75bConfig()

    preprocessor = D75bPreprocessor(config)

    print(f"Loading cameras from {camera_pkl}")
    cameras = preprocessor.load_original_cameras(camera_pkl)
    num_views = len(cameras)
    print(f"Found {num_views} camera views")

    data_path = Path(data_dir)
    video_dir = data_path / "videos_undist"
    mask_dir = data_path / "simpleclick_undist"
    output_path = Path(output_dir)

    # Get frame indices
    cap = cv2.VideoCapture(str(video_dir / "0.mp4"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_indices = [i for i in range(0, total_frames, frame_interval)
                     if i not in JUMP_FRAMES]

    if max_samples:
        frame_indices = frame_indices[:max_samples]

    print(f"Total frames: {total_frames}")
    print(f"Processing {len(frame_indices)} frames")

    # Analyze masks to find object bboxes
    view_bboxes = preprocessor.analyze_masks_for_bbox(
        mask_dir, num_views, frame_indices, sample_every=50
    )

    # Open video captures
    video_caps = []
    mask_caps = []
    for cam_idx in range(num_views):
        video_caps.append(cv2.VideoCapture(str(video_dir / f"{cam_idx}.mp4")))
        mask_caps.append(cv2.VideoCapture(str(mask_dir / f"{cam_idx}.mp4")))

    # Train/val split
    np.random.seed(42)
    shuffled = frame_indices.copy()
    np.random.shuffle(shuffled)
    val_count = int(len(shuffled) * val_ratio)
    val_frames = set(shuffled[:val_count])

    print(f"Train: {len(frame_indices) - val_count}, Val: {val_count}")

    for split in ['train', 'val']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Compute transforms
    print("\nComputing camera transforms (object-aware)...")
    camera_dicts = []
    affine_matrices = []

    for i, cam in enumerate(cameras):
        K = cam['K']
        R = cam['R']
        T = cam['T']

        cam_dict, affine, stats = preprocessor.compute_transform(
            K, R, T, view_bboxes.get(i)
        )
        # Add file_path and view_id for dataset loader compatibility
        cam_dict['file_path'] = f"images/cam_{i:03d}.png"
        cam_dict['view_id'] = i
        camera_dicts.append(cam_dict)
        affine_matrices.append(affine)

        print(f"  View {i}: scale={stats['scale']:.4f}, fx={stats['new_fx']:.0f}, "
              f"coverage={stats['coverage']*100:.1f}%")

    fx_values = [cd['fx'] for cd in camera_dicts]
    print(f"\n  fx range: {min(fx_values):.0f} - {max(fx_values):.0f}")
    print(f"  PP: (256, 256) for all views ✓")

    # Process frames
    print("\nProcessing frames...")
    train_paths = []
    val_paths = []

    for sample_idx, frame_idx in enumerate(tqdm(frame_indices)):
        split = 'val' if frame_idx in val_frames else 'train'
        sample_list = val_paths if split == 'val' else train_paths

        sample_id = f"{len(sample_list):06d}"
        sample_dir = output_path / split / sample_id
        images_out = sample_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        for cam_idx in range(num_views):
            video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = video_caps[cam_idx].read()
            ret_mask, mask = mask_caps[cam_idx].read()

            if not ret or not ret_mask:
                continue

            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            transformed = preprocessor.transform_image(
                frame_rgb, mask_binary, affine_matrices[cam_idx]
            )

            Image.fromarray(transformed).save(images_out / f"cam_{cam_idx:03d}.png")

        camera_json = {
            "frames": camera_dicts,
            "_preprocessing": {
                "method": "D7_5b_object_aware",
                "description": "PP=256 with object-aware maximum scaling",
                "original_frame_idx": int(frame_idx),
            }
        }

        with open(sample_dir / "opencv_cameras.json", 'w') as f:
            json.dump(camera_json, f, indent=2)

        sample_list.append(str(sample_dir) + '/')  # trailing slash for consistency

    for cap in video_caps + mask_caps:
        cap.release()

    for split, samples in [('train', train_paths), ('val', val_paths)]:
        list_path = output_path / f"data_mouse_{split}.txt"
        with open(list_path, 'w') as f:
            f.write('\n'.join(sorted(samples)))
        print(f"Saved {split} list: {len(samples)} samples -> {list_path}")

    print("\n" + "=" * 60)
    print("D7_5b Preprocessing Complete!")
    print("=" * 60)
    print(f"Method: Object-aware optimal scaling with PP=256")
    print(f"Result: Object never clipped, PP=(256,256), fx varies")
    print(f"Safety margin: {config.safety_margin:.0%}")


def main():
    parser = argparse.ArgumentParser(
        description="D7_5b: PP=256 + Object-Aware Maximum Scaling"
    )
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--camera-pkl', required=True)
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--safety-margin', type=float, default=1.05)

    args = parser.parse_args()

    config = D75bConfig(safety_margin=args.safety_margin)

    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        camera_pkl=args.camera_pkl,
        config=config,
        frame_interval=args.frame_interval,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
