#!/usr/bin/env python3
"""
D7_5 Preprocessing: PP=256 + Maximum Coverage
=============================================

Goal: PP=(256,256) exactly while maximizing image coverage (minimizing padding)

Trade-off analysis:
- D7: fx=549 fixed, PP=256, ~56% coverage (lots of padding)
- D7_5: PP=256, 100% coverage, fx varies per view (~695-730)

Mathematical derivation:
  For entire image to fit in 512×512 after PP-centering shift:
  - Top-left (0,0) → (256 - cx*s, 256 - cy*s) >= (0, 0)
  - Bottom-right (W,H) → (256 + (W-cx)*s, 256 + (H-cy)*s) <= (512, 512)

  Maximum scale: s = min(256/cx, 256/cy, 256/(W-cx), 256/(H-cy))

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
class D75Config:
    """D7_5 preprocessing configuration"""
    target_pp: Tuple[float, float] = (256.0, 256.0)  # Target principal point
    target_distance: float = 2.7
    output_size: int = 512
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)


class D75Preprocessor:
    """
    D7_5 Preprocessing: PP=256 + Maximum Coverage

    Unlike D7 which fixes fx=549 (causing padding), D7_5 finds the optimal
    scale that fits the entire image while centering PP at (256, 256).
    """

    def __init__(self, config: D75Config = None):
        self.config = config or D75Config()

    def load_original_cameras(self, pkl_path: str) -> List[Dict]:
        """Load original camera parameters from pickle file"""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def compute_optimal_scale(
        self,
        cx: float,
        cy: float,
        orig_w: int = 1152,
        orig_h: int = 1024
    ) -> float:
        """
        Compute maximum scale that fits entire image in 512×512
        while centering PP at (256, 256).

        Constraints:
        - 256 - cx*s >= 0  → s <= 256/cx
        - 256 - cy*s >= 0  → s <= 256/cy
        - 256 + (W-cx)*s <= 512  → s <= 256/(W-cx)
        - 256 + (H-cy)*s <= 512  → s <= 256/(H-cy)
        """
        target = self.config.target_pp[0]  # 256

        constraints = [
            target / cx,                    # Left edge fits
            target / cy,                    # Top edge fits
            target / (orig_w - cx),         # Right edge fits
            target / (orig_h - cy),         # Bottom edge fits
        ]

        return min(constraints)

    def compute_transform(
        self,
        K: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        orig_size: Tuple[int, int] = (1152, 1024)
    ) -> Tuple[Dict, np.ndarray, dict]:
        """
        Compute camera transform and affine matrix for optimal PP-centered transform.

        Returns:
            camera_dict: New camera parameters with PP = (256, 256)
            affine_matrix: 2x3 affine transform matrix
            stats: Statistics about the transform
        """
        orig_w, orig_h = orig_size
        cfg = self.config

        # Original intrinsics
        orig_fx = K[0, 0]
        orig_fy = K[1, 1]
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]

        # Step 1: Compute optimal scale (maximize coverage while PP=256)
        scale = self.compute_optimal_scale(orig_cx, orig_cy, orig_w, orig_h)

        # Step 2: Compute new focal length (varies per view!)
        new_fx = orig_fx * scale
        new_fy = orig_fy * scale

        # Step 3: Compute shift to move PP to (256, 256)
        scaled_cx = orig_cx * scale
        scaled_cy = orig_cy * scale
        shift_x = cfg.target_pp[0] - scaled_cx
        shift_y = cfg.target_pp[1] - scaled_cy

        # Step 4: Build affine transform matrix
        affine_matrix = np.array([
            [scale, 0, shift_x],
            [0, scale, shift_y]
        ], dtype=np.float32)

        # Step 5: Compute distance normalization (extrinsics)
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

        # Step 6: Compute coverage statistics
        scaled_w = orig_w * scale
        scaled_h = orig_h * scale
        coverage = (scaled_w * scaled_h) / (cfg.output_size ** 2)

        # Step 7: Build camera dict
        camera_dict = {
            "w": cfg.output_size,
            "h": cfg.output_size,
            "fx": new_fx,  # Varies per view!
            "fy": new_fy,
            "cx": cfg.target_pp[0],  # Exactly 256
            "cy": cfg.target_pp[1],  # Exactly 256
            "w2c": new_w2c.tolist(),
            "_original": {
                "fx": float(orig_fx),
                "fy": float(orig_fy),
                "cx": float(orig_cx),
                "cy": float(orig_cy),
                "distance": float(current_distance),
            },
            "_transform": {
                "method": "D7_5_optimal_coverage",
                "scale": float(scale),
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
                "coverage": float(coverage),
                "target_distance": cfg.target_distance,
            },
        }

        stats = {
            "scale": scale,
            "new_fx": new_fx,
            "coverage": coverage,
            "scaled_size": (scaled_w, scaled_h),
        }

        return camera_dict, affine_matrix, stats

    def transform_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        affine_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply affine transform to image (scale + shift)."""
        cfg = self.config
        output_size = cfg.output_size

        # Apply affine warp to image
        warped_img = cv2.warpAffine(
            image,
            affine_matrix,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        # Apply affine warp to mask (nearest neighbor for binary)
        warped_mask = cv2.warpAffine(
            mask,
            affine_matrix,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Combine into RGBA
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
    config: D75Config = None,
    frame_interval: int = 5,
    max_samples: Optional[int] = None,
    val_ratio: float = 0.1
):
    """Process full dataset with D7_5 preprocessing."""
    if config is None:
        config = D75Config()

    preprocessor = D75Preprocessor(config)

    # Load cameras
    print(f"Loading cameras from {camera_pkl}")
    cameras = preprocessor.load_original_cameras(camera_pkl)
    num_views = len(cameras)
    print(f"Found {num_views} camera views")

    # Paths
    data_path = Path(data_dir)
    video_dir = data_path / "videos_undist"
    mask_dir = data_path / "simpleclick_undist"
    output_path = Path(output_dir)

    # Open video captures
    print("Opening video files...")
    video_caps = []
    mask_caps = []

    for cam_idx in range(num_views):
        video_caps.append(cv2.VideoCapture(str(video_dir / f"{cam_idx}.mp4")))
        mask_caps.append(cv2.VideoCapture(str(mask_dir / f"{cam_idx}.mp4")))

    total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Sample frames
    frame_indices = [i for i in range(0, total_frames, frame_interval)
                     if i not in JUMP_FRAMES]

    if max_samples:
        frame_indices = frame_indices[:max_samples]

    print(f"Processing {len(frame_indices)} frames (interval={frame_interval})")

    # Split train/val
    np.random.seed(42)
    shuffled = frame_indices.copy()
    np.random.shuffle(shuffled)
    val_count = int(len(shuffled) * val_ratio)
    val_frames = set(shuffled[:val_count])

    print(f"Train: {len(frame_indices) - val_count}, Val: {val_count}")

    # Create output directories
    for split in ['train', 'val']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Precompute camera transforms
    print("\nComputing camera transforms (optimal coverage)...")
    camera_dicts = []
    affine_matrices = []

    for i, cam in enumerate(cameras):
        K = cam['K']
        R = cam['R']
        T = cam['T']

        cam_dict, affine, stats = preprocessor.compute_transform(K, R, T)
        # Add file_path and view_id for dataset loader compatibility
        cam_dict['file_path'] = f"images/cam_{i:03d}.png"
        cam_dict['view_id'] = i
        camera_dicts.append(cam_dict)
        affine_matrices.append(affine)

        print(f"  View {i}: scale={stats['scale']:.4f}, fx={stats['new_fx']:.0f}, "
              f"coverage={stats['coverage']*100:.1f}%")

    # Summary
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
                print(f"Warning: Failed to read frame {frame_idx} view {cam_idx}")
                continue

            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            transformed = preprocessor.transform_image(
                frame_rgb, mask_binary, affine_matrices[cam_idx]
            )

            out_path = images_out / f"cam_{cam_idx:03d}.png"
            Image.fromarray(transformed).save(out_path)

        camera_json = {
            "frames": camera_dicts,
            "_preprocessing": {
                "method": "D7_5_optimal_coverage",
                "description": "PP=256 with maximum image coverage (fx varies)",
                "original_frame_idx": int(frame_idx),
            }
        }

        with open(sample_dir / "opencv_cameras.json", 'w') as f:
            json.dump(camera_json, f, indent=2)

        sample_list.append(str(sample_dir) + '/')  # trailing slash for consistency

    # Close video captures
    for cap in video_caps + mask_caps:
        cap.release()

    # Save file lists
    for split, samples in [('train', train_paths), ('val', val_paths)]:
        list_path = output_path / f"data_mouse_{split}.txt"
        with open(list_path, 'w') as f:
            f.write('\n'.join(sorted(samples)))
        print(f"Saved {split} list: {len(samples)} samples -> {list_path}")

    print("\n" + "=" * 60)
    print("D7_5 Preprocessing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Method: Optimal coverage with PP=256")
    print(f"Result: PP=(256,256) for all views, fx varies ({min(fx_values):.0f}-{max(fx_values):.0f})")
    print(f"Trade-off: 100% image coverage, but fx != 549")


def main():
    parser = argparse.ArgumentParser(
        description="D7_5 Preprocessing: PP=256 + Maximum Coverage"
    )

    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--camera-pkl', required=True)
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--target-distance', type=float, default=2.7)

    args = parser.parse_args()

    config = D75Config(target_distance=args.target_distance)

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
