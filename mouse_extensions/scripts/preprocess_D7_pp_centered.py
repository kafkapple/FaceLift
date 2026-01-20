#!/usr/bin/env python3
"""
D7 Preprocessing: PP-Centered Shift
====================================

Mathematical justification:
- Shifting image by Δ and updating PP by the same Δ is mathematically equivalent
- u' = u + Δ  =>  cx' = cx + Δ
- This preserves geometric consistency while achieving PP = (256, 256)

Key difference from other versions:
- D4: Object-centered crop + PP=256 forced (WRONG - inconsistent)
- D6-3: Object-centered crop + PP varies (CORRECT)
- D7: PP-centered shift + PP=256 (CORRECT - mathematically consistent)

Trade-off:
- Object may NOT be at image center (up to ~30px off)
- But PP matches FaceLift pretrained model expectation (cx=cy=256)

Reference:
- https://ksimek.github.io/2013/08/13/intrinsic/
- https://stackoverflow.com/questions/74749690/

Created: 2026-01-18
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
class D7Config:
    """D7 preprocessing configuration"""
    target_fx: float = 549.0
    target_fy: float = 549.0  # Square pixels
    scale_mode: str = "fx_only"  # fx_only (D7), individual (D7.1), average (D7.2)
    target_pp: Tuple[float, float] = (256.0, 256.0)  # Target principal point
    target_distance: float = 2.7
    output_size: int = 512
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 0)  # White with alpha=0


class D7Preprocessor:
    """
    D7 Preprocessing: PP-Centered Shift

    Unlike D6-3 which crops around object center, D7 shifts the image
    so that the principal point lands at (256, 256). This is mathematically
    correct and matches FaceLift's expected intrinsics.
    """

    def __init__(self, config: D7Config = None):
        self.config = config or D7Config()

    def load_original_cameras(self, pkl_path: str) -> List[Dict]:
        """Load original camera parameters from pickle file"""
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def compute_transform(
        self,
        K: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        orig_size: Tuple[int, int] = (1152, 1024)
    ) -> Tuple[Dict, np.ndarray]:
        """
        Compute camera transform and affine matrix for PP-centered shift.

        The key insight:
        - We scale the image so fx -> target_fx
        - We shift the image so that the original PP location -> (256, 256)
        - This is mathematically equivalent to: new_cx = 256, new_cy = 256

        Returns:
            camera_dict: New camera parameters with PP = (256, 256)
            affine_matrix: 2x3 affine transform matrix for image warping
        """
        orig_w, orig_h = orig_size
        cfg = self.config

        # Original intrinsics
        orig_fx = K[0, 0]
        orig_fy = K[1, 1]
        orig_cx = K[0, 2]
        orig_cy = K[1, 2]

        # Step 1: Compute scale factor based on scale_mode
        scale_x = cfg.target_fx / orig_fx
        scale_y = cfg.target_fy / orig_fy
        
        if cfg.scale_mode == 'individual':
            # D7.1: Different scale for x and y (geometrically correct)
            pass  # use scale_x, scale_y directly
        elif cfg.scale_mode == 'average':
            # D7.2: Average scale (isotropic)
            avg_scale = (scale_x + scale_y) / 2
            scale_x = scale_y = avg_scale
        else:  # 'fx_only' (D7 default)
            # D7: Use fx scale for both (original behavior)
            scale_y = scale_x

        # Step 2: Compute scaled PP
        scaled_cx = orig_cx * scale_x
        scaled_cy = orig_cy * scale_y

        # Step 3: Compute shift to move PP to target (256, 256)
        # This is the KEY DIFFERENCE from D6-3!
        # D6-3: shift to center the OBJECT
        # D7: shift to center the PP (optical axis)
        shift_x = cfg.target_pp[0] - scaled_cx
        shift_y = cfg.target_pp[1] - scaled_cy

        # Step 4: Build affine transform matrix
        # Combined scale + shift:
        # x' = scale * x + shift_x
        # y' = scale * y + shift_y
        affine_matrix = np.array([
            [scale_x, 0, shift_x],
            [0, scale_y, shift_y]
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

        # Step 6: Build camera dict
        camera_dict = {
            "w": cfg.output_size,
            "h": cfg.output_size,
            "fx": cfg.target_fx if cfg.scale_mode != 'average' else float(orig_fx * scale_x),
            "fy": cfg.target_fy if cfg.scale_mode != 'average' else float(orig_fy * scale_y),
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
                "method": f"D7_PP_centered_shift_{cfg.scale_mode}",
                "scale_mode": cfg.scale_mode,
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "scale_avg": float((scale_x + scale_y) / 2),
                "shift_x": float(shift_x),
                "shift_y": float(shift_y),
                "scaled_pp_before_shift": [float(scaled_cx), float(scaled_cy)],
                "target_distance": cfg.target_distance,
            },
        }

        return camera_dict, affine_matrix

    def transform_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        affine_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply affine transform to image (scale + shift).

        The shift moves the image so that the original PP location
        ends up at (256, 256) in the output image.
        """
        cfg = self.config
        output_size = cfg.output_size

        # Apply affine warp to image
        warped_img = cv2.warpAffine(
            image,
            affine_matrix,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # White background
        )

        # Apply affine warp to mask (nearest neighbor)
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

        # Create RGBA output
        output = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        output[:, :, :3] = warped_img
        output[:, :, 3] = warped_mask

        return output


def process_dataset(
    data_dir: str,
    output_dir: str,
    camera_pkl: str,
    config: D7Config = None,
    frame_interval: int = 5,
    max_samples: Optional[int] = None,
    val_ratio: float = 0.1
):
    """
    Process full dataset with D7 preprocessing.

    Args:
        data_dir: Directory containing videos_undist/ and simpleclick_undist/
        output_dir: Output directory for preprocessed data
        camera_pkl: Path to original camera pickle file
        config: D7 preprocessing config
        frame_interval: Sample every N frames
        max_samples: Maximum samples to process (None for all)
        val_ratio: Fraction of data for validation
    """
    if config is None:
        config = D7Config(scale_mode=args.scale_mode)

    preprocessor = D7Preprocessor(config)

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

    # Sample frames (skip jump frames)
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

    # Precompute camera transforms (same for all frames)
    print("\nComputing camera transforms (PP-centered shift)...")
    camera_dicts = []
    affine_matrices = []

    for i, cam in enumerate(cameras):
        K = cam['K']
        R = cam['R']
        T = cam['T']

        cam_dict, affine = preprocessor.compute_transform(K, R, T)
        # Add file_path and view_id for dataset loader compatibility
        cam_dict['file_path'] = f"images/cam_{i:03d}.png"
        cam_dict['view_id'] = i
        camera_dicts.append(cam_dict)
        affine_matrices.append(affine)

        shift_x = cam_dict['_transform']['shift_x']
        shift_y = cam_dict['_transform']['shift_y']
        print(f"  View {i}: shift=({shift_x:+.1f}, {shift_y:+.1f}) -> PP=(256, 256)")

    # Process frames
    print("\nProcessing frames...")

    train_paths = []
    val_paths = []

    for sample_idx, frame_idx in enumerate(tqdm(frame_indices)):
        # Determine split
        split = 'val' if frame_idx in val_frames else 'train'
        sample_list = val_paths if split == 'val' else train_paths

        # Create sample directory
        sample_id = f"{len(sample_list):06d}"
        sample_dir = output_path / split / sample_id
        images_out = sample_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        # Process each view
        for cam_idx in range(num_views):
            # Read frame from video
            video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = video_caps[cam_idx].read()
            ret_mask, mask = mask_caps[cam_idx].read()

            if not ret or not ret_mask:
                print(f"Warning: Failed to read frame {frame_idx} view {cam_idx}")
                continue

            # Convert mask to grayscale
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply PP-centered transform
            transformed = preprocessor.transform_image(
                frame_rgb, mask_binary, affine_matrices[cam_idx]
            )

            # Save
            out_path = images_out / f"cam_{cam_idx:03d}.png"
            Image.fromarray(transformed).save(out_path)

        # Save camera JSON
        camera_json = {
            "frames": camera_dicts,
            "_preprocessing": {
                "method": "D7_PP_centered_shift",
                "description": "PP-centered shift: cx=cy=256 with geometric consistency",
                "justification": "Image shift by Δ + PP shift by Δ = mathematically equivalent",
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
    print("D7 Preprocessing Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Method: PP-centered shift")
    print(f"Result: cx=cy=256 for all views (mathematically consistent)")
    print(f"Trade-off: Object may be slightly off-center (up to ~30px)")


def main():
    parser = argparse.ArgumentParser(
        description="D7 Preprocessing: PP-Centered Shift for FaceLift compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mathematical Justification:
  When image is shifted by Δ, the projection equation becomes:
    u' = u + Δ = f*X/Z + c + Δ

  This is equivalent to having a new principal point c' = c + Δ.

  D7 computes Δ = 256 - scaled_cx, so the resulting PP is exactly 256.
  This matches FaceLift's pretrained model expectation (cx=cy=256).

Examples:
  # Process with default settings
  python preprocess_D7_pp_centered.py \\
      --data-dir /home/joon/data/markerless_mouse_1_nerf \\
      --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7 \\
      --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl

  # Custom settings
  python preprocess_D7_pp_centered.py \\
      --data-dir /path/to/data \\
      --output-dir /path/to/D7 \\
      --camera-pkl /path/to/cam.pkl \\
      --frame-interval 10 \\
      --max-samples 500
        """
    )

    parser.add_argument('--data-dir', required=True,
                        help='Directory containing videos_undist/ and simpleclick_undist/')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for preprocessed data')
    parser.add_argument('--camera-pkl', required=True,
                        help='Camera pickle file (new_cam.pkl)')
    parser.add_argument('--frame-interval', type=int, default=5,
                        help='Frame sampling interval (default: 5)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to process (default: all)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--target-fx', type=float, default=549.0,
                        help='Target focal length (default: 549)')
    parser.add_argument('--scale-mode', type=str, default='fx_only',
                        choices=['fx_only', 'individual', 'average'],
                        help='Scale mode: fx_only (D7), individual (D7.1), average (D7.2)')
    parser.add_argument('--target-distance', type=float, default=2.7,
                        help='Target camera distance (default: 2.7)')

    args = parser.parse_args()

    config = D7Config(
        target_fx=args.target_fx,
        target_fy=args.target_fx,  # Square pixels
        target_distance=args.target_distance,
        scale_mode=args.scale_mode,
    )

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
