#!/usr
    PreprocessVersion.V13: VersionConfig(
        apply_centering=True,
        correct_cxcy=False,  # cx=cy=256 forced
        preserve_fy_ratio=False,  # fx=fy=549 (FaceLift compatible)
        description="v13: FaceLift compatible (centered + square pixels fx=fy)"
    ),/bin/env python3
"""
Unified Markerless Mouse Preprocessing Script

Supports multiple preprocessing versions via --version flag:
- v5:  Original (cx=cy=256 fixed, centroid centering) - DEPRECATED
- v10: cx,cy corrected (shift + accurate principal point)
- v11: No shift (accurate geometry, object off-center)

Usage:
    python convert_markerless_unified.py --version v10 --input_dir ... --output_dir ...
    python convert_markerless_unified.py --version v11 --input_dir ... --output_dir ...
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# Constants
JUMP_FRAMES = {5900, 11800, 17700}
ORIG_SIZE = (1152, 1024)  # width, height
TARGET_SIZE = 512
UNIT_SCALE = 100.0  # mm → normalized
TARGET_DISTANCE = 2.7
TARGET_FX = 549.0


class PreprocessVersion(Enum):
    V5 = "v5"    # Original (deprecated)
    V10 = "v10"  # cx,cy corrected with shift
    V11 = "v11"  # No shift, accurate geometry
    V12 = "v12"  # Like data_mouse_correct (cx=cy=256 forced)
    V13 = "v13"  # FaceLift compatible (square pixels, centered)


@dataclass
class VersionConfig:
    """Configuration for each preprocessing version."""
    apply_centering: bool
    correct_cxcy: bool
    preserve_fy_ratio: bool  # Whether to preserve orig fy/fx ratio
    description: str


VERSION_CONFIGS = {
    PreprocessVersion.V5: VersionConfig(
        apply_centering=True,
        correct_cxcy=False,  # cx=cy=256 fixed (incorrect!)
        preserve_fy_ratio=False,
        description="Original v5: centroid centering, cx=cy=256 fixed (DEPRECATED)"
    ),
    PreprocessVersion.V10: VersionConfig(
        apply_centering=True,
        correct_cxcy=True,  # cx,cy reflect actual position
        preserve_fy_ratio=True,  # Match data_mouse_correct behavior
        description="v10: centroid centering + cx,cy corrected (RECOMMENDED)"
    ),
    PreprocessVersion.V11: VersionConfig(
        apply_centering=False,
        correct_cxcy=True,  # Original cx,cy preserved
        preserve_fy_ratio=False,  # Force square pixels
        description="v11: No centering, accurate geometry"
    ),
    PreprocessVersion.V12: VersionConfig(
        apply_centering=True,
        correct_cxcy=False,  # cx=cy=256 forced (like data_mouse_correct)
        preserve_fy_ratio=True,  # Preserve fy ratio
        description="v12: centroid centering + cx=cy=256 forced (matches data_mouse_correct)"
    ),
}


def load_camera_params(pkl_path: str) -> List[Dict]:
    """Load camera parameters from pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Compute centroid of foreground pixels in mask."""
    fg_coords = np.where(mask > 127)
    if len(fg_coords[0]) == 0:
        return mask.shape[1] / 2, mask.shape[0] / 2
    centroid_y = fg_coords[0].mean()
    centroid_x = fg_coords[1].mean()
    return centroid_x, centroid_y


def compute_camera_transform(
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
    version_config: VersionConfig,
) -> Tuple[Dict, float, Tuple[float, float]]:
    """
    Compute camera transform based on version configuration.
    """
    orig_w, orig_h = ORIG_SIZE
    orig_fx = K[0, 0]
    orig_fy = K[1, 1]
    orig_cx = K[0, 2]
    orig_cy = K[1, 2]

    # === 1. Extrinsics ===
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    current_dist_mm = np.linalg.norm(cam_pos)
    current_dist_norm = current_dist_mm / UNIT_SCALE

    # === 2. Image scale ===
    # Scale image so that fx becomes TARGET_FX (549)
    # Distance normalization is handled separately in extrinsics (Step 3)
    # DO NOT multiply by dist_ratio - this was a bug causing object size inconsistency
    # Correct formula: total_image_scale = TARGET_FX / orig_fx
    fx_ratio = TARGET_FX / orig_fx
    total_image_scale = fx_ratio  # NOT fx_ratio * dist_ratio!

    # === 3. Distance normalization ===
    cam_direction = cam_pos / current_dist_mm
    new_cam_pos = cam_direction * TARGET_DISTANCE
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = new_cam_pos
    new_w2c = np.linalg.inv(new_c2w)

    # === 4. Scaled principal point ===
    scaled_cx = orig_cx * total_image_scale
    scaled_cy = orig_cy * total_image_scale

    # === 5. Centering (version-dependent) ===
    if version_config.apply_centering:
        centroid_x, centroid_y = compute_mask_centroid(mask)
        scaled_centroid_x = centroid_x * total_image_scale
        scaled_centroid_y = centroid_y * total_image_scale
        center_offset_x = TARGET_SIZE / 2 - scaled_centroid_x
        center_offset_y = TARGET_SIZE / 2 - scaled_centroid_y
    else:
        # v11: No centering - center crop based on scaled principal point
        center_offset_x = TARGET_SIZE / 2 - scaled_cx
        center_offset_y = TARGET_SIZE / 2 - scaled_cy
        scaled_centroid_x = scaled_centroid_y = 0  # Not used

    # === 6. Final cx, cy (version-dependent) ===
    if version_config.correct_cxcy:
        # v10, v11: Accurate cx, cy
        new_cx = scaled_cx + center_offset_x
        new_cy = scaled_cy + center_offset_y
    else:
        # v5: Fixed (incorrect!)
        new_cx = TARGET_SIZE / 2
        new_cy = TARGET_SIZE / 2

    camera_dict = {
        "w": TARGET_SIZE,
        "h": TARGET_SIZE,
        "fx": TARGET_FX,
        "fy": TARGET_FX * (orig_fy / orig_fx) if version_config.preserve_fy_ratio else TARGET_FX,
        "cx": float(new_cx),
        "cy": float(new_cy),
        "w2c": new_w2c.tolist(),
        "_transform": {
            "image_scale": float(total_image_scale),
            "center_offset": [float(center_offset_x), float(center_offset_y)],
            "scaled_cx": float(scaled_cx),
            "scaled_cy": float(scaled_cy),
        },
    }

    return camera_dict, total_image_scale, (center_offset_x, center_offset_y)


def apply_image_transform(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
    center_offset: Tuple[float, float],
) -> np.ndarray:
    """Apply scaling and centering transform to image."""
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w <= 0 or new_h <= 0:
        return np.full((TARGET_SIZE, TARGET_SIZE, 4), [255, 255, 255, 0], dtype=np.uint8)

    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    output = np.full((TARGET_SIZE, TARGET_SIZE, 4), [255, 255, 255, 0], dtype=np.uint8)

    offset_x = int(center_offset[0])
    offset_y = int(center_offset[1])

    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(new_w, TARGET_SIZE - offset_x)
    src_y2 = min(new_h, TARGET_SIZE - offset_y)

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2, :3] = scaled_img[src_y1:src_y2, src_x1:src_x2]
        output[dst_y1:dst_y2, dst_x1:dst_x2, 3] = scaled_mask[src_y1:src_y2, src_x1:src_x2]

    bg_mask = output[:, :, 3] < 127
    output[bg_mask, :3] = 255

    return output


def process_frame(
    frame_idx: int,
    video_caps: List[cv2.VideoCapture],
    mask_caps: List[cv2.VideoCapture],
    cam_params: List[Dict],
    output_dir: Path,
    version_config: VersionConfig,
) -> bool:
    """Process a single frame across all cameras."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    camera_frames = []

    for cam_idx in range(6):
        video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = video_caps[cam_idx].read()
        ret_mask, mask = mask_caps[cam_idx].read()

        if not ret or not ret_mask:
            return False

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cam = cam_params[cam_idx]
        cam_dict, image_scale, center_offset = compute_camera_transform(
            cam["K"], cam["R"], cam["T"], mask_binary, version_config
        )

        transformed = apply_image_transform(
            frame_rgb, mask_binary, image_scale, center_offset
        )

        img_path = images_dir / ("cam_%03d.png" % cam_idx)
        img_pil = Image.fromarray(transformed, mode="RGBA")
        img_pil.save(img_path)

        cam_dict["file_path"] = "images/cam_%03d.png" % cam_idx
        cam_dict["view_id"] = cam_idx
        camera_frames.append(cam_dict)

    cameras_json = {
        "frames": camera_frames,
        "_preprocessing": {
            "method": "convert_markerless_unified",
            "version": version_config.description,
        },
    }

    with open(output_dir / "opencv_cameras.json", "w") as f:
        json.dump(cameras_json, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(description="Unified Mouse Preprocessing")
    parser.add_argument("--version", type=str, required=True, choices=["v5", "v10", "v11", "v12", "v13"])
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    version = PreprocessVersion(args.version)
    version_config = VERSION_CONFIGS[version]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Unified Markerless Mouse Preprocessing")
    print("=" * 70)
    print("Version: %s" % args.version)
    print("Config: %s" % version_config.description)
    print("  - apply_centering: %s" % version_config.apply_centering)
    print("  - correct_cxcy: %s" % version_config.correct_cxcy)
    print("=" * 70)

    cam_params = load_camera_params(input_dir / "new_cam.pkl")
    print("Loaded %d cameras" % len(cam_params))

    video_dir = input_dir / "videos_undist"
    mask_dir = input_dir / "simpleclick_undist"

    video_caps = [cv2.VideoCapture(str(video_dir / ("%d.mp4" % i))) for i in range(6)]
    mask_caps = [cv2.VideoCapture(str(mask_dir / ("%d.mp4" % i))) for i in range(6)]

    total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: %d" % total_frames)

    frame_indices = [i for i in range(0, total_frames, args.frame_interval) if i not in JUMP_FRAMES]

    if args.max_samples:
        frame_indices = frame_indices[:args.max_samples]

    print("Processing %d frames..." % len(frame_indices))

    sample_paths = []
    for idx, frame_idx in enumerate(tqdm(frame_indices)):
        sample_dir = output_dir / ("sample_%06d" % idx)
        if process_frame(frame_idx, video_caps, mask_caps, cam_params, sample_dir, version_config):
            sample_paths.append(str(sample_dir.absolute()))

    for cap in video_caps + mask_caps:
        cap.release()

    num_samples = len(sample_paths)
    num_val = min(100, max(10, int(num_samples * 0.05)))
    num_train = num_samples - num_val

    with open(output_dir / "data_mouse_train.txt", "w") as f:
        f.write("\n".join(sample_paths[:num_train]))
    with open(output_dir / "data_mouse_val.txt", "w") as f:
        f.write("\n".join(sample_paths[num_train:]))

    print("\nDone! %d samples (Train: %d, Val: %d)" % (num_samples, num_train, num_val))


if __name__ == "__main__":
    main()

# Add after VERSION_CONFIGS definition - V12 for cx=cy=256 forced
# This patch adds a new version that matches data_mouse_correct behavior
