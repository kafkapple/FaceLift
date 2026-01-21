#!/usr/bin/env python3
"""
Unified Mouse Preprocessor
==========================

Single entry point for all preprocessing versions (D7+).

Usage:
    # Using preset
    python -m mouse_extensions.preprocessing.preprocess --preset D8.1 \
        --input-dir /path/to/raw --output-dir /path/to/D8.1

    # Custom settings
    python -m mouse_extensions.preprocessing.preprocess \
        --transform homography --zoom 1.3 \
        --input-dir /path/to/raw --output-dir /path/to/custom

Presets:
    D7    : PP-centered shift, affine (fx_only scale)
    D7.1  : PP-centered shift, affine (individual scale) [LEGACY RECOMMENDED]
    D7.2  : PP-centered shift, affine (average scale)
    D8    : Precision homography, skew correction [RECOMMENDED]
    D8.1  : D8 + 1.3x zoom for larger mouse

Created: 2026-01-21
Updated: 2026-01-21 - Added auto config generation
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .presets import PRESETS, get_preset, get_recommended


# Constants
GSLRM_EXACT_FX = 548.9937744140625
JUMP_FRAMES = {5900, 11800, 17700}

# Config generation paths
FACELIFT_ROOT = Path("/home/joon/dev/FaceLift")
DATASET_CONFIG_DIR = FACELIFT_ROOT / "configs" / "datasets"


class TransformType(Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"


class ScaleMode(Enum):
    FX_ONLY = "fx_only"
    INDIVIDUAL = "individual"
    AVERAGE = "average"


@dataclass
class PreprocessConfig:
    """Unified preprocessing configuration."""
    input_dir: Path = None
    output_dir: Path = None
    camera_pkl: Path = None
    frame_interval: int = 5
    max_samples: Optional[int] = None
    val_ratio: float = 0.1
    transform: TransformType = TransformType.HOMOGRAPHY
    scale_mode: ScaleMode = ScaleMode.INDIVIDUAL
    skew_correction: bool = True
    target_fx: float = GSLRM_EXACT_FX
    target_pp: Tuple[float, float] = (256.0, 256.0)
    target_distance: float = 2.7
    output_size: int = 512
    zoom: float = 1.0
    version: str = "D8"

    @classmethod
    def from_preset(cls, preset_name: str, **overrides):
        preset = get_preset(preset_name)
        config = cls()
        config.version = preset_name

        if preset.get('transform') == 'homography' or preset_name.startswith('D8'):
            config.transform = TransformType.HOMOGRAPHY
            config.skew_correction = preset.get('skew_correction', True)
        else:
            config.transform = TransformType.AFFINE
            config.skew_correction = False

        config.scale_mode = ScaleMode(preset.get('scale_mode', 'individual'))
        config.target_fx = preset.get('target_fx', GSLRM_EXACT_FX)
        config.zoom = preset.get('zoom', 1.0)

        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return config


class UnifiedPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.cameras = None
        self.num_views = 0

    def load_cameras(self, pkl_path: Path):
        with open(pkl_path, 'rb') as f:
            self.cameras = pickle.load(f)
        self.num_views = len(self.cameras)

    def compute_affine_transform(self, K: np.ndarray) -> np.ndarray:
        cfg = self.config
        orig_fx, orig_fy = K[0, 0], K[1, 1]
        orig_cx, orig_cy = K[0, 2], K[1, 2]

        scale_x = cfg.target_fx / orig_fx
        scale_y = cfg.target_fx / orig_fy

        if cfg.scale_mode == ScaleMode.FX_ONLY:
            scale_y = scale_x
        elif cfg.scale_mode == ScaleMode.AVERAGE:
            avg = (scale_x + scale_y) / 2
            scale_x = scale_y = avg

        shift_x = cfg.target_pp[0] - orig_cx * scale_x
        shift_y = cfg.target_pp[1] - orig_cy * scale_y

        return np.array([[scale_x, 0, shift_x], [0, scale_y, shift_y]], dtype=np.float32)

    def compute_homography_transform(self, K: np.ndarray) -> np.ndarray:
        cfg = self.config
        K_target = np.array([
            [cfg.target_fx, 0, cfg.target_pp[0]],
            [0, cfg.target_fx, cfg.target_pp[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        return (K_target @ np.linalg.inv(K)).astype(np.float32)

    def compute_transform(self, K: np.ndarray) -> np.ndarray:
        if self.config.transform == TransformType.HOMOGRAPHY:
            return self.compute_homography_transform(K)
        return self.compute_affine_transform(K)

    def apply_transform(self, image: np.ndarray, mask: np.ndarray, transform: np.ndarray):
        cfg = self.config
        size = cfg.output_size

        if cfg.transform == TransformType.HOMOGRAPHY:
            img_out = cv2.warpPerspective(image, transform, (size, size),
                flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            mask_out = cv2.warpPerspective(mask, transform, (size, size),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            img_out = cv2.warpAffine(image, transform, (size, size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
            mask_out = cv2.warpAffine(mask, transform, (size, size),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return img_out, mask_out

    def apply_zoom(self, image: np.ndarray, mask: np.ndarray, zoom: float):
        if zoom <= 1.0:
            return image, mask, (0, 0)

        size = self.config.output_size
        crop_size = int(size / zoom)

        ys, xs = np.where(mask > 127)
        cx = (xs.min() + xs.max()) / 2 if len(xs) > 0 else size / 2
        cy = (ys.min() + ys.max()) / 2 if len(ys) > 0 else size / 2

        crop_x = max(0, min(int(cx - crop_size/2), size - crop_size))
        crop_y = max(0, min(int(cy - crop_size/2), size - crop_size))

        cropped_img = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
        cropped_mask = mask[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

        zoomed_img = cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_LANCZOS4)
        zoomed_mask = cv2.resize(cropped_mask, (size, size), interpolation=cv2.INTER_NEAREST)

        return zoomed_img, zoomed_mask, (crop_x, crop_y)

    def compute_camera_params(self, cam: Dict, zoom: float = 1.0, crop_offset = (0, 0)) -> Dict:
        cfg = self.config
        K, R, T = cam['K'], cam['R'], cam['T']

        if cfg.transform == TransformType.HOMOGRAPHY:
            fx = fy = cfg.target_fx
        else:
            orig_fx, orig_fy = K[0, 0], K[1, 1]
            scale_x = cfg.target_fx / orig_fx
            scale_y = cfg.target_fx / orig_fy
            if cfg.scale_mode == ScaleMode.FX_ONLY:
                scale_y = scale_x
            elif cfg.scale_mode == ScaleMode.AVERAGE:
                scale_x = scale_y = (scale_x + scale_y) / 2
            fx, fy = orig_fx * scale_x, orig_fy * scale_y

        cx, cy = cfg.target_pp

        if zoom > 1.0:
            crop_x, crop_y = crop_offset
            fx, fy = fx * zoom, fy * zoom
            cx, cy = (cx - crop_x) * zoom, (cy - crop_y) * zoom

        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, T.flatten()
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        dist_scale = cfg.target_distance / np.linalg.norm(cam_pos)
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = cam_pos * dist_scale
        new_w2c = np.linalg.inv(new_c2w)

        return {
            "w": cfg.output_size, "h": cfg.output_size,
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "w2c": new_w2c.tolist(),
            "_transform": {"method": cfg.version, "zoom": zoom, "skew_corrected": cfg.skew_correction}
        }

    def process_frame(self, frame_idx: int, video_caps, mask_caps, transforms):
        cfg = self.config
        images, masks = [], []

        for cam_idx in range(self.num_views):
            video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video_caps[cam_idx].read()
            ret_mask, mask = mask_caps[cam_idx].read()
            if not ret or not ret_mask:
                return None
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks.append(np.where(mask > 127, 255, 0).astype(np.uint8))

        if all(m.sum() < 100 for m in masks):
            return None

        proc_images, proc_masks, cam_params = [], [], []
        for cam_idx in range(self.num_views):
            img, mask = self.apply_transform(images[cam_idx], masks[cam_idx], transforms[cam_idx])
            img, mask, crop_offset = self.apply_zoom(img, mask, cfg.zoom)
            proc_images.append(img)
            proc_masks.append(mask)
            params = self.compute_camera_params(self.cameras[cam_idx], cfg.zoom, crop_offset)
            params['file_path'] = f"images/cam_{cam_idx:03d}.png"
            params['view_id'] = cam_idx
            cam_params.append(params)

        return {'images': proc_images, 'masks': proc_masks, 'cameras': cam_params, 'frame_idx': frame_idx}

    def save_sample(self, sample: Dict, sample_dir: Path):
        sample_dir.mkdir(parents=True, exist_ok=True)
        images_dir = sample_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for i, (img, mask) in enumerate(zip(sample['images'], sample['masks'])):
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3], rgba[:, :, 3] = img, mask
            Image.fromarray(rgba).save(images_dir / f"cam_{i:03d}.png")

        with open(sample_dir / "opencv_cameras.json", 'w') as f:
            json.dump({"frames": sample['cameras'], "_preprocessing": {
                "version": self.config.version, "frame_idx": sample['frame_idx']}}, f, indent=2)

    def generate_dataset_config(self, num_train: int, num_val: int):
        """Generate dataset config YAML file for training."""
        cfg = self.config
        
        # Normalize version name for config file (D8.1 -> D8_1)
        config_name = cfg.version.replace('.', '_')
        config_path = DATASET_CONFIG_DIR / f"{config_name}.yaml"
        
        # Get preset info for description
        preset_info = PRESETS.get(cfg.version, {})
        description = preset_info.get('description', f'{cfg.version} preprocessing')
        
        # Determine category and status
        if cfg.version.startswith('D8'):
            category = 'precision_homography'
            status = 'RECOMMENDED' if cfg.version == 'D8' else 'ACTIVE'
        elif cfg.version.startswith('D7'):
            category = 'pp_centered_shift'
            status = 'ACTIVE'
        else:
            category = 'legacy'
            status = 'DEPRECATED'
        
        # Build YAML content
        yaml_content = f"""# =============================================================================
# Dataset: {config_name}
# =============================================================================
# {description}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# =============================================================================

_name: {config_name}
_description: "{description}"
_category: {category}
_status: {status}
_split_type: random
_preprocessing:
  transform: {cfg.transform.value}
  scale_mode: {cfg.scale_mode.value}
  target_fx: {cfg.target_fx}
  zoom: {cfg.zoom}
  skew_correction: {str(cfg.skew_correction).lower()}

training:
  dataset:
    dataset_path: {cfg.output_dir}/data_mouse_train.txt

validation:
  dataset_path: {cfg.output_dir}/data_mouse_val.txt

# Statistics
_stats:
  num_train: {num_train}
  num_val: {num_val}
"""
        
        # Ensure directory exists
        DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Write config file
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n[Config Generated] {config_path}")
        return config_path

    def run(self):
        cfg = self.config
        print(f"Loading cameras: {cfg.camera_pkl}")
        self.load_cameras(cfg.camera_pkl)

        video_dir, mask_dir = cfg.input_dir / "videos_undist", cfg.input_dir / "simpleclick_undist"
        video_caps = [cv2.VideoCapture(str(video_dir / f"{i}.mp4")) for i in range(self.num_views)]
        mask_caps = [cv2.VideoCapture(str(mask_dir / f"{i}.mp4")) for i in range(self.num_views)]

        total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [i for i in range(0, total_frames, cfg.frame_interval) if i not in JUMP_FRAMES]
        if cfg.max_samples:
            frame_indices = frame_indices[:cfg.max_samples]

        print(f"\n{'='*60}\nPreprocessing: {cfg.version}\n{'='*60}")
        print(f"Transform: {cfg.transform.value}, Scale: {cfg.scale_mode.value}, Zoom: {cfg.zoom}x")

        transforms = [self.compute_transform(cam['K']) for cam in self.cameras]
        samples = [s for s in (self.process_frame(f, video_caps, mask_caps, transforms) 
                              for f in tqdm(frame_indices)) if s]

        for cap in video_caps + mask_caps:
            cap.release()

        np.random.seed(42)
        indices = list(range(len(samples)))
        np.random.shuffle(indices)
        val_indices = set(indices[:int(len(samples) * cfg.val_ratio)])

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        train_paths, val_paths = [], []

        for i, sample in enumerate(tqdm(samples, desc="Saving")):
            split = 'val' if i in val_indices else 'train'
            paths = val_paths if split == 'val' else train_paths
            self.save_sample(sample, cfg.output_dir / split / f"{len(paths):06d}")
            paths.append(str(cfg.output_dir / split / f"{len(paths):06d}") + '/')

        for split, paths in [('train', train_paths), ('val', val_paths)]:
            with open(cfg.output_dir / f"data_mouse_{split}.txt", 'w') as f:
                f.write('\n'.join(sorted(paths)))

        with open(cfg.output_dir / "metadata.json", 'w') as f:
            json.dump({"version": cfg.version, "transform": cfg.transform.value,
                      "zoom": cfg.zoom, "num_train": len(train_paths), "num_val": len(val_paths)}, f, indent=2)

        # Auto-generate dataset config
        self.generate_dataset_config(len(train_paths), len(val_paths))

        print(f"\nComplete: {cfg.output_dir}\nTrain: {len(train_paths)}, Val: {len(val_paths)}")


def main():
    parser = argparse.ArgumentParser(description="Unified Mouse Preprocessor")
    parser.add_argument('--preset', '-p', choices=list(PRESETS.keys()))
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--camera-pkl', default=None)
    parser.add_argument('--frame-interval', type=int, default=5)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--transform', choices=['affine', 'homography'])
    parser.add_argument('--scale-mode', choices=['fx_only', 'individual', 'average'])
    parser.add_argument('--zoom', type=float)
    parser.add_argument('--no-config', action='store_true', help='Skip config file generation')
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    camera_pkl = Path(args.camera_pkl) if args.camera_pkl else input_dir / "new_cam.pkl"

    if args.preset:
        config = PreprocessConfig.from_preset(args.preset, input_dir=input_dir, output_dir=output_dir,
            camera_pkl=camera_pkl, frame_interval=args.frame_interval, 
            max_samples=args.max_samples, val_ratio=args.val_ratio)
    else:
        config = PreprocessConfig(input_dir=input_dir, output_dir=output_dir, camera_pkl=camera_pkl,
            frame_interval=args.frame_interval, max_samples=args.max_samples, val_ratio=args.val_ratio)

    if args.transform:
        config.transform = TransformType(args.transform)
    if args.scale_mode:
        config.scale_mode = ScaleMode(args.scale_mode)
    if args.zoom:
        config.zoom = args.zoom

    UnifiedPreprocessor(config).run()


if __name__ == "__main__":
    main()
