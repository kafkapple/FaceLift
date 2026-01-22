#!/usr/bin/env python3
"""
Unified Mouse Preprocessor
==========================

Single entry point for ALL preprocessing versions (D6-D9).

Paradigms:
    geometry_preserving (D6): Accurate PP, various crop strategies
        D6-1: Resize only, letterbox padding
        D6-2: Virtual camera relocation (PP shift)
        D6-3: PP-correct crop with triangulation
    
    pp_centered_shift (D7): Shift image to center PP at 256
        D7  : fx_only scale mode
        D7.1: individual scale mode [RECOMMENDED]
        D7.2: average scale mode
    
    precision_homography (D8): Homography with skew correction
        D8  : Standard precision
        D8.1: With 1.3x zoom
    
    native (D9): Original resolution, no transformation
        For A6000+ GPUs (4.5x memory)

Usage:
    # Using preset
    python -m mouse_extensions.preprocessing.preprocess --preset D7.1 \\
        --input-dir /path/to/raw --output-dir /path/to/D7.1

    # List available presets
    python -m mouse_extensions.preprocessing.preprocess --list-presets

Created: 2026-01-21
Updated: 2026-01-22 - Integrated D6, D9 paradigms
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .presets import PRESETS, get_preset, get_recommended, list_presets as list_preset_names


# Constants
GSLRM_EXACT_FX = 548.9937744140625
JUMP_FRAMES = {5900, 11800, 17700}

# Config generation paths
FACELIFT_ROOT = Path("/home/joon/dev/FaceLift")
DATASET_CONFIG_DIR = FACELIFT_ROOT / "configs" / "datasets"


class Paradigm(Enum):
    OBJECT_CENTERED = "object_centered"
    GEOMETRY_PRESERVING = "geometry_preserving"
    PP_CENTERED_SHIFT = "pp_centered_shift"
    PRECISION_HOMOGRAPHY = "precision_homography"
    NATIVE = "native"


class TransformType(Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"
    NONE = "none"


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
    
    # Transform settings
    paradigm: Paradigm = Paradigm.PP_CENTERED_SHIFT
    transform: TransformType = TransformType.HOMOGRAPHY
    scale_mode: ScaleMode = ScaleMode.INDIVIDUAL
    skew_correction: bool = True
    
    # Target values
    target_fx: float = GSLRM_EXACT_FX
    target_pp: Tuple[float, float] = (256.0, 256.0)
    target_distance: float = 2.7
    output_size: Optional[int] = 512
    zoom: float = 1.0
    
    # D6-specific
    d6_method: str = None  # resize_only, virtual_shift, pp_correct_crop
    object_ratio: float = 0.7
    
    # Normalization flags
    normalize_fx: bool = True
    normalize_translation: bool = True
    
    version: str = "D7.1"

    @classmethod
    def from_preset(cls, preset_name: str, **overrides):
        """Create config from preset."""
        preset = get_preset(preset_name)
        config = cls()
        config.version = preset_name
        
        # Set paradigm
        paradigm_str = preset.get('paradigm', 'pp_centered_shift')
        config.paradigm = Paradigm(paradigm_str)
        
        # ====== GEOMETRY_PRESERVING (D6) ======
        if config.paradigm == Paradigm.GEOMETRY_PRESERVING:
            config.d6_method = preset.get('method')
            config.transform = TransformType.NONE
            config.normalize_fx = preset.get('normalize_fx', True)
            config.normalize_translation = preset.get('normalize_translation', True)
            config.output_size = preset.get('output_size', 512)
            
        # ====== PP_CENTERED_SHIFT (D7) ======
        elif config.paradigm == Paradigm.PP_CENTERED_SHIFT:
            config.transform = TransformType.AFFINE
            config.skew_correction = False
            config.scale_mode = ScaleMode(preset.get('scale_mode', 'individual'))
            config.target_fx = preset.get('target_fx', 549.0)
            
        # ====== PRECISION_HOMOGRAPHY (D8) ======
        elif config.paradigm == Paradigm.PRECISION_HOMOGRAPHY:
            config.transform = TransformType.HOMOGRAPHY
            config.skew_correction = preset.get('skew_correction', True)
            config.scale_mode = ScaleMode(preset.get('scale_mode', 'individual'))
            config.target_fx = preset.get('target_fx', GSLRM_EXACT_FX)
            config.zoom = preset.get('zoom', 1.0)
            
        # ====== NATIVE (D9, D9_norm) ======
        elif config.paradigm == Paradigm.NATIVE:
            config.transform = TransformType.NONE
            config.output_size = None  # Keep original
            # Read normalization settings from preset (D9_norm uses normalize_translation=True)
            config.normalize_fx = preset.get('normalize_fx', False)
            config.normalize_translation = preset.get('normalize_translation', False)
            config.target_distance = preset.get('target_distance', 2.7)

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)

        return config


class UnifiedPreprocessor:
    """Unified preprocessor for all dataset versions."""
    
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.cameras = None
        self.num_views = 0
        self.data_loader = None
        self.center_estimator = None

    def load_cameras(self, pkl_path: Path):
        """Load camera parameters from pickle file."""
        with open(pkl_path, 'rb') as f:
            self.cameras = pickle.load(f)
        self.num_views = len(self.cameras)

    def _init_data_loader(self):
        """Initialize data loader for D6/D9."""
        try:
            from .data_loader import DataLoader
            self.data_loader = DataLoader(self.config.input_dir, source_type="raw")
            self.cameras = self.data_loader.get_camera_matrices()
            self.num_views = len(self.cameras)
        except ImportError:
            raise ImportError("data_loader module required for D6/D9 preprocessing")

    def _init_center_estimator(self):
        """Initialize center estimator for D6-3."""
        try:
            from .center_estimation import CenterEstimator
            self.center_estimator = CenterEstimator(self.cameras, method="triangulation")
        except ImportError:
            raise ImportError("center_estimation module required for D6-3")

    # =========================================================================
    # D7/D8 Methods (existing)
    # =========================================================================
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

    # =========================================================================
    # D6 Methods (geometry_preserving)
    # =========================================================================
    def process_d6_1(self, frame_idx: int) -> Optional[Dict]:
        """D6-1: Resize only, letterbox padding."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        orig_h, orig_w = images[0].shape[:2]
        scale = self.config.output_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        pad_top = (self.config.output_size - new_h) // 2
        pad_left = (self.config.output_size - new_w) // 2

        proc_images, proc_masks, cam_params = [], [], []

        for i in range(self.num_views):
            # Resize + pad
            resized = cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((self.config.output_size, self.config.output_size, 3), 255, dtype=np.uint8)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            proc_images.append(padded)

            mask_resized = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_padded = np.zeros((self.config.output_size, self.config.output_size), dtype=np.uint8)
            mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_resized
            proc_masks.append(mask_padded)

            # Accurate intrinsics
            cam = self.cameras[i]
            params = self._build_d6_camera_params(cam, scale, pad_left, pad_top, i)
            cam_params.append(params)

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, "frame_idx": frame_idx}

    def process_d6_2(self, frame_idx: int) -> Optional[Dict]:
        """D6-2: Virtual camera relocation (PP shift)."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        orig_h, orig_w = images[0].shape[:2]
        scale = self.config.output_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        pad_top = (self.config.output_size - new_h) // 2
        pad_left = (self.config.output_size - new_w) // 2
        target_center = self.config.output_size / 2

        # Compute mouse centroids
        mouse_centroids = []
        for mask in masks:
            coords = np.where(mask > 127)
            if len(coords[0]) > 0:
                cy = coords[0].mean() * scale + pad_top
                cx = coords[1].mean() * scale + pad_left
            else:
                cy, cx = target_center, target_center
            mouse_centroids.append((cx, cy))

        proc_images, proc_masks, cam_params = [], [], []

        for i in range(self.num_views):
            # Same as D6-1
            resized = cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((self.config.output_size, self.config.output_size, 3), 255, dtype=np.uint8)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            proc_images.append(padded)

            mask_resized = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_padded = np.zeros((self.config.output_size, self.config.output_size), dtype=np.uint8)
            mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_resized
            proc_masks.append(mask_padded)

            # Virtual PP shift
            cam = self.cameras[i]
            actual_cx = cam["cx"] * scale + pad_left
            actual_cy = cam["cy"] * scale + pad_top
            
            mouse_cx, mouse_cy = mouse_centroids[i]
            pp_offset_x = mouse_cx - target_center
            pp_offset_y = mouse_cy - target_center
            
            params = self._build_d6_camera_params(cam, scale, pad_left, pad_top, i)
            params["cx"] = actual_cx + pp_offset_x
            params["cy"] = actual_cy + pp_offset_y
            params["_transform"]["pp_offset"] = [pp_offset_x, pp_offset_y]
            cam_params.append(params)

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, "frame_idx": frame_idx}

    def process_d6_3(self, frame_idx: int) -> Optional[Dict]:
        """D6-3: PP-correct crop with triangulation."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        masks_array = np.array(masks)
        result = self.center_estimator.estimate(masks_array)
        centers_2d = [tuple(c) for c in result.centers_2d]

        # Compute global crop size
        max_size = 0
        for mask in masks:
            coords = np.where(mask > 127)
            if len(coords[0]) > 0:
                h = coords[0].max() - coords[0].min()
                w = coords[1].max() - coords[1].min()
                max_size = max(max_size, h, w)
        
        crop_size = max(max_size / self.config.object_ratio, 256)
        scale = self.config.output_size / crop_size

        proc_images, proc_masks, cam_params = [], [], []

        for i in range(self.num_views):
            crop_center = centers_2d[i]
            
            proc_img = self._crop_and_resize(images[i], crop_center, crop_size)
            proc_images.append(proc_img)

            mask_rgb = np.stack([masks[i]] * 3, axis=-1)
            proc_mask = self._crop_and_resize(mask_rgb, crop_center, crop_size, bg=(0, 0, 0))
            proc_masks.append(proc_mask[:, :, 0])

            # ACCURATE PP (not 256!)
            cam = self.cameras[i]
            crop_x = crop_center[0] - crop_size / 2
            crop_y = crop_center[1] - crop_size / 2
            
            params = self._build_d6_camera_params(cam, scale, 0, 0, i)
            params["cx"] = (cam["cx"] - crop_x) * scale
            params["cy"] = (cam["cy"] - crop_y) * scale
            params["_transform"]["crop_center"] = list(crop_center)
            params["_transform"]["crop_size"] = crop_size
            cam_params.append(params)

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, 
                "frame_idx": frame_idx, "center_3d": result.center_3d.tolist() if result.center_3d is not None else None}

    def _build_d6_camera_params(self, cam: Dict, scale: float, pad_left: int, pad_top: int, view_idx: int) -> Dict:
        """Build camera params for D6 methods."""
        cfg = self.config
        
        # Get original values
        fx_orig = cam.get("fx", cam.get("K", np.eye(3))[0, 0])
        fy_orig = cam.get("fy", cam.get("K", np.eye(3))[1, 1])
        cx_orig = cam.get("cx", cam.get("K", np.eye(3))[0, 2])
        cy_orig = cam.get("cy", cam.get("K", np.eye(3))[1, 2])
        
        fx = fx_orig * scale
        fy = fy_orig * scale
        cx = cx_orig * scale + pad_left
        cy = cy_orig * scale + pad_top
        
        # Translation normalization
        w2c_orig = cam.get("w2c", np.eye(4))
        if isinstance(w2c_orig, list):
            w2c_orig = np.array(w2c_orig)
        
        if cfg.normalize_translation:
            c2w = np.linalg.inv(w2c_orig)
            cam_pos = c2w[:3, 3]
            dist_scale = cfg.target_distance / np.linalg.norm(cam_pos)
            c2w[:3, 3] = cam_pos * dist_scale
            w2c = np.linalg.inv(c2w)
        else:
            w2c = w2c_orig.copy()
        
        # fx normalization
        if cfg.normalize_fx:
            fx_norm_scale = cfg.target_fx / fx
            fx = cfg.target_fx
            fy = fy * fx_norm_scale  # Keep aspect ratio
        
        return {
            "file_path": f"images/cam_{view_idx:03d}.png",
            "view_id": view_idx,
            "w": cfg.output_size, "h": cfg.output_size,
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "w2c": w2c.tolist(),
            "_transform": {"method": cfg.version, "scale": scale, "padding": [pad_left, pad_top]}
        }

    def _crop_and_resize(self, image: np.ndarray, crop_center: Tuple[float, float],
                         crop_size: float, bg: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Crop and resize helper for D6-3."""
        h, w = image.shape[:2]
        crop_x = int(crop_center[0] - crop_size / 2)
        crop_y = int(crop_center[1] - crop_size / 2)
        cs = int(crop_size)

        if len(image.shape) == 3:
            output = np.full((cs, cs, 3), bg, dtype=np.uint8)
        else:
            output = np.full((cs, cs), bg[0], dtype=np.uint8)

        src_x1, src_y1 = max(0, crop_x), max(0, crop_y)
        src_x2, src_y2 = min(w, crop_x + cs), min(h, crop_y + cs)
        dst_x1, dst_y1 = max(0, -crop_x), max(0, -crop_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        output[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        return cv2.resize(output, (self.config.output_size, self.config.output_size), 
                         interpolation=cv2.INTER_LINEAR)

    # =========================================================================
    # D9 Method (native)
    # =========================================================================
    def process_d9(self, frame_idx: int) -> Optional[Dict]:
        """D9: Original resolution, no transformation."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        cam_params = []
        for i in range(self.num_views):
            cam = self.cameras[i]
            
            # Keep original intrinsics
            fx = cam.get("fx", cam.get("K", np.eye(3))[0, 0])
            fy = cam.get("fy", cam.get("K", np.eye(3))[1, 1])
            cx = cam.get("cx", cam.get("K", np.eye(3))[0, 2])
            cy = cam.get("cy", cam.get("K", np.eye(3))[1, 2])
            
            w2c = cam.get("w2c", np.eye(4))
            if isinstance(w2c, list):
                w2c = np.array(w2c)
            
            h, w = images[i].shape[:2]
            
            cam_params.append({
                "file_path": f"images/cam_{i:03d}.png",
                "view_id": i,
                "w": w, "h": h,
                "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
                "w2c": w2c.tolist(),
                "_transform": {"method": "D9", "native": True}
            })

        return {"images": images, "masks": masks, "cameras": cam_params, "frame_idx": frame_idx}

    # =========================================================================
    # Main Processing
    # =========================================================================
    def process_frame_d7d8(self, frame_idx: int, video_caps, mask_caps, transforms):
        """Process frame for D7/D8 paradigms."""
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
        """Save processed sample."""
        sample_dir.mkdir(parents=True, exist_ok=True)
        images_dir = sample_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for i, (img, mask) in enumerate(zip(sample['images'], sample['masks'])):
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3], rgba[:, :, 3] = img, mask
            Image.fromarray(rgba).save(images_dir / f"cam_{i:03d}.png")

        with open(sample_dir / "opencv_cameras.json", 'w') as f:
            json.dump({"frames": sample['cameras'], "_preprocessing": {
                "version": self.config.version, "frame_idx": sample['frame_idx']}}, f, indent=2)

    def generate_dataset_config(self, num_train: int, num_val: int):
        """Generate dataset config YAML file."""
        cfg = self.config
        config_name = cfg.version.replace('.', '_').replace('-', '_')
        config_path = DATASET_CONFIG_DIR / f"{config_name}.yaml"
        
        preset_info = PRESETS.get(cfg.version, {})
        description = preset_info.get('description', f'{cfg.version} preprocessing')
        category = preset_info.get('paradigm', 'unknown')
        
        yaml_content = f"""# Dataset: {config_name}
# {description}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

_name: {config_name}
_description: "{description}"
_category: {category}
_preprocessing:
  paradigm: {cfg.paradigm.value}
  transform: {cfg.transform.value}
  target_fx: {cfg.target_fx if cfg.normalize_fx else 'original'}
  output_size: {cfg.output_size or 'original'}

training:
  dataset:
    dataset_path: {cfg.output_dir}/data_mouse_train.txt

validation:
  dataset_path: {cfg.output_dir}/data_mouse_val.txt

_stats:
  num_train: {num_train}
  num_val: {num_val}
"""
        
        DATASET_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n[Config Generated] {config_path}")
        return config_path

    def run(self):
        """Run preprocessing pipeline."""
        cfg = self.config
        
        print(f"\n{'='*60}")
        print(f"Unified Preprocessor: {cfg.version}")
        print(f"{'='*60}")
        print(f"Paradigm: {cfg.paradigm.value}")
        print(f"Output: {cfg.output_dir}")
        
        # ====== D6/D9: Use DataLoader ======
        if cfg.paradigm in [Paradigm.GEOMETRY_PRESERVING, Paradigm.NATIVE]:
            self._init_data_loader()
            if cfg.paradigm == Paradigm.GEOMETRY_PRESERVING and cfg.d6_method == "pp_correct_crop":
                self._init_center_estimator()
            
            total_frames = len(self.data_loader)
            frame_indices = [i for i in range(0, total_frames, cfg.frame_interval) if i not in JUMP_FRAMES]
            if cfg.max_samples:
                frame_indices = frame_indices[:cfg.max_samples]
            
            print(f"Processing {len(frame_indices)} frames...")
            
            # Select method
            if cfg.paradigm == Paradigm.NATIVE:
                process_fn = self.process_d9
            elif cfg.d6_method == "resize_only":
                process_fn = self.process_d6_1
            elif cfg.d6_method == "virtual_shift":
                process_fn = self.process_d6_2
            else:
                process_fn = self.process_d6_3
            
            # Process and save in single loop (streaming - memory efficient)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0
            all_paths = []

            for frame_idx in tqdm(frame_indices, desc="Processing"):
                sample = process_fn(frame_idx)
                if sample:
                    sample_dir = cfg.output_dir / f"{saved_count:06d}"
                    self.save_sample(sample, sample_dir)
                    all_paths.append(str(sample_dir) + '/')
                    saved_count += 1

        # ====== D7/D8: Use video captures ======
        else:
            print(f"Loading cameras: {cfg.camera_pkl}")
            self.load_cameras(cfg.camera_pkl)

            video_dir = cfg.input_dir / "videos_undist"
            mask_dir = cfg.input_dir / "simpleclick_undist"
            video_caps = [cv2.VideoCapture(str(video_dir / f"{i}.mp4")) for i in range(self.num_views)]
            mask_caps = [cv2.VideoCapture(str(mask_dir / f"{i}.mp4")) for i in range(self.num_views)]

            total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = [i for i in range(0, total_frames, cfg.frame_interval) if i not in JUMP_FRAMES]
            if cfg.max_samples:
                frame_indices = frame_indices[:cfg.max_samples]

            print(f"Transform: {cfg.transform.value}, Scale: {cfg.scale_mode.value}, Zoom: {cfg.zoom}x")

            transforms = [self.compute_transform(cam['K']) for cam in self.cameras]

            # Process and save in single loop (streaming - memory efficient)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            saved_count = 0
            all_paths = []

            for frame_idx in tqdm(frame_indices, desc="Processing"):
                sample = self.process_frame_d7d8(frame_idx, video_caps, mask_caps, transforms)
                if sample:
                    sample_dir = cfg.output_dir / f"{saved_count:06d}"
                    self.save_sample(sample, sample_dir)
                    all_paths.append(str(sample_dir) + '/')
                    saved_count += 1

            for cap in video_caps + mask_caps:
                cap.release()

        # Generate split files (separate step - allows reusing same preprocessing with different splits)
        if cfg.val_ratio > 0:
            np.random.seed(42)
            indices = list(range(len(all_paths)))
            np.random.shuffle(indices)
            num_val = int(len(all_paths) * cfg.val_ratio)
            val_indices = set(indices[:num_val])

            train_paths = [all_paths[i] for i in range(len(all_paths)) if i not in val_indices]
            val_paths = [all_paths[i] for i in range(len(all_paths)) if i in val_indices]
        else:
            train_paths = all_paths
            val_paths = []

        for split, paths in [('train', train_paths), ('val', val_paths)]:
            if paths:
                with open(cfg.output_dir / f"data_mouse_{split}.txt", 'w') as f:
                    f.write('\n'.join(sorted(paths)))

        # Also save all paths for later split regeneration
        with open(cfg.output_dir / "data_mouse_all.txt", 'w') as f:
            f.write('\n'.join(sorted(all_paths)))

        with open(cfg.output_dir / "metadata.json", 'w') as f:
            json.dump({
                "version": cfg.version,
                "paradigm": cfg.paradigm.value,
                "total_samples": len(all_paths),
                "num_train": len(train_paths),
                "num_val": len(val_paths),
                "val_ratio": cfg.val_ratio,
            }, f, indent=2)

        self.generate_dataset_config(len(train_paths), len(val_paths))

        print(f"\n{'='*60}")
        print(f"Complete: {cfg.output_dir}")
        print(f"Total: {len(all_paths)}, Train: {len(train_paths)}, Val: {len(val_paths)}")
        print(f"Split files: data_mouse_train.txt, data_mouse_val.txt, data_mouse_all.txt")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Unified Mouse Preprocessor (D6-D9)")
    parser.add_argument('--preset', '-p', choices=list(PRESETS.keys()), help="Preset name")
    parser.add_argument('--list-presets', action='store_true', help="List available presets")
    parser.add_argument('--input-dir', type=str, help="Input directory")
    parser.add_argument('--output-dir', type=str, help="Output directory")
    parser.add_argument('--camera-pkl', default=None, help="Camera pickle path")
    parser.add_argument('--frame-interval', type=int, default=5, help="Frame interval")
    parser.add_argument('--max-samples', type=int, default=None, help="Max samples")
    parser.add_argument('--val-ratio', type=float, default=0.1, help="Validation ratio")
    parser.add_argument('--transform', choices=['affine', 'homography', 'none'])
    parser.add_argument('--scale-mode', choices=['fx_only', 'individual', 'average'])
    parser.add_argument('--zoom', type=float)
    args = parser.parse_args()

    if args.list_presets:
        print("\n=== Available Presets ===")
        recommended = get_recommended()
        for name in list_preset_names():
            info = PRESETS[name]
            rec = " [RECOMMENDED]" if name == recommended else ""
            print(f"  {name:8}{rec}")
            print(f"           {info.get('description', '')}")
        return

    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required")

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
