#!/usr/bin/env python3
"""
Unified Mouse Data Preprocessor

Consolidates preprocessing into single configurable module.
Presets: v13, D1, D2, D3 (recommended)

Usage:
    python -m mouse_extensions.preprocessing.unified_preprocessor \\
        --preset D3 --input_dir /path/to/raw --output_dir /path/to/D3
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from .data_loader import DataLoader
from .center_estimation import CenterEstimator
from .camera_normalizer import normalize_cameras


class CenterMethodType(Enum):
    TRIANGULATION = "triangulation"  # 3D (recommended)
    PER_VIEW_2D = "per_view_2d"      # 2D per view (legacy)
    BBOX = "bbox"                     # Bbox center
    PP_CENTERED = "pp_centered"       # PP centered


class PPMethodType(Enum):
    CORRECT = "correct"       # Compute correct PP
    FORCE_256 = "force_256"   # Force cx=cy=256


@dataclass
class PreprocessConfig:
    input_dir: Path
    output_dir: Path
    source_type: str = "raw"
    start_frame: int = 0
    end_frame: Optional[int] = None
    frame_step: int = 5
    center_method: CenterMethodType = CenterMethodType.TRIANGULATION
    pp_method: PPMethodType = PPMethodType.CORRECT
    target_size: int = 512
    target_fx: float = 549.0
    target_distance: float = 2.7
    normalize_cameras: bool = True
    object_ratio: float = 0.7
    val_ratio: float = 0.1
    version: str = "D3"

    @classmethod
    def from_preset(cls, preset: str, input_dir: Path, output_dir: Path):
        presets = {
            "v13": {"center_method": CenterMethodType.PER_VIEW_2D, "pp_method": PPMethodType.FORCE_256, "version": "v13"},
            "D1": {"center_method": CenterMethodType.PP_CENTERED, "pp_method": PPMethodType.FORCE_256, "version": "D1"},
            "D2": {"center_method": CenterMethodType.BBOX, "pp_method": PPMethodType.CORRECT, "version": "D2"},
            "D3": {"center_method": CenterMethodType.TRIANGULATION, "pp_method": PPMethodType.CORRECT, "version": "D3"},
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        return cls(input_dir=input_dir, output_dir=output_dir, **presets[preset])


class UnifiedPreprocessor:
    def __init__(self, config: PreprocessConfig):
        self.config = config
        self.data_loader = DataLoader(config.input_dir, source_type=config.source_type)
        self.cameras = self.data_loader.get_camera_matrices()
        
        if config.center_method == CenterMethodType.TRIANGULATION:
            self.center_estimator = CenterEstimator(self.cameras, method="triangulation")
        else:
            self.center_estimator = None

    def compute_crop_center(self, masks, images):
        masks_array = np.array(masks)
        
        if self.config.center_method == CenterMethodType.TRIANGULATION:
            result = self.center_estimator.estimate(masks_array)
            return result.center_3d, [tuple(c) for c in result.centers_2d]
        
        elif self.config.center_method == CenterMethodType.PER_VIEW_2D:
            centers_2d = []
            for mask in masks:
                coords = np.where(mask > 127)
                if len(coords[0]) > 0:
                    cy, cx = coords[0].mean(), coords[1].mean()
                else:
                    cy, cx = mask.shape[0]/2, mask.shape[1]/2
                centers_2d.append((cx, cy))
            return None, centers_2d
        
        elif self.config.center_method == CenterMethodType.BBOX:
            centers_2d = []
            for mask in masks:
                coords = np.where(mask > 127)
                if len(coords[0]) > 0:
                    cx = (coords[1].min() + coords[1].max()) / 2
                    cy = (coords[0].min() + coords[0].max()) / 2
                else:
                    cx, cy = mask.shape[1]/2, mask.shape[0]/2
                centers_2d.append((cx, cy))
            return None, centers_2d
        
        elif self.config.center_method == CenterMethodType.PP_CENTERED:
            return None, [(cam["cx"], cam["cy"]) for cam in self.cameras]

    def compute_crop_params(self, masks, centers_2d):
        max_size = 0
        for mask in masks:
            coords = np.where(mask > 127)
            if len(coords[0]) > 0:
                h = coords[0].max() - coords[0].min()
                w = coords[1].max() - coords[1].min()
                max_size = max(max_size, h, w)
        crop_size = max(max_size / self.config.object_ratio, 256)
        scale = self.config.target_size / crop_size
        return {"crop_size": crop_size, "scale": scale, "centers_2d": centers_2d}

    def transform_intrinsics(self, cam, crop_center, crop_size, scale):
        crop_x = crop_center[0] - crop_size / 2
        crop_y = crop_center[1] - crop_size / 2
        new_fx = cam["fx"] * scale
        new_fy = cam["fy"] * scale
        
        if self.config.pp_method == PPMethodType.CORRECT:
            new_cx = (cam["cx"] - crop_x) * scale
            new_cy = (cam["cy"] - crop_y) * scale
        else:
            new_cx = new_cy = self.config.target_size / 2
        
        w2c = cam["w2c"].tolist() if isinstance(cam["w2c"], np.ndarray) else cam["w2c"]
        return {"fx": new_fx, "fy": new_fy, "cx": new_cx, "cy": new_cy, 
                "w": self.config.target_size, "h": self.config.target_size, "w2c": w2c}

    def crop_and_resize(self, image, crop_center, crop_size, bg=(255,255,255)):
        h, w = image.shape[:2]
        crop_x = int(crop_center[0] - crop_size/2)
        crop_y = int(crop_center[1] - crop_size/2)
        cs = int(crop_size)
        
        output = np.full((cs, cs, 3), bg, dtype=np.uint8)
        src_x1, src_y1 = max(0, crop_x), max(0, crop_y)
        src_x2, src_y2 = min(w, crop_x+cs), min(h, crop_y+cs)
        dst_x1, dst_y1 = max(0, -crop_x), max(0, -crop_y)
        dst_x2, dst_y2 = dst_x1 + (src_x2-src_x1), dst_y1 + (src_y2-src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            output[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
        
        return cv2.resize(output, (self.config.target_size, self.config.target_size))

    def process_frame(self, frame_idx):
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None
        
        center_3d, centers_2d = self.compute_crop_center(masks, images)
        crop_params = self.compute_crop_params(masks, centers_2d)
        
        proc_images, proc_masks, cam_params = [], [], {}
        for i in range(len(images)):
            cc, cs, sc = centers_2d[i], crop_params["crop_size"], crop_params["scale"]
            proc_images.append(self.crop_and_resize(images[i], cc, cs))
            mask_rgb = np.stack([masks[i]]*3, axis=-1)
            proc_masks.append(self.crop_and_resize(mask_rgb, cc, cs, bg=(0,0,0))[:,:,0])
            cam_params[f"view{i}"] = self.transform_intrinsics(self.cameras[i], cc, cs, sc)
        
        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params,
                "frame_idx": frame_idx, "center_3d": center_3d.tolist() if center_3d is not None else None}

    def save_sample(self, sample, sample_dir):
        """Save sample in FaceLift-compatible format."""
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images subdirectory
        images_dir = sample_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Save images as cam_XXX.png in images/ subdirectory
        for i, (img, mask) in enumerate(zip(sample["images"], sample["masks"])):
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:,:,:3], rgba[:,:,3] = img, mask
            Image.fromarray(rgba).save(images_dir / f"cam_{i:03d}.png")
        
        # Convert cameras to frames list format for FaceLift compatibility
        frames = []
        for i in range(len(sample["images"])):
            view_key = f"view{i}"
            if view_key in sample["cameras"]:
                frame = sample["cameras"][view_key].copy()
                # Add required fields for data loader
                frame["file_path"] = f"images/cam_{i:03d}.png"
                frame["view_id"] = i
                frames.append(frame)
        
        # Apply camera normalization (fx -> target_fx, distance -> target_distance)
        if self.config.normalize_cameras:
            frames, transform_info = normalize_cameras(
                frames,
                target_fx=self.config.target_fx,
                target_distance=self.config.target_distance
            )
            frames_data = {"frames": frames, "_transform": transform_info}
        else:
            frames_data = {"frames": frames}

        # Save as opencv_cameras.json with frames list
        with open(sample_dir / "opencv_cameras.json", "w") as f:
            json.dump(frames_data, f, indent=2)

    def run(self):
        cfg = self.config
        end = cfg.end_frame or len(self.data_loader)
        frames = list(range(cfg.start_frame, end, cfg.frame_step))
        
        sep = "=" * 60
        print(f"\n{sep}\nUnified Preprocessor - {cfg.version}\n{sep}")
        print(f"Center: {cfg.center_method.value}, PP: {cfg.pp_method.value}")
        print(f"Frames: {len(frames)}\n")
        
        samples = [s for s in (self.process_frame(f) for f in tqdm(frames)) if s]
        print(f"Valid samples: {len(samples)}")
        
        n_val = int(len(samples) * cfg.val_ratio)
        val_step = len(samples) // max(1, n_val)
        val_idx = set(range(0, len(samples), val_step)[:n_val])
        train = [s for i,s in enumerate(samples) if i not in val_idx]
        val = [s for i,s in enumerate(samples) if i in val_idx]
        
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        for split, data in [("train", train), ("val", val)]:
            for i, s in enumerate(tqdm(data, desc=f"Saving {split}")):
                self.save_sample(s, cfg.output_dir / split / f"{i:06d}")
        
        meta = {"version": cfg.version, "center_method": cfg.center_method.value,
                "pp_method": cfg.pp_method.value, "num_train": len(train), "num_val": len(val)}
        with open(cfg.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n✅ Done! {cfg.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Unified Mouse Preprocessor")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preset", choices=["v13", "D1", "D2", "D3"])
    parser.add_argument("--center_method", choices=["triangulation", "per_view_2d", "bbox", "pp_centered"], default="triangulation")
    parser.add_argument("--pp_method", choices=["correct", "force_256"], default="correct")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if args.preset:
        cfg = PreprocessConfig.from_preset(args.preset, input_dir, output_dir)
        cfg.start_frame, cfg.end_frame, cfg.frame_step, cfg.val_ratio = args.start_frame, args.end_frame, args.frame_step, args.val_ratio
    else:
        cfg = PreprocessConfig(input_dir=input_dir, output_dir=output_dir, 
                               center_method=CenterMethodType(args.center_method),
                               pp_method=PPMethodType(args.pp_method),
                               start_frame=args.start_frame, end_frame=args.end_frame,
                               frame_step=args.frame_step, val_ratio=args.val_ratio)
    
    UnifiedPreprocessor(cfg).run()


if __name__ == "__main__":
    main()
