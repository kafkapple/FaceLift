#!/usr/bin/env python3
"""
D6 Preprocessing Methods - Geometry-Preserving Approaches

D6-1: No-Crop + Accurate Intrinsics
    - Resize only (no crop), preserve all geometric relationships
    
D6-2: Virtual Camera Relocation  
    - Keep images, shift PP to virtually center mouse
    
D6-3: PP-Correct Crop
    - Per-view crop with ACCURATE cx,cy (not forced to 256)

Usage:
    python -m mouse_extensions.preprocessing.preprocessor_d6 \
        --method D6-1 --input_dir /path/to/raw --output_dir /path/to/D6-1
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from .data_loader import DataLoader
from .center_estimation import CenterEstimator


@dataclass
class D6Config:
    input_dir: Path
    output_dir: Path
    method: str  # D6-1, D6-2, D6-3
    source_type: str = "raw"
    start_frame: int = 0
    end_frame: Optional[int] = None
    frame_step: int = 5
    target_size: int = 512
    target_fx: float = 549.0
    target_distance: float = 2.7
    object_ratio: float = 0.7
    val_ratio: float = 0.1


class D6Preprocessor:
    """
    D6 Preprocessor - Geometry-Preserving Methods
    
    Key difference from D4: cx,cy are computed ACCURATELY (not forced to 256)
    """

    def __init__(self, config: D6Config):
        self.config = config
        self.data_loader = DataLoader(config.input_dir, source_type=config.source_type)
        self.cameras = self.data_loader.get_camera_matrices()
        self.num_views = len(self.cameras)
        self.center_estimator = CenterEstimator(self.cameras, method="triangulation")

    # =========================================================================
    # D6-1: No-Crop + Accurate Intrinsics
    # =========================================================================
    def process_d6_1(self, frame_idx: int) -> Optional[Dict]:
        """D6-1: Resize only, no cropping. Perfect geometry preservation."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        orig_h, orig_w = images[0].shape[:2]
        
        # Fit to target_size while maintaining aspect ratio
        scale = self.config.target_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        # Center padding
        pad_top = (self.config.target_size - new_h) // 2
        pad_left = (self.config.target_size - new_w) // 2

        proc_images, proc_masks, cam_params = [], [], {}

        for i in range(self.num_views):
            # Resize
            resized = cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((self.config.target_size, self.config.target_size, 3), 255, dtype=np.uint8)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            proc_images.append(padded)

            # Mask
            mask_resized = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_padded = np.zeros((self.config.target_size, self.config.target_size), dtype=np.uint8)
            mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_resized
            proc_masks.append(mask_padded)

            # Intrinsics: scale + pad shift (ACCURATE!)
            cam = self.cameras[i]
            cam_params[f"view{i}"] = {
                "fx": cam["fx"] * scale,
                "fy": cam["fy"] * scale,
                "cx": cam["cx"] * scale + pad_left,  # NOT 256!
                "cy": cam["cy"] * scale + pad_top,   # NOT 256!
                "w": self.config.target_size,
                "h": self.config.target_size,
                "w2c": cam["w2c"].tolist() if isinstance(cam["w2c"], np.ndarray) else cam["w2c"],
                "_method": "D6-1",
                "_scale": scale,
                "_padding": [pad_left, pad_top]
            }

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, "frame_idx": frame_idx}

    # =========================================================================
    # D6-2: Virtual Camera Relocation
    # =========================================================================
    def process_d6_2(self, frame_idx: int) -> Optional[Dict]:
        """D6-2: Keep images, shift PP to virtually center mouse."""
        images, masks = self.data_loader.load_frame(frame_idx)
        if all(m.sum() < 100 for m in masks):
            return None

        orig_h, orig_w = images[0].shape[:2]
        scale = self.config.target_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        pad_top = (self.config.target_size - new_h) // 2
        pad_left = (self.config.target_size - new_w) // 2

        # Compute mouse centroids
        mouse_centroids = []
        for mask in masks:
            coords = np.where(mask > 127)
            if len(coords[0]) > 0:
                cy = coords[0].mean() * scale + pad_top
                cx = coords[1].mean() * scale + pad_left
            else:
                cy, cx = self.config.target_size / 2, self.config.target_size / 2
            mouse_centroids.append((cx, cy))

        proc_images, proc_masks, cam_params = [], [], {}
        target_center = self.config.target_size / 2

        for i in range(self.num_views):
            # Same image processing as D6-1
            resized = cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((self.config.target_size, self.config.target_size, 3), 255, dtype=np.uint8)
            padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            proc_images.append(padded)

            mask_resized = cv2.resize(masks[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_padded = np.zeros((self.config.target_size, self.config.target_size), dtype=np.uint8)
            mask_padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = mask_resized
            proc_masks.append(mask_padded)

            # Virtual PP shift to center mouse
            cam = self.cameras[i]
            actual_cx = cam["cx"] * scale + pad_left
            actual_cy = cam["cy"] * scale + pad_top
            
            mouse_cx, mouse_cy = mouse_centroids[i]
            pp_offset_x = mouse_cx - target_center
            pp_offset_y = mouse_cy - target_center
            
            cam_params[f"view{i}"] = {
                "fx": cam["fx"] * scale,
                "fy": cam["fy"] * scale,
                "cx": actual_cx + pp_offset_x,  # Virtual shift
                "cy": actual_cy + pp_offset_y,
                "w": self.config.target_size,
                "h": self.config.target_size,
                "w2c": cam["w2c"].tolist() if isinstance(cam["w2c"], np.ndarray) else cam["w2c"],
                "_method": "D6-2",
                "_actual_cx": actual_cx,
                "_actual_cy": actual_cy,
                "_pp_offset": [pp_offset_x, pp_offset_y]
            }

        return {"images": proc_images, "masks": proc_masks, "cameras": cam_params, "frame_idx": frame_idx}

    # =========================================================================
    # D6-3: PP-Correct Crop
    # =========================================================================
    def process_d6_3(self, frame_idx: int) -> Optional[Dict]:
        """D6-3: Per-view crop with ACCURATE PP (not forced to 256)."""
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
        scale = self.config.target_size / crop_size

        proc_images, proc_masks, cam_params = [], [], {}

        for i in range(self.num_views):
            crop_center = centers_2d[i]
            
            # Crop
            proc_img = self._crop_and_resize(images[i], crop_center, crop_size)
            proc_images.append(proc_img)

            mask_rgb = np.stack([masks[i]] * 3, axis=-1)
            proc_mask = self._crop_and_resize(mask_rgb, crop_center, crop_size, bg=(0, 0, 0))
            proc_masks.append(proc_mask[:, :, 0])

            # ACCURATE PP calculation (NOT 256!)
            cam = self.cameras[i]
            crop_x = crop_center[0] - crop_size / 2
            crop_y = crop_center[1] - crop_size / 2

            cam_params[f"view{i}"] = {
                "fx": cam["fx"] * scale,
                "fy": cam["fy"] * scale,
                "cx": (cam["cx"] - crop_x) * scale,  # ACCURATE!
                "cy": (cam["cy"] - crop_y) * scale,  # NOT 256!
                "w": self.config.target_size,
                "h": self.config.target_size,
                "w2c": cam["w2c"].tolist() if isinstance(cam["w2c"], np.ndarray) else cam["w2c"],
                "_method": "D6-3",
                "_crop_center": list(crop_center),
                "_crop_size": crop_size
            }

        return {
            "images": proc_images, "masks": proc_masks, "cameras": cam_params,
            "frame_idx": frame_idx, "center_3d": result.center_3d.tolist() if result.center_3d is not None else None
        }

    def _crop_and_resize(self, image: np.ndarray, crop_center: Tuple[float, float],
                         crop_size: float, bg: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
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

        if src_x2 > src_x1 and src_y2 > src_y1:
            output[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

        return cv2.resize(output, (self.config.target_size, self.config.target_size))

    # =========================================================================
    # Common Methods
    # =========================================================================
    def process_frame(self, frame_idx: int) -> Optional[Dict]:
        if self.config.method == "D6-1":
            return self.process_d6_1(frame_idx)
        elif self.config.method == "D6-2":
            return self.process_d6_2(frame_idx)
        elif self.config.method == "D6-3":
            return self.process_d6_3(frame_idx)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def normalize_cameras(self, frames: List[Dict]) -> List[Dict]:
        """Normalize fx and distance to target values."""
        fx_values = [f["fx"] for f in frames]
        distances = []
        for f in frames:
            w2c = np.array(f["w2c"])
            c2w = np.linalg.inv(w2c)
            distances.append(np.linalg.norm(c2w[:3, 3]))

        avg_fx = np.mean(fx_values)
        avg_distance = np.mean(distances)

        fx_scale = self.config.target_fx / avg_fx
        distance_scale = self.config.target_distance / avg_distance

        normalized = []
        for f in frames:
            nf = f.copy()
            nf["fx"] = f["fx"] * fx_scale
            nf["fy"] = f["fy"] * fx_scale
            nf["cx"] = f["cx"] * fx_scale
            nf["cy"] = f["cy"] * fx_scale

            w2c = np.array(f["w2c"])
            c2w = np.linalg.inv(w2c)
            c2w[:3, 3] *= distance_scale
            nf["w2c"] = np.linalg.inv(c2w).tolist()
            normalized.append(nf)

        return normalized

    def save_sample(self, sample: Dict, sample_dir: Path):
        sample_dir.mkdir(parents=True, exist_ok=True)
        images_dir = sample_dir / "images"
        images_dir.mkdir(exist_ok=True)

        for i, (img, mask) in enumerate(zip(sample["images"], sample["masks"])):
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = mask
            Image.fromarray(rgba).save(images_dir / f"cam_{i:03d}.png")

        frames = []
        for i in range(len(sample["images"])):
            view_key = f"view{i}"
            if view_key in sample["cameras"]:
                frame = sample["cameras"][view_key].copy()
                frame["file_path"] = f"images/cam_{i:03d}.png"
                frame["view_id"] = i
                frames.append(frame)

        frames = self.normalize_cameras(frames)

        with open(sample_dir / "opencv_cameras.json", "w") as f:
            json.dump({"frames": frames}, f, indent=2)

    def run(self):
        cfg = self.config
        end = cfg.end_frame or len(self.data_loader)
        frame_indices = list(range(cfg.start_frame, end, cfg.frame_step))

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"D6 Preprocessor - {cfg.method}")
        print(sep)
        print(f"Target: fx={cfg.target_fx}, dist={cfg.target_distance}")
        print(f"Frames: {len(frame_indices)}\n")

        samples = []
        for f in tqdm(frame_indices, desc="Processing"):
            sample = self.process_frame(f)
            if sample is not None:
                samples.append(sample)

        print(f"Valid samples: {len(samples)}")

        n_val = int(len(samples) * cfg.val_ratio)
        val_step = len(samples) // max(1, n_val)
        val_idx = set(range(0, len(samples), val_step)[:n_val])

        train = [s for i, s in enumerate(samples) if i not in val_idx]
        val = [s for i, s in enumerate(samples) if i in val_idx]

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        for split, data in [("train", train), ("val", val)]:
            for i, s in enumerate(tqdm(data, desc=f"Saving {split}")):
                self.save_sample(s, cfg.output_dir / split / f"{i:06d}")

        # Save metadata
        meta = {
            "version": cfg.method,
            "target_fx": cfg.target_fx,
            "target_distance": cfg.target_distance,
            "num_train": len(train),
            "num_val": len(val),
            "note": "cx,cy are ACCURATE (not forced to 256)"
        }
        with open(cfg.output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save data list files
        for split in ["train", "val"]:
            split_dir = cfg.output_dir / split
            sample_dirs = sorted(split_dir.iterdir())
            with open(cfg.output_dir / f"data_mouse_{split}.txt", "w") as f:
                for sd in sample_dirs:
                    f.write(f"{sd.absolute()}/\n")

        print(f"\nDone! Output: {cfg.output_dir}")
        print(f"Train: {len(train)}, Val: {len(val)}")

        # Print PP statistics
        if samples:
            print("\nPP Statistics (first sample):")
            for i in range(self.num_views):
                cam = samples[0]["cameras"][f"view{i}"]
                print(f"  View {i}: cx={cam['cx']:.1f}, cy={cam['cy']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="D6 Geometry-Preserving Preprocessor")
    parser.add_argument("--method", choices=["D6-1", "D6-2", "D6-3"], required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    config = D6Config(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        method=args.method,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
        val_ratio=args.val_ratio
    )

    D6Preprocessor(config).run()


if __name__ == "__main__":
    main()
