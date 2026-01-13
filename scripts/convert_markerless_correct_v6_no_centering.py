#!/usr/bin/env python3
"""
Convert markerless_mouse_1_nerf to FaceLift format (v6)

=== v6: v5에서 Centroid Centering 제거 ===
- 거리 보정: 유지 (필수)
- Centroid centering: 제거 (Principal point centering만)
- 목적: Centering이 Ghost 원인인지 검증
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


JUMP_FRAMES = {5900, 11800, 17700}
ORIG_SIZE = (1152, 1024)
TARGET_SIZE = 512
UNIT_SCALE = 100.0


def load_camera_params(pkl_path: str) -> List[Dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_camera_transform(
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    target_distance: float = 2.7,
    target_fx: float = 549.0,
    target_size: int = 512,
    orig_size: Tuple[int, int] = (1152, 1024),
) -> Tuple[Dict, float, Tuple[float, float]]:
    """
    v6: 거리 보정 + Principal Point centering (Centroid centering 제외)
    """
    orig_w, orig_h = orig_size
    orig_fx = K[0, 0]
    orig_fy = K[1, 1]
    orig_cx = K[0, 2]
    orig_cy = K[1, 2]

    # 카메라 위치 계산
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    current_distance = np.linalg.norm(cam_pos)
    current_distance_norm = current_distance / UNIT_SCALE

    # 거리 정규화
    distance_scale = target_distance / current_distance_norm
    new_cam_pos = cam_pos * distance_scale
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = new_cam_pos
    new_w2c = np.linalg.inv(new_c2w)

    # 이미지 스케일 (거리 보정 포함)
    fx_ratio = target_fx / orig_fx
    dist_ratio = current_distance_norm / target_distance
    total_image_scale = fx_ratio * dist_ratio

    # Principal Point centering (Centroid 대신)
    scaled_cx = orig_cx * total_image_scale
    scaled_cy = orig_cy * total_image_scale
    center_offset_x = target_size / 2 - scaled_cx
    center_offset_y = target_size / 2 - scaled_cy

    camera_dict = {
        "w": target_size,
        "h": target_size,
        "fx": target_fx,
        "fy": target_fx,  # fy = fx 강제
        "cx": target_size / 2,
        "cy": target_size / 2,
        "w2c": new_w2c.tolist(),
        "_original": {
            "fx": float(orig_fx),
            "fy": float(orig_fy),
            "cx": float(orig_cx),
            "cy": float(orig_cy),
            "distance_mm": float(current_distance),
            "distance_norm": float(current_distance_norm),
        },
        "_transform": {
            "image_scale": float(total_image_scale),
            "fx_ratio": float(fx_ratio),
            "dist_ratio": float(dist_ratio),
            "center_offset": [float(center_offset_x), float(center_offset_y)],
            "centering_method": "principal_point",  # v6 차이점
        },
    }

    return camera_dict, total_image_scale, (center_offset_x, center_offset_y)


def process_image(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
    center_offset: Tuple[float, float],
    output_size: int = 512,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """이미지 전처리: 스케일링 + 센터링 + 마스크 적용"""
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w < 1 or new_h < 1:
        new_w = max(1, new_w)
        new_h = max(1, new_h)

    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    output = np.full((output_size, output_size, 4), (*background, 0), dtype=np.uint8)
    
    offset_x = int(center_offset[0])
    offset_y = int(center_offset[1])

    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(new_w, output_size - offset_x)
    src_y2 = min(new_h, output_size - offset_y)

    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2, :3] = scaled_img[src_y1:src_y2, src_x1:src_x2]
        output[dst_y1:dst_y2, dst_x1:dst_x2, 3] = scaled_mask[src_y1:src_y2, src_x1:src_x2]

    return output


def main():
    parser = argparse.ArgumentParser(description="Convert markerless mouse to FaceLift (v6 - no centroid centering)")
    parser.add_argument("--input_dir", type=str, default="/home/joon/data/markerless_mouse_1_nerf")
    parser.add_argument("--output_dir", type=str, default="/home/joon/data/preprocessed/FaceLift_mouse/data_mouse_correct_v6_no_centering")
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--target_distance", type=float, default=2.7)
    parser.add_argument("--target_fx", type=float, default=549.0)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    cam_params = load_camera_params(input_dir / "meta" / "cameras.pkl")
    
    # Video paths
    video_dir = input_dir / "videos_undist"
    mask_dir = input_dir / "segmentations"
    
    video_paths = sorted(video_dir.glob("cam_*.mp4"))
    mask_paths = sorted(mask_dir.glob("cam_*"))
    
    if len(video_paths) != 6 or len(mask_paths) != 6:
        raise ValueError(f"Expected 6 cameras, found {len(video_paths)} videos and {len(mask_paths)} mask dirs")

    # Open videos
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    sample_idx = 0
    train_samples = []
    val_samples = []

    for frame_idx in tqdm(range(0, total_frames, args.frame_interval), desc="Processing frames"):
        if frame_idx in JUMP_FRAMES:
            continue
        if args.max_samples and sample_idx >= args.max_samples:
            break

        sample_name = f"sample_{sample_idx:06d}"
        sample_dir = output_dir / sample_name / "images"
        sample_dir.mkdir(parents=True, exist_ok=True)

        frames_data = {"frames": []}

        for cam_idx in range(6):
            caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = caps[cam_idx].read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mask_path = mask_paths[cam_idx] / f"{frame_idx:06d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255

            K = cam_params[cam_idx]["K"]
            R = cam_params[cam_idx]["R"]
            T = cam_params[cam_idx]["T"]

            camera_dict, img_scale, center_offset = compute_camera_transform(
                K, R, T,
                target_distance=args.target_distance,
                target_fx=args.target_fx,
                target_size=args.target_size,
                orig_size=ORIG_SIZE,
            )

            processed = process_image(
                frame_rgb, mask, img_scale, center_offset,
                output_size=args.target_size,
            )

            output_path = sample_dir / f"cam_{cam_idx:03d}.png"
            Image.fromarray(processed).save(output_path)

            camera_dict["file_path"] = f"images/cam_{cam_idx:03d}.png"
            camera_dict["view_id"] = cam_idx
            frames_data["frames"].append(camera_dict)

        if len(frames_data["frames"]) == 6:
            with open(output_dir / sample_name / "opencv_cameras.json", "w") as f:
                json.dump(frames_data, f, indent=2)

            if sample_idx % 10 == 0:
                val_samples.append(sample_name)
            else:
                train_samples.append(sample_name)

            sample_idx += 1

    for cap in caps:
        cap.release()

    # Save train/val splits
    with open(output_dir / "data_mouse_train.txt", "w") as f:
        for s in train_samples:
            f.write(f"{output_dir}/{s}\n")
    with open(output_dir / "data_mouse_val.txt", "w") as f:
        for s in val_samples:
            f.write(f"{output_dir}/{s}\n")

    print(f"\nDone! Created {sample_idx} samples")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")


if __name__ == "__main__":
    main()
