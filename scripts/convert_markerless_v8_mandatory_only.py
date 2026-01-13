#!/usr/bin/env python3
"""
Convert markerless_mouse_1_nerf to FaceLift format (v8 - Mandatory Only)

=== v8: 필수 요소만 적용 ===

✅ 필수 (적용):
  1. fx = fy 강제 (정사각 픽셀)
  2. 거리 정규화 (dist → 2.7)
  3. 일관된 fx = 549, FOV = 50°

❌ 비필수 (미적용):
  1. 이미지 스케일의 거리 보정 - 미적용
  2. Centroid centering - 미적용 (Principal point만)
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


JUMP_FRAMES = {5900, 11800, 17700}
ORIG_SIZE = (1152, 1024)
TARGET_SIZE = 512
UNIT_SCALE = 100.0
TARGET_FX = 549.0
TARGET_DISTANCE = 2.7


def load_camera_params(pkl_path: str) -> List[Dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_camera_transform_mandatory(K, R, T):
    """v8: 필수 요소만 - 거리 정규화 O, 이미지 스케일 거리보정 X"""
    orig_fx = K[0, 0]
    orig_fy = K[1, 1]
    orig_cx = K[0, 2]
    orig_cy = K[1, 2]

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    
    orig_dist_mm = np.linalg.norm(cam_pos)
    orig_dist_norm = orig_dist_mm / UNIT_SCALE
    
    # 거리 정규화 (필수)
    dist_scale = TARGET_DISTANCE / orig_dist_norm
    new_cam_pos = cam_pos * dist_scale
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = new_cam_pos
    new_w2c = np.linalg.inv(new_c2w)

    # v8: 이미지 스케일 = fx 비율만 (거리 보정 없음)
    image_scale = TARGET_FX / orig_fx

    # Principal point centering
    scaled_cx = orig_cx * image_scale
    scaled_cy = orig_cy * image_scale
    offset_x = TARGET_SIZE / 2 - scaled_cx
    offset_y = TARGET_SIZE / 2 - scaled_cy

    camera_dict = {
        "w": TARGET_SIZE,
        "h": TARGET_SIZE,
        "fx": TARGET_FX,
        "fy": TARGET_FX,
        "cx": TARGET_SIZE / 2,
        "cy": TARGET_SIZE / 2,
        "w2c": new_w2c.tolist(),
        "_original": {
            "fx": float(orig_fx), "fy": float(orig_fy),
            "cx": float(orig_cx), "cy": float(orig_cy),
            "distance_mm": float(orig_dist_mm),
            "distance_norm": float(orig_dist_norm),
        },
        "_transform": {
            "method": "v8_mandatory_only",
            "image_scale": float(image_scale),
            "distance_scale": float(dist_scale),
            "center_offset": [float(offset_x), float(offset_y)],
            "distance_correction_in_scale": False,
        },
    }
    return camera_dict, image_scale, (offset_x, offset_y)


def process_image(image, mask, scale, offset, output_size=512, bg=(255,255,255)):
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < 1: new_w = 1
    if new_h < 1: new_h = 1

    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    output = np.full((output_size, output_size, 4), (*bg, 0), dtype=np.uint8)
    
    ox, oy = int(offset[0]), int(offset[1])
    sx1, sy1 = max(0, -ox), max(0, -oy)
    sx2, sy2 = min(new_w, output_size - ox), min(new_h, output_size - oy)
    dx1, dy1 = max(0, ox), max(0, oy)
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)

    if sx2 > sx1 and sy2 > sy1:
        output[dy1:dy2, dx1:dx2, :3] = scaled_img[sy1:sy2, sx1:sx2]
        output[dy1:dy2, dx1:dx2, 3] = scaled_mask[sy1:sy2, sx1:sx2]
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/joon/data/markerless_mouse_1_nerf")
    parser.add_argument("--output_dir", default="/home/joon/data/preprocessed/FaceLift_mouse/data_mouse_v8_mandatory")
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_params = load_camera_params(input_dir / "new_cam.pkl")
    
    # Video captures
    video_caps = [cv2.VideoCapture(str(input_dir / "videos_undist" / f"{i}.mp4")) for i in range(6)]
    mask_caps = [cv2.VideoCapture(str(input_dir / "simpleclick_undist" / f"{i}.mp4")) for i in range(6)]
    
    total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    sample_idx = 0
    train_samples, val_samples = [], []

    for frame_idx in tqdm(range(0, total_frames, args.frame_interval)):
        if frame_idx in JUMP_FRAMES:
            continue
        if args.max_samples and sample_idx >= args.max_samples:
            break

        sample_name = f"sample_{sample_idx:06d}"
        sample_dir = output_dir / sample_name / "images"
        sample_dir.mkdir(parents=True, exist_ok=True)

        frames_data = {"frames": []}
        valid = True

        for cam_idx in range(6):
            video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret1, frame = video_caps[cam_idx].read()
            ret2, mask_frame = mask_caps[cam_idx].read()
            
            if not ret1:
                valid = False
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY) if ret2 else np.ones(frame.shape[:2], dtype=np.uint8) * 255

            K = np.array(cam_params[cam_idx]["K"])
            R = np.array(cam_params[cam_idx]["R"])
            T = np.array(cam_params[cam_idx]["T"])

            cam_dict, scale, offset = compute_camera_transform_mandatory(K, R, T)
            processed = process_image(frame_rgb, mask, scale, offset)

            Image.fromarray(processed).save(sample_dir / f"cam_{cam_idx:03d}.png")

            cam_dict["file_path"] = f"images/cam_{cam_idx:03d}.png"
            cam_dict["view_id"] = cam_idx
            frames_data["frames"].append(cam_dict)

        if valid and len(frames_data["frames"]) == 6:
            with open(output_dir / sample_name / "opencv_cameras.json", "w") as f:
                json.dump(frames_data, f, indent=2)
            
            (val_samples if sample_idx % 10 == 0 else train_samples).append(sample_name)
            sample_idx += 1

    for cap in video_caps + mask_caps:
        cap.release()

    with open(output_dir / "data_mouse_train.txt", "w") as f:
        f.writelines(f"{output_dir}/{s}\n" for s in train_samples)
    with open(output_dir / "data_mouse_val.txt", "w") as f:
        f.writelines(f"{output_dir}/{s}\n" for s in val_samples)

    print(f"\n=== v8 Dataset Created ===")
    print(f"Samples: {sample_idx} (Train: {len(train_samples)}, Val: {len(val_samples)})")
    
    # Scale comparison
    print(f"\n=== v5 vs v8 Scale Comparison ===")
    for i in range(6):
        K = np.array(cam_params[i]["K"])
        R, T = np.array(cam_params[i]["R"]), np.array(cam_params[i]["T"])
        w2c = np.eye(4); w2c[:3,:3] = R; w2c[:3,3] = T.flatten()
        dist = np.linalg.norm(np.linalg.inv(w2c)[:3,3]) / UNIT_SCALE
        
        v5_scale = (TARGET_FX / K[0,0]) * (dist / TARGET_DISTANCE)
        v8_scale = TARGET_FX / K[0,0]
        print(f"Cam {i}: dist={dist:.2f}, v5={v5_scale:.3f}, v8={v8_scale:.3f}, diff={v5_scale-v8_scale:+.3f}")


if __name__ == "__main__":
    main()
