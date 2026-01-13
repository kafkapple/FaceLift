#!/usr/bin/env python3
"""
Convert markerless_mouse_1_nerf to FaceLift format (v9 - No Centering/Scaling)

=== v9: Centering/Scaling 완전 제외 ===

✅ 적용:
  1. fx = fy 강제 (정사각 픽셀)
  2. 거리 정규화 (dist → 2.7)
  3. 단순 리사이즈 (512/1152 = 0.444)

❌ 미적용:
  1. 이미지 스케일 보정 (fx 비율, 거리 비율 모두 미적용)
  2. Centering 없음 (cx, cy 원본 비율 유지)

=== 결과 ===
- fx = fy = orig_fx * 0.444 (뷰마다 다름: ~725)
- cx, cy = orig_cx * 0.444, orig_cy * 0.444 (뷰마다 다름)
- 거리 = 2.7 (정규화)
- 이미지 = 512x512 (단순 crop/pad)
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
ORIG_SIZE = (1152, 1024)  # width, height
TARGET_SIZE = 512
UNIT_SCALE = 100.0
TARGET_DISTANCE = 2.7


def load_camera_params(pkl_path: str) -> List[Dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_camera_transform_no_centering(K, R, T):
    """
    v9: Centering/Scaling 완전 제외
    - 거리 정규화만 적용
    - 이미지는 단순 리사이즈 (비율 고정)
    - cx, cy는 원본 비율 유지
    """
    orig_fx = K[0, 0]
    orig_fy = K[1, 1]
    orig_cx = K[0, 2]
    orig_cy = K[1, 2]

    # === 1. 카메라 거리 정규화 ===
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    
    orig_dist_mm = np.linalg.norm(cam_pos)
    orig_dist_norm = orig_dist_mm / UNIT_SCALE
    
    # 거리 정규화
    dist_scale = TARGET_DISTANCE / orig_dist_norm
    new_cam_pos = cam_pos * dist_scale
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = new_cam_pos
    new_w2c = np.linalg.inv(new_c2w)

    # === 2. 단순 리사이즈 스케일 ===
    resize_scale = TARGET_SIZE / ORIG_SIZE[0]  # 512/1152 = 0.444

    # === 3. Intrinsics 스케일링 (단순 비율) ===
    # fx, fy를 동일하게 (정사각 픽셀)
    new_fx = orig_fx * resize_scale
    new_fy = orig_fx * resize_scale  # fy = fx 강제
    
    # cx, cy는 원본 비율 유지 (centering 없음!)
    new_cx = orig_cx * resize_scale
    new_cy = orig_cy * resize_scale

    camera_dict = {
        "w": TARGET_SIZE,
        "h": TARGET_SIZE,
        "fx": float(new_fx),
        "fy": float(new_fy),
        "cx": float(new_cx),
        "cy": float(new_cy),
        "w2c": new_w2c.tolist(),
        "_original": {
            "fx": float(orig_fx),
            "fy": float(orig_fy),
            "cx": float(orig_cx),
            "cy": float(orig_cy),
            "distance_mm": float(orig_dist_mm),
            "distance_norm": float(orig_dist_norm),
        },
        "_transform": {
            "method": "v9_no_centering",
            "resize_scale": float(resize_scale),
            "distance_scale": float(dist_scale),
            "centering": False,
            "fx_correction": False,
            "distance_correction_in_scale": False,
        },
    }
    return camera_dict, resize_scale


def process_image_no_centering(image, mask, scale, output_size=512, bg=(255,255,255)):
    """
    단순 리사이즈 + crop/pad (centering 없음)
    이미지 좌상단 기준으로 배치
    """
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    
    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interp)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 512x512 캔버스에 좌상단 정렬 (centering 없음)
    output = np.full((output_size, output_size, 4), (*bg, 0), dtype=np.uint8)
    
    # crop if larger, pad if smaller
    paste_w = min(new_w, output_size)
    paste_h = min(new_h, output_size)
    
    output[:paste_h, :paste_w, :3] = scaled_img[:paste_h, :paste_w]
    output[:paste_h, :paste_w, 3] = scaled_mask[:paste_h, :paste_w]
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/joon/data/markerless_mouse_1_nerf")
    parser.add_argument("--output_dir", default="/home/joon/data/preprocessed/FaceLift_mouse/data_mouse_v9_no_centering")
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_params = load_camera_params(input_dir / "new_cam.pkl")
    
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

            cam_dict, scale = compute_camera_transform_no_centering(K, R, T)
            processed = process_image_no_centering(frame_rgb, mask, scale)

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

    print(f"\n=== v9 Dataset Created (No Centering/Scaling) ===")
    print(f"Samples: {sample_idx} (Train: {len(train_samples)}, Val: {len(val_samples)})")
    
    # 파라미터 출력
    print(f"\n=== Camera Parameters ===")
    resize_scale = TARGET_SIZE / ORIG_SIZE[0]
    for i in range(6):
        K = np.array(cam_params[i]["K"])
        R, T = np.array(cam_params[i]["R"]), np.array(cam_params[i]["T"])
        w2c = np.eye(4); w2c[:3,:3] = R; w2c[:3,3] = T.flatten()
        dist = np.linalg.norm(np.linalg.inv(w2c)[:3,3]) / UNIT_SCALE
        
        new_fx = K[0,0] * resize_scale
        new_cx = K[0,2] * resize_scale
        new_cy = K[1,2] * resize_scale
        
        print(f"Cam {i}: fx={new_fx:.1f}, cx={new_cx:.1f}, cy={new_cy:.1f}, dist_orig={dist:.2f}→2.70")


if __name__ == "__main__":
    main()
