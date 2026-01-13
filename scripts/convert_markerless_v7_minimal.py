#!/usr/bin/env python3
"""
Convert markerless_mouse_1_nerf to FaceLift format (v7 - Minimal Transform)

=== v7: 최소 변환 버전 ===
변환하는 것:
  - fy = fx 강제 (정사각 픽셀)
  - 이미지 512x512 리사이즈 (단순 비율)
  
변환하지 않는 것:
  - 거리 정규화 없음 (원본 거리 유지)
  - Centroid/Principal point centering 없음
  - 거리 기반 이미지 스케일 보정 없음

목적: Centering/Scaling이 Ghost 원인인지 검증
"""

import argparse
import json
import os
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
UNIT_SCALE = 100.0  # mm → normalized unit


def load_camera_params(pkl_path: str) -> List[Dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_camera_transform_minimal(
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    target_size: int = 512,
    orig_size: Tuple[int, int] = (1152, 1024),
) -> Dict:
    """
    v7 최소 변환: 리사이즈 비율만 적용, 거리/센터링 변환 없음
    """
    orig_w, orig_h = orig_size
    orig_fx = K[0, 0]
    orig_fy = K[1, 1]
    orig_cx = K[0, 2]
    orig_cy = K[1, 2]

    # 단순 리사이즈 비율 (정사각형으로 맞춤)
    # 원본 1152x1024 → 512x512
    # 긴 축(width) 기준으로 스케일
    scale = target_size / orig_w  # 512/1152 = 0.4444

    # Intrinsics 스케일링 (단순 비율)
    new_fx = orig_fx * scale
    new_fy = orig_fx * scale  # fy = fx 강제!
    new_cx = orig_cx * scale
    new_cy = orig_cy * scale

    # 원본 w2c 행렬 (변환 없음)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()
    
    # 거리 계산 (정보용, 변환은 안함)
    c2w = np.linalg.inv(w2c)
    distance_mm = np.linalg.norm(c2w[:3, 3])
    distance_norm = distance_mm / UNIT_SCALE

    camera_dict = {
        "w": target_size,
        "h": target_size,
        "fx": float(new_fx),
        "fy": float(new_fy),  # = new_fx
        "cx": float(new_cx),
        "cy": float(new_cy),
        "w2c": w2c.tolist(),
        "_original": {
            "fx": float(orig_fx),
            "fy": float(orig_fy),
            "cx": float(orig_cx),
            "cy": float(orig_cy),
            "distance_mm": float(distance_mm),
            "distance_norm": float(distance_norm),
            "fy_fx_ratio": float(orig_fy / orig_fx),
        },
        "_transform": {
            "method": "minimal_v7",
            "resize_scale": float(scale),
            "distance_normalized": False,
            "centering_applied": False,
        },
    }

    return camera_dict, scale


def process_image_minimal(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
    target_size: int = 512,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    최소 변환: 리사이즈만, centering 없음
    원본 비율 유지하며 512x512에 맞춤 (패딩 추가)
    """
    h, w = image.shape[:2]
    
    # 스케일링
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 512x512 캔버스에 좌상단 정렬 (centering 없음)
    output = np.full((target_size, target_size, 4), (*background, 0), dtype=np.uint8)
    
    # 이미지가 캔버스보다 크면 crop, 작으면 패딩
    paste_w = min(new_w, target_size)
    paste_h = min(new_h, target_size)
    
    output[:paste_h, :paste_w, :3] = scaled_img[:paste_h, :paste_w]
    output[:paste_h, :paste_w, 3] = scaled_mask[:paste_h, :paste_w]
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Convert markerless mouse to FaceLift (v7 - minimal)")
    parser.add_argument("--input_dir", type=str, default="/home/joon/data/markerless_mouse_1_nerf")
    parser.add_argument("--output_dir", type=str, default="/home/joon/data/preprocessed/FaceLift_mouse/data_mouse_v7_minimal")
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera parameters
    cam_params = load_camera_params(input_dir / "new_cam.pkl")
    
    # Video and mask paths
    video_dir = input_dir / "videos_undist"
    mask_dir = input_dir / "simpleclick_undist"
    
    video_paths = sorted(video_dir.glob("cam_*.mp4"))
    
    if len(video_paths) != 6:
        raise ValueError(f"Expected 6 cameras, found {len(video_paths)} videos")

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
        valid_sample = True

        for cam_idx in range(6):
            caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = caps[cam_idx].read()
            if not ret:
                valid_sample = False
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 마스크 로드
            mask_path = mask_dir / f"cam_{cam_idx}" / f"{frame_idx:06d}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255

            K = np.array(cam_params[cam_idx]["K"])
            R = np.array(cam_params[cam_idx]["R"])
            T = np.array(cam_params[cam_idx]["T"])

            camera_dict, img_scale = compute_camera_transform_minimal(
                K, R, T,
                target_size=args.target_size,
                orig_size=ORIG_SIZE,
            )

            processed = process_image_minimal(
                frame_rgb, mask, img_scale,
                target_size=args.target_size,
            )

            output_path = sample_dir / f"cam_{cam_idx:03d}.png"
            Image.fromarray(processed).save(output_path)

            camera_dict["file_path"] = f"images/cam_{cam_idx:03d}.png"
            camera_dict["view_id"] = cam_idx
            frames_data["frames"].append(camera_dict)

        if valid_sample and len(frames_data["frames"]) == 6:
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
    
    # 카메라 파라미터 요약 출력
    print("\n=== Camera Parameters Summary ===")
    with open(output_dir / "sample_000000" / "opencv_cameras.json") as f:
        d = json.load(f)
        for frame in d["frames"]:
            orig = frame["_original"]
            print(f"View {frame['view_id']}: fx={frame['fx']:.1f}, dist={orig['distance_norm']:.2f}, fy/fx_orig={orig['fy_fx_ratio']:.4f}")


if __name__ == "__main__":
    main()
