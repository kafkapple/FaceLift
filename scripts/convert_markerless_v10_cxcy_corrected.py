#!/usr/bin/env python3
"""
Convert markerless_mouse_1_nerf to FaceLift format (v10)

=== v10 핵심 수정 (v5 대비) ===
1. cx, cy 보정: Image shift 후 실제 principal point 위치 반영
   - v5: cx=cy=256 (고정, 부정확)
   - v10: cx=scaled_cx+offset, cy=scaled_cy+offset (정확)

2. 기존 v5 유지:
   - 이미지 스케일에 거리 보정 포함
   - 마스크 Centroid 기반 Centering
   - fy = fx (정사각 픽셀)

=== 기하학적 정확성 개선 ===
v5 문제: Image shift 적용 후 cx,cy=256 고정 → Ray 방향 3-4도 오차
v10 해결: shift offset을 cx,cy에 반영 → Ray 방향 정확

예시:
  scaled_cx = 202, offset_x = 88
  v5:  cx = 256 (오차 34px)
  v10: cx = 202 + 88 = 290 (정확)
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


# 상수
JUMP_FRAMES = {5900, 11800, 17700}
ORIG_SIZE = (1152, 1024)  # width, height
TARGET_SIZE = 512
UNIT_SCALE = 100.0  # mm → normalized
TARGET_DISTANCE = 2.7
TARGET_FX = 549.0


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


def compute_camera_transform_v10(
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
    target_distance: float = TARGET_DISTANCE,
    target_fx: float = TARGET_FX,
    target_size: int = TARGET_SIZE,
    orig_size: Tuple[int, int] = ORIG_SIZE,
) -> Tuple[Dict, float, Tuple[float, float]]:
    """
    Compute camera transform with CORRECTED cx, cy.
    
    v10 핵심 변경:
    - cx, cy = scaled_cx + offset (shift 반영!)
    - v5는 cx=cy=256 고정이었음 (부정확)
    """
    orig_w, orig_h = orig_size
    orig_fx = K[0, 0]
    orig_fy = K[1, 1]
    orig_cx = K[0, 2]
    orig_cy = K[1, 2]

    # === 1. Extrinsics 계산 ===
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.flatten()
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    current_dist_mm = np.linalg.norm(cam_pos)
    current_dist_norm = current_dist_mm / UNIT_SCALE

    # === 2. 이미지 스케일 계산 ===
    fx_ratio = target_fx / orig_fx
    dist_ratio = current_dist_norm / target_distance
    total_image_scale = fx_ratio * dist_ratio

    # === 3. 거리 정규화 (카메라 위치) ===
    cam_direction = cam_pos / current_dist_mm
    new_cam_pos = cam_direction * target_distance
    
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = new_cam_pos
    new_w2c = np.linalg.inv(new_c2w)

    # === 4. Centering: 마스크 Centroid 기반 ===
    centroid_x, centroid_y = compute_mask_centroid(mask)
    
    scaled_centroid_x = centroid_x * total_image_scale
    scaled_centroid_y = centroid_y * total_image_scale
    
    center_offset_x = target_size / 2 - scaled_centroid_x
    center_offset_y = target_size / 2 - scaled_centroid_y

    # === 5. cx, cy 보정 (v10 핵심!) ===
    scaled_cx = orig_cx * total_image_scale
    scaled_cy = orig_cy * total_image_scale
    
    # v5: new_cx = 256, new_cy = 256 (부정확)
    # v10: shift offset 반영 (정확!)
    new_cx = scaled_cx + center_offset_x
    new_cy = scaled_cy + center_offset_y

    # === 6. 카메라 파라미터 구성 ===
    camera_dict = {
        "w": target_size,
        "h": target_size,
        "fx": target_fx,
        "fy": target_fx,  # fy = fx
        "cx": float(new_cx),  # v10: 보정된 값!
        "cy": float(new_cy),  # v10: 보정된 값!
        "w2c": new_w2c.tolist(),
        "_original": {
            "fx": float(orig_fx),
            "fy": float(orig_fy),
            "cx": float(orig_cx),
            "cy": float(orig_cy),
            "distance_mm": float(current_dist_mm),
            "distance_norm": float(current_dist_norm),
        },
        "_transform": {
            "image_scale": float(total_image_scale),
            "centroid_orig": [float(centroid_x), float(centroid_y)],
            "centroid_scaled": [float(scaled_centroid_x), float(scaled_centroid_y)],
            "center_offset": [float(center_offset_x), float(center_offset_y)],
            "scaled_cx": float(scaled_cx),
            "scaled_cy": float(scaled_cy),
            "new_cx": float(new_cx),
            "new_cy": float(new_cy),
            "cx_error_vs_v5": float(new_cx - 256),  # v5 대비 오차
            "cy_error_vs_v5": float(new_cy - 256),
        },
    }

    return camera_dict, total_image_scale, (center_offset_x, center_offset_y)


def apply_image_transform(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
    center_offset: Tuple[float, float],
    output_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Apply scaling and centering transform to image."""
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w <= 0 or new_h <= 0:
        return np.full((output_size, output_size, 4), [255, 255, 255, 0], dtype=np.uint8)

    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    output = np.full((output_size, output_size, 4), [255, 255, 255, 0], dtype=np.uint8)

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

    bg_mask = output[:, :, 3] < 127
    output[bg_mask, :3] = 255

    return output


def process_frame(
    frame_idx: int,
    video_caps: List[cv2.VideoCapture],
    mask_caps: List[cv2.VideoCapture],
    cam_params: List[Dict],
    output_dir: Path,
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
        cam_dict, image_scale, center_offset = compute_camera_transform_v10(
            cam["K"], cam["R"], cam["T"], mask_binary
        )

        transformed = apply_image_transform(
            frame_rgb, mask_binary, image_scale, center_offset
        )

        img_path = images_dir / f"cam_{cam_idx:03d}.png"
        img_pil = Image.fromarray(transformed, mode="RGBA")
        img_pil.save(img_path)

        cam_dict["file_path"] = f"images/cam_{cam_idx:03d}.png"
        cam_dict["view_id"] = cam_idx
        camera_frames.append(cam_dict)

    # 카메라 JSON 저장
    cameras_json = {
        "frames": camera_frames,
        "_preprocessing": {
            "method": "convert_markerless_v10_cxcy_corrected",
            "version": "10.0",
            "changes": [
                "cx, cy corrected to reflect image shift",
                "v5 had cx=cy=256 (incorrect)",
                "v10 has cx=scaled_cx+offset (correct)",
            ],
            "target_distance": TARGET_DISTANCE,
            "target_fx": TARGET_FX,
        },
    }

    with open(output_dir / "opencv_cameras.json", "w") as f:
        json.dump(cameras_json, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(description="v10: cx,cy corrected preprocessing")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Convert Markerless Mouse (v10 - cx,cy corrected)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()
    print("v10 핵심 변경:")
    print("  - v5: cx=cy=256 (고정, 부정확)")
    print("  - v10: cx=scaled_cx+offset (shift 반영, 정확)")
    print("=" * 60)
    print()

    cam_params = load_camera_params(input_dir / "new_cam.pkl")
    print(f"Loaded {len(cam_params)} cameras")

    video_dir = input_dir / "videos_undist"
    mask_dir = input_dir / "simpleclick_undist"

    video_caps = [cv2.VideoCapture(str(video_dir / f"{i}.mp4")) for i in range(6)]
    mask_caps = [cv2.VideoCapture(str(mask_dir / f"{i}.mp4")) for i in range(6)]

    total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    frame_indices = [i for i in range(0, total_frames, args.frame_interval) if i not in JUMP_FRAMES]

    if args.max_samples:
        frame_indices = frame_indices[:args.max_samples]

    print(f"Processing {len(frame_indices)} frames...")

    sample_paths = []
    for idx, frame_idx in enumerate(tqdm(frame_indices)):
        sample_dir = output_dir / f"sample_{idx:06d}"
        if process_frame(frame_idx, video_caps, mask_caps, cam_params, sample_dir):
            sample_paths.append(str(sample_dir.absolute()))

    for cap in video_caps + mask_caps:
        cap.release()

    # Train/Val 분할
    num_samples = len(sample_paths)
    num_val = min(100, max(10, int(num_samples * 0.05)))
    num_train = num_samples - num_val

    with open(output_dir / "data_mouse_train.txt", "w") as f:
        f.write("\n".join(sample_paths[:num_train]))
    with open(output_dir / "data_mouse_val.txt", "w") as f:
        f.write("\n".join(sample_paths[num_train:]))

    print(f"\nDone! {num_samples} samples (Train: {num_train}, Val: {num_val})")

    # 검증 출력
    print("\n" + "=" * 60)
    print("v10 vs v5 비교 (첫 샘플)")
    print("=" * 60)
    
    with open(output_dir / "sample_000000" / "opencv_cameras.json") as f:
        data = json.load(f)
    
    print("\n| View | v10_cx | v10_cy | v5_cx | v5_cy | error_x | error_y |")
    print("|------|--------|--------|-------|-------|---------|---------|")
    
    for frame in data["frames"]:
        t = frame["_transform"]
        print(f"| {frame[view_id]} | {frame[cx]:.1f} | {frame[cy]:.1f} | 256.0 | 256.0 | {t[cx_error_vs_v5]:.1f} | {t[cy_error_vs_v5]:.1f} |")


if __name__ == "__main__":
    main()
