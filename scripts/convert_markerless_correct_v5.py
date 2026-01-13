#!/usr/bin/env python3
"""
Convert markerless_mouse_1_nerf to FaceLift format (v5)

=== v5 핵심 수정 (v3 대비) ===
1. 이미지 스케일에 거리 보정 포함:
   scale = (target_fx / orig_fx) * (orig_dist_norm / target_dist)
   
2. 마스크 Centroid 기반 Centering:
   - Principal point 대신 마스크 무게중심 사용
   - 꼬리 영향 최소화

3. 단위 변환:
   - 원본: mm (246~414)
   - 타겟: normalized (2.7)
   - UNIT_SCALE = 100

=== 이론적 배경 ===
투영 공식: projected_size = fx * (object_size / distance)

같은 물체가 동일 크기로 투영되려면:
orig_fx / orig_dist = target_fx / target_dist (스케일 후)

이미지 스케일:
pixel_target = pixel_orig * scale
scale = (target_fx * orig_dist_norm) / (orig_fx * target_dist)
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


def load_camera_params(pkl_path: str) -> List[Dict]:
    """Load camera parameters from pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Compute centroid of foreground pixels in mask.
    
    Args:
        mask: Binary mask (H, W), foreground > 127
        
    Returns:
        (centroid_x, centroid_y) in pixel coordinates
    """
    fg_coords = np.where(mask > 127)
    if len(fg_coords[0]) == 0:
        # No foreground, return image center
        return mask.shape[1] / 2, mask.shape[0] / 2
    
    centroid_y = fg_coords[0].mean()
    centroid_x = fg_coords[1].mean()
    return centroid_x, centroid_y


def compute_camera_transform(
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,  # v5: 마스크 추가
    target_distance: float = 2.7,
    target_fx: float = 549.0,
    target_size: int = 512,
    orig_size: Tuple[int, int] = (1152, 1024),
) -> Tuple[Dict, float, Tuple[float, float]]:
    """
    Compute camera transform with distance correction and mask-based centering.
    
    v5 핵심 변경:
    1. 이미지 스케일에 거리 보정 포함
    2. 마스크 centroid 기반 centering
    
    Args:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix
        T: 3x1 translation vector
        mask: Binary mask for centroid calculation
        target_distance: Target camera distance (normalized unit)
        target_fx: Target focal length
        target_size: Output image size
        orig_size: Original image size (width, height)
        
    Returns:
        (camera_dict, image_scale, center_offset)
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
    
    # 단위 변환: mm → normalized
    current_dist_norm = current_dist_mm / UNIT_SCALE

    # === 2. 이미지 스케일 계산 (v5 핵심!) ===
    # scale = (target_fx / orig_fx) * (orig_dist_norm / target_dist)
    fx_ratio = target_fx / orig_fx
    dist_ratio = current_dist_norm / target_distance
    total_image_scale = fx_ratio * dist_ratio

    # === 3. 거리 정규화 (카메라 위치) ===
    # 방향 유지, 거리만 target_distance로
    cam_direction = cam_pos / current_dist_mm
    new_cam_pos = cam_direction * target_distance  # 이미 normalized unit
    
    new_c2w = c2w.copy()
    new_c2w[:3, 3] = new_cam_pos
    new_w2c = np.linalg.inv(new_c2w)

    # === 4. Centering: 마스크 Centroid 기반 (v5 핵심!) ===
    centroid_x, centroid_y = compute_mask_centroid(mask)
    
    # 스케일링 후 centroid 위치
    scaled_centroid_x = centroid_x * total_image_scale
    scaled_centroid_y = centroid_y * total_image_scale
    
    # Centroid가 이미지 중앙에 오도록 offset 계산
    center_offset_x = target_size / 2 - scaled_centroid_x
    center_offset_y = target_size / 2 - scaled_centroid_y

    # === 5. 카메라 파라미터 구성 ===
    camera_dict = {
        "w": target_size,
        "h": target_size,
        "fx": target_fx,
        "fy": target_fx,  # v5: fy = fx (정사각 픽셀)
        "cx": target_size / 2,
        "cy": target_size / 2,
        "w2c": new_w2c.tolist(),
        "_original": {
            "fx": float(orig_fx),
            "fy": float(orig_fy),
            "cx": float(orig_cx),
            "cy": float(orig_cy),
            "distance_mm": float(current_dist_mm),
            "distance_norm": float(current_dist_norm),
            "fy_fx_ratio": float(orig_fy / orig_fx),
        },
        "_transform": {
            "image_scale": float(total_image_scale),
            "fx_ratio": float(fx_ratio),
            "dist_ratio": float(dist_ratio),
            "target_distance": target_distance,
            "target_fx": target_fx,
            "centroid_orig": [float(centroid_x), float(centroid_y)],
            "centroid_scaled": [float(scaled_centroid_x), float(scaled_centroid_y)],
            "center_offset": [float(center_offset_x), float(center_offset_y)],
        },
    }

    return camera_dict, total_image_scale, (center_offset_x, center_offset_y)


def apply_image_transform(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
    center_offset: Tuple[float, float],
    output_size: int = 512,
) -> np.ndarray:
    """
    Apply scaling and centering transform to image.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        scale: Image scale factor
        center_offset: (offset_x, offset_y) to center the object
        output_size: Output image size
        
    Returns:
        RGBA image (output_size, output_size, 4)
    """
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w <= 0 or new_h <= 0:
        return np.full(
            (output_size, output_size, 4), [255, 255, 255, 0], dtype=np.uint8
        )

    # 스케일링
    interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 출력 이미지 (흰 배경 + 투명)
    output = np.full((output_size, output_size, 4), [255, 255, 255, 0], dtype=np.uint8)

    # Centering offset 적용
    offset_x = int(center_offset[0])
    offset_y = int(center_offset[1])

    # 소스 영역 (스케일된 이미지에서)
    src_x1 = max(0, -offset_x)
    src_y1 = max(0, -offset_y)
    src_x2 = min(new_w, output_size - offset_x)
    src_y2 = min(new_h, output_size - offset_y)

    # 타겟 영역 (출력 이미지에서)
    dst_x1 = max(0, offset_x)
    dst_y1 = max(0, offset_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        output[dst_y1:dst_y2, dst_x1:dst_x2, :3] = scaled_img[
            src_y1:src_y2, src_x1:src_x2
        ]
        output[dst_y1:dst_y2, dst_x1:dst_x2, 3] = scaled_mask[
            src_y1:src_y2, src_x1:src_x2
        ]

    # 배경 영역을 흰색으로
    bg_mask = output[:, :, 3] < 127
    output[bg_mask, :3] = 255

    return output


def process_frame(
    frame_idx: int,
    video_caps: List[cv2.VideoCapture],
    mask_caps: List[cv2.VideoCapture],
    cam_params: List[Dict],
    output_dir: Path,
    target_distance: float = 2.7,
    target_fx: float = 549.0,
    target_size: int = 512,
) -> bool:
    """Process a single frame across all cameras."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    camera_frames = []

    for cam_idx in range(6):
        # 프레임 읽기
        video_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        mask_caps[cam_idx].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = video_caps[cam_idx].read()
        ret_mask, mask = mask_caps[cam_idx].read()

        if not ret or not ret_mask:
            return False

        # 마스크 처리
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)
        
        # RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 카메라 변환 계산 (v5: 마스크 전달!)
        cam = cam_params[cam_idx]
        cam_dict, image_scale, center_offset = compute_camera_transform(
            cam["K"],
            cam["R"],
            cam["T"],
            mask_binary,  # v5: 마스크 전달
            target_distance=target_distance,
            target_fx=target_fx,
            target_size=target_size,
            orig_size=ORIG_SIZE,
        )

        # 이미지 변환
        transformed = apply_image_transform(
            frame_rgb, mask_binary, image_scale, center_offset, target_size
        )

        # 저장
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
            "method": "convert_markerless_correct_v5",
            "version": "5.0",
            "changes": [
                "Distance correction in image scale",
                "Mask centroid-based centering",
                "Unit conversion (mm → normalized)",
            ],
            "target_distance": target_distance,
            "target_fx": target_fx,
            "target_fy": target_fx,
            "unit_scale": UNIT_SCALE,
            "target_fov_deg": float(
                2 * np.degrees(np.arctan(target_size / 2 / target_fx))
            ),
            "has_alpha": True,
            "square_pixels": True,
            "centroid_centering": True,
        },
    }

    with open(output_dir / "opencv_cameras.json", "w") as f:
        json.dump(cameras_json, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert to FaceLift format (v5 with distance correction + centroid centering)"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frame_interval", type=int, default=5)
    parser.add_argument("--target_distance", type=float, default=2.7)
    parser.add_argument("--target_fx", type=float, default=549)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Convert Markerless Mouse (v5)")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target: distance={args.target_distance}, fx=fy={args.target_fx}")
    print(f"Unit scale: {UNIT_SCALE} (mm → normalized)")
    print()
    print("v5 핵심 변경:")
    print("  1. 이미지 스케일에 거리 보정 포함")
    print("  2. 마스크 Centroid 기반 centering")
    print("=" * 60)
    print()

    # 카메라 파라미터 로드
    cam_params = load_camera_params(input_dir / "new_cam.pkl")
    print(f"Loaded {len(cam_params)} cameras")

    # 비디오 열기
    video_dir = input_dir / "videos_undist"
    mask_dir = input_dir / "simpleclick_undist"

    video_caps = []
    mask_caps = []

    for cam_idx in range(6):
        video_caps.append(cv2.VideoCapture(str(video_dir / f"{cam_idx}.mp4")))
        mask_caps.append(cv2.VideoCapture(str(mask_dir / f"{cam_idx}.mp4")))

    total_frames = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # 처리할 프레임 인덱스
    frame_indices = [
        i for i in range(0, total_frames, args.frame_interval) if i not in JUMP_FRAMES
    ]

    if args.max_samples:
        frame_indices = frame_indices[: args.max_samples]

    print(f"Processing {len(frame_indices)} frames...")

    # 프레임 처리
    sample_paths = []
    for idx, frame_idx in enumerate(tqdm(frame_indices)):
        sample_dir = output_dir / f"sample_{idx:06d}"
        if process_frame(
            frame_idx,
            video_caps,
            mask_caps,
            cam_params,
            sample_dir,
            args.target_distance,
            args.target_fx,
            TARGET_SIZE,
        ):
            sample_paths.append(str(sample_dir.absolute()))

    # 비디오 닫기
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
    print("검증 결과")
    print("=" * 60)
    
    with open(output_dir / "sample_000000" / "opencv_cameras.json") as f:
        data = json.load(f)
    
    print("\n카메라 파라미터:")
    print("| View | fx | fy | distance | image_scale | fx_ratio | dist_ratio |")
    print("|------|-----|-----|----------|-------------|----------|------------|")
    
    for frame in data["frames"]:
        w2c = np.array(frame["w2c"])
        c2w = np.linalg.inv(w2c)
        dist = np.linalg.norm(c2w[:3, 3])
        t = frame["_transform"]
        print(f"| {frame['view_id']} | {frame['fx']:.0f} | {frame['fy']:.0f} | {dist:.3f} | {t['image_scale']:.4f} | {t['fx_ratio']:.4f} | {t['dist_ratio']:.4f} |")


if __name__ == "__main__":
    main()
