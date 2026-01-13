# Step 3: 카메라 정규화 전처리

> 원본 Mouse 데이터를 FaceLift 형식으로 변환하는 전처리 스크립트를 구현합니다.

## 3.1 Why 전처리가 필요한가?

### FaceLift Pretrained Model 가정

모델은 특정 카메라 설정으로 학습되었습니다:

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| fx = fy | 549 | 초점 거리 (정사각 픽셀) |
| cx = cy | 256 | 주점 (512x512 이미지 중앙) |
| distance | 2.7 | 카메라-원점 거리 |
| image_size | 512x512 | 이미지 해상도 |

### 원본 Mouse 데이터 문제

```
View 0: fx=1632, distance=246mm, fx/dist=6.63
View 1: fx=1557, distance=414mm, fx/dist=3.76  ← 비율 다름!
View 2: fx=1630, distance=364mm, fx/dist=4.48
...
```

**문제**: 각 뷰에서 생쥐 크기가 다르게 보임 → 모델 혼란

---

## 3.2 전처리 핵심 수식

### 이미지 스케일 계산

**목표**: 모든 뷰에서 동일한 `fx/distance` 비율 달성

```python
# 핵심 수식
image_scale = (target_fx / orig_fx) * (orig_dist / target_dist)
```

**유도 과정**:
```
원본 투영: pixel_orig = orig_fx * (X / orig_dist)
타겟 투영: pixel_target = target_fx * (X / target_dist)

pixel_target = pixel_orig * scale
→ scale = (target_fx / orig_fx) * (orig_dist / target_dist)
```

### Principal Point (PP) Centered 방식

**권장**: cx=cy=256 고정 (PP 기준 centering)

```python
# PP-centered: 수학적으로 정확
offset = target_size/2 - scaled_cx
shifted_image = shift(image, offset)
# 결과: cx = cy = 256 정확히 보장
```

---

## 3.3 전처리 스크립트 구현

### 파일 생성
`scripts/preprocess_mouse.py`

```python
#!/usr/bin/env python3
"""
Mouse data preprocessing for FaceLift.

Converts raw mouse multi-view data to FaceLift-compatible format:
1. Normalize camera distances to 2.7
2. Adjust intrinsics (fx, fy) proportionally
3. Center images based on principal point
4. Save in opencv_cameras.json format

Usage:
    python scripts/preprocess_mouse.py \
        --input_dir /path/to/raw/mouse \
        --output_dir /path/to/processed \
        --target_fx 549 \
        --target_distance 2.7
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import shift as scipy_shift


# Constants
UNIT_SCALE = 100.0  # mm to normalized units
TARGET_SIZE = 512


def compute_image_scale(orig_fx, orig_dist_mm, target_fx=549, target_dist=2.7):
    """
    Compute image scale factor for normalization.
    
    Args:
        orig_fx: Original focal length in pixels
        orig_dist_mm: Original camera distance in mm
        target_fx: Target focal length (default: 549)
        target_dist: Target camera distance (default: 2.7)
        
    Returns:
        scale: Image scale factor
    """
    orig_dist_norm = orig_dist_mm / UNIT_SCALE
    scale = (target_fx / orig_fx) * (orig_dist_norm / target_dist)
    return scale


def normalize_extrinsics(w2c, target_distance=2.7):
    """
    Normalize camera extrinsics to target distance.
    
    Args:
        w2c: World-to-camera matrix [4, 4]
        target_distance: Target distance from origin
        
    Returns:
        new_w2c: Normalized w2c matrix
    """
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    current_dist = np.linalg.norm(cam_pos)
    
    if current_dist > 0:
        # Scale to target distance (convert from mm to normalized)
        current_dist_norm = current_dist / UNIT_SCALE
        scale = target_distance / current_dist_norm
        new_cam_pos = cam_pos * (scale / UNIT_SCALE)
        
        new_c2w = c2w.copy()
        new_c2w[:3, 3] = new_cam_pos
        new_w2c = np.linalg.inv(new_c2w)
    else:
        new_w2c = w2c.copy()
        
    return new_w2c


def process_view(img_path, mask_path, K, w2c, target_fx=549, target_dist=2.7):
    """
    Process a single view: scale image and normalize camera.
    
    Returns:
        processed_image: PIL Image
        new_K: Normalized intrinsics [3, 3]
        new_w2c: Normalized extrinsics [4, 4]
    """
    # Load image and mask
    img = Image.open(img_path).convert("RGBA")
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
    else:
        mask = None
    
    orig_size = img.size[0]
    
    # Extract camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Compute camera distance (from w2c)
    c2w = np.linalg.inv(w2c)
    cam_pos = c2w[:3, 3]
    orig_dist_mm = np.linalg.norm(cam_pos)
    
    # Compute image scale
    scale = compute_image_scale(fx, orig_dist_mm, target_fx, target_dist)
    
    # Scale image
    new_size = int(orig_size * scale)
    img_scaled = img.resize((new_size, new_size), Image.LANCZOS)
    
    # PP-centered: offset to center principal point
    scaled_cx = cx * scale
    scaled_cy = cy * scale
    offset_x = TARGET_SIZE / 2 - scaled_cx
    offset_y = TARGET_SIZE / 2 - scaled_cy
    
    # Create output image
    output = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255, 0))
    paste_x = int(offset_x + (TARGET_SIZE - new_size) / 2)
    paste_y = int(offset_y + (TARGET_SIZE - new_size) / 2)
    
    # Paste scaled image
    if 0 <= paste_x < TARGET_SIZE and 0 <= paste_y < TARGET_SIZE:
        output.paste(img_scaled, (paste_x, paste_y))
    
    # Normalize intrinsics: fx=fy=target_fx, cx=cy=256
    new_K = np.array([
        [target_fx, 0, TARGET_SIZE / 2],
        [0, target_fx, TARGET_SIZE / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Normalize extrinsics
    new_w2c = normalize_extrinsics(w2c, target_dist)
    
    return output, new_K, new_w2c


def process_sample(input_dir, output_dir, target_fx=549, target_dist=2.7):
    """
    Process a single sample (all views).
    
    Args:
        input_dir: Input sample directory
        output_dir: Output sample directory
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # Load original camera file
    with open(f"{input_dir}/opencv_cameras.json") as f:
        orig_cameras = json.load(f)
    
    new_cameras = {}
    
    for view_idx in range(6):
        cam_key = f"cam_{view_idx:03d}"
        
        # Paths
        img_path = f"{input_dir}/images/{cam_key}.png"
        mask_path = f"{input_dir}/masks/{cam_key}.png"
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping")
            continue
        
        # Load original camera params
        cam = orig_cameras[cam_key]
        K = np.array(cam["K"]).reshape(3, 3)
        w2c = np.array(cam["w2c"]).reshape(4, 4)
        
        # Process view
        processed_img, new_K, new_w2c = process_view(
            img_path, mask_path, K, w2c, target_fx, target_dist
        )
        
        # Save processed image
        processed_img.save(f"{output_dir}/images/{cam_key}.png")
        
        # Save new camera params
        new_cameras[cam_key] = {
            "K": new_K.flatten().tolist(),
            "w2c": new_w2c.flatten().tolist(),
            "img_size": [TARGET_SIZE, TARGET_SIZE]
        }
    
    # Save camera file
    with open(f"{output_dir}/opencv_cameras.json", "w") as f:
        json.dump(new_cameras, f, indent=2)
    
    print(f"Processed: {input_dir} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess mouse data for FaceLift")
    parser.add_argument("--input_dir", required=True, help="Input data directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--target_fx", type=float, default=549, help="Target focal length")
    parser.add_argument("--target_distance", type=float, default=2.7, help="Target camera distance")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # Find all samples (directories with opencv_cameras.json)
    samples = list(input_path.glob("**/opencv_cameras.json"))
    print(f"Found {len(samples)} samples")
    
    for camera_file in samples:
        sample_dir = camera_file.parent
        rel_path = sample_dir.relative_to(input_path)
        out_dir = output_path / rel_path
        
        process_sample(str(sample_dir), str(out_dir), args.target_fx, args.target_distance)
    
    print(f"\nDone! Processed {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
```

---

## 3.4 사용법

### 기본 사용

```bash
python scripts/preprocess_mouse.py \
    --input_dir /path/to/raw/markerless_mouse \
    --output_dir /path/to/processed/facelift_mouse \
    --target_fx 549 \
    --target_distance 2.7
```

### 데이터 분할 생성

```bash
# train/val 파일 리스트 생성
ls /path/to/processed/facelift_mouse/*/opencv_cameras.json | \
    xargs -I {} dirname {} > all_samples.txt

# 80/20 분할
head -n $(($(wc -l < all_samples.txt) * 80 / 100)) all_samples.txt > data_mouse_train.txt
tail -n +$(($(wc -l < all_samples.txt) * 80 / 100 + 1)) all_samples.txt > data_mouse_val.txt
```

---

## 3.5 전처리 결과 검증

```python
# 검증 스크립트
import json
import numpy as np

def verify_preprocessing(sample_dir):
    with open(f"{sample_dir}/opencv_cameras.json") as f:
        cameras = json.load(f)
    
    print("=== Preprocessing Verification ===")
    for cam_key, cam in cameras.items():
        K = np.array(cam["K"]).reshape(3, 3)
        w2c = np.array(cam["w2c"]).reshape(4, 4)
        c2w = np.linalg.inv(w2c)
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        dist = np.linalg.norm(c2w[:3, 3])
        
        print(f"{cam_key}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, dist={dist:.2f}")
    
    print("\n✅ All views should have: fx=fy=549, cx=cy=256, dist=2.7")

verify_preprocessing("/path/to/processed/sample_000")
```

**예상 출력:**
```
=== Preprocessing Verification ===
cam_000: fx=549.0, fy=549.0, cx=256.0, cy=256.0, dist=2.70
cam_001: fx=549.0, fy=549.0, cx=256.0, cy=256.0, dist=2.70
cam_002: fx=549.0, fy=549.0, cx=256.0, cy=256.0, dist=2.70
...

✅ All views should have: fx=fy=549, cx=cy=256, dist=2.7
```

---

## 다음 단계

✅ 전처리 스크립트 구현 완료

→ [Step4: Config 설정](./Step4_Config_Setup.md)

---

*Created: 2026-01-13*
