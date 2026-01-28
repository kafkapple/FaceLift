# 합성 데이터셋 제작 매뉴얼: 저 Coverage 비정형 물체

**Version**: 1.0
**Date**: 2026-01-28
**목적**: 마우스 데이터 특성 반영한 합성 데이터 생성

---

## 1. 마우스 데이터 특성 분석

### 1.1 실제 마우스 데이터 통계

| 특성 | 값 | 영향 |
|------|-----|------|
| **FG Coverage** | ~5% | 배경 90%+, 물체 작음 |
| **물체 위치** | 가변 (±100px) | PP ≠ 물체 중심 |
| **형상** | 비대칭 (꼬리, 사지) | 뷰별 silhouette 다름 |
| **크기 변동** | ±30% | 자세에 따라 bbox 변화 |

### 1.2 합성 데이터가 재현해야 할 조건

```
┌─────────────────────────────────────────────────┐
│                                                 │
│                    배경 (~95%)                   │
│                                                 │
│         ┌───────┐                               │
│         │ 물체  │  ← 작고, 중앙 아님            │
│         │ (~5%) │                               │
│         └───────┘                               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 2. 3D 모델 준비

### 2.1 권장 모델 특성

| 특성 | 요구사항 | 이유 |
|------|----------|------|
| **크기** | Unit sphere의 20-30% | 낮은 coverage |
| **형상** | 비대칭, 세부 특징 | 마우스 유사 |
| **재질** | 단순 diffuse | 조명 영향 최소화 |

### 2.2 모델 옵션

#### Option A: Blender 내장 모델 수정

```python
# Blender Python: 원숭이 머리 (Suzanne) 축소
import bpy

# 기본 Suzanne 생성
bpy.ops.mesh.primitive_monkey_add(size=0.3, location=(0, 0, 0))
obj = bpy.context.active_object
obj.name = "SmallMonkey"

# 크기 조정 (unit sphere의 ~25%)
obj.scale = (0.25, 0.25, 0.25)
bpy.ops.object.transform_apply(scale=True)
```

#### Option B: 단순 비대칭 형상 생성

```python
# 타원체 + 돌출부 (마우스 유사)
import bpy
import bmesh

def create_mouse_like_shape():
    """마우스 유사 비대칭 형상 생성"""
    
    # 기본 타원체 (몸통)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.15, 
        location=(0, 0, 0),
        segments=32, 
        ring_count=16
    )
    body = bpy.context.active_object
    body.name = "MouseBody"
    
    # X축 방향으로 늘리기 (타원형)
    body.scale = (1.5, 0.8, 0.7)
    bpy.ops.object.transform_apply(scale=True)
    
    # 꼬리 (얇은 원뿔)
    bpy.ops.mesh.primitive_cone_add(
        radius1=0.03, 
        radius2=0.005,
        depth=0.3,
        location=(-0.25, 0, 0)
    )
    tail = bpy.context.active_object
    tail.name = "Tail"
    tail.rotation_euler = (0, 1.57, 0)  # 90도 회전
    
    # 머리 (작은 구)
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=0.08,
        location=(0.2, 0, 0.03)
    )
    head = bpy.context.active_object
    head.name = "Head"
    
    # 합치기
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    tail.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    
    return body

# 실행
mouse_obj = create_mouse_like_shape()
```

#### Option C: 외부 모델 다운로드

```bash
# Sketchfab에서 무료 마우스/쥐 모델 다운로드
# 권장: Low-poly mouse, rat 모델

# 또는 Objaverse에서
pip install objaverse
python -c "
import objaverse
# 작은 동물 모델 검색
uids = objaverse.load_uids()
# 다운로드...
"
```

### 2.3 모델 정규화 스크립트

```python
# normalize_model.py - Blender에서 실행
import bpy
import mathutils

def normalize_model_for_gslrm(obj, target_size=0.25):
    """
    모델을 GS-LRM 학습에 적합하게 정규화
    
    Args:
        obj: Blender object
        target_size: Unit sphere 대비 목표 크기 (0.25 = 25%)
    """
    
    # 1. 원점으로 이동
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")
    obj.location = (0, 0, 0)
    
    # 2. 현재 bounding box 크기 계산
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector((min(v.x for v in bbox), 
                                    min(v.y for v in bbox), 
                                    min(v.z for v in bbox)))
    max_corner = mathutils.Vector((max(v.x for v in bbox), 
                                    max(v.y for v in bbox), 
                                    max(v.z for v in bbox)))
    
    current_size = max(max_corner - min_corner)
    
    # 3. 목표 크기로 스케일링
    scale_factor = target_size / current_size
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(scale=True)
    
    print(f"Model normalized: {current_size:.3f} -> {target_size:.3f}")
    print(f"Coverage estimate: ~{(target_size/2)**2 * 100:.1f}% of image")
    
    return obj

# 사용
obj = bpy.context.active_object
normalize_model_for_gslrm(obj, target_size=0.25)
```

---

## 3. 물체 위치 변형 실험

### 3.1 위치 변형 매트릭스

핵심: **물체가 이미지 중앙이 아닐 때의 영향 측정**

| 실험 ID | 물체 위치 (X, Y) | PP 설정 | 예상 결과 |
|---------|------------------|---------|----------|
| **POS_C** | (0, 0) 중앙 | 256, 256 | 기준선 (정상) |
| **POS_R** | (+0.3, 0) 오른쪽 | 256, 256 | PP 불일치 → Ghosting |
| **POS_U** | (0, +0.3) 위쪽 | 256, 256 | PP 불일치 → Ghosting |
| **POS_RU** | (+0.3, +0.3) | 256, 256 | 심한 Ghosting |
| **POS_R_PP** | (+0.3, 0) | 보정된 PP | 정상 (PP 보정 효과) |

### 3.2 위치 오프셋 → PP 오차 계산

```python
import numpy as np

def compute_pp_offset(object_offset_3d, camera_distance=2.7, fx=549):
    """
    물체 3D 위치 오프셋 → 이미지 PP 오프셋 계산
    
    Args:
        object_offset_3d: (dx, dy, dz) 3D 오프셋
        camera_distance: 카메라 거리
        fx: 초점 거리 (pixels)
    
    Returns:
        (dcx, dcy): PP 오프셋 (pixels)
    """
    dx, dy, dz = object_offset_3d
    
    # 투영 (pinhole camera model)
    # 물체가 오른쪽으로 이동 → 이미지에서도 오른쪽으로 이동
    dcx = (dx / camera_distance) * fx
    dcy = (dy / camera_distance) * fx
    
    return dcx, dcy

# 예시: 물체가 (0.3, 0.2, 0) 위치에 있을 때
offset_3d = (0.3, 0.2, 0)
dcx, dcy = compute_pp_offset(offset_3d, camera_distance=2.7, fx=549)
print(f"PP offset: ({dcx:.1f}, {dcy:.1f}) pixels")
# 출력: PP offset: (61.0, 40.7) pixels
```

### 3.3 위치별 렌더링 설정

```python
# position_experiments.py

POSITION_EXPERIMENTS = {
    # 중앙 (기준선)
    "POS_C": {
        "object_location": (0, 0, 0),
        "cx": 256, "cy": 256,
        "description": "Center - baseline"
    },
    
    # 오른쪽으로 오프셋 (PP 불일치)
    "POS_R": {
        "object_location": (0.3, 0, 0),
        "cx": 256, "cy": 256,  # PP는 중앙 유지
        "description": "Right offset - PP mismatch"
    },
    
    # 오른쪽으로 오프셋 (PP 보정)
    "POS_R_PP": {
        "object_location": (0.3, 0, 0),
        "cx": 256 + 61, "cy": 256,  # PP 보정
        "description": "Right offset - PP corrected"
    },
    
    # 위쪽으로 오프셋
    "POS_U": {
        "object_location": (0, 0.2, 0),
        "cx": 256, "cy": 256,
        "description": "Up offset - PP mismatch"
    },
    
    # 대각선 오프셋
    "POS_RU": {
        "object_location": (0.3, 0.2, 0),
        "cx": 256, "cy": 256,
        "description": "Diagonal offset - severe PP mismatch"
    },
    
    # 대각선 오프셋 + PP 보정
    "POS_RU_PP": {
        "object_location": (0.3, 0.2, 0),
        "cx": 256 + 61, "cy": 256 + 41,
        "description": "Diagonal offset - PP corrected"
    },
    
    # 랜덤 위치 (실제 마우스 데이터 시뮬레이션)
    "POS_RANDOM": {
        "object_location": "random",  # 렌더링 시 랜덤 생성
        "cx": 256, "cy": 256,
        "description": "Random position - simulates real mouse data"
    },
}
```

---

## 4. 완전한 렌더링 파이프라인

### 4.1 메인 렌더링 스크립트

```python
#!/usr/bin/env python3
"""
render_lowcoverage_dataset.py

저 coverage 비정형 물체 + 위치 변형 실험용 렌더링 스크립트

Usage:
    blender --background scene.blend -P render_lowcoverage_dataset.py -- \\
        --output_dir /path/to/output \\
        --experiment POS_C \\
        --num_samples 100
"""

import bpy
import numpy as np
import json
import math
import os
import sys
import argparse
import random
from mathutils import Vector, Matrix


# =============================================================================
# 설정
# =============================================================================

EXPERIMENTS = {
    "POS_C": {"loc": (0, 0, 0), "cx": 256, "cy": 256},
    "POS_R": {"loc": (0.3, 0, 0), "cx": 256, "cy": 256},
    "POS_R_PP": {"loc": (0.3, 0, 0), "cx": 317, "cy": 256},
    "POS_U": {"loc": (0, 0.2, 0), "cx": 256, "cy": 256},
    "POS_RU": {"loc": (0.3, 0.2, 0), "cx": 256, "cy": 256},
    "POS_RU_PP": {"loc": (0.3, 0.2, 0), "cx": 317, "cy": 297},
    "POS_RANDOM": {"loc": "random", "cx": 256, "cy": 256},
}

CONFIG = {
    "resolution": 512,
    "fov_deg": 50.0,
    "camera_distance": 2.7,
    "elevation_deg": 20,
    "num_views": 6,
    "object_size": 0.25,  # Unit sphere의 25%
    "render_samples": 64,
}


# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_random_offset(max_offset=0.4):
    """랜덤 위치 오프셋 생성 (XY 평면)"""
    dx = random.uniform(-max_offset, max_offset)
    dy = random.uniform(-max_offset, max_offset)
    return (dx, dy, 0)


def compute_focal_px(fov_deg, resolution):
    """FOV에서 픽셀 초점거리 계산"""
    return resolution / (2 * math.tan(math.radians(fov_deg) / 2))


def setup_camera(fov_deg=50, sensor_width=36):
    """카메라 생성 및 설정"""
    # 기존 카메라 제거
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # 새 카메라
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # 초점거리 설정
    focal_mm = (sensor_width / 2) / math.tan(math.radians(fov_deg) / 2)
    cam_data.lens = focal_mm
    cam_data.sensor_width = sensor_width
    cam_data.shift_x = 0
    cam_data.shift_y = 0
    
    return cam_obj, cam_data


def position_camera(cam_obj, azimuth_deg, elevation_deg, distance, look_at=(0,0,0)):
    """카메라를 구면 좌표에 배치하고 look_at 지점을 바라봄"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    
    # 구면 → 직교 좌표
    x = distance * math.cos(el) * math.sin(az) + look_at[0]
    y = distance * math.cos(el) * math.cos(az) + look_at[1]
    z = distance * math.sin(el) + look_at[2]
    
    cam_obj.location = (x, y, z)
    
    # look_at 방향으로 회전
    direction = Vector(look_at) - Vector((x, y, z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def extract_camera_params(cam_obj, cam_data, view_id, config, cx_override=None, cy_override=None):
    """카메라 파라미터 추출 (GS-LRM 형식)"""
    
    fx = fy = compute_focal_px(config["fov_deg"], config["resolution"])
    cx = cx_override if cx_override else config["resolution"] / 2
    cy = cy_override if cy_override else config["resolution"] / 2
    
    c2w = np.array(cam_obj.matrix_world)
    w2c = np.linalg.inv(c2w)
    
    return {
        "w": config["resolution"],
        "h": config["resolution"],
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w2c": w2c.tolist(),
        "c2w": c2w.tolist(),
        "file_path": f"images/cam_{view_id:03d}.png",
        "view_id": view_id,
    }


def setup_render_engine(config):
    """렌더 엔진 설정"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config["render_samples"]
    scene.cycles.use_denoising = True
    scene.render.resolution_x = config["resolution"]
    scene.render.resolution_y = config["resolution"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True


def setup_lighting():
    """3점 조명 설정"""
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Key light
    key = bpy.data.lights.new("Key", "SUN")
    key.energy = 3.0
    key_obj = bpy.data.objects.new("Key", key)
    bpy.context.collection.objects.link(key_obj)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Fill light
    fill = bpy.data.lights.new("Fill", "SUN")
    fill.energy = 1.0
    fill_obj = bpy.data.objects.new("Fill", fill)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(45), 0, math.radians(-135))


# =============================================================================
# 메인 렌더링 함수
# =============================================================================

def render_sample(
    sample_id: int,
    experiment_name: str,
    output_dir: str,
    model_obj,
    config: dict
):
    """단일 샘플 렌더링 (6개 뷰)"""
    
    exp = EXPERIMENTS[experiment_name]
    
    # 물체 위치 설정
    if exp["loc"] == "random":
        object_loc = get_random_offset(max_offset=0.4)
    else:
        object_loc = exp["loc"]
    
    model_obj.location = object_loc
    bpy.context.view_layer.update()
    
    # 카메라가 바라볼 지점 (물체 위치)
    look_at = object_loc
    
    # PP 설정
    cx, cy = exp["cx"], exp["cy"]
    
    # 샘플 디렉토리
    sample_dir = os.path.join(output_dir, f"sample_{sample_id:05d}")
    os.makedirs(os.path.join(sample_dir, "images"), exist_ok=True)
    
    frames = []
    
    # 6개 뷰 렌더링
    for view_id in range(config["num_views"]):
        azimuth = view_id * (360 / config["num_views"])
        
        # 카메라 배치 (물체 중심으로 공전)
        position_camera(
            cam_obj, azimuth, 
            config["elevation_deg"], 
            config["camera_distance"],
            look_at=look_at
        )
        bpy.context.view_layer.update()
        
        # 렌더링
        filename = f"cam_{view_id:03d}.png"
        bpy.context.scene.render.filepath = os.path.join(sample_dir, "images", filename)
        bpy.ops.render.render(write_still=True)
        
        # 카메라 파라미터 저장
        params = extract_camera_params(cam_obj, cam_data, view_id, config, cx, cy)
        params["object_location"] = list(object_loc)
        params["experiment"] = experiment_name
        frames.append(params)
    
    # JSON 저장
    camera_data = {
        "frames": frames,
        "_metadata": {
            "sample_id": sample_id,
            "experiment": experiment_name,
            "object_location": list(object_loc),
            "cx": cx,
            "cy": cy,
            "fx": frames[0]["fx"],
            "fy": frames[0]["fy"],
        }
    }
    
    with open(os.path.join(sample_dir, "opencv_cameras.json"), "w") as f:
        json.dump(camera_data, f, indent=2)
    
    return sample_dir


def render_dataset(
    experiment_name: str,
    output_dir: str,
    num_samples: int,
    config: dict
):
    """전체 데이터셋 렌더링"""
    
    print(f"\n{=*60}")
    print(f"Rendering: {experiment_name}")
    print(f"Samples: {num_samples}")
    print(f"Output: {output_dir}")
    print(f"{=*60}\n")
    
    # 씬 설정
    setup_render_engine(config)
    setup_lighting()
    global cam_obj, cam_data
    cam_obj, cam_data = setup_camera(config["fov_deg"])
    
    # 모델 찾기
    model_obj = None
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            model_obj = obj
            break
    
    if model_obj is None:
        raise RuntimeError("No mesh object found in scene!")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 샘플별 렌더링
    sample_dirs = []
    for i in range(num_samples):
        print(f"\n[{i+1}/{num_samples}] Rendering sample...")
        sample_dir = render_sample(i, experiment_name, output_dir, model_obj, config)
        sample_dirs.append(sample_dir)
    
    # 데이터 목록 저장
    with open(os.path.join(output_dir, "data_list.txt"), "w") as f:
        for d in sample_dirs:
            f.write(d + "\n")
    
    print(f"\n{=*60}")
    print(f"Dataset complete: {output_dir}")
    print(f"Samples: {num_samples}")
    print(f"{=*60}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # Blender 인자 파싱
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment", default="POS_C", 
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args(argv)
    
    render_dataset(
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        config=CONFIG
    )
```

### 4.2 배치 실행 스크립트

```bash
#!/bin/bash
# run_all_experiments.sh

BLENDER="/path/to/blender"
SCRIPT="render_lowcoverage_dataset.py"
SCENE="mouse_model.blend"
OUTPUT_BASE="/home/joon/data/synthetic/position_experiments"
NUM_SAMPLES=100

EXPERIMENTS=("POS_C" "POS_R" "POS_R_PP" "POS_U" "POS_RU" "POS_RU_PP" "POS_RANDOM")

for EXP in "${EXPERIMENTS[@]}"; do
    echo "========================================"
    echo "Running experiment: $EXP"
    echo "========================================"
    
    OUTPUT_DIR="${OUTPUT_BASE}/${EXP}"
    
    $BLENDER --background $SCENE -P $SCRIPT -- \
        --output_dir "$OUTPUT_DIR" \
        --experiment "$EXP" \
        --num_samples $NUM_SAMPLES
    
    echo "Completed: $EXP"
done

echo "All experiments complete!"
```

---

## 5. Coverage 계산 및 검증

### 5.1 Coverage 측정 스크립트

```python
# compute_coverage.py

import numpy as np
from PIL import Image
from pathlib import Path
import json

def compute_fg_coverage(image_path: str) -> float:
    """이미지의 전경 coverage 계산 (alpha 채널 기반)"""
    img = np.array(Image.open(image_path))
    
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        fg_pixels = np.sum(alpha > 128)
        total_pixels = alpha.size
        coverage = fg_pixels / total_pixels
        return coverage
    else:
        raise ValueError("Image must have alpha channel")


def analyze_dataset_coverage(dataset_dir: str):
    """데이터셋 전체 coverage 분석"""
    
    dataset_path = Path(dataset_dir)
    
    all_coverages = []
    
    for sample_dir in sorted(dataset_path.glob("sample_*")):
        images_dir = sample_dir / "images"
        
        sample_coverages = []
        for img_path in sorted(images_dir.glob("cam_*.png")):
            cov = compute_fg_coverage(str(img_path))
            sample_coverages.append(cov)
        
        avg_cov = np.mean(sample_coverages)
        all_coverages.append(avg_cov)
    
    # 통계
    print(f"\n{=*50}")
    print(f"Coverage Analysis: {dataset_dir}")
    print(f"{=*50}")
    print(f"Samples: {len(all_coverages)}")
    print(f"Mean coverage: {np.mean(all_coverages)*100:.2f}%")
    print(f"Std coverage: {np.std(all_coverages)*100:.2f}%")
    print(f"Min coverage: {np.min(all_coverages)*100:.2f}%")
    print(f"Max coverage: {np.max(all_coverages)*100:.2f}%")
    print(f"{=*50}")
    
    return {
        "mean": np.mean(all_coverages),
        "std": np.std(all_coverages),
        "min": np.min(all_coverages),
        "max": np.max(all_coverages),
    }


if __name__ == "__main__":
    import sys
    analyze_dataset_coverage(sys.argv[1])
```

### 5.2 목표 Coverage

| 조건 | Coverage | 비고 |
|------|----------|------|
| **실제 마우스** | ~5% | 목표 |
| **object_size=0.25** | ~5-7% | 적합 |
| **object_size=0.15** | ~2-3% | 너무 작음 |
| **object_size=0.35** | ~10-12% | 너무 큼 |

---

## 6. GS-LRM 학습 통합

### 6.1 데이터셋 Config 작성

```yaml
# configs/datasets/synthetic_position.yaml

_target_: datasets.synthetic_position
name: synthetic_position
description: "Synthetic low-coverage position experiments"

paths:
  POS_C: /home/joon/data/synthetic/position_experiments/POS_C
  POS_R: /home/joon/data/synthetic/position_experiments/POS_R
  POS_R_PP: /home/joon/data/synthetic/position_experiments/POS_R_PP
  POS_U: /home/joon/data/synthetic/position_experiments/POS_U
  POS_RU: /home/joon/data/synthetic/position_experiments/POS_RU
  POS_RU_PP: /home/joon/data/synthetic/position_experiments/POS_RU_PP
  POS_RANDOM: /home/joon/data/synthetic/position_experiments/POS_RANDOM

# 공통 설정
camera:
  fx: 549
  fy: 549
  # cx, cy는 실험별로 다름 (JSON에서 로드)

preprocessing:
  normalize_cameras: true
  target_camera_distance: 2.7
  normalize_to_z_up: true
```

### 6.2 학습 명령어

```bash
# 기준선 (중앙 위치)
torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -b gslrm_mouse -d synthetic_POS_C -e E0 \
    --wandb_name "synthetic_POS_C"

# PP 불일치 실험
torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -b gslrm_mouse -d synthetic_POS_R -e E0 \
    --wandb_name "synthetic_POS_R_pp_mismatch"

# PP 보정 실험
torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -b gslrm_mouse -d synthetic_POS_R_PP -e E0 \
    --wandb_name "synthetic_POS_R_pp_corrected"
```

---

## 7. 실험 결과 분석

### 7.1 비교 항목

| 실험 쌍 | 비교 목적 |
|---------|----------|
| POS_C vs POS_R | PP 불일치 영향 정량화 |
| POS_R vs POS_R_PP | PP 보정 효과 검증 |
| POS_RU vs POS_RU_PP | 심한 불일치에서 보정 효과 |
| POS_RANDOM vs POS_C | 실제 데이터 시뮬레이션 영향 |

### 7.2 예상 결과

| 실험 | 예상 PSNR | 예상 Ghosting |
|------|----------|--------------|
| POS_C | 높음 (~28) | 없음 |
| POS_R | 낮음 (~22) | Type A 발생 |
| POS_R_PP | 높음 (~27) | 없음/약함 |
| POS_RU | 매우 낮음 (~18) | Type A+B |
| POS_RANDOM | 중간 (~23) | 가변적 |

---

## 8. 체크리스트

### Phase 1: 환경 준비
- [ ] Blender 4.0+ 설치 (gpu03)
- [ ] Python 패키지 확인 (numpy, Pillow)
- [ ] GPU 렌더링 테스트

### Phase 2: 모델 준비
- [ ] 마우스 유사 형상 생성
- [ ] 크기 정규화 (0.25)
- [ ] 머티리얼 설정 (단순 diffuse)
- [ ] Coverage ~5% 확인

### Phase 3: 렌더링
- [ ] 테스트 렌더링 (1 샘플)
- [ ] 전체 실험 렌더링 (7개 × 100 샘플)
- [ ] Coverage 검증
- [ ] 카메라 파라미터 검증

### Phase 4: 학습 및 분석
- [ ] GS-LRM Finetuning (각 실험)
- [ ] Turntable 시각화 비교
- [ ] 정량적 지표 비교
- [ ] 결론 도출

---

*Synthetic Data Manual v1.0 | Low-Coverage Asymmetric Objects | 2026-01-28*

## Synthetic Data Overview (from archive)

# Synthetic Data Generation Overview

**Version**: 1.0.0
**Created**: 2026-01-28
**Purpose**: PP 가설 검증을 위한 합성 데이터 생성

---

## 1. 핵심 가설

> **PP (Principal Point) 불일치가 Ghosting의 주요 원인**

```
물체가 이미지 중앙에 없음 → PP offset 발생 → 광선 방향 오류 → Ghosting
```

---

## 2. FaceLift 원본 학습 설정

### 데이터 형식
```
sample_XXX/
├── images/cam_000.png ~ cam_031.png  (32개 뷰)
└── opencv_cameras.json
```

### 카메라 파라미터
| 항목 | 값 |
|------|-----|
| 뷰 수 | 32 |
| azimuth | 0°~360° (11.25° 간격) |
| elevation | 20° |
| distance | 2.7 |
| fx, fy | 549 |
| cx, cy | 256 |
| resolution | 512×512 |

### 학습 샘플링
```
32개 뷰 → 8개 랜덤 샘플링 → 4개 입력 + 4개 타겟
```

---

## 3. 합성 데이터셋 계획

### 3.1 Position Experiments (단순 모델)

**목적**: 빠른 PP 가설 검증

| 실험 | 위치 | PP | 상태 |
|------|------|-----|------|
| POS_C | 중앙 | 정확 | ✅ 완료 |
| POS_R | 오른쪽 | 불일치 | ✅ 완료 |
| POS_R_PP | 오른쪽 | 보정 | ✅ 완료 |
| POS_RANDOM | 랜덤 | 불일치 | 🔄 진행 |
| POS_RANDOM_PP | 랜덤 | 보정 | ⏳ 대기 |

**위치**: `/home/joon/data/synthetic/position_experiments/`

### 3.2 MAMMAL 32-View (실제 마우스 메시)

**목적**: FaceLift 동일 형식으로 정식 학습

| 실험 | 마우스 위치 | PP | 설명 |
|------|------------|-----|------|
| MAMMAL_CENTER | 원점 (중앙) | cx=cy=256 ✅ | 기준선 |
| MAMMAL_OFFSET | +0.3, +0.2 이동 | cx=cy=256 ❌ | PP 불일치 |
| MAMMAL_OFFSET_PP | +0.3, +0.2 이동 | 보정됨 ✅ | PP 가설 검증 |

**위치**: `/home/joon/data/synthetic/mammal_32view/`

---

## 4. 데이터 소스

### MAMMAL Fitting Results
```
/home/joon/dev/MAMMAL_mouse/results/fitting/
  markerless_mouse_1_nerf_v012345_kp22_20260126_025249/
  ├── obj/step_2_frame_*.obj    (2139개 메시)
  └── params/step_2_frame_*.pkl (포즈 파라미터)
```

### 메시 사양
- 정점: ~14,400
- 스케일: mm 단위 (body ~70mm)
- 프레임 간격: 5

---

## 5. 검증 계획

### Phase 1: Position Experiments
1. GS-LRM으로 각 실험 학습
2. Ghosting 정도 비교
3. PP 보정 효과 정량화

### Phase 2: MAMMAL 32-View
1. FaceLift 동일 설정으로 학습
2. 실제 마우스 데이터와 비교
3. 최종 PP 가설 검증

---

## 6. 파일 위치

| 파일 | 경로 |
|------|------|
| Position 렌더링 | `mouse_extensions/scripts/blender/render_position_experiments.py` |
| MAMMAL 렌더링 | `mouse_extensions/scripts/blender/render_mammal_32view.py` |
| 계획 문서 | `docs/analysis/MAMMAL_MESH_RENDERING_PLAN.md` |
| 매뉴얼 | `docs/analysis/SYNTHETIC_DATA_MANUAL.md` |

---

*FaceLift Synthetic Data | PP Hypothesis Validation*
