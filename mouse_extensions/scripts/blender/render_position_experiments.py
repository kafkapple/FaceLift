#!/usr/bin/env python3
"""
render_position_experiments.py

저 coverage 비정형 물체의 위치 변형 실험용 렌더링 스크립트
물체가 이미지 중앙이 아닌 다양한 위치에 있을 때의 영향 측정

Usage:
    blender --background model.blend -P render_position_experiments.py -- \\
        --output_dir /path/to/output \\
        --experiment POS_C \\
        --num_samples 100

Experiments:
    POS_C     : 중앙 위치 (기준선)
    POS_R     : 오른쪽 오프셋 (PP 불일치)
    POS_R_PP  : 오른쪽 오프셋 + PP 보정
    POS_U     : 위쪽 오프셋
    POS_RU    : 대각선 오프셋
    POS_RU_PP : 대각선 오프셋 + PP 보정
    POS_RANDOM: 랜덤 위치 (실제 마우스 시뮬레이션)
"""

import bpy
import numpy as np
import json
import math
import os
import sys
import argparse
import random
from mathutils import Vector


# =============================================================================
# 실험 설정
# =============================================================================

# PP 보정 계산: offset_3d * (fx / camera_distance)
# fx=549, distance=2.7 → scale = 203.3
PP_SCALE = 549 / 2.7  # ≈ 203

EXPERIMENTS = {
    # 기준선: 중앙 위치
    "POS_C": {
        "offset": (0, 0, 0),
        "pp_correct": False,
        "desc": "Center (baseline)"
    },
    
    # 오른쪽 오프셋: PP 불일치
    "POS_R": {
        "offset": (0.3, 0, 0),
        "pp_correct": False,
        "desc": "Right offset, PP mismatch"
    },
    
    # 오른쪽 오프셋: PP 보정
    "POS_R_PP": {
        "offset": (0.3, 0, 0),
        "pp_correct": True,
        "desc": "Right offset, PP corrected"
    },
    
    # 위쪽 오프셋
    "POS_U": {
        "offset": (0, 0.2, 0),
        "pp_correct": False,
        "desc": "Up offset, PP mismatch"
    },
    
    # 위쪽 오프셋 + PP 보정
    "POS_U_PP": {
        "offset": (0, 0.2, 0),
        "pp_correct": True,
        "desc": "Up offset, PP corrected"
    },
    
    # 대각선 오프셋: 심한 PP 불일치
    "POS_RU": {
        "offset": (0.3, 0.2, 0),
        "pp_correct": False,
        "desc": "Diagonal offset, severe PP mismatch"
    },
    
    # 대각선 오프셋 + PP 보정
    "POS_RU_PP": {
        "offset": (0.3, 0.2, 0),
        "pp_correct": True,
        "desc": "Diagonal offset, PP corrected"
    },
    
    # 랜덤 위치: 실제 마우스 데이터 시뮬레이션
    "POS_RANDOM": {
        "offset": "random",
        "pp_correct": False,
        "desc": "Random position, simulates real mouse data"
    },
    
    # 랜덤 위치 + PP 보정
    "POS_RANDOM_PP": {
        "offset": "random",
        "pp_correct": True,
        "desc": "Random position, PP corrected"
    },
}

CONFIG = {
    "resolution": 512,
    "fov_deg": 50.0,
    "camera_distance": 2.7,
    "elevation_deg": 20,
    "num_views": 6,
    "object_size": 0.25,
    "render_samples": 64,
    "max_random_offset": 0.4,
}


# =============================================================================
# 유틸리티 함수
# =============================================================================

def compute_focal_px(fov_deg, resolution):
    """FOV → 픽셀 초점거리"""
    return resolution / (2 * math.tan(math.radians(fov_deg) / 2))


def compute_pp_offset(offset_3d, fx, camera_distance):
    """3D 오프셋 → PP 오프셋 (pixels)"""
    dx, dy, dz = offset_3d
    dcx = dx * fx / camera_distance
    dcy = dy * fx / camera_distance
    return dcx, dcy


def get_random_offset(max_offset):
    """랜덤 XY 오프셋 생성"""
    dx = random.uniform(-max_offset, max_offset)
    dy = random.uniform(-max_offset, max_offset)
    return (dx, dy, 0)


def spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
    """구면 → 직교 좌표"""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = distance * math.cos(el) * math.sin(az)
    y = distance * math.cos(el) * math.cos(az)
    z = distance * math.sin(el)
    return x, y, z


# =============================================================================
# Blender 설정
# =============================================================================

def setup_render_engine():
    """렌더 엔진 설정"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = CONFIG["render_samples"]
    scene.cycles.use_denoising = True
    
    # GPU 사용 시도
    try:
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        for device in prefs.devices:
            device.use = True
    except:
        print("GPU not available, using CPU")
    
    scene.render.resolution_x = CONFIG["resolution"]
    scene.render.resolution_y = CONFIG["resolution"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True


def setup_lighting():
    """3점 조명"""
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
    
    # Back light
    back = bpy.data.lights.new("Back", "SUN")
    back.energy = 1.5
    back_obj = bpy.data.objects.new("Back", back)
    bpy.context.collection.objects.link(back_obj)
    back_obj.rotation_euler = (math.radians(-30), 0, math.radians(180))


def setup_camera():
    """카메라 생성"""
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    
    cam_data = bpy.data.cameras.new("RenderCam")
    cam_obj = bpy.data.objects.new("RenderCam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # 초점거리
    sensor_width = 36.0
    fov_rad = math.radians(CONFIG["fov_deg"])
    focal_mm = (sensor_width / 2) / math.tan(fov_rad / 2)
    cam_data.lens = focal_mm
    cam_data.sensor_width = sensor_width
    
    return cam_obj, cam_data


def position_camera(cam_obj, azimuth_deg, elevation_deg, distance, look_at=(0,0,0)):
    """카메라 위치 및 방향 설정"""
    # 카메라 위치 (look_at 기준 구면 좌표)
    x, y, z = spherical_to_cartesian(azimuth_deg, elevation_deg, distance)
    cam_obj.location = (x + look_at[0], y + look_at[1], z + look_at[2])
    
    # look_at 방향
    direction = Vector(look_at) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def normalize_model(obj, target_size):
    """모델 크기 정규화"""
    # 현재 크기
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_c = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_c = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    current_size = max(max_c - min_c)
    
    # 스케일링
    if current_size > 0:
        scale = target_size / current_size
        obj.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # 원점으로 이동
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")
    obj.location = (0, 0, 0)
    
    print(f"Model normalized: {current_size:.3f} -> {target_size:.3f}")


# =============================================================================
# 카메라 파라미터 추출
# =============================================================================

def extract_camera_params(cam_obj, view_id, cx, cy):
    """GS-LRM 형식 카메라 파라미터"""
    
    fx = fy = compute_focal_px(CONFIG["fov_deg"], CONFIG["resolution"])
    
    c2w = np.array(cam_obj.matrix_world)
    w2c = np.linalg.inv(c2w)
    
    return {
        "w": CONFIG["resolution"],
        "h": CONFIG["resolution"],
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w2c": w2c.tolist(),
        "c2w": c2w.tolist(),
        "file_path": f"images/cam_{view_id:03d}.png",
        "view_id": view_id,
    }


# =============================================================================
# 렌더링
# =============================================================================

def render_sample(sample_id, experiment_name, output_dir, model_obj, cam_obj):
    """단일 샘플 렌더링 (6뷰)"""
    
    exp = EXPERIMENTS[experiment_name]
    
    # 물체 위치
    if exp["offset"] == "random":
        offset = get_random_offset(CONFIG["max_random_offset"])
    else:
        offset = exp["offset"]
    
    model_obj.location = offset
    bpy.context.view_layer.update()
    
    # PP 계산
    fx = compute_focal_px(CONFIG["fov_deg"], CONFIG["resolution"])
    base_pp = CONFIG["resolution"] / 2  # 256
    
    if exp["pp_correct"]:
        dcx, dcy = compute_pp_offset(offset, fx, CONFIG["camera_distance"])
        cx = base_pp + dcx
        cy = base_pp + dcy
    else:
        cx = base_pp
        cy = base_pp
    
    # 샘플 디렉토리
    sample_dir = os.path.join(output_dir, f"sample_{sample_id:05d}")
    os.makedirs(os.path.join(sample_dir, "images"), exist_ok=True)
    
    frames = []
    
    # 6뷰 렌더링
    for view_id in range(CONFIG["num_views"]):
        azimuth = view_id * (360 / CONFIG["num_views"])
        
        # 카메라 배치 (물체 중심 기준)
        position_camera(
            cam_obj, azimuth,
            CONFIG["elevation_deg"],
            CONFIG["camera_distance"],
            look_at=offset
        )
        bpy.context.view_layer.update()
        
        # 렌더링
        filepath = os.path.join(sample_dir, "images", f"cam_{view_id:03d}.png")
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        
        # 파라미터
        params = extract_camera_params(cam_obj, view_id, cx, cy)
        params["azimuth_deg"] = azimuth
        params["elevation_deg"] = CONFIG["elevation_deg"]
        params["object_offset"] = list(offset)
        frames.append(params)
    
    # JSON 저장
    metadata = {
        "sample_id": sample_id,
        "experiment": experiment_name,
        "experiment_desc": exp["desc"],
        "object_offset": list(offset),
        "pp_corrected": exp["pp_correct"],
        "cx": cx,
        "cy": cy,
        "fx": frames[0]["fx"],
        "camera_distance": CONFIG["camera_distance"],
    }
    
    with open(os.path.join(sample_dir, "opencv_cameras.json"), "w") as f:
        json.dump({"frames": frames, "_metadata": metadata}, f, indent=2)
    
    return sample_dir


def render_experiment(experiment_name, output_dir, num_samples):
    """실험 전체 렌더링"""
    
    print("\n" + "="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Description: {EXPERIMENTS[experiment_name]['desc']}")
    print(f"Samples: {num_samples}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # 설정
    setup_render_engine()
    setup_lighting()
    cam_obj, cam_data = setup_camera()
    
    # 모델 찾기
    model_obj = None
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            model_obj = obj
            break
    
    if not model_obj:
        raise RuntimeError("No mesh object in scene!")
    
    # 모델 정규화
    normalize_model(model_obj, CONFIG["object_size"])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 렌더링
    sample_dirs = []
    for i in range(num_samples):
        print(f"[{i+1}/{num_samples}] Rendering...")
        d = render_sample(i, experiment_name, output_dir, model_obj, cam_obj)
        sample_dirs.append(d)
    
    # 데이터 목록
    with open(os.path.join(output_dir, "data_list.txt"), "w") as f:
        for d in sample_dirs:
            f.write(os.path.basename(d) + "\n")
    
    # 메타데이터
    with open(os.path.join(output_dir, "experiment_info.json"), "w") as f:
        json.dump({
            "experiment": experiment_name,
            "description": EXPERIMENTS[experiment_name]['desc'],
            "num_samples": num_samples,
            "config": CONFIG,
        }, f, indent=2)
    
    print("\n" + "="*60)
    print(f"Complete: {output_dir}")
    print("="*60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment", default="POS_C", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    render_experiment(args.experiment, args.output_dir, args.num_samples)
