# Blender 합성 데이터셋 기반 Ghosting 가설 검증 계획

**Version**: 1.0
**Date**: 2026-01-28
**Status**: Planning
**목적**: 정밀 제어된 카메라 파라미터로 Ghosting 원인 가설 검증

---

## 1. 목표 및 동기

### 1.1 왜 합성 데이터인가?

실제 마우스 데이터의 불확실성 요소:
- 카메라 캘리브레이션 오차
- PP (Principal Point) 추정 오차
- 마스크 불완전성
- 움직임 블러

**합성 데이터의 장점**:
- 완벽한 Ground Truth 카메라 파라미터
- 정확한 3D 모델
- 제어된 조명 및 배경
- 재현 가능한 실험 조건

### 1.2 검증할 가설

| # | 가설 | 검증 방법 |
|---|------|----------|
| H1 | PP 오차가 Ghosting 유발 | PP=256 vs PP 의도적 오프셋 비교 |
| H2 | View-dependent overfitting | Input view 수 변화 (4 vs 6) |
| H3 | 카메라 거리 불일치 | distance=2.7 vs 가변 거리 |
| H4 | fx/fy 불일치 | fx=fy=549 vs fx≠fy |

---

## 2. 데이터 사양

### 2.1 GS-LRM 호환 형식

```
synthetic_dataset/
├── sample_001/
│   ├── images/
│   │   ├── cam_000.png  # 512x512 RGBA
│   │   ├── cam_001.png
│   │   ├── cam_002.png
│   │   ├── cam_003.png
│   │   ├── cam_004.png
│   │   └── cam_005.png
│   └── opencv_cameras.json
├── sample_002/
│   └── ...
└── data_list.txt
```

### 2.2 카메라 파라미터 기준

| 파라미터 | 기준값 | 비고 |
|----------|--------|------|
| **fx, fy** | 549 | Objaverse pretrained 기준 |
| **cx, cy** | 256 | 이미지 중앙 |
| **distance** | 2.7 | 정규화 거리 |
| **elevation** | 20° | 상부 조망 |
| **num_views** | 6 | 60° 간격 |
| **resolution** | 512x512 | 정사각형 |

### 2.3 카메라 배열

```
Top View (Z-up):

      Y
      ↑
  V4  │  V2      Azimuth angles:
   ○  │  ○       V0: 0° (Front)
      │          V1: 60°
──○───┼───○── X  V2: 120°
  V5  │  V1      V3: 180° (Back)
      │          V4: 240°
   ○  │  ○       V5: 300°
  V3  │  V0
```

---

## 3. Blender 렌더링 파이프라인

### 3.1 환경 설정

```bash
# Blender 4.0+ (Python 3.11)
# GPU 렌더링 지원

# 필요 패키지 (Blender 내장 Python)
# numpy, json (기본 포함)
```

### 3.2 렌더링 스크립트 구조

```
blender_synthetic/
├── render_gslrm.py      # 메인 렌더링 스크립트
├── camera_utils.py      # 카메라 유틸리티
├── scene_setup.py       # 씬 설정
├── batch_render.sh      # 배치 렌더링
└── validate_output.py   # 출력 검증
```

### 3.3 핵심 코드 (render_gslrm.py)

```python
import bpy
import numpy as np
import json
import math
import os

# === 카메라 설정 ===
def setup_camera(image_size=512, fov_deg=50.0):
    """GS-LRM 호환 카메라 생성"""
    
    # 기존 카메라 제거
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
            bpy.data.objects.remove(obj)
    
    # 새 카메라 생성
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    
    # FOV 50° → focal length (mm)
    sensor_width = 36.0
    focal_mm = (sensor_width / 2) / math.tan(math.radians(fov_deg) / 2)
    cam_data.lens = focal_mm
    cam_data.sensor_width = sensor_width
    
    return cam_obj, cam_data

# === 카메라 위치 설정 ===
def position_camera(cam_obj, azimuth_deg, elevation_deg=20, distance=2.7):
    """구면 좌표로 카메라 배치"""
    
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    
    x = distance * math.cos(el) * math.sin(az)
    y = distance * math.cos(el) * math.cos(az)
    z = distance * math.sin(el)
    
    cam_obj.location = (x, y, z)
    
    # 원점 바라보기
    direction = mathutils.Vector((x, y, z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

# === 카메라 파라미터 추출 ===
def extract_camera_params(cam_obj, cam_data, image_size=512):
    """Blender 카메라 → GS-LRM 형식"""
    
    # Intrinsics
    sensor_w = cam_data.sensor_width
    focal_mm = cam_data.lens
    fx = fy = (focal_mm / sensor_w) * image_size
    cx = cy = image_size / 2.0
    
    # Extrinsics (w2c)
    c2w = np.array(cam_obj.matrix_world)
    w2c = np.linalg.inv(c2w)
    
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w": image_size,
        "h": image_size,
        "w2c": w2c.tolist()
    }

# === 메인 렌더링 ===
def render_dataset(output_dir, num_views=6, elevation=20, distance=2.7):
    """전체 데이터셋 렌더링"""
    
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    cam_obj, cam_data = setup_camera()
    frames = []
    
    for v in range(num_views):
        azimuth = v * (360 / num_views)
        
        position_camera(cam_obj, azimuth, elevation, distance)
        
        # 렌더링
        filepath = f"{output_dir}/images/cam_{v:03d}.png"
        bpy.context.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        
        # 파라미터 저장
        params = extract_camera_params(cam_obj, cam_data)
        params["file_path"] = f"images/cam_{v:03d}.png"
        params["view_id"] = v
        frames.append(params)
    
    # JSON 저장
    with open(f"{output_dir}/opencv_cameras.json", "w") as f:
        json.dump({"frames": frames}, f, indent=2)

if __name__ == "__main__":
    render_dataset("/tmp/synthetic_test")
```

---

## 4. 실험 설계

### 4.1 실험 매트릭스

| 실험 | 변수 | 기준값 | 변형값 | 목표 |
|------|------|--------|--------|------|
| **Exp1** | PP | (256, 256) | (230, 280) | H1 검증 |
| **Exp2** | Input views | 4 | 6 | H2 검증 |
| **Exp3** | Distance | 2.7 | 2.0~3.5 | H3 검증 |
| **Exp4** | fx/fy | 549/549 | 549/520 | H4 검증 |
| **Exp5** | 기준선 | All correct | - | Baseline |

### 4.2 3D 모델 선정

**후보 모델** (복잡도 순):
1. **Stanford Bunny** - 단순, 대칭
2. **XYZ Dragon** - 중간 복잡도
3. **Simple Mouse Mesh** - 마우스 유사 형상
4. **Objaverse 샘플** - 다양한 형상

**권장**: Simple Mouse Mesh (실제 태스크와 유사)

### 4.3 평가 지표

| 지표 | 설명 | Ghosting 관련성 |
|------|------|----------------|
| **PSNR** | 픽셀 정확도 | 간접 (전체 품질) |
| **SSIM** | 구조적 유사도 | 중간 |
| **LPIPS** | 지각적 품질 | 높음 (ghosting 감지) |
| **Ghost Count** | 수동 분류 | 직접 |
| **Turntable IoU** | Novel view 마스크 정확도 | 높음 |

---

## 5. 구현 단계

### Phase 1: 환경 구축 (1일)

- [ ] Blender 4.0 설치 (gpu03)
- [ ] Python 스크립트 환경 설정
- [ ] 테스트 렌더링 (1개 모델)

### Phase 2: 기준 데이터셋 생성 (2일)

- [ ] 3D 모델 준비 (3-5개)
- [ ] 정확한 파라미터로 렌더링
- [ ] 데이터 검증 (intrinsics, extrinsics 확인)

### Phase 3: 가설 검증 실험 (3일)

- [ ] Exp1-5 데이터셋 생성
- [ ] GS-LRM Finetuning 실행
- [ ] Turntable 시각화 비교

### Phase 4: 분석 및 결론 (1일)

- [ ] 정량적 지표 비교
- [ ] Ghosting 패턴 분석
- [ ] 권장사항 도출

---

## 6. 기대 결과

### 6.1 가설별 예상 결과

| 가설 | 예상 결과 |
|------|----------|
| **H1 (PP 오차)** | PP 오프셋 증가 → Ghosting 증가 (선형 관계) |
| **H2 (Input views)** | 6 views > 4 views (novel view coverage 개선) |
| **H3 (거리 불일치)** | 가변 거리 → Type B 고스트 증가 |
| **H4 (fx/fy 불일치)** | fx≠fy → 비등방적 왜곡 |

### 6.2 성공 기준

1. **정확한 파라미터** 사용 시 Ghosting 없음
2. **각 가설**에서 예상 패턴 관찰
3. **실제 마우스 데이터**에 적용 가능한 가이드라인 도출

---

## 7. 파일 위치

| 항목 | 위치 |
|------|------|
| 렌더링 스크립트 | `/home/joon/dev/FaceLift/mouse_extensions/scripts/blender/` |
| 합성 데이터셋 | `/home/joon/data/synthetic/gslrm_validation/` |
| 실험 config | `/home/joon/dev/FaceLift/configs/experiments/synthetic/` |
| 결과 분석 | `/home/joon/dev/FaceLift/docs/experiments/synthetic_results/` |

---

## 8. 참고 자료

### 8.1 관련 문서

- `docs/analysis/GHOSTING_ANALYSIS_REPORT.md` - 가설 배경
- `docs/theory/camera/coordinate_transformation_guide.md` - 좌표계
- `docs/datasets/CAMERA_CONFIG.md` - 카메라 설정

### 8.2 외부 참조

- Objaverse rendering: https://github.com/allenai/objaverse-rendering
- Blender Python API: https://docs.blender.org/api/current/
- GS-LRM paper: arXiv:2404.xxxx

---

*Blender Synthetic Dataset Validation Plan v1.0 | 2026-01-28*
