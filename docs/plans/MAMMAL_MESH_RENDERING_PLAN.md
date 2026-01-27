# MAMMAL Fitted Mesh를 활용한 Synthetic Data 생성 계획

**Version**: 1.0.0
**Created**: 2026-01-28
**Status**: Planning

---

## 1. 개요

### 목적
MAMMAL로 피팅된 실제 마우스 메시를 활용하여:
1. **Ground Truth 3D geometry** 확보 (실제 마우스 형상)
2. **정확한 카메라 파라미터** 사용 (캘리브레이션된 6개 뷰)
3. **위치/조명 변형 실험** 수행 (PP 가설 검증)

### 장점 vs 단순 Blender 모델

| 항목 | MAMMAL Mesh | Simple Blender Model |
|------|-------------|---------------------|
| **형상 정확도** | ✅ 실제 마우스 | ❌ 근사치 |
| **포즈 다양성** | ✅ 2139 프레임 | ❌ 고정 또는 수동 |
| **카메라 정확도** | ✅ 캘리브레이션됨 | ⚪ 설정값 |
| **Coverage 현실성** | ✅ 실제 ~5% | ⚪ 시뮬레이션 |

---

## 2. 데이터 구조 분석

### MAMMAL Fitting 결과 위치
```
/home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/
├── config.yaml          # 피팅 설정
├── obj/                 # 프레임별 메시 (2139개)
│   ├── step_2_frame_000000.obj
│   ├── step_2_frame_000005.obj
│   └── ...
├── params/              # MAMMAL 파라미터 (4279개)
│   ├── step_1_frame_*.pkl
│   └── step_2_frame_*.pkl
└── render/              # 렌더링 결과
    ├── debug/
    └── keypoints/
```

### OBJ 메시 사양

| 항목 | 값 |
|------|-----|
| **정점 수** | ~14,400 |
| **면 수** | ~14,400 |
| **스케일** | mm 단위 (71mm ≈ 7.1cm) |
| **좌표계** | MAMMAL world space |
| **프레임 간격** | 5 (interval=5) |
| **총 프레임** | 2139개 (0~10690, step 5) |

### PKL 파라미터 구조
```python
{
    "thetas": torch.Size([1, 140, 3]),      # 관절 각도 (140개 관절 × 3축)
    "trans": torch.Size([1, 3]),             # 전역 위치
    "scale": torch.Size([1, 1]),             # 스케일
    "rotation": torch.Size([1, 3]),          # 전역 회전
    "bone_lengths": torch.Size([1, 20]),     # 뼈 길이
    "chest_deformer": torch.Size([1, 1])     # 가슴 변형
}
```

### 카메라 파라미터 (camera_params.h5)
```python
{
    "intrinsic": (6, 3, 3),      # 6개 뷰 × 3×3 intrinsic matrix
    "rotation": (6, 3, 3),       # 6개 뷰 × 3×3 rotation matrix
    "translation": (6, 3)        # 6개 뷰 × 3D translation
}
```

---

## 3. 구현 계획

### Phase 1: 기본 렌더링 파이프라인

**목표**: MAMMAL 메시 + 원본 카메라로 렌더링

```
[OBJ 로드] → [좌표계 변환] → [카메라 설정] → [Blender 렌더링]
```

**스크립트**: `render_mammal_mesh.py`
```python
# 핵심 기능
1. OBJ 메시 import (Blender)
2. MAMMAL → Blender 좌표계 변환
3. H5 카메라 파라미터 적용
4. 6개 뷰 동시 렌더링
5. GS-LRM 형식 출력
```

### Phase 2: 위치 변형 실험

**목표**: PP 가설 검증을 위한 위치 변형

| 실험 | 설명 | PP 보정 |
|------|------|---------|
| **MAMMAL_C** | 원본 위치 (기준선) | 없음 |
| **MAMMAL_R** | 오른쪽 이동 (+X) | 없음 |
| **MAMMAL_R_PP** | 오른쪽 이동 + PP 보정 | ✅ |
| **MAMMAL_RU** | 오른쪽+위 이동 | 없음 |
| **MAMMAL_RU_PP** | 오른쪽+위 + PP 보정 | ✅ |

**위치 변형 방식**:
```python
# 메시 정점 전체에 offset 적용
vertices_new = vertices_original + offset_3d

# PP 보정 (GS-LRM용)
dcx = offset_x * fx / camera_distance
dcy = offset_y * fy / camera_distance
```

### Phase 3: 카메라 정규화

**문제**: MAMMAL 카메라 ≠ GS-LRM 기대값

| 항목 | MAMMAL (예상) | GS-LRM 기대 |
|------|--------------|-------------|
| fx, fy | ~1000+ | 549 |
| cx, cy | ~512 | 256 |
| distance | ~1000mm | 2.7 (normalized) |

**해결**: 정규화 변환 적용
```python
# 스케일 정규화
scale_factor = 549 / fx_original
fx_norm = 549
trans_norm = trans_original * scale_factor / expected_trans_ratio
```

---

## 4. 디렉토리 구조

```
/home/joon/data/synthetic/
├── position_experiments/       # 단순 모델 실험 (진행 중)
│   ├── POS_C/
│   ├── POS_R/
│   └── ...
└── mammal_experiments/         # MAMMAL 메시 실험 (계획)
    ├── MAMMAL_C/               # 원본 위치
    │   ├── sample_00000/
    │   │   ├── images/
    │   │   └── opencv_cameras.json
    │   └── ...
    ├── MAMMAL_R/               # 오른쪽 이동
    ├── MAMMAL_R_PP/            # 오른쪽 + PP 보정
    └── experiment_info.json
```

---

## 5. 구현 단계

### Step 1: 좌표계 분석 (1일)
- [ ] MAMMAL OBJ 좌표계 확인 (Y-up vs Z-up)
- [ ] 원본 카메라 extrinsics와 메시 위치 관계 분석
- [ ] Blender import 후 정렬 확인

### Step 2: 기본 렌더러 구현 (2일)
- [ ] `render_mammal_mesh.py` 작성
- [ ] H5 카메라 → Blender 카메라 변환
- [ ] 단일 프레임 6개 뷰 렌더링 테스트
- [ ] GS-LRM 형식 출력 검증

### Step 3: 카메라 정규화 (1일)
- [ ] fx=549, distance=2.7 정규화 로직
- [ ] 기존 FaceLift 정규화 코드 재사용
- [ ] 정규화 전/후 비교

### Step 4: 위치 변형 실험 (1일)
- [ ] 5가지 위치 변형 구현
- [ ] PP 보정 로직 통합
- [ ] 100 프레임 × 5 실험 = 500 샘플 생성

### Step 5: 검증 및 통합 (1일)
- [ ] GS-LRM 학습 테스트
- [ ] 기존 마우스 데이터와 비교
- [ ] 결과 문서화

---

## 6. 예상 출력

### 샘플당 출력 (1 프레임)
```
sample_XXXXX/
├── images/
│   ├── cam_000.png    # 512×512, RGBA
│   ├── cam_001.png
│   ├── cam_002.png
│   ├── cam_003.png
│   ├── cam_004.png
│   └── cam_005.png
├── opencv_cameras.json  # GS-LRM 형식
└── mesh_info.json       # 원본 OBJ 정보
```

### 데이터셋 규모

| 실험 | 프레임 수 | 뷰 | 이미지 | 용량 (예상) |
|------|----------|-----|--------|------------|
| MAMMAL_C | 100 | 6 | 600 | ~300MB |
| MAMMAL_R | 100 | 6 | 600 | ~300MB |
| MAMMAL_R_PP | 100 | 6 | 600 | ~300MB |
| MAMMAL_RU | 100 | 6 | 600 | ~300MB |
| MAMMAL_RU_PP | 100 | 6 | 600 | ~300MB |
| **Total** | 500 | - | 3000 | ~1.5GB |

---

## 7. 핵심 코드 스니펫

### OBJ 로드 및 변환
```python
import bpy
import numpy as np

def load_mammal_mesh(obj_path, scale=0.001):
    """MAMMAL OBJ 로드 (mm → m 변환)"""
    bpy.ops.import_scene.obj(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    obj.scale = (scale, scale, scale)
    return obj
```

### 카메라 설정
```python
def setup_camera_from_h5(cam_idx, intrinsic, rotation, translation):
    """H5 파라미터로 Blender 카메라 설정"""
    cam = bpy.data.cameras.new(f"Camera_{cam_idx}")
    cam_obj = bpy.data.objects.new(f"Camera_{cam_idx}", cam)
    
    # Intrinsics
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # Blender focal length (mm)
    cam.lens = fx * sensor_width / image_width
    
    # Extrinsics (OpenCV → Blender 변환 필요)
    # ...
    
    return cam_obj
```

### PP 보정
```python
def apply_pp_correction(cameras_json, offset_3d, fx, camera_distance):
    """위치 이동에 따른 PP 보정"""
    dcx = offset_3d[0] * fx / camera_distance
    dcy = offset_3d[1] * fx / camera_distance
    
    for frame in cameras_json["frames"]:
        frame["cx"] += dcx
        frame["cy"] += dcy
    
    return cameras_json
```

---

## 8. 의존성

### 필수
- Blender 4.0+ (설치됨: `~/blender-4.0.2-linux-x64/`)
- Python 3.10+ (Blender 내장)
- h5py (카메라 파라미터 읽기)

### 선택
- trimesh (메시 검증)
- open3d (점군 시각화)

---

## 9. 위험 요소 및 대응

| 위험 | 영향 | 대응 |
|------|------|------|
| 좌표계 불일치 | 렌더링 실패 | 단계별 시각적 검증 |
| 스케일 불일치 | 카메라 범위 벗어남 | mm→m 변환 + 정규화 |
| 메시 품질 이슈 | 아티팩트 | MAMMAL step_2 사용 (최종 결과) |
| 렌더링 시간 | 병목 | GPU 렌더링 + 병렬화 |

---

## 10. 다음 단계

1. **즉시**: 좌표계 분석 스크립트 작성
2. **1일차**: 단일 프레임 렌더링 PoC
3. **2일차**: 전체 파이프라인 구현
4. **3일차**: 위치 실험 + GS-LRM 테스트

---

*FaceLift Synthetic Data | MAMMAL Mesh Integration Plan*

---

## 11. FaceLift 원본 학습 설정 분석

### 데이터셋 구조 (확인됨)

```
sample_XXX/
├── images/
│   ├── cam_000.png ~ cam_031.png   # 32개 뷰
│   └── ...
└── opencv_cameras.json              # 32개 카메라 파라미터
```

### 카메라 배치 (32 views)

| 항목 | 값 |
|------|-----|
| **총 뷰 수** | 32 |
| **fx, fy** | 549 |
| **cx, cy** | 256 |
| **distance** | 2.7 |
| **resolution** | 512×512 |
| **azimuth** | 0°~360° (11.25° 간격) |
| **elevation** | 20° (고정) |

### 학습 시 샘플링 전략

```yaml
# configs/base.yaml
training:
  dataset:
    num_views: 8              # 32개 중 8개 랜덤 샘플링
    num_input_views: 4        # 8개 중 4개 = 입력 (또는 6개)
    target_has_input: true    # 타겟에 입력 포함
    maximize_view_overlap: true  # 인접 뷰 우선 선택
```

**학습 흐름**:
```
32개 뷰 중 8개 랜덤 샘플링
    │
    ├─ 4개 (또는 6개) → 입력 (Input Views)
    │       │
    │       └─ Transformer 인코딩 → Gaussian 예측
    │
    └─ 4개 (또는 2개) → 타겟 (Target Views)
            │
            └─ 렌더링 비교 → Loss 계산
```

### Validation 설정

```yaml
validation:
  enabled: true
  val_every: 5000
  # 동일한 8개 뷰 샘플링, 동일한 4+4 분할
```

---

## 12. MAMMAL 메시 기반 32-View 데이터셋 계획

### 목표

MAMMAL 메시를 FaceLift 원본과 동일한 형식으로 렌더링:

```
┌─────────────────────────────────────────────────────────────┐
│  MAMMAL Mesh (2139 포즈) + 32 Orbit Cameras = GS-LRM 호환   │
└─────────────────────────────────────────────────────────────┘
```

### 카메라 설정

```python
# 32개 카메라 생성 (FaceLift 동일)
cameras_32 = []
for i in range(32):
    azimuth = i * (360 / 32)  # 0°, 11.25°, 22.5°, ...
    elevation = 20
    
    camera = create_orbit_camera(
        azimuth=azimuth,
        elevation=elevation,
        distance=2.7,
        fx=549, fy=549,
        cx=256, cy=256
    )
    cameras_32.append(camera)
```

### 디렉토리 구조 (FaceLift 동일)

```
/home/joon/data/synthetic/mammal_32view/
├── MAMMAL_CENTER/                    # 중앙 배치
│   ├── sample_00000/                 # frame 0
│   │   ├── images/
│   │   │   ├── cam_000.png ~ cam_031.png
│   │   └── opencv_cameras.json
│   ├── sample_00001/                 # frame 5
│   └── ...
├── MAMMAL_OFFSET/                    # 오프셋 (PP 불일치)
├── MAMMAL_OFFSET_PP/                 # 오프셋 + PP 보정
└── data_train.txt                    # 학습 데이터 목록
```

### 데이터셋 규모

| 실험 | 프레임 | 뷰 | 이미지 | 용량 |
|------|--------|-----|--------|------|
| MAMMAL_CENTER | 500 | 32 | 16,000 | ~8GB |
| MAMMAL_OFFSET | 500 | 32 | 16,000 | ~8GB |
| MAMMAL_OFFSET_PP | 500 | 32 | 16,000 | ~8GB |
| **Total** | 1,500 | - | 48,000 | ~24GB |

### 학습 설정 (기존 FaceLift 재사용)

```yaml
# configs/mouse/MAMMAL_32view.yaml
training:
  dataset:
    dataset_path: "data/synthetic/mammal_32view/MAMMAL_CENTER/data_train.txt"
    num_views: 8
    num_input_views: 4
    maximize_view_overlap: true
    background_color: "white"
```

---

## 13. 렌더링 스크립트 계획

### `render_mammal_32view.py`

```python
"""
MAMMAL 메시를 32-view orbit 카메라로 렌더링
FaceLift 원본 데이터 형식과 동일하게 출력
"""

def main():
    # 1. MAMMAL 메시 로드
    mesh = load_mammal_obj(obj_path)
    
    # 2. 중심을 원점으로 이동 + 스케일 정규화
    mesh = normalize_mesh(mesh, target_size=0.25)
    
    # 3. 32개 orbit 카메라 생성
    cameras = create_orbit_cameras_32(
        distance=2.7,
        elevation=20,
        fx=549, cx=256
    )
    
    # 4. 각 카메라에서 렌더링
    for cam_idx, camera in enumerate(cameras):
        render_view(mesh, camera, f"cam_{cam_idx:03d}.png")
    
    # 5. opencv_cameras.json 저장
    save_cameras_json(cameras, output_dir)

# 위치 변형 실험
EXPERIMENTS = {
    "MAMMAL_CENTER": {"offset": (0, 0, 0), "pp_correct": False},
    "MAMMAL_OFFSET": {"offset": (0.3, 0.2, 0), "pp_correct": False},
    "MAMMAL_OFFSET_PP": {"offset": (0.3, 0.2, 0), "pp_correct": True},
}
```

---

## 14. 요약: 3가지 데이터셋

| 데이터셋 | 마우스 위치 | PP | 카메라 | 용도 |
|----------|------------|-----|--------|------|
| **Position Experiments** | 가상 모델 | 다양 | 6뷰 | 빠른 검증 |
| **MAMMAL 6-view** | 실제 메시 | 정확 | 6뷰 | 실제 형상 검증 |
| **MAMMAL 32-view** | 실제 메시 | 정확 | 32뷰 | **FaceLift 동일 학습** |

---

*Updated: 2026-01-28 | FaceLift Training Configuration Analysis*
