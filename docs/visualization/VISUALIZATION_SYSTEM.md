# FaceLift 시각화 시스템 가이드

**Version**: 1.0.0  
**Updated**: 2026-01-27  
**Author**: Claude Code

---

## 1. 시스템 개요

### 1.1 목적

FaceLift 시각화 시스템은 3D Gaussian Splatting 학습 과정을 모니터링하고 디버깅하기 위한 통합 시각화 도구입니다.

**핵심 목표:**
1. **학습 품질 모니터링**: GT vs Rendered 비교로 수렴 상태 확인
2. **3D 재구성 검증**: 다양한 각도에서 Gaussian 렌더링 결과 확인
3. **디버깅 지원**: 마스크, 알파, 에러 분포 시각화

### 1.2 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        시각화 시스템 구조                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Training Loop]                    [Validation Loop]               │
│       │                                   │                         │
│       ▼                                   ▼                         │
│  ┌─────────────┐                   ┌─────────────────┐              │
│  │ gslrm.py    │                   │ validator.py    │              │
│  │ _create_    │                   │ ValidationRunner│              │
│  │ visual()    │                   └────────┬────────┘              │
│  └──────┬──────┘                            │                       │
│         │                                   │                       │
│         ▼                                   ▼                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │           visualization_extensions.py                │           │
│  │  - create_training_visual()                         │           │
│  │  - create_validation_visual()                       │           │
│  │  - create_error_heatmap()                           │           │
│  │  - create_mask_overlay()                            │           │
│  └──────────────────────────────────────────────────────┘           │
│         │                                   │                       │
│         ▼                                   ▼                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              turntable_config.py                     │           │
│  │  - MOUSE_CAMERA_ORDER = [0,4,2,1,3,5]               │           │
│  │  - interpolate_camera_extrinsics()                   │           │
│  │  - create_turntable_trajectory()                     │           │
│  └──────────────────────────────────────────────────────┘           │
│         │                                   │                       │
│         ▼                                   ▼                       │
│  ┌──────────────────────────────────────────────────────┐           │
│  │            gaussians_renderer.py                     │           │
│  │  - render_turntable()         : 360도 회전 렌더링     │           │
│  │  - render_dataset_trajectory(): 카메라 경로 보간      │           │
│  └──────────────────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 시각화 타입별 상세

### 2.1 Supervision 이미지 (GT vs Rendered)

**파일명**: `supervision_{uid1}_{uid2}.jpg`  
**생성 위치**: `gslrm.py::_create_visual()` → `visualization_extensions.py::create_training_visual()`

**구조 (mask_mode != "none" 시):**
```
┌─────────────────────────────────────────┐
│ Row 0: GT Images           (6 views)    │
│ Row 1: Rendered Images     (6 views)    │
│ Row 2: GT + Mask Overlay   (6 views)    │
│ Row 3: Render + Pred Mask  (6 views)    │
│ Row 4: Error Heatmap       (6 views)    │
├─────────────────────────────────────────┤
│ (Batch 2개면 위 패턴이 2번 반복)          │
└─────────────────────────────────────────┘
```

**카메라 순서**: 데이터 로딩 순서 `[0,1,2,3,4,5]`  
**용도**: 학습 수렴 상태, 마스크 품질, per-view 에러 확인

### 2.2 Turntable 비디오/이미지

**파일명**: `turntable_{uid}.mp4`, `turntable_{uid}.jpg`  
**생성 위치**: `validator.py::_create_turntable()`

**원리:**
```python
# 1. 카메라 경로 생성 (SLERP 보간)
trajectory = create_turntable_trajectory(
    camera_c2ws,                    # 6개 카메라 extrinsics
    camera_order=MOUSE_CAMERA_ORDER,  # [0,4,2,1,3,5] CCW
    interpolation_steps=6            # 카메라 쌍 사이 6프레임
)
# 결과: 36개 보간된 카메라 위치 (6 카메라 x 6 steps)

# 2. 각 위치에서 Gaussian 렌더링
for c2w in trajectory:
    frame = render_gaussian(gaussians, c2w, fxfycxcy)
    
# 3. 비디오/그리드 생성
video: 144 frames @ 15fps -> 9.6초 360도 회전
grid:  6x6 = 36 frames 정적 이미지
```

**카메라 순서**: `MOUSE_CAMERA_ORDER = [0,4,2,1,3,5]` (CCW by azimuth)  
**용도**: 3D 재구성 품질의 전방위 확인

### 2.3 Orbit Turntable

**파일명**: `turntable_orbit_{uid}.mp4`  
**생성 위치**: `validator.py::_save_orbit_turntable()`

**원리:**
```python
# 표준 360도 원형 궤도 (데이터셋 카메라와 무관)
render_turntable(
    gaussians,
    rendering_resolution=512,
    num_views=120,
    elevation=20,        # 위에서 20도 내려다봄
    radius=2.7,          # 카메라 거리 (normalized)
    center=gaussian_center  # Gaussian 중심점
)
```

**용도**: 데이터셋 카메라 배치와 무관한 표준 시점에서 품질 확인

### 2.4 Alpha Comparison

**파일명**: `alpha_comparison_{uid}.jpg`  
**생성 위치**: `validator.py::_save_alpha_comparison()`

**구조:**
```
┌─────────────────────────────────────────┐
│ Row 0: GT Alpha Mask       (6 views)    │
│ Row 1: Rendered Alpha      (6 views)    │
│ Row 2: Diff (FP=Red, FN=Blue)           │
└─────────────────────────────────────────┘
```

**용도**: Alpha supervision 품질 확인, mask spreading 감지

---

## 3. 카메라 순서 체계

### 3.1 두 가지 순서 체계

| 체계 | 순서 | 사용처 |
|------|------|--------|
| **Data Order** | `[0,1,2,3,4,5]` | Supervision, 데이터 로딩 |
| **Physical Order** | `[0,4,2,1,3,5]` | Turntable, 360도 시각화 |

### 3.2 Physical Order (MOUSE_CAMERA_ORDER) 유래

```
카메라 물리적 배치 (위에서 본 모습):
              cam2 (3.9도)
                 |
    cam4 (-54도) |   cam1 (56도)
           \     |     /
            \    |    /
             \   |   /
    cam0 ------- * ------- cam3 (101도)
   (-123도)      |
                 |
             cam5 (154도)

CCW 순서 (azimuth 기준): 0 -> 4 -> 2 -> 1 -> 3 -> 5 -> 0
```

### 3.3 순서 변환

```python
# Data -> Physical 변환
MOUSE_CAMERA_ORDER = [0, 4, 2, 1, 3, 5]
physical_views = data_views[MOUSE_CAMERA_ORDER]

# Physical -> Data 변환
inverse_order = [0, 3, 2, 4, 1, 5]  # argsort(MOUSE_CAMERA_ORDER)
data_views = physical_views[inverse_order]
```

---

## 4. Config 설정 가이드

### 4.1 Visualization Config 구조

```yaml
visualization:
  turntable:
    # 기본 설정
    camera_order: [0, 4, 2, 1, 3, 5]  # CCW physical order
    fps: 15                           # 비디오 프레임 레이트
    interpolation_steps: 6            # 카메라 간 보간 프레임
    
    # 해상도
    grid_rows: 6
    grid_cols: 6
    video_views: 144    # 비디오 총 프레임 (smooth)
    grid_views: 36      # 그리드 이미지 프레임
    
    # 저장 옵션
    save_video: true              # turntable_*.mp4
    save_orbit_turntable: true    # turntable_orbit_*.mp4
    smooth_trajectory: true       # SLERP 보간 사용
    
    # Orbit 설정
    orbit_views: 120
    elevation: 20
    radius: 2.7
```

### 4.2 Training Visualization

```yaml
training:
  logging:
    vis_every: 100     # N step마다 시각화 저장
    print_every: 10    # N step마다 loss 출력
```

### 4.3 Validation Visualization

```yaml
validation:
  enabled: true
  val_every: 200       # N step마다 validation 실행
```

---

## 5. 파일 생성 매핑

### 5.1 Training 시각화 (vis_every 마다)

| 파일명 패턴 | 생성 함수 | 내용 |
|------------|----------|------|
| `supervision_{uid1}_{uid2}.jpg` | `_create_visual()` | GT vs Rendered 비교 |
| `input_{uid1}_{uid2}.jpg` | `save_visualization_outputs()` | 입력 뷰 연결 |
| `alpha_comparison_*.jpg` | `save_visualization_outputs()` | Alpha 비교 |

### 5.2 Validation 시각화 (val_every 마다)

| 파일명 패턴 | 생성 함수 | 내용 |
|------------|----------|------|
| `turntable_{uid}.jpg` | `_save_grid()` | 6x6 그리드 |
| `turntable_{uid}.mp4` | `_create_turntable()` | 360도 비디오 |
| `turntable_orbit_{uid}.mp4` | `_save_orbit_turntable()` | 표준 궤도 비디오 |
| `turntable_with_input_{uid}.mp4` | `_save_with_input()` | 입력+렌더링 |
| `aligned_gs_opacity_depth_{uid}.jpg` | `save_visualization_outputs()` | Gaussian 분석 |
| `gaussians_{uid}.ply` | `_save_gaussian_ply()` | 3D 모델 |
| `input_{uid}.jpg` | `_save_input_image()` | 입력 이미지 |
| `gt_vs_pred.png` | `_save_comparison()` | 비교 이미지 |

---

## 6. 코드 위치 참조

| 기능 | 파일 | 주요 함수 |
|------|------|----------|
| Training 시각화 | `gslrm/model/gslrm.py` | `_create_visual()`, `save_visualization_outputs()` |
| Validation 시각화 | `mouse_extensions/validation/validator.py` | `ValidationRunner` 클래스 |
| 시각화 유틸 | `mouse_extensions/model/visualization_extensions.py` | `create_training_visual()` |
| 카메라 설정 | `mouse_extensions/visualization/turntable_config.py` | `MOUSE_CAMERA_ORDER` |
| 렌더링 | `gslrm/model/gaussians_renderer.py` | `render_turntable()` |
| 비디오 저장 | `mouse_extensions/validation/validator.py` | `_safe_video_save()` |

---

*FaceLift Visualization System Guide v1.0 | 2026-01-27*
