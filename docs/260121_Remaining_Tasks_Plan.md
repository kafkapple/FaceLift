# 260121 Remaining Tasks Implementation Plan

> **Date**: 2026-01-21
> **Status**: Planning
> **Purpose**: 남은 리팩토링 작업 체계적 실행 계획

---

## Executive Summary

| Task | Priority | Complexity | Status |
|------|----------|------------|--------|
| Background Loss 삭제 | P0 | Low | Pending |
| Task 3-2: Turntable Config | P2 | Medium | Pending |
| Task 3-3: Camera Trajectory | P2 | Medium | Pending |
| Task 4: Camera Exclusion | P3 | Low | Pending |
| Task 6-1: 3D→2D Projection Doc | P2 | Low | Pending |
| Task 6-2: Pipeline Script | P2 | Medium | Pending |

---

## 1. Background Loss 삭제

### 1.1 현재 상태

Background loss는 현재 코드에 구현되어 있으나 **모든 실험에서 weight=0.0**으로 비활성화됨.

**관련 파일:**
- `gslrm/model/gslrm.py`: 라인 399, 464-474, 612, 1021, 1031
- `mouse_extensions/model/gslrm_patches.py`: 라인 203
- `configs/mouse/*.yaml`: `background_loss_weight: 0.0`

### 1.2 삭제 계획

1. `gslrm.py`에서 background_loss 관련 코드 제거
2. `gslrm_patches.py`에서 config 참조 제거
3. Config 파일에서 `background_loss_weight` 제거

---

## 2. Task 3-2: Turntable Configurable Module

### 2.1 현재 상태

**`gaussians_renderer.py`:**
```python
def get_turntable_cameras(
    hfov=50,          # 하드코딩된 기본값
    num_views=8,      # 하드코딩된 기본값
    w=384, h=384,     # 하드코딩된 기본값
    radius=2.7,       # 하드코딩된 기본값
    elevation=20,     # 하드코딩된 기본값
    ...
):
```

**`gslrm.py:1786`:**
```python
turntable_views = 64  # 하드코딩
turntable_image = render_turntable(..., num_views=turntable_views)
```

**Config:**
```yaml
visualization:
  turntable_fps: 15  # 유일하게 설정 가능한 파라미터
```

### 2.2 구현 계획

**새 Config 구조:**
```yaml
visualization:
  turntable:
    enabled: true
    num_views: 64          # 총 뷰 수
    grid_rows: 8           # WandB용 그리드 행
    grid_cols: 8           # WandB용 그리드 열
    hfov: 50               # Horizontal FOV
    radius: 2.7            # 카메라-객체 거리
    elevation: 20          # 고도각 (degrees)
    fps: 15                # 비디오 FPS
    resolution: 384        # 렌더링 해상도
    center_method: "origin"  # "origin", "camera_mean", "triangulation"
```

### 2.3 수정 파일

1. `gaussians_renderer.py`: `get_turntable_cameras()` 파라미터 전달
2. `gslrm.py`: Config에서 turntable 파라미터 읽기
3. `configs/mouse/*.yaml`: 새 turntable 설정 추가

---

## 3. Task 3-3: Camera Trajectory Mode

### 3.1 목표

학습에 사용된 카메라 경로를 따라 이동하는 시각화 생성.

### 3.2 구현 계획

**새 함수:**
```python
def get_trajectory_cameras(
    input_cameras,      # 학습 카메라들
    num_interpolations, # 카메라 간 보간 프레임 수
    method="slerp",     # "slerp" (rotation) + "lerp" (translation)
):
    """
    카메라 간 보간으로 부드러운 궤적 생성.
    """
```

**Config 추가:**
```yaml
visualization:
  trajectory:
    enabled: true
    interpolation_frames: 10  # 카메라 간 보간 프레임
    method: "slerp"           # "slerp", "linear"
    save_video: true
```

---

## 4. Task 4: Camera Exclusion Training

### 4.1 목표

특정 카메라를 학습에서 제외하고 해당 뷰를 Novel View로 평가.

### 4.2 구현 계획

**Config:**
```yaml
training:
  dataset:
    exclude_camera_indices: [3]     # 제외할 카메라 인덱스
    # 또는
    include_camera_indices: [0, 1, 2, 4, 5]  # 포함할 카메라만 명시
```

**수정 위치:**
- `gslrm/data/dataset.py`: 뷰 필터링 로직 추가

---

## 5. Task 6-1: 3D→2D Projection Documentation

### 5.1 Projection 이론

**World → Camera:**
```
P_cam = R @ P_world + t
      = [R | t] @ [P_world; 1]
```

**Camera → Pixel:**
```
[u]   [fx  0  cx] [X/Z]
[v] = [0  fy  cy] [Y/Z]
[1]   [0   0   1] [ 1 ]

u = fx * X/Z + cx
v = fy * Y/Z + cy
```

### 5.2 GS-LRM 구현 위치

- `gaussians_renderer.py:render_opencv_cam()`: 실제 projection
- `gaussians_renderer.py:get_turntable_cameras()`: 카메라 파라미터 생성

---

## 6. Task 6-2: End-to-End Pipeline Script

### 6.1 파이프라인 개요

```
Single Image → MVDiffusion → 6 Views → GS-LRM → 3D Gaussians → Outputs
                                                      │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                        Turntable Video   Grid Image
```

### 6.2 스크립트 요구사항

```bash
python run_pipeline.py \
    --input image.png \
    --output_dir results/$(date +%y%m%d_%H%M%S) \
    --turntable_views 64 \
    --save_video \
    --save_grid
```

---

## Implementation Order

1. **Phase 1: Cleanup** (즉시)
   - [ ] Background loss 삭제

2. **Phase 2: Configuration** (오늘)
   - [ ] Turntable config 파라미터화
   - [ ] Camera exclusion config

3. **Phase 3: Features** (이번 주)
   - [ ] Camera trajectory 모드
   - [ ] Pipeline script

4. **Phase 4: Documentation**
   - [ ] 3D→2D projection 문서

---

*Created: 2026-01-21*
