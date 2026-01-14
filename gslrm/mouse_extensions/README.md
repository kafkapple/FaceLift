# Mouse Extensions for FaceLift/GS-LRM

Original repo: https://github.com/weijielyu/FaceLift

## Overview

이 모듈은 FaceLift/GS-LRM을 Mouse 데이터에 적용하기 위한 확장 기능을 제공합니다.
원본 코드 수정을 최소화하면서 다음 기능을 추가합니다:

## Module Structure

```
mouse_extensions/
├── __init__.py           # 메인 진입점 (v1.1.0)
├── preprocessing.py      # 카메라 전처리 (NEW)
├── alpha_renderer.py     # Alpha rendering 확장
├── loss_extensions.py    # Mask/Loss 확장
├── visualization.py      # 시각화 확장
├── logging_utils.py      # WandB 로깅 확장
├── apply_patches.py      # 원본 코드 패치
├── PATCHES.md            # 패치 문서
└── README.md
```

## Features

### 1. Preprocessing (preprocessing.py)
- `normalize_camera_distance()`: 카메라 거리 정규화 (target: 2.7)
- `normalize_camera_distance_with_intrinsics()`: 거리 + 내부 파라미터 조정
- `normalize_cameras_to_y_up()`: Y-up 좌표계로 정렬
- `normalize_cameras_to_z_up()`: Z-up 좌표계로 정렬
- `get_bg_color()`: 배경색 파싱
- `preprocess_cameras()`: 전체 전처리 파이프라인
- `PreprocessingConfig`: 전처리 설정 클래스

### 2. Alpha Rendering (alpha_renderer.py)
- `render_opencv_cam_with_alpha()`: Alpha 채널 포함 렌더링
- `DeferredGaussianRenderWithAlpha`: Alpha 지원 deferred rendering

### 3. Loss Extensions (loss_extensions.py)
- `compute_mask_from_config()`: Config 기반 마스크 계산
- `compute_ghost_metrics()`: Ghosting 분석 메트릭
- `MaskType`: 마스크 유형 (NONE, GT, RGB_PRED, ALPHA)

### 4. Visualization (visualization.py)
- `create_threshold_comparison()`: Alpha/RGB 마스크 threshold 비교

### 5. Logging (logging_utils.py)
- `get_experiment_info()`: WandB config 정보 추출
- `get_wandb_log_dict()`: 학습 메트릭 로그 dict
- `get_validation_log_dict()`: 검증 메트릭 로그 dict

## Installation

```bash
# 1. FaceLift repo clone
git clone https://github.com/weijielyu/FaceLift.git
cd FaceLift

# 2. diff_gauss 설치 (alpha 반환 지원)
pip install git+https://github.com/slothfulxtx/diff-gaussian-rasterization.git

# 3. mouse_extensions 복사
cp -r /path/to/mouse_extensions gslrm/

# 4. 패치 적용 (PATCHES.md 참조)
python -m gslrm.mouse_extensions.apply_patches
```

## Usage

```python
# Config
mouse:
  normalize_cameras: true
  target_camera_distance: 2.7
  normalize_to_z_up: true

training:
  dataset:
    random_view_selection: true
    background_color: white
  losses:
    use_rendered_alpha_mask: true
    alpha_mask_threshold: 0.5
```

## Key Changes from Original

| Component | Original | Modified |
|-----------|----------|----------|
| Rasterizer | diff_gaussian_rasterization (2 outputs) | diff_gauss (6 outputs) |
| DeferredRender | returns renders | returns (renders, alphas) |
| Loss mask | GT only | GT / RGB pred / Alpha selectable |
| Visualization | GT/Rendered/Error | + Threshold comparison |
| Preprocessing | None | Camera normalization pipeline |

## Version History

- v1.1.0: Added preprocessing.py module
- v1.0.0: Initial release with alpha rendering, loss extensions, visualization
