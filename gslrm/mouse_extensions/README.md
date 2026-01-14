# Mouse Extensions for FaceLift/GS-LRM

Original repo: https://github.com/weijielyu/FaceLift

## Overview

이 모듈은 FaceLift/GS-LRM을 Mouse 데이터에 적용하기 위한 확장 기능을 제공합니다.
원본 코드 수정을 최소화하면서 다음 기능을 추가합니다:

1. **Alpha Rendering**: Gaussian alpha 값 추출 및 반환
2. **Mask Loss Options**: GT/RGB pred/Alpha mask 선택적 적용
3. **Ghost Metrics**: Ghosting 분석 메트릭
4. **Enhanced Logging**: WandB 로깅 확장
5. **Threshold Visualization**: Alpha/RGB mask 비교 시각화

## Installation

```bash
# 1. FaceLift repo clone
git clone https://github.com/weijielyu/FaceLift.git
cd FaceLift

# 2. diff_gauss 설치 (alpha 반환 지원)
pip install git+https://github.com/slothfulxtx/diff-gaussian-rasterization.git

# 3. mouse_extensions 복사
cp -r /path/to/mouse_extensions gslrm/

# 4. 패치 적용
python -m gslrm.mouse_extensions.apply_patches
```

## Usage

```python
# Config에서 활성화
mouse:
  use_mouse_extensions: true

training:
  dataset:
    random_view_selection: true
  losses:
    use_rendered_alpha_mask: true
    alpha_mask_threshold: 0.5
```

## File Structure

```
mouse_extensions/
├── __init__.py           # 메인 진입점
├── alpha_renderer.py     # Alpha rendering 확장
├── loss_extensions.py    # Mask/Loss 확장
├── visualization.py      # 시각화 확장
├── logging_utils.py      # WandB 로깅 확장
├── apply_patches.py      # 원본 코드 패치
└── README.md
```

## Key Changes from Original

### 1. diff_gauss Rasterizer
원본: `diff_gaussian_rasterization` (2 outputs)
수정: `diff_gauss` (6 outputs: color, depth, norm, alpha, radii, extra)

### 2. DeferredGaussianRender
원본: `return renders`
수정: `return renders, alphas`

### 3. LossComputer._compute_all_losses
원본: GT mask만 사용
수정: GT/RGB pred/Alpha mask 선택 가능

### 4. Visualization
원본: GT/Rendered/Error
수정: + Alpha threshold comparison
