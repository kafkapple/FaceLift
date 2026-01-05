---
date: 2026-01-04
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, preprocessing, image-processing]
project: Mouse-FaceLift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Mouse Data Clipping Issue Analysis

## Summary

Mouse 데이터 전처리 과정에서 신체 일부가 잘리는 현상 분석 및 해결 방안 도출.

## Problem Statement

`preprocess_uniform_scale.py` 적용 후 mouse 이미지에서 신체 일부(꼬리, 다리 등)가 프레임 경계에서 잘리는 현상 발생.

### 증상
- `data_mouse_uniform`: 300개 뷰 중 290개(97%)에서 clipping 발생
- 그 중 238개(79%)가 severe (200+ pixels at edge)
- 주로 확대(scale > 1) 시 발생

### 정량적 분석

| Dataset | Views Analyzed | Clipping | Severe | BBox Margin (min) |
|---------|---------------|----------|--------|-------------------|
| data_mouse_centered | 120 | 0 (0%) | 0 | 94-141 px |
| data_mouse_uniform | 300 | 290 (97%) | 238 (79%) | 0 px |

## Root Cause Analysis

### 1. 현재 스케일링 로직의 문제

`preprocess_uniform_scale.py:scale_image_center()` 함수:

```python
# 이미지 기하학적 중심 기준 스케일링
scaled_center_x = new_w / 2
scaled_center_y = new_h / 2
offset_x = int(output_size / 2 - scaled_center_x)
offset_y = int(output_size / 2 - scaled_center_y)
```

**문제점**: 객체의 Center of Mass(CoM)가 이미지 중심에서 벗어나 있으면, 스케일 확대 시 반대쪽이 프레임 밖으로 나감.

### 2. 구체적 사례 분석

`sample_000000/cam_000.png`:
- 이미지 크기: 512×512
- CoM 위치: (362, 249) - 중심(256, 256)에서 오른쪽으로 106px 치우침
- target_ratio=0.6 적용 시 약 1.95x 확대
- 결과: 오른쪽 부분이 프레임 밖으로 잘림

### 3. 원인 도식화

```
Original (centered data):
+------------------+
|    [  Mouse  ]   |  <- CoM이 중심 근처, 마진 충분
|                  |
+------------------+

After uniform scale (1.95x):
+------------------+
|[    Mouse      ]-|----> 잘림!
|                  |  <- CoM이 치우친 상태로 확대
+------------------+
```

## Solution: Fit-in-Frame Scaling

### 핵심 아이디어

스케일 적용 전에 **BBox가 프레임 안에 완전히 들어가는 최대 스케일**을 계산하여 제한.

### 알고리즘

```python
def calc_safe_scale(bbox, com, output_size, desired_scale):
    """
    BBox가 프레임 안에 완전히 들어가도록 스케일 제한

    Args:
        bbox: (x1, y1, x2, y2) 현재 객체 bounding box
        com: (cx, cy) Center of Mass
        output_size: 출력 이미지 크기 (e.g., 512)
        desired_scale: 목표 스케일 (target_ratio / current_ratio)

    Returns:
        safe_scale: 프레임 내 유지 가능한 최대 스케일
    """
    x1, y1, x2, y2 = bbox
    cx, cy = com

    # CoM에서 BBox 각 꼭지점까지 거리 (스케일 후에도 유지)
    dist_left = cx - x1
    dist_right = x2 - cx
    dist_top = cy - y1
    dist_bottom = y2 - cy

    # 각 방향에서 허용 가능한 최대 거리 (프레임 중앙 기준)
    half_size = output_size / 2

    # 각 방향별 최대 허용 스케일
    max_scales = [
        half_size / dist_left if dist_left > 0 else float('inf'),
        half_size / dist_right if dist_right > 0 else float('inf'),
        half_size / dist_top if dist_top > 0 else float('inf'),
        half_size / dist_bottom if dist_bottom > 0 else float('inf'),
    ]

    # 가장 제약적인 스케일 선택 (5% 마진 추가)
    max_safe_scale = min(max_scales) * 0.95

    return min(desired_scale, max_safe_scale)
```

### 적용 위치

`scripts/preprocess_uniform_scale.py`:
1. `calc_safe_scale()` 함수 추가
2. `process_sample()` 함수에서 스케일 계산 시 적용

## Implementation Plan

1. **Fit-in-Frame 스케일링 함수 추가**
   - `calc_safe_scale()` 구현
   - 기존 `get_object_bbox_from_alpha()` 활용

2. **CoM 기반 centering 후 스케일링**
   - `scale_image_center()` 대신 `transform_image()` 사용 (pixel_based.py에서 가져오기)
   - 또는 새로운 `scale_image_around_com()` 함수 구현

3. **CLI 옵션 추가**
   - `--safe_margin`: 경계 마진 비율 (default: 0.05 = 5%)
   - `--max_scale`: 최대 허용 스케일 (fallback)

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Clipping rate | 97% | 0% |
| Severe clipping | 79% | 0% |
| Target ratio achieved | Variable | ~target (or slightly less) |

## Trade-offs

- **장점**: 잘림 현상 완전 제거
- **단점**: 일부 샘플은 목표 크기(60%)에 도달하지 못할 수 있음
  - 이미 프레임 경계에 가까운 객체는 확대 제한됨
  - 실제로는 대부분 50-60% 범위 내 유지 예상

## References

- `scripts/preprocess_pixel_based.py`: CoM 기반 centering 로직
- `scripts/preprocess_uniform_scale.py`: 현재 스케일링 로직
- `gslrm/data/mouse_dataset.py`: 데이터셋 로딩 코드
