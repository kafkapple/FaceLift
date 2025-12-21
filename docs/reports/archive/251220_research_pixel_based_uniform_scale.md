---
date: 2024-12-20
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, preprocessing, uniform-scale]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Pixel-Based Uniform Scale Preprocessing for Mouse-FaceLift

## 목적

Mouse-FaceLift 파이프라인에서 MVDiffusion 학습 시 **객체 크기 불균일 문제** 해결

## 문제 상황

### 기존 접근법의 한계

1. **카메라 거리 기반 정규화 실패**
   - FaceLift 원본: 사람이 원점에 위치, 카메라 거리 2.7 고정
   - 마우스 데이터: 객체가 원점에 없음, 공간 내 자유롭게 이동
   - 카메라-원점 거리 ≠ 이미지 내 객체 크기

2. **Bounding Box 기반 스케일링 실패**
   - 쥐의 꼬리가 bbox 크기에 불균형하게 영향
   - 꼬리 방향에 따라 bbox가 크게 변동
   - 결과: bbox 비율은 60%로 균일해 보이지만, 실제 픽셀 영역은 5.81%~8.48% (1.46배 차이)

## 해결 방법: Pixel-Based Uniform Scale

### 핵심 아이디어

```
객체 크기 = sqrt(픽셀_수 / 전체_픽셀)
```

- Alpha 마스크에서 실제 객체 픽셀 수를 카운트
- sqrt 적용: 면적 비율 → 선형 크기 비율 변환
- 타겟 비율 0.3 (30%)로 모든 뷰 정규화

### 구현

```python
def get_object_size_ratio(alpha: np.ndarray, threshold: int = 10) -> float:
    """
    Get the ratio of object area to image area using pixel count.
    Using sqrt because we want to scale linearly with size, not area.
    """
    mask = alpha > threshold
    if not mask.any():
        return 0.0

    pixel_count = np.sum(mask)
    h, w = alpha.shape
    total_pixels = h * w

    # sqrt to convert area ratio to linear size ratio
    area_ratio = pixel_count / total_pixels
    size_ratio = np.sqrt(area_ratio)

    return size_ratio
```

## 결과

### 전처리 통계 (100개 샘플, 600개 뷰)

| 지표 | 원본 크기 비율 | 스케일 적용 후 |
|------|---------------|---------------|
| Mean | 27.2% | **30.0%** |
| Std | 4.2% | **0.0%** |
| CV | 15.51% | **0.00%** |
| Min | 17.0% | 30.0% |
| Max | 42.9% | 30.0% |

### 스케일 팩터 분포

- Mean: 1.13
- Std: 0.18
- Range: 0.70 ~ 1.76

## 핵심 인사이트

### Why Pixel-Based > Bbox-Based?

1. **꼬리 독립성**: 픽셀 카운트는 꼬리 방향/길이에 선형적으로 비례
2. **실제 크기 반영**: bbox는 빈 공간을 포함, 픽셀은 실제 객체 영역만 측정
3. **sqrt 변환**: 2D 면적을 1D 크기로 변환하여 적절한 스케일 팩터 계산

### 3D 일관성 trade-off

- 이미지 기반 스케일링은 카메라 파라미터를 수정하지 않음
- 결과: 이미지 내 크기는 균일, 3D 기하학은 원본 유지
- MVDiffusion에는 적합 (2D 이미지 생성), GS-LRM에서는 추가 정규화 필요할 수 있음

## 다음 단계

1. ~~이전 학습 체크포인트 삭제~~
2. data_mouse_uniform_v2로 MVDiffusion 재학습 (pretrained에서 finetune)
3. 학습 결과 검증
4. 합성 데이터 생성
5. GS-LRM 학습

## 관련 파일

- `scripts/preprocess_uniform_scale.py`: 전처리 스크립트
- `data_mouse_uniform_v2/`: 전처리된 데이터셋 (2000 샘플)
- `configs/mouse_mvdiffusion_uniform_v2.yaml`: 학습 설정 (생성 예정)
