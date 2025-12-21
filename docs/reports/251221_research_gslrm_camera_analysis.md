# GS-LRM 카메라 문제 분석 및 Fine-tuning 실패 원인

---
date: 2024-12-21
context_name: "2_Research"
tags: [ai-assisted, gslrm, camera, fine-tuning, catastrophic-forgetting]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

## 1. 연구 배경

### 1.1 목표
- FaceLift/GS-LRM을 마우스 데이터에 적용하여 single-view → multi-view 3D 재구성
- Pretrained 모델(인간 얼굴 학습)을 마우스 도메인에 fine-tuning

### 1.2 문제 현상
모든 fine-tuning 시도에서 예측이 **흰색/희미한 형태**로 변함

---

## 2. 핵심 발견: 카메라 포맷 불일치

### 2.1 Pretrained 모델이 기대하는 카메라 포맷

```json
{
  "fx": 548.993771650447,
  "fy": 548.993771650447,
  "cx": 256.0,
  "cy": 256.0,
  "w2c": [[R], [t]],  // distance ~2-3 units
  "image_size": [512, 512]
}
```

**특징**:
- **정규화된 focal length**: fx = fy = 548.99 (고정값)
- **중심 principal point**: cx = cy = 256 (이미지 중앙)
- **표준화된 거리**: 카메라 거리 ~2.7 units

### 2.2 실제 마우스 데이터 카메라

```json
{
  "fx": 725.47,
  "fy": 819.65,
  "cx": 267.27,
  "cy": 245.61,
  "w2c": [[R], [t]],  // distance ~200-400mm
  "image_size": [512, 512]
}
```

**특징**:
- **실제 캘리브레이션 값**: fx ≠ fy, 뷰마다 다름
- **비중심 principal point**: cx, cy가 256에서 벗어남
- **실제 거리**: mm 단위 (200-400mm)

### 2.3 왜 문제가 되는가: Plücker Ray Encoding

GS-LRM은 **Plücker Ray Encoding**을 사용하여 카메라 정보를 모델에 전달:

```
ray_direction = normalize(K^(-1) @ pixel_coord)
ray_origin = camera_center
plucker = (ray_direction, cross(ray_origin, ray_direction))
```

**핵심**: `K^(-1)` (intrinsics의 역행렬)이 ray direction을 결정

| 카메라 | fx | ray direction (center pixel) |
|--------|-----|------------------------------|
| FAKE (548.99) | 548.99 | [0, 0, 1] |
| REAL (725) | 725.47 | [0, 0, ~0.98] |

**결과**: 같은 픽셀이라도 다른 ray direction → 모델이 완전히 다른 입력으로 인식

### 2.4 `normalize_distance_to`의 한계

```yaml
# config에서
normalize_distance_to: 2.7
```

이 옵션은 **translation만 스케일링**:
```python
distance = np.linalg.norm(translation)
scale = target_distance / distance
translation_new = translation * scale
```

**Intrinsics (fx, fy, cx, cy)는 변경하지 않음** → 근본 문제 해결 불가

---

## 3. 실험 기록

### 3.1 실험 A: Real Camera 데이터 사용

| 항목 | 값 |
|------|-----|
| Config | `mouse_gslrm_real_camera.yaml` |
| 데이터 | `data_mouse_real_camera/` |
| LR | 1e-6 |
| 결과 | **Step 1부터 완전히 흰색** |

**분석**: Plücker ray가 pretrained 분포와 완전히 다름 → 모델 출력 즉시 붕괴

### 3.2 실험 B: Pixel-based 전처리 데이터 사용

| 항목 | 값 |
|------|-----|
| Config | `mouse_gslrm_freeze_all.yaml` |
| 데이터 | `data_mouse_pixel_based/` |
| 카메라 | fx=fy=548.99, cx=cy=256 (FAKE) |
| LR | 1e-4 |
| Freeze | Transformer 24 layers 전체 |

**진행 상황**:

| Step | PSNR | 시각적 결과 |
|------|------|-------------|
| 1 | ~14.0 | 마우스 형태 보임 (pretrained 지식) |
| 301 | ~13.8 | 일부 뷰 희미해짐 |
| 501 | ~13.5 | 대부분 뷰 흰색 |
| 901 | ~14.0 | 줄무늬 아티팩트, 흐릿함 |

**Gradient Explosion 발생**:
```
Step 846: grad_norm = 716 (skip)
Step 854: grad_norm = 876 (skip)
Step 894: grad_norm = 2808 (skip!)  # 임계값의 14배
```

**분석**:
- Transformer를 freeze해도 output decoder (gaussian_upsampler, pixel_gaussian_decoder)가 불안정
- LR=1e-4가 output layer에도 너무 높음
- Pretrained 지식이 점진적으로 파괴됨 (Catastrophic Forgetting)

### 3.3 시각적 비교

```
Step 1 (Freeze All):
┌─────┬─────┬─────┬─────┬─────┬─────┐
│  GT │  GT │  GT │  GT │  GT │  GT │  ← 마우스 형태 명확
├─────┼─────┼─────┼─────┼─────┼─────┤
│ Pred│ Pred│ Pred│ Pred│ Pred│ Pred│  ← 형태 보임 (흐릿)
└─────┴─────┴─────┴─────┴─────┴─────┘

Step 901 (Freeze All):
┌─────┬─────┬─────┬─────┬─────┬─────┐
│  GT │  GT │  GT │  GT │  GT │  GT │  ← 마우스 형태 명확
├─────┼─────┼─────┼─────┼─────┼─────┤
│흐릿 │줄무늬│희미 │흰색 │흰색 │흐릿 │  ← 심각한 품질 저하
└─────┴─────┴─────┴─────┴─────┴─────┘

Step 1 (Real Camera):
┌─────┬─────┬─────┬─────┬─────┬─────┐
│  GT │  GT │  GT │  GT │  GT │  GT │  ← 마우스 형태 명확
├─────┼─────┼─────┼─────┼─────┼─────┤
│흰색 │흰색 │흰색 │흰색 │흰색 │흰색 │  ← 처음부터 완전 실패
└─────┴─────┴─────┴─────┴─────┴─────┘
```

---

## 4. 실패 원인 분석

### 4.1 Domain Gap 분석

| 요소 | Human Face (Pretrained) | Mouse |
|------|-------------------------|-------|
| 형태 | 정면 위주, 대칭적 | 측면/위에서, 비대칭 |
| 텍스처 | 피부, 머리카락 | 털 |
| 크기 비율 | 일정 (0.3-0.4) | 다양 (0.1-0.6) |
| 배경 | 제거됨 (흰색) | 흰색 (동일) |
| 카메라 | 정규화됨 | 실제 캘리브레이션 |

### 4.2 Fine-tuning이 실패하는 이유

1. **카메라 조건부 생성 모델**
   - GS-LRM은 카메라 정보에 강하게 의존
   - Pretrained 분포 외의 카메라 → 즉시 실패

2. **Output Layer 불안정성**
   - Gaussian parameter 예측이 매우 민감
   - 작은 가중치 변화도 렌더링에 큰 영향

3. **Loss Landscape 문제**
   - 새 도메인에서 gradient가 폭발적
   - Local minima 탈출 시 pretrained 지식 파괴

---

## 5. 대안 방향

### 5.1 Zero-shot 사용 (Fine-tuning 포기)

```yaml
# Pretrained 모델 그대로 사용
# 데이터를 모델 기대 포맷에 완벽히 맞춤
```

**필요 조건**:
- 마우스 이미지를 정확히 pretrained 분포에 맞게 전처리
- fx=fy=548.99, cx=cy=256 포맷 유지
- 적절한 이미지 크롭/스케일링

### 5.2 LoRA (Low-Rank Adaptation)

```python
# 가중치 직접 수정 대신 low-rank adapter 추가
# W_new = W_pretrained + A @ B (rank << dim)
```

**장점**:
- Pretrained 가중치 보존
- 훨씬 적은 파라미터 학습
- Catastrophic forgetting 방지

### 5.3 카메라 정규화 파이프라인 개선

```python
def normalize_camera_to_facelift_format(camera, image):
    """
    실제 카메라 → FaceLift 기대 포맷 변환

    1. Intrinsics 정규화: fx, fy → 548.99
    2. Principal point 중앙화: cx, cy → 256
    3. 이미지 크롭/리샘플링으로 보상
    4. 거리 정규화: distance → 2.7
    """
    pass
```

### 5.4 다른 모델 고려

| 모델 | 카메라 의존성 | 적응 용이성 |
|------|--------------|-------------|
| GS-LRM | 매우 높음 (Plücker) | 낮음 |
| LGM | 중간 | 중간 |
| Zero123++ | 낮음 (implicit) | 높음 |
| SV3D | 낮음 | 높음 |

---

## 6. 결론 및 다음 단계

### 6.1 핵심 교훈

1. **카메라 포맷은 협상 불가**: Pretrained 모델이 기대하는 정확한 포맷 사용 필수
2. **Fine-tuning은 위험**: GS-LRM 같은 생성 모델은 fine-tuning에 매우 취약
3. **데이터 전처리가 핵심**: 모델 수정보다 데이터를 모델에 맞추는 것이 효과적

### 6.2 권장 다음 단계

1. **Zero-shot 테스트**
   - Pretrained 모델로 `data_mouse_pixel_based` 직접 inference
   - Fine-tuning 없이 품질 확인

2. **전처리 파이프라인 검증**
   - 현재 pixel-based 전처리가 정확히 FaceLift 포맷인지 확인
   - 필요시 전처리 스크립트 수정

3. **대안 모델 탐색**
   - LGM, Zero123++, SV3D 등 카메라 의존성 낮은 모델 검토

---

## 7. 참고 파일

| 파일 | 설명 |
|------|------|
| `configs/mouse_gslrm_real_camera.yaml` | Real camera 실험 설정 |
| `configs/mouse_gslrm_freeze_all.yaml` | Freeze all 실험 설정 |
| `data_mouse_pixel_based/` | Pixel-based 전처리 데이터 |
| `data_mouse_real_camera/` | Real camera 데이터 |

---

*Generated with Claude Code - 2024-12-21*
