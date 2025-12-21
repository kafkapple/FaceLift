# 📄 GSLRM_Camera_Dependency_Finetuning_Failure

**대화 요약**: GS-LRM 모델의 마우스 데이터 fine-tuning 시도에서 카메라 intrinsics 의존성으로 인한 실패 원인 분석 및 해결 방향 탐색

**주요 다룬 주제**:
1. GS-LRM의 카메라 포맷 의존성과 Plücker Ray Encoding
2. Fine-tuning 실험 결과 및 Catastrophic Forgetting 분석
3. 데이터 전처리 파이프라인과 카메라 정규화

---

## 1. GS-LRM의 카메라 의존성

### 1.1 Pretrained 모델이 기대하는 카메라 포맷

- **핵심개념**: GS-LRM은 특정 카메라 intrinsics에 강하게 의존
- **작동원리**: Plücker Ray Encoding이 카메라 파라미터로 ray direction 계산
- **활용예시**: FaceLift 데이터는 모두 fx=fy=548.99, cx=cy=256으로 정규화됨

```
Pretrained 모델 기대값:
├── fx = fy = 548.993771650447 (고정)
├── cx = cy = 256.0 (이미지 중앙)
├── image_size = 512 × 512
└── camera_distance ≈ 2.7 units
```

### 1.2 Plücker Ray Encoding의 핵심 역할

- **핵심개념**: 6D ray representation으로 카메라 정보를 모델에 전달
- **작동원리**:
  ```python
  ray_direction = normalize(K_inv @ pixel_coord)  # K_inv = intrinsics 역행렬
  ray_origin = camera_center
  plucker = (ray_direction, cross(ray_origin, ray_direction))
  ```
- **중요성**: `K_inv`가 다르면 같은 픽셀도 완전히 다른 ray로 인코딩됨

### 1.3 실제 카메라 vs 정규화 카메라

| 항목 | Pretrained (FAKE) | 실제 마우스 (REAL) |
|------|-------------------|-------------------|
| fx | 548.99 | 725.47 |
| fy | 548.99 | 819.65 |
| cx | 256.0 | 267.27 |
| cy | 256.0 | 245.61 |
| distance | ~2.7 units | ~200-400mm |

---

## 2. Fine-tuning 실험 및 실패 분석

### 2.1 실험 A: Real Camera 데이터 사용

- **문제상황**: 실제 캘리브레이션 카메라 값으로 학습 시도
- **결과**: Step 1부터 완전히 흰색 예측 (즉시 실패)
- **원인**: Plücker ray가 pretrained 분포와 완전히 다름

### 2.2 실험 B: Freeze All Transformer Layers

- **문제상황**: Transformer 24층 전체 freeze, output head만 학습
- **결과**:
  - Step 1: 마우스 형태 보임 (pretrained 지식 활용)
  - Step 301: 일부 뷰 희미해짐
  - Step 501: 대부분 뷰 흰색
  - Step 901: 줄무늬 아티팩트, 심각한 품질 저하
- **원인**: Output decoder (gaussian_upsampler, pixel_gaussian_decoder)도 불안정

### 2.3 Gradient Explosion 현상

```
Step 846: grad_norm = 716    (threshold 200의 3.5배, skip)
Step 854: grad_norm = 876    (threshold 200의 4.4배, skip)
Step 894: grad_norm = 2808   (threshold 200의 14배!, skip)
```

- **주의사항**: Transformer를 freeze해도 output layer gradient가 폭발
- **해결시도**: LR=1e-4에서도 불안정 → 더 낮은 LR 필요했음

---

## 3. 데이터 전처리 파이프라인

### 3.1 Pixel-based 전처리 방식

- **구현 목표**: 실제 카메라 데이터를 pretrained 모델 포맷에 맞춤
- **핵심 로직**:
  - 이미지 center of mass 기반 정렬
  - 객체 크기를 target_size_ratio (0.3)에 맞게 스케일링
  - 카메라 intrinsics를 fake 값 (548.99)으로 대체
- **주요 함수**: `center_of_mass_and_pixel_scale`

### 3.2 `normalize_distance_to`의 한계

```yaml
# config 설정
normalize_distance_to: 2.7
```

- **기능**: 카메라 translation만 스케일링
- **한계**: Intrinsics (fx, fy, cx, cy)는 변경하지 않음
- **결론**: 근본적 카메라 포맷 불일치 해결 불가

---

## 4. Domain Gap 분석

### 4.1 Human Face vs Mouse 비교

| 요소 | Human Face (Pretrained) | Mouse |
|------|-------------------------|-------|
| 형태 | 정면 위주, 대칭적 | 측면/위에서, 비대칭 |
| 텍스처 | 피부, 머리카락 | 털 (uniform dark) |
| 크기 비율 | 일정 (0.3-0.4) | 다양 (0.1-0.6) |
| 포즈 | 제한적 | 다양한 자세 |

### 4.2 Transfer Learning 실패 원인

1. **카메라 조건부 생성 모델**: 카메라 정보에 강하게 의존
2. **Output Layer 민감도**: Gaussian parameter 예측이 매우 민감
3. **Loss Landscape 문제**: 새 도메인에서 gradient 불안정

---

## 5. 대안 방향 탐색

### 5.1 Zero-shot 사용

- Fine-tuning 포기, pretrained 모델 그대로 사용
- 데이터를 모델 기대 포맷에 완벽히 맞춤
- 장점: Pretrained 지식 완전 보존

### 5.2 LoRA (Low-Rank Adaptation)

```python
# 개념
W_new = W_pretrained + A @ B  # rank << dim
```

- 가중치 직접 수정 대신 low-rank adapter 추가
- Pretrained 가중치 보존
- Catastrophic forgetting 방지

### 5.3 다른 모델 고려

| 모델 | 카메라 의존성 | 적응 용이성 |
|------|--------------|-------------|
| GS-LRM | 매우 높음 (Plücker) | 낮음 |
| LGM | 중간 | 중간 |
| Zero123++ | 낮음 (implicit) | 높음 |
| SV3D | 낮음 | 높음 |

---

## 💡 대화에서 얻은 핵심 인사이트

1. **카메라 포맷은 협상 불가**: Pretrained 조건부 생성 모델은 학습 시 사용된 정확한 카메라 포맷을 요구함. ImageNet mean/std normalization과 같은 개념.

2. **Fine-tuning보다 데이터 적응이 효과적**: 모델 가중치를 수정하려 하기보다, 데이터를 모델이 기대하는 포맷에 맞추는 것이 더 안전하고 효과적.

3. **Transformer freeze만으로 불충분**: Output head도 매우 민감하여, 전체 모델을 freeze하지 않으면 catastrophic forgetting 발생.

---

## ❓ 미해결 질문 또는 추가 학습 필요 사항

- LoRA를 GS-LRM에 적용할 수 있는지? 어떤 레이어에 적용해야 하는지?
- Zero-shot으로 pixel-based 데이터 inference 시 품질이 어느 정도인지?
- 카메라 의존성이 낮은 모델들의 마우스 데이터 적용 가능성?

---

## 🔗 참고 자료 및 키워드

**핵심 키워드**:
- Plücker Ray Encoding
- Camera Intrinsics/Extrinsics
- Catastrophic Forgetting
- Domain Adaptation
- Gaussian Splatting

**관련 모델**:
- GS-LRM (Gaussian Splatting Large Reconstruction Model)
- LGM (Large Gaussian Model)
- Zero123++
- SV3D (Stable Video 3D)

**관련 파일**:
- `configs/mouse_gslrm_real_camera.yaml`
- `configs/mouse_gslrm_freeze_all.yaml`
- `data_mouse_pixel_based/` - Pixel-based 전처리 데이터
- `docs/reports/251221_research_gslrm_camera_analysis.md` - 상세 연구 보고서

---

*Created: 2024-12-21 | Tags: #TIL #GS-LRM #Camera #Fine-tuning #Mouse3D*
