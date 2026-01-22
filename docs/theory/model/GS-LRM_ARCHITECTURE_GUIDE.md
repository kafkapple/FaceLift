> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Theory](../) | [Camera](../camera/) | [Loss](../loss/)

# GS-LRM Architecture Guide

> **Version**: 1.0
> **Date**: 2026-01-23
> 3D Gaussian Splatting 기반 Large Reconstruction Model 아키텍처 분석

---

## 1. 핵심 개념

### 1.1 GS-LRM이란?

**GS-LRM (3D Gaussian Splatting Large Reconstruction Model)**:
- Feed-forward 방식의 3D 재구성 모델
- 입력 이미지에서 직접 3D Gaussian 파라미터 예측
- 장면별 최적화 없이 단일 순방향 패스로 3D 생성

### 1.2 핵심 설계 원칙

```
입력 이미지 (H×W pixels)
         │
         ▼
    Transformer Encoder
         │
         ▼
    Pixel-wise Decoder
         │
         ▼
H×W개의 3D Gaussians (각 픽셀 → 1 Gaussian)
         │
         ▼
  Differentiable Rendering
         │
         ▼
    Novel View Image
```

---

## 2. GS-LRM vs 3DGS 비교

### 2.1 공통점

| 항목 | 설명 |
|------|------|
| **3D Gaussian 표현** | 위치($p_i$), 스케일($s_i$), 회전($q_i$), 불투명도($\alpha_i$), 색상($c_i$) |
| **미분 가능 렌더링** | 3D→2D 투영 후 역전파로 파라미터 최적화 |
| **실시간 렌더링** | GPU 가속, 고프레임률 가능 |

### 2.2 핵심 차이점

| 항목 | 3DGS (Original) | GS-LRM |
|------|-----------------|--------|
| **초기화** | SfM sparse points | Pixel-wise prediction |
| **학습 방식** | Per-scene optimization | Feed-forward (일반화) |
| **Densification** | 학습 중 clone/split | ❌ 고정 (H×W개) |
| **새 장면 처리** | 수천 iteration 최적화 | 단일 forward pass |
| **속도** | 수분~수시간 | **< 1초** |

### 2.3 Gaussian 생성 전략 비교

```
3DGS (Traditional):
┌──────────────────────────────────────────────────┐
│ SfM Points (sparse) → Clone/Split → Dense Cloud  │
│     ~1000개         학습 중 증가     ~100K+개     │
└──────────────────────────────────────────────────┘

GS-LRM (Feed-forward):
┌──────────────────────────────────────────────────┐
│ Input Image (H×W) → Transformer → H×W Gaussians  │
│    512×512=262K    고정 구조     262K개 (고정)    │
└──────────────────────────────────────────────────┘
```

---

## 3. "픽셀당 하나의 Gaussian" 설계

### 3.1 개념

> "HW Gaussians for each view, where pixel encodes one 3D Gaussian"
> — FaceLift/GS-LRM 논문

- 입력 이미지의 각 픽셀이 하나의 3D Gaussian에 대응
- 네트워크가 픽셀 위치에서 Gaussian 파라미터 직접 출력
- 구조화된 예측 (structured prediction)

### 3.2 장점

| 장점 | 설명 |
|------|------|
| **Feed-forward 용이** | 고정 크기 출력, 가변 길이 처리 불필요 |
| **속도** | 최적화 루프 없음 (0.23초 ~ 1초) |
| **일관된 밀도** | 해상도에 비례하는 Gaussian 수 |
| **일반화** | 새 장면에 즉시 적용 가능 |

### 3.3 단점

| 단점 | 설명 |
|------|------|
| **중복성** | 빈 공간, 동일 표면에 불필요한 Gaussian |
| **비효율성** | 넓은 평면 = 많은 Gaussian (과잉 표현) |
| **디테일 한계** | 렌더링 에러 기반 밀집화 없음 |
| **메모리** | 고해상도 시 Gaussian 수 급증 |

### 3.4 "픽셀이 인코딩" 의미

```python
# Pixel grid → Gaussian parameters
# 픽셀 위치가 Gaussian 예측의 "앵커" 역할

for each pixel (u, v):
    # 1. Ray 계산
    ray = compute_ray(u, v, camera)

    # 2. Depth 예측
    depth = network.predict_depth(features[u,v])

    # 3. 3D 위치 결정
    position = ray.origin + ray.direction * depth

    # 4. 나머지 파라미터 예측
    scale, rotation, opacity, color = network.predict_params(features[u,v])

    gaussians.append(Gaussian(position, scale, rotation, opacity, color))
```

**중요**: Gaussian이 픽셀에 "고정"되는 것이 아님
- 픽셀 그리드는 예측 구조만 제공
- 최종 Gaussian은 3D 공간에서 뷰 독립적으로 존재

---

## 4. FaceLift Mouse 적용

### 4.1 Multi-view 입력

```
6개 뷰 × 512×512 = 6 × 262K = 1,572,864 Gaussians (이론적 최대)
```

실제로는:
- 중복 영역 처리 (겹치는 뷰)
- Opacity pruning (불투명도 낮은 것 제거)
- 최종 ~100K~500K Gaussians

### 4.2 입력/출력 구조

| 항목 | 값 |
|------|-----|
| **입력 뷰** | 4~5개 (학습 시) |
| **출력 뷰** | 1~2개 (평가 시) |
| **해상도** | 512×512 |
| **Gaussians/view** | 262,144개 |

---

## 5. 메모리 및 연산량

### 5.1 Gaussian 파라미터 크기

```
per Gaussian:
- position: 3 floats (x, y, z)
- scale: 3 floats (sx, sy, sz)
- rotation: 4 floats (quaternion)
- opacity: 1 float
- color (SH): 48 floats (degree 3)
─────────────────────────
Total: 59 floats × 4 bytes = 236 bytes/Gaussian
```

### 5.2 총 메모리 추정

| 해상도 | Gaussians/view | 6 views | 메모리 |
|--------|----------------|---------|--------|
| 256×256 | 65K | 390K | ~92 MB |
| 512×512 | 262K | 1.57M | ~371 MB |
| 1024×1024 | 1M | 6.3M | ~1.5 GB |

---

## 6. 3DGS Densification vs GS-LRM 고정

### 6.1 3DGS Densification 과정

```python
# 3DGS: 학습 중 adaptive densification
for iteration in range(30000):
    render()
    compute_loss()
    backprop()

    if iteration % 100 == 0:
        # 렌더링 에러 높은 영역
        for gaussian in high_error_gaussians:
            if gaussian.scale > threshold:
                split(gaussian)  # 큰 것은 분할
            else:
                clone(gaussian)  # 작은 것은 복제
```

### 6.2 GS-LRM: 고정 구조

```python
# GS-LRM: 고정된 Gaussian 수
gaussians = network(input_images)  # 항상 H×W개
# densification 없음
# pruning만 선택적 적용
```

### 6.3 Trade-off 요약

| 측면 | 3DGS | GS-LRM |
|------|------|--------|
| **품질** | 높음 (adaptive) | 양호 (고정) |
| **속도** | 느림 (최적화) | 빠름 (feed-forward) |
| **일반화** | 없음 (per-scene) | 있음 (learned) |
| **메모리 효율** | 높음 (필요한 곳만) | 낮음 (전체 픽셀) |

---

## 7. FaceLift의 해결 전략

### 7.1 비효율성 극복

1. **Multi-view Fusion**: 여러 뷰에서 중복 Gaussian 자동 병합
2. **Opacity Pruning**: 불투명도 낮은 Gaussian 제거
3. **Scale Regularization**: 과도하게 큰/작은 Gaussian 페널티

### 7.2 디테일 향상

1. **Diffusion Prior**: 다중 뷰 일관성을 위한 확산 모델
2. **Fine-tuning**: 특정 도메인(Mouse)에 맞춤 학습
3. **Perceptual Loss**: LPIPS로 고주파 디테일 보존

---

## 8. 관련 문서

| 문서 | 위치 | 내용 |
|------|------|------|
| 카메라 변환 | [coordinate_transformation_guide.md](../camera/coordinate_transformation_guide.md) | 좌표계, Plücker ray |
| 손실 함수 | [GS-LRM_Loss_Formula.md](../loss/GS-LRM_Loss_Formula.md) | L2, LPIPS, alpha loss |
| 마스크 가이드 | [ALPHA_MASK_COMPLETE_GUIDE.md](../mask/ALPHA_MASK_COMPLETE_GUIDE.md) | mask_mode, threshold |

---

*Updated: 2026-01-23*
*Reference: GS-LRM Paper, 3D Gaussian Splatting Paper*
