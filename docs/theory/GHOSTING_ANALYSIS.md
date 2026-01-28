> Parent: [[INDEX]] > Theory

# Ghosting Analysis & Solution Strategy

**Version**: 2.0
**Date**: 2026-01-27
**Merged from**: GHOSTING_ANALYSIS_REPORT.md, GHOSTING_SOLUTION_STRATEGY.md

---

## 1. 현상 정의

### Ghost Type A: 근접 고스트
- GT 형상 근처에 약간 다른 크기/위치의 이미지 중첩
- **원인**: Intrinsics scaling 불일치, depth ambiguity, PP 미세 불일치

### Ghost Type B: 원거리 고스트
- 90도 회전 위치에 동일 포즈 이미지 배치
- **원인**: View-dependent SH 과적합, Gaussian 3D 위치 오류

---

## 2. 근본 원인: Plucker Ray 분포 불균형

**GS-LRM 입력**: `[RGB(3ch) | Plucker(6ch)] = 9ch per pixel`

**Plucker Coordinates**:
```
P = (d, m) = (d, o × d)
- d: ray 방향 (normalized)
- m: moment (원점에서 ray까지의 수직거리 × 방향)
```

**Mouse 데이터 문제**: Object coverage 2-3%
- 97% 픽셀이 배경 ray → View 간 Plucker 분포 거의 동일
- Transformer가 view 구분 실패 → 독립적 Gaussian 생성 → Ghosting

### PP Mismatch 영향
```
ray_error ≈ arctan(PP_error / focal_length)
예: PP 37px, f=549 → ~3.9° → Ghosting
```

---

## 3. Training vs Validation 차이 분석

| 항목 | Training | Validation | 영향 |
|------|----------|------------|------|
| Turntable 해상도 | input 해상도 사용 | input 해상도 사용 | ✅ 동일 (검증됨) |
| Augmentation | 활성화 | 비활성화 | ⚪ 정상 차이 |
| View selection | 고정 | 고정 | ✅ 동일 |

> **Note**: 해상도 불일치 가설은 코드 분석 후 기각됨 (2026-01-28). Training/Validation 모두 input 해상도 사용.

---

## 4. 데이터 전처리 영향

### zoom_center_mode 영향

| Mode | PP | 특성 |
|------|-----|------|
| `"image"` | 256 고정 | ✅ Pretrained 호환, MVG 정확 |
| `"object"` | 가변 | ⚠️ PP 분산 → Ray 오류 → Ghosting |

**필수**: `zoom_center_mode: "image"`

### M3 시리즈 특성

| Dataset | Zoom | Clipping | 상태 |
|---------|------|----------|------|
| M3_1 | Global | 0% | ✅ 안전 |
| M3_2 | Per-sample [1.0, 1.8] | 6.5% | ⚠️ |

---

## 5. 해결 전략

### 5.1 Object-Centered Zoom (핵심)

```
D7_1 기하학적 정확성 + Object Centering/Scaling
= D7_1_centered (새 프리셋)
```

**목표**:
- FG Coverage: 2% → **5%+**
- Center Offset: 101px → **<15px**
- Ray Error: ~0° 유지
- fx=549, dist=2.7 유지

**파이프라인**:
1. Object Center Detection (3D triangulation)
2. Object-Centered Crop (각 뷰에서 back-project)
3. Intrinsics Adjustment (crop/zoom 반영)
4. Camera Normalization (fx→549, cx/cy→256)

### 5.2 Loss 함수 최적화 (문헌 기반)

```yaml
training:
  losses:
    mask_mode: gt                  # GT mask (안정적)
    normalize_by_mask: true        # Pose Splatter
    alpha_loss_weight: 0.2         # LGM (MSE)
    alpha_loss_type: mse
    bg_loss_weight: 0.3            # Object-Centric 2DGS
    perceptual_loss_weight: 0.5
```

### 5.3 추가 수단

- **Opacity Regularization**: `opacity_reg_weight: 0.01, type: entropy`
- **Input Views 증가**: 4 → 5-6
- **View Augmentation** 활성화

---

## 6. 문헌 근거

| 방법론 | Object Coverage | Centering | Loss |
|--------|-----------------|-----------|------|
| LRM | - | 이미지 중앙 crop | - |
| Objaverse Render | 50-80% | 중앙 정렬 | - |
| LGM | - | - | MSE α |
| Pose Splatter | - | - | Normalized L1 |
| Object-Centric 2DGS | - | - | BG penalty |

### 핵심 인용
- **LRM**: "We crop and resize images to center the object with adjusted camera parameters."
- **LGM**: "MSE loss on both RGB and alpha for faster convergence of the shape."
- **Pose Splatter**: "Division by mask sum normalizes gradients by foreground area."

---

## 7. 예상 결과

| Metric | D7_1 현재 | D7_1_centered 예상 |
|--------|-----------|-------------------|
| Val PSNR | ~20 | 24-26 |
| FG Coverage | 2.16% | 5%+ |
| Center Offset | 101px | <15px |
| Ghosting | 심각 | 경미/없음 |

---

## See Also

- [[MASK_GUIDE]] - 마스크 시스템 상세
- [[FLOATER_ARTIFACT_ANALYSIS]] - Floater 분석
- [[PP_FX_MVG_ANALYSIS]] - PP/MVG 이론
- [[CENTER_ESTIMATION]] - 3D Triangulation

---

*FaceLift Mouse | Ghosting Analysis v2.0 | Last updated: 2026-01-27*

## Floater Mitigation Techniques (from archive)

# Floater Mitigation in 3D Gaussian Splatting

## 1. Problem: Floater Artifacts

### What are Floaters?
Floaters are spurious Gaussian primitives that appear at incorrect depths, causing visual artifacts like:
- **Ghosting**: Double/overlapping objects
- **Haze**: Semi-transparent artifacts in empty space
- **Depth inconsistency**: Objects appearing at wrong distances

### Why Do Floaters Occur?

```
Training View → Gaussians optimize to minimize 2D loss
                ↓
         Ambiguity in 3D position (depth)
                ↓
         Gaussians with intermediate opacity (0.3-0.7)
         get stuck in local minima
                ↓
         These "ghost" Gaussians persist
```

**Root Cause**: 2D supervision alone cannot fully constrain 3D geometry.

---

## 2. Solution: Opacity Regularization

### Theoretical Basis

The key insight: **Real surfaces should have binary opacity** (fully opaque or fully transparent).

Intermediate opacity values (e.g., 0.5) indicate:
- Uncertain Gaussians
- Potential floaters
- Local minima in optimization

### Mathematical Formulation

#### 2.1 Binary Entropy Regularization (Recommended)

$$L_{entropy} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \alpha_i \log(\alpha_i) + (1-\alpha_i) \log(1-\alpha_i) \right]$$

Where $\alpha_i$ is the opacity of Gaussian $i$.

**Properties**:
- Minimum at $\alpha = 0$ or $\alpha = 1$
- Maximum at $\alpha = 0.5$
- Smooth gradient for optimization

```python
entropy = -opacity * log(opacity) - (1 - opacity) * log(1 - opacity)
loss = entropy.mean()
```

#### 2.2 L1 Sparsity

$$L_{sparse} = \left| \frac{1}{N} \sum_{i=1}^{N} \alpha_i - \tau \right|$$

Where $\tau$ is target sparsity (e.g., 0.5).

**Use case**: When you want to control the overall number of opaque Gaussians.

#### 2.3 L2 Binary Distance

$$L_{binary} = \frac{1}{N} \sum_{i=1}^{N} \min(\alpha_i, 1-\alpha_i)^2$$

**Use case**: Direct penalty on intermediate values.

---

## 3. Implementation

### Config Options

```yaml
training:
  losses:
    opacity_reg_weight: 0.01     # 0.0 = disabled
    opacity_reg_type: "entropy"  # entropy | l1_sparse | l2_binary
    opacity_target_sparsity: 0.5 # for l1_sparse only
```

### Recommended Values

| Scenario | Weight | Type |
|----------|--------|------|
| Mild floaters | 0.01 | entropy |
| Severe floaters | 0.05 | entropy |
| Very sparse scenes | 0.01 | l1_sparse |

---

## 4. References

1. **3D Gaussian Splatting** (Kerbl et al., 2023)
   - Original 3DGS paper
   - https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

2. **StableGS: A Floater-Free Framework** (2025)
   - Systematic analysis of floater causes
   - https://arxiv.org/html/2503.18458

3. **Mip-Splatting** (Yu et al., 2024)
   - Anti-aliasing and regularization
   - https://niujinshuchong.github.io/mip-splatting/

4. **GaussianPro** (Cheng et al., 2024)
   - Progressive training with opacity control
   - https://kcheng1021.github.io/gaussianpro.github.io/

---

## 5. Experimental Validation

### Experiments in This Project

| Version | Config | Purpose |
|---------|--------|---------|
| v67 | opacity_reg: 0.01 | Baseline floater mitigation |
| v68 | opacity_reg: 0.05 | Stronger regularization |
| v69 | v13 + opacity_reg | Dataset comparison |

### Expected Outcomes
- Reduced ghosting in turntable visualization
- Cleaner depth maps
- Potentially slight decrease in PSNR (trade-off)

---

*Created: 2026-01-16 | Author: Claude Code*

## Comprehensive Ghosting Analysis (from archive)

# Ghosting 현상 종합 분석 보고서

**Version**: 1.0
**Date**: 2026-01-28
**Author**: AI Research Assistant
**Status**: Investigation Complete

---

## Executive Summary

Training에서 심각한 Ghosting이 발생하는 반면 Validation에서는 정상적인 3D 재구성이 이루어지는 현상의 근본 원인을 분석했습니다.

**핵심 발견**: Turntable 렌더링 해상도 불일치가 주요 원인으로, Training 시각화는 기본 해상도를 사용하고 Validation은 실제 입력 해상도를 사용합니다.

---

## 1. 현상 정의

### 1.1 Ghost Type A: 근접 고스트
- **증상**: GT 형상 근처에 약간 다른 크기/위치의 이미지 중첩
- **특징**: 원본과 유사하나 미세하게 어긋난 복제
- **원인 추정**: Intrinsics scaling 불일치로 인한 ray 방향 오차 (~5°)

### 1.2 Ghost Type B: 원거리 고스트
- **증상**: 90도 회전된 위치에 동일 포즈 이미지 배치
- **특징**: 완전히 다른 공간에 phantom splat 생성
- **원인 추정**: 카메라 좌표계 오정렬 또는 view-dependent 과적합

---

## 2. 이론적 배경

### 2.1 3D Gaussian Splatting의 Multi-view 일관성

3DGS에서 각 Gaussian은 3D 공간의 위치 μ, 공분산 Σ, opacity α, SH 계수를 가집니다. Novel view 렌더링 시:

```
픽셀 색상 = Σ(αᵢ × Tᵢ × cᵢ)
여기서 Tᵢ = Π(1 - αⱼ), j < i (투과율)
```

**핵심**: 모든 뷰에서 동일한 Gaussian이 일관되게 투영되어야 합니다.

### 2.2 Co-adaptation 이론

**문헌**: Barron et al. (2022), Müller et al. (2022)

Training view에서만 최적화되면:
- 여러 Gaussian이 동일 영역을 중복 커버
- Training view에서는 상호 보완하여 정확한 렌더링
- Novel view에서는 중복이 고스트로 나타남

```
[Training View]          [Novel View]
    ○○○                    ○ ○ ○
    |||                    / | \
  정상 렌더링            고스트 발생
```

### 2.3 Principal Point (PP) Mismatch 영향

**문헌**: Hartley & Zisserman (2004) - Multiple View Geometry

PP 오차가 ray 방향에 미치는 영향:
```
ray_direction_error ≈ arctan(PP_error / focal_length)
예: PP 37px 오차, f=549 → ~3.9° 오류
```

37px PP 오차 시 ~5° ray 방향 오류 발생 → Ghosting 유발

### 2.4 GS-LRM 특성

LGM/GS-LRM은 feed-forward 방식으로 Gaussian 파라미터를 예측:
- 입력 이미지 → Transformer → Gaussian params
- Per-view optimization 없이 단일 forward pass
- **취약점**: 입력 intrinsics 불일치에 매우 민감

---

## 3. 실험 조건 분석

### 3.1 Training vs Validation 비교

| 항목 | Training | Validation | 영향 |
|------|----------|------------|------|
| **Turntable 해상도** | 기본값 (미지정) | `input_image.shape[0]` | ⚠️ **핵심 차이** |
| **Intrinsics 스케일링** | 512×512 정규화 | 512×512 정규화 | ✅ 동일 |
| **카메라 정규화** | Z-up, dist=2.7 | Z-up, dist=2.7 | ✅ 동일 |
| **View 선택** | 고정 순서 | 고정 순서 | ✅ 동일 |
| **Augmentation** | 활성화 | 비활성화 | ⚪ 정상 차이 |
| **Loss 계산** | L2 + Perceptual | L2 + Perceptual | ✅ 동일 |

### 3.2 코드 위치 분석

**Training 시각화** (`gslrm.py:1555`):
```python
turntable_image = render_turntable(model_results.gaussians[batch_idx])
# 해상도 파라미터 없음 → 기본값 사용
```

**Validation 시각화** (`gslrm.py:1751-1754`):
```python
render_resolution = input_image.shape[0]  # 실제 입력 해상도
turntable_frames = render_turntable(
    model_results.gaussians[batch_idx], 
    rendering_resolution=render_resolution, ...
)
```

---

## 4. 데이터 전처리 분석

### 4.1 M3 시리즈 전처리 특성

| Dataset | Zoom | PP | fx | Clipping | 상태 |
|---------|------|-----|-----|----------|------|
| M3_1 | Global | 256 | 549 | 0% | ✅ 안전 |
| M3_2 | Per-sample [1.0, 1.8] | 256 | 549 | 6.5% | ⚠️ 일부 클리핑 |
| M3_2b | Per-sample [1.0, 1.5] | 256 | 549 | ~0% | ✅ 보수적 |
| M3_3 | Per-sample + RGB safety | 256 | 549 | TBD | 🔬 검증 필요 |

### 4.2 마우스 데이터 특성

- **움직임**: 프레임 간 평균 99px, 최대 195px
- **형태**: 얇은 꼬리, 사지 → SimpleClick 마스크 불완전
- **시점**: 6 카메라, 60° 간격, 상부 조망
- **크기 변동**: 자세에 따라 bbox 크기 ±30% 변동

### 4.3 zoom_center_mode 영향

| Mode | PP | 특성 |
|------|-----|------|
| `"image"` | 256 고정 | Pretrained 호환, MVG 정확 |
| `"object"` | 가변 | PP 분산 → Ray 오류 → Ghosting |

**권장**: `zoom_center_mode: "image"` 필수

---

## 5. 모델 특성 분석

### 5.1 GS-LRM 아키텍처

```
Multi-view Images (4-6개)
        ↓
    Image Encoder (DINO-ViT)
        ↓
    Multi-view Transformer
        ↓
    Gaussian Decoder
        ↓
    3D Gaussians (per pixel)
        ↓
    Differentiable Rendering
```

### 5.2 Pretrained 모델 기대값

GS-LRM pretrained 모델이 기대하는 입력 분포:
- **fx**: 549 (Objaverse 정규화)
- **PP**: (256, 256) 중앙
- **Translation norm**: ~2.7
- **해상도**: 512×512

**불일치 시 영향**:
- fx 불일치 → 깊이 추정 오류
- PP 불일치 → ray 방향 오류 → Ghosting
- 해상도 불일치 → intrinsics scaling 오류

### 5.3 Feed-forward 특성의 취약점

Per-scene optimization 기반 3DGS와 달리:
- 단일 forward pass로 Gaussian 예측
- 입력 오류에 대한 보정 기회 없음
- Intrinsics 정확성이 결정적

---

## 6. 의심 원인 및 해결책

### 🔴 원인 1: Turntable 렌더링 해상도 불일치 (확신도: 95%)

**문제**:
- Training 시각화: 해상도 미지정 → 기본값 사용
- Validation 시각화: 실제 입력 해상도 사용
- 결과: 다른 intrinsics scaling → ray 방향 불일치

**비판적 분석**:
- ✅ 코드에서 명확히 확인됨
- ✅ Validation에서 정상 동작하는 이유 설명
- ⚠️ 단, 시각화 문제일 뿐 실제 학습에는 영향 없을 수 있음

**해결책**:
```python
# gslrm.py:1555 수정
render_resolution = input_data.image.size(3)  # 실제 입력 해상도
turntable_image = render_turntable(
    model_results.gaussians[batch_idx], 
    rendering_resolution=render_resolution
)
```

**우선순위**: P0 (즉시 수정)

---

### 🟠 원인 2: Co-adaptation으로 인한 Gaussian 중복 (확신도: 70%)

**문제**:
- Training view에서만 최적화 → 특정 뷰에 과적합
- 여러 Gaussian이 동일 영역을 다른 방향에서 커버
- Novel view에서 중복 Gaussian이 고스트로 표현

**비판적 분석**:
- ✅ 3DGS 문헌에서 잘 알려진 현상
- ✅ Training/Validation 차이 설명 가능
- ⚠️ GS-LRM은 feed-forward라 일반 3DGS와 다름
- ⚠️ 실제로 co-adaptation 발생하는지 검증 필요

**해결책**:
1. **Opacity Regularization** 활성화:
   ```yaml
   losses:
     opacity_reg_weight: 0.01
     opacity_reg_type: "entropy"
   ```

2. **Multi-view Consistency Loss** 추가:
   - 학습 중 novel view 렌더링으로 일관성 강제

3. **Gaussian Pruning** 활성화:
   - 낮은 opacity Gaussian 제거

**우선순위**: P1 (실험적 검증 후)

---

### 🟡 원인 3: 입력 해상도 차이 (확신도: 50%)

**문제**:
- Training: 384×384 (config 설정에 따라)
- Validation: 512×512 (원본 해상도)
- 다른 해상도 → 다른 intrinsics scaling

**비판적 분석**:
- ⚠️ 현재 코드에서 둘 다 512로 리사이즈 확인됨
- ⚠️ config에 따라 다를 수 있음
- ❓ 실제 config 확인 필요

**해결책**:
```yaml
# 해상도 통일
model:
  image_tokenizer:
    image_size: 512  # Training과 Validation 동일하게
```

**우선순위**: P2 (config 확인 후)

---

## 7. 권장 조치 순서

### 즉시 (P0)
1. ✅ Training turntable 해상도 명시적 지정
2. ✅ 수정 후 시각화 재확인

### 단기 (P1)
1. 🔬 Opacity regularization 실험
2. 🔬 config 해상도 설정 확인
3. 🔬 실제 intrinsics 값 로깅 추가

### 중기 (P2)
1. 📊 Novel view consistency loss 구현
2. 📊 Gaussian density 분석 도구 개발
3. 📊 Per-view Gaussian 기여도 시각화

---

## 8. 참고 문헌

1. Kerbl et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
2. Zhang et al. (2024). "GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting"
3. Xu et al. (2024). "LGM: Large Multi-View Gaussian Model"
4. Hartley & Zisserman (2004). "Multiple View Geometry in Computer Vision"
5. Barron et al. (2022). "Mip-NeRF 360"

---

## 9. 부록: 코드 위치 참조

| 파일 | 라인 | 내용 |
|------|------|------|
| `gslrm/model/gslrm.py` | 1555 | Training turntable 렌더링 |
| `gslrm/model/gslrm.py` | 1751-1754 | Validation turntable 렌더링 |
| `gslrm/data/mouse_dataset.py` | 전체 | 데이터 로딩 및 전처리 |
| `mouse_extensions/preprocessing/presets.py` | M3_* | 전처리 프리셋 정의 |

---

*FaceLift GS-LRM Ghosting Analysis Report v1.0 | 2026-01-28*

---

## 10. 추가 조사 결과 (2026-01-28 Update)

### 10.1 해상도 불일치 가설 검증

코드 분석 결과, **Turntable 해상도는 이미 일치**합니다:

**Training** (gslrm.py:1437-1439):
- resolution이 None이면 input_data.image.size(3) 사용

**Validation** (validator.py:286):
- input_np.shape[0] 사용

→ **해상도 불일치 가설은 기각됩니다.**

### 10.2 수정된 원인 분석

#### 가설 A: GS-LRM View-dependent Overfitting (확신도: 75%)

**문제**:
GS-LRM은 4-6개 input view에서 Gaussian을 예측합니다. 이 Gaussian들은:
- Input view 방향으로 편향될 수 있음
- Novel view에서 불완전한 coverage

**메커니즘**:
- Input Views: 0, 1, 2, 3 (특정 방향)
- Gaussian 예측: 해당 방향에 집중
- Turntable (360°): Input view 반대 방향에서 Ghosting

**해결 방법**:
1. View Augmentation 활성화
2. 더 많은 Input view 사용 (4 → 5-6)
3. Novel view consistency loss 추가

#### 가설 B: 데이터 특성 차이 (확신도: 60%)

- Training Set vs Validation Set 샘플 특성 차이
- 어려운 포즈가 Training에 편중될 수 있음

#### 가설 C: 시각화 시점 차이 (확신도: 50%)

- Training Visual: vis_every step마다
- Validation Visual: val_every step마다
- 모델 수렴 정도 차이

### 10.3 Ghost Type별 원인 세분화

#### Type A: 근접 고스트 (크기/위치 미세 불일치)
- Gaussian position 예측 오류
- Depth ambiguity
- PP/Intrinsics 미세 불일치 누적

#### Type B: 원거리 고스트 (90° 회전 위치)
- View-dependent SH 계수 과적합
- Gaussian 3D 위치 오류
- 대칭적 위치 (90° 또는 180°)

### 10.4 추가 권장 조사

1. **동일 샘플 비교**: 같은 UID를 Train/Val에서 비교
2. **Gaussian 분포 분석**: 위치 분포 시각화
3. **Per-view 렌더링 비교**: Input vs Novel view 품질
4. **Learning Dynamics**: Step별 Ghosting 변화

---

## 11. 업데이트된 권장 조치

### P0 (완료)
- [x] Turntable 해상도 일치 확인 → 이미 일치

### P1 (단기)
1. 동일 샘플에서 Train/Val turntable 비교
2. Type A vs Type B 고스트 패턴 분석
3. Input view 수 증가 실험 (4 → 5-6)

### P2 (중기)
1. View augmentation 활성화 실험
2. Novel view consistency loss 구현
3. Gaussian density 분석 도구 개발

---

*Updated: 2026-01-28 - Resolution mismatch hypothesis rejected*
