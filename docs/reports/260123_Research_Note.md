# 260123 Research Note

> Date: 2026-01-23

---

## 1. Gaussian Rendering Clamping 분석

### 1.1 개요

본 분석은 **원본 GS-LRM 논문/코드**, **FaceLift 구현**, **Gaussian Splatting 공개 코드**를 기반으로 Gaussian rendering 출력의 범위 제어(normalization/clamping) 방식을 검토합니다.

### 1.2 이론적 배경

#### Gaussian Splatting의 출력 특성

원본 3D Gaussian Splatting (Kerbl et al., 2023)에서 렌더링된 이미지 값(RGB):

```
C(u,v) = Σᵢ cᵢ · αᵢ · Tᵢ
where:
  αᵢ = opacity of Gaussian i at pixel (u,v) ∈ [0, 1]
  Tᵢ = transmittance (accumulated opacity) ∈ [0, 1]  
  cᵢ = color (RGB 또는 Spherical Harmonics 계수)
```

**핵심 특성**:
- Opacity(α)는 이미 [0, 1]로 제약됨 (렌더링 과정)
- Color coefficient(c)는 **학습 가능한 파라미터** → 무제한 범위
- **따라서 최종 출력 C는 수학적으로 [0, 1]로 보장되지 않음**

#### 색상 표현 방식에 따른 범위 제어

| Color 표현 | 범위 제약 | 적용 방법 |
|-----------|---------|---------|
| **RGB (직접)** | [0, 1] | Sigmoid 또는 Clamp |
| **Spherical Harmonics (SH)** | 제약 없음 | Sigmoid/Clamp 필수 |
| **미압축 SH** | 제약 없음 | Clamp만 사용 |

### 1.3 원본 GS-LRM vs FaceLift 비교

#### 손실 함수 정의

**GS-LRM 논문 설정**:
$$\mathcal{L}_{\text{image}} = \frac{1}{M} \sum_{m=1}^{M} \left[ \mathcal{L}_{\text{MSE}}(I^{gt}, I^{pred}) + \lambda \cdot \mathcal{L}_{\text{perc}}(I^{gt}, I^{pred}) \right]$$

- **λ = 0.5** (VGG perceptual loss weight)
- **LPIPS 미사용** (원본 LRM의 λ=2.0과 다름)

#### FaceLift 손실 함수

| Component | Paper (GS-LRM) | FaceLift | 일치 |
|-----------|-----------|---------|------|
| MSE weight | 1.0 | 1.0 | ✅ |
| Perceptual (λ) | **0.5** | **0.5** | ✅ |
| **LPIPS weight** | **0.0** (not used) | **0.0** (not used) | ✅ |

### 1.4 Clamping 구현 분석

#### Hard Clamp의 Gradient 단절 문제

```python
# 현재 구현
rendering_flat = rendering_flat.clamp(0.0, 1.0)
```

**문제점**:
- clamp는 범위 밖 값에서 gradient = 0
- 학습 초기 불안정한 Gaussian이 RGB > 1.0 출력 시, 해당 픽셀에 대한 gradient가 완전히 차단됨

**예시**:
```
rendered = 1.5, target = 0.8
├─ Clamped: 1.0 vs 0.8 → loss = 0.04, grad w.r.t. rendered = 0 ❌
└─ Unclamped: 1.5 vs 0.8 → loss = 0.49, grad = 1.4 ✓
```

#### 권장 대안

```python
# Option A: Soft clamping (gradient 유지)
def soft_clamp(x, min_val=0.0, max_val=1.0, margin=0.1):
    return torch.sigmoid((x - 0.5) * 10) * (max_val - min_val) + min_val

# Option B: Loss 단에서만 clamp (forward는 유지)
with torch.no_grad():
    rendering_for_viz = rendering.clamp(0, 1)
loss = F.mse_loss(rendering, target)  # 원본 사용
```

### 1.5 Loss 함수별 Clamping 필요성

| Loss | Clamping 필요? | 이유 |
|------|---------------|------|
| L2 (MSE) | ⚠️ 선택적 | 큰 오차에 큰 패널티 (의도적일 수 있음) |
| L1 | ⚠️ 선택적 | L2보다 robust |
| LPIPS | ✅ 권장 | VGG는 [0,1] 입력 가정 |
| SSIM | ✅ 필수 | 구조 비교에 범위 가정 |

**권장**: Loss별 선택적 적용
```python
# L2: clamp 없이 (gradient 신호 유지)
l2_loss = F.mse_loss(rendering, target)

# LPIPS: clamp 필수 (사용 시)
lpips_loss = lpips(rendering.clamp(0, 1) * 2 - 1, target * 2 - 1)
```

### 1.6 현재 상태 평가

**장점**:
1. GS-LRM 논문 설정과 일치 (L2=1.0, Perc=0.5, LPIPS=0.0)
2. Perceptual Loss 배경 처리 명시됨 (neutral gray 0.5)
3. 마우스 데이터 특화 확장 (background loss, alpha loss)

**미확인 사항**:
1. Rendering output의 명시적 clamping 여부
2. Loss 범위 안정성 (NaN/Inf 발생 여부)
3. Gradient clipping 설정

### 1.7 권장사항

| 우선순위 | 항목 | 권장 조치 |
|---------|------|---------|
| **P1** | Output clamping 확인 | `gslrm.py` forward() 코드 리뷰 |
| **P1** | Loss 모니터링 | WandB 대시보드에서 loss spike 검사 |
| **P2** | LPIPS 재평가 | weight=0.0 유지 또는 0.1로 실험 |
| **P3** | Soft clamp 도입 | 학습 안정성 향상 시 |

### 1.8 결론

**현재 FaceLift 구현은 GS-LRM 논문 설정과 일치하며 기본적으로 타당함**.

다만:
1. **Rendering output normalization**의 명시적 위치 확인 필수
2. **Loss 계산의 수치 안정성** 모니터링 필수
3. **마우스 데이터 특화 설정** (배경 loss, 마스크 모드) 검증 필수

**참고 자료**:
- GS-LRM Paper: [arXiv:2404.19702](https://arxiv.org/abs/2404.19702)
- 3D Gaussian Splatting: [SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- diff-gaussian-rasterization: [GitHub](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

---

## 2. Temporal Turntable 디버깅

> **Status**: ✅ Resolved (2026-01-23)

### 2.1 현상

| 증상 | 설명 |
|------|------|
| **No Mouse** | 비디오에 생쥐가 전혀 보이지 않음 |
| **Strange Colors** | 기하학적 곡선, 색상들만 표시 (raw Gaussian 같음) |
| **No Temporal Change** | 시간에 따른 변화 없음 |

### 2.2 생성된 파일

| 파일 | 위치 | 용도 |
|------|------|------|
| video_utils.py | `mouse_extensions/utils/video_utils.py` | CV2 기반 MP4 인코딩 |
| temporal_turntable.py | `mouse_extensions/scripts/inference/temporal_turntable.py` | 추론 스크립트 |

### 2.3 학습 vs 추론 데이터 로딩 비교

**학습 시** (`gslrm/data/mouse_dataset.py`):
```python
# 학습 시 데이터 로딩
- DataLoader가 batch를 구성
- image, c2w, fxfycxcy, index 등 edict 형식
- batch.images: [B, V, 3, H, W]
- batch.c2w: [B, V, 4, 4]
- batch.fxfycxcy: [B, V, 4]
```

**추론 시** (`temporal_turntable.py`):
```python
def load_single_sample(sample_dir, sample_idx, device):
    # opencv_cameras.json에서 카메라 파라미터 로드
    # images/*.png에서 이미지 로드
    return edict(
        image=images,      # [1, V, 3, H, W]
        c2w=c2ws,         # [1, V, 4, 4]
        fxfycxcy=fxfycxcy, # [1, V, 4]
        index=index,       # [1, V, 2]
        bg_color=torch.tensor([1.0, 1.0, 1.0]),
    )
```

### 2.4 가능한 원인 분석

1. **데이터 형식 차이**: 학습 시 batch key `images` vs 추론 시 `image`
2. **카메라 파라미터 차이**: intrinsics/extrinsics 범위
3. **Turntable 카메라 설정**: elevation 20°, radius 2.7
4. **Gaussian 출력 품질**: opacity/xyz 분포

### 2.5 근본 원인 및 해결

| 항목 | 학습 (mouse_dataset.py) | 기존 추론 | 수정 후 |
|------|-------------------------|----------|---------|
| **이미지 채널** | 4채널 (RGBA) | 3채널 (RGB) | ✅ 4채널 |
| **Alpha 생성** | 자동 (threshold 250) | 없음 | ✅ 자동 생성 |
| **카메라 정규화** | Z-up + 거리 2.7 | 없음 | ✅ 추가 |
| **Intrinsics 정규화** | 비례 조정 | 없음 | ✅ 추가 |

### 2.6 수정 내용

`load_single_sample()` 함수에 다음 추가:

```python
# 1. Alpha 채널 자동 생성
if auto_generate_mask and alpha is None:
    is_background = np.all(img_np > threshold, axis=2)
    alpha = (~is_background).astype(np.float32)[:, :, np.newaxis]
img_np = np.concatenate([img_np, alpha], axis=2)

# 2. 카메라 정규화
if normalize_to_z_up:
    c2ws_np = normalize_cameras_to_z_up(c2ws_np, up_direction=None)

# 3. 거리 정규화
if target_camera_distance > 0:
    c2ws_np, fxfycxcy_np = normalize_camera_distance_with_intrinsics(
        c2ws_np, fxfycxcy_np, target_camera_distance
    )
```

### 2.7 결과

- Gaussian 통계 정상: xyz [-2.17, 2.67], opacity [0, 1]
- high opacity (>0.5): ~55k (3.5%)
- 테스트 영상 생성 성공

### 2.8 참고 코드 위치

| 코드 | 파일 | 라인 |
|------|------|------|
| 학습 turntable 생성 | `gslrm/model/gslrm.py` | ~1561 |
| render_turntable | `gslrm/model/gaussians_renderer.py` | ~1175 |
| 학습 데이터 로더 | `gslrm/data/mouse_dataset.py` | TBD |
| 추론 데이터 로더 | `mouse_extensions/scripts/inference/temporal_turntable.py` | ~76 |

### 2.9 핵심 교훈

> **학습과 추론의 데이터 전처리는 반드시 동일해야 함**
> - Alpha 채널 (마스크)
> - 카메라 정규화 (좌표계, 거리)
> - Intrinsics 조정

---

*Research Note | 2026-01-23*
