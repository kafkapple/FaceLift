# Gaussian Rendering 출력 범위 Clamping 분석 보고서

> **Date**: 2026-01-23
> **Status**: Initial Analysis

---

## 1. 개요

본 분석은 **원본 GS-LRM 논문/코드**, **FaceLift 구현**, **Gaussian Splatting 공개 코드**를 기반으로 Gaussian rendering 출력의 범위 제어(normalization/clamping) 방식을 검토합니다.

---

## 2. 이론적 배경

### 2.1 Gaussian Splatting의 출력 특성

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

### 2.2 색상 표현 방식에 따른 범위 제어

| Color 표현 | 범위 제약 | 적용 방법 |
|-----------|---------|---------|
| **RGB (직접)** | [0, 1] | Sigmoid 또는 Clamp |
| **Spherical Harmonics (SH)** | 제약 없음 | Sigmoid/Clamp 필수 |
| **미압축 SH** | 제약 없음 | Clamp만 사용 |

---

## 3. 원본 GS-LRM vs FaceLift 비교

### 3.1 손실 함수 정의

**GS-LRM 논문 설정**:
$$\mathcal{L}_{\text{image}} = \frac{1}{M} \sum_{m=1}^{M} \left[ \mathcal{L}_{\text{MSE}}(I^{gt}, I^{pred}) + \lambda \cdot \mathcal{L}_{\text{perc}}(I^{gt}, I^{pred}) \right]$$

- **λ = 0.5** (VGG perceptual loss weight)
- **LPIPS 미사용** (원본 LRM의 λ=2.0과 다름)

### 3.2 FaceLift 손실 함수

| Component | Paper (GS-LRM) | FaceLift | 일치 |
|-----------|-----------|---------|------|
| MSE weight | 1.0 | 1.0 | ✅ |
| Perceptual (λ) | **0.5** | **0.5** | ✅ |
| **LPIPS weight** | **0.0** (not used) | **0.0** (not used) | ✅ |

---

## 4. Clamping 구현 분석

### 4.1 문제점: Hard Clamp의 Gradient 단절

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

### 4.2 권장 대안

```python
# Option A: Soft clamping (gradient 유지)
def soft_clamp(x, min_val=0.0, max_val=1.0, margin=0.1):
    return torch.sigmoid((x - 0.5) * 10) * (max_val - min_val) + min_val

# Option B: Loss 단에서만 clamp (forward는 유지)
with torch.no_grad():
    rendering_for_viz = rendering.clamp(0, 1)
loss = F.mse_loss(rendering, target)  # 원본 사용
```

---

## 5. Loss 함수별 Clamping 필요성

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

---

## 6. 현재 상태 평가

### 6.1 장점

1. GS-LRM 논문 설정과 일치 (L2=1.0, Perc=0.5, LPIPS=0.0)
2. Perceptual Loss 배경 처리 명시됨 (neutral gray 0.5)
3. 마우스 데이터 특화 확장 (background loss, alpha loss)

### 6.2 미확인 사항

1. Rendering output의 명시적 clamping 여부
2. Loss 범위 안정성 (NaN/Inf 발생 여부)
3. Gradient clipping 설정

---

## 7. 권장사항

| 우선순위 | 항목 | 권장 조치 |
|---------|------|---------|
| **P1** | Output clamping 확인 | `gslrm.py` forward() 코드 리뷰 |
| **P1** | Loss 모니터링 | WandB 대시보드에서 loss spike 검사 |
| **P2** | LPIPS 재평가 | weight=0.0 유지 또는 0.1로 실험 |
| **P3** | Soft clamp 도입 | 학습 안정성 향상 시 |

---

## 8. 결론

**현재 FaceLift 구현은 GS-LRM 논문 설정과 일치하며 기본적으로 타당함**.

다만:
1. **Rendering output normalization**의 명시적 위치 확인 필수
2. **Loss 계산의 수치 안정성** 모니터링 필수
3. **마우스 데이터 특화 설정** (배경 loss, 마스크 모드) 검증 필수

---

## 참고 자료

- GS-LRM Paper: [arXiv:2404.19702](https://arxiv.org/abs/2404.19702)
- 3D Gaussian Splatting: [SIGGRAPH 2023](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- diff-gaussian-rasterization: [GitHub](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

---

*Report Date: 2026-01-23*
