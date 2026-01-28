> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Practical](../../practical/) | [Theory](../)

# GS-LRM Loss Formula

> GS-LRM 총 손실 함수 및 실험별 가중치 설정
> Source: `gslrm/model/gslrm.py:651-665`

---

## Original Paper Loss (Reference)

### LRM (Hong et al., 2023)

> Source: [LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400)

$$
\mathcal{L}_{\text{recon}} = \frac{1}{V} \sum_{v=1}^{V} \left[ \mathcal{L}_{\text{MSE}}(\hat{x}_v, x_v^{GT}) + \lambda \cdot \mathcal{L}_{\text{LPIPS}}(\hat{x}_v, x_v^{GT}) \right]
$$

- **$\lambda = 2.0$** (LPIPS weight)
- $V$: supervision view 수
- $\mathcal{L}_{\text{MSE}}$: normalized pixel-wise L2 loss
- $\mathcal{L}_{\text{LPIPS}}$: perceptual image patch similarity

### GS-LRM / Long-LRM (Zhang et al., 2024)

> Source: [GS-LRM](https://arxiv.org/abs/2404.19702), [Long-LRM](https://arxiv.org/abs/2410.12781)

$$
\mathcal{L}_{\text{image}} = \frac{1}{M} \sum_{m=1}^{M} \left[ \mathcal{L}_{\text{MSE}}(I^{gt}, I^{pred}) + \lambda \cdot \mathcal{L}_{\text{perc}}(I^{gt}, I^{pred}) \right]
$$

- **$\lambda = 0.5$** (Perceptual loss weight) ✅
- $\mathcal{L}_{\text{perc}}$: VGG-based perceptual loss

---

## FaceLift Implementation

### Total Loss Formula

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{L2}} \mathcal{L}_{\text{L2}} + \lambda_{\text{LPIPS}} \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}} + \lambda_{\text{pixel}} \mathcal{L}_{\text{pixel}} + \lambda_{\text{pts}} \mathcal{L}_{\text{pts}} + \lambda_{\text{bg}} \mathcal{L}_{\text{bg}} + \lambda_{\alpha} \mathcal{L}_{\alpha}
$$

### Paper vs Implementation 비교

| Component | Original Paper | FaceLift Implementation | 일치 |
|-----------|---------------|------------------------|------|
| MSE/L2 weight | 1.0 | `l2_loss_weight: 1.0` | ✅ |
| Perceptual weight | **0.5** | `perceptual_loss_weight: 0.5` | ✅ |
| LPIPS | 2.0 (LRM) / 0.0 (GS-LRM) | `lpips_loss_weight: 0.0` | ✅ |
| SSIM | Not used | `ssim_loss_weight: 0.0` | ✅ |
| Background | Not in paper | `background_loss_weight` | 🆕 Extension |
| Alpha | Not in paper | `alpha_loss_weight` | 🆕 Extension |

> **결론**: FaceLift 구현은 GS-LRM 논문 설정 ($\lambda_{\text{L2}}=1.0$, $\lambda_{\text{perc}}=0.5$)과 **일치**합니다.
> Background loss와 Alpha loss는 Mouse adaptation을 위한 **확장** 기능입니다.

---

## Individual Loss Definitions

### 1. L2 Loss (MSE)

$$
\mathcal{L}_{\text{L2}} =
\begin{cases}
\frac{1}{N_{\text{fg}}} \sum_{i \in \text{fg}} \| I_{\text{pred}}^i - I_{\text{gt}}^i \|^2 & \text{if masked} \\
\frac{1}{N} \sum_i \| I_{\text{pred}}^i - I_{\text{gt}}^i \|^2 & \text{otherwise}
\end{cases}
$$

- **masked_l2_loss**: foreground 픽셀 ($M > 0.5$) 에서만 계산
- Mask는 `mask_mode` 설정에 따라 결정 (none, gt, alpha, rgb_pred)

### 2. Perceptual Loss (VGG)

$$
\mathcal{L}_{\text{perc}} = \sum_l \| \phi_l(I_{\text{pred}}) - \phi_l(I_{\text{gt}}) \|_1
$$

- $\phi_l$: VGG19 layer $l$ feature map
- **masked_perceptual_loss**: 배경을 neutral gray (0.5)로 설정 후 계산

### 3. LPIPS Loss

$$
\mathcal{L}_{\text{LPIPS}} = \text{LPIPS}(2 \cdot I_{\text{pred}} - 1, 2 \cdot I_{\text{gt}} - 1)
$$

- Input range: $[-1, 1]$ (normalized from $[0, 1]$)
- **Note**: GS-LRM uses VGG perceptual loss, not LPIPS (different from original LRM)

### 4. SSIM Loss

$$
\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(I_{\text{pred}}, I_{\text{gt}})
$$

- **masked_ssim_loss**: 배경을 neutral gray로 설정 후 계산

### 5. Pixel Alignment Loss

$$
\mathcal{L}_{\text{pixel}} = \| \mathbf{p}_{\perp} \|_2
$$

- $\mathbf{p}_{\perp}$: 3D point의 ray 방향 직교 성분
- **masked_pixelalign_loss**: foreground에서만 계산

### 6. Points Distance Loss

$$
\mathcal{L}_{\text{pts}} = \| d_{\text{pred}} - d_{\text{target}} \|^2
$$

- Depth regularization (warmup 시 사용)

### 7. Background Loss (🆕 Mouse Extension)

$$
\mathcal{L}_{\text{bg}} = \frac{1}{N_{\text{bg}}} \sum_{i \in \text{bg}} \| I_{\text{pred}}^i - c_{\text{bg}} \|^2
$$

- $c_{\text{bg}}$: target background color (default: white $[1, 1, 1]$)
- Background 영역 ($M < 0.5$)에서 white color 강제
- **Not in original paper** - added for mouse data with white background

### 8. Alpha Loss (🆕 Mouse Extension)

$$
\mathcal{L}_{\alpha} =
\begin{cases}
\text{BCE}(\alpha_{\text{pred}}, \alpha_{\text{gt}}) & \text{if type = bce} \\
\| \alpha_{\text{pred}} - \alpha_{\text{gt}} \|^2 & \text{if type = mse} \\
1 - \text{Dice}(\alpha_{\text{pred}}, \alpha_{\text{gt}}) & \text{if type = dice}
\end{cases}
$$

- $\alpha_{\text{pred}}$: rendered alpha (accumulated opacity)
- $\alpha_{\text{gt}}$: GT mask alpha channel
- **Not in original paper** - direct alpha channel supervision for foreground/background separation

---

## Experiment Weight Configurations

### 가중치 비교표

| Exp | $\lambda_{\text{L2}}$ | $\lambda_{\text{perc}}$ | $\lambda_{\text{LPIPS}}$ | $\lambda_{\text{SSIM}}$ | $\lambda_{\text{bg}}$ | $\lambda_{\alpha}$ | Mask | Paper 일치 |
|-----|----------------------|------------------------|-------------------------|------------------------|----------------------|-------------------|------|-----------|
| **E1.1** | 1.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | none | ✅ GS-LRM |
| **E1.2** | 1.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | none | ✅ GS-LRM |
| **E2.1** | 1.0 | 0.5 | 0.0 | 0.0 | **1.0** | 0.0 | rgb_pred | 🆕 +bg |
| **E2.2** | 1.0 | 0.5 | 0.0 | 0.0 | **1.0** | 0.0 | gt | 🆕 +bg |
| **E2.3** | 1.0 | 0.5 | 0.0 | 0.0 | **1.0** | 0.0 | alpha | 🆕 +bg |
| **E3.2** | 1.0 | 0.5 | 0.0 | 0.0 | **1.0** | 0.0 | alpha | 🆕 +bg |
| **E4.2** | 1.0 | 0.5 | 0.0 | 0.0 | **1.0** | **0.1** | alpha | 🆕 +bg+α |
| **E5.1** | 1.0 | 0.5 | 0.0 | 0.0 | **1.0** | 0.0 | alpha | 🆕 +bg |

> Note: $\lambda_{\text{pixel}} = 0$, $\lambda_{\text{pts}} = 0$ for all experiments

---

## Effective Loss per Experiment

### E1.1 / E1.2 (Paper Baseline) ✅ GS-LRM 논문과 동일

$$
\mathcal{L}_{\text{E1}} = 1.0 \cdot \mathcal{L}_{\text{L2}} + 0.5 \cdot \mathcal{L}_{\text{perc}}
$$

### E2.x (Masked + Background Loss) 🆕 Extension

$$
\mathcal{L}_{\text{E2}} = 1.0 \cdot \mathcal{L}_{\text{L2}}^{\text{masked}} + 0.5 \cdot \mathcal{L}_{\text{perc}}^{\text{masked}} + 1.0 \cdot \mathcal{L}_{\text{bg}}
$$

- Mask source varies: rgb_pred (E2.1), gt (E2.2), alpha (E2.3)

### E3.2 / E5.1 (5-View + Background Loss) 🆕 Extension

$$
\mathcal{L}_{\text{E3/E5}} = 1.0 \cdot \mathcal{L}_{\text{L2}}^{\text{masked}} + 0.5 \cdot \mathcal{L}_{\text{perc}}^{\text{masked}} + 1.0 \cdot \mathcal{L}_{\text{bg}}
$$

### E4.2 (5-View + Background + Alpha Loss) 🆕 Extension

$$
\mathcal{L}_{\text{E4}} = 1.0 \cdot \mathcal{L}_{\text{L2}}^{\text{masked}} + 0.5 \cdot \mathcal{L}_{\text{perc}}^{\text{masked}} + 1.0 \cdot \mathcal{L}_{\text{bg}} + 0.1 \cdot \mathcal{L}_{\alpha}^{\text{BCE}}
$$

---

## Mask Mode Details

| Mode | Mask Source | Formula |
|------|-------------|---------|
| `none` | No mask | $M = 1$ (all foreground) |
| `gt` | GT alpha channel | $M = \alpha_{\text{gt}}$ |
| `alpha` | Rendered alpha | $M = \mathbb{1}[\alpha_{\text{pred}} > \tau]$, $\tau = 0.5$ |
| `rgb_pred` | RGB prediction | $M = \mathbb{1}[\|I_{\text{pred}} - c_{\text{bg}}\| > \tau]$, $\tau = 0.1$ |

---

## Value Range & Normalization ✅ Verified

### 데이터 흐름

| 단계 | 범위 | 코드 위치 |
|------|------|----------|
| **데이터 로딩** | [0, 255] → [0, 1] | `mouse_dataset.py:398` (`/ 255.0`) |
| **L2 Loss** | [0, 1] 직접 사용 | `gslrm.py:_compute_l2_loss` |
| **VGG Perceptual** | [0, 1] → [~-124, 131] | `utils_losses.py:322` (`* 255 - mean`) |
| **LPIPS** | [0, 1] → [-1, 1] | `gslrm.py:493` (`* 2 - 1`) |
| **SSIM** | [0, 1] 직접 사용 | `utils_losses.py:SsimLoss` |

### VGG ImageNet 정규화

```python
# gslrm/model/utils_losses.py:34
IMAGENET_MEAN = [123.6800, 116.7790, 103.9390]  # BGR

# Line 322-323: 적용
real_img_normalized = real_img * 255.0 - imagenet_mean
```

### LPIPS 정규화

### Gaussian Rendering 출력 범위 ⚠️

| 항목 | 범위 | 소스 |
|------|------|------|
| **GT Image** | [0, 1] | image_np / 255.0 |
| **Rendered Image** | [0, ∞) 이론적 | diff-gaussian-rasterization |
| **Background Color** | (1.0, 1.0, 1.0) | render_opencv_cam 기본값 |

**잠재적 문제**:
- SH (Spherical Harmonics) 평가 시 RGB > 1.0 또는 RGB < 0 가능
- 학습 초기, 불안정한 Gaussian에서 더 자주 발생
- Out-of-range 값이 loss를 불안정하게 만듦

**해결**: Loss 계산 전 clamping 추가
```python
# gslrm/model/gslrm.py - compute_losses()
rendering_flat = rendering_flat.clamp(0.0, 1.0)  # 추가됨
```

| 항목 | 이전 | 이후 |
|------|------|------|
| Rendering 범위 | [-∞, +∞] 가능 | [0, 1] 보장 |
| Loss 안정성 | out-of-range로 불안정 | ✅ 안정적 |
| GT와 일치 | 불일치 가능 | ✅ 일치 |

```python
# gslrm/model/gslrm.py:493-495
return self.lpips_loss_module(
    rendering * 2.0 - 1.0, target * 2.0 - 1.0  # [0,1] → [-1,1]
).mean()
```

---

## Original vs FaceLift Implementation ✅

| 항목 | GS-LRM 원본 | FaceLift | 일치 |
|------|------------|----------|------|
| Perceptual Loss 종류 | VGG | VGG | ✅ |
| Perceptual 가중치 | 0.5 | 0.5 | ✅ |
| Perceptual Mask | ❌ (사용 안함) | ❌ (default False) | ✅ |
| LPIPS | 미사용 (0.0) | 미사용 (0.0) | ✅ |
| Value Range | 표준 VGG 정규화 | 동일 | ✅ |

> **Note**: `masked_perceptual_loss`는 Mouse Extension 기능으로, 원본 GS-LRM에는 없음.
> 원본과 동일하게 사용하려면 `masked_perceptual_loss: false` (기본값) 유지.

---

## References

1. **LRM** (Hong et al., 2023): [arXiv:2311.04400](https://arxiv.org/abs/2311.04400)
   - $\lambda_{\text{LPIPS}} = 2.0$

2. **GS-LRM** (Zhang et al., 2024): [arXiv:2404.19702](https://arxiv.org/abs/2404.19702)
   - $\lambda_{\text{perc}} = 0.5$ (VGG perceptual, not LPIPS)

3. **Long-LRM** (2024): [arXiv:2410.12781](https://arxiv.org/abs/2410.12781)
   - $\lambda_{\text{perc}} = 0.5$, $\lambda_{\text{opacity}} = 0.1$, $\lambda_{\text{depth}} = 0.01$

---

## Code Reference

```python
# gslrm/model/gslrm.py:651-665
def _compute_total_loss(self, losses):
    weights = self.config.training.losses
    bg_weight = weights.get("background_loss_weight", 0.0)
    alpha_weight = weights.get("alpha_loss_weight", 0.0)
    return (
        weights.l2_loss_weight * losses['l2']
        + weights.lpips_loss_weight * losses['lpips']
        + weights.perceptual_loss_weight * losses['perceptual']
        + weights.ssim_loss_weight * losses['ssim']
        + weights.pixelalign_loss_weight * losses['pixelalign']
        + weights.pointsdist_loss_weight * losses['pointsdist']
        + bg_weight * losses['background']
        + alpha_weight * losses['alpha']
    )
```

---

*Created: 2026-01-19*
*Last Updated: 2026-01-22*
*Source: FaceLift Mouse Fork, verified against original papers*

---

## Metrics Guide (from architecture/)


평가 지표 정의, 목표 범위, WandB 로깅 키 매핑.

---

## 1. Image Quality Metrics

### 1.1 PSNR (Peak Signal-to-Noise Ratio)

| 항목 | 값 |
|------|-----|
| **단위** | dB |
| **범위** | 0 ~ ∞ (높을수록 좋음) |
| **목표** | > 25 dB |
| **WandB** | train/psnr, val/psnr |

**해석**:
| PSNR | 품질 |
|------|------|
| < 20 | ⚠️ 불량 |
| 20-25 | 보통 |
| 25-30 | ✅ 양호 |
| > 30 | 🌟 우수 |

### 1.2 SSIM (Structural Similarity Index)

| 항목 | 값 |
|------|-----|
| **범위** | 0 ~ 1 (높을수록 좋음) |
| **목표** | > 0.85 |
| **WandB** | train/ssim, val/ssim |

### 1.3 LPIPS (Learned Perceptual Image Patch Similarity)

| 항목 | 값 |
|------|-----|
| **범위** | 0 ~ 1 (낮을수록 좋음) |
| **목표** | < 0.15 |
| **WandB** | train/lpips, val/lpips |
| **Note** | GS-LRM에서는 VGG perceptual loss 사용 (LPIPS 아님) |

---

## 2. Mask Metrics

### 2.1 mask_iou (Intersection over Union)

| 항목 | 값 |
|------|-----|
| **범위** | 0 ~ 1 (높을수록 좋음) |
| **목표** | > 0.8 |
| **WandB** | train/mask_iou |
| **Note** | 모니터링 전용, 손실 함수에 포함 안됨 |

**해석**:
| mask_iou | 상태 |
|----------|------|
| < 0.3 | ⚠️ 심각한 문제 |
| 0.3-0.6 | 🔄 학습 중 |
| 0.6-0.8 | ✅ 양호 |
| > 0.8 | 🌟 우수 |

### 2.2 fg_coverage (Foreground Coverage)

| 항목 | 값 |
|------|-----|
| **정상 범위** | 0.03 ~ 0.15 (~3-15%) |
| **WandB** | train/fg_coverage |

**⚠️ 이상 징후**:
| fg_coverage | 원인 | 조치 |
|-------------|------|------|
| > 0.3 | mask_mode: alpha 악순환 | threshold ↑ 또는 mask_mode: gt |
| < 0.01 | 마스크 너무 제한적 | threshold ↓ |

---

## 3. Loss Values

### 3.1 정상 범위

| Loss | 정상 범위 | 이상 시 |
|------|-----------|---------|
| train/loss (total) | 0.05 ~ 0.5 | > 1.0: 학습 불안정 |
| train/l2_loss | 0.005 ~ 0.05 | > 0.1: 렌더링 품질 문제 |
| train/perceptual_loss | 0.01 ~ 0.1 | > 0.2: VGG feature 불일치 |
| train/alpha_loss | 0.1 ~ 0.5 | > 0.7: 마스크 학습 실패 |

### 3.2 학습 진행 패턴

**정상**:
- loss: 0.5 → 0.1 → 0.05 (점진적 감소)
- psnr: 15 → 22 → 28 (점진적 증가)
- mask_iou: 0.2 → 0.5 → 0.8 (점진적 증가)

**비정상**:
- loss: 진동 또는 증가
- psnr: 정체 또는 감소
- mask_iou: 0.02 유지 (alpha inversion)

---

## 4. WandB 키 매핑

### Training
| 키 | 설명 |
|----|------|
| train/loss | Total loss |
| train/l2_loss | L2/MSE loss |
| train/perceptual_loss | VGG perceptual loss |
| train/alpha_loss | Alpha BCE loss |
| train/background_loss | Background color loss |
| train/psnr | Training PSNR |
| train/mask_iou | Mask IoU |
| train/fg_coverage | Foreground coverage |

### Validation
| 키 | 설명 |
|----|------|
| val/psnr | Validation PSNR |
| val/ssim | Validation SSIM |
| val/lpips | Validation LPIPS |

---

## 5. 문제 진단 가이드

### 5.1 PSNR 낮음 (< 20)

| 가능한 원인 | 확인 방법 | 해결 |
|-------------|-----------|------|
| 카메라 미정규화 | fx, trans 값 확인 | D7.1 프리셋 사용 |
| 마스크 악순환 | fg_coverage 확인 | mask_mode: gt |
| 학습률 문제 | loss 진동 확인 | lr 조정 |

### 5.2 mask_iou 낮음 (< 0.3)

| 가능한 원인 | 확인 방법 | 해결 |
|-------------|-----------|------|
| Alpha inversion | fg_coverage > 0.5 | alpha_loss_weight ≥ 0.3 |
| Threshold 부적절 | 시각화 확인 | threshold 조정 |

### 5.3 fg_coverage 비정상

| 상황 | 원인 | 해결 |
|------|------|------|
| 0.5+ | 마스크 배경 침범 | threshold ↑, mask_mode: gt |
| < 0.01 | 마스크 너무 작음 | threshold ↓ |

---

## Related Documents

- [GS-LRM_Loss_Formula](../loss/GS-LRM_Loss_Formula.md) - Loss 수식 상세
- [ALPHA_MASK_COMPLETE_GUIDE](../mask/ALPHA_MASK_COMPLETE_GUIDE.md) - mask_iou 정의

---

*FaceLift Mouse | Metrics Guide v1.0 | 2026-01-23*
