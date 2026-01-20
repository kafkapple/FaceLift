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
*Last Updated: 2026-01-19*
*Source: FaceLift Mouse Fork, verified against original papers*
