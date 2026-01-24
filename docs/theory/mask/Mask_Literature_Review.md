# 3DGS Mask Handling Literature Review

> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Theory](../) | [Mask](./)

**Date**: 2026-01-24 (Updated)
**Purpose**: 3D Gaussian Splatting 논문들의 마스크 처리 방식 조사 및 FaceLift 적용

---

## Executive Summary

| 방법론 | Mask in Loss | Alpha Supervision | 핵심 특징 |
|--------|--------------|-------------------|-----------|
| Original 3DGS | ❌ None | ❌ None | 전체 이미지 L1+SSIM |
| GaussianObject | ✅ BCE | ✅ BCE | 명시적 alpha loss |
| **LGM** | ✅ MSE | ✅ **MSE** | Shape 수렴 가속화 |
| **Pose Splatter** | ✅ Normalized L1 | - | Mouse 데이터 검증 |
| Object-Centric 2DGS | - | ✅ BG Penalty | 96% 모델 경량화 |
| Splatfacto-W | ✅ Composite | ✅ Strong α | 배경 분리 |

**핵심 발견**:
- **Rendered alpha로 loss masking하는 논문 없음** (확장 위험)
- Object-centric 방법들은 **GT mask + Alpha Supervision Loss** 조합
- MSE가 BCE보다 gradient 안정적

---

## 상세 분석

### 1. LGM (ECCV 2024 Oral) ★

**Loss 구조**:
```
L_rgb = L_MSE(I_rgb, I_rgb^GT) + λ·L_LPIPS(I_rgb, I_rgb^GT)
L_α = L_MSE(I_α, I_α^GT)  ← Alpha Supervision
```

**핵심 인용**:
> "We use simple MSE loss on both RGB and alpha images for faster convergence of the shape."

**적용**:
- `alpha_loss_type: mse`
- `alpha_loss_weight: 1.0` (RGB와 동등)

### 2. Pose Splatter (NeurIPS 2025) ★

**Loss 구조**:
```
L_mask = Σ|pred - gt|·mask / Σmask  (정규화)
```

**핵심 인용**:
> "Normalized by the mask sum to prevent bias towards larger foregrounds."

**적용**:
- `normalize_by_mask: true` (★ 중요)
- **Mouse/rat 데이터**에서 검증됨

### 3. Object-Centric 2DGS (2025)

**Loss 구조**:
```
L_bg = mean(rendered_α · (1 - GT_mask))
```

**핵심 인용**:
> "Background penalty results in 96% model size reduction and 71% training speedup."

**적용**:
- `bg_loss_weight: 0.3~1.0`
- 배경 Gaussian 생성 억제

### 4. Splatfacto-W (NerfStudio)

**Loss 구조**:
```
pred_composite = pred_rgb · α + bg · (1 - α)
gt_composite = gt_rgb · mask + bg · (1 - mask)
L = MSE(pred_composite, gt_composite)
```

**적용**:
- `mask_mode: composite`
- `background_color: 1.0` (white)
- `alpha_loss_weight: 0.3` (강화)

---

## FaceLift 실험 설정

### 권장 설정 (E2_gt_alpha) ★

```yaml
training:
  losses:
    mask_mode: gt              # GT mask (안정적)
    normalize_by_mask: true    # Pose Splatter
    alpha_loss_weight: 0.1     # LGM (약간 낮춤)
    alpha_loss_type: mse       # BCE보다 안정적
    bg_loss_weight: 0.0
```

### 실험적 설정

| 실험 | 문헌 근거 | 특징 |
|------|----------|------|
| E5_composite_strong | Splatfacto-W | composite + α=0.3 |
| E6_lgm_full | LGM | α=1.0 (동등 가중치) |
| E6_bg_penalty_strong | 2DGS | bg_loss=1.0 |
| E6_combined | 다중 | α=0.2 + bg=0.3 |

---

## mask_mode 동작 상세

### gt (권장)
```python
# GT 마스크로 RGB loss 제한
loss = ((pred - gt)**2 * mask).sum() / mask.sum()
```
- **장점**: 안정적, 마스크 고정
- **단점**: 배경 학습 불가

### alpha (비권장)
```python
# Rendered alpha로 loss 제한
mask = (rendered_alpha > threshold).float()
loss = ((pred - gt)**2 * mask).sum() / mask.sum()
```
- **문제**: 학습 중 마스크 확장 가능 → 악순환
- 문헌에서 사용 사례 없음

### composite
```python
# 배경 합성 후 전체 비교
pred_comp = pred * α + bg * (1 - α)
gt_comp = gt * mask + bg * (1 - mask)
loss = (pred_comp - gt_comp)**2
```
- **장점**: 배경도 학습
- **적합**: 흰 배경 데이터

### none
```python
# 전체 이미지 비교
loss = (pred - gt)**2
```
- **적합**: 마스크 필요 없는 경우

---

## D7_1 마스크 품질 분석

### Alpha Channel 검증
| Threshold | IoU vs GT | 결과 |
|-----------|-----------|------|
| 0.1 ~ 0.9 | **1.0** | 완벽 일치 |

**결론**: D7_1 알파 채널은 깨끗한 이진값 (0/255)

### RGB Prediction (부적합)
| Threshold | IoU | Pred Mask % |
|-----------|-----|-------------|
| 0.02~0.3 | 0.06~0.08 | 37~50% |

**문제**: False positive 과다 (흰색 기준이 마우스 데이터에 부적합)

---

## References

| Paper | Venue | Link |
|-------|-------|------|
| LGM | ECCV 2024 | [arxiv](https://arxiv.org/abs/2402.05054) |
| Pose Splatter | NeurIPS 2025 | [arxiv](https://arxiv.org/html/2505.18342v1) |
| GaussianObject | SIGGRAPH Asia 2024 | [arxiv](https://arxiv.org/abs/2402.10259) |
| Object-Centric 2DGS | 2025 | [arxiv](https://arxiv.org/html/2501.08174v2) |
| Splatfacto-W | NerfStudio | [docs](https://docs.nerf.studio/) |

---

## Related Documents

- [VSCode_Debug_Mask_Guide](../../tutorials/VSCode_Debug_Mask_Guide.md)
- [GHOSTING_DIAGNOSIS](../../analysis/GHOSTING_DIAGNOSIS.md)
- [Quick Reference](../../practical/MOUSE_QUICK_REFERENCE.md)

---

*FaceLift Mouse | Mask Literature Review v2.0 | 2026-01-24*

---

## 중요 개념 구분

### mask_mode vs alpha_loss (혼동 주의)

```
┌─────────────────────────────────────────────────────────────────┐
│ mask_mode: gt                                                   │
│   → RGB loss를 GT mask 영역으로 제한                             │
│   → L_rgb = MSE(pred, gt) * gt_mask                             │
├─────────────────────────────────────────────────────────────────┤
│ alpha_loss_weight: 0.1                                          │
│   → 별도 loss로 rendered α가 GT mask에 수렴하도록 유도           │
│   → L_alpha = MSE(rendered_alpha, gt_mask)                      │
├─────────────────────────────────────────────────────────────────┤
│ ★ 권장 조합: mask_mode: gt + alpha_loss_weight: 0.1             │
│   - RGB: GT mask 영역에서만 계산 (안정적)                        │
│   - Alpha: GT mask로 수렴 유도 (shape 학습)                      │
└─────────────────────────────────────────────────────────────────┘
```

### 잘못된 접근: mask_mode: alpha

```
┌─────────────────────────────────────────────────────────────────┐
│ ❌ mask_mode: alpha                                              │
│   → rendered_alpha를 RGB loss 마스크로 사용                      │
│   → 문제: 학습 중 alpha 확장 → 마스크 확장 → 악순환              │
│   → 문헌에서 이 방식 사용 사례 없음                              │
└─────────────────────────────────────────────────────────────────┘
```

