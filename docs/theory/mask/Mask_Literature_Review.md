# 3DGS Mask Handling Literature Review

> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Theory](../) | [Mask](./)

**Date**: 2026-01-24
**Purpose**: 3D Gaussian Splatting 논문들의 마스크 처리 방식 조사 및 FaceLift 적용 제안

---

## Executive Summary

| 방법론 | Mask in Loss | Alpha Supervision | 핵심 특징 |
|--------|--------------|-------------------|-----------|
| Original 3DGS | ❌ None | ❌ None | 전체 이미지 L1+SSIM |
| GaussianObject | ✅ BCE | ✅ BCE | 명시적 alpha loss |
| LGM | ✅ MSE | ✅ MSE | Shape 수렴 가속화 |
| Pose Splatter | ✅ Normalized L1 | - | Mouse 데이터 검증 |
| Object-Centric 2DGS | - | ✅ BG Penalty | 96% 모델 경량화 |
| GS-LRM/FaceLift | ❌ None | ❌ None | RGB loss only |

**핵심 발견**:
- **Intersection (GT ∩ Alpha) 방식을 사용하는 논문 없음**
- Object-centric 방법들은 **Alpha Supervision Loss**를 별도로 추가
- Rendered alpha로 loss masking하는 방식은 문헌에서 찾기 어려움

---

## Key Papers

### LGM (ECCV 2024 Oral)

```
L_rgb = L_MSE(I_rgb, I_rgb^GT) + λ·L_LPIPS(I_rgb, I_rgb^GT)
L_α = L_MSE(I_α, I_α^GT)  ← Alpha Supervision
```

- Alpha Supervision: **MSE** (rendered α vs GT mask)
- 목적: faster convergence of the shape
- MSE가 BCE보다 gradient 안정적

### Pose Splatter (NeurIPS 2025)

```
L_mask = Σ|pred - gt|·mask / Σmask  (정규화)
```

- **Mouse/rat 데이터**에서 검증
- 마스크 픽셀 수로 정규화 → 작은 전경 편향 방지

### Object-Centric 2DGS (2025)

```
L_bg = mean(rendered_α · (1 - GT_mask))
```

- Background penalty: 배경 opacity 억제
- 결과: 96% 모델 크기 감소, 71% 학습 속도 향상

### Nerfstudio (Composite Loss)

```
pred_composite = pred_rgb · α + bg · (1 - α)
gt_composite = gt_rgb · mask + bg · (1 - mask)
L = MSE(pred_composite, gt_composite)
```

- 암묵적 alpha supervision
- White background에 적합

---

## Recommended Configuration (FaceLift Mouse)

```yaml
# P0: GT Mask + Alpha Supervision (LGM + Pose Splatter)
training:
  losses:
    mask_mode: gt              # GT mask로 RGB loss 제한
    normalize_by_mask: true    # Pose Splatter: 작은 전경 필수
    alpha_loss_weight: 0.1     # LGM: alpha supervision
    alpha_loss_type: mse       # BCE보다 안정적
```

---

## References

| Paper | Venue | Link |
|-------|-------|------|
| LGM | ECCV 2024 | [arxiv](https://arxiv.org/abs/2402.05054) |
| Pose Splatter | NeurIPS 2025 | [arxiv](https://arxiv.org/html/2505.18342v1) |
| GaussianObject | SIGGRAPH Asia 2024 | [arxiv](https://arxiv.org/abs/2402.10259) |
| Object-Centric 2DGS | 2025 | [arxiv](https://arxiv.org/html/2501.08174v2) |

---

## Related Documents

- [ALPHA_MASK_COMPLETE_GUIDE](./ALPHA_MASK_COMPLETE_GUIDE.md) - 기존 마스크 가이드
- [Mask Experiment Priority](../../practical/experiments/Mask_Experiment_Priority.md) - 실험 우선순위
- [Config: mask_exp](../../../configs/mouse/mask_exp/) - 실험 설정

---

*Literature Review v1.0 | 2026-01-24*
