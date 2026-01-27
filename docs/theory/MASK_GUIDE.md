> Parent: [[INDEX]] > Theory

# Mask Guide: Alpha Mask & Loss Complete Reference

**Version**: 2.0
**Date**: 2026-01-27
**Merged from**: ALPHA_MASK_COMPLETE_GUIDE.md, Mask_Literature_Review.md, Research_Note_Mask_Binarization_Issue.md

---

## Table of Contents

1. [Overview](#1-overview)
2. [mask_mode: 마스킹 전략 선택](#2-mask_mode-마스킹-전략-선택)
3. [Alpha Loss: Rendered Alpha 학습](#3-alpha-loss-rendered-alpha-학습)
4. [Alpha Mask Threshold: 이진화 기준](#4-alpha-mask-threshold-이진화-기준)
5. [Masked L2 Loss: RGB 손실 계산](#5-masked-l2-loss-rgb-손실-계산)
6. [mask_iou: 모니터링 지표](#6-mask_iou-모니터링-지표)
7. [Literature Review](#7-literature-review)
8. [Mask Binarization Issue](#8-mask-binarization-issue)
9. [실험 설정 매트릭스](#9-실험-설정-매트릭스)
10. [문제 해결 가이드](#10-문제-해결-가이드)
11. [Quick Reference](#11-quick-reference)

---

## 1. Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FaceLift Mouse 마스킹 시스템                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │  mask_mode  │     │ alpha_loss  │     │  mask_iou   │               │
│  │   (학습용)   │     │   (손실함수)  │     │  (모니터링)  │               │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘               │
│         │                   │                   │                      │
│         ▼                   ▼                   ▼                      │
│  ┌─────────────────────────────────────────────────────┐               │
│  │                Training Pipeline                    │               │
│  │  1. masked_l2_loss: RGB loss × mask                │               │
│  │  2. alpha_loss: BCE(rendered_alpha, gt_mask)       │               │
│  │  3. mask_iou: 예측 마스크 품질 모니터링              │               │
│  └─────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 핵심 개념 구분 (혼동 주의)

```
mask_mode: gt        → RGB loss를 GT mask 영역으로 제한
alpha_loss_weight    → 별도 loss로 rendered α가 GT mask에 수렴하도록 유도

★ 권장 조합: mask_mode: gt + alpha_loss_weight: 0.1
  - RGB: GT mask 영역에서만 계산 (안정적)
  - Alpha: GT mask로 수렴 유도 (shape 학습)

❌ mask_mode: alpha → rendered alpha를 loss mask로 사용 → 악순환 위험
   (문헌에서 이 방식 사용 사례 없음)
```

---

## 2. mask_mode: 마스킹 전략 선택

### 설정
```yaml
training:
  losses:
    mask_mode: alpha  # none | alpha | gt | composite
```

### 모드별 동작

| Mode | masked_l2_loss 마스크 | 특징 | 권장 상황 |
|------|----------------------|------|----------|
| **none** | 전체 이미지 | 마스크 미사용 | 안정적 학습, 배경도 학습 |
| **alpha** | rendered_alpha > threshold | 학습 중 동적 변화 | ⚠️ 비권장 (악순환 위험) |
| **gt** | GT mask 직접 사용 | Oracle (완벽한 마스크) | ✅ 최고 품질 목표 |
| **composite** | 배경 합성 후 전체 비교 | Splatfacto-W 방식 | 흰 배경 데이터 |

### 코드 경로
```python
# gslrm/model/gslrm.py
if mask_mode == "none":    mask = None
elif mask_mode == "alpha": mask = (rendered_alpha > threshold).float()
elif mask_mode == "gt":    mask = gt_mask
elif mask_mode == "composite":
    pred_comp = pred * α + bg * (1 - α)
    gt_comp = gt * mask + bg * (1 - mask)
```

---

## 3. Alpha Loss: Rendered Alpha 학습

Alpha Loss는 3D Gaussian의 불투명도(opacity)가 GT 마스크와 일치하도록 학습시킵니다.

### 수식

**BCE (Binary Cross Entropy)**:
```
L_alpha = -1/N Σ[m·log(α) + (1-m)·log(1-α)]
```

**MSE** (LGM 방식, 권장):
```
L_alpha = 1/N Σ(α - m)²
```

### 설정
```yaml
training:
  losses:
    alpha_loss_weight: 0.3    # 0.0 = 비활성
    alpha_loss_type: mse      # bce | mse | dice | focal
```

### 가중치별 효과

| Weight | RGB 품질 | 마스크 품질 | 권장 |
|--------|----------|------------|------|
| 0.0 | ★★★ | ✗ | GT mask 사용 시 |
| 0.1 | ★★★ | ★ | ⚠️ Alpha 반전 위험 |
| 0.3 | ★★☆ | ★★ | ✅ Conservative |
| 0.5 | ★☆☆ | ★★★ | ✅ Aggressive |

---

## 4. Alpha Mask Threshold: 이진화 기준

```python
pred_mask = (rendered_alpha > threshold).float()
```

```yaml
training:
  losses:
    alpha_mask_threshold: 0.7  # 0.0 ~ 1.0
```

| Threshold | 장점 | 단점 |
|-----------|------|------|
| 0.3 (낮음) | 세밀한 경계 보존 | 배경 침범 |
| 0.5 (중간) | 일반적 상황 | 극단 케이스 취약 |
| 0.7 (높음) | 배경 침범 방지 | 얇은 구조 손실 |

---

## 5. Masked L2 Loss: RGB 손실 계산

```python
# gslrm/model/gslrm.py
def _compute_masked_l2_loss(self, rendering, target, mask):
    mask_binary = (mask > 0.5).float()
    num_valid = mask_binary.sum().clamp(min=1.0)
    squared_error = (rendering - target) ** 2
    masked_error = squared_error * mask_binary
    return masked_error.sum() / (num_valid * 3)
```

```yaml
training:
  losses:
    masked_l2_loss: true
    l2_loss_weight: 1.0
    normalize_by_mask: true    # Pose Splatter 방식 (권장)
```

---

## 6. mask_iou: 모니터링 지표

**모니터링 전용** (손실 함수에 포함되지 않음):

| mask_iou | 의미 | 상태 |
|----------|------|------|
| < 0.3 | 예측 마스크 불량 | ⚠️ |
| 0.3 ~ 0.6 | 보통 | 🔄 학습 중 |
| 0.6 ~ 0.8 | 양호 | ✅ |
| > 0.8 | 우수 | 🌟 |

---

## 7. Literature Review

### 논문별 마스크 처리 요약

| 방법론 | Mask in Loss | Alpha Supervision | 핵심 특징 |
|--------|--------------|-------------------|-----------|
| Original 3DGS | ❌ | ❌ | 전체 이미지 L1+SSIM |
| GaussianObject | ✅ BCE | ✅ BCE | 명시적 alpha loss |
| **LGM** (ECCV 2024) | ✅ MSE | ✅ **MSE** | Shape 수렴 가속화 |
| **Pose Splatter** (NeurIPS 2025) | ✅ Normalized L1 | - | Mouse 데이터 검증 |
| Object-Centric 2DGS | - | ✅ BG Penalty | 96% 모델 경량화 |
| Splatfacto-W | ✅ Composite | ✅ Strong α | 배경 분리 |

### 핵심 발견
- **Rendered alpha로 loss masking하는 논문 없음** (확장 위험)
- Object-centric 방법들은 **GT mask + Alpha Supervision Loss** 조합
- MSE가 BCE보다 gradient 안정적

### LGM 인용
> "We use simple MSE loss on both RGB and alpha images for faster convergence of the shape."

### Pose Splatter 인용
> "Normalized by the mask sum to prevent bias towards larger foregrounds."
- `normalize_by_mask: true` 중요. Mouse/rat 데이터에서 검증됨.

---

## 8. Mask Binarization Issue

### 문제

D4/D6-1 전처리에서 배경에 검은색 아티팩트 관찰. D7은 정상.

### 원인

| Dataset | Semi-transparent 픽셀 | 원인 |
|---------|----------------------|------|
| D4 | 4,827개 | 원본 마스크 비디오 anti-aliasing 보존 |
| D6-1 | 2,520개 | 동일 |
| **D7** | **0개** | 명시적 이진화 (`mask > 127 → 255, else → 0`) |

원본 마스크 비디오가 H.264 압축으로 edge smoothing 발생. D7만 `np.where(mask > 127, 255, 0)` 적용.

### 학습 영향
- **Loss 계산**: 영향 없음 (binary threshold 적용)
- **시각화**: D4/D6-1에서 "dirty" 배경
- **권장**: 신규 전처리 시 D7 방식 마스크 이진화 적용

---

## 9. 실험 설정 매트릭스

```
                     alpha_loss_weight
                    0.0      0.3      0.5
                 ┌────────┬────────┬────────┐
mask_mode   none │ Paper  │   -    │   -    │
            ─────├────────┼────────┼────────┤
            alpha│ E2_3   │ E5_2★  │ E5_3   │
            ─────├────────┼────────┼────────┤
            gt   │ E2_2★★ │   -    │   -    │
                 └────────┴────────┴────────┘
```

| Config | mask_mode | threshold | alpha_loss | mask_iou | l2_loss |
|--------|-----------|-----------|------------|----------|---------|
| E2_2_gt_mask | gt | 0.5 | 0.0 | **0.82** | **0.0069** |
| E5_2_conservative | alpha | 0.7 | 0.3 | 0.53 | 0.0068 |

---

## 10. 문제 해결 가이드

| 문제 | 원인 | 해결 |
|------|------|------|
| 배경이 전경으로 인식 | threshold 낮음 + alpha_loss 약함 | threshold→0.7, alpha_loss→0.3 |
| Alpha 반전 | alpha_loss_weight 너무 낮음 (0.1) | alpha_loss_weight ≥ 0.3 |
| 전경 가장자리 손실 | threshold 너무 높음 | threshold→0.5 |
| Floater (공중 점) | Opacity 정규화 부재 | alpha_loss→0.5, opacity_reg→0.01 |

---

## 11. Quick Reference

### 핵심 설정 요약

| 상황 | mask_mode | threshold | alpha_loss |
|------|-----------|-----------|------------|
| **최고 품질** | gt | - | 0.0 |
| **일반화** | alpha | 0.7 | 0.3 |
| **형상 우선** | alpha | 0.5 | 0.5 |
| **안정적** | none | - | 0.0 |

### 관련 파일

| 파일 | 위치 | 역할 |
|------|------|------|
| loss_extensions.py | mouse_extensions/model/ | Alpha loss 구현 |
| gslrm.py | gslrm/model/ | Masked L2, mask_iou |

## See Also

- [[GHOSTING_ANALYSIS]] - Ghosting과 마스크 관계
- [[FLOATER_ARTIFACT_ANALYSIS]] - Floater 분석
- [[PP_FX_MVG_ANALYSIS]] - PP/MVG 이론

---

*FaceLift Mouse | Mask Guide v2.0 | Last updated: 2026-01-27*
