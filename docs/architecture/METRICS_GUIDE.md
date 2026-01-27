# Metrics Guide

> **Navigation**: [← MoC](../../00_MoC_INDEX.md) | [Practical](../../practical/) | [Theory](../)

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
