> **Navigation**: [← MoC Dashboard](./reports/00_MoC_INDEX.md) | [Quick Reference](./MOUSE_QUICK_REFERENCE.md) | [Preprocessing](./PREPROCESSING_REGISTRY.md) | [Loss Formula](./GS-LRM_Loss_Formula.md)

# Alpha Mask & Loss Complete Guide

**Version**: 1.1
**Date**: 2026-01-22
**Author**: Claude Code Analysis

---

## 1. Overview: 마스킹 시스템 전체 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FaceLift Mouse 마스킹 시스템                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐               │
│  │  mask_mode  │     │ alpha_loss  │     │  mask_iou   │               │
│  │   (학습용)   │     │   (손실함수)  │     │  (모니터링)  │               │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘               │
│         │                   │                   │                      │
│         ▼                   ▼                   ▼                      │
│  ┌─────────────────────────────────────────────────────┐               │
│  │                Training Pipeline                    │               │
│  │                                                     │               │
│  │  1. masked_l2_loss: RGB loss × mask                │               │
│  │  2. alpha_loss: BCE(rendered_alpha, gt_mask)       │               │
│  │  3. mask_iou: 예측 마스크 품질 모니터링              │               │
│  └─────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. mask_mode: 마스킹 전략 선택

### 2.1 설정 위치
```yaml
training:
  losses:
    mask_mode: alpha  # none | alpha | gt
```

### 2.2 모드별 동작

| Mode | masked_l2_loss 마스크 | 특징 | 권장 상황 |
|------|----------------------|------|----------|
| **none** | 전체 이미지 | 마스크 미사용 | 안정적 학습, 배경도 학습 |
| **alpha** | rendered_alpha > threshold | 학습 중 동적 변화 | 일반화 목적 |
| **gt** | GT mask 직접 사용 | Oracle (완벽한 마스크) | 최고 품질 목표 |

### 2.3 코드 경로

```python
# gslrm/model/gslrm.py line ~650
if mask_mode == "none":
    mask = None  # 전체 이미지 사용
elif mask_mode == "alpha":
    mask = (rendered_alpha > threshold).float()
elif mask_mode == "gt":
    mask = gt_mask
```

---

## 3. Alpha Loss: Rendered Alpha 학습

### 3.1 개념

**Alpha Loss**는 3D Gaussian의 불투명도(opacity)가 GT 마스크와 일치하도록 학습시킵니다.

```
rendered_alpha: Gaussian Splatting이 렌더링한 누적 불투명도
gt_mask: Ground Truth 전경/배경 마스크

목표: rendered_alpha ≈ gt_mask
```

### 3.2 수식

**BCE (Binary Cross Entropy)**:
```
L_alpha = BCE(α, m)
        = -1/N Σ[m·log(α) + (1-m)·log(1-α)]

where:
  α = rendered_alpha (예측)
  m = gt_mask (정답)
  N = 픽셀 수
```

**직관적 해석**:
- 전경 픽셀 (m=1): α가 1에 가까워지도록 학습
- 배경 픽셀 (m=0): α가 0에 가까워지도록 학습

### 3.3 설정

```yaml
training:
  losses:
    alpha_loss_weight: 0.3    # 가중치 (0.0 = 비활성)
    alpha_loss_type: bce      # bce | mse | dice | focal
```

### 3.4 가중치별 효과

| Weight | 효과 | RGB 품질 | 마스크 품질 | 권장 |
|--------|------|----------|------------|------|
| 0.0 | Alpha 학습 안함 | ★★★ | ✗ | GT mask 사용 시 |
| 0.1 | 약한 학습 | ★★★ | ★ | ⚠️ Alpha 반전 위험 |
| 0.3 | 균형 | ★★☆ | ★★ | ✅ Conservative |
| 0.5 | 강한 학습 | ★☆☆ | ★★★ | ✅ Aggressive |

### 3.5 Alpha Inversion 문제

**증상**: 배경에서 alpha가 높고, 전경에서 낮음 (반전)

**원인**:
```
alpha_loss_weight가 너무 낮을 때 (0.1):
  RGB loss (1.0) >> alpha_loss (0.1)
  → 모델이 RGB 맞추기에 집중
  → 배경 흰색 맞추려고 배경에도 Gaussian 배치
  → 배경 alpha ↑
```

**해결**: `alpha_loss_weight ≥ 0.3` 권장

---

## 4. Alpha Mask Threshold: 이진화 기준

### 4.1 역할

rendered_alpha를 이진 마스크로 변환할 때 사용:
```python
pred_mask = (rendered_alpha > threshold).float()
```

### 4.2 설정

```yaml
training:
  losses:
    alpha_mask_threshold: 0.7  # 0.0 ~ 1.0
```

### 4.3 threshold별 효과

| Threshold | Pred Mask 특성 | 장점 | 단점 |
|-----------|---------------|------|------|
| 0.3 (낮음) | 넓은 영역 전경 | 세밀한 경계 보존 | 배경 침범 |
| 0.5 (중간) | 균형 | 일반적 상황 | 극단 케이스 취약 |
| 0.7 (높음) | 좁은 영역 전경 | 배경 침범 방지 | 얇은 구조 손실 |

### 4.4 시각적 효과

```
Threshold 0.3:        Threshold 0.7:
┌─────────────┐       ┌─────────────┐
│  ████████   │       │    ████     │
│ ██████████  │       │   ██████    │
│██████████████│       │  ████████   │
│ ██████████  │       │   ██████    │
│  ████████   │       │    ████     │
└─────────────┘       └─────────────┘
  (넓고 흐릿)           (좁고 선명)
```

---

## 5. Masked L2 Loss: RGB 손실 계산

### 5.1 개념

전경 영역에서만 RGB 손실 계산:
```python
loss = ||mask * (pred - gt)||² / num_foreground_pixels
```

### 5.2 설정

```yaml
training:
  losses:
    masked_l2_loss: true      # 마스크 적용 여부
    l2_loss_weight: 1.0       # 가중치
```

### 5.3 마스크 소스 (mask_mode에 따라)

| mask_mode | masked_l2_loss 마스크 소스 |
|-----------|--------------------------|
| none | 전체 이미지 (마스크 없음) |
| alpha | (rendered_alpha > threshold) |
| gt | GT mask 직접 |

### 5.4 핵심 코드

```python
# gslrm/model/gslrm.py line ~550
def _compute_masked_l2_loss(self, rendering, target, mask):
    mask_binary = (mask > 0.5).float()
    num_valid = mask_binary.sum().clamp(min=1.0)
    squared_error = (rendering - target) ** 2
    masked_error = squared_error * mask_binary
    return masked_error.sum() / (num_valid * 3)  # 3 = RGB channels
```

---

## 6. mask_iou: 모니터링 지표

### 6.1 정의

```
IoU = Intersection / Union
    = (pred_mask ∩ gt_mask) / (pred_mask ∪ gt_mask)
```

### 6.2 중요: 학습에 사용되지 않음!

mask_iou는 **모니터링 전용 지표**:
- WandB에 기록
- 학습 진행 상황 파악
- **손실 함수에 포함되지 않음**

### 6.3 해석

| mask_iou | 의미 | 상태 |
|----------|------|------|
| < 0.3 | 예측 마스크 불량 | ⚠️ 문제 |
| 0.3 ~ 0.6 | 보통 | 🔄 학습 중 |
| 0.6 ~ 0.8 | 양호 | ✅ 정상 |
| > 0.8 | 우수 | 🌟 최적 |

---

## 7. 실험 설정 매트릭스

### 7.1 핵심 파라미터 조합

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

★ Conservative (threshold=0.7, loss=0.3)
★★ Best Performance (GT mask, no alpha loss)
```

### 7.2 현재 실험 결과 요약

| Config | mask_mode | threshold | alpha_loss | mask_iou | l2_loss |
|--------|-----------|-----------|------------|----------|---------|
| E2_2_gt_mask | gt | 0.5 | 0.0 | **0.82** | **0.0069** |
| E5_3_aggressive | alpha | 0.5 | 0.5 | 0.60 | 0.0097 |
| E5_2_conservative | alpha | 0.7 | 0.3 | 0.53 | 0.0068 |
| E4_2_alpha_loss | alpha | 0.5 | 0.1 | 0.02 | 0.0055 |

### 7.3 권장 설정

**최고 품질 (GT mask 사용 가능 시)**:
```yaml
mask_mode: gt
alpha_loss_weight: 0.0
```

**일반화 목적 (Alpha mask 필요 시)**:
```yaml
mask_mode: alpha
alpha_mask_threshold: 0.7
alpha_loss_weight: 0.3
```

---

## 8. 문제 해결 가이드

### 8.1 배경이 전경으로 인식됨

**증상**: Pred mask가 배경까지 확장
**원인**: threshold 낮음 + alpha_loss 약함
**해결**:
```yaml
alpha_mask_threshold: 0.7  # ↑
alpha_loss_weight: 0.3     # ↑
```

### 8.2 Alpha Inversion (반전)

**증상**: 배경 alpha > 전경 alpha
**원인**: alpha_loss_weight 너무 낮음 (0.1)
**해결**:
```yaml
alpha_loss_weight: 0.3+    # 최소 0.3
```

### 8.3 전경 가장자리 손실

**증상**: 귀, 꼬리 등 세부 구조 누락
**원인**: threshold 너무 높음
**해결**:
```yaml
alpha_mask_threshold: 0.5  # ↓
```

### 8.4 Floater (공중 점)

**증상**: Novel view에서 물체 주변 점들
**원인**: Opacity 정규화 부재
**해결**:
```yaml
alpha_loss_weight: 0.5
opacity_reg_weight: 0.01
```

---

## 9. 시각화 도구

### 9.1 Threshold 시각화 (학습 없이)

```bash
python scripts/alpha_threshold_visualize.py \
    -c checkpoints/gslrm/D7_1_E5_2/iter_00000801 \
    --thresholds 0.3 0.5 0.7
```

### 9.2 Parameter Sweep

```bash
python scripts/alpha_sweep.py \
    --thresholds 0.5 0.6 0.7 \
    --alpha_losses 0.1 0.2 0.3 \
    --max_steps 300
```

---

## 10. Ghost Gaussian Pruning (실험적)

### 10.1 개념

**Ghost Gaussian**: Object 경계 외부에 존재하는 불필요한 Gaussian

특징:
- 낮은 opacity (< 0.05)
- GT mask 외부에 위치
- Alpha "bleed" 현상 유발 (mask 확장 원인)

### 10.2 GhostGaussianRegularizer

```python
# mouse_extensions/model/gaussian_pruning.py
class GhostGaussianRegularizer:
    """GT 배경 영역에서 alpha 패널티 적용"""

    def compute_loss(self, rendered_alpha, gt_mask):
        # 배경 영역 (gt_mask=0)에서 alpha → 0 유도
        background_mask = 1.0 - gt_mask
        ghost_penalty = (rendered_alpha * background_mask).mean()
        return ghost_penalty
```

### 10.3 설정

```yaml
training:
  losses:
    ghost_reg_weight: 0.01  # 0.0 = 비활성 (기본)
```

---

## 11. Adaptive Threshold (계획됨)

### 11.1 개념

| 방식 | 설명 |
|------|------|
| **Static** | threshold = 0.5 (고정) |
| **Adaptive** | threshold = f(iteration, loss, metrics) |

### 11.2 구현 방향

1. **Validation IoU 기반**: IoU 낮으면 threshold 조정
2. **Ghost area ratio 기반**: ghost 영역 비율로 동적 조정
3. **Multi-threshold 시각화**: 최적값 탐색 후 고정

### 11.3 현재 상태

> ⚠️ **미구현**: 현재는 Static threshold만 지원
>
> Multi-threshold 시각화 도구로 최적값 탐색 후 고정 권장:
> ```bash
> python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
>     --thresholds 0.3 0.5 0.7 0.9
> ```

---

## 12. 데이터 마스크 자동 생성

### 12.1 auto_generate_mask

RGB 이미지에서 배경(흰색)을 감지하여 마스크 자동 생성:

```python
# gslrm/data/mouse_dataset.py
if self.auto_generate_mask and image_np.shape[2] == 3:
    threshold = self.mask_threshold / 255.0  # default: 250
    is_background = np.all(image_np > threshold, axis=2)
    alpha = (~is_background).astype(np.float32)
    image_np = np.concatenate([image_np, alpha[:, :, None]], axis=2)
```

### 12.2 설정

```yaml
dataset:
  auto_generate_mask: true   # RGB → RGBA 자동 변환
  mask_threshold: 250        # 배경 감지 threshold (0-255)
```

### 12.3 마스크 생성 로직

```
RGB 픽셀 (r, g, b) where all > 250/255 → 배경 (alpha=0)
RGB 픽셀 (r, g, b) where any ≤ 250/255 → 전경 (alpha=1)
```

---

## 13. Quick Reference

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
| alpha_threshold_visualize.py | scripts/ | Threshold 시각화 |

---

*FaceLift Mouse Project | Alpha Mask Complete Guide v1.1*
