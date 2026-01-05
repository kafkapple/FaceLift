---
date: 2025-12-30
context_name: "2_Research"
tags: [ai-assisted, gslrm, plucker-coordinates, view-order, masked-loss, ssim, bug-fix]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# 2025-12-30 연구 일지: 뷰 순서 문제 및 Masked Loss 분석

> **통합 문서**: 이전 2개 문서를 핵심 내용 중심으로 통합
> - `251230_research_gslrm_view_order_fix.md` (삭제됨)
> - `241230_masked_ssim_negative_loss_analysis.md` (삭제됨, 날짜 오류 수정)

---

## 1. 뷰 순서 문제

### 1.1 문제 현상
- wandb의 `val/gt_vs_pred`에서 뷰 인덱스가 매번 다르게 표시
- 학습이 진행될수록 출력이 점점 **흐릿해짐**

### 1.2 근본 원인: RandomViewDataset의 뷰 순서 랜덤화

```
Step 1: 뷰 조합 [0, 2, 3, 4, 5, 1] → 인덱스 1의 Plücker: cam_2 기준
Step 2: 뷰 조합 [1, 3, 0, 5, 2, 4] → 인덱스 1의 Plücker: cam_3 기준 (다른 위치!)
```

**결과**: 모델이 일관된 위치 매핑을 학습 불가 → **"평균" 출력 = 흐릿함**

### 1.3 인간 vs 마우스 데이터 차이

| | 인간 데이터 | 마우스 데이터 |
|---|------------|-------------|
| 카메라 배치 | 균등 60° 간격 | 불균등 각도 |
| 랜덤 샘플링 영향 | 낮음 | **높음** |

### 1.4 해결책

**Config 수정**:
```yaml
mouse:
  use_mouse_dataset: true  # ← 핵심! 고정 뷰 순서 사용
```

**네이밍 개선**: `normalize_cameras` → `use_mouse_dataset`

---

## 2. Masked Loss 구현

### 2.1 배경

마우스 이미지 특성:
- **배경**: ~95% (흰색)
- **전경**: ~5%

→ 일반 Loss는 배경이 지배 → "흰색 출력" 학습

### 2.2 구현 내용

**Masked L2 Loss**:
```python
if use_mask and mask is not None:
    mask_binary = (mask > 0.5).float()
    squared_error = (rendering - target) ** 2
    masked_error = squared_error * mask_binary
    return masked_error.sum() / (num_valid * 3)
```

**자동 Mask 생성** (`mouse_dataset.py`):
```python
threshold = self.mask_threshold / 255.0  # default: 250
is_background = np.all(image_np > threshold, axis=2)
alpha = (~is_background).astype(np.float32)
```

**Config**:
```yaml
training:
  losses:
    masked_l2_loss: true
    masked_ssim_loss: true
mouse:
  auto_generate_mask: true
  mask_threshold: 250
```

---

## 3. Masked SSIM 음수 Loss 문제 (해결됨)

### 3.1 문제 현상

```
step 36: ssim_loss: -0.133  # 비정상! (SSIM > 1.0)
```

### 3.2 원인: 수치 불안정성

Masked 영역(neutral_value=0.5)에서:
- 분산이 매우 작음 (상수 영역)
- 부동소수점 오차로 음수 분산 발생
- SSIM > 1.0 계산 → Loss < 0

### 3.3 해결책

```python
# gslrm/model/utils_losses.py
ssim_value = self.ssim_module(x, y)
ssim_value = torch.clamp(ssim_value, 0.0, 1.0)  # [0, 1] 범위 제한
return 1.0 - ssim_value
```

---

## 4. AMP Scaler 상태 오류 (해결됨)

### 4.1 문제
```
RuntimeError: unscale_() has already been called on this optimizer
```

### 4.2 원인
Gradient skip 시 `scaler.update()` 미호출

### 4.3 해결
```python
# scaler.update()는 항상 호출 (step skip 여부와 무관)
if not skip_optimizer_step:
    self.scaler.step(self.optimizer)
self.scaler.update()  # ← 항상 호출!
```

---

## 5. 핵심 교훈

1. **뷰 순서 일관성**: Plücker 좌표 기반 모델에서 필수
2. **Masked Loss**: 배경 비율이 높은 데이터에서 효과적
3. **수치 안정성**: 상수 영역에서 통계 기반 메트릭 주의
4. **AMP 상태 관리**: `update()`는 skip 시에도 호출 필요

---

## 변경된 파일

| 파일 | 변경 내용 |
|------|----------|
| `train_gslrm.py` | use_mouse_dataset 조건, scaler.update() 수정 |
| `gslrm/model/gslrm.py` | Masked L2/SSIM loss 구현 |
| `gslrm/model/utils_losses.py` | SSIM 클램핑 추가 |
| `gslrm/data/mouse_dataset.py` | 자동 mask 생성 기능 |

---

*🤖 Generated with Claude Code - 2025-12-30*
*📝 통합 정리: 2026-01-05*
