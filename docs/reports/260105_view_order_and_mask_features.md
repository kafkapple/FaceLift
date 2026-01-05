# 260105: 뷰 순서 일관성 및 Mask 기능 분석

**날짜:** 2026-01-05
**주제:** RandomViewDataset vs MouseViewDataset 비교, Mask 시각화 기능
**상태:** ✅ 분석 완료, 구현됨

---

## 핵심 요약

| 문제 | 원인 | 해결책 |
|------|------|--------|
| 출력 흐릿함 | RandomViewDataset의 뷰 순서 랜덤화 | `use_mouse_dataset: true` |
| Loss 감소 안됨 | Mask 미적용, 배경(95%)이 loss 지배 | `masked_l2_loss: true` |
| 시각화 부족 | Error heatmap/mask_iou 미구현 | gslrm.py 업데이트 |

---

## 1. 뷰 순서 일관성 문제

### 1.1 문제 현상
- 학습이 진행될수록 출력이 점점 **흐릿해짐**
- Loss는 감소하지만 시각적 품질 저하

### 1.2 근본 원인: RandomViewDataset의 랜덤 뷰 샘플링

**RandomViewDataset (FaceLift 원본)**:
```python
# gslrm/data/dataset.py:147
input_indices = random.sample(all_indices, self.num_input_views)

# gslrm/data/dataset.py:229
image_choices = random.sample(candidates, num_to_select)
```

**문제**: 매 step마다 뷰 순서가 랜덤하게 변경됨
```
Step 1: index[0,1,2,3,4,5] → cameras[3,1,5,0,2,4]  # cam_3이 index 0
Step 2: index[0,1,2,3,4,5] → cameras[1,4,2,5,0,3]  # cam_1이 index 0
Step 3: index[0,1,2,3,4,5] → cameras[5,2,0,3,1,4]  # cam_5가 index 0
```

### 1.3 왜 Human FaceLift에서는 문제 없었나?

| | Human (Objaverse) | Mouse (MAMMAL) |
|---|-------------------|----------------|
| 카메라 배치 | **균등** 60° 간격 turntable | **불균등** 각도 |
| 뷰 수 | 32개 (많은 중복) | 6개 (적음) |
| 랜덤 샘플링 영향 | **낮음** (어떤 뷰든 비슷) | **높음** (뷰마다 크게 다름) |

**결론**: Human 데이터는 균등 배치라서 랜덤 샘플링해도 일관성 유지. Mouse는 불균등이라 치명적.

### 1.4 Plücker Ray와 뷰 순서

GS-LRM은 **Plücker ray 좌표**를 사용:
```
Plücker ray = (ray_origin, ray_direction × ray_origin)
```

- 각 픽셀의 3D 위치는 Plücker ray로 인코딩
- **같은 index가 다른 카메라**를 가리키면 → 모델이 평균 학습 → 흐릿

### 1.5 해결책: MouseViewDataset

**MouseViewDataset**:
```python
# gslrm/data/mouse_dataset.py
# 항상 고정된 순서: [0, 1, 2, 3, 4, 5]
view_indices = list(range(self.num_views))  # 랜덤 없음!
```

**Config 설정**:
```yaml
mouse:
  use_mouse_dataset: true  # ⭐ 핵심! MouseViewDataset 사용
```

---

## 2. Mask 적용 Loss 문제

### 2.1 문제 현상
- Loss가 감소하지 않거나 매우 느리게 감소
- 모델이 흰색 배경만 출력

### 2.2 원인: 배경이 Loss 지배

Mouse 이미지 특성:
```
배경 (흰색): ~95%
전경 (마우스): ~5%
```

**일반 L2 Loss**:
```python
loss = MSE(rendering, target)  # 배경 95%가 loss 지배!
```

→ 모델은 흰색 출력이 가장 쉬운 최적화

### 2.3 해결책: Masked Loss

```python
# gslrm/model/gslrm.py
def _compute_l2_loss(self, rendering, target, mask=None):
    if use_mask and mask is not None:
        mask_binary = (mask > 0.5).float()
        num_valid = mask_binary.sum().clamp(min=1.0)
        squared_error = (rendering - target) ** 2
        masked_error = squared_error * mask_binary
        return masked_error.sum() / (num_valid * 3)  # 전경만!
```

**Config 설정**:
```yaml
training:
  dataset:
    remove_alpha: false      # Alpha 채널 유지 (mask로 사용)
  losses:
    masked_l2_loss: true     # ⭐ 전경에만 L2 loss
    masked_ssim_loss: true   # ⭐ 전경에만 SSIM loss
    background_loss_weight: 0.1  # 배경 흰색 유도
```

---

## 3. Mask 시각화 기능

### 3.1 Error Heatmap
```
색상 범위: Blue (낮음) → Green → Yellow → Red (높음)
배경 영역: Gray (0.3, 0.3, 0.3)
정규화: error / 0.3 (0.3 이상은 빨간색)
```

### 3.2 5행 시각화 그리드
```
Row 1: GT 이미지
Row 2: Rendered 이미지
Row 3: GT + Mask overlay
Row 4: Rendered + Mask overlay
Row 5: Error heatmap (전경만)
```

### 3.3 mask_iou 메트릭
```python
# GT mask vs 예측 mask (배경색 거리로 계산)
color_distance = (rendering - bg_color).abs().mean(dim=1)
pred_mask = (color_distance > threshold).float()
iou = intersection / union
```

---

## 4. 논문 대비 Loss 설정 비교

| Loss | 논문 | 기존 설정 | 수정 설정 |
|------|------|----------|----------|
| L2 (MSE) | 1.0 | 1.0 | 1.0 |
| Perceptual (VGG) | 0.5 | 0.0 | 0.0 ⚠️ Mouse 도메인 문제 |
| LPIPS | 0.0 | 0.0 | 0.0 |
| SSIM | 0.0 | 0.5 | 0.5 |
| **Masked L2** | ❌ | ❌ | ✅ **추가** |
| **Masked SSIM** | ❌ | ❌ | ✅ **추가** |
| **Background** | ❌ | ❌ | ✅ **추가** (0.1) |

**Note**: Perceptual Loss는 VGG(ImageNet/Human 학습)가 Mouse 도메인에서 gradient explosion 발생 → 비활성화 권장

---

## 5. 체크리스트

### 필수 설정
```yaml
mouse:
  use_mouse_dataset: true      # ⭐ 뷰 순서 고정

training:
  dataset:
    remove_alpha: false        # ⭐ Mask 유지
  losses:
    masked_l2_loss: true       # ⭐ 전경 loss
    masked_ssim_loss: true
    background_loss_weight: 0.1
    lpips_loss_weight: 0.0     # Mouse 도메인 비활성화
    perceptual_loss_weight: 0.0
```

### W&B 로깅 항목
- `train/loss`, `train/l2_loss`, `train/ssim_loss`
- `train/background_loss`, `train/mask_iou`
- `val/psnr`, `val/ssim`, `val/lpips`, `val/mask_iou`
- 시각화: GT vs Rendered + Error heatmap

---

## 관련 파일

| 파일 | 역할 |
|------|------|
| `gslrm/data/dataset.py` | RandomViewDataset (원본 FaceLift) |
| `gslrm/data/mouse_dataset.py` | MouseViewDataset (고정 뷰 순서) |
| `gslrm/model/gslrm.py` | Masked loss, Error heatmap 구현 |
| `train_gslrm.py` | Dataset 선택 로직 |

---

*🤖 Generated with Claude Code*
