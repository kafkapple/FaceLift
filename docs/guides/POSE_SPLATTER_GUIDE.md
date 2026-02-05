# Pose Splatter Integration Guide

> **목적**: FaceLift GS-LRM과 Pose Splatter (NeurIPS 2025) 비교 실험
> **Status**: Active

---

## 1. 개요

Pose Splatter (NeurIPS 2025, arXiv:2505.18342)와의 공정한 비교를 위한 설정 가이드.

### Pose Splatter 논문 결과 (Reference)

| 모델 | PSNR | SSIM | IoU |
|------|------|------|-----|
| 5cam baseline | - | 0.9548 | 0.7883 |
| 5cam soft_keypoint | - | **0.9671** | **0.8287** |
| 3cam (최소) | - | - | 0.8289 |

---

## 2. 데이터 Split

### Pose Splatter 프로토콜
- **Temporal 1:1:1 Split**: 연속된 1/3씩 분할
- **목적**: Temporal leakage 방지

### Split 생성

```bash
# Pose Splatter 스타일 (1:1:1 temporal)
python -m mouse_extensions.preprocessing.split_generator \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --strategy temporal \
    --ratios 0.333 0.333 0.334

# 80/10/10 (FaceLift 기본)
python -m mouse_extensions.preprocessing.split_generator \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --strategy random \
    --ratios 0.8 0.1 0.1
```

### 현재 데이터셋

| 데이터셋 | Split | 샘플 수 |
|----------|-------|---------|
| M5t | temporal 1:1:1 | 1198/1198/1204 |
| M5t2 | temporal 8:1:1 | 2880/360/360 |

---

## 3. 평가 메트릭

### MetricsComputer 클래스

```python
from mouse_extensions.evaluation.metrics import MetricsComputer

metrics = MetricsComputer(device='cuda')
results = metrics.compute_all(pred_image, gt_image, pred_mask, gt_mask)
# {'psnr': float, 'ssim': float, 'lpips': float, 'l1': float, 'iou': float}
```

### WandB 로깅 키

| 키 | 내용 |
|----|------|
| test/psnr | Test PSNR |
| test/ssim | Test SSIM |
| test/lpips | Test LPIPS |
| test/iou | Test IoU |

---

## 4. 실험 설정

### View Reconstruction (5-view)

```yaml
# configs/mouse/M5t_E1_5v_recon.yaml
dataset:
  name: mouse_M5t
  num_views: 5
  random_view_selection: true
validation:
  metrics: [psnr, ssim, lpips, l1, iou]
```

### Novel View Synthesis (NVS)

뷰 5를 holdout하여 일반화 능력 평가:
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 --holdout-views 5
```

---

## 5. 실행 방법

### 학습

```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d M5t -e E1_5v_recon
```

### Test 평가 (수동)

```bash
python -m mouse_extensions.scripts.evaluate_test \
    --checkpoint /path/to/best.pt \
    --config configs/mouse/M5t_E1_5v_recon.yaml \
    --output-dir outputs/test_M5t_E1
```

---

## 6. Pose Splatter Loss (선택적)

```yaml
training:
  losses:
    l1_loss_weight: 0.5        # Pose Splatter 스타일 L1
    iou_loss_weight: 0.1       # IoU loss
    masked_l1_loss: true
    normalize_l1_by_mask: true
```

**기본값**: 비활성 (기존 실험 호환)

---

## 7. 모듈 구조

```
mouse_extensions/
├── preprocessing/
│   ├── split_generator.py     # Split 생성
│   └── split_verifier.py      # Split 검증
├── evaluation/
│   └── metrics.py             # MetricsComputer
└── scripts/
    └── evaluate_test.py       # Test 평가
```

---

## 8. 체크리스트

### 실험 전
- [ ] M5t 데이터셋 확인
- [ ] split.json 검증 (temporal 1:1:1)
- [ ] WandB 프로젝트: FaceLift-Mouse

### 실험 후
- [ ] Test 메트릭 기록
- [ ] Pose Splatter 결과와 비교

---

*FaceLift Mouse Extension | Pose Splatter Integration*
