# Pose Splatter Integration Guide

> **목적**: FaceLift GS-LRM과 Pose Splatter (NeurIPS 2025) 비교 실험
> **Status**: Active

---

## 1. 개요

Pose Splatter (NeurIPS 2025, arXiv:2505.18342)와의 공정한 비교를 위한 설정 가이드.

> **⚠️ 데이터 출처 주의**:
> PS 논문은 **자체 녹화한 Duke 데이터**를 사용합니다 (1536×2048, 30fps, 324K frames, 28cm 플라스틱 실린더).
> 우리 데이터(DANNCE/MAMMAL `markerless_mouse_1`, 1152×1024 원본)와 **완전히 별개**입니다.
> 본 비교에서는 PS 코드를 우리 M5 데이터에 적용(`m5_baseline_gs`)하여 동일 조건에서 비교합니다.
> PS 논문의 수치(PSNR ~33.5)는 full-image PSNR(85% white BG 포함)이며, 우리의 PSNR_fg(FG-masked)와 **직접 비교 불가**합니다.

### Pose Splatter 논문 결과 (Reference — 별도 Duke 데이터, full-image PSNR)

| 모델 | PSNR (full-image) | SSIM | IoU |
|------|:------------------:|:----:|:---:|
| 6cam (Mouse, Table 2a) | **33.5** | **0.989** | **0.868** |
| 5cam baseline | — | 0.9548 | 0.7883 |
| 5cam soft_keypoint | — | **0.9671** | **0.8287** |
| 3cam (최소) | — | — | 0.8289 |
| 6cam Rat (Rat7M) | 26.9 | 0.975 | 0.797 |
| Cross-species (Mouse→Rat) | 25.1 | — | — |

> **⚠️ Metric Protocol 차이**:
> - PS 논문 PSNR = **full-image** (white BG ~85-90% 포함 → PSNR 팽창)
> - 우리 PSNR_fg = **FG-masked** (foreground pixels only → 실제 reconstruction quality)
> - PSNR 33.5 (PS paper) vs 13.78 (our fair eval) = **19.7 dB gap**, 이 중 ~15-20 dB은 metric 차이
> - IoU는 유사 정의라 비교 가능: 0.868 (paper) vs 0.846 (fair eval) = comparable
> - **상세 분석**: `FL_vs_PS_comparison.md` §2.4 참조

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
