# Pose Splatter Metrics Integration

## Overview

FaceLift에 Pose Splatter (NeurIPS 2025) 호환 메트릭과 train/val/test split을 통합했습니다.

## Pose Splatter 논문 (NeurIPS 2025)
- **Split**: Temporal consecutive 1/3 (train:val:test = 1:1:1)
- **Metrics**: IoU, L1, PSNR, SSIM (no LPIPS)
- **결과**: Mouse PSNR 29.0, SSIM 0.982, IoU 0.760, L1 0.632
- **평가**: 5cam 학습 → 1 holdout view NVS 평가

## 구현 내용

### 1. Split Generator (`split_generator.py`)

```bash
# Pose Splatter 스타일 (1:1:1 temporal)
python -m mouse_extensions.preprocessing.split_generator \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --strategy temporal \
    --ratios 0.333 0.333 0.334 \
    --holdout_views 5

# 기존 80/10/10 random
python -m mouse_extensions.preprocessing.split_generator \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --strategy random \
    --ratios 0.8 0.1 0.1
```

**출력:**
- `data_mouse_train.txt`, `data_mouse_val.txt`, `data_mouse_test.txt`
- `split.json` (메타데이터)

### 2. L1 Loss (Pose Splatter 스타일)

```yaml
# configs/experiments/pose_splatter_eval.yaml
training:
  losses:
    # 기존 loss (기본값 유지)
    l2_loss_weight: 1.0
    lpips_loss_weight: 0.1
    ssim_loss_weight: 0.0

    # Pose Splatter 추가 loss (기본값: 비활성)
    l1_loss_weight: 0.0        # 0.0 = 비활성, 예: 0.5 = 활성
    iou_loss_weight: 0.0       # 0.0 = 비활성, 예: 0.1 = 활성
    
    # L1 옵션
    masked_l1_loss: true       # Pose Splatter: 마스크 적용
    normalize_l1_by_mask: true # Pose Splatter: 마스크 영역으로 정규화
    
    # IoU 옵션
    iou_bg_threshold: 0.1      # 배경 판별 threshold
```

### 3. Validation/Test Metrics

**자동 로깅 (WandB):**
- `val/psnr`, `val/ssim`, `val/lpips`, `val/l1`, `val/mask_iou`
- `test/psnr`, `test/ssim`, `test/lpips`, `test/l1`, `test/mask_iou`
- `final/test_psnr`, `final/test_ssim`, `final/test_l1`, `final/test_iou`

**Test 종료 시 비교 테이블 출력:**
```
Comparison with Pose Splatter (NeurIPS 2025):
------------------------------------------------------------
Metric       |       Ours | Pose Splatter |       Diff
------------------------------------------------------------
PSNR         |      27.50 |         29.00 |      -1.50
SSIM         |     0.9750 |        0.9820 |    -0.0070
IoU          |     0.7800 |        0.7600 |    +0.0200
L1           |     0.5500 |        0.6320 |    -0.0820
------------------------------------------------------------
```

### 4. Test Dataloader 설정

```yaml
# 기존 설정에 추가
validation:
  enabled: true
  test_enabled: true  # Test dataloader 활성화
```

또는

```yaml
test:
  enabled: true  # Test 평가 활성화
```

## 파일 위치

| 파일 | 역할 |
|------|------|
| `mouse_extensions/preprocessing/split_generator.py` | Split 생성 |
| `gslrm/model/utils_metrics.py` | `compute_l1()` 추가 |
| `gslrm/model/gslrm.py` | L1, IoU loss 계산 |
| `mouse_extensions/validation/validator.py` | Validation L1 추가 |
| `mouse_extensions/scripts/test_evaluation_extension.py` | Test L1 + 비교 테이블 |
| `train_gslrm.py` | test_dataloader 초기화 |

## 기존 실험과의 호환성

- **기본값**: `l1_loss_weight=0.0`, `iou_loss_weight=0.0` (비활성)
- **기존 config**: 변경 없이 그대로 동작
- **새 실험**: yaml에서 weight 설정하여 활성화

## Usage Example

```bash
# 1. Split 생성
python -m mouse_extensions.preprocessing.split_generator \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --strategy temporal \
    --ratios 0.333 0.333 0.334

# 2. 학습 (L1 loss 포함)
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d M5 -e pose_splatter_eval

# 3. 학습 종료 시 자동으로 test 평가 + Pose Splatter 비교 출력
```

---

*Created: 2026-01-29*
*FaceLift GS-LRM Mouse Extension*
