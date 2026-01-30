# Pose Splatter 비교 실험 가이드

## 개요

FaceLift GS-LRM과 Pose Splatter (NeurIPS 2025, arXiv:2505.18342) 모델 비교를 위한 설정 및 실험 가이드.

---

## 1. 데이터 Split 설정

### Pose Splatter 프로토콜
- **Temporal 1:1:1 Split**: 연속된 1/3씩 train/val/test 분할
- **목적**: Temporal leakage 방지 (시간적으로 인접한 프레임이 다른 split에 섞이지 않음)

### FaceLift 구현

**통합 전처리 스크립트 옵션:**
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /path/to/raw \
    --output-dir /path/to/M5 \
    --temporal-variant \
    --split-strategy temporal \
    --split-ratios 0.333 0.333 0.334
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--temporal-variant` | Split 생성 활성화 | False |
| `--split-strategy` | temporal (연속) / random (셔플) | temporal |
| `--split-ratios` | train/val/test 비율 | 0.333 0.333 0.334 |
| `--holdout-views` | NVS 평가용 제외 뷰 | None |

### 현재 데이터셋

| 데이터셋 | Split | 샘플 수 | 위치 |
|----------|-------|---------|------|
| M5t | temporal 1:1:1 | 3600 (1198/1198/1204) | `/home/joon/data/preprocessed/FaceLift_mouse/M5t/` |

---

## 2. 평가 메트릭

### 공통 메트릭 (Pose Splatter 논문 기준)

| 메트릭 | 설명 | 구현 |
|--------|------|------|
| **PSNR** | Peak Signal-to-Noise Ratio | `metrics.py:compute_psnr()` |
| **SSIM** | Structural Similarity | `metrics.py:compute_ssim()` |
| **LPIPS** | Learned Perceptual Image Patch Similarity | `metrics.py:compute_lpips()` |
| **L1** | Mean Absolute Error | `metrics.py:compute_l1()` |
| **IoU** | Intersection over Union (마스크) | `metrics.py:compute_iou()` |

### MetricsComputer 클래스

```python
from mouse_extensions.evaluation.metrics import MetricsComputer

metrics = MetricsComputer(device='cuda')
results = metrics.compute_all(pred_image, gt_image, pred_mask, gt_mask)
# results: {'psnr': float, 'ssim': float, 'lpips': float, 'l1': float, 'iou': float}
```

---

## 3. 실험 설정

### 3.1 View Reconstruction (모든 뷰 사용)

**설정:**
- Input: 5 views (랜덤 선택)
- Target: 동일 5 views
- Split: temporal 1:1:1

```yaml
# configs/mouse/M5t_E1_5v_recon.yaml
dataset:
  name: mouse_M5t
  num_views: 5
  random_view_selection: true
  
validation:
  metrics: [psnr, ssim, lpips, l1, iou]
```

### 3.2 Novel View Synthesis (NVS) - View Holdout

**설정:**
- Input: 4 views (뷰 5 제외)
- Target: 뷰 5 (holdout)
- 목적: 보지 않은 뷰에서의 일반화 능력 평가

```bash
# Holdout 데이터셋 생성
python -m mouse_extensions.preprocessing.preprocess \
    --preset M5 \
    --input-dir /path/to/raw \
    --output-dir /path/to/M5_nvs \
    --temporal-variant \
    --holdout-views 5
```

---

## 4. 실행 방법

### 4.1 학습

```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 train_gslrm.py \
    -d M5t -e E1_5v_recon
```

### 4.2 테스트 평가

학습 종료 시 자동 실행됨 (`train_gslrm.py:1372`).

수동 실행:
```bash
python -m mouse_extensions.scripts.test_evaluation_extension \
    --checkpoint /path/to/best_checkpoint.pt \
    --config configs/mouse/M5t_E1_5v_recon.yaml \
    --output-dir outputs/test_M5t_E1
```

### 4.3 WandB 로깅

| 키 | 내용 |
|----|------|
| `test/psnr` | Test set PSNR |
| `test/ssim` | Test set SSIM |
| `test/lpips` | Test set LPIPS |
| `test/iou` | Test set IoU |
| `test/turntable_*` | Turntable 렌더링 이미지 |
| `test/gt_vs_pred_*` | GT vs Prediction 비교 이미지 |

---

## 5. 모듈 구조

```
mouse_extensions/
├── preprocessing/
│   ├── preprocess.py           # 통합 전처리 (split 옵션 포함)
│   ├── split_generator.py      # Split 생성 (temporal/random)
│   ├── split_verifier.py       # Split 검증
│   └── _archive/               # 중복 모듈 (참조용)
│       ├── create_temporal_split.py
│       └── split_manager.py
├── evaluation/
│   └── metrics.py              # MetricsComputer 클래스
├── scripts/
│   └── test_evaluation_extension.py  # Test 평가 스크립트
└── validation/
    └── validator.py            # Validation 로직
```

---

## 6. Pose Splatter 논문 참조 결과

| 모델 | PSNR | SSIM | IoU | 비고 |
|------|------|------|-----|------|
| Pose Splatter (5cam) | - | 0.9671 | 0.8287 | Soft keypoint |
| Pose Splatter (3cam) | - | - | 0.8289 | 최소 카메라 |

**주의**: 직접 비교를 위해서는 동일 데이터셋, 동일 split 사용 필수.

---

## 7. 체크리스트

### 실험 전
- [ ] M5t 데이터셋 존재 확인
- [ ] split.json 검증 (temporal 1:1:1)
- [ ] WandB 프로젝트 설정 (FaceLift-Mouse)

### 실험 중
- [ ] Validation 메트릭 모니터링
- [ ] Turntable 시각화 확인

### 실험 후
- [ ] Test 메트릭 기록
- [ ] Pose Splatter 결과와 비교
- [ ] 결과 문서화

---

*Created: 2026-01-30*
*Updated: 2026-01-30*
