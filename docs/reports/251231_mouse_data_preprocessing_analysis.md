# Mouse Data Preprocessing 분석 및 wandb 로깅 설정

**날짜**: 2025-12-31
**주제**: Mouse 데이터셋 전처리 상태 확인 및 wandb 로깅 설정 점검
**목적**: 6뷰 생쥐 크기 불일치 문제 해결을 위한 전처리 코드 위치 파악 및 적용 상태 확인

---

## 1. 데이터셋 Pixel Level 전처리 상태

### 현재 상태
- **Camera 레벨 정규화**: 구현됨
  - `normalize_camera_distance()`: 카메라 거리를 2.7로 정규화
  - `normalize_cameras_to_z_up()`: Z-up 좌표계로 정규화

- **Pixel/Image 레벨 정규화**: **구현되어 있으나 데이터셋에 미적용!**
  - 이미지는 단순히 `/ 255.0`으로 0-1 범위 변환만 수행
  - 객체 중앙 이동(centering) 및 크기 정규화(scaling)는 **적용되지 않음**

### 확인 방법
```python
# opencv_cameras.json에서 전처리 여부 확인
# uniform_scale_info가 있으면 전처리 적용됨
cat sample/opencv_cameras.json | jq '.uniform_scale_info'
```

### 현재 데이터셋 상태
| 데이터셋 | uniform_scale_info | 전처리 상태 |
|----------|-------------------|------------|
| `data_mouse_local/` | NOT FOUND | **미적용** |
| `data_mouse_local_rgba/` | NOT FOUND | **미적용** |

---

## 2. 전처리 스크립트 위치 및 기능

### 2.1 Center Align (`scripts/preprocess_center_align_all_views.py`)
```bash
python scripts/preprocess_center_align_all_views.py \
    --input_dir data_mouse_local \
    --output_dir data_mouse_centered \
    --target_ratio 0.7
```

**기능**:
- 각 뷰의 객체 bounding box 탐지 (alpha 채널 또는 흰색 배경 기반)
- **모든 뷰에서 최대 객체 크기 기준으로 scale 계산** → 클리핑 방지
- 각 뷰별로 객체 중심을 이미지 중앙으로 이동
- target_ratio (기본 0.7): 객체가 이미지의 70% 차지하도록 scaling

**핵심 로직** (`preprocess_center_align_all_views.py:233-235`):
```python
# MAX object size across all views → guarantees no clipping
target_size = image_size * target_object_ratio
ref_scale = target_size / max_obj_size
```

### 2.2 Uniform Scale (`scripts/preprocess_uniform_scale.py`)
```bash
python scripts/preprocess_uniform_scale.py \
    --input_dir data_mouse_centered \
    --output_dir data_mouse_uniform \
    --target_ratio 0.6
```

**기능**:
- 각 뷰별로 객체 크기 측정 (픽셀 카운트 기반, sqrt 사용)
- 각 뷰의 객체가 target_ratio (기본 60%)를 차지하도록 **개별** scaling
- 카메라 파라미터는 변경하지 않음 (순수 이미지 기반 접근)

**핵심 로직** (`preprocess_uniform_scale.py:89-91`):
```python
# sqrt로 면적 비율을 선형 크기 비율로 변환
area_ratio = pixel_count / total_pixels
size_ratio = np.sqrt(area_ratio)
```

### 2.3 권장 전처리 파이프라인
```bash
# Step 1: Center align (모든 뷰에서 최대 크기 기준)
python scripts/preprocess_center_align_all_views.py \
    --input_dir data_mouse_local_rgba \
    --output_dir data_mouse_centered \
    --target_ratio 0.7

# Step 2: Uniform scale (각 뷰별 크기 균일화)
python scripts/preprocess_uniform_scale.py \
    --input_dir data_mouse_centered \
    --output_dir data_mouse_uniform \
    --target_ratio 0.6
```

---

## 3. 회색 배경 Mask 문제

### 원인
```python
# mouse_dataset.py:573-577
threshold = self.mask_threshold / 255.0  # 250/255 = 0.98
is_background = np.all(image_np > threshold, axis=2)
```

- **모든 RGB 채널이 0.98 초과**해야만 배경으로 인식
- 회색 배경 (예: RGB = 0.8)은 threshold보다 작아서 **foreground로 오인**
- 결과: 전체 이미지가 foreground(초록색)로 표시

### 해결책
1. **Config에서 threshold 조정**: `mouse.mask_threshold: 200` (0.78)
2. **데이터 전처리**: 배경을 순수 흰색(255)으로 변환
3. **RGBA 사용**: alpha 채널이 있으면 auto_generate_mask가 skip됨

---

## 4. wandb 로깅 설정 분석

### 4.1 로깅 주기 (configs/mouse_gslrm_normalized_synthetic.yaml)
| 항목 | 설정값 | 설명 |
|------|--------|------|
| `logging.print_every` | 10 | 콘솔 출력 주기 |
| `logging.vis_every` | 100 | 시각화 저장 주기 |
| `logging.wandb.log_every` | 10 | **메트릭 wandb 로깅 주기** |
| `validation.val_every` | 200 | **validation 실행 주기** |

### 4.2 wandb 로깅 구현 위치

**메트릭 로깅** (`train_gslrm.py:722-738`):
```python
if self.fwdbwd_pass_step % self.config.training.logging.wandb.log_every == 0:
    log_dict = {
        "metrics/iter": self.fwdbwd_pass_step,
        "metrics/lr": self.optimizer.param_groups[0]["lr"],
        ...
    }
    log_dict.update({"train/" + k: v for k, v in loss_name2value})
    wandb.log(log_dict, step=self.fwdbwd_pass_step)
```

**시각화 로깅** (`train_gslrm.py:774-776`):
```python
# vis_every 주기마다 train 시각화 로깅
self._log_visuals_to_wandb(vis_dir, prefix="train")
```

**Validation 로깅** (`train_gslrm.py:926-938`):
```python
# val_every 주기마다 실행
wandb_log_val_metrics = {
    "val/psnr": ...,
    "val/ssim": ...,
    "val/lpips": ...,
    "val/mask_iou": ...,
}
wandb.log(wandb_log_val_metrics, step=self.fwdbwd_pass_step)

# Validation 시각화도 로깅
self._log_visuals_to_wandb(val_vis_dir, prefix="val")
```

### 4.3 wandb UI 구조
```
images/
├── train/         # vis_every마다 로깅
│   ├── supervision_*
│   ├── input_*
│   └── turntable_*
└── val/           # val_every마다 로깅
    ├── supervision_*
    └── gt_vs_pred_*

train/             # log_every마다 로깅
├── loss
├── l2
└── ssim

val/               # val_every마다 로깅
├── psnr
├── ssim
├── lpips
└── mask_iou
```

---

## 5. Mask/Background ROI 비교

### 디버그 스크립트
```bash
python scripts/debug_gslrm_mask_visualization.py \
    --config configs/mouse_gslrm_local_rtx3060.yaml \
    --checkpoint checkpoints/gslrm/ckpt_xxx.pt \
    --num_samples 3 \
    --output_dir experiments/debug_mask
```

### 출력물
- `sample_*_gt_vs_rendered.png`: GT/Rendered 이미지 및 마스크 비교
- `sample_*_mask_diff.png`: 마스크 차이 시각화
  - Green = 일치
  - Red = 누락 (GT foreground, Rendered background)
  - Blue = 추가 (GT background, Rendered foreground)

### wandb mask_iou 로깅
`gslrm/model/gslrm.py:377-420`에서 `_compute_mask_iou()` 계산 후 `val/mask_iou`로 로깅

---

## 6. Action Items

### 필수 작업
- [ ] `data_mouse_local_rgba/`에 centering + scaling 전처리 적용
- [ ] 전처리 후 데이터셋으로 config 업데이트
- [ ] 회색 배경 샘플에 대한 mask_threshold 조정 또는 배경 순백 변환

### 권장 작업
- [ ] wandb 로깅이 실제로 되는지 확인 (학습 시작 후 wandb UI 체크)
- [ ] validation 시각화가 `images/val/` 섹션에 나타나는지 확인

---

## 7. 핵심 교훈

1. **전처리 스크립트는 존재하지만 데이터셋에 적용되지 않았음**
   - `uniform_scale_info` 필드로 적용 여부 확인 가능

2. **wandb 로깅 주기 분리**:
   - 메트릭: `log_every` (빈번)
   - 시각화: `vis_every` (덜 빈번)
   - Validation: `val_every` (가장 덜 빈번)

3. **Mask 생성 threshold 문제**:
   - 회색 배경은 threshold 250으로 감지 불가
   - RGBA 데이터 사용 권장
