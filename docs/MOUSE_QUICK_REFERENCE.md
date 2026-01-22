# FaceLift Mouse Extension - Quick Reference

> **Last Updated**: 2026-01-22
> **Full Documentation**: Obsidian `30_Projects/_CODES/code_Face_Lift/docs/`

---

## 1. Preprocessing (통합 스크립트)

### 1.1 기본 명령어

```bash
cd /home/joon/dev/FaceLift

# 프리셋 목록 확인
python -m mouse_extensions.preprocessing.preprocess --list-presets

# D7.1 (권장)
python -m mouse_extensions.preprocessing.preprocess --preset D7.1 --input-dir /home/joon/data/raw/markerless_mouse_1_nerf --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1

# D9 (원본 해상도 1152x1024, A6000+ 필요)
python -m mouse_extensions.preprocessing.preprocess --preset D9 --input-dir /home/joon/data/raw/markerless_mouse_1_nerf --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D9
```

### 1.2 프리셋 목록

| Preset | 특징 | GPU 메모리 |
|--------|------|-----------|
| **D7.1** | PP-centered shift, fx=549 | ~16GB |
| D8 | Precision homography + skew | ~16GB |
| D9 | 원본 해상도 (1152x1024) | ~48GB+ |

### 1.3 Split 재생성 + Config 업데이트

```bash
# Split 생성 + D9 config 자동 업데이트 (권장)
python -m mouse_extensions.preprocessing.generate_split \
    --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
    --val-ratio 0.1 \
    --update-config D9

# Split만 재생성 (config 수동 업데이트 필요)
python -m mouse_extensions.preprocessing.generate_split \
    --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
    --val-ratio 0.1

# 다른 split 비율/시드
python -m mouse_extensions.preprocessing.generate_split \
    --data-dir /home/joon/data/preprocessed/FaceLift_mouse/D9 \
    --val-ratio 0.2 --seed 123 --update-config D9
```

**출력:**
- `data_mouse_train.txt` - train split
- `data_mouse_val.txt` - val split
- `data_mouse_all.txt` - 전체 목록
- `configs/datasets/D9.yaml` 자동 업데이트 (--update-config 사용 시)

---

## 2. Inference (추론)

### 2.1 Pretrained 체크포인트 (기본)

```bash
cd /home/joon/dev/FaceLift

# Pretrained 체크포인트 위치
ls checkpoints/gslrm/ckpt_0000000000021125.pt  # 3.7GB

# 기본 추론 (6-view 샘플에서)
CUDA_VISIBLE_DEVICES=4 python inference_mouse.py \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1/train/000000 \
    --checkpoint checkpoints/gslrm/ckpt_0000000000021125.pt \
    --config configs/gslrm.yaml \
    --output_dir outputs/inference/sample_000000
```

### 2.2 학습된 체크포인트 사용

```bash
# D7.1 학습 체크포인트
CUDA_VISIBLE_DEVICES=4 python inference_mouse.py \
    --sample_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1/train/000000 \
    --checkpoint checkpoints/gslrm/D7_1_E1_1_paper_random/ckpt_step_2300.pt \
    --config checkpoints/gslrm/D7_1_E1_1_paper_random/config.yaml \
    --output_dir outputs/inference/D7_1_E1_1
```

### 2.3 Single Image (Zero123++)

```bash
# 이미지 한 장으로 6-view 생성 후 추론
python inference_mouse.py \
    --input_image examples/mouse.png \
    --use_zero123pp \
    --checkpoint checkpoints/gslrm/ckpt_0000000000021125.pt \
    --output_dir outputs/single_image/
```

### 2.4 D7_1 샘플 데이터 위치

```
/home/joon/data/preprocessed/FaceLift_mouse/D7_1/
├── train/
│   ├── 000000/           # 첫번째 샘플
│   │   ├── images/
│   │   │   ├── cam_000.png  # RGBA (mask in alpha)
│   │   │   ├── cam_001.png
│   │   │   └── ...
│   │   └── opencv_cameras.json
│   ├── 000001/
│   └── ...
├── val/
└── data_mouse_train.txt
```

---

## 3. Training (학습)

```bash
cd /home/joon/dev/FaceLift

# 기본 학습
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 train_gslrm.py -d D7_1 -e E1_1_paper_random

# Background 실행
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 train_gslrm.py -d D7_1 -e E1_1_paper_random > logs/d7_1_e1_1.log 2>&1 &
```

---

## 4. Turntable Visualization

### 4.1 기본 설정 (YAML)

```yaml
visualization:
  turntable:
    num_views: 64           # 그리드 뷰 수
    resolution: 384
    elevation: 20           # 카메라 고도 (도)
    radius: 2.7             # 중심으로부터 거리
    trajectory_mode: "turntable"  # 기본값
```

### 4.2 Trajectory 모드

| Mode | 설명 |
|------|------|
| `turntable` | 고정 elevation, 360도 수평 회전 (기본) |
| `spiral` | Elevation 점진 변화 + 회전 |
| `figure8` | 8자 패턴 |
| `arc` | 고정 azimuth, elevation 변화 |
| `dataset_cameras` | 데이터셋 6개 카메라 간 보간 이동 |

### 4.3 Dataset Camera Trajectory (a)

**6개 카메라를 지정 순서로 순회 (실제 extrinsic 기반)**

```python
from gslrm.model.gaussians_renderer import render_dataset_trajectory

# 카메라 순서: 1→3→5→0→4→2→1 (360도)
frames, segments = render_dataset_trajectory(
    gaussians,
    dataset_c2ws,        # [6, 4, 4] - 데이터셋 카메라 pose
    dataset_fxfycxcy,    # [6, 4]
    num_views=150,       # 프레임 수
    camera_order=[1, 3, 5, 0, 4, 2],  # 방문 순서
    loop=True,           # 처음으로 돌아감
    show_overlay=True    # "Cam 1 -> Cam 3" 텍스트
)
```

### 4.4 Object-Centered 360 Rotation (b)

**생쥐 중심 기준 360도 회전 (turntable 기본)**

```yaml
visualization:
  turntable:
    trajectory_mode: "turntable"
    num_views: 120
    elevation: 15         # 카메라 고도
    radius: 2.7           # 생쥐 중심 ~ 카메라 거리
```

또는 코드:

```python
from gslrm.model.gaussians_renderer import render_turntable

frames = render_turntable(
    gaussians,
    h=384, w=384,
    num_views=120,
    radius=2.7,           # 중심으로부터 거리
    elevation=15,         # 고도
    trajectory_mode="turntable"
)
```

---

## 5. Mask Mode Analysis

### 5.1 체크포인트 기반 분석 (필수!)

```bash
cd /home/joon/dev/FaceLift

# GPU 4-7 (A6000) 사용 필수 (Blackwell 호환 안됨)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint checkpoints/_temp/D7_1_E1_1_paper_random/ckpt_0000000000002300.pt \
    --config configs/mouse/D7_1_E1_1_paper_random.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_analysis/D7_1_E1_1 \
    --sample_idx 0
```

### 5.2 주의사항

- **시뮬레이션 금지**: GT mask + noise 사용 금지
- **실제 데이터**: 체크포인트 + 실제 생쥐 데이터 + 가우시안 렌더링
- **GPU 호환성**: A6000 (GPU 4-7) 사용, Blackwell (GPU 0-3) 미지원

### 5.3 출력물 (WandB Style)

```
alpha_analysis/{experiment}/
├── figures/
│   └── gt_vs_pred_wandb.png   # WandB 스타일 시각화
└── alpha_analysis_report.md
```

**시각화 레이아웃:**
- Row 1: GT RGB (6개 뷰)
- Row 2: Rendered RGB
- Row 3: GT + GT Mask overlay (녹색)
- Row 4: Rendered + Alpha Mask overlay (파란색)
- Row 5: Error heatmap

### 5.4 Alpha Threshold 비교 시각화 ⭐

**기존 체크포인트에서 threshold만 변경해 foreground mask 비교:**

```bash
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/quick_alpha/ckpt_0000000000000500.pt \
    --config configs/mouse/D7_1_quick_alpha.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir alpha_threshold_comparison \
    --thresholds 0.3 0.5 0.7 0.9 0.99
```

**출력물:**
```
alpha_threshold_comparison/
├── threshold_comparison.png  # 각 threshold별 mask 시각화
└── report.md
```

### 5.5 mask_mode vs alpha_loss (중요!)

| 설정 | 역할 | 독립성 |
|------|------|--------|
| **mask_mode** | L2 loss 계산 영역 (none/gt/alpha) | ✓ |
| **alpha_loss_weight** | rendered alpha → GT mask supervision | ✓ |

- `mask_mode: gt` + `alpha_loss: 0` → GT mask 영역에서만 L2, alpha supervision 없음
- `mask_mode: none` + `alpha_loss: 0.1` → 전체 이미지 L2 + alpha supervision

### 5.6 빠른 마스크 모드 학습

**mask_mode=gt로 빠르게 학습 (500 steps, 모듈화 config):**

```bash
# E_quick_alpha: mask_mode=gt, alpha_loss_weight=0.0, 500 steps
# D7_1 데이터셋으로 테스트
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E_quick_alpha \
    > logs/d7_1_quick_alpha.log 2>&1 &

# D9 데이터셋으로 테스트
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D9 -e E_quick_alpha \
    > logs/d9_quick_alpha.log 2>&1 &
```

**실험 config 위치**: `configs/experiments/E_quick_alpha.yaml`

### 5.7 빠른 마스크 모드 비교 (다중, 실험적)

```bash
# 모든 마스크 모드 비교 (100 step 학습)
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.scripts.analysis.quick_mask_mode_compare \
    --config configs/mouse/D7_1_E1_1_paper_random.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir mask_mode_comparison \
    --mask_modes none gt alpha \
    --steps 100
```

---

## 6. GPU Reference

| GPU Index | Model | Compute Cap | PyTorch 호환 |
|-----------|-------|-------------|--------------|
| 0-3 | RTX PRO 6000 Blackwell | sm_120 | X |
| 4-7 | RTX A6000 | sm_86 | O |

---

## 7. Checkpoints (체크포인트)

### Pretrained (기본)
```
checkpoints/gslrm/ckpt_0000000000021125.pt  # 3.7GB, Objaverse 학습
```

### 학습된 (최근)
```bash
find checkpoints/_temp -name '*.pt' -mtime -7 | sort
```

---

## 8. Troubleshooting

### CUDA 호환성 오류
```
CUDA capability sm_120 is not compatible
```
-> GPU 4-7 (A6000) 사용: `CUDA_VISIBLE_DEVICES=4`

### Validation 폴더 오류
```
FileNotFoundError: experiments/validation/...
```
-> 2026-01-22 수정됨

---

*Updated: 2026-01-22*
