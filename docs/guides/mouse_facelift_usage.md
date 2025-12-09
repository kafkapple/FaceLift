# Mouse-FaceLift Usage Guide

---
date: 2024-12-04
context_name: "2_Research"
tags: [ai-assisted, mouse-reconstruction, multi-view, 3d-reconstruction]
project: FaceLift
status: in-progress
generator: ai-assisted
generator_tool: claude-code
---

## Overview

Mouse-FaceLift adapts the FaceLift 3D reconstruction pipeline for mouse multi-view data.
This guide covers environment setup, data preprocessing, training, and inference.

## Quick Start (gpu05)

```bash
# 1. SSH to gpu05
ssh gpu05
cd /home/joon/FaceLift

# 2. Activate environment (conda activate만 하면 됨!)
conda activate mouse_facelift

# 3. Process mouse video data
python scripts/process_mouse_data.py \
    --video_dir /home/joon/data/markerless_mouse_1_nerf/videos_undist \
    --meta_dir /home/joon/data/markerless_mouse_1_nerf \
    --output_dir data_mouse \
    --num_samples 2000

# 4. Run overfitting test (verify setup)
python train_mouse.py --config configs/mouse_config.yaml --overfit 10

# 5. Full training (multi-GPU)
torchrun --nproc_per_node 4 train_mouse.py --config configs/mouse_config.yaml
```

---

## GPU05 환경 설정 (처음 1회)

### Step 1: Conda 환경 생성 (이미 완료됨)
```bash
# 환경이 이미 생성되어 있음. 확인:
conda env list | grep mouse_facelift
# 출력: mouse_facelift    /home/joon/anaconda3/envs/mouse_facelift
```

### Step 2: 환경 활성화
```bash
ssh gpu05
cd /home/joon/FaceLift
conda activate mouse_facelift   # CUDA/GCC 환경변수 자동 설정됨!
```

**참고**: 환경변수가 conda 환경에 영구 설정되어 있음
- 위치: `~/anaconda3/envs/mouse_facelift/etc/conda/activate.d/env_vars.sh`
- `conda activate` 시 CUDA 11.8, GCC-9 자동 설정
- `conda deactivate` 시 원래 환경으로 자동 복원

### 환경 활성화 확인
```bash
# 확인 방법:
echo $CUDA_HOME    # /usr/local/cuda-11.8
echo $CC           # /usr/bin/gcc-9
nvcc --version     # CUDA 11.8
python -c "import torch; print(torch.cuda.is_available())"  # True
```

---

## 데이터 전처리 (현재 마우스 데이터 기준)

### 현재 데이터 위치
```
/home/joon/data/markerless_mouse_1_nerf/
├── videos_undist/          # 6개 동기화된 비디오
│   ├── 0.mp4 (25.8MB)
│   ├── 1.mp4 (17.6MB)
│   ├── 2.mp4 (23.4MB)
│   ├── 3.mp4 (21.6MB)
│   ├── 4.mp4 (19.9MB)
│   └── 5.mp4 (25.0MB)
├── simpleclick_undist/     # 마스크 비디오
│   ├── 0.mp4 ~ 5.mp4
├── new_cam.pkl             # 카메라 캘리브레이션
└── keypoints2d_undist/     # 2D 키포인트 (선택)
```

### 전처리 실행
```bash
# gpu05에서 환경 활성화 후:
source activate_gpu05.sh

# 데이터 전처리 (약 2000개 샘플 추출)
python scripts/process_mouse_data.py \
    --video_dir /home/joon/data/markerless_mouse_1_nerf/videos_undist \
    --meta_dir /home/joon/data/markerless_mouse_1_nerf \
    --output_dir data_mouse \
    --num_samples 2000 \
    --image_size 512 \
    --num_views 6

# 출력 확인
ls data_mouse/
# data_mouse_train.txt, data_mouse_val.txt, sample_000000/, ...
```

### 출력 구조
```
data_mouse/
├── data_mouse_train.txt    # 학습 샘플 경로 목록
├── data_mouse_val.txt      # 검증 샘플 경로 목록
├── sample_000000/
│   ├── images/
│   │   ├── cam_000.png     # 512x512 RGBA
│   │   ├── cam_001.png
│   │   ├── cam_002.png
│   │   ├── cam_003.png
│   │   ├── cam_004.png
│   │   └── cam_005.png
│   └── opencv_cameras.json # 카메라 파라미터
├── sample_000001/
│   └── ...
└── ...
```

---

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Mouse-FaceLift 전체 파이프라인                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Single Image ──┬──► MVDiffusion (fine-tuned) ──► 6 Views ──► GSLRM ──► PLY │
│                 │                                  ↑                        │
│                 └──► Zero123++ (pretrained) ───────┘                        │
│                                                                             │
│  두 모델은 별도 학습 가능 (Stage 1: MVDiffusion, Stage 2: GSLRM)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 학습

### Stage 1: MVDiffusion Fine-tune (Single View → 6 Views)

```bash
# MVDiffusion 학습 (single GPU)
python train_diffusion.py --config configs/mouse_mvdiffusion.yaml

# MVDiffusion 학습 (multi GPU)
accelerate launch --num_processes 4 \
    train_diffusion.py --config configs/mouse_mvdiffusion.yaml
```

**Config**: `configs/mouse_mvdiffusion.yaml`
- `max_train_steps`: 30,000
- `learning_rate`: 5e-5
- `train_batch_size`: 4
- `gradient_accumulation_steps`: 4 (effective batch = 16)

### Stage 2: GSLRM Fine-tune (6 Views → 3D Gaussian)

#### Step 1: Overfitting 테스트 (필수 권장)
```bash
# 10개 샘플로 코드 정상 동작 확인
python train_mouse.py --config configs/mouse_config.yaml --overfit 10
```

**기대 결과**:
- Loss가 0에 가깝게 감소
- 입력 이미지가 완벽하게 복원됨
- 이것이 성공해야 전체 학습 진행

#### Step 2: 전체 학습

```bash
# 단일 GPU
python train_mouse.py --config configs/mouse_config.yaml

# 멀티 GPU (권장)
torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id ${RANDOM} --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    train_mouse.py --config configs/mouse_config.yaml
```

#### Step 3: 체크포인트에서 재개
```bash
python train_mouse.py --config configs/mouse_config.yaml \
    --load checkpoints/gslrm/mouse/
```

---

## 추론

### 옵션 1: Zero123++ (pretrained) + GSLRM

```bash
# Single image → Zero123++ → GSLRM → PLY/OBJ/Video
python inference_mouse.py \
    --input_image path/to/mouse.png \
    --use_zero123pp \
    --checkpoint checkpoints/gslrm/mouse/ \
    --output_dir outputs/
```

**특징**:
- Zero123++는 pretrained 모델 사용 (HuggingFace에서 자동 다운로드)
- 빠르게 테스트 가능

### 옵션 2: MVDiffusion (fine-tuned) + GSLRM

```bash
# Single image → MVDiffusion → GSLRM → PLY/OBJ/Video
python inference_mouse.py \
    --input_image path/to/mouse.png \
    --mvdiffusion_checkpoint checkpoints/experiments/train/mouse_mvdiffusion/pipeckpts \
    --checkpoint checkpoints/gslrm/mouse/ \
    --output_dir outputs/
```

**특징**:
- Mouse 데이터로 fine-tuned MVDiffusion 사용
- 더 정확한 multi-view 생성 기대

### 옵션 3: 6-view 샘플에서 직접 추론

```bash
# 이미 6개 뷰가 있는 경우
python inference_mouse.py \
    --sample_dir data_mouse/sample_000000 \
    --checkpoint checkpoints/gslrm/mouse/ \
    --output_dir outputs/
```

### 추론 파라미터

| 파라미터 | Zero123++ | MVDiffusion | 설명 |
|---------|-----------|-------------|------|
| `--*_steps` | 75 | 50 | Diffusion steps |
| `--*_guidance` | 4.0 | 3.0 | CFG guidance scale |
| `--seed` | 42 | 42 | Random seed |

### 출력 파일

```
outputs/{sample_name}/
├── gaussians.ply           # 3D Gaussian splat (Blender/MeshLab 호환)
├── mesh.obj                # Mesh (Poisson reconstruction)
├── turntable.mp4           # 360° 회전 비디오
├── render_view_*.png       # 각 뷰 렌더링
├── render_grid.png         # 6개 뷰 그리드
└── generated_views/        # MVDiffusion/Zero123++ 생성 이미지
    ├── view_00.png ~ view_05.png
```

---

## 설정 파일 (Config)

### Config 파일 비교표

| Config 파일 | 용도 | max_steps | warmup | LR | batch_size |
|-------------|------|-----------|--------|-----|------------|
| `mouse_config.yaml` | 기본 학습 (scratch) | **100,000** | 200 | 5e-5 | 2 |
| `mouse_config_finetune.yaml` | FaceLift pretrained fine-tune | **20,000** | 100 | 2e-5 | 2 |
| `mouse_config_debug.yaml` | 빠른 테스트 (~10-30분) | **1,000** | 50 | 1e-4 | 4 |

### 공통 모델 설정

```yaml
model:
  image_tokenizer:
    image_size: 512            # 입력 이미지 크기
    patch_size: 8              # ViT 패치 크기
    in_channels: 9             # 3 RGB + 3 direction + 3 Reference

  transformer:
    d: 1024                    # 히든 차원
    d_head: 64                 # 어텐션 헤드 차원
    n_layer: 24                # 트랜스포머 레이어 수

  gaussians:
    n_gaussians: 2             # 12288 (실제)
    sh_degree: 0
```

### 데이터 설정

```yaml
training:
  dataset:
    dataset_path: "data_mouse/data_mouse_train.txt"
    num_views: 6               # 총 6개 뷰
    num_input_views: 1         # 입력: 단일 뷰
    target_has_input: true     # 타겟에 입력 포함
    background_color: "white"
```

### Loss 가중치

```yaml
losses:
  l2_loss_weight: 1.0          # MSE 손실
  lpips_loss_weight: 0.5       # LPIPS 지각 손실
  perceptual_loss_weight: 0.5  # VGG 지각 손실
  ssim_loss_weight: 0.2        # 구조적 유사도 손실
  pixelalign_loss_weight: 0.0  # 비활성화
  pointsdist_loss_weight: 0.0  # 비활성화
```

### 체크포인트 설정

| Config | resume_ckpt | checkpoint_dir |
|--------|-------------|----------------|
| 기본 | `checkpoints/gslrm` | `checkpoints/gslrm/mouse` |
| fine-tune | `ckpt_0000000000021125.pt` (FaceLift pretrained) | `checkpoints/gslrm/mouse_finetune` |
| debug | `checkpoints/gslrm/stage_2` | `checkpoints/gslrm/mouse_debug` |

### Validation 설정

```yaml
validation:
  enabled: true                # 기본/fine-tune: true, debug: false
  val_every: 500               # 500 steps 마다 검증
  dataset_path: "data_mouse/data_mouse_val.txt"
```

### Inference 설정

```yaml
inference:
  enabled: false               # 현재 모든 config에서 비활성화
  output_dir: "experiments/inference/mouse"
```

### Mouse 특화 설정

```yaml
mouse:
  camera:
    num_views: 6
    camera_distance: 2.7

  augmentation:
    enabled: true              # debug에서는 false
    horizontal_flip: true
    brightness_range: [0.9, 1.1]
    contrast_range: [0.9, 1.1]
```

### 권장 사용 시나리오

| 시나리오 | Config 파일 | 설명 |
|----------|-------------|------|
| 빠른 테스트 | `mouse_config_debug.yaml` | 1000 steps, ~10-30분, wandb offline |
| Fine-tune | `mouse_config_finetune.yaml` | FaceLift pretrained → 20k steps |
| Full 학습 | `mouse_config.yaml` | scratch → 100k steps |

```bash
# 예시: Fine-tune 실행
python train_mouse.py --config configs/mouse_config_finetune.yaml

# 예시: Debug 모드 실행
python train_mouse.py --config configs/mouse_config_debug.yaml
```

---

## 문제 해결

### CUDA Out of Memory
```yaml
# batch_size 줄이기
training:
  dataloader:
    batch_size_per_gpu: 1
```

### CUDA 버전 불일치 에러
```bash
# 반드시 activate_gpu05.sh로 환경 활성화
source activate_gpu05.sh

# 확인
echo $CUDA_HOME  # /usr/local/cuda-11.8 이어야 함
```

### 학습이 수렴하지 않음
1. Overfitting 테스트 먼저 실행
2. `checkpoints/*/data_examples/` 에서 데이터 시각화 확인
3. 카메라 파라미터 검증
4. 학습률 낮추기

---

## Git 동기화 워크플로우

### 로컬에서 코드 수정 후
```bash
cd /home/joon/dev/FaceLift
git add -A
git commit -m "feat(mouse): description"
git push
```

### gpu05에서 학습 전
```bash
ssh gpu05
cd /home/joon/FaceLift
git pull
source activate_gpu05.sh
```

### gpu05에서 학습 후
```bash
# 체크포인트 커밋 (선택)
git add checkpoints/ outputs/
git commit -m "chore: add training checkpoints"
git push

# 로컬에서 pull
cd /home/joon/dev/FaceLift
git pull
```

---

## 파일 참조

### GSLRM (Stage 2: 6 Views → 3D)

| 파일 | 용도 |
|------|------|
| `train_mouse.py` | GSLRM 학습 스크립트 |
| `inference_mouse.py` | 통합 추론 스크립트 (Zero123++/MVDiffusion + GSLRM) |
| `gslrm/data/mouse_dataset.py` | GSLRM용 PyTorch Dataset |
| `configs/mouse_config.yaml` | GSLRM 기본 학습 (100k steps) |
| `configs/mouse_config_finetune.yaml` | GSLRM Fine-tune (20k steps) |
| `configs/mouse_config_debug.yaml` | GSLRM 디버그 (1k steps) |

### MVDiffusion (Stage 1: Single View → 6 Views)

| 파일 | 용도 |
|------|------|
| `train_diffusion.py` | MVDiffusion 학습 스크립트 |
| `configs/mouse_mvdiffusion.yaml` | MVDiffusion fine-tune 설정 (30k steps) |
| `mvdiffusion/data/mouse_dataset.py` | MVDiffusion용 PyTorch Dataset |
| `mvdiffusion/pipelines/pipeline_mvdiffusion_unclip.py` | MVDiffusion 추론 파이프라인 |
| `mvdiffusion/pipelines/zero123pp_pipeline.py` | Zero123++ 추론 파이프라인 |

### 환경 및 유틸리티

| 파일 | 용도 |
|------|------|
| `setup_mouse_env.sh` | Conda 환경 설정 (1회) |
| `scripts/process_mouse_data.py` | 비디오 → FaceLift 포맷 변환 |
| `scripts/download_weights.py` | Pretrained 가중치 다운로드 |

---

## 체크리스트

- [ ] `source activate_gpu05.sh` 실행 확인
- [ ] 데이터 전처리 완료 (`data_mouse/` 생성)
- [ ] Overfitting 테스트 통과
- [ ] 전체 학습 실행
- [ ] 결과 평가
