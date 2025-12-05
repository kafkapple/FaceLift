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

## 학습

### Step 1: Overfitting 테스트 (필수 권장)
```bash
# 10개 샘플로 코드 정상 동작 확인
python train_mouse.py --config configs/mouse_config.yaml --overfit 10
```

**기대 결과**:
- Loss가 0에 가깝게 감소
- 입력 이미지가 완벽하게 복원됨
- 이것이 성공해야 전체 학습 진행

### Step 2: 전체 학습

#### 단일 GPU
```bash
python train_mouse.py --config configs/mouse_config.yaml
```

#### 멀티 GPU (권장)
```bash
# 4 GPU 사용
torchrun --nproc_per_node 4 --nnodes 1 \
    --rdzv_id ${RANDOM} --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    train_mouse.py --config configs/mouse_config.yaml
```

### Step 3: 체크포인트에서 재개
```bash
python train_mouse.py --config configs/mouse_config.yaml \
    --load checkpoints/gslrm/mouse/
```

---

## 추론

### 단일 이미지
```bash
python inference_mouse.py \
    --input_image path/to/mouse.png \
    --output_dir outputs/ \
    --checkpoint checkpoints/gslrm/mouse/ \
    --save_video
```

### 디렉토리 내 모든 이미지
```bash
python inference_mouse.py \
    --input_dir examples/mouse/ \
    --output_dir outputs/mouse/ \
    --checkpoint checkpoints/gslrm/mouse/ \
    --num_views 6
```

---

## 설정 파일

### configs/mouse_config.yaml 주요 설정
```yaml
training:
  dataset:
    dataset_path: "data_mouse/data_mouse_train.txt"  # 전처리 출력 경로
    num_views: 6               # 총 뷰 수
    num_input_views: 1         # 입력 뷰 (단일 이미지)

  dataloader:
    batch_size_per_gpu: 2      # GPU 메모리에 따라 조절
    num_workers: 4

  optimizer:
    lr: 0.00005                # 학습률

  checkpointing:
    resume_ckpt: "checkpoints/gslrm/stage_2"  # 사전학습 가중치
    checkpoint_dir: "checkpoints/gslrm/mouse"
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

| 파일 | 용도 |
|------|------|
| `setup_mouse_env.sh` | Conda 환경 최초 설정 (1회, 환경변수 영구 설정 포함) |
| `scripts/process_mouse_data.py` | 비디오 → FaceLift 포맷 변환 |
| `gslrm/data/mouse_dataset.py` | PyTorch Dataset 클래스 |
| `configs/mouse_config.yaml` | 학습 설정 |
| `train_mouse.py` | 학습 스크립트 |
| `inference_mouse.py` | 추론 스크립트 |

---

## 체크리스트

- [ ] `source activate_gpu05.sh` 실행 확인
- [ ] 데이터 전처리 완료 (`data_mouse/` 생성)
- [ ] Overfitting 테스트 통과
- [ ] 전체 학습 실행
- [ ] 결과 평가
