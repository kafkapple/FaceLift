# Mouse-FaceLift Usage Guide

---
date: 2025-12-13
context_name: "2_Research"
tags: [ai-assisted, mouse-reconstruction, multi-view, 3d-reconstruction, mvdiffusion, gslrm]
project: FaceLift
status: active
generator: ai-assisted
generator_tool: claude-code
last_updated: 2025-12-13
---

## Quick Start - 2단계 학습 파이프라인

> **핵심**: MVDiffusion → 합성 데이터 → GS-LRM 순차 학습으로 도메인 정렬

### 현재 상태 (2025-12-13)

| 단계 | 모델 | 상태 | Config | Checkpoint |
|------|------|:----:|--------|------------|
| **Phase 1** | MVDiffusion | 🔄 학습중 | `mouse_mvdiffusion_6x_aug.yaml` | `mouse_embeds_6x_aug/` |
| **Phase 2** | 합성 데이터 | ⏳ 대기 | - | `data_mouse_synthetic/` |
| **Phase 3** | GS-LRM | ⏳ 대기 | `mouse_gslrm_synthetic.yaml` | `mouse_synthetic/` |

**WandB**: https://wandb.ai → project: `mouse_facelift`

### 실험 결과에 따른 다음 단계

> **상세 가이드**: [mouse_experiment_options.md](./mouse_experiment_options.md)

| 결과 | 다음 단계 | Config |
|------|----------|--------|
| ✅ 수렴 성공 | Phase 2 진행 | - |
| ⚠️ 수렴 느림/실패 | FaceLift 프롬프트 실험 | `mouse_mvdiffusion_facelift_prompt.yaml` |

> **Note**: Realistic 프롬프트는 권장하지 않음 (Pretrained가 `rendering` 도메인 학습)

**프롬프트 대안 생성**:
```bash
python scripts/generate_mouse_prompt_embeds_realistic.py --list-styles
python scripts/generate_mouse_prompt_embeds_realistic.py --style [facelift|realistic|hybrid]
```

---

### Phase 1: MVDiffusion Fine-tune (1뷰 → 6뷰)

```bash
# gpu05 접속
ssh gpu05
cd /home/joon/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

# 학습 시작 (GPU 1만 사용!)
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 accelerate launch train_diffusion.py \
    --config configs/mouse_mvdiffusion_6x_aug.yaml' \
    > logs/train_mvdiff_6x_gpu1.log 2>&1 &

# 모니터링
tail -f logs/train_mvdiff_6x_gpu1.log
nvidia-smi
```

| 설정 | 값 |
|------|-----|
| Config | `configs/mouse_mvdiffusion_6x_aug.yaml` |
| Checkpoint | `checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/` |
| Prompt Embeds | `mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt` |
| Steps | 20,000 |
| 예상 시간 | ~61시간 (~11초/step) |

---

### Phase 2: 합성 데이터 생성

```bash
# Phase 1 완료 후 실행 (checkpoint-10000 이상 권장)
python scripts/generate_gslrm_training_data.py \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-20000 \
    --input_data data_mouse/data_mouse_train.txt \
    --output_dir data_mouse_synthetic \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --camera_json data_mouse/sample_000000/opencv_cameras.json \
    --augment_all_views

# 결과 확인
ls data_mouse_synthetic/
# data_train.txt, data_val.txt, sample_000000/, ...
```

| 설정 | 값 |
|------|-----|
| Script | `scripts/generate_gslrm_training_data.py` |
| 입력 | 1,799 train 샘플 × 6뷰 = 10,794 합성 샘플 |
| 출력 | `data_mouse_synthetic/` |
| 예상 시간 | ~2-4시간 |

---

### Phase 3: GS-LRM Fine-tune (합성 6뷰 → 3D)

```bash
# Phase 2 완료 후 실행
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse_gslrm_synthetic.yaml' \
    > logs/train_gslrm_synthetic.log 2>&1 &

# 모니터링
tail -f logs/train_gslrm_synthetic.log
```

| 설정 | 값 |
|------|-----|
| Config | `configs/mouse_gslrm_synthetic.yaml` |
| Dataset | `data_mouse_synthetic/data_train.txt` |
| Start From | `checkpoints/gslrm/ckpt_0000000000021125.pt` (human pretrained) |
| Checkpoint | `checkpoints/gslrm/mouse_synthetic/` |
| Steps | 30,000 |

---

### Phase 4: 최종 추론

```bash
# 전체 파이프라인 테스트
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-20000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_synthetic \
    --output_dir outputs/pipeline_test
```

---

## 데이터셋 구성

### 원본 데이터

| 항목 | 값 | 설명 |
|------|-----|------|
| 소스 | 6개 동기화 카메라 비디오 | MAMMAL 스타일 촬영 |
| 샘플링 | 2,000 프레임 | 비디오에서 균등 추출 |
| 각 샘플 | 6개 뷰 | 동시 촬영된 카메라 뷰 |
| 총 이미지 | **12,000장** | 2,000 × 6 뷰 |
| 이미지 크기 | 512 × 512 | RGBA (배경 제거됨) |

### Train/Val Split

| 구분 | 샘플 수 | 비율 |
|------|---------|------|
| Train | 1,799 | 90% |
| Val | 199 | 10% |
| **합계** | 1,998 | 100% |

- Split 방식: `np.random.permutation` + `seed(42)` (재현 가능)
- 중복 없음 검증 완료

### 데이터 충분성 분석

| 비교 대상 | 샘플 수 | 이미지 수 |
|----------|---------|-----------|
| **Mouse 데이터** | 2,000 | 12,000 |
| FaceLift Human | ~50,000 | ~300,000 |
| Zero123++ | ~800,000 | ~800,000 |
| MVDream | ~10,000 | ~40,000 |

**결론**:
- 2,000 샘플은 fine-tuning에 충분 (pretrained 모델 활용)
- 6x 증강 (`reference_view_idx: "random"`)으로 effective ~12,000 샘플
- 추가 데이터 확보 시 성능 향상 가능

---

## 주요 파일 경로

### Configs

| Config | 용도 | 경로 |
|--------|------|------|
| MVDiffusion 6x | Phase 1 학습 | `configs/mouse_mvdiffusion_6x_aug.yaml` |
| GS-LRM Synthetic | Phase 3 학습 | `configs/mouse_gslrm_synthetic.yaml` |
| Mouse Prompt Embeds | 경사 6뷰 임베딩 | `mvdiffusion/data/mouse_prompt_embeds_6view/` |

### Scripts

| Script | 용도 |
|--------|------|
| `scripts/process_mouse_data.py` | 비디오 → FaceLift 포맷 변환 |
| `scripts/generate_mouse_prompt_embeds_simple.py` | Mouse prompt embeds 생성 |
| `scripts/generate_mouse_prompt_embeds_realistic.py` | 다양한 스타일 prompt embeds 생성 |
| `scripts/generate_gslrm_training_data.py` | Phase 2 합성 데이터 생성 |
| `scripts/check_server_resources.sh` | 서버 리소스 모니터링 |

### Checkpoints

| Checkpoint | 경로 |
|------------|------|
| Human Pretrained GS-LRM | `checkpoints/gslrm/ckpt_0000000000021125.pt` |
| MVDiffusion Pretrained | `checkpoints/mvdiffusion/pipeckpts/` |
| MVDiffusion Mouse (학습중) | `checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/` |
| GS-LRM Synthetic (예정) | `checkpoints/gslrm/mouse_synthetic/` |

---

## 모니터링 명령어

```bash
# GPU 상태
ssh gpu05 "nvidia-smi"

# 프로세스 확인
ssh gpu05 "ps aux | grep train | grep -v grep"

# 로그 확인
ssh gpu05 "tail -f /home/joon/FaceLift/logs/train_mvdiff_6x_gpu1.log"

# 체크포인트 확인
ssh gpu05 "ls -la /home/joon/FaceLift/checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/"

# 학습 중단
ssh gpu05 "pkill -f train_diffusion"
```

---

## Overview

Mouse-FaceLift adapts the FaceLift 3D reconstruction pipeline for mouse multi-view data.
This guide covers environment setup, data preprocessing, training, and inference.

### 2단계 학습 전략 (2025-12-13)

```
┌─────────────────────────────────────────────────────────────────┐
│  문제: 카메라/Prompt 불일치                                      │
├─────────────────────────────────────────────────────────────────┤
│  MVDiffusion: FaceLift prompt_embeds (수평 뷰)로 학습됨          │
│  GS-LRM: Mouse 카메라 (경사 뷰 ~20°)로 학습됨                    │
│  → MVDiffusion 출력 ≠ GS-LRM 기대 입력 → 3D 복원 실패           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  해결: 2단계 순차 학습                                           │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: MVDiffusion + Mouse prompt_embeds (경사 뷰) 학습      │
│  Phase 2: 학습된 MVDiffusion으로 합성 데이터 생성                │
│  Phase 3: GS-LRM을 합성 데이터로 학습 (도메인 정렬)              │
└─────────────────────────────────────────────────────────────────┘
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
source ~/anaconda3/etc/profile.d/conda.sh
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
conda activate mouse_facelift

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
├── data_mouse_train.txt    # 학습 샘플 경로 목록 (1,799)
├── data_mouse_val.txt      # 검증 샘플 경로 목록 (199)
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

## 문제 해결

### CLIPTokenizer merges.txt 오류
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

**원인**: CLIPTokenizer에 필요한 `merges.txt` 파일 누락

**해결**: 자동 다운로드 로직이 포함되어 있음 (v2024.12.10+). 수동 해결 필요 시:
```bash
cd checkpoints/mvdiffusion/pipeckpts/tokenizer
wget https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt
```

### CUDA Out of Memory
```yaml
# batch_size 줄이기
training:
  dataloader:
    batch_size_per_gpu: 1
```

### OmegaConf ValidationError (reference_view_idx)
```
Value 'random' of type 'str' could not be converted to Integer
```

**해결**: Python 캐시 삭제 후 재시도
```bash
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null
```

### GPU 사용 제한 (공용 서버)
```bash
# GPU 0 사용 금지! GPU 1만 사용
CUDA_VISIBLE_DEVICES=1 accelerate launch ...
```

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
conda activate mouse_facelift
```

---

## 파일 참조

### GSLRM (Stage 2: 6 Views → 3D)

| 파일 | 용도 |
|------|------|
| `train_gslrm.py` | GSLRM 학습 스크립트 |
| `inference_mouse.py` | 통합 추론 스크립트 |
| `gslrm/data/mouse_dataset.py` | GSLRM용 PyTorch Dataset |
| `configs/mouse_gslrm_synthetic.yaml` | 합성 데이터 학습 (Phase 3) |

### MVDiffusion (Stage 1: Single View → 6 Views)

| 파일 | 용도 |
|------|------|
| `train_diffusion.py` | MVDiffusion 학습 스크립트 |
| `configs/mouse_mvdiffusion_6x_aug.yaml` | 6x 증강 학습 (Phase 1) |
| `mvdiffusion/data/mouse_dataset.py` | MVDiffusion용 Dataset (random ref view 지원) |
| `mvdiffusion/data/mouse_prompt_embeds_6view/` | Mouse 경사 뷰 prompt embeddings |

### 환경 및 유틸리티

| 파일 | 용도 |
|------|------|
| `scripts/process_mouse_data.py` | 비디오 → FaceLift 포맷 변환 |
| `scripts/generate_mouse_prompt_embeds_simple.py` | Prompt embeddings 생성 |
| `scripts/generate_gslrm_training_data.py` | Phase 2 합성 데이터 생성 |

---

## 체크리스트

### Phase 1 시작 전
- [x] `mouse_prompt_embeds_6view/clr_embeds.pt` 존재 확인
- [x] GPU 1 사용 가능 확인
- [x] Python 캐시 정리

### Phase 2 시작 전
- [ ] MVDiffusion 학습 완료 확인 (WandB)
- [ ] 체크포인트 존재 확인 (checkpoint-XXXXX)
- [ ] 디스크 공간 확인 (~50GB)

### Phase 3 시작 전
- [ ] 합성 데이터 생성 완료
- [ ] `data_mouse_synthetic/data_train.txt` 존재 확인
- [ ] Human pretrained 체크포인트 준비

---

## 관련 문서

- [실험 옵션 가이드](./mouse_experiment_options.md) - 상황별 실험 전략
- [Prompt Embedding 연구](../reports/251213_research_prompt_embedding_adaptation.md) - 프롬프트 적응 분석
- [2단계 학습 전략 연구노트](../reports/251213_research_two_phase_training_strategy.md)
- [MVDiffusion 체크포인트 이슈](../reports/251212_research_mvdiffusion_training_checkpoint_issue.md)
- [CLIP Tokenizer 문제 해결](../troubleshooting/clip_tokenizer_merges_error.md)
