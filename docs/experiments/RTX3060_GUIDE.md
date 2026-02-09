# RTX 3060 (12GB VRAM) 실험 가이드

> FaceLift 실험을 소비자 GPU (RTX 3060, 12GB)에서 실행하기 위한 설정.
>
> Created: 2026-02-07

---

## 1. 환경 설정

### 1.1 요구사항

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX 3060 12GB (Ampere, CC 8.6) |
| CUDA | 12.4+ |
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| OS | Ubuntu 20.04+ / Windows WSL2 |

### 1.2 Conda 환경 설치

```bash
# 1. 환경 생성
conda create -n facelift python=3.11 -y
conda activate facelift

# 2. PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 의존성
bash mouse_extensions/scripts/setup/setup_env.sh

# 또는 전체 자동 설치:
bash mouse_extensions/scripts/setup/setup_rtx3060.sh
```

### 1.3 A6000 vs RTX 3060 차이

| 항목 | A6000 | RTX 3060 | 대응 |
|------|-------|----------|------|
| VRAM | 49 GB | 12 GB | batch 축소 |
| Compute | GA102 | GA106 | 동일 CC 8.6 |
| bf16 | 효율적 | 가능하나 fp16 권장 | amp_dtype 변경 |
| Memory BW | 768 GB/s | 360 GB/s | 학습 속도 저하 |

---

## 2. Config 파일

### 2.1 GS-LRM

**파일**: `configs/mouse/rtx3060/gslrm_3060.yaml`

| 설정 | A6000 (기본) | RTX 3060 | 영향 |
|------|:---:|:---:|------|
| batch_size | 2 | **1** | VRAM ~50% 감소 |
| amp_dtype | bf16 | **fp16** | RTX 3060 최적 |
| turntable views | 144 | **36** | validation VRAM spike 방지 |
| val_every | 100 | **500** | 시간 절약 |
| vis_every | 500 | **1000** | 시간 절약 |

**예상 VRAM**: ~10-11 GB (4-view)

### 2.2 MVDiffusion

**파일**: `configs/mouse/rtx3060/mvdiffusion_3060.yaml`

| 설정 | A6000 (기본) | RTX 3060 | 영향 |
|------|:---:|:---:|------|
| train_batch_size | 4 | **1** | VRAM ~75% 감소 |
| gradient_accumulation | 4 | **16** | effective batch 유지 (16) |
| validation_batch_size | 2 | **1** | VRAM 절약 |
| num_workers | 8 | **4** | RAM 절약 |

**예상 VRAM**: ~5-6 GB

---

## 3. 실행 명령어

### 3.1 GS-LRM 학습

```bash
cd /path/to/FaceLift

# 4-view baseline (RTX 3060)
CUDA_VISIBLE_DEVICES=0 python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/rtx3060/gslrm_3060.yaml
```

### 3.2 MVDiffusion 학습

```bash
cd /path/to/FaceLift/mvdiffusion

# MVDiffusion M5t2 (RTX 3060)
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file 1gpu.yaml \
  train_diffusion.py \
  --config ../configs/mouse/rtx3060/mvdiffusion_3060.yaml
```

### 3.3 E2E 추론

```bash
cd /path/to/FaceLift

# E2E inference (RTX 3060 - 추론은 학습보다 VRAM 적음)
CUDA_VISIBLE_DEVICES=0 python -m mouse_extensions.scripts.inference.run_e2e_inference \
  --dataset_path /path/to/M5/data_mouse_t2_test.txt \
  --mvdiff_checkpoint /path/to/mvdiffusion/checkpoint-5000 \
  --gslrm_checkpoint /path/to/gslrm/best_psnr.pt \
  --output_dir outputs/rtx3060_test \
  --num_frames 5
```

---

## 4. VRAM 최적화 팁

### 4.1 VRAM 부족 시 추가 조치

```yaml
# GS-LRM: image_size 축소 (최후의 수단)
model:
  image_tokenizer:
    image_size: 256    # 512 -> 256 (VRAM ~75% 감소, 품질 저하)

# GS-LRM: input views 축소
model:
  num_input_views: 2   # 4 -> 2 (VRAM 추가 감소)
```

### 4.2 PyTorch 메모리 최적화

```bash
# 환경 변수
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 4.3 학습 속도 참고

| 모델 | A6000 | RTX 3060 (추정) |
|------|-------|-----------------|
| GS-LRM (15K steps) | ~12h | ~36-48h |
| MVDiffusion (10K steps) | ~24h | ~48-72h |

---

## 5. 데이터 준비

### 5.1 데이터셋 다운로드 (로컬)

```bash
# gpu03에서 로컬로 복사
scp -r gpu03:/home/joon/data/preprocessed/FaceLift_mouse/M5/ /local/path/data/M5/
```

### 5.2 Split 파일 경로 수정

로컬 환경에서는 split 파일 내 경로를 수정해야 합니다:
```bash
# 경로 치환
sed -i 's|/home/joon/data/preprocessed/FaceLift_mouse|/local/path/data|g' \
  /local/path/data/M5/data_mouse_t2_*.txt
```

### 5.3 Pretrained Checkpoint

```bash
# GS-LRM pretrained
scp gpu03:/home/joon/dev/FaceLift/checkpoints/gslrm/ckpt_0000000000021125.pt \
  /local/path/FaceLift/checkpoints/gslrm/

# MVDiffusion pretrained pipeline
scp -r gpu03:/home/joon/dev/FaceLift/checkpoints/mvdiffusion/pipeckpts/ \
  /local/path/FaceLift/checkpoints/mvdiffusion/

# MVDiffusion M5t2 fine-tuned (optional)
scp -r gpu03:/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2/checkpoint-5000/ \
  /local/path/checkpoints/mvdiffusion/mouse_M5t2/checkpoint-5000/
```

---

## 6. 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| CUDA OOM | VRAM 초과 | batch_size 더 줄이기, image_size 256 |
| bf16 NaN | RTX 3060 bf16 정밀도 | fp16 사용 (config에 이미 설정) |
| diff-gaussian-rasterization 빌드 실패 | CUDA toolkit 미설치 | `conda install cuda-nvcc=12.4` |
| xformers 호환 오류 | PyTorch/xformers 버전 불일치 | 동일 CUDA 12.4 기반 설치 |
| 느린 학습 | Memory BW 차이 | 정상 (A6000의 ~3배 소요) |

---

*RTX 3060 Guide v1.0 | 2026-02-07*
