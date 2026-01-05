---
date: 2024-12-30
context_name: "2_Research"
tags: [ai-assisted, gslrm, plucker-coordinates, view-order, bug-fix]
project: mouse-facelift
status: in-progress
generator: ai-assisted
generator_tool: claude-code
---

# GS-LRM 뷰 순서 문제 분석 및 수정

## 목적

GS-LRM fine-tuning 시 학습이 진행될수록 출력이 흐릿해지는 문제의 원인 분석 및 해결

## 문제 발견

### 증상
- wandb의 `val/gt_vs_pred` 시각화에서 뷰 인덱스가 매번 다르게 표시됨
- 학습이 진행될수록 출력이 점점 흐릿해짐

### 근본 원인: RandomViewDataset의 뷰 순서 랜덤화

```
═══════════════════════════════════════════════════════════════════════════════
                    Plücker 좌표 불일치 문제
═══════════════════════════════════════════════════════════════════════════════

Step 1: 뷰 조합 [0, 2, 3, 4, 5, 1]
  인덱스 1의 Plücker: cam_2 기준 (상단에서 아래로)

Step 2: 뷰 조합 [1, 3, 0, 5, 2, 4]
  인덱스 1의 Plücker: cam_3 기준 (우측 뒤에서)  ← 완전히 다른 위치!

Step 3: 뷰 조합 [4, 1, 2, 0, 3, 5]
  인덱스 1의 Plücker: cam_1 기준 (좌측 앞에서)  ← 또 다른 위치!

→ 모델이 "인덱스 N = 특정 공간 위치"라는 일관된 매핑을 학습할 수 없음
→ 모든 가능한 위치의 "평균"을 학습 → 흐릿한 출력
```

### 인간 데이터 vs 마우스 데이터

| | 인간 데이터 | 마우스 데이터 |
|---|------------|-------------|
| 카메라 배치 | 균등 60° 간격 | 불균등 각도 |
| 랜덤 샘플링 | 어떤 조합도 유사 커버리지 | 조합마다 커버리지 다름 |
| 문제 심각도 | 낮음 | **높음** |

## 코드 분석

### 문제의 코드 위치

1. **train_gslrm.py:113** (수정 전)
```python
use_mouse_dataset = self.config.get("mouse", {}).get("normalize_cameras", False)
```
- `normalize_cameras`가 없으면 기본값 `False` → RandomViewDataset 사용
- 네이밍 혼란: "normalize"가 데이터셋 클래스 선택에 사용됨

2. **RandomViewDataset (dataset.py:229)**
```python
image_choices = random.sample(range(len(cameras)), self.num_views)
```
- 매 스텝마다 다른 뷰 조합

3. **MouseViewDataset (mouse_dataset.py:443)**
```python
input_indices = list(range(self.num_input_views))  # 고정 순서
```
- 항상 [0,1,2,3,4,5] 순서

## 수정 내용

### 1. train_gslrm.py 수정
```python
# Before
use_mouse_dataset = self.config.get("mouse", {}).get("normalize_cameras", False)

# After
use_mouse_dataset = self.config.get("mouse", {}).get("use_mouse_dataset", False)
```
- 명확한 네이밍: `use_mouse_dataset`
- 주석으로 Plücker 좌표 문제 설명 추가

### 2. Config 파일들에 설정 추가

**mouse_gslrm_normalized_synthetic.yaml, mouse_gslrm_lowmem.yaml:**
```yaml
mouse:
  use_mouse_dataset: true      # ← 핵심! 고정 뷰 순서

  normalize_cameras: true       # Z-up 정규화
  target_camera_distance: 2.7   # 거리 정규화
  normalize_to_z_up: true
```

### 3. RTX 3060용 로컬 config 생성

`configs/mouse_gslrm_local_rtx3060.yaml`:
- batch_size: 1 (12GB VRAM)
- grad_accum_steps: 4 (effective batch = 4)
- fp16 (bf16 대신)
- LPIPS/Perceptual: 0.0 (마우스 도메인 문제)

## 데이터 상태

### 현재 data_mouse_local/
- 샘플 수: 3,496 train + 99 val
- 카메라 거리: **246mm** (원본 실험 좌표)
- 정규화 필요: MouseViewDataset이 자동으로 2.7로 정규화

### MouseViewDataset 카메라 처리
```python
# mouse_dataset.py:586-589
if self.target_camera_distance > 0:
    input_c2ws = normalize_camera_distance(
        input_c2ws, self.target_camera_distance  # 2.7
    )
```
→ 런타임에 자동 정규화 (데이터 전처리 불필요)

## 실행 환경

### 로컬 머신
- GPU: RTX 3060 12GB
- VRAM 최적화 설정 적용

### 예상 학습 시간
- 3,496 샘플 × 4 grad_accum = 14,000 steps/epoch
- 20,000 steps = ~1.5 epochs
- 예상: 약 4-6시간

## 실행 명령어

```bash
# 1. 체크포인트 확인
ls -la checkpoints/gslrm/ckpt_0000000000021125.pt

# 2. 로컬 RTX 3060에서 학습 시작
python train_gslrm.py --config configs/mouse_gslrm_local_rtx3060.yaml

# 3. wandb 모니터링
# → val/gt_vs_pred 에서 뷰 인덱스가 항상 [0,1,2,3,4,5]로 고정되는지 확인
```

## 검증 방법

### 성공 지표
1. wandb `val/gt_vs_pred`에서 뷰 순서가 항상 일정
2. 학습 진행에 따라 출력이 선명해짐 (흐릿해지지 않음)
3. PSNR/SSIM 지표가 개선됨

### 실패 시 추가 조치
1. 카메라 정규화 값 검증 (distance, up vector)
2. Plücker 좌표 시각화로 일관성 확인
3. 데이터 전처리 단계에서 정규화 수행

## 핵심 학습 포인트

1. **네이밍의 중요성**: `normalize_cameras`가 두 가지 의미로 사용되어 혼란 발생
2. **데이터 특성 고려**: 인간/마우스 데이터의 카메라 배치 차이가 알고리즘에 미치는 영향
3. **Plücker 좌표**: 픽셀별 3D 광선 인코딩 → 일관된 매핑이 핵심

---

## 환경 설정

### 로컬 시스템 사양

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 22.04.5 LTS |
| GPU | RTX 3060 12GB |
| CUDA | 11.8 / 12.4 (둘 다 설치됨) |
| Python | 3.10 |
| gcc | 11.4.0 |

### Conda 환경 설치

```bash
# 1. 설치 스크립트 실행
bash setup_local_rtx3060.sh facelift_rtx3060

# 2. 환경 활성화
conda activate facelift_rtx3060

# 3. 검증
python -c "import torch; print(torch.cuda.is_available())"
```

### RTX 3060 메모리 최적화 설정

| 설정 | 값 | 이유 |
|------|-----|------|
| batch_size | 1 | 12GB VRAM 한계 |
| grad_accum_steps | 4 | effective batch = 4 |
| amp_dtype | fp16 | bf16보다 consumer GPU에 적합 |
| use_tf32 | false | RTX 3060 미지원 |
| grad_checkpoint_every | 1 | 최대 메모리 절약 |
| LPIPS/Perceptual | 0.0 | 마우스 도메인 문제 회피 |

### 환경 변수 (자동 설정됨)

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

---

## 추가 분석: Masked Loss 문제 (2024-12-30)

### 발견된 핵심 문제

1. **이미지에 alpha 채널 없음**
   - 마우스 이미지: RGB 모드 (3 채널)
   - 배경: 순수 흰색 (255, 255, 255)
   - **배경 비율: ~95%**, 전경(마우스): ~5%

2. **기존 Loss 문제**
   - L2/SSIM loss가 mask 없이 전체 픽셀에 적용
   - 95%의 loss가 배경 픽셀에서 발생
   - 모델이 "흰색 이미지 출력"을 학습하게 됨

### 구현된 수정사항

#### 1. Masked Loss 구현 (`gslrm/model/gslrm.py`)

```python
# _compute_l2_loss(): mask 옵션 추가
if use_mask and mask is not None:
    mask_binary = (mask > 0.5).float()
    num_valid = mask_binary.sum().clamp(min=1.0)
    squared_error = (rendering - target) ** 2
    masked_error = squared_error * mask_binary
    return masked_error.sum() / (num_valid * 3)

# _compute_ssim_loss(): mask 옵션 추가
if use_mask and mask is not None:
    mask_binary = (mask > 0.5).float()
    neutral_value = 0.5
    masked_rendering = rendering * mask_binary + neutral_value * (1 - mask_binary)
    masked_target = target * mask_binary + neutral_value * (1 - mask_binary)
    return self.ssim_loss_module(masked_rendering, masked_target)
```

#### 2. 자동 Mask 생성 (`gslrm/data/mouse_dataset.py`)

```python
# 흰 배경에서 자동 mask 생성
if self.auto_generate_mask and image_np.shape[2] == 3:
    threshold = self.mask_threshold / 255.0
    is_background = np.all(image_np > threshold, axis=2)
    alpha = (~is_background).astype(np.float32)
    image_np = np.concatenate([image_np, alpha[:, :, None]], axis=2)
```

#### 3. Config 업데이트 (`configs/mouse_gslrm_local_rtx3060.yaml`)

```yaml
training:
  losses:
    masked_l2_loss: true      # L2 loss를 foreground에만 적용
    masked_ssim_loss: true    # SSIM loss를 foreground에만 적용

mouse:
  auto_generate_mask: true    # 흰 배경에서 mask 자동 생성
  mask_threshold: 250         # RGB > 250 = 배경
```

### 디버깅 스크립트

```bash
# mask 시각화 디버깅
python scripts/debug_mask_visualization.py \
    --config configs/mouse_gslrm_local_rtx3060.yaml \
    --num_samples 5 \
    --output_dir experiments/debug
```

### 수정 파일 목록

| 파일 | 변경 내용 |
|------|-----------|
| `gslrm/model/gslrm.py` | Masked L2/SSIM loss 구현 |
| `gslrm/data/mouse_dataset.py` | 자동 mask 생성 기능 |
| `configs/mouse_gslrm_local_rtx3060.yaml` | masked loss 및 auto_generate_mask 설정 |
| `scripts/debug_mask_visualization.py` | 디버깅 스크립트 (신규) |

---

## 다음 단계

- [x] Masked L2/SSIM loss 구현
- [x] 자동 mask 생성 기능 추가
- [x] Config 업데이트
- [ ] conda 환경 설치: `bash setup_local_rtx3060.sh`
- [ ] 디버깅 스크립트로 mask 시각화 확인
- [ ] 로컬에서 학습 실행
- [ ] wandb 모니터링으로 loss 감소 확인 (목표: < 0.5, PSNR > 10)
- [ ] 500 steps 후 시각화 품질 비교
