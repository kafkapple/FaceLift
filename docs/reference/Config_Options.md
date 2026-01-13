# Config 옵션 참조

> Mouse GS-LRM config의 모든 옵션 상세 설명

## 1. Model 섹션

### model.class_name

```yaml
model:
  class_name: gslrm.model.gslrm.GSLRM
```

모델 클래스의 전체 경로. 동적 import 사용.

### model.image_tokenizer

```yaml
image_tokenizer:
  image_size: 512     # 입력 이미지 크기
  patch_size: 8       # ViT 패치 크기 (512/8 = 64 패치)
  in_channels: 9      # RGB(3) + Plücker(3) + Reference(3)
```

### model.transformer

```yaml
transformer:
  d: 1024             # Hidden dimension
  d_head: 64          # Attention head dimension (d/d_head = 16 heads)
  n_layer: 24         # Transformer 레이어 수
```

### model.gaussians

```yaml
gaussians:
  n_gaussians: 2      # 픽셀당 Gaussian 수
  sh_degree: 0        # Spherical Harmonics 차수 (0 = RGB only)
  upsampler:
    upsample_factor: 1  # 업샘플링 배율
```

**Gaussian 출력 크기**:
- n_gaussians=2, 512x512 이미지 → 524,288 Gaussians

### model flags

```yaml
add_refsrc_marker: false    # Reference/Source 마커 추가
hard_pixelalign: true       # 픽셀 정렬 강제
use_custom_plucker: true    # 커스텀 Plücker 좌표 사용
clip_xyz: true              # 3D 좌표 클리핑
```

---

## 2. Training 섹션

### training.runtime

```yaml
runtime:
  use_tf32: true          # TensorFloat-32 (A100 가속)
  use_amp: true           # Automatic Mixed Precision
  amp_dtype: "bf16"       # bf16 (권장) 또는 fp16
  torch_compile: false    # torch.compile (실험적)
  grad_accum_steps: 2     # Gradient accumulation
  grad_clip_norm: 50.0    # Gradient clipping norm
  grad_checkpoint_every: 1  # Gradient checkpointing 빈도
```

**amp_dtype 선택**:
- `bf16`: 더 넓은 동적 범위, 안정적 (권장)
- `fp16`: 더 빠를 수 있음, overflow 주의

### training.dataset

```yaml
dataset:
  dataset_path: "data_mouse_train.txt"   # 학습 데이터 경로
  num_views: 6                           # 총 뷰 수
  num_input_views: 5                     # 입력 뷰 수 (나머지가 target)
  target_has_input: true                 # Target에 input 뷰 포함
  maximize_view_overlap: false           # 뷰 오버랩 최대화 (Mouse에서 false)
  normalize_distance_to: 0.0             # 거리 정규화 (0=비활성화)
  remove_alpha: false                    # 알파 채널 제거
  background_color: "white"              # 배경색 (white/black/gray/random)
```

**num_views vs num_input_views**:
- `num_views=6, num_input_views=5`: 1개 입력 → 6개 출력
- `target_has_input=true`: 출력에 입력 뷰도 포함

### training.dataloader

```yaml
dataloader:
  batch_size_per_gpu: 2   # GPU당 배치 크기
  num_workers: 8          # DataLoader 워커 수
  num_threads: 16         # CPU 스레드 수
  prefetch_factor: 8      # 프리페치 배수
```

**메모리 부족 시**:
```yaml
batch_size_per_gpu: 1
grad_accum_steps: 4  # effective batch = 1 * 4 = 4
```

### training.losses

```yaml
losses:
  # Primary losses
  l2_loss_weight: 1.0           # L2 (MSE) reconstruction loss
  perceptual_loss_weight: 0.5   # VGG perceptual loss
  lpips_loss_weight: 0.0        # LPIPS (느림, 선택적)
  ssim_loss_weight: 0.0         # SSIM loss
  
  # Auxiliary losses
  background_loss_weight: 1.0   # 배경 영역 loss
  pixelalign_loss_weight: 0.0   # 픽셀 정렬 loss
  pointsdist_loss_weight: 0.0   # 점 분포 loss
  distill_loss_weight: 0.0      # Knowledge distillation
  
  # Masking options
  masked_l2_loss: true          # 마스크 영역만 L2
  masked_pixelalign_loss: true  # 마스크 영역만 pixelalign
  masked_ssim_loss: true        # 마스크 영역만 SSIM
  clamp_rendering: true         # 렌더링 값 클램핑
```

**권장 조합**:
```yaml
l2_loss_weight: 1.0
perceptual_loss_weight: 0.5
background_loss_weight: 1.0
masked_l2_loss: true
```

### training.optimizer

```yaml
optimizer:
  lr: 1.0e-6              # Learning rate
  beta1: 0.9              # Adam β1
  beta2: 0.95             # Adam β2
  weight_decay: 0.05      # L2 regularization
  reset_lr: true          # Checkpoint 로드 시 LR 리셋
  reset_weight_decay: false
  reset_training_state: true  # Optimizer 상태 리셋
```

**Fine-tuning LR 가이드**:
| 상황 | LR |
|------|------|
| From scratch | 1e-4 |
| Fine-tuning | 1e-5 ~ 1e-6 |
| 매우 적은 데이터 | 1e-6 |

### training.schedule

```yaml
schedule:
  num_epochs: 50000           # 최대 에폭
  early_stop_after_epochs: 50000
  max_fwdbwd_passes: 10000    # 최대 학습 스텝
  warmup: 500                 # LR warmup 스텝
  l2_warmup_steps: 500        # L2 only warmup
```

**max_fwdbwd_passes**: 실제 학습은 이 값에 도달하면 종료

### training.checkpointing

```yaml
checkpointing:
  resume_ckpt: "checkpoints/gslrm/ckpt_*.pt"  # Pretrained 체크포인트
  checkpoint_dir: "checkpoints/mouse_gslrm"   # 저장 디렉토리
  checkpoint_every: 100                        # 저장 빈도
```

### training.logging

```yaml
logging:
  print_every: 10      # 콘솔 출력 빈도
  vis_every: 100       # 시각화 저장 빈도
  
  wandb:
    project: "facelift_mouse"   # WandB 프로젝트
    group: "gslrm"              # 실험 그룹
    job_type: "finetune"        # 작업 유형
    exp_name: "mouse_v1"        # 실험 이름 (고유)
    log_every: 10               # WandB 로깅 빈도
    offline: false              # 오프라인 모드
```

---

## 3. Mouse 섹션

```yaml
mouse:
  use_mouse_dataset: true       # ⭐ MouseViewDataset 사용
  
  # Camera normalization
  normalize_cameras: false      # Dataset 내 카메라 정규화
  target_camera_distance: 2.7   # 목표 카메라 거리
  
  # Coordinate system
  normalize_to_z_up: true       # Z-up 좌표계 정규화
  
  # View selection
  random_view_selection: false  # 랜덤 뷰 선택 (false 권장)
  
  # Augmentation
  augmentation:
    enabled: false              # 데이터 증강
```

**use_mouse_dataset**: 가장 중요한 플래그
- `true`: MouseViewDataset (고정 뷰 순서)
- `false`: RandomViewDataset (원본)

---

## 4. Validation 섹션

```yaml
validation:
  enabled: true                                    # 검증 활성화
  dataset_path: "data_mouse_val.txt"              # 검증 데이터
  output_dir: "experiments/validation/mouse"       # 출력 디렉토리
  val_every: 200                                   # 검증 빈도 (스텝)
```

---

## 5. Inference 섹션

```yaml
inference:
  enabled: false                              # 추론 모드
  output_dir: "experiments/inference/mouse"   # 출력 디렉토리
```

추론 모드 실행:
```bash
python train_gslrm.py --config configs/mouse/gslrm.yaml --inference
```

---

## 6. 설정 예시: 다양한 시나리오

### 빠른 테스트

```yaml
training:
  dataloader:
    batch_size_per_gpu: 1
  schedule:
    max_fwdbwd_passes: 100
  logging:
    vis_every: 10
```

### 메모리 제한 환경

```yaml
training:
  dataloader:
    batch_size_per_gpu: 1
    num_workers: 2
  runtime:
    grad_accum_steps: 4
```

### 고품질 학습

```yaml
training:
  losses:
    l2_loss_weight: 1.0
    perceptual_loss_weight: 1.0
    lpips_loss_weight: 0.5
  schedule:
    max_fwdbwd_passes: 50000
```

---

*Created: 2026-01-13*
