# Step 4: Config 설정

> Mouse 학습을 위한 config 파일을 설정합니다.

## 4.1 Config 구조 이해

### 원본 gslrm.yaml 주요 섹션

```yaml
model:           # 모델 아키텍처
training:        # 학습 설정
  runtime:       # - 런타임 옵션 (AMP, TF32 등)
  dataset:       # - 데이터셋 설정
  dataloader:    # - 데이터로더 설정
  losses:        # - 손실 함수 가중치
  optimizer:     # - 옵티마이저 설정
  schedule:      # - 학습 스케줄
  checkpointing: # - 체크포인트 설정
  logging:       # - 로깅 설정
validation:      # 검증 설정
inference:       # 추론 설정
```

---

## 4.2 Mouse Config 생성

### 파일 생성
`configs/mouse/gslrm.yaml`

```yaml
# ============================================================================
# Mouse GS-LRM Configuration
# 
# FaceLift pretrained model 기반 mouse 데이터 fine-tuning
# 핵심: use_mouse_dataset: true 로 MouseViewDataset 사용
# ============================================================================

profile: false
debug: false

# ============================================================================
# Model Configuration (원본과 동일하게 유지)
# ============================================================================
model:
  class_name: gslrm.model.gslrm.GSLRM
  
  image_tokenizer:
    image_size: 512
    patch_size: 8
    in_channels: 9  # RGB(3) + Ray Direction(3) + Reference(3)
  
  transformer:
    d: 1024         # Hidden dimension
    d_head: 64      # Attention head dimension
    n_layer: 24     # Number of transformer layers
  
  gaussians:
    n_gaussians: 2  # Gaussians per pixel (after upsampling)
    sh_degree: 0    # Spherical harmonics degree
    upsampler:
      upsample_factor: 1
  
  # Model behavior flags
  add_refsrc_marker: false
  hard_pixelalign: true
  use_custom_plucker: true
  clip_xyz: true

# ============================================================================
# Training Configuration
# ============================================================================
training:
  runtime:
    use_tf32: true          # TensorFloat-32 가속
    use_amp: true           # Mixed precision
    amp_dtype: "bf16"       # BFloat16 (더 안정적)
    torch_compile: false
    grad_accum_steps: 2     # Gradient accumulation
    grad_clip_norm: 50.0    # Gradient clipping
    grad_checkpoint_every: 1

  # Dataset configuration
  dataset:
    # 전처리된 데이터 경로 (Step 3에서 생성)
    dataset_path: "data_mouse_train.txt"
    
    # View configuration
    num_views: 6              # Mouse: 6 views (not 32)
    num_input_views: 5        # 5 input -> 6 output (1 is always input)
    target_has_input: true    # Target includes input views
    maximize_view_overlap: false
    
    # Preprocessing (이미 전처리된 경우 False)
    normalize_distance_to: 0.0
    remove_alpha: false
    background_color: "white"

  dataloader:
    batch_size_per_gpu: 2     # GPU 메모리에 따라 조정
    num_workers: 8
    num_threads: 16
    prefetch_factor: 8

  # Loss weights
  losses:
    l2_loss_weight: 1.0           # Primary reconstruction loss
    perceptual_loss_weight: 0.5   # VGG perceptual loss
    lpips_loss_weight: 0.0        # LPIPS (비활성화 - 느림)
    ssim_loss_weight: 0.0         # SSIM
    background_loss_weight: 1.0   # Background loss
    pixelalign_loss_weight: 0.0
    pointsdist_loss_weight: 0.0
    distill_loss_weight: 0.0
    
    # Masked losses (권장: 마스크 영역만 계산)
    masked_l2_loss: true
    masked_pixelalign_loss: true
    masked_ssim_loss: true
    clamp_rendering: true

  # Optimizer
  optimizer:
    lr: 1.0e-6              # Very low LR for fine-tuning
    beta1: 0.9
    beta2: 0.95
    weight_decay: 0.05
    reset_lr: true          # Reset LR when loading checkpoint
    reset_weight_decay: false
    reset_training_state: true

  # Schedule
  schedule:
    num_epochs: 50000       # Will be limited by max_fwdbwd_passes
    early_stop_after_epochs: 50000
    max_fwdbwd_passes: 10000  # 실제 학습 스텝 수
    warmup: 500
    l2_warmup_steps: 500

  # Checkpointing
  checkpointing:
    # Pretrained checkpoint (HuggingFace에서 다운로드)
    resume_ckpt: "checkpoints/gslrm/ckpt_0000000000021125.pt"
    checkpoint_dir: "checkpoints/mouse_gslrm"
    checkpoint_every: 100

  # Logging
  logging:
    print_every: 10
    vis_every: 100
    
    wandb:
      project: "facelift_mouse"
      group: "gslrm"
      job_type: "finetune"
      exp_name: "mouse_gslrm_v1"
      log_every: 10
      offline: false

# ============================================================================
# Mouse-Specific Configuration (NEW)
# ============================================================================
mouse:
  use_mouse_dataset: true     # ⭐ 핵심: MouseViewDataset 사용
  
  # Camera normalization (전처리에서 이미 된 경우 false)
  normalize_cameras: false
  target_camera_distance: 2.7
  
  # Coordinate system
  normalize_to_z_up: true
  
  # View selection
  random_view_selection: false  # 고정 뷰 순서 사용
  
  # Augmentation (선택적)
  augmentation:
    enabled: false

# ============================================================================
# Validation Configuration
# ============================================================================
validation:
  enabled: true
  dataset_path: "data_mouse_val.txt"
  output_dir: "experiments/validation/mouse_gslrm"
  val_every: 200

# ============================================================================
# Inference Configuration
# ============================================================================
inference:
  enabled: false
  output_dir: "experiments/inference/mouse_gslrm"
```

---

## 4.3 주요 설정 설명

### 핵심 변경 (Mouse 적용)

| 설정 | 원본 (Human) | Mouse | 이유 |
|------|-------------|-------|------|
| `num_views` | 8 | **6** | Mouse 데이터는 6개 뷰 |
| `num_input_views` | 4 | **5** | 1 input + 5 target = 6 total |
| `use_mouse_dataset` | false | **true** | MouseViewDataset 사용 |
| `lr` | 1e-4 | **1e-6** | Fine-tuning은 낮은 LR |
| `max_fwdbwd_passes` | 100000 | **10000** | 적은 데이터로 빠른 학습 |

### Loss 설정 권장값

```yaml
losses:
  l2_loss_weight: 1.0           # 기본 reconstruction
  perceptual_loss_weight: 0.5   # 텍스처 품질 향상
  background_loss_weight: 1.0   # 배경 분리 개선
  
  # 마스크 기반 loss (권장)
  masked_l2_loss: true
  masked_ssim_loss: true
```

---

## 4.4 데이터 경로 설정

### data_mouse_train.txt 형식

```text
/path/to/processed/sample_000
/path/to/processed/sample_001
/path/to/processed/sample_002
...
```

각 줄은 `opencv_cameras.json`이 있는 샘플 디렉토리 경로입니다.

### 상대 경로 vs 절대 경로

```yaml
# 절대 경로 (권장)
dataset_path: "/home/user/data/facelift_mouse/data_mouse_train.txt"

# 상대 경로 (프로젝트 루트 기준)
dataset_path: "data_mouse_train.txt"
```

---

## 4.5 WandB 설정

### 로컬 로그인

```bash
wandb login
# API key 입력 (wandb.ai 에서 확인)
```

### 오프라인 모드

```yaml
wandb:
  offline: true  # 인터넷 없이 로컬 저장
```

### 프로젝트 구조

```yaml
wandb:
  project: "facelift_mouse"    # 프로젝트 이름
  group: "gslrm"               # 그룹 (여러 실험 묶음)
  exp_name: "mouse_v1"         # 실험 이름 (고유해야 함)
```

---

## 4.6 설정 검증

```python
# Config 로드 테스트
import yaml
from easydict import EasyDict as edict

with open("configs/mouse/gslrm.yaml") as f:
    config = edict(yaml.safe_load(f))

# 필수 설정 확인
assert config.mouse.use_mouse_dataset == True
assert config.training.dataset.num_views == 6
assert config.training.dataset.num_input_views == 5

print("✅ Config validation passed!")
print(f"  - Dataset: {config.training.dataset.dataset_path}")
print(f"  - Checkpoint: {config.training.checkpointing.resume_ckpt}")
print(f"  - LR: {config.training.optimizer.lr}")
```

---

## 다음 단계

✅ Config 설정 완료

→ [Step5: 학습 실행](./Step5_Training.md)

---

*Created: 2026-01-13*
