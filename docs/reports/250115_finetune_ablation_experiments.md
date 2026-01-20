# Mouse GS-LRM Finetune Ablation Experiments

**Date**: 2025-01-15  
**Base Checkpoint**: `ckpt_0000000000021125.pt` (step 21k, Human pretrained)  
**Training Mode**: Finetune (`reset_training_state: true` → step 0부터 시작)

---

## 1. Experiment Overview

4가지 실험을 통해 Mouse 데이터에 대한 최적 설정을 탐색한다.

| Experiment | Config | Purpose |
|------------|--------|---------|
| v15_reproduce | `gslrm_v15_reproduce.yaml` | 기준선 재현 (Mask Loss + v12) |
| v48 | `gslrm_v48_v13_square.yaml` | 정사각 픽셀 데이터셋 효과 |
| v49 | `gslrm_v49_no_mask.yaml` | Mask Loss 제거 효과 |
| v54 | `gslrm_v54_paper.yaml` | 원논문 설정 재현 |

---

## 2. Configuration Comparison

### 2.1 Key Differences

| Setting | v15_reproduce | v48 | v49 | v54 (Paper) |
|---------|---------------|-----|-----|-------------|
| **Dataset** | v12_centered | v13_original_ratio | v12_centered | v13_original_ratio |
| **Input Views** | 5 | 5 | 5 | **4** |
| **Masked L2 Loss** | ✅ | ✅ | ❌ | ❌ |
| **Masked SSIM Loss** | ✅ | ✅ | ❌ | ❌ |
| **Masked PixelAlign** | ✅ | ✅ | ❌ | ❌ |
| **BG Loss Weight** | 1.0 | 1.0 | 0.0 | 0.0 |
| **Perceptual Weight** | 0.5 | 0.5 | 0.5 | **0.1** |

### 2.2 Dataset Specifications

| Dataset | fx (focal) | Pixel Type | Description |
|---------|------------|------------|-------------|
| **v12_centered** | 551.35 | Rectangular | 중심 정렬, FaceLift Human과 동일한 fx |
| **v13_original_ratio** | 549 | Square | 원본 비율 유지, 정사각 픽셀 |

### 2.3 Common Settings (All Experiments)

```yaml
# Model
image_size: 512
patch_size: 8
n_gaussians: 2
hard_pixelalign: true

# Data
num_views: 6  # total views per sample
background_color: white

# Loss
l2_loss_weight: 1.0
lpips_loss_weight: 0.0
ssim_loss_weight: 0.0

# Optimizer
lr: 1.0e-06
beta1: 0.9
beta2: 0.95
weight_decay: 0.05

# Training
batch_size: 2 (x grad_accum 2 = effective 4)
max_fwdbwd_passes: 15000
val_every: 200
```

---

## 3. Experimental Design Rationale

### 3.1 v15_reproduce (Baseline)
- **목적**: 이전 best 설정 재현
- **핵심**: Mask를 통해 foreground에 집중, 배경도 학습

### 3.2 v48 (Square Pixel Dataset)
- **목적**: 데이터 전처리 방식의 영향 검증
- **가설**: 정사각 픽셀이 모델의 공간 이해에 더 적합할 수 있음
- **변수**: Dataset only (v12 → v13)

### 3.3 v49 (No Mask Loss)
- **목적**: Mask Loss의 효과 검증
- **가설**: 배경 포함 전체 이미지 학습이 더 robust할 수 있음
- **변수**: Mask flags + BG loss

### 3.4 v54 (Paper Setting)
- **목적**: FaceLift 원논문 설정 재현
- **핵심 차이**:
  - 4 input views (vs 5): 더 challenging한 설정
  - perceptual 0.1 (vs 0.5): L2에 더 집중

---

## 4. Execution Commands

```bash
# v15_reproduce (GPU 4)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
  train_gslrm.py --config configs/mouse/gslrm_v15_reproduce.yaml \
  > logs/v15_reproduce.log 2>&1 &

# v48 (GPU 4)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
  train_gslrm.py --config configs/mouse/gslrm_v48_v13_square.yaml \
  > logs/v48_v13.log 2>&1 &

# v49 (GPU 5)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
  train_gslrm.py --config configs/mouse/gslrm_v49_no_mask.yaml \
  > logs/v49_nomask.log 2>&1 &

# v54 (GPU 5)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
  train_gslrm.py --config configs/mouse/gslrm_v54_paper.yaml \
  > logs/v54_paper.log 2>&1 &
```

---

## 5. WandB Tracking

| Experiment | WandB Name | Project | Group |
|------------|------------|---------|-------|
| v15_reproduce | mouse_v15_reproduce | FaceLift-Mouse | gslrm |
| v48 | mouse_v48_v13_square | FaceLift-Mouse | gslrm |
| v49 | mouse_v49_no_mask | FaceLift-Mouse | gslrm |
| v54 | mouse_v54_paper | FaceLift-Mouse | gslrm |

---

## 6. Expected Outcomes & Metrics

### Primary Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity (higher is better)
- **LPIPS**: Learned Perceptual (lower is better)

### Evaluation Focus
1. Foreground reconstruction quality
2. Background handling
3. Novel view synthesis consistency
4. Convergence speed

---

## 7. Future Experiments (Priority Ordered)

### P0: Critical (Current Ablation 결과에 따라)

#### 7.1 v55: Paper + Mask Loss (v54 vs v15 hybrid)
- **근거**: v54(paper)와 v15(mask) 중 어느 요소가 더 중요한지 분리 검증
- **설정**: 4 input views + perceptual 0.1 + **Mask Loss ON**
- **목적**: Mask Loss가 4-view 설정에서도 유효한지 확인

#### 7.2 v56: v13 Dataset + No Mask (v48 vs v49 hybrid)
- **근거**: Dataset과 Mask를 동시에 바꾼 실험이 없음
- **설정**: v13_original_ratio + Mask OFF + BG 0.0
- **목적**: 정사각 픽셀 + 전체 이미지 학습의 조합 효과

#### 7.3 v57: 6 Input Views (Maximum Information)
- **근거**: 모든 뷰를 input으로 사용하여 reconstruction 품질 극대화
- **설정**: num_input_views=6, target_has_input=true
- **목적**: 최대 정보량으로 학습 시 성능 상한선 확인

### P1: High Priority

#### 7.4 Learning Rate Exploration
- **근거**: 현재 모든 실험이 lr=1e-6 고정
- **설정**: lr in {5e-7, 2e-6, 5e-6} 탐색
- **목적**: Finetune 최적 학습률 탐색

#### 7.5 Perceptual Loss Weight Ablation
- **근거**: 0.1 (paper) vs 0.5 (current) 큰 차이
- **설정**: perceptual in {0.0, 0.1, 0.3, 0.5}
- **목적**: Mouse 데이터에 최적인 perceptual weight 탐색

### P2: Medium Priority

#### 7.6 Input Views Ablation
- **근거**: 4 vs 5 vs 6 views의 trade-off 불명확
- **설정**: num_input_views in {3, 4, 5, 6}
- **목적**: Mouse 데이터에서 최소 필요 view 수 파악

#### 7.7 LPIPS Loss 추가
- **근거**: 현재 모든 실험 lpips=0.0
- **설정**: lpips_loss_weight in {0.05, 0.1}
- **목적**: Perceptual quality 향상 여부 확인

### P3: Low Priority (Long-term)

#### 7.8 Data Augmentation
- **근거**: 현재 augmentation disabled
- **설정**: 회전, 스케일, 색상 augmentation
- **목적**: Generalization 향상

#### 7.9 Multi-GPU Training
- **설정**: 2-4 GPU with larger batch
- **목적**: 학습 속도 향상 및 batch size 효과 검증

---

## 8. Decision Tree (Next Steps)

```
Current Experiments Complete
           |
           v
   +-------------------+
   | v15 vs v49 비교   | -> Mask Loss 효과 판단
   +-------------------+
           |
     +-----+-----+
     v           v
  Mask 승    No-Mask 승
     |           |
     v           v
   v55 실행    v56 실행
  (Paper+Mask) (v13+NoMask)
           |
           v
   +-------------------+
   | v48 vs v15 비교   | -> Dataset 효과 판단
   +-------------------+
           |
           v
   최적 Dataset + Mask 조합 확정
           |
           v
   v57 (6-view) 실험으로 상한선 확인
           |
           v
   LR / Perceptual 탐색 (P1)
```

---

## Appendix: Config File Locations

```
configs/mouse/
├── gslrm_v15_reproduce.yaml   # Baseline
├── gslrm_v48_v13_square.yaml  # Square pixel
├── gslrm_v49_no_mask.yaml     # No mask loss
└── gslrm_v54_paper.yaml       # Paper setting
```
