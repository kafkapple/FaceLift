# Experiment Registry (실험 레지스트리)

> **Navigation**: [← Index](../INDEX.md) | [Quick Ref](../MOUSE_QUICK_REFERENCE.md)

> **SSOT**: 모든 실험 ID, 명령어, 설정의 중앙 관리 문서
> **최종 업데이트**: 2026-01-25

---

## 명명 규칙

```
E{카테고리}_{번호}_{이름}                    - 기본 설정
E{카테고리}_{번호}_{서브번호}_{이름}_{변형}   - 변형 설정

예시:
E0_1_facelift                   - E0 카테고리, 1번, FaceLift 기본
E0_1_1_facelift_alpha           - E0_1_facelift의 1번 변형 (alpha 추가)
E1_2_3_alpha_fixed              - E1_2_alpha의 3번 변형 (fixed view)
```

---

## 카테고리 체계

| 카테고리 | mask_mode | 설명 | 상태 |
|----------|-----------|------|------|
| **E0** | none | FaceLift Baseline | ⭐ **권장** |
| **E1** | gt | GT Mask 기반 | 활성 |
| **E2** | none | Alpha only | 실험적 |

---

## E0: FaceLift Baseline ⭐

> mask_mode=none, FaceLift 논문 기반

### 기본 설정

| ID | 파일 | alpha | view | 설명 |
|----|------|-------|------|------|
| **E0_1_facelift** | `E0_1_facelift.yaml` | 0.0 | random | FaceLift 논문 원본 |
| E0_2_mouse | `E0_2_mouse.yaml` | 0.0 | random | Mouse-adapted (lr 조정) |

### E0_1 변형

| ID | 파일 | alpha | view | 변형 내용 |
|----|------|-------|------|----------|
| E0_1_1_facelift_alpha | `E0_1_1_facelift_alpha.yaml` | **0.1** | random | +alpha |
| E0_1_2_facelift_fixed | `E0_1_2_facelift_fixed.yaml` | 0.0 | **fixed** | +fixed |
| E0_1_3_facelift_alpha_fixed | `E0_1_3_facelift_alpha_fixed.yaml` | **0.1** | **fixed** | +alpha+fixed |

---

## E1: GT Mask

> mask_mode=gt, RGB loss를 GT mask 영역으로 제한

### 기본 설정

| ID | 파일 | alpha | 설명 |
|----|------|-------|------|
| E1_1_base | `E1_1_base.yaml` | 0.0 | GT mask만 |
| **E1_2_alpha** | `E1_2_alpha.yaml` | 0.1 | GT mask + alpha |
| E1_3_lgm | `E1_3_lgm.yaml` | 1.0 | LGM 스타일 |

### E1_2 변형

| ID | 파일 | 변형 내용 |
|----|------|----------|
| E1_2_1_alpha_3v | `E1_2_1_alpha_3v.yaml` | 3 view input |
| E1_2_2_alpha_5v | `E1_2_2_alpha_5v.yaml` | 5 view input |
| E1_2_3_alpha_fixed | `E1_2_3_alpha_fixed.yaml` | fixed view selection |
| E1_2_4_alpha_overfit | `E1_2_4_alpha_overfit.yaml` | overfit 테스트 |

---

## E2: Alpha Only

> mask_mode=none, alpha supervision만

| ID | 파일 | alpha | 설명 |
|----|------|-------|------|
| E2_1_alpha | `E2_1_alpha.yaml` | 0.1 | 마스크 없이 alpha만 |

---

## 데이터셋

### Production

| Alias | Coverage | fx | PP | Val PSNR | 설명 |
|-------|----------|-----|-----|----------|------|
| **D3_normalized** | **84%** | 549 | 256 | **27.1** | ⭐ SOTA |
| M1 (D7_1) | 50% | 549 | 256 | 20.9 | Baseline |
| M2 (D8) | 50% | 549 | 256 | 20.2 | Homography |

### 실험용

| Alias | Coverage | fx | PP | 가설 |
|-------|----------|-----|-----|------|
| M3 | 78% | **739** ❌ | 가변 | ⛔ 사용금지 |
| ~~M3_norm~~ | 78% | 549 | **가변** | ⛔ 폐기 (ray 13.62°) |
| ~~M3_persample~~ | 50% | 549 | **가변** | ⛔ 폐기 (ray 16.15°) |
| **M3_1** | 80% | 549 | **256** | ✅ MVG-correct (Global) |
| **M3_2** | 80% | 549 | **256** | ✅ MVG-correct (Per-sample) ⭐ |

---

## 실행 명령어

### 기본 형식

```bash
CUDA_VISIBLE_DEVICES={GPU} torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d {DATASET} -e {EXPERIMENT}
```

### 권장 실험 (D3_normalized)

```bash
# SOTA 확인
-d D3_normalized -e E0_1_facelift

# +Alpha
-d D3_normalized -e E0_1_1_facelift_alpha

# +Fixed
-d D3_normalized -e E0_1_2_facelift_fixed

# +Alpha+Fixed
-d D3_normalized -e E0_1_3_facelift_alpha_fixed
```

---

## 변경 이력

| 날짜 | 버전 | 변경 |
|------|------|------|
| 2026-01-25 | v2.4 | 명명규칙 통일 (E{cat}_{num}_{subnum}_{name}_{variant}) |
| 2026-01-25 | v2.3 | E0_paper_* 추가 (폐기) |
| 2026-01-25 | v2.2 | random_view_selection 버그 수정 |

---

*Experiment Registry v2.4 | 2026-01-25*

## Historical Experiment Results (from archive)

# Experiment Results Registry

> **SSOT**: 실험 결과 비교표
> **최종 업데이트**: 2026-01-26

---

## Validation PSNR 비교

| Dataset | Transform | Zoom | PP | Val PSNR | 비고 |
|---------|-----------|------|-----|----------|------|
| **D3_normalized** | ? | ? | 256 | **27.09** | ⭐ 최고 (전처리 불명) |
| **D7_1** (M1) | Affine | ❌ | 256 | 20.93 | 기준선 |
| **D8** (M2) | Homography | ❌ | 256 | 20.21 | Affine과 유사 |
| M3_norm | Homo+Zoom | ✅ | **가변** | 17.09 | ❌ PP 불일치 |
| **M3_2** | Homo+Zoom | ✅ | 256 | TBD | ⏳ 검증 예정 ⭐ |

---

## 주요 발견

| 요인 | 영향 | 근거 |
|------|------|------|
| **PP=256 정합** | +3~4 PSNR | M3_norm(17) vs D7_1(21) |
| **Coverage** | +6 PSNR | D7_1(21) vs D3(27) |
| **Transform** | ~0.7 PSNR | Affine ≈ Homography |

---

## 실험 명령어

```bash
# 권장: M3_2 + E1_2_alpha
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1     train_gslrm.py -d M3_2 -e E1_2_alpha
```

---

## 관련 문서

- [[HYPOTHESIS_VERIFICATION]] - 가설 검증 (H1-H6)
- [[../practical/MOUSE_QUICK_REFERENCE]] - 전체 명령어

---

*EXPERIMENT_RESULTS v4.0 | 2026-01-26 | 중복 제거*

## Finetune Ablation Results (from archive)

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

---

## 9. M5t2 View Ablation Study (2026-02-04)

### 9.1 목적
- 균일한 간격의 뷰 수에 따른 3D 재구성 품질 변화 측정
- Minimum viable view count 확인

### 9.2 실험 설계

| Config | 뷰 수 | 간격 | 선택 뷰 | 설명 |
|--------|-------|------|---------|------|
| `M5t2_view_ablation_2v` | 2 | 180° | 0, 3 | 최소 (정면/후면) |
| `M5t2_view_ablation_3v` | 3 | 120° | 0, 2, 4 | 균등 삼각 |
| `M5t2_view_ablation_6v` | 6 | 60° | all | Baseline (전체) |

### 9.3 Config 위치
```
configs/experiments/
├── M5t2_view_ablation_2v.yaml
├── M5t2_view_ablation_3v.yaml
└── M5t2_view_ablation_6v.yaml
```

### 9.4 핵심 설정
```yaml
model:
  num_views: 6
  num_input_views: N  # 2, 3, or 6

training:
  dataset:
    random_view_selection: false
    include_camera_indices: [...]  # 균일 간격 선택
```

### 9.5 학습 명령어

```bash
cd /home/joon/dev/FaceLift

# 2v (180°) - GPU 5
export CUDA_VISIBLE_DEVICES=5 && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e M5t2_view_ablation_2v \
    > logs/gslrm_M5t2_2v.log 2>&1 &

# 3v (120°) - GPU 6
export CUDA_VISIBLE_DEVICES=6 && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e M5t2_view_ablation_3v \
    > logs/gslrm_M5t2_3v.log 2>&1 &

# 6v (60°, baseline) - GPU 7
export CUDA_VISIBLE_DEVICES=7 && nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e M5t2_view_ablation_6v \
    > logs/gslrm_M5t2_6v.log 2>&1 &
```

### 9.6 평가 지표
- PSNR, SSIM, LPIPS (held-out views)
- Alpha IoU, F1 (mask quality)
- Turntable visual quality

### 9.7 결과 (예정)

| Config | PSNR | SSIM | Alpha IoU | 비고 |
|--------|------|------|-----------|------|
| 2v | TBD | TBD | TBD | |
| 3v | TBD | TBD | TBD | |
| 6v | TBD | TBD | TBD | Baseline |

---

## H1 Diagnosis Experiments (260205) ✅

### 목적
GS-LRM 품질 오류 원인 진단: MVDiffusion vs Test 일반화

### 실험 목록

| ID | 모드 | Split | Checkpoint | 결과 |
|----|------|-------|-----------|------|
| h1a | GS-LRM only | Train | M5t2_E0_1_facelift/best_psnr.pt | ✅ |
| h1b | GS-LRM only | Test | M5t2_E0_1_facelift/best_psnr.pt | ✅ |
| h1c | E2E | Train | MVDiff cfgr-10k + GS-LRM best | ✅ |
| h1d | E2E | Test | MVDiff cfgr-10k + GS-LRM best | ✅ |

### 결론
- E2E vs GS-LRM 차이 > Train vs Test 차이
- MVDiffusion이 품질 저하의 주 원인

→ [[260205_FaceLift_Hypothesis_Experiments#8-h1-실험-결과]]

---

## View Ablation Experiments (260205) ⏳

### 목적
MVDiffusion reference view 선택의 영향 분석

| ID | Input View | 상태 |
|----|------------|------|
| h1d | 0 | ✅ 완료 |
| view_1 | 1 | ⏳ 대기 |
| view_2 | 2 | ⏳ 대기 |
| view_3 | 3 | ⏳ 대기 |
| view_4 | 4 | ⏳ 대기 |
| view_5 | 5 | ⏳ 대기 |

→ [[260205_FaceLift_Hypothesis_Experiments#9-view-ablation-실험]]

---

## Cyclic Reference View Augmentation (260205) 🔄

### 설정
- Config: `mouse_mvdiffusion_M5t2_cyclic.yaml`
- reference_view_idx: [0,1,2,3,4,5] (6x augmentation)
- WandB: `mvdiff_M5t2_cyclic`

### 상태
- GPU 4에서 학습 중
- ~80/10000 steps

→ [[260205_FaceLift_Hypothesis_Experiments#10-cyclic-reference-view-augmentation]]

---

## H1 M5t Diagnosis (260205) 🔄

### 목적
M5t2 vs M5t 데이터 양 비교 (2880 vs 1198 Train samples)

### 실험 목록

| ID | 모드 | Split | Dataset | GPU | 상태 |
|----|------|-------|---------|-----|------|
| h1a | GS-LRM | Train | M5t | 5 | 🔄 |
| h1b | GS-LRM | Test | M5t | 5 | ⏳ |
| h1c | E2E | Train | M5t | 6 | 🔄 |
| h1d | E2E | Test | M5t | 7 | 🔄 |

### 비교 대상
- `h1_diagnosis_M5t2` (완료) vs `h1_diagnosis_M5t` (진행중)

→ [[260205_FaceLift_Hypothesis_Experiments#12-m5t2-vs-m5t-데이터-양-비교-실험]]

---

## GS-LRM Input View Ablation (260205) 🔄

### 목적
- GS-LRM 입력 뷰 수에 따른 재구성 품질 변화 측정
- 1-view (sanity check): 이론적으로 불가능 → 구현 검증
- 최소 필요 뷰 수 확인

### Config 위치
```
configs/experiments/view_ablation/
├── E0_1_1view.yaml   # Sanity check (PSNR ~10-15 예상)
├── E0_1_2view.yaml   # Minimal (180°)
├── E0_1_3view.yaml   # Triangle (120°)
├── E0_1_5view.yaml   # Near-full
└── E0_1_6view.yaml   # Full baseline
```

### 뷰 선택 전략

| Views | 선택 | 각도 간격 | 비고 |
|-------|------|----------|------|
| 1 | [0] | - | Sanity check (불가능) |
| 2 | [0, 3] | 180° | 정면/후면 |
| 3 | [0, 2, 4] | 120° | 균등 삼각 |
| 5 | [0, 1, 2, 4, 5] | 72° | Near-full |
| 6 | [0-5] | 60° | Baseline |

### 학습 명령어

```bash
cd /home/joon/dev/FaceLift

# 1-view (Sanity Check) - GPU X
CUDA_VISIBLE_DEVICES=X nohup bash -c 'python train_gslrm.py -d M5t2 -e view_ablation/E0_1_1view' > logs/view_ablation_1view_training.log 2>&1 &

# 2-view - GPU 6
CUDA_VISIBLE_DEVICES=6 nohup bash -c 'python train_gslrm.py -d M5t2 -e view_ablation/E0_1_2view' > logs/view_ablation_2view_training.log 2>&1 &

# 3-view - GPU 5
CUDA_VISIBLE_DEVICES=5 nohup bash -c 'python train_gslrm.py -d M5t2 -e view_ablation/E0_1_3view' > logs/view_ablation_3view_training.log 2>&1 &

# 5-view - GPU X
CUDA_VISIBLE_DEVICES=X nohup bash -c 'python train_gslrm.py -d M5t2 -e view_ablation/E0_1_5view' > logs/view_ablation_5view_training.log 2>&1 &

# 6-view (Baseline) - GPU X
CUDA_VISIBLE_DEVICES=X nohup bash -c 'python train_gslrm.py -d M5t2 -e view_ablation/E0_1_6view' > logs/view_ablation_6view_training.log 2>&1 &
```

### Inference-time Ablation 결과 (260205)

| Views | PSNR | 비고 |
|-------|------|------|
| **3** | **21.12** | ⭐ 최고 |
| 2 | 20.20 | |
| 4 | 19.58 | Baseline |
| 5 | 19.58 | = 4-view |
| 6 | 19.58 | = 4-view |

**발견**: 3-view (120° 간격)가 4-view baseline보다 우수

### Training-time 결과 (진행 중)

| Views | GPU | Train PSNR | Val PSNR | 상태 |
|-------|-----|-----------|----------|------|
| 1 | - | - | - | ⏳ 대기 |
| 2 | 6 | 28.4 | TBD | 🔄 진행 |
| 3 | 5 | 31.0 | TBD | 🔄 진행 |

---

*Updated: 260205 20:30*
