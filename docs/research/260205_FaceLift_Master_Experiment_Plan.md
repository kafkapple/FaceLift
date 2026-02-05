# FaceLift Master Experiment Plan

> **Ultimate Goal**: Single-view 이미지/영상 → 시간적으로 일관된 3D 생쥐 재구성
> **핵심 문제**: E2E 추론 시 프레임 간 겹침/떨림 현상 (MVDiffusion 뷰 불일치 추정)

---

## 1. 시스템 아키텍처 및 모듈별 책임

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FaceLift E2E Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Input]          [MVDiffusion]           [GS-LRM]         [Deformation]   │
│  Single View  →   Multi-view Gen   →   3D Gaussians  →   Temporal Reg     │
│  (1 view)         (1→6 views)           (4 input)         (t, t+1 pair)    │
│                                                                             │
│  ▼ 책임                                                                     │
│  - 단일 뷰           - 뷰 일관성          - 3D 품질          - 시간 일관성    │
│                    - 프롬프트 영향        - 카메라 정규화     - 드리프트 방지  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 현재 문제 진단

### 2.1 E2E 추론 시 프레임 겹침 현상

| 증상 | 추정 원인 | 검증 방법 |
|------|-----------|-----------|
| 프레임 간 겹침/떨림 | MVDiffusion 뷰 불일치 | H1 실험 (E2E vs GS-LRM) |
| 시간 흐름 무시 | 프레임별 독립 추론 | Deformation V2 적용 |
| Test set 품질 저하 | 일반화 실패 | Train/Test 비교 |

### 2.2 현재 완료된 검증

| 항목 | 결과 | 상태 |
|------|------|------|
| V1 vs V2 드리프트 | V1=0.78, V2=0.00 | ✅ V2 우월 |
| 50프레임 추출 | 연속 순차 | ⚠️ 샘플링 개선 필요 |
| MVDiffusion M5t2_cfgr | 학습 중 | 🔄 진행 |

---

## 3. 모듈별 가설 및 실험 계획

### 3.1 MVDiffusion 모듈

#### 핵심 가설

| ID | 가설 | 우선순위 | 상태 |
|----|------|----------|------|
| **D1** | CFG dropout이 뷰 일관성에 영향 | P1 | 🔄 M5t2_cfgr 실험 중 |
| **D2** | Full attention이 sparse보다 일관성 향상 | P1 | ✅ M5t2_consistent 완료 |
| **D3** | 학습 스텝 부족 (10k → 20k) | P2 | 📋 계획 |
| **D4** | 프롬프트 타입별 품질 차이 | P3 | 📋 계획 |

#### 실험 설정 비교

| Config | CFG Drop | Attention | Steps | 목적 |
|--------|----------|-----------|-------|------|
| M5t2 | 0.05 | Sparse | 10k | Baseline |
| M5t2_consistent | 0.0 | **Full** | 10k | 뷰 일관성 |
| M5t2_cfgr | **0.05** | Full | 10k | CFG 복원 테스트 |
| **M5t2_20k** | 0.05 | Full | **20k** | 장기 학습 |

#### 명령어 (20k 스텝 확장 버전)

```bash
# D3: 20k Steps Extended Training
ssh gpu03
export CUDA_VISIBLE_DEVICES=6 && cd /home/joon/dev/FaceLift && \
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift && \
python train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_20k.yaml \
    2>&1 | tee logs/mvdiff_M5t2_20k.log &
```

### 3.2 GS-LRM 모듈

#### 핵심 가설

| ID | 가설 | 우선순위 | 상태 |
|----|------|----------|------|
| **G1** | Train vs Test 일반화 문제 | P1 | 📋 H1 실험 필요 |
| **G2** | 카메라 정규화 영향 (M5_4, M5_5) | P2 | ✅ 학습 완료 |
| **G3** | 입력 뷰 수 (4 vs 5 vs 6) | P3 | 📋 계획 |

#### H1 실험: 오류 원인 진단 (2×2 Factorial)

| 실험 | Pipeline | Data | 예상 결과 |
|------|----------|------|-----------|
| H1a | GS-LRM only | Train | 최고 품질 (upper bound) |
| H1b | GS-LRM only | Test | 일반화 테스트 |
| H1c | E2E | Train | MVDiffusion 영향 측정 |
| H1d | E2E | Test | 실제 사용 시나리오 |

**해석 가이드**:
- H1a ≈ H1b → Test 일반화 OK
- H1c << H1a → MVDiffusion이 병목
- H1b << H1a → GS-LRM 일반화 실패

```bash
# H1a: GS-LRM + Train
python -m mouse_extensions.scripts.inference.run_evaluation \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5_E0_1_facelift/latest.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --num_samples 50 --output_dir /node_data/joon/outputs/FaceLift/eval/H1a_gslrm_train

# H1b: GS-LRM + Test
python -m mouse_extensions.scripts.inference.run_evaluation \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5_E0_1_facelift/latest.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --num_samples 50 --output_dir /node_data/joon/outputs/FaceLift/eval/H1b_gslrm_test

# H1c: E2E + Train
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiff_ckpt /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000 \
    --gslrm_ckpt /node_data/joon/checkpoints/FaceLift/gslrm/M5_E0_1_facelift/latest.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --num_samples 50 --output_dir /node_data/joon/outputs/FaceLift/eval/H1c_e2e_train

# H1d: E2E + Test
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --mvdiff_ckpt /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr/checkpoint-10000 \
    --gslrm_ckpt /node_data/joon/checkpoints/FaceLift/gslrm/M5_E0_1_facelift/latest.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --num_samples 50 --output_dir /node_data/joon/outputs/FaceLift/eval/H1d_e2e_test
```

### 3.3 Deformation 모듈

#### 핵심 가설

| ID | 가설 | 우선순위 | 상태 |
|----|------|----------|------|
| **F1** | V2 Per-frame이 드리프트 해결 | P1 | ✅ 검증 완료 (drift 0→0.78) |
| **F2** | ARAP + Velocity loss 효과 | P1 | 📋 학습 필요 |
| **F3** | 50 프레임 충분성 | P2 | ⚠️ 샘플링 개선 권장 |

#### 샘플링 전략 (문헌 기반)

| 전략 | 방법 | 장점 | 단점 |
|------|------|------|------|
| **Sequential** (현재) | 처음 N개 | 연속성 | 분포 편향 |
| **Uniform Stride** | 매 K번째 | 전체 커버 | 급격한 변화 놓침 |
| **Stratified** | 구간별 균등 | 균형 | 구현 복잡 |
| **Motion-aware** | 움직임 큰 프레임 | 정보량 최대 | 사전 계산 필요 |

**권장**: 검증용=Sequential(50), 학습용=Stratified(200)+Motion-aware

```bash
# Deformation V2 Training (200 frames, stratified)
python -m mouse_extensions.scripts.train_deformation_v2 \
    --gaussian_dir /node_data/joon/outputs/FaceLift/gaussians/M5t2_stratified200 \
    --output_dir /node_data/joon/checkpoints/FaceLift/deformation/v2_stratified \
    --num_epochs 100 --batch_size 8
```

---

## 4. 데이터셋 정리

### 4.1 Split 비교

| Split | 비율 | Train/Val/Test | 특징 | 상태 |
|-------|------|----------------|------|------|
| **M5** | 8:1:1 | 2880/360/360 | 랜덤 (data leakage 위험) | ⚠️ 참조용 |
| **M5t** | 1:1:1 | 1200/1200/1200 | 시간순 | 📋 비교용 |
| **M5t2** | 8:1:1 | 2880/360/360 | 시간순 + 많은 학습 데이터 | ✅ **권장** |

### 4.2 뷰 사용 현황

| 모델 | Input Views | Target Views | 평가 뷰 |
|------|-------------|--------------|---------|
| MVDiffusion | 1 (view 0) | 6 (all) | 생성 뷰 5개 |
| GS-LRM | 4 (random) | 6 (all) | 나머지 2개 |
| E2E | 1 | 6 → 4 → 6 | Turntable 360° |

---

## 5. 우선순위 실험 로드맵

### Phase 1: 병목 진단 (현재)

| 순서 | 실험 | 목적 | 예상 기간 |
|------|------|------|-----------|
| 1.1 | H1 (4개 조합) | E2E vs GS-LRM 병목 식별 | 1일 |
| 1.2 | M5t2_cfgr 완료 대기 | CFG 영향 확인 | 진행 중 |
| 1.3 | Deformation V2 학습 | 시간 일관성 | 1일 |

### Phase 2: 모듈 최적화

| 순서 | 실험 | 조건 | 예상 기간 |
|------|------|------|-----------|
| 2.1 | MVDiff 20k steps | Phase 1 결과 따라 | 2일 |
| 2.2 | GS-LRM view ablation | H1b 낮으면 | 1일 |
| 2.3 | Deformation 200 frames | F2 성공 시 | 1일 |

### Phase 3: 통합 평가

| 순서 | 실험 | 목적 |
|------|------|------|
| 3.1 | E2E + Deformation V2 | 시간 일관성 통합 |
| 3.2 | Test set 영상 생성 | 최종 평가 |
| 3.3 | 정량/정성 비교표 | 논문 준비 |

---

## 6. Quick Start Commands

### 6.1 현재 최우선 실험

```bash
# 1. H1 실험 (4개 병렬 실행)
ssh gpu03
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

# GPU 4: H1a (GS-LRM + Train)
export CUDA_VISIBLE_DEVICES=4 && python -m mouse_extensions.scripts.inference.run_evaluation \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5_E0_1_facelift/latest.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --num_samples 50 --output_dir /node_data/joon/outputs/FaceLift/eval/H1a &

# GPU 5: H1b (GS-LRM + Test)
export CUDA_VISIBLE_DEVICES=5 && python -m mouse_extensions.scripts.inference.run_evaluation \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5_E0_1_facelift/latest.pt \
    --data_list /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --num_samples 50 --output_dir /node_data/joon/outputs/FaceLift/eval/H1b &
```

### 6.2 MVDiffusion 20k Steps Config 생성

```bash
# Config 파일 생성 (기존 M5t2_cfgr 기반)
cat > configs/mvdiffusion/mouse_mvdiffusion_M5t2_20k.yaml << 'EOF'
# M5t2 + 20k steps (2x extended training)
# Based on M5t2_cfgr (CFG restored + full attention)

n_views: 6
img_wh: 512
reference_view_idx: 0
dataset_type: "mouse"

pretrained_model_name_or_path: "checkpoints/mvdiffusion/pipeckpts"
prompt_embed_path: "mvdiffusion/data/mouse_prompt_embeds_6view_1024/clr_embeds.pt"

train_dataset:
  path: /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt
  bg_color: "three_choices"
  augmentation: true

validation_dataset:
  path: /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt
  bg_color: "white"

output_dir: "/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_20k"
val_out_dir: "/node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_20k/val/"

seed: 42
train_batch_size: 4
max_train_steps: 20000  # 2x extended
gradient_accumulation_steps: 4
gradient_checkpointing: true
learning_rate: 5e-5
lr_scheduler: "piecewise_constant"
lr_warmup_steps: 100
use_ema: true

checkpointing_steps: 2000
checkpoints_total_limit: 3
validation_steps: 400

wandb_exp_name: "mvdiff_M5t2_20k"
tracker_project_name: "FaceLift-MVDiffusion"

use_classifier_free_guidance: true
condition_drop_rate: 0.05
sparse_mv_attention: false  # Full attention
EOF
```

---

## 7. 용량 관리

### 현재 사용량

| 경로 | 용량 | 상태 |
|------|------|------|
| `/node_data/joon/checkpoints/FaceLift/` | ~50GB | ✅ 유지 |
| `/node_data/joon/outputs/FaceLift/gaussians/` | ~3GB | ✅ 최적화됨 |
| `e2e_test/`, `inference_v2/` | ~400GB | ⚠️ 삭제 권장 |

### 삭제 명령어

```bash
# 불필요 캐시 삭제 (492GB 절약)
rm -rf /node_data/joon/outputs/FaceLift/gaussian_cache
rm -rf /node_data/joon/outputs/FaceLift/e2e_test
rm -rf /node_data/joon/outputs/FaceLift/inference_v2
```

---

## 8. 시각화 결과 위치

| 실험 | 위치 | 형식 |
|------|------|------|
| Deformation V1 vs V2 | `/node_data/joon/outputs/FaceLift/deformation/v1_v2_comparison/` | PNG/MP4 |
| MVDiffusion validation | `checkpoints/mvdiffusion/*/val/` | Grid PNG |
| GS-LRM turntable | `outputs/*/turntable_*.mp4` | MP4 |
| E2E inference | `outputs/*/e2e_*.mp4` | MP4 |

---

## 9. 핵심 메트릭

| 메트릭 | 용도 | 목표값 |
|--------|------|--------|
| **PSNR** | 재구성 품질 | > 25 dB |
| **SSIM** | 구조 유사도 | > 0.90 |
| **LPIPS** | 지각 품질 | < 0.15 |
| **Temporal Consistency** | 프레임 일관성 | drift < 0.1 |
| **FID** (MVDiff) | 생성 다양성 | < 50 |

---

## 10. 문헌 참조

### Gaussian Splatting Temporal

| 논문 | 샘플 수 | 전략 | 인사이트 |
|------|---------|------|----------|
| Dynamic 3DGS (CVPR24) | 100-300 | Uniform | 전체 분포 커버 |
| 4D-GS (CVPR24) | 50-150 | Consecutive | 시간 일관성 |
| SC-GS (CVPR24) | ~100 | Motion-aware | 동작 구간 강조 |
| Deformable 3DGS | 200+ | Full | 장기 학습 |

### MVDiffusion / Multi-view

| 논문 | CFG | Attention | 인사이트 |
|------|-----|-----------|----------|
| MVDiffusion | 0.05 | Sparse | 효율적 |
| Era3D | 0.0 | Full | 일관성 우선 |
| Zero123++ | 0.1 | Cross-attn | 균형 |

---

*Created: 2026-02-05 | FaceLift Master Plan v1.0*
