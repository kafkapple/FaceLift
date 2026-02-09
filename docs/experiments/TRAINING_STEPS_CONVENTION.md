# Training Steps Convention Report

학습량 표기 기준과 FaceLift 프로젝트 내 현황을 정리한다.

---

## 1. 분야별 표기 관행

### 1.1 Steps/Iterations 기반 (주류)

3D reconstruction, multi-view generation 분야에서는 **steps (iterations)** 이 표준이다.

| 논문/모델 | 표기 | 설정값 |
|-----------|------|--------|
| **GS-LRM** (ECCV 2024) | steps | pretrain 80K, finetune 20K |
| **NeRF** (2020) | iterations | 30K iterations |
| **Zero-1-to-3** (2023) | steps | 105K steps |
| **MVDiffusion** (NeurIPS 2023) | steps | step 기반 |
| **Stable Diffusion** | steps | 800K steps |

**Why steps?**
- Batch size, GPU 수에 독립적인 단위
- Learning rate schedule이 step 기반으로 설계됨
- 다른 연구자가 재현할 때 가장 명확

### 1.2 Epochs 기반 (소수)

| 논문/모델 | 표기 | 설정값 |
|-----------|------|--------|
| **LGM** (ECCV 2024) | epochs | 30 epochs finetune |
| **Pose-Splatter** (NeurIPS 2025) | epochs | 40-75 epochs + early stopping |

**Why epochs?**
- 데이터셋 크기 대비 학습량 직관적
- 데이터셋이 바뀌면 epoch 수 유지하고 step은 자동 조정

### 1.3 권장 표기 (논문 작성 시)

```
We train for 15K steps (10 epochs) on 2880 training samples
with batch size 2 on a single A6000 GPU (~8 hours).
```

**Steps + epochs 병기 + hardware 명시**가 가장 완전한 표기이다.

---

## 2. FaceLift 현황

### 2.1 GS-LRM (train_gslrm.py)

**Stop mechanism**: `max_fwdbwd_passes` (step 기반)

| 설정 | 값 | 의미 |
|------|-----|------|
| `max_fwdbwd_passes` | 15000 | 실제 stop 조건 |
| `num_epochs` | 50000 | 사실상 비활성 (min으로 선택 안됨) |
| `batch_size_per_gpu` | 2 | |
| `grad_accum_steps` | 1 | |
| **effective batch** | **2** | |
| **steps/epoch** | **1440** | 2880 samples / 2 |

**⚠️ +1 Epoch 반올림 문제**:

```python
# gslrm/model/utils_train.py:196
num_epochs = min(num_epochs, int(max_fwdbwd_passes / fwdbwd_per_epoch) + 1)
#          = min(50000,      int(15000 / 1440) + 1)
#          = min(50000,      10 + 1) = 11

actual_steps = 1440 × 11 = 15840   # 15000이 아닌 15840
```

코드는 항상 **정수 epoch 경계**에서 종료한다.
`max_fwdbwd_passes`는 "최소 이만큼"의 의미이며, 실제로는 올림된 epoch × steps/epoch이 적용된다.

| 의도한 steps | 계산 | 실제 epochs | 실제 steps |
|-------------|------|------------|------------|
| 15000 | int(10.41)+1=11 | 11 | **15840** |
| 14400 | int(10.0)+1=11 | 11 | **15840** (동일!) |
| 14399 | int(9.99)+1=10 | 10 | **14400** |
| 10000 | int(6.94)+1=7 | 7 | **10080** |
| 20000 | int(13.88)+1=14 | 14 | **20160** |

**대안**: `num_epochs`를 직접 설정하면 정확한 epoch 수 제어 가능.
```yaml
num_epochs: 10              # → 정확히 10 epochs = 14400 steps
max_fwdbwd_passes: 999999   # epoch가 제한
```

### 2.2 MVDiffusion (train_diffusion.py)

**Stop mechanism**: `max_train_steps` (step 기반, 정확히 끊김)

| 설정 | 값 | 의미 |
|------|-----|------|
| `max_train_steps` | 10000 | 정확히 이 step에서 종료 |
| `train_batch_size` | 4 | per device |
| `gradient_accumulation_steps` | 4 | |
| **effective batch** | **16** | 4 × 1 GPU × 4 accum |
| **optimizer steps/epoch** | **180** | 2880 / 4 / 4 |

```python
# train_diffusion.py:1165
if global_step >= cfg.max_train_steps:
    break  # 정확히 step에서 끊김, epoch 경계 무관
```

| Config | max_train_steps | ~epochs |
|--------|----------------|---------|
| M5t2 (기본) | 10000 | ~55.6 |
| M5t2_20k | 20000 | ~111.1 |
| M5t2_consistent | 10000 | ~55.6 |
| M5t2_cfgr | 10000 | ~55.6 |

**MVDiffusion은 +1 반올림 문제 없음** — step에서 정확히 종료.

---

## 3. 비교 요약

| | GS-LRM | MVDiffusion |
|--|--------|-------------|
| Stop 단위 | `max_fwdbwd_passes` | `max_train_steps` |
| 정확도 | epoch 경계로 올림 (+1) | **정확히** step에서 종료 |
| 설정값 | 15000 | 10000 |
| **실제 steps** | **15840** (11 epochs) | **10000** (~55.6 epochs) |
| Effective batch | 2 | 16 |
| Train samples | 2880 (M5t2) | 2880 (M5t2) |
| Steps/epoch | 1440 | 180 |

두 모델의 epoch 수는 직접 비교 대상이 아니다.
모델 아키텍처, loss function, learning rate 모두 다르다.
**각 모델의 step 수를 기준으로 보고**하는 것이 적절하다.

---

## 4. 현재 진행 중 실험의 실제 step 수

### H4 View Ablation (GS-LRM)

모든 실험: `max_fwdbwd_passes: 15000` → **실제 15840 steps (11 epochs)**

| 실험 | GPU | 실제 steps |
|------|-----|-----------|
| baseline_v2 | - | 1 (재실행 예정: true zero-shot) |
| 1view_v2 | 6 | 15840 |
| 2view_v2 | 6 | 15840 |
| 3view_v2 | 7 | 15840 |
| 4view_v2 | 5 | 15840 |
| 5view_v2 | 7 | 15840 |
| 6view_v2 | (미실행) | 15840 |

### H5 MVDiffusion

| 실험 | 실제 steps |
|------|-----------|
| M5t2 (기본) | 10000 |
| M5t2_consistent (cyclic) | 10000 |

### H6 Alpha Mask (GS-LRM, 계획)

base_uniform_v2 상속 → **실제 15840 steps (11 epochs)**

### H7 SSIM Weight (GS-LRM, 계획)

base_uniform_v2 상속 → **실제 15840 steps (11 epochs)**

---

## 5. 향후 고려사항

### Option A: 현행 유지
- 모든 GS-LRM 실험이 동일하게 15840 steps
- 비교 공정성 유지
- 논문 표기: "~15.8K steps (11 epochs)"

### Option B: 10 epochs로 통일
- `num_epochs: 10`, `max_fwdbwd_passes: 999999`
- 정확히 14400 steps
- 논문 표기: "14.4K steps (10 epochs)"
- **현재 진행 중인 실험 재시작 필요**

### 결정 보류
- 현재 실험 완료 후 결과 확인
- 필요 시 다음 라운드에서 변경

---

*Created: 2026-02-07 | FaceLift Training Convention Report*
