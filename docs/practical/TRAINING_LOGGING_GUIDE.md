# Training & Logging Guide

> **목적**: 학습 진행 상황 모니터링, WandB 로깅 이해
> **최종 업데이트**: 2026-01-25

---

## 1. 학습 단위 (Training Units)

### 1.1 핵심 개념

| 단위 | 변수명 | 설명 | 기본값 |
|------|--------|------|--------|
| **Step** | `fwdbwd_pass_step` | Forward-Backward pass 횟수 | - |
| **Update** | `param_update_step` | 파라미터 업데이트 횟수 | step / grad_accum |
| **Epoch** | `epoch` | 데이터셋 전체 순회 횟수 | - |

### 1.2 관계

```
Step = Forward-Backward Pass (1 batch 처리)
Update = Step / grad_accum_steps (파라미터 실제 갱신)
Epoch = 전체 데이터셋 1회 순회

예시 (grad_accum_steps=1):
  Step 100 = Update 100 = Epoch (100 / batch_count)
```

### 1.3 학습 종료 조건

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `max_fwdbwd_passes` | **15000** | 최대 step 수 (주요 종료 조건) |
| `num_epochs` | 50000 | 최대 epoch 수 (보조) |
| `early_stop_after_epochs` | 50000 | Early stopping |

**실제 종료**: `max_fwdbwd_passes` (15000 step) 도달 시

---

## 2. Schedule 설정

### 2.1 기본 설정 (gslrm_mouse.yaml)

```yaml
training:
  schedule:
    num_epochs: 50000
    max_fwdbwd_passes: 15000   # ★ 실제 종료 조건
    warmup: 500                 # Warmup steps

  runtime:
    grad_accum_steps: 1         # Gradient accumulation
```

### 2.2 Warmup

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `warmup` | 500 | LR warmup steps |
| `l2_warmup_steps` | 500 | L2 loss warmup |

---

## 3. WandB Logging

### 3.1 로깅 주기

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `print_every` | 10 | 콘솔 출력 주기 |
| `log_every` | 10 | WandB 로깅 주기 |
| `vis_every` | 100 | 시각화 생성 주기 |
| `val_every` | 200 | Validation 주기 |
| `checkpoint_every` | 100 | 체크포인트 저장 주기 |

### 3.2 로깅 메트릭

#### Train 메트릭 (`train/*`)

| 메트릭 | 설명 | 정상 범위 |
|--------|------|----------|
| `train/loss` | 총 손실 | 0.1 ~ 0.5 |
| `train/psnr` | Train PSNR | 20 ~ 30 |
| `train/l2_loss` | L2 loss | 0.01 ~ 0.1 |
| `train/perceptual_loss` | Perceptual loss | 0.05 ~ 0.2 |
| `train/alpha_loss` | Alpha loss (E1/E2) | 0.01 ~ 0.1 |

#### Validation 메트릭 (`val/*`)

| 메트릭 | 설명 | 좋은 값 |
|--------|------|--------|
| `val/psnr` | Validation PSNR | **20+** (D7_1), **27+** (D3) |
| `val/ssim` | Structural Similarity | 0.95+ |
| `val/lpips` | Perceptual Distance | 0.02 이하 |
| `val/mask_iou` | Mask IoU | 0.8+ |

#### 기타

| 메트릭 | 설명 |
|--------|------|
| `grad_norm` | Gradient norm |
| `lr` | Learning rate |
| `ghost/*` | Ghosting 관련 (있을 경우) |

### 3.3 WandB에서 현재 Step 확인

```
WandB Dashboard → Run 선택 → Charts

X축: step (= fwdbwd_pass_step)
주요 확인:
- train/psnr: 상승 추세
- train/loss: 하락 추세
- val/psnr: 목표 PSNR 도달 여부
```

---

## 4. 진행 상황 확인 방법

### 4.1 콘솔 로그

```bash
# 실시간 로그 확인
tail -f logs/D3_E0_1_facelift.log

# 현재 step 확인
grep step= logs/*.log | tail -5

# PSNR 추이 확인
grep psnr logs/*.log | tail -10
```

### 4.2 WandB 대시보드

```
https://wandb.ai/{user}/FaceLift-Mouse

필터:
- Group: mouse
- State: running / finished
```

### 4.3 진행률 계산

```python
# 현재 진행률
progress = current_step / max_fwdbwd_passes * 100

# 예시: step 5000 / 15000
# progress = 5000 / 15000 * 100 = 33.3%
```

---

## 5. 주요 Step 이정표

| Step | 진행률 | 예상 상태 |
|------|--------|----------|
| 0 | 0% | 시작 |
| 500 | 3% | Warmup 완료 |
| 1000 | 7% | 초기 수렴 확인 |
| 3000 | 20% | 중간 평가 |
| 5000 | 33% | 주요 추세 확립 |
| 10000 | 67% | 후반 미세조정 |
| 15000 | 100% | 학습 완료 |

---

## 6. Validation 일정

```
val_every: 200 (기본값)

Validation 실행 step:
  1, 201, 401, 601, 801, 1001, ...

각 validation에서:
  - val/psnr 계산
  - val/ssim, val/lpips 계산
  - Turntable 렌더링 (vis_every마다)
```

---

## 7. 체크포인트

### 7.1 저장 위치

```
outputs/{exp_name}/checkpoints/
├── ckpt_{step}.pt       # 각 checkpoint_every step
└── ckpt_latest.pt       # 최신
```

### 7.2 Resume

```bash
# 자동 resume (같은 exp_name)
torchrun ... train_gslrm.py -d D3_normalized -e E0_1_facelift

# 수동 resume
# config에서:
training:
  checkpointing:
    resume_ckpt: outputs/exp_name/checkpoints/ckpt_5000.pt
```

---

## 8. 트러블슈팅

### 8.1 학습 진행 안 됨

```
증상: step이 증가하지 않음
확인:
1. GPU 메모리 (nvidia-smi)
2. 로그 에러 메시지
3. wandb offline 모드 여부
```

### 8.2 PSNR 낮음

```
증상: val/psnr < 15
가능 원인:
1. fx 미정규화 (M3 버그)
2. 데이터셋 문제
3. config 오류

확인: wandb에서 train/psnr vs val/psnr gap
```

### 8.3 NaN 발생

```
증상: loss = NaN
가능 원인:
1. lr 너무 높음
2. bf16 + log 연산
3. 데이터 손상

해결: grad_clip_norm 추가, lr 낮춤
```

---

## 9. 설정 예시

### 9.1 빠른 테스트 (1000 step)

```yaml
training:
  schedule:
    max_fwdbwd_passes: 1000
```

### 9.2 긴 학습 (30000 step)

```yaml
training:
  schedule:
    max_fwdbwd_passes: 30000
```

---

## 10. 관련 문서

| 문서 | 내용 |
|------|------|
| [EXPERIMENT_SCHEMA](./EXPERIMENT_SCHEMA.md) | 실험 명명 규칙 |
| [EXPERIMENT_REGISTRY](./EXPERIMENT_REGISTRY.md) | 실험 목록 |

---

*Training & Logging Guide v1.0 | 2026-01-25*
