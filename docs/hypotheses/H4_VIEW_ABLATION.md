# H4: View Ablation

> ← [RESEARCH_HYPOTHESES.md](../RESEARCH_HYPOTHESES.md) | **상태**: 🔄 학습중 | **Updated**: 2026-02-09
**목표**: 통일된 조건에서 1-6개 입력 뷰에 따른 GS-LRM 재구성 품질 비교

---

## 1. 연구 가설

### H1: 최적 뷰 수 존재 (Primary)

**가설**: 3-4개 뷰가 최적 성능을 보일 것이다.

**근거**:
- 이전 R1(비균일): 3-view (21.12) > 4-view (19.58) → **v2(uniform)에서 반전: 단조 증가**
- 뷰가 너무 적으면 정보 부족, 너무 많으면 과적합/노이즈 증폭
- GS-LRM 논문: "4 views are sufficient for high-quality reconstruction"

**검증 방법**: 1-6 view 실험 후 PSNR/SSIM/LPIPS 곡선 분석

**예상 결과**: ~~1v < 2v < 3v ≈ 4v > 5v > 6v (inverted-U shape)~~ → **실제: 단조 증가 (1v < 2v < 3v < 4v < 5v < 6v)**

---

### H2: Fine-tuning 효과 (Secondary)

**가설**: Zero-shot 대비 fine-tuning이 유의미한 개선을 보일 것이다.

**근거**:
- Pretrained 모델은 Objaverse 데이터 기반
- Mouse 데이터는 domain shift 존재 (동물 vs 일반 객체)
- 15K fwdbwd passes fine-tuning으로 domain adaptation 기대

**검증 방법**: baseline (zero-shot) vs 1view 이상 비교

**예상 결과**: baseline < 1view, PSNR 개선폭 +2~5 dB 예상

---

### H3: 6-view Overfitting 위험 (Tertiary)

**가설**: 6-view 학습은 training views에 과적합될 위험이 있다.

**근거**:
- 6-view = 모든 카메라 사용 → target과 input 중복 가능
- target_has_input: true 설정 시 same-view 평가
- Hold-out view가 없어 일반화 능력 측정 불가

**검증 방법**: 5v vs 6v val PSNR 비교

**주의사항**:
- 6-view 학습 시 target_has_input: true → 전체 뷰 평가 (same-view 포함)
- target_has_input: false → target 빈 리스트 → validation 불가

---

## ⚠️ Dead Config 주의사항

`train_gslrm.py`에는 **실제로 동작하지 않는 config 경로 3가지**가 존재한다.
실험 config 작성 시 반드시 인지해야 한다.

| Dead Config | 왜 동작 안 하는가 | 실제 동작하는 대안 |
|-------------|-------------------|-------------------|
| `max_steps` | `train_gslrm.py`가 읽지 않음. 종료 조건은 `max_fwdbwd_passes`만 사용 | `max_fwdbwd_passes` |
| `training.schedule.val_every` | 코드에서 참조하지 않음. validation 주기에 영향 없음 | `validation.val_every` |
| `max_fwdbwd_passes` (소수값) | epoch 단위로 올림 처리됨. 예: `1` → 1 epoch = 1440 steps 실행 | `early_stop_after_epochs` (정확한 epoch 제어) |

**핵심**: `max_fwdbwd_passes`는 epoch 경계에서만 체크되므로, "1 pass 후 즉시 종료"는 불가능하다.
Zero-shot baseline 구현 시 `early_stop_after_epochs: 1`이 실제 종료 메커니즘이다.

---

## 2. 실험 설계

### 2.1 통일 조건 (base_uniform_v2.yaml)

| 항목 | 값 | 이유 |
|------|-----|------|
| max_fwdbwd_passes | 15,000 | ~11 epochs (15840 actual, +1 rounding) |
| seed | 42 | 재현성 |
| batch_size | 2 | GPU 메모리 최적화 |
| random_view_selection | true | 뷰 조합 다양성 |
| dataset | M5t2 | 80:10:10 temporal split |
| pretrained | ckpt_21125 | 동일 초기화 |

### 2.2 실험별 변수

| 실험 | num_input_views | 비고 |
|------|-----------------|------|
| baseline | 6 | **True Zero-shot** (validate_before_training, no gradient update) |
| 1view | 1 | 최소 뷰 |
| 2view | 2 | |
| 3view | 3 | |
| 4view | 4 | E0_1_facelift 동등 |
| 5view | 5 | |
| 6view | 6 | 최대 뷰 |

### 2.2.1 baseline_v2 수정 이력 (260207)

Zero-shot baseline 구현 과정에서 dead config와 코드 구조 문제를 순차적으로 발견:

1. `max_steps: 1` - Dead config (코드 미참조)
2. `max_fwdbwd_passes: 1` - +1 epoch 반올림으로 1440 steps 실행
3. `training.schedule.val_every: 1` - Dead config (코드는 `validation.val_every` 참조)
4. `early_stop_after_epochs: 1` - 1 step 학습 후 validation (true zero-shot 아님)
5. 코드 구조: train_step - optimizer_step - validation 순서로 1회 gradient update 포함

**최종 수정**: `train()`에 `validate_before_training` 플래그 추가 (코드 수정)

- `early_stop_after_epochs: 0` - training loop 진입 안 함
- `validate_before_training: true` - train() 시작 시 즉시 validation

**동작**: pretrained 가중치 로드 -> validation 1회 -> training loop skip -> 종료.
gradient update 0회. **진정한 zero-shot 평가**.

-> 상세: [TRAINING_STEPS_CONVENTION.md](../experiments/TRAINING_STEPS_CONVENTION.md)

### 2.3 평가 메트릭

| 메트릭 | 설명 | 참조 |
|--------|------|------|
| PSNR | Peak Signal-to-Noise Ratio | 높을수록 좋음 |
| SSIM | Structural Similarity | 0-1, 높을수록 좋음 |
| LPIPS | Perceptual Similarity | 낮을수록 좋음 |
| Mask IoU | Foreground segmentation 품질 | 높을수록 좋음 |

---

## 3. 실험 명령어

### 3.1 GPU 할당

- GPU 0-3: Blackwell (PyTorch 미지원) ❌
- GPU 4-7: A6000 (48GB) ✅

### 3.2 현재 GPU 배치 상태

| GPU | 실험 | 상태 |
|-----|------|------|
| GPU 4 | available (baseline + 6view 예정) | 대기 |
| GPU 5 | 4view_v2 | 실행 중 |
| GPU 6 | 1view_v2 → 2view_v2 (순차) | 실행 중 |
| GPU 7 | 3view_v2 → 5view_v2 (순차) | 실행 중 |

### 3.3 실행 명령어

```bash
cd /home/joon/dev/FaceLift
mkdir -p logs

# GPU 5: 4-view (E0_1 equivalent)
export CUDA_VISIBLE_DEVICES=5 && \
nohup python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/4view_v2.yaml \
  > logs/uniform_4view_v2.log 2>&1 &

# GPU 6: 1-view → 2-view (순차)
export CUDA_VISIBLE_DEVICES=6 && \
nohup bash -c '
python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/1view_v2.yaml \
  > logs/uniform_1view_v2.log 2>&1 && \
python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/2view_v2.yaml \
  > logs/uniform_2view_v2.log 2>&1
' &

# GPU 7: 3-view → 5-view (순차)
export CUDA_VISIBLE_DEVICES=7 && \
nohup bash -c '
python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/3view_v2.yaml \
  > logs/uniform_3view_v2.log 2>&1 && \
python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/5view_v2.yaml \
  > logs/uniform_5view_v2.log 2>&1
' &

# GPU 4: Baseline (zero-shot) + 6-view (순차, 위 실험들 완료 후)
export CUDA_VISIBLE_DEVICES=5 && \
nohup bash -c '
python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/baseline_v2.yaml \
  > logs/uniform_baseline_v2.log 2>&1 && \
python train_gslrm.py \
  -b configs/mouse/uniform/base_uniform_v2.yaml \
  -e configs/mouse/uniform/6view_v2.yaml \
  > logs/uniform_6view_v2.log 2>&1
' &
```

### 3.4 진행 상황 모니터링

```bash
# 로그 실시간 확인
tail -f logs/uniform_*_v2.log

# GPU 사용량 확인
watch -n 5 nvidia-smi

# WandB에서 확인
# https://wandb.ai/[username]/FaceLift-Mouse
```

---

## 4. 예상 결과 및 분석 계획

### 4.1 예상 소요 시간

| 항목 | 시간 |
|------|------|
| 1 실험 (15840 steps) | ~12-14 시간 |
| 1-5view (병렬, GPU 5-7) | ~24-28 시간 (순차 2개 GPU 기준) |
| baseline + 6view (GPU 4) | ~14 시간 (baseline 빠름 + 6view) |
| 총 소요 시간 | ~28 시간 (병렬 고려) |

### 4.2 결과 분석 계획

1. PSNR vs num_views 그래프 작성
2. 최적 뷰 수 도출 (H1 검증)
3. Zero-shot vs Fine-tuned 개선폭 계산 (H2 검증)
4. 5v vs 6v 비교로 과적합 여부 확인 (H3 검증)

### 4.3 후속 실험

| 조건 | 후속 실험 |
|------|----------|
| 3-4 view 최적 확인 시 | 더 긴 학습 (30K fwdbwd passes) |
| 6-view 과적합 확인 시 | Regularization 추가 |
| 1-view도 준수 시 | Single-view 최적화 연구 |

---

## 5. Config 파일 위치

```
configs/mouse/uniform/
├── base_uniform_v2.yaml    # 공통 base config
├── baseline_v2.yaml        # Zero-shot (early_stop_after_epochs=1)
├── 1view_v2.yaml          # 1-view 학습
├── 2view_v2.yaml          # 2-view 학습
├── 3view_v2.yaml          # 3-view 학습
├── 4view_v2.yaml          # 4-view 학습 (E0_1 equiv)
├── 5view_v2.yaml          # 5-view 학습
└── 6view_v2.yaml          # 6-view 학습
```

**⚠️ Config 작성 시 주의**: Dead config 3가지 (`max_steps`, `training.schedule.val_every`,
`max_fwdbwd_passes` rounding) 숙지 필수. 상단 "Dead Config 주의사항" 섹션 참조.
특히 validation 주기는 반드시 `validation.val_every`로 설정할 것.

---

## 5. Uniform v2 결과 (260209)

> ⚠️ 아래는 val PSNR (학습 중 모니터링). 최종 test evaluation은 학습 완료 후 진행 예정.

| Views | Val PSNR | Best Step | 상태 |
|-------|----------|-----------|------|
| **6** | **23.46** | 601 | 🔄 학습중 |
| 5 | 22.63 | 2901 | 🔄 학습중 |
| 4 | 21.50 | 3801 | 🔄 학습중 |
| 3 | 19.92 | 3801 | 🔄 학습중 |
| 2 | 17.70 | 8801 | ✅ completed |
| 1 | 11.08 | 2401 | ✅ completed |
| baseline | 15.99 | 0 | ✅ zero-shot |

### 핵심 발견

1. **단조 증가**: 뷰 수 ↑ = PSNR ↑ (R1과 반전)
2. **원인**: R1은 비균일 뷰 샘플링, v2는 균일 → 조건 통일의 중요성
3. **baseline**: zero-shot 4-view = 15.99 → 1-view(11.08)보다 높음

→ 상세 결과: [[RESEARCH_HYPOTHESES|RESEARCH_HYPOTHESES.md]]

## 6. 참고 문헌

1. GS-LRM: "GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting" (2024)
2. pixelSplat: "pixelSplat: 3D Gaussian Splats from Image Pairs" (CVPR 2024)
3. LGM: "LGM: Large Multi-View Gaussian Model" (2024)

---

*Created: 2026-02-07 | Updated: 2026-02-09 | Project: FaceLift*
