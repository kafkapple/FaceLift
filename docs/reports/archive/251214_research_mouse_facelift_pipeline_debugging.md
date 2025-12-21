---
date: 2025-12-14
tags: [mouse-facelift, mvdiffusion, gslrm, 3d-reconstruction, debugging]
project: mouse-facelift
status: completed
---

# Mouse_FaceLift_3D_Reconstruction_Pipeline_Debugging

**대화 요약**: Mouse-FaceLift 3D 재구성 파이프라인의 MVDiffusion 학습 문제(mode collapse) 해결, Phase 2/3 파이프라인 최적화, 그리고 GS-LRM 전이학습 시 gradient explosion 문제 해결

**주요 다룬 주제**:

1. MVDiffusion 6x Augmentation 실패 원인 분석 및 해결
2. 3-Phase 파이프라인 최적화 (7시간 테스트 모드 설계)
3. GS-LRM 전이학습 시 Gradient Explosion 해결

---

## 1. MVDiffusion 학습 문제 해결

### 1.1 6x Augmentation Mode Collapse 문제

**문제상황**:
- 1500 step에서 조기 수렴, 모든 view가 유사한 출력 생성
- 91% step이 실제로 학습되지 않음

**근본 원인**:
```
6x Augmentation (random reference view) 문제점:
1. Identity Mapping: 1/6 확률로 input = target (자기 자신 복사 학습)
2. High Prompt Similarity: View 간 프롬프트 유사도 0.85-0.99
   → 모델이 view 구분 없이 입력 복사하는 것이 최적해가 됨
```

**해결방법**:
- `reference_view_idx: "random"` → `reference_view_idx: 0` (고정)
- FaceLift 원본 설정으로 복귀
- **결과**: 1300 step만에 정확한 예측 달성

### 1.2 Prompt Embedding 분석

**핵심개념**: MVDiffusion은 CLIP 프롬프트 임베딩으로 view 방향 구분

**비교 분석**:
| 항목 | FaceLift 원본 | Mouse 수정 |
|------|--------------|-----------|
| 프롬프트 | `"front view"` | `"top-front view, from above"` |
| Cosine Similarity | - | 0.70 (30% 차이) |
| View 구분력 | 0.93 | 0.90 |

**활용예시**:
```python
# FaceLift 원본 프롬프트 (권장)
prompt_embed_path: "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"

# Mouse 커스텀 프롬프트 (수렴 느림)
prompt_embed_path: "mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt"
```

---

## 2. 3-Phase 파이프라인 최적화

### 2.1 파이프라인 시간 분석

**문제상황**: 전체 파이프라인 ~25시간 소요, 7시간 내 검증 필요

**해결방법 - 테스트 모드 설계**:

| Phase | Full Train | Test Mode | 시간 절약 |
|-------|-----------|-----------|----------|
| Phase 2 (Synthetic) | 300x6=1800개, 50 steps | 100x6=600개, 25 steps | 8h → 1.5h |
| Phase 3 (GS-LRM) | 30,000 iters | 10,000 iters | 15h → 4.5h |
| **Total** | ~25h | ~7h | **72% 절약** |

**설계 근거**:
- 600개 합성 샘플: GS-LRM 학습에 최소한의 다양성 확보
- 10,000 iterations = ~67 epochs: 수렴 확인 충분
- 25 diffusion steps: 50 steps 대비 품질 유사, 2배 빠름

### 2.2 Phase 2 속도 최적화

**병목 원인**:
```
1799 샘플 x 6 views x 50 steps x 0.35초/step ≈ 52시간
```

**최적화 옵션**:
```bash
# 옵션 1: steps 감소 (25 steps)
--num_steps 25  # 2배 빠름

# 옵션 2: Augmentation 비활성화
--no-augment_all_views  # 6배 빠름

# 옵션 3: 샘플 수 감소 (테스트용)
--input_data data_mouse/data_mouse_test_100.txt
```

---

## 3. GS-LRM 전이학습 문제 해결

### 3.1 Gradient Explosion 문제

**문제상황**:
```
WARNING: step 766 grad norm too large 58.02 > 20.0, skipping optimizer step
param_update_step: 69/766 (9%만 실제 업데이트)
```

**근본 원인**:
1. Learning rate 2e-5가 synthetic 데이터에 너무 높음
2. `reset_training_state: true`로 warmup 리셋
3. Human face pretrained → Mouse synthetic 도메인 차이

**해결방법**:
```yaml
# Before
lr: 0.00002        # 2e-5
warmup: 100

# After
lr: 0.000005       # 5e-6 (4배 감소)
warmup: 500        # 5배 증가
l2_warmup_steps: 500
```

**결과**: 47% 업데이트로 개선 (9% → 47%)

### 3.2 Training State 리셋 문제

**문제상황**: 학습 시작 즉시 종료
```
Before training: fwdbwd_pass_step=21125
max_fwdbwd_passes: 10000
→ 21125 >= 10000 이므로 즉시 완료 판단
```

**해결방법**:
```yaml
# Checkpoint에서 step counter 리셋
reset_training_state: true
```

### 3.3 체크포인트 분리 원칙

**핵심개념**: 다른 데이터 도메인은 별도 체크포인트 관리

```
checkpoints/gslrm/
├── ckpt_0000000000021125.pt    # Human face pretrained (원본)
├── mouse_test/                  # Synthetic 데이터 학습 (별도)
└── archive/                     # 이전 실험 (정리 대상)
```

---

## 4. 서버 리소스 관리

### 4.1 디스크 정리 계획

**분석 결과**: 170GB 중 93GB 삭제 가능

| 폴더 | 크기 | 상태 | 조치 |
|------|------|------|------|
| `mouse_embeds_6x_aug/` | 39GB | 실패 실험 | 삭제 |
| `original_facelift_embeds/` | 39GB | 이전 버전 | 삭제 |
| `archive/mouse_finetune/` | 11GB | 이전 실험 | 삭제 |
| `facelift_prompt_6x/` | 65GB | **현재 사용** | 유지 |
| `pipeckpts/` | 5.3GB | Pretrained | 유지 |

### 4.2 병렬 실행 전략

**GPU 활용**:
```
GPU 0: Phase 2 합성 데이터 생성 (MVDiffusion inference)
GPU 1: Phase 3 GS-LRM 학습
```

---

## 5. 주요 설정 파일 및 명령어

### 5.1 MVDiffusion Config

```yaml
# configs/mouse_mvdiffusion_facelift_prompt.yaml
reference_view_idx: 0  # 고정 (random 아님!)
prompt_embed_path: "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
checkpointing_steps: 200
resume_from_checkpoint: "latest"
```

### 5.2 GS-LRM Test Config

```yaml
# configs/mouse_gslrm_test.yaml
lr: 0.000005              # Gradient explosion 방지
warmup: 500               # 안정화
reset_training_state: true # Step 리셋
max_fwdbwd_passes: 10000   # 테스트용 축소
checkpoint_dir: "checkpoints/gslrm/mouse_test"
```

### 5.3 핵심 명령어

```bash
# Phase 2: Synthetic 데이터 생성
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/generate_gslrm_training_data.py \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse/facelift_prompt_6x/checkpoint-1200 \
    --num_steps 25

# Phase 3: GS-LRM 학습 (torchrun 필수)
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29501 \
    train_gslrm.py --config configs/mouse_gslrm_test.yaml
```

---

## 핵심 인사이트

1. **Data Augmentation 설계 시 Identity Mapping 주의**: Random reference view는 자기 복사 학습의 지름길이 될 수 있음

2. **전이학습 시 Learning Rate 조정 필수**: 새로운 도메인 데이터로 fine-tuning 시 LR을 4-10배 낮추고, warmup 충분히

3. **파이프라인 검증은 소규모로 먼저**: 전체 데이터로 25시간 학습 전에, 33% 샘플로 7시간 테스트

---

## 참고 자료

- WandB: https://wandb.ai/kafkapple-joon-kaist/mouse_facelift
- 관련 문서: `docs/guides/mouse_facelift_usage.md`
- 실험 옵션: `docs/guides/mouse_experiment_options.md`
