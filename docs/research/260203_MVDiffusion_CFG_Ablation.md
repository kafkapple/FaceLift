# MVDiffusion CFG Dropout & Attention Ablation

**Date**: 2026-02-03
**Status**: Planning
**Tags**: #mvdiffusion #ablation #cfg #attention

---

## 1. Background

### 문제 제기
M5t2_consistent 모델의 결과가 기존 M5t baseline보다 안 좋아 보임.
변경된 설정이 오히려 품질 저하를 유발했을 가능성 조사.

### 변경 이력

| 버전 | 날짜 | 주요 변경 |
|------|------|-----------|
| M5t | 2026-01-28 | Baseline (original prompt, 1to1 split) |
| M5t_mp | 2026-02-01 | Mouse prompt 적용 |
| M5t2 | 2026-02-01 | t2 split (80:10:10) |
| M5t2_consistent | 2026-02-02 | CFG dropout 제거, Full attention |

---

## 2. 설정 비교

### 핵심 파라미터

| 파라미터 | M5t (baseline) | M5t2 | M5t2_consistent | 역할 |
|----------|----------------|------|-----------------|------|
| `prompt_embed_path` | original (768d) | mouse (1024d) | mouse (1024d) | View 설명 임베딩 |
| `train_dataset` | 1to1 train | t2 train | t2 train | 학습 데이터 |
| `condition_drop_rate` | **0.05** | **0.05** | **0.0** | CFG dropout 확률 |
| `sparse_mv_attention` | **true** | **true** | **false** | Multi-view attention 타입 |
| `max_train_steps` | 10000 | 10000 | 10000 | 최대 학습 스텝 |

### 체크포인트 현황

| 모델 | 사용 가능 ckpt | 최신 |
|------|----------------|------|
| M5t | 6000, 8000 | 8000 |
| M5t2 | 3000, 4000, 5000 | 5000 |
| M5t2_consistent | 4000, 5000, 6000 | 6000 |

---

## 3. 가설

### H1: CFG Dropout 제거가 품질 저하의 원인

**메커니즘**:
- `condition_drop_rate=0.05`: 학습 중 5% 확률로 conditioning 드롭
- 이를 통해 모델이 unconditional generation도 학습
- Inference 시 `guidance_scale > 1.0` 적용하면 조건부/비조건부 차이로 guidance 수행

**문제**:
- `condition_drop_rate=0.0`: unconditional 학습 없음
- Inference에서 guidance_scale 적용해도 효과 미미
- 결과적으로 conditioning이 약해지거나 일관성 저하

**검증 방법**:
- M5t2 (CFG=0.05) vs M5t2_consistent (CFG=0.0) 비교
- 동일 데이터, 동일 스텝에서 guidance_scale 변화 실험

### H2: Full Attention이 과적합 유발

**메커니즘**:
- `sparse_mv_attention=true`: 인접 뷰 간만 attention (regularization 효과)
- `sparse_mv_attention=false`: 모든 뷰 쌍 간 attention (더 많은 파라미터)

**문제**:
- Full attention은 학습 데이터에 과적합될 가능성
- 특히 데이터가 적을 때 (t2 train: ~2400 samples)

**검증 방법**:
- 학습 loss vs validation loss 비교 (WandB)
- Train set vs Test set 생성 품질 비교

### H3: Prompt 임베딩이 원인 (가능성 낮음)

- M5t_mp (mouse prompt, 1to1)가 M5t와 비슷하다면 prompt는 원인 아님
- 별도 검증 필요

---

## 4. 실험 설계

### 4.1 비교 실험 (E2E Inference)

**조건 통제**:
- 동일 테스트 데이터: `data_mouse_1to1_test.txt`
- 동일 GS-LRM: `M5t_E0_1_facelift/best_psnr.pt`
- 동일 입력 뷰: view 0 (top-front)
- 동일 샘플 수: 10 frames

| 실험 | MVDiffusion ckpt | 역할 |
|------|------------------|------|
| **Ctrl** | M5t/checkpoint-8000 | 대조군 (baseline) |
| **Exp-A** | M5t2/checkpoint-5000 | Split+Prompt 변경 |
| **Exp-B** | M5t2_consistent/checkpoint-6000 | +CFG제거+FullAttn |

### 4.2 평가 지표

| 지표 | 측정 방법 | 기대 |
|------|-----------|------|
| **Visual Quality** | 생성 6-view 이미지 검토 | Ctrl > Exp-B? |
| **View Consistency** | 6뷰 간 형태/색상 일관성 | Ctrl ≥ Exp-A > Exp-B? |
| **3D Reconstruction** | GS-LRM turntable 렌더링 | Ctrl > Exp-B? |
| **Artifact Check** | Ghosting, 불일치 확인 | Exp-B에서 더 많을 것으로 예상 |

### 4.3 추가 실험 (결과에 따라)

**만약 H1 확인 시**:
```yaml
# Exp-C: CFG 복원 실험
condition_drop_rate: 0.05  # 원복
sparse_mv_attention: false  # full 유지
```

**만약 H2 확인 시**:
- WandB에서 train/val loss 곡선 분석
- 과적합 징후 확인

---

## 5. 실험 명령어

```bash
cd /home/joon/dev/FaceLift

# === Ctrl: M5t baseline ===
export CUDA_VISIBLE_DEVICES=4 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t/checkpoint-8000 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --input_view_idx 0 --num_frames 10 \
    --output_dir outputs/compare_mvdiff/ctrl_M5t_8k \
    > logs/compare_ctrl_M5t.log 2>&1 &

# === Exp-A: M5t2 (CFG=0.05, sparse=true) ===
export CUDA_VISIBLE_DEVICES=6 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2/checkpoint-5000 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --input_view_idx 0 --num_frames 10 \
    --output_dir outputs/compare_mvdiff/exp_A_M5t2_5k \
    > logs/compare_expA_M5t2.log 2>&1 &

# === Exp-B: M5t2_consistent (CFG=0, sparse=false) ===
export CUDA_VISIBLE_DEVICES=7 && nohup python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --model M5t \
    --mvdiffusion_checkpoint /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_consistent/checkpoint-6000 \
    --data_dir ~/data/preprocessed/FaceLift_mouse/M5 \
    --split ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_1to1_test.txt \
    --input_view_idx 0 --num_frames 10 \
    --output_dir outputs/compare_mvdiff/exp_B_M5t2_consistent_6k \
    > logs/compare_expB_M5t2_consistent.log 2>&1 &
```

---

## 6. 예상 결과

### 시나리오 A: CFG Dropout이 주 원인
- Exp-A ≈ Ctrl (둘 다 CFG=0.05)
- Exp-B < Ctrl (CFG=0.0의 영향)
- **Action**: M5t2_consistent config에서 `condition_drop_rate: 0.05` 복원 후 재학습

### 시나리오 B: Full Attention이 주 원인
- Exp-A ≈ Ctrl
- Exp-B에서 과적합 징후 (train set에서는 좋지만 test set에서 나쁨)
- **Action**: `sparse_mv_attention: true` 복원

### 시나리오 C: 복합 원인
- 두 변경 모두 영향
- **Action**: 둘 다 원복하거나, 각각 단독 실험 추가

---

## 7. 관련 문헌

### Classifier-Free Guidance (Ho & Salimans, 2022)
- Training: `p_drop` 확률로 conditioning 드롭
- Inference: `guidance_scale * (cond - uncond) + uncond`
- 학습 시 unconditional 경험 없으면 guidance 효과 없음

### MVDiffusion 원본 설정
- `condition_drop_rate: 0.05` (표준)
- `sparse_mv_attention: true` (효율성 + regularization)

---

## 8. References

- Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance.
- Shi, Y., et al. (2023). MVDiffusion: Enabling holistic multi-view image generation.
- Config files: `configs/mvdiffusion/mouse_mvdiffusion_*.yaml`

---

*Created: 2026-02-03 | FaceLift Mouse Project*

---

## 9. 후속 학습 실험 (결과에 따라)

### 9.1 Exp-C: CFG 복원 (H1 확인 시)

**설정**: `condition_drop_rate: 0.05` 복원, `sparse_mv_attention: false` 유지

```bash
cd /home/joon/dev/FaceLift

# 새 config 생성 (M5t2_cfgr.yaml)
export CUDA_VISIBLE_DEVICES=6 && nohup accelerate launch \
    --config_file configs/accelerate/1gpu.yaml \
    train_diffusion.py \
    --config configs/mvdiffusion/mouse_mvdiffusion_M5t2_cfgr.yaml \
    > logs/mvdiff_M5t2_cfgr.log 2>&1 &
```

**필요 config 변경**:
```yaml
# mouse_mvdiffusion_M5t2_cfgr.yaml
condition_drop_rate: 0.05  # 복원 (consistent는 0.0)
sparse_mv_attention: false  # full 유지
output_dir: /node_data/joon/checkpoints/FaceLift/mvdiffusion/mouse_M5t2_cfgr
```

### 9.2 Exp-D: Sparse Attention 복원 (H2 확인 시)

**설정**: `sparse_mv_attention: true` 복원, `condition_drop_rate: 0.0` 유지

---

## 10. 근거 및 이론적 배경

### 10.1 Classifier-Free Guidance (CFG) 원리

**논문**: Ho & Salimans, 2022 - "Classifier-Free Diffusion Guidance"

**핵심 메커니즘**:
```
# Training phase
if random() < p_drop:
    condition = ∅  # null/unconditional
else:
    condition = actual_condition

# Inference phase
ε_guided = ε_uncond + guidance_scale × (ε_cond - ε_uncond)
```

**왜 p_drop > 0 이 필수인가**:
1. `ε_uncond` 추정 필요 → unconditional 학습 경험 필수
2. `p_drop=0`이면 `ε_uncond`를 학습하지 않음
3. Inference에서 `guidance_scale > 1.0` 적용해도 `ε_uncond`가 부정확
4. 결과: guidance 효과 감소, conditioning 약화

**표준 값**:
- Stable Diffusion: `p_drop=0.1`
- MVDiffusion 원본: `p_drop=0.05`
- 일반적 범위: `0.05 ~ 0.15`

### 10.2 Sparse vs Full Multi-view Attention

**MVDiffusion 원본 설계 (Shi et al., 2023)**:
- Sparse Attention: 인접 뷰 쌍만 attention, O(N) complexity, Regularization 효과
- Full Attention: 모든 뷰 쌍 간 attention, O(N²) complexity, 과적합 위험

---

## 11. 방법론 상세

### 11.1 변수 통제 전략

| 변수 | 값 | 이유 |
|------|-----|------|
| Test data | `data_mouse_1to1_test.txt` | 공정 비교 |
| GS-LRM | `M5t_E0_1_facelift/best_psnr.pt` | MVDiffusion 영향만 측정 |
| Input view | 0 (top-front) | 정보량 최대 |
| Sample count | 10 frames | 빠른 검증 |

### 11.2 결과 해석 가이드

| 패턴 | 해석 | Action |
|------|------|--------|
| Exp-A ≈ Ctrl, Exp-B < Ctrl | CFG가 주 원인 | CFG 복원 학습 |
| Exp-A ≈ Ctrl, Exp-B train > test | Full Attn 과적합 | Sparse 복원 |
| Exp-A < Ctrl | Prompt/Split 영향 | 원인 분리 필요 |

---

*Updated: 2026-02-03*
