# Mouse MVDiffusion 실험 옵션 가이드

**작성일**: 2025-12-13
**현재 실험**: `mouse_mvdiffusion_6x_aug.yaml` (Step ~1000/20000)

---

## 1. 현재 실험 상태

### 설정 요약
| 항목 | 값 |
|------|-----|
| Config | `configs/mouse_mvdiffusion_6x_aug.yaml` |
| Prompt | Mouse elevated (`top-front`, `from above at an angle`) |
| Augmentation | 6x (random reference view) |
| Checkpoint 주기 | 500 steps (~1.5시간) |
| Validation 주기 | 100 steps (~20분) |
| WandB | `mouse_facelift` / `mvdiff_mouse_6x_aug` |

### 모니터링 명령어
```bash
# 실시간 로그
ssh gpu05 "tail -f /home/joon/FaceLift/logs/train_mvdiff_6x_gpu1.log"

# 서버 리소스
ssh gpu05 "/home/joon/FaceLift/scripts/check_server_resources.sh"

# WandB 대시보드
# https://wandb.ai/[username]/mouse_facelift
```

---

## 2. 평가 기준 (Step 2000 시점)

### 수렴 판단 기준
| 지표 | 성공 | 실패 |
|------|------|------|
| **View 일관성** | 6개 view가 동일 객체로 보임 | View마다 다른 형태 |
| **GT 유사도** | Pred ≈ GT (형태, 색상) | 큰 차이 |
| **Loss 추이** | 안정적 감소/수렴 | 진동 또는 정체 |

### WandB 확인 포인트
1. `validation/images` - 6-view 그리드 이미지
2. `train/loss` - 학습 손실 곡선
3. View별 생성 품질 비교

---

## 3. 상황별 실험 옵션

### Case A: 현재 실험 수렴 성공 ✅

**다음 단계**: Phase 2 (GS-LRM 학습 데이터 생성)로 진행

```bash
# Phase 2: Synthetic data generation
python scripts/generate_gslrm_training_data.py \
    --mvdiff_checkpoint checkpoints/mvdiffusion/mouse/mouse_embeds_6x_aug/checkpoint-XXXX \
    --output_dir data_mouse/synthetic_6view

# Phase 3: GS-LRM fine-tuning
CUDA_VISIBLE_DEVICES=1 accelerate launch train_gslrm.py \
    --config configs/mouse_gslrm.yaml
```

---

### Case B: 수렴 느림 (형태는 나오지만 view 불일치) ⚠️

**원인 추정**: 프롬프트 차이 (Cosine Sim 0.70)
**해결책**: FaceLift 원본 프롬프트로 전환

```bash
# gpu05에서 실행

# 1. 현재 학습 종료
pkill -TERM -f "train_diffusion.py"

# 2. FaceLift 프롬프트로 새 실험
CUDA_VISIBLE_DEVICES=1 nohup accelerate launch train_diffusion.py \
    --config configs/mouse_mvdiffusion_facelift_prompt.yaml \
    > logs/train_mvdiff_facelift_prompt.log 2>&1 &
```

**Config 차이점**:
- `prompt_embed_path`: `fixed_prompt_embeds_6view/clr_embeds.pt` (원본)
- `output_dir`: `checkpoints/mvdiffusion/mouse/facelift_prompt_6x`
- `wandb_exp_name`: `mvdiff_facelift_prompt`

---

### Case C: 수렴 실패 (전혀 다른 이미지 생성) ❌

**원인 추정**: 도메인 불일치 (rendering vs real video)
**해결책**: 현실적 프롬프트 사용

```bash
# gpu05에서 실행

# 1. 현재 학습 종료
pkill -TERM -f "train_diffusion.py"

# 2. 현실적 프롬프트 임베딩 생성
cd /home/joon/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mouse_facelift

python scripts/generate_mouse_prompt_embeds_realistic.py --style realistic

# 3. 새 실험 시작
CUDA_VISIBLE_DEVICES=1 nohup accelerate launch train_diffusion.py \
    --config configs/mouse_mvdiffusion_realistic_prompt.yaml \
    > logs/train_mvdiff_realistic_prompt.log 2>&1 &
```

**Config 차이점**:
- `prompt_embed_path`: `mouse_prompt_embeds_realistic/clr_embeds.pt`
- 프롬프트: `"a photograph of a mouse, {view} view, from above at an angle."`

---

### Case D: 부분 수렴 (일부 view만 성공) 🔄

**원인 추정**: 특정 view에서 데이터 부족 또는 프롬프트 불일치
**해결책**: 단계별 프롬프트 fine-tuning (Curriculum Learning)

```bash
# Stage 1: FaceLift 프롬프트로 기본 수렴 (5000 steps)
CUDA_VISIBLE_DEVICES=1 accelerate launch train_diffusion.py \
    --config configs/mouse_mvdiffusion_facelift_prompt.yaml

# Stage 2: 체크포인트에서 mouse 프롬프트로 전환
# configs/mouse_mvdiffusion_6x_aug.yaml 수정:
#   resume_from_checkpoint: "checkpoints/mvdiffusion/mouse/facelift_prompt_6x/checkpoint-5000"
```

---

## 4. 프롬프트 스타일 비교

| 스타일 | 프롬프트 예시 | 특징 |
|--------|--------------|------|
| **facelift** | `"a rendering image of 3D models, front view, color map."` | 빠른 수렴, pretrained와 일치 |
| **mouse_elevated** | `"a rendering image of a 3D model, top-front view, from above at an angle, color map."` | 카메라 각도 반영 |
| **realistic** | `"a photograph of a mouse, front view, from above at an angle."` | 실제 영상 도메인 |
| **hybrid** | `"a multi-view image of a mouse, front view, elevated camera."` | 균형 |
| **simple** | `"a mouse, front view, top-down angle."` | 도메인 중립 |

### 프롬프트 생성 명령어
```bash
# 스타일 목록 확인
python scripts/generate_mouse_prompt_embeds_realistic.py --list-styles

# 특정 스타일 생성
python scripts/generate_mouse_prompt_embeds_realistic.py --style [STYLE_NAME]

# 커스텀 출력 경로
python scripts/generate_mouse_prompt_embeds_realistic.py --style realistic \
    --output-dir mvdiffusion/data/my_custom_embeds
```

---

## 5. Config 파일 요약

| Config | 프롬프트 | 용도 |
|--------|---------|------|
| `mouse_mvdiffusion_6x_aug.yaml` | mouse elevated | **현재 실험** |
| `mouse_mvdiffusion_facelift_prompt.yaml` | facelift original | 빠른 수렴 테스트 |
| `mouse_mvdiffusion_realistic_prompt.yaml` | realistic | 도메인 매칭 테스트 |

---

## 6. 실험 비교 체크리스트

현재 실험 완료 후 기록:

- [ ] Step 2000 도달 시점 기록: ____
- [ ] Loss 값: ____
- [ ] View 일관성 (1-5): ____
- [ ] GT 유사도 (1-5): ____
- [ ] 결정: Case A / B / C / D

다음 실험 시작 전:
- [ ] 이전 프로세스 완전 종료 확인
- [ ] WandB에서 이전 run 종료 처리
- [ ] 새 config 확인
- [ ] 로그 파일명 변경

---

## 7. 참고 자료

- 연구 보고서: `docs/reports/251213_research_prompt_embedding_adaptation.md`
- 사용 가이드: `docs/guides/mouse_facelift_usage.md`
- 데이터 전처리: `docs/guides/mouse_data_preprocessing.md`
