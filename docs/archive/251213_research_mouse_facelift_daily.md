---
date: 2024-12-13
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, mvdiffusion, gslrm, prompt-embeds]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Mouse-FaceLift 파이프라인 정렬 문제 해결

## 핵심 요약

| 항목 | 내용 |
|------|------|
| **문제** | MVDiffusion 출력과 GS-LRM 카메라 불일치 → 3D 결과 이상 |
| **원인** | FaceLift prompt_embeds (수평 6뷰) ↔ Mouse 카메라 (경사 20°) |
| **해결** | Mouse prompt_embeds 생성 및 적용 |
| **결과** | 추가 학습 없이 문제 해결됨 |

---

## 1. 문제 상황

### 1.1 초기 증상
- MVDiffusion → GS-LRM 전체 파이프라인 실행 시 3D 결과가 "기묘함"
- 각 단계 개별 테스트 시에는 정상으로 보임

### 1.2 분석 과정
```
MVDiffusion (FaceLift embeds) → 수평 6뷰 생성
                ↓
GS-LRM mouse_finetune (경사 카메라 기대)
                ↓
        ❌ 카메라 불일치 → 3D 복원 실패
```

### 1.3 핵심 발견
- **prompt_embeds의 역할**: MVDiffusion 출력 뷰 방향을 결정
- FaceLift embeds: `front, front_right, right, back, left, front_left` (수평)
- Mouse 카메라: elevation ~20° 경사 뷰
- **불일치**: MVDiffusion은 수평 뷰 생성 → GS-LRM은 경사 뷰 기대

---

## 2. 해결책

### 2.1 Mouse prompt_embeds 생성

**스크립트**: `scripts/generate_mouse_prompt_embeds.py`

```python
views = ["top-front", "top-front-right", "top-right",
         "top-back", "top-left", "top-front-left"]

color_prompts = [
    f"a rendering image of a 3D model, {view} view, from above at an angle, color map."
    for view in views
]
```

**생성 위치**: `mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt`

### 2.2 적용 결과
```
MVDiffusion (Mouse embeds) → 경사 6뷰 생성
                ↓
GS-LRM mouse_finetune (경사 카메라 기대)
                ↓
        ✅ 카메라 일치 → 정상 3D 복원
```

---

## 3. 실험 결과

### 3.1 Stage 1: MVDiffusion 비교 (prompt_embeds만 변경)

| 테스트 | prompt_embeds | 결과 |
|--------|---------------|------|
| Test A | FaceLift (수평) | 품질 좋음 |
| Test B | Mouse (경사) | 품질 좋음 |

**발견**: 둘 다 품질 좋음 → MVDiffusion UNet이 잘 학습됨

### 3.2 Stage 2: Full Pipeline 비교

| 조합 | prompt_embeds | GS-LRM | 결과 |
|------|---------------|--------|------|
| 이전 | FaceLift | mouse_finetune | ❌ 이상 |
| 신규 | **Mouse** | mouse_finetune | ✅ 개선 |
| Baseline | FaceLift | pretrained (인간) | ✅ 작동 |

**핵심**: Mouse embeds 적용만으로 문제 해결됨

---

## 4. 추가 최적화 (선택)

### 4.1 GS-LRM v2 Fine-tuning

**목적**: 품질 추가 개선 (필수 아님)

**변경 사항**:
| 파라미터 | v1 | v2 | 이유 |
|----------|-----|-----|------|
| resume_from | pretrained | v1 (20k) | 빠른 학습 |
| Learning Rate | 2e-05 | 1e-05 | 안정적 fine-tuning |
| LPIPS weight | 0.5 | 0.8 | 시각적 품질 강화 |
| Perceptual weight | 0.5 | 0.8 | 시각적 품질 강화 |
| 추가 steps | - | +30,000 | 충분한 학습 |

**config**: `configs/mouse_gslrm_v2.yaml`

### 4.2 학습 명령어
```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse_gslrm_v2.yaml
```

### 4.3 WandB 로깅
- Project: `mouse_facelift`
- Group: `gslrm`
- Exp: `mouse_gslrm_v2`
- 로깅 항목: loss, PSNR, LPIPS, SSIM, visualization (train/val)

---

## 5. 체크포인트 구조

```
checkpoints/
├── mvdiffusion/mouse/
│   └── original_facelift_embeds/
│       └── checkpoint-6000/unet     ← MVDiffusion (사용 중)
│
└── gslrm/
    ├── ckpt_0000000000021125.pt     ← pretrained (인간)
    ├── mouse_finetune/
    │   └── ckpt_0000000000020000.pt ← v1 (현재 사용 가능)
    └── mouse_v2/
        └── ckpt_*.pt                 ← v2 (학습 중)
```

---

## 6. 최종 파이프라인 명령어

### 현재 사용 가능 (v1)
```bash
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_finetune \
    --output_dir outputs/pipeline_v1
```

### v2 학습 완료 후
```bash
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/original_facelift_embeds/checkpoint-6000/unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_v2 \
    --output_dir outputs/pipeline_v2
```

---

## 7. Prompt Embedding 상세 설명

### 7.1 Prompt vs Prompt Embedding

```
Prompt (텍스트)          Prompt Embedding (벡터)
      │                         │
      ▼                         ▼
"front view"    ──CLIP──→   [77, 1024] 텐서
      │         인코딩           │
  사람이 읽음                모델이 읽음
```

| 구분 | Prompt | Prompt Embedding |
|------|--------|------------------|
| 형태 | 텍스트 문자열 | 숫자 벡터 [6, 77, 1024] |
| 용도 | 사람이 작성 | 모델 입력 (cross-attention) |
| 저장 | 텍스트 파일 | .pt 파일 (빠른 로딩) |

### 7.2 왜 .pt로 저장?

```python
# 매번 CLIP 인코딩 (느림, 수 초)
embeds = clip_encoder("front view")

# 미리 저장 후 로딩 (빠름, 즉시)
embeds = torch.load("clr_embeds.pt")
```

### 7.3 FaceLift vs Mouse Embeds 비교

| 뷰 | FaceLift (수평) | Mouse (경사) |
|----|-----------------|--------------|
| 0 | "front view" | "**top-front** view, **from above at an angle**" |
| 1 | "front_right view" | "**top-front-right** view, from above at an angle" |
| 2 | "right view" | "**top-right** view, from above at an angle" |

**핵심 차이**:
- `top-` 접두사: 위에서 보는 뷰임을 명시
- `from above at an angle`: 경사 각도 설명

### 7.4 정말 텍스트만으로 해결?

**예!** 정확한 카메라 수치 (elevation=20°) 없이, 자연어 설명만으로 CLIP이 방향 개념 인코딩

```python
# 이것이 해결책의 전부
views = ["top-front", "top-front-right", "top-right", ...]
prompts = [f"...{view} view, from above at an angle..." for view in views]
```

### 7.5 추가 개선 가능성

| 방법 | 설명 | 난이도 |
|------|------|--------|
| 프롬프트에 수치 추가 | "elevation 20 degrees" | 쉬움 |
| 카메라 임베딩 별도 | camera_embeds 추가 | 모델 수정 필요 |

---

## 8. 핵심 교훈

### 7.1 prompt_embeds의 중요성
- MVDiffusion 출력 뷰 방향을 결정하는 **핵심 요소**
- 학습 시 사용한 embeds ↔ 추론 시 embeds ↔ GS-LRM 카메라 **모두 일치해야 함**

### 7.2 디버깅 접근법
- 전체 파이프라인 문제 시 → 각 단계 분리 테스트
- 입력/출력 데이터 형식 및 파라미터 정렬 확인

### 7.3 실험 관리
- 체크포인트 폴더 구조화 (실험별 분리)
- config 파일로 실험 설정 관리
- WandB로 학습 모니터링

---

## 8. 다음 단계

1. [ ] GS-LRM v2 학습 완료 대기 (~8시간)
2. [ ] v1 vs v2 품질 비교
3. [ ] 다양한 샘플로 테스트
4. [ ] 최종 모델 선정

---

## 관련 파일

| 파일 | 용도 |
|------|------|
| `scripts/generate_mouse_prompt_embeds.py` | Mouse embeds 생성 |
| `configs/mouse_gslrm_v2.yaml` | GS-LRM v2 학습 config |
| `docs/guides/experiment_commands.md` | 실험 명령어 가이드 |
| `docs/guides/gslrm_training_strategy.md` | 학습 전략 가이드 |
| `docs/guides/prompt_embeds_design.md` | prompt_embeds 설계 가이드 |
