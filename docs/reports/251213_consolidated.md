# 251213: Prompt Embeddings 해결 및 2단계 학습 전략

**날짜:** 2025-12-13
**주제:** MVDiffusion-GS-LRM 파이프라인 정렬 문제 해결
**상태:** ✅ 핵심 문제 해결됨

---

## 핵심 요약

| 항목 | 내용 |
|------|------|
| **문제** | MVDiffusion 출력 (수평 뷰) ↔ GS-LRM 기대 (경사 뷰) 불일치 |
| **원인** | FaceLift prompt_embeds가 수평 6뷰 방향 인코딩 |
| **해결** | Mouse용 prompt_embeds 생성 (경사 20° 뷰) |
| **결과** | 추가 학습 없이 문제 해결 |

---

## 1. Prompt Embeddings 이해

### Prompt vs Prompt Embedding
```
Prompt (텍스트)          Prompt Embedding (벡터)
      │                         │
      ▼                         ▼
"front view"    ──CLIP──→   [77, 1024] 텐서
      │         인코딩           │
  사람이 읽음                모델이 읽음
```

### FaceLift vs Mouse Embeds 비교
| 뷰 | FaceLift (수평) | Mouse (경사) |
|----|-----------------|--------------|
| 0 | "front view" | "**top-front** view, from above at an angle" |
| 1 | "front_right view" | "**top-front-right** view, from above at an angle" |

### 해결: Mouse Prompt Embeds 생성
```python
# scripts/generate_mouse_prompt_embeds.py
views = ["top-front", "top-front-right", "top-right",
         "top-back", "top-left", "top-front-left"]

color_prompts = [
    f"a rendering image of a 3D model, {view} view, from above at an angle, color map."
    for view in views
]
```

**저장 위치**: `mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt`

---

## 2. 적용 결과

### Before (문제)
```
MVDiffusion (FaceLift embeds) → 수평 6뷰 생성
                ↓
GS-LRM mouse_finetune (경사 카메라 기대)
                ↓
        ❌ 카메라 불일치 → 3D 실패
```

### After (해결)
```
MVDiffusion (Mouse embeds) → 경사 6뷰 생성
                ↓
GS-LRM mouse_finetune (경사 카메라 기대)
                ↓
        ✅ 카메라 일치 → 3D 성공
```

---

## 3. 2단계 순차 학습 전략

### 전체 파이프라인
```
Phase 1: MVDiffusion Fine-tuning
─────────────────────────────────
입력: 실제 마우스 2000 샘플 × 6뷰 = 12,000 (6배 증강)
설정: reference_view_idx: "random", mouse prompt_embeds
학습: 임의 뷰 → 전체 6뷰 생성
         ↓
Phase 2: 합성 데이터 생성
─────────────────────────────────
입력: 실제 마우스 1뷰씩 (2000 × 6 = 12,000)
출력: MVDiffusion 생성 6뷰 (12,000 샘플)
카메라: mouse prompt_embeds 기준 (경사 뷰)
         ↓
Phase 3: GS-LRM Fine-tuning
─────────────────────────────────
데이터: Phase 2 합성 6뷰 12,000 샘플
시작: human pretrained
결과: MVDiffusion 출력에 최적화된 GS-LRM
```

### 예상 타임라인
| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 1 | MVDiffusion 학습 (20k steps) | ~12-24h |
| 2 | 합성 데이터 생성 (12k samples) | ~2-4h |
| 3 | GS-LRM 학습 (30k steps) | ~12-24h |
| **총** | | **~30-50h** |

---

## 4. Prompt Embedding 적응 연구

### 프롬프트 비교 (Cosine Similarity)
| 버전 | 프롬프트 예시 | Similarity |
|------|---------------|------------|
| FaceLift | "front view, color map" | 1.00 (baseline) |
| Mouse | "top-front view, from above at an angle" | **0.70** |

### 단계별 적응 전략 (Curriculum Learning)
```
Stage 1 (0-2000 steps): FaceLift 원본 프롬프트 → 빠른 초기 수렴
Stage 2 (2000-5000): 경사 시점 정보 추가
Stage 3 (5000-20000): 실제 데이터 도메인으로 전환
```

### 현재 선택: 단일 단계
- **장점**: 구현 단순, 최종 목표 프롬프트로 직접 학습
- **단점**: 초기 수렴 느림, 더 많은 step 필요
- **예상 수렴**: ~3000-5000 steps

---

## 5. 최적 설정

### 현재 사용 가능 파이프라인 (v1)
```bash
python test_full_pipeline.py \
    --input_image data_mouse/sample_000000/images/cam_000.png \
    --mvdiff_unet checkpoints/mvdiffusion/mouse/.../unet \
    --prompt_embeds mvdiffusion/data/mouse_prompt_embeds_6view/clr_embeds.pt \
    --gslrm_checkpoint checkpoints/gslrm/mouse_finetune \
    --output_dir outputs/pipeline_v1
```

### 체크포인트 구조
```
checkpoints/
├── mvdiffusion/mouse/
│   └── original_facelift_embeds/checkpoint-6000/unet
└── gslrm/
    ├── ckpt_0000000000021125.pt     ← pretrained (human)
    └── mouse_finetune/
        └── ckpt_0000000000020000.pt ← v1 (사용 가능)
```

---

## 6. 핵심 교훈

1. **prompt_embeds의 중요성**: MVDiffusion 출력 뷰 방향 결정하는 핵심 요소
2. **정렬 필수**: 학습 embeds ↔ 추론 embeds ↔ GS-LRM 카메라 모두 일치해야 함
3. **디버깅 접근**: 전체 파이프라인 문제 시 → 각 단계 분리 테스트

---

## 관련 파일
- `scripts/generate_mouse_prompt_embeds.py` - Mouse embeds 생성
- `mvdiffusion/data/mouse_prompt_embeds_6view/` - Mouse prompt embeddings
- `configs/mouse_mvdiffusion_6x_aug.yaml` - MVDiffusion Phase 1 config

---

*통합 문서: 251213_pipeline_alignment_resolved.md + 251213_research_mouse_facelift_daily.md + 251213_research_prompt_embedding_adaptation.md + 251213_research_two_phase_training_strategy.md*
