---
date: 2025-12-14
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, mvdiffusion, gslrm, debugging]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# 2025-12-14 연구 일지: Pipeline Debugging

## 1. 연구 목표

MVDiffusion mode collapse 문제 해결 및 GS-LRM 전이학습 안정화

---

## 2. 진행 내용

### 2.1 MVDiffusion Mode Collapse 해결

**문제상황**:
- 1500 step에서 조기 수렴
- 모든 view가 유사한 출력 생성
- 91% step이 실제로 학습되지 않음

**근본 원인**:
```
6x Augmentation (random reference view) 문제점:
1. Identity Mapping: 1/6 확률로 input = target (자기 자신 복사 학습)
2. High Prompt Similarity: View 간 프롬프트 유사도 0.85-0.99
   → 모델이 view 구분 없이 입력 복사하는 것이 최적해가 됨
```

**해결책**:
- `reference_view_idx: "random"` → `reference_view_idx: 0` (고정)
- FaceLift 원본 설정으로 복귀
- **결과**: 1300 step만에 정확한 예측 달성

### 2.2 GS-LRM Gradient Explosion 해결

**문제상황**:
```
WARNING: step 766 grad norm too large 58.02 > 20.0, skipping optimizer step
param_update_step: 69/766 (9%만 실제 업데이트)
```

**근본 원인**:
1. Learning rate 2e-5가 synthetic 데이터에 너무 높음
2. `reset_training_state: true`로 warmup 리셋
3. Human face pretrained → Mouse synthetic 도메인 차이

**해결책**:
```yaml
# Before
lr: 0.00002        # 2e-5
warmup: 100

# After
lr: 0.000005       # 5e-6 (4배 감소)
warmup: 500        # 5배 증가
```

**결과**: 9% → 47% 업데이트 비율 개선

### 2.3 파이프라인 시간 최적화

**테스트 모드 설계**:
| Phase | Full Train | Test Mode | 시간 절약 |
|-------|-----------|-----------|----------|
| Phase 2 (Synthetic) | 1800개, 50 steps | 600개, 25 steps | 8h → 1.5h |
| Phase 3 (GS-LRM) | 30,000 iters | 10,000 iters | 15h → 4.5h |
| **Total** | ~25h | ~7h | **72% 절약** |

---

## 3. 주요 교훈

1. **Random Reference View 위험**: Identity mapping 학습의 지름길이 될 수 있음
2. **도메인 전이 시 LR 조정 필수**: 4-10배 낮추고, warmup 충분히
3. **파이프라인 검증은 소규모로 먼저**: 전체 데이터 전에 33% 샘플로 테스트

---

## 4. 핵심 설정 정리

### MVDiffusion Config
```yaml
reference_view_idx: 0  # 고정 (random 아님!)
prompt_embed_path: "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
```

### GS-LRM Config
```yaml
lr: 0.000005              # Gradient explosion 방지
warmup: 500               # 안정화
reset_training_state: true # Step 리셋
```

---

## 5. 다음 단계

- [x] reference_view_idx: 0 고정 설정
- [x] GS-LRM learning rate 조정
- [ ] 전체 파이프라인 테스트 (7시간 모드)

---

*통합 출처*:
- `251214_research_mouse_facelift_pipeline_debugging.md`

