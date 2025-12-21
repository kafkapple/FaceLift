---
date: 2025-12-13
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, mvdiffusion, training-strategy]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# 2025-12-13 연구 일지: Mouse-FaceLift 학습 전략 수립

## 1. 연구 목표

FaceLift 파이프라인을 마우스 도메인에 적응하기 위한 초기 전략 수립

---

## 2. 진행 내용

### 2.1 문제 진단

**이전 실험 결과 분석**:
| 테스트 | GS-LRM 모델 | 카메라 | 결과 |
|--------|-------------|--------|------|
| pretrained (human) | pretrained | FaceLift | ✅ 성공 |
| mouse_finetune | fine-tuned | FaceLift | ❌ 실패 |
| mouse_finetune | fine-tuned | mouse | ❌ 실패 |

**핵심 발견**: 카메라 설정 불일치 체인
```
MVDiffusion 학습: FaceLift prompt_embeds (수평 뷰)
        ↓
MVDiffusion 출력: 수평 6뷰 (elevation ≈ 0°)
        ↓
GS-LRM mouse_finetune: Mouse 카메라 (elevation ≈ 20°)
        ↓
결과: MVDiffusion 출력 ≠ GS-LRM 기대 입력 → 3D 복원 실패
```

### 2.2 2단계 학습 전략 설계

**Stage 1: MVDiffusion Fine-tuning**
- FaceLift pretrained 모델에서 시작
- Mouse 6-view 데이터로 fine-tune
- 프롬프트 임베딩은 FaceLift 원본 유지

**Stage 2: GS-LRM Fine-tuning**
- MVDiffusion 출력 형식에 맞춘 카메라로 학습
- FaceLift 카메라 파라미터 사용
- 합성 데이터 활용 고려

### 2.3 Prompt Embedding 분석

| 버전 | 프롬프트 | Cosine Sim |
|------|---------|------------|
| FaceLift (Original) | "front view, color map" | 1.00 |
| Mouse (수정) | "top-front view, from above" | 0.70 |

**결론**: 0.70 유사도는 너무 낮음 → 수렴 속도 저하 예상

---

## 3. 주요 교훈

1. **도메인 정렬 중요성**: MVDiffusion → GS-LRM 간 카메라 설정 일치 필수
2. **Prompt Embedding**: 큰 변경보다 원본 유지가 안전
3. **순차적 학습**: Stage 1 완료 후 Stage 2 진행 권장

---

## 4. 다음 단계

- [ ] MVDiffusion mouse fine-tuning 시작
- [ ] FaceLift 카메라 설정으로 데이터 변환
- [ ] reference_view_idx: 0 고정 설정

---

*통합 출처*:
- `251213_research_two_phase_training_strategy.md`
- `251213_research_prompt_embedding_adaptation.md`
