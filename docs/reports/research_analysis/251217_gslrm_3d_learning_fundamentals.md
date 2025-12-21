---
date: 2025-12-17
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, gs-lrm, 3d-learning, objaverse]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# GS-LRM 3D 학습 원리 분석

> Objaverse에서 진정한 3D 학습이 가능한 이유와 Mouse 데이터의 한계

## 1. 핵심 질문

> **"왜 Objaverse는 진정한 3D 학습이 가능했고, Mouse 데이터는 안 되는가?"**

---

## 2. Objaverse vs Mouse 데이터 비교

### 데이터 생성 방식

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Objaverse 데이터 생성 과정                        │
├─────────────────────────────────────────────────────────────────────┤
│   [3D 모델]  ──렌더링──>  [Multi-view 이미지]                        │
│      │                         │                                    │
│      ▼                         ▼                                    │
│   완벽한 3D              모든 뷰가 동일한 3D에서                     │
│   기하학 정보             수학적으로 정확하게 생성                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Mouse 데이터 촬영 과정                            │
├─────────────────────────────────────────────────────────────────────┤
│   [실제 마우스]  ──촬영──>  [6개 카메라 이미지]                       │
│      │                         │                                    │
│      ▼                         ▼                                    │
│   3D Ground Truth          시간에 따른 마우스 움직임                  │
│   존재하지 않음             조명 변화, 카메라 오차                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 정량적 비교

| 항목 | Objaverse | Mouse |
|------|-----------|-------|
| 3D Ground Truth | ✅ 있음 | ❌ 없음 |
| 뷰 간 기하학적 일관성 | ✅ 완벽 | ⚠️ 노이즈 있음 |
| 카메라 파라미터 | ✅ 정확 | ⚠️ 캘리브레이션 오차 |
| 객체 위치 | ✅ 원점 고정 | ❌ 자유 이동 |
| 조명 조건 | ✅ 제어됨 | ⚠️ 변동 있음 |

---

## 3. 3D 학습의 핵심 조건

### Multi-View Consistency
모든 뷰가 **동일한 3D 기하학**에서 생성되어야 함

```python
# Objaverse: 완벽한 일관성
for view in views:
    image = render(3D_model, camera[view])  # 동일 3D에서 렌더링

# Mouse: 불완전한 일관성
for cam in cameras:
    image = capture(moving_mouse, cam, time)  # 움직이는 객체 촬영
```

### 2D-only Supervision의 한계

**문제**: Ground Truth 3D가 없으면 모델이 "올바른" 3D를 학습했는지 검증 불가

```
2D Loss만으로 학습 시:
- 여러 3D 해석이 동일한 2D 렌더링을 생성할 수 있음
- 모델이 "평균적인" 해를 찾아 mode collapse 발생
- 특히 뷰가 적을수록 (6뷰) ambiguity 증가
```

---

## 4. Mouse 데이터 개선 전략

### 전략 1: 합성 데이터 활용
- MAMMAL body model로 3D 마우스 생성
- 다양한 포즈/뷰에서 렌더링
- Objaverse와 유사한 조건 확보

### 전략 2: 데이터 정규화
- 객체 중앙 정렬 (CoM 기반)
- 크기 균일화 (pixel-based scaling)
- 카메라 파라미터 정규화

### 전략 3: 추가 감독 신호
- Temporal consistency (프레임 간 일관성)
- Silhouette supervision
- Depth hints (가능하다면)

---

## 5. 합성 데이터 생성 파이프라인 (FaceLift 원본)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FaceLift GS-LRM 학습 파이프라인                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Objaverse Pretrain (일반 3D prior)                        │
│  ───────────────────────────────────────────                        │
│  - ~80K 3D 모델에서 렌더링                                          │
│  - 32개 뷰 렌더링                                                    │
│  - 일반적인 3D geometry prior 학습                                  │
│                                                                      │
│            ↓                                                        │
│                                                                      │
│  Stage 2: Synthetic Head Finetune (도메인 적응)                     │
│  ──────────────────────────────────────────────                     │
│  - 합성 인간 헤드 3D 모델에서 렌더링                                 │
│  - 32개 뷰 렌더링 (랜덤 HDR 조명)                                    │
│  - 매 step: 8개 뷰 랜덤 샘플 (4 input + 4 supervision)              │
│  - Human face 도메인 적응                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 결론

### 왜 Objaverse가 성공하는가?
1. **완벽한 3D Ground Truth** 존재
2. **수학적으로 정확한** multi-view 일관성
3. **대규모 다양성** (~80K 모델)

### 왜 Mouse가 어려운가?
1. **3D GT 부재**: 2D supervision만으로는 한계
2. **동적 객체**: 촬영 중 움직임
3. **제한된 뷰**: 6뷰만으로 3D ambiguity 증가

### 해결 방향
1. **합성 데이터**: MAMMAL body model + 렌더링
2. **철저한 전처리**: 위치/크기 정규화
3. **적절한 Loss**: Perceptual loss 비활성화

---

*출처*: `251217_research_gslrm_3d_learning_analysis.md`
