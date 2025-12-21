---
date: 2025-12-20
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, preprocessing, pixel-based, generalization]
project: mouse-facelift
status: in_progress
generator: ai-assisted
generator_tool: claude-code
---

# 2025-12-20 연구 일지: Pixel-Based 전처리 및 일반화 전략

## 1. 연구 목표

1. Bounding Box 기반 전처리의 근본 문제 해결
2. 다양한 카메라 환경에 대한 일반화 전략 수립
3. Gaussian Mouse Avatar 구현 계획 수립

---

## 2. 진행 내용

### 2.1 전처리 품질 문제 진단

**기존 방식의 문제**:
| 단계 | 기존 방식 | 문제점 |
|------|----------|--------|
| Centering | Bbox 중심 | 꼬리 방향에 따라 중심점 왜곡 |
| Scaling | Bbox 비율 | 꼬리가 bbox에 불균형 영향 |

**꼬리의 영향 예시**:
- 꼬리 방향에 따라 bbox 중심이 크게 달라짐
- Bbox 비율 60%로 균일해 보여도, 실제 픽셀 영역은 5.81%~8.48% (1.46배 차이)

### 2.2 Pixel-Based 전처리 도입

**Center of Mass (CoM) 기반 Centering**:
```python
# 픽셀 좌표 가중 평균으로 중심점 계산
cy = (y_coords * alpha_flat).sum() / alpha_flat.sum()
cx = (x_coords * alpha_flat).sum() / alpha_flat.sum()
```

**Pixel-Count 기반 Scaling**:
```python
# 실제 객체 픽셀 수 기반 크기 비율
size_ratio = sqrt(object_pixels / total_pixels)
```

### 2.3 전처리 품질 비교 결과

| Dataset | Size CV | CoM Offset | Grade |
|---------|---------|------------|-------|
| **data_mouse_pixel_based** | **0.16%** | **1.5px** | **A** |
| data_mouse | 7.31% | 119.2px | D |
| data_mouse_centered | 12.82% | 36.0px | D |
| data_mouse_uniform | 8.92% | 38.3px | D |
| data_mouse_uniform_v2 | 0.20% | 42.0px | D |

**결론**: `data_mouse_pixel_based`가 유일하게 Grade A 달성

### 2.4 일반화 전략 분석

**핵심 질문**: 현재 모델이 다른 카메라 환경에 적용 가능한가?

| 변화 요소 | 예상 영향 | 심각도 |
|-----------|----------|--------|
| 다른 생쥐 개체 | 낮음 | ★☆☆ |
| 다른 종/나이 | 낮음-중간 | ★★☆ |
| 다른 해상도 | 중간 | ★★☆ |
| **다른 카메라 각도** | **높음** | **★★★** |
| 다른 뷰 수 (4뷰 등) | 높음 | ★★★ |

**병목**: MVDiffusion은 학습된 카메라 각도 관계에 강하게 의존

### 2.5 Gaussian Mouse Avatar 구현 계획

**목적**: 합성 데이터로 다양한 카메라 환경에 일반화

**모듈 구조**:
```
src/
├── mouse_avatar/
│   ├── body_model.py      # MAMMAL LBS wrapper
│   ├── gaussian_avatar.py # Gaussian 표현
│   └── camera.py          # Camera sampler
├── data_generation/
│   └── synthetic_dataset.py
└── evaluation/
    └── quality_metrics.py
```

**합성 데이터 계획**:
- Azimuth: 0° - 360° (12 samples)
- Elevation: -30° - 60° (4 samples)
- Distance: 0.8x - 2.0x (3 samples)
- 총 288 카메라 구성 × 포즈 변화 = ~1,000 샘플

---

## 3. 실험 현황

| 실험 | 상태 | 비고 |
|------|------|------|
| MVDiffusion pixel_based | 진행 중 (Step ~70/5000) | Grade A 데이터 사용 |
| GS-LRM pixel_based | 대기 | MVDiff 완료 후 |
| End-to-end 테스트 | 대기 | 전체 파이프라인 검증 |

---

## 4. 주요 교훈

1. **Bbox 기반 전처리 실패**: 불규칙 형상(꼬리)이 있는 객체에 부적합
2. **Pixel-based 전처리 성공**: CoM + pixel count 방식이 robust
3. **일반화 한계**: 카메라 배치 변화에는 합성 데이터 필요

---

## 5. 다음 단계

- [ ] MVDiffusion pixel_based 학습 완료 (5000 steps)
- [ ] GS-LRM pixel_based 학습
- [ ] Gaussian Mouse Avatar 모듈 구현 시작
- [ ] 품질 평가 자동화 스크립트 작성

---

*통합 출처*:
- `251220_research_pixel_based_preprocessing.md`
- `251220_research_pixel_based_uniform_scale.md`
- `251220_research_mouse_facelift_generalization.md`
- `251220_research_gaussian_mouse_avatar_implementation.md`
