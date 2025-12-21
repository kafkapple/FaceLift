---
date: 2024-12-18
context_name: "2_Research"
tags: [ai-assisted, facelift, gslrm, camera-analysis, poc-test]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# GS-LRM Camera Parameter PoC Test Results

## Summary

**결론: 카메라 정규화는 문제의 근본 원인이 아님. 이미지 도메인 차이가 더 큰 영향을 미침.**

## 실험 개요

### 목적
Pretrained GS-LRM 모델의 흐릿한 출력이 카메라 파라미터 차이 때문인지 검증

### 테스트 조건
Mouse 이미지 6장을 다양한 카메라 설정으로 추론:
1. Original Mouse cameras
2. Z-up normalized Mouse cameras
3. Y-up normalized Mouse cameras
4. Human cameras (완전히 다른 카메라 패턴)

## 카메라 분석 결과

### Human Camera Parameters
```
Distances: [2.70, 2.70, 2.70, 2.70, 2.70, 2.70]  # 완벽하게 일정
Up Vector Z: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]      # 모두 정확히 1.0
Up Vector X: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]      # 모두 0
```

### Mouse Camera Parameters (Original)
```
Distances: [2.00, 3.38, 2.96, 2.77, 2.59, 2.49]  # 가변
Up Vector Z: [1.00, 0.97, 1.00, 1.00, 0.96, 0.93] # 가변
Up Vector X: [-0.00, -0.19, -0.02, -0.02, 0.25, -0.06] # 가변
```

### Mouse Camera Parameters (Z-up Normalized)
```
Distances: [2.70, 2.70, 2.70, 2.70, 2.70, 2.70]  # 정규화됨 ✓
Up Vector Z: [1.00, 0.97, 1.00, 1.00, 0.96, 0.93] # 여전히 가변!
Up Vector X: [0.01, -0.18, -0.01, -0.00, 0.27, -0.05] # 여전히 가변!
```

## 핵심 발견

### 1. Z-up 정규화의 한계
- **정규화는 world 좌표계를 회전**시킴
- **개별 카메라의 기울기(tilt) 차이는 수정하지 않음**
- Mouse 데이터는 각 카메라마다 서로 다른 elevation angle을 가짐

### 2. Human 카메라 사용 시 결과가 더 나빠짐!
```
Test Results (시각적 품질):
1. Original Mouse cameras: 흐릿함
2. Z-up normalized: 흐릿함
3. Y-up normalized: 흐릿함
4. Human cameras: 가장 흐릿함! ← 예상 외 결과
```

### 3. 이미지 도메인 차이가 근본 원인
- Human face로 학습된 pretrained 모델
- Mouse body라는 완전히 다른 도메인
- 카메라 파라미터가 완벽해도 도메인 불일치로 인해 품질 저하

## 시각화

![PoC Test Comparison](../experiments/camera_poc_test/camera_poc_comparison.png)

## 결론 및 다음 단계

### 확인된 사실
1. **카메라 정규화는 부차적 문제** - 최적화해도 큰 개선 없음
2. **이미지 도메인 차이가 주요 원인** - Human face vs Mouse body
3. **Fine-tuning이 필수** - Pretrained 모델만으로는 한계

### 권장 다음 단계

#### 단기 (Fine-tuning 개선)
1. **더 긴 학습** - 현재 학습이 충분히 진행되었는지 확인
2. **Learning Rate 조정** - 더 높은 LR로 빠른 adaptation
3. **Perceptual Loss 강화** - 선명도 개선을 위한 loss weight 조정

#### 중기 (데이터 개선)
1. **카메라 패턴 일관성 확보**
   - 모든 샘플에서 동일한 elevation angle 사용
   - 거리 정규화 일관되게 적용
2. **배경 처리**
   - Human 데이터처럼 깔끔한 흰색 배경으로 전처리

#### 장기 (아키텍처 고려)
1. **Scratch training** 검토 - 도메인 차이가 심하면 pretrained weight가 오히려 방해
2. **Domain adaptation** 기법 적용

## 파일 위치
- PoC 테스트 스크립트: `scripts/test_gslrm_with_human_cameras.py`
- 결과 이미지: `experiments/camera_poc_test/`
- 카메라 분석 스크립트: `scripts/analyze_human_vs_mouse_cameras.py`
