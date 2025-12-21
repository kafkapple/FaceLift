---
date: 2024-12-18
context_name: "2_Research"
tags: [ai-assisted, facelift, gslrm, debugging, fine-tuning]
project: mouse-facelift
status: in_progress
generator: ai-assisted
generator_tool: claude-code
---

# GS-LRM Mouse Fine-tuning 문제 원인 분석 및 해결

## Executive Summary

Mouse 데이터로 GS-LRM 모델을 fine-tuning할 때 발생하던 **학습 불안정성**과 **성능 저하** 문제의 근본 원인을 파악하고 해결책을 검증했습니다.

### 핵심 발견
| 문제 | 영향도 | 해결 |
|------|--------|------|
| **num_input_views=1 설정** | **매우 높음** | 5개 입력, 1개 평가로 변경 |
| **LPIPS/Perceptual Loss** | **높음** | 비활성화 (weight=0) |
| 카메라 정규화 | 낮음 | 이미 적용됨, 근본 원인 아님 |

---

## 1. 문제 현상

### 1.1 증상
- 학습 전보다 학습 후 결과가 더 나쁨 (객체가 사라지고 흰색으로 변함)
- Gradient explosion 빈번 발생 (grad_norm > 100)
- Optimizer step 스킵 빈도 높음
- PSNR 13-15 수준에서 정체

### 1.2 기존 설정 (문제 설정)
```yaml
training:
  dataset:
    num_views: 6
    num_input_views: 1        # 문제!
  losses:
    lpips_loss_weight: 1.0    # 문제!
    perceptual_loss_weight: 1.0  # 문제!
```

---

## 2. 원인 분석

### 2.1 핵심 원인 #1: num_input_views 불일치

#### 원본 GS-LRM (Human Face)
```
입력: 6개 뷰 → 모델 → 출력: 2개 뷰 예측
```
- Pretrained 모델은 **6개 입력에서 2개를 예측**하도록 학습됨
- 다양한 시점 정보를 종합하여 reconstruction

#### 기존 Mouse 설정 (문제)
```
입력: 1개 뷰 → 모델 → 출력: 5개 뷰 예측
```
- **완전히 다른 task**: 1개에서 5개를 예측하는 것은 훨씬 어려움
- Pretrained weight가 오히려 방해가 됨
- 모델이 "1개 → 다수" 패턴을 학습하려다 불안정해짐

#### 수정된 설정
```yaml
num_views: 6
num_input_views: 5    # 5개 입력, 1개 평가
```

**왜 이전에 1개로 설정했나?**
- Mouse 데이터가 6개 뷰만 있어서 "최대한 많이 예측"하려는 의도
- 하지만 pretrained 모델의 학습 패턴과 완전히 불일치

### 2.2 핵심 원인 #2: Perceptual Loss의 Domain Mismatch

#### VGG-based Loss의 문제
- LPIPS와 Perceptual Loss는 **ImageNet/Human Face**로 학습된 VGG 사용
- Mouse body는 완전히 다른 도메인
- Out-of-distribution 입력에서 **큰 gradient** 발생

#### 실험 결과
| 실험 | LPIPS | Perceptual | 결과 |
|------|-------|------------|------|
| Baseline | 1.0 | 1.0 | Grad explosion 심각 |
| Exp A | 0.0 | 1.0 | 여전히 explosion (145-342) |
| Exp C | 0.0 | 0.0 | **Explosion 없음** |

### 2.3 부차적 요인: 카메라 정규화

#### 분석 결과
- 카메라 거리 정규화 (→ 2.7)는 이미 적용됨
- PoC 테스트에서 Human 카메라 사용 시 오히려 결과가 더 나빠짐
- **결론: 카메라 정규화는 근본 원인이 아님**

Human vs Mouse 카메라 차이:
```
Human: 거리 일정(2.7), Up vector 완전 정렬(Z=1.0)
Mouse: 거리 정규화됨, 개별 카메라 tilt/elevation 다양
```

---

## 3. 제어 실험 결과

### 3.1 실험 설계
| 실험 | 변경 사항 | 목적 |
|------|-----------|------|
| Exp A | LPIPS=0 only | LPIPS 단독 영향 확인 |
| Exp B | num_input_views=5 only | 입력 뷰 수 영향 확인 |
| Exp C | 5views + LPIPS=0 + Perceptual=0 | 종합 해결책 검증 |

### 3.2 결과

| 실험 | PSNR (Train) | Val PSNR | Grad Explosion | 안정성 |
|------|--------------|----------|----------------|--------|
| Baseline | 13-15 | ~15 | 매우 심각 | 불안정 |
| Exp A | 13-15 | ~15 | 발생 (145-342) | 불안정 |
| **Exp B** | **20-23** | **23.6** | 발생 (172-531) | 개선됨 |
| **Exp C** | 19-20 | 23.6 | **없음 (0회)** | **안정** |

### 3.3 핵심 인사이트

1. **num_input_views=5가 PSNR 개선의 핵심**
   - Exp A (LPIPS=0): PSNR 변화 없음
   - Exp B (5views): PSNR 약 7dB 개선

2. **Perceptual Loss 제거가 안정성의 핵심**
   - Exp B: PSNR 높지만 여전히 explosion
   - Exp C: explosion 완전 제거

3. **두 가지 모두 필요**
   - 입력 뷰 수 증가: 성능 개선
   - Perceptual loss 제거: 안정성 확보

---

## 4. 최종 권장 설정

```yaml
# configs/mouse_gslrm_recommended.yaml
training:
  dataset:
    num_views: 6
    num_input_views: 5        # 핵심!

  losses:
    l2_loss_weight: 1.0       # 유지
    lpips_loss_weight: 0.0    # 제거
    perceptual_loss_weight: 0.0  # 제거
    ssim_loss_weight: 0.5     # 유지 (안정적)

  runtime:
    grad_clip_norm: 0.5       # 강화 (1.0 → 0.5)
    allowed_gradnorm_factor: 50  # 낮춤 (100 → 50)
```

---

## 5. 향후 계획

### 단기 (검증)
- [ ] Exp C 장기 학습 안정성 확인 (500+ steps)
- [ ] Validation 지표 모니터링
- [ ] 시각적 품질 확인

### 중기 (개선)
- [ ] SSIM loss weight 최적화
- [ ] Learning rate schedule 조정
- [ ] 더 많은 샘플로 검증

### 장기 (고려사항)
- Perceptual loss 없이 선명도 확보하는 방법 연구
- Mouse 도메인 특화 perceptual loss 탐색

---

## 6. 파일 위치

### 설정 파일
- `configs/mouse_gslrm_exp_c.yaml` - 검증된 안정 설정

### 로그
- `logs/train_exp_a.log` - Exp A 결과
- `logs/train_exp_b.log` - Exp B 결과
- `logs/train_exp_c.log` - Exp C 결과

### 관련 문서
- `docs/reports/251218_research_camera_poc_test_results.md` - 카메라 PoC 테스트

---

## 7. 핵심 교훈

> **"Pretrained 모델 fine-tuning 시, 원본 학습 설정과의 일관성이 매우 중요하다."**

1. **Task 난이도 유지**: 원본 모델의 입출력 비율 유지
2. **Loss function 도메인 고려**: VGG 기반 loss는 도메인 변경에 취약
3. **단계적 검증**: 변수를 하나씩 통제하며 실험
