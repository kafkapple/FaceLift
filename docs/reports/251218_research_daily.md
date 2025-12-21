---
date: 2025-12-18
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, camera-normalization, gslrm-debugging]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# 2025-12-18 연구 일지: 카메라 정규화 및 GS-LRM 디버깅

## 1. 연구 목표

GS-LRM 학습 시 발생하는 blurry output 문제의 근본 원인 파악 및 해결

---

## 2. 진행 내용

### 2.1 카메라 파라미터 비교 분석

**인간 vs 생쥐 데이터 비교**:
| 항목 | 인간 | 생쥐 | 상태 |
|------|------|------|------|
| 이미지 크기 | 512×512 | 512×512 | ✅ MATCH |
| Focal Length | 548.99 | 548.99 | ✅ MATCH |
| 카메라 거리 | 2.700 ± 0.000 | 2.700 ± 0.422 | ⚠️ 분산 차이 |
| Up 방향 | Z-up [0,0,1] | ~Z-up [0,0.015,1] | ⚠️ 미세 불일치 |
| Elevation | -70° ~ +70° | -51° ~ +78° | ⚠️ 분포 차이 |

**핵심 발견**: GS-LRM pretrained 모델은 **Z-up 좌표계** 사용!
- 이전 `normalize_cameras_to_y_up()`이 오히려 역효과

### 2.2 PoC 테스트 결과

**테스트 조건**:
1. Original Mouse cameras
2. Z-up normalized Mouse cameras
3. Y-up normalized Mouse cameras
4. Human cameras (완전히 다른 패턴)

**결론**: **카메라 정규화는 근본 원인이 아님**
- Human camera로 mouse 이미지 추론 시에도 비슷한 결과
- 이미지 도메인 차이가 더 큰 영향

### 2.3 GS-LRM Fine-tune 문제 발견

| 문제 | 영향도 | 해결 |
|------|--------|------|
| `num_input_views=1` 설정 | **매우 높음** | 5개 입력, 1개 평가로 변경 |
| LPIPS/Perceptual Loss | **높음** | 비활성화 (weight=0) |
| 카메라 정규화 | 낮음 | 이미 적용됨 |

**증상**:
- 학습 후 결과가 더 나빠짐 (객체 사라짐, 흰색 출력)
- Gradient explosion (grad_norm > 100)
- PSNR 13-15 정체

**원인**: `num_input_views=1`은 추론 모드 설정
- 학습 시에는 `num_input_views=5`로 5개 입력, 1개 supervision 필요

---

## 3. 주요 교훈

1. **좌표계 확인 필수**: pretrained 모델의 좌표계(Z-up vs Y-up) 먼저 확인
2. **학습 설정 검증**: 추론 설정과 학습 설정 구분 필요
3. **Perceptual Loss 주의**: 도메인 변화 시 VGG 기반 loss가 오히려 해로움

---

## 4. 해결된 설정

```yaml
# 수정된 configs/mouse_gslrm.yaml
training:
  dataset:
    num_views: 6
    num_input_views: 5    # 5개 입력 (원래 1)
  losses:
    lpips_loss_weight: 0.0    # 비활성화 (원래 1.0)
    perceptual_loss_weight: 0.0  # 비활성화
```

---

## 5. 다음 단계

- [x] num_input_views=5 설정으로 재학습
- [x] LPIPS loss 비활성화
- [ ] 이미지 중앙 정렬 문제 확인

---

*통합 출처*:
- `251218_camera_spec_comparison_report.md`
- `251218_research_camera_normalization_issue.md`
- `251218_research_camera_poc_test_results.md`
- `251218_research_gslrm_finetune_debugging.md`
