---
date: 2024-12-15
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, gs-lrm, lgm, 3d-reconstruction, gaussian-splatting]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Stage 2: GS-LRM/LGM 3D Reconstruction 실험 기록

## 1. 목표

**6개 뷰 이미지에서 3D Gaussian Splatting 모델 생성**

```
Input: 6개 뷰 마우스 이미지 (0°, 60°, 120°, 180°, 240°, 300°)
Output: 3D Gaussian Splatting (위치, 색상, 불투명도, 공분산)
```

---

## 2. 테스트한 모델들

### 2.1 GS-LRM (Human Pretrained)
| 항목 | 내용 |
|------|------|
| **출처** | FaceLift 논문의 GS-LRM |
| **Pretrain** | Human 얼굴 데이터 |
| **구조** | Transformer + Gaussian decoder |
| **문제** | Domain gap (Human → Mouse) |

### 2.2 LGM (Objaverse Pretrained)
| 항목 | 내용 |
|------|------|
| **출처** | LGM 논문 |
| **Pretrain** | Objaverse 3D 모델 |
| **구조** | UNet + View-aligned Gaussian |
| **장점** | 일반 객체에 대한 3D prior |

---

## 3. 주요 문제점 및 실험

### 3.1 Mode Collapse 현상

#### 증상
학습 진행에 따라 예측 이미지가 회색 평균으로 수렴

```
Step 1:     GT와 유사한 형태, 색상 유지
Step 5000:  색상 흐릿해짐
Step 20000: 거의 회색 blob으로 수렴
```

#### Supervision 이미지 분석
```
이미지 포맷: [GT(회색 배경) | Pred(흰 배경)] 쌍이 가로로 나열
- 초기: GT와 Pred 유사
- 후기: Pred가 회색 평균으로 collapse
```

#### 원인 분석
1. **2D-only supervision**: L1 + LPIPS loss만 사용
   - 모든 뷰의 평균이 loss 최소화
   - 3D geometry 학습 유도 없음

2. **Domain gap**: Human pretrained 모델
   - 마우스 형태/텍스처와 불일치
   - Fine-tuning 시 prior 손실

3. **데이터 다양성 부족**: ~1,600 샘플
   - 쉽게 overfitting
   - 일반화 어려움

### 3.2 FaceLift 논문과의 차이점

#### FaceLift 성공 요인
```
1. Objaverse Pretrain: 다양한 3D 객체로 일반적인 3D prior 학습
2. Depth Supervision: 2D loss + depth loss 함께 사용
3. Minimal Fine-tune: 적은 양의 fine-tuning으로 domain adaptation
```

#### Mouse-FaceLift 실패 요인
```
1. Human Pretrained: 마우스와 형태 차이 큼
2. 2D-only Loss: 깊이 정보 없이 학습
3. Aggressive Fine-tune: Prior 손실 (catastrophic forgetting)
```

### 3.3 Freeze Layer 실험

#### 목적
Pretrained prior를 보존하면서 domain adaptation

#### 실험 설정
| 실험 | Frozen Layers | Learning Rate | 결과 |
|------|--------------|---------------|------|
| Baseline | None | 1e-4 | Mode collapse |
| Conservative | Encoder + Mid | 1e-5 | 조금 개선 |
| Decoder-only | All except decoder | 1e-5 | 형태 유지 |
| Minimal | All except last layer | 1e-6 | Prior 유지 |

#### 결과
```yaml
# 가장 나은 설정
freeze_encoder: true
freeze_mid: true
learning_rate: 1e-5
trainable_params: ~20% (decoder만)
```

하지만 **여전히 진정한 3D 학습은 되지 않음**

### 3.4 Ray Embedding 버그

#### 발견된 버그
`provider_mouse_6view.py`에서 인덱싱 오류

```python
# 버그 있는 코드
cam_poses = transform.unsqueeze(0) @ cam_poses  # [1, 6, 4, 4]
# cam_poses[i]가 [6, 4, 4] 반환 (잘못됨)

# 수정된 코드
cam_poses = transform.unsqueeze(0) @ cam_poses  # [1, 6, 4, 4]
cam_poses = cam_poses.squeeze(0)  # [6, 4, 4] 추가
# 이제 cam_poses[i]가 [4, 4] 반환 (정상)
```

#### 영향
- Ray embedding이 모든 뷰에서 동일하게 생성됨
- 뷰 구분이 안 되어 3D 학습 불가능
- 수정 후에도 근본적 한계는 존재

### 3.5 LGM v1~v8 실험 이력

| 버전 | 변경사항 | Train PSNR | Val PSNR | 비고 |
|------|----------|------------|----------|------|
| v1 | 초기 설정 | 35.2 | 32.1 | 기본 |
| v2 | 로깅 수정 | 36.1 | 33.5 | - |
| v3 | 데이터 경로 수정 | 38.4 | 35.2 | - |
| v4 | 체크포인트 재개 | 40.1 | 36.8 | - |
| v5 | 카메라 수정 | 42.3 | 38.5 | - |
| v6 | kiui 카메라 | 44.7 | 41.2 | - |
| v7 | 정확한 카메라 | 47.2 | 44.1 | - |
| **v8** | **Ray embedding 수정** | **49.59** | **46.45** | **최종** |

---

## 4. View-Aligned Gaussian의 근본적 한계

### 구조적 제약
```python
# LGM의 Gaussian 생성 방식
# 각 뷰당 128x128 = 16,384 Gaussians
# 총 6 뷰 * 16,384 = 98,304 Gaussians

# 각 뷰의 Gaussian이 독립적으로 분포
View 0: Gaussians spread in [-0.66, 0.66], std ~0.31
View 1: Gaussians spread in [-0.66, 0.66], std ~0.31
...
```

### 문제점
1. **독립적 분포**: 각 뷰의 Gaussian이 서로 연결되지 않음
2. **3D prior 부재**: 진정한 3D geometry가 아닌 2D 이미지의 중첩
3. **360° 회전 시**: 각 뷰의 2D 이미지가 번갈아 보임

### 시각적 증상
```
360° 비디오 출력:
- 6개의 2D 이미지가 각 방향에 배치된 것처럼 보임
- 매끄러운 3D 회전이 아님
- 뷰 사이에서 갑작스러운 전환
```

---

## 5. Confounding Factors (미분리 요소)

### 5.1 Mode Collapse vs Domain Gap
| 가설 | 검증 방법 | 상태 |
|------|----------|------|
| Domain gap이 주원인 | Objaverse pretrained로 테스트 | 부분 검증 |
| 학습 방식이 주원인 | 같은 domain으로 테스트 | 미검증 |

### 5.2 Ray Embedding Bug vs Architecture Limitation
- 버그 수정 후에도 3D 품질 개선 제한적
- 근본적으로 view-aligned 구조의 한계 존재

### 5.3 Freeze Layer vs Learning Rate
- Conservative training에서 두 요소 동시 변경
- 개별 효과 분리 필요

---

## 6. 결론

### 핵심 발견
1. **2D supervision만으로 3D 학습은 근본적으로 어려움**
2. **Mode collapse는 domain gap보다 loss 설계의 문제**
3. **View-aligned Gaussian은 진정한 3D prior 학습 불가**

### 현재 최선의 접근
```
Pretrained LGM/GS-LRM을 Fine-tuning 없이 직접 사용

이유:
- Objaverse prior가 마우스에도 어느 정도 일반화
- Fine-tuning은 prior 손실 초래
- 높은 PSNR ≠ 좋은 3D 품질
```

### 향후 연구 방향

#### 단기
1. Depth supervision 추가 (마우스 3D 스캔 활용)
2. Multi-view consistency loss 강화
3. Perceptual loss 가중치 조정

#### 장기
1. View-aligned가 아닌 global Gaussian 구조 연구
2. 마우스 전용 3D prior 학습
3. Objaverse에 마우스 유사 객체 추가 후 재학습

---

## 7. 관련 파일

| 파일 | 설명 |
|------|------|
| `LGM/core/provider_mouse_6view.py` | 데이터셋 (ray embedding 버그 수정) |
| `scripts/train_lgm_mouse_6view.py` | LGM 학습 스크립트 |
| `configs/mouse_gslrm_conservative.yaml` | Conservative training 설정 |
| `checkpoints/lgm/mouse_6view_v8/` | 최종 학습 체크포인트 |

---

## 8. 실험 시각화 자료

### Supervision 이미지 해석
```
파일: experiments/archive/gslrm_run1/visualizations/step_XXXXX/supervision_*.jpg

형식: [GT(회색배경) | Pred(흰배경)] 쌍이 가로로 12개 뷰
     - GT와 Pred가 나란히 배치
     - 학습 초기: GT ≈ Pred
     - 학습 후기: Pred → 회색 blob (mode collapse)
```

### 360° 비디오 출력
```
파일: /tmp/360_video_*.mp4

증상: 연속적인 3D 회전이 아닌
      6개의 2D 이미지가 각도별로 보임
원인: View-aligned Gaussian + 2D-only loss
```
