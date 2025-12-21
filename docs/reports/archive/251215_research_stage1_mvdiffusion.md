---
date: 2024-12-15
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, mvdiffusion, multi-view-generation]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Stage 1: MVDiffusion Multi-View Generation 실험 기록

## 1. 목표

**단일 마우스 이미지에서 일관된 6개 뷰 이미지 생성**

```
Input: 단일 마우스 이미지 (정면 또는 측면)
Output: 6개 뷰 이미지 (0°, 60°, 120°, 180°, 240°, 300°)
```

---

## 2. 베이스 모델 및 구조

### Zero123++ 기반 MVDiffusion
- **베이스**: Stable Diffusion 2.1
- **구조**: UNet with cross-attention for multi-view consistency
- **Pretrain**: Objaverse 3D 모델에서 렌더링된 이미지

### 입출력 형식
```python
# Input
cond_image: [B, 3, 320, 320]  # 조건 이미지
camera_embeds: [B, 6, 1280]   # 카메라 위치 임베딩

# Output
generated_views: [B, 6, 320, 320, 3]  # 6개 뷰 이미지
```

---

## 3. 주요 실험 및 결과

### 3.1 카메라 정보 주입 실험

#### 문제 상황
FaceLift 논문에서 카메라 파라미터를 UNet에 주입하는 방식이 불명확

#### 시도한 방법들

| 방법 | 구현 | 결과 |
|------|------|------|
| **Cross-attention injection** | 카메라 임베딩을 cross-attn에 추가 | 학습 불안정 |
| **AdaLN injection** | Timestep과 함께 주입 | 비교적 안정적 |
| **Concatenation** | 입력에 직접 concat | 효과 미미 |

#### 최종 선택
```python
# AdaLN 방식으로 카메라 정보 주입
camera_emb = self.camera_proj(camera_embeds)  # [B, 6, 1280]
time_emb = time_emb + camera_emb  # timestep embedding에 더함
```

### 3.2 프롬프트 임베딩 적응 실험

#### 문제 상황
Zero123++는 "a 3D rendering of..." 형태의 프롬프트로 학습됨
마우스 도메인에 맞는 프롬프트 임베딩 필요

#### 시도한 방법들

| 방법 | 내용 | 결과 |
|------|------|------|
| **텍스트 프롬프트 변경** | "a laboratory mouse" | 효과 있음 |
| **Learnable embedding** | 학습 가능한 임베딩 벡터 | Overfitting 우려 |
| **Domain-specific tokens** | [MOUSE] 토큰 추가 | 추가 학습 필요 |

#### 최종 선택
- 프롬프트: `"a laboratory mouse, white background, high quality"`
- 추가 임베딩 학습 없이 텍스트 프롬프트만 변경

### 3.3 Fine-tuning vs Pretrained

#### 실험 설정
```yaml
# Baseline: Pretrained Zero123++ 직접 사용
# Fine-tuned: 마우스 데이터로 5 epoch 학습
learning_rate: 1e-5
batch_size: 4
augmentation: random_flip, color_jitter
```

#### 결과 비교
| 모델 | PSNR | SSIM | View Consistency |
|------|------|------|------------------|
| Pretrained | 18.2 | 0.72 | 낮음 |
| Fine-tuned (1 epoch) | 22.5 | 0.81 | 중간 |
| Fine-tuned (5 epoch) | 25.1 | 0.85 | 높음 |

---

## 4. 발견된 문제점

### 4.1 학습 시 오히려 성능 저하되는 경우

#### 증상
- 초기에는 loss 감소
- 5-10 epoch 후 생성 품질 저하
- 색상 왜곡, 형태 변형 발생

#### 원인 분석
1. **Overfitting**: 마우스 데이터셋 크기 제한 (~1,600 샘플)
2. **Catastrophic forgetting**: Objaverse prior 손실
3. **Learning rate 과다**: 1e-4는 너무 높음

#### 해결책
```yaml
# Conservative fine-tuning 설정
learning_rate: 1e-5  # 낮은 학습률
num_epochs: 5        # 적은 epoch
freeze_encoder: true # 인코더 동결
augmentation:
  random_flip: true
  color_jitter: 0.1
  random_crop: true
```

### 4.2 카메라 파라미터 불일치

#### 문제
- FaceLift 카메라: elevation 30°, azimuth 60° 간격
- 마우스 데이터: elevation 0°, azimuth 60° 간격

#### 해결
```python
# 마우스 데이터에 맞게 카메라 파라미터 수정
elevations = [0, 0, 0, 0, 0, 0]  # 모두 0°
azimuths = [0, 60, 120, 180, 240, 300]  # 60° 간격
```

---

## 5. Confounding Factors (미분리 요소)

> 아래 요소들은 실험에서 동시에 변경되어 개별 효과 분리 어려움

### 5.1 카메라 파라미터 vs 데이터 전처리
- 카메라 수정 시점에 데이터 정규화 방식도 변경됨
- 개별 효과 정량화 필요

### 5.2 Augmentation vs Learning Rate
- Augmentation 추가 시 learning rate도 함께 조정
- 둘 중 어느 것이 주요 효과인지 불명확

### 5.3 프롬프트 vs Fine-tuning 효과
- 프롬프트 변경과 fine-tuning 동시 적용
- 프롬프트만의 효과 측정 필요

---

## 6. 결론 및 권장사항

### Stage 1 성공 요인
1. **Conservative fine-tuning**: 낮은 학습률 + 적은 epoch
2. **적절한 augmentation**: Overfitting 방지
3. **카메라 파라미터 정렬**: 데이터와 모델 일치

### 권장 설정
```yaml
# MVDiffusion Fine-tuning 권장 설정
learning_rate: 1e-5
num_epochs: 3-5
batch_size: 4
freeze_encoder: true  # 처음 절반 epoch

augmentation:
  random_flip: true
  color_jitter: 0.1
  random_crop_ratio: 0.95
```

### 남은 과제
1. View consistency 정량적 평가 메트릭 개발
2. 프롬프트 임베딩 최적화 실험
3. 더 큰 데이터셋으로 검증

---

## 7. 관련 파일

| 파일 | 설명 |
|------|------|
| `configs/mouse_mvdiffusion_6x_aug.yaml` | 최종 학습 설정 |
| `gslrm/data/mouse_dataset.py` | 데이터셋 클래스 |
| `train_diffusion.py` | 학습 스크립트 |
| `docs/guides/prompt_embeds_design.md` | 프롬프트 임베딩 설계 |
