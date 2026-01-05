# Mouse-FaceLift 실험 요약 및 진행 현황

**작성일:** 2025-01-05
**작성자:** 연구팀
**프로젝트:** Mouse 3D/4D Reconstruction using FaceLift Pipeline

---

## 1. 연구 목표

FaceLift (Human Face 3D Reconstruction) 파이프라인을 Mouse 데이터에 적용하여:
1. Multi-view 이미지에서 3D Gaussian Splatting 기반 재구성
2. MVDiffusion → GS-LRM 2단계 파이프라인 구축
3. Mouse 특화 전처리 및 학습 전략 개발

---

## 2. 데이터 전처리 분석

### 2.1 전처리 방법 비교

| 방법 | 스크립트 | Centering | Scaling | 카메라 정규화 | BG 제거 |
|------|----------|:---------:|:-------:|:-------------:|:-------:|
| **pixel_based** ⭐ | `preprocess_pixel_based.py` | CoM 기반 | 픽셀 비율 | ✅ 완전 정규화 | ✅ |
| centered | `preprocess_center_align_all_views.py` | Bbox 중심 | ❌ | ❌ | ✅ |
| uniform | `preprocess_uniform_scale.py` | ❌ | 균일 스케일 | ❌ | ✅ |

### 2.2 FaceLift 원본과의 호환성

```
FaceLift Human 원본:    fx=548.99, fy=548.99, cx=256.00, cy=256.00
Mouse pixel_based:      fx=548.99, fy=548.99, cx=256.00, cy=256.00  ✅ 완벽 일치
Mouse centered/uniform: fx≠fy, cx≠cy (다름)                        ❌ 불일치
```

**결론:** `pixel_based` 전처리가 FaceLift pretrained 모델과 가장 호환성이 높음

### 2.3 pixel_based 전처리 상세

```python
# Center of Mass (CoM) 기반 Centering
# - Bbox 중심 대신 픽셀 질량 중심 사용
# - 꼬리 방향에 따른 bias 최소화

CoM = Σ(position × alpha) / Σ(alpha)

# Pixel 기반 Scaling
# - Bbox 크기 대신 실제 픽셀 점유율 사용
# - 포즈 변화에 robust

size_ratio = sqrt(pixel_count / total_pixels)
scale = target_ratio / size_ratio  # target_ratio = 0.3
```

---

## 3. 데이터셋 현황

### 3.1 서버별 데이터

| 서버 | 데이터셋 | 샘플 수 | 형식 | 상태 |
|------|----------|---------|------|------|
| gpu03 | data_mouse (원본) | 2,000 | RGBA, 정규화 카메라 | ✅ 사용 가능 |
| gpu03 | data_mouse_pixel_based | 2,000 | RGBA, 정규화 카메라 | ✅ 처리 완료 |
| joon | data_mouse_centered | 3,597 | RGBA, 원본 카메라 | ⚠️ 카메라 불일치 |
| joon | data_mouse_uniform | 3,597 | RGBA, 원본 카메라 | ⚠️ 카메라 불일치 |

### 3.2 데이터 파이프라인

```
원본 Markerless Mouse 데이터
    ↓
배경 제거 + 카메라 정규화 (data_mouse)
    ↓
pixel_based 전처리 (CoM centering + pixel scaling)
    ↓
data_mouse_pixel_based (GS-LRM 학습용)
```

---

## 4. Mask 처리 문제 및 해결

### 4.1 발견된 문제

기존 설정에서 mask가 loss에 적용되지 않았음:

```yaml
# 기존 설정 (문제)
remove_alpha: true        # 알파 채널 제거 → mask 손실
masked_l2_loss: (없음)    # L2 loss에 mask 미적용
masked_ssim_loss: (없음)  # SSIM loss에 mask 미적용
```

### 4.2 해결책

```yaml
# 수정된 설정 (mouse_gslrm_pixel_based_v2.yaml)
remove_alpha: false           # 알파 채널 유지
masked_l2_loss: true          # 전경에만 L2 loss
masked_ssim_loss: true        # 전경에만 SSIM loss
background_loss_weight: 0.1   # 배경 흰색 유도
```

### 4.3 Mask 처리 흐름

```
RGBA 이미지 로드 (4채널)
         ↓
remove_alpha: false → 알파 채널 유지
         ↓
forward() 에서 분리: RGB(3) + Mask(1)
         ↓
    ┌────┴────┐
    ↓         ↓
L2 Loss   SSIM Loss
(mask>0.5만) (mask>0.5만)
    ↓         ↓
    └────┬────┘
         ↓
Background Loss (mask<0.5 → 흰색)
         ↓
Mask IoU 로깅 (GT mask vs 렌더링 mask)
```

---

## 5. 실험 환경

### 5.1 gpu03 서버 설정

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 24.04 LTS |
| GPU | RTX A6000 49GB (device 4 사용) |
| CUDA | 12.4 (conda-forge) |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11 |
| Conda 환경 | facelift |

### 5.2 학습 설정

| 항목 | 값 |
|------|-----|
| batch_size_per_gpu | 2 |
| grad_accum_steps | 2 |
| learning_rate | 1e-6 |
| num_views | 6 |
| num_input_views | 5 |
| image_size | 512 |
| pretrained | ckpt_0000000000021125.pt |

---

## 6. 실행 명령어

### 6.1 환경 활성화

```bash
ssh gpu03
conda activate facelift
cd /home/joon/dev/FaceLift
```

### 6.2 데이터 전처리 (선택)

```bash
# pixel_based 전처리 재실행 (필요시)
python scripts/preprocess_pixel_based.py \
    --input_dir data_mouse \
    --output_dir data_mouse_pixel_based_v2 \
    --target_size_ratio 0.3 \
    --output_size 512
```

### 6.3 GS-LRM 학습

```bash
# Mask 적용된 새 config로 학습
python train_gslrm.py --config configs/mouse_gslrm_pixel_based_v2.yaml
```

---

## 7. Config 변경 요약

| 항목 | 기존 (v1) | 수정 (v2) |
|------|-----------|-----------|
| remove_alpha | true | **false** |
| masked_l2_loss | (없음) | **true** |
| masked_ssim_loss | (없음) | **true** |
| background_loss_weight | (없음) | **0.1** |
| checkpoint_dir | mouse_pixel_based | mouse_pixel_based_v2 |
| exp_name | pixel_based | pixel_based_v2_masked |

---

## 8. 예상 결과 및 평가 지표

### 8.1 평가 지표

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Mask IoU**: GT mask vs 렌더링 mask 일치도

### 8.2 예상 개선점

1. **전경 재구성 품질 향상**: masked loss로 배경 noise 제거
2. **경계 선명도 개선**: background loss로 명확한 foreground/background 분리
3. **수렴 속도 향상**: 불필요한 배경 학습 제거

---

## 9. 다음 단계

1. [ ] GS-LRM 학습 실행 및 모니터링
2. [ ] Validation 결과 분석
3. [ ] Human pretrained vs Mouse fine-tuned 비교
4. [ ] MVDiffusion 학습 진행
5. [ ] End-to-end 파이프라인 테스트

---

## 10. 참고 자료

- FaceLift 원본 논문/코드
- GS-LRM (Gaussian Splatting Large Reconstruction Model)
- MVDiffusion (Multi-View Diffusion)

---

*이 문서는 Claude Code를 활용하여 자동 생성되었습니다.*
