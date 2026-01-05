# Mouse-FaceLift 실험 요약 및 진행 현황

**작성일:** 2026-01-05
**최종 업데이트:** 2026-01-05
**프로젝트:** Mouse 3D/4D Reconstruction using FaceLift Pipeline

---

## 📋 연구 목표

FaceLift (Human Face 3D Reconstruction) 파이프라인을 Mouse 데이터에 적용:
1. Multi-view 이미지에서 3D Gaussian Splatting 기반 재구성
2. MVDiffusion → GS-LRM 2단계 파이프라인 구축
3. Mouse 특화 전처리 및 학습 전략 개발

---

## 🔬 데이터 전처리 분석

### 전처리 방법 비교

| 방법 | 스크립트 | Centering | Scaling | 카메라 정규화 | 권장 |
|------|----------|:---------:|:-------:|:-------------:|:----:|
| **pixel_based** | `preprocess_pixel_based.py` | CoM 기반 | 픽셀 비율 | ✅ 완전 정규화 | ⭐ |
| centered | `preprocess_center_align_all_views.py` | Bbox 중심 | ❌ | ❌ | |
| uniform | `preprocess_uniform_scale.py` | ❌ | 균일 스케일 | ❌ | |

### FaceLift 원본과의 호환성
```
FaceLift Human 원본:    fx=548.99, fy=548.99, cx=256.00, cy=256.00
Mouse pixel_based:      fx=548.99, fy=548.99, cx=256.00, cy=256.00  ✅ 완벽 일치
Mouse centered/uniform: fx≠fy, cx≠cy                               ❌ 불일치
```

### pixel_based 전처리 상세
```python
# Center of Mass (CoM) 기반 Centering - 꼬리 방향 bias 최소화
CoM = Σ(position × alpha) / Σ(alpha)

# Pixel 기반 Scaling - 포즈 변화에 robust
size_ratio = sqrt(pixel_count / total_pixels)
scale = target_ratio / size_ratio  # target_ratio = 0.3
```

---

## 📊 데이터셋 현황

### 서버별 데이터

| 서버 | 데이터셋 | 샘플 수 | 상태 |
|------|----------|---------|------|
| gpu03 | data_mouse (원본) | 2,000 | ✅ 사용 가능 |
| gpu03 | data_mouse_pixel_based | 2,000 | ✅ 처리 완료 |
| joon | data_mouse_centered | 3,597 | ⚠️ 카메라 불일치 |

### 데이터 파이프라인
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

## 🔧 Mask 처리 문제 및 해결

### 발견된 문제
```yaml
# 기존 설정 (문제)
remove_alpha: true        # 알파 채널 제거 → mask 손실
masked_l2_loss: (없음)    # L2 loss에 mask 미적용
```

### 해결책
```yaml
# 수정된 설정 (mouse_gslrm_pixel_based_v2.yaml)
remove_alpha: false           # 알파 채널 유지
masked_l2_loss: true          # 전경에만 L2 loss
masked_ssim_loss: true        # 전경에만 SSIM loss
background_loss_weight: 0.1   # 배경 흰색 유도
```

---

## 💻 실험 환경

### gpu03 서버 설정

| 항목 | 값 |
|------|-----|
| OS | Ubuntu 24.04 LTS |
| GPU | RTX A6000 49GB (device 4 사용) |
| CUDA | 12.4 (conda-forge) |
| PyTorch | 2.6.0+cu124 |
| Python | 3.11 |
| Conda 환경 | facelift |
| xformers | 0.0.29.post3 |

### 학습 설정

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

## 🚀 실행 명령어

### 환경 활성화
```bash
ssh gpu03
conda activate facelift
cd /home/joon/dev/FaceLift
```

### 데이터 전처리 (선택)
```bash
python scripts/preprocess_pixel_based.py \
    --input_dir data_mouse \
    --output_dir data_mouse_pixel_based_v2 \
    --target_size_ratio 0.3 \
    --output_size 512
```

### GS-LRM 학습
```bash
# nohup으로 백그라운드 실행
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse_gslrm_pixel_based_v2.yaml \
    > train_log.txt 2>&1 &
```

### 로그 확인
```bash
tail -f train_log.txt                  # 실시간 모니터링
grep "loss" train_log.txt | tail -20    # loss 값 확인
```

### W&B 대시보드
- https://wandb.ai/kafkapple-joon-kaist/mouse_facelift

---

## ⚙️ Config 변경 요약

| 항목 | 기존 (v1) | 수정 (v2) |
|------|-----------|-----------|
| remove_alpha | true | **false** |
| masked_l2_loss | (없음) | **true** |
| masked_ssim_loss | (없음) | **true** |
| background_loss_weight | (없음) | **0.1** |
| checkpoint_dir | mouse_pixel_based | mouse_pixel_based_v2 |
| exp_name | pixel_based | pixel_based_v2_masked |

---

## 📈 평가 지표

| 지표 | 목표 | 설명 |
|------|------|------|
| PSNR | >25 dB | Peak Signal-to-Noise Ratio |
| SSIM | >0.9 | Structural Similarity Index |
| LPIPS | <0.15 | Learned Perceptual Similarity |
| Mask IoU | >0.9 | GT mask vs 렌더링 mask 일치도 |

### 예상 개선점
1. **전경 재구성 품질 향상**: masked loss로 배경 noise 제거
2. **경계 선명도 개선**: background loss로 명확한 분리
3. **수렴 속도 향상**: 불필요한 배경 학습 제거

---

## 📝 다음 단계

1. [ ] GS-LRM 학습 실행 및 모니터링
2. [ ] Validation 결과 분석
3. [ ] Human pretrained vs Mouse fine-tuned 비교
4. [ ] MVDiffusion 학습 진행
5. [ ] End-to-end 파이프라인 테스트

---

## 📚 관련 문서

- [00_MoC_INDEX.md](./00_MoC_INDEX.md) - 전체 보고서 목차
- [241208_consolidated.md](./241208_consolidated.md) - 카메라 파라미터 분석
- [251213_consolidated.md](./251213_consolidated.md) - prompt_embeds 해결
- [251219_consolidated.md](./251219_consolidated.md) - 알려진 이슈 종합

---

*🤖 Generated with Claude Code*
