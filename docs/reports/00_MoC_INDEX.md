# Mouse-FaceLift 연구 보고서 목차 (Map of Contents)

**프로젝트:** Mouse 3D/4D Reconstruction using FaceLift Pipeline
**기간:** 2024-12-08 ~ 2026-01-05
**최종 업데이트:** 2026-01-05

---

## 📋 프로젝트 개요

FaceLift (Human Face 3D Reconstruction) 파이프라인을 Mouse 데이터에 적용하여 Multi-view 이미지로부터 3D Gaussian Splatting 기반 재구성 수행.

### 핵심 파이프라인
```
Single View Image → MVDiffusion → 6 Multi-View Images → GS-LRM → 3D Gaussians
```

### 주요 성과
- ✅ 카메라 정규화 파이프라인 구축 (거리 2.7, FOV 50°)
- ✅ pixel_based 전처리 방법 개발 (CoM centering + pixel scaling)
- ✅ prompt_embeds 정렬 문제 해결
- ✅ Mask 적용 학습 설정 완료
- 🔄 GS-LRM 학습 진행 중

---

## 📅 날짜별 보고서 목록

| 날짜 | 파일명 | 주요 내용 |
|------|--------|----------|
| **2024-12-08** | [241208_consolidated.md](./241208_consolidated.md) | 카메라 파라미터 분석, 좌표계 변환, 버그 수정 |
| **2025-12-10** | [251210_finetune_strategy.md](./251210_research_mouse_facelift_finetune_strategy.md) | 2단계 파이프라인 파인튜닝 전략 |
| **2025-12-12** | [251212_consolidated.md](./251212_consolidated.md) | 파이프라인 분석, 도메인 갭 이슈 |
| **2025-12-13** | [251213_consolidated.md](./251213_consolidated.md) | prompt_embeds 해결, 2단계 학습 전략 |
| **2025-12-18** | [251218_camera_spec_comparison.md](./251218_camera_spec_comparison_report.md) | Human vs Mouse 카메라 정량 비교 |
| **2025-12-19** | [251219_consolidated.md](./251219_consolidated.md) | 알려진 이슈 종합, 카메라 정렬 |
| **2025-12-20** | [251220_synthetic_pipeline.md](./251220_critical_synthetic_data_pipeline.md) | 합성 데이터 파이프라인 핵심 사항 |
| **2026-01-05** | [260105_experiment_summary.md](./260105_mouse_facelift_experiment_summary.md) | 최신 실험 요약, 환경 설정 |

---

## 🔑 핵심 이슈 및 해결책 요약

### 1. 카메라 정규화 (Critical)
- **문제**: Mouse 원본 카메라 거리 (2.0~3.4) ≠ FaceLift 표준 (2.7)
- **해결**: `preprocess_pixel_based.py`로 fx=fy=548.99, cx=cy=256.0 정규화
- **참조**: 241208, 251218, 251219

### 2. prompt_embeds 불일치 (Critical)
- **문제**: FaceLift embeds (수평 뷰) ≠ Mouse 카메라 (경사 20°)
- **해결**: Mouse용 prompt_embeds 생성 (`mouse_prompt_embeds_6view/`)
- **참조**: 251212, 251213

### 3. Mask 미적용 (High)
- **문제**: `remove_alpha: true`로 mask 손실
- **해결**: `remove_alpha: false`, `masked_l2_loss: true`, `masked_ssim_loss: true`
- **참조**: 260105

### 4. num_input_views 설정 (High)
- **문제**: num_input_views=1 (너무 어려움)
- **해결**: num_input_views=5 (pretrained와 유사)
- **참조**: 251219

### 5. Perceptual Loss 도메인 불일치 (High)
- **문제**: VGG 기반 loss가 Mouse 도메인에서 gradient explosion
- **해결**: `lpips_loss_weight: 0.0`, `perceptual_loss_weight: 0.0`
- **참조**: 251219

---

## 📁 핵심 파일 위치

### 전처리 스크립트
| 스크립트 | 용도 |
|----------|------|
| `scripts/preprocess_pixel_based.py` | ⭐ 권장: CoM centering + pixel scaling |
| `scripts/convert_markerless_to_facelift.py` | 원본 → FaceLift 형식 변환 |
| `scripts/generate_synthetic_data.py` | 합성 데이터 생성 |

### 학습 Config
| Config | 용도 |
|--------|------|
| `configs/mouse_gslrm_pixel_based_v2.yaml` | ⭐ 권장: GS-LRM 학습 (mask 적용) |
| `configs/mouse_mvdiffusion.yaml` | MVDiffusion 학습 |

### 데이터셋
| 경로 | 상태 | 설명 |
|------|------|------|
| `data_mouse` | ✅ | 원본 (2,000 샘플) |
| `data_mouse_pixel_based` | ✅ | pixel_based 전처리 완료 |

### 체크포인트
| 경로 | 용도 |
|------|------|
| `checkpoints/gslrm/ckpt_0000000000021125.pt` | Human pretrained |
| `checkpoints/gslrm/mouse_pixel_based_v2/` | Mouse fine-tuned |

---

## 🚀 빠른 시작 가이드

### 환경 설정 (gpu03)
```bash
ssh gpu03
conda activate facelift
cd /home/joon/dev/FaceLift
```

### GS-LRM 학습 실행
```bash
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py --config configs/mouse_gslrm_pixel_based_v2.yaml \
    > train_log.txt 2>&1 &
```

### 학습 모니터링
```bash
tail -f train_log.txt
# 또는 W&B: https://wandb.ai/kafkapple-joon-kaist/mouse_facelift
```

---

## 📊 평가 지표

| 지표 | 목표 | 설명 |
|------|------|------|
| PSNR | >25 dB | Peak Signal-to-Noise Ratio |
| SSIM | >0.9 | Structural Similarity Index |
| LPIPS | <0.15 | Learned Perceptual Similarity (낮을수록 좋음) |
| Mask IoU | >0.9 | GT mask vs 렌더링 mask 일치도 |

---

## 📝 문서 관리 규칙

1. **날짜 형식**: YYMMDD (예: 260105 = 2026-01-05)
2. **통합 원칙**: 같은 날짜의 보고서는 하나로 통합
3. **MoC 유지**: 새 보고서 추가 시 이 파일 업데이트
4. **핵심 위주**: 상세 내용보다 결론과 해결책 중심으로 작성

---

*🤖 Generated with Claude Code*
