# FaceLift Experiment Quick Start

> **Updated**: 2026-01-27
> **관련 문서**: [DATASET_MASTER_REFERENCE.md](datasets/DATASET_MASTER_REFERENCE.md)

---

## 1. 현재 우선순위 실험

### P1: Coverage 가설 검증 (최우선)

| 순위 | 실험 | 명령어 | 목적 |
|------|------|--------|------|
| **1** | M3_2b + alpha | `-d M3_2b -e E1_2_alpha` | H2 baseline (낮은 zoom) |
| **2** | M3_3 + alpha | `-d M3_3 -e E1_2_alpha` | H2 test (높은 zoom, safe) |

**가설**: Coverage 증가 → PSNR 향상?

| Dataset | Coverage | 예상 |
|---------|----------|------|
| D7_1/D8 | ~3% | PSNR ~20-21 |
| M3_2b | ~4% | PSNR ? |
| M3_3 | ~5-6% | PSNR ? (D3_norm 참고: 27) |

### P2: View Selection / Count

| 순위 | 실험 | 명령어 | 목적 |
|------|------|--------|------|
| 3 | Fixed view | `-d M3_2b -e E1_2_3_alpha_fixed` | Random vs Fixed |
| 4 | 5 views | `-d M3_2b -e E1_2_2_alpha_5v` | View 수 증가 |
| 5 | 3 views | `-d M3_2b -e E1_2_1_alpha_3v` | 최소 view |

### P3: Sanity Check

| 순위 | 실험 | 명령어 | 목적 |
|------|------|--------|------|
| 6 | Overfit | `-d M3_2b -e E1_2_4_alpha_overfit` | PSNR 40+ 확인 |

---

## 2. 실험 카테고리 체계

### Loss 설정 비교

| 카테고리 | mask_mode | alpha | normalize | 용도 |
|----------|-----------|-------|-----------|------|
| **E0** | none | 0 | - | 논문 원본 baseline |
| **E1** | gt | 0~0.1 | true | Production |
| E2 | none | 0.1 | - | Alpha only (실험) |

### E0 계열 (No Mask)

| Config | 설명 |
|--------|------|
| E0_1_facelift | GS-LRM 논문 원본 |
| E0_2_mouse | + Mouse LR/grad_clip 조정 |

### E1 계열 (GT Mask)

| Config | alpha | views | 설명 |
|--------|-------|-------|------|
| E1_1_base | 0 | 4 | Mask만 (Pose Splatter) |
| **E1_2_alpha** | **0.1** | **4** | **Production** |
| E1_2_1_alpha_3v | 0.1 | 3 | 3 input views |
| E1_2_2_alpha_5v | 0.1 | 5 | 5 input views |
| E1_2_3_alpha_fixed | 0.1 | 4 | Fixed view selection |
| E1_2_4_alpha_overfit | 0.1 | 1 | 1 sample overfit |

---

## 3. 데이터셋 선택

### 권장

| Dataset | 용도 | PP | Coverage |
|---------|------|-----|----------|
| **M3_2b** | 실험 baseline | 256 | ~4% |
| **M3_3** | Coverage 테스트 | 256 | ~5-6% |
| M3_1 | 안전한 fallback | 256 | ~4-5% |

### 기준선 (비교용)

| Dataset | 용도 |
|---------|------|
| D7_1 (M1) | Affine baseline |
| D8 (M2) | Homography baseline |

### 실패 (사용 금지)

| Dataset | 이유 |
|---------|------|
| M4 | PP 가변 - Ray Error |
| M3_persample | zoom_center_mode: object |
| D10.3 (M3) | fx=739 버그 |

---

## 4. Quick Commands

```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

# P1: Coverage 가설 검증
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_2b -e E1_2_alpha

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M3_3 -e E1_2_alpha

# 설정 override 예시
train_gslrm.py -d M3_2b -e E1_2_alpha \
    -s training.losses.alpha_loss_weight 0.2
```

---

## 5. 핵심 Wandb 메트릭

| 메트릭 | 의미 | 기대값 |
|--------|------|--------|
| train/psnr | 학습 PSNR | 상승 |
| train/alpha_loss | Alpha supervision | >0 |
| train/mask_iou | Mask 일치도 | 상승 |
| val/psnr | 검증 PSNR | 최종 지표 |

---

## 6. 관련 문서

| 문서 | 내용 |
|------|------|
| [DATASET_MASTER_REFERENCE.md](datasets/DATASET_MASTER_REFERENCE.md) | 데이터셋 SSOT |
| [PREPROCESSING_REGISTRY.md](PREPROCESSING_REGISTRY.md) | 전처리 이력 |

---

*FaceLift Experiment Guide v1.0 | 2026-01-27*
