# FaceLift Mouse Quick Reference v5.2

> Last Updated: 2026-01-24 | Modular Mode | D7_1 Verified

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Baseline Comparison](#baseline-comparison)
3. [Experiment Matrix](#experiment-matrix)
4. [Datasets](#datasets)
5. [Commands](#commands)
6. [Monitoring](#monitoring)

---

## Quick Start

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# 권장 실험 (D7_1 검증됨)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha
```

---

## Baseline Comparison

### E0_paper_original vs E0_mouse_baseline

| 설정 | E0_paper_original | E0_mouse_baseline | 비고 |
|------|-------------------|-------------------|------|
| **lr** | **1e-4** | 1e-5 | 원본 vs finetuning |
| **grad_clip_norm** | 1.0 | 5.0 | 원본 vs 완화 |
| **maximize_view_overlap** | true | false | 원본 vs 등거리 뷰 |
| random_view_selection | true | true | 동일 |
| num_input_views | 4 | 4 | 동일 |
| mask_mode | none | none | 동일 |
| perceptual_loss | 0.5 | 0.5 | 동일 |

### 선택 기준

| 상황 | 권장 설정 |
|------|----------|
| 빠른 수렴, 논문 재현 | **E0_paper_original** (lr=1e-4) |
| 안정적 finetuning | **E0_mouse_baseline** (lr=1e-5) |

### 베이스라인 비교 실험

```bash
# 원본 논문 설정 (lr=1e-4)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_paper_original

# 생쥐 적응 설정 (lr=1e-5)
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_mouse_baseline
```

---

## Experiment Matrix

### Mask Mode 실험

| Experiment | mask_mode | alpha_loss | lr | Priority | 설명 |
|------------|-----------|------------|-----|----------|------|
| **E0_paper_original** | none | 0.0 | 1e-4 | P1 | 논문 원본 baseline |
| **E0_mouse_baseline** | none | 0.0 | 1e-5 | P1 | 생쥐 finetuning |
| **E2_gt_alpha** ⭐ | gt | 0.1 | base | **P0** | **GT + alpha (권장)** |
| E2_gt_alpha_3v | gt | 0.1 | base | P2 | 3-view 강건성 |
| E2_gt_alpha_5v | gt | 0.1 | base | P2 | 5-view 최대 정보 |
| E1_gt | gt | 0.0 | base | P3 | GT mask only |
| E7_alpha_optimized | alpha | 0.1 | base | P4 | Alpha threshold=0.6 |

### 우선순위 설명

- **P0**: 권장 실험 (먼저 실행)
- **P1**: 베이스라인 비교 (lr 차이 분석)
- **P2**: View ablation (3v vs 4v vs 5v)
- **P3-P4**: 보조 실험

---

## Datasets

### 검증 상태

| Dataset | 상태 | 특징 |
|---------|------|------|
| **D7_1** ⭐ | ✅ 검증됨 | Affine, 기본 데이터셋 |
| D8 | ⚠️ 테스트 필요 | Homography + skew |
| D8_1 | ⚠️ 테스트 필요 | 1.3x zoom |
| D9, D10 | 🗄️ 아카이브됨 | 미검증 |

### 데이터 경로

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/       ✅ (3238 train, 359 val)
├── D8/         ⚠️ 테스트 필요
└── D8_1/       ⚠️ 테스트 필요
```

---

## Commands

### 기본 명령어

```bash
# 권장 실험
python train_gslrm.py -d D7_1 -e E2_gt_alpha

# 베이스라인 비교
python train_gslrm.py -d D7_1 -e E0_paper_original   # lr=1e-4
python train_gslrm.py -d D7_1 -e E0_mouse_baseline   # lr=1e-5

# View ablation
python train_gslrm.py -d D7_1 -e E2_gt_alpha_3v      # 3 input views
python train_gslrm.py -d D7_1 -e E2_gt_alpha_5v      # 5 input views
```

### GPU 할당 예시

```bash
# 병렬 실험
CUDA_VISIBLE_DEVICES=0 python train_gslrm.py -d D7_1 -e E0_paper_original &
CUDA_VISIBLE_DEVICES=1 python train_gslrm.py -d D7_1 -e E0_mouse_baseline &
CUDA_VISIBLE_DEVICES=2 python train_gslrm.py -d D7_1 -e E2_gt_alpha &
```

---

## Monitoring

### WandB

- Project: `FaceLift-Mouse`
- Key metrics: `train/loss`, `train/psnr`, `val/psnr`

### 로그 확인

```bash
# 실시간 로그
tail -f outputs/D7_1_E2_gt_alpha/*/train.log

# 최근 체크포인트
ls -lt outputs/D7_1_*/checkpoints/ | head
```

---

## File Locations

| 항목 | 경로 |
|------|------|
| Base config | `configs/base/gslrm_mouse.yaml` |
| Dataset configs | `configs/datasets/D*.yaml` |
| Experiment configs | `configs/experiments/E*.yaml` |
| Outputs | `outputs/{dataset}_{experiment}/` |
| Checkpoints | `checkpoints/gslrm/` |

---

*v5.2 | 2026-01-24 | D7_1 Verified*
