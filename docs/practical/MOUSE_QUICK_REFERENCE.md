# FaceLift Mouse Quick Reference v5.9

> Last Updated: 2026-01-24 (M3/D10.3 완료, E0 명명 변경) | Modular Mode | D7_1 Verified | Complete Guide

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Baseline Comparison](#baseline-comparison)
3. [Experiment Priority Matrix](#experiment-priority-matrix)
4. [Datasets](#datasets)
5. [Experiments](#experiments)
6. [Preprocessing](#preprocessing)
7. [Complete Commands](#complete-commands)
8. [Visualization & Analysis](#visualization--analysis)
9. [Monitoring](#monitoring)
10. [File Locations](#file-locations)
11. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# P0 권장 실험 (D7_1 검증됨)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha
```

---

## Baseline Comparison

### E0_1_facelift vs E0_2_mouse

| 설정 | E0_1_facelift | E0_2_mouse | 비고 |
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
| 빠른 수렴, 논문 재현 | **E0_1_facelift** (lr=1e-4) |
| 안정적 finetuning | **E0_2_mouse** (lr=1e-5) |

### 베이스라인 비교 실험

```bash
# 원본 논문 설정 (lr=1e-4)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_1_facelift > logs/D7_1_E0_paper.log 2>&1 &

# 생쥐 적응 설정 (lr=1e-5)
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_2_mouse > logs/D7_1_E0_mouse.log 2>&1 &
```

---

## Experiment Priority Matrix

### 전체 우선순위 (Dataset × Experiment)

| Priority | Dataset | Experiment | 특징 | 상태 |
|----------|---------|------------|------|------|
| **P0** | **D7_1** | **E1_2_gt_alpha** | Affine + GT mask + α | ✅ Verified |
| **P1** | D7_1 | E0_1_facelift | 논문 원본 baseline | ✅ Ready |
| **P1** | D7_1 | E0_2_mouse | 생쥐 finetuning | ✅ Ready |
| **P2** | D7_1 | E1_2_gt_alpha_3v | 3-view 강건성 | ✅ Ready |
| **P2** | D7_1 | E1_2_gt_alpha_5v | 5-view 최대 정보 | ✅ Ready |
| **P3** | D8 | E1_2_gt_alpha | Homography | ⚠️ Test needed |
| **P4** | D8_1 | E1_2_gt_alpha | 1.3x zoom | ⚠️ Test needed |

### 실험 목적별 분류

| 목적 | Dataset | Experiment | 설명 |
|------|---------|------------|------|
| **기준선** | D7_1 | E1_2_gt_alpha | 모든 비교의 기준 |
| **베이스라인 비교** | D7_1 | E0_1_facelift, E0_2_mouse | lr 영향 분석 |
| **강건성** | D7_1 | E1_2_gt_alpha_3v | 적은 입력에서 성능 |
| **최대 품질** | D7_1 | E1_2_gt_alpha_5v | 최대 정보 활용 |
| **기하학 정밀** | D8 | E1_2_gt_alpha | Homography + skew |

---

## Datasets

### 데이터셋 비교표

| Dataset | Paradigm | Transform | Zoom | PP | Pretrained 호환 | 상태 |
|---------|----------|-----------|------|-----|-----------------|------|
| **D7_1** ⭐ | pp_centered | Affine (individual) | 1.0x | 256 | ✅ | ✅ Verified |
| D7_2 | pp_centered | Affine (average) | 1.0x | 256 | ✅ | ✅ Ready |
| D8 | precision | Homography+skew | 1.0x | 256 | ✅ | ⚠️ Test needed |
| D8_1 | precision | Homography+skew | 1.3x | var | ✅ | ⚠️ Test needed |
| **D8_2** 🆕 | precision | Homography+skew | **Adaptive** | 256 | ✅ | ⚠️ 전처리 중 |
| D9 | native | None | - | orig | ✅ | ⚠️ 검증 중 |
| D10 | up_aligned | Homography | 1.0x | 256 | ❌ (93° 회전) | ⛔ 비권장 |
| D10.1 | up_aligned | Homography | Adaptive | var | ❌ (93° 회전) | ⛔ 비권장 |
| **D10.3 (M3)** ⭐ | up_aligned_zoom | Homography | Coverage 5% | 256 | ✅ | ✅ **Ready** |

### 데이터 경로

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/       ✅ Verified (3238 train, 359 val)
├── D7_2/       ✅ Ready
├── D8/         ⚠️ Test needed
├── D8_1/       ⚠️ Test needed
├── D9/         ⚠️ 검증 중
├── D10/        ⛔ 비권장 (93° 회전)
└── M3/         ✅ Ready (3597 total: 3238 train, 359 val)
```

---

## Experiments

### Modular Mode 구조

```
train_gslrm.py -d <DATASET> -e <EXPERIMENT>

configs/
├── base/gslrm_mouse.yaml      # 공통 설정 (자동 로드)
├── datasets/<DATASET>.yaml    # 데이터 경로
└── experiments/<EXPERIMENT>.yaml  # 실험 설정
```

### Mask Mode 실험 (E0-E3)

**카테고리 정의:**
- **E0**: Baseline (mask=none)
- **E1**: GT Mask + Alpha (권장) - E1_1_gt, E1_2_gt_alpha⭐, E1_3_gt_alpha_lgm
- **E2**: Alpha Only (마스크 없이 alpha loss만)
- **E3**: Advanced (composite, bg penalty 등)

| Experiment | mask_mode | alpha_loss | lr | Priority | 설명 |
|------------|-----------|------------|-----|----------|------|
| **E0_1_facelift** | none | 0.0 | 1e-4 | P1 | 논문 원본 baseline |
| **E0_2_mouse** | none | 0.0 | 1e-5 | P1 | 생쥐 finetuning |
| E1_1_gt | gt | 0.0 | base | P4 | GT mask only |
| **E1_2_gt_alpha** ⭐ | gt | 0.1 | base | **P0** | GT + alpha supervision |
| E2_1_alpha | none | 0.1 | base | P5 | Alpha supervision only |
| E3_3_bg | none | 0.1+bg | base | P4 | Background penalty |
| E3_1_composite | composite | 0.05 | base | ⛔ | ~~Nerfstudio~~ (alpha 확장 문제) |
| **E3_1_composite_strong** | composite | 0.3 | base | P3 | Splatfacto-W 스타일 |
| **E1_3_lgm_full** | gt | 1.0 | base | P2 | LGM 스타일 (alpha=RGB) |
| E3_4_bg_strong | none | 0.1 | base | P3 | Object-Centric 2DGS |
| E3_5_combined | gt | 0.2+bg | base | P3 | 다중 문헌 조합 |
| **E3_6_clean_bg** 🆕 | gt | 0.2 | base | **P1** | 배경 Gaussian 최소화 |

### View Ablation 실험

| Experiment | Input | Holdout | Loss 계산 | 용도 |
|------------|-------|---------|-----------|------|
| E1_2_gt_alpha_3v | 3 | 3 | 3개 평균 | 강건성 테스트 |
| **E1_2_gt_alpha** | 4 | 2 | 2개 평균 | **기본 (권장)** |
| E1_2_gt_alpha_5v | 5 | 1 | 1개 | 최대 정보 |

### View Selection

| Experiment | random_view_selection | 설명 |
|------------|----------------------|------|
| E1_2_gt_alpha | true (default) | Random view order |
| E1_2_gt_alpha_fixed | false | Fixed view order |

---

## Preprocessing

### 통합 전처리 (Unified Preprocessor)

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# D7_1 (기본)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7_1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1

# D10.3 / M3 (Coverage-based zoom, 5% target)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.3 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3
```

### M-Series (새 명명 체계)

| Alias | Base Preset | 특징 | 권장 |
|-------|-------------|------|------|
| **M1** | D7.1 | Affine, individual scale | 검증됨 |
| **M2** | D8 | Homography + skew | 정밀 기하학 |
| **M3** | D10.3 | Coverage 4.55%, offset 48px (검증됨) | **★ 권장** |

### Preset 목록

```bash
# 사용 가능한 preset 확인
python -m mouse_extensions.preprocessing.preprocess --list-presets
```

### Preset 비교표

| Preset | Paradigm | Zoom | Up-Align | Pretrained 호환 | 상태 |
|--------|----------|------|----------|-----------------|------|
| **D7_1 (M1)** | up_aligned | None | Yes | ✅ | **검증됨** |
| **D8 (M2)** | precision_homography | None | No | ✅ | 안정 |
| **D10.3 (M3)** | up_aligned_zoom | Coverage 5% | Yes | ✅ | **★ 신규** |

### M3 (D10.3) - 권장 신규 데이터셋 ⭐

| 항목 | 값 |
|------|-----|
| **Preset** | D10.3 (precision_homography + coverage zoom) |
| **경로** | `/home/joon/data/preprocessed/FaceLift_mouse/M3/` |
| **샘플 수** | 3597 (Train: 3238, Val: 359) |
| **Split** | 9:1 (data_mouse_train.txt, data_mouse_val.txt) |
| **Adaptive Zoom** | 1.35x (목표 coverage 5%, 실제 ~9.7%) |
| **Config** | `configs/datasets/D10_3.yaml` |
| **상태** | ✅ 전처리 완료, 실험 대기 |

**D10.3 vs D10/D10.1 차이점**:
- D10/D10.1: `up_alignment=True` → 93도 좌표 회전 문제 발생
- **D10.3**: `up_alignment=False` + `zoom_after_transform=True` → 문제 해결
- D10.3은 **homography 변환 후** adaptive zoom 적용 → 기하학적 정확도 유지

**M3 전처리 명령어**:
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.3 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3
```

**M3 실험 실행**:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D10_3 -e E1_2_gt_alpha
```

**D10.3 (M3) 설정 상세:**
```yaml
paradigm: up_aligned_zoom
adaptive_zoom: true
zoom_method: coverage_based
target_fg_coverage: 0.05    # 목표 FG 5% (pretrained 분포 ~5-8%)
zoom_range: [1.0, 2.5]
single_folder: true         # 유연한 split 지원
```

### 전처리 검증

```bash
# 카메라 파라미터 검증
python -m mouse_extensions.scripts.diagnostics.validate_cameras \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

### Split 관리 (Flexible Split System)

```bash
# Random split (M3 → M3.r)
python -m mouse_extensions.preprocessing.split_manager \
    --data-dir /path/to/M3 --preset random

# Temporal stratified (Pose Splatter 방식, M3 → M3.s)
python -m mouse_extensions.preprocessing.split_manager \
    --data-dir /path/to/M3 --preset pose_splatter

# Temporal 3-way split (M3 → M3.t)
python -m mouse_extensions.preprocessing.split_manager \
    --data-dir /path/to/M3 --preset temporal_3way

# Custom YAML config
python -m mouse_extensions.preprocessing.split_manager \
    --data-dir /path/to/M3 --config configs/splits/custom.yaml
```

**Split 버전 접미사:**
| 접미사 | 전략 | 설명 |
|--------|------|------|
| `.r` | random | 무작위 셔플 |
| `.t` | temporal | 시간순 3분할 |
| `.s` | temporal_stratified | Pose Splatter 방식 |

---

## Complete Commands

### 1. 권장 실험 (P0)

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# P0: D7_1 + E1_2_gt_alpha (권장, 검증됨)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha > logs/D7_1_E1_2_gt_alpha.log 2>&1 &
```

### 2. 베이스라인 비교 (P1)

```bash
# P1: 논문 원본 baseline (lr=1e-4)
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_1_facelift > logs/D7_1_E0_paper.log 2>&1 &

# P1: 생쥐 finetuning baseline (lr=1e-5)
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_2_mouse > logs/D7_1_E0_mouse.log 2>&1 &
```

### 3. View Ablation (P2)

```bash
# P2: 3-view input (강건성)
CUDA_VISIBLE_DEVICES=3 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha_3v > logs/D7_1_E2_3v.log 2>&1 &

# P2: 5-view input (최대 정보)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha_5v > logs/D7_1_E2_5v.log 2>&1 &
```

### 4. 기타 Mask Mode (P3-P4)

```bash
# P3: GT mask only
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_1_gt > logs/D7_1_E1_1_gt.log 2>&1 &

# P4: Background penalty
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E3_3_bg > logs/D7_1_E4_bg.log 2>&1 &
```

### 5. D8 테스트 (미검증)

```bash
# P3: D8 (Homography) - 테스트 필요
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E1_2_gt_alpha > logs/D8_E1_2_gt_alpha.log 2>&1 &
```

### 6. 병렬 실행 예시

```bash
# GPU 0-5에서 6개 실험 동시 실행 (각 GPU 1개 실험)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha > logs/D7_1_E2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_1_facelift > logs/D7_1_E0_paper.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_2_mouse > logs/D7_1_E0_mouse.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha_3v > logs/D7_1_E2_3v.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha_5v > logs/D7_1_E2_5v.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D10_3 -e E1_2_gt_alpha > logs/M3_E1_2.log 2>&1 &

# 추가 데이터셋 테스트 (GPU 6-7)
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E0_1_facelift > logs/D8_E0_paper.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8_2 -e E0_1_facelift > logs/D8_2_E0_paper.log 2>&1 &
```

---

## Visualization & Analysis

### 1. Mask Mode 분석

```bash
# 마스크 모드 비교 분석
python -m mouse_extensions.scripts.analysis.mask_mode_analysis \
    --checkpoint checkpoints/gslrm/D7_1_E1_2_gt_alpha/latest.pt \
    --output_dir outputs/mask_analysis
```

### 2. Alpha 분석

```bash
# Rendered alpha 분석
python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint checkpoints/gslrm/D7_1_E1_2_gt_alpha/latest.pt

# Alpha threshold 분석
python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E1_2_gt_alpha/latest.pt
```

### 3. 카메라 시각화

```bash
# 카메라 설정 시각화
python -m mouse_extensions.scripts.visualize_camera_setup_v4 \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --output_dir outputs/camera_viz
```

### 4. Temporal Turntable (동영상 생성)

```bash
# Turntable 동영상 생성
python -m mouse_extensions.scripts.inference.temporal_turntable \
    --checkpoint checkpoints/gslrm/D7_1_E1_2_gt_alpha/latest.pt \
    --output_dir outputs/turntable \
    --num_frames 60
```

### 5. Checkpoint 렌더링

```bash
# Checkpoint에서 렌더링
python -m mouse_extensions.scripts.render_from_checkpoint \
    --checkpoint checkpoints/gslrm/D7_1_E1_2_gt_alpha/latest.pt \
    --input_dir /path/to/test/data \
    --output_dir outputs/renders
```

---

## Monitoring

### 로그 확인

```bash
# 실시간 로그
tail -f logs/D7_1_E1_2_gt_alpha.log

# 최근 로그
tail -100 logs/D7_1_E1_2_gt_alpha.log

# 에러만 확인
grep -i error logs/D7_1_E1_2_gt_alpha.log

# 여러 로그 동시 확인
tail -f logs/D7_1_*.log
```

### GPU 상태

```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi

# 특정 GPU만
nvidia-smi -i 0,1,2,3

# GPU 프로세스 확인
nvidia-smi --query-compute-apps=pid,name,gpu_name,used_memory --format=csv
```

### WandB

```
Project: https://wandb.ai/joon/FaceLift
Key metrics: train/loss, train/psnr, val/psnr, mask/fg_coverage
```

### 프로세스 관리

```bash
# 실행 중인 학습 확인
ps aux | grep train_gslrm

# 특정 GPU 프로세스
nvidia-smi -i 0 --query-compute-apps=pid,name --format=csv

# 프로세스 종료 (PID 확인 후)
kill <PID>

# ⚠️ 절대 사용 금지: killall python (다른 실험 종료됨)
```

### 주요 메트릭

| 메트릭 | 설명 | 정상 범위 |
|--------|------|----------|
| train/loss | Total loss | 0.1~0.5 |
| train/psnr | PSNR | 15~25 |
| train/l2_loss | L2 reconstruction | 0.01~0.1 |
| val/psnr | Validation PSNR | 15~25 |
| mask/fg_coverage | Foreground coverage | 0.3~0.7 |

---

## File Locations

### Config

| 항목 | 위치 |
|------|------|
| Base config | `configs/base/gslrm_mouse.yaml` |
| Dataset configs | `configs/datasets/D*.yaml` |
| Experiment configs | `configs/experiments/E*.yaml` |
| Combined configs | `configs/mouse/D*_E*.yaml` |

### Data

| 항목 | 위치 |
|------|------|
| Raw data | `/home/joon/data/raw/markerless_mouse_*` |
| Preprocessed | `/home/joon/data/preprocessed/FaceLift_mouse/` |
| Train list | `{dataset}/data_mouse_train.txt` |
| Val list | `{dataset}/data_mouse_val.txt` |

### Model

| 항목 | 위치 |
|------|------|
| Pretrained | `checkpoints/gslrm/ckpt_0000000000021125.pt` |
| Experiment checkpoints | `checkpoints/gslrm/{dataset}_{experiment}/` |
| Outputs | `outputs/{dataset}_{experiment}/` |

### Code

| 항목 | 위치 |
|------|------|
| Training script | `train_gslrm.py` |
| Model | `gslrm/model/gslrm.py` |
| Mouse extensions | `mouse_extensions/` |
| Preprocessing | `mouse_extensions/preprocessing/` |
| Analysis scripts | `mouse_extensions/scripts/analysis/` |
| Visualization | `mouse_extensions/visualization/` |

### Documentation

| 항목 | 위치 |
|------|------|
| This document | `docs/practical/MOUSE_QUICK_REFERENCE.md` |
| Preprocessing registry | `docs/PREPROCESSING_REGISTRY.md` |
| Experiment registry | `docs/practical/experiments/EXPERIMENT_REGISTRY.md` |

---

## Troubleshooting

---

## Mask Experiments

### mask_mode 비교

| Mode | RGB Loss 영역 | 특징 | 권장 |
|------|--------------|------|------|
| **gt** | GT mask만 | 안정적 + alpha_loss 조합 권장 | ★★★ |
| alpha | Rendered α | 동적, 확장 위험 | ★☆☆ |
| composite | 전체 (배경 합성) | 흰 배경 적합 | ★★☆ |
| none | 전체 이미지 | 마스크 미사용 | ★★☆ |

### 실험 설정 비교

| 실험 | mask_mode | alpha_w | bg_w | 문헌 | 상태 |
|------|-----------|---------|------|------|------|
| **E1_2_gt_alpha** ★ | gt | 0.1 | 0.0 | LGM+PS | ✅ 권장 |
| E3_1_composite_strong | composite | 0.3 | 0.0 | Splatfacto-W | 실험적 |
| E1_3_lgm_full | gt | 1.0 | 0.0 | LGM | 실험적 |
| E3_4_bg_strong | none | 0.1 | 1.0 | 2DGS | 실험적 |
| E3_5_combined | gt | 0.2 | 0.3 | 다중 | 실험적 |

### 마스크 실험 명령어

```bash
# E1_2_gt_alpha (권장)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_2_gt_alpha > logs/E1_2_gt_alpha.log 2>&1 &

# E1_3_lgm_full (LGM 스타일)
CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_3_lgm_full > logs/E6_lgm.log 2>&1 &

# E3_5_combined (다중 문헌)
CUDA_VISIBLE_DEVICES=2 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E3_5_combined > logs/E3_5_combined.log 2>&1 &
```

### 핵심 발견

1. **D7_1 마스크 품질**: 깨끗한 이진값 (0/255), 모든 threshold에서 GT와 IoU=1.0
2. **rgb_pred 부적합**: IoU 0.06-0.08, false positive 과다
3. **권장**: `mask_mode: gt` + `alpha_loss_weight: 0.1`


### Config 오류

```bash
# Modular mode 권장 (Legacy mode 대신)
python train_gslrm.py -d D7_1 -e E1_2_gt_alpha    # ✅ Modular
python train_gslrm.py --config file.yaml        # ⚠️ Legacy
```

### 메모리 부족

- `batch_size_per_gpu` 줄이기 (base config에서 2 → 1)
- `num_workers` 줄이기

### CUDA 오류

```bash
# GPU 상태 확인
nvidia-smi

# 특정 GPU 지정
CUDA_VISIBLE_DEVICES=5 torchrun ...
```

### Pretrained checkpoint 오류

```bash
# 항상 pretrained 사용 확인
ls checkpoints/gslrm/ckpt_0000000000021125.pt

# Base config에 포함되어 있으므로 Modular mode 사용 시 자동 적용
```

---

## See Also

- [CONFIG_SCHEMA.md](config/CONFIG_SCHEMA.md) - Config 구조 상세
- [PREPROCESSING_REGISTRY.md](../PREPROCESSING_REGISTRY.md) - 전처리 버전 기록
- [configs/README.md](../../configs/README.md) - Config 시스템 개요
- [EXPERIMENT_REGISTRY.md](experiments/EXPERIMENT_REGISTRY.md) - 실험 목록

---

*FaceLift Mouse Quick Reference v5.9 | Complete Guide | 2026-01-24*

---

## D8.2: Adaptive Zoom Preprocessing (NEW)

### 특징
- **D8 기반**: Homography + skew correction (기하학적 정확도 유지)
- **Adaptive Zoom**: bbox 기반 자동 확대 (생쥐가 프레임의 85% 차지)
- **NO Up-Alignment**: D10의 좌표계 회전 문제 없음

### 명령어
```bash
python -m mouse_extensions.preprocessing.preprocess \
    --preset D8.2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D8_2
```

### 기대 출력
```
Computing adaptive zoom (target fill: 0.85)
  Adaptive zoom: 1.XXx
Transform: homography, Scale: individual, Zoom: 1.XXx
```

### 기하학적 정확도 + Zoom 원리
```
Homography H = K' · K⁻¹ (projective transform)
- fx' = fx × scale × zoom
- PP shift: (cx,cy) → (256,256)
- 3D→2D projection 관계 정확히 보존
- Ray direction 보존
```

---

## Temporal Turntable Video Generation

### 용도
연속 프레임에서 시간에 따른 3D 재구성 변화를 영상으로 생성

### 명령어
```bash
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=0 python -m mouse_extensions.scripts.inference.temporal_turntable \
    --checkpoint checkpoints/gslrm/D7_1_E0_1_facelift/iter_XXXXX/model.pt \
    --config checkpoints/gslrm/D7_1_E0_1_facelift/config.yaml \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1/train \
    --start_frame 0 \
    --end_frame 30 \
    --frame_step 1 \
    --mode rotating \
    --num_views 60 \
    --resolution 384 \
    --output_dir outputs/temporal_test
```

### 모드 옵션
| 모드 | 설명 | 출력 |
|------|------|------|
| `temporal_single_angle` | 고정 시점, 시간만 변화 | T frames |
| `rotating` | 시간 + 회전 동시 | T frames |
| `full_turntable` | 전체 시간 × 전체 각도 | T × N frames |

### 주요 파라미터
| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--start_frame` | 시작 프레임 | 0 |
| `--end_frame` | 끝 프레임 | 30 |
| `--frame_step` | 프레임 간격 | 1 |
| `--num_views` | 360° 분할 수 | 60 |
| `--resolution` | 출력 해상도 | 384 |
| `--elevation` | 카메라 고도 (도) | 15.0 |
| `--radius` | 카메라 거리 | 2.0 |

---


---

## Background Gaussian Reduction

### 문제
GS-LRM은 모든 픽셀에 Gaussian 할당 → 작은 생쥐 주변 흰색 배경에도 Gaussian 생성

### 해결책 1: 학습 중 (E3_6_clean_bg)

```yaml
# configs/experiments/E3_6_clean_bg.yaml
training:
  losses:
    mask_mode: gt              # RGB loss → 전경 영역만
    alpha_loss_weight: 0.2     # rendered α → GT mask
    bg_loss_weight: 0.5        # 배경 α → 0
    opacity_reg_weight: 0.01   # opacity → 0 or 1
    ghost_reg_weight: 0.1      # 배경 영역 α 패널티
```

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E3_6_clean_bg
```

### 해결책 2: 후처리 Pruning

```bash
# Low-opacity Gaussian 제거
python -m mouse_extensions.scripts.inference.prune_gaussians \
    --input gaussians.ply \
    --output gaussians_pruned.ply \
    --opacity_threshold 0.1
```

| Threshold | 효과 |
|-----------|------|
| 0.05 | 보수적 |
| 0.1 | 권장 |
| 0.2 | 적극적 |

### 문헌 근거

| 방법 | 논문 | 설정 |
|------|------|------|
| Alpha Supervision | LGM (ECCV 2024) | `alpha_loss_weight` |
| Background Penalty | Object-Centric 2DGS | `bg_loss_weight` |
| Opacity Regularization | StableGS | `opacity_reg_weight` |
| Ghost Regularization | (자체 구현) | `ghost_reg_weight` |

## Known Issues & Solutions

### D10 렌더링 실패
| 문제 | Up-alignment가 ~93° 좌표계 회전 적용 |
|------|--------------------------------------|
| 증상 | PSNR ~18 (D7_1: ~23), 렌더링 왜곡 |
| 원인 | vertical_lines.npz의 up 벡터가 X축 방향 |
| 해결 | **D8.2 사용** (up-alignment 없음) |
### GSLRM.train() 반환값 버그 (수정됨)
| 문제 | `model.to(device).eval()` 시 `None` 반환 |
|------|------------------------------------------|
| 원인 | `GSLRM.train()` 메서드가 `return self` 누락 |
| 해결 | `return self` 추가 (commit `aec73ff`) |



### D8.2 Adaptive Zoom 미작동 (수정됨)
| 문제 | zoom=1.0으로 고정 |
|------|-------------------|
| 원인 | PRECISION_HOMOGRAPHY에서 adaptive_zoom 미로드 |
| 해결 | preprocess.py 수정 (commit `178cbcb`) |

---

## Dataset Selection Guide

### 빠른 선택

| 목적 | 권장 Dataset | 이유 |
|------|-------------|------|
| **기준선/안정성** | D7_1 | ✅ 검증됨, 안정적 |
| **생쥐 크기 최대화** | D8.2 | Adaptive zoom, ray 정확 |
| **고정 확대** | D8.1 | 1.3x zoom, 간단 |
| **원본 해상도** | D9 | 변환 없음, 메모리 4.5x |

### 결정 흐름도

```
                    ┌──────────────────────────────────────┐
                    │    생쥐 영역 최대화 필요?            │
                    └─────────────┬────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                   YES                          NO
                    │                           │
                    ▼                           ▼
    ┌───────────────────────────────┐   ┌───────────────┐
    │  프레임마다 zoom 다르게?      │   │    D7_1       │
    └───────────────┬───────────────┘   │  Baseline     │
                    │                   │  (✅ 검증됨)  │
          ┌─────────┴─────────┐         └───────────────┘
          │                   │
         YES                  NO
          │                   │
          ▼                   ▼
   ┌──────────────┐   ┌──────────────┐
   │   D8.2       │   │   D8.1       │
   │ Adaptive     │   │ Fixed 1.3x   │
   │ ★ 권장      │   │              │
   └──────────────┘   └──────────────┘
```

### ⚠️ 피해야 할 설정

| Dataset | 문제 |
|---------|------|
| D10, D10.1, D10.2 | Up-alignment로 ~93° 좌표 회전 → Pretrained 불호환 |
| D1-D4 | Deprecated (cross-view inconsistency) |

---

*v5.6 | 2026-01-24 | E6 실험 추가, Background Gaussian Reduction, Pruning 스크립트*
