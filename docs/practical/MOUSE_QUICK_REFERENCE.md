# FaceLift Mouse Quick Reference v5.1

> Last Updated: 2026-01-24 | Modular Mode | Complete Guide

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Experiment Priority Matrix](#experiment-priority-matrix)
3. [Datasets](#datasets)
4. [Experiments](#experiments)
5. [Preprocessing](#preprocessing)
6. [Complete Commands](#complete-commands)
7. [Monitoring](#monitoring)
8. [File Locations](#file-locations)

---

## Quick Start

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# P0 권장 실험
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha
```

---

## Experiment Priority Matrix

### 전체 우선순위 (Dataset × Experiment)

| Priority | Dataset | Experiment | 특징 | 상태 |
|----------|---------|------------|------|------|
| **P0** | **D8** | **E2_gt_alpha** | Homography + GT mask + α | ✅ Ready |
| **P1** | D7_1 | E2_gt_alpha | Baseline 비교 | ✅ Ready |
| **P2** | D8_1 | E2_gt_alpha | 1.3x zoom (큰 마우스) | ✅ Ready |
| **P3** | D8 | E2_gt_alpha_3v | 3-view 강건성 | ✅ Ready |
| **P4** | D8 | E2_gt_alpha_5v | 5-view 최대 정보 | ✅ Ready |
| **P5** | D10 | E2_gt_alpha | Up-alignment | ⚠️ 전처리 필요 |
| **P6** | D10_1 | E2_gt_alpha | Up + Adaptive zoom | ⚠️ 전처리 필요 |
| **P7** | D9_norm | E2_gt_alpha_native | Full-res (A6000+) | ✅ Ready |

### 실험 목적별 분류

| 목적 | Dataset | Experiment | 설명 |
|------|---------|------------|------|
| **기준선** | D8 | E2_gt_alpha | 모든 비교의 기준 |
| **강건성** | D8 | E2_gt_alpha_3v | 적은 입력에서 성능 |
| **최대 품질** | D8 | E2_gt_alpha_5v | 최대 정보 활용 |
| **마우스 크기** | D8_1, D10_1 | E2_gt_alpha | 프레임 내 마우스 크기 ↑ |
| **좌표 정렬** | D10 | E2_gt_alpha | Turntable 일관성 |
| **고해상도** | D9_norm | E2_gt_alpha_native | 정보 손실 없음 |

---

## Datasets

### 데이터셋 비교표

| Dataset | Paradigm | Transform | Zoom | Up-Align | PP | 상태 |
|---------|----------|-----------|------|----------|-----|------|
| D7_1 | pp_centered | Affine | 1.0x | ❌ | 256 | ✅ Ready |
| **D8** ⭐ | precision | Homography+skew | 1.0x | ❌ | 256 | ✅ Ready |
| D8_1 | precision | Homography+skew | 1.3x | ❌ | var | ✅ Ready |
| D9_norm | native | None | - | ❌ | orig | ✅ Ready |
| **D10** | up_aligned | Homography+skew | 1.0x | ✅ | 256 | ⚠️ 전처리 |
| **D10_1** | up_aligned | Homography+skew | adaptive | ✅ | 256 | ⚠️ 전처리 |

### 데이터 경로

```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/       ✅ Ready (3238 train, 359 val)
├── D8/         ✅ Ready (3238 train, 359 val)
├── D8_1/       ✅ Ready
├── D9_norm/    ✅ Ready
├── D10/        ⚠️ Needs preprocessing
└── D10_1/      ⚠️ Needs preprocessing
```

---

## Experiments

### Mask Mode 실험 (E0-E7)

| Experiment | mask_mode | alpha_loss | threshold | Priority | 설명 |
|------------|-----------|------------|-----------|----------|------|
| **E2_gt_alpha** ⭐ | gt | 0.1 | - | **P0** | **GT + alpha (권장)** |
| **E7_alpha_optimized** | alpha | 0.1 | 0.6 | P7 | Optimized alpha |
| E0_none | none | 0.0 | - | P8 | No mask baseline |
| E1_gt | gt | 0.0 | - | P9 | GT mask only |
| E3_alpha | none | 0.1 | 0.5 | P10 | Alpha only |
| E4_bg_penalty | none | 0.1+bg | - | P9 | Background penalty |

### View Ablation 실험

| Experiment | Input | Holdout | Loss 계산 | 용도 |
|------------|-------|---------|-----------|------|
| E2_gt_alpha_3v | 3 | 3 | 3개 평균 | 강건성 테스트 |
| **E2_gt_alpha** | 4 | 2 | 2개 평균 | **기본 (권장)** |
| E2_gt_alpha_5v | 5 | 1 | 1개 | 최대 정보 |
| E2_gt_alpha_overfit | 1 | 5 | 5개 평균 | 오버핏 테스트 |

### Special Variants

| Experiment | 특징 | 용도 |
|------------|------|------|
| E2_gt_alpha_fixed | random_view=false | 고정 뷰 순서 |
| E2_gt_alpha_native | batch=1, size=1024 | D9/D9_norm 전용 |

---

## Preprocessing

### 전처리 명령어

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# D10: Up-alignment (zoom 1.0x)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D10

# D10.1: Up-alignment + Adaptive zoom (~1.4x)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D10_1
```

### Preset 목록

```bash
# 사용 가능한 preset 확인
python -m mouse_extensions.preprocessing.preprocess --list-presets

# Available: D6-1, D6-2, D6-3, D7, D7.1, D7.2, D8, D8.1, 
#            D9, D9_norm, D9_resized, D10, D10.1, D10.2
```

---

## Complete Commands

### 1. 기존 데이터셋 실험 (즉시 실행)

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

###
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha_3v > logs/D7_1_E2_gt_alpha_3v.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha_fixed > logs/D7_1_E2_gt_alpha_fixed.log 2>&1 &

######

CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E0_none > logs/D8_E0.log 2>&1 &

####

# P0: D8 + E2_gt_alpha (권장)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha > logs/D8_E2.log 2>&1 &

# P1: D7_1 baseline 비교
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha > logs/D7_1_E2.log 2>&1 &

# P2: D8_1 (1.3x zoom)
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8_1 -e E2_gt_alpha > logs/D8_1_E2.log 2>&1 &

# P7: E7_alpha_optimized (GT mask 없이)
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E7_alpha_optimized > logs/D8_E7.log 2>&1 &
```

### 2. View Ablation 실험

```bash
# P3: 3-view input (강건성)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha_3v > logs/D8_E2_3v.log 2>&1 &

# P4: 5-view input (최대 정보)
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha_5v > logs/D8_E2_5v.log 2>&1 &
```

### 3. D10 전처리 + 실험

```bash
# Step 1: D10 전처리
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D10

# Step 2: D10.1 전처리
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D10_1

# Step 3: 실험 실행
# P5: D10 (up-alignment)
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D10 -e E2_gt_alpha > logs/D10_E2.log 2>&1 &

# P6: D10_1 (up + adaptive zoom)
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D10_1 -e E2_gt_alpha > logs/D10_1_E2.log 2>&1 &
```

### 4. High-Resolution 실험 (A6000+ 필요)

```bash
# P7: D9_norm (1152x1024 원본)
CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D9_norm -e E2_gt_alpha_native > logs/D9_norm_E2.log 2>&1 &
```

### 5. 병렬 실행 예시

```bash
# GPU 4-7에서 4개 실험 동시 실행
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha > logs/D8_E2.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha > logs/D7_1_E2.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha_3v > logs/D8_E2_3v.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha_5v > logs/D8_E2_5v.log 2>&1 &
```

---

## Monitoring

### 로그 확인

```bash
# 실시간 로그
tail -f logs/D8_E2.log

# 최근 로그
tail -100 logs/D8_E2.log

# 에러만 확인
grep -i error logs/D8_E2.log
```

### GPU 상태

```bash
# 실시간 GPU 모니터링
watch -n 1 nvidia-smi

# 특정 GPU만
nvidia-smi -i 4,5,6,7
```

### WandB

```
Project: https://wandb.ai/joon/FaceLift
```

### 프로세스 관리

```bash
# 실행 중인 학습 확인
ps aux | grep train_gslrm

# 특정 GPU 프로세스
nvidia-smi -i 4 --query-compute-apps=pid,name --format=csv

# 프로세스 종료 (PID 확인 후)
kill <PID>
```

---

## File Locations

### 코드

```
/home/joon/dev/FaceLift/
├── train_gslrm.py                    # 학습 진입점
├── configs/
│   ├── base/gslrm_mouse.yaml         # 공통 설정
│   ├── datasets/                     # 데이터셋 configs
│   └── experiments/                  # 실험 configs
├── mouse_extensions/
│   └── preprocessing/
│       ├── preprocess.py             # 통합 전처리
│       └── presets.py                # 프리셋 정의
└── docs/
    ├── practical/MOUSE_QUICK_REFERENCE.md  # 이 문서
    └── reference/PREPROCESSING_REGISTRY.md # 전처리 상세
```

### 데이터

```
/home/joon/data/
├── raw/markerless_mouse_1_nerf/      # 원본 데이터
│   ├── videos_undist/                # 비디오
│   ├── simpleclick_undist/           # 마스크
│   ├── new_cam.pkl                   # 카메라
│   └── vertical_lines.npz            # Up 방향
└── preprocessed/FaceLift_mouse/
    ├── D7_1/                         # 전처리된 데이터
    ├── D8/
    └── ...
```

---

## Analysis Reports

### Mask Mode Analysis (D9)

**위치**: 

| Mode | IoU | Best Threshold |
|------|-----|----------------|
| alpha | 0.99 | **0.6** |
| gt | 1.00 | - |
| rgb_pred | 0.07 | 0.3 |

**권장**:  (E2_gt_alpha) 또는  (E7)

---

## See Also

- [PREPROCESSING_REGISTRY.md](../reference/PREPROCESSING_REGISTRY.md) - 전처리 상세 문서
- [experiments/README.md](../../configs/experiments/README.md) - 실험 설정 가이드

---

*FaceLift Mouse Quick Reference v5.1 | 2026-01-24*
