# FaceLift Mouse Quick Reference v4.1

> Last Updated: 2026-01-24 | Modular Mode

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Experiments](#experiments)
3. [Datasets](#datasets)
4. [Preprocessing](#preprocessing)
5. [Visualization & Analysis](#visualization--analysis)
6. [Monitoring](#monitoring)
7. [File Locations](#file-locations)

---

## Quick Start

```bash
cd /home/joon/dev/FaceLift
conda activate facelift

# 권장 실험 (P0)
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha
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

### 실험 카테고리

#### 1. Mask Mode (E0-E5)
| Experiment | mask_mode | alpha_loss | bg_loss | Priority | 설명 |
|------------|-----------|------------|---------|----------|------|
| **E0_none** | none | 0.0 | 0.0 | P2 | Baseline, no mask |
| **E1_gt** | gt | 0.0 | 0.0 | P3 | GT mask only |
| **E2_gt_alpha** ⭐ | gt | 0.1 | 0.0 | **P0** | GT + alpha supervision |
| **E3_alpha** | none | 0.1 | 0.0 | P4 | Alpha supervision only |
| **E4_bg_penalty** | none | 0.1 | 0.5 | P3 | Background penalty |
| **E5_composite** | composite | 0.0 | 0.0 | P5 | Nerfstudio style |
| **E6_rgb_pred** | rgb_pred | 0.0 | 0.0 | P5 | RGB mask (deprecated) |

#### 2. View Ablation
| Experiment | num_input_views | holdout | 설명 |
|------------|-----------------|---------|------|
| **E2_gt_alpha_3v** | 3 | 3 | 강건성 테스트 (적은 입력) |
| E2_gt_alpha | 4 (default) | 2 | 기본 설정 |
| **E2_gt_alpha_5v** | 5 | 1 | 최대 정보 |
| **E2_gt_alpha_overfit** | 1 | 5 | 오버핏 테스트 |

**Loss 계산**: Holdout 뷰들의 loss **평균**

**권장**:
- 일반 학습: 4개 입력 (균형)
- 강건성: 3개 입력
- 최대 품질: 5개 입력

#### 3. View Selection
| Experiment | random_view_selection | 설명 |
|------------|----------------------|------|
| E2_gt_alpha | true (default) | Random view order |
| **E2_gt_alpha_fixed** | false | Fixed view order |

### 우선순위 기준

| Priority | 의미 | 실험 |
|----------|------|------|
| **P0** | 최우선 실행 | E2_gt_alpha |
| **P1** | 핵심 비교군 | - |
| **P2** | Baseline | E0_none |
| **P3** | 주요 ablation | E1_gt, E4_bg_penalty |
| **P4** | 보조 ablation | E3_alpha, E2_gt_alpha_4v |
| **P5** | 실험적 | E5_composite, E6_rgb_pred |

### 병렬 실행 예시

```bash
# P0: 권장 실험
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha > logs/D7_1_E2_gt_alpha.log 2>&1 &

# P2: Baseline
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E0_none > logs/D7_1_E0_none.log 2>&1 &

# P3: GT mask only
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E1_gt > logs/D7_1_E1_gt.log 2>&1 &

# View ablation (4v)
CUDA_VISIBLE_DEVICES=7 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha_4v > logs/D7_1_E2_gt_alpha_4v.log 2>&1 &

# overfit
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha > logs/D7_1_E2_gt_alpha.log 2>&1 &
```

---

## Datasets

### 권장 데이터셋

| Dataset | 설명 | 상태 | Priority |
|---------|------|------|----------|
| **D7_1** ⭐ | Individual scale, PP=256 | **Stable** | P1 |
| **D8** ⭐ | Homography + skew correction | **Precision** | P0 |
| D9_norm | Full resolution + trans_norm | Experimental | P2 |
| **D10** | D8 + Up-alignment | **Proposed** | P3 |

### 실험 우선순위

| Priority | Dataset | Experiment | 설명 |
|----------|---------|------------|------|
| **P0** | D8 | E2_gt_alpha | Homography + GT mask + α |
| **P1** | D7_1 | E2_gt_alpha | Baseline comparison |
| **P2** | D9_norm | E2_gt_alpha_native | Full-resolution test |
| **P3** | D10 | E2_gt_alpha | Up-aligned (needs preprocessing) |

### 전체 데이터셋 목록

| Dataset | Paradigm | PP | Transform | Ray Error | 상태 |
|---------|----------|-----|-----------|-----------|------|
| v13 | Legacy | 256 | None | ~5° | Legacy |
| D1-D4 | object_centered | 256 (bug) | Affine | 5-13° | ⛔ Deprecated |
| D6-1,2,3 | geometry_preserving | Accurate | Various | 0° | Active |
| **D7_1** | pp_centered_shift | 256 | Affine (individual) | ~0° | ✅ **Recommended** |
| D7_2 | pp_centered_shift | 256 | Affine (average) | ~0.2° | ✅ Active |
| **D8** | precision_homography | 256 | Homography + skew | ~0° | ✅ **Precision** |
| D8_1 | precision_homography | Variable | Homography + 1.3x | ~0° | ⚠️ cx/cy varies |
| D9 | native | Original | None | 0° | Experimental |
| D9_norm | native | Original | None + trans_norm | 0° | Experimental |
| **D10** | up_aligned_zoom | 256 | Homography + up | ~0° | ✅ **Proposed** |

### 데이터 경로
```
/home/joon/data/preprocessed/FaceLift_mouse/
├── D7_1/           # Stable baseline
├── D8/             # Precision (RECOMMENDED)
├── D9_norm/        # Full-resolution
└── (D10/ - TBD)    # Up-aligned
```

### 실행 명령어 예시

```bash
# P0: D8 + E2_gt_alpha (권장)
CUDA_VISIBLE_DEVICES=4 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D8 -e E2_gt_alpha > logs/D8_E2_gt_alpha.log 2>&1 &

# P1: D7_1 baseline
CUDA_VISIBLE_DEVICES=5 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D7_1 -e E2_gt_alpha > logs/D7_1_E2_gt_alpha.log 2>&1 &

# P2: D9_norm (requires 48GB+ GPU)
CUDA_VISIBLE_DEVICES=6 nohup torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d D9_norm -e E2_gt_alpha_native > logs/D9_norm_E2.log 2>&1 &
```

---


## Preprocessing

### 통합 전처리

```bash
cd /home/joon/dev/FaceLift

# D7.1 전처리 (권장)
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7_1 \
    --input_dir /path/to/raw/data \
    --output_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

### 프리셋 옵션

| Preset | Center | Transform | 용도 |
|--------|--------|-----------|------|
| D7_1 | PP-shift | Affine (individual) | 일반 학습 |
| D7_2 | PP-shift | Affine (average) | 안정적 |
| D8 | PP-shift | Homography | 정밀 보정 |

### 전처리 검증

```bash
# 단일 데이터셋 검증
python -m mouse_extensions.scripts.validate_preprocessing \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1

# 카메라 파라미터 검증
python -m mouse_extensions.scripts.diagnostics.validate_cameras \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1
```

### Split 생성

```bash
# Train/Val split 생성 (8:2)
python -m mouse_extensions.preprocessing.generate_split \
    --dataset_dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
    --train_ratio 0.8
```

---

## Visualization & Analysis

### 1. Mask Mode 분석

```bash
# 마스크 모드 비교 분석
python -m mouse_extensions.scripts.analysis.mask_mode_analysis \
    --checkpoint checkpoints/gslrm/D7_1_E2_gt_alpha/latest.pt \
    --output_dir outputs/mask_analysis

# 빠른 비교 (wandb 이미지 기반)
python -m mouse_extensions.scripts.analysis.quick_mask_mode_compare \
    --wandb_run facelift/D7_1_E2_gt_alpha
```

### 2. Alpha 분석

```bash
# Rendered alpha 분석
python -m mouse_extensions.scripts.analysis.analyze_rendered_alpha \
    --checkpoint checkpoints/gslrm/D7_1_E2_gt_alpha/latest.pt

# Alpha threshold 분석
python -m mouse_extensions.scripts.analysis.analyze_alpha_thresholds \
    --checkpoint checkpoints/gslrm/D7_1_E2_gt_alpha/latest.pt
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
    --checkpoint checkpoints/gslrm/D7_1_E2_gt_alpha/latest.pt \
    --output_dir outputs/turntable \
    --num_frames 60
```

### 5. Checkpoint 렌더링

```bash
# Checkpoint에서 렌더링
python -m mouse_extensions.scripts.render_from_checkpoint \
    --checkpoint checkpoints/gslrm/D7_1_E2_gt_alpha/latest.pt \
    --input_dir /path/to/test/data \
    --output_dir outputs/renders
```

### 6. 리포트 생성

```bash
# 통합 리포트 생성
python -m mouse_extensions.scripts.report_generator.generate_report \
    --datasets D7_1 D7_2 D8 \
    --output_dir outputs/reports
```

---

## Monitoring

### 학습 모니터링

```bash
# 로그 실시간 확인
tail -f logs/D7_1_E2_gt_alpha.log

# GPU 사용량
nvidia-smi -l 5

# 프로세스 확인
ps aux | grep train_gslrm

# 특정 프로세스 종료 (PID 확인 후)
kill <PID>
```

### WandB
- Project: `FaceLift-Mouse`
- URL: https://wandb.ai/your-team/FaceLift-Mouse

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
| Dataset configs | `configs/datasets/` |
| Experiment configs | `configs/experiments/` |
| Config validation | `scripts/validate_config.py` |

### Data
| 항목 | 위치 |
|------|------|
| Raw data | `/home/joon/data/raw/markerless_mouse_*` |
| Preprocessed | `/home/joon/data/preprocessed/FaceLift_mouse/` |

### Model
| 항목 | 위치 |
|------|------|
| Pretrained | `checkpoints/gslrm/ckpt_0000000000021125.pt` |
| Experiment checkpoints | `checkpoints/gslrm/{dataset}_{experiment}/` |

### Outputs
| 항목 | 위치 |
|------|------|
| Training logs | `logs/` |
| Visualizations | `outputs/` |
| WandB logs | `wandb_logs/` |

### Code
| 항목 | 위치 |
|------|------|
| Training script | `train_gslrm.py` |
| Model | `gslrm/model/gslrm.py` |
| Mouse extensions | `mouse_extensions/` |
| Preprocessing | `mouse_extensions/preprocessing/` |
| Analysis scripts | `mouse_extensions/scripts/analysis/` |

---

## Troubleshooting

### Config 오류
```bash
# Config 검증
python scripts/validate_config.py configs/experiments/E2_gt_alpha.yaml
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

---

## See Also

- [CONFIG_SCHEMA.md](config/CONFIG_SCHEMA.md) - Config 구조 상세
- [PREPROCESSING_REGISTRY.md](../PREPROCESSING_REGISTRY.md) - 전처리 버전 기록
- [configs/README.md](../../configs/README.md) - Config 시스템 개요

---

*FaceLift Mouse v4.0 | Comprehensive Reference | 2026-01-23*
